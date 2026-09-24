import unittest, itertools, torch, numpy as np
from tinygrad import Tensor, Device, dtypes, nn, GlobalCounters
from tinygrad.helpers import VIZ
from tinygrad.renderer.isa import ISARenderer, IselContext, Register
from tinygrad.uop.ops import graph_rewrite, PatternMatcher, UPat, UOp, Ops, ProgramInfo, AddrSpace, GroupOp
from tinygrad.codegen import full_rewrite_to_sink, pm_to_program
from tinygrad.engine.realize import _get_call_to_compile, run_linear
from test.backend.test_ops import prepare_test_op

def _cross_exec(graph:Tensor) -> int:
  device = Device[Device.DEFAULT]
  isa_ren, final_ren = device.renderer, next(r for r in device.renderers if not issubclass(r, ISARenderer))
  final_ren = final_ren(isa_ren.target)

  def transmute(ast:UOp) -> UOp:
    sink = full_rewrite_to_sink(ast, isa_ren)
    # perform instruction selection
    sink = graph_rewrite(sink, isa_ren.pre_isel_matcher, ctx=itertools.count(-1,-1), name="pre instruction selection", bottom_up=True)
    sink = graph_rewrite(sink, isa_ren.isel_matcher, ctx=IselContext(sink), name="instruction selection", bottom_up=True)
    sink = graph_rewrite(sink, PatternMatcher([]), name="view machine code")

    # re-expand CALL graphs
    # TODO: make this better, sucks (could add binding metadata in InstInfo?)
    pm_embed_bodies = PatternMatcher([(UPat(Ops.CALL, name="c"), lambda c: graph_rewrite(c,
      PatternMatcher([(UPat(Ops.PARAM, name="p"), lambda ctx,p: ctx[p.arg.slot] if p.addrspace is AddrSpace.REG else None)]),
      ctx=c.src[1:], enter_calls=True).body),
    ])
    # strip tags on round trip to enable UOp coalescence
    pm_strip_tags = PatternMatcher([
      (UPat(GroupOp.All, name="x"), lambda x: x.replace(tag=None) if (isinstance(x.tag, tuple) and isinstance(x.tag[0], Register)) \
        or x.tag is True else None),
    ])
    sink = graph_rewrite(sink, pm_embed_bodies, name="implement as UOps (embed bodies)")
    sink = graph_rewrite(sink, pm_strip_tags, name="remove register references")

    # plug through non-assembly backend's render pass
    prg_info = ProgramInfo.from_sink(sink, final_ren.target)
    prg = UOp(Ops.PROGRAM, src=(sink,), arg=prg_info)
    prg = graph_rewrite(prg, pm_to_program, ctx=final_ren, name="linearize/render")
    if VIZ: graph_rewrite(prg, PatternMatcher([]), name="View Program")
    return prg

  # compile kernels and swap calls
  linear = graph.schedule_linear()
  calls = {c: (c.src[0], final_ren) for c in linear.toposort() if c.op is Ops.CALL and _get_call_to_compile(c) is not None}
  prgs = {c: transmute(a[0]) for c,a in calls.items()}
  linear = linear.substitute({c: c.replace(src=(c.src[0].substitute({a[0]: prgs[c]}), *c.src[1:])) for c,a in calls.items()})
  GlobalCounters.reset()
  run_linear(linear, jit=True)
  return GlobalCounters.kernel_count

# TODO: verify post-linearize round trip?
@unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, ISARenderer), "cross compilation is for asm backends")
class TestRetarget(unittest.TestCase):
  def _helper_test_cross(self, ts:tuple[Tensor,...], op, atol=1e-6, rtol=1e-3):
    Tensor.realize(*ts)
    GlobalCounters.reset()
    truth = op(*ts).realize()
    expected = GlobalCounters.kernel_count
    self.assertEqual(_cross_exec((out := op(*ts))), expected)
    np.testing.assert_allclose(out.numpy(), truth.numpy(), atol=atol, rtol=rtol)

  def test_transfer_plus(self):
    self._helper_test_cross((Tensor([1,2,3,4]), Tensor([27, 26, 25, 24])), lambda a,b: a.float() + b.float())

  def test_transfer_hplus(self):
    self._helper_test_cross((Tensor([1,2,3,4]), Tensor([27, 26, 25, 24])), lambda a,b: a.half() + b.half())
  
  def test_transfer_gemm(self):
    self._helper_test_cross((Tensor.rand(32,32), Tensor.rand(32,32)), Tensor.matmul)

  def test_transfer_hgemm(self):
    self._helper_test_cross((Tensor.rand(32,32), Tensor.rand(32,32)), lambda x,y: x.half().matmul(y.half()))

  def test_transfer_idiv(self):
    self._helper_test_cross((Tensor([5,6,7]),Tensor([1,2,3])), lambda x,y: x//y)

  def test_transfer_mnist(self):
    layers = [
      nn.Conv2d(1, 32, 5), Tensor.relu,
      nn.Conv2d(32, 32, 5), Tensor.relu,
      nn.BatchNorm(32), Tensor.max_pool2d,
      nn.Conv2d(32, 64, 3), Tensor.relu,
      nn.Conv2d(64, 64, 3), Tensor.relu,
      nn.BatchNorm(64), Tensor.max_pool2d,
      lambda x: x.flatten(1), nn.Linear(576, 1)]

    Tensor.realize(*[p.replace(Tensor.ones_like(p).contiguous()) for p in nn.state.get_parameters(layers)])
    self._helper_test_cross((Tensor.rand(1, 1, 28, 28),), lambda x: x.sequential(layers))

  def test_transfer_loop(self):
    from test.backend.test_wait_loop import wait_loop_kernel
    def mk(): return Tensor.custom_kernel(Tensor.empty(1, dtype=dtypes.int), fxn=wait_loop_kernel)[0]
    ref, out = mk(), mk()
    GlobalCounters.reset()
    truth = ref.item()
    native = GlobalCounters.kernel_count
    cross = _cross_exec(out)
    self.assertEqual(native, cross)
    self.assertEqual(truth, out.item())

if __name__ == '__main__':
  np.random.seed(2973)
  unittest.main(verbosity=2)
