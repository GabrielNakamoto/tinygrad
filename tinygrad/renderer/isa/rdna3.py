from __future__ import annotations
# flake8: noqa: E702
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.helpers import Target
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.renderer.amd.elf import assemble_linear
from tinygrad.renderer.cstyle import HIPRenderer
import tinygrad.renderer.amd.dsl as dsl
import tinygrad.runtime.autogen.amd.rdna3.ins as RDNA3Ops
from tinygrad.renderer.isa import IselContext, ISARenderer, vreg, regs, vdef, rdef, register_choices, physical_reg

SGPR = tuple(UOp.placeholder((1,), dtypes.uint32, i, AddrSpace.REG, name=f"s{i}") for i in range(32))
VGPR = tuple(UOp.placeholder((1,), dtypes.uint32, 256+i, AddrSpace.REG, name=f"v{i}") for i in range(32))
KERNARG_PTR = SGPR[:2]
def const(v, dtype=dtypes.uint32): return UOp.cconst(v, dtype)
def is_const(x:UOp) -> bool: return x.op is Ops.CONST or x.op is Ops.CAST and x.src[0].op is Ops.CONST
def const_val(x:UOp): return x.val if x.op is Ops.CONST else x.src[0].val
def rafter(x:UOp, _:bool=False) -> UOp: return x.without_after

extra_matcher = pre_isel_matcher = PatternMatcher([])

def abi(ctx:IselContext, x:UOp) -> UOp|None:
  if x.op is Ops.SPECIAL: return VGPR[0].cast(x.dtype) if x.arg == 'lidx0' else None
  if x.addrspace is not AddrSpace.GLOBAL: return None
  pair = SGPR[4+2*x.arg.slot:6+2*x.arg.slot]
  return UOp.ins(RDNA3Ops.s_load_b64, (KERNARG_PTR[0], const(x.arg.slot*8)), dtypes.uint64, *pair)

def alloc_vregs(ctx:IselContext, x:UOp) -> UOp|None:
  if x.dtype is dtypes.void: return None
  if (dst:=vdef(x)) is not None:
    if physical_reg(dst) is None: return None
    defs = tuple(p for choices in register_choices(x).values() for p in ctx.vreg(x.dtype, choices))
  else: defs = ctx.vreg(x.dtype, SGPR[12:] if x.body.arg.code.func is RDNA3Ops.SMEM else VGPR[1:])
  return UOp.ins(x.body.arg.code, x.src_without_body, x.dtype, *defs)

def load(x:UOp, idx:UOp) -> UOp:
  if idx.addrspace is not AddrSpace.GLOBAL or x.dtype is not dtypes.float32: raise NotImplementedError(x)
  return UOp.ins(RDNA3Ops.global_load_b32, (idx.src[1] << const(2, dtypes.int32), idx.src[0]), x.dtype)

def store(x:UOp, idx:UOp, val:UOp) -> UOp:
  if idx.addrspace is not AddrSpace.GLOBAL or val.dtype is not dtypes.float32: raise NotImplementedError(x)
  if is_const(val): val = UOp.ins(RDNA3Ops.v_mov_b32_e32, (val,), val.dtype)
  return UOp.ins(RDNA3Ops.global_store_b32, (idx.src[1] << const(2, dtypes.int32), idx.src[0], val), dtypes.void)

isel_matcher = PatternMatcher([
  (UPat(Ops.SHL, name="x"), lambda x: UOp.ins(RDNA3Ops.v_mov_b32_e32, (const(const_val(x.src[0]) << const_val(x.src[1]), x.dtype),), x.dtype)
   if all(map(is_const, x.src)) else UOp.ins(RDNA3Ops.v_lshlrev_b32_e32, x.src[::-1], x.dtype)),
  (UPat(Ops.ADD, dtypes.float32, name="x"), lambda x: UOp.ins(RDNA3Ops.v_add_f32_e32, x.src, x.dtype)),
  (UPat.var("idx").store(UPat.var("val"), allow_any_len=True).named("x"), store),
  (UPat.var("idx").load(name="x", allow_any_len=True), load),
  (UPat((Ops.PARAM, Ops.SPECIAL), name="x"), abi),
  (UPat(Ops.CALL, name="x"), alloc_vregs),
])

def lower_range(x:UOp) -> tuple[UOp, list[UOp]]:
  idx = UOp.ins(RDNA3Ops.v_mov_b32_e32, (const(0),), x.dtype, *vreg(0, x.dtype, VGPR[1:], name=x.key.hex()))
  rng = x.replace(src=x.src+(idx,))
  return idx.after(rng), [idx, rng]

def lower_end(x:UOp, c:UOp) -> tuple[UOp, list[UOp]]:
  inc = UOp.ins(RDNA3Ops.v_add_nc_u32_e32, (const(1), c), c.dtype, *regs(c.src[0]))
  end = x.replace(src=(x.src[0], c.src[1], inc))
  return end, [inc, end]

pre_regalloc_matcher = PatternMatcher([
  (UPat(Ops.RANGE, name="x"), lower_range),
  (UPat(Ops.END, src=(UPat(), UPat(Ops.AFTER, name="c")), name="x"), lower_end),
])

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.SINK, name="x"), lambda x: (x, [UOp.ins(RDNA3Ops.s_endpgm, (), dtypes.void)])),
  (UPat(Ops.RANGE, name="x"), lambda x: (x, [x, UOp.ins(RDNA3Ops.v_cmp_gt_u32_e32, (x.src[0], x.src[-1]), dtypes.void),
                                                  UOp.ins(RDNA3Ops.s_cbranch_vccz, (x,), dtypes.void)])),
  (UPat(Ops.END, name="x"), lambda x: (x, [UOp.ins(RDNA3Ops.v_cmp_gt_u32_e32, (x.src[1].src[0], x.src[2]), dtypes.void),
                                                UOp.ins(RDNA3Ops.s_cbranch_vccnz, (x.src[1],), dtypes.void), x])),
  (UPat((Ops.CONST, Ops.CAST, Ops.BITCAST, Ops.AFTER, Ops.BUFFER, Ops.PARAM, Ops.SPECIAL, Ops.STACK), name="x"), lambda x: (x, [])),
])

def encode(x:UOp):
  def encfield(x:UOp):
    if is_const(x): return const_val(rafter(x))
    r = rdef(x)
    assert r is not None, x
    off = r.arg.slot
    return dsl.v[off-256] if off >= 256 else dsl.s[off:off+1] if x.dtype.itemsize == 8 else dsl.s[off]

  op, src = x.body.arg.code, x.src_without_body
  if op.func is RDNA3Ops.SMEM: return op(sdata=encfield(x), sbase=dsl.s[0:1], offset=encfield(src[1]))
  if op.func is RDNA3Ops.GLOBAL:
    fields = dict(addr=encfield(src[0]), saddr=encfield(src[1]))
    if x.dtype is dtypes.void: fields['data'] = encfield(src[2])
    else: fields['vdst'] = encfield(x)
    return op(**fields)
  if op.func is RDNA3Ops.VOP2: return op(vdst=encfield(x), src0=encfield(src[0]), vsrc1=encfield(src[1]))
  if op.func is RDNA3Ops.VOP1: return op(vdst=encfield(x), src0=encfield(src[0]))
  if op.func is RDNA3Ops.VOPC: return op(src0=encfield(src[0]), vsrc1=encfield(src[1]))
  if op.func is RDNA3Ops.SOPP: return op(simm16=0)
  raise NotImplementedError(op)


class RDNA3Renderer(ISARenderer):
  device = "AMD"
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  extra_matcher = extra_matcher
  post_regalloc_matcher = post_regalloc_matcher
  pre_regalloc_matcher = pre_regalloc_matcher
  code_for_op = {x: lambda: None for x in ( Ops.SQRT, Ops.LOG2, Ops.EXP2, Ops.SUB, Ops.RECIP, Ops.TRUNC, Ops.CMPLT,
                                              Ops.CMPEQ, Ops.CMPNE, Ops.XOR, Ops.SHR, Ops.SHL, Ops.MAX, Ops.MULACC)}

  def __init__(self, target: Target):
    if target.arch.split(",")[0] != "gfx1100": raise RuntimeError(f"RDNA3Renderer only supports gfx1100, got {target.arch}")
    super().__init__(target)
    from tinygrad.renderer.tc import get_amd
    self.shared_max, self.tensor_cores = HIPRenderer.shared_max, get_amd(target.arch)

  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d not in dtypes.fp8s}
  def asm_str(self, uops:list[UOp], function_name:str) -> str: return '\n'.join(str(encode(u)) for u in uops)

  def render(self, uops:list[UOp]) -> str:
    raise RuntimeError(f"Unlowered UOps remain in RDNA3 graph, cannot assemble: {[u.op for u in uops if u.op is not Ops.CALL]}")

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    insts:list[tuple[UOp|None, dsl.Inst, int]] = []
    starts, ends = {}, {}
    pc = 0
    for u in lin.src:
      if u.op is Ops.RANGE: starts[u] = pc; continue
      if u.op is Ops.END: ends[u.src[1]] = pc; continue
      if u.op is not Ops.CALL: raise RuntimeError(f"cannot assemble {u.op}")
      insts.append((u, encode(u), pc))
      pc += insts[-1][1].size()
      if u.body.arg.code.func in (RDNA3Ops.SMEM, RDNA3Ops.GLOBAL) and u.dtype is not dtypes.void:
        insts.append((None, RDNA3Ops.s_waitcnt(simm16=0), pc))
        pc += insts[-1][1].size()
    out = []
    for call,inst,pc in insts:
      if call is not None and call.body.arg.code in (RDNA3Ops.s_cbranch_vccz, RDNA3Ops.s_cbranch_vccnz):
        rng = call.src[1]
        target = ends[rng] if call.body.arg.code is RDNA3Ops.s_cbranch_vccz else starts[rng]
        inst = call.body.arg.code(simm16=((target-pc-inst.size())//4)&0xffff)
      out.append(UOp.native(inst).call())
    return assemble_linear(prg, lin.replace(src=tuple(out)), self.target.arch)
