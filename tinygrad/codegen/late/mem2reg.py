from tinygrad.renderer import Renderer
from tinygrad.renderer.isa import VRegister, rdef, rdefs, ISARenderer
from tinygrad.uop.ops import PatternMatcher, UOp, UPat, Ops, ParamArg, AddrSpace
import itertools

def bptr(x:UOp) -> tuple[UOp, int]:
  buf,idx = x.src
  while buf.op is Ops.AFTER: buf=buf.src[0]
  return (buf,idx.src[0].val)

# promotes REG space BUFFER memory loads/stores to SSA registers through control flow analysis/PHI resolution
# https://llvm.org/docs/Passes.html#mem2reg-promote-memory-to-register
class Mem2regContext:
  # in tinygrad phis are only necessary for loop carried dependencies ex.
  # stores that occur between load and one or more backedges
  def __init__(self, lst:list[UOp], ren:Renderer):
    assert isinstance(ren, ISARenderer), "mem2reg only supported for assembly backends"
    self.ren = ren
    self.current: dict[UOp, UOp] = {}
    self.nl: dict[tuple[UOp, int], int] = {}

    lane_ctr = itertools.count()
    rng_stack: list[UOp] = []
    current: dict[tuple[UOp, int], UOp] = {}
    # NOTE: could calculate this better by just saving all ops
    # that happen in a rng and analyzing at END
    rng_head: dict[UOp, dict[tuple[UOp, int], UOp]] = {}
    rng_backedge: dict[UOp, dict[tuple[UOp, int], UOp]] = {}
    # loads -> last store
    flat: dict[tuple[UOp, int], dict[UOp, UOp]] = {}
    self.phis: dict[tuple[tuple[UOp, int], int], UOp] = {}
    nl: dict[UOp, int] = {}
    for u in lst:
      if u.op in {Ops.STORE, Ops.LOAD}:
        buf, i = bptr(u.src[0])
        if buf.addrspace is not AddrSpace.REG: continue
        # register first load and last store of each range
        if len(rng_stack):
          rng = rng_stack[-1]
          if u.op is Ops.LOAD and (buf,i) not in rng_head.setdefault(rng, {}):
            if rng in rng_backedge and (buf,i) in rng_backedge[rng]: rng_backedge[rng].pop((buf,i))
            rng_head[rng][(buf,i)]=u
          if u.op is Ops.STORE:
            rng_backedge.setdefault(rng, {})[(buf,i)] = u
        if u.op is Ops.STORE: current[(buf,i)] = u
        if u.op is Ops.LOAD:
          flat.setdefault((buf,i), {})[u] = current[(buf,i)]
          nl[u] = len(flat[(buf,i)])
      if u.op is Ops.RANGE: rng_stack.append(u)
      if u.op is Ops.END:
        if (rng := rng_stack.pop()) in rng_head:
          for ptr,l in rng_head.pop(rng).items():
            if rng in rng_backedge and ptr in rng_backedge[rng]:
              # phi between last flat store and store backedge
              cur, bedge = rdef(flat[ptr][l]), rdef(rng_backedge[rng][ptr])
              vr = ren.vreg(cur.cons, width=cur.width, alignment=cur.alignment, phi=(cur,bedge))
              phi = UOp.placeholder((1,), ptr[0].dtype, next(lane_ctr), AddrSpace.REG).replace(tag=(vr,))
              self.phis[(ptr, nl[l])] = phi

  def try_phi(self, idx:UOp, x:UOp) -> UOp|None:
    ptr = bptr(idx)
    self.nl[ptr] = self.nl.setdefault(ptr, 0) + 1
    phi = self.phis.get((ptr, self.nl[ptr]), None)
    return (phi, [phi]) if phi is not None else None

pm_insert_phis = PatternMatcher([
  (UPat.var("idx").load(name="x"), lambda ctx,idx,x: ctx.try_phi(idx, x)),
])

# REG store copy/coalesce is handled by regalloc PHI logic
# simply rewrite LOADs to equivalent SSA node
def update(ctx, val:UOp, x:UOp, idx:UOp):
  nx = ctx.ren.vcopy(val, rdef(x))
  ctx.current[bptr(idx)] = nx
  return (nx, [nx])

# only deterministic LOADs remain
pm_promote_regbufs = PatternMatcher([
  (UPat.var("idx").store(UPat.var("val"), name="x"), update),
  (UPat.var("idx").load(name="x"), lambda ctx,idx,x: (ctx.current[bptr(idx)], [])),
])
