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
  def r(self): self.rc += 1
  # in tinygrad phis are only necessary for loop carried dependencies ex.
  # stores that occur between load and one or more backedges
  def __init__(self, lst:list[UOp], ren:Renderer):
    assert isinstance(ren, ISARenderer), "mem2reg only supported for assembly backends"
    self.ren, self.rc = ren, 0
    lane_ctr = itertools.count()
    rng_stack: list[UOp] = []
    current: dict[tuple[UOp, int], UOp] = {}
    rng_head: dict[UOp, dict[tuple[UOp, int], UOp]] = {}
    rng_backedge: dict[UOp, dict[tuple[UOp, int], UOp]] = {}
    redge: dict[UOp, VRegister] = {}
    self.phis: dict[tuple[UOp, int], dict[int, UOp]] = {}
    n = 0
    for u in lst:
      if u.op in {Ops.STORE, Ops.LOAD}:
        buf, i = bptr(u.src[0])
        if buf.addrspace is not AddrSpace.REG: continue

        # register first load and last store of each range
        if len(rng_stack):
          if u.op is Ops.LOAD and (buf,i) not in rng_head.setdefault(rng_stack[-1], {}):
            rng_head[rng_stack[-1]][(buf,i)]=u
          if u.op is Ops.STORE:
            rng_backedge.setdefault(rng_stack[-1], {})[(buf,i)] = u

        # otherwise load maps to last stores value?
        if u.op is Ops.STORE: current[(buf,i)] = u
        if u.op is Ops.LOAD: redge[u] = rdef(current[(buf,i)])
      if u.op is Ops.RANGE:
        n += 1
        rng_stack.append(u)
      if u.op is Ops.END:
        rng = rng_stack.pop()
        if rng in rng_head:
          for ptr,l in rng_head[rng].items():
            if rng in rng_backedge and ptr in rng_backedge[rng]:
              # phi between last flat store and store backedge
              cur, bedge = redge[l], rdef(rng_backedge[rng][ptr])
              vr = ren.vreg(cur.cons, width=cur.width, alignment=cur.width, phi=(cur,bedge))
              phi = UOp.placeholder((1,), ptr[0].dtype, next(lane_ctr), AddrSpace.REG).replace(tag=(vr,))
              self.phis.setdefault(ptr, {})[n] = phi
    self.current: dict[UOp, UOp] = {}

pm_insert_phis = PatternMatcher([
  (UPat(Ops.RANGE, name="rng"), lambda ctx,rng: ctx.r()),
  (UPat.var("idx").load(name="x"), lambda ctx,idx,x:
    ((nx := ctx.phis[bptr(idx)][ctx.rc]), [nx]) if bptr(idx) in ctx.phis else None),
])

# REG store copy/coalesce is handled by regalloc PHI logic
# simply rewrite LOADs to equivalent SSA node
def update(ctx, val:UOp, x:UOp, idx:UOp):
  nx = ctx.ren.copy(val, rdef(x))
  ctx.current[bptr(idx)] = nx
  return (nx, [nx])

# only deterministic LOADs remain
pm_promote_regbufs = PatternMatcher([
  (UPat.var("idx").store(UPat.var("val"), name="x"), update),
  (UPat.var("idx").load(name="x"), lambda ctx,idx,x: (ctx.current[bptr(idx)], [])),
])
