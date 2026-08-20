from tinygrad.renderer import Renderer
from tinygrad.renderer.isa import Phi, rdef, rdefs
from tinygrad.uop.ops import PatternMatcher, UOp, UPat, Ops, ParamArg, AddrSpace
import itertools

# promotes REG space BUFFER memory loads/stores to SSA registers through control flow analysis/PHI resolution
# https://llvm.org/docs/Passes.html#mem2reg-promote-memory-to-register
class Mem2regContext:
  # in tinygrad phis are only necessary for loop carried dependencies ex.
  # stores that occur between load and one or more backedges
  def __init__(self, lst:list[UOp], ren:Renderer):
    lane_ctr = itertools.count()
    rng_stack: list[UOp] = []
    rng_uses: dict[tuple[UOp, int], list[UOp]] = {}
    last_linear: dict[tuple[UOp, int], UOp] = {}
    rewrite: dict[UOp, UOp] = {}
    for u in lst:
      if u.op in {Ops.STORE, Ops.LOAD}:
        buf, i = u.src[0].src[0], u.src[0].src[1].src[0].val
        while buf.op is Ops.AFTER: buf=buf.src[0]
        if buf.addrspace is not AddrSpace.REG: continue
        if len(rng_stack):
          rng_uses.setdefault((buf,i), []).append(u)
        if u.op is Ops.STORE: last_linear[(buf,i)] = u
        if u.op is Ops.LOAD: rewrite[u] = last_linear[(buf,i)]
      if u.op is Ops.RANGE: rng_stack.append(u)
      if u.op is Ops.END:
        rng_stack.pop()
        # once all ranges have been processed analyze uses
        if len(rng_stack) == 0:
          # insert phi merge if necessary
          for ptr,uses in rng_uses.items():
            for i,u in enumerate(uses):
              # if loop carried dependencies come after load, overwrite linear reference with merge
              if u.op is Ops.LOAD and any(s.op is Ops.STORE for s in uses[i+1:]):
                # regalloc emits copiy/coalesces from PHI knowledge?
                phi = Phi(tuple(rdef(s) for s in uses[i+1:] if s.op is Ops.STORE))
                print("Inserting phi", phi.edges)
                rewrite[u] = UOp.placeholder((1,), ptr[0].dtype, next(lane_ctr), AddrSpace.REG).replace(tag=(phi,))
          rng_uses.clear()

# REG store copy/coalesce is handled by regalloc PHI logic
# simply rewrite LOADs to equivalent SSA node
pm_promote_regbufs = PatternMatcher([
  (UPat(Ops.LOAD, name="x"), lambda x: ((nx := rewrite[x]), [nx]) if x in rewrite else None),
])
