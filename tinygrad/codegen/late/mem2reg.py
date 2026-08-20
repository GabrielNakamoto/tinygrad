from tinygrad.renderer import Renderer
from tinygrad.renderer.isa import VRegister, rdef, rdefs, ISARenderer
from tinygrad.uop.ops import PatternMatcher, UOp, UPat, Ops, ParamArg, AddrSpace
import itertools

# promotes REG space BUFFER memory loads/stores to SSA registers through control flow analysis/PHI resolution
# https://llvm.org/docs/Passes.html#mem2reg-promote-memory-to-register
class Mem2regContext:
  # in tinygrad phis are only necessary for loop carried dependencies ex.
  # stores that occur between load and one or more backedges
  def __init__(self, lst:list[UOp], ren:Renderer):
    assert isinstance(ren, ISARenderer), "mem2reg only supported for assembly backends"

    lane_ctr = itertools.count()
    rng_stack: list[UOp] = []
    current: dict[tuple[UOp, int], UOp] = {}
    rng_head: dict[UOp, dict[tuple[UOp, int], UOp]] = {}
    rng_backedge: dict[UOp, dict[tuple[UOp, int], UOp]] = {}
    self.rewrite: dict[UOp, tuple[UOp, list[UOp]]] = {}
    for u in lst:
      if u.op in {Ops.RANGE, Ops.END}: print(u.op)
      if u.op in {Ops.STORE, Ops.LOAD}:
        buf, i = u.src[0].src[0], u.src[0].src[1].src[0].val
        while buf.op is Ops.AFTER: buf=buf.src[0]
        if buf.addrspace is not AddrSpace.REG: continue

        print(u.op, buf.arg, i, rdef(u))
        # register first load and last store of each range
        if len(rng_stack):
          if u.op is Ops.LOAD and (buf,i) not in rng_head.setdefault(rng_stack[-1], {}):
            rng_head[rng_stack[-1]][(buf,i)]=u
          if u.op is Ops.STORE:
            rng_backedge.setdefault(rng_stack[-1], {})[(buf,i)] = u

        # otherwise load maps to last stores value?
        if u.op is Ops.STORE: current[(buf,i)] = u
        if u.op is Ops.LOAD: self.rewrite[u] = (current[(buf,i)], [])

      if u.op is Ops.RANGE: rng_stack.append(u)
      if u.op is Ops.END:
        rng = rng_stack.pop()
        for ptr,l in rng_head[rng].items():
          if rng in rng_backedge and ptr in rng_backedge[rng]:
            # phi between last flat store and store backedge
            cur, bedge = rdef(self.rewrite[l][0]), rdef(rng_backedge[rng][ptr])
            vr = ren.vreg(cur.cons, width=cur.width, alignment=cur.width, phi=(cur,bedge))
            print("inserting backedge phi", ptr[0].arg, ptr[1], vr.phi)
            phi = UOp.placeholder((1,), ptr[0].dtype, next(lane_ctr), AddrSpace.REG).replace(tag=(vr,))
            self.rewrite[u] = (phi, [phi])

# REG store copy/coalesce is handled by regalloc PHI logic
# simply rewrite LOADs to equivalent SSA node
pm_promote_regbufs = PatternMatcher([
  (UPat(Ops.LOAD, name="x"), lambda ctx,x: ctx.rewrite[x] if x in ctx.rewrite else None),
])
