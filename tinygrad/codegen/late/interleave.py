from tinygrad.uop import Ops
from tinygrad.uop.ops import AddrSpace, PatternMatcher, UPat, UOp

class WMMASchedulePolicy:
  def __init__(self, sink:UOp):
    self.schedule_past: dict[UOp, tuple[UOp,...]] = {}

    # wmma -> load deps per operand
    wmma_deps: dict[UOp, list[set[UOp]]] = {}
    # find loads used by wmma
    def _seek(u:UOp) -> UOp:
      while u.op in {Ops.CAST, Ops.BITCAST, Ops.INDEX}: u=u.src[0]
      return u
    for u in sink.toposort():
      if u.op is not Ops.WMMA: continue
      for s in u.src:
        assert s.op is Ops.STACK
        deps = set()
        for l in s.src:
          l = _seek(l)
          if l.op is Ops.LOAD and l.src[0].addrspace is not AddrSpace.REG:
            deps.add(l)
        wmma_deps.setdefault(u, []).append(set(deps))


    # heuristics:
    # - scheduled by fixed shared operand (A) of WMMAs
    # - track other load dependencies and schedule lazily

    sched_groups: dict[frozenset[UOp], list[UOp]] = {}
    wmma_a_deps = {k:v[0] for k,v in wmma_deps.items()}
    for w,ls in wmma_a_deps.items():
      # group by shared A fragment
      best = max((l.intersection(ls) for l in wmma_a_deps.values() if l is not ls), key=lambda lis: len(lis))
      sched_groups.setdefault(frozenset(best), []).append(w)

    # TODO: try search, optimally orient the placements to minimize peak pressure?
    if len(sched_groups) == 0: return
    groups = list(sched_groups.values())
    scheduled: set[UOp] = set()
    wmmas: list[UOp] = []
    for i,g in enumerate(groups):
      deps = set()
      for w in g:
        for ls in wmma_deps[w]: deps.update(ls)
      if i > 0:
        for l in deps:
          if l in scheduled: continue
          self.schedule_past[l] = tuple(wmmas)
      scheduled.update(deps)
      wmmas.extend(g)

pm_schedule_interleave_wmma = PatternMatcher([
  (UPat(Ops.LOAD, name="x"), lambda ctx,x: x.replace(src=(x.src[0].after(*ctx.schedule_past[x]),)+x.src[1:]) if x in ctx.schedule_past else None),
])
