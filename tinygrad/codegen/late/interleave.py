from tinygrad.uop import Ops
from tinygrad.uop.ops import AddrSpace, PatternMatcher, UPat, UOp

# heuristics:
#   - scheduled by fixed shared operand (A) of WMMAs
#   - track other load dependencies and schedule lazily
#   - also track WMMA reg stores to schedule at end of each group
#   to end reg lifetimes (bit of a hack, would be solved if reg buf semantics were correct)
class WMMASchedulePolicy:
  def __init__(self, sink:UOp):
    self.schedule_past: dict[UOp, tuple[UOp,...]] = {}

    # wmma -> load deps per operand
    wmma_deps: dict[UOp, list[set[UOp]]] = {}
    wmma_consumers: dict[UOp, list[UOp]] = {}
    # find loads used by wmma
    def _seek(u:UOp) -> UOp:
      while u.op in {Ops.CAST, Ops.BITCAST, Ops.INDEX}: u=u.src[0]
      return u
    for u in sink.toposort():
      if u.op is Ops.WMMA:
        for s in u.src:
          assert s.op is Ops.STACK
          deps = set()
          for l in s.src:
            l = _seek(l)
            if l.op is Ops.LOAD and l.src[0].addrspace is not AddrSpace.REG:
              deps.add(l)
          wmma_deps.setdefault(u, []).append(set(deps))
      if u.op is Ops.STORE:
        if u.src[1].op is Ops.INDEX:
          w = u.src[1].src[0]
          if w.op is not Ops.WMMA: continue
          wmma_consumers.setdefault(w, []).append(u)
    if len(wmma_deps) < 4: return

    sched_groups: dict[frozenset[UOp], list[UOp]] = {}
    wmma_a_deps = {k:v[0] for k,v in wmma_deps.items()}
    for w,ls in wmma_a_deps.items():
      # group by shared A fragment
      best = max((l.intersection(ls) for l in wmma_a_deps.values() if l is not ls), key=lambda lis: len(lis))
      sched_groups.setdefault(frozenset(best), []).append(w)

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
      for w in g: wmmas.extend(wmma_consumers[w])

pm_schedule_interleave_wmma = PatternMatcher([
  (UPat((Ops.LOAD, Ops.STORE), name="x"), lambda ctx,x: x.replace(src=(x.src[0].after(*ctx.schedule_past[x]),)+x.src[1:]) if x in ctx.schedule_past else None),
])
