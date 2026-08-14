from tinygrad.uop import Ops
from tinygrad.uop.ops import AddrSpace, PatternMatcher, UPat

class WMMASchedulePolicy:
  def __init__(self, sink:UOp):
    lst = list(sink.toposort())
    self.schedule_past: dict[UOp, tuple[UOp,...]] = {}

    loads_into: dict[UOp, set[UOp]] = {}
    for u in lst:
      # find loads used by wmma
      if u.op is not Ops.WMMA: continue
      for s in u.src:
        assert s.op is Ops.STACK
        for l in s.src:
          while l.op in {Ops.CAST, Ops.BITCAST}: l=l.src[0]
          if l.op is Ops.LOAD and l.src[0].addrspace is not AddrSpace.REG:
            loads_into.setdefault(u, set()).add(l)

    sched_groups: dict[tuple[UOp,...], list[UOp]] = {}
    for w,ls in loads_into.items():
      sched_groups.setdefault(tuple(ls), []).append(w)

    if len(sched_groups) == 0: return

    done: list[UOp] = list(sched_groups.values())[0]
    for ls,ws in list(sched_groups.items())[1:]:
      for l in ls: self.schedule_past[l] = done
      done.extend(ws)

pm_schedule_interleave_wmma = PatternMatcher([
  (UPat(Ops.LOAD, name="x"), lambda ctx,x: x.replace(src=(x.src[0].after(*ctx.schedule_past[x]),)+x.src[1:]) if x in ctx.schedule_past else None),
])
