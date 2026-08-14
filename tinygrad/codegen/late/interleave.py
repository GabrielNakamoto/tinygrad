from tinygrad.uop import Ops
from tinygrad.uop.ops import AddrSpace, PatternMatcher, UPat

class WMMASchedulePolicy:
  def __init__(self, sink:UOp):
    lst = list(sink.toposort())

    loads_into: dict[UOp, list[UOp]] = {}
    for u in lst:
      # find loads used by wmma
      if u.op is not Ops.WMMA: continue
      for s in u.src:
        assert s.op is Ops.STACK
        for l in s.src:
          while l.op in {Ops.CAST, Ops.BITCAST}: l=l.src[0]
          if l.op is Ops.LOAD and l.src[0].addrspace is not AddrSpace.REG:
            loads_into.setdefault(u, []).append(l)

    # detect overlapping loads? used for multiple WMMAS
    # ususally multiple wmmas use the same outputs and should be placed in the same block
    overlaps: dict[UOp, UOp] = {}
    for w,ls in loads_into.items():
      if w in overlaps.values(): continue
      for ow,ols in loads_into.items():
        if w is ow: continue
        if (n := len(set(ls).intersection(set(ols)))) > 0:
          overlaps[w]=ow

    # for all blocks but the first insert AFTER targeting both previous overlapping WMMAs
    self.schedule_past: dict[UOp, tuple[UOp,UOp]] = {}
    scheds = list(overlaps.items())
    for i,(a,b) in enumerate(scheds):
      if i == 0: continue
      self.schedule_past[b] = self.schedule_past[a] = scheds[i-1]

    print(len(self.schedule_past))

pm_schedule_interleave_wmma = PatternMatcher([
  (UPat(Ops.WMMA, name="w"), lambda ctx,w: w.after(*ctx.schedule_past[w]) if w in ctx.schedule_past else None),
])
