from __future__ import annotations
import itertools
from dataclasses import dataclass, field, replace
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, AddrSpace
from tinygrad.dtype import dtypes
from typing import Any

@dataclass(frozen=True)
class Register:
  name: str
  index: int
  _cons: tuple[Register, ...] = field(default_factory=tuple)
  # vreg size represents the area an instructions output occupies, not necessarily the entire register
  size: int = 8
  @property
  def cons(self): return self._cons or (self,)
  def __repr__(self): return self.name

class IselContext:
  def __init__(self, sink:UOp):
    self.reg_n = itertools.count()
    def arg_key(u:UOp): return (1, u.arg) if u.op is Ops.SPECIAL else (0, u.arg.slot)
    self.func_args = sorted([u for u in sink.toposort() if u.op in {Ops.PARAM, Ops.SPECIAL}], key=arg_key)

  def vreg(self, cons:tuple[Register, ...]|Register, size:int=0):
    cons = cons if isinstance(cons, tuple) else (cons,)
    return Register(f"v{next(self.reg_n)}", 0, cons, size or cons[0].size)

def rdef(u:UOp):
  if u.op in {Ops.NOOP, Ops.AFTER, Ops.BITCAST} and u.src: return rdef(u.src[0])
  r = u.tag[0] if isinstance(u.tag, tuple) else u.tag
  return r if isinstance(r, Register) else None

# automatically implement machine instruction as semantically equivalent minimal subset of upstream graph, operands bound as REG PARAMs
def impl(u:UOp, opc:Any, oprs:tuple[UOp,...], **kwargs):
  def bind(i:int) -> UOp:
    if (o:=oprs[i]).is_ins: return UOp.param(i, o.dtype, addrspace=AddrSpace.REG)
    return (p := o.param_like(i)).replace(arg=replace(p.arg, addrspace=AddrSpace.REG))
  bound = {o:bind(i) for i,o in enumerate(oprs)}
  pruned = u.topovisit(lambda u: u.replace(src=tuple(bound.get(s,s) for s in u.src)), bound)
  return pruned.rtag().ins(opc, *oprs, **kwargs)

class LinearContext:
  def __init__(self, ren:ISARenderer):
    self.ren, self.stack_size = ren, 0
    self.loop_label: dict[UOp, str] = {}
  def assign_spill_slot(self, r:Register, u:UOp) -> Any: raise NotImplementedError("arch specific")

class ISARenderer(Renderer):
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  pre_regalloc_matcher: PatternMatcher
  post_regalloc_matcher: PatternMatcher
  linear_ctx_type: type = LinearContext

  def is_two_address(self, x:UOp) -> bool: return False
  def spill(self, spill_slot:Any, x:UOp) -> UOp: raise NotImplementedError("arch specific")
  def fill(self, spill_slot:Any, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def asm_str(self, uops:list[UOp], function_name:str) -> str: raise NotImplementedError("arch specific")
