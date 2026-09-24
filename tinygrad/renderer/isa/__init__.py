from __future__ import annotations
import itertools
from dataclasses import dataclass, field, replace
from tinygrad.renderer import Renderer
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, GroupOp, AddrSpace
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
  return u.tag[0] if isinstance(u.tag, tuple) else u.tag

def bind_opr(u:UOp, slot:int): return (p := u.param_like(slot)).replace(arg=replace(p.arg, addrspace=AddrSpace.REG))
def bind(src:tuple[UOp,...], graph:tuple[UOp,...]):
  bound = {u:bind_opr(u,i) for i,u in enumerate(src) if src in graph}
  return tuple(bound.get(u,u) for u in graph)

# NOTE: should this be in the UOP constructor? semi arch dependent, nice to minimize hidden side effects
# NOTE: NOOPs are sometimes used as encoding padding, discluded from arity
def impl_ins(x:UOp, src:tuple[UOp,...]):
  if x.op in {Ops.NOOP, Ops.RANGE}: return x
  if x.op in {Ops.STACK, Ops.GROUP}:
    return x.replace(src=tuple(bind_opr(s,i) if s.dtype is not dtypes.void else x.src[i] for i,s in enumerate(src)))
  if len(src) == 1 and len(x.src) == 0: return bind_opr(*src, 0)
  if x.op in {Ops.CAST, Ops.BITCAST}:
    # cant rely on arity because of NOOP padding?, should we use null edges for encoding metadata? probably belongs in arg (InstInfo)
    i,value = next((j,s) for j,s in enumerate(src) if s.dtype is not dtypes.void)
    return x.replace(src=(bind_opr(value, i),))
  if x.op in GroupOp.ALU|{Ops.INDEX}:
    if len(x.src) == len(src):
      return x.replace(src=tuple(bind_opr(s,i) for i,s in enumerate(src)))
    return x.replace(src=bind(src, x.src))
  raise NotImplementedError(f"cannot automatically implement op: {x.op}")

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
