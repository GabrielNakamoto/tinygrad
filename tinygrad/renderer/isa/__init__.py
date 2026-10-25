from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from tinygrad.renderer import Renderer
from tinygrad.device import TinyELF
from tinygrad.dtype import DType, AddrSpace
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, GroupOp
from typing import Any

class IselContext:
  def __init__(self, sink:UOp):
    self.reg_n = itertools.count()
    params = sorted([u for u in sink.toposort() if u.op is Ops.PARAM], key=lambda u: u.arg.slot)
    self.arg_offsets = {u: o for u, (o, _) in zip(params, TinyELF.iter_sig(tuple((None, u.addrspace, u.dtype, ()) for u in params)))}

  def vreg(self, dt:DType, cons:str) -> UOp: return UOp.alloc((1,), dt, addrspace=AddrSpace.REG, name=f"{cons}@{next(self.reg_n)}")

def is_reg(x:UOp, classes:dict) -> bool: return x.op in GroupOp.Defines and x.addrspace == AddrSpace.REG \
  and isinstance(x.arg.name, str) and any(c in x.arg.name for c in classes.keys())

def rdef(u:UOp):
  if u.op in {Ops.AFTER, Ops.BITCAST}: return rdef(u.src[0])
  return r[0] if (r:=regs(u)) else None

class LinearContext:
  def __init__(self, ren:ISARenderer):
    self.ren, self.stack_size = ren, 0
  def insert_linearized(self, uops:list[UOp]) -> list[UOp]: return uops
  def assign_spill_slot(self, r:Register, u:UOp) -> Any: raise NotImplementedError("arch specific")

class ISARenderer(Renderer):
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  pre_regalloc_matcher: PatternMatcher
  post_regalloc_matcher: PatternMatcher
  vreg_alloc_matcher: PatternMatcher
  linear_ctx_type: type = LinearContext

  def is_two_address(self, x:UOp) -> bool: return False
  def spill(self, spill_slot:Any, x:UOp) -> UOp: raise NotImplementedError("arch specific")
  def fill(self, spill_slot:Any, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def asm_str(self, uops:list[UOp], function_name:str) -> str: raise NotImplementedError("arch specific")
