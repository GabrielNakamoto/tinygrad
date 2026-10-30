from __future__ import annotations
import itertools
from dataclasses import dataclass, field, replace
from tinygrad.renderer import Renderer
from tinygrad.device import TinyELF
from tinygrad.dtype import DType, AddrSpace
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, GroupOp, UPat
from typing import Any

# specify callee reserved registers that either cannot be touched
# or must be refilled before end of kernel (ex. x86 callee registers,
# RDNA3 kernel arg sgprs and thread/block ids
@dataclass(frozen=True)
class ABIInfo:
  reserved: set[str]

# how to model the relation of vregister strings to particular register
# class elegantly for regalloc
@dataclass(frozen=True)
class RegisterClass:
  name: str
  names: list[str]
  spill_size: int
  belongs: Callable[[UOp], bool]

# all non scalar reg buffers should get rewritten to LOCAL for isa backends?
# match reg addrspace knowing they are scalar and model as individual regs
pm_alloc_vregs = PatternMatcher([
  (UPat(Ops.INS, name="u"), lambda ctx,u: u.replace(arg=replace(u.arg, name=f"{rc.name}{next(ctx[1])}"))
    if (rc:=next((rc for rc in ctx[0].reg_classes if rc.belongs(u)), None)) is not None else None),
])

class IselContext:
  def __init__(self, sink:UOp):
    params = sorted([u for u in sink.toposort() if u.op is Ops.PARAM], key=lambda u: u.arg.slot)
    self.arg_offsets = {u: o for u, (o, _) in zip(params, TinyELF.iter_sig(tuple((None, u.addrspace, u.dtype, ()) for u in params)))}

def is_reg(x:UOp, classes:dict) -> bool: return x.op in GroupOp.Defines and x.addrspace == AddrSpace.REG \
  and isinstance(x.arg.name, str) and any(c in x.arg.name for c in classes.keys())

def rdef(u:UOp, classes:dict):
  if u.op in {Ops.AFTER, Ops.BITCAST}: return rdef(u.src[0])
  return u.arg.name if is_reg(u, classes) else None

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
  linear_ctx_type: type = LinearContext
  reg_classes: list[RegisterClass]
  abi_info: ABIInfo

  def is_two_address(self, x:UOp) -> bool: return False
  def spill(self, spill_slot:Any, x:UOp) -> UOp: raise NotImplementedError("arch specific")
  def fill(self, spill_slot:Any, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def asm_str(self, uops:list[UOp], function_name:str) -> str: raise NotImplementedError("arch specific")
