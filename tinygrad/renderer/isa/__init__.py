import itertools
from tinygrad.dtype import DType, AddrSpace, dtypes
from tinygrad.uop.ops import UOp, Ops, PatternMatcher
from tinygrad.renderer import Renderer

def physical_reg(u:UOp) -> UOp|None:
  if u.arg.name and u.arg.name[0] in 'sv' and u.arg.name[1:].isdigit():
    slot = int(u.arg.name[1:]) + (256 if u.arg.name[0] == 'v' else 0)
    return UOp.placeholder((1,), dtypes.uint32, slot, AddrSpace.REG, name=u.arg.name)
  from tinygrad.renderer.isa.x86 import physical_reg as x86_reg
  return x86_reg(u)

def regbuf_reg(u:UOp) -> tuple[UOp, ...]:
  if (r:=physical_reg(u)) is not None: return (r,)
  from tinygrad.renderer.isa.x86 import regbuf_reg as x86_regbuf
  return x86_regbuf(u)

def vreg(slot:int, dtype:DType, choices:tuple[UOp, ...], name:str="vreg") -> tuple[UOp, ...]:
  return UOp.param(slot, dtype, name=name, addrspace=AddrSpace.REG), UOp(Ops.STACK, src=choices)

class IselContext:
  def __init__(self): self.reg_n = itertools.count()
  def vreg(self, dtype:DType, choices:tuple[UOp, ...]): return vreg(next(self.reg_n), dtype, choices)

# the registers an ins defines exist in the src of NATIVE, with additional definitions in an AFTER
def regs(u:UOp) -> tuple[UOp, ...]:
  if u.op is Ops.BUFFER and u.addrspace is AddrSpace.REG: return regbuf_reg(u)
  if u.op is not Ops.CALL: return ()
  src = u.body.src[0]
  return src.src if src.op is Ops.AFTER else (src,)

def vdef(u:UOp) -> UOp|None:
  # a BITCAST or CAST that survives isel is no instruction, it reads the register of its source
  if u.op in {Ops.AFTER, Ops.BITCAST, Ops.CAST}: return vdef(u.src[0])
  operands = regs(u)
  return operands[0] if operands and operands[0].addrspace is AddrSpace.REG else None

def rdef(u:UOp) -> UOp|None: return physical_reg(p) if (p:=vdef(u)) is not None else None

def register_choices(u:UOp) -> dict[UOp, tuple[UOp, ...]]:
  operands = regs(u)
  return {p:(r,) if (r:=physical_reg(p)) is not None else operands[i+1].src
          for i,p in enumerate(operands) if p.op in (Ops.PARAM, Ops.BUFFER) and p.addrspace is AddrSpace.REG}

class LinearContext:
  def __init__(self, ren:'ISARenderer'): self.ren, self.stack_size = ren, 0
  def insert_linearized(self, uops:list[UOp]) -> list[UOp]: return uops
  def assign_spill_slot(self, r:UOp) -> int: raise NotImplementedError("arch specific")

class ISARenderer(Renderer):
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  pre_regalloc_matcher: PatternMatcher
  post_regalloc_matcher: PatternMatcher
  linear_ctx_type: type[LinearContext] = LinearContext

  def is_two_address(self, x:UOp) -> bool: return False
  def spill(self, spill_slot:int, x:UOp) -> UOp: raise NotImplementedError("arch specific")
  def fill(self, spill_slot:int, x:UOp, reg:UOp) -> UOp: raise NotImplementedError("arch specific")
