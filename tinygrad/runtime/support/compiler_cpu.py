import subprocess, shutil, tempfile, os
from tinygrad.device import Compiler
from tinygrad.helpers import getenv, capstone_flatdump, cpu_objdump

class ClangCompiler(Compiler):
  def __init__(self, arch:list[str], cachekey="compile_clang_obj"):
    assert len(arch) >= 2, f"invalid arch string: {','.join(arch)!r}, expected '<arch>,<cpu>,[<feats>]' (eg. 'x86_64,znver2')"
    self.arch, cpu, *feats = arch
    match self.arch:
      case "x86_64": self.args = [f"-march={cpu}"] + [f"-mno{f}" if f.startswith("-") else f"-m{f}" for f in feats]
      # on arm march means "runs on this arch and superset" instead of "optimize for this arch". x86 march == arm mcpu
      # x18 is a reserved platform register. It is clobbered on context switch in macos and is used to store TEB pointer in windows on arm
      case "arm64": self.args = ["-ffixed-x18", "-mcpu=" + "+".join([cpu] + ["no"+f[1:] if f.startswith("-") else f for f in feats])]
      case "riscv64": self.args = ["-march=" + "_".join(["rv64g" if cpu == "native" else cpu] + feats)]
      case _: raise RuntimeError(f"unsupported arch: {self.arch!r}")
    super().__init__(f"{cachekey}_{'_'.join(arch)}")

  def compile(self, src:str) -> bytes:
    """Compile C source to ELF object file (before linking)."""
    # -fno-math-errno is required for __builtin_sqrt to become an instruction instead of a function call
    return subprocess.check_output([getenv("CC", 'clang'), '-c', '-x', 'c', '-O2', '-fPIC', '-ffreestanding', '-fno-math-errno', '-nostdlib',
                                    '-fno-ident', f'--target={self.arch}-none-unknown-elf', *self.args, '-', '-o', '-'], input=src.encode('utf-8'))

  def disassemble(self, lib: bytes): cpu_objdump(lib)

class TCCCompiler(Compiler):
  def __init__(self, arch:list[str], cachekey="compile_tcc_obj"):
    if shutil.which("tcc") is None: raise RuntimeError("tcc not found in PATH")
    self.arch, cpu, *feats = arch
    self.args = ()
    super().__init__(f"{cachekey}_{'_'.join(arch)}")

  def compile(self, src:str) -> bytes:
    fd, fn = tempfile.mkstemp(suffix=".o")
    os.close(fd)
    try:
      subprocess.check_output(["tcc", "-c", "-x", "c", "-", "-o", fn], input=src.encode('utf-8'))
      with open(fn, "rb") as f: return f.read()
    finally: os.unlink(fn)

  def disassemble(self, lib: bytes): cpu_objdump(lib)

class X86Compiler(Compiler):
  def __init__(self): super().__init__(None)
  def compile(self, src:str) -> bytes: return bytes.fromhex(src)
  def disassemble(self, lib:bytes): return capstone_flatdump(lib, "x86_64")
