#!/usr/bin/env python3
"""Fail when an x86-64 library shares a weak function built for an ISA tier.

The AVX2/AVX512 copies of the dispatched CPU kernels are compiled with their
ISA flags for the whole TU, so every inline function or template
instantiation they emit out of line is a weak definition that may use those
instructions.  The linker keeps one definition per weak symbol for the whole
library; when it keeps a tier copy, code outside the dispatcher runs AVX512
on CPUs without it and dies with SIGILL (see Note [Linking ISA-tier kernel
copies] in p10/CMakeLists.txt).

A weak function may legitimately carry AVX/AVX512 code when it belongs to a
CPU_CAPABILITY tier namespace (``tensorplay::cpu::AVX512::...``) or is an
ISA-specific helper whose name says so (``row_max_f32_512``,
``dot_f32_avx2``): those only run behind a runtime ISA check.  Any other
weak function with such code is a leaked tier copy.

Usage:
    python tools/check_isa_tier_leaks.py tensorplay/lib/libp10.so
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_FUNCTION_HEADER = re.compile(r"^[0-9a-f]+ <(.+)>:$")
# EVEX-only operands: zmm, mask registers, the upper 16 xmm/ymm registers,
# and broadcasts from a general-purpose register.
_AVX512_OPERAND = re.compile(r"%zmm|%k[0-7]\b|%[xy]mm(?:1[6-9]|2[0-9]|3[01])\b|vpbroadcast[bwdq]\s+%[re]")
# Any VEX/EVEX-encoded instruction; baseline x86-64 code uses none.
_AVX_MNEMONIC = re.compile(r"^\s*[0-9a-f]+:\s+v[a-z0-9]+\s")
# Names that declare the ISA they were built for.
_ISA_NAMED = re.compile(r"AVX2|AVX512|avx2|avx512|_256\b|_512\b")


def _weak_symbols(lib: Path) -> set[str]:
    out = subprocess.run(["nm", "--defined-only", str(lib)], capture_output=True, text=True, check=True).stdout
    weak = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == "W":
            weak.add(parts[2])
    return weak


def _isa_functions(lib: Path, weak: set[str]) -> dict[str, str]:
    """Map each weak function that executes AVX or AVX512 to that ISA."""
    found: dict[str, str] = {}
    current = None
    proc = subprocess.Popen(
        ["objdump", "-d", "--no-show-raw-insn", "-j", ".text", str(lib)],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1 << 20,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        header = _FUNCTION_HEADER.match(line)
        if header:
            current = header.group(1) if header.group(1) in weak else None
        elif current is not None and found.get(current) != "AVX512":
            if _AVX512_OPERAND.search(line):
                found[current] = "AVX512"
            elif _AVX_MNEMONIC.match(line):
                found[current] = "AVX"
    if proc.wait() != 0:
        raise RuntimeError(f"objdump failed on {lib}")
    return found


def _demangle(names: list[str]) -> list[str]:
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True, check=True).stdout
    return out.splitlines()


def find_leaks(lib: Path) -> list[tuple[str, str]]:
    found = _isa_functions(lib, _weak_symbols(lib))
    names = sorted(found)
    return [(found[name], pretty) for name, pretty in zip(names, _demangle(names)) if not _ISA_NAMED.search(pretty)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("libs", nargs="+", type=Path, help="x86-64 ELF libraries")
    args = parser.parse_args()
    status = 0
    for lib in args.libs:
        leaks = find_leaks(lib)
        if not leaks:
            print(f"{lib}: no weak function carries ISA-tier code")
            continue
        status = 1
        print(f"{lib}: {len(leaks)} weak function(s) carry ISA-tier code:")
        for isa, name in leaks:
            print(f"  {isa:6s} {name}")
    return status


if __name__ == "__main__":
    sys.exit(main())
