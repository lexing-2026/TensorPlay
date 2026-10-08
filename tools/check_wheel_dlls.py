#!/usr/bin/env python3
"""Fail when a Windows wheel needs a DLL it neither ships nor finds in Windows.

Python resolves an extension module's dependencies from the directories the
package registers with ``os.add_dll_directory`` (``tensorplay/lib``) and from
System32, never from PATH.  A DLL that the build links against through its
import library but that the wheel does not bundle therefore links fine on the
builder, where the toolkit is installed, and fails every import elsewhere
with "DLL load failed while importing _C", which names no DLL.  The CUDA
DLLs are bundled from a hand-kept pattern list in CMakeLists.txt, so a new
link (nvjpeg for the image IO bindings, for one) has to be added there too.

This reads the load-time import table of every DLL and extension module in
the wheel and reports each imported DLL that is not in the wheel, not an API
set, not the interpreter's own DLL, and not present in the system directory.
Delay-loaded imports are skipped: they resolve on first use, behind the
code's own error handling.

Usage:
    python tools/check_wheel_dlls.py dist/tensorplay-*-win_amd64.whl
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import sys
import zipfile
from pathlib import Path

# API sets are virtual names the loader maps to system DLLs.
_API_SET = re.compile(r"^(api|ext)-ms-", re.IGNORECASE)
# The interpreter's own DLL; it is loaded before any extension module.
_PYTHON_DLL = re.compile(r"^python3\d*\.dll$", re.IGNORECASE)


def _imported_dlls(image: bytes) -> list[str]:
    """Names in the load-time import directory of a PE image."""
    pe = struct.unpack_from("<I", image, 0x3C)[0]
    if image[pe:pe + 4] != b"PE\0\0":
        raise ValueError("not a PE image")
    num_sections, = struct.unpack_from("<H", image, pe + 6)
    optional_size, = struct.unpack_from("<H", image, pe + 20)
    optional = pe + 24
    magic, = struct.unpack_from("<H", image, optional)
    data_dirs = optional + (112 if magic == 0x20B else 96)
    sections = []
    for i in range(num_sections):
        header = optional + optional_size + 40 * i
        virtual_size, virtual_address = struct.unpack_from("<II", image, header + 8)
        raw_size, raw_offset = struct.unpack_from("<II", image, header + 16)
        sections.append((virtual_address, max(virtual_size, raw_size), raw_offset))

    def offset(rva: int) -> int:
        for address, size, raw in sections:
            if address <= rva < address + size:
                return rva - address + raw
        raise ValueError(f"RVA {rva:#x} is outside every section")

    import_rva, = struct.unpack_from("<I", image, data_dirs + 8)
    names = []
    if import_rva:
        descriptor = offset(import_rva)
        while True:
            name_rva, = struct.unpack_from("<I", image, descriptor + 12)
            if not name_rva:
                break
            start = offset(name_rva)
            names.append(image[start:image.index(b"\0", start)].decode("ascii"))
            descriptor += 20
    return names


def _missing_imports(wheel: Path, system_dir: Path) -> dict[str, list[str]]:
    with zipfile.ZipFile(wheel) as archive:
        binaries = [n for n in archive.namelist() if n.lower().endswith((".dll", ".pyd"))]
        shipped = {Path(n).name.lower() for n in binaries}
        missing: dict[str, list[str]] = {}
        for name in binaries:
            for dll in _imported_dlls(archive.read(name)):
                if (dll.lower() in shipped or _API_SET.match(dll)
                        or _PYTHON_DLL.match(dll) or (system_dir / dll).exists()):
                    continue
                missing.setdefault(dll, []).append(name)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument(
        "--system-dir", type=Path,
        default=Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32",
        help="directory whose DLLs every Windows machine provides")
    args = parser.parse_args()
    failed = False
    for wheel in args.wheels:
        missing = _missing_imports(wheel, args.system_dir)
        if not missing:
            print(f"{wheel.name}: every imported DLL ships in the wheel or with Windows")
            continue
        failed = True
        print(f"{wheel.name}: imports DLLs it does not ship:", file=sys.stderr)
        for dll, importers in sorted(missing.items()):
            print(f"  {dll}  <- {', '.join(sorted(importers))}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
