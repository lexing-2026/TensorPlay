"""CPU SIMD capability selection for runtime code generation.

A ``VecISA`` instance bundles the build macros, compiler architecture
flags, and per-dtype lane counts of one SIMD tier.  The host toolchain is
verified with a dry compile + dlopen probe: a CPU that reports AVX2/AVX512
cannot use the tier unless the system compiler can actually produce a
working shared object for it.  The probe verdict is persisted next to the
kernel cache and keyed by a fingerprint of (compiler version, ISA flags,
package version), so an upgraded toolchain re-probes instead of inheriting
a stale verdict.

``pick_vec_isa()`` selects the widest supported tier.  ``TP_STAX_CPU_TIER``
overrides the choice with a tier name (``avx512``/``avx2``/``default``),
mirroring the ``ATEN_CPU_CAPABILITY`` override convention used by the
in-tree kernels.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
from typing import Any, Optional

from .codecache import default_cache

# Lane counts per element width for each SIMD tier (used by codegen to size
# the per-chunk element tile).
_NELEMENTS_F32 = {"avx512": 16, "avx2": 8, "default": 4}


class VecISA:
    """Description of one SIMD tier and its toolchain feasibility."""

    name = "invalid"
    bit_width = 0
    macros: tuple[str, ...] = ()
    arch_flags: tuple[str, ...] = ()

    def __init__(self, runtime_dirs: Optional[tuple[str, str, str]] = None) -> None:
        # ``(include_dir, generated_include_dir, lib_dir)`` from
        # :func:`cpp_builder.package_paths`; ``None`` defers resolution to
        # probe time so standalone instances still work.
        self._dirs = runtime_dirs
        self._feasible: Optional[bool] = None

    def nelements(self) -> int:
        return _NELEMENTS_F32.get(self.name, 4)

    def definitions(self) -> list[str]:
        macros = list(self.macros)
        macros.append(f"CPU_CAPABILITY={self.name.upper()}")
        return [f"-D{macro}" for macro in macros]

    def build_arch_flags(self) -> list[str]:
        return list(self.arch_flags)

    def __str__(self) -> str:
        return self.name

    def __bool__(self) -> bool:
        return self.name != "invalid" and self.is_feasible()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"VecISA({self.name})"

    # -- feasibility probe ---------------------------------------------------

    def fingerprint(self) -> str:
        from .cpp_builder import get_compiler_version_info, get_cpp_compiler

        return "|".join(
            (
                "stax-cpu-isa-v1",
                self.name,
                get_compiler_version_info(get_cpp_compiler()),
                _package_version(),
            )
        )

    def is_feasible(self) -> bool:
        """Dry-compile a vector probe and dlopen the result.

        The verdict is cached per process and persisted next to the kernel
        cache keyed by the toolchain fingerprint, so the roughly one-second
        probe runs once per (compiler, flags, package) combination.
        """

        if self._feasible is not None:
            return self._feasible
        self._feasible = self._probe_impl()
        return self._feasible

    def _probe_impl(self) -> bool:
        if self._dirs is None:
            from .cpp_builder import package_paths

            self._dirs = package_paths()
        if self.name == "invalid" or self._dirs is None:
            return False
        include_dir, generated_include_dir, lib_dir = self._dirs
        try:
            from .cpp_builder import get_cpp_compiler

            compiler = get_cpp_compiler()
        except Exception:
            return False
        if not compiler:
            return False

        cache = default_cache("stax-cpu-isa")
        key = cache.cache_key(
            _PROBE_SOURCE,
            entry=f"probe_{self.name}",
            options={"fp": self.fingerprint()},
        )
        marker = cache.path_for(key, "load_ok")
        if os.path.exists(marker):
            return True

        from .codecache import file_lock

        source_path = cache.path_for(key, "cpp")
        output_path = cache.path_for(key, "so")
        try:
            os.makedirs(os.path.dirname(source_path), exist_ok=True)
            if not os.path.exists(output_path):
                with file_lock(output_path + ".lock"):
                    if not os.path.exists(output_path):
                        with open(source_path, "w") as fh:
                            fh.write(_PROBE_SOURCE)
                        from .cpp_builder import CppBuilder, CppOptions

                        # The vector transcendentals resolve to SLEEF
                        # symbols exported by the runtime library, so the
                        # probe must link against it like real kernels do.
                        options = CppOptions(
                            compiler=compiler,
                            include_dirs=[include_dir, generated_include_dir],
                            cflags=[
                                "-std=c++20",
                                "-O1",
                                "-fPIC",
                                "-shared",
                                *self.build_arch_flags(),
                            ],
                            definitions=self.definitions(),
                            library_dirs=[lib_dir],
                            # ``tpx`` is a header-only interface target: its
                            # ops namespace is compiled into libp10, so the
                            # probe must link libp10 only — naming libtpx
                            # fails the build and every JIT route with it.
                            libraries=["p10"],
                            ldflags=[f"-Wl,-rpath,{lib_dir}"],
                        )
                        builder = CppBuilder(
                            name=os.path.basename(output_path),
                            sources=[source_path],
                            options=options,
                            output_dir=os.path.dirname(source_path),
                        )
                        builder.build()
        except Exception:
            return False

        load_ok = _dlopen_probe(output_path)
        if load_ok:
            try:
                with open(marker, "w") as fh:
                    fh.write("ok\n")
            except OSError:
                pass
        return load_ok


def _dlopen_probe(output_path: str) -> bool:
    """Load the probe artifact in this process and run its entry point."""

    try:
        lib = ctypes.CDLL(output_path)
        entry = getattr(lib, "tp_isa_probe_entry")
        entry.restype = ctypes.c_int
        entry.argtypes = []
        return int(entry()) == 0
    except Exception:
        return False


_PROBE_SOURCE = """
#include "cpu/vec/vec.h"
using V = tensorplay::vec::Vectorized<float>;

alignas(64) float tp_isa_input[64] = {0};

extern "C" int tp_isa_probe_entry() {
    V a = V::loadu(tp_isa_input, V::size());
    V b = tensorplay::vec::maximum(a, V(0.0f)) + a.exp();
    V m = (a > V(0.0f));
    V c = V::blendv(a, b, m);
    __attribute__((aligned(64))) float out[64];
    c.store(out, V::size());
    return (out[0] == out[0]) ? 0 : 1;
}
"""


def _package_version() -> str:
    try:
        from ....version import __version__ as pkg_version

        return str(pkg_version)
    except Exception:
        return "unknown"


class VecAVX512(VecISA):
    name = "avx512"
    bit_width = 512
    macros = (
        "CPU_CAPABILITY_AVX512",
        "HAVE_AVX512_CPU_DEFINITION",
        "HAVE_AVX2_CPU_DEFINITION",
    )
    arch_flags = ("-mavx512f", "-mavx512dq", "-mavx512vl", "-mavx512bw", "-mfma")


class VecAVX2(VecISA):
    name = "avx2"
    bit_width = 256
    macros = (
        "CPU_CAPABILITY_AVX2",
        "HAVE_AVX2_CPU_DEFINITION",
    )
    arch_flags = ("-mavx2", "-mfma", "-mf16c")


class VecDefault(VecISA):
    name = "default"
    bit_width = 256
    macros = ("CPU_CAPABILITY_DEFAULT", "HAVE_AVX2_CPU_DEFINITION")
    arch_flags = ()


class InvalidVecISA(VecISA):
    name = "invalid"
    macros = ()
    arch_flags = ()

    def is_feasible(self) -> bool:
        return False


def _cpu_has(flags: set[str], *needed: str) -> bool:
    return set(needed).issubset(flags)


def _host_isa_flags() -> set[str]:
    flags: set[str] = set()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    break
    except OSError:
        pass
    if not flags:
        try:
            import tensorplay

            probe = getattr(tensorplay._C, "_stax_cpu_capability", None)
            if callable(probe):
                caps = int(probe())
                if caps >= 2:
                    flags.update(("avx2", "avx512f"))
                elif caps == 1:
                    flags.add("avx2")
        except Exception:
            pass
    return flags


def valid_vec_isa_list(runtime_dirs: Optional[tuple[str, str, str]] = None) -> list[VecISA]:
    flags = _host_isa_flags()
    candidates: list[VecISA] = []
    if _cpu_has(flags, "avx512f", "avx512dq", "avx512vl", "avx512bw"):
        candidates.append(VecAVX512(runtime_dirs))
    if _cpu_has(flags, "avx2"):
        candidates.append(VecAVX2(runtime_dirs))
    candidates.append(VecDefault(runtime_dirs))
    return candidates


def pick_vec_isa(runtime_dirs: Optional[tuple[str, str, str]] = None) -> VecISA:
    """Return the widest toolchain-verified SIMD tier for the host."""

    if runtime_dirs is None:
        from .cpp_builder import package_paths

        runtime_dirs = package_paths()
    override = os.environ.get("TP_STAX_CPU_TIER", "").strip().lower()
    wanted = {
        "avx512": VecAVX512,
        "avx2": VecAVX2,
        "default": VecDefault,
        "invalid": InvalidVecISA,
    }.get(override)
    if wanted is not None:
        return wanted(runtime_dirs)
    for isa in valid_vec_isa_list(runtime_dirs):
        if isa.is_feasible():
            return isa
    return InvalidVecISA(runtime_dirs)
