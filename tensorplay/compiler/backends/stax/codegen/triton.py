"""Writing a launch for the kernel-writing runtime.

A launch written here is not a whole kernel: it is the part of one that is the
same whatever the kernel computes.  Which runtime built it, which machine it
is for, and which names the runtime reads as its own are all settled before a
single line of the kernel's own arithmetic exists, because all three decide
what the generated module has to import and what has to be written into the
record the runtime reads back.

The base class arrives with the shared kernel machinery.  Until it does, the
two things here -- what a generated module imports, and what is recorded about
how it was written -- are whole without it, and are read by the wrapper that
writes the module.  The import is resolved when this module is loaded, so the
base is picked up as soon as it exists rather than having to be written in
twice.
"""

from __future__ import annotations

from typing import Any

from .. import config
from ..utils import IndentedBuffer


try:  # The shared kernel machinery this builds on.
    from .simd import SIMDKernel
except ImportError:  # pragma: no cover - until that machinery is here
    SIMDKernel = object


class TritonKernel(SIMDKernel):  # type: ignore[misc,valid-type]
    """A launch written for the kernel-writing runtime.

    Only the parts that are the same for every such launch live here.  What the
    launch computes is the base class's business, along with the shared
    machinery for reading a value, computing one, and keeping the result.
    """

    @classmethod
    def gen_common_triton_imports(cls) -> str:
        """The imports every generated launch needs, as source to splice in.

        A generated module is a module of its own, so it cannot borrow the
        names of the module that wrote it: the runtime, the language it is
        written in, and the device library have to be imported by name.  The
        device library is imported under a second name because the generated
        arithmetic says that name, and the runtime's own helpers are imported
        under theirs because the generated launcher says theirs.

        Anything only some launches need -- a tracing profiler, a set of
        hardware descriptors -- is left out here and written by the code that
        knows the launch needs it, because a module that imports something it
        never uses pays for it on every launch.
        """

        imports = IndentedBuffer()
        imports.splice(
            """
            import triton
            import triton.language as tl
            """
        )
        try:
            import triton.language.extra.cuda.libdevice as libdevice  # noqa: F401

            imports.splice(
                """
                import triton.language.extra.cuda.libdevice as libdevice
                """
            )
        except ImportError:
            pass
        try:
            import triton.language.extra.tlx  # noqa: F401

            imports.splice(
                """
                import triton.language.extra.tlx as tlx  # noqa: F401
                """
            )
        except ImportError:
            pass
        imports.splice(
            """
            from tensorplay.compiler.backends.stax.runtime import (
                triton_helpers,
                triton_heuristics,
            )
            from tensorplay.compiler.backends.stax.runtime.triton_helpers import (
                libdevice,
                math as tl_math,
            )
            from tensorplay.compiler.backends.stax.runtime.hints import (
                AutotuneHint,
                DeviceProperties,
                ReductionHint,
                TileHint,
            )
            """
        )
        if config.triton.proton_profiling:
            imports.splice(
                """
                import triton.profiler as proton
                import triton.profiler.language as pl
                pl.enable_semantic('triton')
                """
            )
        return imports.getvalue()

    @classmethod
    def triton_meta_common(cls) -> dict[str, Any]:
        """What the runtime is told about how to treat the launch itself.

        These are the runtime's own switches rather than ours: whether it may
        fold the arithmetic of the launch, whether it may launch a dependent
        launch early, and whether it flushes denormals to zero.  The last is
        off, because a denormal flushed to zero is a different answer rather
        than a faster one, and a caller asking for exact arithmetic has not
        asked for that.
        """

        return {
            "enable_fp_fusion": not config.emulate_precision_casts,
            "launch_pdl": False,
            "disable_ftz": False,
        }

    @classmethod
    def inductor_meta_common(cls) -> dict[str, Any]:
        """What is recorded about the settings this launch was written under.

        A launch is cached, and a cache entry is only good for the settings it
        was written under -- a launch written to check an index is not
        interchangeable with one written not to, and a launch written to be
        deterministic is not interchangeable with one that was not.  So every
        setting that changes the generated code is written into the record, and
        the runtime compares it before reusing an entry.

        The runtime and the machine are in the record too, and they are asked
        of the runtime rather than assumed, since a launch built by one runtime
        for one machine is not readable by another.
        """

        from tensorplay.utils._triton import triton_hash_with_backend

        inductor_meta = {
            "backend_hash": triton_hash_with_backend(),
            "assert_indirect_indexing": config.assert_indirect_indexing,
            "max_autotune": config.max_autotune,
            "deterministic": config.deterministic,
            "emulate_precision_casts": config.emulate_precision_casts,
            "force_disable_caches": config.force_disable_caches,
            "store_cubin": config.triton.store_cubin,
            "force_filter_reduction_configs": (
                config.test_configs.force_filter_reduction_configs
            ),
        }

        if config.profile_bandwidth:
            inductor_meta["profile_bandwidth"] = config.profile_bandwidth
            inductor_meta["profile_bandwidth_output"] = config.profile_bandwidth_output
            inductor_meta["profile_bandwidth_with_do_bench_using_profiling"] = (
                config.profile_bandwidth_with_do_bench_using_profiling
            )

        return inductor_meta
