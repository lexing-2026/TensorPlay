from __future__ import annotations

"""Measuring a candidate kernel, in a process of its own when it may crash.

A candidate is measured by building the tensors its metadata describes,
running it, and reading the time.  A candidate that reads a tensor wrongly, or
one whose code does not compile, does not raise -- it corrupts memory or
aborts the process.  So the request describing what to measure is built in the
parent, sent to a child, and answered with a number; the child may die, and
the number it did not send is the answer that the candidate does not run.

Nothing here holds a tensor.  A request crosses a process boundary, and what
crosses is the geometry of what the kernel reads and writes, never the data:
the child allocates its own tensors from that geometry, so two measurements of
the same candidate are of the same work even though the data differs.
"""

import tensorplay as tp


from . import config
from .compile_worker.timer import Timer
from .utils import apply_subprocess_env, clear_caches
import atexit
import ctypes
import functools
import logging
import multiprocessing as mp
import os
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

from .kernel_cache import default_cache

log = logging.getLogger(__name__)

__all__ = [
    "BenchmarkRequest",
    "CPUDeviceBenchmarkMixin",
    "CppBenchmarkRequest",
    "NonzeroWorkspaceNotSupportedError",
    "TensorMeta",
]


def _compile_cpp(source: str):
    """Build a host kernel from source text and return the loaded library.

    The artifact is a shared library, because that is what a candidate has to
    be to be called with the addresses of the tensors it reads: a call into a
    loaded library is the only way to reach a kernel that was just written.
    """

    import ctypes
    import os
    import tempfile

    from .cpp_builder import CppBuilder, CppOptions, package_paths

    _, include_dir, lib_dir = package_paths()
    options = CppOptions(
        include_dirs=[include_dir],
        cflags=["-std=c++20", "-O3", "-fPIC", "-shared"],
        definitions=[],
        library_dirs=[lib_dir],
        libraries=["p10"],
        ldflags=[f"-Wl,-rpath,{lib_dir}"],
    )
    with tempfile.TemporaryDirectory(prefix="tp_cpp_bench_") as workdir:
        source_path = os.path.join(workdir, "kernel.cpp")
        with open(source_path, "w") as fh:
            fh.write(source)
        builder = CppBuilder(
            name="kernel.so",
            sources=[source_path],
            options=options,
            output_dir=workdir,
        )
        builder.build()
        return ctypes.CDLL(builder.get_target_file_path())


class NonzeroWorkspaceNotSupportedError(Exception):
    """A candidate that needs scratch space cannot be measured this way."""


def rand_strided(
    sizes,
    strides,
    *,
    device,
    dtype,
    extra_size: int = 0,
):
    """A tensor of the given geometry, filled with arbitrary values.

    The values are arbitrary because what is being measured is the kernel, not
    the data: a kernel that branches on its contents would be measured on the
    branch it happened to take, which is a worse answer than measuring it on
    every branch and is not available at all for values it did not see.
    """

    from tensorplay import functional

    n = 1 + extra_size
    for extent, stride in zip(sizes, strides):
        n += (int(extent) - 1) * int(stride)
    return functional.empty_strided(
        [int(x) for x in sizes],
        [int(x) for x in strides],
        dtype=dtype,
        device=device,
    )


@dataclass
class TensorMeta:
    """The geometry of a tensor, and the name it was known by.

    A measurement needs the geometry and nothing else, so this is deliberately
    not a tensor: it can be built where the tensor does not exist and sent
    across a process boundary.
    """

    device: Any
    dtype: Any
    sizes: tuple
    strides: tuple
    offset: int
    name: str | None = None

    @classmethod
    def from_irnodes(cls, irnodes):
        """The metadata of a result the region described, or of several of them."""

        if isinstance(irnodes, Sequence):
            result: list[Any] = [cls.from_irnodes(x) for x in irnodes]
            if not all(isinstance(x, TensorMeta) for x in result):
                raise AssertionError(
                    f"Expected all elements to be TensorMeta, got types: "
                    f"{[type(x) for x in result if not isinstance(x, TensorMeta)]}"
                )
            return result

        node = irnodes
        from .ir import Buffer, Layout
        from .loops import V

        if isinstance(node, Layout):
            node = Buffer("fake", node)

        dtype = node.get_dtype()
        if dtype is None:
            raise AssertionError(
                f"Expected node to have a dtype, but get_dtype() returned None for node '{node}'"
            )
        device = node.get_device()
        if device is None:
            raise AssertionError(
                f"Expected node to have a device, but get_device() returned None for node '{node}'"
            )

        return TensorMeta(
            device=device,
            dtype=dtype,
            sizes=V.graph.sizevars.optimization_hints(node.get_size()),
            strides=V.graph.sizevars.optimization_hints(node.layout.stride),
            offset=V.graph.sizevars.optimization_hint(node.layout.offset),
            name=node.name,
        )

    def to_tensor(self):
        """A tensor of this geometry, filled with arbitrary values."""

        return rand_strided(
            self.sizes,
            self.strides,
            device=self.device,
            dtype=self.dtype,
            extra_size=self.offset,
        )


class BenchmarkRequest:
    """One candidate, described well enough to be measured somewhere else.

    A request has to survive being sent to another process, so it holds the
    kernel's name, the geometry of what it reads and writes, and its extra
    arguments -- and nothing that is not one of those.
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: "TensorMeta | list[TensorMeta]",
        output_tensor_meta: "TensorMeta | list[TensorMeta]",
        extra_args: Iterable[Any],
    ) -> None:
        self.kernel_name = kernel_name

        if isinstance(input_tensor_meta, TensorMeta):
            self.input_tensor_meta: list[TensorMeta] = [input_tensor_meta]
        else:
            self.input_tensor_meta: list[TensorMeta] = input_tensor_meta

        if output_tensor_meta and isinstance(output_tensor_meta, (tuple, list)):
            if len(output_tensor_meta) > 1:
                # Several results with one description is a grouped product, and
                # they are written by the same kernel, so they have to be the
                # same geometry or the kernel is being asked two questions.
                if not all(
                    getattr(output_tensor_meta[0], attr) == getattr(x, attr)
                    for x in output_tensor_meta
                    for attr in ["device", "dtype", "sizes", "strides", "offset"]
                ):
                    raise AssertionError(
                        "All output tensor metas in a Grouped GEMM must have matching "
                        "device, dtype, sizes, strides, and offset"
                    )
            self.output_tensor_meta = output_tensor_meta[0]
        else:
            self.output_tensor_meta: TensorMeta = output_tensor_meta

        self.extra_args = extra_args
        self.benchmark_with_cudagraphs = False

    def make_run_fn(self, *input_tensors, out):
        """The closure that runs the candidate once it is built."""

        raise NotImplementedError

    def cleanup_run_fn(self) -> None:
        pass

    def do_bench(self, fn, *input_tensors, out=None) -> float:
        raise NotImplementedError

    def benchmark(self, *input_tensors, out=None) -> float:
        """How long the candidate takes, in the units the machine is timed in.

        The tensors are built here rather than being passed in, so that the
        measurement is of the kernel rather than of a caller that already had
        them, and so that a candidate that fails on the tensors it is handed
        fails on tensors built the same way every time.
        """

        debug = log.isEnabledFor(logging.DEBUG)
        if debug:
            start_ts = time.time()

        if out is None:
            if self.input_tensor_meta is None or not isinstance(
                self.output_tensor_meta, TensorMeta
            ):
                raise AssertionError(
                    "Input and output tensor meta must be populated when out is None"
                )
            if not len(input_tensors) == 0:
                raise AssertionError(
                    f"Expected no input_tensors when out is None, but got {len(input_tensors)}"
                )
            input_tensors = tuple(x.to_tensor() for x in self.input_tensor_meta)
            out = self.output_tensor_meta.to_tensor()

        if debug:
            create_tensor_elapse = time.time() - start_ts
            start_ts = time.time()
        try:
            try:
                fn = self.make_run_fn(*input_tensors, out=out)
            except NonzeroWorkspaceNotSupportedError:
                # A candidate that needs scratch space is not measured here;
                # the time reported is one no candidate can beat, so it is
                # never chosen and does not have to be understood.
                log.info("Skipping op due to nonzero workspace requirement")
                return float("inf")

            if debug:
                load_elapse = time.time() - start_ts
                start_ts = time.time()

            res = self.do_bench(fn, *input_tensors, out)

            if debug:
                bench_elapse = time.time() - start_ts
                log.debug(
                    "InChildProcess %s: load %f, create tensor %f, bench %f",
                    self,
                    load_elapse,
                    create_tensor_elapse,
                    bench_elapse,
                )
            return res
        finally:
            self.cleanup_run_fn()


class CPUDeviceBenchmarkMixin:
    """How a candidate is timed when it runs on the processor."""

    def do_bench(self, fn, *input_tensors, out=None) -> float:
        """The median of a few runs, which is what a single run cannot be.

        A processor shares its units with everything else on the machine, so a
        single run measures the machine as much as the kernel; the median of
        several is the one number that survives a neighbour arriving once.
        """

        import time as _time

        run_times: list[float] = []
        for _ in range(3):
            start = _time.perf_counter()
            fn()
            run_times.append(_time.perf_counter() - start)
        run_times.sort()
        return run_times[len(run_times) // 2]


class GPUDeviceBenchmarkMixin:
    """How a candidate is timed when it runs on an accelerator.

    Timed on the device it will run on, and not moved there first: a candidate
    measured on one device and used on another was measured as something else.
    The device is taken from the values rather than asked for, because the values
    are what the candidate will actually be handed.
    """

    def do_bench(
        self,
        fn,
        *input_tensors,
        out=None,
    ) -> float:
        import tensorplay as tp

        from .runtime.benchmarking import benchmarker, get_interface_for_device

        # One device, or the measurement is of nothing: a candidate that reads
        # from two devices is two candidates, and timing them together times the
        # transfer as much as the work.
        indices = {
            tensor.device.index
            for tensor in [*input_tensors, out]
            if isinstance(tensor, tp.Tensor) and _is_gpu_device(tensor.device)
            and tensor.device.index is not None
        }
        if len(indices) > 1:
            raise AssertionError(f"Can not mix devices {sorted(indices)}")
        device_type = next(
            (
                tensor.device.type
                for tensor in input_tensors
                if _is_gpu_device(tensor.device)
            ),
            "cuda",
        )
        device_interface = get_interface_for_device(device_type)
        device_idx = (
            next(iter(indices))
            if len(indices) == 1
            else device_interface.current_device()
        )
        with device_interface.device(device_idx):
            result = benchmarker.benchmark(fn, device=device_type)
            # Waiting here rather than at the end of the measurement turns a
            # failure inside the candidate into a failure here, where the
            # candidate is still known, instead of into one several measurements
            # later attributed to whichever candidate ran next.
            device_interface.synchronize()
        return result


def _is_gpu_device(device) -> bool:
    """Whether a device is one whose work is timed rather than waited for."""

    return getattr(device, "type", None) in {"cuda", "xpu", "mtia"}


class SubgraphBenchmarkRequest(BenchmarkRequest):
    """A candidate that is a whole subgraph, already built in this process.

    A subgraph is compiled here rather than where it is measured, because
    compiling it is most of what it costs and the measurement is meant to be of
    running it.  So what travels is not the subgraph but where it was written and
    the key it was written under, which is enough to load the same one and
    nothing else.
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta,
        output_tensor_meta,
        extra_args,
        module_path: str,
        module_cache_key: str,
        sym_input_values: list[int],
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.module_path = module_path
        self.module_cache_key = module_cache_key
        self.sym_input_values = list(sym_input_values)

    def make_run_fn(self, *input_tensors, out=None):
        """A closure that runs the subgraph on the values it was handed.

        The sizes that were only known when it was built come first, because a
        built subgraph is entered by position and those sizes are its arguments
        as much as the values are.
        """

        from ..codecache import load_by_key_path

        module = load_by_key_path(
            self.module_cache_key, self.module_path, set_sys_modules=False
        )
        sym_values = self.sym_input_values
        # A fresh list each time: the call consumes the list it is given, so a
        # shared one would be empty the second time round.
        return lambda: module.call([*sym_values, *input_tensors])

    def precompile(self) -> None:
        """Load the built subgraph now, so that loading is not timed.

        Nothing is built here -- it was built in the process that chose it -- but
        loading maps a file, and a measurement that included a mapping would be
        measuring the file system.
        """

        from ..codecache import load_by_key_path

        load_by_key_path(
            self.module_cache_key, self.module_path, set_sys_modules=False
        )

    def __str__(self) -> str:
        return (
            f"SubgraphBenchmarkRequest({self.kernel_name}, {self.module_path})"
        )


class SubgraphGPUBenchmarkRequest(GPUDeviceBenchmarkMixin, SubgraphBenchmarkRequest):
    """A built subgraph, measured on an accelerator."""


class SubgraphCPUBenchmarkRequest(CPUDeviceBenchmarkMixin, SubgraphBenchmarkRequest):
    """A built subgraph, measured on the processor."""



class CppBenchmarkRequest(CPUDeviceBenchmarkMixin, BenchmarkRequest):
    """A candidate whose body is source text compiled on the measuring host.

    The source travels with the request rather than being looked up, because
    the process that measures has not necessarily seen the module that emitted
    it, and because a source that has been edited since it was cached has to be
    measured as the new text rather than as whatever was compiled before.
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: "TensorMeta | list[TensorMeta]",
        output_tensor_meta: "TensorMeta | list[TensorMeta]",
        extra_args: Iterable[Any],
        source_code: str,
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.source_code = source_code
        self.cache = default_cache("cpp")
        self.hash_key = self.cache.cache_key(source_code)
        self.DLL = None

    def precompile(self):
        """Build the candidate before it is timed, so the build is not timed."""

        log.debug("Precompiling %s", self)
        self._load()
        log.debug("Done precompiling %s", self)

    def _load(self):
        """The built candidate for this request's source, building it if needed."""

        return self.cache.compile_or_load(
            lambda source: _compile_cpp(source),
            self.source_code,
        )

    def make_run_fn(self, *input_tensors, out) -> Callable[[], None]:
        """The closure that calls the candidate with the tensors' addresses."""

        self.DLL = self._load()
        args = [tensor.data_ptr() for tensor in list(input_tensors) + [out]]
        log.debug(
            "make_run_fn: self.kernel_name=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.kernel_name,
            self.DLL,
            args,
            self.extra_args,
        )
        run_method = getattr(self.DLL, self.kernel_name)
        # An extra argument is a size, so it is a pointer-sized integer; a
        # caller that passed something else is being refused here rather than
        # being handed to the kernel as an address it will call.
        if not all(isinstance(arg, ctypes.c_ulonglong) for arg in self.extra_args):
            raise AssertionError(
                f"Expected all extra_args to be ctypes.c_ulonglong, got types: "
                f"{[type(arg) for arg in self.extra_args if not isinstance(arg, ctypes.c_ulonglong)]}"
            )
        run_method.argtypes = [ctypes.c_ulonglong] * (
            len(args) + len(list(self.extra_args))
        )

        return functools.partial(
            run_method,
            *args,
            *self.extra_args,
        )

    def __str__(self) -> str:
        return f"{self.kernel_name=}"



# ---------------------------------------------------------------------------
# the pool that measures kernels
# ---------------------------------------------------------------------------


#: How long a pool may sit with nothing to do before it stops itself.
#:
#: Measured candidates are expensive and a burst of them is followed by a long
#: quiet period, so a pool left over from a burst holds workers for nothing.
#: Zero turns the stopping off, which is what a machine that should keep its
#: workers between runs asks for.
AUTOTUNE_POOL_INACTIVITY_TIMEOUT = int(
    os.environ.get("TP_AUTOTUNE_POOL_INACTIVITY_TIMEOUT", "600")
)

autotuning_log = tp.getArtifactLogger(__name__, "autotuning")


def _cache_env_for_subprocess() -> dict[str, str | None]:
    env_vars = [
        "TP_CACHE_DIR",
        "TP_TRITON_CACHE_DIR",
        "TP_FLYDSL_RUNTIME_CACHE_DIR",
    ]
    return {v: os.environ.get(v) for v in env_vars}


# The environment last applied to this process, so that a change is applied
# once rather than at every task.  Compared whole: applying it again would be
# harmless, and emptying the caches again would not be.
_last_applied_cache_env: dict[str, str | None] | None = None


def _apply_subprocess_env_and_clear_caches(
    extra_env: dict[str, str | None] | None,
) -> None:
    global _last_applied_cache_env

    if extra_env is None:
        return

    if extra_env != _last_applied_cache_env:
        clear_caches()
        _last_applied_cache_env = extra_env.copy()
    apply_subprocess_env(extra_env)


def _init_autotune_subprocess(fp32_precision: Any) -> bool:
    """What a worker does before it is given anything to measure.

    Two things, both of which would otherwise be paid for by the first
    measurement rather than before it.  The context for the device has to
    exist, and creating it takes long enough to be worth doing while nothing
    is being timed.  And the setting that says whether single-precision
    matmul goes to the tensor cores has to be the one the parent measured
    under, or the measurement is of different work than the one the choice
    was about.
    """
    if tp.cuda.is_available():
        tp.zeros(1, device="cuda")

    tp.backends.cuda.matmul.allow_tf32 = bool(fp32_precision)

    return True


def _run_with_subprocess_env(
    fn: Callable[..., Any],
    extra_env: dict[str, str | None],
    *args: Any,
    **kwargs: Any,
) -> Any:
    _apply_subprocess_env_and_clear_caches(extra_env)
    return fn(*args, **kwargs)


class AutotuneProcessPool:
    """
    Singleton pool manager for running autotuning (precompilation + benchmarking)
    in a separate process.
    """

    _instance: AutotuneProcessPool | None = None
    _lock: threading.Lock = threading.Lock()
    _shutdown_for_inactivity: bool = False

    def __init__(self):
        self._pool: ProcessPoolExecutor | None = self._init_pool()
        self._warmup_future: Future[Any] | None = None
        self._warmup_start_time: float | None = None
        self._timer: Timer | None = self._init_timer()

    @classmethod
    def get_instance(cls):
        """Get or create the singleton pool instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # num_workers=1 to avoid GPU contention during benchmarking
                    cls._instance = cls()
        return cls._instance

    @property
    def pool(self):
        """Get the process pool."""
        if not config.pipeline_max_autotune_gemm:
            raise AssertionError(
                "To use AutotuneProcessPool, pipeline_max_autotune_gemm must be enabled"
            )
        if self._pool is None:
            self._pool = self._init_pool()
            self._timer = self._init_timer()
        return self._pool

    def _init_timer(self) -> Timer | None:
        if AUTOTUNE_POOL_INACTIVITY_TIMEOUT > 0:
            return Timer(AUTOTUNE_POOL_INACTIVITY_TIMEOUT, self._on_inactivity_timeout)
        return None

    def _record_activity(self) -> None:
        if self._timer is not None:
            self._timer.record_call()

    def _on_inactivity_timeout(self) -> None:
        autotuning_log.info(
            "AutotuneProcessPool shutting down due to inactivity (timeout=%ds)",
            AUTOTUNE_POOL_INACTIVITY_TIMEOUT,
        )

        with self._lock:
            if self._pool is not None:
                self._pool.shutdown(wait=False)
                self._pool = None
            self._timer = None

            # Mark that the pool was shut down for inactivity.
            # This prevents the pool from being recreated on recompiles
            # which likely do not require large amounts of autotuning.
            AutotuneProcessPool._shutdown_for_inactivity = True

    def _init_pool(self):
        """
        Get or create the process pool.

        Uses ProcessPoolExecutor with 'spawn' context for CUDA safety.
        ProcessPoolExecutor is lazily initialized - workers are not spawned
        until the first submit() call, making this property non-blocking.
        """
        # Use 'spawn' context to avoid CUDA fork issues
        # Workers are spawned lazily on first submit(), not here
        ctx = mp.get_context("spawn")
        pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=ctx,
        )
        atexit.register(self._shutdown)
        autotuning_log.info("AutotuneProcessPool created (workers spawn lazily)")

        return pool

    def warm_up(self) -> Future[Any]:
        """
        Submit a warmup job to eagerly spawn workers and initialize CUDA.

        This is optional - call it early to hide spawn latency.
        Returns the warmup future which can be ignored or awaited.
        """
        if self._warmup_future is None:
            with self._lock:
                if self._warmup_future is None:
                    self._warmup_start_time = time.perf_counter()
                    self._warmup_future = self.submit(
                        _init_autotune_subprocess,
                        fp32_precision=tp.backends.cuda.matmul.allow_tf32,
                    )
                    self._warmup_future.add_done_callback(self._on_warmup_complete)
                    autotuning_log.info("Warmup job submitted")
        # pyrefly: ignore[bad-return]
        return self._warmup_future

    def _on_warmup_complete(self, future: Future[Any]) -> None:
        """Callback invoked when the warmup job completes."""
        warmup_elapsed_time = None
        if self._warmup_start_time is not None:
            warmup_elapsed_time = time.perf_counter() - self._warmup_start_time

        try:
            result = future.result()
            autotuning_log.info(
                "AutotuneProcessPool warmup completed successfully in %.4f seconds: %s",
                warmup_elapsed_time,
                result,
            )
            self._record_activity()
        except Exception as e:
            autotuning_log.error(
                "AutotuneProcessPool warmup failed after %.4f seconds",
                warmup_elapsed_time,
            )
            raise e

    def submit(self, fn, *args, **kwargs) -> Future[Any]:
        """Submit a job to the pool and return a Future."""
        future = self.pool.submit(
            _run_with_subprocess_env,
            fn,
            _cache_env_for_subprocess(),
            *args,
            **kwargs,
        )
        if self._timer is not None:
            future.add_done_callback(lambda _: self._record_activity())
        return future

    def _shutdown(self):
        """Shutdown the pool on exit."""
        if self._timer is not None:
            self._timer.quit()
            self._timer = None
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    @classmethod
    def shutdown_instance(cls):
        """Explicitly shutdown the singleton instance."""
        if cls._instance is not None:
            with cls._lock:
                if cls._instance is not None:
                    cls._instance._shutdown()
                    cls._instance = None
