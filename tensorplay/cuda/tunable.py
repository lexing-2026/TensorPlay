# mypy: allow-untyped-defs
r"""Runtime tuning controls for CUDA GEMM kernels.

Matrix products dispatched through the cuBLASLt plan path can execute with
more than one library algorithm. For a given (dtype, shape, transpose,
bias-fusion) combination, the fastest algorithm depends on the GPU and the
library build, and the library heuristic's top estimate is not always the
measured winner. This module records, per GEMM signature, the algorithm
that measured fastest on the current hardware and replays that choice on
later runs instead of re-measuring.

Tuning is enabled separately from the feature itself:

* :func:`enable` turns the feature on. Every GEMM on the tunable path then
  first checks whether a winner is already recorded for its signature; if
  so, that algorithm runs immediately and no measurement happens.
* :func:`tuning_enable` controls what happens when no winner is recorded:
  with tuning on, every candidate algorithm is timed and the fastest is
  recorded; with tuning off, the library heuristic's top choice runs.

The results are persisted to a CSV file so a machine is tuned once and
reused across runs. New winners are appended to the file as they are
measured while tuning is on, and :func:`write_file` rewrites the file on
demand. :func:`read_file` merges an existing file into the in-memory
database. The file carries validator lines (library and device identity);
a file whose validators do not match the current build is rejected, since
its entries would no longer describe runnable choices.

GEMMs that run without a tuned choice can be logged to a separate untuned
file with :func:`record_untuned_enable`, which is useful for sizing how
much a workload would benefit from tuning.
"""

from tensorplay import _C

__all__ = [
    "enable",
    "disable",
    "is_enabled",
    "tuning_enable",
    "tuning_disable",
    "tuning_is_enabled",
    "record_untuned_enable",
    "record_untuned_disable",
    "record_untuned_is_enabled",
    "set_verbose",
    "is_verbose",
    "set_max_tuning_duration",
    "get_max_tuning_duration",
    "set_max_tuning_samples",
    "get_max_tuning_samples",
    "set_filename",
    "get_filename",
    "get_results",
    "read_file",
    "write_file",
]


def enable(val: bool = True) -> None:
    r"""Turn GEMM kernel tuning on or off.

    While on, every GEMM on the tunable path reuses the algorithm recorded
    for its signature (or measures one, when tuning is also enabled and no
    winner is recorded yet). While off, dispatch is exactly the untuned
    behavior.
    """
    _C._cuda_tunableop_enable(val)


def disable() -> None:
    r"""Turn GEMM kernel tuning off. See :func:`enable`."""
    _C._cuda_tunableop_enable(False)


def is_enabled() -> bool:
    r"""Return whether GEMM kernel tuning is enabled."""
    return _C._cuda_tunableop_is_enabled()


def tuning_enable(val: bool = True) -> None:
    r"""Control whether untuned GEMM shapes are measured.

    When enabled, a GEMM without a recorded winner times every candidate
    algorithm and records the fastest; newly found winners are appended to
    the results file as they are measured. When disabled, such a GEMM runs
    the library heuristic's top choice.
    """
    _C._cuda_tunableop_tuning_enable(val)


def tuning_disable() -> None:
    r"""Stop measuring untuned GEMM shapes. See :func:`tuning_enable`."""
    _C._cuda_tunableop_tuning_enable(False)


def tuning_is_enabled() -> bool:
    r"""Return whether untuned GEMM shapes are measured."""
    return _C._cuda_tunableop_tuning_is_enabled()


def record_untuned_enable(val: bool = True) -> None:
    r"""Control logging of GEMMs that ran without a tuned choice.

    When enabled, every unique signature that runs untuned is appended to
    the untuned file (``tunableop_untuned<device>.csv`` in the working
    directory), one line per signature.
    """
    _C._cuda_tunableop_record_untuned_enable(val)


def record_untuned_disable() -> None:
    r"""Stop logging untuned GEMMs. See :func:`record_untuned_enable`."""
    _C._cuda_tunableop_record_untuned_enable(False)


def record_untuned_is_enabled() -> bool:
    r"""Return whether untuned GEMMs are being logged."""
    return _C._cuda_tunableop_record_untuned_is_enabled()


def set_verbose(val: bool) -> None:
    r"""Turn diagnostic logging of the tuning context on or off.

    Verbose output goes to stderr and reports state changes, measurement
    passes and file activity. It is meant for debugging.
    """
    _C._cuda_tunableop_set_verbose(val)


def is_verbose() -> bool:
    r"""Return whether diagnostic logging is on."""
    return _C._cuda_tunableop_is_verbose()


def set_max_tuning_duration(duration_ms: int) -> None:
    r"""Bound the time spent measuring one candidate, in milliseconds.

    Zero disables the limit. When both this and :func:`set_max_tuning_samples`
    are set, the smaller bound wins; a measurement always runs at least one
    timed sample.
    """
    _C._cuda_tunableop_set_max_tuning_duration(duration_ms)


def get_max_tuning_duration() -> int:
    r"""Return the per-candidate measurement time limit in milliseconds."""
    return _C._cuda_tunableop_get_max_tuning_duration()


def set_max_tuning_samples(samples: int) -> None:
    r"""Bound the timed samples spent measuring one candidate.

    Zero disables the limit. When both this and :func:`set_max_tuning_duration`
    are set, the smaller bound wins; a measurement always runs at least one
    timed sample.
    """
    _C._cuda_tunableop_set_max_tuning_samples(samples)


def get_max_tuning_samples() -> int:
    r"""Return the per-candidate sample limit."""
    return _C._cuda_tunableop_get_max_tuning_samples()


def set_filename(filename: str, insert_device_ordinal: bool = False) -> None:
    r"""Set the file used to persist tuning results.

    If :attr:`insert_device_ordinal` is ``True``, the current device ordinal
    is embedded in the name (a ``%d`` token is replaced in place, otherwise
    the ordinal lands before the extension). This keeps one-process-per-device
    runs from sharing a file. An empty filename turns file persistence off.
    """
    _C._cuda_tunableop_set_filename(filename, insert_device_ordinal)


def get_filename() -> str:
    r"""Return the configured results filename (empty until set or first use)."""
    return _C._cuda_tunableop_get_filename()


def get_results() -> list[tuple[str, str, str, float]]:
    r"""Return every recorded winner as ``(op, params, kernel, time_ms)``."""
    return _C._cuda_tunableop_get_results()


def read_file(filename: str | None = None) -> bool:
    r"""Merge a tuning results file into the in-memory database.

    The file's validator lines must match the current build and device;
    otherwise the file is rejected and ``False`` is returned. Entries
    already measured in this process win over the file's. If
    :attr:`filename` is not given, the configured results file is used.
    """
    if filename is None:
        filename = get_filename()
    return _C._cuda_tunableop_read_file(filename)


def write_file() -> None:
    r"""Rewrite the results file with every recorded winner.

    The file is recreated with fresh validator lines followed by all
    in-memory results (both those read from files and those measured in
    this process).
    """
    _C._cuda_tunableop_write_file()
