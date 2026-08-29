# mypy: allow-untyped-defs
r"""CUDA tunable-operation controls.

The TunableOp tuning framework requires runtime instrumentation this
TensorPlay build does not expose; all controls are present and report as
disabled / raise on use.
"""

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
    "read_file",
    "write_file",
]


def _fail(*args, **kwargs):
    raise RuntimeError("TunableOp is not supported by this TensorPlay build")


def enable(val: bool = True) -> None:
    r"""Enable TunableOp."""
    if val:
        _fail()


def disable() -> None:
    r"""Disable TunableOp."""


def is_enabled() -> bool:
    r"""Read TunableOp enabled status. Always ``False`` here."""
    return False


def tuning_enable(val: bool = True) -> None:
    r"""Enable writing of TuningResultsFiles."""
    if val:
        _fail()


def tuning_disable() -> None:
    r"""Disable writing of TuningResultsFiles."""


def tuning_is_enabled() -> bool:
    r"""Read whether TuningResultsFiles are enabled. Always ``False`` here."""
    return False


def record_untuned_enable(val: bool = True) -> None:
    r"""Enable recording of untuned operations."""
    if val:
        _fail()


def record_untuned_disable() -> None:
    r"""Disable recording of untuned operations."""


def record_untuned_is_enabled() -> bool:
    r"""Read whether recording of untuned operations is enabled."""
    return False


def set_verbose(val: bool) -> None:
    r"""Set verbosity of TunableOp."""


def is_verbose() -> bool:
    r"""Read verbosity of TunableOp. Always ``False`` here."""
    return False


set_max_tuning_duration = _fail
get_max_tuning_duration = _fail
set_max_tuning_samples = _fail
get_max_tuning_samples = _fail
read_file = _fail
write_file = _fail
