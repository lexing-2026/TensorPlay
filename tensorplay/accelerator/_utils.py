# mypy: allow-untyped-defs
"""Shared index helpers for the accelerator namespace.

All device arguments below accept a device object, a spelling such as
``"cuda:1"``, an integer index, or ``None`` for the current device.
"""

import tensorplay


def _native():
    try:
        import tensorplay._C as _mod
    except ImportError:
        return None
    return _mod


def _native_accelerator():
    mod = _native()
    if mod is None:
        return None
    probe = getattr(mod, "_accelerator_getAccelerator", None)
    if not callable(probe):
        return None
    return probe()


def _parse_index(device=None) -> int:
    if device is None:
        return -1
    if isinstance(device, int):
        return device
    if isinstance(device, tensorplay.device):
        return device.index
    if isinstance(device, str):
        text = device.strip()
        if ":" in text:
            _, _, idx = text.partition(":")
            return int(idx)
        return -1
    raise RuntimeError(
        f"Invalid device value '{device}', expect device, str, int, or None"
    )


def _resolve_device(device=None, optional: bool = False) -> int:
    """Index for ``device`` after checking it names the accelerator type."""
    if isinstance(device, (str, tensorplay.device)):
        spelling = (
            device if isinstance(device, str) else f"{device.type}"
            + (f":{device.index}" if device.index >= 0 else "")
        )
        parsed = tensorplay.device(spelling)
        from . import current_accelerator as _current

        acc = _current()
        if acc is None:
            raise RuntimeError("Accelerator expected")
        if acc.type != parsed.type:
            raise ValueError(
                f"{parsed.type} doesn't match the current accelerator {acc}."
            )
        if parsed.index is None or parsed.index < 0:
            if not optional:
                raise ValueError(
                    "Expected a device with a specified index or an "
                    f"integer, but got: {device}"
                )
            from . import current_device_index as _current_index

            return _current_index()
        return parsed.index
    if isinstance(device, int):
        return device
    if device is None:
        if not optional:
            raise ValueError(
                "Expected a device with a specified index or an integer, "
                f"but got: {device}"
            )
        from . import current_device_index as _current_index

        return _current_index()
    raise RuntimeError(
        f"Invalid device value '{device}', expect device, str, int, or None"
    )


def _require_accelerator():
    from . import current_accelerator as _current

    acc = _current()
    if acc is None:
        raise RuntimeError("No accelerator device in this build")
    return acc


def _device_module_for_accelerator():
    acc = _require_accelerator()
    return tensorplay.get_device_module(acc)


__all__ = ["_parse_index", "_resolve_device"]
