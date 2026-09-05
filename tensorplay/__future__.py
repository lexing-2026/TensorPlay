"""Optional module-conversion behaviors."""

_overwrite_module_params_on_conversion: bool = False
_swap_module_params_on_conversion: bool = False


def set_overwrite_module_params_on_conversion(value: bool) -> None:
    """Choose whether conversion assigns replacement parameter objects."""
    global _overwrite_module_params_on_conversion
    _overwrite_module_params_on_conversion = value


def get_overwrite_module_params_on_conversion() -> bool:
    """Return whether conversion assigns replacement parameter objects."""
    return _overwrite_module_params_on_conversion


def set_swap_module_params_on_conversion(value: bool) -> None:
    """Choose whether conversion exchanges parameter storage in place."""
    global _swap_module_params_on_conversion
    _swap_module_params_on_conversion = value


def get_swap_module_params_on_conversion() -> bool:
    """Return whether conversion exchanges parameter storage in place."""
    return _swap_module_params_on_conversion


__all__ = [
    "get_overwrite_module_params_on_conversion",
    "get_swap_module_params_on_conversion",
    "set_overwrite_module_params_on_conversion",
    "set_swap_module_params_on_conversion",
]
