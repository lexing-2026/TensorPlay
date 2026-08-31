"""CPU backend capability queries."""

import tensorplay

__all__ = ["get_cpu_capability"]


def get_cpu_capability() -> str:
    """Return the CPU instruction set selected for this build."""
    return str(tensorplay._C._get_build_info().get("CPU_CAPABILITY", ""))
