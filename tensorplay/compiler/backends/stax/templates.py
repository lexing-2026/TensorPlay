"""Operator templates: one place where a kernel's implementation is chosen.

A template owns an operation's candidate space and the rule for picking one.
The compiled program records the choice instead of re-deciding it per call,
so "which kernel runs this operator" has a single answer per region.

The gemm template is the first: its candidates are the operator as the
framework already runs it plus a curated tile set, and the choice is measured
once per shape, validated against the operator it replaces, and cached.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from .codegen.triton_gemm import GEMM_CANDIDATE_CONFIGS
from .loops import TemplateKernel


class KernelTemplate:
    """One operator, implemented by a kernel chosen from a config space."""

    def __init__(self, name: str):
        self.name = name

    def candidates(self) -> list:
        """The implementations worth measuring, best guess first."""

        raise NotImplementedError

    def resolve(self, launch: Callable[[list], Any], feed: Sequence[Any], meta: dict):
        """Bake a launcher for this operator, or ``None`` to keep the plain one."""

        raise NotImplementedError


class GemmTemplate(KernelTemplate):
    """Matmul-shaped operators, measured against the framework's own gemm.

    The operator is always a candidate, so a measured tile can never leave the
    region slower than not measuring at all, and a tile that disagrees with
    the operator on a deterministic probe is rejected rather than trusted.
    """

    def __init__(self):
        super().__init__("gemm")

    def candidates(self) -> list:
        return [("native",), *(("triton", cfg) for cfg in GEMM_CANDIDATE_CONFIGS)]

    def resolve(self, launch, feed, meta):
        from .codegen.triton_gemm import tuned_matmul_launch

        operand_specs = meta.get("operand_specs")
        out_shape = meta.get("out_shape")
        if operand_specs is None or out_shape is None:
            return None
        return tuned_matmul_launch(
            launch,
            feed,
            operand_specs,
            out_shape,
            bias_spec=meta.get("bias_spec"),
            b_transposed=bool(meta.get("b_transposed", False)),
        )


GEMM = GemmTemplate()

TEMPLATES: dict[str, KernelTemplate] = {GEMM.name: GEMM}


def template_for(name: str) -> KernelTemplate | None:
    return TEMPLATES.get(name)


__all__ = [
    "GEMM",
    "GemmTemplate",
    "KernelTemplate",
    "TEMPLATES",
    "TemplateKernel",
    "template_for",
]
