"""Operator templates: a template owns an operator's implementation space.

A template says three things about the operation it owns: what its result
looks like, which configurations are worth considering, and how to turn one
configuration into a kernel.  The compiler asks for the result spec up front,
enumerates the choices, and keeps the one it settled on -- so a region's
kernels are chosen while it is compiled, and the choice is keyed to the code
that produced it.

The gemm template is the first.  Its result is the product's shape, dtype and
layout; its configurations are the operator as the framework already runs it
plus a curated tile set; and the operator is always among them, so a measured
tile can never leave a region slower than not measuring at all.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Iterator

from .codegen.triton_gemm import GEMM_CANDIDATE_CONFIGS
from .loops import TemplateKernel


# ---------------------------------------------------------------------------
# configurations
# ---------------------------------------------------------------------------


class TemplateParams(ABC):
    """One configuration of a template.

    A configuration is a value, not a tuple: it knows how to become keyword
    arguments for the emitter and how to become a record that survives being
    written down and read back.
    """

    @abstractmethod
    def to_kwargs(self) -> dict[str, Any]:
        """The configuration as emitter arguments."""

    @abstractmethod
    def to_record(self) -> dict[str, Any]:
        """The configuration as something that can be stored and read back."""

    @classmethod
    @abstractmethod
    def from_record(cls, record: dict[str, Any]) -> "TemplateParams":
        """The configuration a stored record describes."""


class DictParams(TemplateParams):
    """A configuration given as the emitter's own arguments."""

    def __init__(self, kwargs: dict[str, Any]):
        self.kwargs = dict(kwargs)

    def to_kwargs(self) -> dict[str, Any]:
        return dict(self.kwargs)

    def to_record(self) -> dict[str, Any]:
        return dict(self.kwargs)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "DictParams":
        return cls(record)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DictParams) and other.kwargs == self.kwargs

    def __hash__(self) -> int:
        return hash(tuple(sorted((k, repr(v)) for k, v in self.kwargs.items())))

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in sorted(self.kwargs.items()))
        return f"DictParams({inner})"


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


class OutSpec:
    """What a template says about one of its results.

    A template knows its own result -- a matmul's product shape, dtype and
    layout -- so the caller does not have to infer it and cannot get it
    subtly wrong for a form the caller has never seen.
    """

    __slots__ = ("dtype", "size", "stride", "offset", "device", "requires_grad")

    def __init__(self, dtype, size, stride=None, offset: int = 0, device=None,
                 requires_grad: bool = False):
        self.dtype = dtype
        self.size = tuple(int(extent) for extent in size)
        self.stride = None if stride is None else tuple(int(s) for s in stride)
        self.offset = int(offset)
        self.device = device
        self.requires_grad = bool(requires_grad)

    @property
    def numel(self) -> int:
        total = 1
        for extent in self.size:
            total *= max(int(extent), 1)
        return total

    def __repr__(self) -> str:
        return (
            f"OutSpec({self.dtype}, {self.size}, stride={self.stride}, "
            f"requires_grad={self.requires_grad})"
        )


def contiguous_stride(size) -> tuple:
    stride = []
    running = 1
    for extent in reversed(tuple(size)):
        stride.append(running)
        running *= max(int(extent), 1)
    return tuple(reversed(stride))


# ---------------------------------------------------------------------------
# choices
# ---------------------------------------------------------------------------


class TemplateChoice:
    """One configuration of one template for one call site.

    The kernel is built when the choice is first asked for and kept, including
    when the build turns out not to apply: a configuration that does not fit
    this call is an answer, not a retry.
    """

    def __init__(self, template: "KernelTemplate", params: TemplateParams,
                 out_specs: tuple, meta: dict):
        self.template = template
        self.params = params
        self.out_specs = out_specs
        self.meta = meta
        self._choice = None
        self._resolved = False

    @property
    def key(self) -> tuple:
        return (self.template.uid, self.params.to_record() and repr(
            self.params.to_record()
        ), tuple(spec.size for spec in self.out_specs), str(self.meta.get("device")))

    def resolve(self, plain_launch):
        """The kernel for this configuration, or ``None`` when it does not fit."""

        if not self._resolved:
            self._resolved = True
            try:
                self._choice = self.template.choice_or_none(
                    self.params, self.out_specs, self.meta, plain_launch
                )
            except Exception:  # noqa: BLE001 - a config that fails is not a fit
                self._choice = None
        return self._choice

    def __repr__(self) -> str:
        return f"TemplateChoice({self.template.name}, {self.params!r})"


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------


class KernelTemplate:
    """One operation, implemented by kernels chosen from a configuration space."""

    def __init__(self, name: str):
        self.name = name

    # identity ------------------------------------------------------------
    def emitter(self) -> str:
        """The source a configuration is turned into, for change detection."""

        raise NotImplementedError

    def uid(self) -> str:
        """A key that changes when the template's meaning changes."""

        digest = hashlib.sha1(self.emitter().encode()).hexdigest()[:16]
        return f"{self.name}:{digest}"

    # results -------------------------------------------------------------
    def out_specs(self, meta: dict) -> tuple:
        """What this call produces."""

        raise NotImplementedError

    # configurations ------------------------------------------------------
    def configurations(self, out_specs: tuple, meta: dict) -> Iterator[TemplateParams]:
        """The configurations worth considering for this call."""

        raise NotImplementedError

    def choice_or_none(self, params: TemplateParams, out_specs: tuple, meta: dict,
                       plain_launch):
        """The kernel for one configuration, or ``None`` when it does not fit."""

        raise NotImplementedError

    # enumeration ---------------------------------------------------------
    def choices(self, meta: dict) -> list:
        """Every configuration that applies to this call, best guess first."""

        specs = self.out_specs(meta)
        out = []
        for params in self.configurations(specs, meta):
            choice = TemplateChoice(self, params, specs, meta)
            if choice.resolve(None) is not None:
                out.append(choice)
        return out


class GemmTemplate(KernelTemplate):
    """Matmul-shaped operators, measured against the framework's own gemm.

    Validity belongs to the template: a configuration that does not fit this
    call -- not two-dimensional, not the dtype the tiles are written for, not
    this device -- is refused here rather than by whoever is asking.
    """

    def __init__(self):
        super().__init__("gemm")

    def emitter(self) -> str:
        """The tile kernel this template emits, identified by its source.

        The digest covers the kernel body and the tuning version, so a stored
        decision is not reused once either has moved.
        """

        from .codegen.triton_gemm import (
            GEMM_TUNING_VERSION,
            _kernel_source_digest,
        )

        return f"{GEMM_TUNING_VERSION}:{_kernel_source_digest()}"

    def out_specs(self, meta: dict) -> tuple:
        size = meta.get("out_size")
        if size is None:
            raise NotImplementedError("a gemm call without a result shape")
        dtype = meta.get("out_dtype")
        return (
            OutSpec(
                dtype,
                size,
                stride=contiguous_stride(size),
                device=meta.get("device"),
                requires_grad=bool(meta.get("requires_grad", False)),
            ),
        )

    def configurations(self, out_specs: tuple, meta: dict):
        yield DictParams({"choice": "native"})
        for config in GEMM_CANDIDATE_CONFIGS:
            yield DictParams(
                {
                    "choice": "triton",
                    "BLOCK_M": config[0],
                    "BLOCK_N": config[1],
                    "BLOCK_K": config[2],
                    "num_warps": config[3],
                    "num_stages": config[4],
                }
            )

    def choice_or_none(self, params, out_specs, meta, plain_launch):
        from .codegen.triton_gemm import tuned_matmul_launch

        if params.to_kwargs().get("choice") == "native":
            return plain_launch
        spec = out_specs[0]
        # A tile kernel is written for the two-dimensional contiguous case in
        # the dtype it accumulates in; anything else keeps the operator.
        if len(spec.size) != 2 or meta.get("transposed"):
            return None
        if meta.get("operand_dtype") not in ("float32",):
            return None
        if not meta.get("qualifies", False):
            return None
        return tuned_matmul_launch(
            plain_launch,
            meta.get("feed", ()),
            meta["operand_specs"],
            spec.size,
            bias_spec=meta.get("bias_spec"),
            b_transposed=bool(meta.get("b_transposed", False)),
        )


GEMM = GemmTemplate()

TEMPLATES: dict[str, KernelTemplate] = {GEMM.name: GEMM}


def template_for(name: str) -> KernelTemplate | None:
    return TEMPLATES.get(name)


__all__ = [
    "GEMM",
    "DictParams",
    "GemmTemplate",
    "KernelTemplate",
    "OutSpec",
    "TEMPLATES",
    "TemplateChoice",
    "TemplateKernel",
    "TemplateParams",
    "contiguous_stride",
    "template_for",
]
