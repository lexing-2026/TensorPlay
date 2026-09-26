"""The templates this route can choose between.

One table, keyed by the name each template's operators are declared under, and
a check that no two entries in it answer to the same identity: a stored decision
names a choice by its identity, so a collision would make every stored decision
ambiguous between the one that was measured and the one that was not.

Each module here answers to one upstream module, and the split is the split:

  params        what a configuration is
  ir            where a result lands, what a call is, what a built choice is
  config        what a tile would like
  heuristics    which of it fits this call
  choices       which tables a device is measured with
  select        the operation itself, as a peer of the templates
  common        the four things every template shares
  choice        one template with one configuration, deferred
  bridge        the layer only a loop region needs
  mm            products
  conv          convolutions, forward and both ways back
  subgraph      a region of the graph, kept as a unit
"""

from __future__ import annotations

from .bridge import LoopTemplate
from .common import KernelTemplate
from .conv import (
    CONV,
    CONV_BWD_INPUT,
    CONV_BWD_WEIGHT,
    CONV_TEMPLATES,
    DEPTHWISE_CONV,
    ConvBwdInputTemplate,
    ConvBwdWeightTemplate,
    ConvConfigHeuristics,
    ConvGradientConfigHeuristics,
    ConvTemplate,
    DepthwiseConvConfigHeuristics,
    DepthwiseConvTemplate,
    conv1x1_via_mm,
    conv1x1_via_product,
    framework_convolution,
)
from .mm import GEMM, GemmConfigHeuristics, GemmTemplate
from .select import ExternKernelChoice
from .subgraph import SubgraphTemplate

#: Templates by the name their operators are declared under.
TEMPLATES: dict[str, KernelTemplate] = {
    GEMM.name: GEMM,
    CONV.name: CONV,
    DEPTHWISE_CONV.name: DEPTHWISE_CONV,
    CONV_BWD_INPUT.name: CONV_BWD_INPUT,
    CONV_BWD_WEIGHT.name: CONV_BWD_WEIGHT,
}


def template_for(name: str) -> KernelTemplate | None:
    return TEMPLATES.get(name)


def assert_uids_unique() -> None:
    """No two choices may answer to the same identity."""

    seen: dict[str, str] = {}
    for choice in (*TEMPLATES.values(), *ExternKernelChoice._registry.values()):
        identity = choice.uid
        owner = seen.get(identity)
        if owner is not None and owner != choice.name:
            raise AssertionError(
                f"two choices share the identity {identity!r}: "
                f"{owner} and {choice.name}"
            )
        seen[identity] = choice.name


assert_uids_unique()

__all__ = [
    "CONV",
    "CONV_BWD_INPUT",
    "CONV_BWD_WEIGHT",
    "CONV_TEMPLATES",
    "DEPTHWISE_CONV",
    "GEMM",
    "TEMPLATES",
    "ConvBwdInputTemplate",
    "ConvBwdWeightTemplate",
    "ConvConfigHeuristics",
    "ConvGradientConfigHeuristics",
    "ConvTemplate",
    "DepthwiseConvConfigHeuristics",
    "DepthwiseConvTemplate",
    "ExternKernelChoice",
    "GemmConfigHeuristics",
    "GemmTemplate",
    "KernelTemplate",
    "LoopTemplate",
    "SubgraphTemplate",
    "assert_uids_unique",
    "conv1x1_via_mm",
    "conv1x1_via_product",
    "framework_convolution",
    "template_for",
]
