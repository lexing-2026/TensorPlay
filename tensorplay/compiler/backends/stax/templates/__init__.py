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
  mm            products, plain and swept
  conv          convolutions, forward and both ways back
  subgraph      a region of the graph, kept as a unit
"""

from __future__ import annotations

from .select_algorithm import KernelTemplateChoice, make_ktc_generator
from ..codegen.common import KernelTemplate
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
from .mm import (
    BLACKWELL_WS_PERSISTENT_TMA,
    GEMM,
    GEMM_EPILOGUE_SCALING,
    GEMM_MAIN_LOOP_SCALING,
    GEMM_PERSISTENT,
    GEMM_PERSISTENT_TMA,
    GEMM_TEMPLATES,
    GemmConfigHeuristics,
    GemmTemplate,
    PersistentGemmTemplate,
    mm_grid,
    persistent_mm_grid,
)
from .bmm import (
    BMM,
    BMM_SHARED_A,
    BMM_TEMPLATES,
    BmmConfigHeuristics,
    BmmSharedAConfigHeuristics,
    BmmSharedATemplate,
    BMMTemplate,
    bmm_grid,
    bmm_shared_a_grid,
)
from .mm_grouped import (
    GROUPED_MM,
    GROUPED_MM_TEMPLATES,
    GroupedMmConfigHeuristics,
    GroupedMmTemplate,
    grouped_extents,
    grouped_mm_grid,
)
from .mm_plus_mm import (
    MM_PLUS_MM,
    MM_PLUS_MM_TEMPLATES,
    MmPlusMmConfigHeuristics,
    MmPlusMmTemplate,
)
from .mm_common import num_sms
from .params import (
    DictKernelTemplateParams,
    GemmTemplateParams,
    KernelTemplateParams,
)
from .select_algorithm import ExternKernelChoice
from .subgraph import SubgraphChoiceCaller, SubgraphTemplate, subgraph_template

#: Templates by the name their operators are declared under.
TEMPLATES: dict[str, KernelTemplate] = {
    GEMM.name: GEMM,
    GEMM_PERSISTENT.name: GEMM_PERSISTENT,
    BMM.name: BMM,
    BMM_SHARED_A.name: BMM_SHARED_A,
    MM_PLUS_MM.name: MM_PLUS_MM,
    GROUPED_MM.name: GROUPED_MM,
    GEMM_PERSISTENT_TMA.name: GEMM_PERSISTENT_TMA,
    BLACKWELL_WS_PERSISTENT_TMA.name: BLACKWELL_WS_PERSISTENT_TMA,
    GEMM_MAIN_LOOP_SCALING.name: GEMM_MAIN_LOOP_SCALING,
    GEMM_EPILOGUE_SCALING.name: GEMM_EPILOGUE_SCALING,
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
    "BMM",
    "BMM_SHARED_A",
    "CONV",
    "CONV_BWD_INPUT",
    "CONV_BWD_WEIGHT",
    "CONV_TEMPLATES",
    "DEPTHWISE_CONV",
    "GEMM",
    "GEMM_TEMPLATES",
    "GEMM_EPILOGUE_SCALING",
    "GEMM_MAIN_LOOP_SCALING",
    "BLACKWELL_WS_PERSISTENT_TMA",
    "GEMM_PERSISTENT_TMA",
    "GEMM_PERSISTENT",
    "GROUPED_MM",
    "MM_PLUS_MM",
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
    "PersistentGemmTemplate",
    "BmmConfigHeuristics",
    "BmmSharedAConfigHeuristics",
    "BmmSharedATemplate",
    "BMMTemplate",
    "GroupedMmConfigHeuristics",
    "GroupedMmTemplate",
    "MmPlusMmConfigHeuristics",
    "MmPlusMmTemplate",
    "bmm_grid",
    "bmm_shared_a_grid",
    "grouped_extents",
    "grouped_mm_grid",
    "mm_grid",
    "num_sms",
    "persistent_mm_grid",
    "KernelTemplate",
    "KernelTemplateChoice",
    "KernelTemplateParams",
    "DictKernelTemplateParams",
    "GemmTemplateParams",
    "make_ktc_generator",
    "SubgraphChoiceCaller",
    "SubgraphTemplate",
    "subgraph_template",
    "assert_uids_unique",
    "conv1x1_via_mm",
    "conv1x1_via_product",
    "framework_convolution",
    "template_for",
]
