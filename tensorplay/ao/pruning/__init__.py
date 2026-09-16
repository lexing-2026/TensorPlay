"""Pruning of module tensors under various sparsity recipes.

Exposes the method classes, the container used for iterative pruning, and
the functional entry points that install or remove the mask
reparameterization described in :mod:`tensorplay.ao.pruning.base_classes`.
"""

from .base_classes import BasePruningMethod as BasePruningMethod
from .base_classes import PruningBase as PruningBase
from .base_classes import PruningContainer as PruningContainer
from .methods import CustomFromMask as CustomFromMask
from .methods import Identity as Identity
from .methods import L1Unstructured as L1Unstructured
from .methods import LnStructured as LnStructured
from .methods import RandomStructured as RandomStructured
from .methods import RandomUnstructured as RandomUnstructured
from .methods import custom_from_mask as custom_from_mask
from .methods import global_unstructured as global_unstructured
from .methods import identity as identity
from .methods import is_pruned as is_pruned
from .methods import l1_unstructured as l1_unstructured
from .methods import ln_structured as ln_structured
from .methods import random_structured as random_structured
from .methods import random_unstructured as random_unstructured
from .methods import remove as remove
from .utils import compute_nparams_to_prune as compute_nparams_to_prune
from .utils import validate_pruning_amount as validate_pruning_amount

__all__ = [
    "PruningBase",
    "BasePruningMethod",
    "PruningContainer",
    "Identity",
    "RandomUnstructured",
    "RandomStructured",
    "L1Unstructured",
    "LnStructured",
    "CustomFromMask",
    "identity",
    "random_unstructured",
    "random_structured",
    "l1_unstructured",
    "ln_structured",
    "global_unstructured",
    "custom_from_mask",
    "remove",
    "is_pruned",
    "validate_pruning_amount",
    "compute_nparams_to_prune",
]
