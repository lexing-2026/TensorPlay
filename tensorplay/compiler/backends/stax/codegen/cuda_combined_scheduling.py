"""Scheduling for a device that runs its own kernels.

The kernels a device runs are written as source and compiled, and the writing
is done by a printer that belongs to the device rather than to the host.  So
the device has a scheduling of its own, and this is it: it holds the one that
writes device kernels and answers for the device.  It exists as a class of its
own rather than as that printer directly so that a device whose work is
sometimes written one way and sometimes another has somewhere to say which, and
so that the place such a decision is made is a name rather than a branch in
every method.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..kernel_scheduler import BaseScheduling
from .triton import TritonScheduling

#: Where this module's messages go.
log = logging.getLogger(__name__)


class CUDACombinedScheduling(BaseScheduling):
    """Scheduling for a device that runs its own kernels.

    Answers for the device by the printer that writes its kernels, and says so
    in one place: which printer a node is written by, which nodes may be written
    together, and how each thing is written.
    """

    def __init__(self, scheduler=None) -> None:
        super().__init__(scheduler)
        self._triton_scheduling = TritonScheduling(scheduler)

    def get_backend_features(self, device) -> Any:
        return self._triton_scheduling.get_backend_features(device)

    def has_sub_parent_epilogue(self, nodes: Sequence) -> bool:
        return self._triton_scheduling.has_sub_parent_epilogue(nodes)

    def choose_node_backend(self, node) -> BaseScheduling:
        return self._triton_scheduling

    def can_fuse_vertical(self, node1, node2) -> bool:
        return self._triton_scheduling.can_fuse_vertical(node1, node2)

    def can_fuse_horizontal(self, node1, node2) -> bool:
        return self._triton_scheduling.can_fuse_horizontal(node1, node2)

    def can_fuse_reduction_epilogue(self, node1, node2) -> bool:
        # Only a template that carries its own reduction can be written into the
        # tail of another reduction, and the one that does is not written here.
        return False

    def group_fn(self, sizes: Sequence) -> tuple:
        return self._triton_scheduling.group_fn(sizes)

    def codegen_template(
        self,
        template_node,
        epilogue_nodes: Sequence,
        prologue_nodes: Sequence,
    ) -> None:
        return self._triton_scheduling.codegen_template(
            template_node, epilogue_nodes, prologue_nodes
        )

    def codegen_mix_order_reduction(self, node) -> None:
        return self._triton_scheduling.codegen_mix_order_reduction(node)

    def codegen_staged_reduction(self, node) -> None:
        return self._triton_scheduling.codegen_staged_reduction(node)

    def codegen_node(self, node) -> None:
        return self._triton_scheduling.codegen_node(node)

    def codegen_sync(self) -> None:
        return self._triton_scheduling.codegen_sync()

    def flush(self) -> None:
        return self._triton_scheduling.flush()

    def codegen_combo_kernel(self, *args: Any, **kwargs: Any) -> None:
        return self._triton_scheduling.codegen_combo_kernel(*args, **kwargs)

    def benchmark_fused_nodes(self, nodes: Sequence):
        return self._triton_scheduling.benchmark_fused_nodes(nodes)

    def benchmark_codegened_module(self, module):
        return self._triton_scheduling.benchmark_codegened_module(module)
