"""A template whose kernel is a region of the graph, emitted as one.

Some operations are not worth writing by hand: fusing a run of elementwise
operations into a single pass is a matter of capturing the region, not of expressing
it.
"""

from __future__ import annotations

from typing import Any

from .common import KernelTemplate
from .ir import SubgraphChoiceCaller

class SubgraphTemplate(KernelTemplate):
    """A template whose kernel is a region of the graph, emitted as one.

    Some operations are not worth writing by hand: the fusion of a run of
    elementwise operations into a single pass is a matter of capturing the
    region, not of expressing it.  This template therefore takes the region as
    it stands, and its one decision is whether the region is worth keeping as
    a unit -- which is the same question every other template answers about its
    own configuration, asked about a different thing.
    """

    def __init__(self, name: str, hash: str | None = None):
        super().__init__(name, hash)
        #: Set by the caller that owns the region, before any choice is built.
        self.graph = None
        self.decomposition = None
        self.decomposition_kwargs: dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> SubgraphChoiceCaller:
        if self.graph is None:
            raise NotImplementedError(f"{self.name} has no region to emit")
        layout = kwargs.get("layout")
        return SubgraphChoiceCaller(
            name=self.name,
            input_nodes=tuple(kwargs.get("input_nodes") or ()),
            layout=layout,
            description=repr(sorted(kwargs.get("overrides", {}).items())),
            graph=self.graph,
            decomposition=self.decomposition,
            decomposition_kwargs=self.decomposition_kwargs,
        )

    def choice_or_none(self, **kwargs: Any) -> SubgraphChoiceCaller | None:
        return super().choice_or_none(**kwargs)
