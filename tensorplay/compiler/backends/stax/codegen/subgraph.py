"""Turning a chosen subgraph into nodes of the region it was chosen in.

A subgraph is compiled as a program of its own, and a program of its own has a
boundary: what it computes is stored, handed back, and read again.  A boundary
that exists only because of how the code was written is a cost nobody asked for,
so the chosen subgraph is instead taken apart and its operations become
operations of the surrounding region, which can then be fused with what is around
them as if they had been written there.
"""


def inline_subgraph_to_ir_nodes(gm, inputs, name):
    """The value a subgraph yields, with its operations lowered into this region.

    The subgraph is walked against the region it came from rather than against a
    region of its own, so each operation it contains becomes an operation of that
    region.  The region's own module is stood in for the duration, because an
    operation is lowered by looking at what the module it belongs to declares,
    and an operation of the subgraph must be lowered as one of the module that
    holds it.

    Returns a value standing for the subgraph's last operation, which is the
    value the subgraph produces.
    """

    from ..loops import V

    original_module = V.graph.module
    try:
        V.graph.module = gm
        return V.graph.process_subgraph_nodes(gm, inputs)
    finally:
        V.graph.module = original_module
