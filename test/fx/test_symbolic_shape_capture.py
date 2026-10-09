"""Shape reads remain graph values instead of example constants."""

import tensorplay as tp
from tensorplay.graph.experimental.proxy_tensor import make_graph


def test_symbolic_capture_preserves_size_arithmetic():
    fn = lambda x: x.reshape(x.size(0) * x.size(1))
    gm = make_graph(fn, tracing_mode="symbolic")(tp.randn(3, 7))
    assert gm.meta["symbolic_shapes"]
    assert any(node.target is tp.ops.tp.sym_size.int for node in gm.graph.nodes)
    for rows, cols in ((3, 7), (5, 11)):
        x = tp.randn(rows, cols)
        assert tp.equal(gm(x), fn(x))


def test_symbolic_capture_records_shape_branch_constraint():
    def fn(x):
        return x + 1 if x.shape[0] > 4 else x * 2

    gm = make_graph(fn, tracing_mode="symbolic")(tp.randn(7))
    dims = gm.meta["symbolic_dims"]
    assert dims.guards
    assert bool(dims.guards[-1].fact.xreplace(dims.hints))
