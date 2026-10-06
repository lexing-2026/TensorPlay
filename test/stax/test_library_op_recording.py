"""A library operation is one node of a traced graph, whatever its body runs.

The body of a library operation runs while the graph is being traced; the
operations it makes belong to the node the operation records, so the trace
must neither record them again nor decompose them back into the operation.
"""

import tensorplay as tp
from tensorplay.graph.experimental._dispatch_trace import dispatch_make_graph


@tp.library.custom_op("tplibrecord::double_sin", mutates_args=())
def double_sin(x: tp.Tensor) -> tp.Tensor:
    return x.sin() * 2


@tp.library.custom_op("tplibrecord::negate", mutates_args=())
def negate(x: tp.Tensor) -> tp.Tensor:
    return tp.neg(x)


def _targets(gm):
    return [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]


def test_the_body_of_a_library_operation_is_not_recorded_again():
    gm = dispatch_make_graph(lambda x: double_sin(x) + 1)(tp.randn(3))
    assert _targets(gm) == ["<custom_op tplibrecord::double_sin>", "tp.add.Scalar"]
    x = tp.randn(5)
    assert tp.allclose(gm(x), x.sin() * 2 + 1)


def test_a_body_calling_the_operation_it_decomposes_from_finishes():
    # neg decomposes into the library operation, whose body calls neg: the
    # body runs as the operation's kernel, outside the decomposing trace.
    table = {tp.ops.tp.neg.default: lambda x: negate(x)}
    gm = dispatch_make_graph(lambda x: tp.neg(x) * 3, decomposition_table=table)(tp.randn(3))
    assert _targets(gm) == ["<custom_op tplibrecord::negate>", "tp.mul.Scalar"]
    x = tp.randn(4)
    assert tp.allclose(gm(x), -x * 3)


def test_a_compiled_real_transform_runs_through_its_primitive():
    # rfft decomposes onto the real-to-complex primitive, whose eager body is
    # the transform itself.
    tp.manual_seed(0)
    a = tp.randn(6, 10)
    got = tp.compile(lambda t: tp.fft.rfft(t).abs() * 2, strict_native=True)(a)
    assert tp.allclose(got, tp.fft.rfft(a).abs() * 2, atol=1e-5)
