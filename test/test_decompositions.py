"""L5-M4: decomposition pass rewrites composites into differentiable primitives."""

import operator

import pytest

import tensorplay as tp
from tensorplay.compiler import DecomposePass
from tensorplay.compiler.graph import Tracer
from tensorplay.compiler import build_aot


def _trace(fn, sample):
    return Tracer().trace(fn, sample_inputs=sample)


def test_sigmoid_decomposed_into_primitives():
    def fn(x):
        return x.sigmoid().sum()

    smap = {"x": tp.tensor([1.0, 2.0])}
    gm = _trace(fn, smap)
    targets = {n.target for n in gm.graph.nodes if n.op == "call_method"}
    assert "sigmoid" in targets
    res = DecomposePass()(gm)
    assert res.modified is True
    methods = {n.target for n in gm.graph.nodes if n.op == "call_method"}
    funcs = {getattr(n.target, "__name__", n.target) for n in gm.graph.nodes if n.op == "call_function"}
    assert "sigmoid" not in methods
    assert "exp" in methods
    assert "truediv" in funcs or operator.truediv in funcs


def test_decomposed_graph_is_differentiable_structurally():
    def fn(x):
        return x.sigmoid().sum()

    smap = {"x": tp.tensor([1.0, 2.0])}
    gm = _trace(fn, smap)
    DecomposePass()(gm)
    result = build_aot(gm, sample_inputs=smap)
    phs = list(result.backward_gm.graph.placeholders)
    kinds = result.input_kinds
    assert len(phs) == len(kinds)
    assert sum(k == "tangent" for k in kinds) == 1


def test_pass_idempotent_when_no_composites():
    def fn(x, w):
        return (x * w).relu().sum()

    smap = {"x": tp.tensor([1.0]), "w": tp.tensor([1.0])}
    gm = _trace(fn, smap)
    res = DecomposePass()(gm)
    assert res.changed is False
