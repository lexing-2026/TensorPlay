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
    assert res.modified is False


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

import math

from tensorplay.backends.stax import stax


# sec/csc/cot/tanhshrink/squared_difference/swish 的表层包装尚未生成
# （当前 schema 尚缺条目，分解规则保留）；
# 测试仅覆盖当前已暴露的算子面。
_CASES = {
    "softplus": (lambda x: tp.softplus(x), math.log1p(math.exp(0.7))),
    "mish": (lambda x: tp.mish(x), 0.7 * math.tanh(math.log1p(math.exp(0.7)))),
    "log1p": (lambda x: tp.log1p(x), math.log1p(0.7)),
    "expm1": (lambda x: tp.expm1(x), math.expm1(0.7)),
    "logit": (lambda x: tp.logit(x), math.log(0.7 / 0.3)),
    "sinh": (lambda x: tp.sinh(x), math.sinh(0.7)),
    "cosh": (lambda x: tp.cosh(x), math.cosh(0.7)),
    "asinh": (lambda x: tp.asinh(x), math.asinh(0.7)),
    "atanh": (lambda x: tp.atanh(x), math.atanh(0.7)),

}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_decomposed_ops_match_reference(name):
    eager_fn, expected = _CASES[name]
    x = tp.tensor([0.7])
    got = eager_fn(x)
    assert abs(float(got[0]) - expected) < 1e-5


@pytest.mark.parametrize("name", sorted(_CASES))
def test_decomposed_op_compiles_to_native_graph(name):
    """每个新分解条目都必须落进原生图——覆盖面倍增的直接证明。

    （仅推理：compile 训练路径依赖 AOT 分解，被 tools/codegen 重构暂时
    阻断，见 alignment plan L5-M4 遗留项。）
    """
    fn, _ = _CASES[name]
    x = tp.tensor([0.7])
    gm = Tracer().trace(fn, sample_inputs={"x": x})
    # 复现真实编译管线：分解 → 消死代码 → 融合标注，再交给后端
    from tensorplay.compiler.decompositions import DecomposePass
    from tensorplay.compiler.passes import (
        ConstFold,
        DeadCodeElimination,
        PassManager,
    )
    from tensorplay.compiler.fx_passes import (
        NormalizeOperators,
        PointwiseFusionHint,
    )

    PassManager(
        [
            NormalizeOperators(),
            ConstFold(),
            DecomposePass(),
            DeadCodeElimination(),
            PointwiseFusionHint(),
        ]
    )(gm)
    compiled = stax(gm, [x])
    assert getattr(gm, "_stax_native_graph", None) is not None, (
        f"{name} decomposition did not lower natively"
    )
    assert abs(float(compiled(x)[0]) - float(fn(x)[0])) < 1e-5


_DECOMP_GRAD_CASES = {
    "softplus": lambda x: tp.softplus(x),
    "mish": lambda x: tp.mish(x),
    "silu": lambda x: tp.silu(x),
    "logit": lambda x: tp.logit(x),
    "sinh": lambda x: tp.sinh(x),
    "cosh": lambda x: tp.cosh(x),
    "atanh": lambda x: tp.atanh(x),
}


@pytest.mark.parametrize("name", sorted(_DECOMP_GRAD_CASES))
def test_decomposed_grad_matches_eager(name):
    """分解链的 VJP 组合必须与复合算子的原生求导一致（解释器执行真原语）。"""
    fn = _DECOMP_GRAD_CASES[name]

    def loss(v):
        return fn(v * v).sum()

    xa = tp.tensor([0.7], requires_grad=True)
    loss(xa).backward()
    grad_eager = [float(g) for g in xa.grad.tolist()]

    gm = Tracer().trace(loss, sample_inputs={"v": tp.tensor([0.7], requires_grad=True)})
    res = DecomposePass()(gm)
    assert res.modified is True
    interpreted = gm.recompile()
    xb = tp.tensor([0.7], requires_grad=True)
    interpreted(xb).backward()
    assert [float(g) for g in xb.grad.tolist()] == pytest.approx(
        grad_eager, rel=1e-5
    )


def test_lerp_and_addcmul_decompose():
    s = tp.tensor([1.0])
    e = tp.tensor([3.0])
    assert abs(float(tp.lerp(s, e, 0.25)[0]) - 1.5) < 1e-6

    inp = tp.tensor([10.0])
    t1 = tp.tensor([2.0])
    t2 = tp.tensor([5.0])
    got = tp.addcmul(inp, t1, t2)
    assert abs(float(got[0]) - 20.0) < 1e-6

    got_div = tp.addcdiv(inp, t1, t2)
    assert abs(float(got_div[0]) - 10.4) < 1e-6


def test_compile_pipeline_applies_decomposition():
    """默认管线接入：tp.compile 的图里不应再出现已注册复合算子。"""
    seen = {}

    from tensorplay.compiler import register_backend

    @register_backend(name="_decomp_probe")
    def probe(gm, example_inputs, **kw):
        seen["names"] = [
            getattr(n.target, "__name__", n.target)
            for n in gm.graph.nodes
            if n.op in ("call_function", "call_method")
        ]
        return gm.recompile()

    try:
        x = tp.tensor([0.7])
        tp.compile(lambda a: tp.mish(a), backend="_decomp_probe")(x)
        assert "mish" not in seen["names"]
        assert "softplus" not in seen["names"]
        assert "tanh" in seen["names"]
    finally:
        tp.compiler.unregister_backend("_decomp_probe")
