"""P4: guard expression chains for compiled specializations (L6)."""

import pytest

import tensorplay as tp
from tensorplay.compiler.guards import GuardChain, format_recompile_reasons


@pytest.fixture(autouse=True)
def _reset_compiler():
    yield
    tp.compiler.reset()


@pytest.fixture()
def probe_calls():
    calls = []

    def backend(graph_module, example_inputs, **kwargs):
        calls.append([(n.op, n.target) for n in graph_module.graph.nodes])
        return graph_module.forward

    tp.compiler.register_backend(name="l6_probe")(backend)
    yield calls
    tp.compiler.unregister_backend("l6_probe")



def test_guard_chain_renders_tensor_conditions():
    x = tp.zeros(4, 3)
    from tensorplay.compiler.api import _input_signature

    key = (_input_signature((x,), {}, dynamic=False), ())
    chain = GuardChain(key, param_names=(), dynamic=False)

    exprs = [guard.expr for guard in chain.guards]
    assert any("pytype" in expr for expr in exprs)
    assert any(".shape" in expr and "4" in expr for expr in exprs)
    assert any("dtype" in expr for expr in exprs)
    assert chain.source.startswith("def guard(")


def test_evaluate_and_explain_on_live_args(probe_calls):
    def fn(x, w):
        return (x * w).sum()

    x = tp.tensor([1.0, 2.0])
    w = tp.tensor([3.0, 4.0])
    compiled = tp.compile(fn, backend="l6_probe")
    compiled(x, w)

    chains = list(compiled._tensorplay_guard_chains.values())
    assert len(chains) == 1
    chain = chains[0]

    # Same metadata -> guards pass.
    assert chain.evaluate((tp.tensor([9.0, 9.0]), tp.tensor([1.0, 1.0])), {})
    # Different shape -> shape guard fails with a rendered reason.
    failures = chain.explain((tp.tensor([1.0, 2.0, 3.0]), w), {})
    assert failures
    assert any("[2]" in guard.expr or "shape" in guard.expr for guard in failures)
    text = format_recompile_reasons(failures)
    assert "inputs" in text


def test_recompile_reasons_recorded_under_dynamic(probe_calls):
    def fn(x):
        return x * 2 + 1

    compiled = tp.compile(fn, backend="l6_probe", dynamic=True)
    compiled(tp.tensor([1.0, 2.0]))
    compiled(tp.tensor([[1.0]]))  # rank 1 -> rank 2 -> new specialization

    reasons = compiled._tensorplay_last_recompile_reasons
    assert reasons
    # Shape guards render as indexed expressions ("inputs[0][0][2]") rather
    # than literal prose, so match the same disjunction the chain-rendering
    # test above accepts.
    assert any(
        "rank" in guard.expr or "shape" in guard.expr or "[2]" in guard.expr
        for guard in reasons
    )
    assert len(probe_calls) == 2


def test_recompile_warning_via_env(probe_calls, recwarn, monkeypatch):
    monkeypatch.setenv("TP_LOG_RECOMPILES", "1")

    def fn(x):
        return x * 2 + 1

    compiled = tp.compile(fn, backend="l6_probe", dynamic=True)
    compiled(tp.tensor([1.0, 2.0]))
    n_warnings = len(recwarn)
    compiled(tp.tensor([[1.0]]))
    assert len(recwarn) > n_warnings
    message = str(recwarn.pop(UserWarning).message)
    assert "recompiling" in message


def test_reset_clears_guard_chains(probe_calls):
    def fn(x):
        return x + 1

    compiled = tp.compile(fn, backend="l6_probe")
    compiled(tp.tensor([1.0, 2.0]))
    assert compiled._tensorplay_guard_chains
    tp.compiler.reset()
    assert not compiled._tensorplay_guard_chains
