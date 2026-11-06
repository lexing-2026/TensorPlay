"""P4: guard expression chains for compiled specializations (L6)."""

import pytest

import tensorplay as tp
from tensorplay._stax.guards import GuardChain, format_recompile_reasons


@pytest.fixture(autouse=True)
def _reset_compiler():
    yield
    tp._stax.reset()


@pytest.fixture()
def probe_calls():
    calls = []

    def backend(graph_module, example_inputs, **kwargs):
        calls.append([(n.op, n.target) for n in graph_module.graph.nodes])
        return graph_module.forward

    tp._stax.register_backend(name="l6_probe")(backend)
    yield calls
    tp._stax.unregister_backend("l6_probe")



def test_guard_chain_renders_tensor_conditions():
    x = tp.zeros(4, 3)
    from tensorplay._stax.api import _input_signature

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
    tp._stax.reset()
    assert not compiled._tensorplay_guard_chains


class _TrainingScale(tp.nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale if self.training else x


class _TrainingTree(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = tp.nn.Sequential(_TrainingScale())

    def forward(self, x):
        return self.child(x * 2 if self.training else x)


@pytest.fixture(params=["python", "quick", "fast"])
def training_dispatch(request, monkeypatch):
    from tensorplay._C import _stax as native
    from tensorplay.compiler._core import api

    events = {"captures": 0, "fast": 0}
    compile_region = api._compile_region

    def record(*args, **kwargs):
        events["captures"] += 1
        compiled, gm = compile_region(*args, **kwargs)
        if request.param == "fast":
            def entry(*values, **keywords):
                return compiled(*values, **keywords)

            def fast(*values):
                events["fast"] += 1
                return compiled(*values)

            entry._fast_call = fast
            return entry, gm
        return compiled, gm

    monkeypatch.setattr(api, "_compile_region", record)
    if request.param == "python":
        monkeypatch.setattr(native, "make_call_dispatcher", lambda fn: None)
    events["path"] = request.param
    return events


@pytest.mark.parametrize("entry", ["module", "bound", "closure", "global", "lifted"])
def test_training_modes_guard_all_dispatch_paths(entry, training_dispatch):
    net = _TrainingTree().eval()
    x = tp.tensor([2.0])
    if entry == "module":
        program = net
    elif entry == "bound":
        program = net.forward
    elif entry == "global":
        scope = {"model": net}
        exec("def program(x):\n    return model(x)", scope)
        program = scope["program"]
    elif entry == "lifted":
        offset = tp.tensor([0.0])

        def program(x):
            return net(x) + offset
    else:
        def program(x):
            return net(x)

    compiled = tp.compile(program, backend="eager", fullgraph=True, recompile_limit=3)
    with tp.no_grad():
        assert compiled(x).tolist() == [2.0]
        assert compiled(x).tolist() == [2.0]
        assert training_dispatch["captures"] == 1
        if training_dispatch["path"] == "fast":
            assert training_dispatch["fast"] > 0

        eval_chain = next(iter(compiled._tensorplay_guard_chains.values()))
        # Lifted calls have extra tensor inputs; inspect their chain separately.
        if entry != "lifted":
            assert eval_chain.evaluate((x,), {})

        net.child[0].train()
        assert net.training is False
        if entry != "lifted":
            assert not eval_chain.evaluate((x,), {})
            assert any(g.path == "module-state" for g in eval_chain.explain((x,), {}))
        assert compiled(x).tolist() == program(x).tolist() == [4.0]
        assert compiled(x).tolist() == [4.0]

        net.training = True
        assert compiled(x).tolist() == program(x).tolist() == [8.0]
        assert compiled(x).tolist() == [8.0]
        net.training = False
        assert compiled(x).tolist() == [4.0]
        net.child[0].eval()
        assert compiled(x).tolist() == [2.0]
        assert training_dispatch["captures"] == 3
        assert len(compiled._tensorplay_cache) == 3
        if entry != "lifted":
            assert eval_chain.evaluate((x,), {})


@pytest.mark.parametrize("entry", ["closure", "global"])
def test_training_module_sources_are_read_after_rebinding(entry, training_dispatch):
    net = _TrainingScale().eval()
    if entry == "global":
        scope = {"model": net}
        exec("def program(x):\n    return model(x)", scope)
        program = scope["program"]
    else:
        def program(x):
            return net(x)

    compiled = tp.compile(program, backend="eager", fullgraph=True)
    x = tp.tensor([2.0])
    original = net
    with tp.no_grad():
        assert compiled(x).tolist() == [2.0]
        assert compiled(x).tolist() == [2.0]
        net = _TrainingScale(3)
        if entry == "global":
            scope["model"] = net
        assert compiled(x).tolist() == program(x).tolist() == [6.0]
        net.eval()
        assert compiled(x).tolist() == [2.0]
        net.train()
        assert compiled(x).tolist() == [6.0]
        net = original
        if entry == "global":
            scope["model"] = net
        assert compiled(x).tolist() == [2.0]
    assert training_dispatch["captures"] == 3


def test_training_child_replacement_and_aliases_are_guarded(training_dispatch):
    net = _TrainingTree().eval()
    net.child[0].train()
    compiled = tp.compile(net, backend="eager", fullgraph=True)
    x = tp.tensor([2.0])
    with tp.no_grad():
        assert compiled(x).tolist() == [4.0]
        assert compiled(x).tolist() == [4.0]
        original = net.child[0]
        net.child[0] = _TrainingScale(3)
        assert compiled(x).tolist() == net(x).tolist() == [6.0]
        net.child.add_module("alias", net.child[0])
        assert compiled(x).tolist() == net(x).tolist() == [18.0]
        del net.child.alias
        assert compiled(x).tolist() == [6.0]
        net.child[0] = original
        assert compiled(x).tolist() == [4.0]
    assert training_dispatch["captures"] == 3


def test_training_cache_with_fresh_and_keyword_inputs():
    net = _TrainingScale().eval()
    compiled = tp.compile(net, backend="eager", fullgraph=True)
    with tp.no_grad():
        assert compiled(x=tp.tensor([2.0])).tolist() == [2.0]
        net.train()
        assert compiled(x=tp.tensor([3.0])).tolist() == [6.0]
        net.eval()
        assert compiled(x=tp.tensor([4.0])).tolist() == [4.0]
    assert len(compiled._tensorplay_cache) == 2
