"""Kernel codecache wiring + compile-time autotune logic."""

import json
from types import SimpleNamespace

import pytest

import tensorplay as tp
import tensorplay.compiler.backends.stax.runtime.stax_autotune as sa
from tensorplay.compiler.backends.stax.runtime.stax_autotune import (
    CANDIDATE_CONFIGS,
    EXHAUSTIVE_CANDIDATE_CONFIGS,
)
import tensorplay.compiler.backends.stax.runtime.coordinate_descent as cd
import tensorplay.compiler.backends.stax.codegen.triton as st
from tensorplay.compiler.backends.stax.kernel_cache import CodeCache


@pytest.fixture()
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_CACHE_DIR", str(tmp_path))
    # default_cache memoizes per process; reset for isolation.
    import tensorplay._stax.codecache as cc

    monkeypatch.setattr(cc, "_default_caches", {})
    return tmp_path


# --- M2 helpers ---------------------------------------------------------------


def test_xnumel_bucket_powers_of_two():
    assert sa.xnumel_bucket(0) == 128
    assert sa.xnumel_bucket(1) == 128
    assert sa.xnumel_bucket(127) == 128
    assert sa.xnumel_bucket(128) == 128
    assert sa.xnumel_bucket(129) == 256
    assert sa.xnumel_bucket(1000) == 1024
    assert sa.xnumel_bucket(5000) == 8192


def test_decision_roundtrip(cache_root):
    digest = sa.program_digest([1, 0, 2, -1, 3], [1.5], (4,))
    assert sa.load_decision(digest, 256, "cuda:0") is None
    sa.store_decision(digest, 256, "cuda:0", (512, 8))
    assert sa.load_decision(digest, 256, "cuda:0") == (512, 8)
    assert sa.load_decision(digest, 256, "cuda:1") is None
    assert sa.load_decision(digest, 512, "cuda:0") is None


def test_load_decision_rejects_unknown_config(cache_root):
    digest = sa.program_digest([1], [], (0,))
    payload = json.dumps({"xblock": 333, "warps": 7}).encode()
    sa._decision_cache().store(sa.decision_key(digest, 128, "d"), payload,
                               ext="json")
    assert sa.load_decision(digest, 128, "d") is None


def test_pick_config_benchmarks_once_then_uses_decision(cache_root):
    digest = sa.program_digest([1, 2], [], (0,))
    builds: list[tuple[int, int]] = []
    benches: list[tuple[int, int]] = []

    times = {c: float(len(CANDIDATE_CONFIGS) - i) for i, c in enumerate(CANDIDATE_CONFIGS)}
    times[CANDIDATE_CONFIGS[-1]] = 0.5  # last config wins

    def build_launch(config):
        builds.append(config)

        class FakeLaunch:
            pass

        fake = FakeLaunch()
        fake._cfg = config
        return fake

    def bench_fn(launch, args):
        benches.append("called")  # type: ignore[attr-defined]
        return times[launch._cfg]

    config, launch = sa.pick_config(digest, 300, "cuda:0", build_launch,
                                    [], bench_fn=bench_fn)
    assert config == CANDIDATE_CONFIGS[-1]  # fastest candidate wins
    assert builds == list(sa.CANDIDATE_CONFIGS)  # every candidate compiled once
    # Interleaved rounds (bench_candidates): every candidate is benched once
    # per round so a clock transient hits the whole table, not one config.
    assert len(benches) == 2 * len(CANDIDATE_CONFIGS)

    # Second call hits the persisted decision: single rebuild, no benchmarking.
    builds.clear()
    benches.clear()
    config2, launch2 = sa.pick_config(digest, 300, "cuda:0", build_launch,
                                      [], bench_fn=bench_fn)
    assert config2 == CANDIDATE_CONFIGS[-1]
    assert builds == [CANDIDATE_CONFIGS[-1]]
    assert benches == []


def test_pick_config_skips_failing_candidates(cache_root):
    digest = sa.program_digest([7], [], (0,))

    def build_launch(config):
        if config[0] < 512:
            raise RuntimeError("nope")
        return object()

    calls = {"n": 0}

    def bench_fn(launch, args):
        calls["n"] += 1
        return 2.0

    eligible = [c for c in CANDIDATE_CONFIGS if c[0] >= 512]
    config, _ = sa.pick_config(digest, 64, "cuda:0", build_launch, [],
                               bench_fn=bench_fn)
    assert config == eligible[0]
    assert calls["n"] == 2 * len(eligible)  # two interleaved rounds


def test_pick_config_all_fail_raises(cache_root):
    digest = sa.program_digest([9], [], (0,))

    def build_launch(config):
        raise RuntimeError("always")

    with pytest.raises(RuntimeError, match="all candidate configs"):
        sa.pick_config(digest, 10, "cuda:0", build_launch, [],
                       bench_fn=lambda l, a: 1.0)


def test_bench_candidates_later_round_failure_keeps_earlier_result():
    """A candidate that benched fine in round 0 must not be discarded
    because a later round hiccuped — that transient is exactly the clock
    noise round-interleaving absorbs."""

    candidates = [("a",), ("b",)]
    built = []

    def build(config):
        built.append(config)
        return object()

    calls = {"n": 0}

    def bench_fn(launch, args):
        calls["n"] += 1
        if calls["n"] == 3:  # ("b",) round 1: transient failure
            raise RuntimeError("flaky")
        return 1.0 if config_of(launch) == ("a",) else 5.0

    launch_by_config = {}

    def config_of(launch):
        for cfg, obj in launch_by_config.items():
            if obj is launch:
                return cfg
        raise AssertionError("unknown launch")

    def build2(config):
        obj = build(config)
        launch_by_config[config] = obj
        return obj

    best_config, best_launch, best_time = sa.bench_candidates(
        build2, candidates, [], bench_fn=bench_fn
    )
    assert best_config == ("a",)
    assert best_time == 1.0
    assert best_launch is launch_by_config[("a",)]


def test_bench_candidates_min_across_rounds():
    """Round-2 measurements below round-1 must lower the kept time (min)."""

    candidates = [("only",)]
    seq = [4.0, 2.0]
    state = {"i": 0}

    def build(config):
        return object()

    def bench_fn(launch, args):
        value = seq[min(state["i"], len(seq) - 1)]
        state["i"] += 1
        return value

    _, _, best_time = sa.bench_candidates(build, candidates, [], bench_fn=bench_fn)
    assert best_time == 2.0


def test_disabled_env(monkeypatch):
    monkeypatch.setenv(sa._DISABLE_ENV, "1")
    assert sa.disabled() is True
    monkeypatch.setenv(sa._DISABLE_ENV, "0")
    assert sa.disabled() is False


# --- M1/codegen emission ------------------------------------------------------


def _cuda_inputs():
    return [tp.rand(300, device=tp.device("cuda", 0))]


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA required")
def test_program_digest_shape_independent():
    x = _cuda_inputs()[0]
    digest_a = sa.program_digest([3, -1, 1], [1.5], (2,))
    digest_b = sa.program_digest([3, -1, 1], [1.5], (2,))
    assert digest_a == digest_b
    y = x * 2  # same program, different tensor -> same digest
    assert sa.program_digest([3, -1, 1], [1.5], (2,)) == digest_a
    del y


def test_exhaustive_candidates_extend_baseline():
    assert set(CANDIDATE_CONFIGS) <= set(EXHAUSTIVE_CANDIDATE_CONFIGS)
    # the widened tier completes the (XBLOCK, warps) cross product the
    # baseline omits
    assert (128, 8) in EXHAUSTIVE_CANDIDATE_CONFIGS
    assert (256, 8) in EXHAUSTIVE_CANDIDATE_CONFIGS
    assert len(EXHAUSTIVE_CANDIDATE_CONFIGS) == 10


def test_decision_tiers_are_isolated(cache_root):
    digest = sa.program_digest([1, 0, 2], [1.0], (4,))
    sa.store_decision(digest, 256, "cuda:0", (1024, 8))
    sa.store_decision(digest, 256, "cuda:0", (96, 8), tier="coordesc")
    assert sa.load_decision(digest, 256, "cuda:0") == (1024, 8)
    # the descent tier accepts configs off the curated table
    assert sa.load_decision(digest, 256, "cuda:0", tier="coordesc") == (96, 8)
    # table tiers never read the descent record
    assert sa.load_decision(digest, 256, "cuda:0", tier="exhaustive") is None


def test_pick_config_exhaustive_tier_persists_and_replays(cache_root):
    digest = sa.program_digest([1], [], (0,))
    builds: list[tuple[int, int]] = []

    def build(config):
        builds.append(config)
        return SimpleNamespace(config=config)

    timings = {c: (0.1 if c == (128, 8) else 1.0) for c in
               EXHAUSTIVE_CANDIDATE_CONFIGS}
    config, launch = sa.pick_config(
        digest, 256, "cuda:0", build, [],
        candidates=EXHAUSTIVE_CANDIDATE_CONFIGS,
        tier="exhaustive",
        bench_fn=lambda launch, args: timings[launch.config],
    )
    assert config == (128, 8)

    # a same-tier reload hits the decision cache and never benchmarks
    builds.clear()
    config2, _ = sa.pick_config(
        digest, 256, "cuda:0", build, [],
        candidates=EXHAUSTIVE_CANDIDATE_CONFIGS,
        tier="exhaustive",
        bench_fn=lambda launch, args: pytest.fail("must not bench"),
    )
    assert config2 == (128, 8)
    assert builds == [(128, 8)]


def test_pick_config_applies_refiner(cache_root):
    digest = sa.program_digest([2], [], (0,))
    timings = {c: 1.0 for c in CANDIDATE_CONFIGS}

    def refiner(build, config, args):
        return (64, 2), SimpleNamespace(config=(64, 2))

    def build(config):
        return SimpleNamespace(config=config)

    config, launch = sa.pick_config(
        digest, 256, "cuda:0", build, [],
        refiner=refiner,
        tier="coordesc",
        bench_fn=lambda launch, args: timings[launch.config],
    )
    assert config == (64, 2)
    assert sa.load_decision(digest, 256, "cuda:0", tier="coordesc") == (64, 2)


def test_coordinate_descent_walks_to_better_neighbour():
    cost = {(128, 4): 1.0, (256, 4): 0.5, (256, 8): 0.6, (128, 8): 0.9,
            (64, 4): 1.2, (128, 2): 1.1}
    tuner = cd.CoordinateDescentTuner(
        cd.POINTWISE_FIELDS, bench_fn=lambda launch, args: cost[launch]
    )
    config, launch = tuner.refine(lambda c: c, (128, 4), [])
    assert config == (256, 4)
    assert launch == (256, 4)


def test_coordinate_descent_chains_across_fields():
    # (128, 4) -> (256, 4) through XBLOCK, then (256, 8) through warps: the
    # walk must pick the chain up without a restart
    cost = {(128, 4): 1.0, (256, 4): 0.5, (256, 8): 0.4, (512, 8): 0.9,
            (512, 4): 0.8, (128, 8): 0.9, (64, 4): 1.2, (128, 2): 1.1,
            (256, 2): 0.7}
    tuner = cd.CoordinateDescentTuner(
        cd.POINTWISE_FIELDS, bench_fn=lambda launch, args: cost[launch]
    )
    config, _ = tuner.refine(lambda c: c, (128, 4), [])
    assert config == (256, 8)


def test_coordinate_descent_disqualifies_unbuildable():
    def build(config):
        if config[0] > 256:
            raise RuntimeError("tile too large")
        return config

    cost = {(128, 4): 1.0, (256, 4): 0.5, (256, 8): 0.4, (128, 8): 0.9,
            (64, 4): 1.2, (128, 2): 1.1, (256, 2): 0.7}
    tuner = cd.CoordinateDescentTuner(
        cd.POINTWISE_FIELDS, bench_fn=lambda launch, args: cost[launch]
    )
    config, _ = tuner.refine(build, (128, 4), [])
    # every config above XBLOCK 256 benches as disqualified, so the walk
    # settles on the best buildable one
    assert config == (256, 8)


def test_coordinate_descent_bounds_and_radius():
    warps = cd.TunableField(1, "pow2", hi=32)
    tuner = cd.CoordinateDescentTuner((warps,))
    assert tuner._neighbour_values(warps, 32) == [16]
    assert tuner._neighbour_values(warps, 4) == [8, 2]

    stages = cd.TunableField(0, "stages", hi=4)
    tuner = cd.CoordinateDescentTuner((stages,))
    assert tuner._neighbour_values(stages, 4) == [3]
    assert tuner._neighbour_values(stages, 1) == [2]

    wide = cd.CoordinateDescentTuner((cd.TunableField(0, "pow2"),), radius=2)
    assert wide._neighbour_values(cd.TunableField(0, "pow2"), 128) == [
        256, 512, 64, 32
    ]

    with pytest.raises(ValueError):
        cd.CoordinateDescentTuner(cd.POINTWISE_FIELDS, radius=0)


def test_coordinate_descent_all_directions_sweep():
    # a diagonal-only basin: no single-field neighbour improves, only the
    # simultaneous (XBLOCK, warps) step does
    cost = {(128, 4): 1.0, (256, 8): 0.5}

    def bench(launch, args):
        try:
            return cost[launch]
        except KeyError:
            raise RuntimeError("unbuildable")

    stalled = cd.CoordinateDescentTuner(cd.POINTWISE_FIELDS, bench_fn=bench)
    config, _ = stalled.refine(lambda c: c, (128, 4), [])
    assert config == (128, 4)

    swept = cd.CoordinateDescentTuner(
        cd.POINTWISE_FIELDS, check_all_directions=True, bench_fn=bench
    )
    config, _ = swept.refine(lambda c: c, (128, 4), [])
    assert config == (256, 8)


def test_refiner_for_matches_pick_config_protocol(cache_root):
    digest = sa.program_digest([9], [], (0,))
    cost = {(128, 4): 1.0, (256, 4): 0.5, (256, 8): 0.4, (128, 8): 0.9,
            (64, 4): 1.2, (128, 2): 1.1, (256, 2): 0.7, (512, 4): 0.8,
            (512, 8): 0.9, (1024, 4): 1.0, (1024, 8): 1.0, (2048, 4): 1.0,
            (2048, 8): 1.0}
    config, launch = sa.pick_config(
        digest, 256, "cuda:0", lambda c: c, [],
        refiner=cd.refiner_for(
            cd.POINTWISE_FIELDS, bench_fn=lambda launch, args: cost[launch]
        ),
        tier="coordesc",
        bench_fn=lambda launch, args: cost[launch],
    )
    assert config == (256, 8)


def test_split_fields_dispatch_matches_config_arity():
    # the classic two-kernel (XBLOCK, warps) pair and the persistent
    # (XBLOCK, warps, NPROG) triple refine with their own field layouts
    assert cd.split_fields(2) == cd.POINTWISE_FIELDS
    assert cd.split_fields(3) == cd.SPLIT_FIELDS
    with pytest.raises(ValueError):
        cd.split_fields(4)


def test_coordinate_descent_refines_two_slot_split_winner():
    # a classic-form winner walks its own two slots; the walk never touches
    # a third (persistent-only) tuple position
    cost = {(2048, 8): 1.0, (4096, 8): 0.6, (2048, 4): 0.9, (1024, 8): 1.1}
    tuner = cd.CoordinateDescentTuner(
        cd.split_fields(2), bench_fn=lambda launch, args: cost[launch]
    )
    config, launch = tuner.refine(lambda c: c, (2048, 8), [])
    assert config == (4096, 8)
    assert launch == (4096, 8)
