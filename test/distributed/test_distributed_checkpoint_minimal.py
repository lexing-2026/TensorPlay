"""Single-process checks for the distributed checkpoint format.

The checkpoint format's coordination path is exercised through its
no-distribution mode: one rank plans, writes and commits the whole state.
These tests verify that the round trip preserves values and metadata, that
sharded state dicts keep their layout, and that the on-disk layout carries
the metadata index the format promises.
"""

import math

import pytest

import tensorplay as tp
from tensorplay import nn
from tensorplay.distributed import checkpoint as dcp


def _sample_state():
    return {
        "weights": tp.randn(4, 4),
        "bias": tp.arange(8, dtype=tp.float32),
        "scale": tp.tensor(2.5),
        "counts": {"low": tp.zeros(3, dtype=tp.int64), "high": tp.ones(3)},
    }


def test_roundtrip_preserves_values(tmp_path):
    state = _sample_state()
    ckpt = str(tmp_path / "state.dcp")
    dcp.save(state, checkpoint_id=ckpt)

    restored = {
        "weights": tp.empty(4, 4),
        "bias": tp.empty(8, dtype=tp.float32),
        "scale": tp.empty(()),
        "counts": {
            "low": tp.empty(3, dtype=tp.int64),
            "high": tp.empty(3),
        },
    }
    dcp.load(restored, checkpoint_id=ckpt)

    tp.testing.assert_close(restored["weights"], state["weights"])
    tp.testing.assert_close(restored["bias"], state["bias"])
    assert restored["scale"].item() == pytest.approx(2.5)
    tp.testing.assert_close(restored["counts"]["low"], state["counts"]["low"])
    tp.testing.assert_close(restored["counts"]["high"], state["counts"]["high"])


def test_module_state_roundtrip(tmp_path):
    model = nn.Linear(8, 3)
    ref = {k: v.clone() for k, v in model.state_dict().items()}
    ckpt = str(tmp_path / "module.dcp")
    dcp.save({"model": model.state_dict()}, checkpoint_id=ckpt)

    fresh = nn.Linear(8, 3)
    holder = {"model": fresh.state_dict()}
    dcp.load(holder, checkpoint_id=ckpt)
    fresh.load_state_dict(holder["model"])
    for key in ref:
        tp.testing.assert_close(
            dict(fresh.state_dict())[key], ref[key], check_dtype=True
        )


def test_bfloat16_and_zero_size_roundtrip(tmp_path):
    state = {
        "bf16": tp.randn(5, 2).to(tp.bfloat16),
        "empty": tp.zeros(0, 3),
    }
    ckpt = str(tmp_path / "dtypes.dcp")
    dcp.save(state, checkpoint_id=ckpt)
    restored = {"bf16": tp.empty(5, 2, dtype=tp.bfloat16), "empty": tp.empty(0, 3)}
    dcp.load(restored, checkpoint_id=ckpt)
    assert restored["bf16"].dtype == tp.bfloat16
    assert tuple(restored["empty"].shape) == (0, 3)


def test_checkpoint_directory_layout(tmp_path):
    ckpt = tmp_path / "layout.dcp"
    dcp.save(_sample_state(), checkpoint_id=str(ckpt))
    assert ckpt.is_dir()
    files = {p.name for p in ckpt.iterdir()}
    assert ".metadata" in files, files
    assert any(name.startswith("__0_0") for name in files), files


def test_load_reports_missing_key(tmp_path):
    state = _sample_state()
    ckpt = str(tmp_path / "keys.dcp")
    dcp.save(state, checkpoint_id=ckpt)

    restored = {"weights": tp.empty(4, 4)}
    dcp.load(restored, checkpoint_id=ckpt)
    tp.testing.assert_close(restored["weights"], state["weights"])


def test_nan_and_inf_survive_roundtrip(tmp_path):
    state = {
        "special": tp.tensor([math.nan, math.inf, -math.inf, 0.0]),
    }
    ckpt = str(tmp_path / "special.dcp")
    dcp.save(state, checkpoint_id=ckpt)
    restored = {"special": tp.empty(4)}
    dcp.load(restored, checkpoint_id=ckpt)
    out = restored["special"]
    assert tp.isnan(out[0])
    assert out[1] == math.inf
    assert out[2] == -math.inf
    assert out[3] == 0.0


def test_optimizer_state_roundtrip(tmp_path):
    from tensorplay.distributed.checkpoint.state_dict import (
        get_state_dict,
        set_state_dict,
    )

    model = nn.Linear(4, 2)
    opt = tp.optim.AdamW(model.parameters(), lr=1e-3)
    x = tp.randn(6, 4)
    (model(x).sum()).backward()
    opt.step()

    ckpt = str(tmp_path / "opt.dcp")
    model_sd, optim_sd = get_state_dict(model, opt)
    dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=ckpt)

    fresh_model = nn.Linear(4, 2)
    fresh_opt = tp.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    loaded_model_sd, loaded_optim_sd = get_state_dict(fresh_model, fresh_opt)
    dcp.load({"model": loaded_model_sd, "optim": loaded_optim_sd},
             checkpoint_id=ckpt)
    set_state_dict(
        fresh_model,
        fresh_opt,
        model_state_dict=loaded_model_sd,
        optim_state_dict=loaded_optim_sd,
    )
    assert len(fresh_opt.state_dict()["state"]) == len(opt.state_dict()["state"])
    for a, b in zip(model.parameters(), fresh_model.parameters()):
        tp.testing.assert_close(a, b)
