"""Find which part of the prefill the reference disagrees with.

One model load costs minutes, so the prefill is run several times inside a
single process with one piece of the code put back the way it was written
before: the rotary tables gathered by address instead of by a row lookup, the
projection weights recovered on every forward instead of held, the module's own
normalization kernel, the module's own activation, no rotation at all.  Each
run is compared against the stored reference scores, so the piece whose
restoration makes the difference collapse is the one that is wrong.

A run also reports the shape of its own logits -- how many entries are not
finite, the extremes and the mean -- because a tower that has collapsed to a
constant and a tower that has overflowed look the same through a single
distance number.

    python probe_isolation.py --model <dir> --ref ref.npz [--device mlu]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402

from qwen3vl import model as M  # noqa: E402


def pick_device(name):
    if name == "mlu":
        import tensorplay_mlu

        tensorplay_mlu.load()
    return name


def prefill_logits(model, ids, device):
    from tensorplay.nn.cache import DynamicCache

    positions = tp.arange(ids.shape[1], device=device).unsqueeze(0)
    cache = DynamicCache()
    with tp.no_grad():
        logits = model(ids, position_ids=positions, past_key_values=cache)
    return logits[0, ids.shape[1] - 1].to(tp.float32).cpu().numpy()


def rope_by_address(self, x, position_ids=None, max_position=None):
    """The table read with the address tensor itself."""
    if position_ids is None:
        raise ValueError("a rotary table needs positions")
    cos_table, sin_table = self._tables(x.device)
    if position_ids.ndim == 2:
        cos = cos_table[position_ids]
        sin = sin_table[position_ids]
    else:
        table_cos = cos_table[position_ids]
        table_sin = sin_table[position_ids]
        cos = table_cos[0].clone()
        sin = table_sin[0].clone()
        for axis in (1, 2):
            length = self.mrope_section[axis] * 3
            cos[..., axis:length:3] = table_cos[axis][..., axis:length:3]
            sin[..., axis:length:3] = table_sin[axis][..., axis:length:3]
    return cos.to(x.dtype), sin.to(x.dtype)


def no_rotation(x, cos, sin):
    """The rotation skipped, to see whether the traffic through it is the one."""
    return x


def weights_recovered(self, x):
    """The values recovered from the codes on every forward."""
    dense = self.dense_weight().to(x.dtype)
    return x @ dense.t().contiguous()


def weights_held_transposed(self, x):
    """The values held in the other order and read directly."""
    return x @ self._dense_cache


def norm_module(self, x):
    """The normalization kernel the module reaches."""
    return tp.nn.functional.rms_norm(x, self.normalized_shape, self.weight, self.eps)


def mlp_module_activation(self, x):
    """The feed-forward with the activation reached through the module."""
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def describe(scores):
    finite = np.isfinite(scores)
    return (
        f"nonfinite {int((~finite).sum()):6d}  "
        f"min {np.nanmin(scores):10.4f}  max {np.nanmax(scores):10.4f}  "
        f"mean {np.nanmean(scores):9.4f}  argmax {int(np.nanargmax(scores))}"
    )


def run(name, model, ids, device, reference, extra=""):
    try:
        scores = prefill_logits(model, ids, device)
    except Exception as exc:  # noqa: BLE001
        print(f"  {name:28} {type(exc).__name__}: {str(exc)[:70]}")
        return
    difference = float(np.abs(scores - reference).max())
    print(f"  {name:28} diff {difference:10.4f}  {describe(scores)}{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--device", default="mlu")
    args = ap.parse_args()

    device = pick_device(args.device)
    ref = np.load(args.ref, allow_pickle=True)
    prompt_ids = ref["prompt_ids"].reshape(1, -1)
    reference = np.asarray(ref["scores"][0], dtype=np.float32)
    ids = tp.from_numpy(prompt_ids.astype(np.int64)).to(device=device)

    dtype = tp.float16 if device == "mlu" else tp.bfloat16
    model, _config = M.load_qwen3vl(
        args.model, dtype=dtype, device=device, weight_only=True, progress=print
    )
    cached = M.cache_dense_weights(model, dtype)
    print(f"recovered {cached} projection weights once")
    count = M.use_written_normalization(model, "written")
    print(f"normalizations run through the written body ({count})")

    norms = [m for m in model.language_model.modules() if isinstance(m, M.nn.RMSNorm)]
    projections = [m for m in model.modules() if isinstance(m, M.WeightOnlyInt4Linear)]

    print(f"\nreference logits: {describe(reference)}\n")

    report = lambda label: run(label, model, ids, device, reference)
    report("as it stands")

    saved_rope = M.MultimodalRotaryEmbedding.forward
    M.MultimodalRotaryEmbedding.forward = rope_by_address
    report("rotary read by address")
    M.MultimodalRotaryEmbedding.forward = saved_rope

    saved_forward = M.WeightOnlyInt4Linear.forward
    M.WeightOnlyInt4Linear.forward = weights_recovered
    report("weights recovered per forward")
    M.WeightOnlyInt4Linear.forward = weights_held_transposed
    for module in projections:
        module._dense_cache = module._dense_cache.t().contiguous()
    report("weights held transposed")
    for module in projections:
        module._dense_cache = module._dense_cache.t().contiguous()
    M.WeightOnlyInt4Linear.forward = saved_forward

    saved_norms = [module.forward for module in norms]
    for module in norms:
        module.forward = norm_module.__get__(module)
    report("module normalization kernel")
    for module, body in zip(norms, saved_norms):
        module.forward = body

    saved_mlp = M.Qwen3VLMLP.forward
    M.Qwen3VLMLP.forward = mlp_module_activation
    report("module activation")
    M.Qwen3VLMLP.forward = saved_mlp

    saved_rotate = M._rotate_heads
    M._rotate_heads = no_rotation
    report("rotation skipped")
    M._rotate_heads = saved_rotate

    trace = []
    saved_layer = M.Qwen3VLDecoderLayer.forward
    saved_language = M.Qwen3VLLanguageModel.forward

    def traced_layer(self, hidden_states, **kwargs):
        out = saved_layer(self, hidden_states, **kwargs)
        trace.append(float(out.to(tp.float32).abs().max().item()))
        return out

    def traced_language(self, hidden_states, **kwargs):
        del trace[:]
        out = saved_language(self, hidden_states, **kwargs)
        print(f"\n  hidden state after each block (max abs), then the final norm:")
        print("   ", " ".join(f"{value:.3g}" for value in trace))
        print(f"    final norm max abs {float(out.to(tp.float32).abs().max().item()):.6g}")
        return out

    M.Qwen3VLDecoderLayer.forward = traced_layer
    M.Qwen3VLLanguageModel.forward = traced_language
    report("per-block trace")
    M.Qwen3VLDecoderLayer.forward = saved_layer
    M.Qwen3VLLanguageModel.forward = saved_language


if __name__ == "__main__":
    main()
