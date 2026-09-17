"""

End-to-end paired test on a local Qwen3-MoE checkpoint.

The reference side executes the canonical modeling stack unmodified.  The
native side installs a module-level alias before any modeling import, so
the same stack binds every framework symbol to tensorplay at import
time — no upstream source is copied or rewritten.  The alias re-exports
the tensorplay surface under the expected names and fills the few missing
spellings (type aliases, single-argument predicate selection, the
serialization loader entry).

Outputs:
  match   — logits/argmax agreement plus greedy generation agreement
  speed   — prefill latency and decode ms/token, interleaved min-of-R
"""

import argparse
import importlib
import importlib.abc
import importlib.util
import io
import json
import re
import sys
import time
import types
from pathlib import Path

# Pin the whole package to the repo tree: the editable finder maps the
# package root here but resolves submodules to the installed copy, which
# mixes stale files into the import.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

SCENARIOS = {"short": (128, 24), "medium": (512, 12), "long": (1024, 6)}
PROMPT = "The little robot walked into the lab and"
REF_CACHE = Path("/tmp/qwen3_moe_e2e_ref.npz")


def encode_ids(model_dir):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    return np.asarray([tok.encode(PROMPT).ids], dtype=np.int64), tok


def greedy_steps(fwd, ids, steps, cat):
    cur = ids
    for _ in range(steps):
        nxt = fwd(cur)[:, -1:, :].argmax(-1)
        cur = cat([cur, nxt], 1)
    return cur


# --------------------------------------------------------------------------
# reference side: canonical stack on its home runtime
# --------------------------------------------------------------------------

def run_reference(args):
    import torch
    from transformers import AutoModelForCausalLM

    model_dir = args.model
    ids_np, tok = encode_ids(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.float16).cuda().eval()
    ids = torch.from_numpy(ids_np).cuda()

    def fwd(t):
        with torch.no_grad():
            return model(t).logits

    logits = fwd(ids).float().cpu().numpy()
    gen = greedy_steps(fwd, ids, args.gen_tokens, torch.cat)
    gen = gen.cpu().numpy()[0, ids_np.shape[1]:]

    def bench(prefill, steps):
        ids_r = torch.randint(0, model.config.vocab_size, (1, prefill),
                              device="cuda")

        def sync():
            torch.cuda.synchronize()

        def best(fn, runs):
            fn(); sync()
            b = 1e18
            for _ in range(runs):
                t0 = time.perf_counter(); fn(); sync()
                b = min(b, (time.perf_counter() - t0) * 1e3)
            return b

        with torch.no_grad():
            pf = best(lambda: model(ids_r).logits, args.timing_runs)

            def dec():
                cur = ids_r
                for _ in range(steps):
                    cur = torch.cat([cur, model(cur).logits[:, -1:, :].argmax(-1)], 1)
            dc = best(dec, args.timing_runs)
        return pf, dc / steps

    np.savez(REF_CACHE, logits=logits, gen=gen)
    print(f"reference: logits saved to {REF_CACHE} "
          f"shape={logits.shape} top1_last={int(logits[0, -1].argmax())}")
    print(f"reference greedy {args.gen_tokens}: "
          f"{tok.decode(gen.tolist())!r}")
    for name, (prefill, steps) in SCENARIOS.items():
        pf, mt = bench(prefill, steps)
        print(f"  {name}: decode={mt:7.2f} ms/tok prefill={pf:6.2f} ms")
    return logits, gen


# --------------------------------------------------------------------------
# native side: import-space alias
# --------------------------------------------------------------------------

class _AliasLoader(importlib.abc.Loader):
    def __init__(self, mod):
        self._mod = mod

    def create_module(self, spec):
        return self._mod

    def exec_module(self, module):
        pass


def _register(fullname, mod):
    """Present an aliased module to the import system with full metadata."""
    loader = _AliasLoader(mod)
    spec = importlib.util.spec_from_loader(fullname, loader)
    mod.__name__ = fullname
    mod.__spec__ = spec
    mod.__loader__ = loader
    mod.__package__ = fullname.rpartition(".")[0]
    sys.modules[fullname] = mod
    return mod


class _Permissive:
    """Stand-in for optional framework corners the model path never uses."""
    def __getattr__(self, name):
        return self

    def __call__(self, *a, **k):
        return self

    def __getitem__(self, k):
        return self

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False


def _stub(fullname):
    mod = types.ModuleType(fullname)
    mod.__getattr__ = lambda name: _Permissive()
    _register(fullname, mod)
    return mod


def install_alias():
    """Bind the framework module name to tensorplay for all later imports."""
    import tensorplay as tp

    def where(cond, input=None, other=None):
        if input is None:
            idx = tp.nonzero(cond)
            return tuple(idx[:, i] for i in range(len(idx.shape)))
        return tp.where(cond, input, other)

    class _Alias(types.ModuleType):
        def __getattr__(self, name):
            return getattr(tp, name)

    class _LegacyTensorAlias:
        """Stand-in for the legacy dtype-tensor spellings, which the native
        stack does not carry; only annotation evaluation needs the name."""

    alias = _Alias("torch")
    alias.where = where
    alias.LongTensor = _LegacyTensorAlias
    alias.FloatTensor = _LegacyTensorAlias
    alias.HalfTensor = _LegacyTensorAlias
    alias.BFloat16Tensor = _LegacyTensorAlias
    alias.IntTensor = _LegacyTensorAlias
    alias.BoolTensor = _LegacyTensorAlias
    alias.__path__ = tp.__path__
    alias.__file__ = tp.__file__
    _register("torch", alias)
    _register("torch.nn", tp.nn)
    _register("torch.nn.functional", tp.nn.functional)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if not fullname.startswith("torch.") or fullname in sys.modules:
                return None
            try:
                mod = importlib.import_module("tensorplay." + fullname[6:])
            except ImportError:
                return None
            _register(fullname, mod)
            return mod.__spec__

    sys.meta_path.insert(0, _Finder())

    # serialization loader entry routed through the native archive reader
    import tensorplay.serialization as tp_ser

    st = types.ModuleType("safetensors.torch")

    def _to_device(sd, device):
        return sd if device == "cpu" else {k: v.to(device) for k, v in sd.items()}

    def load_file(filename, device="cpu"):
        return _to_device(tp_ser.load(filename), device)

    def load(data, device="cpu"):
        return _to_device(tp_ser.load(io.BytesIO(data)), device)

    def save_file(tensors, filename, metadata=None):
        tp_ser.save(dict(tensors), filename, metadata=metadata)

    st.load_file = load_file
    st.load = load
    st.save_file = save_file
    st.save = save_file
    st.storage_ptr = lambda tensor: tensor.untyped_storage().data_ptr()
    st.storage_size = lambda tensor: tensor.untyped_storage().nbytes()
    _register("safetensors.torch", st)

    # The vision companion package ships compiled kernels bound to the
    # reference runtime, so the real import cannot work under the alias.
    # Only the interpolation/decode names the modeling stack reads at import
    # time are provided; the image paths themselves stay unimplemented.
    import enum as _enum

    tv = types.ModuleType("torchvision")
    tv.__version__ = "0.0.0+alias"
    tv.__path__ = []

    class ImageReadMode(_enum.Enum):
        UNCHANGED = 0
        GRAY = 1
        GRAY_ALPHA = 2
        RGB = 3
        RGB_ALPHA = 4

    def _unavailable(*args, **kwargs):
        raise NotImplementedError(
            "image decoding is not provided under the alias import"
        )

    tv_io = types.ModuleType("torchvision.io")
    tv_io.ImageReadMode = ImageReadMode
    tv_io.decode_image = _unavailable
    tv.io = tv_io

    class InterpolationMode(_enum.Enum):
        NEAREST = "nearest"
        NEAREST_EXACT = "nearest-exact"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        BOX = "box"
        HAMMING = "hamming"
        LANCZOS = "lanczos"

    tv_tr = types.ModuleType("torchvision.transforms")
    tv_tr.InterpolationMode = InterpolationMode
    tv.transforms = tv_tr

    tv_trf = types.ModuleType("torchvision.transforms.functional")
    tv_trf.pil_to_tensor = _unavailable
    tv_tr.functional = tv_trf

    for name, mod in (
        ("torchvision", tv),
        ("torchvision.io", tv_io),
        ("torchvision.transforms", tv_tr),
        ("torchvision.transforms.functional", tv_trf),
    ):
        _register(name, mod)


def build_native_model(model_dir):
    """Patch the few machinery pieces, then build and load the model."""
    import tensorplay as tp
    import tensorplay.serialization as tp_ser
    import transformers
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

    import transformers.masking_utils as _mask
    _mask.create_causal_mask = lambda **kw: None
    _mask.create_sliding_window_causal_mask = lambda **kw: None

    def sdpa_forward(module, query, key, value, attention_mask=None,
                     dropout=0.0, scaling=None, sliding_window=None, **kw):
        groups = getattr(module, "num_key_value_groups", 1)
        if groups > 1:
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        out = tp.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=(attention_mask is None))
        # interface contract: sequence-major [B, S, H*D] back to the caller
        out = out.transpose(1, 2).contiguous()
        return out.reshape(query.shape[0], query.shape[2], -1), None

    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_forward

    from transformers.models.qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    # the generic init pass is bypassed: every parameter arrives from the
    # checkpoint below, and the tied head is shared explicitly
    PreTrainedModel.post_init = lambda self: None
    config = Qwen3MoeConfig.from_pretrained(str(model_dir))
    config._attn_implementation = "sdpa"
    model = Qwen3MoeForCausalLM(config)
    model.lm_head.weight = model.model.embed_tokens.weight

    sd = tp_ser.load(str(model_dir / "model.safetensors"))

    # The checkpoint stores one nn.Linear per expert; the model consumes
    # grouped weights shaped (num_experts, ...).  Regroup gate/up per expert
    # along dim 1 and stack the expert dimension along dim 0.
    pattern = re.compile(
        r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
    )
    per_expert = {}
    merged_sd = {}
    for key, value in sd.items():
        m = pattern.match(key)
        if m is None:
            merged_sd[key] = value
            continue
        layer, expert, proj = m.group(1), int(m.group(2)), m.group(3)
        per_expert.setdefault((layer, proj), {})[expert] = value
    for (layer, proj), tensors in per_expert.items():
        stacked = tp.stack([tensors[e] for e in sorted(tensors)], 0)
        if proj == "down_proj":
            merged_sd[f"model.layers.{layer}.mlp.experts.down_proj"] = stacked
        else:
            merged_sd.setdefault(
                f"model.layers.{layer}.mlp.experts.gate_up_proj", []
            ).append(stacked)
    for key, parts in list(merged_sd.items()):
        if isinstance(parts, list):
            merged_sd[key] = tp.cat(parts, 1)

    missing, unexpected = model.load_state_dict(merged_sd, strict=False)
    if missing or unexpected:
        print(f"native: missing={list(missing)} unexpected={list(unexpected)}")
    model = model.to("cuda").eval()
    return model


def run_native(args):
    install_alias()
    model = build_native_model(args.model)
    import tensorplay as tp

    ids_np, tok = encode_ids(args.model)
    ids = tp.from_numpy(ids_np).to("cuda")

    def fwd(t):
        with tp.no_grad():
            return model(input_ids=t, use_cache=False).logits

    logits = fwd(ids).float().cpu().numpy()
    # Co-tenant training jobs shrink the free pool between runs; release
    # every cached block before the autoregressive loop peaks.
    import gc as _gc
    _gc.collect()
    tp.cuda.empty_cache()
    gen = greedy_steps(fwd, ids, args.gen_tokens, tp.cat)
    gen = gen.cpu().numpy()[0, ids_np.shape[1]:]

    ref = np.load(REF_CACHE)
    d = np.abs(ref["logits"] - logits)
    match = (ref["logits"].argmax(-1) == logits.argmax(-1)).mean()
    agree = int((ref["gen"] == gen).sum())
    print(f"match vs reference: shape={logits.shape} max|diff|={d.max():.5f} "
          f"mean|diff|={d.mean():.6f} top1-match={match:.4f} "
          f"greedy-agree={agree}/{len(ref['gen'])}")
    print(f"native greedy {args.gen_tokens}: {tok.decode(gen.tolist())!r}")

    for name, (prefill, steps) in SCENARIOS.items():
        ids_r = tp.from_numpy(
            np.random.randint(0, model.config.vocab_size, (1, prefill))
        ).to("cuda")

        def sync():
            tp.cuda.synchronize()

        def best(fn, runs):
            fn(); sync()
            b = 1e18
            for _ in range(runs):
                t0 = time.perf_counter(); fn(); sync()
                b = min(b, (time.perf_counter() - t0) * 1e3)
            return b

        with tp.no_grad():
            pf = best(lambda: fwd(ids_r), args.timing_runs)

            def dec():
                cur = ids_r
                for _ in range(steps):
                    cur = tp.cat([cur, fwd(cur)[:, -1:, :].argmax(-1)], 1)
            dc = best(dec, args.timing_runs)
        print(f"  {name}: decode={dc / steps:7.2f} ms/tok prefill={pf:6.2f} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("side", choices=("ref", "native"))
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--timing-runs", type=int, default=3)
    args = ap.parse_args()
    if args.side == "ref":
        run_reference(args)
    else:
        if not REF_CACHE.exists():
            raise SystemExit("reference cache missing; run the ref side first")
        run_native(args)


if __name__ == "__main__":
    main()
