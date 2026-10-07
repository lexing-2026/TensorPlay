# mypy: allow-untyped-defs
"""Autoregressive generation loop.

``generate`` drives a causal language model token by token: a prefill pass over
the prompt, then ``max_new_tokens`` decode steps. When the model supports a
cache the prompt is processed once and every following step feeds a single
token; otherwise each step re-feeds the whole sequence.

Model contract
--------------
``model(input_ids, position_ids=None, past_key_values=None)`` returns logits of
shape ``[batch, seq, vocab]``. ``position_ids`` is ``[batch, seq]``. A model
that wants caching accepts ``past_key_values``, mutates it, and should expose
``create_empty_cache()`` returning its preferred cache object (used by
:func:`generate` when the caller passes none).
"""

from __future__ import annotations

import tensorplay
from tensorplay import Tensor

from .cache import DynamicCache

__all__ = ["generate", "top_k_top_p_filtering"]


def _filter_logits(logits: Tensor, top_k: int, top_p: float) -> Tensor:
    """Set the probability mass below the top-``k`` / nucleus (top-``p``)
    thresholds to ``-inf``.

    ``top_k <= 0`` keeps every candidate; ``top_p >= 1.0`` disables the nucleus
    cutoff. Both filters operate on a copy, leaving the input untouched.
    """
    logits = logits.clone()
    vocab = logits.shape[-1]
    if top_k <= 0 or top_k > vocab:
        top_k = vocab
    if top_k < vocab:
        kth = tensorplay.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = tensorplay.sort(logits, dim=-1, descending=False)
        cum = tensorplay.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # With the ascending sort the cumulative mass ends at 1.0, so the last
        # candidate is never removed and the support is never empty.
        remove_sorted = cum <= (1.0 - top_p)
        remove = tensorplay.zeros_like(remove_sorted).scatter_(-1, sorted_idx, remove_sorted)
        logits = logits.masked_fill(remove, float("-inf"))
    return logits


def top_k_top_p_filtering(logits: Tensor, top_k: int = 0, top_p: float = 1.0) -> Tensor:
    """Public wrapper around the logit filters used by :func:`generate`."""
    return _filter_logits(logits, top_k, top_p)


def _apply_repetition_penalty(logits: Tensor, sequences: Tensor, penalty: float) -> Tensor:
    """Down-weight tokens already present in each row's sequence.

    Scores above zero are divided by ``penalty``, scores below zero are
    multiplied, so both over- and under-confident repeats are discouraged.
    """
    if penalty == 1.0:
        return logits
    scores = tensorplay.gather(logits, 1, sequences)
    scores = tensorplay.where(scores > 0, scores / penalty, scores * penalty)
    return logits.clone().scatter_(1, sequences, scores)


@tensorplay.no_grad()
def generate(
    model,
    input_ids: Tensor,
    max_new_tokens: int = 20,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    use_cache: bool = True,
    past_key_values=None,
) -> Tensor:
    """Generate ``max_new_tokens`` tokens for each row of ``input_ids``.

    Args:
        model: causal LM following the module contract described above.
        input_ids: prompt tokens, ``[batch, seq]``.
        max_new_tokens: number of decode steps.
        do_sample: sample from the (filtered) distribution when True,
            otherwise take the per-step argmax.
        temperature: softmax temperature for sampling; values below ~1e-5
            fall back to greedy decoding.
        top_k: sampling support limited to the ``top_k`` highest logits
            (``0`` keeps the full vocabulary).
        top_p: nucleus cutoff keeping the smallest prefix of the sorted
            distribution whose mass exceeds ``top_p`` (``1.0`` disables).
        repetition_penalty: multiplicative penalty applied to every token
            already emitted in the same row.
        eos_token_id: stop token; rows that emit it keep receiving
            ``pad_token_id`` for the remaining steps.
        pad_token_id: filler for finished rows (defaults to ``eos_token_id``,
            then ``0``).
        use_cache: run through the model's cache when it provides one.
        past_key_values: optional pre-built cache; a fresh one is created via
            ``model.create_empty_cache()`` or :class:`DynamicCache` otherwise.

    Returns:
        Sequence tensor ``[batch, prompt_len + max_new_tokens]`` containing the
        prompt followed by the generated (or padding) tokens.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be [batch, seq], got shape {tuple(input_ids.shape)}")
    if max_new_tokens <= 0:
        return input_ids
    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else 0

    if use_cache and past_key_values is None:
        create = getattr(model, "create_empty_cache", None)
        past_key_values = create() if create is not None else DynamicCache()

    sequences = input_ids
    batch = input_ids.shape[0]
    device = input_ids.device
    finished = tensorplay.zeros(batch, dtype=tensorplay.bool, device=device)

    # Prefill consumes the whole prompt; every decode step consumes one token.
    next_input = input_ids
    for step in range(max_new_tokens):
        position_ids = tensorplay.arange(
            sequences.shape[1] - next_input.shape[1],
            sequences.shape[1],
            device=device,
        ).unsqueeze(0).expand(batch, -1)
        logits = model(
            next_input, position_ids=position_ids, past_key_values=past_key_values
        )[:, -1, :]

        if not do_sample or temperature <= 1e-5:
            token = logits.argmax(dim=-1)
        elif repetition_penalty != 1.0:
            penalized = _apply_repetition_penalty(logits.float(), sequences, repetition_penalty)
            filtered = _filter_logits(penalized, top_k, top_p)
            token = tensorplay.multinomial(tensorplay.softmax(filtered, dim=-1), num_samples=1).squeeze(1)
        else:
            token = tensorplay.sample(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

        token = tensorplay.where(finished, tensorplay.full_like(token, pad_token_id), token)
        sequences = tensorplay.cat([sequences, token.unsqueeze(1)], dim=1)

        if eos_token_id is not None:
            finished = finished | (token == eos_token_id)
            if bool(finished.all()):
                break

        # Cached models consume one token per step; without a cache the model
        # must re-see the entire prefix, causal masking keeps it consistent.
        next_input = token.unsqueeze(1) if use_cache else sequences
    return sequences
