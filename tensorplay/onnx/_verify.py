"""Numerical verification of an exported model against eager TensorPlay."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ._type_mapping import _to_numpy

__all__ = ["VerificationError", "VerificationResult", "verify_model"]


class VerificationError(AssertionError):
    """Raised when the exported graph disagrees with eager execution."""


@dataclass
class VerificationResult:
    """Per-output comparison between eager TensorPlay and onnxruntime."""

    matched: bool
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0
    mismatches: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.matched


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    return [value]


def _run_onnxruntime(model: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise VerificationError(
            "verify=True requires onnxruntime; install it or pass verify=False"
        ) from exc

    buffer = io.BytesIO()
    from onnx import save_model

    save_model(model, buffer)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        buffer.getvalue(), options, providers=["CPUExecutionProvider"]
    )
    return session.run(None, feeds)


def verify_model(
    model: Any,
    *,
    expected: Any,
    input_names: Sequence[str],
    example_inputs: dict[str, Any],
    rtol: float = 1e-4,
    atol: float = 1e-5,
    raise_on_mismatch: bool = True,
) -> VerificationResult:
    """Compare ``model`` under onnxruntime against the eager result.

    Args:
        model: the built ``ModelProto``.
        expected: what eager TensorPlay produced for ``example_inputs`` —
            a tensor, or an arbitrarily nested tuple/list of them.
        input_names: ONNX graph input names, in placeholder order.
        example_inputs: placeholder name -> example tensor.
        rtol/atol: tolerances forwarded to :func:`numpy.allclose`.
        raise_on_mismatch: raise :class:`VerificationError` instead of
            returning a failed result.
    """

    feeds = {
        onnx_name: _to_numpy(example_inputs[placeholder])
        for onnx_name, placeholder in zip(input_names, example_inputs)
    }
    actual = _run_onnxruntime(model, feeds)
    expected = [_to_numpy(item) for item in _flatten(expected)]

    result = VerificationResult(matched=True)
    if len(expected) != len(actual):
        result.matched = False
        result.mismatches.append(
            f"output count differs: eager produced {len(expected)}, "
            f"onnxruntime produced {len(actual)}"
        )
    for index, (reference, candidate) in enumerate(zip(expected, actual)):
        candidate = np.asarray(candidate)
        if reference.shape != candidate.shape:
            result.matched = False
            result.mismatches.append(
                f"output {index}: shape {reference.shape} != {candidate.shape}"
            )
            continue
        if reference.dtype.kind in "fc" or candidate.dtype.kind in "fc":
            difference = np.abs(
                reference.astype(np.float64) - candidate.astype(np.float64)
            )
            scale = np.maximum(np.abs(reference.astype(np.float64)), 1e-12)
            result.max_abs_diff = max(result.max_abs_diff, float(difference.max(initial=0.0)))
            result.max_rel_diff = max(
                result.max_rel_diff, float((difference / scale).max(initial=0.0))
            )
            close = np.allclose(reference, candidate, rtol=rtol, atol=atol, equal_nan=True)
        else:
            close = bool(np.array_equal(reference, candidate))
        if not close:
            result.matched = False
            result.mismatches.append(
                f"output {index}: values differ (max abs diff {result.max_abs_diff:.3e})"
            )

    if not result.matched and raise_on_mismatch:
        raise VerificationError(
            "exported ONNX model does not match eager execution:\n  "
            + "\n  ".join(result.mismatches)
        )
    return result
