"""Testing helpers for ONNX export.

Runs an exported model under onnxruntime and compares against eager
execution; useful from test suites and REPL debugging alike.
"""

from __future__ import annotations

import io
from typing import Any, Sequence

import numpy as np

from ._verify import VerificationError, verify_model
from .errors import OnnxExporterError

__all__ = ["export_to_bytes", "run_model_test"]


def export_to_bytes(model_and_args: Sequence[Any], **kwargs) -> bytes:
    """Export ``(model, *args)`` and return the serialized ``ModelProto``.

    Keyword arguments forward to :func:`tensorplay.onnx.export`.
    """
    from . import export

    proto = export(tuple(model_and_args), None, **kwargs)
    if proto is None:
        raise OnnxExporterError(
            "export_to_bytes(): the export wrote to a file instead of "
            "returning the model proto")
    buffer = io.BytesIO()
    buffer.write(proto.SerializeToString())
    return buffer.getvalue()


def run_model_test(
    model: Any,
    *example_args: Any,
    input_names: Sequence[str] | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> list[np.ndarray]:
    """Export ``model``, execute it under onnxruntime, and return outputs.

    Raises :class:`VerificationError` when the runtime outputs disagree with
    eager execution beyond the given tolerances.

    Args:
        model: the model to export and execute.
        example_args: positional example inputs, capturing the model.
        input_names: names for the graph inputs, in argument order.
        rtol/atol: tolerances for the eager-versus-runtime comparison.
    """
    import onnxruntime as ort

    from . import export
    from ._type_mapping import _to_numpy

    if input_names is None:
        raise ValueError(
            "run_model_test(): input_names is required to name the graph "
            "inputs")
    example_inputs = dict(zip(input_names, example_args))

    proto = export((model, *example_args), None, input_names=list(input_names))
    if proto is None:
        raise OnnxExporterError("run_model_test(): export returned no model")

    expected = model(*example_args)
    verify_model(
        proto,
        expected=expected,
        input_names=[value.name for value in proto.graph.input],
        example_inputs=example_inputs,
        rtol=rtol,
        atol=atol,
    )

    session = ort.InferenceSession(proto.SerializeToString())
    feeds = {
        value.name: np.asarray(_to_numpy(example_inputs[value.name]))
        for value in proto.graph.input
        if value.name in example_inputs
    }
    return list(session.run(None, feeds))
