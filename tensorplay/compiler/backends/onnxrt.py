"""ONNX Runtime backend selected by ``tensorplay.compile(backend="onnxrt")``.

Thin core shell: the canonical :class:`GraphModule` is exported to an ONNX
model, compiled once into an ONNX Runtime session, and executed through
DLPack (TensorPlay tensors implement ``__dlpack__``/``from_dlpack``; the
runtime consumes both directions natively), so data crosses without host
copies for contiguous device buffers the runtime can address directly.

The runtime is an optional dependency.  A missing dependency hides the
backend from :func:`list_backends` and selecting it by name raises an
error that names the package to install.  Only inference regions lower;
training regions are wrapped ahead-of-time by the frontend (forward here,
backward as traced).

Execution providers follow installation: the CPU provider always exists;
CUDA/TensorRT providers are tried in order when the runtime exposes them
and the region lives on that device.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ...graph import GraphModule
from .._core.registry import BackendCapabilities, declares_capabilities

__all__ = ["onnxrt"]

log = logging.getLogger(__name__)


def _providers_for(device_type: str) -> list[str]:
    import onnxruntime

    available = onnxruntime.get_available_providers()
    ordered: list[str] = []
    for provider, device_prefix in (
        ("TensorrtExecutionProvider", "tensorrt"),
        ("CUDAExecutionProvider", "cuda"),
        ("CPUExecutionProvider", "cpu"),
    ):
        if provider in available and (device_type == device_prefix or device_prefix == "cpu"):
            ordered.append(provider)
    return ordered or ["CPUExecutionProvider"]


@declares_capabilities(
    BackendCapabilities(
        inference_only=True,
        handles_training=False,
        optional_deps=("onnxruntime",),
    )
)
def onnxrt(
    graph_module: GraphModule,
    example_inputs: list[Any],
    **kwargs: Any,
) -> Callable[..., Any]:
    if kwargs:
        log.warning("onnxrt backend ignoring extra kwargs %s", kwargs)

    try:
        import onnxruntime
    except ImportError as exc:
        raise RuntimeError(
            "The onnxrt backend requires onnxruntime. Install it with "
            "`pip install onnxruntime` (CPU) or `pip install onnxruntime-gpu` "
            "(CUDA), then retry."
        ) from exc

    import numpy as np

    from tensorplay import onnx as tp_onnx

    device_type = "cpu"
    for sample in example_inputs:
        device = getattr(sample, "device", None)
        if device is not None:
            device_type = str(getattr(device, "type", device))
            break

    model = tp_onnx.export((graph_module, *example_inputs))
    session = onnxruntime.InferenceSession(
        model.SerializeToString(),
        providers=_providers_for(device_type),
    )
    input_names = [value.name for value in session.get_inputs()]
    single_output = len(session.get_outputs()) == 1

    def _to_array(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value
        target = value
        if getattr(target.device, "is_cuda", lambda: False)():
            # The runtime consumes host buffers through the plain session API;
            # IOBinding zero-copy is future work.
            target = target.cpu()
        return np.from_dlpack(target.contiguous())

    def run(*args: Any, **call_kwargs: Any) -> Any:
        import tensorplay

        feeds = {name: _to_array(value) for name, value in zip(input_names, args)}
        outputs = session.run(None, feeds)
        tensors = tuple(
            tensorplay.from_numpy(np.array(output, copy=True)) for output in outputs
        )
        if device_type != "cpu":
            tensors = tuple(t.to(device_type) for t in tensors)
        return tensors[0] if single_output and len(tensors) == 1 else tensors

    return run
