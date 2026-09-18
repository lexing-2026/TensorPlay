"""Lazy registration of built-in TensorPlay compiler backends."""

from __future__ import annotations

from .registry import BackendCapabilities, register_backend


def register() -> None:
    from .stax import stax

    register_backend(stax, name="stax")
    # Importing the module registers the debug-tagged backends.
    from . import debugging  # noqa: F401

    # Apache-TVM is an optional dependency; the backend module itself stays
    # import-light and validates availability at compile time (its own
    # actionable error), so it stays visible in list_backends.  TVM lowering
    # covers inference regions; training regions are wrapped ahead-of-time by
    # the frontend (forward through TVM, backward as traced).
    from .tvm import tvm as tvm_backend

    register_backend(
        tvm_backend,
        name="tvm",
        capabilities=BackendCapabilities(
            inference_only=True,
            handles_training=False,
        ),
    )

    from .cudagraphs import CudagraphsBackend

    register_backend(CudagraphsBackend(), name="cudagraphs")

    # ONNX Runtime is an optional dependency; a missing install hides the
    # backend from list_backends() and selecting it by name explains what to
    # install.  The backend callable itself carries the declaration via
    # @declares_capabilities; see tensorplay._stax.onnxrt.
    from .onnxrt import onnxrt

    register_backend(onnxrt, name="onnxrt")
