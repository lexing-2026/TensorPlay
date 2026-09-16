"""ONNX exporter exceptions."""

from __future__ import annotations

__all__ = [
    "OnnxExporterWarning",
    "OnnxExporterError",
    "UnsupportedOperatorError",
]


class OnnxExporterWarning(UserWarning):
    """Warnings raised during ONNX export."""


class OnnxExporterError(RuntimeError):
    """Base class for errors raised by the ONNX exporter."""


class UnsupportedOperatorError(OnnxExporterError, NotImplementedError):
    """Raised when a captured operation has no ONNX lowering."""

    def __init__(self, name: str, version: int | None = None,
                 supported_version: int | None = None) -> None:
        if version is not None and supported_version is not None:
            msg = (
                f"Exporting the operator '{name}' to ONNX opset version "
                f"{version} is not supported.  Support for this operator "
                f"was added in opset version {supported_version}; try "
                "exporting with that version"
            )
        elif version is not None:
            msg = (
                f"Exporting the operator '{name}' to ONNX opset version "
                f"{version} is not supported"
            )
        else:
            msg = (
                f"ONNX export failed on the operator '{name}'.  If this is "
                "a custom operator, register a lowering for it before "
                "exporting"
            )
        super().__init__(msg)
