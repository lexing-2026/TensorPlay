"""Public verification API for exported models.

Wraps the numerical comparison between eager execution and onnxruntime so
export call sites can report and assert agreement.
"""

from __future__ import annotations

from ._verify import VerificationError, VerificationResult, verify_model

__all__ = ["VerificationError", "VerificationResult", "verify_model"]
