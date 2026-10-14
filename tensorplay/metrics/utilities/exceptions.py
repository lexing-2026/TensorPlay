"""Metric-specific exception types."""


class MetricsUserError(Exception):
    """Raised when the Metric API is used in an unsupported combination."""


class MetricsUserWarning(Warning):
    """Warning category used for user-facing metric API warnings."""