"""Functional API for all metrics.

The functional interface exposes stateless metric computations. Functions are
grouped by domain into the ``functional`` subpackage (classification,
regression, image, text) and re-exported here for direct access, e.g.
``tensorplay.metrics.functional.mean_squared_error``.
"""

from tensorplay.metrics.functional.classification import *  # noqa: F401,F403
from tensorplay.metrics.functional.regression import *  # noqa: F401,F403
from tensorplay.metrics.functional.image import *  # noqa: F401,F403
from tensorplay.metrics.functional.text import *  # noqa: F401,F403
from tensorplay.metrics.functional import classification  # noqa: F401
from tensorplay.metrics.functional import regression  # noqa: F401
from tensorplay.metrics.functional import image  # noqa: F401
from tensorplay.metrics.functional import text  # noqa: F401

__all__ = (
    classification.__all__
    + regression.__all__
    + image.__all__
    + text.__all__
    + ["classification", "regression", "image", "text"]
)