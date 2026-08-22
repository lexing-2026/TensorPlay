"""tensorplay.vision — torchvision-compatible vision components.

Structure mirrors torchvision: ``datasets``, ``transforms``, ``models``,
``ops`` and ``io`` subpackages plus the ``make_grid``/``save_image`` helpers
from ``utils``.  The legacy module-level helpers ``from_file`` and
``from_image`` are kept for backwards compatibility.
"""

from . import datasets
from . import transforms
from . import models
from . import io as io  # noqa: F401
from .backend import set_backend, get_backend
from .transforms.functional import to_tensor, from_image, from_file  # legacy names
from .utils import make_grid, save_image

__all__ = [
    "datasets",
    "transforms",
    "models",
    "io",
    "set_backend",
    "get_backend",
    "to_tensor",
    "from_image",
    "from_file",
    "make_grid",
    "save_image",
]
