from __future__ import annotations

import os
from typing import Any


def _storage_setup(storage: Any, checkpoint_id: str | os.PathLike[str] | None = None) -> Any:
    if checkpoint_id is not None:
        storage.reset(checkpoint_id)
    return storage
