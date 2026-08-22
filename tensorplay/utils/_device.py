"""Default-device context, mirroring ``torch.utils._device``.

``tensorplay.device.__enter__``/``__exit__`` (C++ side) push/pop the
thread-local default device so factory functions allocate on it; this module
exposes the equivalent ``DeviceContext`` object for API parity with torch.
"""

from __future__ import annotations

import tensorplay


class DeviceContext:
    def __init__(self, device) -> None:
        self.device = tensorplay.device(device)

    def __enter__(self):
        tensorplay._C._push_default_device(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        tensorplay._C._pop_default_device()
        return False
