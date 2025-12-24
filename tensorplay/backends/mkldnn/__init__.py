import sys
import tensorplay._C as _C
from contextlib import contextmanager

class MkldnnModule:
    def __getattr__(self, name):
        if name == 'is_available':
            return self.is_available
        if name == 'flags':
            return self.flags
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    @staticmethod
    def is_available():
        return _C.has_mkldnn()

    @property
    def enabled(self):
        return _C.is_mkldnn_enabled()

    @enabled.setter
    def enabled(self, val):
        _C.set_mkldnn_enabled(val)

    @contextmanager
    def flags(self, enabled=None):
        original = _C.is_mkldnn_enabled()
        if enabled is not None:
            _C.set_mkldnn_enabled(enabled)
        try:
            yield
        finally:
            _C.set_mkldnn_enabled(original)

# Replace the module with an instance of the class
sys.modules[__name__] = MkldnnModule()
