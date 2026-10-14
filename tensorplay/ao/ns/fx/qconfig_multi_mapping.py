"""QConfig multi-mapping for the FX numeric suite."""

__all__ = ["QConfigMultiMapping"]


class QConfigMultiMapping:
    def __init__(self):
        self.qconfig_mappings = {}

    def set_mapping(self, key, qconfig_mapping):
        self.qconfig_mappings[key] = qconfig_mapping
        return self

    def get_mapping(self, key):
        return self.qconfig_mappings.get(key)

    def to_dict(self):
        return self.qconfig_mappings
