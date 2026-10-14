"""Custom configuration containers for the graph quantization workflow."""

from dataclasses import dataclass, field

__all__ = ["PrepareCustomConfig", "ConvertCustomConfig", "FuseCustomConfig"]


@dataclass
class PrepareCustomConfig:
    """Custom configuration for :func:`prepare_fx`."""

    concrete_args: dict = field(default_factory=dict)
    skipped_module_names: list = field(default_factory=list)
    skipped_module_classes: list = field(default_factory=list)
    non_traceable_module_names: list = field(default_factory=list)
    non_traceable_module_classes: list = field(default_factory=list)
    preserve_attributes: list = field(default_factory=list)
    additional_quant_funcs: dict = field(default_factory=dict)

    def set_concrete_args(self, concrete_args):
        self.concrete_args = concrete_args
        return self

    def set_skip_module_name(self, module_name):
        self.skipped_module_names.append(module_name)
        return self

    def set_skip_module_class(self, module_class):
        self.skipped_module_classes.append(module_class)
        return self

    def set_non_traceable_module_name(self, module_name):
        self.non_traceable_module_names.append(module_name)
        return self

    def set_non_traceable_module_class(self, module_class):
        self.non_traceable_module_classes.append(module_class)
        return self

    def set_preserved_attributes(self, attributes):
        self.preserve_attributes = list(attributes)
        return self

    def to_dict(self):
        return {
            "concrete_args": self.concrete_args,
            "skipped_module_names": self.skipped_module_names,
            "skipped_module_classes": self.skipped_module_classes,
            "non_traceable_module_names": self.non_traceable_module_names,
            "non_traceable_module_classes": self.non_traceable_module_classes,
            "preserve_attributes": self.preserve_attributes,
        }


@dataclass
class ConvertCustomConfig:
    """Custom configuration for :func:`convert_fx`."""

    additional_conversions: dict = field(default_factory=dict)

    def set_additional_conversion(self, pattern, conversion):
        self.additional_conversions[pattern] = conversion
        return self

    def to_dict(self):
        return {"additional_conversions": self.additional_conversions}


@dataclass
class FuseCustomConfig:
    """Custom configuration for :func:`fuse_fx`."""

    additional_fuser_method_mapping: dict = field(default_factory=dict)
    fuse_custom_config_dict: dict = field(default_factory=dict)

    def set_fuser_method(self, pattern, fuser_method):
        self.additional_fuser_method_mapping[pattern] = fuser_method
        return self

    def to_dict(self):
        return {
            "additional_fuser_method_mapping": self.additional_fuser_method_mapping,
            "fuse_custom_config_dict": self.fuse_custom_config_dict,
        }