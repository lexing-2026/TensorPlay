"""QConfig multi-mapping for the FX numeric suite.

A :class:`QConfigMultiMapping` stores a list of :class:`QConfigMapping`s so
that multiple qconfigs can be specified per matching style.  Each ``set_*``
method takes a **list** of qconfigs instead of a single one; the list is
spread across the internal mappings, preserving match priority semantics for
``prepare_n_shadows_model``-style workflows.
"""

from __future__ import annotations

import copy
from typing import Any, TYPE_CHECKING

from tensorplay.ao.quantization import QConfigMapping, qconfig_equals

if TYPE_CHECKING:
    from collections.abc import Callable

    from tensorplay.ao.quantization import QConfigAny

__all__ = ["QConfigMultiMapping"]

_QCONFIG_STYLE_ORDER = [
    "global_qconfig",
    "object_type_qconfigs",
    "module_name_regex_qconfigs",
    "module_name_qconfigs",
    "module_name_object_type_order_qconfigs",
]

_QCONFIG_STYLE_TO_METHOD: dict[str, str] = {
    "global_qconfig": "set_global",
    "object_type_qconfigs": "set_object_type",
    "module_name_regex_qconfigs": "set_module_name_regex",
    "module_name_qconfigs": "set_module_name",
    "module_name_object_type_order_qconfigs": "set_module_name_object_type_order",
}


def _remove_duplicates_and_none(qconfig_list) -> None:
    to_remove = []
    for index, cur_qconfig in enumerate(qconfig_list):
        if cur_qconfig is None:
            to_remove.append(index)
            break
        for checked_qconfig in qconfig_list[:index]:
            if qconfig_equals(cur_qconfig, checked_qconfig):
                to_remove.append(index)
                break
    for index in reversed(to_remove):
        qconfig_list.pop(index)


class QConfigMultiMapping:
    """Store multiple qconfigs for each qconfig matching style.

    Example usage::

        mapping = QConfigMultiMapping() \\
            .set_global([qconfig1, qconfig2]) \\
            .set_object_type(nn.Linear, [qconfig2, qconfig3]) \\
            .set_module_name("module1", [None]) \\
            .set_module_name_object_type_order("foo.bar", F.linear, 0, [qconfig3])
    """

    def __init__(self) -> None:
        # initialize with one QConfigMapping to avoid corner cases
        self.qconfig_mappings_list: list[QConfigMapping] = [QConfigMapping()]

    def _handle_list_size_mismatch(self, qconfig_list, style: str) -> None:
        # Maintain the invariant that if any QConfigMapping has a qconfig for
        # a given style+key, all QConfigMappings must have either a qconfig or
        # None for that same style+key.
        if len(qconfig_list) > len(self.qconfig_mappings_list):
            new_qconfig_mapping = QConfigMapping()
            for qconfig_mapping in self.qconfig_mappings_list:
                for check_style in _QCONFIG_STYLE_ORDER[1:]:
                    qconfigs_dict = getattr(qconfig_mapping, check_style)
                    target_qconfigs_dict = getattr(new_qconfig_mapping, check_style)
                    for key in qconfigs_dict:
                        target_qconfigs_dict[key] = None
                break
            while len(qconfig_list) > len(self.qconfig_mappings_list):
                self.qconfig_mappings_list.append(copy.deepcopy(new_qconfig_mapping))
        else:
            while len(qconfig_list) < len(self.qconfig_mappings_list):
                qconfig_list.append(None)

    def _insert_qconfig_list(self, style: str, args, qconfig_list) -> None:
        _remove_duplicates_and_none(qconfig_list)
        self._handle_list_size_mismatch(qconfig_list, style)
        method_name = _QCONFIG_STYLE_TO_METHOD[style]
        for qconfig_mapping, qconfig in zip(self.qconfig_mappings_list, qconfig_list):
            set_method = getattr(qconfig_mapping, method_name)
            set_method(*args, qconfig)

    def set_global(self, global_qconfig_list) -> "QConfigMultiMapping":
        self._insert_qconfig_list("global_qconfig", [], global_qconfig_list)
        return self

    def set_object_type(self, object_type, qconfig_list) -> "QConfigMultiMapping":
        self._insert_qconfig_list("object_type_qconfigs", [object_type], qconfig_list)
        return self

    def set_module_name_regex(self, module_name_regex, qconfig_list) -> "QConfigMultiMapping":
        self._insert_qconfig_list("module_name_regex_qconfigs", [module_name_regex], qconfig_list)
        return self

    def set_module_name(self, module_name, qconfig_list) -> "QConfigMultiMapping":
        self._insert_qconfig_list("module_name_qconfigs", [module_name], qconfig_list)
        return self

    def set_module_name_object_type_order(
        self, module_name, object_type, index, qconfig_list
    ) -> "QConfigMultiMapping":
        self._insert_qconfig_list(
            "module_name_object_type_order_qconfigs",
            [module_name, object_type, index],
            qconfig_list,
        )
        return self

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ["
            + "".join(
                f"\n{qconfig_mapping.__repr__()},"
                for qconfig_mapping in self.qconfig_mappings_list
            )
            + "\n]"
        )

    @classmethod
    def from_list_qconfig_mapping(cls, qconfig_mapping_list):
        new = cls()
        new.qconfig_mappings_list = copy.deepcopy(qconfig_mapping_list)
        for style in _QCONFIG_STYLE_ORDER[1:]:
            qconfig_dict_list: dict[Any, list] = {}
            for qconfig_mapping in qconfig_mapping_list:
                qconfig_dict = getattr(qconfig_mapping, style)
                for key, qconfig in qconfig_dict.items():
                    if key not in qconfig_dict_list:
                        qconfig_dict_list[key] = []
                    qconfig_dict_list[key].append(qconfig)
            set_method_name = _QCONFIG_STYLE_TO_METHOD[style]
            set_method = getattr(new, set_method_name)
            for key, qconfig_list in qconfig_dict_list.items():
                if isinstance(key, tuple):
                    set_method(*key, qconfig_list)
                else:
                    set_method(key, qconfig_list)
        return new