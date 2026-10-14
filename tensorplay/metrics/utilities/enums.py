"""Enums used across the metrics package.

The string enums behave like plain strings so that values can be compared
directly with ``str``, while also providing ``from_str`` parsing with friendly
error messages.
"""

import enum
from typing import ClassVar, Literal


class EnumStr(str, enum.Enum):
    """Base class for string-valued enums."""

    @staticmethod
    def _name() -> str:
        return "Task"

    @classmethod
    def _allowed_matches(cls, source: Literal["key", "value", "any"]) -> list[str]:
        if source == "key":
            return [m.name.lower() for m in cls]
        if source == "value":
            return [str(m.value).lower() for m in cls]
        return sorted(set(cls._allowed_matches("key") + cls._allowed_matches("value")))

    @classmethod
    def from_str(cls, value: str, source: Literal["key", "value", "any"] = "key") -> "EnumStr":
        """Load an enum member from a string.

        Args:
            value: string to match against the enum
            source: whether to match against member names (``key``), values
                (``value``) or both (``any``)

        Raises:
            ValueError: if ``value`` does not match any member

        """
        normalized = value.replace("-", "_").lower()
        if source in ("key", "any"):
            for member in cls:
                if member.name.lower() == normalized:
                    return member
        if source in ("value", "any"):
            for member in cls:
                if str(member.value).lower() == normalized:
                    return member
        raise ValueError(
            f"Invalid {cls._name()}: expected one of {cls._allowed_matches(source)}, but got {value}."
        )


class DataType(EnumStr):
    """Data type of a classification problem."""

    @staticmethod
    def _name() -> str:
        return "Data type"

    BINARY = "binary"
    MULTILABEL = "multi-label"
    MULTICLASS = "multi-class"
    MULTIDIM_MULTICLASS = "multi-dim multi-class"


class AverageMethod(EnumStr):
    """Average method for classification metrics."""

    @staticmethod
    def _name() -> str:
        return "Average method"

    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"
    NONE = "none"
    SAMPLES = "samples"


class MDMCAverageMethod(EnumStr):
    """Average method for multi-dim multi-class metrics."""

    @staticmethod
    def _name() -> str:
        return "MDMC Average method"

    GLOBAL = "global"
    SAMPLEWISE = "samplewise"


class ClassificationTask(EnumStr):
    """Classification task type."""

    @staticmethod
    def _name() -> str:
        return "Classification"

    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


class ClassificationTaskNoBinary(EnumStr):
    """Classification task type without the binary option."""

    @staticmethod
    def _name() -> str:
        return "Classification"

    MULTILABEL = "multilabel"
    MULTICLASS = "multiclass"


class ClassificationTaskNoMultilabel(EnumStr):
    """Classification task type without the multilabel option."""

    @staticmethod
    def _name() -> str:
        return "Classification"

    BINARY = "binary"
    MULTICLASS = "multiclass"