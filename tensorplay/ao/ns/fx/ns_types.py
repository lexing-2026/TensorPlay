"""Type aliases for the FX numeric suite."""

from typing import Any, Union

NSNodeTargetType = Union[type, str]
NSSubgraph = tuple[Any, Any, Any]
NSResultsType = dict[str, dict[str, Any]]