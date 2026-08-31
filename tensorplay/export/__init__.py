"""Graph capture, dynamic shape declarations, and exported programs."""

from typing import Any

from .decomp_utils import CustomDecompTable
from .custom_obj import register_dataclass
from .pt2_archive import load_pt2, package_pt2
from .dynamic_shapes import (
    AdditionalInputs,
    Constraint,
    Dim,
    ShapesCollection,
    dims,
    refine_dynamic_shapes_from_suggested_fixes,
)
from .exported_program import (
    ExportedProgram,
    ModuleCallEntry,
    ModuleCallSignature,
    default_decompositions,
)
from .graph_signature import (
    ArgumentSpec,
    ConstantArgument,
    CustomObjArgument,
    ExportBackwardSignature,
    ExportGraphSignature,
    GraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    SymBoolArgument,
    SymFloatArgument,
    SymIntArgument,
    TensorArgument,
    TokenArgument,
)
from ._trace import export, export_for_training
from ._draft_export import draft_export

from .unflatten import FlatArgsAdapter, UnflattenedModule, unflatten

__all__ = [
    "AdditionalInputs",
    "ArgumentSpec",
    "ConstantArgument",
    "Constraint",
    "CustomDecompTable",
    "CustomObjArgument",
    "Dim",
    "ExportBackwardSignature",
    "ExportGraphSignature",
    "ExportedProgram",
    "FlatArgsAdapter",
    "GraphSignature",
    "InputKind",
    "InputSpec",
    "ModuleCallEntry",
    "ModuleCallSignature",
    "OutputKind",
    "OutputSpec",
    "ShapesCollection",
    "SymBoolArgument",
    "SymFloatArgument",
    "SymIntArgument",
    "TensorArgument",
    "TokenArgument",
    "UnflattenedModule",
    "default_decompositions",
    "dims",
    "draft_export",
    "export",
    "export_for_training",
    "refine_dynamic_shapes_from_suggested_fixes",
    "register_dataclass",
    "save",
    "load",
    "unflatten",
]


def save(
    ep: Any,
    f: Any,
    *,
    extra_files: dict[str, Any] | None = None,
    opset_version: dict[str, int] | None = None,
    pickle_protocol: int = 4,
) -> None:
    """Save an :class:`ExportedProgram` to a file or writable buffer."""

    if not isinstance(ep, ExportedProgram):
        raise TypeError(
            f"The 'ep' parameter must be an ExportedProgram, got {type(ep).__name__}"
        )
    package_pt2(
        f,
        exported_programs=ep,
        extra_files=extra_files,
        opset_version=opset_version,
        pickle_protocol=pickle_protocol,
    )


def load(
    f: Any,
    *,
    extra_files: dict[str, Any] | None = None,
    expected_opset_version: dict[str, int] | None = None,
) -> Any:
    """Load an :class:`ExportedProgram` previously written by :func:`save`."""

    contents = load_pt2(f, expected_opset_version=expected_opset_version)
    if extra_files is not None:
        extra_files.update(contents.extra_files)
    if not contents.exported_programs:
        raise ValueError("the archive does not contain an exported program")
    return contents.exported_programs["model"]
