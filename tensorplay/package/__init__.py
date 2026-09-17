"""Bundle Python code, pickled data, and resources into self-contained archives.

The two central entry points are :class:`PackageExporter`, which writes a
package, and :class:`PackageImporter`, which loads code and resources from a
package in a hermetic way (resolved against the package itself rather than the
surrounding interpreter).
"""

from types import ModuleType
from typing import Any

from .file_structure_representation import Directory
from .glob_group import GlobGroup
from .importer import (
    Importer,
    ObjMismatchError,
    ObjNotFoundError,
    OrderedImporter,
    sys_importer,
)
from ._mangling import is_mangled
from .package_exporter import (
    EmptyMatchError,
    PackageExporter,
    PackagingError,
    PackagingErrorReason,
)
from .package_importer import PackageImporter


__all__ = [
    "Directory",
    "EmptyMatchError",
    "GlobGroup",
    "Importer",
    "ObjMismatchError",
    "ObjNotFoundError",
    "OrderedImporter",
    "PackageExporter",
    "PackageImporter",
    "PackagingError",
    "PackagingErrorReason",
    "is_from_package",
    "sys_importer",
]


def is_from_package(obj: Any) -> bool:
    """
    Return whether an object was loaded from a package.

    Note: packaged objects from externed modules will return ``False``.
    """
    if type(obj) is ModuleType:
        return is_mangled(obj.__name__)
    else:
        return is_mangled(type(obj).__module__)
