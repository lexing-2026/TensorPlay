"""Where an operation that is not written out has its own spelling kept.

Most operations that reach the wrapper are printed from the buffer operations
describe.  A few are not expressions over buffers at all: they are calls the
user wrote, and what they print is the call itself rather than arithmetic on
memory.  Those are registered here instead of being written into the wrapper's
body, so that one spelling serves every wrapper that has to emit the call, and
so that adding one is a line rather than an edit inside a generator.

A registry entry names an operation and gives the spelling for each language a
wrapper might be written in.  A language left empty is one this operation has
no spelling for, which is different from an operation that is absent: the
first is a fallback the wrapper can take, the second is a hole.

The registry starts empty because this route has no operation whose printed form
is a call rather than arithmetic, and an entry for an operation that does not
exist would be a spelling nothing can ask for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CustomCodegen:
    """What one operation is printed as, per language.

    An entry with neither field set is an operation that is registered but not
    printable, which is the honest way to say "this is known and not written".
    """

    python: Any = None
    cpp: Any = None


#: Operations with a spelling of their own, by operation name.  A wrapper looks
#: an operation up here before it falls back to printing the call.
#:
#: To add one: write a function taking the node and a ``writeline`` callable
#: that appends one line of generated source, and register it under the
#: operation's name.
CUSTOM_EXTERN_KERNEL_CODEGEN: dict[str, CustomCodegen] = {}


__all__ = [
    "CUSTOM_EXTERN_KERNEL_CODEGEN",
    "CustomCodegen",
]
