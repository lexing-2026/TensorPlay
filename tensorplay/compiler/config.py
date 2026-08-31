"""Live configuration for the TensorPlay compiler frontend.

Only settings with consumers in the current capture pipeline are exposed here.
Changing a setting affects compiler wrappers created afterwards.  A wrapper
keeps the policy values that were selected when it was created, so changing a
setting does not silently change an already populated specialization cache.
Call :func:`tensorplay.compiler.reset` when a process needs to discard the
specializations owned by existing wrappers.
"""

from __future__ import annotations

__all__ = [
    "dynamic_shapes",
    "assume_static_by_default",
    "recompile_limit",
    "accumulated_recompile_limit",
    "verbose",
    "fail_on_recompile_limit_hit",
    "force_disable_caches",
]


dynamic_shapes: bool | None = None
"""Default shape policy used when a compile call omits both shape arguments.

``True`` makes the frontend retain tensor rank while omitting concrete sizes
from the specialization signature.  The resulting cache entry can therefore
be reused across size changes that the captured graph and backend can handle.
``False`` keeps concrete tensor sizes in the signature and creates a separate
specialization when a size changes.  ``None`` selects the frontend default,
which is the concrete-size policy used by the current capture path.

The public compile entry reads this value only when neither ``dynamic`` nor
``dynamic_shapes`` is supplied explicitly.  It forwards the resolved boolean
policy to capture, signature construction, and guard-chain creation, so the
setting affects both specialization reuse and the conditions checked before a
cached artifact runs.
"""


recompile_limit: int = 8
"""Maximum number of cached specializations kept by one compiler wrapper.

Every distinct input signature, shape guard result, or data-dependent branch
outcome consumes one entry.  Once this limit is reached, the frontend raises
when the policy forbids another specialization; it never installs an
uncaptured callable as the result of a compile request.  The value must be a
positive integer.

The public compile entry reads this setting when its ``recompile_limit``
argument is omitted and passes the resolved limit into the frontend cache.  The
same limit is stored on the wrapper used by the reset path, so
``tensorplay.compiler.reset()`` clears the corresponding specialization and
guard state together.
"""


accumulated_recompile_limit: int = 256
"""Maximum number of specializations created by one compiler wrapper.

``recompile_limit`` limits the live entries used for one input signature
family.  This cumulative limit counts every compilation attempt, including
entries later invalidated when guard metadata is promoted.  The next compile
attempt after this limit is rejected by the frontend.
"""


assume_static_by_default: bool = True
"""Use concrete tensor sizes when no dynamic shape policy is supplied.

When false, an omitted shape policy uses rank-only tensor signatures.  Native
lowerings still receive the real runtime tensors and are responsible for
accepting the resulting shape range.
"""


verbose: bool = False
"""Emit a warning with guard reasons whenever a specialization is rebuilt.

The environment switch ``TP_LOG_RECOMPILES`` remains an independent way to
enable the same diagnostics for a process.
"""


fail_on_recompile_limit_hit: bool = False
"""Raise at the specialization limit boundary.

When false, the current uncached call is compiled once more without being
inserted into the bounded live cache.  When true, that call raises at the
boundary.  Both modes keep execution on a captured native compiler path.
"""


force_disable_caches: bool = False
"""Disable frontend, guard-chain, process and disk cache reuse.

Every invocation goes through capture and backend compilation while this is
true.  Artifacts are not read from or written to the process or disk cache.
"""
