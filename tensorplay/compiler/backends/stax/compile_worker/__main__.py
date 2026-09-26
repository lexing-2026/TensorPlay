# mypy: allow-untyped-defs
#
# Run as a script rather than as a module: the pool names this file
# directly so that the worker is the interpreter that started it and
# not whatever a module named on the path would resolve to.  A script
# has no package, so the imports below name their full path.
import argparse
import base64
import functools
import importlib
import logging
import os
import sys
from typing import TypeVar

from tensorplay.compiler.backends.stax.runtime.compile_tasks import pre_fork_setup
from tensorplay.compiler.backends.stax.codecache import package_key
from tensorplay.compiler.backends.stax.compile_worker.subproc_pool import (
    SubprocKind,
    SubprocException,
    SubprocMain,
    SubprocPickler,
)
from tensorplay.compiler.backends.stax.compile_worker.utils import _async_compile_initializer
from tensorplay.compiler.backends.stax.runtime.compile_tasks import _set_triton_ptxas_path


_T = TypeVar("_T")


log = logging.getLogger(__name__)

_set_triton_ptxas_path()

try:
    import triton

    if triton is None:
        raise AssertionError("triton failed to preload in parent")
except ImportError:
    pass


def _lookup_and_create_type(base: type[_T], qname: str) -> _T:
    """
    Given a base type and qualified name: import & lookup that name, check
    that it's of the given type and then instantiate it.
    """
    pkg, name = qname.rsplit(".", 1)
    mod = importlib.import_module(pkg)
    ty = getattr(mod, name)
    if not issubclass(ty, base):
        raise TypeError(f"Type {ty} is not a subtype of {base}")
    return ty()


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--pickler", type=functools.partial(_lookup_and_create_type, SubprocPickler)
        )
        parser.add_argument("--kind", type=SubprocKind)
        parser.add_argument("--workers", type=int)
        parser.add_argument("--parent", type=int)
        parser.add_argument("--read-fd", type=int)
        parser.add_argument("--write-fd", type=int)
        parser.add_argument("--package-key", type=str)
        args = parser.parse_args()
        if os.getppid() != args.parent:
            sys.exit(0)
        read_fd = os.fdopen(args.read_fd, "rb")
        write_fd = os.fdopen(args.write_fd, "wb")

        pre_fork_setup()

        # The parent computed this before starting us so that every worker
        # answers from one cache.  Recomputing it here and comparing is how a
        # worker notices it was started by something built differently.
        if base64.b64decode(args.package_key.encode("utf-8")) != package_key():
            raise SubprocException(
                "worker was started by a build with a different key"
            )

        _async_compile_initializer(args.parent)

        SubprocMain(args.pickler, args.kind, args.workers, read_fd, write_fd).main()
    except Exception:
        log.exception("Uncaught exception in compile_worker subprocess")


if __name__ == "__main__":
    main()
