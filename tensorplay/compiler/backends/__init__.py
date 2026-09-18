"""Compiler backends.

Each module turns a captured :class:`~tensorplay.graph.GraphModule` plus
example inputs into an executable callable.  Registration (names, tags,
capabilities) happens lazily through ``backends.builtins``; see
:mod:`tensorplay.compiler._core.registry`.
"""
