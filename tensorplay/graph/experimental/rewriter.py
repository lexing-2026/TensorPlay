from __future__ import annotations

import ast
import copy
import functools
import inspect
import textwrap
from types import FunctionType
from typing import Any, cast

from ..annotate import annotate
from ..graph import Graph
from ..graph_module import GraphModule
from ..tracer import Tracer

__all__ = ["AST_Rewriter", "RewritingTracer", "_rewrite"]


def _graph_assert(condition: Any, message: Any = "") -> None:
    if not condition:
        raise AssertionError(message)


class AST_Rewriter(ast.NodeTransformer):
    """Rewrite Python syntax that needs an explicit graph operation."""

    def rewrite(self, fn: FunctionType) -> FunctionType:
        if not isinstance(fn, FunctionType):
            raise TypeError(f"expected a Python function, got {type(fn).__name__}")
        try:
            source = textwrap.dedent("".join(inspect.getsourcelines(fn)[0]))
        except (OSError, IOError, TypeError) as exc:
            raise OSError(f"source for {fn!r} is unavailable") from exc
        source_ast = ast.parse(source)
        destination = ast.fix_missing_locations(self.visit(source_ast))
        code = compile(destination, inspect.getsourcefile(fn) or "<graph-rewrite>", "exec")
        globals_dict = copy.copy(fn.__globals__)
        globals_dict.setdefault("annotate", annotate)
        globals_dict.setdefault("graph_assert", _graph_assert)
        before = set(globals_dict)
        exec(code, globals_dict)
        added = [name for name in globals_dict if name not in before]
        functions = [name for name in added if isinstance(globals_dict[name], FunctionType)]
        if len(functions) != 1:
            raise AssertionError(f"expected one rewritten function, got {functions!r}")
        compiled = globals_dict[functions[0]]
        result = FunctionType(
            compiled.__code__,
            fn.__globals__,
            name=fn.__name__,
            argdefs=fn.__defaults__,
            closure=fn.__closure__,
        )
        result = functools.update_wrapper(result, fn)
        result.__kwdefaults__ = copy.copy(fn.__kwdefaults__)
        return result

    def visit_Assert(self, node: ast.Assert) -> ast.Expr:
        message = node.msg if node.msg is not None else ast.Constant(value="")
        call = ast.Call(
            func=ast.Name(id="graph_assert", ctx=ast.Load()),
            args=[node.test, message],
            keywords=[],
        )
        return ast.copy_location(ast.Expr(value=call), node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.Assign:
        if node.value is None:
            raise SyntaxError("annotated assignment requires a value in a traced function")
        call = ast.Call(
            func=ast.Name(id="annotate", ctx=ast.Load()),
            args=[node.value, node.annotation],
            keywords=[],
        )
        return ast.copy_location(ast.Assign(targets=[node.target], value=call), node)


class RewritingTracer(Tracer):
    """Tracer that applies :class:`AST_Rewriter` before capture."""

    def trace(
        self, root: Any, concrete_args: dict[str, Any] | None = None
    ) -> GraphModule:
        return super().trace(_rewrite(root), sample_inputs=concrete_args)


def _rewrite(fn: Any) -> Any:
    """Return a callable with assertion and annotation syntax rewritten."""

    module_type = None
    try:
        from tensorplay.nn import Module

        module_type = Module
    except ImportError:
        pass

    if module_type is not None and isinstance(fn, module_type):
        def rewrite_module(module: Any) -> Any:
            rewritten = copy.copy(module)
            if hasattr(module, "_modules"):
                rewritten._modules = {}
            for name, value in vars(module).items():
                if name in {"_modules", "forward"}:
                    continue
                try:
                    setattr(rewritten, name, copy.copy(value))
                except (AttributeError, TypeError):
                    pass
            for name, child in module.named_children():
                rewritten.add_module(name, rewrite_module(child))
            rewritten.forward = AST_Rewriter().rewrite(cast(FunctionType, module.forward))
            return rewritten

        return rewrite_module(fn)
    if not isinstance(fn, FunctionType):
        raise TypeError(f"expected a function or module, got {type(fn).__name__}")
    return AST_Rewriter().rewrite(fn)

