"""(see docstring above)"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Callable

if __package__ in (None, ''):
    # Direct-script invocation: re-enter through the package so every
    # generator's relative imports resolve, then delegate.
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from tools.codegen.main import main  # noqa: E402
    raise SystemExit(main())

from .model import parse_native_yaml  # noqa: F401 (re-exported)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

_GENERATORS: dict[str, Callable] = {}


def register_generator(name: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        _GENERATORS[name] = fn
        return fn
    return deco


@dataclass
class CodegenContext:
    funcs: list  # of model.NativeFunction
    native_by_opname: dict[str, "NativeFunction"]
    derivatives: dict  # op name -> OpDerivatives
    autocast_ops: set[str]
    autograd_ops: dict[str, str]  # dispatcher op name -> backward node
    out_dir: str
    pkg_out: str | None = None
    written: dict[str, str] = field(default_factory=dict)

    def write(self, filename: str, content: str) -> None:
        path = os.path.join(self.out_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(content)
        self.written[filename] = path
        print(f'Generated "{path}"')

    def write_pkg(self, relname: str, content: str) -> None:
        assert self.pkg_out, "pkg_out not provided"
        path = os.path.join(self.pkg_out, relname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(content)
        print(f'Generated "{path}"')


# ---------------------------------------------------------------------------
# Generators (one per output file group)
# ---------------------------------------------------------------------------

@register_generator("TensorMethods")
def _gen_tensor_methods(ctx: CodegenContext) -> None:
    from .gen_api import generate_cpp, generate_header
    ctx.write("TensorGenerated.h",
              generate_header(ctx.funcs))
    ctx.write("TensorGenerated.cpp",
              generate_cpp(ctx.funcs, autocast_ops=ctx.autocast_ops,
                           autograd_ops=ctx.autograd_ops))


@register_generator("Redispatch")
def _gen_redispatch(ctx: CodegenContext) -> None:
    from .gen_api import generate_redispatch_header
    ctx.write("TensorRedispatchGenerated.h",
              generate_redispatch_header(ctx.funcs))


@register_generator("AutogradNodes")
def _gen_autograd_nodes(ctx: CodegenContext) -> None:
    from .gen_autograd import generate_autograd_nodes
    ctx.write("AutogradNodesGenerated.h",
              generate_autograd_nodes(ctx.derivatives))


@register_generator("TPXOps")
def _gen_tpx_ops(ctx: CodegenContext) -> None:
    from .gen_tpx import generate_tpx_ops_cpp, generate_tpx_ops_h
    ctx.write("TPXOpsGenerated.h", generate_tpx_ops_h(ctx.funcs))
    ctx.write("TPXOpsGenerated.cpp",
              generate_tpx_ops_cpp(ctx.funcs, autocast_ops=ctx.autocast_ops,
                                   derivatives=ctx.derivatives))


@register_generator("AutogradRegistration")
def _gen_autograd_registration(ctx: CodegenContext) -> None:
    from .gen_tpx import generate_autograd_registration
    ctx.write("TPXAutogradRegistration.cpp",
              generate_autograd_registration(ctx.funcs,
                                             derivatives=ctx.derivatives))


@register_generator("Bindings")
def _gen_bindings(ctx: CodegenContext) -> None:
    from .gen_bindings import generate_bindings
    ctx.write("TensorBindingsGenerated.h", generate_bindings(ctx.funcs))


@register_generator("Autocast")
def _gen_autocast(ctx: CodegenContext) -> None:
    from .gen_autocast import generate_autocast_registration
    ctx.write("AutocastGenerated.cpp",
              generate_autocast_registration(ctx.funcs))


@register_generator("PythonCAPI")
def _gen_python_capi(ctx: CodegenContext) -> None:
    from .gen_python_c import _gen_python_capi as gen
    gen(ctx)


@register_generator("Structured")
def _gen_structured(ctx: CodegenContext) -> None:
    from .gen_structured import _gen_structured as gen
    gen(ctx)


@register_generator("InplaceOrView")
def _gen_inplace_or_view(ctx: CodegenContext) -> None:
    from .gen_inplace_or_view import _gen_inplace_or_view as gen
    gen(ctx)


@register_generator("PythonFunctional")
def _gen_python_functional(ctx: CodegenContext) -> None:
    from .gen_python import generate_functional_py
    if ctx.pkg_out:
        ctx.write_pkg("functional.py", generate_functional_py(ctx.funcs))


@register_generator("PyiStub")
def _gen_pyi(ctx: CodegenContext) -> None:
    from .gen_pyi import generate_pyi
    if ctx.pyi_template and getattr(ctx, "pyi_out", None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dtype_header = os.path.join(script_dir, "../../p10/include/DType.h")
        path = ctx.pyi_out
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(generate_pyi(ctx.funcs, ctx.pyi_template, dtype_header))
        print(f'Generated "{path}"')


DEFAULT_TARGETS = ["TensorMethods", "Redispatch", "AutogradNodes", "TPXOps",
                   "AutogradRegistration", "Bindings", "Autocast",
                   "InplaceOrView", "Structured", "PythonFunctional",
                   "PythonCAPI"]


def run_gen(ctx: CodegenContext, targets: list[str]) -> None:
    for name in targets:
        _GENERATORS[name](ctx)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--pyi_template")
    parser.add_argument("--pyi_out")
    parser.add_argument("--derivatives")
    parser.add_argument("--pkg_out")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                        help="comma-separated generator names (run_gen subset)")
    args = parser.parse_args(argv)

    funcs = parse_native_yaml(args.yaml)
    native_by_opname = {f.func_name: f for f in funcs}

    derivatives = {}
    if args.derivatives and os.path.exists(args.derivatives):
        from .gen_autograd import load_derivatives
        derivatives = load_derivatives(args.derivatives, native_by_opname)

    from .gen_autocast import autocast_registered_ops
    autograd_ops = {op: dv.node_name for op, dv in derivatives.items()}
    autograd_ops.setdefault("relu_", "ReluBackward")

    ctx = CodegenContext(
        funcs=funcs,
        native_by_opname=native_by_opname,
        derivatives=derivatives,
        autocast_ops=autocast_registered_ops(),
        autograd_ops=autograd_ops,
        out_dir=args.out_dir,
        pkg_out=args.pkg_out,
    )
    ctx.pyi_template = args.pyi_template
    ctx.pyi_out = args.pyi_out

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    unknown = [t for t in targets if t not in _GENERATORS]
    if unknown:
        raise SystemExit(f"unknown generators: {unknown}; "
                         f"available: {sorted(_GENERATORS)}")
    run_gen(ctx, targets)


if __name__ == "__main__":
    main()
