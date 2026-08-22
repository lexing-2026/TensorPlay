"""Derivatives loading, expression AST, and backward-node generation.

Mirrors PyTorch's tools/autograd layer:

* ``load_derivatives`` parses derivatives.yaml entries (ATen spelling:
  ``- name: <schema>`` plus one gradient formula per differentiable input /
  output) into typed objects.
* Gradient formulas are compiled through a real expression AST (tokenizer +
  precedence-climbing parser + emitter) instead of regex rewriting.
* Saved-variable analysis walks the AST once to decide which forward inputs /
  outputs each backward node stores, so the node struct, its constructor, and
  every call site agree by construction.
* Ops whose backward cannot be expressed in the formula DSL (list-mapping
  backwards like ``cat``/``stack``/``roll``) are declared in
  ``MANUAL_DERIVATIVES`` -- the same information upstream records as
  hand-written ``derivatives.yaml`` entries -- instead of being special-cased
  inline in the orchestrator.
"""

from __future__ import annotations


_COMPARISON_OPS = {"<=": "le", ">=": "ge", "==": "eq", "!=": "ne", "<": "lt", ">": "gt"}


def _normalize_comparisons(formula: str) -> str:
    """Rewrite (a OP b) comparisons into dispatched op calls.

    torch's derivatives.yaml permits comparison operators inside derivative
    formulas (they produce bool masks); TensorPlay's expression DSL has no
    infix comparisons, so translate them to the dispatched gt/lt/... ops.
    """
    out = formula
    for _ in range(10):
        m = re.search(r"\(([^()]+?)\s*(<=|>=|==|!=|<|>)\s*([^()]+?)\)", out)
        if not m:
            break
        op = _COMPARISON_OPS[m.group(2)]
        out = out[:m.start()] + f"{op}({m.group(1)}, {m.group(3)})" + out[m.end():]
    return out


import re
from dataclasses import dataclass, field

from .api_types import autograd_node_name, node_member_type
from .model import Argument, NativeFunction


# ---------------------------------------------------------------------------
# Expression AST
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Expr:
    pass

@dataclass(frozen=True)
class Num(Expr):
    text: str

@dataclass(frozen=True)
class BoolLit(Expr):
    text: str

@dataclass(frozen=True)
class StrLit(Expr):
    text: str

@dataclass(frozen=True)
class Var(Expr):
    name: str

@dataclass(frozen=True)
class Call(Expr):
    callee: str            # bare name or qualified (`std::get<0>`, `tpx_helper`)
    args: tuple[Expr, ...]

@dataclass(frozen=True)
class Method(Expr):
    receiver: Expr
    name: str
    args: tuple[Expr, ...]

@dataclass(frozen=True)
class BinOp(Expr):
    op: str                # + - * /
    left: Expr
    right: Expr

@dataclass(frozen=True)
class Braced(Expr):
    """C++ braced-init-list argument, e.g. `{dim}` or `{0, 1}`."""
    items: tuple[Expr, ...]

@dataclass(frozen=True)
class Neg(Expr):
    value: Expr


_TOKEN_RE = re.compile(
    r"""\s*(?:
        (?P<num>-?\d+\.\d+(?:[eE][+-]?\d+)?|-?\.\d+|-?\d+)
      | (?P<ident>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*(?:<\d+>)?)
      | (?P<str>"(?:[^"\\]|\\.)*")
      | (?P<bool>true|false)
      | (?P<cmp><=|>=|==|!=|<|>)
      | (?P<punct>[().,\[\]{}])
      | (?P<op>[-+*/])
    )""",
    re.VERBOSE,
)


def tokenize_expr(s: str):
    pos, toks = 0, []
    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if not m:
            if s[pos:].strip() == "":
                break
            raise ValueError(f"Cannot tokenize derivative formula at: {s[pos:]!r}")
        pos = m.end()
        for g in ("num", "ident", "str", "bool", "cmp", "punct", "op"):
            v = m.group(g)
            if v is not None:
                toks.append((g, v))
                break
    return toks


class ExprParser:
    """Precedence-climbing parser: +- < */ < unary- < call/method/postfix."""

    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def take(self):
        t = self.peek()
        self.i += 1
        return t

    def expect(self, val):
        k, v = self.take()
        if v != val:
            raise ValueError(f"Expected {val!r}, got {v!r}")

    def parse(self) -> Expr:
        e = self.parse_cmp()
        if self.i != len(self.toks):
            raise ValueError(f"Trailing tokens in expression: {self.toks[self.i:]}")
        return e

    def parse_cmp(self) -> Expr:
        e = self.parse_add()
        while self.peek()[1] in ("<=", ">=", "==", "!=", "<", ">"):
            op = self.take()[1]
            e = BinOp(op, e, self.parse_add())
        return e

    def parse_add(self) -> Expr:
        e = self.parse_mul()
        while self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            e = BinOp(op, e, self.parse_mul())
        return e

    def parse_mul(self) -> Expr:
        e = self.parse_unary()
        while self.peek()[1] in ("*", "/"):
            op = self.take()[1]
            e = BinOp(op, e, self.parse_unary())
        return e

    def parse_unary(self) -> Expr:
        if self.peek()[1] == "-":
            self.take()
            return Neg(self.parse_unary())
        return self.parse_postfix()

    def parse_postfix(self) -> Expr:
        e = self.parse_primary()
        while self.peek()[1] == ".":
            self.take()
            kind, name = self.take()
            if kind != "ident":
                raise ValueError("Expected method name after '.'")
            args = ()
            if self.peek()[1] == "(":
                args = tuple(self.parse_call_args())
            e = Method(e, name.split("::")[-1], args)
        return e

    def parse_call_args(self) -> list[Expr]:
        self.expect("(")
        out = []
        if self.peek()[1] != ")":
            out.append(self.parse_add())
            while self.peek()[1] == ",":
                self.take()
                out.append(self.parse_add())
        self.expect(")")
        return out

    def parse_primary(self) -> Expr:
        kind, val = self.take()
        if kind == "num":
            return Num(val)
        if kind == "bool":
            return BoolLit(val)
        if kind == "str":
            return StrLit(val)
        if kind == "ident":
            if self.peek()[1] == "(":
                return Call(val, tuple(self.parse_call_args()))
            return Var(val)
        if val == "(":
            e = self.parse_add()
            self.expect(")")
            return e
        if val == "{":
            items = []
            if self.peek()[1] != "}":
                items.append(self.parse_add())
                while self.peek()[1] == ",":
                    self.take()
                    items.append(self.parse_add())
            self.expect("}")
            return Braced(tuple(items))
        raise ValueError(f"Unexpected token {val!r}")


def parse_expr(formula: str) -> Expr:
    return ExprParser(tokenize_expr(formula)).parse()


# ---------------------------------------------------------------------------
# Formula rendering (the equivalent of torchgen's autograd codegen emit)
# ---------------------------------------------------------------------------

# Free functions invoked by formulas that live in tensorplay::tpx::ops and
# are numeric backward kernels rather than autograd building blocks.
BACKWARD_HELPERS = {
    "clamp_backward", "threshold_backward", "nll_loss_backward",
    "mse_loss_backward", "max_pool2d_backward", "adaptive_avg_pool2d_backward",
    "adaptive_max_pool2d_backward", "batch_norm_backward", "layer_norm_backward",
    "group_norm_backward", "instance_norm_backward", "constant_pad_nd_backward",
    "conv1d_grad_input", "conv1d_grad_weight", "conv1d_grad_bias",
    "conv2d_grad_input", "conv2d_grad_weight", "conv2d_grad_bias",
    "conv3d_grad_input", "conv3d_grad_weight", "conv3d_grad_bias",
    "conv_transpose2d_grad_input", "conv_transpose2d_grad_weight",
    "conv_transpose2d_grad_bias", "conv_transpose3d_grad_input",
    "conv_transpose3d_grad_weight", "conv_transpose3d_grad_bias",
    "embedding_dense_backward", "permute_backward", "squeeze_backward",
}

# Tensor-returning methods that the DSL lowers to tpx::ops free functions,
# matching upstream's treatment of view/shape primitives in derivatives.
TENSOR_METHODS = {
    "neg": "neg", "t": "t", "mm": "mm", "matmul": "matmul",
    "transpose": "transpose", "squeeze": "squeeze", "unsqueeze": "unsqueeze",
    "permute": "permute", "view": "view", "reshape": "reshape",
    "expand": "expand", "sum": "sum", "mean": "mean", "pow": "pow",
    "sqrt": "sqrt", "sin": "sin", "cos": "cos", "exp": "exp", "log": "log",
    "tanh": "tanh", "sigmoid": "sigmoid", "relu": "relu", "softmax": "softmax",
    "log_softmax": "log_softmax", "abs": "abs", "square": "square",
    "sign": "sign", "mul": "mul", "add": "add", "sub": "sub", "div": "div",
    "atan2": "atan2", "clamp": "clamp", "lerp": "lerp", "clone": "clone",
    "detach": "detach", "contiguous": "contiguous", "select": "select",
    "slice": "slice", "t_": "t_",
}

_GRAD_SYMBOLS = {"grad", "grad_output"}


class Emitter:
    """Renders an Expr back to C++, lowering tensor arithmetic onto the
    tpx::ops free functions (add/sub/mul/div/neg) exactly like the reference
    translation tables."""

    def __init__(self, tensor_syms: set[str], member_names: set[str]):
        self.tensor_syms = set(tensor_syms) | _GRAD_SYMBOLS
        self.members = set(member_names)

    # -- static tensor-ness analysis ----------------------------------------
    def _is_tensor(self, e: Expr) -> bool:
        if isinstance(e, Var):
            return e.name in self.tensor_syms
        if isinstance(e, Neg):
            return self._looks_tensor(e.value)
        if isinstance(e, Method):
            base = e.name
            while base.endswith("_"):
                base = base[:-1]
            return base in TENSOR_METHODS
        if isinstance(e, Call):
            leaf = e.callee.split("::")[-1].split("<")[0]
            return leaf not in ("Scalar",)
        if isinstance(e, BinOp):
            return self._is_tensor(e.left) or self._looks_tensor(e.right)
        return False

    def _looks_tensor(self, e: Expr) -> bool:
        if self._is_tensor(e):
            return True
        if isinstance(e, BinOp):
            return True
        return False

    def var_name(self, name: str) -> str:
        return f"{name}_" if name in self.members else name

    def emit(self, e: Expr) -> str:
        if isinstance(e, Num):
            return e.text
        if isinstance(e, BoolLit):
            return e.text
        if isinstance(e, StrLit):
            return e.text
        if isinstance(e, Var):
            return self.var_name(e.name)
        if isinstance(e, Neg):
            inner = self.emit(e.value)
            if self._looks_tensor(e.value):
                return f"neg({inner})"
            return f"-{inner}"
        if isinstance(e, Braced):
            return "{" + ", ".join(self.emit(a) for a in e.items) + "}"
        if isinstance(e, Call):
            args = ", ".join(self.emit(a) for a in e.args)
            leaf = e.callee.split("::")[-1].split("<")[0]
            if leaf == "Scalar":
                return f"Scalar({args})" if args else "Scalar()"
            return f"{e.callee}({args})"
        if isinstance(e, Method):
            recv = self.emit(e.receiver)
            base = e.name[:-1] if e.name.endswith("_") and e.name[:-1] in TENSOR_METHODS else e.name
            args = ", ".join(self.emit(a) for a in e.args)
            if base in TENSOR_METHODS:
                return f"{TENSOR_METHODS[base]}({recv}, {args})" if args else f"{TENSOR_METHODS[base]}({recv})"
            return f"{recv}.{e.name}({args})" if args else f"{recv}.{e.name}()"
        if isinstance(e, BinOp):
            l_txt = self.emit(e.left)
            r_txt = self.emit(e.right)
            # Comparison masks stay textual C++ (Tensor/Scalar operator<).
            if e.op in ("<=", ">=", "==", "!=", "<", ">"):
                return f"{l_txt} {e.op} {r_txt}"
            l_tensor = self._is_tensor(e.left)
            r_tensor = self._looks_tensor(e.right)
            if e.op in "+-" and l_tensor:
                return f"{'add' if e.op == '+' else 'sub'}({l_txt}, {r_txt})"
            if e.op == "*" and l_tensor:
                return f"mul({l_txt}, {r_txt})"
            if e.op == "/" and l_tensor:
                return f"div({l_txt}, {r_txt})"
            if e.op == "*" and r_tensor:
                return f"mul({r_txt}, {l_txt})"
            if e.op == "-" and r_tensor:
                return f"neg(sub({r_txt}, {l_txt}))"
            return f"{l_txt} {e.op} {r_txt}"
        raise ValueError(f"Unsupported expr node: {e!r}")


def render_formula(expr: Expr, tensor_syms: set[str], member_names: set[str]) -> str:
    return Emitter(tensor_syms, member_names).emit(expr)


def _iter_call_nodes(expr: Expr):
    """Yield every Call/Method subtree of the expression."""
    stack = [expr]
    while stack:
        e = stack.pop()
        if isinstance(e, (Call, Method)):
            yield e
        if isinstance(e, Neg):
            stack.append(e.value)
        elif isinstance(e, Braced):
            stack.extend(e.items)
        elif isinstance(e, Call):
            stack.extend(e.args)
        elif isinstance(e, Method):
            stack.append(e.receiver)
            stack.extend(e.args)
        elif isinstance(e, BinOp):
            stack.append(e.left)
            stack.append(e.right)


def collect_vars(expr: Expr, out: set[str]) -> None:
    if isinstance(expr, Var):
        out.add(expr.name)
    elif isinstance(expr, Neg):
        collect_vars(expr.value, out)
    elif isinstance(expr, Call):
        for a in expr.args:
            collect_vars(a, out)
    elif isinstance(expr, Braced):
        for a in expr.items:
            collect_vars(a, out)
    elif isinstance(expr, Method):
        collect_vars(expr.receiver, out)
        for a in expr.args:
            collect_vars(a, out)
    elif isinstance(expr, BinOp):
        collect_vars(expr.left, out)
        collect_vars(expr.right, out)


# ---------------------------------------------------------------------------
# Typed derivatives
# ---------------------------------------------------------------------------

@dataclass
class OpDerivatives:
    func: NativeFunction
    node_name: str
    formulas: dict[str, Expr]           # gradient slot arg name -> d(out)/d(arg)
    grad_slots: list[Argument]          # tensor-like forward args, schema order
    members: list[tuple[str, str]]      # saved state: (member name, C++ type)
    used_input_names: set[str] = field(default_factory=set)
    used_output_names: set[str] = field(default_factory=set)


# Backwards that cannot be written in the formula DSL because they map over a
# tensor list.  Declared here (upstream keeps equivalent hand-written nodes);
# `saved` lists forward inputs stored by the manual node in ManualNodes.h.
MANUAL_DERIVATIVES: dict[str, dict] = {
    "cat": {"saved": ["tensors", "dim"]},
    "stack": {"saved": ["tensors", "dim"]},
    "roll": {"saved": ["shifts", "dims"]},
}

# Ops whose backward node is provided hand-written elsewhere; skip emitting a
# generated class even though derivatives exist.
EXTERNAL_NODES = {"scaled_dot_product_attention", "mean"}


def compute_op_derivatives(func: NativeFunction, raw_formulas: dict[str, str],
                           node_name: str | None = None) -> OpDerivatives:
    """Analyze one op's derivative formulas into node layout + call info."""
    node_name = node_name or autograd_node_name(func.func_name)

    parsed = {k: parse_expr(v) for k, v in raw_formulas.items()}
    # Formula keys may be gradient slots (forward arg names) or named outputs
    # (`result` / tuple element names).

    arg_names = {a.name for a in func.args}
    output_names = {"result"}
    if func.cpp_return_kind == "tuple":
        from .api_types import tuple_element_names
        output_names.update(tuple_element_names(func))

    used: set[str] = set()
    for e in parsed.values():
        collect_vars(e, used)
    used &= arg_names | output_names

    members: list[tuple[str, str]] = []
    for a in func.args:
        if a.name in used:
            members.append((a.name, node_member_type(a.type)))
    if func.cpp_return_kind == "tuple":
        from .api_types import tuple_element_cpp_types, tuple_element_names
        for i, nm in enumerate(tuple_element_names(func)):
            if nm in used:
                members.append((nm, tuple_element_cpp_types(func)[i]))
    elif "result" in used:
        rt = func.returns[0].type
        members.append(("result", "std::vector<Tensor>" if rt.is_list else "Tensor"))

    grad_slots = [
        a for a in func.args
        if (a.type.is_tensor_like and not a.type.is_list) or a.type.is_mutable_ref
    ]
    return OpDerivatives(
        func=func, node_name=node_name, formulas=parsed,
        grad_slots=grad_slots, members=members,
        used_input_names={m for m, _ in members} & arg_names,
        used_output_names={m for m, _ in members} & output_names,
    )


def load_derivatives(path: str, native_by_opname: dict[str, NativeFunction]) \
        -> dict[str, OpDerivatives]:
    """Parse derivatives.yaml keyed by dispatcher op name."""
    from .model import parse_schema, parse_derivatives_yaml

    out: dict[str, OpDerivatives] = {}
    for item in parse_derivatives_yaml(path):
        f = parse_schema(item["name"])
        op = f.func_name
        native = native_by_opname.get(op)
        if native is None:
            continue
        if op in EXTERNAL_NODES:
            continue
        raw = {}
        for key in ("self", "result") + tuple(a.name for a in native.args):
            if key in item:
                raw[key] = _normalize_comparisons(item[key])
        for decl in native.returns:
            if decl.name and decl.name in item:
                raw[decl.name] = item[decl.name]
        if raw:
            out[op] = compute_op_derivatives(native, raw)

    # Manual (hand-written) backwards: register their saved-state layout so
    # wrapper generation treats them uniformly.
    for base, spec in MANUAL_DERIVATIVES.items():
        cand = [n for op, n in native_by_opname.items() if op.split(".")[0] == base]
        if not cand:
            continue
        native = sorted(cand, key=lambda n: len(n.overload_name))[0]
        saved = [a for a in native.args if a.name in spec["saved"]]
        members = [(a.name, node_member_type(a.type)) for a in saved]
        out[native.func_name] = OpDerivatives(
            func=native, node_name=autograd_node_name(base),
            formulas={}, grad_slots=native.tensor_args,
            members=members,
            used_input_names=set(spec["saved"]), used_output_names=set(),
        )
    return out


# ---------------------------------------------------------------------------
# AutogradNodesGenerated.h
# ---------------------------------------------------------------------------

_HEADER_PRELUDE = """// Generated by tools/codegen/gen.py -- DO NOT EDIT
#pragma once
#include "Node.h"
#include "Autograd.h"
#include "ManualNodes.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include <algorithm>
#include <utility>
#include "Scalar.h"
#include <vector>
#include <cstdint>
#include <cstdio>

namespace tensorplay {
namespace tpx {
using namespace ops;
"""


def generate_autograd_nodes(derivatives: dict[str, OpDerivatives]) -> str:
    lines = [_HEADER_PRELUDE.rstrip("\n"), ""]

    emitted: set[str] = set()
    for op, dv in derivatives.items():
        if dv.node_name in emitted or not dv.formulas:
            continue
        emitted.add(dv.node_name)
        f = dv.func
        member_names = {m for m, _ in dv.members}
        tensor_syms = {a.name for a in f.args if a.type.is_tensor_like}
        # Saved forward outputs referenced by formulas (`result`, named tuple
        # elements) are tensor symbols too.
        if "result" in member_names:
            tensor_syms.add("result")
        if f.cpp_return_kind == "tuple":
            from .api_types import tuple_element_names
            tensor_syms.update(n for n in tuple_element_names(f) if n in member_names)

        lines.append(f"struct {dv.node_name} : public Node {{")
        for m, t in dv.members:
            lines.append(f"    {t} {m}_;")
        lines.append("")
        ctor_args = [f"{t} {m}" for m, t in dv.members]
        ctor_inits = [f"{m}_({m})" for m, _ in dv.members]
        lines.append(f"    explicit {dv.node_name}({', '.join(ctor_args)})")
        if ctor_inits:
            lines.append(f"        : {', '.join(ctor_inits)} {{}}")
        else:
            lines.append("        {}")
        lines.append("")
        lines.append("    variable_list apply(variable_list&& inputs) override {")

        n_slots = len(dv.grad_slots)
        undef = ", ".join(["Tensor()"] * n_slots)
        lines.append(f"        if (inputs.empty() || !inputs[0].defined()) return {{{undef}}};")
        lines.append("        const Tensor& grad = inputs[0];")
        lines.append("")
        lines.append("        variable_list grads;")

        # Common-subexpression elimination: identical Call sub-expressions
        # shared across gradient slots are evaluated once (generalizes
        # upstream's hand-written `shared` blocks, e.g. batch_norm's
        # three-way backward kernel call).
        cse_temps: dict[str, str] = {}
        call_counts: dict[str, int] = {}
        em = Emitter(tensor_syms, member_names)
        for a in dv.grad_slots:
            expr = dv.formulas.get(a.name)
            if expr is None:
                continue
            for node in _iter_call_nodes(expr):
                txt = em.emit(node)
                call_counts[txt] = call_counts.get(txt, 0) + 1

        lines.append("        variable_list grads;")
        for txt, count in call_counts.items():
            if count < 2:
                continue
            temp = f"__shared_{len(cse_temps)}"
            cse_temps[txt] = temp
            lines.append(f"        auto {temp} = {txt};")

        for a in dv.grad_slots:
            expr = dv.formulas.get(a.name)
            if expr is None:
                lines.append("        grads.push_back(Tensor());")
                continue
            txt = render_formula(expr, tensor_syms, member_names)
            # Splice shared temporaries into the rendered formula (longest
            # first so nested shared calls splice cleanly).
            for t in sorted(cse_temps, key=len, reverse=True):
                if t in txt:
                    txt = txt.replace(t, cse_temps[t])
            if txt in cse_temps:
                lines.append(f"        grads.push_back({cse_temps[txt]});")
                continue
            var = f"__grad_{a.name}"
            lines.append(f"        auto {var} = {txt};")
            lines.append(f"        grads.push_back({var});")
        lines.append("        return grads;")
        lines.append("    }")
        lines.append("};")
        lines.append("")

    lines.append("} // namespace tpx")
    lines.append("} // namespace tensorplay")
    return "\n".join(lines) + "\n"
