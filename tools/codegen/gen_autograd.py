"""Load derivative formulas and generate backward-node metadata.

The input contains a schema name plus one gradient formula per differentiable input /
  output) into typed objects.
* Gradient formulas are compiled through a real expression AST (tokenizer +
  precedence-climbing parser + emitter) instead of regex rewriting.
* Saved-variable analysis walks the AST once to decide which forward inputs /
  outputs each backward node stores, so the node struct, its constructor, and
  every call site agree by construction.
* Ops whose backward cannot be expressed in the formula DSL (list-mapping
  backwards like ``cat``/``stack``/``roll``) are declared in
  ``MANUAL_DERIVATIVES`` instead of being special-cased inline in the
  orchestrator.
"""

from __future__ import annotations


_COMPARISON_OPS = {"<=": "le", ">=": "ge", "==": "eq", "!=": "ne", "<": "lt", ">": "gt"}


def _normalize_comparisons(formula: str) -> str:
    """Rewrite (a OP b) comparisons into dispatched op calls.

    formulas (they produce bool masks); TensorPlay's expression DSL has no
    infix comparisons, so translate them to the dispatched gt/lt/... ops.
    Handles both parenthesized groups and bare `a > b` operands used by
    clamp_min/clamp_max formulas.
    """
    out = formula
    for _ in range(10):
        m = re.search(r"\(([^()]+?)\s*(<=|>=|==|!=|<|>)\s*([^()]+?)\)", out)
        if not m:
            break
        op = _COMPARISON_OPS[m.group(2)]
        out = out[:m.start()] + f"{op}({m.group(1)}, {m.group(3)})" + out[m.end():]

    def _bare(m):
        return f"{_COMPARISON_OPS[m.group(2)]}({m.group(1)}, {m.group(3)})"

    out = re.sub(
        r"(?<![\w.])([A-Za-z_][\w.:]*)\s*(<=|>=|==|!=|<|>)\s*([A-Za-z_][\w.:]*)",
        _bare,
        out,
    )
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
class Not(Expr):
    value: Expr


@dataclass(frozen=True)
class Ternary(Expr):
    cond: Expr
    then: Expr
    other: Expr


@dataclass(frozen=True)
class Braced(Expr):
    """C++ braced-init-list argument, e.g. `{dim}` or `{0, 1}`."""
    items: tuple[Expr, ...]

@dataclass(frozen=True)
class Paren(Expr):
    """Parenthesized subexpression; keeps the author's grouping so re-emitted
    formulas do not lose precedence (e.g. `1.0 / (1.0 - p)`)."""
    value: Expr

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
      | (?P<logic>&&|\|\||!)
      | (?P<quest>\?)
      | (?P<colon>:)
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
        for g in ("num", "ident", "str", "bool", "cmp", "logic", "quest",
                  "colon", "punct", "op"):
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
        e = self.parse_ternary()
        if self.i != len(self.toks):
            raise ValueError(f"Trailing tokens in expression: {self.toks[self.i:]}")
        return e

    def parse_ternary(self) -> Expr:
        cond = self.parse_or()
        if self.peek()[1] == "?":
            self.take()
            then = self.parse_ternary()
            self.expect(":")
            return Ternary(cond, then, self.parse_ternary())
        return cond

    def parse_or(self) -> Expr:
        e = self.parse_and()
        while self.peek()[1] == "||":
            self.take()
            e = BinOp("||", e, self.parse_and())
        return e

    def parse_and(self) -> Expr:
        e = self.parse_cmp()
        while self.peek()[1] == "&&":
            self.take()
            e = BinOp("&&", e, self.parse_cmp())
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
        if self.peek()[1] == "!":
            self.take()
            return Not(self.parse_unary())
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
            out.append(self.parse_ternary())
            while self.peek()[1] == ",":
                self.take()
                out.append(self.parse_ternary())
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
            e = self.parse_ternary()
            self.expect(")")
            return Paren(e)
        if val == "{":
            items = []
            if self.peek()[1] != "}":
                items.append(self.parse_ternary())
                while self.peek()[1] == ",":
                    self.take()
                    items.append(self.parse_ternary())
            self.expect("}")
            return Braced(tuple(items))
        raise ValueError(f"Unexpected token {val!r}")


def parse_expr(formula: str) -> Expr:
    return ExprParser(tokenize_expr(formula)).parse()


# ---------------------------------------------------------------------------
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
    "_alpha_dropout_backward", "_feature_dropout_backward",
    "_trapezoid_backward", "_cumulative_trapezoid_backward",
    "_cov_backward", "_corrcoef_backward",
    "_scatter_reduce_backward_self", "_scatter_reduce_backward_src",
    "_index_reduce_backward_self", "_index_reduce_backward_src",
}

# Tensor-returning methods that the DSL lowers to tpx::ops free functions,
# used for view/shape primitives in derivatives.
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
    "slice": "slice", "t_": "t_", "conj": "conj",
    # A type conversion inside a formula stays on the recorded graph.
    "to": "tensorplay::tpx::to",
}

_GRAD_SYMBOLS = {"grad", "grad_output"}


class Emitter:
    """Renders an Expr back to C++, lowering tensor arithmetic onto the
    tpx::ops free functions (add/sub/mul/div/neg) for generated translation
    tables."""

    def __init__(self, tensor_syms: set[str], member_names: set[str],
                 tensor_member_names: set[str] = frozenset(),
                 native_op_names: set[str] = frozenset(),
                 attribute_members: dict[str, tuple[str, str]] | None = None):
        self.tensor_syms = set(tensor_syms) | _GRAD_SYMBOLS
        self.members = set(member_names)
        # Saved forward tensors become SavedVariable members and are unpacked
        # into `{name}_sv` locals at the top of apply(); formulas reference
        # those locals so every use goes through the version check.
        self.tensor_members = set(tensor_member_names)
        self.native_op_names = set(native_op_names)
        # `arg.attr()` of a tensor saved only by its attributes reads the
        # member holding that attribute.
        self.attributes = {key: member
                           for member, key in (attribute_members or {}).items()}

    def op_name(self, name: str) -> str:
        if "::" not in name and name in self.native_op_names:
            return f"ops::{name}"
        return name

    # -- static tensor-ness analysis ----------------------------------------
    def _is_tensor(self, e: Expr) -> bool:
        if isinstance(e, Paren):
            return self._is_tensor(e.value)
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
        if isinstance(e, Paren):
            return self._looks_tensor(e.value)
        if self._is_tensor(e):
            return True
        if isinstance(e, BinOp):
            return True
        return False

    def var_name(self, name: str) -> str:
        if name in self.tensor_members:
            return f"{name}_sv"
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
        if isinstance(e, Not):
            return f"!{self.emit(e.value)}"
        if isinstance(e, Ternary):
            return (f"{self.emit(e.cond)} ? {self.emit(e.then)}"
                    f" : {self.emit(e.other)}")
        if isinstance(e, Braced):
            return "{" + ", ".join(self.emit(a) for a in e.items) + "}"
        if isinstance(e, Paren):
            return f"({self.emit(e.value)})"
        if isinstance(e, Call):
            args = ", ".join(self.emit(a) for a in e.args)
            leaf = e.callee.split("::")[-1].split("<")[0]
            if leaf == "Scalar":
                return f"Scalar({args})" if args else "Scalar()"
            return f"{self.op_name(e.callee)}({args})"
        if isinstance(e, Method):
            if (isinstance(e.receiver, Var) and not e.args
                    and (e.receiver.name, e.name) in self.attributes):
                return f"{self.attributes[(e.receiver.name, e.name)]}_"
            recv = self.emit(e.receiver)
            base = e.name[:-1] if e.name.endswith("_") and e.name[:-1] in TENSOR_METHODS else e.name
            args = ", ".join(self.emit(a) for a in e.args)
            if base in TENSOR_METHODS:
                name = self.op_name(TENSOR_METHODS[base])
                return f"{name}({recv}, {args})" if args else f"{name}({recv})"
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


def render_formula(expr: Expr, tensor_syms: set[str], member_names: set[str],
                   tensor_member_names: set[str] = frozenset(),
                   native_op_names: set[str] = frozenset(),
                   attribute_members: dict[str, tuple[str, str]] | None = None) -> str:
    return Emitter(tensor_syms, member_names, tensor_member_names,
                   native_op_names, attribute_members).emit(expr)


def _iter_call_nodes(expr: Expr):
    """Yield every Call/Method subtree of the expression."""
    stack = [expr]
    while stack:
        e = stack.pop()
        if isinstance(e, (Call, Method)):
            yield e
        if isinstance(e, Neg):
            stack.append(e.value)
        elif isinstance(e, Paren):
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
    elif isinstance(expr, Paren):
        collect_vars(expr.value, out)
    elif isinstance(expr, Not):
        collect_vars(expr.value, out)
    elif isinstance(expr, Ternary):
        collect_vars(expr.cond, out)
        collect_vars(expr.then, out)
        collect_vars(expr.other, out)
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
    # Members holding one attribute of a forward tensor the formulas read
    # nothing else of: member name -> (argument, attribute method).
    attribute_members: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Saved tensors only some gradient slots read: member name -> the slot
    # arguments whose wanting a gradient is the reason to keep it.
    conditional_members: dict[str, list[str]] = field(default_factory=dict)
    # All forward outputs are marked non-differentiable: the autograd wrapper
    # still registers (so the dispatch chain resolves) but builds no backward
    # node and leaves the outputs detached.
    non_differentiable_output: bool = False
    differentiable_outputs: list[bool] | None = None
    # Inputs declared `non_differentiable`: read for their shape or values
    # only, so they never make the output require a gradient.
    non_differentiable_args: frozenset[str] = frozenset()
    # Forward-mode derivatives: output name -> jvp formula over each
    # argument's primal value "{arg}_p" and tangent "{arg}_t".
    fw_formulas: dict[str, Expr] = field(default_factory=dict)
    # Arguments whose tangent / primal value the jvp formula reads; drives
    # the local declarations and the any-tangent-defined guard emitted into
    # the generated wrapper.
    fw_required_tangent: list[str] = field(default_factory=list)
    fw_required_primal: list[str] = field(default_factory=list)


_IDENT_BOUNDARY = r"(?<![\w.]){name}(?![\w])"


def _fw_requirements(expr: Expr, arg_names: set[str]) -> tuple[list[str], list[str]]:
    """Names whose tangent / primal the formula reads, in schema order."""
    used: set[str] = set()
    collect_vars(expr, used)
    tangent = {v[:-2] for v in used if v.endswith("_t") and v[:-2] in arg_names}
    primal = {v[:-2] for v in used if v.endswith("_p") and v[:-2] in arg_names}
    return tangent, primal


def _expand_fw_formula(func: NativeFunction, out_name: str, formula: str,
                       raw_formulas: dict[str, str]) -> tuple[Expr, list[str], list[str]]:
    """Resolve one forward-derivative formula.

    ``auto_linear`` re-evaluates the op on the tangents of every
    differentiable argument (valid because the Jacobian of a linear map is
    the map itself); ``auto_element_wise`` reuses the backward formula with
    ``grad`` replaced by the conjugated tangent and the input by its primal
    (valid for single-input element-wise maps, where the Jacobian is
    diagonal).  Both yield the conjugate-wrapped result required for
    complex-valued inputs.
    """
    arg_names = {a.name for a in func.args}
    diff_args = [a.name for a in func.args if a.name in raw_formulas]
    if formula == "auto_linear":
        if out_name != "result":
            raise ValueError(
                f"auto_linear forward derivative of {func.func_name} requires "
                "a single 'result' output")
        if not diff_args:
            raise ValueError(
                f"auto_linear forward derivative of {func.func_name} needs at "
                "least one differentiable argument")
        new_args = [a.name + "_t" if a.name in diff_args else a.name
                    for a in func.args]
        return parse_expr(f"{func.base_name}({', '.join(new_args)})"), \
            list(diff_args), []
    if formula == "auto_element_wise":
        if len(diff_args) != 1 or out_name != "result":
            raise ValueError(
                f"auto_element_wise forward derivative of {func.func_name} "
                "requires a single differentiable argument and a single result")
        inp = diff_args[0]
        backward = raw_formulas.get(inp)
        if backward is None:
            raise ValueError(
                f"auto_element_wise forward derivative of {func.func_name} "
                f"requires a backward formula for {inp!r}")
        fw = re.sub(_IDENT_BOUNDARY.format(name="grad"),
                    f"{inp}_t.conj()", backward)
        fw = re.sub(_IDENT_BOUNDARY.format(name=inp), f"{inp}_p", fw)
        return parse_expr(f"({fw}).conj()"), [inp], [inp]
    expr = parse_expr(formula)
    tangent, primal = _fw_requirements(expr, arg_names)
    return expr, list(tangent), list(primal)


# Backwards that cannot be written in the formula DSL because they map over a
# tensor list.  Declared here as hand-written nodes;
# `saved` lists forward inputs stored by the manual node in ManualNodes.h.
MANUAL_DERIVATIVES: dict[str, dict] = {
    "block_diag": {"saved": ["tensors"]},
    "mean": {"saved": ["self"]},
    "cat": {"saved": ["tensors", "dim"]},
    "stack": {"saved": ["tensors", "dim"]},
    "roll": {"saved": ["shifts", "dims"]},
    # Dim-dependent case split (dot / vec@mat / mat@vec / batched); the node
    # composes recordable primitives so double-backward through `@` records.
    "matmul": {"saved": ["self", "other"]},
    # List-gradient alignment: apply() pads the index slots so outputs line
    # up with the per-element edges collected for the Tensor[] indices.
    "index": {"saved": ["self", "indices"]},
    "index_put": {"saved": ["indices", "values", "accumulate"]},
    "index_put_": {"saved": ["indices", "values", "accumulate"]},
    "_index_put_impl_": {"saved": ["indices", "values", "accumulate"],
                         "node": "IndexPutBackward"},
    # Two differentiable outputs: the node reads grads[0]/grads[1].
    "aminmax": {"saved": ["self", "dim", "keepdim"]},
    "std_mean": {"saved": ["self", "dim", "unbiased", "keepdim"]},
    "var_mean": {"saved": ["self", "dim", "unbiased", "keepdim"]},
    # RNN layer backwards: replay-based native nodes (RNNBackward.h); grads
    # flow to input, every hx element and every parameter in schema order.
    "lstm": {"saved": ["input", "hx", "params", "has_biases", "num_layers",
                       "dropout_p", "training", "bidirectional", "batch_first"]},
    "gru": {"saved": ["input", "hx", "params", "has_biases", "num_layers",
                      "dropout_p", "training", "bidirectional", "batch_first"]},
    "rnn_tanh": {"saved": ["input", "hx", "params", "has_biases", "num_layers",
                           "dropout_p", "training", "bidirectional",
                           "batch_first"]},
    "rnn_relu": {"saved": ["input", "hx", "params", "has_biases", "num_layers",
                           "dropout_p", "training", "bidirectional",
                           "batch_first"]},
    # Factorizations with several differentiable outputs (LinalgBackward.h):
    # the node receives one gradient per output.  `saved_outputs` are handed
    # to the node after the saved inputs, in the order listed.
    "_linalg_svd": {"saved": ["full_matrices", "compute_uv"],
                    "saved_outputs": ["U", "S", "Vh"],
                    "node": "LinalgSvdBackward"},
    "_linalg_eigh": {"saved": ["compute_v"],
                     "saved_outputs": ["eigenvalues", "eigenvectors"],
                     "node": "LinalgEighBackward"},
    "linalg_eig": {"saved": ["A"],
                   "saved_outputs": ["eigenvalues", "eigenvectors"]},
    "linalg_qr": {"saved": ["mode"], "saved_outputs": ["Q", "R"]},
    # The two-output loss backward kernels return one gradient per input
    # (LossBackward.h); the node receives one incoming gradient for each.
    "tp_margin_ranking_loss_backward": {
        "saved": ["grad_output", "input1", "input2", "target", "margin", "reduction"],
        "node": "TpMarginRankingLossBackwardBackward"},
    "tp_cosine_embedding_loss_backward": {
        "saved": ["grad_output", "input1", "input2", "target", "margin", "reduction"],
        "node": "TpCosineEmbeddingLossBackwardBackward"},
    # Convolution and normalization backward kernels return one gradient per
    # input of the forward op (ConvNormBackward.h); the node receives one
    # incoming gradient for each.
    "convolution_backward": {
        "saved": ["grad_output", "input", "weight", "stride", "padding", "dilation",
                  "transposed", "output_padding", "groups"],
        "node": "ConvolutionBackwardBackward"},
    "convolution_backward_overrideable": {
        "saved": ["grad_output", "input", "weight", "stride", "padding", "dilation",
                  "transposed", "output_padding", "groups"],
        "node": "ConvolutionBackwardOverrideableBackward"},
    "batch_norm_backward": {
        "saved": ["grad_output", "input", "weight", "running_mean", "running_var",
                  "training", "eps"],
        "node": "BatchNormBackwardBackward"},
    "instance_norm_backward": {
        "saved": ["grad_output", "input", "weight", "bias", "running_mean",
                  "running_var", "use_input_stats", "eps"],
        "node": "InstanceNormBackwardBackward"},
    "group_norm_backward": {
        "saved": ["grad_output", "input", "num_groups", "weight", "bias", "eps"],
        "node": "GroupNormBackwardBackward"},
    "native_group_norm_backward": {
        "saved": ["grad_out", "input", "mean", "rstd", "weight", "N", "C", "HxW", "group"],
        "node": "NativeGroupNormBackwardBackward"},
    "native_layer_norm_backward": {
        "saved": ["grad_out", "input", "normalized_shape", "mean", "rstd", "weight", "bias"],
        "node": "NativeLayerNormBackwardBackward"},
    # The grid sampler backwards differentiate again (GridSamplerBackward.h).
    "grid_sampler_2d_backward": {
        "saved": ["grad_output", "input", "grid", "interpolation_mode", "padding_mode",
                  "align_corners"],
        "node": "GridSampler2dBackwardBackward"},
    "grid_sampler_3d_backward": {
        "saved": ["grad_output", "input", "grid", "interpolation_mode", "padding_mode",
                  "align_corners"],
        "node": "GridSampler3dBackwardBackward"},
    # Multi-output backward kernels with no derivative of their own
    # (MiscKernelBackward.h): a pass that needs one raises.
    "deform_conv2d_backward": {"saved": [], "node": "DeformConv2dBackwardBackward"},
    "_flash_attention_backward": {"saved": [], "node": "FlashAttentionBackwardBackward"},
    "_efficient_attention_backward": {"saved": [], "node": "EfficientAttentionBackwardBackward"},
    # The attention backwards differentiate again at dropout_p == 0
    # (SdpaBackward.h), recomposed from the operands with recordable
    # primitives.
    "_scaled_dot_product_flash_attention_for_cpu_backward": {
        "saved": ["grad_out", "query", "key", "value", "dropout_p", "is_causal",
                  "attn_mask", "scale"],
        "node": "ScaledDotProductFlashAttentionForCpuBackwardBackward"},
    "scaled_dot_product_attention_backward": {
        "saved": ["grad_output", "query", "key", "value", "attn_mask",
                  "dropout_p", "is_causal", "scale", "enable_gqa"],
        "node": "ScaledDotProductAttentionBackwardBackward"},
    "_scaled_dot_product_attention_backward_with_lse": {
        "saved": ["grad_output", "query", "key", "value", "output", "logsumexp",
                  "is_causal", "impl"],
        "node": "ScaledDotProductAttentionBackwardWithLseBackward"},
    "linalg_lu": {"saved": ["pivot"], "saved_outputs": ["P", "L", "U"],
                  "output_differentiability": [False, True, True]},
    "lu_unpack": {"saved": ["LU_data"],
                  "output_differentiability": [False, True, True]},
    "_linalg_slogdet": {"saved": ["A"], "saved_outputs": ["sign"],
                        "output_differentiability": [True, True, False, False],
                        "node": "LinalgSlogdetBackward"},
    "linalg_lstsq": {"saved": ["A", "B"], "saved_outputs": ["solution"],
                     "output_differentiability": [True, True, False, False]},
    "triangular_solve": {"saved": ["self", "A", "upper", "transpose",
                                   "unitriangular"],
                         "saved_outputs": ["solution"]},
    # The node keeps the input's geometry, not the input.
    "as_strided": {"saved": ["self", "size", "stride", "storage_offset"],
                   "node": "AsStridedBackward"},
}

# Ops whose backward node is provided hand-written elsewhere; skip emitting a
# generated class even though derivatives exist.
EXTERNAL_NODES: set[str] = set()


# Tensor attributes a node can hold in place of the tensor, with the member
# type each is stored as.
ATTRIBUTE_METHODS: dict[str, str] = {
    "shape": "std::vector<int64_t>",
    "sizes": "std::vector<int64_t>",
    "dtype": "DType",
    "scalar_type": "DType",
    "numel": "int64_t",
    "dim": "int64_t",
    "device": "Device",
}


def tensor_uses(expr: Expr, name: str) -> tuple[set[str], bool]:
    """The attributes of `name` a formula reads, and whether it reads the
    tensor's values anywhere else."""
    found: set[str] = set()
    value_use = False

    def walk(e: Expr) -> None:
        nonlocal value_use
        if (isinstance(e, Method) and isinstance(e.receiver, Var)
                and e.receiver.name == name and not e.args
                and e.name in ATTRIBUTE_METHODS):
            found.add(e.name)
            return
        if isinstance(e, Var):
            if e.name == name:
                value_use = True
            return
        if isinstance(e, (Neg, Paren, Not)):
            walk(e.value)
        elif isinstance(e, Ternary):
            walk(e.cond)
            walk(e.then)
            walk(e.other)
        elif isinstance(e, Call):
            for a in e.args:
                walk(a)
        elif isinstance(e, Braced):
            for a in e.items:
                walk(a)
        elif isinstance(e, Method):
            walk(e.receiver)
            for a in e.args:
                walk(a)
        elif isinstance(e, BinOp):
            walk(e.left)
            walk(e.right)

    walk(expr)
    return found, value_use


def saved_output_member_type(cpp_type: str) -> str:
    """Node member type of a saved forward output.

    A symbolic size the forward hands back is concrete by the time the node
    runs, and the backward ops take it as a plain integer.
    """
    return "int64_t" if cpp_type == "SymInt" else cpp_type


def compute_op_derivatives(func: NativeFunction, raw_formulas: dict[str, str],
                           node_name: str | None = None,
                           fw_raw: dict[str, str] | None = None,
                           differentiable_outputs: list[bool] | None = None
                           ) -> OpDerivatives:
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

    fw_formulas: dict[str, Expr] = {}
    fw_required_tangent: list[str] = []
    fw_required_primal: list[str] = []
    for out_name, formula in (fw_raw or {}).items():
        expr, tangent, primal = _expand_fw_formula(
            func, out_name, formula, raw_formulas)
        fw_formulas[out_name] = expr
        fw_required_tangent.extend(n for n in tangent
                                   if n not in fw_required_tangent)
        fw_required_primal.extend(n for n in primal
                                  if n not in fw_required_primal)

    used: set[str] = set()
    for e in parsed.values():
        collect_vars(e, used)
    used &= arg_names | output_names

    # What a formula measures of a forward tensor (its shape, type, element
    # count, ...) is held as that attribute; the tensor itself is kept only
    # for the slots that read its values, so a tensor that is only measured
    # may be freed or updated in place before the backward pass.
    members: list[tuple[str, str]] = []
    attribute_members: dict[str, tuple[str, str]] = {}
    value_readers: dict[str, list[str]] = {}
    for a in func.args:
        if a.name not in used:
            continue
        t = a.type
        # The in-place self qualifies too: an in-place op keeps its shape and
        # type, and its node shares the out-of-place spelling's layout.
        if not (t.is_tensor_like and not t.is_list and not t.is_opt):
            members.append((a.name, node_member_type(a.type)))
            continue
        attributes: set[str] = set()
        readers: list[str] = []
        for slot, e in parsed.items():
            found, reads_value = tensor_uses(e, a.name)
            attributes |= found
            if reads_value:
                readers.append(slot)
        for attr in sorted(attributes):
            member = f"{a.name}_{attr}"
            members.append((member, ATTRIBUTE_METHODS[attr]))
            attribute_members[member] = (a.name, attr)
        if readers:
            members.append((a.name, node_member_type(a.type)))
            value_readers[a.name] = readers
    if func.cpp_return_kind == "tuple":
        from .api_types import tuple_element_cpp_types, tuple_element_names
        cpp_types = tuple_element_cpp_types(func)
        for i, nm in enumerate(tuple_element_names(func)):
            # A name an argument also carries refers to the argument.
            if nm in used and nm not in arg_names:
                members.append((nm, saved_output_member_type(cpp_types[i])))
    elif "result" in used:
        rt = func.returns[0].type
        members.append(("result", "std::vector<Tensor>" if rt.is_list else "Tensor"))

    grad_slots = [
        a for a in func.args
        if (a.type.is_tensor_like and not a.type.is_list) or a.type.is_mutable_ref
    ]
    # A saved tensor that only some slots read is kept only when one of
    # those inputs wants a gradient, so a product with a constant factor does
    # not hold on to the factor that receives no gradient.
    conditional_members: dict[str, list[str]] = {}
    formula_slots = [a.name for a in grad_slots if a.name in parsed]
    if len(formula_slots) > 1:
        for m, t in members:
            if t != "Tensor":
                continue
            if m in value_readers:
                if any(n not in formula_slots for n in value_readers[m]):
                    continue  # read by a formula that is not a gradient slot
                readers = [n for n in formula_slots if n in value_readers[m]]
            else:
                # A saved output: every slot whose formula names it.
                readers = []
                for n in formula_slots:
                    names: set[str] = set()
                    collect_vars(parsed[n], names)
                    if m in names:
                        readers.append(n)
            if readers and len(readers) < len(formula_slots):
                conditional_members[m] = readers
    return OpDerivatives(
        func=func, node_name=node_name, formulas=parsed,
        grad_slots=grad_slots, members=members,
        used_input_names={m for m, _ in members} & arg_names,
        used_output_names={m for m, _ in members} & output_names,
        attribute_members=attribute_members,
        conditional_members=conditional_members,
        differentiable_outputs=differentiable_outputs,
        fw_formulas=fw_formulas,
        fw_required_tangent=fw_required_tangent,
        fw_required_primal=fw_required_primal,
    )


def load_derivatives(path: str, native_by_opname: dict[str, NativeFunction]) \
        -> dict[str, OpDerivatives]:
    """Parse derivatives.yaml keyed by dispatcher op name."""
    from .model import parse_schema, parse_derivatives_yaml

    out: dict[str, OpDerivatives] = {}
    # The backward formulas of every entry, for the in-place spellings that
    # differentiate like their functional twin.
    raw_by_op: dict[str, tuple[dict[str, str], list[bool] | None]] = {}
    seen_names: set[str] = set()
    for item in parse_derivatives_yaml(path):
        f = parse_schema(item["name"])
        op = f.func_name
        # One entry per operator: a second one would silently replace the
        # first, and the two need not agree.  Keyed by the operator rather
        # than the spelled schema, which can differ in defaults alone.
        if op in seen_names:
            raise ValueError(f"derivatives.yaml defines '{op}' more than once")
        seen_names.add(op)
        native = native_by_opname.get(op)
        if native is None:
            continue
        if op in EXTERNAL_NODES:
            continue
        # Keys naming a forward output define the forward-mode (jvp)
        # derivative; every other key is a backward gradient slot.  Named
        # tuple-element keys are backward slots when the op's backward uses
        # them and forward slots when the schema output carries the formula,
        # so outputs with a declared name always route forward.
        output_keys = {"result"}
        output_keys.update(d.name for d in native.returns if d.name)
        output_differentiability = item.get("output_differentiability")
        if output_differentiability is not None:
            if not isinstance(output_differentiability, list) or any(
                    type(value) is not bool for value in output_differentiability):
                raise ValueError(
                    f"output_differentiability for '{op}' must be a list of booleans")
            if len(output_differentiability) != len(native.returns):
                raise ValueError(
                    f"output_differentiability for '{op}' has "
                    f"{len(output_differentiability)} entries for "
                    f"{len(native.returns)} outputs")
        arg_names = {a.name for a in native.args}
        raw: dict[str, str] = {}
        fw_raw: dict[str, str] = {}
        non_differentiable: set[str] = set()

        def split_names(raw_names: str) -> tuple[str, ...]:
            """Given "foo, bar", return ("foo", "bar")."""
            return tuple(x.strip() for x in raw_names.split(","))

        for key, value in item.items():
            if key in ("name", "dispatch", "output_differentiability") \
                    or not isinstance(value, str):
                continue
            names = split_names(key)
            formula = _normalize_comparisons(value)
            for name in names:
                if name in arg_names and name in output_keys:
                    raise ValueError(
                        f"Derivative key '{name}' for '{op}' names both a "
                        f"schema argument and an output")
            # A key naming schema arguments assigns the backward gradient
            # slots; any other key defines the forward-mode (jvp) derivative
            # and must name declared outputs.  An unmatchable key is a stale
            # schema spelling and fails the generation instead of silently
            # dropping that gradient.
            if names[0] in arg_names:
                for name in names:
                    if name not in arg_names:
                        raise ValueError(
                            f"Derivative key '{key}' for '{op}' mixes the "
                            f"argument '{names[0]}' with the unknown name "
                            f"'{name}'")
                    if formula.strip() == "non_differentiable":
                        non_differentiable.add(name)
                    else:
                        raw[name] = formula
            else:
                for name in names:
                    if name not in output_keys:
                        raise ValueError(
                            f"Unknown derivative key '{key}' for '{op}': "
                            f"'{name}' is neither a schema argument "
                            f"{sorted(arg_names)} nor a declared output")
                    fw_raw[name] = formula
        if raw:
            raw_by_op[op] = (dict(raw), output_differentiability)
        if raw or fw_raw:
            out[op] = compute_op_derivatives(
                native, raw, fw_raw=fw_raw or None,
                differentiable_outputs=output_differentiability)
            out[op].non_differentiable_args = frozenset(non_differentiable)
        elif output_differentiability is not None and not any(
                output_differentiability):
            # Non-differentiable output: register the autograd wrapper so the
            # dispatch chain resolves above the backend key, but emit no
            # backward node and keep the outputs detached.
            out[op] = OpDerivatives(
                func=native,
                node_name=autograd_node_name(native.func_name),
                formulas={},
                grad_slots=native.tensor_args,
                members=[],
                used_input_names=set(),
                used_output_names=set(),
                non_differentiable_output=True,
                differentiable_outputs=output_differentiability,
            )

    # An in-place spelling without an entry of its own differentiates like
    # the functional operator with its signature: the wrapper captures the
    # input before the update for formulas that read `self`, and `result` is
    # the updated tensor.  Without this the in-place call recorded nothing
    # and backward treated it as the identity.
    def signature(fn: NativeFunction) -> list[tuple[str, str]]:
        return [(a.name, str(a.type).replace("(a!)", "")) for a in fn.args]

    functional_by_base: dict[str, list[NativeFunction]] = {}
    for fn in native_by_opname.values():
        functional_by_base.setdefault(fn.base_name, []).append(fn)
    for op, native in native_by_opname.items():
        base = native.base_name
        if (op in out or not base.endswith("_") or base.startswith("_foreach")
                or base.endswith("__") or not native.args
                or not native.args[0].type.is_mutable_ref
                or native.cpp_return_kind != "mut_ref"):
            continue
        twins = [fn for fn in functional_by_base.get(base[:-1], ())
                 if fn.func_name in raw_by_op and signature(fn) == signature(native)]
        if not twins:
            continue
        twin_raw, differentiability = raw_by_op[twins[0].func_name]
        # The twin's node serves both spellings.
        out[op] = compute_op_derivatives(
            native, twin_raw, node_name=out[twins[0].func_name].node_name,
            differentiable_outputs=differentiability)

    # Manual (hand-written) backwards: register their saved-state layout so
    # wrapper generation treats them uniformly.  Forward-mode formulas parsed
    # from the yaml entry survive the overwrite: the hand-written node only
    # replaces the backward side.
    for base, spec in MANUAL_DERIVATIVES.items():
        cand = [n for op, n in native_by_opname.items() if op.split(".")[0] == base]
        if not cand:
            continue
        native = sorted(cand, key=lambda n: len(n.overload_name))[0]
        saved = [a for a in native.args if a.name in spec["saved"]]
        members = [(a.name, node_member_type(a.type)) for a in saved]
        # Outputs the node reads are handed to it after the forward ran, in
        # the order listed.
        saved_outputs = list(spec.get("saved_outputs", ()))
        if saved_outputs:
            from .api_types import tuple_element_cpp_types, tuple_element_names
            names = tuple_element_names(native)
            types = tuple_element_cpp_types(native)
            for name in saved_outputs:
                if name not in names:
                    raise ValueError(
                        f"manual derivative for '{base}' saves the unknown "
                        f"output '{name}'")
                members.append(
                    (name, saved_output_member_type(types[names.index(name)])))
        prev = out.get(native.func_name)
        out[native.func_name] = OpDerivatives(
            func=native, node_name=spec.get("node", autograd_node_name(base)),
            formulas={}, grad_slots=native.tensor_args,
            members=members,
            used_input_names=set(spec["saved"]),
            used_output_names=set(saved_outputs),
            fw_formulas=prev.fw_formulas if prev else {},
            differentiable_outputs=(
                spec.get("output_differentiability")
                or (prev.differentiable_outputs if prev else None)),
            fw_required_tangent=prev.fw_required_tangent if prev else [],
            fw_required_primal=prev.fw_required_primal if prev else [],
        )
    return out


# ---------------------------------------------------------------------------
# AutogradNodesGenerated.h
# ---------------------------------------------------------------------------

_HEADER_PRELUDE = """// Generated by tools/codegen/main.py -- DO NOT EDIT
#pragma once
#include "Node.h"
#include "Autograd.h"
#include "ManualNodes.h"
#include "SavedVariable.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include <algorithm>
#include <optional>
#include <utility>
#include "Scalar.h"
#include <vector>
#include <cstdint>
#include <cstdio>

namespace tensorplay {
namespace tpx {
using namespace ops;
"""


def generate_autograd_nodes(
        derivatives: dict[str, OpDerivatives],
        native_op_names: set[str] = frozenset()) -> str:
    lines = [_HEADER_PRELUDE.rstrip("\n"), ""]

    emitted: set[str] = set()
    for op, dv in derivatives.items():
        if dv.node_name in emitted or not dv.formulas:
            continue
        emitted.add(dv.node_name)
        f = dv.func
        member_names = {m for m, _ in dv.members}
        # Saved forward tensors (Tensor-typed forward args) go into
        # SavedVariable with version checking; scalars/dims stay plain.
        tensor_members = {m for m, t in dv.members if t == "Tensor"}
        member_names = {m for m, _ in dv.members}
        tensor_syms = {a.name for a in f.args if a.type.is_tensor_like}
        # Saved forward outputs referenced by formulas (`result`, named tuple
        # elements) are tensor symbols too.
        if "result" in member_names:
            tensor_syms.add("result")
        if f.cpp_return_kind == "tuple":
            from .api_types import tuple_element_names
            tensor_syms.update(n for n in tuple_element_names(f)
                               if n in tensor_members)
        # A saved output can be undefined from the start (an output a call
        # does not produce); only one that was defined and is now gone has
        # been released.
        arg_names = {a.name for a in f.args}
        saved_outputs = [m for m, _t in dv.members
                         if m in tensor_members and m not in arg_names]
        # The output index each saved output had in the forward's result.
        output_index = {"result": 0}
        if f.cpp_return_kind == "tuple":
            from .api_types import tuple_element_names
            output_index.update(
                {n: i for i, n in enumerate(tuple_element_names(f))})
        # An input kept only for some slots is likewise undefined from the
        # start when none of those slots wanted a gradient.
        output_tensor_members = saved_outputs + [
            m for m, _t in dv.members
            if m in tensor_members and m in arg_names
            and m in dv.conditional_members]

        lines.append(f"struct {dv.node_name} : public Node {{")
        for m, t in dv.members:
            if m in tensor_members:
                lines.append(f"    SavedVariable {m}_;")
            else:
                lines.append(f"    {t} {m}_;")
        for m in output_tensor_members:
            lines.append(f"    bool {m}_saved_undefined_;")
        lines.append("")
        ctor_args = [f"{t} {m}" for m, t in dv.members]
        # A saved output is held without its autograd metadata (no cycle
        # through this node).
        ctor_inits = [f"{m}_({m}, true)" if m in saved_outputs else f"{m}_({m})"
                      for m, _ in dv.members]
        ctor_inits += [f"{m}_saved_undefined_(!{m}.defined())"
                       for m in output_tensor_members]
        lines.append(f"    explicit {dv.node_name}({', '.join(ctor_args)})")
        if ctor_inits:
            lines.append(f"        : {', '.join(ctor_inits)} {{}}")
        else:
            lines.append("        {}")
        lines.append("")
        # Single-grad-input contract: the forward op is single-output, so
        # upstream grads always arrive at slot 0. Sizing the engine's input
        # buffer by next_edges would pad phantom slots that apply() never
        # reads, and the engine's grad materialization would allocate and
        # fill zeros for each of them on every backward pass.
        lines.append("    size_t num_inputs() const override { return 1; }")
        lines.append("")
        lines.append("    variable_list apply(variable_list&& inputs) override {")

        n_slots = len(dv.grad_slots)
        undef = ", ".join(["Tensor()"] * n_slots)
        lines.append(f"        if (inputs.empty() || !inputs[0].defined()) return {{{undef}}};")
        lines.append("        const Tensor& grad = inputs[0];")

        if tensor_members:
            lines.append("")
            for m, _t in dv.members:
                if m in saved_outputs:
                    lines.append(
                        f"        const Tensor {m}_sv = {m}_.unpack_output("
                        f"shared_from_this(), {output_index.get(m, 0)});")
                elif m in tensor_members:
                    lines.append(f"        const Tensor {m}_sv = {m}_.unpack();")
            # A backward pass without retain_graph releases saved state once
            # the walk finishes. Re-entering such a node must not touch the
            # released storage: the subgraph is already consumed, so every
            # grad slot goes out undefined and propagation stops here.
            required = [m for m, _t in dv.members if m in tensor_members]
            cond = " || ".join(
                f"(!{m}_sv.defined() && !{m}_saved_undefined_)"
                if m in output_tensor_members else f"!{m}_sv.defined()"
                for m in required)
            lines.append(f"        if ({cond}) return {{{undef}}};")
        lines.append("")
        lines.append("        variable_list grads;")

        uses_grad_input_mask = False
        for expr in dv.formulas.values():
            formula_vars: set[str] = set()
            collect_vars(expr, formula_vars)
            uses_grad_input_mask = uses_grad_input_mask or (
                "grad_input_mask" in formula_vars)
        if uses_grad_input_mask:
            lines.append("        std::vector<bool> grad_input_mask;")
            lines.append("        grad_input_mask.reserve(next_edges().size());")
            for slot_idx in range(n_slots):
                lines.append(
                    f"        grad_input_mask.push_back(should_compute_output({slot_idx}));")

        # Common-subexpression elimination: identical Call sub-expressions
        # shared across gradient slots are evaluated once (generalizes
        # hand-written `shared` blocks, e.g. batch_norm's
        # three-way backward kernel call). Shared temporaries record which
        # slots reference them, so a temp used only by guarded slots is
        # itself guarded (any referenced slot valid -> compute once).
        cse_slots: dict[str, list[int]] = {}
        cse_temps: dict[str, str] = {}
        call_counts: dict[str, int] = {}
        em = Emitter(tensor_syms, member_names, tensor_members,
                     native_op_names, dv.attribute_members)
        for slot_idx, a in enumerate(dv.grad_slots):
            expr = dv.formulas.get(a.name)
            if expr is None:
                continue
            for node in _iter_call_nodes(expr):
                txt = em.emit(node)
                call_counts[txt] = call_counts.get(txt, 0) + 1

        for txt, count in call_counts.items():
            if count < 2:
                continue
            temp = f"__shared_{len(cse_temps)}"
            cse_temps[txt] = temp

        for slot_idx, a in enumerate(dv.grad_slots):
            expr = dv.formulas.get(a.name)
            if expr is None:
                continue
            txt = render_formula(expr, tensor_syms, member_names,
                                 tensor_members, native_op_names,
                                 dv.attribute_members)
            for t in sorted(cse_temps, key=len, reverse=True):
                if t in txt:
                    cse_slots.setdefault(cse_temps[t], []).append(slot_idx)

        # Shared temporaries are guarded by the validity of any slot that
        # references them: an invalid edge means the slot's gradient is
        # discarded downstream, so a temp consumed only by invalid slots
        # must not launch its kernels either. The lazy-call form keeps the
        # kernel's own return type (tuple returns stay tuples); slot
        # formulas splice in a dereference of the optional.
        use_token: dict[str, str] = {}
        for txt, temp in cse_temps.items():
            users = cse_slots.get(temp, [])
            if n_slots > 1 and users:
                cond = " || ".join(
                    f"should_compute_output({i})" for i in users)
                lines.append(f"        auto {temp}_compute = [&] {{ return {txt}; }};")
                lines.append(
                    f"        std::optional<decltype({temp}_compute())> {temp}_val;")
                lines.append(f"        if ({cond}) {{")
                lines.append(f"            {temp}_val = {temp}_compute();")
                lines.append("        }")
                use_token[txt] = f"(*{temp}_val)"
            else:
                lines.append(f"        auto {temp} = {txt};")
                use_token[txt] = temp

        for slot_idx, a in enumerate(dv.grad_slots):
            expr = dv.formulas.get(a.name)
            if expr is None:
                lines.append("        grads.push_back(Tensor());")
                continue
            txt = render_formula(expr, tensor_syms, member_names,
                                 tensor_members, native_op_names,
                                 dv.attribute_members)
            # Splice shared temporaries into the rendered formula (longest
            # first so nested shared calls splice cleanly). Guarded temps
            # splice in as an optional dereference.
            for t in sorted(use_token, key=len, reverse=True):
                if t in txt:
                    txt = txt.replace(t, use_token[t])
            if txt in use_token:
                txt = use_token[txt]
            var = f"__grad_{a.name}"
            # A slot whose next edge is invalid has no consumer: the wrapper
            # (gen_tpx::_emit_edges) drops edges for forward inputs that do
            # not require grad, so any kernel launches here would be
            # discarded downstream. Slot i maps to next edge i because
            # grad_slots lists tensor args in declaration order and edges
            # are registered in that same order (no list-arg op generates a
            # node; those are all hand-written with their own alignment).
            # Single-slot nodes are skipped: a node exists only when at
            # least one forward input required grad, so its only edge is
            # always valid.
            if n_slots > 1:
                lines.append(f"        Tensor {var};")
                lines.append(f"        if (should_compute_output({slot_idx})) {{")
                lines.append(f"            {var} = {txt};")
                lines.append("        }")
            else:
                lines.append(f"        auto {var} = {txt};")
            lines.append(f"        grads.push_back({var});")
        lines.append("        return grads;")
        lines.append("    }")
        if tensor_members:
            lines.append("")
            lines.append("    void release_variables() override {")
            lines.append("        Node::release_variables();")
            for m, _t in dv.members:
                if m in tensor_members:
                    lines.append(f"        {m}_.reset_data();")
            lines.append("    }")
        lines.append("};")
        lines.append("")

    lines.append("} // namespace tpx")
    lines.append("} // namespace tensorplay")
    return "\n".join(lines) + "\n"
