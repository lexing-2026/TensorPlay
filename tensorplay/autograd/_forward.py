"""Native forward-mode AD (JVP) through fused ``forward_*`` kernels.

A :class:`DualTensor` carries a (primal, tangent) pair.  Operators and the
supported method set propagate BOTH components forward in a single kernel
pass per op — no backward graph, O(1) memory.  This mirrors torch.func.jvp
(native dual numbers) as opposed to the double-backward trick used by
``autograd.functional.jvp(mode="reversed")``.

Supported ops: +, -, *, /, **, neg, exp, log, sin, cos, sqrt, tanh,
sigmoid, relu, mm/matmul (2-D), sum.  Anything else raises a clear
NotImplementedError so callers can fall back to reverse mode.
"""

import tensorplay

__all__ = ["DualTensor", "forward_jvp"]

_SUPPORTED = (
    "+ - * / ** neg exp log sin cos sqrt tanh sigmoid relu mm matmul sum"
).split()


def _promote_primal(t):
    if t.dtype in (tensorplay.float32, tensorplay.float64):
        return t
    return t.to(tensorplay.float32)


class DualTensor:
    __slots__ = ("primal", "tangent")

    def __init__(self, primal, tangent):
        if not isinstance(primal, tensorplay.Tensor):
            raise TypeError("DualTensor primal must be a tensor")
        if not isinstance(tangent, tensorplay.Tensor):
            raise TypeError("DualTensor tangent must be a tensor")
        if tangent.shape != primal.shape:
            raise ValueError(
                f"DualTensor tangent shape {tuple(tangent.shape)} does not "
                f"match primal shape {tuple(primal.shape)}")
        self.primal = _promote_primal(primal)
        self.tangent = _promote_primal(tangent)

    # ---- introspection -------------------------------------------------
    @property
    def shape(self):
        return self.primal.shape

    def __repr__(self):
        return (f"DualTensor(primal={self.primal}, "
                f"tangent={self.tangent})")

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _coerce(other, like):
        """Wraps scalars / tensors as constant duals for mixed arithmetic.

        Scalars are broadcast to the dual's primal shape because the native
        forward kernels require operand/tangent shapes to match exactly.
        """
        if isinstance(other, DualTensor):
            return other
        if isinstance(other, tensorplay.Tensor):
            if other.shape != like.primal.shape:
                other = other + tensorplay.zeros_like(like.primal)
                if other.shape != like.primal.shape:
                    raise ValueError(
                        "forward AD: cannot broadcast tensor operand to the "
                        "dual's primal shape")
            return DualTensor(other, tensorplay.zeros_like(other))
        # Python scalar: materialize a constant dual over `like`'s layout.
        const = tensorplay.full_like(like.primal, float(other))
        return DualTensor(const, tensorplay.zeros_like(const))

    def _binary(self, other, op):
        other = self._coerce(other, self)
        r, dr = op(self.primal, self.tangent, other.primal, other.tangent)
        return DualTensor._from_parts(r, dr)

    def _unary(self, op):
        r, dr = op(self.primal, self.tangent)
        return DualTensor._from_parts(r, dr)

    @classmethod
    def _from_parts(cls, primal, tangent):
        obj = cls.__new__(cls)
        obj.primal = primal
        obj.tangent = tangent
        return obj

    # ---- operators -------------------------------------------------------
    def __add__(self, other):
        return self._binary(other, tensorplay._C.forward_add)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, tensorplay._C.forward_sub)

    def __rsub__(self, other):
        other = self._coerce(other, self)
        return other.__sub__(self)

    def __mul__(self, other):
        return self._binary(other, tensorplay._C.forward_mul)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary(other, tensorplay._C.forward_div)

    def __rtruediv__(self, other):
        other = self._coerce(other, self)
        return other.__truediv__(self)

    def __pow__(self, other):
        return self._binary(other, tensorplay._C.forward_pow)

    def __neg__(self):
        return self._unary(tensorplay._C.forward_neg)

    def __matmul__(self, other):
        if isinstance(other, DualTensor):
            # Product rule over the raw matmuls: supports the full
            # vector/matrix broadcast semantics of `@` (unlike the fused
            # 2-D forward_mm kernel).
            primal = self.primal @ other.primal
            tangent = (self.tangent @ other.primal) + (self.primal @ other.tangent)
            return DualTensor._from_parts(primal, tangent)
        if isinstance(other, tensorplay.Tensor):
            return DualTensor._from_parts(self.primal @ other,
                                          self.tangent @ other)
        return self._binary(other, tensorplay._C.forward_mm)

    def __rmatmul__(self, other):
        if isinstance(other, tensorplay.Tensor):
            return DualTensor._from_parts(other @ self.primal,
                                          other @ self.tangent)
        raise NotImplementedError(
            "forward AD: DualTensor.__rmatmul__ with a dual right operand is "
            "not supported; reorder the product")

    # ---- named ops -------------------------------------------------------
    def exp(self):
        return self._unary(tensorplay._C.forward_exp)

    def log(self):
        return self._unary(tensorplay._C.forward_log)

    def sin(self):
        return self._unary(tensorplay._C.forward_sin)

    def cos(self):
        return self._unary(tensorplay._C.forward_cos)

    def sqrt(self):
        return self._unary(tensorplay._C.forward_sqrt)

    def tanh(self):
        return self._unary(tensorplay._C.forward_tanh)

    def sigmoid(self):
        return self._unary(tensorplay._C.forward_sigmoid)

    def relu(self):
        return self._unary(tensorplay._C.forward_relu)

    def mm(self, other):
        return self._binary(other, tensorplay._C.forward_mm)

    def matmul(self, other):
        other = self._coerce(other, self)
        if (self.primal.dim() == 2 and other.primal.dim() == 2):
            return self._binary(other, tensorplay._C.forward_mm)
        raise NotImplementedError(
            "forward-mode matmul supports 2-D operands only; reshape or use "
            "mode='reversed'")

    def sum(self, dim=None):
        # Reductions are linear, so the tangent reduces identically; this is
        # an exact composite over the native reduction kernels.
        if dim is None:
            return DualTensor._from_parts(self.primal.sum(),
                                          self.tangent.sum())
        return DualTensor._from_parts(
            self.primal.sum(dim), self.tangent.sum(dim))

    def detach(self):
        return DualTensor._from_parts(self.primal.detach(),
                                      self.tangent.detach())

    def __getitem__(self, idx):
        return DualTensor._from_parts(self.primal[idx], self.tangent[idx])


def _unwrap(output):
    """Splits a function output into (primals, tangents) like jvp."""
    if isinstance(output, DualTensor):
        return output.primal, output.tangent
    if isinstance(output, (tuple, list)):
        primals, tangents = zip(*(_unwrap(o) for o in output))
        return (type(output)(primals), type(output)(tangents)) \
            if isinstance(output, tuple) else (list(primals), list(tangents))
    raise NotImplementedError(
        f"forward-mode AD cannot trace {type(output).__name__} outputs; "
        "func must return tensors (or tuples of them) computed from its "
        f"inputs via the supported ops: {_SUPPORTED}")


def forward_jvp(func, inputs, v=None):
    """Computes Jacobian-vector products by propagating tangents forward.

    Args:
        func: callable taking one argument per input.  Its body must operate
            on the inputs through operators or the supported method set
            (free functions like ``tensorplay.exp(x)`` do NOT intercept
            DualTensors).
        inputs: tensor or tuple of tensors.
        v: direction(s); defaults to ones.  Same layout as ``inputs``.

    Returns:
        (output, jvp): the function value and its derivative along ``v``,
        matching the structure of ``inputs``/``v``.
    """
    single = isinstance(inputs, tensorplay.Tensor)
    inputs_t = (inputs,) if single else tuple(inputs)
    if v is None:
        v_t = tuple(tensorplay.ones_like(x) for x in inputs_t)
    else:
        v_t = (v,) if isinstance(v, tensorplay.Tensor) else tuple(v)
    if len(v_t) != len(inputs_t):
        raise ValueError("jvp: v must match inputs element-for-element")

    duals = tuple(DualTensor(x, dv) for x, dv in zip(inputs_t, v_t))
    output = func(*duals)
    return _unwrap(output)


def _as_plain_tuple(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)


def jacfwd(func, inputs):
    """Computes the full Jacobian by forward-mode column scanning.

    Mirrors ``torch.func.jacfwd`` loosely: returns one Jacobian block per
    (output, input) pair with shape ``out_shape + in_shape``, computed with
    num_cols(func cost) forward passes and no backward graph.
    """
    single = isinstance(inputs, tensorplay.Tensor)
    ins = (inputs,) if single else tuple(inputs)

    primals = func(*ins)
    primal_tuple = _as_plain_tuple(primals)
    for p in primal_tuple:
        if not isinstance(p, tensorplay.Tensor):
            raise NotImplementedError(
                "jacfwd: outputs must be tensors (or tuples of tensors)")

    jacobian_blocks = [[None] * len(ins) for _ in primal_tuple]
    for i, x in enumerate(ins):
        flat_x = x.reshape(-1)
        in_numel = flat_x.numel()
        for o, primal in enumerate(primal_tuple):
            out_shape = tuple(primal.shape)
            out_numel = primal.numel()
            columns = []
            for k in range(in_numel):
                tangent = tensorplay.zeros_like(flat_x)
                tangent[k] = 1.0
                _, js = forward_jvp(
                    func, ins,
                    tuple(tensorplay.zeros_like(t) for t in ins[:i])
                    + (tangent.reshape(x.shape),)
                    + tuple(tensorplay.zeros_like(t) for t in ins[i + 1:]))
                jt = _as_plain_tuple(js)[o]
                # Column k of the Jacobian: d(output_e)/d(input_k).
                columns.append(jt.reshape(-1))
            if columns:
                j_flat = tensorplay.stack(columns, dim=1)
                jacobian_blocks[o][i] = j_flat.reshape(out_shape +
                                                       tuple(x.shape))
            else:
                jacobian_blocks[o][i] = tensorplay.zeros(out_shape +
                                                         tuple(x.shape))

    # torch.autograd.functional.jacobian conventions:
    #   single in/out            -> Tensor
    #   one side a tuple         -> tuple of Tensors
    #   both sides tuples        -> tuple of tuples (Jacobian[i][j])
    multi_out = len(jacobian_blocks) > 1
    multi_in = len(ins) > 1
    if not multi_out and not multi_in:
        return jacobian_blocks[0][0]
    if not multi_in:
        return tuple(jacobian_blocks[o][0] for o in range(len(jacobian_blocks)))
    if not multi_out:
        return tuple(jacobian_blocks[0][i] for i in range(len(ins)))
    return tuple(tuple(_block(o, i) for i in range(len(ins)))
                 for o in range(len(jacobian_blocks)))
