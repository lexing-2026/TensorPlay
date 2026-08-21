# Python layer for einsum, mirroring torch/functional.py's structure: this
# shim only parses torch's sublist calling format and computes an optional
# opt-einsum contraction path; the actual algorithm lives in the native
# kernel (p10/src/Einsum.cpp), a faithful port of ATen's einsum.

import itertools


def _parse_subscript(n):
    if n is Ellipsis:
        return "..."
    if isinstance(n, int) and 0 <= n < 52:
        return chr(ord("A") + n) if n < 26 else chr(ord("a") + n - 26)
    raise ValueError(
        "einsum(): subscript in subscript list is not within the valid range [0, 52)"
    )


def _try_opt_einsum_path(equation, operands):
    """Return a flattened contraction path, or None when opt-einsum is not
    installed (the native kernel then contracts left to right)."""
    try:
        import opt_einsum
    except ImportError:
        return None
    tupled_path = opt_einsum.contract_path(equation, *operands, optimize="auto")[0]
    return [*itertools.chain.from_iterable(tupled_path)]


def einsum(*args):
    """einsum(equation, *operands) -> Tensor

    Sums the product of the elements of the input ``operands`` along dimensions
    specified using a notation based on the Einstein summation convention,
    e.g. ``tp.einsum("ij,jk->ik", A, B)`` computes a matrix multiplication.

    Also supports torch's sublist format:
    ``tp.einsum(A, [..., 0, 1], B, [..., 1, 2], [..., 0, 2])``.
    """
    import tensorplay

    if len(args) < 2:
        raise ValueError(
            "einsum(): must specify the equation string and at least one operand, "
            "or at least one operand and its subscripts list"
        )

    if isinstance(args[0], tensorplay.Tensor):
        # Convert the subscript list format (an interleaving of operand and
        # its subscripts list with an optional output subscripts list at the
        # end) to the equation string format.
        equation = ",".join(
            "".join(_parse_subscript(s) for s in lst) for lst in args[1::2]
        )
        if len(args) % 2 == 1:
            equation += "->" + "".join(_parse_subscript(s) for s in args[-1])
            operands = args[:-1:2]
        else:
            operands = args[::2]
    else:
        equation = args[0]
        operands = args[1:]

    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        # The old interface of passing the operands as one list argument.
        operands = tuple(operands[0])

    # Contracting 0 or 1 times is already optimal; otherwise let opt-einsum
    # pick the order when available.
    path = []
    if len(operands) > 2:
        computed = _try_opt_einsum_path(equation, operands)
        if computed is not None:
            path = computed
    return tensorplay._C.einsum(equation=equation, operands=list(operands), path=path)
