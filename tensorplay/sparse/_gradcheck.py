"""Gradient checking through sparse inputs and outputs."""
from typing import Any

import tensorplay

__all__ = ["as_sparse_gradcheck"]

_STRIDED_REPRESENTATION = "__STRIDED_REPRESENTATION__"


def as_sparse_gradcheck(gradcheck):
    """Extends a gradcheck function with support for sparse tensors.

    Wraps :func:`tensorplay.autograd.gradcheck` (or a ``functools.partial``
    of it) so the checked function may take and return sparse tensors.  The
    wrapped gradcheck itself only ever sees strided tensors: each sparse
    input is decomposed into its index buffers plus a differentiable values
    tensor, and rebuilt inside the function under test.

    Pass ``masked=True`` to check the gradient only at the stored
    coordinates; by default unspecified entries are materialized as zeros so
    that the gradient with respect to them is checked as well.

    Example::

        gradcheck = tensorplay.sparse.as_sparse_gradcheck(
            tensorplay.autograd.gradcheck
        )
        x = tensorplay.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=tensorplay.float64)
        x = x.to_sparse().requires_grad_(True)
        gradcheck(lambda t: tensorplay.sparse.sum(t), x)
    """

    def gradcheck_with_sparse_support(func, inputs, **kwargs):
        """Same as the wrapped gradcheck, with sparse inputs/outputs support."""
        masked = kwargs.pop("masked", False)

        def convert_to_strided_representation(args):
            """Replaces each differentiable sparse tensor by its buffers."""
            if not isinstance(args, (list, tuple)):
                args = (args,)
            new_args: list[Any] = []
            for obj in args:
                if (
                    isinstance(obj, tensorplay.Tensor)
                    and obj.requires_grad
                    and obj.is_sparse
                ):
                    d = {"layout": obj.layout, "shape": list(obj.shape)}
                    if not masked:
                        # Materialize unspecified elements with zero values so
                        # the numerical Jacobian covers them too.
                        full_mask = tensorplay.ones(
                            list(obj.shape), dtype=tensorplay.bool, device=obj.device
                        ).to_sparse()
                        obj = obj.to_dense().sparse_mask(full_mask)
                    if obj.layout == tensorplay.sparse_csr:
                        d.update(
                            compressed_indices=obj.crow_indices(),
                            plain_indices=obj.col_indices(),
                        )
                        values = obj.values()
                    else:
                        d.update(
                            indices=obj._indices(),
                            is_coalesced=obj.is_coalesced(),
                        )
                        values = obj.values()
                    new_args.extend(
                        (_STRIDED_REPRESENTATION, d, values.requires_grad_(True))
                    )
                else:
                    new_args.append(obj)
            return tuple(new_args)

        def restore_from_strided_representation(args):
            """Rebuilds the sparse tensors from the buffers, keeping the graph."""
            new_args = []
            args = list(args)
            while args:
                a = args.pop(0)
                if isinstance(a, str) and a == _STRIDED_REPRESENTATION:
                    d, values = args.pop(0), args.pop(0)
                    if d["layout"] == tensorplay.sparse_csr:
                        raise NotImplementedError(
                            "rebuilding a CSR tensor from its strided representation "
                            "needs a compressed-layout constructor, which this build "
                            "does not expose yet; convert the input to COO first"
                        )
                    a = tensorplay.sparse.sparse_coo_tensor(
                        d["indices"],
                        values,
                        d["shape"],
                        is_coalesced=d["is_coalesced"],
                    )
                new_args.append(a)
            return tuple(new_args)

        def func_wrapper(*args, **kwargs):
            restored_args = restore_from_strided_representation(args)
            outputs = func(*restored_args, **kwargs)

            is_sequence = isinstance(outputs, (list, tuple))
            strided_outputs = tuple(outputs) if is_sequence else (outputs,)
            strided_outputs = tuple(
                (
                    o.to_dense()
                    if isinstance(o, tensorplay.Tensor)
                    and o.requires_grad
                    and o.is_sparse
                    else o
                )
                for o in strided_outputs
            )
            return strided_outputs if is_sequence else strided_outputs[0]

        return gradcheck(func_wrapper, convert_to_strided_representation(inputs), **kwargs)

    return gradcheck_with_sparse_support
