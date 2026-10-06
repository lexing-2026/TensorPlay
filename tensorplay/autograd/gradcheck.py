# mypy: allow-untyped-defs
r"""Gradient checking via finite differences.

:func:`gradcheck` compares the Jacobians that reverse-mode (and, on request,
forward-mode) AD produce with central differences, either element by element
(slow mode) or projected onto random vectors (fast mode).  Complex inputs and
outputs are checked through their real and imaginary parts.
"""

import collections
import functools
import warnings
from itertools import product

import tensorplay

from .function import Function


__all__ = [
    "gradcheck",
    "gradgradcheck",
    "GradcheckError",
    "get_numerical_jacobian_wrt_specific_input",
]


def _is_tensor_like(x):
    return isinstance(x, tensorplay.Tensor)


def _is_sparse_any_tensor(obj):
    return _is_tensor_like(obj) and obj.is_sparse


def _is_float_or_complex_tensor(obj):
    return _is_tensor_like(obj) and (obj.is_floating_point() or obj.is_complex())


class GradcheckError(RuntimeError):
    r"""Error raised by :func:`gradcheck` and :func:`gradgradcheck`."""


class _UndefinedGrad(Function):
    # Identity in the forward; the backward ignores whatever gradient it
    # receives and propagates an undefined (None) one instead.  Used to check
    # that functions handle undefined output gradients.
    @staticmethod
    def forward(ctx, inp):
        ctx.set_materialize_grads(False)
        return inp.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return None


def _allocate_jacobians_with_inputs(input_tensors, numel_output):
    # Makes zero-filled tensors from inputs. For each tensor in
    # `input_tensors`, returns a new zero-filled tensor with height of
    # `t.numel` and width of `numel_output`.
    out: list[tensorplay.Tensor] = []
    for t in input_tensors:
        if _is_float_or_complex_tensor(t) and t.requires_grad:
            out.append(tensorplay.zeros((t.numel(), numel_output), dtype=t.dtype, device=t.device))
    return tuple(out)


def _allocate_jacobians_with_outputs(output_tensors, numel_input, dtype=None, device=None):
    # Makes zero-filled tensors from outputs. For each tensor in
    # `output_tensors`, returns a new zero-filled tensor with height of
    # `numel_input` and width of `t.numel`.
    out: list[tensorplay.Tensor] = []
    for t in output_tensors:
        if _is_float_or_complex_tensor(t):
            out.append(tensorplay.zeros(
                (numel_input, t.numel()),
                dtype=dtype if dtype is not None else t.dtype,
                device=device if device is not None else t.device,
            ))
    return tuple(out)


def _iter_tensors(x, only_requiring_grad: bool = False):
    if _is_tensor_like(x):
        if x.requires_grad or not only_requiring_grad:
            yield x
    elif isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        for elem in x:
            yield from _iter_tensors(elem, only_requiring_grad)


def _densify(x):
    # Sparse inputs are checked through their dense equivalents.
    if isinstance(x, (list, tuple)):
        return type(x)(map(_densify, x))
    if _is_sparse_any_tensor(x):
        dense = x.to_dense()
        return dense.requires_grad_(x.requires_grad)
    return x


def _iter_tensor(x_tensor):
    # (Only used for slow gradcheck) Returns a generator that yields the following
    # elements at each iteration:
    #  1) a tensor: the same tensor is returned across all iterations. The tensor
    #     is not the same as the original x_tensor as given as input - it is
    #     prepared so that it can be modified in-place (it shares storage with
    #     x_tensor).
    #  2) a tuple of indices that can be used with advanced indexing (yielded in
    #     dictionary order)
    #  3) flattened index that will be used to index into the Jacobian tensor
    #
    # For a tensor t with size (2, 2), _iter_tensor yields:
    #     `x, (0, 0), 0`, `x, (0, 1), 1`, `x, (1, 0), 2`, `x, (1, 1), 3`
    #
    # where x is the t.data of the original tensor. Perturbing the entry of x
    # at index (1, 1) yields the 3rd column of the overall Jacobian matrix.
    # Use .data here to get around the version check
    x_tensor = x_tensor.data
    for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.size()])):
        yield x_tensor, x_idx, d_idx


def _get_numerical_jacobian(fn, inputs, outputs=None, target=None, eps=1e-3, is_forward_ad=False):
    """Compute the numerical Jacobian of `fn(inputs)` with respect to `target`.

    If not specified, targets are the input. Returns M * N Jacobians where N is the
    number of tensors in target that require grad and M is the number of non-integral
    outputs.

    Args:
        fn: the function to compute the jacobian for
        inputs: inputs to `fn`
        outputs: provide precomputed outputs to avoid one extra invocation of fn
        target: the Tensors wrt whom Jacobians are calculated (default=`inputs`)
        eps: the magnitude of the perturbation during finite differencing
             (default=`1e-3`)
        is_forward_ad: if this numerical jacobian is computed to be checked wrt
                       forward AD gradients (this is used for error checking only)

    Returns:
        A list of M N-tuples of tensors

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    jacobians: list[tuple[tensorplay.Tensor, ...]] = []
    if outputs is None:
        outputs = _as_tuple(fn(*_as_tuple(inputs)))
    if not is_forward_ad and any(o.is_complex() for o in outputs):
        raise ValueError(
            "Expected output to be non-complex. get_numerical_jacobian no "
            "longer supports functions that return complex outputs."
        )
    if target is None:
        target = inputs
    inp_indices = [i for i, a in enumerate(target) if _is_tensor_like(a) and a.requires_grad]
    for inp, inp_idx in zip(_iter_tensors(target, True), inp_indices):
        jacobians += [
            get_numerical_jacobian_wrt_specific_input(
                fn, inp_idx, inputs, outputs, eps, input=inp, is_forward_ad=is_forward_ad
            )
        ]
    return jacobians


def _compute_numerical_gradient(fn, entry, v, norm_v, nbhd_checks_fn):
    # Computes numerical directional derivative as finite difference
    # of function `fn` at input `entry`, perturbed by vector `v`.
    orig = entry.clone()
    entry.copy_(orig - v)
    outa = fn()
    entry.copy_(orig + v)
    outb = fn()
    entry.copy_(orig)

    def compute(a, b):
        nbhd_checks_fn(a, b)
        ret = (b - a) / (2 * norm_v)  # use central difference approx
        return ret.detach().reshape(-1)

    return tuple(compute(a, b) for (a, b) in zip(outa, outb))


def _compute_numerical_jvps_wrt_specific_input(jvp_fn, delta, input_is_complex, is_forward_ad=False):
    # Computing the jacobian only works for real delta
    # For details on the algorithm used here, refer:
    # Section 3.5.3 https://arxiv.org/pdf/1701.00392.pdf
    # s = fn(z) where z = x for real valued input
    # and z = x + yj for complex valued input
    jvps: list[tensorplay.Tensor] = []
    ds_dx_tup = jvp_fn(delta[0] if isinstance(delta, tuple) else delta)

    if input_is_complex:  # C -> R
        ds_dy_tup = jvp_fn(delta[1] * 1j) if isinstance(delta, tuple) else jvp_fn(delta * 1j)
        for ds_dx, ds_dy in zip(ds_dx_tup, ds_dy_tup):
            if ds_dx.is_complex():
                raise AssertionError("Expected ds_dx to be real-valued, not complex")
            # conjugate wirtinger derivative
            conj_w_d = ds_dx + ds_dy * 1j
            jvps.append(conj_w_d)
    else:
        for ds_dx in ds_dx_tup:  # R -> R or (R -> C for the forward AD case)
            if not is_forward_ad and ds_dx.is_complex():
                raise AssertionError("Expected ds_dx to be real-valued, not complex.")
            jvps.append(ds_dx)
    return jvps


def _combine_jacobian_cols(jacobians_cols: dict, outputs, input, numel):
    # jacobian_cols maps column_idx -> output_idx -> single column of jacobian Tensor
    # we return a list that maps output_idx -> full jacobian Tensor
    jacobians = _allocate_jacobians_with_outputs(
        outputs, numel, dtype=input.dtype if input.is_complex() else None
    )
    for i, jacobian in enumerate(jacobians):
        for k, v in jacobians_cols.items():
            jacobian.select(0, k).copy_(v[i].reshape(-1))
    return jacobians


def _check_outputs_same_dtype_and_shape(output1, output2, eps, idx=None) -> None:
    # Check that the returned outputs don't have different dtype or shape when you
    # perturb the input
    on_index = f"on index {idx} " if idx is not None else ""
    if tuple(output1.shape) != tuple(output2.shape):
        raise AssertionError(
            f"Expected `func` to return outputs with the same shape"
            f" when inputs are perturbed {on_index}by {eps}, but got:"
            f" shapes {tuple(output1.shape)} and {tuple(output2.shape)}."
        )
    if output1.dtype != output2.dtype:
        raise AssertionError(
            f"Expected `func` to return outputs with the same dtype"
            f" when inputs are perturbed {on_index}by {eps}, but got:"
            f" dtypes {output1.dtype} and {output2.dtype}."
        )


def get_numerical_jacobian_wrt_specific_input(
    fn, input_idx, inputs, outputs, eps, input=None, is_forward_ad=False
) -> tuple:
    # Computes the numerical jacobians wrt to a single input. Returns N jacobian
    # tensors, where N is the number of outputs. We use a dictionary for
    # jacobian_cols because indices aren't necessarily consecutive for sparse inputs
    # When we perturb only a single element of the input tensor at a time, the jvp
    # is equivalent to a single col of the Jacobian matrix of fn.
    jacobian_cols: dict[int, list[tensorplay.Tensor]] = {}
    input = inputs[input_idx] if input is None else input
    if not input.requires_grad:
        raise AssertionError("Expected input to have requires_grad=True")
    for x, idx, d_idx in _iter_tensor(input):
        wrapped_fn = _with_prepare_inputs(fn, inputs, input_idx, x)
        input_to_perturb = x[idx]
        nbhd_checks_fn = functools.partial(_check_outputs_same_dtype_and_shape, idx=idx, eps=eps)
        jvp_fn = _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn)
        jacobian_cols[d_idx] = _compute_numerical_jvps_wrt_specific_input(
            jvp_fn, eps, x.is_complex(), is_forward_ad
        )
    return _combine_jacobian_cols(jacobian_cols, outputs, input, input.numel())


def _get_analytical_jacobian_forward_ad(fn, inputs, outputs, *, check_grad_dtypes=False, all_u=None) -> tuple:
    """Compute the analytical Jacobian of `fn(inputs)` using forward mode AD.

    Return N * M Jacobians where N is the number of tensors in target that require grad and
    M is the number of non-integral outputs.
    Contrary to other functions here, this function requires "inputs" to actually be used by the function.
    The computed value is expected to be wrong if the function captures the inputs by side effect instead of
    using the passed ones.

    Args:
        fn: the function to compute the jacobian for
        inputs: inputs to `fn`
        outputs: provide precomputed outputs to avoid one extra invocation of fn
        check_grad_dtypes: if True, will check that the gradient dtype are valid
        all_u (optional): if provided, the Jacobian will be right multiplied with this vector

    Returns:
        A tuple of M N-tuples of tensors
    """
    fwAD = tensorplay.autograd.forward_ad

    tensor_inputs = tuple(i for i in inputs if _is_tensor_like(i) and i.requires_grad)

    if any(i.is_complex() for i in tensor_inputs):
        raise ValueError(
            "Expected inputs to be non-complex for _get_analytical_jacobian_forward_ad."
        )

    if all_u:
        jacobians = tuple(_allocate_jacobians_with_outputs(outputs, 1) for i in tensor_inputs)
    else:
        jacobians = tuple(_allocate_jacobians_with_outputs(outputs, i.numel()) for i in tensor_inputs)

    def read_tangents(i, row):
        raw_outputs = _as_tuple(fn(*dual_inputs))
        dual_outputs = filter(_is_float_or_complex_tensor, raw_outputs)
        for index_o, d_o in enumerate(dual_outputs):
            val, res = fwAD.unpack_dual(d_o)
            if check_grad_dtypes and res is not None and val.is_complex() != res.is_complex():
                raise GradcheckError("Forward AD gradient has dtype mismatch.")
            dest = jacobians[i][index_o].select(0, row)
            if res is None:
                dest.zero_()
            else:
                dest.copy_(res.reshape(-1))

    with fwAD.dual_level():
        fw_grads = []
        dual_inputs = []
        for inp in inputs:
            if _is_tensor_like(inp) and inp.requires_grad:
                inp = fwAD.make_dual(inp.detach(), tensorplay.zeros_like(inp))
                # If inp is a differentiable view, the dual might not be the tangent given to
                # make_dual, so read it explicitly from the dual tensor
                fw_grads.append(fwAD.unpack_dual(inp)[1])
            dual_inputs.append(inp)

        if all_u:
            # Do the full reduction in one pass
            # To be consistent with numerical evaluation, we actually compute one reduction per input
            for i, (fw_grad, u) in enumerate(zip(fw_grads, all_u)):
                fw_grad.copy_(u.view_as(fw_grad))
                read_tangents(i, 0)
                fw_grad.zero_()
            # Remove extra dimension of size 1 corresponding to the reduced input
            jacobians = tuple(tuple(j.squeeze(0) for j in per_input) for per_input in jacobians)
        else:
            # Reconstruct the full Jacobian column by column
            for i, fw_grad in enumerate(fw_grads):
                for lin_idx, grad_idx in enumerate(product(*[range(m) for m in fw_grad.size()])):
                    fw_grad[grad_idx] = 1.0
                    read_tangents(i, lin_idx)
                    fw_grad[grad_idx] = 0.0

    return jacobians


def _get_input_to_perturb(input):
    # Prepare the input so that it can be modified in-place. Dense tensors
    # use .data to bypass the version counter.
    if _is_sparse_any_tensor(input):
        # Clone because input may require grad, and copy_ calls resize_,
        # which is not allowed for .data
        return input.clone()
    return input.data


def _with_prepare_inputs(fn, inputs, input_idx, input_to_perturb, fast_mode=False):
    # Wraps `fn` so that its inputs are already supplied.  Dense inputs are
    # passed as-is: `entry` (`input.data`) cannot stand in for them when
    # gradgrad has to work, because `fn` (in the gradgrad case) needs to
    # compute grad wrt the original input object.
    def wrapped_fn():
        inp = tuple(
            input_to_perturb
            if fast_mode and i == input_idx and _is_sparse_any_tensor(a)
            else a
            for i, a in enumerate(_as_tuple(inputs))
        )
        return tuple(a.clone() for a in _as_tuple(fn(*inp)))

    return wrapped_fn


def _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn):
    # Wraps jvp_fn so that certain arguments are already supplied
    def jvp_fn(delta):
        return _compute_numerical_gradient(wrapped_fn, input_to_perturb, delta, eps, nbhd_checks_fn)

    return jvp_fn


def _reshape_tensor_or_tuple(u, shape):
    if isinstance(u, tuple):
        return (u[0].reshape(shape), u[1].reshape(shape))
    return u.reshape(shape)


def _mul_tensor_or_tuple(u, k):
    if isinstance(u, tuple):
        return (k * u[0], k * u[1])
    return k * u


def _get_numerical_jvp_wrt_specific_input(fn, input_idx, inputs, u, eps, is_forward_ad=False) -> list:
    input = inputs[input_idx]
    input_to_perturb = _get_input_to_perturb(input)
    wrapped_fn = _with_prepare_inputs(fn, inputs, input_idx, input_to_perturb, True)
    nbhd_checks_fn = functools.partial(_check_outputs_same_dtype_and_shape, eps=eps)
    jvp_fn = _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn)
    u = _reshape_tensor_or_tuple(u, input_to_perturb.shape)
    u = _mul_tensor_or_tuple(u, eps)
    return _compute_numerical_jvps_wrt_specific_input(jvp_fn, u, input.is_complex(), is_forward_ad)


def _get_numerical_vJu(fn, inputs, inp_indices, func_out, all_u, all_v, eps, is_forward_ad):
    # Note that all_v can also be None, in that case, this function only computes Ju.
    reduced_jacobians: list[list[tensorplay.Tensor]] = []
    for inp_idx, u in zip(inp_indices, all_u):
        all_Ju = _get_numerical_jvp_wrt_specific_input(fn, inp_idx, inputs, u, eps, is_forward_ad)
        # Filter out the Ju for non floating point outputs
        filtered_Ju = []
        func_out = _as_tuple(func_out)
        if len(all_Ju) != len(func_out):
            raise AssertionError(
                f"Expected all_Ju and func_out to have the same length, "
                f"but got {len(all_Ju)} and {len(func_out)}"
            )
        for Ju, output in zip(all_Ju, func_out):
            if _is_float_or_complex_tensor(output):
                filtered_Ju.append(Ju)
        if all_v is not None:
            jacobian_scalars: list[tensorplay.Tensor] = []
            for v, Ju in zip(all_v, filtered_Ju):
                jacobian_scalars.append(_dot_with_type_promotion(v, Ju))
            reduced_jacobians.append(jacobian_scalars)
        else:
            reduced_jacobians.append(filtered_Ju)
    return reduced_jacobians


def _check_jacobians_equal(j1, j2, atol):
    # Check whether the max difference between two Jacobian tensors are within some
    # tolerance `atol`.
    for j1_x, j2_x in zip(j1, j2):
        if j1_x.numel() != 0 and (j1_x - j2_x).abs().max().item() > atol:
            return False
    return True


def _stack_and_check_tensors(list_of_list_of_tensors, inputs, numel_outputs):
    # For the ith tensor in the inner list checks whether it has the same size and
    # dtype as the ith differentiable input.
    out_jacobians = _allocate_jacobians_with_inputs(inputs, numel_outputs)
    diff_input_list = list(_iter_tensors(inputs, True))
    correct_grad_sizes = True
    correct_grad_types = True
    for i, tensor_list in enumerate(list_of_list_of_tensors):
        inp = diff_input_list[i]
        out_jacobian = out_jacobians[i]
        for j, tensor in enumerate(tensor_list):
            col = out_jacobian.select(1, j)
            if tensor is not None and tuple(tensor.shape) != tuple(inp.shape):
                correct_grad_sizes = False
            elif tensor is not None and tensor.dtype != inp.dtype:
                correct_grad_types = False
            if tensor is None:
                col.zero_()
            else:
                dense = tensor.to_dense() if _is_sparse_any_tensor(tensor) else tensor
                dense = dense.reshape(-1)
                if col.numel() != dense.numel():
                    raise AssertionError(
                        f"Expected out_jacobian column to have {dense.numel()} elements, "
                        f"but got {col.numel()}"
                    )
                col.copy_(dense)
    return out_jacobians, correct_grad_sizes, correct_grad_types


FAILED_NONDET_MSG = """\n
NOTE: If your op relies on non-deterministic operations this failure might be
expected.

If you are adding a new operator and the test manually invokes
gradcheck/gradgradcheck, call gradcheck/gradgradcheck with
`nondet_tol=<tol>` as a keyword argument.
"""


def _check_analytical_jacobian_attributes(
    inputs, output, nondet_tol, check_grad_dtypes, fast_mode=False, v=None
) -> tuple:
    # This is used by both fast and slow mode:
    #  - For slow mode, vjps[i][j] is the jth row of the Jacobian wrt the ith
    #    input.
    #  - For fast mode, vjps[i][0] is a linear combination of the rows
    #    of the Jacobian wrt the ith input
    diff_input_list = list(_iter_tensors(inputs, True))

    def vjp_fn(grad_output):
        return tensorplay.autograd.grad(
            output, diff_input_list, grad_output, retain_graph=True, allow_unused=True
        )

    # Compute everything twice to check for nondeterminism (which we call reentrancy)
    if fast_mode:
        vjps1 = _get_analytical_vjps_wrt_specific_output(vjp_fn, output.clone(), v)
        vjps2 = _get_analytical_vjps_wrt_specific_output(vjp_fn, output.clone(), v)
    else:
        vjps1 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
        vjps2 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())

    output_numel = output.numel() if not fast_mode else 1
    jacobians1, types_ok, sizes_ok = _stack_and_check_tensors(vjps1, inputs, output_numel)
    jacobians2, _, _ = _stack_and_check_tensors(vjps2, inputs, output_numel)
    reentrant = _check_jacobians_equal(jacobians1, jacobians2, nondet_tol)

    if not types_ok and check_grad_dtypes:
        raise GradcheckError("Gradient has dtype mismatch")
    if not sizes_ok:
        raise GradcheckError("Analytical gradient has incorrect size")
    if not reentrant:
        raise GradcheckError(
            "Backward is not reentrant, i.e., running backward with "
            "same input and grad_output multiple times gives different values, "
            "although analytical gradient matches numerical gradient."
            f"The tolerance for nondeterminism was {nondet_tol}." + FAILED_NONDET_MSG
        )
    return jacobians1


def _get_analytical_vJu_backward_mode(inputs, outputs, nondet_tol, check_grad_dtypes, all_v, all_u):
    reduced_jacobians: list[list[tensorplay.Tensor]] = []
    for output, v in zip(outputs, all_v):
        all_vJ = _check_analytical_jacobian_attributes(
            inputs, output, nondet_tol, check_grad_dtypes, fast_mode=True, v=v
        )
        jacobian_scalars: list[tensorplay.Tensor] = []
        for vJ, u in zip(all_vJ, all_u):
            # vJ is a 2-d tensor so that the error checking logic of slow
            # mode applies; its single column is the product.
            vJ = vJ.T.squeeze(0)
            if vJ.is_complex():  # C -> R
                tv = tensorplay.view_as_real(vJ.resolve_conj())
                tr = tv.select(-1, 0)
                ti = tv.select(-1, 1)
                jacobian_scalars.append(tr.dot(u[0]) + 1j * ti.dot(u[1]))
            else:  # R -> R
                jacobian_scalars.append(vJ.dot(u))
        reduced_jacobians.append(jacobian_scalars)
    return reduced_jacobians


def _get_analytical_jacobian(inputs, outputs, input_idx, output_idx):
    # Computes the analytical Jacobian in slow mode for a single input-output pair.
    # Forgoes performing checks on dtype, shape, and reentrancy.
    jacobians = _check_analytical_jacobian_attributes(
        inputs, outputs[output_idx], nondet_tol=float("inf"), check_grad_dtypes=False
    )
    return jacobians[input_idx]


def _compute_analytical_jacobian_rows(vjp_fn, sample_output):
    # Computes Jacobian row-by-row by projecting `vjp_fn` = v^T J on standard basis
    # vectors: vjp_fn(e) = e^T J is a corresponding row of the Jacobian.
    # NB: this function does not assume vjp_fn(v) to return tensors with the same
    # number of elements for different v. This is checked when we later combine the
    # rows into a single tensor.
    grad_out_base = tensorplay.zeros_like(sample_output)
    flat_grad_out = grad_out_base.view(-1)
    # jacobians_rows[i][j] is the Jacobian jth row for the ith input
    jacobians_rows: list[list[tensorplay.Tensor | None]] = []
    for j in range(flat_grad_out.numel()):
        flat_grad_out.zero_()
        flat_grad_out[j] = 1.0  # projection for jth row of Jacobian
        grad_inputs = vjp_fn(grad_out_base)
        for i, d_x in enumerate(grad_inputs):
            if j == 0:
                jacobians_rows.append([])
            jacobians_rows[i] += [d_x.clone() if isinstance(d_x, tensorplay.Tensor) else None]
    return jacobians_rows


def _get_analytical_vjps_wrt_specific_output(vjp_fn, sample_output, v):
    grad_inputs = vjp_fn(v.reshape(sample_output.shape))
    return [[vjp.clone() if isinstance(vjp, tensorplay.Tensor) else None] for vjp in grad_inputs]


def _check_inputs(tupled_inputs) -> bool:
    # Make sure that gradients are saved for at least one input
    any_input_requiring_grad = False
    for idx, inp in enumerate(tupled_inputs):
        if _is_tensor_like(inp) and inp.requires_grad:
            if not (inp.dtype == tensorplay.float64 or inp.dtype == tensorplay.complex128):
                warnings.warn(
                    f"Input #{idx} requires gradient and "
                    "is not a double precision floating point or complex. "
                    "This check will likely fail if all the inputs are "
                    "not of double precision floating point or complex. ",
                    stacklevel=2,
                )
            content = inp._values() if _is_sparse_any_tensor(inp) else inp
            # TODO: To cover more problematic cases, replace stride = 0 check with
            # "any overlap in memory" once we have a proper function to check it.
            if not all(st > 0 or sz <= 1 for st, sz in zip(content.stride(), content.size())):
                raise RuntimeError(
                    f"The {idx}th input has a dimension with stride 0. gradcheck only "
                    "supports inputs that are non-overlapping to be able to "
                    "compute the numerical gradients correctly. You should call "
                    ".contiguous on the input before passing it to gradcheck."
                )
            any_input_requiring_grad = True

    if not any_input_requiring_grad:
        raise ValueError(
            "gradcheck expects at least one input tensor to require gradient, "
            "but none of them have requires_grad=True."
        )
    return True


def _check_outputs(outputs) -> None:
    if any(_is_sparse_any_tensor(t) for t in outputs):
        # it is easier to call to_dense() on the sparse output than
        # to modify analytical jacobian
        raise ValueError(
            "Sparse output is not supported at gradcheck yet. "
            "Please call to_dense() on the output of fn for gradcheck."
        )


def _check_no_differentiable_outputs(func, inputs, func_out, eps, *, is_forward_ad) -> bool:
    # When there are no differentiable outputs, numerical gradient for a function is
    # expected to be zero.
    jacobians_all_inputs_outputs = _get_numerical_jacobian(
        func, inputs, func_out, eps=eps, is_forward_ad=is_forward_ad
    )
    for jacobians_all_outputs_and_fixed_input in jacobians_all_inputs_outputs:
        for jacobian in jacobians_all_outputs_and_fixed_input:
            if tensorplay.ne(jacobian, 0).sum().item() > 0:
                raise GradcheckError("Numerical gradient for function expected to be zero")
    return True


def _check_no_differentiable_outputs_fast(func, func_out, all_inputs, inputs_indices, all_u, eps, nondet_tol):
    for inp_idx, u in zip(inputs_indices, all_u):
        jvps = _get_numerical_jvp_wrt_specific_input(func, inp_idx, all_inputs, u, eps)
        for jvp in jvps:
            if jvp.numel() == 0:
                continue
            if jvp.abs().max().item() > nondet_tol:
                raise GradcheckError("Numerical gradient for function expected to be zero")
    return True


FAILED_BATCHED_GRAD_MSG = """
gradcheck or gradgradcheck failed while testing batched gradient computation.

To compute batched grads (e.g., jacobians, hessians), we vmap over the backward
computation. The most common failure case is a backward formula that runs an
operation vmap cannot batch, or that reads or writes its gradient in place.

If the test manually invokes gradcheck/gradgradcheck and the operator does not
support batched gradients yet, call gradcheck/gradgradcheck with
`check_batched_grad=False` as a keyword argument.
""".strip()

FAILED_BATCHED_GRAD_MSG_FWD_AD = """
gradcheck failed while testing batched gradient computation with forward-mode AD.
This test is enabled automatically when both `check_batched_grad=True`
and `check_forward_ad=True`.  If the test manually invokes gradcheck and the
operator does not support batched forward gradients yet, call gradcheck with
`check_batched_forward_grad=False` as a keyword argument.
""".strip()


def _get_failed_batched_grad_test_msg(output_idx, input_idx, res, exp, is_forward_ad=False):
    return f"""
For output {output_idx} and input {input_idx}:

{FAILED_BATCHED_GRAD_MSG_FWD_AD if is_forward_ad else FAILED_BATCHED_GRAD_MSG}

Got:
{res}

Expected:
{exp}
""".strip()


def _test_batched_grad_forward_ad(func, inputs) -> bool:
    fwAD = tensorplay.autograd.forward_ad
    if not isinstance(inputs, tuple):
        raise AssertionError("Expected inputs to be a tuple")

    for input_idx, current_input in enumerate(inputs):
        if not (_is_tensor_like(current_input) and current_input.requires_grad):
            continue

        def jvp(tangent):
            with fwAD.dual_level():
                dual = fwAD.make_dual(current_input.detach(), tangent)
                inputs_with_dual = tuple(
                    dual if idx == input_idx else (inp.detach() if _is_tensor_like(inp) else inp)
                    for idx, inp in enumerate(inputs)
                )
                dual_outputs = _as_tuple(func(*inputs_with_dual))
                ret = []
                for dual_output in dual_outputs:
                    if dual_output is None:
                        continue
                    primal_out, tangent_out = fwAD.unpack_dual(dual_output)
                    if tangent_out is not None:
                        ret.append(tangent_out)
                    else:
                        ret.append(
                            tensorplay.zeros([], dtype=primal_out.dtype, device=primal_out.device)
                            .expand(primal_out.shape)
                        )
                return tuple(ret)

        if not _is_float_or_complex_tensor(current_input):
            continue

        tangents = [tensorplay.randn_like(current_input) for _ in range(2)]
        expected = [jvp(t) for t in tangents]
        expected = [tensorplay.stack(shards) for shards in zip(*expected)]

        try:
            result = tensorplay.vmap(jvp)(tensorplay.stack(tangents))
        except RuntimeError as ex:
            # Rethrow to provide a better error message
            raise GradcheckError(
                f"While computing batched gradients, got: {ex}\n\n{FAILED_BATCHED_GRAD_MSG_FWD_AD}"
            ) from ex

        for output_idx, (res, exp) in enumerate(zip(result, expected)):
            if tensorplay.allclose(res, exp):
                continue
            raise GradcheckError(
                _get_failed_batched_grad_test_msg(output_idx, input_idx, res, exp, is_forward_ad=True)
            )
    return True


def _test_batched_grad(input, output, output_idx) -> bool:
    # NB: _test_batched_grad compares two autograd.grad invocations with a single
    # vmap(autograd.grad) invocation. It's not exactly a "gradcheck" in the
    # sense that we're not comparing an analytical jacobian with a numeric one,
    # but it is morally similar (we could have computed a full analytic jac
    # via vmap, but that is potentially slow)
    diff_input_list = list(_iter_tensors(input, True))
    grad = functools.partial(
        tensorplay.autograd.grad,
        output,
        diff_input_list,
        retain_graph=True,
        allow_unused=True,
    )

    def vjp(v):
        results = grad(v)
        return tuple(
            g if g is not None
            else tensorplay.zeros([], dtype=inp.dtype, device=inp.device).expand(inp.shape)
            for g, inp in zip(results, diff_input_list)
        )

    grad_outputs = [tensorplay.randn_like(output) for _ in range(2)]

    expected = [vjp(gO) for gO in grad_outputs]
    expected = [tensorplay.stack(shards) for shards in zip(*expected)]

    try:
        result = tensorplay.vmap(vjp)(tensorplay.stack(grad_outputs))
    except RuntimeError as ex:
        # The error surfaces here rather than at the backward formula that
        # raised it; the message carries the formula's own report.
        raise GradcheckError(
            f"While computing batched gradients, got: {ex}\n\n{FAILED_BATCHED_GRAD_MSG}"
        ) from ex

    for input_idx, (res, exp) in enumerate(zip(result, expected)):
        if tensorplay.allclose(res, exp):
            continue
        raise GradcheckError(_get_failed_batched_grad_test_msg(output_idx, input_idx, res, exp))
    return True


def _test_backward_mul_by_grad_output(outputs, inputs, masked) -> bool:
    # Tests that backward is multiplied by grad_output
    diff_input_list: list[tensorplay.Tensor] = list(_iter_tensors(inputs, True))
    if not diff_input_list:
        raise GradcheckError("no Tensors requiring grad found in input")
    grads_input = tensorplay.autograd.grad(
        outputs,
        diff_input_list,
        [tensorplay.zeros_like(o) for o in outputs],
        allow_unused=True,
    )
    for gi, di in zip(grads_input, diff_input_list):
        if gi is None:
            continue
        if _is_sparse_any_tensor(gi) or _is_sparse_any_tensor(di):
            if gi.layout != di.layout:
                raise GradcheckError(f"grad is incorrect layout ({gi.layout} is not {di.layout})")
            if _is_sparse_any_tensor(gi):
                if gi.sparse_dim() != di.sparse_dim():
                    raise GradcheckError(
                        "grad is sparse tensor, but has incorrect sparse_dim"
                        f" {gi.sparse_dim()}, expected {di.sparse_dim()}"
                    )
                if gi.dense_dim() != di.dense_dim():
                    raise GradcheckError(
                        "grad is sparse tensor, but has incorrect dense_dim"
                        f" {gi.dense_dim()}, expected {di.dense_dim()}"
                    )
            gi = gi.to_dense()
            di = di.to_dense()
        if masked:
            if not tensorplay.allclose(gi, tensorplay.zeros_like(gi)):
                raise GradcheckError("backward not multiplied by grad_output")
        elif not gi.eq(0).all().item():
            raise GradcheckError("backward not multiplied by grad_output")
        if gi.dtype != di.dtype:
            raise GradcheckError("grad is incorrect type")
        if gi.device != di.device:
            raise GradcheckError("grad is incorrect device")
        if tuple(gi.size()) != tuple(di.size()):
            raise GradcheckError("grad is incorrect size")
    return True


def _test_undefined_forward_mode(func, outputs, inputs):
    fwAD = tensorplay.autograd.forward_ad

    _inp_tensors_idx, inp_tensors = _get_inp_tensors(inputs)
    _all_v, all_u, _all_u_dense = _make_vectors(inp_tensors, outputs, use_forward_ad=True)

    with fwAD.dual_level():
        fw_grads = []
        dual_inputs = []
        tensor_indices = set()
        for i, inp in enumerate(inputs):
            if _is_tensor_like(inp) and inp.requires_grad:
                inp = fwAD.make_dual(inp.detach(), tensorplay.zeros_like(inp))
                # If inp is a differentiable view, the dual might not be the tangent given to
                # make_dual, so read it explicitly from the dual tensor
                fw_grads.append(fwAD.unpack_dual(inp)[1])
                tensor_indices.add(i)
            dual_inputs.append(inp)

        for fw_grad, u in zip(fw_grads, all_u):
            fw_grad.copy_(u.view_as(fw_grad))

        for idx, inp in enumerate(inputs):
            if idx not in tensor_indices:
                continue
            dual_inp_obj = dual_inputs[idx]

            # case 1 (Materialized Zero Tensor Tangent)
            dual_inputs[idx] = fwAD.make_dual(inp.detach(), tensorplay.zeros_like(inp))
            raw_outputs = _as_tuple(func(*dual_inputs))
            dual_outputs1 = filter(_is_float_or_complex_tensor, raw_outputs)

            # case 2 (Efficient Zero Tensor Tangent since we don't make a dual object and pass a regular tensor)
            dual_inputs[idx] = inp.detach()
            raw_outputs = _as_tuple(func(*dual_inputs))
            dual_outputs2 = filter(_is_float_or_complex_tensor, raw_outputs)

            # reset
            dual_inputs[idx] = dual_inp_obj

            for index_o, (d_o1, d_o2) in enumerate(zip(dual_outputs1, dual_outputs2)):
                _val1, res1 = fwAD.unpack_dual(d_o1)
                _val2, res2 = fwAD.unpack_dual(d_o2)

                if not (res1 is None or res2 is None):
                    if not tensorplay.allclose(res1, res2):
                        raise GradcheckError(
                            f"Mismatch in tangent values for output with index: {index_o}"
                            f" when input: {inp} has an undefined tangent value. "
                            f" Got: {res1} but expected: {res2}"
                        )
    return True


def _test_undefined_backward_mode(func, outputs, inputs) -> bool:
    diff_input_list: list[tensorplay.Tensor] = list(_iter_tensors(inputs, True))
    if not diff_input_list:
        raise GradcheckError("no Tensors requiring grad found in input")

    def warn_bc_breaking():
        warnings.warn(
            "Backwards compatibility: New undefined gradient support checking "
            "feature is enabled by default, but it may break existing callers "
            "of this function. If this is true for you, you can call this "
            'function with "check_undefined_grad=False" to disable the feature',
            stacklevel=2,
        )

    def check_undefined_grad_support(output_to_check):
        grads_output = [tensorplay.zeros_like(o) for o in output_to_check]
        try:
            grads_input = tensorplay.autograd.grad(
                output_to_check, diff_input_list, grads_output, allow_unused=True
            )
        except RuntimeError as e:
            warn_bc_breaking()
            raise GradcheckError(
                "Expected backward function to handle undefined output grads. "
            ) from e

        for gi in grads_input:
            if (gi is not None) and (not gi.eq(0).all().item()):
                warn_bc_breaking()
                raise GradcheckError(
                    "Expected all input grads to be undefined or zero when all output grads are undefined "
                    "or zero."
                )
        return True

    # All backward functions must work properly if all output grads are undefined
    outputs_to_check = [
        [
            _UndefinedGrad.apply(o)
            for o in _differentiable_outputs(func(*inputs))
            # This check filters out Tensor-likes that aren't instances of Tensor.
            if isinstance(o, tensorplay.Tensor)
        ]
    ]

    # If there are multiple output grads, we should be able to undef one at a time without error
    if len(outputs_to_check[0]) > 1:
        for undef_grad_idx in range(len(outputs)):
            output_to_check = _differentiable_outputs(func(*inputs))
            outputs_to_check.append(
                [
                    _UndefinedGrad.apply(o) if idx == undef_grad_idx else o
                    for idx, o in enumerate(output_to_check)
                ]
            )

    return all(check_undefined_grad_support(output) for output in outputs_to_check)


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)


def _differentiable_outputs(x):
    return tuple(o for o in _as_tuple(x) if o.requires_grad)


def _get_notallclose_msg(
    analytical, numerical, output_idx, input_idx, complex_indices=None, test_imag=False, is_forward_ad=False
) -> str:
    out_is_complex = (not is_forward_ad) and complex_indices and output_idx in complex_indices
    inp_is_complex = is_forward_ad and complex_indices and input_idx in complex_indices
    part = "imaginary" if test_imag else "real"
    element = "inputs" if is_forward_ad else "outputs"
    prefix = (
        ""
        if not (out_is_complex or inp_is_complex)
        else f"While considering the {part} part of complex {element} only, "
    )
    mode = "computed with forward mode " if is_forward_ad else ""
    return (
        prefix
        + f"Jacobian {mode}mismatch for output {output_idx:d} with respect to input {input_idx:d},\n"
        f"numerical:{numerical}\nanalytical:{analytical}\n"
    )


def _transpose(matrix_of_tensors):
    # returns list of tuples
    return list(zip(*matrix_of_tensors))


def _real_and_imag_output(fn):
    # returns new functions real(fn), and imag(fn) where real(fn) and imag(fn) behave the same as
    # the original fn, except tensorplay.real or tensorplay.imag are applied to the complex outputs
    def apply_to_c_outs(fn, fn_to_apply):
        def wrapped_fn(*inputs):
            outs = _as_tuple(fn(*inputs))
            return tuple(fn_to_apply(o) if o.is_complex() else o for o in outs)

        return wrapped_fn

    return apply_to_c_outs(fn, tensorplay.real), apply_to_c_outs(fn, tensorplay.imag)


def _real_and_imag_input(fn, complex_inp_indices, tupled_inputs):
    # returns new functions that take real inputs instead of complex inputs as
    # (x, y) -> fn(x + y * 1j). And it computes: inp -> fn(inp + y * 1j) and inp -> fn(x + inp * 1j).
    # In each case, the other part is considered constant.
    # We do not use 0 for the constant here to make sure we always call the user function with a valid input.
    def apply_to_c_inps(fn, fn_to_apply):
        def wrapped_fn(*inputs):
            new_inputs = list(inputs)
            for should_be_complex in complex_inp_indices:
                new_inputs[should_be_complex] = fn_to_apply(
                    new_inputs[should_be_complex], tupled_inputs[should_be_complex]
                )
            return _as_tuple(fn(*new_inputs))

        return wrapped_fn

    real_fn = apply_to_c_inps(fn, lambda inp, orig: inp + orig.imag * 1j)
    imag_fn = apply_to_c_inps(fn, lambda inp, orig: orig.real + inp * 1j)
    return real_fn, imag_fn


def _gradcheck_real_imag(
    gradcheck_fn,
    func,
    func_out,
    tupled_inputs,
    outputs,
    eps,
    rtol,
    atol,
    check_grad_dtypes,
    check_forward_ad,
    check_backward_ad,
    nondet_tol,
    check_undefined_grad,
):
    complex_out_indices = [i for i, o in enumerate(outputs) if o.is_complex()]
    has_any_complex_output = any(o.is_complex() for o in _as_tuple(func_out))
    if check_backward_ad:
        if has_any_complex_output:
            real_fn, imag_fn = _real_and_imag_output(func)

            imag_func_out = imag_fn(*tupled_inputs)
            imag_outputs = _differentiable_outputs(imag_func_out)
            gradcheck_fn(
                imag_fn,
                imag_func_out,
                tupled_inputs,
                imag_outputs,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                complex_indices=complex_out_indices,
                test_imag=True,
            )

            real_func_out = real_fn(*tupled_inputs)
            real_outputs = _differentiable_outputs(real_func_out)
            gradcheck_fn(
                real_fn,
                real_func_out,
                tupled_inputs,
                real_outputs,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                complex_indices=complex_out_indices,
            )
        else:
            gradcheck_fn(
                func,
                func_out,
                tupled_inputs,
                outputs,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
            )

    if check_forward_ad:
        complex_inp_indices = [
            i for i, inp in enumerate(tupled_inputs) if _is_tensor_like(inp) and inp.is_complex()
        ]
        if complex_inp_indices:
            real_fn, imag_fn = _real_and_imag_input(func, complex_inp_indices, tupled_inputs)

            imag_inputs = [
                _requires_grad_like(inp.imag, inp) if _is_tensor_like(inp) and inp.is_complex() else inp
                for inp in tupled_inputs
            ]
            imag_func_out = imag_fn(*imag_inputs)
            diff_imag_func_out = _differentiable_outputs(imag_func_out)
            gradcheck_fn(
                imag_fn,
                imag_func_out,
                imag_inputs,
                diff_imag_func_out,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                complex_indices=complex_inp_indices,
                test_imag=True,
                use_forward_ad=True,
            )

            real_inputs = [
                _requires_grad_like(inp.real, inp) if _is_tensor_like(inp) and inp.is_complex() else inp
                for inp in tupled_inputs
            ]
            real_func_out = real_fn(*real_inputs)
            diff_real_func_out = _differentiable_outputs(real_func_out)
            gradcheck_fn(
                real_fn,
                real_func_out,
                real_inputs,
                diff_real_func_out,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                complex_indices=complex_inp_indices,
                use_forward_ad=True,
            )
            if check_undefined_grad:
                _test_undefined_forward_mode(imag_fn, imag_func_out, imag_inputs)
                _test_undefined_forward_mode(real_fn, real_func_out, real_inputs)
        else:
            gradcheck_fn(
                func,
                func_out,
                tupled_inputs,
                outputs,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                use_forward_ad=True,
            )
            if check_undefined_grad:
                _test_undefined_forward_mode(func, outputs, tupled_inputs)


def _requires_grad_like(part, inp):
    # The real and imaginary parts of a complex input stand in for it as
    # separate inputs; they are leaves that require grad when it does.
    part = part.detach().clone()
    return part.requires_grad_(inp.requires_grad)


def _slow_gradcheck(
    func,
    func_out,
    tupled_inputs,
    outputs,
    eps,
    rtol,
    atol,
    check_grad_dtypes,
    nondet_tol,
    *,
    use_forward_ad=False,
    complex_indices=None,
    test_imag=False,
    masked=False,
):
    func_out = _as_tuple(func_out)
    if not outputs:
        return _check_no_differentiable_outputs(
            func, tupled_inputs, func_out, eps=eps, is_forward_ad=use_forward_ad
        )
    tupled_inputs_numerical = tupled_inputs if masked else _densify(tupled_inputs)

    numerical = _transpose(
        _get_numerical_jacobian(
            func, tupled_inputs_numerical, func_out, eps=eps, is_forward_ad=use_forward_ad
        )
    )
    # Note: [numerical vs analytical output length]
    # The numerical path returns jacobian quantity for all outputs, even if requires_grad of that
    # output is False. This behavior is necessary for _check_no_differentiable_outputs to work.
    numerical = [nj for o, nj in zip(func_out, numerical) if o.requires_grad]
    if use_forward_ad:
        analytical_forward = _get_analytical_jacobian_forward_ad(
            func, tupled_inputs, func_out, check_grad_dtypes=check_grad_dtypes
        )

        for i, n_per_out in enumerate(numerical):
            for j, n in enumerate(n_per_out):
                a = analytical_forward[j][i]
                if not _allclose_with_type_promotion(a, n.to(a.device), rtol, atol):
                    raise GradcheckError(
                        _get_notallclose_msg(a, n, i, j, complex_indices, test_imag, is_forward_ad=True)
                    )
    else:
        for i, o in enumerate(outputs):
            analytical = _check_analytical_jacobian_attributes(tupled_inputs, o, nondet_tol, check_grad_dtypes)

            for j, (a, n) in enumerate(zip(analytical, numerical[i])):
                if not _allclose_with_type_promotion(a, n.to(a.device), rtol, atol):
                    raise GradcheckError(_get_notallclose_msg(a, n, i, j, complex_indices, test_imag))

    return True


def _dot_with_type_promotion(u, v):
    if u.dim() != 1 or v.dim() != 1:
        raise AssertionError(f"Expected u and v to be 1D tensors, but got dims {u.dim()} and {v.dim()}")
    return (u * v).sum()


def _allclose_with_type_promotion(a, b, rtol, atol):
    promoted_type = tensorplay.promote_types(a.dtype, b.dtype)
    a = a.to(dtype=promoted_type)
    b = b.to(dtype=promoted_type)
    return tensorplay.allclose(a, b, rtol, atol)


def _to_real_dtype(dtype):
    if dtype == tensorplay.complex128:
        return tensorplay.float64
    elif dtype == tensorplay.complex64:
        return tensorplay.float32
    else:
        return dtype


def _vec_from_tensor(x, generator, downcast_complex=False):
    # Create a random vector with the same number of elements as x and the same
    # dtype/device. If x is complex and downcast_complex is False, we create a
    # complex tensor with only real component.
    dtype = _to_real_dtype(x.dtype) if downcast_complex else x.dtype
    if _is_sparse_any_tensor(x):
        # For sparse, create a random sparse vec with random values in the same
        # indices. Make sure size is set so that it isn't inferred to be smaller.
        x_values = x._values()
        values = (
            tensorplay.rand(x_values.numel(), generator=generator, dtype=tensorplay.float64)
            .to(dtype=dtype, device=x.device)
            .view(x_values.shape)
        )
        values /= values.norm()
        return tensorplay.sparse_coo_tensor(x._indices(), values, x.size(), device=x.device)
    # Drawn on the CPU, where the generator lives, then moved.
    vec = tensorplay.rand(x.numel(), generator=generator, dtype=tensorplay.float64).to(
        dtype=dtype, device=x.device
    )
    vec /= vec.norm()
    return vec


def _get_inp_tensors(tupled_inputs):
    inp_idx_tup = [(i, t) for i, t in enumerate(tupled_inputs) if _is_tensor_like(t) and t.requires_grad]
    return [tup[0] for tup in inp_idx_tup], [tup[1] for tup in inp_idx_tup]


def _adjusted_atol(atol, u, v):
    # In slow gradcheck, we compare A and B element-wise, i.e., for some a, b we
    # allow: |a - b| < atol + rtol * b. But since we now compare q1 = v^T A u and
    # q2 = v^T B u, we must allow |q1 - q2| < v^T E u + rtol * v^T B u, where E is
    # the correctly sized matrix in which each entry is atol.
    #
    # We see that atol needs to be scaled by v^T M u (where M is an all-ones M x N
    # matrix): v^T M u = \sum_{i} \sum_{j} u[i] * v[j] = sum(u) * sum(v).

    # For the case of complex inputs, u has re and im. components: u = (ur, ui).
    # Let q = a - b = (qr, qi) in the above notation, eg. q is the difference between analytic and numerical
    # Jacobians. Then the transformed tolerance checks being done in the allclose ops
    # are of form abs(Re{q_r u_r - q_i u_i}) < atol * (abs(u_r) + abs(u_i)), and equivalently
    # for imaginary components abs(Im{q_r u_r - q_i ui}) < atol * (abs(u_r) + abs(u_i)).
    # Since u is drawn randomly non-negative, the end effect is a factor
    # (sum(u_r) + sum(u_i)) * sum(v) increase in atol for complex inputs, eg.
    # a statistical factor of 2 as compared to the real case.

    sum_v = 1.0 if v is None else _real_sum(v)

    if isinstance(u, tuple):
        # case of complex input
        ur, ui = u[0], u[1]
        return atol * (_real_sum(ur) + _real_sum(ui)) * float(sum_v)

    # case of real input
    return atol * _real_sum(u) * float(sum_v)


def _real_sum(t):
    s = _to_flat_dense_if_sparse(t).sum()
    if s.is_complex():
        s = tensorplay.real(s)
    return float(s.item())


FAST_FAIL_SLOW_OK_MSG = """
Fast gradcheck failed but element-wise differences are small. This means that the
test might've passed in slow_mode!

If the test manually invokes gradcheck/gradgradcheck, call
gradcheck/gradgradcheck with `fast_mode=False` as a keyword argument.
""".strip()


def _run_slow_mode_and_get_error(func, tupled_inputs, outputs, input_idx, output_idx, rtol, atol, eps, is_forward_ad):
    # Compute jacobians in slow mode for better error message
    slow_numerical = _get_numerical_jacobian(func, tupled_inputs, outputs, eps=eps, is_forward_ad=is_forward_ad)[
        input_idx
    ][output_idx]
    if is_forward_ad:

        def new_fn(inp):
            new_inputs = list(tupled_inputs)
            new_inputs[input_idx] = inp
            return _as_tuple(func(*new_inputs))[output_idx]

        slow_analytical = _get_analytical_jacobian_forward_ad(
            new_fn, (tupled_inputs[input_idx],), (outputs[output_idx],)
        )[0][0]
    else:
        slow_analytical = _get_analytical_jacobian(tupled_inputs, outputs, input_idx, output_idx)

    # Assume jacobians are non-empty and have the same shape
    slow_max_diff = (slow_numerical - slow_analytical).abs().max().item()

    slow_allclose = tensorplay.allclose(slow_analytical, slow_numerical, rtol, atol)
    msg = (
        "\nThe above quantities relating the numerical and analytical jacobians are computed \n"
        "in fast mode, which compares v^T J u for random vectors u and v instead of whole \n"
        "jacobians. Below, we recompute numerical and analytical jacobians in slow mode:\n\n"
        f"Numerical:\n {slow_numerical}\n"
        f"Analytical:\n{slow_analytical}\n\n"
        f"The max per-element difference (slow mode) is: {slow_max_diff}.\n"
    )
    if slow_allclose:
        # Slow gradcheck would've passed!
        msg += FAST_FAIL_SLOW_OK_MSG
    return msg


def _to_flat_dense_if_sparse(tensor):
    if _is_sparse_any_tensor(tensor):
        return tensor.to_dense().reshape(-1)
    return tensor


def _make_vectors(inp_tensors, outputs, *, use_forward_ad):
    # Use our own generator to avoid messing with the user's RNG state
    g_cpu = tensorplay.Generator()

    all_u = []
    all_u_dense = []
    for inp in inp_tensors:
        ur = _vec_from_tensor(inp, g_cpu, True)
        ur_dense = _to_flat_dense_if_sparse(ur)
        if inp.is_complex():
            ui = _vec_from_tensor(inp, g_cpu, True)
            all_u.append((ur, ui))
            ui_dense = _to_flat_dense_if_sparse(ui)
            all_u_dense.append((ur_dense, ui_dense))
        else:
            all_u.append(ur)
            all_u_dense.append(ur_dense)
    all_v = None if use_forward_ad else [_vec_from_tensor(out, g_cpu) for out in outputs]
    return all_v, all_u, all_u_dense


def _check_analytical_numerical_equal(
    all_analytical,
    all_numerical,
    complex_indices,
    tupled_inputs,
    outputs,
    func,
    all_v,
    all_u,
    rtol,
    atol,
    eps,
    test_imag,
    *,
    is_forward_ad=False,
):
    for i, all_numerical_for_input_i in enumerate(all_numerical):
        for j, n in enumerate(all_numerical_for_input_i):
            # Forward AD generates the transpose of what this function expects
            if is_forward_ad:
                a = all_analytical[i][j]
            else:
                a = all_analytical[j][i]
            n = n.to(device=a.device)
            updated_atol = _adjusted_atol(atol, all_u[i], all_v[j] if all_v else None)
            if not _allclose_with_type_promotion(a, n.to(a.device), rtol, updated_atol):
                jacobians_str = _run_slow_mode_and_get_error(
                    func, tupled_inputs, outputs, i, j, rtol, atol, eps, is_forward_ad
                )
                raise GradcheckError(
                    _get_notallclose_msg(a, n, j, i, complex_indices, test_imag, is_forward_ad) + jacobians_str
                )


def _fast_gradcheck(
    func,
    func_out,
    inputs,
    outputs,
    eps,
    rtol,
    atol,
    check_grad_dtypes,
    nondet_tol,
    *,
    use_forward_ad=False,
    complex_indices=None,
    test_imag=False,
    masked=False,
):
    inp_tensors_idx, inp_tensors = _get_inp_tensors(inputs)
    # Backward mode computes v^T * J (VJP)
    # Since we computed J * u (JVP) through finite difference method, we perform an equality check
    # between VJP * u, v * JVP
    # ----
    # Forward mode computes J * u (JVP)
    # Since we already compute JVP through finite difference method,
    # we don't need v for correctness check here as asserted below
    all_v, all_u, all_u_dense = _make_vectors(inp_tensors, outputs, use_forward_ad=use_forward_ad)

    inputs_numerical, all_u_numerical, all_v_numerical = (
        (inputs, all_u, all_v) if masked else _densify((inputs, all_u, all_v))
    )

    numerical_vJu = _get_numerical_vJu(
        func,
        inputs_numerical,
        inp_tensors_idx,
        func_out,
        all_u_numerical,
        all_v_numerical,
        eps,
        is_forward_ad=use_forward_ad,
    )
    if use_forward_ad:
        if all_v is not None:
            raise AssertionError("Expected all_v to be None.")
        analytical_vJu = _get_analytical_jacobian_forward_ad(
            func,
            inputs,
            _as_tuple(func_out),
            all_u=all_u,
            check_grad_dtypes=check_grad_dtypes,
        )
    else:
        if not outputs:
            _check_no_differentiable_outputs_fast(func, func_out, inputs, inp_tensors_idx, all_u, eps, nondet_tol)

        analytical_vJu = _get_analytical_vJu_backward_mode(
            inputs, outputs, nondet_tol, check_grad_dtypes, all_v, all_u_dense
        )

    _check_analytical_numerical_equal(
        analytical_vJu,
        numerical_vJu,
        complex_indices,
        inputs,
        outputs,
        func,
        all_v,
        all_u,
        rtol,
        atol,
        eps,
        test_imag,
        is_forward_ad=use_forward_ad,
    )

    return True


# Note [VarArg of Tensors]
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 'func' accepts a vararg of tensors, which isn't expressible in the type system at the moment.
# For now, we permit any input.
def gradcheck(
    func,  # See Note [VarArg of Tensors]
    inputs,
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
    nondet_tol: float = 0.0,
    check_undefined_grad: bool = True,
    check_grad_dtypes: bool = False,
    check_batched_grad: bool = False,
    check_batched_forward_grad: bool = False,
    check_forward_ad: bool = False,
    check_backward_ad: bool = True,
    fast_mode: bool = False,
    masked: bool | None = None,
) -> bool:
    r"""Check gradients computed via small finite differences against analytical
    gradients wrt tensors in :attr:`inputs` that are of floating point or complex type
    and with ``requires_grad=True``.

    The check between numerical and analytical gradients uses :func:`~tensorplay.allclose`.

    For most of the complex functions we consider for optimization purposes, no notion of
    Jacobian exists. Instead, gradcheck verifies if the numerical and analytical values of
    the Wirtinger and Conjugate Wirtinger derivatives are consistent. Because the gradient
    computation is done under the assumption that the overall function has a real-valued
    output, we treat functions with complex output in a special way. For these functions,
    gradcheck is applied to two real-valued functions corresponding to taking the real
    components of the complex outputs for the first, and taking the imaginary components
    of the complex outputs for the second.

    .. note::
        The default values are designed for :attr:`input` of double precision.
        This check will likely fail if :attr:`input` is of less precision, e.g.,
        ``FloatTensor``.

    .. note::
        Gradcheck may fail when evaluated on non-differentiable points
        because the numerically computed gradients via finite differencing may differ
        those computed analytically (not necessarily because either is incorrect).

    .. warning::
       If any checked tensor in :attr:`input` has overlapping memory, i.e.,
       different indices pointing to the same memory address (e.g., from
       :meth:`expand <tensorplay.Tensor.expand>`), this check will likely fail because the numerical
       gradients computed by point perturbation at such indices will change
       values at all other indices that share the same memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
        nondet_tol (float, optional): tolerance for non-determinism. When running
            identical inputs through the differentiation, the results must either match
            exactly (default, 0.0) or be within this tolerance.
        check_undefined_grad (bool, optional): if ``True``, check if undefined output grads
            are supported and treated as zeros, for ``Tensor`` outputs.
        check_grad_dtypes (bool, optional): if ``True``, check that the gradient
            dtypes match the ones from the numerical computation. Defaults to ``False``.
        check_batched_grad (bool, optional): if ``True``, check if we can compute
            batched gradients using :func:`tensorplay.vmap`. Defaults to False.
        check_batched_forward_grad (bool, optional): if ``True``, checks if we can compute
            batched forward gradients using forward ad and :func:`tensorplay.vmap`.
            Defaults to ``False``.
        check_forward_ad (bool, optional): if ``True``, check that the gradients computed with forward
            mode AD match the numerical ones. Defaults to ``False``.
        check_backward_ad (bool, optional): if ``False``, do not perform any checks that rely on
            backward mode AD to be implemented. Defaults to ``True``.
        fast_mode (bool, optional): if ``True``, compare the jacobians projected onto
            random vectors instead of element by element, which costs one backward
            (or forward) pass per output instead of one per output element.
            Defaults to ``False``.
        masked (bool, optional): if ``True``, the gradients of unspecified elements of
            sparse tensors are ignored. Defaults to ``False``.
    Returns:
        ``True`` if all differences satisfy allclose condition

    """
    if not (check_forward_ad or check_backward_ad):
        raise AssertionError("Expected at least one of check_forward_ad or check_backward_ad to be True")
    if check_batched_grad and not check_backward_ad:
        raise AssertionError("Setting check_batched_grad=True requires check_backward_ad to be True")
    if check_batched_forward_grad and not check_forward_ad:
        raise AssertionError("Setting check_batched_forward_grad=True requires check_forward_ad to be True")
    args = locals().copy()
    args.pop("raise_exception")
    if not raise_exception:
        try:
            return _gradcheck_helper(**args)
        except GradcheckError:
            return False
    else:
        return _gradcheck_helper(**args)


def _gradcheck_helper(
    func,
    inputs,
    eps,
    atol,
    rtol,
    nondet_tol,
    check_undefined_grad,
    check_grad_dtypes,
    check_batched_grad,
    check_batched_forward_grad,
    check_forward_ad,
    check_backward_ad,
    fast_mode,
    masked,
):
    tupled_inputs = _as_tuple(inputs)
    _check_inputs(tupled_inputs)

    func_out = func(*tupled_inputs)
    outputs = _differentiable_outputs(func_out)
    _check_outputs(outputs)

    gradcheck_fn = functools.partial(_fast_gradcheck if fast_mode else _slow_gradcheck, masked=masked)
    _gradcheck_real_imag(
        gradcheck_fn,
        func,
        func_out,
        tupled_inputs,
        outputs,
        eps,
        rtol,
        atol,
        check_grad_dtypes,
        check_forward_ad=check_forward_ad,
        check_backward_ad=check_backward_ad,
        nondet_tol=nondet_tol,
        check_undefined_grad=check_undefined_grad,
    )

    if check_batched_forward_grad:
        _test_batched_grad_forward_ad(func, tupled_inputs)

    # Short circuit because remaining tests rely on backward AD to be implemented
    if not check_backward_ad:
        return True

    for i, o in enumerate(outputs):
        if check_batched_grad:
            _test_batched_grad(tupled_inputs, o, i)

    _test_backward_mul_by_grad_output(outputs, tupled_inputs, masked)

    if check_undefined_grad and check_backward_ad:
        _test_undefined_backward_mode(func, outputs, tupled_inputs)
    return True


def gradgradcheck(
    func,  # See Note [VarArg of Tensors]
    inputs,
    grad_outputs=None,
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    gen_non_contig_grad_outputs: bool = False,
    raise_exception: bool = True,
    nondet_tol: float = 0.0,
    check_undefined_grad: bool = True,
    check_grad_dtypes: bool = False,
    check_batched_grad: bool = False,
    check_fwd_over_rev: bool = False,
    check_rev_over_rev: bool = True,
    fast_mode: bool = False,
    masked: bool = False,
) -> bool:
    r"""Check gradients of gradients computed via small finite differences
    against analytical gradients wrt tensors in :attr:`inputs` and
    :attr:`grad_outputs` that are of floating point or complex type and with
    ``requires_grad=True``.

    This function checks that backpropagating through the gradients computed
    to the given :attr:`grad_outputs` are correct.

    The check between numerical and analytical gradients uses :func:`~tensorplay.allclose`.

    .. note::
        The default values are designed for :attr:`input` and
        :attr:`grad_outputs` of double precision. This check will likely fail if
        they are of less precision, e.g., ``FloatTensor``.

    .. warning::
       If any checked tensor in :attr:`input` and :attr:`grad_outputs` has
       overlapping memory, i.e., different indices pointing to the same memory
       address (e.g., from :meth:`expand <tensorplay.Tensor.expand>`), this check
       will likely fail because the numerical gradients computed by point
       perturbation at such indices will change values at all other indices that
       share the same memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        grad_outputs (tuple of [Tensor or None] or Tensor, optional): The gradients with
            respect to the function's outputs.
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        gen_non_contig_grad_outputs (bool, optional): if :attr:`grad_outputs` is
            ``None`` and :attr:`gen_non_contig_grad_outputs` is ``True``, the
            randomly generated gradient outputs are made to be noncontiguous
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
        nondet_tol (float, optional): tolerance for non-determinism. When running
            identical inputs through the differentiation, the results must either match
            exactly (default, 0.0) or be within this tolerance. Note that a small amount
            of nondeterminism in the gradient will lead to larger inaccuracies in
            the second derivative.
        check_undefined_grad (bool, optional): if True, check if undefined output grads
            are supported and treated as zeros
        check_batched_grad (bool, optional): if True, check if we can compute
            batched gradients using :func:`tensorplay.vmap`. Defaults to False.
        check_fwd_over_rev (bool, optional): if True, check the second derivative
            with forward-mode AD over the first-order backward. Defaults to False.
        check_rev_over_rev (bool, optional): if True, check the second derivative
            with reverse-mode AD over the first-order backward. Defaults to True.
        fast_mode (bool, optional): if True, run a faster implementation of gradgradcheck that
            no longer computes the entire jacobian.
        masked (bool, optional): if True, the gradients of unspecified elements of
            sparse tensors are ignored (default, False).
    Returns:
        True if all differences satisfy allclose condition
    """
    if not (check_fwd_over_rev or check_rev_over_rev):
        raise AssertionError("Expected at least one of check_fwd_over_rev or check_rev_over_rev to be True")
    if check_undefined_grad and not check_rev_over_rev:
        raise AssertionError("Setting check_undefined_grad=True requires check_rev_over_rev to be True")
    if check_batched_grad and not check_rev_over_rev:
        raise AssertionError("Setting check_batched_grad=True requires check_rev_over_rev to be True")
    tupled_inputs = _as_tuple(inputs)

    if grad_outputs is None:
        # If grad_outputs is not specified, create random Tensors of the same shape, type, and device as the outputs
        outputs = _differentiable_outputs(func(*tupled_inputs))
        tupled_grad_outputs = tuple(
            tensorplay.testing.make_tensor(
                tuple(x.shape),
                dtype=x.dtype if x.is_floating_point() or x.is_complex() else tensorplay.float64,
                device=x.device,
                low=-1,
                high=1,
                requires_grad=True,
                noncontiguous=gen_non_contig_grad_outputs,
            )
            for x in outputs
        )
    else:
        tupled_grad_outputs = _as_tuple(grad_outputs)

    num_outputs = len(tupled_grad_outputs)

    # NB: We need to save the requires_grad information about the inputs here because gradcheck detaches inputs
    #     before running forward mode AD
    diff_input_args_indices = {i for i, x in enumerate(tupled_inputs) if _is_tensor_like(x) and x.requires_grad}
    diff_grad_output_indices = {i for i, x in enumerate(tupled_grad_outputs) if x.requires_grad}

    def new_func(*args):
        # Restore the requires_grad information
        input_args = tuple(
            x.requires_grad_() if i in diff_input_args_indices else x for i, x in enumerate(args[:-num_outputs])
        )
        outputs = _differentiable_outputs(func(*input_args))
        grad_outputs = tuple(
            x.requires_grad_() if i in diff_grad_output_indices else x for i, x in enumerate(args[-num_outputs:])
        )
        diff_input_args = tuple(x for i, x in enumerate(input_args) if i in diff_input_args_indices)
        grad_inputs = tensorplay.autograd.grad(
            outputs, diff_input_args, grad_outputs, create_graph=True, allow_unused=True
        )
        grad_inputs = tuple(g for g in grad_inputs if g is not None)
        return grad_inputs

    return gradcheck(
        new_func,
        tupled_inputs + tupled_grad_outputs,
        eps=eps,
        atol=atol,
        rtol=rtol,
        raise_exception=raise_exception,
        nondet_tol=nondet_tol,
        check_undefined_grad=check_undefined_grad,
        check_grad_dtypes=check_grad_dtypes,
        check_batched_grad=check_batched_grad,
        fast_mode=fast_mode,
        check_forward_ad=check_fwd_over_rev,
        check_backward_ad=check_rev_over_rev,
        masked=masked,
    )
