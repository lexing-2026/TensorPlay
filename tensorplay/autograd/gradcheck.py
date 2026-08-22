# mypy: allow-untyped-defs
r"""Gradient checking via finite differences.

Ported from ``torch/autograd/gradcheck.py`` (slow mode). Features that rely on
machinery this engine does not have yet -- forward-mode AD, vmap/batched
gradients, sparse/mkldnn layouts and fast_mode -- raise
:class:`NotImplementedError` with an explicit message.
"""

import collections
import functools
import warnings
from itertools import product

import tensorplay


# Note: `get_*_jacobian` functions are added here even though we didn't intend to make them public
# since they have been exposed from before we added `__all__`  and we already maintain BC for them
__all__ = [
    "gradcheck",
    "gradgradcheck",
    "GradcheckError",
    "get_numerical_jacobian",
    "get_analytical_jacobian",
    "get_numerical_jacobian_wrt_specific_input",
]


def _is_tensor_like(x):
    return isinstance(x, tensorplay.Tensor)


def _is_sparse_any_tensor(obj):
    # This engine has no sparse layouts yet.
    return False


def _is_float_or_complex_tensor(obj):
    return _is_tensor_like(obj) and (obj.is_floating_point() or obj.is_complex())


def _promote_types(a, b):
    # Minimal dtype promotion over the float types relevant to gradcheck.
    order = [
        tensorplay.float16,
        getattr(tensorplay._C.DType, "bfloat16", tensorplay.float16),
        tensorplay.float32,
        tensorplay.float64,
    ]
    ia = order.index(a) if a in order else len(order) - 1
    ib = order.index(b) if b in order else len(order) - 1
    return order[max(ia, ib)]


class GradcheckError(RuntimeError):
    r"""Error raised by :func:`gradcheck` and :func:`gradgradcheck`."""


class _UndefinedGrad(tensorplay.autograd.Function):
    # Port of torch._C._functions.UndefinedGrad: passes the input through in
    # the forward but makes backward *ignore* whatever gradient it receives
    # and propagate undefined (None) grads instead. Used by gradcheck to test
    # that functions handle undefined output gradients.
    @staticmethod
    def forward(ctx, inp):
        ctx.set_materialize_grads(False)
        return inp.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return None


def _allocate_jacobians_with_inputs(input_tensors, numel_output):
    # Makes zero-filled tensors from inputs. If `numel_output` is not None, for
    # each tensor in `input_tensors`, returns a new zero-filled tensor with height
    # of `t.numel` and width of `numel_output`. Otherwise, for each tensor, returns
    # a 1-d tensor with size `(t.numel,)`.
    out: list[tensorplay.Tensor] = []
    for t in input_tensors:
        if _is_float_or_complex_tensor(t) and t.requires_grad:
            out.append(tensorplay.zeros((t.numel(), numel_output), dtype=t.dtype, device=t.device))
    return tuple(out)


def _allocate_jacobians_with_outputs(output_tensors, numel_input, dtype=None, device=None):
    # Makes zero-filled tensors from outputs. If `dim` is not None, for each tensor
    # in `output_tensors`, returns a new zero-filled tensor with height of `dim` and
    # width of `t.numel`. Otherwise, for each tensor, returns a 1-d tensor with size
    # (t.numel,).
    out: list[tensorplay.Tensor] = []
    for t in output_tensors:
        if _is_float_or_complex_tensor(t):
            kwargs = {}
            kwargs["dtype"] = dtype if dtype is not None else t.dtype
            kwargs["device"] = device if device is not None else t.device
            out.append(tensorplay.zeros((numel_input, t.numel()), **kwargs))
    return tuple(out)


def _iter_tensors(x, only_requiring_grad: bool = False):
    if _is_tensor_like(x):
        if x.requires_grad or not only_requiring_grad:
            yield x
    elif isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        for elem in x:
            yield from _iter_tensors(elem, only_requiring_grad)


def _densify(x):
    # No sparse layouts in this engine: identity.
    if isinstance(x, (list, tuple)):
        return type(x)(map(_densify, x))
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


def _get_numerical_jacobian(fn, inputs, outputs=None, target=None, eps=1e-3):
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

    Returns:
        A list of M N-tuples of tensors

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    jacobians: list[tuple[tensorplay.Tensor, ...]] = []
    if outputs is None:
        outputs = _as_tuple(fn(*_as_tuple(inputs)))
    if any(o.is_complex() for o in outputs):
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
                fn, inp_idx, inputs, outputs, eps, input=inp
            )
        ]
    return jacobians


def get_numerical_jacobian(fn, inputs, target=None, eps=1e-3, grad_out=1.0):
    """Compute the numerical Jacobian for a given fn and its inputs.

    .. warning::
        ``get_numerical_jacobian`` was part of PyTorch's private API and is kept
        only for backward compatibility.

    Args:
        fn: the function to compute the Jacobian for (must take inputs as a tuple)
        inputs: input to `fn`
        target: the Tensors wrt whom Jacobians are calculated (default=`input`)
        eps: the magnitude of the perturbation during finite differencing
             (default=`1e-3`)
        grad_out: defaults to 1.0.

    Returns:
        A list of Jacobians of `fn` (restricted to its first output) with respect to
        each input or target, if provided.

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    warnings.warn(
        "`get_numerical_jacobian` was part of PyTorch's private API and not "
        "meant to be exposed. We are deprecating it and it will be removed "
        "in a future version.",
        FutureWarning,
        stacklevel=2,
    )
    if grad_out != 1.0:  # grad_out param is only kept for backward compatibility reasons
        raise ValueError(
            "Expected grad_out to be 1.0. get_numerical_jacobian no longer "
            "supports values of grad_out != 1.0."
        )

    def fn_pack_inps(*inps):
        return fn(inps)

    jacobians = _get_numerical_jacobian(fn_pack_inps, inputs, None, target, eps)

    return tuple(jacobian_for_each_output[0] for jacobian_for_each_output in jacobians)


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
    jvps: list[tensorplay.Tensor] = []
    ds_dx_tup = jvp_fn(delta[0] if isinstance(delta, tuple) else delta)
    for ds_dx in ds_dx_tup:  # R -> R
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


def _prepare_input(input, maybe_perturbed_input=None, fast_mode=False):
    # Prepares the inputs to be passed into the function while including the new
    # modified input. Dense tensors are passed as-is: we cannot use `entry`
    # (`input.data`) when we want gradgrad to work because `fn` (in the
    # gradgrad case) needs to compute grad wrt the original input object.
    return input


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


def get_numerical_jacobian_wrt_specific_input(fn, input_idx, inputs, outputs, eps, input=None) -> tuple:
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
        jacobian_cols[d_idx] = _compute_numerical_jvps_wrt_specific_input(jvp_fn, eps, x.is_complex())
    return _combine_jacobian_cols(jacobian_cols, outputs, input, input.numel())


def _get_input_to_perturb(input):
    # Prepare the input so that it can be modified in-place. For dense
    # tensors use .data to bypass the version counter.
    return input.data


def _with_prepare_inputs(fn, inputs, input_idx, input_to_perturb, fast_mode=False):
    # Wraps `fn` so that its inputs are already supplied
    def wrapped_fn():
        inp = tuple(
            input_to_perturb if i == input_idx else a
            for i, a in enumerate(_as_tuple(inputs))
        )
        return tuple(a.clone() for a in _as_tuple(fn(*inp)))

    return wrapped_fn


def _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn):
    # Wraps jvp_fn so that certain arguments are already supplied
    def jvp_fn(delta):
        return _compute_numerical_gradient(wrapped_fn, input_to_perturb, delta, eps, nbhd_checks_fn)

    return jvp_fn


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
                dense = tensor.reshape(-1)
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

If you are adding a new operator and the test
- manually invokes gradcheck/gradgradcheck, then call gradcheck/gradgradcheck
  with `nondet_tol=<tol>` as a keyword argument.
"""



def _check_analytical_jacobian_attributes(inputs, output, nondet_tol, check_grad_dtypes) -> tuple:
    # This computes the analytical jacobian rows for the given output:
    # vjps[i][j] is the jth row of the Jacobian wrt the ith input.
    diff_input_list = list(_iter_tensors(inputs, True))

    def vjp_fn(grad_output):
        return tensorplay.autograd.grad(
            output, diff_input_list, grad_output, retain_graph=True, allow_unused=True
        )

    # Compute everything twice to check for nondeterminism (which we call reentrancy)
    vjps1 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
    vjps2 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())

    output_numel = output.numel()
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


def get_analytical_jacobian(inputs, output, nondet_tol=0.0, grad_out=1.0):
    # Replicates the behavior of the old get_analytical_jacobian before the refactor
    # This shares much of its code with _check_analytical_jacobian_attributes
    warnings.warn(
        "`get_analytical_jacobian` was part of PyTorch's private API and not "
        "meant to be exposed. We are deprecating it and it will be removed "
        "in a future version.",
        FutureWarning,
        stacklevel=2,
    )
    if grad_out != 1.0:  # grad_out param is only kept for backward compatibility reasons
        raise ValueError(
            "Expected grad_out to be 1.0. get_analytical_jacobian no longer "
            "supports values of grad_out != 1.0."
        )
    if output.is_complex():
        raise ValueError(
            "Expected output to be non-complex. get_analytical_jacobian no "
            "longer supports functions that return complex outputs."
        )
    diff_input_list = list(_iter_tensors(inputs, True))

    def vjp_fn(grad_output):
        return tensorplay.autograd.grad(
            output, diff_input_list, grad_output, retain_graph=True, allow_unused=True
        )

    # Compute everything twice to check for nondeterminism (which we call reentrancy)
    vjps1 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
    vjps2 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())

    output_numel = output.numel()
    jacobians1, types_ok, sizes_ok = _stack_and_check_tensors(vjps1, inputs, output_numel)
    jacobians2, _, _ = _stack_and_check_tensors(vjps2, inputs, output_numel)
    reentrant = _check_jacobians_equal(jacobians1, jacobians2, nondet_tol)

    return jacobians1, reentrant, sizes_ok, types_ok


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


def _check_inputs(tupled_inputs) -> bool:
    # Make sure that gradients are saved for at least one input
    any_input_requiring_grad = False
    complex128 = getattr(tensorplay._C.DType, "complex128", None)
    for idx, inp in enumerate(tupled_inputs):
        if _is_tensor_like(inp) and inp.requires_grad:
            if not (inp.dtype == tensorplay.float64 or (complex128 is not None and inp.dtype == complex128)):
                warnings.warn(
                    f"Input #{idx} requires gradient and "
                    "is not a double precision floating point or complex. "
                    "This check will likely fail if all the inputs are "
                    "not of double precision floating point or complex. ",
                    stacklevel=2,
                )
            content = inp
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
    pass  # No sparse/mkldnn layouts in this engine.


def _check_no_differentiable_outputs(func, inputs, func_out, eps) -> bool:
    # When there are no differentiable outputs, numerical gradient for a function is
    # expected to be zero.
    jacobians_all_inputs_outputs = _get_numerical_jacobian(func, inputs, func_out, eps=eps)
    for jacobians_all_outputs_and_fixed_input in jacobians_all_inputs_outputs:
        for jacobian in jacobians_all_outputs_and_fixed_input:
            if (jacobian != 0).sum().item() > 0:
                raise GradcheckError("Numerical gradient for function expected to be zero")
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
        if not gi.eq(0).all().item():
            raise GradcheckError("backward not multiplied by grad_output")
        if gi.dtype != di.dtype:
            raise GradcheckError("grad is incorrect type")
        if gi.device != di.device:
            raise GradcheckError("grad is incorrect device")
        if tuple(gi.size()) != tuple(di.size()):
            raise GradcheckError("grad is incorrect size")
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
            _UndefinedGrad()(o)
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
                    _UndefinedGrad()(o) if idx == undef_grad_idx else o
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


def _allclose_with_type_promotion(a, b, rtol, atol):
    promoted_type = _promote_types(a.dtype, b.dtype)
    a = a.to(dtype=promoted_type)
    b = b.to(dtype=promoted_type)
    return tensorplay.allclose(a, b, rtol, atol)


def _get_notallclose_msg(analytical, numerical, output_idx, input_idx) -> str:
    return (
        f"Jacobian mismatch for output {output_idx:d} with respect to input {input_idx:d},\n"
        f"numerical:{numerical}\nanalytical:{analytical}\n"
    )


def _transpose(matrix_of_tensors):
    # returns list of tuples
    return list(zip(*matrix_of_tensors))


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
):
    func_out = _as_tuple(func_out)
    if not outputs:
        return _check_no_differentiable_outputs(func, tupled_inputs, func_out, eps=eps)
    tupled_inputs_numerical = _densify(tupled_inputs)

    numerical = _transpose(
        _get_numerical_jacobian(func, tupled_inputs_numerical, func_out, eps=eps)
    )
    # Note: [numerical vs analytical output length]
    # The numerical path returns jacobian quantity for all outputs, even if requires_grad of that
    # output is False. This behavior is necessary for _check_no_differentiable_outputs to work.
    numerical = [nj for o, nj in zip(func_out, numerical) if o.requires_grad]
    for i, o in enumerate(outputs):
        analytical = _check_analytical_jacobian_attributes(tupled_inputs, o, nondet_tol, check_grad_dtypes)

        for j, (a, n) in enumerate(zip(analytical, numerical[i])):
            if not _allclose_with_type_promotion(a, n.to(a.device), rtol, atol):
                raise GradcheckError(_get_notallclose_msg(a, n, i, j))

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
        check_batched_grad (bool, optional): Not supported by this engine yet;
            ``True`` raises :class:`NotImplementedError`. Defaults to False.
        check_batched_forward_grad (bool, optional): Requires forward AD; ``True``
            raises :class:`NotImplementedError`. Defaults to False.
        check_forward_ad (bool, optional): Requires forward AD; ``True`` raises
            :class:`NotImplementedError`. Defaults to False.
        check_backward_ad (bool, optional): if ``False``, do not perform any checks that rely on
            backward mode AD to be implemented. Defaults to ``True``.
        fast_mode (bool, optional): Only the slow implementation exists in this
            engine; ``True`` raises :class:`NotImplementedError`. Defaults to False.
        masked (bool, optional): Kept for signature parity with torch; has no
            effect since this engine has no sparse layouts. Defaults to ``False``.
    Returns:
        ``True`` if all differences satisfy allclose condition

    """
    if check_forward_ad or check_batched_forward_grad or check_batched_grad or fast_mode:
        raise NotImplementedError(
            "tensorplay.autograd.gradcheck: forward-mode AD, batched gradients "
            "(vmap) and fast_mode are not supported by this engine yet; keep "
            "`check_forward_ad=False`, `check_batched_grad=False`, "
            "`check_batched_forward_grad=False` and `fast_mode=False`."
        )
    if not check_backward_ad:
        raise AssertionError(
            "Expected at least one of check_forward_ad or check_backward_ad to be True"
        )
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

    if any(isinstance(t, tensorplay.Tensor) and t.is_complex() for t in (*tupled_inputs, *outputs)):
        raise NotImplementedError(
            "tensorplay.autograd.gradcheck: complex tensor support is not "
            "available in this engine yet; gradcheck only supports real-valued "
            "inputs and outputs."
        )

    _slow_gradcheck(
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

    for i, o in enumerate(outputs):
        pass  # check_batched_grad unsupported (rejected above)

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

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        grad_outputs (tuple of [Tensor or None] or Tensor, optional): The gradients with
            respect to the function's outputs.
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        gen_non_contig_grad_outputs (bool, optional): Not supported by this
            engine yet; ``True`` raises :class:`NotImplementedError`.
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
        check_batched_grad (bool, optional): Not supported by this engine yet.
        fast_mode (bool, optional): Not supported by this engine yet.
        masked (bool, optional): Kept for signature parity with torch.
    Returns:
        True if all differences satisfy allclose condition
    """
    if gen_non_contig_grad_outputs:
        raise NotImplementedError(
            "tensorplay.autograd.gradgradcheck: `gen_non_contig_grad_outputs="
            "True` is not supported by this engine yet."
        )
    tupled_inputs = _as_tuple(inputs)

    if grad_outputs is None:
        # If grad_outputs is not specified, create random Tensors of the same shape, type, and device as the outputs
        outputs = _differentiable_outputs(func(*tupled_inputs))
        tupled_grad_outputs = tuple(
            tensorplay.rand(
                tuple(x.shape),
                dtype=x.dtype if x.is_floating_point() else tensorplay.float64,
                device=x.device,
                requires_grad=True,
            ) * 2 - 1
            for x in outputs
        )
    else:
        tupled_grad_outputs = _as_tuple(grad_outputs)

    num_outputs = len(tupled_grad_outputs)

    # NB: We need to save the requires_grad information about the inputs here because gradcheck detaches inputs
    #     before running forward mode AD
    diff_input_args_indices = {
        i for i, x in enumerate(tupled_inputs) if _is_tensor_like(x) and x.requires_grad
    }
    diff_grad_output_indices = {i for i, x in enumerate(tupled_grad_outputs) if x.requires_grad}

    def new_func(*args):
        # Restore the requires_grad information
        input_args = tuple(
            x.requires_grad_() if i in diff_input_args_indices else x
            for i, x in enumerate(args[:-num_outputs])
        )
        outputs = _differentiable_outputs(func(*input_args))
        grad_outputs = tuple(
            x.requires_grad_() if i in diff_grad_output_indices else x
            for i, x in enumerate(args[-num_outputs:])
        )
        diff_input_args = tuple(
            x for i, x in enumerate(input_args) if i in diff_input_args_indices
        )
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
