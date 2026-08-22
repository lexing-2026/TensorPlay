# mypy: allow-untyped-defs
r"""Functional utilities for computing higher-order derivatives.

Ported from ``torch/autograd/functional.py``. The engine has no forward-mode
AD or vmap support yet, so ``vectorize=True`` and ``strategy="forward-mode"``
raise ``NotImplementedError``; everything else follows torch's semantics.
"""

import tensorplay


__all__ = ["vjp", "jvp", "jacobian", "hessian", "hvp", "vhp"]

# Utility functions


def _as_tuple_nocheck(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)


def _as_tuple(inp, arg_name=None, fn_name=None):
    # Ensures that inp is a tuple of Tensors
    # Returns whether or not the original inp was a tuple and the tupled version of the input
    if arg_name is None and fn_name is None:
        return _as_tuple_nocheck(inp)

    is_inp_tuple = True
    if not isinstance(inp, tuple):
        inp = (inp,)
        is_inp_tuple = False

    for i, el in enumerate(inp):
        if not isinstance(el, tensorplay.Tensor):
            if is_inp_tuple:
                raise TypeError(
                    f"The {arg_name} given to {fn_name} must be either a Tensor or a tuple of Tensors but the"
                    f" value at index {i} has type {type(el)}."
                )
            else:
                raise TypeError(
                    f"The {arg_name} given to {fn_name} must be either a Tensor or a tuple of Tensors but the"
                    f" given {arg_name} has type {type(el)}."
                )

    return is_inp_tuple, inp


def _tuple_postprocess(res, to_unpack):
    # Unpacks a potentially nested tuple of Tensors
    # to_unpack should be a single boolean or a tuple of two booleans.
    # It is used to:
    # - invert _as_tuple when res should match the inp given to _as_tuple
    # - optionally remove nesting of two tuples created by multiple calls to _as_tuple
    if isinstance(to_unpack, tuple):
        if len(to_unpack) != 2:
            raise AssertionError("Expected to_unpack tuple to have exactly 2 elements")
        if not to_unpack[1]:
            res = tuple(el[0] for el in res)
        if not to_unpack[0]:
            res = res[0]
    else:
        if not to_unpack:
            res = res[0]
    return res


def _grad_preprocess(inputs, create_graph, need_graph):
    # Preprocess the inputs to make sure they require gradient
    # inputs is a tuple of Tensors to preprocess
    # create_graph specifies if the user wants gradients to flow back to the Tensors in inputs
    # need_graph specifies if we internally want gradients to flow back to the Tensors in res
    # Note that we *always* create a new Tensor object to be able to see the difference between
    # inputs given as arguments and the same Tensors automatically captured by the user function.
    res = []
    for inp in inputs:
        if create_graph and inp.requires_grad:
            # Create at least a new Tensor object in a differentiable way
            # Use .view() to get a shallow copy
            res.append(inp.view(tuple(inp.shape)))
        else:
            res.append(inp.detach().requires_grad_(need_graph))
    return tuple(res)


def _grad_postprocess(inputs, create_graph):
    # Postprocess the generated Tensors to avoid returning Tensors with history when the user did not
    # request it.
    if isinstance(inputs[0], tensorplay.Tensor):
        if not create_graph:
            return tuple(inp.detach() for inp in inputs)
        else:
            return inputs
    else:
        return tuple(_grad_postprocess(inp, create_graph) for inp in inputs)


def _validate_v(v, other, is_other_tuple):
    # This assumes that other is the correct shape, and v should match
    # Both are assumed to be tuples of Tensors
    if len(other) != len(v):
        if is_other_tuple:
            raise RuntimeError(
                f"v is a tuple of invalid length: should be {len(other)} but got {len(v)}."
            )
        else:
            raise RuntimeError("The given v should contain a single Tensor.")

    for idx, (el_v, el_other) in enumerate(zip(v, other)):
        if el_v.size() != el_other.size():
            prepend = ""
            if is_other_tuple:
                prepend = f"Entry {idx} in "
            raise RuntimeError(
                f"{prepend}v has invalid size: should be {el_other.size()} but got {el_v.size()}."
            )


def _check_requires_grad(inputs, input_type, strict):
    # Used to make all the necessary checks to raise nice errors in strict mode.
    if not strict:
        return

    if input_type not in ["outputs", "grad_inputs", "jacobian", "hessian"]:
        raise RuntimeError("Invalid input_type to _check_requires_grad")
    for i, inp in enumerate(inputs):
        if inp is None:
            # This can only be reached for grad_inputs.
            raise RuntimeError(
                f"The output of the user-provided function is independent of input {i}."
                " This is not allowed in strict mode."
            )
        if not inp.requires_grad:
            if input_type == "hessian":
                raise RuntimeError(
                    f"The hessian of the user-provided function with respect to input {i}"
                    " is independent of the input. This is not allowed in strict mode."
                    " You should ensure that your function is thrice differentiable and that"
                    " the hessian depends on the inputs."
                )
            elif input_type == "jacobian":
                raise RuntimeError(
                    "While computing the hessian, found that the jacobian of the user-provided"
                    f" function with respect to input {i} is independent of the input. This is not"
                    " allowed in strict mode. You should ensure that your function is twice"
                    " differentiable and that the jacobian depends on the inputs (this would be"
                    " violated by a linear function for example)."
                )
            elif input_type == "grad_inputs":
                raise RuntimeError(
                    f"The gradient with respect to input {i} is independent of the inputs of the"
                    " user-provided function. This is not allowed in strict mode."
                )
            else:
                raise RuntimeError(
                    f"Output {i} of the user-provided function does not require gradients."
                    " The outputs must be computed in a differentiable manner from the input"
                    " when running in strict mode."
                )


def _autograd_grad(
    outputs,
    inputs,
    grad_outputs=None,
    create_graph=False,
    retain_graph=None,
):
    # Version of autograd.grad that accepts `None` in outputs and do not compute gradients for them.
    # This has the extra constraint that inputs has to be a tuple
    if not isinstance(outputs, tuple):
        raise AssertionError("Expected outputs to be a tuple")
    if grad_outputs is None:
        grad_outputs = (None,) * len(outputs)
    if not isinstance(grad_outputs, tuple):
        raise AssertionError("Expected grad_outputs to be a tuple")
    if len(outputs) != len(grad_outputs):
        raise AssertionError(
            f"Expected outputs and grad_outputs to have the same length, "
            f"but got {len(outputs)} and {len(grad_outputs)}"
        )

    new_outputs: tuple[tensorplay.Tensor, ...] = ()
    new_grad_outputs: tuple[tensorplay.Tensor | None, ...] = ()
    for out, grad_out in zip(outputs, grad_outputs):
        if out is not None and out.requires_grad:
            new_outputs += (out,)
            new_grad_outputs += (grad_out,)

    if len(new_outputs) == 0:
        # No differentiable output, we don't need to call the autograd engine
        return (None,) * len(inputs)
    else:
        return tensorplay.autograd.grad(
            new_outputs,
            inputs,
            new_grad_outputs,
            allow_unused=True,
            create_graph=create_graph,
            retain_graph=retain_graph,
        )


def _fill_in_zeros(grads, refs, strict, create_graph, stage):
    # Used to detect None in the grads and depending on the flags, either replace them
    # with Tensors full of 0s of the appropriate size based on the refs or raise an error.
    # strict and create graph allow us to detect when it is appropriate to raise an error
    # stage gives us information of which backward call we consider to give good error message
    if stage not in ["back", "back_trick", "double_back", "double_back_trick"]:
        raise RuntimeError(f"Invalid stage argument '{stage}' to _fill_in_zeros")

    res: tuple[tensorplay.Tensor, ...] = ()
    for i, grads_i in enumerate(grads):
        if grads_i is None:
            if strict:
                if stage == "back":
                    raise RuntimeError(
                        "The output of the user-provided function is independent of "
                        f"input {i}. This is not allowed in strict mode."
                    )
                elif stage == "back_trick":
                    raise RuntimeError(
                        f"The gradient with respect to the input is independent of entry {i}"
                        " in the grad_outputs when using the double backward trick to compute"
                        " forward mode gradients. This is not allowed in strict mode."
                    )
                elif stage == "double_back":
                    raise RuntimeError(
                        "The jacobian of the user-provided function is independent of "
                        f"input {i}. This is not allowed in strict mode."
                    )
                else:
                    raise RuntimeError(
                        "The hessian of the user-provided function is independent of "
                        f"entry {i} in the grad_jacobian. This is not allowed in strict "
                        "mode as it prevents from using the double backward trick to "
                        "replace forward mode AD."
                    )

            grads_i = tensorplay.zeros_like(refs[i])
        else:
            if strict and create_graph and not grads_i.requires_grad:
                if "double" not in stage:
                    raise RuntimeError(
                        "The jacobian of the user-provided function is independent of "
                        f"input {i}. This is not allowed in strict mode when create_graph=True."
                    )
                else:
                    raise RuntimeError(
                        "The hessian of the user-provided function is independent of "
                        f"input {i}. This is not allowed in strict mode when create_graph=True."
                    )

        res += (grads_i,)

    return res


# Public API


def vjp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Compute the dot product between a vector ``v`` and the Jacobian of the given function at the point given by the inputs.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the vector
            Jacobian product is computed.  Must be the same size as the output
            of ``func``. This argument is optional when the output of ``func``
            contains a single element and (if it is not provided) will be set
            as a Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result
            will be computed in a differentiable way. Note that when ``strict``
            is ``False``, the result can not require gradients or be
            disconnected from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            vjp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            vjpval (tuple of Tensors or Tensor): result of the dot product with
            the same shape as the inputs.

    Example:

        >>> def exp_reducer(x):
        ...     return x.exp().sum(dim=1)
        >>> inputs = tensorplay.rand(4, 4)
        >>> v = tensorplay.ones(4)
        >>> vjp(exp_reducer, inputs, v)
        (tensor([5.7817, 7.2458, 5.7830, 6.7782]),
         tensor([[1.4458, 1.3962, 1.3042, 1.6354],
                [2.1288, 1.0652, 1.5483, 2.5035],
                [2.2046, 1.1292, 1.1432, 1.3059],
                [1.3225, 1.6652, 1.7753, 2.0152]]))
    """
    with tensorplay.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "vjp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "vjp"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)

        if v is not None:
            _, v = _as_tuple(v, "v", "vjp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, outputs, is_outputs_tuple)
        else:
            if len(outputs) != 1 or outputs[0].numel() != 1:
                raise RuntimeError(
                    "The vector v can only be None if the "
                    "user-provided function returns "
                    "a single Tensor with a single element."
                )

    enable_grad = True if create_graph else tensorplay.is_grad_enabled()
    with tensorplay.set_grad_enabled(enable_grad):
        grad_res = _autograd_grad(outputs, inputs, v, create_graph=create_graph)
        vjpval = _fill_in_zeros(grad_res, inputs, strict, create_graph, "back")

    # Cleanup objects and return them to the user
    outputs = _grad_postprocess(outputs, create_graph)
    vjpval = _grad_postprocess(vjpval, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        vjpval, is_inputs_tuple
    )


def jvp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Compute the dot product between the Jacobian of the given function at the point given by the inputs and a vector ``v``.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the Jacobian
            vector product is computed. Must be the same size as the input of
            ``func``. This argument is optional when the input to ``func``
            contains a single element and (if it is not provided) will be set
            as a Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result
            will be computed in a differentiable way. Note that when ``strict``
            is ``False``, the result can not require gradients or be
            disconnected from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            jvp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            jvp (tuple of Tensors or Tensor): result of the dot product with
            the same shape as the output.

    Note:
        ``autograd.functional.jvp`` computes the jvp by using the backward of
        the backward (sometimes called the double backwards trick). This is not
        the most performant way of computing the jvp.

    Example:

        >>> def exp_reducer(x):
        ...     return x.exp().sum(dim=1)
        >>> inputs = tensorplay.rand(4, 4)
        >>> v = tensorplay.ones(4, 4)
        >>> jvp(exp_reducer, inputs, v)
        (tensor([6.3090, 4.6742, 7.9114, 8.2106]),
         tensor([6.3090, 4.6742, 7.9114, 8.2106]))

        >>> def adder(x, y):
        ...     return 2 * x + 3 * y
        >>> inputs = (tensorplay.rand(2), tensorplay.rand(2))
        >>> v = (tensorplay.ones(2), tensorplay.ones(2))
        >>> jvp(adder, inputs, v)
        (tensor([2.2399, 2.5005]),
         tensor([5., 5.]))

    """
    with tensorplay.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jvp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        if v is not None:
            _, v = _as_tuple(v, "v", "jvp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        else:
            if len(inputs) != 1 or inputs[0].numel() != 1:
                raise RuntimeError(
                    "The vector v can only be None if the input to "
                    "the user-provided function is a single Tensor "
                    "with a single element."
                )

        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "jvp"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)
        # The backward is linear so the value of grad_outputs is not important as
        # it won't appear in the double backward graph. We only need to ensure that
        # it does not contain inf or nan.
        grad_outputs = tuple(
            tensorplay.zeros_like(out, requires_grad=True) for out in outputs
        )

        grad_inputs = _autograd_grad(outputs, inputs, grad_outputs, create_graph=True)
        _check_requires_grad(grad_inputs, "grad_inputs", strict=strict)

    if create_graph:
        with tensorplay.enable_grad():
            grad_res = _autograd_grad(
                grad_inputs, grad_outputs, v, create_graph=create_graph
            )
            jvp_val = _fill_in_zeros(grad_res, outputs, strict, create_graph, "back_trick")
    else:
        grad_res = _autograd_grad(
            grad_inputs, grad_outputs, v, create_graph=create_graph
        )
        jvp_val = _fill_in_zeros(grad_res, outputs, strict, create_graph, "back_trick")

    # Cleanup objects and return them to the user
    outputs = _grad_postprocess(outputs, create_graph)
    jvp_val = _grad_postprocess(jvp_val, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        jvp_val, is_outputs_tuple
    )


def jacobian(
    func,
    inputs,
    create_graph=False,
    strict=False,
    vectorize=False,
    strategy="reverse-mode",
):
    r"""Compute the Jacobian of a given function.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        create_graph (bool, optional): If ``True``, the Jacobian will be
            computed in a differentiable manner. Note that when ``strict`` is
            ``False``, the result can not require gradients or be disconnected
            from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            jacobian for said inputs, which is the expected mathematical value.
            Defaults to ``False``.
        vectorize (bool, optional): Not supported by this engine yet; passing
            ``True`` raises :class:`NotImplementedError`.
        strategy (str, optional): Set to ``"reverse-mode"`` (default) or
            ``"forward-mode"``. Forward-mode AD is not supported by this engine
            yet; passing it raises :class:`NotImplementedError`.

    Returns:
        Jacobian (Tensor or nested tuple of Tensors): if there is a single
        input and output, this will be a single Tensor containing the
        Jacobian for the linearized inputs and output. If one of the two is
        a tuple, then the Jacobian will be a tuple of Tensors. If both of
        them are tuples, then the Jacobian will be a tuple of tuple of
        Tensors where ``Jacobian[i][j]`` will contain the Jacobian of the
        ``i``\th output and ``j``\th input and will have as size the
        concatenation of the sizes of the corresponding output and the
        corresponding input and will have same dtype and device as the
        corresponding input.

    Example:

        >>> def exp_reducer(x):
        ...     return x.exp().sum(dim=1)
        >>> inputs = tensorplay.rand(2, 2)
        >>> jacobian(exp_reducer, inputs)
        tensor([[[1.4917, 2.4352],
                 [0.0000, 0.0000]],
                [[0.0000, 0.0000],
                 [2.4369, 2.3799]]])
    """
    if strategy == "forward-mode":
        raise NotImplementedError(
            "tensorplay.autograd.functional.jacobian: `strategy='forward-mode'` "
            "requires forward-mode AD, which this engine does not support yet. "
            'Please use `strategy="reverse-mode"`.'
        )
    if strategy != "reverse-mode":
        raise AssertionError(
            'Expected strategy to be either "forward-mode" or "reverse-mode". Hint: If your '
            'function has more outputs than inputs, "forward-mode" tends to be more performant. '
            'Otherwise, prefer to use "reverse-mode".'
        )
    if vectorize:
        # Requires the vmap prototype feature (is_grads_batched); not available
        # in this engine yet.
        raise NotImplementedError(
            "tensorplay.autograd.functional.jacobian: `vectorize=True` requires "
            "batched gradients (vmap), which this engine does not support yet. "
            "Please use `vectorize=False`."
        )

    with tensorplay.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jacobian")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "jacobian"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)

        jacobian_result: tuple[tensorplay.Tensor, ...] = ()

        for i, out in enumerate(outputs):
            jac_i: tuple[list[tensorplay.Tensor]] = tuple([] for _ in range(len(inputs)))
            for j in range(out.numel()):
                vj = _autograd_grad(
                    (out.reshape(-1)[j],),
                    inputs,
                    retain_graph=True,
                    create_graph=create_graph,
                )

                for el_idx, (jac_i_el, vj_el, inp_el) in enumerate(
                    zip(jac_i, vj, inputs)
                ):
                    if vj_el is not None:
                        if strict and create_graph and not vj_el.requires_grad:
                            msg = (
                                "The jacobian of the user-provided function is "
                                f"independent of input {i}. This is not allowed in "
                                "strict mode when create_graph=True."
                            )
                            raise RuntimeError(msg)
                        jac_i_el.append(vj_el)
                    else:
                        if strict:
                            msg = (
                                f"Output {i} of the user-provided function is "
                                f"independent of input {el_idx}. This is not allowed in "
                                "strict mode."
                            )
                            raise RuntimeError(msg)
                        jac_i_el.append(tensorplay.zeros_like(inp_el))

            jacobian_result += (
                tuple(
                    tensorplay.stack(jac_i_el, dim=0).view(
                        tuple(out.size()) + tuple(inputs[el_idx].size())
                    )
                    for (el_idx, jac_i_el) in enumerate(jac_i)
                ),
            )

        jacobian_result = _grad_postprocess(jacobian_result, create_graph)

        return _tuple_postprocess(jacobian_result, (is_outputs_tuple, is_inputs_tuple))


def hessian(
    func,
    inputs,
    create_graph=False,
    strict=False,
    vectorize=False,
    outer_jacobian_strategy="reverse-mode",
):
    r"""Compute the Hessian of a given scalar function.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        create_graph (bool, optional): If ``True``, the Hessian will be computed in
            a differentiable manner. Note that when ``strict`` is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return a Tensor of zeros as the
            hessian for said inputs, which is the expected mathematical value.
            Defaults to ``False``.
        vectorize (bool, optional): Not supported by this engine yet; passing
            ``True`` raises :class:`NotImplementedError`.
        outer_jacobian_strategy (str, optional): Only ``"reverse-mode"`` is
            supported; forward-mode AD raises :class:`NotImplementedError`.

    Returns:
        Hessian (Tensor or a tuple of tuple of Tensors): if there is a single input,
        this will be a single Tensor containing the Hessian for the input.
        If it is a tuple, then the Hessian will be a tuple of tuples where
        ``Hessian[i][j]`` will contain the Hessian of the ``i``\th input
        and ``j``\th input with size the sum of the size of the ``i``\th input plus
        the size of the ``j``\th input. ``Hessian[i][j]`` will have the same
        dtype and device as the corresponding ``i``\th input.

    Example:

        >>> def pow_reducer(x):
        ...     return x.pow(3).sum()
        >>> inputs = tensorplay.rand(2, 2)
        >>> hessian(pow_reducer, inputs)
        tensor([[[[5.2265, 0.0000],
                  [0.0000, 0.0000]],
                 [[0.0000, 4.8221],
                  [0.0000, 0.0000]]],
                [[[0.0000, 0.0000],
                  [1.9456, 0.0000]],
                 [[0.0000, 0.0000],
                  [0.0000, 3.2550]]]])
    """
    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "hessian")
    if outer_jacobian_strategy not in (
        "forward-mode",
        "reverse-mode",
    ):
        raise AssertionError(
            'Expected strategy to be either "forward-mode" or "reverse-mode".'
        )

    def ensure_single_output_function(*inp):
        out = func(*inp)
        is_out_tuple, t_out = _as_tuple(
            out, "outputs of the user-provided function", "hessian"
        )
        _check_requires_grad(t_out, "outputs", strict=strict)

        if is_out_tuple or not isinstance(out, tensorplay.Tensor):
            raise RuntimeError(
                "The function given to hessian should return a single Tensor"
            )

        if out.numel() != 1:
            raise RuntimeError(
                "The Tensor returned by the function given to hessian should contain a single element"
            )

        return out.squeeze()

    def jac_func(*inp):
        jac = jacobian(ensure_single_output_function, inp, create_graph=True)
        _check_requires_grad(jac, "jacobian", strict=strict)
        return jac

    res = jacobian(
        jac_func,
        inputs,
        create_graph=create_graph,
        strict=strict,
        vectorize=vectorize,
        strategy=outer_jacobian_strategy,
    )
    return _tuple_postprocess(res, (is_inputs_tuple, is_inputs_tuple))


def vhp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Compute the dot product between vector ``v`` and Hessian of a  given scalar function at a specified point.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the vector Hessian
            product is computed. Must be the same size as the input of
            ``func``. This argument is optional when ``func``'s input contains
            a single element and (if it is not provided) will be set as a
            Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result
            will be computed in a differentiable way. Note that when ``strict``
            is ``False``, the result can not require gradients or be
            disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            vhp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            vhp (tuple of Tensors or Tensor): result of the dot product with the
            same shape as the inputs.

    Example:

        >>> def pow_reducer(x):
        ...     return x.pow(3).sum()
        >>> inputs = tensorplay.rand(2, 2)
        >>> v = tensorplay.ones(2, 2)
        >>> output = vhp(pow_reducer, inputs, v)
        >>> output[0]
        tensor(0.5591)
        >>> output[1]
        tensor([[1.0689, 1.2431],
                [3.0989, 4.4456]])
    """
    with tensorplay.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "vhp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        if v is not None:
            _, v = _as_tuple(v, "v", "vhp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        else:
            if len(inputs) != 1 or inputs[0].numel() != 1:
                raise RuntimeError(
                    "The vector v can only be None if the input to the user-provided function "
                    "is a single Tensor with a single element."
                )
        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "vhp"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)

        if is_outputs_tuple or not isinstance(outputs[0], tensorplay.Tensor):
            raise RuntimeError(
                "The function given to vhp should return a single Tensor"
            )

        if outputs[0].numel() != 1:
            raise RuntimeError(
                "The Tensor returned by the function given to vhp should contain a single element"
            )

        jac = _autograd_grad(outputs, inputs, create_graph=True)
        _check_requires_grad(jac, "jacobian", strict=strict)

    enable_grad = True if create_graph else tensorplay.is_grad_enabled()
    with tensorplay.set_grad_enabled(enable_grad):
        grad_res = _autograd_grad(jac, inputs, v, create_graph=create_graph)
        vhp_val = _fill_in_zeros(grad_res, inputs, strict, create_graph, "double_back")

    outputs = _grad_postprocess(outputs, create_graph)
    vhp_val = _grad_postprocess(vhp_val, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        vhp_val, is_inputs_tuple
    )


def hvp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Compute the dot product between the scalar function's Hessian and a vector ``v`` at a specified point.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the Hessian vector
            product is computed. Must be the same size as the input of
            ``func``. This argument is optional when ``func``'s input contains
            a single element and (if it is not provided) will be set as a
            Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result will be
            computed in a differentiable way. Note that when ``strict`` is
            ``False``, the result can not require gradients or be disconnected
            from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            hvp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.
    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            hvp (tuple of Tensors or Tensor): result of the dot product with
            the same shape as the inputs.

    Example:

        >>> def pow_reducer(x):
        ...     return x.pow(3).sum()
        >>> inputs = tensorplay.rand(2, 2)
        >>> v = tensorplay.ones(2, 2)
        >>> output = hvp(pow_reducer, inputs, v)
        >>> output[0]
        tensor(0.1448)
        >>> output[1]
        tensor([[2.0239, 1.6456],
                [2.4988, 1.4310]])

    Note:

        This function is significantly slower than `vhp` due to backward mode AD constraints.
        If your function is twice continuously differentiable, then hvp = vhp.t(). So if you
        know that your function satisfies this condition, you should use vhp instead that is
        much faster with the current implementation.

    """
    with tensorplay.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "hvp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        if v is not None:
            _, v = _as_tuple(v, "v", "hvp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        else:
            if len(inputs) != 1 or inputs[0].numel() != 1:
                raise RuntimeError(
                    "The vector v can only be None if the input to the user-provided function "
                    "is a single Tensor with a single element."
                )
        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "hvp"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)

        if is_outputs_tuple or not isinstance(outputs[0], tensorplay.Tensor):
            raise RuntimeError(
                "The function given to hvp should return a single Tensor"
            )

        if outputs[0].numel() != 1:
            raise RuntimeError(
                "The Tensor returned by the function given to hvp should contain a single element"
            )

        jac = _autograd_grad(outputs, inputs, create_graph=True)
        _check_requires_grad(jac, "jacobian", strict=strict)

        grad_jac = tuple(tensorplay.zeros_like(inp, requires_grad=True) for inp in inputs)

        double_back = _autograd_grad(jac, inputs, grad_jac, create_graph=True)
        _check_requires_grad(jac, "hessian", strict=strict)

    enable_grad = True if create_graph else tensorplay.is_grad_enabled()
    with tensorplay.set_grad_enabled(enable_grad):
        grad_res = _autograd_grad(double_back, grad_jac, v, create_graph=create_graph)
        hvp_val = _fill_in_zeros(
            grad_res, inputs, strict, create_graph, "double_back_trick"
        )

    outputs = _grad_postprocess(outputs, create_graph)
    hvp_val = _grad_postprocess(hvp_val, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        hvp_val, is_inputs_tuple
    )
