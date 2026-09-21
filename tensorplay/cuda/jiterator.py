# mypy: allow-untyped-defs
import re

import tensorplay
from tensorplay import Tensor


__all__ = ["_JiteratorFunction"]


class _CodeParser:
    """Extract the entry functor name from a jiterator source string.

    The last ``template <...>`` definition is the entry point, matching the
    convention that helper functors may precede it.
    """

    def __init__(self, code_string: str):
        optional_ws = r"\s*"
        required_ws = r"\s+"
        template_params = r"(?P<template_params>\<.+\>)"
        return_type = r"(?P<return_type>\w+)"
        function_name = r"(?P<function_name>\w+)"
        function_params = r"(?P<function_params>\(.+\))"
        function_body = r"(?P<function_body>\{.+\})"

        pattern = (
            optional_ws
            + "template"
            + optional_ws
            + template_params
            + optional_ws
            + return_type
            + required_ws
            + function_name
            + optional_ws
            + function_params
            + optional_ws
            + function_body
            + optional_ws
        )

        result = re.match(pattern, code_string, re.DOTALL)
        if result is None:
            raise RuntimeError(
                f"Couldn't parse code, please check correctness:\n {code_string}"
            )

        self.template_params = result["template_params"]
        self.return_type = result["return_type"]
        self.function_name = result["function_name"]
        self.function_params = result["function_params"]
        self.function_body = result["function_body"]


class _JiteratorFunction:
    """Callable that launches a jiterator-generated kernel.

    The ``code_string`` must contain a valid CUDA functor describing the
    computation of a single element. Tensors are broadcast, promoted to a
    common dtype, and copied to contiguous layout before the kernel runs.
    """

    def __init__(
        self, code_string: str, return_by_ref: bool, num_outputs: int, **kwargs
    ):
        self.code_string = code_string

        if not (return_by_ref or num_outputs == 1):
            raise AssertionError("Return by value only works for single output.")
        self.return_by_ref = return_by_ref
        self.num_outputs = num_outputs

        parsed_code = _CodeParser(code_string)
        self.kernel_name = parsed_code.function_name

        self.kwargs_dict = kwargs
        self.is_cuda_available = tensorplay.cuda.is_available()

    def __call__(self, *tensors: Tensor, **kwargs):
        # kernel availability is deferred to invoke time, mirroring the
        # lazy initialization behavior of the CUDA layer
        if not self.is_cuda_available:
            raise AssertionError(
                "Jiterator is only supported on CUDA GPUs, none are available."
            )

        if len(tensors) > 8:
            raise AssertionError(
                f"jiterator only supports up to 8 tensor inputs, got {len(tensors)}"
            )

        expanded_kwargs = self.kwargs_dict.copy()
        for key, value in kwargs.items():
            if key in self.kwargs_dict:
                expanded_kwargs[key] = value
            else:
                raise KeyError(f"{key} is not declared in function definition")

        normalized = self._normalize_inputs(tensors)
        shape = tensorplay.broadcast_shapes(
            *(tuple(tensor.shape) for tensor in tensors)
        )
        result = tensorplay._C._cuda_jiterator_compile_and_launch_kernel(
            self.code_string,
            self.kernel_name,
            self.return_by_ref,
            self.num_outputs,
            tuple(normalized),
            expanded_kwargs,
        )
        if self.num_outputs == 1:
            return result.view(shape)
        return tuple(out.view(shape) for out in result)

    def _normalize_inputs(self, tensors: tuple) -> tuple:
        for tensor in tensors:
            if not tensor.device.is_cuda():
                raise RuntimeError(
                    "jiterator inputs must live on a cuda device"
                )
        shape = tensorplay.broadcast_shapes(
            *(tuple(tensor.shape) for tensor in tensors)
        )
        dtype = tensors[0].dtype
        for tensor in tensors[1:]:
            dtype = tensorplay.promote_types(dtype, tensor.dtype)
        return tuple(
            tensor.expand(shape).to(dtype).contiguous() for tensor in tensors
        )


def _create_jiterator_fn(code_string: str, **kwargs) -> _JiteratorFunction:
    """
    Create a jiterator-generated cuda kernel for an elementwise op.

    The code string has to be a valid CUDA function that describes the
    computation for a single element. The code string has to follow the
    c++ template pattern, as shown in the example below. This function will
    be inlined into an elementwise kernel template, and compiled on the
    fly. The compiled kernel is cached in memory.

    Jiterator-generated kernels accept noncontiguous tensors, and support
    broadcasting and type promotion.

    Args:
        code_string (str): CUDA code string to be compiled by jiterator. The
            entry functor must return by value.
        kwargs (Dict, optional): Keyword arguments for generated function

    Example::

        code_string = "template <typename T> T my_kernel(T x, T y, T alpha) { return -x + alpha * y; }"
        jitted_fn = create_jiterator_fn(code_string, alpha=1.0)
        a = tensorplay.rand(3, device="cuda")
        b = tensorplay.rand(3, device="cuda")
        # invoke jitted function like a regular python function
        result = jitted_fn(a, b, alpha=3.14)

    code_string also allows multiple function definitions, and the last
    function will be treated as the entry function.

    .. warning::
        This API is in beta and may change in future releases.

    .. warning::
        This API only supports up to 8 inputs and 1 output

    .. warning::
        All input tensors must live in CUDA device
    """
    return _JiteratorFunction(
        code_string, return_by_ref=False, num_outputs=1, **kwargs
    )


def _create_multi_output_jiterator_fn(
    code_string: str, num_outputs: int, **kwargs
) -> _JiteratorFunction:
    """
    Create a jiterator-generated cuda kernel for an elementwise op that
    supports returning one or more outputs.

    Args:
        code_string (str): CUDA code string to be compiled by jiterator. The
            entry functor must return value by reference.
        num_outputs(int): number of outputs return by the kernel
        kwargs (Dict, optional): Keyword arguments for generated function

    .. warning::
        This API only supports up to 8 inputs and 8 outputs
    """
    return _JiteratorFunction(
        code_string, return_by_ref=True, num_outputs=num_outputs, **kwargs
    )
