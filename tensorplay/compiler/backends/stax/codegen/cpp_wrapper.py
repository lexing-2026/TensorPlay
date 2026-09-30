"""C++ wrapper generation for ahead-of-time compiled graphs.

The Python wrapper calls kernels through Python objects; the C++ wrapper
compiles the same schedule into a single translation unit that exposes one
entry point:

    extern "C" void call(void** tensor_data, long* sizevars);

``tensor_data`` holds the data pointer of every buffer the graph touches
(graph inputs, outputs and intermediates), addressed by an index assigned
during codegen.  ``sizevars`` holds the value of every symbolic size in the
graph.  The Python runtime allocates the buffers, fills ``tensor_data`` and
``sizevars``, and invokes ``call`` through the loaded shared object.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import sympy

import tensorplay as tp

from .. import config
from ..loops import V

from .. import ir
from ..utils import LineContext
from .common import IndentedBuffer
from .wrapper import (
    AllocateLine,
    FreeLine,
    MemoryPlanningLine,
    PythonWrapperCodegen,
    WrapperLine,
)


@dataclass
class CppWrapperCode:
    """The printed C++ wrapper plus everything needed to build and run it."""

    value: str
    line_map: list[tuple[int, LineContext]]
    buffer_meta: dict[str, tuple[Any, Any, Any, Any]]
    output_names: list[str]
    tensor_index: dict[str, int]
    sizevar_index: dict[Any, int]
    num_tensors: int
    num_sizevars: int
    graph_input_names: list[str]


class _TpTensorBuffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("sizes", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("dim", ctypes.c_int64),
        ("dtype", ctypes.c_int32),
        ("device_type", ctypes.c_int32),
    ]


_DEVICE_TYPE_TO_INT = {
    "cpu": 0,
    "cuda": 1,
    "hip": 2,
    "mps": 3,
    "mtia": 4,
    "xpu": 5,
    "hpu": 6,
    "privateuse1": 7,
    "vulkan": 8,
    "meta": 9,
    "unknown": 10,
}


class CppWrapperModule:
    """A built C++ wrapper, loaded and ready to be called.

    The shared object exposes ``call(TpTensorBuffer* tensor_data,
    long* sizevars)``.  This class owns the Python-side buffers: graph
    inputs are aliased from the caller's tensors, while intermediate and
    output buffers are allocated here according to the metadata recorded
    during codegen.
    """

    def __init__(
        self, call_fn: Any, wrapper_code: CppWrapperCode
    ) -> None:
        self._call_fn = call_fn
        self._tensor_index = wrapper_code.tensor_index
        self._num_tensors = wrapper_code.num_tensors
        self._num_sizevars = wrapper_code.num_sizevars
        self._output_names = wrapper_code.output_names
        self._buffer_meta = wrapper_code.buffer_meta
        self._graph_input_names = wrapper_code.graph_input_names

    @staticmethod
    def _fill_buffer(
        buf: _TpTensorBuffer,
        data_ptr: int,
        sizes: tuple[int, ...],
        strides: tuple[int, ...],
        dtype: Any,
        device_type: str,
        keep_alive: list,
    ) -> None:
        dim = len(sizes)
        size_arr = (ctypes.c_int64 * max(dim, 1))(*sizes)
        stride_arr = (ctypes.c_int64 * max(dim, 1))(*strides)
        buf.data = ctypes.c_void_p(data_ptr)
        buf.sizes = ctypes.cast(size_arr, ctypes.POINTER(ctypes.c_int64))
        buf.strides = ctypes.cast(stride_arr, ctypes.POINTER(ctypes.c_int64))
        buf.dim = dim
        buf.dtype = int(dtype)
        buf.device_type = _DEVICE_TYPE_TO_INT.get(device_type, 10)
        keep_alive.extend([size_arr, stride_arr])

    def call(self, args: list) -> tuple[Any, ...]:
        import ctypes

        if len(args) != len(self._graph_input_names):
            raise RuntimeError(
                f"expected {len(self._graph_input_names)} arguments, "
                f"got {len(args)}"
            )
        num_tensors = self._num_tensors
        tensor_data = (_TpTensorBuffer * num_tensors)()
        keep_alive: list[Any] = []
        for name, tensor in zip(self._graph_input_names, args):
            idx = self._tensor_index.get(name)
            if idx is None:
                continue
            buf = _TpTensorBuffer()
            self._fill_buffer(
                buf,
                tensor.data_ptr(),
                tuple(int(s) for s in tensor.shape),
                tuple(int(s) for s in tensor.stride()),
                tensor.dtype,
                tensor.device.type,
                keep_alive,
            )
            tensor_data[idx] = buf

        allocated: dict[str, Any] = {}
        for name, meta in self._buffer_meta.items():
            idx = self._tensor_index.get(name)
            if idx is None or name in self._graph_input_names:
                continue
            if name not in allocated:
                shape, stride, dtype, device = meta
                if shape is None:
                    continue
                try:
                    shape_i = tuple(int(s) for s in shape)
                    stride_i = tuple(int(s) for s in stride)
                except (TypeError, ValueError):
                    raise RuntimeError(
                        "C++ wrapper does not support dynamic shapes for "
                        f"buffer {name!r}"
                    ) from None
                allocated[name] = tp.empty_strided(
                    shape_i, stride_i, dtype=dtype, device=device
                )
            buf = _TpTensorBuffer()
            self._fill_buffer(
                buf,
                allocated[name].data_ptr(),
                tuple(int(s) for s in allocated[name].shape),
                tuple(int(s) for s in allocated[name].stride()),
                allocated[name].dtype,
                allocated[name].device.type,
                keep_alive,
            )
            tensor_data[idx] = buf

        sizevars = (ctypes.c_long * self._num_sizevars)()
        self._call_fn(tensor_data, sizevars)

        outputs = []
        for name in self._output_names:
            if name not in allocated:
                raise RuntimeError(
                    f"output buffer {name!r} was not allocated"
                )
            outputs.append(allocated[name])
        return tuple(outputs)


_DISPATCH_BINARY_TEMPLATE = (
    "tensorplay::DispatchStub<tensorplay::Tensor, "
    "const tensorplay::Tensor&, const tensorplay::Tensor&>::call("
    'tensorplay::Dispatcher::singleton().findHandle("{op}"), '
    "tensorplay::DispatchKey::CPU, {0}, {1})"
)

_DISPATCH_UNARY_TEMPLATE = (
    "tensorplay::DispatchStub<tensorplay::Tensor, "
    "const tensorplay::Tensor&>::call("
    'tensorplay::Dispatcher::singleton().findHandle("{op}"), '
    "tensorplay::DispatchKey::CPU, {0})"
)

_EXTERN_KERNEL_SHIMS: dict[str, tuple[str, str]] = {
    "matmul": (
        "tp_cpu_matmul",
        _DISPATCH_BINARY_TEMPLATE.replace("{op}", "matmul"),
    ),
    "mm": ("tp_cpu_mm", _DISPATCH_BINARY_TEMPLATE.replace("{op}", "mm")),
    "bmm": ("tp_cpu_bmm", _DISPATCH_BINARY_TEMPLATE.replace("{op}", "bmm")),
    "mul": ("tp_cpu_mul", "({0} * {1})"),
    "add": ("tp_cpu_add", "({0} + {1})"),
    "sub": ("tp_cpu_sub", "({0} - {1})"),
    "div": ("tp_cpu_div", "({0} / {1})"),
    "relu": ("tp_cpu_relu", _DISPATCH_UNARY_TEMPLATE.replace("{op}", "relu")),
    "sigmoid": (
        "tp_cpu_sigmoid",
        _DISPATCH_UNARY_TEMPLATE.replace("{op}", "sigmoid"),
    ),
    "silu": ("tp_cpu_silu", _DISPATCH_UNARY_TEMPLATE.replace("{op}", "silu")),
    "gelu": ("tp_cpu_gelu", _DISPATCH_UNARY_TEMPLATE.replace("{op}", "gelu")),
    "tanh": ("tp_cpu_tanh", _DISPATCH_UNARY_TEMPLATE.replace("{op}", "tanh")),
    "exp": ("tp_cpu_exp", _DISPATCH_UNARY_TEMPLATE.replace("{op}", "exp")),
    "sqrt": ("tp_cpu_sqrt", _DISPATCH_UNARY_TEMPLATE.replace("{op}", "sqrt")),
    "abs": ("tp_cpu_abs", _DISPATCH_UNARY_TEMPLATE.replace("{op}", "abs")),
    "neg": ("tp_cpu_neg", _DISPATCH_UNARY_TEMPLATE.replace("{op}", "neg")),
}


def _build_extern_shim_impl(
    shim_name: str, template: str, n_tensors: int
) -> str:
    params = ", ".join(
        [f"TpTensorBuffer* arg{i}" for i in range(n_tensors)]
        + ["TpTensorBuffer* out"]
    )
    lines = []
    for i in range(n_tensors):
        lines.append(
            f"tensorplay::Tensor t{i} = tp_buffer_to_tensor(*arg{i});"
        )
    expr = template.format(*[f"t{i}" for i in range(n_tensors)])
    lines.append(f"tensorplay::Tensor result = {expr};")
    lines.append(
        "std::memcpy(out->data, result.data_ptr(),"
        " result.numel() * result.itemsize());"
    )
    body = "\n        ".join(lines)
    return f"""
extern "C" int64_t {shim_name}({params}) {{
    try {{
        {body}
        return 0;
    }} catch (const std::exception& e) {{
        fprintf(stderr, "%s failed: %s\\n", "{shim_name}", e.what());
        return 1;
    }} catch (...) {{
        fprintf(stderr, "%s failed: unknown error\\n", "{shim_name}");
        return 1;
    }}
}}
"""


class CppWrapperCodegen(PythonWrapperCodegen):
    """Generate a C++ wrapper around the compiled C++ kernels.

    Currently supports CPU graphs whose schedule consists of C++ kernels
    (the ``CppScheduling`` backend).  Triton kernels and extern-kernel-only
    schedules raise ``NotImplementedError`` until the corresponding C++
    wrappers are implemented.
    """

    def __init__(self) -> None:
        self._kernel_decls: list[str] = []
        self._kernel_sources: list[str] = []
        self._extern_shim_sources: list[str] = []
        super().__init__()
        self.declare = "auto "
        self.declare_maybe_reference = "decltype(auto) "
        self.ending = ";"
        self.comment = "//"
        self.none_str = "nullptr"
        self.supports_intermediate_hooks = False

        # Buffers are addressed through a flat pointer array at runtime.
        self._tensor_index: dict[str, int] = {}
        self._sizevar_index: dict[Any, int] = {}
        self._num_tensors = 0
        self._num_sizevars = 0
        self._tensor_data_name = "tensor_data"
        self._sizevar_name = "sizevars"

        #: Metadata for buffers referenced by the generated wrapper, keyed by
        #: buffer name: (shape, stride, dtype, device).  Consumed by the
        #: Python side that builds ``tensor_data`` before calling ``call``.
        self.buffer_meta: dict[str, tuple[Any, Any, Any, Any]] = {}

        #: Names of the graph outputs, in return order.
        self.output_names: list[str] = []

    # -- argument/state plumbing ----------------------------------------

    def _register_tensor(self, name: str) -> int:
        if name not in self._tensor_index:
            self._tensor_index[name] = self._num_tensors
            self._num_tensors += 1
        return self._tensor_index[name]

    def _register_sizevar(self, sym: Any) -> int:
        key = sym if isinstance(sym, sympy.Basic) else sympy.sympify(sym)
        if key not in self._sizevar_index:
            self._sizevar_index[key] = self._num_sizevars
            self._num_sizevars += 1
        return self._sizevar_index[key]

    def _tensor_ref(self, name: str, cpp_dtype: str) -> str:
        idx = self._register_tensor(name)
        return f"({cpp_dtype}){self._tensor_data_name}[{idx}].data"

    def _tensor_buf_ref(self, name: str) -> str:
        idx = self._register_tensor(name)
        return f"&{self._tensor_data_name}[{idx}]"

    def _sizevar_ref(self, sym: Any) -> str:
        idx = self._register_sizevar(sym)
        return f"{self._sizevar_name}[{idx}]"

    def _record_buffer_meta(self, name: str) -> None:
        buf = V.graph.try_get_buffer(name)
        if buf is None:
            return
        try:
            shape = tuple(int(s) for s in buf.get_size())
            stride = tuple(int(s) for s in buf.get_stride())
        except (TypeError, ValueError):
            shape = tuple(buf.get_size())
            stride = tuple(buf.get_stride())
        self.buffer_meta[name] = (
            shape,
            stride,
            buf.get_dtype(),
            buf.get_device(),
        )

    # -- overrides -------------------------------------------------------

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: "PythonWrapperCodegen | None",
        partition_signatures: Any = None,
    ) -> "CppWrapperCodegen":
        if is_subgraph:
            raise NotImplementedError(
                "C++ wrapper does not support subgraphs yet"
            )
        return CppWrapperCodegen()

    def set_launcher_fn_name(self) -> None:
        self.launcher_fn_name = "call"

    def write_constant(self, name: str, hashed: str) -> None:
        # Constants are embedded in the generated C++ or passed from Python;
        # there is no separate constant namespace in the C++ wrapper.
        pass

    def write_header(self) -> None:
        self.header.splice('#include "tensorplay/GeneratedCode.h"')
        self.header.splice('#include "Tensor.h"')
        self.header.splice("#include <cstring>")
        self.header.splice("#include <cstdio>")
        self.header.splice("#include <vector>")
        self.header.writeline("")
        self.header.splice(
            """
            struct TpTensorBuffer {
                void* data;
                const int64_t* sizes;
                const int64_t* strides;
                int64_t dim;
                int32_t dtype;
                int32_t device_type;
            };

            static tensorplay::Tensor tp_buffer_to_tensor(const TpTensorBuffer& buf) {
                std::vector<int64_t> sizes(buf.sizes, buf.sizes + buf.dim);
                std::vector<int64_t> strides(buf.strides, buf.strides + buf.dim);
                int64_t numel = 1;
                for (int64_t s : sizes) numel *= s;
                tensorplay::Device dev(static_cast<tensorplay::DeviceType>(buf.device_type));
                tensorplay::Storage storage(
                    tensorplay::DataPtr(buf.data, tensorplay::deleteNothing, dev),
                    numel * tensorplay::elementSize(static_cast<tensorplay::ScalarType>(buf.dtype)),
                    nullptr);
                return tensorplay::Tensor(
                    storage, sizes, strides,
                    static_cast<tensorplay::ScalarType>(buf.dtype), 0);
            }
            """
        )
        self.header.writeline("")

    def write_prefix(self) -> None:
        # Input handling and buffer allocation happen on the Python side of
        # the C++ wrapper; nothing is written into the generated translation
        # unit before the kernels are called.
        pass

    def _define_kernel_helper(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: str | None = None,
        gpu: bool = True,
        cpp_definition: str | None = None,
    ) -> None:
        if cpp_definition is None:
            raise NotImplementedError(
                "C++ wrapper requires cpp_definition for kernel declarations"
            )
        self._kernel_decls.append(cpp_definition)
        self._kernel_sources.append(kernel_body)

    def codegen_sizevar(self, x: sympy.Expr) -> str:
        x = sympy.sympify(x)
        if x.is_number:
            return str(int(x))
        return self._sizevar_ref(x)

    def write_triton_header_once(self) -> None:
        pass

    def write_get_raw_stream_header_once(self) -> None:
        pass

    def write_get_raw_stream(self, device_idx: int, graph_name: str) -> str:
        return "0"

    def generate_profiler_mark_wrapper_call(self, stack) -> None:
        pass

    def generate_start_graph(self) -> None:
        pass

    def generate_end_graph(self) -> None:
        pass

    def generate_proton_finalize(self) -> None:
        pass

    def generate_debug_sync(self, buffer) -> None:
        pass

    @staticmethod
    def _op_name_from_kernel(kernel_name: str) -> str:
        """Derive the registered op name from a generated C++ kernel name.

        ``"tp_cpu_matmul"`` -> ``"matmul"``, ``"mm_out"`` -> ``"mm"``,
        ``"tp_mm_out"`` -> ``"mm"``.
        """

        name = kernel_name
        if name.startswith("tp_cpu_"):
            name = name[len("tp_cpu_") :]
        if name.endswith("_out"):
            name = name[: -len("_out")]
        name = name.removeprefix("tp_").removeprefix("_")
        return name

    def _generate_extern_kernel_alloc_helper(
        self, extern_kernel, args
    ) -> None:
        if isinstance(extern_kernel.layout, ir.NoneLayout):
            return
        if getattr(extern_kernel, "outputs", None):
            self._generate_fallback_kernel_helper(
                extern_kernel.get_name(),
                [o.get_name() for o in extern_kernel.outputs],
                args,
                extern_kernel.get_device(),
                extern_kernel.get_stack_traces(),
            )
            return
        output_name = extern_kernel.get_name()
        self._register_tensor(output_name)
        self._record_buffer_meta(output_name)

        tensor_args = list(getattr(extern_kernel, "tensor_args", None) or [])
        if not tensor_args:
            tensor_args = [
                a for a in extern_kernel.inputs if isinstance(a, ir.Buffer)
            ]
        if not tensor_args:
            raise NotImplementedError(
                "C++ wrapper does not support extern kernels without "
                "tensor inputs"
            )

        target = getattr(extern_kernel, "kernel", None) or extern_kernel.op_overload
        name = getattr(target, "__name__", None)
        if name is None:
            name = getattr(extern_kernel, "cpp_kernel_name", None)
            if name:
                name = name.split("::")[-1].replace("_", "", 1)
        shim_name, template = _EXTERN_KERNEL_SHIMS.get(
            name, (None, None)
        )
        if shim_name is None:
            raise NotImplementedError(
                f"C++ wrapper has no shim for extern kernel {name!r}"
            )
        impl = _build_extern_shim_impl(shim_name, template, len(tensor_args))
        if impl not in self._extern_shim_sources:
            self._extern_shim_sources.append(impl)

        arg_refs = [
            self._tensor_buf_ref(t.get_name()) for t in tensor_args
        ]
        arg_refs.append(self._tensor_buf_ref(output_name))
        self.writeline(
            f"TP_CHECK({shim_name}({', '.join(arg_refs)}) == 0, "
            f'"extern kernel {shim_name} failed");'
        )

    def _generate_extern_kernel_out_helper(
        self,
        kernel_name,
        output_name,
        output_view_name,
        args,
        device,
        stack_traces,
    ) -> None:
        self._register_tensor(output_name)
        self._record_buffer_meta(output_name)
        op_name = self._op_name_from_kernel(kernel_name)
        shim_name, template = _EXTERN_KERNEL_SHIMS.get(
            op_name, (None, None)
        )
        if shim_name is None:
            raise NotImplementedError(
                f"C++ wrapper has no shim for extern kernel {op_name!r}"
            )
        tensor_refs = []
        for arg in args:
            if isinstance(arg, str) and arg in V.graph.name_to_buffer:
                self._register_tensor(arg)
                tensor_refs.append(self._tensor_buf_ref(arg))
            else:
                raise NotImplementedError(
                    "C++ wrapper extern kernel out calls support only "
                    f"tensor arguments; got {arg!r} for {op_name!r}"
                )
        impl = _build_extern_shim_impl(
            shim_name, template, len(tensor_refs)
        )
        if impl not in self._extern_shim_sources:
            self._extern_shim_sources.append(impl)
        out_ref = self._tensor_buf_ref(output_name)
        self.writeline(
            f"TP_CHECK({shim_name}({', '.join(tensor_refs + [out_ref])}) == 0, "
            f'"extern kernel {shim_name} failed");'
        )

    def _generate_extern_kernel_multi_out_helper(
        self, kernel_name, output_names, args, device, stack_traces
    ) -> None:
        raise NotImplementedError(
            "C++ wrapper does not support extern kernel multi-out calls yet"
        )

    def _generate_fallback_kernel_helper(
        self,
        kernel_name,
        output_names,
        args,
        device,
        stack_traces,
    ) -> None:
        raise NotImplementedError(
            "C++ wrapper does not support fallback kernel calls yet"
        )

    def _generate_index_put_fallback_helper(self, node) -> None:
        raise NotImplementedError(
            "C++ wrapper does not support index_put fallback yet"
        )

    def _generate_scatter_fallback_helper(self, node) -> None:
        raise NotImplementedError(
            "C++ wrapper does not support scatter fallback yet"
        )

    def _generate_kernel_call_helper(
        self,
        kernel_name: str,
        call_args,
        *,
        device=None,
        triton=True,
        arg_types=None,
        raw_keys=None,
        raw_args=None,
        triton_meta=None,
        tp_meta=None,
        graph_name="",
        original_fxnode_name=None,
        current_stream_idx=None,
    ) -> None:
        if triton:
            raise NotImplementedError(
                "C++ wrapper does not support triton kernels yet"
            )
        if device is not None and device.type != "cpu":
            raise NotImplementedError(
                "C++ wrapper currently supports CPU kernels only"
            )
        cpp_args = []
        for arg, arg_type in zip(call_args, arg_types):
            if arg_type.endswith("*"):
                cpp_args.append(self._tensor_ref(arg, arg_type))
                self._record_buffer_meta(arg)
            else:
                expr = sympy.sympify(arg)
                if expr.is_number:
                    cpp_args.append(str(int(expr)))
                else:
                    cpp_args.append(self._sizevar_ref(expr))
        self.writeline(f"{kernel_name}({', '.join(cpp_args)});")

    def generate_return(self, output_refs) -> None:
        # Outputs are written through tensor_data; there is no return value.
        pass

    def generate_before_suffix(self, result) -> None:
        result.writeline("")
        result.writeline(
            f'extern "C" void call(TpTensorBuffer* {self._tensor_data_name}, '
            f"long* {self._sizevar_name}) {{"
        )

    def generate_after_suffix(self, result) -> None:
        pass

    def generate_end(self, result) -> None:
        result.writeline("}")

    def add_benchmark_harness(self, output) -> None:
        pass

    def _generate(self, is_inference):
        import contextlib

        with contextlib.ExitStack() as stack:
            stack.enter_context(self.wrapper_call.indent())
            if config.profiler_mark_wrapper_call:
                self.generate_profiler_mark_wrapper_call(stack)

            with self.set_writeline(self.wrapper_call, self.wrapper_call.writeline):
                for line in self.lines:
                    if isinstance(line, MemoryPlanningLine):
                        # Buffer allocation and reuse are handled by the
                        # Python runtime of the C++ wrapper; the generated
                        # translation unit only references buffers through
                        # the flat tensor_data array.
                        if isinstance(line, AllocateLine):
                            self._register_tensor(line.node.get_name())
                            self._record_buffer_meta(line.node.get_name())
                        continue
                    if isinstance(line, FreeLine):
                        continue
                    if isinstance(line, WrapperLine):
                        line.codegen(self.wrapper_call)
                    else:
                        self.wrapper_call.writeline(line)

            output_refs = self.get_output_refs()
            self.output_names = list(output_refs)
            for name in output_refs:
                self._register_tensor(name)
                self._record_buffer_meta(name)
            self.generate_return(output_refs)

        result = IndentedBuffer()
        result.splice(self.imports)
        result.writeline("")
        result.splice(self.header)
        for decl in self._kernel_decls:
            result.writeline(decl)
        result.writeline("")
        for impl in self._extern_shim_sources:
            result.splice(impl)
            result.writeline("")
        result.splice(self.subgraph_definitions)
        self.finalize_prefix()
        result.splice(self.prefix)

        self.generate_before_suffix(result)

        with result.indent(self.get_wrapper_call_indent()):
            result.splice(self.wrapper_call)

        result.splice(self.suffix)
        self.generate_after_suffix(result)
        self.generate_end(result)

        full_source = result.getvaluewithlinemap()
        value = full_source.value
        if self._kernel_sources:
            value = value + "\n\n" + "\n\n".join(self._kernel_sources)

        for name in list(self._tensor_index):
            if name not in self.buffer_meta:
                self._record_buffer_meta(name)

        tensor_input_names = [
            name
            for name in V.graph.graph_input_names
            if name in V.graph.graph_inputs_original
        ]
        code = CppWrapperCode(
            value=value,
            line_map=full_source.line_map,
            buffer_meta=self.buffer_meta,
            output_names=self.output_names,
            tensor_index=self._tensor_index,
            sizevar_index=self._sizevar_index,
            num_tensors=self._num_tensors,
            num_sizevars=self._num_sizevars,
            graph_input_names=tensor_input_names,
        )
        return (
            code,
            self.kernel_declarations.getvaluewithlinemap(),
        )
