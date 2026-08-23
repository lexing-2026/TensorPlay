#include "python_bindings.h"
#include "tensorplay/ops/Config.h"
#include "tensorplay/ops/TensorBindingsGenerated.h"
#include "tensorplay/ops/TensorCPythonGenerated.h"
#include "Context.h"
#include "utils.h"
#include <filesystem>

using namespace tensorplay::python;

// Declaration of create_tensor (defined in Tensor.cpp)
Tensor create_tensor(py::object data, std::optional<DType> dtype, std::optional<Device> device);

namespace {
Tensor mark_requires_grad(Tensor t, bool requires_grad) {
    if (requires_grad) tensorplay::tpx::impl::set_requires_grad(t, true);
    return t;
}

// Factory functions fall back to the global default dtype/device when the
// caller passes None, mirroring torch's factory resolution.
DType resolve_default_dtype(std::optional<DType> dtype) {
    return dtype.has_value() ? *dtype : tensorplay::globalContext().defaultDType();
}
Device resolve_default_device(std::optional<Device> device) {
    return device.has_value() ? *device : tensorplay::globalContext().defaultDevice();
}
}

void init_ops(py::module_& m) {
    // Module functions
    m.def("tensor", [](py::object data, std::optional<DType> dtype, std::optional<Device> device,
                        bool pin_memory, bool requires_grad) {
         Tensor t = create_tensor(data, dtype, device);
         if (pin_memory) t = Tensor(t.pin_memory());
         if (requires_grad) {
             tensorplay::tpx::impl::set_requires_grad(t, true);
         }
         return t;
    }, "data"_a, py::kw_only(), "dtype"_a = py::none(), "device"_a = py::none(),
       "pin_memory"_a = false, "requires_grad"_a = false,
    "tensor(data, *, dtype: Optional[DType] = None, device: Optional[Device] = None, pin_memory: bool = False, requires_grad: bool = False) -> Tensor");

    {
        return mark_requires_grad(Tensor::zeros(size, resolve_default_dtype(dtype),
                             resolve_default_device(device), pin_memory), requires_grad);
    }, "size"_a, py::kw_only(), "dtype"_a = py::none(), "device"_a = py::none(),
       "pin_memory"_a = false, "requires_grad"_a = false,
    "zeros(size: Sequence[int], *, dtype: Optional[DType] = None, device: Optional[Device] = None, pin_memory: bool = False, requires_grad: bool = False) -> Tensor");

    m.def("zeros", [](py::args args, py::kwargs kwargs) -> Tensor {
        DType dtype = tensorplay::globalContext().defaultDType();
        Device device = tensorplay::glob = false;
        bool requires_grad = false;
        if (kwargs.contains("dtype") && !kwargs["dtype"].is_none()) dtype = py::cast<DType>(kwargs["dtype"]);
        if (kwargs.contains("device") && !kwargs["device"].is_none()) device = py::cast<Device>(kwargs["device"]);
        if (kwargs.contains("pin_memory") && !kwargs["pin_memory"].is_none()) pin_memory = py::cast<bool>(kwargs["pin_memory"]);
        if (kwargs.contains("requires_grad") && !kwargs["requires_grad"].is_none()) requires_grad = py::cast<bool>(kwargs["requires_grad"]);
        return mark_requires_grad(Tensor::zeros(parse_shape_args(args), dtype, device, pin_memory), requires_grad);
    }, "zeros(*size: int, dtype: Optional[DType] = None, device: Optional[Device] = None, pin_memory: bool = False, requires_grad: bool = False) -> Tensor");

    // Ops submodule
    py::module_ ops = m.def_submodule("ops", "Operator registry");
    ops.def("load_library", [](const std::string& path) {
        namespace fs = std::filesystem;
        fs::path p(path);
        if (!fs::exists(p)) {
            TP_THROW(RuntimeError, "Library file not found: ", path);
        }
        
        py::object importlib_util = py::module_::import("importlib.util");
        std::string name = p.stem().string();
        
        // Remove ABI tags (everything after first dot)
        size_t first_dot = name.find('.');
        if (first_dot != std::string::npos) {
            name = name.substr(0, first_dot);
        }
        
        // Remove "lib" prefix if present (common in Unix)
        if (name.size() > 3 && name.rfind("lib", 0) == 0) {
            name = name.substr(3);
        }

        py::object spec = importlib_util.attr("spec_from_file_location")(name, path);
        if (spec.is_none()) {
            TP_THROW(RuntimeError, "Could not load library specification from: ", path);
        }
        
        py::object module = importlib_util.attr("module_from_spec")(spec);
        spec.attr("loader").attr("exec_module")(module);
        
        // Register under tensorplay.ops
        py::object tp = py::module_::import("tensorplay");
        if (py::hasattr(tp, "ops")) {
            tp.attr("ops").attr(name.c_str()) = module;
        }
    }, "path"_a);

    // Bind generated functions (includes *_like, transpose, permute, etc.)
    bind_generated_op_functions(m);

    // NOTE: the METH_FASTCALL layer (register_generated_cpython_functions)
    // is installed at the end of PYBIND11_MODULE so it can never shadow a
    // hand-written pybind overload -- it only fills names nothing else bound.

    // Config
    m.def("_show_config", &tensorplay::show_config);
    
    // Manual bindings for varargs/complex factories
    {
        return mark_requires_grad(Tensor::linspace(start, end, steps, resolve_default_dtype(dtype), resolve_default_device(device)), requires_grad);
    }, "start"_a, "end"_a, "steps"_a, py::kw_only(), "dtype"_a = py::none(), "device"_a = py::none(), "requires_grad"_a = false,
    "linspace(start: float, end: float, steps: int, *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");
    {
        return mark_requires_grad(Tensor::logspace(start, end, steps, base, resolve_default_dtype(dtype), resolve_default_device(device)), requires_grad);
    }, "start"_a, "end"_a, "steps"_a, "base"_a = 10.0, py::kw_only(), "dtype"_a = py::none(), "device"_a = py::none(), "requires_grad"_a = false,
    "logspace(start: float, end: float, steps: int, base: float = 10.0, *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");
}
