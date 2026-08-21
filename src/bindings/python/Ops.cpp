#include "python_bindings.h"
#include "tensorplay/ops/Config.h"
#include "tensorplay/ops/TensorBindingsGenerated.h"
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

    m.def("zeros", [](const std::vector<int64_t>& size, std::optional<DType> dtype,
                       std::optional<Device> device, bool pin_memory, bool requires_grad) {
        return mark_requires_grad(Tensor::zeros(size, dtype.value_or(DType::Float32),
                             device.value_or(Device(DeviceType::CPU)), pin_memory), requires_grad);
    }, "size"_a, py::kw_only(), "dtype"_a = py::none(), "device"_a = py::none(),
       "pin_memory"_a = false, "requires_grad"_a = false,
    "zeros(size: Sequence[int], *, dtype: Optional[DType] = None, device: Optional[Device] = None, pin_memory: bool = False, requires_grad: bool = False) -> Tensor");

    m.def("zeros", [](py::args args, py::kwargs kwargs) -> Tensor {
        DType dtype = DType::Float32;
        Device device = Device(DeviceType::CPU);
        bool pin_memory = false;
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

    m.def("ones", [](const std::vector<int64_t>& size, std::optional<DType> dtype,
                      std::optional<Device> device, bool pin_memory, bool requires_grad) {
        return mark_requires_grad(Tensor::ones(size, dtype.value_or(DType::Float32),
                            device.value_or(Device(DeviceType::CPU)), pin_memory), requires_grad);
    }, "size"_a, py::kw_only(), "dtype"_a = py::none(), "device"_a = py::none(),
       "pin_memory"_a = false, "requires_grad"_a = false,
    "ones(size: Sequence[int], *, dtype: Optional[DType] = None, device: Optional[Device] = None, pin_memory: bool = False, requires_grad: bool = False) -> Tensor");

    m.def("ones", [](py::args args, py::kwargs kwargs) -> Tensor {
        DType dtype = DType::Float32;
        Device device = Device(DeviceType::CPU);
        bool pin_memory = false;
        bool requires_grad = false;
        if (kwargs.contains("dtype") && !kwargs["dtype"].is_none()) dtype = py::cast<DType>(kwargs["dtype"]);
        if (kwargs.contains("device") && !kwargs["device"].is_none()) device = py::cast<Device>(kwargs["device"]);
        if (kwargs.contains("pin_memory") && !kwargs["pin_memory"].is_none()) pin_memory = py::cast<bool>(kwargs["pin_memory"]);
        if (kwargs.contains("requires_grad") && !kwargs["requires_grad"].is_none()) requires_grad = py::cast<bool>(kwargs["requires_grad"]);
        auto shape = parse_shape_args(args);
        return mark_requires_grad(Tensor::ones(shape, dtype, device, pin_memory), requires_grad);
    }, "ones(*size: int, dtype: Optional[DType] = None, device: Optional[Device] = None, pin_memory: bool = False, requires_grad: bool = False) -> Tensor");

    m.def("eye", [](int64_t n, int64_t m, DType dtype, Device device, bool requires_grad) {
        return mark_requires_grad(Tensor::eye(n, m, dtype, device), requires_grad);
    }, "n"_a, "m"_a = -1, py::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false,
    "eye(n: int, m: int = -1, *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");
    
    m.def("empty", [](const std::vector<int64_t>& size, std::optional<DType> dtype,
                       std::optional<Device> device, bool pin_memory, bool requires_grad) {
        return mark_requires_grad(Tensor::empty(size, dtype.value_or(DType::Float32),
                             device.value_or(Device(DeviceType::CPU)), pin_memory), requires_grad);
    }, "size"_a, py::kw_only(), "dtype"_a = py::none(), "device"_a = py::none(),
       "pin_memory"_a = false, "requires_grad"_a = false,
    "empty(size: Sequence[int], *, dtype: Optional[DType] = None, device: Optional[Device] = None, pin_memory: bool = False, requires_grad: bool = False) -> Tensor");

    m.def("empty", [](py::args args, py::kwargs kwargs) -> Tensor {
        DType dtype = DType::Float32;
        Device device = Device(DeviceType::CPU);
        bool pin_memory = false;
        bool requires_grad = false;
        if (kwargs.contains("dtype") && !kwargs["dtype"].is_none()) dtype = py::cast<DType>(kwargs["dtype"]);
        if (kwargs.contains("device") && !kwargs["device"].is_none()) device = py::cast<Device>(kwargs["device"]);
        if (kwargs.contains("pin_memory") && !kwargs["pin_memory"].is_none()) pin_memory = py::cast<bool>(kwargs["pin_memory"]);
        if (kwargs.contains("requires_grad") && !kwargs["requires_grad"].is_none()) requires_grad = py::cast<bool>(kwargs["requires_grad"]);
        return mark_requires_grad(Tensor::empty(parse_shape_args(args), dtype, device, pin_memory), requires_grad);
    }, "empty(*size: int, dtype: Optional[DType] = None, device: Optional[Device] = None, pin_memory: bool = False, requires_grad: bool = False) -> Tensor");

    m.def("rand", [](const std::vector<int64_t>& size, std::optional<DType> dtype, std::optional<Device> device, bool requires_grad) {
        return mark_requires_grad(Tensor::rand(size, dtype.value_or(DType::Float32), device.value_or(Device(DeviceType::CPU))), requires_grad);
    }, "size"_a, py::kw_only(), "dtype"_a = py::none(), "device"_a = py::none(), "requires_grad"_a = false,
    "rand(size: Sequence[int], *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");

    m.def("rand", [](py::args args, py::kwargs kwargs) -> Tensor {
        DType dtype = DType::Float32;
        Device device = Device(DeviceType::CPU);
        bool requires_grad = false;
        if (kwargs.contains("dtype") && !kwargs["dtype"].is_none()) dtype = py::cast<DType>(kwargs["dtype"]);
        if (kwargs.contains("device") && !kwargs["device"].is_none()) device = py::cast<Device>(kwargs["device"]);
        if (kwargs.contains("requires_grad") && !kwargs["requires_grad"].is_none()) requires_grad = py::cast<bool>(kwargs["requires_grad"]);
        return mark_requires_grad(Tensor::rand(parse_shape_args(args), dtype, device), requires_grad);
    }, "rand(*size: int, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");

    m.def("randint", [](int64_t low, int64_t high, const std::vector<int64_t>& size, DType dtype, Device device, bool requires_grad) {
        return mark_requires_grad(Tensor::randint(low, high, size, dtype, device), requires_grad);
    }, "low"_a, "high"_a, "size"_a, py::kw_only(), "dtype"_a = DType::Int64, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false,
    "randint(low: int, high: int, size: Sequence[int], *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");

    m.def("randn", [](const std::vector<int64_t>& size, std::optional<DType> dtype, std::optional<Device> device, bool requires_grad) {
        return mark_requires_grad(Tensor::randn(size, dtype.value_or(DType::Float32), device.value_or(Device(DeviceType::CPU))), requires_grad);
    }, "size"_a, py::kw_only(), "dtype"_a = py::none(), "device"_a = py::none(), "requires_grad"_a = false,
    "randn(size: Sequence[int], *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");

    m.def("randn", [](py::args args, py::kwargs kwargs) -> Tensor {
        DType dtype = DType::Float32;
        Device device = Device(DeviceType::CPU);
        bool requires_grad = false;
        if (kwargs.contains("dtype") && !kwargs["dtype"].is_none()) dtype = py::cast<DType>(kwargs["dtype"]);
        if (kwargs.contains("device") && !kwargs["device"].is_none()) device = py::cast<Device>(kwargs["device"]);
        if (kwargs.contains("requires_grad") && !kwargs["requires_grad"].is_none()) requires_grad = py::cast<bool>(kwargs["requires_grad"]);
        return mark_requires_grad(Tensor::randn(parse_shape_args(args), dtype, device), requires_grad);
    }, "randn(*size: int, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");

    m.def("randperm", [](int64_t n, DType dtype, Device device, bool requires_grad) {
        return mark_requires_grad(Tensor::randperm(n, dtype, device), requires_grad);
    }, "n"_a, py::kw_only(), "dtype"_a = DType::Int64, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false,
    "randperm(n: int, *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");
    
    m.def("full", [](const std::vector<int64_t>& shape, Scalar fill_value, DType dtype, Device device, bool pin_memory, bool requires_grad) {
        return mark_requires_grad(Tensor::full(shape, fill_value, dtype, device, pin_memory), requires_grad);
    }, "shape"_a, "fill_value"_a, py::kw_only(),
          "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU),
          "pin_memory"_a = false, "requires_grad"_a = false,
    "full(shape: Sequence[int], fill_value: Union[float, int], *, dtype: Optional[DType] = None, device: Optional[Device] = None, pin_memory: bool = False, requires_grad: bool = False) -> Tensor");
    
    // Bind generated functions (includes *_like, transpose, permute, etc.)
    bind_generated_op_functions(m);

    // Config
    m.def("_show_config", &tensorplay::show_config);
    
    // Manual bindings for varargs/complex factories
    m.def("linspace", [](Scalar start, Scalar end, int64_t steps, DType dtype, Device device, bool requires_grad) {
        return mark_requires_grad(Tensor::linspace(start, end, steps, dtype, device), requires_grad);
    }, "start"_a, "end"_a, "steps"_a, py::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false,
    "linspace(start: float, end: float, steps: int, *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");
    m.def("logspace", [](Scalar start, Scalar end, int64_t steps, double base, DType dtype, Device device, bool requires_grad) {
        return mark_requires_grad(Tensor::logspace(start, end, steps, base, dtype, device), requires_grad);
    }, "start"_a, "end"_a, "steps"_a, "base"_a = 10.0, py::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false,
    "logspace(start: float, end: float, steps: int, base: float = 10.0, *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");
}
