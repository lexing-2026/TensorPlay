#include "python_bindings.h"
#include "tensorplay/ops/Config.h"
#include "tensorplay/ops/TensorBindingsGenerated.h"
#include "utils.h"
#include <filesystem>

using namespace tensorplay::python;

// Declaration of create_tensor (defined in Tensor.cpp)
Tensor create_tensor(nb::object data, std::optional<DType> dtype, std::optional<Device> device);

void init_ops(nb::module_& m) {
    // Module functions
    m.def("tensor", [](nb::object data, std::optional<DType> dtype, std::optional<Device> device, bool requires_grad) {
         Tensor t = create_tensor(data, dtype, device);
         if (requires_grad) {
             t.set_requires_grad(true);
         }
         return t;
    }, "data"_a, nb::kw_only(), "dtype"_a = nb::none(), "device"_a = nb::none(), "requires_grad"_a = false,
    "tensor(data, *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");

    m.def("zeros", [](const std::vector<int64_t>& size, std::optional<DType> dtype, std::optional<Device> device, bool requires_grad) {
        return Tensor::zeros(size, dtype.value_or(DType::Float32), device.value_or(Device(DeviceType::CPU)), requires_grad);
    }, "size"_a, nb::kw_only(), "dtype"_a = nb::none(), "device"_a = nb::none(), "requires_grad"_a = false,
    "zeros(size: Sequence[int], *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");

    m.def("zeros", [](nb::args args, nb::kwargs kwargs) {
        DType dtype = DType::Float32;
        Device device = Device(DeviceType::CPU);
        bool requires_grad = false;
        if (kwargs.contains("dtype") && !kwargs["dtype"].is_none()) dtype = nb::cast<DType>(kwargs["dtype"]);
        if (kwargs.contains("device") && !kwargs["device"].is_none()) device = nb::cast<Device>(kwargs["device"]);
        if (kwargs.contains("requires_grad") && !kwargs["requires_grad"].is_none()) requires_grad = nb::cast<bool>(kwargs["requires_grad"]);
        return Tensor::zeros(parse_shape_args(args), dtype, device, requires_grad);
    }, "zeros(*size: int, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");

    // Ops submodule
    nb::module_ ops = m.def_submodule("ops", "Operator registry");
    ops.def("load_library", [](const std::string& path) {
        namespace fs = std::filesystem;
        fs::path p(path);
        if (!fs::exists(p)) {
            TP_THROW(RuntimeError, "Library file not found: ", path);
        }
        
        nb::object importlib_util = nb::module_::import_("importlib.util");
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

        nb::object spec = importlib_util.attr("spec_from_file_location")(name, path);
        if (spec.is_none()) {
            TP_THROW(RuntimeError, "Could not load library specification from: ", path);
        }
        
        nb::object module = importlib_util.attr("module_from_spec")(spec);
        spec.attr("loader").attr("exec_module")(module);
        
        // Register under tensorplay.ops
        nb::object tp = nb::module_::import_("tensorplay");
        if (nb::hasattr(tp, "ops")) {
            tp.attr("ops").attr(name.c_str()) = module;
        }
    }, "path"_a);

    m.def("ones", [](const std::vector<int64_t>& size, std::optional<DType> dtype, std::optional<Device> device, bool requires_grad) {
        return Tensor::ones(size, dtype.value_or(DType::Float32), device.value_or(Device(DeviceType::CPU)), requires_grad);
    }, "size"_a, nb::kw_only(), "dtype"_a = nb::none(), "device"_a = nb::none(), "requires_grad"_a = false,
    "ones(size: Sequence[int], *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");

    m.def("ones", [](nb::args args, nb::kwargs kwargs) {
        DType dtype = DType::Float32;
        Device device = Device(DeviceType::CPU);
        bool requires_grad = false;
        if (kwargs.contains("dtype") && !kwargs["dtype"].is_none()) dtype = nb::cast<DType>(kwargs["dtype"]);
        if (kwargs.contains("device") && !kwargs["device"].is_none()) device = nb::cast<Device>(kwargs["device"]);
        if (kwargs.contains("requires_grad") && !kwargs["requires_grad"].is_none()) requires_grad = nb::cast<bool>(kwargs["requires_grad"]);
        auto shape = parse_shape_args(args);
        return Tensor::ones(shape, dtype, device, requires_grad);
    }, "ones(*size: int, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");

    m.def("eye", &Tensor::eye, "n"_a, "m"_a = -1, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false,
    "eye(n: int, m: int = -1, *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");
    
    m.def("empty", [](const std::vector<int64_t>& size, std::optional<DType> dtype, std::optional<Device> device, bool requires_grad) {
        return Tensor::empty(size, dtype.value_or(DType::Float32), device.value_or(Device(DeviceType::CPU)), requires_grad);
    }, "size"_a, nb::kw_only(), "dtype"_a = nb::none(), "device"_a = nb::none(), "requires_grad"_a = false,
    "empty(size: Sequence[int], *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");

    m.def("empty", [](nb::args args, nb::kwargs kwargs) {
        DType dtype = DType::Float32;
        Device device = Device(DeviceType::CPU);
        bool requires_grad = false;
        if (kwargs.contains("dtype") && !kwargs["dtype"].is_none()) dtype = nb::cast<DType>(kwargs["dtype"]);
        if (kwargs.contains("device") && !kwargs["device"].is_none()) device = nb::cast<Device>(kwargs["device"]);
        if (kwargs.contains("requires_grad") && !kwargs["requires_grad"].is_none()) requires_grad = nb::cast<bool>(kwargs["requires_grad"]);
        return Tensor::empty(parse_shape_args(args), dtype, device, requires_grad);
    }, "empty(*size: int, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");

    m.def("rand", [](const std::vector<int64_t>& size, std::optional<DType> dtype, std::optional<Device> device, bool requires_grad) {
        return Tensor::rand(size, dtype.value_or(DType::Float32), device.value_or(Device(DeviceType::CPU)), requires_grad);
    }, "size"_a, nb::kw_only(), "dtype"_a = nb::none(), "device"_a = nb::none(), "requires_grad"_a = false,
    "rand(size: Sequence[int], *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");

    m.def("rand", [](nb::args args, nb::kwargs kwargs) {
        DType dtype = DType::Float32;
        Device device = Device(DeviceType::CPU);
        bool requires_grad = false;
        if (kwargs.contains("dtype") && !kwargs["dtype"].is_none()) dtype = nb::cast<DType>(kwargs["dtype"]);
        if (kwargs.contains("device") && !kwargs["device"].is_none()) device = nb::cast<Device>(kwargs["device"]);
        if (kwargs.contains("requires_grad") && !kwargs["requires_grad"].is_none()) requires_grad = nb::cast<bool>(kwargs["requires_grad"]);
        return Tensor::rand(parse_shape_args(args), dtype, device, requires_grad);
    }, "rand(*size: int, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");

    m.def("randint", &Tensor::randint, "low"_a, "high"_a, "size"_a, nb::kw_only(), "dtype"_a = DType::Int64, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false,
    "randint(low: int, high: int, size: Sequence[int], *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");

    m.def("randn", [](const std::vector<int64_t>& size, std::optional<DType> dtype, std::optional<Device> device, bool requires_grad) {
        return Tensor::randn(size, dtype.value_or(DType::Float32), device.value_or(Device(DeviceType::CPU)), requires_grad);
    }, "size"_a, nb::kw_only(), "dtype"_a = nb::none(), "device"_a = nb::none(), "requires_grad"_a = false,
    "randn(size: Sequence[int], *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");

    m.def("randn", [](nb::args args, nb::kwargs kwargs) {
        DType dtype = DType::Float32;
        Device device = Device(DeviceType::CPU);
        bool requires_grad = false;
        if (kwargs.contains("dtype") && !kwargs["dtype"].is_none()) dtype = nb::cast<DType>(kwargs["dtype"]);
        if (kwargs.contains("device") && !kwargs["device"].is_none()) device = nb::cast<Device>(kwargs["device"]);
        if (kwargs.contains("requires_grad") && !kwargs["requires_grad"].is_none()) requires_grad = nb::cast<bool>(kwargs["requires_grad"]);
        return Tensor::randn(parse_shape_args(args), dtype, device, requires_grad);
    }, "randn(*size: int, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");

    m.def("randperm", &Tensor::randperm, "n"_a, nb::kw_only(), "dtype"_a = DType::Int64, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false,
    "randperm(n: int, *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");
    
    m.def("full", &Tensor::full, "shape"_a, "fill_value"_a, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false,
    "full(shape: Sequence[int], fill_value: Union[float, int], *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");
    
    // Bind generated functions (includes *_like, transpose, permute, etc.)
    bind_generated_op_functions(m);

    // Config
    m.def("_show_config", &tensorplay::show_config);
    
    // Manual bindings for varargs/complex factories
    m.def("linspace", &Tensor::linspace, "start"_a, "end"_a, "steps"_a, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false,
    "linspace(start: float, end: float, steps: int, *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");
    m.def("logspace", &Tensor::logspace, "start"_a, "end"_a, "steps"_a, "base"_a = 10.0, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false,
    "logspace(start: float, end: float, steps: int, base: float = 10.0, *, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> Tensor");
}
