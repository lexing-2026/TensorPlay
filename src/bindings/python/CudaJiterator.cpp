#include "python_bindings.h"

#ifdef USE_CUDA
#include "backend/cuda/JitUtils.h"
#endif

#include <variant>

namespace py = pybind11;

#ifdef USE_CUDA

namespace {

tensorplay::cuda::jit::JitExtraArg parse_extra_arg(const py::handle& value) {
    if (py::isinstance<py::bool_>(value)) {
        return {"bool", value.cast<bool>()};
    }
    if (py::isinstance<py::int_>(value)) {
        return {"long long", value.cast<long long>()};
    }
    if (py::isinstance<py::float_>(value)) {
        return {"double", value.cast<double>()};
    }
    throw py::type_error("jiterator extra arguments must be bool, int or float");
}

}  // namespace

void init_cuda_jiterator(py::module_& m) {
    m.def(
        "_cuda_jiterator_compile_and_launch_kernel",
        [](const std::string& code_string, const std::string& kernel_name,
           bool return_by_ref, int num_outputs, const py::tuple& tensors_py,
           const py::dict& kwargs) -> py::object {
            std::vector<Tensor> tensors;
            tensors.reserve(tensors_py.size());
            for (const py::handle handle : tensors_py) {
                const Tensor& t = handle.cast<const Tensor&>();
                TP_CHECK(t.device().is_cuda(),
                         "jiterator inputs must live on a CUDA device");
                tensors.push_back(t);
            }

            std::vector<tensorplay::cuda::jit::JitExtraArg> extra_args;
            for (const auto& item : kwargs) {
                extra_args.push_back(parse_extra_arg(item.second));
            }

            std::vector<Tensor> outputs =
                tensorplay::cuda::compile_and_launch_jiterator(
                    code_string, kernel_name, num_outputs, tensors,
                    extra_args, return_by_ref);

            if (num_outputs == 1) {
                return py::cast(outputs[0]);
            }
            py::tuple result(num_outputs);
            for (int i = 0; i < num_outputs; ++i) {
                result[i] = py::cast(outputs[static_cast<size_t>(i)]);
            }
            return result;
        },
        py::arg("code_string"), py::arg("kernel_name"),
        py::arg("return_by_ref"), py::arg("num_outputs"), py::arg("tensors"),
        py::arg("kwargs"));
}

#else

void init_cuda_jiterator(py::module_& m) {
    m.def(
        "_cuda_jiterator_compile_and_launch_kernel",
        [](const py::object&, const py::object&, const py::object&,
           const py::object&, const py::object&, const py::object&) {
            throw py::runtime_error(
                "jiterator requires a CUDA-enabled TensorPlay build");
        },
        py::arg("code_string"), py::arg("kernel_name"),
        py::arg("return_by_ref"), py::arg("num_outputs"), py::arg("tensors"),
        py::arg("kwargs"));
}

#endif  // USE_CUDA
