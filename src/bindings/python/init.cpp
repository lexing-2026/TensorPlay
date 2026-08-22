#include "python_bindings.h"
#include "tensorplay/ops/Config.h"
#include "Context.h"
#include "OneDNNContext.h"
#include <cstdlib>

// Extern declarations (if not in header)
void init_scalar(py::module_& m);

PYBIND11_MODULE(_C, m) {
    m.doc() = "The C extension module of tensorplay";

    // Exception translation
    py::register_exception_translator([](std::exception_ptr p) {
        auto set_error = [](PyObject* type, const tensorplay::Exception& e) {
            std::string msg = e.msg();
            const char* env_val = std::getenv("TENSORPLAY_SHOW_CPP_STACKTRACES");
            if (env_val && std::string(env_val) == "1" && !e.stacktrace().empty()) {
                msg += "\n\n" + e.stacktrace();
            }
            PyErr_SetString(type, msg.c_str());
        };

        try {
            std::rethrow_exception(p);
        } catch (const tensorplay::IndexError &e) {
            set_error(PyExc_IndexError, e);
        } catch (const tensorplay::ValueError &e) {
            set_error(PyExc_ValueError, e);
        } catch (const tensorplay::TypeError &e) {
            set_error(PyExc_TypeError, e);
        } catch (const tensorplay::NotImplementedError &e) {
            set_error(PyExc_NotImplementedError, e);
        } catch (const tensorplay::Exception &e) {
            set_error(PyExc_RuntimeError, e);
        }
        // Generic std::exception (incl. pybind11 internal exceptions like
        // stop_iteration) is left to pybind11's builtin translator, which
        // converts them appropriately (e.g. StopIteration).
    });

    // Warning handler
    tensorplay::setWarningHandler([](const tensorplay::SourceLocation& source, const std::string& msg) {
        py::gil_scoped_acquire gil;
        PyErr_WarnEx(PyExc_UserWarning, msg.c_str(), 1);
    });

    init_dtype(m);
    init_device(m);
    init_scalar(m); // Initialize scalar after DType/Device as it might be used? Actually scalar is independent.
    init_size(m);
    init_generator(m);
    init_tensor(m);
    init_autograd(m);
    init_autocast(m);
    init_ops(m);
    init_stax(m);
    init_parallel(m);
    init_distributed(m);

    // CUDA availability
    m.def("is_cuda_available", []() {
#ifdef TENSORPLAY_USE_CUDA
        return true;
#else
        return false;
#endif
    });

    // Config
    m.def("_show_config", &tensorplay::show_config);
    m.def("_cxx_flags", &tensorplay::_cxx_flags);
    m.def("_parallel_info", &tensorplay::_parallel_info);
    m.def("_get_build_info", &tensorplay::get_build_info);

    // Global default dtype/device (mirrors torch._C bindings of the same
    // names; see torch/csrc/Module.cpp and torch/__init__.py).
    m.def("get_default_dtype", []() {
        return tensorplay::globalContext().defaultDType();
    }, "get_default_dtype() -> DType\n\n"
       "Gets the current default floating point dtype.");

    m.def("_set_default_dtype", [](DType dtype) {
        tensorplay::globalContext().setDefaultDType(dtype);
    }, "dtype"_a);

    m.def("get_default_device", []() {
        return tensorplay::globalContext().defaultDevice();
    }, "get_default_device() -> Device\n\n"
       "Gets the default ``Tensor`` to be allocated on ``device``");

    m.def("_set_default_device", [](std::optional<Device> device) {
        if (device.has_value()) {
            tensorplay::globalContext().setDefaultDevice(device);
        } else {
            tensorplay::globalContext().clearDefaultDevice();
        }
    }, "device"_a.none());

    m.def("_push_default_device", [](const Device& device) {
        tensorplay::globalContext().pushDefaultDevice(device);
    }, "device"_a);
    m.def("_pop_default_device", []() {
        tensorplay::globalContext().popDefaultDevice();
    });

    // Deterministic algorithms (torch/csrc/Module.cpp:
    // _set_deterministic_algorithms / _get_deterministic_algorithms /
    // _get_deterministic_algorithms_warn_only)
    m.def("_set_deterministic_algorithms",
          [](bool mode, bool warn_only) {
              tensorplay::globalContext().setDeterministicAlgorithms(mode, warn_only);
          },
          "mode"_a, py::kw_only(), "warn_only"_a = false);
    m.def("_get_deterministic_algorithms", []() {
        return tensorplay::globalContext().deterministicAlgorithms();
    });
    m.def("_get_deterministic_algorithms_warn_only", []() {
        return tensorplay::globalContext().deterministicAlgorithmsWarnOnly();
    });

    // Float32 matmul precision (torch/csrc/Module.cpp:
    // _set_float32_matmul_precision / _get_float32_matmul_precision)
    m.def("get_float32_matmul_precision", []() {
        return tensorplay::globalContext().getFloat32MatmulPrecisionStr();
    }, "get_float32_matmul_precision() -> str\n\n"
       "Returns the current value of float32 matrix multiplication precision.");
    m.def("_set_float32_matmul_precision", [](const std::string& precision) {
        tensorplay::globalContext().setFloat32MatmulPrecision(precision);
    }, "precision"_a);

    // TF32 flags (torch/csrc/Module.cpp: _get/_set_cublas_allow_tf32,
    // _get/_set_cudnn_allow_tf32)
    m.def("_get_cublas_allow_tf32", []() {
        return tensorplay::globalContext().allowTF32CuBLAS();
    });
    m.def("_set_cublas_allow_tf32", [](bool enabled) {
        tensorplay::globalContext().setAllowTF32CuBLAS(enabled);
    }, "enabled"_a);
    m.def("_get_cudnn_allow_tf32", []() {
        return tensorplay::globalContext().allowTF32CuDNN();
    });
    m.def("_set_cudnn_allow_tf32", [](bool enabled) {
        tensorplay::globalContext().setAllowTF32CuDNN(enabled);
    }, "enabled"_a);

    m.def("set_printoptions", &tensorplay::set_printoptions, 
          "Set print options", 
          py::arg("edge_items") = -1, 
          py::arg("threshold") = -1, 
          py::arg("precision") = -1, 
          py::arg("linewidth") = -1);

    // Backends
    m.def("has_mkldnn", &tensorplay::OneDNNContext::is_available);
    m.def("is_mkldnn_enabled", &tensorplay::OneDNNContext::is_enabled);
    m.def("set_mkldnn_enabled", &tensorplay::OneDNNContext::set_enabled);
    
    m.def("has_mkl", []() {
#ifdef USE_MKL
        return true;
#else
        return false;
#endif
    });
    
    m.def("has_openmp", []() {
#ifdef _OPENMP
        return true;
#else
        return false;
#endif
    });

    m.def("_add_docstr", [](py::object obj, const std::string& doc) -> py::object {
         if (obj.is_none()) {
              return py::none();
         }
         try {
             if (py::hasattr(obj, "__doc__")) {
                  py::setattr(obj, "__doc__", py::str(doc.c_str()));
             }
         } catch (...) {
             // Ignore errors if docstring cannot be set (e.g. read-only attribute)
         }
         return obj;
     }, py::arg("obj").none(), py::arg("doc"), "Adds or replaces the docstring of a Python object.");

    m.def("_set_module_name", [](py::object obj, const std::string& name) {
        PyObject* o = obj.ptr();
        PyObject* name_obj = PyUnicode_FromString(name.c_str());
        if (!name_obj) {
             PyErr_Clear();
             return;
        }
        if (PyObject_SetAttrString(o, "__module__", name_obj) != 0) {
            PyErr_Clear();
        }
        Py_DECREF(name_obj);
    });
}