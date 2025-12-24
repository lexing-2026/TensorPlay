#include "python_bindings.h"
#include "tensorplay/ops/Config.h"
#include "OneDNNContext.h"
#include <cstdlib>

// Extern declarations (if not in header)
void init_scalar(nb::module_& m);

NB_MODULE(_C, m) {
    m.doc() = "The C extension module of tensorplay";

    // Exception translation
    nb::register_exception_translator([](const std::exception_ptr &p, void * /* unused */) {
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
        } catch (const std::exception &e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    // Warning handler
    tensorplay::setWarningHandler([](const tensorplay::SourceLocation& source, const std::string& msg) {
        nb::gil_scoped_acquire gil;
        PyErr_WarnEx(PyExc_UserWarning, msg.c_str(), 1);
    });

    init_dtype(m);
    init_device(m);
    init_scalar(m); // Initialize scalar after DType/Device as it might be used? Actually scalar is independent.
    init_size(m);
    init_generator(m);
    init_tensor(m);
    init_autograd(m);
    init_ops(m);
    init_stax(m);

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

    m.def("set_printoptions", &tensorplay::set_printoptions, 
          "Set print options", 
          nb::arg("edge_items") = -1, 
          nb::arg("threshold") = -1, 
          nb::arg("precision") = -1, 
          nb::arg("linewidth") = -1);

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

    m.def("_add_docstr", [](nb::object obj, const std::string& doc) {
         if (obj.is_none()) {
              return nb::none();
         }
         try {
             if (nb::hasattr(obj, "__doc__")) {
                  nb::setattr(obj, "__doc__", nb::str(doc.c_str()));
             }
         } catch (...) {
             // Ignore errors if docstring cannot be set (e.g. read-only attribute)
         }
         return obj;
     }, nb::arg("obj").none(), nb::arg("doc"), "Adds or replaces the docstring of a Python object.");

    m.def("_set_module_name", [](nb::object obj, const std::string& name) {
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