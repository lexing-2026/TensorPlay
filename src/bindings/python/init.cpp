#include "python_bindings.h"
#include "tensorplay/ops/Config.h"
#include "tensorplay/ops/TensorCPythonGenerated.h"
#include "CPythonBridge.h"
#include "Context.h"
#include "OneDNNContext.h"
#include "Profiler.h"
#include "Graph.h"
#include <cstdlib>
#include <cstring>

// Extern declarations (if not in header)
void init_scalar(py::module_& m);
void init_filecheck(py::module_& m);

namespace tensorplay {
namespace python {

// with_stack hook: called from every generated METH_FASTCALL entry while the
// GIL is held, before its GIL-releasing invoke.  Extracts the caller's full
// Python frame chain (C functions add no frames, so the chain starts at user
// code) and hands plain bytes to the profiler; the next OpRecord on this
// thread adopts it and clears the slot, so composite inner ops record no
// stack instead of inheriting the outermost call's.
void tpx_prof_capture_site() {
    if (!tensorplay::prof::g_active.load(std::memory_order_acquire)) return;
    if (!tensorplay::prof::g_capture_sites.load(std::memory_order_acquire)) {
        return;
    }
    PyFrameObject* frame = PyEval_GetFrame();  // borrowed
    if (frame == nullptr) return;
    std::vector<tensorplay::prof::ProfFrame> frames;
    frames.reserve(8);
    PyFrameObject* owned = nullptr;  // frame refs from PyFrame_GetBack
    int depth = 0;
    while (frame != nullptr && depth < 64) {
        PyCodeObject* code = PyFrame_GetCode(frame);  // new reference
#if PY_VERSION_HEX >= 0x030D0000
        const char* file = PyCode_GetFilename(code);   // borrowed (3.13+)
#else
        // pre-3.13: co_filename is a directly accessible member
        const char* file = PyUnicode_AsUTF8(code->co_filename);
#endif
        // co_name stays an immediate PyCodeObject member through 3.13
        const char* func = PyUnicode_AsUTF8(code->co_name);
        const int line = PyFrame_GetLineNumber(frame);
        frames.push_back({file ? file : "<unknown>",
                          func ? func : "<module>", line});
        Py_DECREF(code);
        PyFrameObject* next = PyFrame_GetBack(frame);  // new reference
        Py_XDECREF(owned);
        owned = next;
        frame = next;
        ++depth;
    }
    Py_XDECREF(owned);
    if (frames.empty()) return;
    tensorplay::prof::set_python_stack(std::move(frames));
}

} // namespace python
} // namespace tensorplay

namespace {

// ---------------------------------------------------------------------------
// Factory fast paths
//
// The generated METH_FASTCALL layer already parses empty/zeros/ones/rand/
// wrappers in front of them cost ~0.5us of frame/checks per call -- which is
// rebinds those public names onto C trampolines:
//
//   * while the native capture state reports an active compiler region the
//     call vectors straight to the original Python wrapper, so
//     capture_call/proxy-scan semantics are preserved bit-for-bit;
//   * otherwise it normalizes just the spellings the generated parser rejects
//     (device strings, lone-int or iterable-only sizes, bare-int fill_value)
//     and vectors directly into the fastcall entry -- zero Python frames.
//
// Called from the tail of tensorplay/functional.py (which owns the wrappers),
// so both refs are resolved eagerly before the fast call entry is installed.
// ---------------------------------------------------------------------------

struct FactoryHook {
    PyObject* raw;      // generated fastcall entry (strong ref)
    PyObject* wrapper;  // original Python functional wrapper (strong ref)
    PyObject* tensor_type;  // tensorplay.Tensor (strong, may be null)
    PyObject* scalar_type;  // tensorplay.Scalar (strong, may be null)
    bool varargs;           // empty family: fold *size positionals
};

bool factory_trace_depth(long* out) {
    *out = static_cast<long>(
        tensorplay::stax::currentCaptureState().compile_depth);
    return true;
}

// A single size argument the generated int[] parser consumes natively.
bool factory_size_seq(PyObject* o) {
    return PyList_Check(o) || PyTuple_Check(o) || PyRange_Check(o);
}

PyObject* factory_trampoline(PyObject* self, PyObject* const* args,
                             Py_ssize_t nargs, PyObject* kwnames) {
    FactoryHook* h = (FactoryHook*)PyCapsule_GetPointer(self, nullptr);
    if (!h) return nullptr;

    long depth;
    if (!factory_trace_depth(&depth) || depth > 0)
        return PyObject_Vectorcall(h->wrapper, args, (size_t)nargs, kwnames);

    Py_ssize_t nkw = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;

    // Classify the positional size spelling.
    PyObject* owned_size = nullptr;   // list built for fold/lift conversions
    Py_ssize_t call_nargs = nargs;
    bool fold_all = false, lift_first = false, listify_single = false;
    if (h->varargs) {
        if (nargs != 1 || !factory_size_seq(args[0])) {
            if (nargs == 1 && PyObject_HasAttrString(args[0], "__iter__")) {
                listify_single = true;   // generator/range-like: consume like list(x)
            } else {
                fold_all = true;         // *size ints (or ()) -> single list
            }
        }
    } else if (nargs >= 1 && PyLong_Check(args[0]) && !PyBool_Check(args[0])) {
        lift_first = true;               // full(3, v) -> full([3], v)
    }

    // full(): numbers/Tensors/Scalars go straight through; anything exotic
    // (numpy scalars, Decimal, ...) takes the wrapper's Scalar() promotion.
    if (!h->varargs && !lift_first && nargs >= 2 && h->tensor_type &&
        h->scalar_type) {
        PyObject* fv = args[nargs - 1];
        if (!PyLong_Check(fv) && !PyFloat_Check(fv) && !PyBool_Check(fv) &&
            Py_TYPE(fv) != (PyTypeObject*)h->tensor_type &&
            Py_TYPE(fv) != (PyTypeObject*)h->scalar_type) {
            PyObject* r = PyObject_Vectorcall(h->wrapper, args,
                                              (size_t)nargs, kwnames);
            return r;
        }
    }

    if (!fold_all && !lift_first && !listify_single)
        return PyObject_Vectorcall(h->raw, args, (size_t)nargs, kwnames);

    // Slow-shaped eager call: build a rewritten argument buffer.
    PyObject* stack[24];
    PyObject** buf = stack;
    PyObject** heap = nullptr;
    if (nargs + nkw + 1 > 24) {
        heap = (PyObject**)PyMem_Malloc(sizeof(PyObject*) * (size_t)(nargs + nkw + 1));
        if (!heap) return PyErr_NoMemory();
        buf = heap;
    }
    Py_ssize_t out = 0;
    if (fold_all) {
        owned_size = PyList_New(nargs);
        if (!owned_size) goto fail;
        for (Py_ssize_t i = 0; i < nargs; ++i) {
            Py_INCREF(args[i]);
            PyList_SET_ITEM(owned_size, i, args[i]);
        }
        buf[out++] = owned_size;
    } else if (lift_first) {
        owned_size = PyList_New(1);
        if (!owned_size) goto fail;
        Py_INCREF(args[0]);
        PyList_SET_ITEM(owned_size, 0, args[0]);
        buf[out++] = owned_size;
        for (Py_ssize_t i = 1; i < nargs; ++i) buf[out++] = args[i];
    } else if (listify_single) {
        owned_size = PySequence_List(args[0]);
        if (!owned_size) goto fail;
        buf[out++] = owned_size;
    } else {
        for (Py_ssize_t i = 0; i < nargs; ++i) buf[out++] = args[i];
    }
    for (Py_ssize_t j = 0; j < nkw; ++j)
        buf[out++] = args[nargs + j];

    {
        // nargsf counts POSITIONALS only; kw values trail after them and are
        // counted via kwnames.
        PyObject* r = PyObject_Vectorcall(h->raw, buf, (size_t)(out - nkw), kwnames);
        Py_XDECREF(owned_size);
        PyMem_Free(heap);
        return r;
    }

fail:
    Py_XDECREF(owned_size);
    PyMem_Free(heap);
    return nullptr;
}

PyMethodDef factory_defs[] = {
    {"empty",  (PyCFunction)(void(*)(void))factory_trampoline,
        METH_FASTCALL | METH_KEYWORDS, nullptr},
    {"zeros",  (PyCFunction)(void(*)(void))factory_trampoline,
        METH_FASTCALL | METH_KEYWORDS, nullptr},
    {"ones",   (PyCFunction)(void(*)(void))factory_trampoline,
        METH_FASTCALL | METH_KEYWORDS, nullptr},
    {"rand",   (PyCFunction)(void(*)(void))factory_trampoline,
        METH_FASTCALL | METH_KEYWORDS, nullptr},
    {"randn",  (PyCFunction)(void(*)(void))factory_trampoline,
        METH_FASTCALL | METH_KEYWORDS, nullptr},
    {"full",   (PyCFunction)(void(*)(void))factory_trampoline,
        METH_FASTCALL | METH_KEYWORDS, nullptr},
};
const char* factory_names[] = {"empty", "zeros", "ones",
                               "rand", "randn", "full"};

int install_factory_fast_paths_impl(py::module_& m, py::dict wrappers) {
    static bool done = false;
    if (done) return 0;
    done = true;

    PyObject* tp_mod = PyImport_ImportModule("tensorplay");
    PyObject* tensor_type = nullptr, *scalar_type = nullptr;
    if (tp_mod) {
        tensor_type = PyObject_GetAttrString(tp_mod, "Tensor");
        if (!tensor_type) PyErr_Clear();
        scalar_type = PyObject_GetAttrString(tp_mod, "Scalar");
        if (!scalar_type) PyErr_Clear();
        Py_DECREF(tp_mod);
    } else {
        PyErr_Clear();
    }

    for (size_t i = 0; i < sizeof(factory_names)/sizeof(factory_names[0]); ++i) {
        PyObject* raw = PyObject_GetAttrString(m.ptr(), factory_names[i]);
        if (!raw) { PyErr_Clear(); continue; }  // unbound op: leave as-is
        PyObject* w = PyDict_GetItemString(wrappers.ptr(), factory_names[i]);
        if (!w) { Py_DECREF(raw); continue; }

        FactoryHook* h = new FactoryHook();
        h->raw = raw;                       // ownership moved from GetAttr
        h->wrapper = Py_NewRef(w);
        h->tensor_type = tensor_type ? Py_NewRef(tensor_type) : nullptr;
        h->scalar_type = scalar_type ? Py_NewRef(scalar_type) : nullptr;
        h->varargs = (strcmp(factory_names[i], "full") != 0);

        PyObject* cap = PyCapsule_New((void*)h, nullptr,
                                      [](PyObject* c) {
            auto* hh = (FactoryHook*)PyCapsule_GetPointer(c, nullptr);
            if (hh) {
                Py_XDECREF(hh->raw);
                Py_XDECREF(hh->wrapper);
                Py_XDECREF(hh->tensor_type);
                Py_XDECREF(hh->scalar_type);
                delete hh;
            }
        });
        if (!cap) { PyErr_Clear(); continue; }
        PyObject* fn = PyCFunction_New(&factory_defs[i], cap);
        Py_DECREF(cap);  // fn holds the self reference
        if (!fn) { PyErr_Clear(); continue; }
        // Inherit the generated entry's docstring (best effort).
        PyObject* doc = PyObject_GetAttrString(raw, "__doc__");
        if (doc) {
            if (PyObject_SetAttrString(fn, "__doc__", doc) != 0) PyErr_Clear();
            Py_DECREF(doc);
        } else {
            PyErr_Clear();
        }
        std::string fast_name = std::string(factory_names[i]) + "_fast";
        if (PyObject_SetAttrString(m.ptr(), fast_name.c_str(), fn) != 0)
            PyErr_Clear();
        Py_DECREF(fn);
    }
    Py_XDECREF(tensor_type);
    Py_XDECREF(scalar_type);
    return 0;
}

} // anonymous namespace

PYBIND11_MODULE(_C, m) {
    m.doc() = "The C extension module of tensorplay";

    // Catchable device-mismatch exception (RuntimeError subclass), so users
    // FakeTensorDeviceMismatchError but for real tensors. Auto re-exported at
    // top level by tensorplay/__init__.py.
    tensorplay::set_device_mismatch_error_type(
        py::register_exception<tensorplay::DeviceMismatchError>(
            m, "DeviceMismatchError", PyExc_RuntimeError)
            .ptr());

    // Exception translation
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            std::rethrow_exception(p);
        } catch (const tensorplay::Exception &e) {
            std::string msg = e.msg();
            const char* env_val = std::getenv("TENSORPLAY_SHOW_CPP_STACKTRACES");
            if (env_val && std::string(env_val) == "1" && !e.stacktrace().empty()) {
                msg += "\n\n" + e.stacktrace();
            }
            PyErr_SetString(translate_exception(e), msg.c_str());
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
    init_symint(m);
    init_size(m);
    init_generator(m);
    init_storage(m);
    init_tensor(m);
    init_autograd(m);
    init_autocast(m);
    init_transforms(m);
    init_ops(m);
    init_dispatch(m);
    init_stax(m);
    init_parallel(m);
    init_distributed(m);
    init_futures(m);
    init_rpc(m);
    init_distributed_autograd(m);
    init_filecheck(m);
    init_cuda_graph(m);

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

    // Python dispatch state is thread-local and is consumed by both the
    // generated fastcall entries and the public override helpers.
    m.def("_get_tensor_function_state", []() {
        return tensorplay::python_c::tpx_py_get_function_state();
    });
    m.def("_set_tensor_function_state", [](int state) {
        if (!tensorplay::python_c::tpx_py_set_function_state(state)) {
            throw py::error_already_set();
        }
    }, "state"_a);
    m.def("_exchange_tensor_function_skip_next", [](bool value) {
        return tensorplay::python_c::tpx_py_exchange_skip_next(value);
    }, "value"_a);
    m.def("_peek_tensor_function_skip_next", []() {
        return tensorplay::python_c::tpx_py_peek_skip_next();
    });
    m.def("_exchange_tensor_subclass_skip_next", [](bool value) {
        return tensorplay::python_c::tpx_py_exchange_subclass_skip_next(value);
    }, "value"_a);
    m.def("_peek_tensor_subclass_skip_next", []() {
        return tensorplay::python_c::tpx_py_peek_subclass_skip_next();
    });
    m.def("_get_tensor_dispatch_layer", []() {
        return tensorplay::python_c::tpx_py_get_dispatch_layer();
    });
    m.def("_push_tensor_function_mode", [](py::object mode) {
        tensorplay::python_c::tpx_py_push_function_mode(mode.ptr());
    }, "mode"_a);
    m.def("_pop_tensor_function_mode", []() {
        PyObject* mode = tensorplay::python_c::tpx_py_pop_function_mode();
        if (mode == nullptr) throw py::error_already_set();
        return py::reinterpret_steal<py::object>(mode);
    });
    m.def("_get_tensor_function_mode", [](Py_ssize_t index) {
        PyObject* mode = tensorplay::python_c::tpx_py_get_function_mode(index);
        if (mode == nullptr) throw py::error_already_set();
        return py::reinterpret_steal<py::object>(mode);
    }, "index"_a);
    m.def("_len_tensor_function_mode", []() {
        return tensorplay::python_c::tpx_py_function_mode_len();
    });
    m.def("_is_tensor_function_mode_enabled", []() {
        return tensorplay::python_c::tpx_py_get_function_state() !=
                   tensorplay::python_c::TPX_ALL_DISABLED &&
               tensorplay::python_c::tpx_py_function_mode_len() != 0;
    });

    m.def("_get_nnpack_enabled", []() {
        return tensorplay::globalContext().userEnabledNNPACK();
    });
    m.def("_set_nnpack_enabled", [](bool e) {
        tensorplay::globalContext().setUserEnabledNNPACK(e);
    });
    m.def("_get_mkldnn_enabled", []() {
        return tensorplay::globalContext().userEnabledMkldnn();
    });
    m.def("_set_mkldnn_enabled", [](bool e) {
        tensorplay::globalContext().setUserEnabledMkldnn(e);
    });

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

    // _set_float32_matmul_precision / _get_float32_matmul_precision)
    m.def("get_float32_matmul_precision", []() {
        return tensorplay::globalContext().getFloat32MatmulPrecisionStr();
    }, "get_float32_matmul_precision() -> str\n\n"
       "Returns the current value of float32 matrix multiplication precision.");
    m.def("_set_float32_matmul_precision", [](const std::string& precision) {
        tensorplay::globalContext().setFloat32MatmulPrecision(precision);
    }, "precision"_a);

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
    m.def("_get_cudnn_benchmark", []() {
        return tensorplay::globalContext().cudnnBenchmark();
    });
    m.def("_set_cudnn_benchmark", [](bool enabled) {
        tensorplay::globalContext().setCudnnBenchmark(enabled);
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

    using tensorplay::prof::Event;
    m.def("_profiler_start", [](bool capture_shapes, bool with_stack,
                                bool gpu_timing, bool gpu_trace,
                                bool mem_capture) {
        // This is a session option, not a process-global sticky mode.  In
        // particular, a later CPU profile must not arm CUDA events for CPU
        // redispatches after a timed CUDA profile has finished.
        tensorplay::prof::g_gpu_timing.store(
            gpu_timing, std::memory_order_release);
        tensorplay::prof::g_mem_capture.store(
            mem_capture, std::memory_order_release);
        if (gpu_trace) {
            // Arm CUPTI before the first op so no kernel is missed; flag is
            // stored first because GpuTimerPair::arm consults it from the
            // very first redispatch of the session.
            tensorplay::prof::g_gpu_trace.store(true,
                                                std::memory_order_release);
            if (!tensorplay::prof::cupti_start()) {
                tensorplay::prof::g_gpu_trace.store(
                    false, std::memory_order_release);
                const std::string reason =
                    tensorplay::prof::cupti_last_error();
                PyErr_WarnEx(PyExc_RuntimeWarning,
                             ("gpu_trace unavailable: " + reason).c_str(),
                             1);
            }
        }
        if (with_stack) {
            tensorplay::prof::profiler_start_full();
        } else if (capture_shapes) {
            tensorplay::prof::profiler_start_with_shapes();
        } else {
            tensorplay::prof::profiler_start();
        }
    }, "capture_shapes"_a = false, "with_stack"_a = false,
       "gpu_timing"_a = false, "gpu_trace"_a = false,
       "mem_capture"_a = false);
    // Returns (op_events, gpu_activities, mem_events).
    //   op_events: (name, kind, start_ns, end_ns, tid, shapes|None,
    //     dtypes|None, site_str|None, gpu_ms, out_bytes, stack|None,
    //     kernel_count) tuples ordered by start.
    //   gpu_activities: (name, kind, start_ns, end_ns, device, stream,
    //     correlation, external_id, tid, cbid, bytes, copy_kind, value).
    //   mem_events: (ts_ns, ptr, bytes, is_alloc, is_cuda, device, stream,
    //     tid).
    m.def("_profiler_stop", []() {
        py::gil_scoped_release release;
        std::vector<Event> events = tensorplay::prof::profiler_stop();
        // Resolve GPU pairs with bounded waits on the recorded stream tails;
        // this deliberately avoids a device-wide synchronize and writes
        // gpu_ms into the events.
        tensorplay::prof::gpu_resolve_all(
            events, [](Event&, float) {});
        tensorplay::prof::g_gpu_timing.store(false,
                                             std::memory_order_release);
        const bool trace_on =
            tensorplay::prof::g_gpu_trace.exchange(false,
                std::memory_order_acq_rel);
        std::vector<tensorplay::prof::GpuActivity> gpu_acts;
        if (trace_on) {
            tensorplay::prof::cupti_stop_and_collect(gpu_acts);
            // Correlate GPU activity back to the op that launched it via
            // the external-correlation id (the OpRecord slot).
            for (auto& a : gpu_acts) {
                if (a.kind == 'r' || a.kind == 'd') continue;
                if (a.external_id == tensorplay::prof::GpuActivity::kNoExt ||
                    a.external_id >= events.size()) {
                    continue;
                }
                auto& op = events[a.external_id];
                const double ms =
                    static_cast<double>(a.end_ns - a.start_ns) / 1e6;
                op.gpu_ms = (op.gpu_ms < 0.f ? 0.f : op.gpu_ms) +
                            static_cast<float>(ms);
                op.kernel_count += 1;
            }
        }
        std::vector<tensorplay::prof::MemEvent> mem_events =
            tensorplay::prof::mem_take();
        py::gil_scoped_acquire acquire;
        py::list out_ops;
        for (const auto& e : events) {
            py::object shapes = py::none();
            if (e.shapes) {
                py::list sh;
                for (const auto& s : *e.shapes) sh.append(s);
                shapes = std::move(sh);
            }
            py::object dtypes = py::none();
            if (e.dtypes) {
                dtypes = py::cast(*e.dtypes);
            }
            py::object site = py::none();
            if (e.site_id != Event::kNoSite) {
                site = py::str(tensorplay::prof::site_string(e.site_id));
            }
            py::object stack = py::none();
            if (e.stack_id != Event::kNoSite) {
                py::list fr;
                for (const auto& f : tensorplay::prof::stack_frames(e.stack_id)) {
                    fr.append(f.file + ":" + std::to_string(f.line) +
                              " (" + f.func + ")");
                }
                stack = std::move(fr);
            }
            // FLOP estimate for the aggregation view: computed here (once
            // per event, off the hot path) from the already-captured input
            // shapes; 0 when shapes are absent or the op is not covered.
            const int64_t flops =
                e.shapes ? tensorplay::prof::estimate_flops(e.name, *e.shapes)
                         : 0;
            out_ops.append(py::make_tuple(
                std::string(e.name), char(e.kind), e.start_ns, e.end_ns,
                e.tid, std::move(shapes), std::move(dtypes),
                std::move(site), e.gpu_ms, e.out_bytes,
                std::move(stack), e.kernel_count, flops));
        }
        py::list out_gpu;
        for (const auto& a : gpu_acts) {
            out_gpu.append(py::make_tuple(
                std::string(a.name), char(a.kind), a.start_ns, a.end_ns,
                a.device, a.stream, a.correlation, a.external_id,
                a.thread_id, a.cbid, a.bytes, a.copy_kind, a.value));
        }
        py::list out_mem;
        for (const auto& e : mem_events) {
            out_mem.append(py::make_tuple(
                e.ts_ns, reinterpret_cast<uintptr_t>(e.ptr), e.bytes,
                e.alloc, e.cuda, e.device, e.stream, e.tid));
        }
        return py::make_tuple(std::move(out_ops), std::move(out_gpu),
                              std::move(out_mem));
    });
    m.def("_profiler_is_active", []() {
        // One relaxed-enough atomic load; gates Python-side span emission
        // (custom-op wrappers) at zero cost when no session runs.
        return tensorplay::prof::g_active.load(std::memory_order_acquire);
    });
    m.def("_profiler_user_begin", [](const std::string& name) {
        // Capture the caller's frame while we still hold the GIL; the span
        // adopts it like any op would.
        tensorplay::python::tpx_prof_capture_site();
        tensorplay::prof::user_span_begin(name);
    });
    m.def("_profiler_user_end", []() {
        tensorplay::prof::user_span_end();
    });
    m.def("_profiler_emit_nvtx", [](bool on) {
        tensorplay::prof::g_emit_nvtx.store(on,
                                            std::memory_order_release);
    });
    m.def("_profiler_emit_itt", [](bool on) {
        tensorplay::prof::g_emit_itt.store(on,
                                           std::memory_order_release);
    });
    // CUPTI library version for trace-export schema metadata (0 = n/a).
    m.def("_profiler_cupti_version", []() {
        return static_cast<int64_t>(tensorplay::prof::cupti_version());
    });

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

    // METH_FASTCALL function layer goes in LAST, after every pybind11
    // binding above: it only fills names nothing else bound and must never
    // shadow a hand-written overload.  TP_NO_FASTCALL=1 disables it (escape
    // hatch while a py3.12-specific crash in the bridge is investigated).
    if (getenv("TP_NO_FASTCALL") == nullptr) {
        if (tensorplay::python_c::register_generated_cpython_functions(m.ptr()) != 0) {
            throw py::error_already_set();
        }
        // The op-functions submodule carries the same fill-only layer so the
        // METH_FASTCALL-bound ops are reachable from both module surfaces.
        if (py::hasattr(m, "_VariableFunctions")) {
            py::object variable_functions = m.attr("_VariableFunctions");
            if (tensorplay::python_c::register_generated_cpython_functions(
                    variable_functions.ptr()) != 0) {
                throw py::error_already_set();
            }
        }
    }

    // Factory fast paths: opted into from the tail of functional.py, which
    // hands over the Python wrappers the trampolines divert to under capture.
    m.def("install_factory_fast_paths",
          [](py::module_& mod, py::dict wrappers) {
              install_factory_fast_paths_impl(mod, wrappers);
          }, "mod"_a, "wrappers"_a);
}
