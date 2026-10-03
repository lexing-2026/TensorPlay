#include "python_bindings.h"
#include "tensorplay/ops/TensorBindingsGenerated.h"
#include "tensorplay/ops/TensorCPythonGenerated.h"
#include "Dispatcher.h"
#include "Graph.h"
#include "Context.h"
#include "PythonUtils.h"
#include <filesystem>
#include <cctype>
#include <mutex>
#include <unordered_map>
#include <optional>

using namespace tensorplay::python;

// Declaration of create_tensor (defined in Tensor.cpp)
Tensor create_tensor(py::object data, std::optional<DType> dtype, std::optional<Device> device);

namespace {
}

namespace {

// ---------------------------------------------------------------------------
//
// Kernels registered from Python via ``tensorplay.library`` are mirrored into
// the p10 Dispatcher under their qualified ``"ns::op"`` name so native code
// (and the bindings below) can resolve and invoke them through the real
// dispatch path -- DispatchTable lookup plus the autocast choke point --
// instead of bypassing it.  The canonical unboxed calling convention for
// Python-backed operators is
//     std::vector<Tensor>(const std::vector<Tensor>&)
// i.e. tensors in, tensors out; scalar arguments stay on the Python dispatch
// ---------------------------------------------------------------------------

struct PyOpKernelEntry {
    py::object cpu;
    py::object cuda;
    py::object composite;  // device-agnostic kernel (device_types=None)
};

// Kernel slots for Python-backed operators.  Every reader and writer runs
// under the GIL -- all entry points live in this file and acquire it before
// touching the map -- so the plain map carries no lock of its own and a
// rehash during registration cannot interleave with a kernel selection.
// Leaky by design: the dispatcher table (also process-lifetime) keeps raw
// trampoline pointers.
PyOpKernelEntry* py_op_entry(const std::string& op_name) {
    static auto* map = new std::unordered_map<std::string, PyOpKernelEntry>();
    auto it = map->find(op_name);
    if (it == map->end()) {
        it = map->emplace(op_name, PyOpKernelEntry{}).first;
    }
    return &it->second;
}

// The trampoline cannot receive per-op state through the type-erased
// KernelFunction pointer, so the caller names the operator through this
// thread-local right before invoking DispatchStub.  All entry points live in
// this translation unit and hold the GIL, making the handoff atomic.
thread_local std::string t_active_python_op;

py::object select_py_kernel(const std::string& op_name, const std::vector<Tensor>& inputs) {
    const bool is_cuda = !inputs.empty() && inputs[0].device().is_cuda();
    PyOpKernelEntry* entry = py_op_entry(op_name);
    py::object fn = (is_cuda ? entry->cuda : entry->cpu);
    // A device-specific kernel shadows the composite slot; an operator
    // registered without device_types covers every backend, matching
    if (!fn) {
        fn = entry->composite;
    }
    if (!fn) {
        TP_THROW(NotImplementedError,
            "Python kernel not found for op: ", op_name,
            is_cuda ? " on CUDA" : " on CPU");
    }
    return fn;
}

std::vector<Tensor> tensor_results_from_python(py::object result) {
    if (!result) {
        throw py::error_already_set();
    }
    // Single-output kernels may return a bare Tensor.
    if (py::isinstance<Tensor>(result)) {
        return {result.cast<Tensor>()};
    }
    return py::cast<std::vector<Tensor>>(result);
}

// Acquires the GIL only when the calling thread does not already hold it;
// every other caller pays one thread-state check instead of the full
// ensure/release pair.
struct GilIfExternal {
    const bool held = PyGILState_Check() == 1;
    PyGILState_STATE state;
    GilIfExternal() : state(held ? PyGILState_UNLOCKED : PyGILState_Ensure()) {}
    ~GilIfExternal() {
        if (!held) {
            PyGILState_Release(state);
        }
    }
};

std::vector<Tensor> invoke_python_kernel_by_name(
    const std::string& op_name,
    const std::vector<Tensor>& inputs) {
    GilIfExternal gil;
    py::object fn = select_py_kernel(op_name, inputs);
    // User kernels take per-tensor parameters, matching their signature at
    // registration time; results come back as one or many tensors.
    py::tuple py_args = py::cast(inputs);
    py::object result = py::reinterpret_steal<py::object>(
        PyObject_Call(fn.ptr(), py_args.ptr(), nullptr));
    return tensor_results_from_python(std::move(result));
}

py::object resolve_python_eager_call(const std::string& op_name) {
    static auto* cache = new std::unordered_map<std::string, py::object>();
    auto it = cache->find(op_name);
    if (it != cache->end()) {
        return it->second;
    }
    py::object lib = py::module_::import("tensorplay.library");
    py::object op = lib.attr("get_op")(op_name);
    py::object callable = op.attr("_eager_call");
    auto inserted = cache->emplace(op_name, std::move(callable)).first;
    return inserted->second;
}

std::vector<Tensor> python_op_trampoline(const std::vector<Tensor>& inputs) {
    // A reference: the name lives in TLS and nothing between here and the
    // last read can dispatch another operator to overwrite it.
    const std::string& op_name = t_active_python_op;
    if (op_name.empty()) {
        TP_THROW(RuntimeError,
            "Python op trampoline invoked without an active operator name");
    }
    return invoke_python_kernel_by_name(op_name, inputs);
}

// Executor for stax native-graph "custom_op" nodes: routes through the
// Python operator entry so device dispatch AND register_autograd semantics
// Autograd key the same way).
void ensure_stax_custom_op_executor() {
    static bool installed = false;
    if (installed) {
        return;
    }
    tensorplay::stax::setCustomOpExecutor(
        [](const std::string& op_name,
           const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
            GilIfExternal gil;
            py::object result;
            if (op_name.rfind("tp_stax::pointwise_", 0) == 0) {
                py::object kernel = select_py_kernel(op_name, inputs);
                py::tuple py_args(inputs.size());
                for (size_t i = 0; i < inputs.size(); ++i) {
                    py_args[i] = py::cast(inputs[i]);
                }
                result = py::reinterpret_steal<py::object>(
                    PyObject_Call(kernel.ptr(), py_args.ptr(), nullptr));
            } else {
                py::object callable = resolve_python_eager_call(op_name);
                py::tuple tensor_args(inputs.size());
                for (size_t i = 0; i < inputs.size(); ++i) {
                    tensor_args[i] = py::cast(inputs[i]);
                }
                py::tuple call_args(2);
                call_args[0] = std::move(tensor_args);
                call_args[1] = py::dict();
                result = py::reinterpret_steal<py::object>(
                    PyObject_Call(callable.ptr(), call_args.ptr(), nullptr));
            }
            return tensor_results_from_python(std::move(result));
        });
    installed = true;
}

// Registration slots for Python-backed operators: concrete backend keys
// occupy dispatcher entries, the composite slot holds device-agnostic
// kernels that never reach the dispatcher.
enum class PyOpSlot { CPU, CUDA, COMPOSITE };

PyOpSlot parse_bridge_key(const std::string& device_type) {
    std::string lowered;
    lowered.reserve(device_type.size());
    for (char c : device_type) {
        lowered.push_back(static_cast<char>(::tolower(static_cast<unsigned char>(c))));
    }
    if (lowered == "cpu") return PyOpSlot::CPU;
    if (lowered == "cuda") return PyOpSlot::CUDA;
    if (lowered == "default" || lowered == "composite" ||
        lowered == "compositeexplicitautograd" ||
        lowered == "compositeimplicitautograd") {
        return PyOpSlot::COMPOSITE;
    }
    TP_THROW(ValueError,
        "native bridge supports CPU/CUDA/composite kernels, got: ",
        device_type);
}

} // namespace

namespace {

// Device-key selection for the Python hot path: the first tensor argument
// decides which registered kernel serves the call.  One crossing returns
// the device type as an interned lowercase string ("cpu", "cuda", ...),
// replacing a per-argument Python attribute walk.  The name cache is
// process-lifetime, matching the other leaky registries in this file.
py::object first_device_key(py::args args) {
    constexpr int kDeviceTypeSlots = 16;
    static PyObject* interned[kDeviceTypeSlots] = {};
    for (py::handle item : args) {
        if (!py::isinstance<Tensor>(item)) {
            continue;
        }
        const auto type = static_cast<int>(
            item.cast<const Tensor&>().device().type());
        if (type < 0 || type >= kDeviceTypeSlots) {
            break;
        }
        PyObject* name = interned[type];
        if (name == nullptr) {
            name = PyUnicode_InternFromString(
                tensorplay::Device(static_cast<tensorplay::DeviceType>(type))
                    .toString()
                    .c_str());
            interned[type] = name;
        }
        return py::reinterpret_borrow<py::object>(name);
    }
    return py::none();
}

// tensor() is a flagship constructor, so its argument errors and repr are
// part of the public face.  The pybind11 typed-arg surface answers a mismatch
// with an aggregate "incompatible function arguments" dump that spells out
// internal type names, and its function-record object leaks a mangled helper
// type into repr.  A plain METH_FASTCALL entry with the bridge's shared
// parser keeps the same accepted argument surface while errors quote only
// the public argument names and repr reads as a built-in function.
PyObject* tensor_fastcall(PyObject*, PyObject* const* args, Py_ssize_t nargs,
                          PyObject* kwnames) {
    try {
        static const char* kwlist[] = {"data", "dtype", "device", "pin_memory",
                                       "requires_grad", nullptr};
        if (nargs > 1) {
            throw std::invalid_argument("tensor: too many positional arguments");
        }
        PyObject* slots[5];
        tensorplay::python_c::tpx_py_parse_into(args, nargs, kwnames, kwlist, 5,
                                                "tensor", slots);
        if (slots[0] == nullptr) {
            throw std::invalid_argument(
                "tensor: missing required argument \"data\"");
        }
        py::object data = py::reinterpret_borrow<py::object>(slots[0]);
        std::optional<DType> dtype;
        if (slots[1] != nullptr && slots[1] != Py_None) {
            if (!py::isinstance<DType>(py::handle(slots[1]))) {
                throw std::invalid_argument(
                    std::string("tensor: argument 'dtype' must be dtype, not ")
                    + Py_TYPE(slots[1])->tp_name);
            }
            dtype = py::reinterpret_borrow<py::object>(slots[1]).cast<DType>();
        }
        std::optional<Device> device;
        if (slots[2] != nullptr && slots[2] != Py_None) {
            if (!PyUnicode_Check(slots[2])
                && !py::isinstance<Device>(py::handle(slots[2]))) {
                throw std::invalid_argument(
                    std::string("tensor: argument 'device' must be Device, not ")
                    + Py_TYPE(slots[2])->tp_name);
            }
            device = tensorplay::python_c::tpx_py_device(slots[2]);
        }
        auto bool_arg = [&](PyObject* slot, const char* name) {
            if (slot == nullptr || slot == Py_False) return false;
            if (slot == Py_True) return true;
            throw std::invalid_argument(
                std::string("tensor: argument '") + name + "' must be bool, not "
                + Py_TYPE(slot)->tp_name);
        };
        bool pin_memory = bool_arg(slots[3], "pin_memory");
        bool requires_grad = bool_arg(slots[4], "requires_grad");
        Tensor t = create_tensor(data, dtype, device);
        if (pin_memory) t = Tensor(t.pin_memory());
        if (requires_grad) {
            tensorplay::tpx::impl::set_requires_grad(t, true);
        }
        return tensorplay::python_c::tpx_py_wrap(t);
    } catch (const std::exception& e) {
        tensorplay::python_c::tpx_py_set_error(e);
        return nullptr;
    }
}

PyMethodDef tensor_def = {
    "tensor", (PyCFunction)(void*)tensor_fastcall,
    METH_FASTCALL | METH_KEYWORDS,
    "tensor(data, *, dtype: Optional[DType] = None, device: Optional[Device] "
    "= None, pin_memory: bool = False, requires_grad: bool = False) -> Tensor"};

}  // namespace

void init_ops(py::module_& m) {
    // Module functions
    static PyObject* module_name = PyUnicode_InternFromString("tensorplay._C");
    m.add_object("tensor", py::reinterpret_steal<py::object>(
        PyCFunction_NewEx(&tensor_def, nullptr, module_name)));


    // Python implementation is functionally correct but spends most of a
    // small multi-tensor optimizer step in per-element Python list/dict work.
    // Keep the same contract here: grouping is keyed by the first tensor list;
    // later lists may be empty or contain None, and their original positions
    // are retained when with_indices is requested.
    m.def("_group_tensors_by_device_and_dtype",
          [](py::object nested_object, bool with_indices) {
              if (!PySequence_Check(nested_object.ptr())) {
                  TP_THROW(TypeError,
                      "Expected a sequence of nested tensor lists");
              }
              const py::sequence nested =
                  py::reinterpret_borrow<py::sequence>(nested_object);
              if (py::len(nested) == 0 || py::len(nested[0]) == 0) {
                  TP_THROW(ValueError,
                      "Expected the first nested tensor list to be non-empty");
              }
              std::vector<py::sequence> sources;
              sources.reserve(static_cast<size_t>(py::len(nested)));
              for (const py::handle item : nested) {
                  if (!PySequence_Check(item.ptr())) {
                      TP_THROW(TypeError,
                          "Expected every nested tensor list to be a sequence");
                  }
                  sources.push_back(
                      py::reinterpret_borrow<py::sequence>(item));
              }
              const size_t num_tensors =
                  static_cast<size_t>(py::len(sources[0]));
              for (size_t list_index = 1; list_index < sources.size();
                   ++list_index) {
                  const size_t size =
                      static_cast<size_t>(py::len(sources[list_index]));
                  if (size != 0 && size != num_tensors) {
                      TP_THROW(ValueError,
                          "Expected every nested tensor list to have the same "
                          "length as the first list or to be empty");
                  }
              }

              struct Group {
                  Device device;
                  DType dtype;
                  std::vector<size_t> indices;
              };
              std::vector<Group> groups;
              groups.reserve(2);
              for (size_t tensor_index = 0; tensor_index < num_tensors;
                   ++tensor_index) {
                  const py::handle first_object = sources[0][tensor_index];
                  if (first_object.is_none()) {
                      TP_THROW(ValueError,
                          "Tensors of the first list of nested Tensor lists "
                          "are supposed to be defined");
                  }
                  const Tensor& first = py::cast<const Tensor&>(first_object);
                  const Device device = first.device();
                  const DType dtype = first.dtype();
                  size_t group_index = 0;
                  for (; group_index < groups.size(); ++group_index) {
                      if (groups[group_index].device == device &&
                          groups[group_index].dtype == dtype) {
                          break;
                      }
                  }
                  if (group_index == groups.size()) {
                      groups.push_back(Group{device, dtype, {}});
                  }
                  groups[group_index].indices.push_back(tensor_index);
              }

              py::dict result;
              for (const auto& group : groups) {
                  py::list grouped_lists;
                  for (const auto& source : sources) {
                      py::list grouped;
                      if (py::len(source) != 0) {
                          for (const size_t tensor_index : group.indices) {
                              const py::handle value = source[tensor_index];
                              if (!value.is_none()) {
                                  // Reuse the original Python Tensor wrapper;
                                  // constructing a fresh py::cast(Tensor)
                                  // here costs more than the native grouping.
                                  grouped.append(value);
                              } else {
                                  grouped.append(py::none());
                              }
                          }
                      }
                      grouped_lists.append(std::move(grouped));
                  }
                  py::list indices;
                  if (with_indices) {
                      for (const size_t tensor_index : group.indices) {
                          indices.append(py::int_(tensor_index));
                      }
                  }
                  result[py::make_tuple(py::cast(group.device),
                                        py::cast(group.dtype))] =
                      py::make_tuple(std::move(grouped_lists),
                                     std::move(indices));
              }
              return result;
          }, "tensorlistlist"_a, "with_indices"_a = false);

    // Fast-path restrictions of the fused multi-tensor kernels, checked in one
    // native pass: every tensor in ``tensor_lists`` shares the device and
    // element type of the first one, index-aligned tensors share one shape,
    // and every tensor is dense, contiguous and not sparse.  Empty lists are
    // allowed.  ``steps`` (may be empty) must hold one contiguous element per
    // tensor, on the host when ``host_steps`` is set and on the tensors'
    // device otherwise.
    m.def("_foreach_fast_path_ready",
          [](const py::sequence& tensor_lists, const py::sequence& steps,
             bool host_steps) -> bool {
              const size_t list_count = static_cast<size_t>(py::len(tensor_lists));
              if (list_count == 0) {
                  return false;
              }
              const py::sequence first_list =
                  py::reinterpret_borrow<py::sequence>(tensor_lists[0]);
              const size_t count = static_cast<size_t>(py::len(first_list));
              if (count == 0) {
                  return false;
              }
              const Tensor& reference = py::cast<const Tensor&>(first_list[0]);
              const Device device = reference.device();
              const DType dtype = reference.dtype();
              std::vector<const Tensor*> anchors(count, nullptr);
              for (size_t list_index = 0; list_index < list_count; ++list_index) {
                  const py::sequence list =
                      py::reinterpret_borrow<py::sequence>(tensor_lists[list_index]);
                  const size_t size = static_cast<size_t>(py::len(list));
                  if (size == 0) {
                      continue;
                  }
                  if (size != count) {
                      return false;
                  }
                  for (size_t index = 0; index < count; ++index) {
                      const py::handle item = list[index];
                      if (item.is_none() || !py::isinstance<Tensor>(item)) {
                          return false;
                      }
                      const Tensor& tensor = py::cast<const Tensor&>(item);
                      if (tensor.is_sparse() || tensor.device() != device ||
                          tensor.dtype() != dtype || !tensor.is_contiguous()) {
                          return false;
                      }
                      if (anchors[index] == nullptr) {
                          anchors[index] = &tensor;
                      } else if (anchors[index]->sizes() != tensor.sizes()) {
                          return false;
                      }
                  }
              }
              const size_t step_count = static_cast<size_t>(py::len(steps));
              if (step_count == 0) {
                  return true;
              }
              if (step_count != count) {
                  return false;
              }
              for (size_t index = 0; index < step_count; ++index) {
                  const py::handle item = steps[index];
                  if (item.is_none() || !py::isinstance<Tensor>(item)) {
                      return false;
                  }
                  const Tensor& step = py::cast<const Tensor&>(item);
                  const bool placed = host_steps ? step.device().is_cpu()
                                                 : step.device() == device;
                  if (!placed || step.numel() != 1 || !step.is_contiguous()) {
                      return false;
                  }
              }
              return true;
          }, "tensor_lists"_a, "steps"_a, "host_steps"_a);


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

    // Python custom-op bridge: expose tensorplay.library kernels through the
    // native Dispatcher and invoke them through the real dispatch path.
    m.def("_register_python_op_kernel", [](const std::string& op_name,
                                           const std::string& device_type,
                                           py::object kernel) {
        if (op_name.find("::") == std::string::npos) {
            TP_THROW(ValueError,
                "op_name must be qualified like 'ns::op', got: ", op_name);
        }
        ensure_stax_custom_op_executor();
        tensorplay::DispatchKey dispatch_key;
        PyOpSlot slot = parse_bridge_key(device_type);
        PyOpKernelEntry* entry = py_op_entry(op_name);
        switch (slot) {
            case PyOpSlot::CPU:
                entry->cpu = std::move(kernel);
                dispatch_key = tensorplay::DispatchKey::CPU;
                break;
            case PyOpSlot::CUDA:
                entry->cuda = std::move(kernel);
                dispatch_key = tensorplay::DispatchKey::CUDA;
                break;
            case PyOpSlot::COMPOSITE:
                // A device-agnostic kernel must stay natively dispatchable:
                // register the trampoline on both backend keys; the
                // trampoline resolves the composite implementation itself.
                entry->composite = std::move(kernel);
                tensorplay::Dispatcher::singleton().registerKernel(
                    op_name, tensorplay::DispatchKey::CPU, &python_op_trampoline);
                tensorplay::Dispatcher::singleton().registerKernel(
                    op_name, tensorplay::DispatchKey::CUDA, &python_op_trampoline);
                return;
        }
        tensorplay::Dispatcher::singleton().registerKernel(
            op_name, dispatch_key, &python_op_trampoline);
    }, "op_name"_a, "device_type"_a, "kernel"_a);

    m.def("_call_native_op", [](const std::string& op_name,
                                std::vector<Tensor> inputs,
                                std::optional<std::string> device_type) {
        tensorplay::DispatchKey key;
        if (device_type) {
            PyOpSlot slot = parse_bridge_key(*device_type);
            if (slot == PyOpSlot::COMPOSITE) {
                TP_THROW(ValueError,
                    "_call_native_op dispatches through a backend key; "
                    "pass 'cpu' or 'cuda', not a composite spelling");
            }
            key = slot == PyOpSlot::CUDA ? tensorplay::DispatchKey::CUDA : tensorplay::DispatchKey::CPU;
        } else {
            if (inputs.empty()) {
                TP_THROW(ValueError,
                    "_call_native_op needs at least one tensor input to "
                    "infer the dispatch key; pass device_type explicitly");
            }
            key = tensorplay::computeDispatchKey(inputs[0].device());
        }
        t_active_python_op = op_name;
        struct TlsReset {
            ~TlsReset() { t_active_python_op.clear(); }
        } reset;
        return tensorplay::DispatchStub<std::vector<Tensor>,
                                        const std::vector<Tensor>&>::call(
            op_name, key, inputs);
    }, "op_name"_a, "inputs"_a, "device_type"_a = py::none());

    m.def("_has_native_kernel", [](const std::string& op_name,
                                   std::optional<std::string> device_type) {
        const bool want_cuda = device_type &&
            parse_bridge_key(*device_type) == PyOpSlot::CUDA;
        const bool composite_only = device_type &&
            parse_bridge_key(*device_type) == PyOpSlot::COMPOSITE;
        PyOpKernelEntry* entry = py_op_entry(op_name);
        py::object fn = want_cuda ? entry->cuda : entry->cpu;
        if (!fn) {
            fn = entry->composite;
        }
        if (!fn || composite_only) {
            return static_cast<bool>(fn);
        }
        tensorplay::DispatchKey key = want_cuda ? tensorplay::DispatchKey::CUDA : tensorplay::DispatchKey::CPU;
        return tensorplay::Dispatcher::singleton().getKernel(op_name, key) != nullptr;
    }, "op_name"_a, "device_type"_a = py::none());

    m.def("_first_device_key", &first_device_key);

    // Bind generated functions (includes *_like, transpose, permute, etc.)
    // onto the dedicated op-functions submodule; every bound name is then
    // aliased onto the root module so `from ._C import <op>` keeps resolving
    // to the same objects.  Hand-written bindings on the root always win.
    py::module_ variable_functions = m.def_submodule("_VariableFunctions");
    bind_generated_op_functions(variable_functions);
    py::object module_dir = py::module_::import("builtins").attr("dir");
    for (py::handle h : module_dir(variable_functions)) {
        const std::string name = py::str(h);
        if (name.empty() || name[0] == '_') {
            continue;
        }
        if (PyObject_HasAttrString(m.ptr(), name.c_str())) {
            continue;
        }
        m.attr(name.c_str()) = variable_functions.attr(name.c_str());
    }

    // NOTE: the METH_FASTCALL layer (register_generated_cpython_functions)
    // is installed at the end of PYBIND11_MODULE so it can never shadow a
    // hand-written pybind overload -- it only fills names nothing else bound.


}
