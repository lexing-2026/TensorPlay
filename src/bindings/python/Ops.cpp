#include "python_bindings.h"
#include "tensorplay/ops/TensorBindingsGenerated.h"
#include "tensorplay/ops/TensorCPythonGenerated.h"
#include "Dispatcher.h"
#include "Graph.h"
#include "Context.h"
#include "utils.h"
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

// Leaky singleton: entries must outlive interpreter shutdown because the
// dispatcher table (also process-lifetime) keeps raw trampoline pointers.
PyOpKernelEntry* py_op_entry(const std::string& op_name) {
    static auto* map = new std::unordered_map<std::string, PyOpKernelEntry>();
    static auto* mutex = new std::mutex();
    std::lock_guard<std::mutex> lock(*mutex);
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

std::vector<Tensor> invoke_python_kernel_by_name(
    const std::string& op_name,
    const std::vector<Tensor>& inputs) {
    py::gil_scoped_acquire acquire;
    py::object fn = select_py_kernel(op_name, inputs);
    // User kernels take per-tensor parameters, matching their signature at
    // registration time; results come back as one or many tensors.
    py::tuple py_args = py::cast(inputs);
    py::object result = py::reinterpret_steal<py::object>(
        PyObject_Call(fn.ptr(), py_args.ptr(), nullptr));
    if (!result) {
        throw py::error_already_set();
    }
    // Single-output kernels may return a bare Tensor.
    if (py::isinstance<Tensor>(result)) {
        return {result.cast<Tensor>()};
    }
    return py::cast<std::vector<Tensor>>(result);
}

std::vector<Tensor> python_op_trampoline(const std::vector<Tensor>& inputs) {
    const std::string op_name = t_active_python_op;
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
            py::gil_scoped_acquire acquire;
            py::object lib = py::module_::import("tensorplay.library");
            py::object entry = lib.attr("_native_invoke");
            py::tuple py_args(inputs.size() + 1);
            py_args[0] = py::cast(op_name);
            for (size_t i = 0; i < inputs.size(); ++i) {
                py_args[i + 1] = py::cast(inputs[i]);
            }
            py::object result = py::reinterpret_steal<py::object>(
                PyObject_Call(entry.ptr(), py_args.ptr(), nullptr));
            if (!result) {
                throw py::error_already_set();
            }
            if (py::isinstance<Tensor>(result)) {
                return {result.cast<Tensor>()};
            }
            return py::cast<std::vector<Tensor>>(result);
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
                    op_name, tensorplay::DispatchKey::CPU,
                    reinterpret_cast<tensorplay::KernelFunction>(
                        &python_op_trampoline));
                tensorplay::Dispatcher::singleton().registerKernel(
                    op_name, tensorplay::DispatchKey::CUDA,
                    reinterpret_cast<tensorplay::KernelFunction>(
                        &python_op_trampoline));
                return;
        }
        tensorplay::Dispatcher::singleton().registerKernel(
            op_name, dispatch_key,
            reinterpret_cast<tensorplay::KernelFunction>(
                &python_op_trampoline));
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
