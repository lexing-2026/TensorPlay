#include "python_bindings.h"
#include "Dispatcher.h"
#include "Tensor.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

// Dispatcher introspection surface: registration dumps, op listings and
// key conversions used by debugging tools to inspect where an operator
// resolves (which backend serves it, whether autograd/autocast layers hold
// entries, whether a backend falls through to a composite kernel).

namespace {

// Key order follows the dispatch walk: backend, then autograd, autocast and
// vmap layers, then the backend-neutral composite key.
const tensorplay::DispatchKey kDumpKeys[] = {
    tensorplay::DispatchKey::CPU,
    tensorplay::DispatchKey::CUDA,
    tensorplay::DispatchKey::Vulkan,
    tensorplay::DispatchKey::AutogradCPU,
    tensorplay::DispatchKey::AutogradCUDA,
    tensorplay::DispatchKey::AutogradVulkan,
    tensorplay::DispatchKey::AutocastCPU,
    tensorplay::DispatchKey::AutocastCUDA,
    tensorplay::DispatchKey::AutocastVulkan,
    tensorplay::DispatchKey::VmapCPU,
    tensorplay::DispatchKey::VmapCUDA,
    tensorplay::DispatchKey::VmapVulkan,
    tensorplay::DispatchKey::Composite,
};

tensorplay::DispatchKey parse_key_or_throw(const std::string& name) {
    static const std::unordered_map<std::string, tensorplay::DispatchKey>
        kByName = {
            {"CPU", tensorplay::DispatchKey::CPU},
            {"CUDA", tensorplay::DispatchKey::CUDA},
            {"Vulkan", tensorplay::DispatchKey::Vulkan},
            {"AutogradCPU", tensorplay::DispatchKey::AutogradCPU},
            {"AutogradCUDA", tensorplay::DispatchKey::AutogradCUDA},
            {"AutogradVulkan", tensorplay::DispatchKey::AutogradVulkan},
            {"AutocastCPU", tensorplay::DispatchKey::AutocastCPU},
            {"AutocastCUDA", tensorplay::DispatchKey::AutocastCUDA},
            {"AutocastVulkan", tensorplay::DispatchKey::AutocastVulkan},
            {"VmapCPU", tensorplay::DispatchKey::VmapCPU},
            {"VmapCUDA", tensorplay::DispatchKey::VmapCUDA},
            {"VmapVulkan", tensorplay::DispatchKey::VmapVulkan},
            {"Composite", tensorplay::DispatchKey::Composite},
        };
    auto it = kByName.find(name);
    if (it == kByName.end()) {
        TP_THROW(RuntimeError, "unknown dispatch key: ", name);
    }
    return it->second;
}

} // namespace

void init_dispatch(py::module_& m) {
    // Registration state of one operator, keyed by dispatch key name. A
    // backend slot with no kernel of its own reports "composite fallback"
    // when the composite key holds a kernel, matching what
    // OperatorHandle::getKernel would resolve. Returns None for operators
    // with no registration anywhere.
    m.def("_dispatch_dump", [](const std::string& op_name) -> py::object {
        auto& dispatcher = tensorplay::Dispatcher::singleton();
        py::dict dump;
        bool any = false;
        for (auto key : kDumpKeys) {
            if (dispatcher.has_kernel(op_name, key)) {
                any = true;
                dump[py::str(tensorplay::toString(key))] = py::str("registered");
            } else if (tensorplay::is_backend_key(key) &&
                       dispatcher.has_kernel(op_name,
                                             tensorplay::DispatchKey::Composite)) {
                dump[py::str(tensorplay::toString(key))] =
                    py::str("composite fallback");
            }
        }
        if (!any) return py::none();
        return std::move(dump);
    }, py::arg("op_name"),
        "_dispatch_dump(op_name) -> dict[str, str] | None\n\n"
        "Dump the kernel registrations for an operator. Keys are dispatch\n"
        "key names (\"CPU\", \"AutogradCPU\", \"Composite\", ...); values are\n"
        "\"registered\" or \"composite fallback\". None means the operator\n"
        "name has no registration under any key.\n\n"
        "Op names follow the schema spelling: \"add.Tensor\", \"mul.Tensor\",\n"
        "\"sum.dim_IntList\", ...");

    // Single-point registration queries (used by test suites that assert a
    // given backend implements an op without dumping the whole table).
    m.def("_dispatch_has_kernel", [](const std::string& op_name) -> bool {
        auto& dispatcher = tensorplay::Dispatcher::singleton();
        for (auto key : kDumpKeys) {
            if (dispatcher.has_kernel(op_name, key)) return true;
        }
        return false;
    }, py::arg("op_name"),
        "_dispatch_has_kernel(op_name) -> bool\n\n"
        "Whether the operator has a kernel registered under any dispatch key.");

    m.def("_dispatch_has_kernel_for_dispatch_key",
          [](const std::string& op_name, const std::string& key_name) -> bool {
              return tensorplay::Dispatcher::singleton().has_kernel(
                  op_name, parse_key_or_throw(key_name));
          }, py::arg("op_name"), py::arg("key"),
        "_dispatch_has_kernel_for_dispatch_key(op_name, key) -> bool\n\n"
        "Whether a direct kernel registration exists for the\n"
        "<op_name, dispatch key> pair (no composite fallback resolution).");

    // Every operator name known to the dispatcher, sorted for stable
    // comparison across runs and builds.
    m.def("_dispatch_ops", []() {
        std::vector<std::string> names =
            tensorplay::Dispatcher::singleton().operator_names();
        std::sort(names.begin(), names.end());
        return names;
    }, "_dispatch_ops() -> list[str]\n\n"
       "List every operator name registered with the dispatcher, sorted.");

    // Dispatch key <-> name conversions (round trip: the enum exposes the
    // same names, so parse(toString(k)) == k).
    py::enum_<tensorplay::DispatchKey>(m, "DispatchKey",
        "Dispatch keys of the runtime, in dispatch-priority order: autocast\n"
        "keys outrank autograd keys, which outrank backend keys.")
        .value("CPU", tensorplay::DispatchKey::CPU)
        .value("CUDA", tensorplay::DispatchKey::CUDA)
        .value("Vulkan", tensorplay::DispatchKey::Vulkan)
        .value("AutogradCPU", tensorplay::DispatchKey::AutogradCPU)
        .value("AutogradCUDA", tensorplay::DispatchKey::AutogradCUDA)
        .value("AutogradVulkan", tensorplay::DispatchKey::AutogradVulkan)
        .value("AutocastCPU", tensorplay::DispatchKey::AutocastCPU)
        .value("AutocastCUDA", tensorplay::DispatchKey::AutocastCUDA)
        .value("AutocastVulkan", tensorplay::DispatchKey::AutocastVulkan)
        .value("VmapCPU", tensorplay::DispatchKey::VmapCPU)
        .value("VmapCUDA", tensorplay::DispatchKey::VmapCUDA)
        .value("VmapVulkan", tensorplay::DispatchKey::VmapVulkan)
        .value("Composite", tensorplay::DispatchKey::Composite)
        .export_values();

    m.def("_dispatch_key_name", [](tensorplay::DispatchKey key) {
        return tensorplay::toString(key);
    }, py::arg("key"), "DispatchKey -> name string.");

    m.def("_dispatch_key_parse", [](const std::string& name) {
        return parse_key_or_throw(name);
    }, py::arg("name"), "Dispatch key name string -> DispatchKey.");

    // The backend component keys a layered key (autograd/autocast/vmap)
    // resolves to; identity for backend and composite keys.
    m.def("_dispatch_key_to_backend_key", [](tensorplay::DispatchKey key) {
        return tensorplay::toBackendKey(key);
    }, py::arg("key"), "Layered DispatchKey -> its backend component key.");

    // The tensor's dispatch keys, names sorted by numeric key value
    // (highest priority last, matching the walk order of the dispatcher).
    m.def("_dispatch_keys", [](const Tensor& t) {
        std::vector<std::string> names;
        if (!t.defined()) return names;
        for (auto key : kDumpKeys) {
            if (key == tensorplay::DispatchKey::Composite) continue;
            if (t.key_set().has(key)) {
                names.push_back(tensorplay::toString(key));
            }
        }
        return names;
    }, py::arg("tensor"),
        "_dispatch_keys(tensor) -> list[str]\n\n"
        "Dispatch key names carried by the tensor's key set.");

    // The composite fallthrough contract: a backend lookup with no kernel
    // of its own resolves through the composite key.
    m.def("_dispatch_get_backend_keyset_from_composite", []() {
        std::vector<std::string> names;
        for (auto key : kDumpKeys) {
            if (tensorplay::is_backend_key(key)) {
                names.push_back(tensorplay::toString(key));
            }
        }
        return names;
    },
        "_dispatch_get_backend_keyset_from_composite() -> list[str]\n\n"
        "Backend key names a Composite registration serves.");
}
