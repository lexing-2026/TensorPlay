#include "python_bindings.h"
#include "CpuFallback.h"
#include "Dispatcher.h"
#include "Tensor.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <typeinfo>
#if defined(__GNUG__)
#include <cxxabi.h>
#endif
#include <string>
#include <unordered_map>
#include <vector>

// Dispatcher introspection surface: registration dumps, op listings and
// key conversions used by debugging tools to inspect where an operator
// resolves (which backend serves it, whether autograd/autocast layers hold
// entries, whether a backend falls through to a composite kernel).

namespace {

// Every dispatch key in numeric order: backends, the Python key, then the
// autograd, autocast and vmap layers, then the backend-neutral composite key
// and the transform modes.  Built from the key enumeration itself so a
// backend added there is listed here without further edits.
const std::vector<tensorplay::DispatchKey>& dump_keys() {
    static const std::vector<tensorplay::DispatchKey> keys = [] {
        std::vector<tensorplay::DispatchKey> out;
        const auto end = static_cast<uint8_t>(tensorplay::DispatchKey::EndOfKeys);
        for (uint8_t k = 0; k < end; ++k) {
            out.push_back(static_cast<tensorplay::DispatchKey>(k));
        }
        return out;
    }();
    return keys;
}

tensorplay::DispatchKey parse_key_or_throw(const std::string& name) {
    static const std::unordered_map<std::string, tensorplay::DispatchKey>
        kByName = [] {
            std::unordered_map<std::string, tensorplay::DispatchKey> out;
            for (auto key : dump_keys()) out.emplace(tensorplay::toString(key), key);
            return out;
        }();
    auto it = kByName.find(name);
    if (it == kByName.end()) {
        TP_THROW(RuntimeError, "unknown dispatch key: ", name);
    }
    return it->second;
}

std::string readable_type(const std::type_info* type) {
    if (type == nullptr) return "<untyped>";
#if defined(__GNUG__)
    int status = 0;
    std::unique_ptr<char, void (*)(void*)> name(
        abi::__cxa_demangle(type->name(), nullptr, nullptr, &status), std::free);
    if (status == 0 && name) return name.get();
#endif
    return type->name();
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
        for (auto key : dump_keys()) {
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

    m.def("_set_host_fallback_counting", [](bool enabled) {
        tensorplay::cpu_fallback::set_host_call_counting(enabled);
    }, py::arg("enabled"));
    m.def("_host_fallback_counts", [] {
        py::dict out;
        for (const auto& [name, count] : tensorplay::cpu_fallback::host_call_counts()) {
            out[py::str(name)] = count;
        }
        return out;
    }, "Per-operator calls served by the host fallback since the last reset.");
    m.def("_reset_host_fallback_counts", [] {
        tensorplay::cpu_fallback::reset_host_call_counts();
    });

    // Single-point registration queries (used by test suites that assert a
    // given backend implements an op without dumping the whole table).
    m.def("_dispatch_has_kernel", [](const std::string& op_name) -> bool {
        auto& dispatcher = tensorplay::Dispatcher::singleton();
        for (auto key : dump_keys()) {
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
    py::enum_<tensorplay::DispatchKey> dispatch_key_enum(m, "DispatchKey",
        "Dispatch keys of the runtime, in dispatch-priority order: vmap keys\n"
        "outrank autocast keys, which outrank autograd keys, which outrank\n"
        "backend keys.");
    for (auto key : dump_keys()) {
        dispatch_key_enum.value(tensorplay::toString(key).c_str(), key);
    }
    dispatch_key_enum
        .export_values();

    // Kernels whose registered signature differs from the operator's
    // dispatcher signature (the one its Python-key kernel is registered
    // with).  Every caller casts the slot to that signature, so each entry is
    // a call with a mismatched ABI.  Returns (op, key, registered, expected).
    m.def("_dispatch_abi_mismatches", []() {
        auto& dispatcher = tensorplay::Dispatcher::singleton();
        py::list out;
        for (const auto& op : dispatcher.operator_names()) {
            const std::type_info* expected =
                dispatcher.signature(op, tensorplay::DispatchKey::Python);
            if (expected == nullptr) continue;
            for (auto key : dump_keys()) {
                if (key == tensorplay::DispatchKey::Python) continue;
                if (!dispatcher.has_kernel(op, key)) continue;
                const std::type_info* registered = dispatcher.signature(op, key);
                if (registered != nullptr && *registered == *expected) continue;
                out.append(py::make_tuple(op, tensorplay::toString(key),
                                          readable_type(registered),
                                          readable_type(expected)));
            }
        }
        return out;
    });

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
        for (auto key : dump_keys()) {
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
        for (auto key : dump_keys()) {
            if (tensorplay::is_backend_key(key)) {
                names.push_back(tensorplay::toString(key));
            }
        }
        return names;
    },
        "_dispatch_get_backend_keyset_from_composite() -> list[str]\n\n"
        "Backend key names a Composite registration serves.");
}
