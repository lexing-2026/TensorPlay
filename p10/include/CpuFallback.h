#pragma once

// Run an operator of a device backend on the host.
//
// A new backend rarely starts with a kernel for every operator.  This header
// turns any operator's CPU kernel into a kernel for the backend: device
// tensors among the arguments are copied to the host, the CPU kernel runs,
// results are copied back to the device the inputs came from, and arguments
// the operator writes receive the written values.
//
// A backend can register it for one operator:
//
//   #include "CpuFallback.h"
//   #include "tensorplay/ops/KernelSignatures.h"
//
//   TENSORPLAY_LIBRARY_IMPL(PrivateUse1, MyBackendFallbacks) {
//       m.impl("mm", cpu_fallback::kernel<"mm", kernel_signatures::mm>);
//   }
//
// or for every operator that is still missing a kernel, once its own
// kernels are registered:
//
//   cpu_fallback::register_host_fallbacks(DispatchKey::PrivateUse1);
//
// Requirements and limits:
// * the backend implements "empty" and "copy_" natively (copy_ in both
//   directions between the host and the device), plus "resize_" for
//   operators that resize an output: the fallback moves data with exactly
//   those operators;
// * the operator returns fresh tensors or its written arguments.  Operators
//   whose result aliases an input (views) cannot run this way;
// * every call pays two transfers per tensor, so the fallback is a way to
//   bring a model up, not to run it fast.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "Device.h"
#include "Dispatcher.h"
#include "LocalDispatchKeySet.h"
#include "Macros.h"
#include "Tensor.h"

namespace tensorplay {
namespace cpu_fallback {

// An operator name usable as a template argument: cpu_fallback::kernel<"mm", ...>.
template <size_t N>
struct OpName {
    char value[N];
    constexpr OpName(const char (&text)[N]) {
        for (size_t i = 0; i < N; ++i) value[i] = text[i];
    }
};

// Register the host fallback for every operator that has a CPU kernel but
// neither a kernel under ``key`` nor a backend-neutral composite kernel.
// Operators that only rearrange metadata (views), allocate or copy are
// never covered: the backend provides those.  Returns how many operators
// were registered.
P10_API int register_host_fallbacks(DispatchKey key);

// Per-operator counts of calls that ran on the host through this fallback,
// for finding the operators a backend still lacks.  Counting is off until
// enabled; the counts persist until reset.
P10_API void set_host_call_counting(bool enabled);
P10_API std::vector<std::pair<std::string, int64_t>> host_call_counts();
P10_API void reset_host_call_counts();

namespace detail {

P10_API extern std::atomic<bool> host_call_counting;
P10_API void count_host_call(const char* name);

// Where results go: the device of the first non-host tensor (or device
// argument) the call names.
struct Home {
    std::optional<Device> device;
    void note(const Device& d) {
        if (!device && !d.is_cpu()) device = d;
    }
};

inline Tensor to_host(const Tensor& t, Home& home) {
    if (!t.defined() || t.device().is_cpu()) return t;
    home.note(t.device());
    return t.to(Device(DeviceType::CPU));
}

// Host copy of one argument, stored for the length of the call.  Device
// arguments are rewritten to the host so factory operators allocate there.
template <typename Arg>
struct Staged {
    using Value = std::decay_t<Arg>;
    static Value stage(const Value& arg, Home&) { return arg; }
};

template <typename Arg>
    requires std::is_same_v<std::decay_t<Arg>, Tensor>
struct Staged<Arg> {
    using Value = Tensor;
    static Value stage(const Tensor& arg, Home& home) { return to_host(arg, home); }
};

template <typename Arg>
    requires std::is_same_v<std::decay_t<Arg>, std::optional<Tensor>>
struct Staged<Arg> {
    using Value = std::optional<Tensor>;
    static Value stage(const std::optional<Tensor>& arg, Home& home) {
        if (!arg) return std::nullopt;
        return to_host(*arg, home);
    }
};

template <typename Arg>
    requires std::is_same_v<std::decay_t<Arg>, std::vector<Tensor>>
struct Staged<Arg> {
    using Value = std::vector<Tensor>;
    static Value stage(const std::vector<Tensor>& arg, Home& home) {
        Value out;
        out.reserve(arg.size());
        for (const auto& t : arg) out.push_back(to_host(t, home));
        return out;
    }
};

template <typename Arg>
    requires std::is_same_v<std::decay_t<Arg>, std::vector<std::optional<Tensor>>>
struct Staged<Arg> {
    using Value = std::vector<std::optional<Tensor>>;
    static Value stage(const Value& arg, Home& home) {
        Value out;
        out.reserve(arg.size());
        for (const auto& t : arg) {
            out.push_back(t ? std::optional<Tensor>(to_host(*t, home)) : std::nullopt);
        }
        return out;
    }
};

template <typename Arg>
    requires std::is_same_v<std::decay_t<Arg>, Device>
struct Staged<Arg> {
    using Value = Device;
    static Value stage(const Device& arg, Home& home) {
        home.note(arg);
        return Device(DeviceType::CPU);
    }
};

template <typename Arg>
    requires std::is_same_v<std::decay_t<Arg>, std::optional<Device>>
struct Staged<Arg> {
    using Value = std::optional<Device>;
    static Value stage(const std::optional<Device>& arg, Home& home) {
        if (!arg) return std::nullopt;
        home.note(*arg);
        return arg->is_cpu() ? arg : std::optional<Device>(Device(DeviceType::CPU));
    }
};

// Arguments the operator writes: a mutable tensor reference, or a mutable
// tensor list (passed by value in the kernel ABI).
template <typename Arg>
constexpr bool is_written_tensor = std::is_same_v<Arg, Tensor&>;
template <typename Arg>
constexpr bool is_written_list = std::is_same_v<Arg, std::vector<Tensor>>;

// The written device tensors of one call, each with the host copy the CPU
// kernel wrote.
struct Written {
    std::vector<std::pair<Tensor, Tensor>> pairs;  // (original, staged)

    void add(const Tensor& original, const Tensor& staged) {
        if (original.defined() && !original.device().is_cpu()) {
            pairs.emplace_back(original, staged);
        }
    }

    void write_back() const {
        for (const auto& [original, staged] : pairs) {
            Tensor target = original;
            if (target.sizes().vec() != staged.sizes().vec()) {
                target.resize_(staged.sizes().vec());
            }
            target.copy_(staged);
        }
    }

    // The original a returned host tensor stands for, when it is the host
    // copy of a written argument.
    const Tensor* original_of(const Tensor& returned) const {
        if (!returned.defined()) return nullptr;
        for (const auto& [original, staged] : pairs) {
            if (staged.unsafeGetTensorImpl() == returned.unsafeGetTensorImpl()) {
                return &original;
            }
        }
        return nullptr;
    }
};

template <typename Arg>
void note_written(Written& written, Arg& original, const typename Staged<Arg>::Value& staged) {
    if constexpr (is_written_tensor<Arg>) {
        written.add(original, staged);
    } else if constexpr (is_written_list<Arg>) {
        for (size_t i = 0; i < original.size(); ++i) written.add(original[i], staged[i]);
    }
}

inline Tensor tensor_to_home(const Tensor& t, const Home& home, const Written& written) {
    if (const Tensor* original = written.original_of(t)) return *original;
    if (!t.defined() || !home.device || !t.device().is_cpu()) return t;
    return t.to(*home.device);
}

template <typename T>
T result_to_home(T value, const Home& home, const Written& written) {
    if constexpr (std::is_same_v<T, Tensor>) {
        return tensor_to_home(value, home, written);
    } else if constexpr (std::is_same_v<T, std::vector<Tensor>>) {
        for (auto& t : value) t = tensor_to_home(t, home, written);
        return value;
    } else {
        return value;
    }
}

template <typename... Ts>
std::tuple<Ts...> result_to_home(std::tuple<Ts...> value, const Home& home,
                                 const Written& written) {
    return std::apply(
        [&](auto&... items) {
            return std::tuple<Ts...>(result_to_home(std::move(items), home, written)...);
        },
        value);
}

} // namespace detail

template <OpName Name, typename Signature>
struct Kernel;

template <OpName Name, typename Ret, typename... Args>
struct Kernel<Name, Ret(Args...)> {
    static Ret call(Args... args) {
        static const OperatorHandle handle = Dispatcher::singleton().findHandle(Name.value);
        if (detail::host_call_counting.load(std::memory_order_relaxed)) {
            detail::count_host_call(Name.value);
        }
        detail::Home home;
        detail::Written written;
        std::tuple<typename detail::Staged<Args>::Value...> staged{
            detail::Staged<Args>::stage(args, home)...};
        auto originals = std::forward_as_tuple(args...);
        [&]<size_t... I>(std::index_sequence<I...>) {
            (detail::note_written<std::tuple_element_t<I, std::tuple<Args...>>>(
                 written, std::get<I>(originals), std::get<I>(staged)),
             ...);
        }(std::index_sequence_for<Args...>{});

        // The host kernel runs below autograd, autocast and vmap: the call
        // that reached this kernel has already passed through them.  Moving
        // results back is an ordinary call again.
        auto run = [&]() -> Ret {
            impl::ExcludeDispatchKeyGuard below(DispatchKeySet::autograd_keys() |
                                                DispatchKeySet::autocast_keys() |
                                                DispatchKeySet::vmap_keys());
            return std::apply(
                [&](auto&... values) -> Ret {
                    return DispatchStub<Ret, Args...>::call(handle, DispatchKey::CPU,
                                                            static_cast<Args>(values)...);
                },
                staged);
        };

        if constexpr (std::is_void_v<Ret>) {
            run();
            write_back(written);
        } else if constexpr (std::is_same_v<Ret, Tensor&>) {
            Tensor& returned = run();
            write_back(written);
            // Hand back the argument whose host copy (or itself, for a host
            // argument) the kernel returned.
            Tensor* original = nullptr;
            [&]<size_t... I>(std::index_sequence<I...>) {
                (match_returned<std::tuple_element_t<I, std::tuple<Args...>>>(
                     returned, std::get<I>(staged), std::get<I>(originals), original),
                 ...);
            }(std::index_sequence_for<Args...>{});
            TP_CHECK(original != nullptr, "cpu_fallback for '", Name.value,
                     "': the kernel returned a tensor that is not one of its "
                     "written arguments");
            return *original;
        } else {
            Ret result = run();
            write_back(written);
            return detail::result_to_home(std::move(result), home, written);
        }
    }

private:
    static void write_back(const detail::Written& written) {
        impl::ExcludeDispatchKeyGuard below(DispatchKeySet::autograd_keys());
        written.write_back();
    }

    template <typename Arg, typename StagedValue, typename Original>
    static void match_returned(const Tensor& returned, const StagedValue& staged,
                               Original& original, Tensor*& found) {
        if constexpr (detail::is_written_tensor<Arg>) {
            const auto& impl = returned.unsafeGetTensorImpl();
            if (found == nullptr && (staged.unsafeGetTensorImpl() == impl ||
                                     original.unsafeGetTensorImpl() == impl)) {
                found = &original;
            }
        }
    }
};

// The fallback kernel of operator ``Name`` whose kernel signature is
// ``Signature`` (from tensorplay/ops/KernelSignatures.h).
template <OpName Name, typename Signature>
inline constexpr auto kernel = &Kernel<Name, Signature>::call;

namespace detail {

// Register the fallback for one operator when the operator needs it (see
// register_host_fallbacks).  Used by the generated registration tables.
template <OpName Name, typename Signature>
void add_host_fallback(DispatchKey key, int& count) {
    auto& dispatcher = Dispatcher::singleton();
    const std::string name(Name.value);
    if (dispatcher.has_kernel(name, key) ||
        dispatcher.has_kernel(name, DispatchKey::Composite) ||
        !dispatcher.has_kernel(name, DispatchKey::CPU)) {
        return;
    }
    dispatcher.registerKernel(name, key, &Kernel<Name, Signature>::call);
    ++count;
}

} // namespace detail

} // namespace cpu_fallback
} // namespace tensorplay
