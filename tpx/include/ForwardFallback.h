#pragma once

// Forward-mode derivatives of operations that declare no forward formula.
//
// The generated wrapper of every differentiable operation answers the
// tangent of its outputs one of these ways:
//
//   * a forward formula from derivatives.yaml;
//   * a view applies itself to its input's tangent, so the output's tangent
//     views the input's;
//   * an in-place update takes the tangent its functional twin gives on the
//     same inputs, written into the tangent the updated tensor already has;
//   * any other operation takes it from its backward: the backward is linear
//     in the gradient it receives, vjp(u) = J^H u, and differentiating that
//     product in u along the input tangents t gives (d vjp / d u)^H t = J t;
//   * an operation none of these can serve -- it writes an argument other
//     than its result, is an in-place update without a functional twin, or
//     fills an out= tensor -- refuses the tangent.
//
// While the kernel runs, tangent reads are off: the wrapper answers for its
// outputs, so the operations a kernel composes need not answer for theirs.

#include "Autograd.h"
#include "ForwardGrad.h"
#include "Generator.h"

#include <cstddef>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

namespace tensorplay {
namespace tpx {
namespace impl {

using ForwardRerun = std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;

// `inputs` are the operation's differentiable tensor arguments in order, a
// list argument's tensors in place and an absent one as an undefined tensor;
// `rerun` evaluates the operation with them replaced and returns its
// outputs, an undefined tensor for an output that is not a tensor.  Returns
// one tangent per output, undefined where the output has none.
TENSORPLAY_API std::vector<Tensor> fw_grads_from_backward(
    const char* op_name, const std::vector<Tensor>& inputs, const ForwardRerun& rerun);

// The random streams an operation draws from, taken before it draws, so a
// rerun draws the same numbers: on the CPU the generator it was handed or
// the default one, and the generator of every CUDA device an input is on.
class TENSORPLAY_API RandomReplay {
public:
    RandomReplay(const std::vector<Tensor>& inputs, const std::optional<Generator>& generator);

    // Runs `fn` on the streams as they were taken, then puts back the
    // streams as they were before the call.
    std::vector<Tensor> run(const std::function<std::vector<Tensor>()>& fn) const;

private:
    std::optional<Generator> generator_;
    Tensor cpu_state_;
    std::vector<std::pair<int, Tensor>> cuda_states_;
};

// Turns tangent reads off for its lifetime.
class FwGradSuspend {
public:
    FwGradSuspend() : active_(FwGradMode::is_enabled()) {
        if (active_) FwGradMode::set_enabled(false);
    }
    ~FwGradSuspend() {
        if (active_) FwGradMode::set_enabled(true);
    }
    FwGradSuspend(const FwGradSuspend&) = delete;
    FwGradSuspend& operator=(const FwGradSuspend&) = delete;

private:
    bool active_;
};

// Whether any tensor in a list argument carries a tangent at `level`.
inline bool any_fw_grad_defined(const std::vector<Tensor>& ts, uint64_t level) {
    for (const auto& t : ts) {
        if (is_fw_grad_defined(t, level)) return true;
    }
    return false;
}

inline bool any_fw_grad_defined(const std::vector<std::optional<Tensor>>& ts, uint64_t level) {
    for (const auto& t : ts) {
        if (is_fw_grad_defined(t, level)) return true;
    }
    return false;
}

// A list argument's tensors flattened into the rerun's inputs, and read
// back out of them.
inline void append_tensors(std::vector<Tensor>& out, const std::vector<Tensor>& ts) {
    out.insert(out.end(), ts.begin(), ts.end());
}

inline void append_tensors(std::vector<Tensor>& out, const std::vector<std::optional<Tensor>>& ts) {
    for (const auto& t : ts) out.push_back(t.has_value() ? *t : Tensor());
}

inline std::vector<Tensor> tensors_from(const std::vector<Tensor>& p, size_t at, size_t n) {
    return std::vector<Tensor>(p.begin() + at, p.begin() + at + n);
}

inline std::optional<Tensor> optional_from(const Tensor& t) {
    return t.defined() ? std::optional<Tensor>(t) : std::nullopt;
}

inline std::vector<std::optional<Tensor>> optional_tensors_from(
    const std::vector<Tensor>& p, size_t at, size_t n) {
    std::vector<std::optional<Tensor>> out;
    out.reserve(n);
    for (size_t i = at; i < at + n; ++i) out.push_back(optional_from(p[i]));
    return out;
}

// Gives `out` the tangent `tangent`, unless it already has one: an output
// that is one of the inputs, or a view reading its base's tangent.
TENSORPLAY_API void set_output_fw_grad(const Tensor& out, const Tensor& tangent);

// After an in-place update, `self` takes `tangent`: copied into the tangent
// it already has, so views sharing that tangent see the update, or attached
// as its first one.
TENSORPLAY_API void set_inplace_fw_grad(const Tensor& self, const Tensor& tangent);

// Raised when a tangent is due that the operation cannot carry.
[[noreturn]] TENSORPLAY_API void refuse_forward_ad(const char* op_name, bool out_variant);

} // namespace impl
} // namespace tpx
} // namespace tensorplay
