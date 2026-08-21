#pragma once
#include <vector>
#include <cstddef>
#include <utility>
#include "Tensor.h"
#include "Autograd.h"
#include "Node.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

namespace tensorplay {
namespace tpx {

// True if `v` is a "vanilla" contiguous tensor that we hold the last reference
// to (both the TensorImpl and its Storage), so accumulating into it in-place
// with add_() is safe.
inline bool can_accumulate_inplace(const Tensor& v) {
    if (!v.is_contiguous()) return false;
    // unsafeGetTensorImpl()/impl() return temporary shared_ptr copies; a
    // use_count of 2 means the tensor itself is the only other holder.
    if (v.unsafeGetTensorImpl().use_count() != 2) return false;
    if (!v.impl()->has_storage()) return false;
    if (v.impl()->storage().use_count() != 1) return false;
    return true;
}

// `grad_mode` is passed explicitly (rather than read from thread-local
// GradMode) because with the multithreaded engine accumulation may run on a
// worker thread whose TLS does not describe this GraphTask.
inline void accumulate(std::vector<Tensor>& buffer, size_t pos, Tensor&& var, bool grad_mode) {
    auto& old_var = buffer[pos];
    if (grad_mode) {
        // Under GradMode (e.g. create_graph backward) accumulate through the
        // autograd-aware ops so the second-order graph is built.
        buffer[pos] = tensorplay::tpx::ops::add(old_var, var);
    } else if (can_accumulate_inplace(old_var)) {
        buffer[pos] = old_var.add_(var);
    } else {
        buffer[pos] = old_var + var;
    }
}

// Accumulates gradients for a single Node input at a fixed index (input_nr).
// Mirrors torch/csrc/autograd/input_buffer.{h,cpp}.
struct InputBuffer {
    InputBuffer() = default;
    explicit InputBuffer(size_t size) : buffer(size) {}
    InputBuffer(variable_list&& inputs) : buffer(std::move(inputs)) {}
    InputBuffer(InputBuffer&&) = default;
    InputBuffer& operator=(InputBuffer&&) = default;
    InputBuffer(const InputBuffer&) = delete;
    InputBuffer& operator=(const InputBuffer&) = delete;

    void add(size_t pos, Tensor&& var, bool grad_mode = false) {
        if (pos >= buffer.size() || !var.defined()) return;
        if (!buffer[pos].defined()) {
            buffer[pos] = std::move(var);
        } else {
            accumulate(buffer, pos, std::move(var), grad_mode);
        }
    }

    Tensor operator[](size_t pos) { return buffer[pos]; }

    // Device of the first defined input; used by the engine to route a
    // NodeTask to the owning device's ready queue (mirrors
    // torch::autograd::InputBuffer::device()).
    int device_index() const {
        for (const auto& t : buffer) {
            if (t.defined() && t.device().is_cuda()) {
                return static_cast<int>(t.device().index());
            }
        }
        return -1; // CPU / unspecified
    }

    static variable_list variables(InputBuffer&& g) {
        return std::move(g.buffer);
    }

    std::vector<Tensor> buffer;
};

} // namespace tpx
} // namespace tensorplay
