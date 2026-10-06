#include "SavedVariable.h"

#include "Autograd.h"
#include "Exception.h"
#include "GradMode.h"

#include <utility>
#include <vector>

namespace tensorplay {
namespace tpx {

namespace {
thread_local std::vector<std::shared_ptr<SavedVariableHooks>> g_hooks;
}

std::shared_ptr<SavedVariableHooks> current_saved_variable_hooks() {
    if (g_hooks.empty()) return nullptr;
    return g_hooks.back();
}

void push_saved_variable_hooks(std::shared_ptr<SavedVariableHooks> hooks) {
    g_hooks.push_back(std::move(hooks));
}

void pop_saved_variable_hooks() {
    if (g_hooks.empty()) {
        TP_THROW(RuntimeError, "saved variable hook stack is empty");
    }
    g_hooks.pop_back();
}

const char* backward_twice_message() {
    return "Trying to backward through the graph a second time (or directly "
           "access saved tensors after they have already been freed). Saved "
           "intermediate values of the graph are freed when you call "
           ".backward() or autograd.grad(). Specify retain_graph=True if you "
           "need to backward through the graph a second time or if you need "
           "to access saved tensors after calling backward.";
}

void SavedVariable::save(const Tensor& tensor, bool is_output) {
    if (!tensor.defined()) {
        was_default_constructed_ = true;
        data_ = Tensor();
        packed_.reset();
        hooks_.reset();
        saved_version_ = 0;
        return;
    }
    was_default_constructed_ = false;
    hooks_ = current_saved_variable_hooks();
    data_ = is_output ? tensor.detach() : tensor;
    saved_version_ = tensor.unsafeGetTensorImpl()->version();
    // The detached output loses its tangent; a forward-over-reverse pass
    // reads it back, so it is kept as it is now.  An input is the tensor
    // itself and carries its own.
    fw_grad_.reset();
    if (is_output && ForwardADLevel::has_any_level()) {
        const Tensor tangent = impl::fw_grad(tensor, 0);
        if (tangent.defined()) {
            fw_grad_ = std::make_shared<ForwardGrad>();
            fw_grad_->set_value(tangent, 0);
        }
    }
    if (hooks_) {
        packed_ = hooks_->pack(data_);
        data_ = Tensor();
    } else {
        packed_.reset();
    }
}

Tensor SavedVariable::with_fw_grad(const Tensor& value) const {
    if (!fw_grad_ || !value.defined()) return value;
    const Tensor& tangent = fw_grad_->value(0);
    if (!tangent.defined()) return value;
    // A fresh alias per unpack: the tangent of a tensor cannot be set twice.
    Tensor out = value.detach();
    impl::set_fw_grad(out, tangent, 0, /* is_inplace_op */ false);
    return out;
}

Tensor SavedVariable::unpack_output(const std::shared_ptr<Node>& owner,
                                    uint32_t output_nr) const {
    Tensor value = unpack();
    if (!value.defined() || !owner || !GradMode::is_enabled() ||
        !isFloatingOrComplexType(value.dtype())) {
        return value;
    }
    const Tensor tangent = impl::fw_grad(value, 0);
    Tensor attached = value.detach();
    impl::set_requires_grad(attached, true);
    impl::set_grad_fn(attached, owner, output_nr);
    if (tangent.defined()) impl::set_fw_grad(attached, tangent, 0, /* is_inplace_op */ false);
    return attached;
}

Tensor SavedVariable::unpack() const {
    if (hooks_) {
        return with_fw_grad(hooks_->unpack(packed_));
    }
    if (!data_.defined()) {
        if (!was_default_constructed_) {
            TP_THROW(RuntimeError, backward_twice_message());
        }
        return Tensor();
    }
    uint32_t current = data_.unsafeGetTensorImpl()->version();
    if (current != saved_version_) {
        TP_THROW(RuntimeError,
                 "one of the variables needed for gradient computation has "
                 "been modified by an inplace operation: [saved version: " +
                     std::to_string(saved_version_) +
                     "; current version: " + std::to_string(current) + "]");
    }
    return with_fw_grad(data_);
}

} // namespace tpx
} // namespace tensorplay
