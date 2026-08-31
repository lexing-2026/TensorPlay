#include "sendrpc_backward.h"

#include <stdexcept>

namespace tensorplay::distributed::autograd {

tensorplay::tpx::variable_list SendRpcBackward::apply(
    tensorplay::tpx::variable_list&& inputs) {
    (void)inputs;
    std::lock_guard<std::mutex> lock(mutex_);
    return grads_;
}

size_t SendRpcBackward::num_inputs() const {
    return next_edges().size();
}

void SendRpcBackward::set_grads(
    const tensorplay::tpx::variable_list& grads) {
    for (const auto& grad : grads) {
        if (!grad.defined()) {
            throw std::invalid_argument(
                "distributed autograd received an undefined gradient");
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    grads_ = grads;
}

tensorplay::tpx::variable_list SendRpcBackward::grads() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return grads_;
}

void SendRpcBackward::add_leaf(Tensor variable, Tensor baseline) {
    if (!variable.defined()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    leaves_.emplace_back(std::move(variable), std::move(baseline));
}

std::vector<std::pair<SendRpcBackward::Tensor, SendRpcBackward::Tensor>>
SendRpcBackward::leaves() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return leaves_;
}

}  // namespace tensorplay::distributed::autograd
