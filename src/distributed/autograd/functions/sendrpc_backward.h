#pragma once

#include "Node.h"

#include <mutex>
#include <utility>
#include <vector>

namespace tensorplay::distributed::autograd {

class SendRpcBackward final : public tensorplay::tpx::Node {
public:
    using Tensor = tensorplay::Tensor;

    tensorplay::tpx::variable_list apply(
        tensorplay::tpx::variable_list&& inputs) override;
    size_t num_inputs() const override;

    void set_grads(const tensorplay::tpx::variable_list& grads);
    tensorplay::tpx::variable_list grads() const;

    void add_leaf(Tensor variable, Tensor baseline);
    std::vector<std::pair<Tensor, Tensor>> leaves() const;

private:
    mutable std::mutex mutex_;
    tensorplay::tpx::variable_list grads_;
    std::vector<std::pair<Tensor, Tensor>> leaves_;
};

}  // namespace tensorplay::distributed::autograd
