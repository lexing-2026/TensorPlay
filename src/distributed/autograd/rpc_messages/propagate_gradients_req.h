#pragma once

#include "autograd_metadata.h"
#include "rpc/message.h"

#include "Tensor.h"

#include <memory>
#include <vector>

namespace tensorplay::distributed::autograd {

class PropagateGradientsReq final {
public:
    PropagateGradientsReq(
        AutogradMetadata metadata,
        std::vector<tensorplay::Tensor> gradients,
        bool retain_graph);

    rpc::MessagePtr to_message() const;
    static PropagateGradientsReq from_message(const rpc::Message& message);

    const AutogradMetadata& metadata() const noexcept;
    const std::vector<tensorplay::Tensor>& gradients() const noexcept;
    bool retain_graph() const noexcept;

private:
    AutogradMetadata metadata_;
    std::vector<tensorplay::Tensor> gradients_;
    bool retain_graph_ = false;
};

}  // namespace tensorplay::distributed::autograd
