#pragma once

#include "Node.h"
#include "context/context.h"
#include "rpc/tensorpipe_utils.h"
#include "rpc_messages/autograd_metadata.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace tensorplay::distributed::autograd {

class RecvRpcBackward final : public tensorplay::tpx::Node {
public:
    RecvRpcBackward(
        AutogradMetadata metadata,
        ContextPtr context,
        rpc::worker_id_t from_worker,
        rpc::DeviceMap device_map,
        std::vector<tensorplay::Tensor> inputs);

    tensorplay::tpx::variable_list apply(
        tensorplay::tpx::variable_list&& grads) override;
    size_t num_inputs() const override;

private:
    AutogradMetadata metadata_;
    std::weak_ptr<DistAutogradContext> context_;
    rpc::worker_id_t from_worker_;
    rpc::DeviceMap device_map_;
    std::vector<tensorplay::Tensor> inputs_;
};

}  // namespace tensorplay::distributed::autograd
