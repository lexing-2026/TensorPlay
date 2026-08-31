#include "recvrpc_backward.h"

#include "rpc/rpc_agent.h"
#include "rpc_messages/propagate_gradients_req.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <stdexcept>

namespace tensorplay::distributed::autograd {

RecvRpcBackward::RecvRpcBackward(
    AutogradMetadata metadata,
    ContextPtr context,
    rpc::worker_id_t from_worker,
    rpc::DeviceMap device_map,
    std::vector<tensorplay::Tensor> inputs)
    : metadata_(metadata),
      context_(std::move(context)),
      from_worker_(from_worker),
      device_map_(std::move(device_map)),
      inputs_(std::move(inputs)) {}

size_t RecvRpcBackward::num_inputs() const {
    return inputs_.size();
}

tensorplay::tpx::variable_list RecvRpcBackward::apply(
    tensorplay::tpx::variable_list&& grads) {
    auto context = context_.lock();
    if (!context) {
        throw std::runtime_error(
            "distributed autograd context is no longer available");
    }
    if (grads.size() != inputs_.size()) {
        throw std::runtime_error(
            "distributed autograd gradient count does not match the RPC tensors");
    }

    tensorplay::tpx::variable_list output_grads;
    output_grads.reserve(grads.size());
    for (size_t index = 0; index < grads.size(); ++index) {
        if (grads[index].defined()) {
            output_grads.push_back(grads[index]);
        } else {
            output_grads.push_back(
                tensorplay::tpx::ops::zeros_like(inputs_[index]));
        }
    }

    rpc::RpcAgent* agent = context->agent();
    if (agent == nullptr) {
        agent = rpc::RpcAgent::current_rpc_agent();
    }
    if (agent == nullptr) {
        throw std::runtime_error(
            "distributed autograd has no active RPC agent");
    }

    auto message = PropagateGradientsReq(
        metadata_, output_grads, context->retain_graph()).to_message();
    auto future = agent->send(
        agent->get_worker_info(from_worker_),
        std::move(message),
        -1.0,
        device_map_);
    context->add_outstanding_rpc(future);
    return {};
}

}  // namespace tensorplay::distributed::autograd
