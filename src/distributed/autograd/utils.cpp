#include "utils.h"

#include "AccumulateGrad.h"
#include "Autograd.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <pybind11/stl.h>

#include <stdexcept>
#include <unordered_set>

namespace tensorplay::distributed::autograd {
namespace {

void collect_leaves(
    const std::shared_ptr<tensorplay::tpx::Node>& function,
    SendRpcBackward& send,
    std::unordered_set<tensorplay::tpx::Node*>& seen) {
    if (!function || !seen.insert(function.get()).second) {
        return;
    }
    if (auto* accumulate =
            dynamic_cast<tensorplay::tpx::AccumulateGrad*>(function.get())) {
        const tensorplay::Tensor variable = accumulate->value_;
        const auto gradient = tensorplay::tpx::impl::grad(variable);
        const auto snapshot = gradient.defined()
            ? tensorplay::tpx::ops::clone(
                  tensorplay::tpx::ops::detach(gradient))
            : tensorplay::Tensor();
        send.add_leaf(variable, snapshot);
        return;
    }
    for (const auto& edge : function->next_edges()) {
        collect_leaves(edge.function, send, seen);
    }
}

}  // namespace

void add_send_rpc_backward(
    const ContextPtr& context,
    const AutogradMetadata& metadata,
    std::vector<pybind11::object>& tensors) {
    if (!context || !metadata.valid()) {
        throw std::invalid_argument(
            "distributed autograd send metadata is invalid");
    }

    std::vector<tensorplay::Tensor> requiring_grad;
    requiring_grad.reserve(tensors.size());
    for (const auto& object : tensors) {
        const auto tensor = object.cast<tensorplay::Tensor>();
        if (tensor.requires_grad()) {
            requiring_grad.push_back(tensor);
        }
    }
    auto function = std::make_shared<SendRpcBackward>();
    std::vector<tensorplay::tpx::Edge> edges;
    edges.reserve(requiring_grad.size());
    std::unordered_set<tensorplay::tpx::Node*> seen;
    for (const auto& tensor : requiring_grad) {
        auto tensor_edges = tensorplay::tpx::collect_next_edges(tensor);
        for (auto& edge : tensor_edges) {
            if (edge.function) {
                collect_leaves(edge.function, *function, seen);
            }
            edges.push_back(std::move(edge));
        }
    }
    function->add_next_edge_list(std::move(edges));
    context->add_send(metadata.message_id, function);
}

ContextPtr add_recv_rpc_backward(
    const AutogradMetadata& metadata,
    std::vector<pybind11::object>& tensors,
    rpc::worker_id_t from_worker,
    rpc::DeviceMap device_map) {
    if (!metadata.valid()) {
        throw std::invalid_argument(
            "distributed autograd receive metadata is invalid");
    }
    auto& container = DistAutogradContainer::instance();
    auto context = container.get_or_create(metadata.context_id);
    std::vector<tensorplay::Tensor> requiring_grad;
    requiring_grad.reserve(tensors.size());
    for (const auto& object : tensors) {
        const auto tensor = object.cast<tensorplay::Tensor>();
        if (tensor.requires_grad()) {
            requiring_grad.push_back(tensor);
        }
    }
    if (!requiring_grad.empty()) {
        auto function = std::make_shared<RecvRpcBackward>(
            metadata,
            context,
            from_worker,
            std::move(device_map),
            requiring_grad);
        size_t output_nr = 0;
        for (const auto& object : tensors) {
            auto tensor = object.cast<tensorplay::Tensor>();
            if (!tensor.requires_grad()) {
                continue;
            }
            tensorplay::tpx::impl::set_grad_fn(tensor, function, output_nr++);
        }
        context->add_recv(metadata.message_id, function);
    }
    return context;
}

}  // namespace tensorplay::distributed::autograd
