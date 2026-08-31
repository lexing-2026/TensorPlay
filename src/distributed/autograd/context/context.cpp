#include "context.h"

#include "rpc/rpc_agent.h"

#include <pybind11/gil.h>

#include <stdexcept>

namespace tensorplay::distributed::autograd {

DistAutogradContext::DistAutogradContext(
    int64_t context_id,
    rpc::RpcAgent* agent)
    : context_id_(context_id), agent_(agent) {}

int64_t DistAutogradContext::context_id() const noexcept {
    return context_id_;
}

void DistAutogradContext::add_send(
    int64_t message_id,
    const std::shared_ptr<SendRpcBackward>& function) {
    if (!function) {
        throw std::invalid_argument("distributed autograd send function is null");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!send_functions_.emplace(message_id, function).second) {
        throw std::runtime_error("distributed autograd message id is already registered");
    }
}

void DistAutogradContext::add_recv(
    int64_t message_id,
    const std::shared_ptr<RecvRpcBackward>& function) {
    if (!function) {
        throw std::invalid_argument("distributed autograd receive function is null");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!recv_functions_.emplace(message_id, function).second) {
        throw std::runtime_error("distributed autograd message id is already registered");
    }
}

std::shared_ptr<SendRpcBackward> DistAutogradContext::send_function(
    int64_t message_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iterator = send_functions_.find(message_id);
    if (iterator == send_functions_.end()) {
        throw std::runtime_error(
            "distributed autograd send function was not found");
    }
    return iterator->second;
}

std::unordered_map<int64_t, std::shared_ptr<SendRpcBackward>>
DistAutogradContext::send_functions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return send_functions_;
}

std::unordered_map<int64_t, std::shared_ptr<RecvRpcBackward>>
DistAutogradContext::recv_functions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return recv_functions_;
}

void DistAutogradContext::add_known_worker(rpc::worker_id_t worker_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    known_workers_.insert(worker_id);
}

std::unordered_set<rpc::worker_id_t> DistAutogradContext::known_workers() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return known_workers_;
}

void DistAutogradContext::accumulate_grad(
    const tensorplay::Tensor& variable,
    const tensorplay::Tensor& grad) {
    if (!variable.defined() || !grad.defined()) {
        return;
    }
    const void* key = variable.unsafeGetTensorImpl().get();
    std::lock_guard<std::mutex> lock(mutex_);
    auto iterator = accumulated_grads_.find(key);
    if (iterator == accumulated_grads_.end()) {
        accumulated_grads_.emplace(key, std::make_pair(variable, grad));
    } else {
        iterator->second.second = iterator->second.second + grad;
    }
}

std::vector<std::pair<tensorplay::Tensor, tensorplay::Tensor>>
DistAutogradContext::gradients() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<tensorplay::Tensor, tensorplay::Tensor>> result;
    result.reserve(accumulated_grads_.size());
    for (const auto& entry : accumulated_grads_) {
        result.push_back(entry.second);
    }
    return result;
}

void DistAutogradContext::add_outstanding_rpc(
    const rpc::RpcFuturePtr& future) {
    if (!future) {
        throw std::invalid_argument("distributed autograd RPC future is null");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    outstanding_rpcs_.push_back(future);
}

void DistAutogradContext::wait_outstanding_rpcs() {
    std::vector<rpc::RpcFuturePtr> futures;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        futures = outstanding_rpcs_;
    }
    for (const auto& future : futures) {
        pybind11::gil_scoped_acquire gil;
        pybind11::object ignored = future->wait(-1.0);
        (void)ignored;
    }
}

void DistAutogradContext::clear_outstanding_rpcs() {
    std::lock_guard<std::mutex> lock(mutex_);
    outstanding_rpcs_.clear();
}

rpc::RpcAgent* DistAutogradContext::agent() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return agent_;
}

void DistAutogradContext::set_agent(rpc::RpcAgent* agent) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    agent_ = agent;
}

bool DistAutogradContext::retain_graph() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return retain_graph_;
}

void DistAutogradContext::set_retain_graph(bool value) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    retain_graph_ = value;
}

namespace {
thread_local ContextPtr current_context;
}

ThreadLocalDistAutogradContext::ThreadLocalDistAutogradContext(
    ContextPtr context)
    : previous_(std::move(current_context)) {
    current_context = std::move(context);
}

ThreadLocalDistAutogradContext::~ThreadLocalDistAutogradContext() {
    current_context = std::move(previous_);
}

ContextPtr ThreadLocalDistAutogradContext::current() {
    return current_context;
}

}  // namespace tensorplay::distributed::autograd
