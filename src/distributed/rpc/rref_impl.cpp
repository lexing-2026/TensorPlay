#include "rref_impl.h"

#include "../autograd/autograd.h"
#include "../autograd/rpc_messages/rref_backward_req.h"
#include "rpc_agent.h"
#include "rpc_runtime.h"

#include "Autograd.h"
#include "Tensor.h"

#include <sstream>

namespace tensorplay::distributed::rpc {

RpcRRef::RpcRRef(
    RpcRuntime* runtime,
    WorkerInfo owner,
    RRefId rref_id,
    ForkId fork_id,
    RpcFuturePtr creation,
    std::shared_ptr<RRefState> local_state,
    std::weak_ptr<std::atomic<bool>> runtime_lifetime)
    : runtime_(runtime),
      owner_(std::move(owner)),
      rref_id_(rref_id),
      fork_id_(fork_id),
      creation_(std::move(creation)),
      local_state_(std::move(local_state)),
      runtime_lifetime_(std::move(runtime_lifetime)) {}

RpcRRef::~RpcRRef() {
    try {
        const auto lifetime = runtime_lifetime_.lock();
        if (!lifetime || !lifetime->load(std::memory_order_acquire)) {
            return;
        }
        if (runtime_ != nullptr && runtime_->initialized() && !is_owner()) {
            runtime_->delete_rref(*this);
        } else if (runtime_ != nullptr && runtime_->initialized() && local_state_) {
            runtime_->delete_rref(*this);
        }
    } catch (...) {
    }
}

const WorkerInfo& RpcRRef::owner() const noexcept {
    return owner_;
}

const RRefId& RpcRRef::rref_id() const noexcept {
    return rref_id_;
}

const ForkId& RpcRRef::fork_id() const noexcept {
    return fork_id_;
}

bool RpcRRef::is_owner() const {
    return runtime_ != nullptr && runtime_->initialized() &&
        owner_.id == runtime_->current_worker().id;
}

bool RpcRRef::confirmed_by_owner() const noexcept {
    if (creation_ == nullptr || !creation_->done()) {
        return false;
    }
    try {
        return creation_->exception(0.0).is_none();
    } catch (...) {
        return false;
    }
}

py::object RpcRRef::to_here(double timeout_seconds) const {
    if (creation_ != nullptr) {
        creation_->wait(timeout_seconds);
    }
    if (runtime_ == nullptr) {
        throw std::runtime_error("RRef is detached from its runtime");
    }
    return runtime_->fetch_rref(*this, timeout_seconds);
}

py::object RpcRRef::local_value() const {
    if (!is_owner()) {
        throw std::runtime_error("local_value is only available on the owner");
    }
    if (runtime_ == nullptr) {
        throw std::runtime_error("RRef is detached from its runtime");
    }
    return runtime_->fetch_rref(*this, -1.0);
}

void RpcRRef::backward(int64_t context_id, bool retain_graph) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("RRef is detached from its runtime");
    }
    if (is_owner()) {
        const py::object value = local_value();
        const auto root = value.cast<tensorplay::Tensor>();
        if (context_id < 0) {
            tensorplay::tpx::backward(root);
        } else {
            tensorplay::distributed::autograd::backward(
                context_id, {root}, retain_graph);
        }
        return;
    }
    if (context_id < 0) {
        throw std::invalid_argument(
            "user RRefs require a distributed autograd context");
    }
    auto* agent = RpcAgent::current_rpc_agent();
    if (agent == nullptr) {
        throw std::runtime_error("RPC runtime is not active");
    }
    auto future = agent->send(
        agent->get_worker_info(owner_.id),
        tensorplay::distributed::autograd::RRefBackwardReq(
            rref_id_, context_id, retain_graph)
            .to_message());
    future->wait(-1.0);
}

std::shared_ptr<RpcRRef> RpcRRef::fork() const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("RRef is detached from its runtime");
    }
    runtime_->fork_rref(*this);
    const ForkId fork_id(runtime_->current_worker().id,
                         runtime_->next_local_id_.fetch_add(1));
    return std::make_shared<RpcRRef>(
        runtime_,
        owner_,
        rref_id_,
        fork_id,
        creation_,
        local_state_,
        runtime_lifetime_);
}

std::string RpcRRef::repr() const {
    std::ostringstream stream;
    stream << "RRef(owner='" << owner_.name << "', id=" << rref_id_
           << ", fork=" << fork_id_ << ')';
    return stream.str();
}

}  // namespace tensorplay::distributed::rpc
