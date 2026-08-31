#include "container.h"

#include "../rpc_messages/cleanup_autograd_context_req.h"
#include "../../rpc/rpc_agent.h"

#include <limits>
#include <stdexcept>
#include <thread>

namespace tensorplay::distributed::autograd {
namespace {

constexpr int kIdBits = 48;
constexpr int64_t kIdMask = (int64_t{1} << kIdBits) - 1;
constexpr int64_t kInvalidContextId = -1;
thread_local int64_t t_current_context_id = kInvalidContextId;
std::mutex initialization_mutex;

int64_t make_first_id(rpc::worker_id_t worker_id) {
    const auto value = static_cast<uint64_t>(
        static_cast<uint16_t>(worker_id));
    return static_cast<int64_t>(value << kIdBits);
}

}  // namespace

DistAutogradContainer::DistAutogradContainer(uint32_t shards) {
    (void)shards;
}

uint32_t DistAutogradContainer::compute_shards() {
    uint32_t shards = 1;
    const unsigned count = std::thread::hardware_concurrency();
    const unsigned target = count == 0 ? 128U : count * 2U;
    while (shards < target) {
        shards <<= 1;
    }
    return shards;
}

DistAutogradContainer& DistAutogradContainer::instance_internal() {
    static DistAutogradContainer* container =
        new DistAutogradContainer(compute_shards());
    return *container;
}

bool DistAutogradContainer::is_initialized_global() noexcept {
    return instance_internal().is_initialized();
}

DistAutogradContainer& DistAutogradContainer::init(
    rpc::worker_id_t worker_id,
    rpc::RpcAgent* agent) {
    if (worker_id < 0) {
        throw std::invalid_argument(
            "distributed autograd worker id must be non-negative");
    }
    std::lock_guard<std::mutex> guard(initialization_mutex);
    auto& container = instance_internal();
    if (container.initialized_ && container.worker_id_ != worker_id) {
        throw std::runtime_error(
            "distributed autograd is already initialized for another worker");
    }
    if (!container.initialized_) {
        container.worker_id_ = worker_id;
        const int64_t first = make_first_id(worker_id);
        container.next_context_id_.store(first);
        container.next_message_id_.store(first);
        container.max_id_ = first | kIdMask;
        container.initialized_ = true;
    }
    if (agent != nullptr) {
        container.agent_ = agent;
        for (const auto& entry : container.contexts_) {
            entry.second->set_agent(agent);
        }
    }
    return container;
}

DistAutogradContainer& DistAutogradContainer::instance() {
    auto& container = instance_internal();
    std::lock_guard<std::mutex> guard(initialization_mutex);
    if (!container.initialized_) {
        throw std::runtime_error(
            "distributed autograd has not been initialized");
    }
    return container;
}

ContextPtr DistAutogradContainer::new_context() {
    if (has_current()) {
        throw std::runtime_error(
            "the current thread already owns a distributed autograd context");
    }
    const int64_t context_id = next_context_id_.fetch_add(1);
    if (context_id >= max_id_) {
        throw std::overflow_error(
            "distributed autograd context id space is exhausted");
    }
    auto context = std::make_shared<DistAutogradContext>(context_id, agent_);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        contexts_.emplace(context_id, context);
    }
    t_current_context_id = context_id;
    return context;
}

ContextPtr DistAutogradContainer::get_or_create(int64_t context_id) {
    if (context_id < 0) {
        throw std::invalid_argument(
            "distributed autograd context id must be non-negative");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto iterator = contexts_.find(context_id);
    if (iterator != contexts_.end()) {
        return iterator->second;
    }
    auto context = std::make_shared<DistAutogradContext>(context_id, agent_);
    contexts_.emplace(context_id, context);
    return context;
}

ContextPtr DistAutogradContainer::retrieve(int64_t context_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iterator = contexts_.find(context_id);
    if (iterator == contexts_.end()) {
        throw std::runtime_error(
            "distributed autograd context was not found");
    }
    return iterator->second;
}

void DistAutogradContainer::release(int64_t context_id) {
    ContextPtr context;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto iterator = contexts_.find(context_id);
        if (iterator == contexts_.end()) {
            throw std::runtime_error(
                "distributed autograd context was not found");
        }
        context = std::move(iterator->second);
        contexts_.erase(iterator);
    }
    if (t_current_context_id == context_id) {
        t_current_context_id = kInvalidContextId;
    }
    const auto agent = context->agent();
    if (agent == nullptr) {
        return;
    }
    for (const auto worker_id : context->known_workers()) {
        if (worker_id == worker_id_) {
            continue;
        }
        try {
            agent->send_with_retries(
                agent->get_worker_info(worker_id),
                CleanupAutogradContextReq(context_id).to_message());
        } catch (...) {
        }
    }
}

void DistAutogradContainer::release_if_present(int64_t context_id) {
    ContextPtr context;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto iterator = contexts_.find(context_id);
        if (iterator == contexts_.end()) {
            return;
        }
        context = std::move(iterator->second);
        contexts_.erase(iterator);
    }
    if (t_current_context_id == context_id) {
        t_current_context_id = kInvalidContextId;
    }
    const auto agent = context->agent();
    if (agent == nullptr) {
        return;
    }
    for (const auto worker_id : context->known_workers()) {
        if (worker_id == worker_id_) {
            continue;
        }
        try {
            agent->send_with_retries(
                agent->get_worker_info(worker_id),
                CleanupAutogradContextReq(context_id).to_message());
        } catch (...) {
        }
    }
}

void DistAutogradContainer::validate(int64_t context_id) const {
    (void)retrieve(context_id);
}

bool DistAutogradContainer::is_initialized() const noexcept {
    std::lock_guard<std::mutex> guard(initialization_mutex);
    return initialized_;
}

ContextPtr DistAutogradContainer::current() const {
    if (!has_current()) {
        throw std::runtime_error(
            "the current thread has no distributed autograd context");
    }
    return retrieve(t_current_context_id);
}

bool DistAutogradContainer::has_current() noexcept {
    return t_current_context_id != kInvalidContextId;
}

int64_t DistAutogradContainer::current_context_id() noexcept {
    return t_current_context_id;
}

void DistAutogradContainer::set_current_context_id(int64_t context_id) {
    if (has_current()) {
        throw std::runtime_error(
            "the current thread already owns a distributed autograd context");
    }
    t_current_context_id = context_id;
}

void DistAutogradContainer::force_current_context_id(int64_t context_id) noexcept {
    t_current_context_id = context_id;
}

void DistAutogradContainer::clear_current_context() const noexcept {
    t_current_context_id = kInvalidContextId;
}

int64_t DistAutogradContainer::new_message_id() {
    const int64_t message_id = next_message_id_.fetch_add(1);
    if (message_id >= max_id_) {
        throw std::overflow_error(
            "distributed autograd message id space is exhausted");
    }
    return message_id;
}

int64_t DistAutogradContainer::max_id() const noexcept {
    return max_id_;
}

rpc::worker_id_t DistAutogradContainer::worker_id() const noexcept {
    return worker_id_;
}

size_t DistAutogradContainer::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return contexts_.size();
}

void DistAutogradContainer::set_agent(rpc::RpcAgent* agent) {
    std::lock_guard<std::mutex> lock(mutex_);
    agent_ = agent;
    for (const auto& entry : contexts_) {
        entry.second->set_agent(agent);
    }
}

rpc::RpcAgent* DistAutogradContainer::agent() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return agent_;
}

ContextGuard::ContextGuard(int64_t context_id)
    : previous_(DistAutogradContainer::current_context_id()) {
    DistAutogradContainer::force_current_context_id(context_id);
}

ContextGuard::~ContextGuard() {
    DistAutogradContainer::force_current_context_id(previous_);
}

}  // namespace tensorplay::distributed::autograd
