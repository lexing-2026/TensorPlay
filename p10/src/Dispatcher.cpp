#include "Dispatcher.h"
#include <functional>
#include <iostream>
#include <stdexcept>

namespace tensorplay {

Dispatcher& Dispatcher::singleton() {
    static Dispatcher* instance = new Dispatcher();
    return *instance;
}

Dispatcher::Dispatcher() {
    for (auto& slot : op_slots_) {
        slot.store(nullptr, std::memory_order_relaxed);
    }
}

void Dispatcher::install_slot(DispatchTable* table) {
    const std::size_t mask = kOpSlotCount - 1;
    std::size_t h = std::hash<std::string>{}(table->name);
    for (std::size_t probe = 0; probe < kOpSlotCount; ++probe) {
        std::atomic<DispatchTable*>& slot = op_slots_[(h + probe) & mask];
        DispatchTable* occupant = slot.load(std::memory_order_relaxed);
        if (occupant == nullptr) {
            slot.store(table, std::memory_order_release);
            return;
        }
        if (occupant->name == table->name) {
            return;
        }
    }
}

void Dispatcher::registerKernel(const std::string& op_name, DispatchKey key, KernelFunction kernel,
                                const std::type_info* signature) {
    if (dispatchKeyIndex(key) >= kDispatchKeyCount) {
        throw std::invalid_argument("invalid dispatch key for operator: " + op_name);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto& table = operators_[op_name];
    if (!table) {
        table = std::make_unique<DispatchTable>(op_name);
        install_slot(table.get());
    }
    table->signatures[dispatchKeyIndex(key)].store(signature, std::memory_order_release);
    table->kernels[dispatchKeyIndex(key)].store(kernel, std::memory_order_release);
}

const std::type_info* Dispatcher::signature(const std::string& op_name, DispatchKey key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = operators_.find(op_name);
    if (it == operators_.end() || dispatchKeyIndex(key) >= kDispatchKeyCount) {
        return nullptr;
    }
    return it->second->signatures[dispatchKeyIndex(key)].load(std::memory_order_acquire);
}

KernelFunction Dispatcher::getKernel(const std::string& op_name, DispatchKey key) {
    return findHandle(op_name).getKernel(key);
}

OperatorHandle Dispatcher::findHandle(const std::string& op_name) {
    // Hot path: registered tables are published once and never reassigned,
    // so the probe array answers hits without the registry mutex.  A miss
    // falls back to the mutex and creates the table, keeping handle
    // identity stable for registrations that land later.
    const std::size_t mask = kOpSlotCount - 1;
    std::size_t h = std::hash<std::string>{}(op_name);
    for (std::size_t probe = 0; probe < kOpSlotCount; ++probe) {
        DispatchTable* table =
            op_slots_[(h + probe) & mask].load(std::memory_order_acquire);
        if (table == nullptr) {
            break;
        }
        if (table->name == op_name) {
            return OperatorHandle(table);
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto& table = operators_[op_name];
    if (!table) {
        table = std::make_unique<DispatchTable>(op_name);
        install_slot(table.get());
    }
    return OperatorHandle(table.get());
}

std::vector<std::string> Dispatcher::operator_names() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(operators_.size());
    for (const auto& entry : operators_) {
        names.push_back(entry.first);
    }
    return names;
}

KernelFunction Dispatcher::direct_kernel(const std::string& op_name, DispatchKey key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = operators_.find(op_name);
    if (it == operators_.end() || dispatchKeyIndex(key) >= kDispatchKeyCount) {
        return nullptr;
    }
    return it->second->kernels[dispatchKeyIndex(key)].load(std::memory_order_acquire);
}

bool Dispatcher::has_kernel(const std::string& op_name, DispatchKey key) const {
    return direct_kernel(op_name, key) != nullptr;
}

} // namespace tensorplay
