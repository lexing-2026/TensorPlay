#include "Dispatcher.h"
#include <iostream>
#include <stdexcept>

namespace tensorplay {

Dispatcher& Dispatcher::singleton() {
    static Dispatcher* instance = new Dispatcher();
    return *instance;
}

void Dispatcher::registerKernel(const std::string& op_name, DispatchKey key, KernelFunction kernel) {
    if (dispatchKeyIndex(key) >= kDispatchKeyCount) {
        throw std::invalid_argument("invalid dispatch key for operator: " + op_name);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto& table = operators_[op_name];
    if (!table) {
        table = std::make_unique<DispatchTable>(op_name);
    }
    table->kernels[dispatchKeyIndex(key)].store(kernel, std::memory_order_release);
}

KernelFunction Dispatcher::getKernel(const std::string& op_name, DispatchKey key) {
    return findHandle(op_name).getKernel(key);
}

OperatorHandle Dispatcher::findHandle(const std::string& op_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& table = operators_[op_name];
    if (!table) {
        table = std::make_unique<DispatchTable>(op_name);
    }
    return OperatorHandle(table.get());
}

} // namespace tensorplay
