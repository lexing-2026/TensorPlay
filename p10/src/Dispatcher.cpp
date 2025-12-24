#include "Dispatcher.h"
#include <iostream>

namespace tensorplay {

Dispatcher& Dispatcher::singleton() {
    static Dispatcher* instance = new Dispatcher();
    return *instance;
}

void Dispatcher::registerKernel(const std::string& op_name, DispatchKey key, KernelFunction kernel) {
    std::lock_guard<std::mutex> lock(mutex_);
    operators_[op_name].kernels[key] = kernel;
}

KernelFunction Dispatcher::getKernel(const std::string& op_name, DispatchKey key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = operators_.find(op_name);
    if (it == operators_.end()) {
        return nullptr;
    }
    auto kit = it->second.kernels.find(key);
    if (kit == it->second.kernels.end()) {
        return nullptr;
    }
    return kit->second;
}

} // namespace tensorplay
