#pragma once
#include "DType.h"
#include "Exception.h"
#include <array>

namespace tensorplay {

// StructuredKernel: A registry for DType-specific implementations
// This allows separating the dispatch logic (choosing the implementation based on DType)
// from the kernel implementation itself.
// Usage:
//   StructuredKernel<void(const Tensor&, const Tensor&)> my_kernel;
//   my_kernel.registerImpl(DType::Float32, &my_float_impl);
//   my_kernel(tensor.dtype(), tensor, other);

template<typename FuncType>
class StructuredKernel;

template<typename Return, typename... Args>
class StructuredKernel<Return(Args...)> {
public:
    using FuncPtr = Return(*)(Args...);
    
    void registerImpl(ScalarType dtype, FuncPtr impl) {
        if (dtype == ScalarType::Undefined || dtype == ScalarType::NumOptions) return;
        impls_[static_cast<int>(dtype)] = impl;
    }
    
    FuncPtr getImpl(ScalarType dtype) const {
        if (dtype == ScalarType::Undefined || dtype == ScalarType::NumOptions) {
            return nullptr;
        }
        return impls_[static_cast<int>(dtype)];
    }
    
    // Dispatch helper
    Return operator()(ScalarType dtype, Args... args) const {
        auto impl = getImpl(dtype);
        if (!impl) {
            TP_THROW(NotImplementedError, "Operator not implemented for this dtype");
        }
        return impl(std::forward<Args>(args)...);
    }

private:
    std::array<FuncPtr, static_cast<int>(ScalarType::NumOptions)> impls_ = {nullptr};
};

} // namespace tensorplay
