#include "SymbolicShapes.h"
#include "ManualNodes.h"
#include "LocalDispatchKeySet.h"

namespace tensorplay::tpx {
namespace {

bool symbolic(const std::vector<SymInt>& sizes) {
    return std::any_of(sizes.begin(), sizes.end(),
                       [](const SymInt& size) { return size.is_symbolic(); });
}

std::vector<int64_t> concrete(const std::vector<SymInt>& sizes) {
    std::vector<int64_t> result;
    result.reserve(sizes.size());
    for (const auto& size : sizes) result.push_back(size.guard_int(__FILE__, __LINE__));
    return result;
}

Tensor dispatch_shape(const char* name, const Tensor& value,
                      const std::vector<SymInt>& sizes) {
    const auto handle = Dispatcher::singleton().findHandle(name);
    return DispatchStub<Tensor, const Tensor&, const std::vector<SymInt>&>::call(
        handle, DispatchKey::Python, value, sizes);
}

} // namespace

std::vector<SymInt> symbolic_sizes(const Tensor& value) {
    std::vector<SymInt> result;
    result.reserve(value.dim());
    const auto keys = tensorplay::impl::tls_local_dispatch_key_set();
    const bool tracing = keys.included.has(DispatchKey::Python) && !keys.excluded.has(DispatchKey::Python);
    for (int64_t axis = 0; axis < value.dim(); ++axis) {
        result.push_back(tracing ? Tensor::sym_size(value, axis) : SymInt(value.size(axis)));
    }
    return result;
}

Tensor expand_symint(const Tensor& value, const std::vector<SymInt>& sizes) {
    if (symbolic(sizes) && tensorplay::impl::tls_local_dispatch_key_set().included.has(DispatchKey::Python)) {
        return dispatch_shape("_symbolic.expand", value, sizes);
    }
    return ops::expand(value, concrete(sizes));
}

Tensor reshape_symint(const Tensor& value, const std::vector<SymInt>& sizes) {
    if (symbolic(sizes) && tensorplay::impl::tls_local_dispatch_key_set().included.has(DispatchKey::Python)) {
        return dispatch_shape("_symbolic.reshape", value, sizes);
    }
    return ops::reshape(value, concrete(sizes));
}

Tensor div_symint(const Tensor& value, const SymInt& divisor) {
    if (divisor.is_symbolic() && tensorplay::impl::tls_local_dispatch_key_set().included.has(DispatchKey::Python)) {
        const auto handle = Dispatcher::singleton().findHandle("_symbolic.div");
        return DispatchStub<Tensor, const Tensor&, const SymInt&>::call(
            handle, DispatchKey::Python, value, divisor);
    }
    return ops::div(value, Scalar(divisor.guard_int(__FILE__, __LINE__)));
}

Tensor sum_backward_symint(const Tensor& grad, const std::vector<SymInt>& sizes,
                           std::vector<int64_t> dims, bool keepdim) {
    if (sizes.empty()) return grad;
    dims = wrap_dims(dims, static_cast<int64_t>(sizes.size()));
    if (dims.empty()) dims = all_dims(static_cast<int64_t>(sizes.size()));
    Tensor restored = grad;
    if (!keepdim) {
        std::sort(dims.begin(), dims.end());
        for (const auto dim : dims) restored = ops::unsqueeze(restored, dim);
    }
    return expand_symint(restored, sizes);
}

Tensor mean_backward_symint(const Tensor& grad, const std::vector<SymInt>& sizes,
                            std::vector<int64_t> dims, bool keepdim) {
    dims = wrap_dims(dims, static_cast<int64_t>(sizes.size()));
    if (dims.empty()) dims = all_dims(static_cast<int64_t>(sizes.size()));
    SymInt count(1);
    for (const auto dim : dims) count *= sizes[dim];
    return sum_backward_symint(div_symint(grad, count), sizes, dims, keepdim);
}

} // namespace tensorplay::tpx
