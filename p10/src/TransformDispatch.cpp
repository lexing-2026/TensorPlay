#include "TransformDispatch.h"

#include <algorithm>

#include "Exception.h"

namespace tensorplay {
namespace transform {
namespace {

thread_local std::vector<Layer> layers;
thread_local int64_t next_level = 0;
thread_local size_t disabled_depth = 0;

int64_t normalize_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(IndexError, "batch dimension is out of range");
    }
    return dim;
}

} // namespace

DisableTransformsGuard::DisableTransformsGuard() {
    ++disabled_depth;
}

DisableTransformsGuard::~DisableTransformsGuard() {
    if (!active_) return;
    if (disabled_depth > 0) --disabled_depth;
    active_ = false;
}

int64_t push_vmap(int64_t batch_size, Randomness randomness) {
    if (batch_size < 0) {
        TP_THROW(ValueError, "vmap batch size must be non-negative");
    }
    Layer layer;
    layer.kind = Kind::Vmap;
    layer.level = next_level++;
    layer.batch_size = batch_size;
    layer.randomness = randomness;
    layers.push_back(layer);
    return layer.level;
}

Layer pop_layer() {
    if (layers.empty()) {
        TP_THROW(RuntimeError, "cannot pop an empty transform stack");
    }
    Layer layer = layers.back();
    layers.pop_back();
    return layer;
}

std::optional<Layer> current_layer() {
    if (layers.empty()) return std::nullopt;
    return layers.back();
}

std::vector<Layer> layer_stack() {
    return layers;
}

void clear_layers() {
    layers.clear();
}

bool are_transforms_active() {
    return !layers.empty() && disabled_depth == 0;
}

DispatchKey dispatch_key_for_random(DispatchKey backend) {
    if (are_transforms_active() && layers.back().kind == Kind::Vmap) {
        return toVmapKey(backend);
    }
    return backend;
}

Tensor make_batched(const Tensor& value, int64_t dim, int64_t level) {
    if (!value.defined()) {
        TP_THROW(ValueError, "cannot batch an undefined tensor");
    }
    if (level < 0) {
        TP_THROW(ValueError, "transform level must be non-negative");
    }
    const int64_t public_ndim = value.dim();
    const int64_t bdim = normalize_dim(dim, public_ndim);
    const auto sizes = static_cast<std::vector<int64_t>>(value.shape());
    const auto strides = value.strides();
    std::vector<int64_t> logical_sizes;
    std::vector<int64_t> logical_strides;
    logical_sizes.reserve(sizes.size() - 1);
    logical_strides.reserve(strides.size() - 1);
    for (int64_t d = 0; d < static_cast<int64_t>(sizes.size()); ++d) {
        if (d == bdim) continue;
        logical_sizes.push_back(sizes[static_cast<size_t>(d)]);
        logical_strides.push_back(strides[static_cast<size_t>(d)]);
    }
    auto impl = std::make_shared<TensorImpl>(value.unsafeGetTensorImpl(),
                                             logical_sizes,
                                             logical_strides);
    impl->set_transform_value(value.unsafeGetTensorImpl(), bdim, level);
    return Tensor(std::move(impl));
}

std::tuple<Tensor, std::optional<int64_t>> unwrap_at_level(
    const Tensor& value, int64_t level) {
    if (!value.defined() || !value.is_batched() || value.batch_level() != level) {
        return {value, std::nullopt};
    }
    return {value.transform_value(), value.batch_dim()};
}

Tensor unwrap_all(const Tensor& value) {
    Tensor result = value;
    while (result.defined() && result.is_batched()) {
        result = result.transform_value();
    }
    return result;
}

bool is_batched_at_level(const Tensor& value, int64_t level) {
    return value.defined() && value.is_batched() && value.batch_level() == level;
}

int64_t actual_dim(const Tensor& value, int64_t public_dim, bool wrap_dim) {
    if (!value.is_batched()) {
        if (!wrap_dim && public_dim < 0) return public_dim;
        return normalize_dim(public_dim, value.dim());
    }
    int64_t dim = public_dim;
    if (wrap_dim && dim < 0) dim += value.dim();
    if (dim < 0 || dim >= value.dim()) {
        TP_THROW(IndexError, "public dimension is out of range");
    }
    return dim < value.batch_dim() ? dim : dim + 1;
}

Tensor move_batch_dim(const Tensor& value, int64_t dim) {
    if (!value.is_batched()) return value;
    const int64_t from = normalize_dim(value.batch_dim(), value.dim() + 1);
    const int64_t to = normalize_dim(dim, value.dim() + 1);
    if (from == to) return value;

    std::vector<int64_t> permutation;
    permutation.reserve(static_cast<size_t>(value.dim() + 1));
    for (int64_t d = 0; d < value.dim() + 1; ++d) {
        if (d == from) continue;
        permutation.push_back(d);
    }
    permutation.insert(permutation.begin() + to, from);
    return value.transform_value().permute(permutation);
}

} // namespace transform
} // namespace tensorplay
