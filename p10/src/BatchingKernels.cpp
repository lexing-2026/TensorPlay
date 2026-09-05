#include "BatchingKernels.h"

#include <algorithm>
#include <string>

#include "Context.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "TransformDispatch.h"

namespace tensorplay {
namespace transform {
namespace batch {
namespace {

struct Operand {
    Tensor value;
    std::optional<int64_t> bdim;
    int64_t level = -1;
};

const Layer& layer_for(int64_t level) {
    static thread_local Layer selected;
    const auto stack = layer_stack();
    for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
        if (it->level == level) {
            selected = *it;
            return selected;
        }
    }
    TP_THROW(RuntimeError, "batch tensor refers to an inactive transform level");
}

Operand unwrap_operand(const Tensor& input) {
    if (!input.is_batched()) {
        return {input, std::nullopt, -1};
    }
    const int64_t level = input.batch_level();
    layer_for(level);
    auto unwrapped = unwrap_at_level(input, level);
    if (!std::get<1>(unwrapped).has_value()) {
        TP_THROW(RuntimeError, "failed to unwrap a batch tensor");
    }
    Tensor value = std::get<0>(unwrapped);
    return {std::move(value), std::get<1>(unwrapped), level};
}

int64_t normalize_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(IndexError, "batch rule dimension is out of range");
    }
    return dim;
}

Tensor move_to_front(const Tensor& value, int64_t dim) {
    if (dim == 0) return value;
    const int64_t ndim = value.dim();
    std::vector<int64_t> permutation;
    permutation.reserve(static_cast<size_t>(ndim));
    permutation.push_back(dim);
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != dim) permutation.push_back(d);
    }
    return value.permute(permutation);
}

Tensor expand_unbatched(const Tensor& value, int64_t batch_size,
                        const std::vector<int64_t>& target_shape) {
    std::vector<int64_t> shape;
    shape.reserve(target_shape.size() + 1);
    shape.push_back(batch_size);
    shape.insert(shape.end(), target_shape.begin(), target_shape.end());
    return value.unsqueeze(0).expand(shape);
}

std::vector<int64_t> logical_shape(const Operand& operand) {
    auto physical = static_cast<std::vector<int64_t>>(operand.value.shape());
    if (!operand.bdim.has_value()) return physical;
    const int64_t dim = *operand.bdim;
    if (dim < 0 || dim >= static_cast<int64_t>(physical.size())) {
        TP_THROW(RuntimeError, "batch dimension is inconsistent with tensor metadata");
    }
    physical.erase(physical.begin() + dim);
    return physical;
}

// Unwrap an operand only when it is batched at `level`; tensors batched at
// other levels keep their wrapper, so their public shape hides their own
// batch dim and the underlying operator re-dispatches through their level.
Operand operand_at_level(const Tensor& input, int64_t level) {
    if (!input.is_batched() || input.batch_level() != level) {
        return {input, std::nullopt, -1};
    }
    auto unwrapped = unwrap_at_level(input, level);
    if (!std::get<1>(unwrapped).has_value()) {
        TP_THROW(RuntimeError, "failed to unwrap a batch tensor");
    }
    return {std::get<0>(unwrapped), std::get<1>(unwrapped), level};
}

// Move the batch dim to the front and insert 1-sized dims right after it so
// the batch-dim-free rank reaches `logical_rank`.  Only operands mapped at
// the current level are padded; rank alignment from the right then lets the
// underlying operator broadcast the batch dim against everything else.
Tensor pad_to_logical_rank(const Tensor& value, int64_t bdim,
                           int64_t logical_rank) {
    Tensor front = move_to_front(value, bdim);
    const int64_t logical = front.dim() - 1;
    if (logical >= logical_rank) {
        return front;
    }
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(logical_rank + 1));
    shape.push_back(front.size(0));
    for (int64_t i = logical; i < logical_rank; ++i) {
        shape.push_back(1);
    }
    for (int64_t d = 1; d < front.dim(); ++d) {
        shape.push_back(front.size(static_cast<size_t>(d)));
    }
    return front.view(shape);
}

// Broadcast two operands at the current level: mapped batch dims move to the
// front, mapped operands are padded with 1s so both logical ranks match, and
// the underlying operator performs the elementwise broadcast.  Operands
// without a batch dim at this level — plain tensors, or tensors batched at
// outer levels whose public shape already hides their own batch dim — pass
// through unchanged so the operator re-dispatches through their level.
std::pair<Tensor, Tensor> broadcast_values(const Operand& left,
                                           const Operand& right) {
    const int64_t left_rank = static_cast<int64_t>(left.value.dim()) -
        (left.bdim.has_value() ? 1 : 0);
    const int64_t right_rank = static_cast<int64_t>(right.value.dim()) -
        (right.bdim.has_value() ? 1 : 0);
    const int64_t max_rank = std::max(left_rank, right_rank);
    Tensor left_value = left.bdim.has_value()
        ? pad_to_logical_rank(left.value, *left.bdim, max_rank)
        : left.value;
    Tensor right_value = right.bdim.has_value()
        ? pad_to_logical_rank(right.value, *right.bdim, max_rank)
        : right.value;
    if (left.bdim.has_value() && right.bdim.has_value() &&
        left_value.size(0) != right_value.size(0)) {
        TP_THROW(RuntimeError,
                 "operands mapped at the same level disagree on batch size: ",
                 left_value.size(0), " vs ", right_value.size(0));
    }
    return {std::move(left_value), std::move(right_value)};
}

template <typename Return, typename... Args>
Return call_next(const char* op, const Tensor& device_source, Args... args) {
    DispatchKey dispatch_key = dispatchKeyForTensorArgs(args...);
    if (dispatch_key == DispatchKey::EndOfKeys) {
        dispatch_key = computeDispatchKey(device_source.device());
    }
    return DispatchStub<Return, Args...>::call(
        std::string(op), dispatch_key,
        std::forward<Args>(args)...);
}

template <typename Return, typename... Args>
Return call_device(const char* op, const Device& device, Args... args) {
    return DispatchStub<Return, Args...>::call(
        std::string(op), computeDispatchKey(device),
        std::forward<Args>(args)...);
}

Layer current_vmap_layer() {
    const auto active = current_layer();
    if (!active.has_value() || active->kind != Kind::Vmap) {
        TP_THROW(RuntimeError, "random batch rule requires an active vectorizing layer");
    }
    return *active;
}

void check_randomness(const Layer& layer) {
    if (layer.randomness == Randomness::Error) {
        TP_THROW(RuntimeError,
                 "random operations are not allowed under randomness=error");
    }
}

DType resolve_random_dtype(const std::optional<DType>& dtype) {
    if (dtype.has_value() && *dtype != DType::Undefined) return *dtype;
    return globalContext().defaultDType();
}

Device resolve_random_device(const std::optional<Device>& device) {
    return device.has_value() ? *device : globalContext().defaultDevice();
}

std::vector<int64_t> prepend_batch_size(const std::vector<int64_t>& shape,
                                        int64_t batch_size) {
    std::vector<int64_t> physical;
    physical.reserve(shape.size() + 1);
    physical.push_back(batch_size);
    physical.insert(physical.end(), shape.begin(), shape.end());
    return physical;
}

Tensor expand_same_random(Tensor sample, const Layer& layer) {
    const auto shape = static_cast<std::vector<int64_t>>(sample.shape());
    Tensor expanded = sample.unsqueeze(0).expand(
        prepend_batch_size(shape, layer.batch_size));
    return make_batched(expanded, 0, layer.level);
}

Tensor random_factory(const char* op, const std::vector<int64_t>& shape,
                      std::optional<DType> dtype,
                      std::optional<Device> device) {
    const Layer layer = current_vmap_layer();
    check_randomness(layer);
    const Device target = resolve_random_device(device);
    if (layer.randomness == Randomness::Different) {
        Tensor result = call_device<Tensor, const std::vector<int64_t>&,
                                    std::optional<DType>, std::optional<Device>>(
            op, target, prepend_batch_size(shape, layer.batch_size), dtype,
            device);
        return make_batched(result, 0, layer.level);
    }
    Tensor sample = call_device<Tensor, const std::vector<int64_t>&,
                                std::optional<DType>, std::optional<Device>>(
        op, target, shape, dtype, device);
    return expand_same_random(std::move(sample), layer);
}

Tensor random_int_factory(const char* op, int64_t low, int64_t high,
                          const std::vector<int64_t>& shape, DType dtype,
                          std::optional<Device> device) {
    const Layer layer = current_vmap_layer();
    check_randomness(layer);
    const Device target = resolve_random_device(device);
    if (layer.randomness == Randomness::Different) {
        Tensor result = call_device<Tensor, int64_t, int64_t,
                                    const std::vector<int64_t>&, DType,
                                    std::optional<Device>>(
            op, target, low, high,
            prepend_batch_size(shape, layer.batch_size), dtype, device);
        return make_batched(result, 0, layer.level);
    }
    Tensor sample = call_device<Tensor, int64_t, int64_t,
                                const std::vector<int64_t>&, DType,
                                std::optional<Device>>(
        op, target, low, high, shape, dtype, device);
    return expand_same_random(std::move(sample), layer);
}

Tensor randperm_factory(int64_t n, DType dtype,
                        std::optional<Device> device) {
    const Layer layer = current_vmap_layer();
    check_randomness(layer);
    const Device target = resolve_random_device(device);
    const auto sample = [&]() {
        return call_device<Tensor, int64_t, DType, std::optional<Device>>(
            "randperm", target, n, dtype, device);
    };
    if (layer.randomness != Randomness::Different) {
        return expand_same_random(sample(), layer);
    }
    if (layer.batch_size == 0) {
        Tensor empty(std::vector<int64_t>{0, n}, dtype, target);
        return make_batched(empty, 0, layer.level);
    }
    std::vector<Tensor> values;
    values.reserve(static_cast<size_t>(layer.batch_size));
    for (int64_t i = 0; i < layer.batch_size; ++i) {
        values.push_back(sample());
    }
    Tensor result = call_device<Tensor, const std::vector<Tensor>&, int64_t>(
        "stack", target, values, 0);
    return make_batched(result, 0, layer.level);
}

Tensor random_like_factory(const char* op, const Tensor& input,
                           DType dtype, std::optional<Device> device) {
    const Layer active = current_vmap_layer();
    check_randomness(active);
    const DType output_dtype = dtype == DType::Undefined ? input.dtype() : dtype;
    const auto call = [&](const Tensor& value) {
        return call_next<Tensor, const Tensor&, DType, std::optional<Device>>(
            op, value, value, output_dtype, device);
    };

    if (input.is_batched() && input.batch_level() == active.level) {
        Operand operand = unwrap_operand(input);
        if (active.randomness == Randomness::Different) {
            Tensor result = call(operand.value);
            return make_batched(result, *operand.bdim, active.level);
        }
        Tensor sample = call_next<Tensor, const Tensor&, int64_t, int64_t>(
        "select.int", operand.value, operand.value, *operand.bdim, 0);
        sample = call(sample);
        Tensor expanded = sample.unsqueeze(*operand.bdim).expand(
            static_cast<std::vector<int64_t>>(operand.value.shape()));
        return make_batched(expanded, *operand.bdim, active.level);
    }

    if (active.randomness == Randomness::Same) {
        return expand_same_random(call(input), active);
    }
    Tensor template_value = input.unsqueeze(0).expand(
        prepend_batch_size(static_cast<std::vector<int64_t>>(input.shape()),
                           active.batch_size));
    Tensor result = call(template_value);
    return make_batched(result, 0, active.level);
}

template <typename Function>
Tensor random_like_impl(const Tensor& input, DType dtype,
                        std::optional<Device> device, Function&& call) {
    const Layer active = current_vmap_layer();
    check_randomness(active);
    const DType output_dtype = dtype == DType::Undefined ? input.dtype() : dtype;
    if (input.is_batched() && input.batch_level() == active.level) {
        Operand operand = unwrap_operand(input);
        if (active.randomness == Randomness::Different) {
            return make_batched(call(operand.value, output_dtype, device),
                                *operand.bdim, active.level);
        }
        Tensor sample = call_next<Tensor, const Tensor&, int64_t, int64_t>(
        "select.int", operand.value, operand.value, *operand.bdim, 0);
        sample = call(sample, output_dtype, device);
        Tensor expanded = sample.unsqueeze(*operand.bdim).expand(
            static_cast<std::vector<int64_t>>(operand.value.shape()));
        return make_batched(expanded, *operand.bdim, active.level);
    }
    if (active.randomness == Randomness::Same) {
        return expand_same_random(call(input, output_dtype, device), active);
    }
    Tensor template_value = input.unsqueeze(0).expand(
        prepend_batch_size(static_cast<std::vector<int64_t>>(input.shape()),
                           active.batch_size));
    return make_batched(call(template_value, output_dtype, device), 0,
                        active.level);
}

template <typename Function>
Tensor unary_impl(const Tensor& input, Function&& call) {
    Operand operand = unwrap_operand(input);
    if (!operand.bdim.has_value()) {
        TP_THROW(RuntimeError, "unary batch rule received an unbatched operand");
    }
    Tensor result = call(operand.value);
    return make_batched(result, *operand.bdim, operand.level);
}

template <typename Function>
Tensor binary_impl(const Tensor& left, const Tensor& right,
                   Function&& call) {
    // Each invocation peels only the innermost mapped level; operands batched
    // at outer levels keep their wrapper and re-dispatch through call_next,
    // which routes them to their own level's batch rule.  The result therefore
    // can itself still be batched at an outer level when this level's tag is
    // attached on top of it.
    const int64_t left_level = left.defined() && left.is_batched()
        ? left.batch_level() : -1;
    const int64_t right_level = right.defined() && right.is_batched()
        ? right.batch_level() : -1;
    const int64_t level = std::max(left_level, right_level);
    if (level < 0) {
        TP_THROW(RuntimeError, "binary batch rule received no mapped operand");
    }
    auto values = broadcast_values(operand_at_level(left, level),
                                   operand_at_level(right, level));
    Tensor result = call(values.first, values.second);
    return make_batched(result, 0, level);
}

} // namespace

Tensor unary(const char* op, const Tensor& input) {
    return unary_impl(input, [&](const Tensor& value) {
        return call_next<Tensor, const Tensor&>(op, value, value);
    });
}

Tensor binary(const char* op, const Tensor& left, const Tensor& right) {
    return binary_impl(left, right, [&](const Tensor& a, const Tensor& b) {
        return call_next<Tensor, const Tensor&, const Tensor&>(op, a, a, b);
    });
}

Tensor binary_alpha(const char* op, const Tensor& left, const Tensor& right,
                    Scalar alpha) {
    return binary_impl(left, right,
                       [&](const Tensor& a, const Tensor& b) {
        return call_next<Tensor, const Tensor&, const Tensor&, Scalar>(
            op, a, a, b, alpha);
    });
}

Tensor scalar(const char* op, const Tensor& input, Scalar value) {
    return unary_impl(input, [&](const Tensor& base) {
        return call_next<Tensor, const Tensor&, Scalar>(op, base, base, value);
    });
}

// Scalar-first overload: the constant is the leading argument of the op, so
// the re-dispatch order flips while the batch dimension still comes from the
// mapped operand.
Tensor scalar_left(const char* op, const Tensor& input, Scalar value) {
    return unary_impl(input, [&](const Tensor& base) {
        return call_next<Tensor, Scalar, const Tensor&>(op, base, value, base);
    });
}

Tensor scalar_alpha(const char* op, const Tensor& input, Scalar value,
                    Scalar alpha) {
    return unary_impl(input, [&](const Tensor& base) {
        return call_next<Tensor, const Tensor&, Scalar, Scalar>(
            op, base, base, value, alpha);
    });
}

Tensor tensor_pow(const Tensor& left, const Tensor& right) {
    return binary_impl(left, right,
                       [&](const Tensor& a, const Tensor& b) {
        return call_next<Tensor, const Tensor&, const Tensor&>(
            "pow.Tensor_Tensor", a, a, b);
    });
}

Tensor sum_all(const Tensor& input, DType dtype) {
    Operand operand = unwrap_operand(input);
    if (!operand.bdim.has_value()) {
        TP_THROW(RuntimeError, "sum batch rule received an unbatched operand");
    }
    const int64_t actual_bdim = *operand.bdim;
    const int64_t level = operand.level;
    const int64_t ndim = operand.value.dim();
    std::vector<int64_t> dims;
    dims.reserve(static_cast<size_t>(std::max<int64_t>(0, ndim - 1)));
    for (int64_t dim = 0; dim < ndim; ++dim) {
        if (dim != actual_bdim) dims.push_back(dim);
    }
    Tensor result = call_next<Tensor, const Tensor&, const std::vector<int64_t>&,
                                 bool, DType>(
        "sum.dim_IntList", operand.value, operand.value, dims, false, dtype);
    return make_batched(result, 0, level);
}

Tensor sum_dim(const Tensor& input, const std::vector<int64_t>& dims,
               bool keepdim, DType dtype) {
    Operand operand = unwrap_operand(input);
    if (!operand.bdim.has_value()) {
        TP_THROW(RuntimeError, "sum batch rule received an unbatched operand");
    }
    const int64_t old_bdim = *operand.bdim;
    const int64_t level = operand.level;
    std::vector<int64_t> actual_dims;
    actual_dims.reserve(dims.size());
    for (int64_t dim : dims) {
        const int64_t public_dim = normalize_dim(dim, input.dim());
        const int64_t actual = public_dim < old_bdim ? public_dim : public_dim + 1;
        if (std::find(actual_dims.begin(), actual_dims.end(), actual) == actual_dims.end()) {
            actual_dims.push_back(actual);
        }
    }
    Tensor result = call_next<Tensor, const Tensor&, const std::vector<int64_t>&,
                                 bool, DType>(
        "sum.dim_IntList", operand.value, operand.value, actual_dims,
        keepdim, dtype);
    int64_t result_bdim = old_bdim;
    if (!keepdim) {
        result_bdim -= static_cast<int64_t>(std::count_if(
            actual_dims.begin(), actual_dims.end(),
            [old_bdim](int64_t dim) { return dim < old_bdim; }));
    }
    return make_batched(result, result_bdim, level);
}

Tensor view(const Tensor& input, const std::vector<int64_t>& shape) {
    Operand operand = unwrap_operand(input);
    if (!operand.bdim.has_value()) {
        TP_THROW(RuntimeError, "view batch rule received an unbatched operand");
    }
    const int64_t batch_size = layer_for(operand.level).batch_size;
    Tensor value = move_to_front(operand.value, *operand.bdim);
    std::vector<int64_t> physical_shape;
    physical_shape.reserve(shape.size() + 1);
    physical_shape.push_back(batch_size);
    physical_shape.insert(physical_shape.end(), shape.begin(), shape.end());
    Tensor result = value.view(physical_shape);
    return make_batched(result, 0, operand.level);
}

Tensor permute(const Tensor& input, const std::vector<int64_t>& dims) {
    Operand operand = unwrap_operand(input);
    if (!operand.bdim.has_value()) {
        TP_THROW(RuntimeError, "permute batch rule received an unbatched operand");
    }
    const int64_t ndim = input.dim();
    if (static_cast<int64_t>(dims.size()) != ndim) {
        TP_THROW(ValueError, "permute dimensions must cover every public dimension");
    }
    std::vector<int64_t> actual_dims;
    actual_dims.reserve(dims.size() + 1);
    int64_t new_bdim = -1;
    for (int64_t out_dim = 0; out_dim < ndim; ++out_dim) {
        const int64_t public_dim = normalize_dim(dims[static_cast<size_t>(out_dim)], ndim);
        actual_dims.push_back(public_dim < *operand.bdim
                                  ? public_dim : public_dim + 1);
    }
    new_bdim = *operand.bdim;
    std::vector<int64_t> physical_permutation;
    physical_permutation.reserve(actual_dims.size() + 1);
    size_t public_index = 0;
    for (int64_t physical_dim = 0;
         physical_dim < static_cast<int64_t>(actual_dims.size() + 1);
         ++physical_dim) {
        if (physical_dim == new_bdim) {
            physical_permutation.push_back(*operand.bdim);
        } else {
            physical_permutation.push_back(actual_dims[public_index++]);
        }
    }
    Tensor result = call_next<Tensor, const Tensor&, const std::vector<int64_t>&>(
        "permute", operand.value, operand.value, physical_permutation);
    return make_batched(result, new_bdim, operand.level);
}

Tensor transpose(const Tensor& input, int64_t dim0, int64_t dim1) {
    Operand operand = unwrap_operand(input);
    if (!operand.bdim.has_value()) {
        TP_THROW(RuntimeError, "transpose batch rule received an unbatched operand");
    }
    const int64_t ndim = input.dim();
    const int64_t public_dim0 = normalize_dim(dim0, ndim);
    const int64_t public_dim1 = normalize_dim(dim1, ndim);
    const int64_t actual_dim0 = public_dim0 < *operand.bdim
        ? public_dim0 : public_dim0 + 1;
    const int64_t actual_dim1 = public_dim1 < *operand.bdim
        ? public_dim1 : public_dim1 + 1;
    Tensor result = call_next<Tensor, const Tensor&, int64_t, int64_t>(
        "transpose", operand.value, operand.value, actual_dim0, actual_dim1);
    return make_batched(result, *operand.bdim, operand.level);
}

std::vector<int64_t> movedim_permutation(
    int64_t ndim, const std::vector<int64_t>& source,
    const std::vector<int64_t>& destination) {
    if (source.size() != destination.size()) {
        TP_THROW(ValueError, "movedim source and destination must have the same length");
    }
    std::vector<int64_t> normalized_source;
    std::vector<int64_t> normalized_destination;
    normalized_source.reserve(source.size());
    normalized_destination.reserve(destination.size());
    for (int64_t dim : source) {
        normalized_source.push_back(normalize_dim(dim, ndim));
    }
    for (int64_t dim : destination) {
        normalized_destination.push_back(normalize_dim(dim, ndim));
    }
    auto has_duplicate = [](const std::vector<int64_t>& dims) {
        std::vector<int64_t> sorted = dims;
        std::sort(sorted.begin(), sorted.end());
        return std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end();
    };
    if (has_duplicate(normalized_source) || has_duplicate(normalized_destination)) {
        TP_THROW(ValueError, "movedim source and destination must be unique");
    }
    std::vector<bool> source_seen(static_cast<size_t>(ndim), false);
    std::vector<bool> destination_seen(static_cast<size_t>(ndim), false);
    for (int64_t dim : normalized_source) source_seen[static_cast<size_t>(dim)] = true;
    for (int64_t dim : normalized_destination) {
        destination_seen[static_cast<size_t>(dim)] = true;
    }
    std::vector<int64_t> permutation(static_cast<size_t>(ndim), -1);
    for (size_t index = 0; index < normalized_source.size(); ++index) {
        permutation[static_cast<size_t>(normalized_destination[index])] =
            normalized_source[index];
    }
    int64_t cursor = 0;
    for (int64_t output_dim = 0; output_dim < ndim; ++output_dim) {
        if (destination_seen[static_cast<size_t>(output_dim)]) continue;
        while (cursor < ndim && source_seen[static_cast<size_t>(cursor)]) ++cursor;
        permutation[static_cast<size_t>(output_dim)] = cursor++;
    }
    return permutation;
}

Tensor movedim(const Tensor& input, const std::vector<int64_t>& source,
               const std::vector<int64_t>& destination) {
    Operand operand = unwrap_operand(input);
    if (!operand.bdim.has_value()) {
        TP_THROW(RuntimeError, "movedim batch rule received an unbatched operand");
    }
    return permute(input, movedim_permutation(input.dim(), source, destination));
}

Tensor reshape(const Tensor& input, const std::vector<int64_t>& shape) {
    Operand operand = unwrap_operand(input);
    if (!operand.bdim.has_value()) {
        TP_THROW(RuntimeError, "reshape batch rule received an unbatched operand");
    }
    const int64_t batch_size = layer_for(operand.level).batch_size;
    std::vector<int64_t> physical_shape;
    physical_shape.reserve(shape.size() + 1);
    physical_shape.push_back(batch_size);
    physical_shape.insert(physical_shape.end(), shape.begin(), shape.end());
    Tensor value = move_to_front(operand.value, *operand.bdim);
    Tensor result = call_next<Tensor, const Tensor&, const std::vector<int64_t>&>(
        "reshape", value, value, physical_shape);
    return make_batched(result, 0, operand.level);
}

Tensor expand(const Tensor& input, const std::vector<int64_t>& shape,
              bool implicit) {
    Operand operand = unwrap_operand(input);
    if (!operand.bdim.has_value()) {
        TP_THROW(RuntimeError, "expand batch rule received an unbatched operand");
    }
    std::vector<int64_t> physical_shape = shape;
    physical_shape.insert(physical_shape.begin(),
                          layer_for(operand.level).batch_size);
    Tensor result = call_next<Tensor, const Tensor&, const std::vector<int64_t>&, bool>(
        "expand", operand.value, operand.value, physical_shape, implicit);
    return make_batched(result, *operand.bdim, operand.level);
}

Tensor squeeze(const Tensor& input) {
    Operand operand = unwrap_operand(input);
    if (!operand.bdim.has_value()) {
        TP_THROW(RuntimeError, "squeeze batch rule received an unbatched operand");
    }
    std::vector<int64_t> actual_dims;
    for (int64_t public_dim = 0; public_dim < input.dim(); ++public_dim) {
        const int64_t actual_dim = public_dim < *operand.bdim
            ? public_dim : public_dim + 1;
        if (operand.value.size(static_cast<size_t>(actual_dim)) == 1) {
            actual_dims.push_back(actual_dim);
        }
    }
    if (actual_dims.empty()) return input;
    Tensor result = call_next<Tensor, const Tensor&, const std::vector<int64_t>&>(
        "squeeze.dims", operand.value, operand.value, actual_dims);
    const int64_t result_bdim = *operand.bdim - static_cast<int64_t>(
        std::count_if(actual_dims.begin(), actual_dims.end(),
                      [bdim = *operand.bdim](int64_t dim) { return dim < bdim; }));
    return make_batched(result, result_bdim, operand.level);
}

Tensor squeeze_dim(const Tensor& input, int64_t dim) {
    Operand operand = unwrap_operand(input);
    if (!operand.bdim.has_value()) {
        TP_THROW(RuntimeError, "squeeze batch rule received an unbatched operand");
    }
    const int64_t public_dim = normalize_dim(dim, input.dim());
    const int64_t actual_dim = public_dim < *operand.bdim
        ? public_dim : public_dim + 1;
    if (operand.value.size(static_cast<size_t>(actual_dim)) != 1) return input;
    Tensor result = call_next<Tensor, const Tensor&, int64_t>(
        "squeeze.dim", operand.value, operand.value, actual_dim);
    const int64_t result_bdim = actual_dim < *operand.bdim
        ? *operand.bdim - 1 : *operand.bdim;
    return make_batched(result, result_bdim, operand.level);
}

Tensor squeeze_dims(const Tensor& input, const std::vector<int64_t>& dims) {
    Operand operand = unwrap_operand(input);
    if (!operand.bdim.has_value()) {
        TP_THROW(RuntimeError, "squeeze batch rule received an unbatched operand");
    }
    std::vector<int64_t> actual_dims;
    for (int64_t dim : dims) {
        const int64_t public_dim = normalize_dim(dim, input.dim());
        const int64_t actual_dim = public_dim < *operand.bdim
            ? public_dim : public_dim + 1;
        if (std::find(actual_dims.begin(), actual_dims.end(), actual_dim) ==
            actual_dims.end()) {
            actual_dims.push_back(actual_dim);
        }
    }
    std::sort(actual_dims.begin(), actual_dims.end());
    std::vector<int64_t> removed;
    for (int64_t actual_dim : actual_dims) {
        if (operand.value.size(static_cast<size_t>(actual_dim)) == 1) {
            removed.push_back(actual_dim);
        }
    }
    if (removed.empty()) return input;
    Tensor result = call_next<Tensor, const Tensor&, const std::vector<int64_t>&>(
        "squeeze.dims", operand.value, operand.value, removed);
    const int64_t result_bdim = *operand.bdim - static_cast<int64_t>(
        std::count_if(removed.begin(), removed.end(),
                      [bdim = *operand.bdim](int64_t dim) { return dim < bdim; }));
    return make_batched(result, result_bdim, operand.level);
}

Tensor unsqueeze(const Tensor& input, int64_t dim) {
    Operand operand = unwrap_operand(input);
    if (!operand.bdim.has_value()) {
        TP_THROW(RuntimeError, "unsqueeze batch rule received an unbatched operand");
    }
    const int64_t public_dim = normalize_dim(dim, input.dim() + 1);
    const int64_t actual_dim = public_dim <= *operand.bdim
        ? public_dim : public_dim + 1;
    Tensor result = call_next<Tensor, const Tensor&, int64_t>(
        "unsqueeze", operand.value, operand.value, actual_dim);
    const int64_t result_bdim = actual_dim <= *operand.bdim
        ? *operand.bdim + 1 : *operand.bdim;
    return make_batched(result, result_bdim, operand.level);
}

Tensor contiguous(const Tensor& input, int64_t memory_format) {
    Operand operand = unwrap_operand(input);
    if (!operand.bdim.has_value()) {
        TP_THROW(RuntimeError, "contiguous batch rule received an unbatched operand");
    }
    Tensor result = call_next<Tensor, const Tensor&, int64_t>(
        "contiguous", operand.value, operand.value, memory_format);
    return make_batched(result, *operand.bdim, operand.level);
}

Tensor select(const Tensor& input, int64_t dim, int64_t index) {
    Operand operand = unwrap_operand(input);
    if (!operand.bdim.has_value()) {
        TP_THROW(RuntimeError, "select batch rule received an unbatched operand");
    }
    const int64_t public_dim = normalize_dim(dim, input.dim());
    const int64_t actual_dim = public_dim < *operand.bdim ? public_dim : public_dim + 1;
    Tensor result = call_next<Tensor, const Tensor&, int64_t, int64_t>(
        "select.int", operand.value, operand.value, actual_dim, index);
    const int64_t result_bdim = actual_dim < *operand.bdim
        ? *operand.bdim - 1 : *operand.bdim;
    return make_batched(result, result_bdim, operand.level);
}

Tensor slice(const Tensor& input, int64_t dim,
             std::optional<int64_t> start, std::optional<int64_t> end,
             int64_t step) {
    Operand operand = unwrap_operand(input);
    if (!operand.bdim.has_value()) {
        TP_THROW(RuntimeError, "slice batch rule received an unbatched operand");
    }
    const int64_t public_dim = normalize_dim(dim, input.dim());
    const int64_t actual_dim = public_dim < *operand.bdim ? public_dim : public_dim + 1;
    Tensor result = call_next<Tensor, const Tensor&, int64_t,
                                 std::optional<int64_t>, std::optional<int64_t>, int64_t>(
        "slice", operand.value, operand.value, actual_dim, start, end, step);
    return make_batched(result, actual_dim < *operand.bdim
                                    ? *operand.bdim : *operand.bdim,
                        operand.level);
}

Tensor index_select(const Tensor& input, int64_t dim, const Tensor& index) {
    Operand operand = unwrap_operand(input);
    if (!operand.bdim.has_value()) {
        TP_THROW(RuntimeError, "index_select batch rule received an unbatched operand");
    }
    if (index.is_batched()) {
        TP_THROW(NotImplementedError,
                 "index_select with a mapped index requires an index batch rule");
    }
    const int64_t public_dim = normalize_dim(dim, input.dim());
    const int64_t actual_dim = public_dim < *operand.bdim ? public_dim : public_dim + 1;
    Tensor result = call_next<Tensor, const Tensor&, int64_t, const Tensor&>(
        "index_select", operand.value, operand.value, actual_dim, index);
    return make_batched(result, actual_dim < *operand.bdim
                                    ? *operand.bdim : *operand.bdim,
                        operand.level);
}

Tensor narrow(const Tensor& input, int64_t dim, int64_t start, int64_t length) {
    Operand operand = unwrap_operand(input);
    if (!operand.bdim.has_value()) {
        TP_THROW(RuntimeError, "narrow batch rule received an unbatched operand");
    }
    const int64_t public_dim = normalize_dim(dim, input.dim());
    const int64_t actual_dim = public_dim < *operand.bdim
        ? public_dim : public_dim + 1;
    Tensor result = call_next<Tensor, const Tensor&, int64_t, int64_t, int64_t>(
        "narrow", operand.value, operand.value, actual_dim, start, length);
    return make_batched(result, *operand.bdim, operand.level);
}

std::pair<std::vector<Tensor>, int64_t> align_tensor_list(
    const std::vector<Tensor>& inputs) {
    const auto active = current_layer();
    if (!active.has_value()) {
        TP_THROW(RuntimeError, "tensor-list batch rule requires an active transform");
    }
    std::optional<Operand> mapped;
    std::vector<Operand> operands;
    operands.reserve(inputs.size());
    for (const Tensor& input : inputs) {
        Operand operand = unwrap_operand(input);
        if (operand.bdim.has_value() && !mapped.has_value()) mapped = operand;
        operands.push_back(std::move(operand));
    }
    if (!mapped.has_value()) {
        TP_THROW(RuntimeError, "tensor-list batch rule received no mapped operand");
    }
    const int64_t batch_size = active->batch_size;
    std::vector<Tensor> aligned;
    aligned.reserve(operands.size());
    for (const Operand& operand : operands) {
        if (operand.bdim.has_value()) {
            aligned.push_back(move_to_front(operand.value, *operand.bdim));
        } else {
            aligned.push_back(expand_unbatched(
                operand.value, batch_size, logical_shape(operand)));
        }
    }
    return {std::move(aligned), mapped->level};
}

Tensor cat(const std::vector<Tensor>& inputs, int64_t dim) {
    if (inputs.empty()) {
        TP_THROW(ValueError, "cat batch rule received an empty tensor list");
    }
    auto aligned = align_tensor_list(inputs);
    const int64_t logical_ndim = inputs.front().is_batched()
        ? inputs.front().dim() : aligned.first.front().dim() - 1;
    const int64_t public_dim = normalize_dim(dim, logical_ndim);
    Tensor result = call_next<Tensor, const std::vector<Tensor>&, int64_t>(
        "cat", aligned.first.front(), aligned.first, public_dim + 1);
    return make_batched(result, 0, aligned.second);
}

Tensor stack(const std::vector<Tensor>& inputs, int64_t dim) {
    if (inputs.empty()) {
        TP_THROW(ValueError, "stack batch rule received an empty tensor list");
    }
    auto aligned = align_tensor_list(inputs);
    const int64_t logical_ndim = inputs.front().is_batched()
        ? inputs.front().dim() : aligned.first.front().dim() - 1;
    const int64_t public_dim = normalize_dim(dim, logical_ndim + 1);
    Tensor result = call_next<Tensor, const std::vector<Tensor>&, int64_t>(
        "stack", aligned.first.front(), aligned.first, public_dim + 1);
    return make_batched(result, 0, aligned.second);
}

// Shared shape machinery for matmul-like rules.  Each operand unwraps only
// at the innermost mapped level; operands batched at outer levels keep their
// wrapper so their public shape hides their own batch dim and the matmul call
// re-dispatches through their level.  Mapped batch dims move to the front and
// the underlying matmul broadcasts mismatched batch extents.
std::pair<Tensor, Tensor> matmul_values(const Tensor& left, const Tensor& right,
                                        const Operand& a, const Operand& b) {
    Tensor a_value = a.bdim.has_value() ? move_to_front(a.value, *a.bdim)
                                        : left;
    Tensor b_value = b.bdim.has_value() ? move_to_front(b.value, *b.bdim)
                                        : right;
    return {std::move(a_value), std::move(b_value)};
}

Tensor mm(const Tensor& left, const Tensor& right) {
    const int64_t level = std::max(
        left.defined() && left.is_batched() ? left.batch_level() : -1,
        right.defined() && right.is_batched() ? right.batch_level() : -1);
    if (level < 0) {
        TP_THROW(RuntimeError, "mm batch rule received no mapped operand");
    }
    Operand a = operand_at_level(left, level);
    Operand b = operand_at_level(right, level);
    const int64_t left_rank =
        static_cast<int64_t>(a.value.dim()) - (a.bdim.has_value() ? 1 : 0);
    const int64_t right_rank =
        static_cast<int64_t>(b.value.dim()) - (b.bdim.has_value() ? 1 : 0);
    if (left_rank != 2 || right_rank != 2) {
        TP_THROW(RuntimeError,
                 "Shape mismatch: Got incorrect dims for mm(a, b). a has dim ",
                 left_rank, " and b has dim ", right_rank,
                 " but expected them to have dim 2 and dim 2");
    }
    auto values = matmul_values(left, right, a, b);
    Tensor result = call_next<Tensor, const Tensor&, const Tensor&>(
        "matmul", values.first, values.first, values.second);
    return make_batched(result, 0, level);
}

Tensor matmul(const Tensor& left, const Tensor& right) {
    const int64_t level = std::max(
        left.defined() && left.is_batched() ? left.batch_level() : -1,
        right.defined() && right.is_batched() ? right.batch_level() : -1);
    if (level < 0) {
        TP_THROW(RuntimeError,
                 "matmul batch rule received no mapped operand");
    }
    Operand a = operand_at_level(left, level);
    Operand b = operand_at_level(right, level);
    const int64_t left_ndim =
        static_cast<int64_t>(a.value.dim()) - (a.bdim.has_value() ? 1 : 0);
    const int64_t right_ndim =
        static_cast<int64_t>(b.value.dim()) - (b.bdim.has_value() ? 1 : 0);
    if (left_ndim == 1 && right_ndim == 1) {
        // Vector-vector product: mapped operands pair up through their batch
        // dims; an unmapped operand contracts against the transposed other.
        auto values = matmul_values(left, right, a, b);
        Tensor result;
        if (a.bdim.has_value() && b.bdim.has_value()) {
            Tensor a_col = call_next<Tensor, const Tensor&, int64_t>(
                "unsqueeze", values.first, values.first, -2);
            Tensor b_row = call_next<Tensor, const Tensor&, int64_t>(
                "unsqueeze", values.second, values.second, -1);
            Tensor product = call_next<Tensor, const Tensor&, const Tensor&>(
                "matmul", a_col, a_col, b_row);
            Tensor first = call_next<Tensor, const Tensor&, int64_t>(
                "squeeze.dim", product, product, -1);
            result = call_next<Tensor, const Tensor&, int64_t>(
                "squeeze.dim", first, first, -1);
        } else {
            Tensor b_transposed = call_next<Tensor, const Tensor&, int64_t,
                                            int64_t>(
                "transpose", values.second, values.second, 0, 1);
            result = call_next<Tensor, const Tensor&, const Tensor&>(
                "matmul", values.first, values.first, b_transposed);
        }
        return make_batched(result, 0, level);
    }
    auto values = matmul_values(left, right, a, b);
    Tensor result = call_next<Tensor, const Tensor&, const Tensor&>(
        "matmul", values.first, values.first, values.second);
    return make_batched(result, 0, level);
}

Tensor bmm(const Tensor& left, const Tensor& right) {
    const int64_t level = std::max(
        left.defined() && left.is_batched() ? left.batch_level() : -1,
        right.defined() && right.is_batched() ? right.batch_level() : -1);
    if (level < 0) {
        TP_THROW(RuntimeError, "bmm batch rule received no mapped operand");
    }
    Operand a = operand_at_level(left, level);
    Operand b = operand_at_level(right, level);
    const int64_t left_rank =
        static_cast<int64_t>(a.value.dim()) - (a.bdim.has_value() ? 1 : 0);
    const int64_t right_rank =
        static_cast<int64_t>(b.value.dim()) - (b.bdim.has_value() ? 1 : 0);
    if (left_rank != 3 || right_rank != 3) {
        TP_THROW(RuntimeError,
                 "Shape mismatch: Got incorrect dims for bmm(a, b). "
                 "a has dim ", left_rank, " and b has dim ", right_rank,
                 " but expected them to have dim 3 and dim 3");
    }
    auto values = matmul_values(left, right, a, b);
    Tensor result = call_next<Tensor, const Tensor&, const Tensor&>(
        "matmul", values.first, values.first, values.second);
    return make_batched(result, 0, level);
}

Tensor linear(const Tensor& input, const Tensor& weight,
              std::optional<Tensor> bias) {
    Operand input_operand = unwrap_operand(input);
    Operand weight_operand = unwrap_operand(weight);
    std::optional<Operand> bias_operand;
    if (bias.has_value() && bias->defined()) {
        bias_operand = unwrap_operand(*bias);
    }

    const Operand* mapped = input_operand.bdim.has_value()
        ? &input_operand
        : (weight_operand.bdim.has_value()
            ? &weight_operand
            : (bias_operand.has_value() && bias_operand->bdim.has_value()
                ? &*bias_operand : nullptr));
    if (mapped == nullptr) {
        TP_THROW(RuntimeError, "linear batch rule received no mapped operand");
    }
    const Layer& active = layer_for(mapped->level);
    const auto weight_shape = logical_shape(weight_operand);
    if (weight_shape.size() != 2) {
        TP_THROW(RuntimeError, "linear batch rule expects a 2D weight");
    }

    const auto align = [&](const Operand& operand) {
        if (operand.bdim.has_value()) {
            return move_to_front(operand.value, *operand.bdim);
        }
        return expand_unbatched(operand.value, active.batch_size,
                                logical_shape(operand));
    };
    Tensor input_value = align(input_operand);
    Tensor weight_value = align(weight_operand);
    Tensor transposed_weight = call_next<Tensor, const Tensor&, int64_t, int64_t>(
        "transpose", weight_value, weight_value,
        weight_value.dim() - 2, weight_value.dim() - 1);
    Tensor result = call_next<Tensor, const Tensor&, const Tensor&>(
        "matmul", input_value, input_value, transposed_weight);
    if (bias_operand.has_value()) {
        Tensor bias_value = align(*bias_operand);
        while (bias_value.dim() < result.dim()) {
            bias_value = call_next<Tensor, const Tensor&, int64_t>(
                "unsqueeze", bias_value, bias_value, 1);
        }
        result = call_next<Tensor, const Tensor&, const Tensor&, Scalar>(
            "add.Tensor", result, result, bias_value, Scalar(1));
    }
    return make_batched(result, 0, mapped->level);
}

Tensor rand(const std::vector<int64_t>& shape, std::optional<DType> dtype,
            std::optional<Device> device) {
    return random_factory("rand", shape, dtype, device);
}

Tensor randn(const std::vector<int64_t>& shape, std::optional<DType> dtype,
             std::optional<Device> device) {
    return random_factory("randn", shape, dtype, device);
}

Tensor randint(int64_t low, int64_t high,
               const std::vector<int64_t>& shape, DType dtype,
               std::optional<Device> device) {
    return random_int_factory("randint", low, high, shape, dtype, device);
}

Tensor randperm(int64_t n, DType dtype, std::optional<Device> device) {
    return randperm_factory(n, dtype, device);
}

Tensor rand_like(const Tensor& input, DType dtype,
                 std::optional<Device> device) {
    return random_like_factory("rand_like", input, dtype, device);
}

Tensor randint_like(const Tensor& input, int64_t low, int64_t high,
                    DType dtype, std::optional<Device> device) {
    return random_like_impl(
        input, dtype, device,
        [&](const Tensor& value, DType output_dtype,
            std::optional<Device> output_device) {
            return call_next<Tensor, const Tensor&, int64_t, int64_t, DType,
                             std::optional<Device>>(
                "randint_like", value, value, low, high, output_dtype,
                output_device);
        });
}

Tensor randn_like(const Tensor& input, DType dtype,
                  std::optional<Device> device) {
    return random_like_factory("randn_like", input, dtype, device);
}

} // namespace batch
} // namespace transform
} // namespace tensorplay

namespace {

using namespace tensorplay;
using namespace tensorplay::transform::batch;
using tensorplay::transform::make_batched;

#define TP_BATCH_UNARY(NAME, OP) \
    Tensor NAME(const Tensor& input) { return unary(OP, input); }
#define TP_BATCH_BINARY(NAME, OP) \
    Tensor NAME(const Tensor& left, const Tensor& right) { return binary(OP, left, right); }

TP_BATCH_UNARY(batch_neg, "neg")
TP_BATCH_UNARY(batch_negative, "negative")
TP_BATCH_UNARY(batch_abs, "abs")
TP_BATCH_UNARY(batch_exp, "exp")
TP_BATCH_UNARY(batch_log, "log")
TP_BATCH_UNARY(batch_sin, "sin")
TP_BATCH_UNARY(batch_cos, "cos")
TP_BATCH_UNARY(batch_sinh, "sinh")
TP_BATCH_UNARY(batch_cosh, "cosh")
TP_BATCH_UNARY(batch_tanh, "tanh")
TP_BATCH_UNARY(batch_sqrt, "sqrt")
TP_BATCH_UNARY(batch_rsqrt, "rsqrt")
TP_BATCH_UNARY(batch_sigmoid, "sigmoid")
TP_BATCH_UNARY(batch_relu, "relu")
TP_BATCH_UNARY(batch_floor, "floor")
TP_BATCH_UNARY(batch_ceil, "ceil")
TP_BATCH_UNARY(batch_round, "round")
TP_BATCH_UNARY(batch_trunc, "trunc")
TP_BATCH_UNARY(batch_erf, "erf")
TP_BATCH_UNARY(batch_erfc, "erfc")
TP_BATCH_UNARY(batch_log1p, "log1p")
TP_BATCH_UNARY(batch_expm1, "expm1")
TP_BATCH_UNARY(batch_bitwise_not, "bitwise_not")

TP_BATCH_BINARY(batch_mul, "mul.Tensor")
TP_BATCH_BINARY(batch_div, "div.Tensor")
TP_BATCH_BINARY(batch_maximum, "maximum")
TP_BATCH_BINARY(batch_minimum, "minimum")
TP_BATCH_BINARY(batch_logical_and, "logical_and")
TP_BATCH_BINARY(batch_logical_or, "logical_or")
TP_BATCH_BINARY(batch_logical_xor, "logical_xor")
TP_BATCH_BINARY(batch_bitwise_and, "bitwise_and.Tensor")
TP_BATCH_BINARY(batch_bitwise_or, "bitwise_or.Tensor")
TP_BATCH_BINARY(batch_bitwise_xor, "bitwise_xor.Tensor")
TP_BATCH_BINARY(batch_bitwise_lshift, "bitwise_left_shift.Tensor")
TP_BATCH_BINARY(batch_bitwise_rshift, "bitwise_right_shift.Tensor")

Tensor batch_add(const Tensor& left, const Tensor& right, Scalar alpha) {
    return binary_alpha("add.Tensor", left, right, alpha);
}
Tensor batch_sub(const Tensor& left, const Tensor& right, Scalar alpha) {
    return binary_alpha("sub.Tensor", left, right, alpha);
}
Tensor batch_add_scalar(const Tensor& input, Scalar value, Scalar alpha) {
    return scalar_alpha("add.Scalar", input, value, alpha);
}
Tensor batch_sub_scalar(const Tensor& input, Scalar value, Scalar alpha) {
    return scalar_alpha("sub.Scalar", input, value, alpha);
}
Tensor batch_mul_scalar(const Tensor& input, Scalar value) {
    return scalar("mul.Scalar", input, value);
}
Tensor batch_div_scalar(const Tensor& input, Scalar value) {
    return scalar("div.Scalar", input, value);
}
Tensor batch_bitwise_and_scalar(const Tensor& input, Scalar value) {
    return scalar("bitwise_and.Scalar", input, value);
}
Tensor batch_bitwise_or_scalar(const Tensor& input, Scalar value) {
    return scalar("bitwise_or.Scalar", input, value);
}
Tensor batch_bitwise_xor_scalar(const Tensor& input, Scalar value) {
    return scalar("bitwise_xor.Scalar", input, value);
}
Tensor batch_bitwise_lshift_scalar(const Tensor& input, Scalar value) {
    return scalar("bitwise_left_shift.Tensor_Scalar", input, value);
}
Tensor batch_bitwise_rshift_scalar(const Tensor& input, Scalar value) {
    return scalar("bitwise_right_shift.Tensor_Scalar", input, value);
}
Tensor batch_bitwise_and_stensor(Scalar value, const Tensor& input) {
    return scalar_left("bitwise_and.Scalar_Tensor", input, value);
}
Tensor batch_bitwise_or_stensor(Scalar value, const Tensor& input) {
    return scalar_left("bitwise_or.Scalar_Tensor", input, value);
}
Tensor batch_bitwise_xor_stensor(Scalar value, const Tensor& input) {
    return scalar_left("bitwise_xor.Scalar_Tensor", input, value);
}
Tensor batch_bitwise_lshift_stensor(Scalar value, const Tensor& input) {
    return scalar_left("bitwise_left_shift.Scalar_Tensor", input, value);
}
Tensor batch_bitwise_rshift_stensor(Scalar value, const Tensor& input) {
    return scalar_left("bitwise_right_shift.Scalar_Tensor", input, value);
}
Tensor batch_pow_scalar(const Tensor& input, Scalar exponent) {
    return scalar("pow.Tensor_Scalar", input, exponent);
}
Tensor batch_pow_tensor(const Tensor& left, const Tensor& right) {
    return tensor_pow(left, right);
}
Tensor batch_sum(const Tensor& input, DType dtype) { return sum_all(input, dtype); }
Tensor batch_sum_dim(const Tensor& input, const std::vector<int64_t>& dims,
                     bool keepdim, DType dtype) {
    return sum_dim(input, dims, keepdim, dtype);
}
Tensor batch_permute(const Tensor& input, const std::vector<int64_t>& dims) {
    return permute(input, dims);
}
Tensor batch_view(const Tensor& input, const std::vector<int64_t>& shape) {
    return view(input, shape);
}
Tensor batch_transpose(const Tensor& input, int64_t dim0, int64_t dim1) {
    return transpose(input, dim0, dim1);
}
Tensor batch_movedim(const Tensor& input, const std::vector<int64_t>& source,
                     const std::vector<int64_t>& destination) {
    return movedim(input, source, destination);
}
Tensor batch_reshape(const Tensor& input, const std::vector<int64_t>& shape) {
    return reshape(input, shape);
}
Tensor batch_expand(const Tensor& input, const std::vector<int64_t>& shape,
                    bool implicit) {
    return expand(input, shape, implicit);
}
Tensor batch_squeeze(const Tensor& input) { return squeeze(input); }
Tensor batch_squeeze_dim(const Tensor& input, int64_t dim) {
    return squeeze_dim(input, dim);
}
Tensor batch_squeeze_dims(const Tensor& input, const std::vector<int64_t>& dims) {
    return squeeze_dims(input, dims);
}
Tensor batch_unsqueeze(const Tensor& input, int64_t dim) {
    return unsqueeze(input, dim);
}
Tensor batch_contiguous(const Tensor& input, int64_t memory_format) {
    return contiguous(input, memory_format);
}
Tensor batch_select(const Tensor& input, int64_t dim, int64_t index) {
    return select(input, dim, index);
}
Tensor batch_slice(const Tensor& input, int64_t dim,
                  std::optional<int64_t> start, std::optional<int64_t> end,
                  int64_t step) {
    return slice(input, dim, start, end, step);
}
Tensor batch_narrow(const Tensor& input, int64_t dim, int64_t start,
                    int64_t length) {
    return narrow(input, dim, start, length);
}
Tensor batch_index_select(const Tensor& input, int64_t dim, const Tensor& index) {
    return index_select(input, dim, index);
}
Tensor batch_cat(const std::vector<Tensor>& inputs, int64_t dim) {
    return cat(inputs, dim);
}
Tensor batch_stack(const std::vector<Tensor>& inputs, int64_t dim) {
    return stack(inputs, dim);
}
Tensor batch_mm(const Tensor& left, const Tensor& right) { return mm(left, right); }
Tensor batch_matmul(const Tensor& left, const Tensor& right) {
    return matmul(left, right);
}
Tensor batch_bmm(const Tensor& left, const Tensor& right) {
    return bmm(left, right);
}
Tensor batch_linear(const Tensor& input, const Tensor& weight,
                    std::optional<Tensor> bias) {
    return linear(input, weight, std::move(bias));
}

Tensor batch_rand(const std::vector<int64_t>& shape,
                  std::optional<DType> dtype,
                  std::optional<Device> device) {
    return rand(shape, dtype, device);
}
Tensor batch_randn(const std::vector<int64_t>& shape,
                   std::optional<DType> dtype,
                   std::optional<Device> device) {
    return randn(shape, dtype, device);
}
Tensor batch_randint(int64_t low, int64_t high,
                     const std::vector<int64_t>& shape, DType dtype,
                     std::optional<Device> device) {
    return randint(low, high, shape, dtype, device);
}
Tensor batch_randperm(int64_t n, DType dtype,
                      std::optional<Device> device) {
    return randperm(n, dtype, device);
}
Tensor batch_rand_like(const Tensor& input, DType dtype,
                       std::optional<Device> device) {
    return rand_like(input, dtype, device);
}
Tensor batch_randint_like(const Tensor& input, int64_t low, int64_t high,
                          DType dtype, std::optional<Device> device) {
    return randint_like(input, low, high, dtype, device);
}
Tensor batch_randn_like(const Tensor& input, DType dtype,
                        std::optional<Device> device) {
    return randn_like(input, dtype, device);
}

// ---------------------------------------------------------------------------
// Comparison predicates.
// ---------------------------------------------------------------------------

#define TP_BATCH_PREDICATE_TENSOR(NAME, OP) \
    Tensor NAME(const Tensor& left, const Tensor& right) { return binary(OP, left, right); }
#define TP_BATCH_PREDICATE_SCALAR(NAME, OP) \
    Tensor NAME(const Tensor& input, Scalar value) { return scalar(OP, input, value); }

TP_BATCH_PREDICATE_TENSOR(batch_eq_tensor, "eq.Tensor")
TP_BATCH_PREDICATE_TENSOR(batch_ne_tensor, "ne.Tensor")
TP_BATCH_PREDICATE_TENSOR(batch_lt_tensor, "lt.Tensor")
TP_BATCH_PREDICATE_TENSOR(batch_le_tensor, "le.Tensor")
TP_BATCH_PREDICATE_TENSOR(batch_gt_tensor, "gt.Tensor")
TP_BATCH_PREDICATE_TENSOR(batch_ge_tensor, "ge.Tensor")
TP_BATCH_PREDICATE_SCALAR(batch_eq_scalar, "eq.Scalar")
TP_BATCH_PREDICATE_SCALAR(batch_ne_scalar, "ne.Scalar")
TP_BATCH_PREDICATE_SCALAR(batch_lt_scalar, "lt.Scalar")
TP_BATCH_PREDICATE_SCALAR(batch_le_scalar, "le.Scalar")
TP_BATCH_PREDICATE_SCALAR(batch_gt_scalar, "gt.Scalar")
TP_BATCH_PREDICATE_SCALAR(batch_ge_scalar, "ge.Scalar")

// ---------------------------------------------------------------------------
// where: every overload maps the tensor operands onto the batch dimension.
// ---------------------------------------------------------------------------

Tensor batch_where_self(const Tensor& condition, const Tensor& self,
                        const Tensor& other) {
    auto aligned = align_tensor_list({condition, self, other});
    Tensor result = call_next<Tensor, const Tensor&, const Tensor&, const Tensor&>(
        "where.self", aligned.first[0], aligned.first[0], aligned.first[1],
        aligned.first[2]);
    return make_batched(result, 0, aligned.second);
}

Tensor batch_where_scalar_self(const Tensor& condition, Scalar self,
                               const Tensor& other) {
    auto aligned = align_tensor_list({condition, other});
    Tensor result = call_next<Tensor, const Tensor&, Scalar, const Tensor&>(
        "where.ScalarSelf", aligned.first[0], aligned.first[0], self,
        aligned.first[1]);
    return make_batched(result, 0, aligned.second);
}

Tensor batch_where_scalar_other(const Tensor& condition, const Tensor& self,
                                Scalar other) {
    auto aligned = align_tensor_list({condition, self});
    Tensor result = call_next<Tensor, const Tensor&, const Tensor&, Scalar>(
        "where.ScalarOther", aligned.first[0], aligned.first[0],
        aligned.first[1], other);
    return make_batched(result, 0, aligned.second);
}

Tensor batch_where_scalar(const Tensor& condition, Scalar self, Scalar other) {
    return unary_impl(condition, [&](const Tensor& value) {
        return call_next<Tensor, const Tensor&, Scalar, Scalar>(
            "where.Scalar", value, value, self, other);
    });
}

// ---------------------------------------------------------------------------
// Elementwise and reduction rules.  Dim-taking ops shift the public dim past
// the batch dimension and keep (or re-locate) the batch dimension on the
// output, mirroring the shape rules of sum_dim above.
// ---------------------------------------------------------------------------

Tensor batch_clamp(const Tensor& input, std::optional<Scalar> min,
                   std::optional<Scalar> max) {
    return unary_impl(input, [&](const Tensor& value) {
        return call_next<Tensor, const Tensor&, std::optional<Scalar>,
                         std::optional<Scalar>>("clamp", value, value, min, max);
    });
}

Tensor batch_cumsum(const Tensor& input, int64_t dim,
                    std::optional<DType> dtype) {
    Operand operand = unwrap_operand(input);
    const int64_t public_dim = normalize_dim(dim, input.dim());
    const int64_t actual =
        public_dim < *operand.bdim ? public_dim : public_dim + 1;
    Tensor result = call_next<Tensor, const Tensor&, int64_t, std::optional<DType>>(
        "cumsum", operand.value, operand.value, actual, dtype);
    return make_batched(result, *operand.bdim, operand.level);
}

Tensor batch_logsumexp(const Tensor& input, int64_t dim, bool keepdim) {
    Operand operand = unwrap_operand(input);
    const int64_t public_dim = normalize_dim(dim, input.dim());
    const int64_t actual =
        public_dim < *operand.bdim ? public_dim : public_dim + 1;
    Tensor result = call_next<Tensor, const Tensor&, int64_t, bool>(
        "logsumexp", operand.value, operand.value, actual, keepdim);
    const int64_t result_bdim =
        keepdim ? *operand.bdim
                : (actual < *operand.bdim ? *operand.bdim - 1 : *operand.bdim);
    return make_batched(result, result_bdim, operand.level);
}

Tensor batch_all_dim(const Tensor& input, int64_t dim, bool keepdim) {
    Operand operand = unwrap_operand(input);
    const int64_t public_dim = normalize_dim(dim, input.dim());
    const int64_t actual =
        public_dim < *operand.bdim ? public_dim : public_dim + 1;
    Tensor result = call_next<Tensor, const Tensor&, int64_t, bool>(
        "all.dim", operand.value, operand.value, actual, keepdim);
    const int64_t result_bdim =
        keepdim ? *operand.bdim
                : (actual < *operand.bdim ? *operand.bdim - 1 : *operand.bdim);
    return make_batched(result, result_bdim, operand.level);
}

std::tuple<Tensor, Tensor> batch_max_dim(const Tensor& input, int64_t dim,
                                         bool keepdim) {
    Operand operand = unwrap_operand(input);
    const int64_t public_dim = normalize_dim(dim, input.dim());
    const int64_t actual =
        public_dim < *operand.bdim ? public_dim : public_dim + 1;
    auto result =
        call_next<std::tuple<Tensor, Tensor>, const Tensor&, int64_t, bool>(
            "max.dim", operand.value, operand.value, actual, keepdim);
    const int64_t result_bdim =
        keepdim ? *operand.bdim
                : (actual < *operand.bdim ? *operand.bdim - 1 : *operand.bdim);
    return std::make_tuple(
        make_batched(std::get<0>(result), result_bdim, operand.level),
        make_batched(std::get<1>(result), result_bdim, operand.level));
}

Tensor batch_argsort(const Tensor& input, int64_t dim, bool descending) {
    Operand operand = unwrap_operand(input);
    const int64_t public_dim = normalize_dim(dim, input.dim());
    const int64_t actual =
        public_dim < *operand.bdim ? public_dim : public_dim + 1;
    Tensor result = call_next<Tensor, const Tensor&, int64_t, bool>(
        "argsort", operand.value, operand.value, actual, descending);
    return make_batched(result, *operand.bdim, operand.level);
}

Tensor batch_argsort_stable(const Tensor& input, bool stable, int64_t dim,
                            bool descending) {
    Operand operand = unwrap_operand(input);
    const int64_t public_dim = normalize_dim(dim, input.dim());
    const int64_t actual =
        public_dim < *operand.bdim ? public_dim : public_dim + 1;
    Tensor result = call_next<Tensor, const Tensor&, bool, int64_t, bool>(
        "argsort.stable", operand.value, operand.value, stable, actual,
        descending);
    return make_batched(result, *operand.bdim, operand.level);
}

Tensor batch_gather(const Tensor& input, int64_t dim, const Tensor& index) {
    if (index.is_batched()) {
        TP_THROW(NotImplementedError,
                 "gather with a mapped index requires an index batch rule");
    }
    Operand operand = unwrap_operand(input);
    const int64_t public_dim = normalize_dim(dim, input.dim());
    const int64_t actual =
        public_dim < *operand.bdim ? public_dim : public_dim + 1;
    Tensor result = call_next<Tensor, const Tensor&, int64_t, const Tensor&>(
        "gather", operand.value, operand.value, actual, index);
    return make_batched(result, *operand.bdim, operand.level);
}

Tensor batch_repeat_interleave_self_int(const Tensor& input, int64_t repeats,
                                        std::optional<int64_t> dim,
                                        std::optional<int64_t> output_size) {
    if (!dim.has_value()) {
        TP_THROW(NotImplementedError,
                 "repeat_interleave without dim requires a flatten batch rule");
    }
    Operand operand = unwrap_operand(input);
    const int64_t public_dim = normalize_dim(*dim, input.dim());
    const int64_t actual =
        public_dim < *operand.bdim ? public_dim : public_dim + 1;
    Tensor result =
        call_next<Tensor, const Tensor&, int64_t, std::optional<int64_t>,
                  std::optional<int64_t>>(
            "repeat_interleave.self_int", operand.value, operand.value,
            repeats, actual, output_size);
    return make_batched(result, *operand.bdim, operand.level);
}

Tensor batch_to_dtype(const Tensor& input, DType dtype, bool non_blocking,
                      bool copy, std::optional<int64_t> memory_format) {
    return unary_impl(input, [&](const Tensor& value) {
        return call_next<Tensor, const Tensor&, DType, bool, bool,
                         std::optional<int64_t>>(
            "to.dtype", value, value, dtype, non_blocking, copy, memory_format);
    });
}

Tensor batch_tril(const Tensor& input, int64_t diagonal) {
    return unary_impl(input, [&](const Tensor& value) {
        return call_next<Tensor, const Tensor&, int64_t>(
            "tril", value, value, diagonal);
    });
}

// ---------------------------------------------------------------------------
// Factory-with-like-input rules: new_zeros/new_ones take the logical shape
// from the caller and gain a fresh batch dimension.
// ---------------------------------------------------------------------------

Tensor batch_new_like(const char* factory, const Tensor& self,
                      const std::vector<int64_t>& size,
                      std::optional<DType> dtype,
                      std::optional<int64_t> layout,
                      std::optional<Device> device,
                      std::optional<bool> pin_memory) {
    (void)layout;
    Operand operand = unwrap_operand(self);
    const DType out_dtype =
        dtype.has_value() && *dtype != DType::Undefined ? *dtype : self.dtype();
    const Device out_device = device.has_value() ? *device : self.device();
    Tensor result =
        call_device<Tensor, const std::vector<int64_t>&, std::optional<DType>,
                    std::optional<Device>, bool>(
            factory, out_device, size, out_dtype, out_device,
            pin_memory.value_or(false));
    if (!operand.bdim.has_value()) {
        return result;
    }
    return make_batched(result, 0, operand.level);
}

Tensor batch_new_zeros(const Tensor& self, const std::vector<int64_t>& size,
                       std::optional<DType> dtype, std::optional<int64_t> layout,
                       std::optional<Device> device,
                       std::optional<bool> pin_memory) {
    return batch_new_like("zeros", self, size, dtype, layout, device,
                          pin_memory);
}

Tensor batch_new_ones(const Tensor& self, const std::vector<int64_t>& size,
                      std::optional<DType> dtype, std::optional<int64_t> layout,
                      std::optional<Device> device,
                      std::optional<bool> pin_memory) {
    return batch_new_like("ones", self, size, dtype, layout, device,
                          pin_memory);
}

// ---------------------------------------------------------------------------
// Advanced indexing: align the indexed tensor with every present index
// tensor (and the assigned values for index_put_) so each batch element
// gathers with its own indices.
// ---------------------------------------------------------------------------

std::vector<Tensor> index_present_tensors(
    const std::vector<std::optional<Tensor>>& indices) {
    std::vector<Tensor> present;
    present.reserve(indices.size());
    for (const auto& index : indices) {
        if (index.has_value()) present.push_back(*index);
    }
    return present;
}

std::vector<std::optional<Tensor>> index_aligned_slots(
    const std::vector<std::optional<Tensor>>& indices,
    const std::vector<Tensor>& aligned, size_t aligned_count) {
    // `aligned` holds the aligned form of every present index in order; walk
    // the original slots and reinsert them positionally.
    std::vector<std::optional<Tensor>> slots;
    slots.reserve(indices.size());
    size_t next = 0;
    for (const auto& index : indices) {
        if (index.has_value() && next < aligned_count) {
            slots.push_back(aligned[next]);
            ++next;
        } else {
            slots.push_back(std::nullopt);
        }
    }
    return slots;
}

Tensor batch_index(const Tensor& self,
                   const std::vector<std::optional<Tensor>>& indices) {
    std::vector<Tensor> operands;
    operands.push_back(self);
    for (const Tensor& index : index_present_tensors(indices)) {
        operands.push_back(index);
    }
    auto aligned = align_tensor_list(operands);
    std::vector<std::optional<Tensor>> aligned_indices = index_aligned_slots(
        indices, aligned.first, aligned.first.size());
    Tensor result = call_next<Tensor, const Tensor&,
                              const std::vector<std::optional<Tensor>>&>(
        "index.Tensor", aligned.first[0], aligned.first[0], aligned_indices);
    return make_batched(result, 0, aligned.second);
}

Tensor batch_index_put_(Tensor self,
                        const std::vector<std::optional<Tensor>>& indices,
                        const Tensor& values, bool accumulate) {
    std::vector<Tensor> operands;
    operands.push_back(self);
    for (const Tensor& index : index_present_tensors(indices)) {
        operands.push_back(index);
    }
    operands.push_back(values);
    auto aligned = align_tensor_list(operands);
    std::vector<std::optional<Tensor>> aligned_indices = index_aligned_slots(
        indices, aligned.first, aligned.first.size());
    Tensor result = call_next<Tensor, Tensor&,
                              const std::vector<std::optional<Tensor>>&,
                              const Tensor&, bool>(
        "index_put_", aligned.first[0], aligned.first[0], aligned_indices,
        aligned.first.back(), accumulate);
    return make_batched(result, 0, aligned.second);
}

void register_batch_rules(tensorplay::Library& library) {
    library.impl("neg", &batch_neg);
    library.impl("negative", &batch_negative);
    library.impl("abs", &batch_abs);
    library.impl("exp", &batch_exp);
    library.impl("log", &batch_log);
    library.impl("sin", &batch_sin);
    library.impl("cos", &batch_cos);
    library.impl("sinh", &batch_sinh);
    library.impl("cosh", &batch_cosh);
    library.impl("tanh", &batch_tanh);
    library.impl("sqrt", &batch_sqrt);
    library.impl("rsqrt", &batch_rsqrt);
    library.impl("sigmoid", &batch_sigmoid);
    library.impl("relu", &batch_relu);
    library.impl("floor", &batch_floor);
    library.impl("ceil", &batch_ceil);
    library.impl("round", &batch_round);
    library.impl("trunc", &batch_trunc);
    library.impl("erf", &batch_erf);
    library.impl("erfc", &batch_erfc);
    library.impl("log1p", &batch_log1p);
    library.impl("expm1", &batch_expm1);
    library.impl("mul.Tensor", &batch_mul);
    library.impl("div.Tensor", &batch_div);
    library.impl("maximum", &batch_maximum);
    library.impl("minimum", &batch_minimum);
    library.impl("logical_and", &batch_logical_and);
    library.impl("logical_or", &batch_logical_or);
    library.impl("logical_xor", &batch_logical_xor);
    library.impl("bitwise_not", &batch_bitwise_not);
    library.impl("bitwise_and.Tensor", &batch_bitwise_and);
    library.impl("bitwise_or.Tensor", &batch_bitwise_or);
    library.impl("bitwise_xor.Tensor", &batch_bitwise_xor);
    library.impl("bitwise_left_shift.Tensor", &batch_bitwise_lshift);
    library.impl("bitwise_right_shift.Tensor", &batch_bitwise_rshift);
    library.impl("bitwise_and.Scalar", &batch_bitwise_and_scalar);
    library.impl("bitwise_or.Scalar", &batch_bitwise_or_scalar);
    library.impl("bitwise_xor.Scalar", &batch_bitwise_xor_scalar);
    library.impl("bitwise_left_shift.Tensor_Scalar", &batch_bitwise_lshift_scalar);
    library.impl("bitwise_right_shift.Tensor_Scalar", &batch_bitwise_rshift_scalar);
    library.impl("bitwise_and.Scalar_Tensor", &batch_bitwise_and_stensor);
    library.impl("bitwise_or.Scalar_Tensor", &batch_bitwise_or_stensor);
    library.impl("bitwise_xor.Scalar_Tensor", &batch_bitwise_xor_stensor);
    library.impl("bitwise_left_shift.Scalar_Tensor", &batch_bitwise_lshift_stensor);
    library.impl("bitwise_right_shift.Scalar_Tensor", &batch_bitwise_rshift_stensor);
    library.impl("add.Scalar", &batch_add_scalar);
    library.impl("sub.Scalar", &batch_sub_scalar);
    library.impl("mul.Scalar", &batch_mul_scalar);
    library.impl("div.Scalar", &batch_div_scalar);
    library.impl("add.Tensor", &batch_add);
    library.impl("sub.Tensor", &batch_sub);
    library.impl("pow.Tensor_Scalar", &batch_pow_scalar);
    library.impl("pow.Tensor_Tensor", &batch_pow_tensor);
    library.impl("sum", &batch_sum);
    library.impl("sum.dim_IntList", &batch_sum_dim);
    library.impl("view", &batch_view);
    library.impl("permute", &batch_permute);
    library.impl("transpose", &batch_transpose);
    library.impl("movedim", &batch_movedim);
    library.impl("reshape", &batch_reshape);
    library.impl("expand", &batch_expand);
    library.impl("squeeze", &batch_squeeze);
    library.impl("squeeze.dim", &batch_squeeze_dim);
    library.impl("squeeze.dims", &batch_squeeze_dims);
    library.impl("unsqueeze", &batch_unsqueeze);
    library.impl("contiguous", &batch_contiguous);
    library.impl("select.int", &batch_select);
    library.impl("slice", &batch_slice);
    library.impl("narrow", &batch_narrow);
    library.impl("index_select", &batch_index_select);
    library.impl("cat", &batch_cat);
    library.impl("stack", &batch_stack);
    library.impl("mm", &batch_mm);
    library.impl("matmul", &batch_matmul);
    library.impl("bmm", &batch_bmm);
    library.impl("linear", &batch_linear);
    library.impl("rand", &batch_rand);
    library.impl("randn", &batch_randn);
    library.impl("randint", &batch_randint);
    library.impl("randperm", &batch_randperm);
    library.impl("rand_like", &batch_rand_like);
    library.impl("randint_like", &batch_randint_like);
    library.impl("randn_like", &batch_randn_like);
    library.impl("eq.Tensor", &batch_eq_tensor);
    library.impl("ne.Tensor", &batch_ne_tensor);
    library.impl("lt.Tensor", &batch_lt_tensor);
    library.impl("le.Tensor", &batch_le_tensor);
    library.impl("gt.Tensor", &batch_gt_tensor);
    library.impl("ge.Tensor", &batch_ge_tensor);
    library.impl("eq.Scalar", &batch_eq_scalar);
    library.impl("ne.Scalar", &batch_ne_scalar);
    library.impl("lt.Scalar", &batch_lt_scalar);
    library.impl("le.Scalar", &batch_le_scalar);
    library.impl("gt.Scalar", &batch_gt_scalar);
    library.impl("ge.Scalar", &batch_ge_scalar);
    library.impl("where.self", &batch_where_self);
    library.impl("where.ScalarSelf", &batch_where_scalar_self);
    library.impl("where.ScalarOther", &batch_where_scalar_other);
    library.impl("where.Scalar", &batch_where_scalar);
    library.impl("clamp", &batch_clamp);
    library.impl("cumsum", &batch_cumsum);
    library.impl("logsumexp", &batch_logsumexp);
    library.impl("all.dim", &batch_all_dim);
    library.impl("max.dim", &batch_max_dim);
    library.impl("argsort", &batch_argsort);
    library.impl("argsort.stable", &batch_argsort_stable);
    library.impl("gather", &batch_gather);
    library.impl("repeat_interleave.self_int", &batch_repeat_interleave_self_int);
    library.impl("to.dtype", &batch_to_dtype);
    library.impl("tril", &batch_tril);
    library.impl("new_zeros", &batch_new_zeros);
    library.impl("new_ones", &batch_new_ones);
    library.impl("index.Tensor", &batch_index);
    library.impl("index_put_", &batch_index_put_);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(VmapCPU, NativeBatchRulesCPU) {
    register_batch_rules(m);
}

TENSORPLAY_LIBRARY_IMPL(VmapCUDA, NativeBatchRulesCUDA) {
    register_batch_rules(m);
}
