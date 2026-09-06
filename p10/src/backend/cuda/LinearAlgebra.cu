// Linear-algebra native wrappers.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

namespace tensorplay::cuda {

namespace ops = tensorplay::tpx::ops;

namespace {

Tensor linalg_multi_dot_impl(const std::vector<Tensor>& tensors) {
    const size_t count = tensors.size();
    if (count < 2) {
        TP_THROW(RuntimeError,
                 "linalg.multi_dot(): expected at least 2 tensors");
    }

    std::vector<Tensor> matrices(count);
    std::vector<int64_t> output_shape;
    const Tensor& first = tensors.front();
    const Tensor& last = tensors.back();
    if (!first.defined() || (first.dim() != 1 && first.dim() != 2)) {
        TP_THROW(RuntimeError,
                 "linalg.multi_dot(): the first tensor must be 1-D or 2-D");
    }
    if (!last.defined() || (last.dim() != 1 && last.dim() != 2)) {
        TP_THROW(RuntimeError,
                 "linalg.multi_dot(): the last tensor must be 1-D or 2-D");
    }

    const bool first_vector = first.dim() == 1;
    const bool last_vector = last.dim() == 1;
    matrices[0] = first_vector ? ops::unsqueeze(first, 0) : first;
    matrices[count - 1] = last_vector ? ops::unsqueeze(last, -1) : last;
    if (!first_vector) output_shape.push_back(first.size(0));
    if (!last_vector) output_shape.push_back(last.size(-1));

    for (size_t i = 1; i + 1 < count; ++i) {
        if (!tensors[i].defined() || tensors[i].dim() != 2) {
            TP_THROW(RuntimeError,
                     "linalg.multi_dot(): middle tensors must be 2-D");
        }
        matrices[i] = tensors[i];
    }

    const DType dtype = matrices[0].dtype();
    const Device device = matrices[0].device();
    for (size_t i = 1; i < count; ++i) {
        if (matrices[i].dtype() != dtype) {
            TP_THROW(TypeError,
                     "linalg.multi_dot(): all tensors must have the same dtype");
        }
        if (matrices[i].device() != device) {
            TP_THROW(DeviceMismatchError,
                     "linalg.multi_dot(): all tensors must be on the same device");
        }
        if (matrices[i - 1].size(-1) != matrices[i].size(0)) {
            TP_THROW(RuntimeError,
                     "linalg.multi_dot(): tensor shapes cannot be multiplied");
        }
    }

    std::vector<int64_t> dimensions(count + 1);
    dimensions[0] = matrices[0].size(0);
    for (size_t i = 0; i < count; ++i) {
        dimensions[i + 1] = matrices[i].size(1);
    }

    std::vector<std::vector<int64_t>> costs(
        count, std::vector<int64_t>(count, 0));
    std::vector<std::vector<size_t>> splits(
        count, std::vector<size_t>(count, 0));
    for (size_t length = 2; length <= count; ++length) {
        for (size_t start = 0; start + length <= count; ++start) {
            const size_t end = start + length - 1;
            int64_t best = std::numeric_limits<int64_t>::max();
            for (size_t middle = start; middle < end; ++middle) {
                const int64_t candidate =
                    costs[start][middle] + costs[middle + 1][end] +
                    dimensions[start] * dimensions[middle + 1] *
                        dimensions[end + 1];
                if (candidate < best) {
                    best = candidate;
                    splits[start][end] = middle;
                }
            }
            costs[start][end] = best;
        }
    }

    std::function<Tensor(size_t, size_t)> multiply =
        [&](size_t start, size_t end) -> Tensor {
        if (start == end) return matrices[start];
        const size_t middle = splits[start][end];
        return ops::matmul(multiply(start, middle),
                           multiply(middle + 1, end));
    };
    return ops::view(multiply(0, count - 1), output_shape);
}

int64_t shape_product(const std::vector<int64_t>& shape) {
    int64_t product = 1;
    for (int64_t size : shape) product *= size;
    return product;
}

Tensor linalg_tensorinv_impl(const Tensor& self, int64_t ind) {
    if (ind <= 0) {
        TP_THROW(RuntimeError,
                 "linalg.tensorinv(): ind must be a positive integer");
    }
    if (ind > self.dim()) {
        TP_THROW(RuntimeError,
                 "linalg.tensorinv(): ind must not exceed the tensor rank");
    }

    const std::vector<int64_t> shape =
        static_cast<std::vector<int64_t>>(self.shape());
    const auto split = shape.begin() + ind;
    const std::vector<int64_t> front(shape.begin(), split);
    const std::vector<int64_t> tail(split, shape.end());
    const int64_t front_product = shape_product(front);
    const int64_t tail_product = shape_product(tail);
    if (front_product != tail_product) {
        TP_THROW(RuntimeError,
                 "linalg.tensorinv(): products of dimensions must match");
    }

    std::vector<int64_t> result_shape = tail;
    result_shape.insert(result_shape.end(), front.begin(), front.end());
    Tensor matrix = ops::reshape(self, {tail_product, front_product});
    return ops::reshape(ops::linalg_inv(matrix), result_shape);
}

Tensor linalg_tensorsolve_impl(
    const Tensor& self, const Tensor& other,
    const std::optional<std::vector<int64_t>>& dims) {
    Tensor working = self;
    if (dims.has_value()) {
        const int64_t ndim = self.dim();
        std::vector<int64_t> normalized;
        normalized.reserve(dims->size());
        std::vector<bool> selected(static_cast<size_t>(ndim), false);
        for (int64_t dim : *dims) {
            if (dim < 0) dim += ndim;
            if (dim < 0 || dim >= ndim) {
                TP_THROW(IndexError,
                         "linalg.tensorsolve(): dimension is out of range");
            }
            if (selected[static_cast<size_t>(dim)]) {
                TP_THROW(ValueError,
                         "linalg.tensorsolve(): dimensions must be unique");
            }
            selected[static_cast<size_t>(dim)] = true;
            normalized.push_back(dim);
        }
        std::vector<int64_t> destination;
        destination.reserve(normalized.size());
        const int64_t first_destination =
            ndim - static_cast<int64_t>(normalized.size());
        for (size_t i = 0; i < normalized.size(); ++i) {
            destination.push_back(first_destination + static_cast<int64_t>(i));
        }
        working = ops::movedim(working, normalized, destination);
    }

    if (other.dim() > working.dim()) {
        TP_THROW(RuntimeError,
                 "linalg.tensorsolve(): right-hand side rank is too large");
    }
    const std::vector<int64_t> working_shape =
        static_cast<std::vector<int64_t>>(working.shape());
    const auto split = working_shape.begin() + other.dim();
    const std::vector<int64_t> result_shape(split, working_shape.end());
    const int64_t result_product = shape_product(result_shape);
    const int64_t other_product =
        shape_product(static_cast<std::vector<int64_t>>(other.shape()));
    if (result_product != other_product) {
        TP_THROW(RuntimeError,
                 "linalg.tensorsolve(): flattened dimensions must match");
    }

    Tensor matrix = ops::reshape(working, {result_product, result_product});
    Tensor rhs = ops::reshape(other, {other_product});
    return ops::reshape(ops::linalg_solve(matrix, rhs), result_shape);
}

}  // namespace

Tensor ger_native_cuda(const Tensor& self, const Tensor& vec2) {
    return ops::outer(self, vec2);
}

Tensor kron_native_cuda(const Tensor& self, const Tensor& other) {
    const int64_t maxdim = std::max(self.dim(), other.dim());
    const int64_t pad_self = maxdim - self.dim();
    const int64_t pad_other = maxdim - other.dim();
    std::vector<int64_t> a_shape(2 * maxdim);
    std::vector<int64_t> b_shape(2 * maxdim);
    std::vector<int64_t> result_shape(maxdim);
    for (int64_t i = 0; i < maxdim; ++i) {
        a_shape[2 * i] = i >= pad_self ? self.size(i - pad_self) : 1;
        a_shape[2 * i + 1] = 1;
        b_shape[2 * i] = 1;
        b_shape[2 * i + 1] = i >= pad_other ? other.size(i - pad_other) : 1;
        result_shape[i] = a_shape[2 * i] * b_shape[2 * i + 1];
    }
    return ops::view(
        ops::mul(ops::view(self, a_shape), ops::view(other, b_shape)),
        result_shape);
}

Tensor linalg_multi_dot_native_cuda(const std::vector<Tensor>& tensors) {
    return linalg_multi_dot_impl(tensors);
}

Tensor& linalg_multi_dot_native_cuda_out(const std::vector<Tensor>& tensors,
                                         Tensor& out) {
    Tensor result = linalg_multi_dot_impl(tensors);
    if (!out.defined()) {
        out = result;
        return out;
    }
    if (out.dtype() != result.dtype()) {
        TP_THROW(TypeError,
                 "linalg.multi_dot(): output dtype must match result dtype");
    }
    if (out.device() != result.device()) {
        TP_THROW(DeviceMismatchError,
                 "linalg.multi_dot(): output device must match input device");
    }
    out.resize_(static_cast<std::vector<int64_t>>(result.shape()));
    out.copy_(result);
    return out;
}

Tensor linalg_tensorinv_native_cuda(const Tensor& self, int64_t ind) {
    return linalg_tensorinv_impl(self, ind);
}

Tensor& linalg_tensorinv_native_cuda_out(const Tensor& self, int64_t ind,
                                         Tensor& out) {
    if (out.defined()) {
        if (out.dtype() != self.dtype()) {
            TP_THROW(TypeError,
                     "linalg.tensorinv(): output dtype must match input dtype");
        }
        if (out.device() != self.device()) {
            TP_THROW(DeviceMismatchError,
                     "linalg.tensorinv(): output device must match input device");
        }
    }
    Tensor result = linalg_tensorinv_impl(self, ind);
    if (!out.defined()) {
        out = result;
        return out;
    }
    out.resize_(static_cast<std::vector<int64_t>>(result.shape()));
    out.copy_(result);
    return out;
}

Tensor linalg_tensorsolve_native_cuda(
    const Tensor& self, const Tensor& other,
    std::optional<std::vector<int64_t>> dims) {
    return linalg_tensorsolve_impl(self, other, dims);
}

Tensor& linalg_tensorsolve_native_cuda_out(
    const Tensor& self, const Tensor& other,
    std::optional<std::vector<int64_t>> dims, Tensor& out) {
    if (out.defined()) {
        if (out.dtype() != self.dtype()) {
            TP_THROW(TypeError,
                     "linalg.tensorsolve(): output dtype must match input dtype");
        }
        if (out.device() != self.device()) {
            TP_THROW(DeviceMismatchError,
                     "linalg.tensorsolve(): output device must match input device");
        }
    }
    Tensor result = linalg_tensorsolve_impl(self, other, dims);
    if (!out.defined()) {
        out = result;
        return out;
    }
    out.resize_(static_cast<std::vector<int64_t>>(result.shape()));
    out.copy_(result);
    return out;
}

Tensor matrix_power_native_cuda(const Tensor& self, int64_t n) {
    if (self.dim() < 2 || self.size(-2) != self.size(-1)) {
        TP_THROW(RuntimeError, "matrix_power(): expected a square matrix");
    }
    const int64_t order = self.size(-1);
    if (n == 0) {
        Tensor result = ops::clone(self, 0);
        Tensor identity = ops::eye(order, order, self.dtype(),
                                   std::optional<Device>(self.device()), false);
        if (self.dim() > 2) {
            const std::vector<int64_t> shape =
                static_cast<std::vector<int64_t>>(self.shape());
            identity = ops::clone(ops::expand(identity, shape, false), 0);
        }
        return ops::copy_(result, identity, false);
    }
    if (n == 1) return ops::clone(self, 0);
    if (n == std::numeric_limits<int64_t>::min()) {
        TP_THROW(RuntimeError, "matrix_power(): exponent is too small");
    }
    Tensor base = n < 0 ? ops::linalg_inv(self) : self;
    if (n == -1) return base;
    n = std::abs(n);
    if (n == 2) return ops::matmul(base, base);
    if (n == 3) return ops::matmul(ops::matmul(base, base), base);

    Tensor z;
    Tensor result;
    while (n > 0) {
        const int64_t bit = n % 2;
        n /= 2;
        z = z.defined() ? ops::matmul(z, z) : base;
        if (bit == 1) {
            result = result.defined() ? ops::matmul(result, z) : z;
        }
    }
    return result;
}

Tensor& matrix_power_native_cuda_out(const Tensor& self, int64_t n, Tensor& out) {
    if (out.device() != self.device()) {
        TP_THROW(DeviceMismatchError,
                 "matrix_power: output must be on the same device as input");
    }
    if (out.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "matrix_power: output dtype must match input dtype");
    }
    out.resize_(static_cast<std::vector<int64_t>>(self.shape()));
    out.copy_(matrix_power_native_cuda(self, n));
    return out;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, NativeLinearAlgebra) {
    m.impl("ger", ger_native_cuda);
    m.impl("kron", kron_native_cuda);
    m.impl("linalg_multi_dot", linalg_multi_dot_native_cuda);
    m.impl("linalg_multi_dot.out", linalg_multi_dot_native_cuda_out);
    m.impl("linalg_tensorinv", linalg_tensorinv_native_cuda);
    m.impl("linalg_tensorinv.out", linalg_tensorinv_native_cuda_out);
    m.impl("linalg_tensorsolve", linalg_tensorsolve_native_cuda);
    m.impl("linalg_tensorsolve.out", linalg_tensorsolve_native_cuda_out);
    m.impl("matrix_power", matrix_power_native_cuda);
    m.impl("matrix_power.out", matrix_power_native_cuda_out);
    m.impl("linalg_matrix_power", matrix_power_native_cuda);
    m.impl("linalg_matrix_power.out", matrix_power_native_cuda_out);
}

} // namespace tensorplay::cuda
