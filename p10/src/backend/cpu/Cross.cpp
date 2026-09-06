// Cross-product kernels. Keep the default-dimension behavior in this
// translation unit and the device implementation in cuda/Cross.cu.

#include "Tensor.h"
#include "Complex.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"
#include "Utils.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <optional>
#include <type_traits>
#include <vector>

namespace tensorplay::cpu {

namespace {

int64_t wrap_cross_dim(int64_t dim, int64_t ndim) {
    if (ndim <= 0 || dim < -ndim || dim >= ndim) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 -ndim, ", ", ndim - 1, "], but got ", dim, ")");
    }
    return dim < 0 ? dim + ndim : dim;
}

int64_t default_cross_dim(const std::optional<int64_t>& dimension,
                          const Tensor& input) {
    if (dimension.has_value()) {
        return *dimension;
    }

    // Emit this warning once per process. The default rule is the first input
    // dimension whose length is three.
    static std::once_flag warning_once;
    std::call_once(warning_once, [] {
        TP_WARN("Using cross without specifying the dim arg is deprecated.\n",
                "Please either pass the dim explicitly or use linalg_cross.\n",
                "The default value of dim will change to agree with that of linalg.cross in a future release.");
    });

    for (int64_t dim = 0; dim < input.dim(); ++dim) {
        if (input.size(dim) == 3) {
            return dim;
        }
    }
    TP_THROW(RuntimeError, "no dimension of size 3 in input");
}

std::vector<int64_t> cross_output_shape(const Tensor& input,
                                        const Tensor& other,
                                        int64_t& dim) {
    if (input.dim() != other.dim()) {
        TP_THROW(RuntimeError,
                 "linalg.cross: inputs must have the same number of dimensions.");
    }
    if (input.dim() == 0) {
        TP_THROW(IndexError,
                 "Dimension out of range (expected to be in range of [-1, 0], but got -1)");
    }

    dim = wrap_cross_dim(dim, input.dim());
    if (input.size(dim) != 3 || other.size(dim) != 3) {
        TP_THROW(RuntimeError, "linalg.cross: inputs dimension ", dim,
                 " must have length 3. Got ", input.size(dim), " and ",
                 other.size(dim));
    }
    if (input.dtype() != other.dtype()) {
        TP_THROW(RuntimeError, "expected scalar type ", toString(input.dtype()),
                 " but found ", toString(other.dtype()));
    }
    if (input.device() != other.device()) {
        TP_THROW(DeviceMismatchError,
                 "Expected all tensors to be on the same device, but got ",
                 input.device().toString(), " and ", other.device().toString());
    }
    if (!input.device().is_cpu()) {
        TP_THROW(DeviceMismatchError,
                 "cross CPU kernel received non-CPU tensors");
    }
    if (input.dtype() == DType::Bool) {
        TP_THROW(NotImplementedError, "\"cross\" not implemented for 'Bool'");
    }
    if (input.dtype() == DType::ComplexHalf ||
        input.dtype() == DType::BComplex32) {
        TP_THROW(NotImplementedError, "\"cross\" not implemented for reduced complex dtype ",
                 toString(input.dtype()));
    }

    return broadcast_shapes(static_cast<std::vector<int64_t>>(input.shape()),
                             static_cast<std::vector<int64_t>>(other.shape()));
}

template <typename scalar_t>
void apply_cross(const Tensor& result, const Tensor& a, const Tensor& b,
                 int64_t dim) {
    const int64_t total = result.numel() / 3;
    if (total == 0) {
        return;
    }

    const int64_t a_stride = a.stride(dim);
    const int64_t b_stride = b.stride(dim);
    const int64_t r_stride = result.stride(dim);
    const scalar_t* a_ptr = a.data_ptr<scalar_t>();
    const scalar_t* b_ptr = b.data_ptr<scalar_t>();
    scalar_t* r_ptr = result.data_ptr<scalar_t>();
    const int64_t ndim = result.dim();

    // The cross-product dimension is squashed, while remaining dimensions are
    // walked through their real strides. Expanded inputs keep zero strides,
    // and arbitrary non-contiguous views remain valid.
    parallel::parallel_for(0, total, parallel::GRAIN_SIZE,
                           [&](int64_t begin, int64_t end) {
        std::vector<int64_t> position(static_cast<size_t>(ndim), 0);
        int64_t linear = begin;
        int64_t a_start = 0;
        int64_t b_start = 0;
        int64_t r_start = 0;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            if (d == dim) {
                continue;
            }
            const int64_t size = result.size(d);
            position[static_cast<size_t>(d)] = linear % size;
            linear /= size;
            a_start += position[static_cast<size_t>(d)] * a.stride(d);
            b_start += position[static_cast<size_t>(d)] * b.stride(d);
            r_start += position[static_cast<size_t>(d)] * result.stride(d);
        }

        for (int64_t row = begin; row < end; ++row) {
            r_ptr[r_start + 0 * r_stride] =
                a_ptr[a_start + 1 * a_stride] * b_ptr[b_start + 2 * b_stride] -
                a_ptr[a_start + 2 * a_stride] * b_ptr[b_start + 1 * b_stride];
            r_ptr[r_start + 1 * r_stride] =
                a_ptr[a_start + 2 * a_stride] * b_ptr[b_start + 0 * b_stride] -
                a_ptr[a_start + 0 * a_stride] * b_ptr[b_start + 2 * b_stride];
            r_ptr[r_start + 2 * r_stride] =
                a_ptr[a_start + 0 * a_stride] * b_ptr[b_start + 1 * b_stride] -
                a_ptr[a_start + 1 * a_stride] * b_ptr[b_start + 0 * b_stride];

            for (int64_t d = ndim - 1; d >= 0; --d) {
                if (d == dim) {
                    continue;
                }
                ++position[static_cast<size_t>(d)];
                a_start += a.stride(d);
                b_start += b.stride(d);
                r_start += result.stride(d);
                if (position[static_cast<size_t>(d)] == result.size(d)) {
                    position[static_cast<size_t>(d)] = 0;
                    a_start -= result.size(d) * a.stride(d);
                    b_start -= result.size(d) * b.stride(d);
                    r_start -= result.size(d) * result.stride(d);
                } else {
                    break;
                }
            }
        }
    });
}

template <typename scalar_t>
void cross_dtype(const Tensor& result, const Tensor& a, const Tensor& b,
                 int64_t dim) {
    apply_cross<scalar_t>(result, a, b, dim);
}

Tensor cross_impl(const Tensor& input, const Tensor& other, int64_t dim) {
    const std::vector<int64_t> out_shape =
        cross_output_shape(input, other, dim);
    Tensor a = input.expand(out_shape);
    Tensor b = other.expand(out_shape);
    Tensor result = Tensor::empty(out_shape, input.dtype(), input.device());
    if (result.numel() == 0) {
        return result;
    }

    switch (input.dtype()) {
#define TP_CROSS_REAL_CASE(ctype, name) \
        case DType::name: \
            cross_dtype<ctype>(result, a, b, dim); \
            break;
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_CROSS_REAL_CASE)
#undef TP_CROSS_REAL_CASE
        case DType::ComplexFloat:
            cross_dtype<tensorplay::complex<float>>(result, a, b, dim);
            break;
        case DType::ComplexDouble:
            cross_dtype<tensorplay::complex<double>>(result, a, b, dim);
            break;
        case DType::ComplexHalf:
        case DType::BComplex32:
            TP_THROW(NotImplementedError,
                     "cross: reduced complex dtypes are not supported");
        default:
            TP_THROW(NotImplementedError, "cross is not implemented for dtype ",
                     toString(input.dtype()));
    }
    return result;
}

} // namespace

Tensor linalg_cross_cpu(const Tensor& input, const Tensor& other, int64_t dim) {
    return cross_impl(input, other, dim);
}

Tensor cross_cpu(const Tensor& input, const Tensor& other,
                 std::optional<int64_t> dimension) {
    const int64_t dim = default_cross_dim(dimension, input);
    return cross_impl(input, other, dim);
}

TENSORPLAY_LIBRARY_IMPL(CPU, NativeCross) {
    m.impl("cross", cross_cpu);
    m.impl("linalg_cross", linalg_cross_cpu);
}

} // namespace tensorplay::cpu
