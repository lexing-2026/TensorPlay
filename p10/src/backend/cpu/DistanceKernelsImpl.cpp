// Vectorized p-norm distance kernels, compiled once per CPU capability tier
// (see TP_CPU_KERNEL_SRCS in p10/CMakeLists.txt).  Each copy lands in the
// CPU_CAPABILITY inline namespace and registers its own slots on the stubs
// declared in cpu/DistanceKernels.h; DispatchStub picks the best tier at
// runtime.  The pdist/cdist entry points, batch bookkeeping and the matmul
// shortcut stay in the base tier (DistanceKernels.cpp).

#include "cpu/DistanceKernels.h"
#include "DType.h"
#include "Exception.h"
#include "Parallel.h"

#include "cpu/vec/vec.h"

#include <cmath>
#include <cstdint>

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

namespace {

enum class Norm { Zero, One, Two, Infinity, General };

// Distance between one pair of rows.  The vector loop consumes full lanes of
// the difference; the scalar tail finishes the remaining columns.
template <Norm mode, typename T>
T row_distance(const T* lhs, const T* rhs, int64_t width, double p) {
    using Vec = tensorplay::vec::Vectorized<T>;
    Vec aggregate(static_cast<T>(0));
    const int64_t vector_width = Vec::size();
    int64_t column = 0;
    for (; column + vector_width <= width; column += vector_width) {
        const Vec diff =
            (Vec::loadu(lhs + column) - Vec::loadu(rhs + column)).abs();
        if constexpr (mode == Norm::One) {
            aggregate = aggregate + diff;
        } else if constexpr (mode == Norm::Two) {
            aggregate = aggregate + diff * diff;
        } else if constexpr (mode == Norm::Infinity) {
            aggregate = tensorplay::vec::maximum(aggregate, diff);
        } else {
            aggregate = aggregate + diff.pow(Vec(static_cast<T>(p)));
        }
    }

    T result = mode == Norm::Infinity
        ? aggregate.reduce_max()
        : aggregate.reduce_add();
    for (; column < width; ++column) {
        const T diff = static_cast<T>(
            std::abs(lhs[column] - rhs[column]));
        if constexpr (mode == Norm::One) {
            result += diff;
        } else if constexpr (mode == Norm::Two) {
            result += diff * diff;
        } else if constexpr (mode == Norm::Infinity) {
            result = std::max(result, diff);
        } else {
            result += static_cast<T>(std::pow(diff, p));
        }
    }

    if constexpr (mode == Norm::Two) {
        return static_cast<T>(std::sqrt(result));
    } else if constexpr (mode == Norm::General) {
        return static_cast<T>(std::pow(result, 1.0 / p));
    } else {
        return result;
    }
}

// p == 0 counts differing lanes; no arithmetic beyond comparison.
template <typename T>
T row_hamming(const T* lhs, const T* rhs, int64_t width) {
    int64_t count = 0;
    for (int64_t column = 0; column < width; ++column) {
        count += lhs[column] != rhs[column];
    }
    return static_cast<T>(count);
}

template <Norm mode, typename T>
inline T row_pair_distance(const T* lhs, const T* rhs, int64_t width,
                           double p) {
    if constexpr (mode == Norm::Zero) {
        return row_hamming(lhs, rhs, width);
    } else {
        return row_distance<mode>(lhs, rhs, width, p);
    }
}

// Condensed upper-triangle pdist: walking the flat output index backwards
// recovers the row pair (i, j), i < j, without materializing the full matrix.
template <Norm mode, typename T>
void pdist_loop(const T* data, T* output, int64_t n, int64_t width,
                double p) {
    const int64_t outn = n * (n - 1) / 2;
    parallel_for(0, outn, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        const double n2 = static_cast<double>(n) - 0.5;
        int64_t i = static_cast<int64_t>(
            n2 - std::sqrt(n2 * n2 - 2.0 * static_cast<double>(begin) - 1.0));
        int64_t j = begin - n * i + i * (i + 1) / 2 + i + 1;
        for (int64_t index = begin; index < end; ++index) {
            const T* lhs = data + i * width;
            const T* rhs = data + j * width;
            output[index] = row_pair_distance<mode>(lhs, rhs, width, p);
            ++j;
            if (j == n) {
                ++i;
                j = i + 1;
            }
        }
    });
}

template <typename T>
void pdist_typed(const T* data, T* output, int64_t n, int64_t width,
                 double p, int mode) {
    switch (mode) {
        case static_cast<int>(Norm::Zero):
            pdist_loop<Norm::Zero>(data, output, n, width, p);
            break;
        case static_cast<int>(Norm::One):
            pdist_loop<Norm::One>(data, output, n, width, p);
            break;
        case static_cast<int>(Norm::Two):
            pdist_loop<Norm::Two>(data, output, n, width, p);
            break;
        case static_cast<int>(Norm::Infinity):
            pdist_loop<Norm::Infinity>(data, output, n, width, p);
            break;
        default:
            pdist_loop<Norm::General>(data, output, n, width, p);
            break;
    }
}

void pdist_impl(const void* data, void* out, int64_t n, int64_t width,
                double p, int mode, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            pdist_typed<double>(static_cast<const double*>(data),
                                static_cast<double*>(out), n, width, p, mode);
            break;
        case DType::Float32:
            pdist_typed<float>(static_cast<const float*>(data),
                               static_cast<float*>(out), n, width, p, mode);
            break;
        default:
            TP_THROW(TypeError, "pdist: unsupported dtype");
    }
}

template <Norm mode, typename T>
void cdist_loop(const T* lhs_data, const T* rhs_data, T* output_data,
                int64_t batches, int64_t rows1, int64_t rows2, int64_t width,
                double p) {
    const int64_t pair_count = batches * rows1 * rows2;
    const int64_t grain = std::max<int64_t>(1, GRAIN_SIZE / width);
    parallel_for(0, pair_count, grain, [&](int64_t begin, int64_t end) {
        for (int64_t linear = begin; linear < end; ++linear) {
            const int64_t batch_pair = linear % (rows1 * rows2);
            const int64_t batch = linear / (rows1 * rows2);
            const int64_t row1 = batch_pair / rows2;
            const int64_t row2 = batch_pair % rows2;
            const T* lhs_row = lhs_data + (batch * rows1 + row1) * width;
            const T* rhs_row = rhs_data + (batch * rows2 + row2) * width;
            output_data[linear] =
                row_pair_distance<mode>(lhs_row, rhs_row, width, p);
        }
    });
}

template <typename T>
void cdist_typed(const T* lhs_data, const T* rhs_data, T* output_data,
                 int64_t batches, int64_t rows1, int64_t rows2, int64_t width,
                 double p, int mode) {
    switch (mode) {
        case static_cast<int>(Norm::Zero):
            cdist_loop<Norm::Zero>(lhs_data, rhs_data, output_data, batches,
                                   rows1, rows2, width, p);
            break;
        case static_cast<int>(Norm::One):
            cdist_loop<Norm::One>(lhs_data, rhs_data, output_data, batches,
                                  rows1, rows2, width, p);
            break;
        case static_cast<int>(Norm::Two):
            cdist_loop<Norm::Two>(lhs_data, rhs_data, output_data, batches,
                                  rows1, rows2, width, p);
            break;
        case static_cast<int>(Norm::Infinity):
            cdist_loop<Norm::Infinity>(lhs_data, rhs_data, output_data,
                                       batches, rows1, rows2, width, p);
            break;
        default:
            cdist_loop<Norm::General>(lhs_data, rhs_data, output_data, batches,
                                      rows1, rows2, width, p);
            break;
    }
}

void cdist_impl(const void* lhs, const void* rhs, void* out, int64_t batches,
                int64_t rows1, int64_t rows2, int64_t width, double p,
                int mode, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            cdist_typed<double>(static_cast<const double*>(lhs),
                                static_cast<const double*>(rhs),
                                static_cast<double*>(out), batches, rows1,
                                rows2, width, p, mode);
            break;
        case DType::Float32:
            cdist_typed<float>(static_cast<const float*>(lhs),
                               static_cast<const float*>(rhs),
                               static_cast<float*>(out), batches, rows1, rows2,
                               width, p, mode);
            break;
        default:
            TP_THROW(TypeError, "cdist: unsupported dtype");
    }
}

} // namespace

// One slot per tier TU (the specializations live outside the capability
// namespace, so cross-tier duplicates collide at link time): DEFAULT/AVX2
// copies register their own slot; the AVX512 copy uses ALSO_ instead of
// REGISTER_DISPATCH, which would otherwise null its slot (opt-in design).
#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(pdist_stub, &pdist_impl);
REGISTER_DISPATCH(cdist_stub, &cdist_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(pdist_stub, &pdist_impl);
ALSO_REGISTER_AVX512_DISPATCH(cdist_stub, &cdist_impl);
#endif

} // namespace cpu
} // namespace tensorplay
