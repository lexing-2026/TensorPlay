// Vectorized arange fill, compiled once per CPU capability tier (see
// TP_CPU_KERNEL_SRCS in p10/CMakeLists.txt).  Each copy lands in the
// CPU_CAPABILITY inline namespace and registers its own slot on the stub
// declared in cpu/RangeKernels.h; DispatchStub picks the best tier at
// runtime.  Length computation, dtype resolution and dispatcher registration
// stay in the base tier (FactoryKernels.cpp).

#include "cpu/RangeKernels.h"
#include "DType.h"
#include "Exception.h"
#include "Parallel.h"

#include "cpu/vec/vec.h"

#include "Half.h"
#include "BFloat16.h"

#include <cstdint>

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

namespace {

template <typename T>
void arange_fill_typed(T* data, int64_t steps, double start, double step) {
    using Vec = tensorplay::vec::Vectorized<T>;
    constexpr int64_t width = Vec::size();
    parallel_for(0, steps, GRAIN_SIZE,
        [&](int64_t begin, int64_t last) {
            int64_t index = begin;
            const int64_t vector_end = begin +
                ((last - begin) / width) * width;
            for (; index < vector_end; index += width) {
                const T base = static_cast<T>(
                    start + static_cast<double>(index) * step);
                Vec::arange(base, step).store(data + index);
            }
            for (; index < last; ++index) {
                data[index] = static_cast<T>(
                    start + static_cast<double>(index) * step);
            }
        });
}

void arange_fill_impl(void* data, int64_t steps, double start, double step,
                      int dtype) {
#define TP_ARANGE_FILL_CASE(ctype, name_)                        \
    case DType::name_:                                           \
        arange_fill_typed<ctype>(                                \
            static_cast<ctype*>(data), steps, start, step);      \
        break;
    switch (static_cast<DType>(dtype)) {
        TP_ARANGE_FILL_CASE(uint8_t, UInt8)
        TP_ARANGE_FILL_CASE(int8_t, Int8)
        TP_ARANGE_FILL_CASE(int16_t, Int16)
        TP_ARANGE_FILL_CASE(int32_t, Int32)
        TP_ARANGE_FILL_CASE(int64_t, Int64)
        TP_ARANGE_FILL_CASE(float, Float32)
        TP_ARANGE_FILL_CASE(double, Float64)
        TP_ARANGE_FILL_CASE(tensorplay::Half, Float16)
        TP_ARANGE_FILL_CASE(tensorplay::BFloat16, BFloat16)
        default:
            TP_THROW(NotImplementedError,
                     "\"arange\" not implemented for this dtype");
    }
#undef TP_ARANGE_FILL_CASE
}

} // namespace

// One slot per tier TU (the specializations live outside the capability
// namespace, so cross-tier duplicates collide at link time): DEFAULT/AVX2
// copies register their own slot; the AVX512 copy uses ALSO_ instead of
// REGISTER_DISPATCH, which would otherwise null its slot (opt-in design).
#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(arange_fill_stub, &arange_fill_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(arange_fill_stub, &arange_fill_impl);
#endif

} // namespace cpu
} // namespace tensorplay
