#include "cpu/QuantizedElementwiseKernels.h"
#include "cpu/vec/vec.h"
#ifndef CPU_CAPABILITY_AVX512
#include "cpu/vec/vec256/vec256_qint.h"
#endif

namespace tensorplay::cpu {
inline namespace CPU_CAPABILITY {
namespace {
template<class Q>
void clip(const void* input, void* output, int64_t size, int64_t lo, int64_t hi) {
    using V = vec::Vectorized<Q>;
    using T = typename Q::underlying;
    const V lower(Q(static_cast<T>(lo))), upper(Q(static_cast<T>(hi)));
    const auto* x = static_cast<const T*>(input);
    auto* y = static_cast<T*>(output);
    int64_t i = 0;
    for (; i + V::size() <= size; i += V::size())
        V::loadu(x + i).relu6(lower, upper).store(y + i);
    if (i < size) V::loadu(x + i, size - i).relu6(lower, upper).store(y + i, size - i);
}

void clip_impl(const void* x, void* y, int64_t size, DType dtype, int64_t lo, int64_t hi) {
    switch (dtype) {
        case DType::QInt8: clip<qint8>(x, y, size, lo, hi); break;
        case DType::QUInt8: clip<quint8>(x, y, size, lo, hi); break;
        case DType::QInt32: clip<qint32>(x, y, size, lo, hi); break;
        default: break;
    }
}

void lookup_impl(const uint8_t* x, uint8_t* y, int64_t size, const float* table) {
    using QV = vec::Vectorized<quint8>;
    using FV = vec::Vectorized<float>;
    const FV one(1.f), zero(0.f);
    int64_t i = 0;
    for (; i + QV::size() <= size; i += QV::size()) {
        auto values = QV::loadu(x + i).dequantize(one, zero);
        for (auto& v : values)
            v = vec::gather<4>(table, vec::convert_to_int_of_same_size<float>(v));
        QV::quantize(values, 1.f, 0, 1.f).store(y + i);
    }
    for (; i < size; ++i) y[i] = static_cast<uint8_t>(table[x[i]]);
}
}
}

#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(quantized_clip_stub, &clip_impl);
REGISTER_DISPATCH(quantized_lookup_stub, &lookup_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(quantized_clip_stub, &clip_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_lookup_stub, &lookup_impl);
#endif
}
