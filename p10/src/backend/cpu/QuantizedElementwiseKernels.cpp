#include "Macros.h"
#include "cpu/QuantizedElementwiseKernels.h"
#include "cpu/vec/functional.h"
#include "cpu/vec/vec.h"
#include "cpu/vec/vec_convert.h"
#include "QuantElementwise.h"
#include "QuantRowwise.h"
#include "QuantPacked.h"
#include "QuantNative.h"
#include "Parallel.h"
#include <cfenv>
#ifndef CPU_CAPABILITY_AVX512
#include "cpu/vec/vec256/vec256_qint.h"
#endif

namespace tensorplay::cpu {
inline namespace CPU_CAPABILITY {
namespace {
constexpr int int4_columns = vec::Vectorized<float>::size() * 4;

void unsigned_pack_impl(const float* input, uint8_t* output, int64_t size, float scale, int64_t zp) {
    using V = vec::Vectorized<float>;
    using QV = vec::Vectorized<quint8>;
    const float inverse = 1.f / scale;
    int64_t i = 0;
#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
    for (; i + QV::size() <= size; i += QV::size()) {
        typename QV::float_vec_return_type values;
        for (size_t j = 0; j < values.size(); ++j) values[j] = V::loadu(input + i + j * V::size());
        QV::quantize(values, scale, static_cast<int32_t>(zp), inverse).store(output + i);
    }
#else
    for (; i + V::size() <= size; i += V::size()) {
        alignas(64) float values[V::size()];
        (V::loadu(input + i) * V(inverse)).round().store(values);
        for (int lane = 0; lane < V::size(); ++lane)
            output[i + lane] = static_cast<uint8_t>(std::clamp<int64_t>(static_cast<int64_t>(values[lane]) + zp, 0, 255));
    }
#endif
    for (; i < size; ++i) output[i] = quant::encode<uint8_t>(input[i], inverse, zp);
}

#if defined(CPU_CAPABILITY_AVX2)
// A code no wider than its slot needs no decision per lane: the two neighbours
// sharing a byte merge inside their own 128-bit half by shifting one byte of each
// pair over the other, and a byte gather keeps the positions that are now whole.
// Running the same merge again folds four codes into a byte, so the halves only
// have to be brought together once per level.
template <int Bits>
__attribute__((always_inline)) inline __m256i codes_to_items(__m256i lower, __m256i upper) {
    // Each narrow below discards a code once it no longer fits: a negative one
    // becomes zero, so only the top of the grid has to be stated, and once per
    // sixteen codes instead of twice per eight.
    __m256i wide = _mm256_min_epi16(_mm256_packs_epi32(lower, upper),
                                    _mm256_set1_epi16((1 << Bits) - 1));
    wide = _mm256_permute4x64_epi64(wide, 0xD8);
    return _mm256_packus_epi16(wide, wide);
}

template <int Shift>
__attribute__((always_inline)) inline __m256i merge_pairs(__m256i items) {
    const __m256i whole = _mm256_or_si256(items,
        _mm256_slli_epi16(_mm256_srli_si256(items, 1), Shift));
    // Bytes 0, 2, 4 and 6 of each half hold a finished byte of codes.
    return _mm256_shuffle_epi8(whole,
        _mm256_set1_epi64x(static_cast<long long>(0xFFFFFFFF06040200ULL)));
}

inline __m256i join_halves(__m256i packed) {
    return _mm256_permutevar8x32_epi32(packed, _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0));
}

// Sixteen codes leave as the bytes that hold them: eight for a nibble, four for a
// pair of bits.
template <int Bits>
__attribute__((always_inline)) inline void pack_sixteen(__m256 lower, __m256 upper,
                                                        uint8_t* output) {
    const __m256i items =
        codes_to_items<Bits>(_mm256_cvtps_epi32(lower), _mm256_cvtps_epi32(upper));
    if constexpr (Bits == 4) {
        _mm_storel_epi64(reinterpret_cast<__m128i*>(output),
                         _mm256_castsi256_si128(join_halves(merge_pairs<4>(items))));
    } else {
        const __m256i halves = join_halves(merge_pairs<2>(items));
        _mm_storeu_si32(output, _mm256_castsi256_si128(merge_pairs<4>(halves)));
    }
}

// The codes of a packed row are its values scaled onto the grid. Widening the
// product to an integer rounds it the same way the tail of the row does, so the
// two vectors of a chunk are all the packer needs.
template <int Bits>
void pack_row_chunk(const float* source, int64_t first, float lo, float inverse,
                    uint8_t* output) {
    using V = vec::Vectorized<float>;
    constexpr int lanes = V::size();
    const V base(lo), step(inverse);
    pack_sixteen<Bits>((V::loadu(source + first) - base) * step,
                       (V::loadu(source + first + lanes) - base) * step, output);
}
#endif

template<int Bits>
void subbyte_pack(const float* input,uint8_t* output,int64_t size,float inverse,float zp) {
    using V=vec::Vectorized<float>;
    constexpr int lanes=V::size(),pack=8/Bits,maximum=(1<<Bits)-1;
    const V inv(inverse),zero_point(zp),lo(0.f),hi(static_cast<float>(maximum)),limit(0x1p63f);
    int64_t i=0;
    const int64_t vector_end=std::fegetround()==FE_TONEAREST ? size : 0;
    for (;i+lanes<=vector_end;i+=lanes) {
        alignas(64) float codes[lanes];
        const V code=(V::loadu(input+i)*inv).round()+zero_point;
        const V valid=(code>=limit.neg())&(code<limit);
        V::blendv(lo,vec::clamp_max(vec::clamp_min(code,lo),hi),valid).store(codes);
        for (int c=0;c<lanes;c+=pack) {
            uint8_t value=0;
            for (int j=0;j<pack;++j) value|=static_cast<uint8_t>(codes[c+j])<<(j*Bits);
            output[(i+c)/pack]=value;
        }
    }
    for (;i<size;i+=pack) {
        uint8_t value=0;
        for (int c=0;c<pack && i+c<size;++c) {
            const float position=std::nearbyint(input[i+c]*inverse)+zp;
            const float code=position>=-0x1p63f && position<0x1p63f
                ? std::clamp(position,0.f,static_cast<float>(maximum)) : 0.f;
            value|=static_cast<uint8_t>(code)<<(c*Bits);
        }
        output[i/pack]=value;
    }
}

void subbyte_pack_impl(const float* input,uint8_t* output,int64_t size,float inverse,float zp,int bits) {
    if (bits==4) subbyte_pack<4>(input,output,size,inverse,zp);
    else if (bits==2) subbyte_pack<2>(input,output,size,inverse,zp);
    else subbyte_pack<1>(input,output,size,inverse,zp);
}

template<int Bits>
void subbyte_unpack(const uint8_t* input,float* output,int64_t size,float scale,float zp) {
    using V=vec::Vectorized<float>;
    constexpr int lanes=V::size(),pack=8/Bits,mask=(1<<Bits)-1;
    const V multiplier(scale),zero_point(zp);
    int64_t i=0;
    for (;i+lanes<=size;i+=lanes) {
        alignas(64) float codes[lanes];
        for (int c=0;c<lanes;++c) codes[c]=(input[(i+c)/pack]>>((c%pack)*Bits))&mask;
        ((V::loadu(codes)-zero_point)*multiplier).store(output+i);
    }
    for (;i<size;++i) output[i]=(static_cast<float>((input[i/pack]>>((i%pack)*Bits))&mask)-zp)*scale;
}

void subbyte_unpack_impl(const uint8_t* input,float* output,int64_t size,float scale,float zp,int bits) {
    if (bits==4) subbyte_unpack<4>(input,output,size,scale,zp);
    else if (bits==2) subbyte_unpack<2>(input,output,size,scale,zp);
    else subbyte_unpack<1>(input,output,size,scale,zp);
}

int64_t dynamic_dot_impl(const uint8_t* x,const int8_t* w,int64_t size) {
    int64_t total=0,k=0;
#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
    using V=vec::Vectorized<int32_t>;
    constexpr int bytes=V::size()*4;
    while (k+bytes<=size) {
        const int64_t stop=std::min<int64_t>(k+8192,size-size%bytes);
        V accum(0);
        for (;k<stop;k+=bytes) {
#ifdef CPU_CAPABILITY_AVX512
            const auto a=_mm512_loadu_si512(x+k),b=_mm512_loadu_si512(w+k);
            accum=accum+V(_mm512_madd_epi16(_mm512_maddubs_epi16(a,b),_mm512_set1_epi16(1)));
#else
            const auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x+k));
            const auto b=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w+k));
            accum=accum+V(_mm256_madd_epi16(_mm256_maddubs_epi16(a,b),_mm256_set1_epi16(1)));
#endif
        }
        alignas(64) int32_t lanes[V::size()];
        accum.store(lanes);
        for (const auto value:lanes) total+=value;
    }
#endif
    for (;k<size;k+=2) {
        int value=static_cast<int>(x[k])*w[k];
        if (k+1<size) value+=static_cast<int>(x[k+1])*w[k+1];
        total+=std::clamp(value,-32768,32767);
    }
    return total;
}

template<int Rows,int Columns>
void dynamic_block_codes(const uint8_t* x,const int8_t* w,int64_t* totals,
        int64_t size,int64_t weight_stride) {
    constexpr int rows=Rows,columns=Columns;
    std::fill(totals,totals+8,0);
    int64_t k=0;
#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
    using V=vec::Vectorized<int32_t>;
    constexpr int bytes=V::size()*4;
    while (k+bytes<=size) {
        const int64_t stop=std::min<int64_t>(k+8192,size-size%bytes);
        V accum[2][4];
        for (int r=0;r<2;++r) for (int c=0;c<4;++c) accum[r][c]=V(0);
        for (;k<stop;k+=bytes) {
#ifdef CPU_CAPABILITY_AVX512
            __m512i weights[4];
            for (int c=0;c<columns;++c) weights[c]=_mm512_loadu_si512(w+c*weight_stride+k);
            #pragma GCC unroll 2
            for (int r=0;r<rows;++r) {
                const auto values=_mm512_loadu_si512(x+r*size+k);
                #pragma GCC unroll 4
                for (int c=0;c<columns;++c)
                    accum[r][c]=accum[r][c]+V(_mm512_madd_epi16(_mm512_maddubs_epi16(values,weights[c]),_mm512_set1_epi16(1)));
            }
#else
            __m256i weights[4];
            for (int c=0;c<columns;++c) weights[c]=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w+c*weight_stride+k));
            #pragma GCC unroll 2
            for (int r=0;r<rows;++r) {
                const auto values=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x+r*size+k));
                #pragma GCC unroll 4
                for (int c=0;c<columns;++c)
                    accum[r][c]=accum[r][c]+V(_mm256_madd_epi16(_mm256_maddubs_epi16(values,weights[c]),_mm256_set1_epi16(1)));
            }
#endif
        }
        alignas(64) int32_t lanes[V::size()];
        for (int r=0;r<rows;++r) for (int c=0;c<columns;++c) {
            accum[r][c].store(lanes);
            for (const auto value:lanes) totals[r*4+c]+=value;
        }
    }
#endif
    for (;k<size;k+=2) for (int r=0;r<rows;++r) for (int c=0;c<columns;++c) {
        int value=static_cast<int>(x[r*size+k])*w[c*weight_stride+k];
        if (k+1<size) value+=static_cast<int>(x[r*size+k+1])*w[c*weight_stride+k+1];
        totals[r*4+c]+=std::clamp(value,-32768,32767);
    }
}

void dynamic_block_impl(const uint8_t* x,const int8_t* w,int64_t* totals,
        int64_t size,int64_t weight_stride,int64_t rows,int64_t columns) {
    auto run=[&]<int Rows>() {
        switch (columns) {
            case 1: dynamic_block_codes<Rows,1>(x,w,totals,size,weight_stride); break;
            case 2: dynamic_block_codes<Rows,2>(x,w,totals,size,weight_stride); break;
            case 3: dynamic_block_codes<Rows,3>(x,w,totals,size,weight_stride); break;
            default: dynamic_block_codes<Rows,4>(x,w,totals,size,weight_stride); break;
        }
    };
    if (rows==1) run.template operator()<1>();
    else run.template operator()<2>();
}

template<class X>
void linear_epilogue_codes(const int32_t* raw, const int64_t* offsets,
        const float* scales, const int64_t* zps, const float* bias, void* output,
        int64_t count, int64_t sum, int64_t k, int64_t xz, double xs, double ys,
        int64_t yz, bool relu, bool output_float, int64_t activation, double slope) {
    using IV = vec::Vectorized<int64_t>;
    using DV = vec::Vectorized<double>;
    using FV = vec::Vectorized<float>;
    constexpr int lanes = DV::size();
    constexpr double lo = std::is_same_v<X, uint8_t> ? 0. : -128.;
    auto* real_output = output_float ? static_cast<float*>(output) : nullptr;
    auto* code_output = output_float ? nullptr : static_cast<X*>(output);
    int64_t n = 0;
    if (!output_float && activation != 2 && std::fegetround() == FE_TONEAREST) {
        const IV zero_point(xz), channel_sum(k * xz - sum);
        const DV input_scale(xs), output_scale(ys), output_zero(yz), zero(0.);
        const DV minimum(lo), maximum(lo + 255.), negative_slope(slope);
        for (; n + lanes <= count; n += lanes) {
            const IV accumulator = vec::convert<int64_t>(vec::Vectorized<int32_t>::loadu(raw + n, lanes))
                - zero_point * IV::loadu(offsets + n) + channel_sum * IV::loadu(zps + n);
            DV value = vec::convert<double>(accumulator) * input_scale
                * vec::convert<double>(FV::loadu(scales + n, lanes));
            if (bias) value = value + vec::convert<double>(FV::loadu(bias + n, lanes));
            if (activation == 1) value = DV::blendv(value * negative_slope, value, value >= zero);
            if (relu) value = vec::clamp_min(value, zero);
            const DV rounded = (value / output_scale).round() + output_zero;
            const DV codes = vec::clamp_min(vec::clamp_max(rounded, maximum), minimum);
            alignas(64) double values[lanes];
            codes.store(values);
            for (int lane = 0; lane < lanes; ++lane) code_output[n + lane] = static_cast<X>(values[lane]);
        }
    }
    for (; n < count; ++n) {
        const int64_t accumulator = static_cast<int64_t>(raw[n]) - xz * offsets[n]
            - zps[n] * sum + k * xz * zps[n];
        double value = quant::packed_real_value(accumulator, xs, scales[n], bias ? bias[n] : 0.f, output_float);
        if (activation == 1) value = value >= 0. ? value : value * slope;
        else if (activation == 2) value = ::tanh(value);
        if (relu) value = std::max(value, 0.);
        if (output_float) real_output[n] = static_cast<float>(value);
        else code_output[n] = static_cast<X>(std::clamp(std::nearbyint(value / ys) + yz, lo, lo + 255.));
    }
}

void linear_epilogue_impl(const int32_t* raw, const int64_t* offsets,
        const float* scales, const int64_t* zps, const float* bias, void* output,
        DType dtype, int64_t count, int64_t sum, int64_t k, int64_t xz, double xs,
        double ys, int64_t yz, bool relu, bool output_float, int64_t activation, double slope) {
    if (dtype == DType::QUInt8) linear_epilogue_codes<uint8_t>(raw, offsets, scales, zps, bias,
        output, count, sum, k, xz, xs, ys, yz, relu, output_float, activation, slope);
    else linear_epilogue_codes<int8_t>(raw, offsets, scales, zps, bias,
        output, count, sum, k, xz, xs, ys, yz, relu, output_float, activation, slope);
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif
void recurrent_step_impl(const float* input,const float* recurrent,const float* hidden,
        const float* cell,float* hy,float* cy,int64_t rows,int64_t width,int mode) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    using V=vec::Vectorized<float>;
    const V one(1.f),zero(0.f);
    const int gates=mode==2 ? 4 : mode==3 ? 3 : 1;
    auto sigmoid=[&](V value) { return (one+(-value).exp()).reciprocal(); };
    for (int64_t row=0;row<rows;++row) {
        const int64_t start=row*gates*width,state=row*width;
        int64_t col=0;
        for (;col+V::size()<=width;col+=V::size()) {
            const int64_t count=V::size(),base=start+col;
            auto a=[&](int gate) { return V::loadu(input+base+gate*width,count); };
            auto b=[&](int gate) { return V::loadu(recurrent+base+gate*width,count); };
            V value;
            if (mode==2) {
                const V ingate=sigmoid(a(0)+b(0)),forgetgate=sigmoid(a(1)+b(1));
                const V candidate=(a(2)+b(2)).tanh(),outgate=sigmoid(a(3)+b(3));
                const V next=forgetgate*V::loadu(cell+state+col,count)+ingate*candidate;
                next.store(cy+state+col,count);
                value=outgate*next.tanh();
            } else if (mode==3) {
                const V reset=sigmoid(b(0)+a(0)),update=sigmoid(b(1)+a(1));
                const V candidate=(a(2)+b(2)*reset).tanh();
                value=(V::loadu(hidden+state+col,count)-candidate)*update+candidate;
            } else {
                value=a(0)+b(0);
                value=mode==0 ? vec::clamp_min(value,zero) : value.tanh();
            }
            value.store(hy+state+col,count);
        }
        for (;col<width;++col) {
            const int64_t base=start+col,index=state+col;
            auto a=[&](int gate) { return input[base+gate*width]; };
            auto b=[&](int gate) { return recurrent[base+gate*width]; };
            auto sigmoid_scalar=[](float value) { return 1.f/(1.f+::expf(-value)); };
            if (mode==2) {
                const float i=sigmoid_scalar(a(0)+b(0)),f=sigmoid_scalar(a(1)+b(1));
                const float g=::tanhf(a(2)+b(2)),o=sigmoid_scalar(a(3)+b(3));
                cy[index]=f*cell[index]+i*g;
                hy[index]=o*::tanhf(cy[index]);
            } else if (mode==3) {
                const float reset=sigmoid_scalar(b(0)+a(0)),update=sigmoid_scalar(b(1)+a(1));
                const float candidate=::tanhf(a(2)+b(2)*reset);
                hy[index]=(hidden[index]-candidate)*update+candidate;
            } else {
                const float value=a(0)+b(0);
                hy[index]=mode==0 ? std::max(value,0.f) : ::tanhf(value);
            }
        }
    }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

void int4_pack_impl(const int32_t* src, uint8_t* dst, int64_t n, int64_t k) {
    const int64_t blocks = (n + int4_columns - 1) / int4_columns;
    parallel::parallel_for(0, blocks, 1, [&](int64_t begin, int64_t end) {
        for (int64_t block = begin; block < end; ++block) {
            const int64_t first = block * int4_columns;
            const int64_t width = std::min<int64_t>(int4_columns, n-first);
            for (int64_t feature = 0; feature < k; ++feature)
                for (int64_t lane = 0; lane < width/2; ++lane) {
#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
                    const int64_t low = width == int4_columns ? lane : lane*2;
                    const int64_t high = width == int4_columns ? lane+width/2 : lane*2+1;
#else
                    const int64_t low = lane*2, high = lane*2+1;
#endif
                    dst[first*k/2 + feature*width/2 + lane] =
                        static_cast<uint8_t>(src[(first+low)*k+feature]) |
                        (static_cast<uint8_t>(src[(first+high)*k+feature]) << 4);
                }
        }
    });
}

template<class T, class X = T, class P = T>
void int4_mm_codes(const X* TP_RESTRICT x, const uint8_t* TP_RESTRICT packed,
        const P* TP_RESTRICT params, T* TP_RESTRICT output,
        int64_t m, int64_t n, int64_t k, int64_t group) {
    using FV = vec::Vectorized<float>;
    constexpr int lanes = FV::size(), columns = int4_columns;
#ifdef CPU_CAPABILITY_AVX512
    const __m512 lookup = _mm512_setr_ps(-8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f,
                                        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f);
#endif
    const int64_t nb = (n+columns-1)/columns, mb = (m+3)/4;
    parallel::parallel_for(0, mb*nb, 1, [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
            const int64_t row = task/nb*4, first = task%nb*columns;
            const int active_width = std::min<int64_t>(columns,n-first), active_rows = std::min<int64_t>(4,m-row);
            auto compute=[&]<int Rows,bool Full>() {
            const int width=Full ? columns : active_width,rows=Full ? Rows : active_rows;
            const int vectors = Full ? 4 : (width+lanes-1)/lanes;
            FV acc[4][4];
            for (int r=0;r<rows;++r) for (int c=0;c<vectors;++c) acc[r][c]=FV(0.f);
            FV scales[4], zeros[4];
            int64_t next_scale=0,group_index=0;
            for (int64_t feature=0;feature<k;++feature) {
                if (feature==next_scale) {
                    #pragma GCC unroll 4
                    for (int c=0;c<vectors;++c) {
                        if (c*lanes+lanes<=width) {
                            const int64_t index=(group_index*n+first+c*lanes)*2;
                            FV a,b;
                            if constexpr (std::is_same_v<P,float>) {
                                a=FV::loadu(params+index); b=FV::loadu(params+index+lanes);
                            } else {
                                using TV=vec::Vectorized<P>;
                                const auto values=vec::convert<float,2>(vec::VectorizedN<P,1>(TV::loadu(params+index)));
                                a=values[0]; b=values[1];
                            }
                            std::tie(scales[c],zeros[c])=vec::deinterleave2(a,b);
                            continue;
                        }
                        alignas(64) float s[lanes]{}, z[lanes]{};
                        for (int lane=0;lane<lanes && c*lanes+lane<width;++lane) {
                            const int64_t index=(group_index*n+first+c*lanes+lane)*2;
                            s[lane]=static_cast<float>(params[index]); z[lane]=static_cast<float>(params[index+1]);
                        }
                        scales[c]=FV::loadu(s); zeros[c]=FV::loadu(z);
                    }
                    next_scale+=group;
                    ++group_index;
                }
                FV activation[4];
                for (int r=0;r<rows;++r) activation[r]=FV(static_cast<float>(x[(row+r)*k+feature]));
                #pragma GCC unroll 4
                for (int c=0;c<vectors;++c) {
                    FV code_vector;
#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
                    if constexpr (Full) {
                        const int offset=c%(vectors/2)*lanes,shift=(c>=vectors/2)*4;
                        const auto* src=packed+first*k/2+feature*width/2+offset;
#ifdef CPU_CAPABILITY_AVX512
                        auto values=_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src)));
                        code_vector=FV(_mm512_permutexvar_ps(_mm512_srli_epi32(values,shift),lookup));
#else
                        auto values=_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src)));
                        values=_mm256_and_si256(_mm256_srli_epi32(values,shift),_mm256_set1_epi32(15));
                        code_vector=FV(_mm256_cvtepi32_ps(_mm256_sub_epi32(values,_mm256_set1_epi32(8))));
#endif
                    } else
#endif
                    {
                    alignas(64) float codes[lanes]{};
                    for (int lane=0;lane<lanes && c*lanes+lane<width;++lane) {
                        const int col=c*lanes+lane;
#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
                        const int byte=width==columns ? col%(width/2) : col/2;
                        const int shift=width==columns ? (col>=width/2)*4 : (col%2)*4;
#else
                        const int byte=col/2, shift=(col%2)*4;
#endif
                        codes[lane]=static_cast<float>(((packed[first*k/2+feature*width/2+byte]>>shift)&15)-8);
                    }
                    code_vector=FV::loadu(codes);
                    }
                    const FV weight=vec::fmadd(code_vector,scales[c],zeros[c]);
                    #pragma GCC unroll 4
                    for (int r=0;r<rows;++r)
                        acc[r][c]=vec::fmadd(activation[r],weight,acc[r][c]);
                }
            }
            for (int r=0;r<rows;++r) {
                if constexpr (Full) {
                    if constexpr (std::is_same_v<T,float>) {
                        for (int c=0;c<vectors;++c) acc[r][c].store(output+(row+r)*n+first+c*lanes);
                    } else {
                        for (int c=0;c<vectors;c+=2) {
                            const auto values=vec::convert<T,1>(vec::VectorizedN<float,2>(acc[r][c],acc[r][c+1]));
                            values.store(output+(row+r)*n+first+c*lanes);
                        }
                    }
                } else for (int c=0;c<vectors;++c) {
                    alignas(64) float values[lanes]; acc[r][c].store(values);
                    for (int lane=0;lane<lanes && c*lanes+lane<width;++lane)
                        output[(row+r)*n+first+c*lanes+lane]=static_cast<T>(values[lane]);
                }
            }
            };
            if (active_width==columns) {
                switch (active_rows) {
                    case 1: compute.template operator()<1,true>(); break;
                    case 2: compute.template operator()<2,true>(); break;
                    case 3: compute.template operator()<3,true>(); break;
                    default: compute.template operator()<4,true>(); break;
                }
            } else compute.template operator()<4,false>();
        }
    });
}

#ifdef CPU_CAPABILITY_AVX512
template<int Rows, int Columns>
void int4_bfloat16_rows(const BFloat16* TP_RESTRICT x, const uint8_t* TP_RESTRICT packed,
        const BFloat16* TP_RESTRICT params, BFloat16* TP_RESTRICT output,
        int64_t m, int64_t n, int64_t k, int64_t group) {
    using V = vec::Vectorized<float>;
    using B = vec::Vectorized<BFloat16>;
    constexpr int vectors = Columns / 16;
    const __m512 lookup = _mm512_setr_ps(-8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f,
                                        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f);
    parallel::parallel_for(0, m / Rows * (n / Columns), 1, [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
            const int64_t row = task / (n / Columns) * Rows, col = task % (n / Columns) * Columns;
            const int shift = col % 64 < 32 ? 0 : 4;
            const uint8_t* weight = packed + col / 64 * 32 * k + col % 32;
            V totals[Rows][vectors];
            #pragma GCC unroll 16
            for (int r = 0; r < Rows; ++r) {
                #pragma GCC unroll 2
                for (int c = 0; c < vectors; ++c) totals[r][c] = V(0.f);
            }
            V scales[vectors], zeros[vectors];
            int64_t next_group = 0;
            for (int64_t feature = 0; feature < k; ++feature) {
                if (feature == next_group) {
                    #pragma GCC unroll 2
                    for (int c = 0; c < vectors; ++c) {
                        const auto values = vec::convert<float, 2>(vec::VectorizedN<BFloat16, 1>(
                            B::loadu(params + (feature / group * n + col + c * 16) * 2)));
                        std::tie(scales[c], zeros[c]) = vec::deinterleave2(values[0], values[1]);
                    }
                    next_group += group;
                }
                V decoded[vectors];
                #pragma GCC unroll 2
                for (int c = 0; c < vectors; ++c) {
                    const auto raw = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(weight + feature * 32 + c * 16)));
                    decoded[c] = vec::fmadd(V(_mm512_permutexvar_ps(_mm512_srli_epi32(raw, shift), lookup)), scales[c], zeros[c]);
                }
                #pragma GCC unroll 16
                for (int r = 0; r < Rows; ++r) {
                    const V activation(static_cast<float>(x[(row + r) * k + feature]));
                    #pragma GCC unroll 2
                    for (int c = 0; c < vectors; ++c) totals[r][c] = vec::fmadd(activation, decoded[c], totals[r][c]);
                }
            }
            #pragma GCC unroll 16
            for (int r = 0; r < Rows; ++r) {
                if constexpr (vectors == 2)
                    vec::convert<BFloat16, 1>(vec::VectorizedN<float, 2>(totals[r][0], totals[r][1])).store(output + (row + r) * n + col);
                else vec::convert<BFloat16>(totals[r][0]).store(output + (row + r) * n + col, Columns);
            }
        }
    });
}
#endif

#if defined(CPU_CAPABILITY_AVX2) && !defined(CPU_CAPABILITY_AVX512)
template<int Rows>
void int4_bfloat16_avx2_tile(const BFloat16* TP_RESTRICT input,
        const uint8_t* TP_RESTRICT packed, const BFloat16* TP_RESTRICT params,
        BFloat16* TP_RESTRICT output, int64_t n, int64_t k, int64_t group) {
    constexpr int columns = 32;
    constexpr int vectors = columns / 8;
    constexpr int prefetch_features = 16 * 4;
    const int prefetch_groups = (prefetch_features + group - 1) / group;
    const int groups = k / group;
    __m256 activation, weights[vectors], accumulators[Rows * vectors];
    __m256 scales[vectors], zeros[vectors];
    const __m256i lane_order = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    const __m256 offset = _mm256_set1_ps(-8.f);

    auto load_scales = [&](int col, int group_index) {
        const auto values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
            params + group_index * n * 2 + 16 * col));
        if (group_index + prefetch_groups < groups)
            _mm_prefetch(reinterpret_cast<const char*>(params +
                (group_index + prefetch_groups) * n * 2 + 16 * col), _MM_HINT_T0);
        __m256 a, b;
        vec::cvtbf16_fp32(values, a, b);
        a = _mm256_permutevar8x32_ps(a, lane_order);
        b = _mm256_permutevar8x32_ps(b, lane_order);
        scales[col] = _mm256_permute2f128_ps(a, b, 0b00100000);
        zeros[col] = _mm256_permute2f128_ps(a, b, 0b00110001);
        zeros[col] = _mm256_fmadd_ps(scales[col], offset, zeros[col]);
    };

    auto initialize = [&](auto i) { accumulators[i] = _mm256_setzero_ps(); };
    tensorplay::ForcedUnroll<Rows * vectors>{}(initialize);

    auto compute = [&](auto i, int feature) {
        constexpr int row = i / vectors;
        constexpr int col = i % vectors;
        if constexpr (col == 0) {
            const float value = static_cast<float>(input[row * k + feature]);
            if (feature + prefetch_features < k)
                _mm_prefetch(reinterpret_cast<const char*>(input + row * k + feature + prefetch_features), _MM_HINT_T0);
            activation = _mm256_set1_ps(value);
        }
        if constexpr (row == 0) {
            if constexpr (col == 0) {
                const __m256i mask = _mm256_set1_epi32(0x0f);
                const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + feature * 16));
                if (feature + prefetch_features < k)
                    _mm_prefetch(reinterpret_cast<const char*>(packed +
                        (feature + prefetch_features) * 16), _MM_HINT_T0);
                __m256i codes = _mm256_cvtepu8_epi32(bytes);
                weights[0] = _mm256_cvtepi32_ps(_mm256_and_si256(codes, mask));
                weights[0] = _mm256_fmadd_ps(weights[0], scales[0], zeros[0]);
                weights[2] = _mm256_cvtepi32_ps(_mm256_srli_epi32(codes, 4));
                weights[2] = _mm256_fmadd_ps(weights[2], scales[2], zeros[2]);
                codes = _mm256_cvtepu8_epi32(_mm_shuffle_epi32(bytes, _MM_SHUFFLE(3, 2, 3, 2)));
                weights[1] = _mm256_cvtepi32_ps(_mm256_and_si256(codes, mask));
                weights[1] = _mm256_fmadd_ps(weights[1], scales[1], zeros[1]);
                weights[3] = _mm256_cvtepi32_ps(_mm256_srli_epi32(codes, 4));
                weights[3] = _mm256_fmadd_ps(weights[3], scales[3], zeros[3]);
            }
        }
        constexpr int index = row * vectors + col;
        accumulators[index] = _mm256_fmadd_ps(activation, weights[col], accumulators[index]);
    };

    for (int feature = 0, group_index = 0; feature < k; ++feature) {
        if (feature % group == 0) {
            tensorplay::ForcedUnroll<vectors>{}(load_scales, group_index++);
        }
        tensorplay::ForcedUnroll<Rows * vectors>{}(compute, feature);
    }

    auto store = [&](auto i) {
        constexpr int row = i / vectors;
        constexpr int col = i % vectors;
        if constexpr (col % 2 == 0) {
            const __m256i result = vec::cvtfp32_bf16(
                accumulators[row * vectors + col], accumulators[row * vectors + col + 1]);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + row * n + col * 8), result);
        }
    };
    tensorplay::ForcedUnroll<Rows * vectors>{}(store);
}

void int4_bfloat16_avx2(const BFloat16* input, const uint8_t* packed,
        const BFloat16* params, BFloat16* output, int64_t m, int64_t n,
        int64_t k, int64_t group) {
    const int64_t column_blocks = n / 32;
    const int64_t row_blocks = (m + 3) / 4;
    parallel::parallel_for(0, row_blocks * column_blocks, 1, [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
            const int64_t row = task / column_blocks * 4;
            const int64_t column = task % column_blocks * 32;
            const int rows = std::min<int64_t>(4, m - row);
            const auto* a = input + row * k;
            const auto* b = packed + column * k / 2;
            const auto* s = params + column * 2;
            auto* c = output + row * n + column;
            switch (rows) {
                case 1: int4_bfloat16_avx2_tile<1>(a, b, s, c, n, k, group); break;
                case 2: int4_bfloat16_avx2_tile<2>(a, b, s, c, n, k, group); break;
                case 3: int4_bfloat16_avx2_tile<3>(a, b, s, c, n, k, group); break;
                default: int4_bfloat16_avx2_tile<4>(a, b, s, c, n, k, group); break;
            }
        }
    });
}
#endif

void int4_mm_impl(const void* x, const uint8_t* packed, const void* params, void* output,
        DType dtype, int64_t m, int64_t n, int64_t k, int64_t group) {
    switch (dtype) {
        case DType::Float32: int4_mm_codes(static_cast<const float*>(x),packed,static_cast<const float*>(params),static_cast<float*>(output),m,n,k,group); break;
        case DType::Float16: int4_mm_codes(static_cast<const Half*>(x),packed,static_cast<const Half*>(params),static_cast<Half*>(output),m,n,k,group); break;
        case DType::BFloat16: {
            const auto* input = static_cast<const BFloat16*>(x);
#ifdef CPU_CAPABILITY_AVX512
            if (m >= 8 && m % 8 == 0 && n % 64 == 0) {
                if (m % 16 == 0)
                    int4_bfloat16_rows<16, 16>(input, packed, static_cast<const BFloat16*>(params), static_cast<BFloat16*>(output), m, n, k, group);
                else int4_bfloat16_rows<8, 32>(input, packed, static_cast<const BFloat16*>(params), static_cast<BFloat16*>(output), m, n, k, group);
                break;
            }
#endif
#if defined(CPU_CAPABILITY_AVX2) && !defined(CPU_CAPABILITY_AVX512)
            if (n % 32 == 0) {
                int4_bfloat16_avx2(input, packed, static_cast<const BFloat16*>(params),
                    static_cast<BFloat16*>(output), m, n, k, group);
                break;
            }
#endif
            if (m < 8) {
                int4_mm_codes(input,packed,static_cast<const BFloat16*>(params),static_cast<BFloat16*>(output),m,n,k,group);
                break;
            }
            // Convert each activation once, then reuse the full-precision
            // values across output tiles without changing accumulation order.
            thread_local std::vector<float> real, coefficients;
            real.resize(m * k);
            using V = vec::Vectorized<BFloat16>;
            int64_t i = 0;
            for (; i + V::size() <= m * k; i += V::size()) {
                const auto values = vec::convert<float,2>(vec::VectorizedN<BFloat16,1>(V::loadu(input + i)));
                values[0].store(real.data() + i);
                values[1].store(real.data() + i + vec::Vectorized<float>::size());
            }
            for (; i < m * k; ++i) real[i] = static_cast<float>(input[i]);
            const auto* source=static_cast<const BFloat16*>(params);
            const int64_t coefficient_count=k/group*n*2;
            coefficients.resize(coefficient_count);
            i=0;
            for (;i+V::size()<=coefficient_count;i+=V::size()) {
                const auto values=vec::convert<float,2>(vec::VectorizedN<BFloat16,1>(V::loadu(source+i)));
                values[0].store(coefficients.data()+i);
                values[1].store(coefficients.data()+i+vec::Vectorized<float>::size());
            }
            for (;i<coefficient_count;++i) coefficients[i]=static_cast<float>(source[i]);
            int4_mm_codes<BFloat16,float,float>(real.data(),packed,coefficients.data(),
                static_cast<BFloat16*>(output),m,n,k,group);
            break;
        }
        default: break;
    }
}

template<class Q>
void batch_norm_codes(const void* xx, void* yy, int64_t channels,
        const float* alpha, const float* beta, int64_t xz, int64_t yz, bool relu,int64_t rows) {
    using T = typename Q::underlying;
    using QV = vec::Vectorized<Q>;
    using FV = vec::Vectorized<float>;
    const FV one(1.f), zero(0.f), zp(static_cast<float>(xz));
    for (int64_t row=0;row<rows;++row) {
    const auto* x = static_cast<const T*>(xx)+row*channels;
    auto* y = static_cast<T*>(yy)+row*channels;
    int64_t c = 0;
    for (; c + QV::size() <= channels; c += QV::size()) {
        auto values = QV::loadu(x+c).dequantize(one, zp);
        for (size_t j = 0; j < values.size(); ++j) {
            const int64_t offset = c + j * FV::size();
            values[j] = vec::fmadd(FV::loadu(alpha+offset), values[j], FV::loadu(beta+offset));
            if (relu) values[j] = vec::clamp_min(values[j], zero);
        }
        QV::quantize(values, 1.f, yz, 1.f).store(y+c);
    }
    for (; c < channels; ++c) {
        float value = alpha[c] * static_cast<float>(static_cast<int64_t>(x[c])-xz) + beta[c];
        if (relu) value = std::max(value, 0.f);
        y[c] = quant::encode<T>(value, 1.f, yz);
    }
    }
}

void batch_norm_impl(const void* x, void* y, int64_t channels, DType dtype,
        const float* alpha, const float* beta, int64_t xz, int64_t yz, bool relu,int64_t rows) {
    switch (dtype) {
        case DType::QInt8: batch_norm_codes<qint8>(x,y,channels,alpha,beta,xz,yz,relu,rows); break;
        case DType::QUInt8: batch_norm_codes<quint8>(x,y,channels,alpha,beta,xz,yz,relu,rows); break;
        case DType::QInt32: batch_norm_codes<qint32>(x,y,channels,alpha,beta,xz,yz,relu,rows); break;
        default: break;
    }
}

template<class Q>
void bilinear_codes(const void* aa, const void* bb, const void* cc, const void* dd,
        void* yy, int64_t size, float h, float w, int64_t zp) {
    using T = typename Q::underlying;
    using QV = vec::Vectorized<Q>;
    using FV = vec::Vectorized<float>;
    const auto *a = static_cast<const T*>(aa), *b = static_cast<const T*>(bb),
        *c = static_cast<const T*>(cc), *d = static_cast<const T*>(dd);
    auto* y = static_cast<T*>(yy);
    const FV one(1.f), zero(0.f), hv(h), h0(1.f-h), wv(w), w0(1.f-w), z(zp);
    int64_t i = 0;
    for (; i + QV::size() <= size; i += QV::size()) {
        auto av = QV::loadu(a+i).dequantize(one, zero);
        const auto bv = QV::loadu(b+i).dequantize(one, zero);
        const auto cv = QV::loadu(c+i).dequantize(one, zero);
        const auto dv = QV::loadu(d+i).dequantize(one, zero);
        for (size_t j = 0; j < av.size(); ++j)
            av[j] = h0 * (w0 * av[j] + wv * bv[j]) + hv * (w0 * cv[j] + wv * dv[j]) - z;
        QV::quantize(av, 1.f, zp, 1.f).store(y+i);
    }
    for (; i < size; ++i) {
        const float value = (1.f-h) * ((1.f-w) * a[i] + w * b[i])
            + h * ((1.f-w) * c[i] + w * d[i]);
        y[i] = quant::encode<T>(value-static_cast<float>(zp), 1.f, zp);
    }
}

void bilinear_impl(const void* a, const void* b, const void* c, const void* d,
        void* y, int64_t size, DType dtype, float h, float w, int64_t zp) {
    switch (dtype) {
        case DType::QInt8: bilinear_codes<qint8>(a,b,c,d,y,size,h,w,zp); break;
        case DType::QUInt8: bilinear_codes<quint8>(a,b,c,d,y,size,h,w,zp); break;
        case DType::QInt32: bilinear_codes<qint32>(a,b,c,d,y,size,h,w,zp); break;
        default: break;
    }
}

template<class Q>
void clip(const void* input, void* output, int64_t size, int64_t lo, int64_t hi) {
    using V = vec::Vectorized<Q>;
    using T = typename Q::underlying;
    const V lower(Q(static_cast<T>(lo))), upper(Q(static_cast<T>(hi)));
    const auto* x = static_cast<const T*>(input);
    auto* y = static_cast<T*>(output);
    int64_t i = 0;
    if (hi == std::numeric_limits<T>::max()) {
        for (; i + 2 * V::size() <= size; i += 2 * V::size()) {
            V first = V::loadu(x + i), second = V::loadu(x + i + V::size());
            first.relu(lower).store(y + i);
            second.relu(lower).store(y + i + V::size());
        }
        for (; i + V::size() <= size; i += V::size()) V::loadu(x + i).relu(lower).store(y + i);
        if (i < size) V::loadu(x + i, size - i).relu(lower).store(y + i, size - i);
    } else {
        for (; i + 2 * V::size() <= size; i += 2 * V::size()) {
            V first = V::loadu(x + i), second = V::loadu(x + i + V::size());
            first.relu6(lower, upper).store(y + i);
            second.relu6(lower, upper).store(y + i + V::size());
        }
        for (; i + V::size() <= size; i += V::size()) V::loadu(x + i).relu6(lower, upper).store(y + i);
        if (i < size) V::loadu(x + i, size - i).relu6(lower, upper).store(y + i, size - i);
    }
}

template<class T>
int64_t byte_sum(const T* values, int64_t count) {
    int64_t sum = 0, i = 0;
#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
    using V = vec::Vectorized<int64_t>;
    constexpr int bytes = V::size() * 8;
    V total(0);
    for (; i + bytes <= count; i += bytes) {
#ifdef CPU_CAPABILITY_AVX512
        auto raw = _mm512_loadu_si512(values + i);
        if constexpr (std::is_signed_v<T>) raw = _mm512_xor_si512(raw, _mm512_set1_epi8(-128));
        total = total + V(_mm512_sad_epu8(raw, _mm512_setzero_si512()));
#else
        auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values + i));
        if constexpr (std::is_signed_v<T>) raw = _mm256_xor_si256(raw, _mm256_set1_epi8(-128));
        total = total + V(_mm256_sad_epu8(raw, _mm256_setzero_si256()));
#endif
    }
    alignas(64) int64_t lanes[V::size()];
    total.store(lanes);
    for (auto value : lanes) sum += value;
    if constexpr (std::is_signed_v<T>) sum -= i * 128;
#endif
    for (; i < count; ++i) sum += values[i];
    return sum;
}

template<class T>
int64_t byte_square_sum(const T* values, int64_t count) {
    int64_t sum = 0, i = 0;
#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
    using V = vec::Vectorized<int32_t>;
    constexpr int bytes = V::size() * 2;
    while (i + bytes <= count) {
        const int64_t end = std::min<int64_t>(i + 65536, count - count % bytes);
        V total(0);
        for (; i < end; i += bytes) {
#ifdef CPU_CAPABILITY_AVX512
            const auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values + i));
            const auto wide = std::is_signed_v<T> ? _mm512_cvtepi8_epi16(raw) : _mm512_cvtepu8_epi16(raw);
            total = total + V(_mm512_madd_epi16(wide, wide));
#else
            const auto raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(values + i));
            const auto wide = std::is_signed_v<T> ? _mm256_cvtepi8_epi16(raw) : _mm256_cvtepu8_epi16(raw);
            total = total + V(_mm256_madd_epi16(wide, wide));
#endif
        }
        alignas(64) int32_t lanes[V::size()];
        total.store(lanes);
        for (auto value : lanes) sum += value;
    }
#endif
    for (; i < count; ++i) sum += static_cast<int64_t>(values[i]) * values[i];
    return sum;
}

template<class Q>
void moment_codes(const void* input, void* output, int64_t count, int64_t rows,
                  double scale, int64_t zp, bool deviation, double correction) {
    using T = typename Q::underlying;
    using FV = vec::Vectorized<float>;
    const auto* x = static_cast<const T*>(input);
    auto* y = static_cast<T*>(output);
    for (int64_t row = 0; row < rows; ++row) {
        const auto* values = x + row * count;
        if constexpr (sizeof(T) == 1) {
            if (!deviation) {
                y[row] = static_cast<T>(std::nearbyint(static_cast<float>(byte_sum(values, count)) / count));
                continue;
            }
        }
        int64_t sum = 0, integer_square_sum = 0;
        double square_sum = 0.;
        int64_t i = 0;
        if constexpr (sizeof(T) == 1) {
            sum = byte_sum(values, count);
            integer_square_sum = byte_square_sum(values, count);
            square_sum = static_cast<double>(integer_square_sum);
        } else {
            for (int64_t j = 0; j < count; ++j) sum += values[j];
            if (deviation) {
                float total = 0.f;
#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
                FV squares(0.f);
                for (; i + FV::size() <= count; i += FV::size()) {
                    const FV v = vec::convert<float>(vec::Vectorized<int32_t>::loadu(values + i));
                    squares = squares + v * v;
                }
                alignas(64) float lanes[FV::size()];
                squares.store(lanes);
                for (const auto value : lanes) total += value;
#endif
                for (; i < count; ++i) { const float v = static_cast<float>(values[i]); total += v * v; }
                square_sum = total;
            }
        }
        if (!deviation) { y[row] = static_cast<T>(std::nearbyint(static_cast<float>(sum) / count)); continue; }
        const double denominator = std::max(static_cast<double>(count) - correction, 0.);
        const double mean = static_cast<double>(sum) / count, unbiased = static_cast<double>(sum) / denominator;
        const double variance = std::max(square_sum / denominator - 2. * mean * unbiased + mean * mean * count / denominator, 0.);
        const float value = static_cast<float>(std::sqrt(variance) * scale);
        y[row] = std::isnan(value) ? (sizeof(T) == 4 ? std::numeric_limits<T>::lowest() : T(0)) :
            quant::encode<T>(value, 1.f / static_cast<float>(scale), zp);
    }
}

void moment_impl(const void* x, void* y, int64_t count, int64_t rows, DType dtype,
                 double scale, int64_t zp, bool deviation, double correction) {
    switch (dtype) {
        case DType::QInt8: moment_codes<qint8>(x,y,count,rows,scale,zp,deviation,correction); break;
        case DType::QUInt8: moment_codes<quint8>(x,y,count,rows,scale,zp,deviation,correction); break;
        case DType::QInt32: moment_codes<qint32>(x,y,count,rows,scale,zp,deviation,correction); break;
        default: break;
    }
}

// A run whose affine coefficients are fixed for every element needs neither an
// index walk nor a per-element decision: one slope and one shift ride through
// the vector lanes next to the codes.
template<class Q>
void normalize_run(const typename Q::underlying* source, typename Q::underlying* destination,
                   int64_t length, float x_zp, float mean_div_scale, float slope, float bias,
                   float y_scale, float inverse, int64_t y_zp) {
    using T = typename Q::underlying;
    using QV = vec::Vectorized<Q>;
    using FV = vec::Vectorized<float>;
    const FV one(1.f), shift(x_zp), mean(mean_div_scale), gradient(slope), offset(bias);
    int64_t i = 0;
    for (; i + QV::size() <= length; i += QV::size()) {
        auto values = QV::loadu(source + i).dequantize(one, shift);
        for (size_t lane = 0; lane < values.size(); ++lane)
            values[lane] = (values[lane] - mean) * gradient + offset;
        QV::quantize(values, y_scale, static_cast<int32_t>(y_zp), inverse).store(destination + i);
    }
    for (; i < length; ++i) {
        const float code = static_cast<float>(source[i]) - x_zp;
        destination[i] = quant::encode<T>((code - mean_div_scale) * slope + bias, inverse, y_zp);
    }
}

// Each row is normalized by its own moments, so the pair of sums runs over a
// dense byte run, and the second pass feeds the shifted codes straight through
// the vector round. A wide row is one run; a group whose affine coefficients
// repeat per channel splits into one run per channel instead. A group held
// channel innermost occupies one short run per spatial position, so both walks
// step by the channel count rather than decode an element's location again.
template<class Q>
void normalize_codes(const void* input, const float* gamma, const float* beta, void* output,
                     int64_t count, int64_t rows, bool affine_per_channel, bool channels_last,
                     int64_t channels, int64_t channel_groups, float x_scale, int64_t x_zp,
                     float y_scale, int64_t y_zp, double eps) {
    using T = typename Q::underlying;
    using QV = vec::Vectorized<Q>;
    using FV = vec::Vectorized<float>;
    const auto* x = static_cast<const T*>(input);
    auto* y = static_cast<T*>(output);
    const float inverse = 1.f / y_scale, shift = static_cast<float>(x_zp);
    const int64_t spans = channels_last ? channels / channel_groups
        : (affine_per_channel ? channels / channel_groups : 1);
    const int64_t span = count / spans;
    parallel::parallel_for(0, rows, 1, [&](int64_t first, int64_t last) {
        for (int64_t row = first; row < last; ++row) {
            const int64_t slice = channels_last ? (row % channel_groups) * spans : 0;
            const int64_t base = channels_last
                ? (row / channel_groups) * count * channel_groups + slice : row * count;
            const T* values = x + base;
            T* codes = y + base;
            int64_t sum = 0;
            double square_sum = 0.;
            if (channels_last) {
                for (int64_t plane = 0; plane < span; ++plane) {
                    const T* run = values + plane * channels;
                    for (int64_t c = 0; c < spans; ++c) {
                        sum += run[c];
                        square_sum += static_cast<double>(run[c]) * run[c];
                    }
                }
            } else if constexpr (sizeof(T) == 1) {
                sum = byte_sum(values, count);
                square_sum = static_cast<double>(byte_square_sum(values, count));
            } else {
                for (int64_t k = 0; k < count; ++k) {
                    sum += values[k];
                    square_sum += static_cast<double>(values[k]) * values[k];
                }
            }
            const float mean_code = static_cast<float>(sum) / count;
            const float mean_div_scale = mean_code - shift;
            const float variance = std::max(static_cast<float>(square_sum) / count
                                            - mean_code * mean_code, 0.f);
            const float scale_div_std = x_scale / std::sqrt(variance * x_scale * x_scale + eps);
            if (channels_last) {
                for (int64_t plane = 0; plane < span; ++plane) {
                    const T* run = values + plane * channels;
                    T* write = codes + plane * channels;
                    for (int64_t c = 0; c < spans; ++c) {
                        const int64_t channel = affine_per_channel ? slice + c
                                                                   : plane * spans + c;
                        const float code = static_cast<float>(run[c]) - shift;
                        const float value = (code - mean_div_scale)
                            * (scale_div_std * (gamma ? gamma[channel] : 1.f))
                            + (beta ? beta[channel] : 0.f);
                        write[c] = quant::encode<T>(value, inverse, y_zp);
                    }
                }
                continue;
            }
            if (affine_per_channel) {
                for (int64_t part = 0; part < spans; ++part) {
                    const int64_t channel = (row % channel_groups) * spans + part;
                    normalize_run<Q>(values + part * span, codes + part * span, span, shift,
                        mean_div_scale, scale_div_std * (gamma ? gamma[channel] : 1.f),
                        beta ? beta[channel] : 0.f, y_scale, inverse, y_zp);
                }
                continue;
            }
            const FV one(1.f), zp(shift), mean(mean_div_scale), gradient(scale_div_std), zero(0.f);
            int64_t k = 0;
            for (; k + QV::size() <= count; k += QV::size()) {
                auto normalized = QV::loadu(values + k).dequantize(one, zp);
                for (size_t lane = 0; lane < normalized.size(); ++lane) {
                    const int64_t at = k + static_cast<int64_t>(lane) * FV::size();
                    const FV weight = gamma ? FV::loadu(gamma + at) : one;
                    const FV offset = beta ? FV::loadu(beta + at) : zero;
                    normalized[lane] = ((normalized[lane] - mean) * gradient) * weight + offset;
                }
                QV::quantize(normalized, y_scale, static_cast<int32_t>(y_zp), inverse)
                    .store(codes + k);
            }
            for (; k < count; ++k) {
                const float code = static_cast<float>(values[k]) - shift;
                const float value = ((code - mean_div_scale) * scale_div_std) *
                    (gamma ? gamma[k] : 1.f) + (beta ? beta[k] : 0.f);
                codes[k] = quant::encode<T>(value, inverse, y_zp);
            }
        }
    });
}

void normalize_impl(const void* x, const float* gamma, const float* beta, void* y, DType dtype,
                    int64_t count, int64_t rows, bool affine_per_channel, bool channels_last,
                    int64_t channels, int64_t channel_groups, float x_scale, int64_t x_zp,
                    float y_scale, int64_t y_zp, double eps) {
    switch (dtype) {
        case DType::QInt8: normalize_codes<qint8>(x,gamma,beta,y,count,rows,affine_per_channel,
            channels_last,channels,channel_groups,x_scale,x_zp,y_scale,y_zp,eps); break;
        case DType::QUInt8: normalize_codes<quint8>(x,gamma,beta,y,count,rows,affine_per_channel,
            channels_last,channels,channel_groups,x_scale,x_zp,y_scale,y_zp,eps); break;
        case DType::QInt32: normalize_codes<qint32>(x,gamma,beta,y,count,rows,affine_per_channel,
            channels_last,channels,channel_groups,x_scale,x_zp,y_scale,y_zp,eps); break;
        default: break;
    }
}

template<class T>
void pool_channels_last_codes(const void* input, void* output, int64_t first, int64_t last,
                             const quant::PoolGeometry& g) {
    using IV = vec::Vectorized<int32_t>;
    using FV = vec::Vectorized<float>;
    constexpr int lanes = IV::size(), block = 4 * lanes;
    const auto* x = static_cast<const T*>(input);
    auto* y = static_cast<T*>(output);
    const FV zero(static_cast<float>(g.zp)), lo(static_cast<float>(std::numeric_limits<T>::lowest())),
             hi(static_cast<float>(std::numeric_limits<T>::max()));
    for (int64_t location = first; location < last; ++location) {
        int64_t p = location % g.out_size, count = 1, valid = 1;
        int64_t start[3]{}, end[3]{};
        for (int d = 2; d >= 0; --d) {
            const int64_t o = p % g.out[d]; p /= g.out[d];
            if (g.mode == 1) {
                start[d] = o * g.in[d] / g.out[d];
                end[d] = ((o + 1) * g.in[d] + g.out[d] - 1) / g.out[d];
            } else {
                start[d] = o * g.stride[d] - g.pad[d];
                end[d] = start[d] + g.kernel[d];
                count *= std::min(end[d], g.in[d] + g.pad[d]) - start[d];
            }
            start[d] = std::max<int64_t>(start[d], 0);
            end[d] = std::min(end[d], g.in[d]);
            valid *= end[d] - start[d];
        }
        const int64_t divisor = g.divisor ? g.divisor : (g.mode == 1 || !g.include_pad ? valid : count);
        const FV inverse(divisor ? 1.f / divisor : 0.f);
        const int32_t initial = static_cast<int32_t>(-g.zp * valid);
        const auto* row = x + location / g.out_size * g.batch_stride;
        auto* dst = y + location * g.channels;
        int64_t channel = 0;
        for (; channel + lanes <= g.channels;) {
            const int vectors = static_cast<int>(std::min<int64_t>(4, (g.channels - channel) / lanes));
            IV sums[4]{IV(initial), IV(initial), IV(initial), IV(initial)};
            for (int64_t d = start[0]; d < end[0]; ++d)
                for (int64_t h = start[1]; h < end[1]; ++h)
                    for (int64_t w = start[2]; w < end[2]; ++w) {
                        const auto* src = row + d * g.spatial_strides[0] + h * g.spatial_strides[1] +
                                          w * g.spatial_strides[2] + channel;
                        for (int v = 0; v < vectors; ++v) {
#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
                            sums[v] = sums[v] + vec::convert_to_int32<T>(src + v * lanes);
#else
                            alignas(64) int32_t values[lanes];
                            for (int lane = 0; lane < lanes; ++lane) values[lane] = src[v * lanes + lane];
                            sums[v] = sums[v] + IV::loadu(values);
#endif
                        }
                    }
            alignas(64) float codes[block];
            for (int v = 0; v < vectors; ++v)
                vec::clamp_max(vec::clamp_min((vec::convert<float>(sums[v]) * inverse + zero).round(), lo), hi).store(codes + v * lanes);
            for (int c = 0; c < vectors * lanes; ++c) dst[channel + c] = static_cast<T>(codes[c]);
            channel += vectors * lanes;
        }
        for (; channel < g.channels; ++channel) {
            int32_t sum = initial;
            for (int64_t d = start[0]; d < end[0]; ++d)
                for (int64_t h = start[1]; h < end[1]; ++h)
                    for (int64_t w = start[2]; w < end[2]; ++w)
                        sum += row[d * g.spatial_strides[0] + h * g.spatial_strides[1] + w * g.spatial_strides[2] + channel];
            const float code = divisor ? std::nearbyint(static_cast<float>(sum) * (1.f / divisor) + static_cast<float>(g.zp)) : g.zp;
            dst[channel] = static_cast<T>(std::clamp(code, static_cast<float>(std::numeric_limits<T>::lowest()),
                                                   static_cast<float>(std::numeric_limits<T>::max())));
        }
    }
}

// The largest code of a window is the largest value it stands for, so a maximum
// window never leaves integer space: no scale, no rounding, no clamp beyond the
// storage type. Channels sit next to each other in a channel-innermost tensor, so
// a window's taps are strided vectors and the running maximum is one instruction
// per tap; a pixel covered only by padding keeps the lowest code.
template<class T>
void pool_max_channels_last_codes(const void* input, void* output, int64_t first, int64_t last,
                                  const quant::PoolGeometry& g) {
    using QV = vec::Vectorized<T>;
    constexpr int lanes = QV::size();
    const auto* x = static_cast<const T*>(input);
    auto* y = static_cast<T*>(output);
    const QV lowest(std::numeric_limits<T>::lowest());
    for (int64_t location = first; location < last; ++location) {
        int64_t start[3]{}, end[3]{1,1,1};
        int64_t p = location % g.out_size;
        for (int d = 2; d >= 0; --d) {
            const int64_t o = p % g.out[d]; p /= g.out[d];
            start[d] = o * g.stride[d] - g.pad[d];
            end[d] = start[d] + (g.kernel[d] - 1) * g.dilation[d] + 1;
            while (start[d] < 0) start[d] += g.dilation[d];
            end[d] = std::min(end[d], g.in[d]);
        }
        const T* row = x + location / g.out_size * g.batch_stride;
        T* dst = y + location * g.channels;
        int64_t channel = 0;
        for (; channel + lanes <= g.channels; channel += lanes) {
            QV largest = lowest;
            for (int64_t d = start[0]; d < end[0]; d += g.dilation[0])
                for (int64_t h = start[1]; h < end[1]; h += g.dilation[1])
                    for (int64_t w = start[2]; w < end[2]; w += g.dilation[2])
                        largest = vec::maximum(largest, QV::loadu(row +
                            d * g.spatial_strides[0] + h * g.spatial_strides[1] +
                            w * g.spatial_strides[2] + channel));
            largest.store(dst + channel);
        }
        for (; channel < g.channels; ++channel) {
            T largest = std::numeric_limits<T>::lowest();
            for (int64_t d = start[0]; d < end[0]; d += g.dilation[0])
                for (int64_t h = start[1]; h < end[1]; h += g.dilation[1])
                    for (int64_t w = start[2]; w < end[2]; w += g.dilation[2]) {
                        const T code = row[d * g.spatial_strides[0] + h * g.spatial_strides[1] +
                                           w * g.spatial_strides[2] + channel];
                        if (code > largest) largest = code;
                    }
            dst[channel] = largest;
        }
    }
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif
template<class A, class Load>
std::array<A, 4> spatial_cascade(int64_t count, Load load) {
    int power = 0;
    for (uint64_t n = count > 1 ? static_cast<uint64_t>(count - 1) : 0; n; n >>= 1) ++power;
    power = std::max(4, power / 4);
    const int64_t step = int64_t{1} << power;
    std::array<std::array<A, 4>, 4> sums;
    for (auto& level : sums) level.fill(A(0.f));
    int64_t i = 0;
    for (; i + step <= count;) {
        for (int64_t j = 0; j < step; ++j, ++i) {
            #pragma GCC unroll 4
            for (int row = 0; row < 4; ++row) sums[0][row] = sums[0][row] + load(i, row);
        }
        for (int level = 1; level < 4; ++level) {
            #pragma GCC unroll 4
            for (int row = 0; row < 4; ++row) {
                sums[level][row] = sums[level][row] + sums[level - 1][row];
                sums[level - 1][row] = A(0.f);
            }
            if (static_cast<uint64_t>(i) & (static_cast<uint64_t>(step - 1) << (level * power))) break;
        }
    }
    for (; i < count; ++i) {
        #pragma GCC unroll 4
        for (int row = 0; row < 4; ++row) sums[0][row] = sums[0][row] + load(i, row);
    }
    for (int level = 1; level < 4; ++level) {
        #pragma GCC unroll 4
        for (int row = 0; row < 4; ++row) sums[0][row] = sums[0][row] + sums[level][row];
    }
    return sums[0];
}

template<class T>
void spatial_mean_codes(const void* input, void* output, int64_t count, int64_t channels,
                        int64_t begin, int64_t end, float scale, int64_t zp) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    using V = vec::Vectorized<float>;
    using IV = vec::Vectorized<int32_t>;
    constexpr int lanes = V::size();
    const auto* x = static_cast<const T*>(input);
    auto* y = static_cast<T*>(output);
    const V grid_scale(scale), zero(static_cast<float>(zp)), divisor(static_cast<float>(count)), inverse(1.f / scale);
    const V lower(static_cast<float>(std::numeric_limits<T>::lowest())), upper(static_cast<float>(std::numeric_limits<T>::max()));
    auto load = [&](int64_t spatial, int64_t channel) {
        const T* src = x + spatial * channels + channel;
#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
        const IV codes = vec::convert_to_int32<T>(src);
#else
        alignas(64) int32_t values[IV::size()];
        for (int lane = 0; lane < lanes; ++lane) values[lane] = src[lane];
        const IV codes = IV::loadu(values);
#endif
        return (vec::convert<float>(codes) - zero) * grid_scale;
    };
    auto store = [&](int64_t channel, V total) {
        alignas(64) float values[lanes];
        vec::clamp_min(vec::clamp_max(((total / divisor) * inverse + zero).round(), upper), lower).store(values);
        for (int lane = 0; lane < lanes; ++lane) y[channel + lane] = static_cast<T>(values[lane]);
    };
    auto scalar_load = [&](int64_t spatial, int64_t channel) {
        return (static_cast<float>(x[spatial * channels + channel]) - static_cast<float>(zp)) * scale;
    };
    auto scalar_store = [&](int64_t channel, float total) {
        const float code = std::nearbyint((total / static_cast<float>(count)) * (1.f / scale) + static_cast<float>(zp));
        y[channel] = static_cast<T>(std::fmax(static_cast<float>(std::numeric_limits<T>::lowest()),
            std::fmin(static_cast<float>(std::numeric_limits<T>::max()), code)));
    };
    int64_t channel = begin;
    if (end - begin < lanes) {
        for (; channel + 4 <= end; channel += 4) {
            const auto totals = spatial_cascade<float>(count, [&](int64_t i, int row) { return scalar_load(i, channel + row); });
            for (int row = 0; row < 4; ++row) scalar_store(channel + row, totals[row]);
        }
    }
    for (; channel + 4 * lanes <= end; channel += 4 * lanes) {
        const auto totals = spatial_cascade<V>(count, [&](int64_t i, int row) { return load(i, channel + row * lanes); });
        for (int row = 0; row < 4; ++row) store(channel + row * lanes, totals[row]);
    }
    for (; channel + lanes <= end; channel += lanes) {
        auto totals = spatial_cascade<V>(count / 4, [&](int64_t i, int row) { return load(i * 4 + row, channel); });
        for (int64_t i = count / 4 * 4; i < count; ++i) totals[0] = totals[0] + load(i, channel);
        for (int row = 1; row < 4; ++row) totals[0] = totals[0] + totals[row];
        store(channel, totals[0]);
    }
    for (; channel < end; ++channel) {
        auto totals = spatial_cascade<float>(count / 4, [&](int64_t i, int row) { return scalar_load(i * 4 + row, channel); });
        for (int64_t i = count / 4 * 4; i < count; ++i) totals[0] += scalar_load(i, channel);
        for (int row = 1; row < 4; ++row) totals[0] += totals[row];
        scalar_store(channel, totals[0]);
    }
}

void spatial_mean_impl(const void* input, void* output, int64_t count, int64_t channels,
                       int64_t begin, int64_t end, DType dtype, float scale, int64_t zp) {
    if (dtype == DType::QInt8) spatial_mean_codes<int8_t>(input, output, count, channels, begin, end, scale, zp);
    else spatial_mean_codes<uint8_t>(input, output, count, channels, begin, end, scale, zp);
}

template<class T>
void pool_spatial_codes(const void* input, void* output, int64_t first, int64_t last,
                        const quant::PoolGeometry& g, const quant::PoolWindow* windows) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    using V = vec::Vectorized<float>;
    const auto* x = static_cast<const T*>(input);
    auto* y = static_cast<T*>(output);
    const bool raw_adaptive = g.mode == 1 && !g.channels_last;
    const V zero(raw_adaptive ? 0.f : static_cast<float>(g.zp));
    for (int64_t outer = first; outer < last; ++outer) {
        const int64_t batch = outer / g.channels, channel = outer % g.channels;
        const T* row = x + batch * g.batch_stride + channel * g.channel_stride;
        for (int64_t location = 0; location < g.out_size; location += V::size()) {
            const int lanes = static_cast<int>(std::min<int64_t>(V::size(), g.out_size - location));
            alignas(64) float sums[V::size()]{}, inverses[V::size()]{}, result[V::size()];
            for (int lane = 0; lane < lanes; ++lane) {
                const auto& window = windows[location + lane];
                int64_t raw_sum = 0;
                uint32_t wrapped_sum = window.initial;
                for (int64_t d = window.start[0]; d < window.end[0]; ++d)
                    for (int64_t h = window.start[1]; h < window.end[1]; ++h) {
                        const T* src = row + d * g.spatial_strides[0] + h * g.spatial_strides[1];
                        for (int64_t w = window.start[2]; w < window.end[2]; ++w) {
                            const T code = src[w * g.spatial_strides[2]];
                            if (raw_adaptive) raw_sum += code;
                            else wrapped_sum += static_cast<uint32_t>(code);
                        }
                    }
                sums[lane] = raw_adaptive ? static_cast<float>(raw_sum)
                                         : static_cast<float>(static_cast<int32_t>(wrapped_sum));
                inverses[lane] = window.inverse;
            }
            (V::loadu(sums) * V::loadu(inverses) + zero).round().store(result);
            for (int lane = 0; lane < lanes; ++lane) {
                const int64_t index = g.channels_last ? (batch * g.out_size + location + lane) * g.channels + channel
                                                     : outer * g.out_size + location + lane;
                if (raw_adaptive) {
                    if constexpr (sizeof(T) == 4)
                        y[index] = result[lane] >= 0x1p31f || result[lane] < -0x1p31f
                            ? std::numeric_limits<T>::lowest() : static_cast<T>(result[lane]);
                    else y[index] = static_cast<T>(result[lane]);
                } else y[index] = static_cast<T>(std::clamp(static_cast<double>(result[lane]),
                    static_cast<double>(std::numeric_limits<T>::lowest()), static_cast<double>(std::numeric_limits<T>::max())));
            }
        }
    }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

void pool_impl(const void* input, void* output, int64_t first, int64_t last, DType dtype,
               const quant::PoolGeometry& g, const quant::PoolWindow* windows) {
    if (windows) {
        switch (dtype) {
            case DType::QInt8: pool_spatial_codes<int8_t>(input,output,first,last,g,windows); break;
            case DType::QUInt8: pool_spatial_codes<uint8_t>(input,output,first,last,g,windows); break;
            case DType::QInt32: pool_spatial_codes<int32_t>(input,output,first,last,g,windows); break;
            default: break;
        }
        return;
    }
    if (g.mode == 2) {
        if (dtype == DType::QInt8)
            pool_max_channels_last_codes<int8_t>(input, output, first, last, g);
        else
            pool_max_channels_last_codes<uint8_t>(input, output, first, last, g);
        return;
    }
    if (dtype == DType::QInt8) pool_channels_last_codes<int8_t>(input, output, first, last, g);
    else pool_channels_last_codes<uint8_t>(input, output, first, last, g);
}

template<bool Hyperbolic>
void word_activation_codes(const int32_t* x, int32_t* y, int64_t size, float scale, int64_t zp,
                   double output_scale, int64_t output_zp, bool vectorize) {
    using QV = vec::Vectorized<qint32>;
    using FV = vec::Vectorized<float>;
    const FV input_scale(scale), input_zero(static_cast<float>(zp)), one(1.f);
    const FV scaled_zero = input_scale * input_zero.neg();
    const float inverse = 1.f / static_cast<float>(output_scale);
    const int64_t vector_end = vectorize ? size / (2 * QV::size()) * (2 * QV::size()) : 0;
    int64_t i = 0;
    for (; i < vector_end; i += QV::size()) {
        auto values = Hyperbolic ? QV::loadu(x + i).dequantize(input_scale, input_zero, scaled_zero)
                                 : QV::loadu(x + i).dequantize(input_scale, input_zero);
        for (auto& value : values) {
            if constexpr (Hyperbolic) value = value.tanh();
            else value = (one + value.neg().exp()).reciprocal();
        }
        alignas(64) float real[QV::size()];
        values[0].store(real);
        for (int lane = 0; lane < QV::size(); ++lane)
            y[i + lane] = quant::encode<int32_t>(real[lane], inverse, output_zp);
    }
    for (; i < size; ++i) {
        const float real = static_cast<float>((static_cast<double>(x[i]) - zp) * scale);
        const float value = Hyperbolic ? std::tanh(real) : static_cast<float>(1.0f / (1.0 + std::exp(-real)));
        y[i] = quant::encode<int32_t>(value, inverse, output_zp);
    }
}

template<class Q>
void prelu_codes(const void* input, const void* weight, void* output, int64_t size,
                 bool scalar_weight, int64_t xz, int64_t wz, int64_t yz, int64_t one, float multiplier) {
    using T = typename Q::underlying;
    using V = vec::Vectorized<Q>;
    using IV = vec::Vectorized<qint32>;
    const auto* x = static_cast<const T*>(input);
    const auto* w = static_cast<const T*>(weight);
    auto* y = static_cast<T*>(output);
    const V input_zero(Q(static_cast<T>(xz))), weight_zero(Q(static_cast<T>(wz)));
    const IV unit(qint32(static_cast<int32_t>(one)));
    const IV scalar_slope(qint32(scalar_weight ? static_cast<int32_t>(static_cast<int64_t>(w[0]) - wz) : 0));
    int64_t i = 0;
    for (; i + V::size() <= size; i += V::size()) {
        const V a = V::loadu(x + i);
        auto positive = a.maximum(input_zero).widening_subtract(input_zero);
        const auto negative = a.minimum(input_zero).widening_subtract(input_zero);
        if (scalar_weight) {
            for (size_t j = 0; j < positive.size(); ++j)
                positive[j] = positive[j] * unit + negative[j] * scalar_slope;
        } else {
            const auto slope = V::loadu(w + i).widening_subtract(weight_zero);
            for (size_t j = 0; j < positive.size(); ++j)
                positive[j] = positive[j] * unit + negative[j] * slope[j];
        }
        V::requantize_from_int(positive, multiplier, yz).store(y + i);
    }
    for (; i < size; ++i) {
        const int64_t a = static_cast<int64_t>(x[i]) - xz;
        const int64_t b = static_cast<int64_t>(w[scalar_weight ? 0 : i]) - wz;
        const int32_t value = static_cast<int32_t>(std::max<int64_t>(a, 0) * one + std::min<int64_t>(a, 0) * b);
        y[i] = quant::encode<T>(static_cast<float>(value), multiplier, yz);
    }
}

void prelu_impl(const void* x, const void* w, void* y, int64_t size, bool scalar_weight,
                DType dtype, int64_t xz, int64_t wz, int64_t yz, int64_t one, float multiplier) {
    switch (dtype) {
        case DType::QInt8: prelu_codes<qint8>(x,w,y,size,scalar_weight,xz,wz,yz,one,multiplier); break;
        case DType::QUInt8: prelu_codes<quint8>(x,w,y,size,scalar_weight,xz,wz,yz,one,multiplier); break;
        case DType::QInt32: prelu_codes<qint32>(x,w,y,size,scalar_weight,xz,wz,yz,one,multiplier); break;
        default: break;
    }
}

template<class T>
void float_prelu_codes(const void* input, const float* weight, uint8_t* y, int64_t size,
                      bool scalar_weight, int64_t zp, float scale, float inverse, int64_t output_zp) {
    using V = vec::Vectorized<float>;
    const auto* x = static_cast<const T*>(input);
    const V multiplier(scale), inv(inverse), zero(0.f), offset(static_cast<float>(output_zp));
    int64_t i = 0;
    for (; i + V::size() <= size; i += V::size()) {
        alignas(64) float values[V::size()], result[V::size()];
        for (int lane = 0; lane < V::size(); ++lane)
            values[lane] = static_cast<float>(static_cast<int64_t>(x[i + lane]) - zp);
        const V input_value = V::loadu(values) * multiplier;
        const V slope = scalar_weight ? V(weight[0]) : V::loadu(weight + i);
        const V value = V::blendv(input_value * slope, input_value, input_value > zero);
        const V rounded = (value * inv + offset).round();
        V::blendv(vec::clamp_max(vec::clamp_min(rounded, zero), V(255.f)), V(255.f),
                  rounded != rounded).store(result);
        for (int lane = 0; lane < V::size(); ++lane) y[i + lane] = static_cast<uint8_t>(result[lane]);
    }
    for (; i < size; ++i) {
        const float real = static_cast<float>(static_cast<int64_t>(x[i]) - zp) * scale;
        const float value = real > 0.f ? real : real * weight[scalar_weight ? 0 : i];
        y[i] = static_cast<uint8_t>(std::fmax(0.f, std::fmin(255.f,
            std::nearbyint(value * inverse + static_cast<float>(output_zp)))));
    }
}

void float_prelu_impl(const void* x, const float* w, uint8_t* y, int64_t size,
                      bool scalar_weight, DType dtype, int64_t zp, float scale, float inverse, int64_t output_zp) {
    switch (dtype) {
        case DType::QInt8: float_prelu_codes<int8_t>(x,w,y,size,scalar_weight,zp,scale,inverse,output_zp); break;
        case DType::QUInt8: float_prelu_codes<uint8_t>(x,w,y,size,scalar_weight,zp,scale,inverse,output_zp); break;
        case DType::QInt32: float_prelu_codes<int32_t>(x,w,y,size,scalar_weight,zp,scale,inverse,output_zp); break;
        default: break;
    }
}

void clip_impl(const void* x, void* y, int64_t size, DType dtype, int64_t lo, int64_t hi) {
    switch (dtype) {
        case DType::QInt8: clip<qint8>(x, y, size, lo, hi); break;
        case DType::QUInt8: clip<quint8>(x, y, size, lo, hi); break;
        case DType::QInt32: clip<qint32>(x, y, size, lo, hi); break;
        default: break;
    }
}

// The register gather and the buffer gather deduce to the same signature over a
// float table, and MSVC cannot order that pair. Spelling the element type at the
// call resolves it, but selects the buffer form on every compiler; naming the
// scale alone keeps the register form wherever the instruction exists.
inline vec::Vectorized<float> table_gather(const float* table,
        const vec::Vectorized<int32_t>& index) {
#ifdef _MSC_VER
    return vec::gather<4, float>(table, index);
#else
    return vec::gather<4>(table, index);
#endif
}

void lookup_impl(const uint8_t* x, uint8_t* y, int64_t size, const float* table) {
    using QV = vec::Vectorized<quint8>;
    using FV = vec::Vectorized<float>;
    const FV one(1.f), zero(0.f);
    int64_t i = 0;
    for (; i + QV::size() <= size; i += QV::size()) {
        auto values = QV::loadu(x + i).dequantize(one, zero);
        for (auto& v : values)
            v = table_gather(table, vec::convert_to_int_of_same_size<float>(v));
        QV::quantize(values, 1.f, 0, 1.f).store(y + i);
    }
    for (; i < size; ++i) y[i] = static_cast<uint8_t>(table[x[i]]);
}
// The operation and whether to clamp at zero are chosen once per call, so they
// enter as template parameters and the element loop holds no branch at all.
template<class Q, int Op, bool Relu>
void binary_codes(const void* aa, const void* bb, void* yy, int64_t size,
        double as, int64_t az, double bs, int64_t bz, double ys, int64_t yz) {
    using T = typename Q::underlying;
    using QV = vec::Vectorized<Q>;
    using FV = vec::Vectorized<float>;
    const T* a = static_cast<const T*>(aa); const T* b = static_cast<const T*>(bb); T* y = static_cast<T*>(yy);
    const FV asc(as), bsc(bs);
    const FV azf(static_cast<float>(az)), bzf(static_cast<float>(bz));
    // A code stands for scale*(code - zero_point), whose two terms share the
    // scale factor; folding it leaves one multiply-add per lane instead of a
    // subtraction and a multiply.
    const FV ashift(-static_cast<float>(as) * static_cast<float>(az));
    const FV bshift(-static_cast<float>(bs) * static_cast<float>(bz));
    const float inv_ys = 1.f / static_cast<float>(ys);
    const FV lower((std::numeric_limits<T>::lowest() - yz) * static_cast<float>(ys));
    const FV upper((std::numeric_limits<T>::max() - yz) * static_cast<float>(ys));
    // The grid's lowest code sits at or below zero, so a relu floor already
    // implies that bound and the two clamps collapse into one.
    const FV at_least = Relu ? FV(0.f) : lower;
    int64_t i = 0;
    for (; i + QV::size() <= size; i += QV::size()) {
        auto av = QV::loadu(a + i).dequantize(asc, azf, ashift);
        auto bv = QV::loadu(b + i).dequantize(bsc, bzf, bshift);
        for (size_t j = 0; j < av.size(); ++j) {
            if constexpr (Op == 0) av[j] = av[j] + bv[j];
            else if constexpr (Op == 1) av[j] = av[j] - bv[j];
            else if constexpr (Op == 2) av[j] = av[j] * bv[j];
            else av[j] = av[j] / bv[j];
            // Keeping a value inside the grid is the pair of compares that a
            // single instruction each already performs; the wider form that
            // also carries a NaN into the result spends an extra compare and
            // blend per bound, and a NaN is turned into a boundary code by the
            // conversion below regardless of which bound reached it.
            av[j] = vec::clamp_max(vec::clamp_min(av[j], at_least), upper);
        }
        QV::quantize(av, ys, yz, inv_ys).store(y + i);
    }
    for (; i < size; ++i) {
        const float av = static_cast<float>(static_cast<int64_t>(a[i]) - az) * static_cast<float>(as);
        const float bv = static_cast<float>(static_cast<int64_t>(b[i]) - bz) * static_cast<float>(bs);
        float v;
        if constexpr (Op == 0) v = av + bv;
        else if constexpr (Op == 1) v = av - bv;
        else if constexpr (Op == 2) v = av * bv;
        else v = av / bv;
        if constexpr (Relu) v = std::fmax(v, 0.f);
        y[i] = quant::encode<T>(v, 1.f / static_cast<float>(ys), yz);
    }
}

template<class Q, int Op>
void binary_relu(const void* a, const void* b, void* y, int64_t size,
        double as, int64_t az, double bs, int64_t bz, double ys, int64_t yz, bool relu) {
    if (relu) binary_codes<Q, Op, true>(a, b, y, size, as, az, bs, bz, ys, yz);
    else binary_codes<Q, Op, false>(a, b, y, size, as, az, bs, bz, ys, yz);
}

template<class Q>
void binary_typed(const void* a, const void* b, void* y, int64_t size,
        double as, int64_t az, double bs, int64_t bz, double ys, int64_t yz, int op, bool relu) {
    switch (op) {
        case 0: binary_relu<Q, 0>(a, b, y, size, as, az, bs, bz, ys, yz, relu); break;
        case 1: binary_relu<Q, 1>(a, b, y, size, as, az, bs, bz, ys, yz, relu); break;
        case 2: binary_relu<Q, 2>(a, b, y, size, as, az, bs, bz, ys, yz, relu); break;
        default: binary_relu<Q, 3>(a, b, y, size, as, az, bs, bz, ys, yz, relu); break;
    }
}

void binary_impl(const void* a, const void* b, void* y, int64_t size, DType dtype,
        double as, int64_t az, double bs, int64_t bz, double ys, int64_t yz, int op, bool relu) {
    if (dtype == DType::QInt8) binary_typed<qint8>(a,b,y,size,as,az,bs,bz,ys,yz,op,relu);
    else if (dtype == DType::QUInt8) binary_typed<quint8>(a,b,y,size,as,az,bs,bz,ys,yz,op,relu);
}

template<class T>
void softmax_rows(const T* x, T* y, int64_t width, int64_t inner, int64_t first, int64_t last,
        int64_t zp, float scale, float inverse) {
    using V = vec::Vectorized<float>;
    constexpr int lanes = V::size();
    std::vector<float> exponentials;
    if (inner == 1) exponentials.resize(width);
    for (int64_t row = first; row < last; ++row) {
        const int64_t base = row / inner * width * inner + row % inner;
        int64_t maximum = std::numeric_limits<T>::lowest();
        for (int64_t c = 0; c < width; ++c) maximum = std::max<int64_t>(maximum, x[base + c * inner]);
        if (inner == 1) {
            V partial(0.f);
            alignas(64) float values[lanes];
            int64_t c = 0;
            for (; c + lanes <= width; c += lanes) {
                for (int lane = 0; lane < lanes; ++lane) values[lane] = static_cast<float>(x[base + c + lane]);
                const V exp = ((V::loadu(values) - V(static_cast<float>(maximum))) * V(scale)).exp();
                exp.store(exponentials.data() + c);
                partial += exp;
            }
            float sum = 0.f;
            partial.store(values);
            for (float value : values) sum += value;
            for (; c < width; ++c) {
                const float exp = ::expf(static_cast<float>(static_cast<int64_t>(x[base + c]) - maximum) * scale);
                exponentials[c] = exp;
                sum += exp;
            }
            const float normalization = 1.f / sum;
            int64_t output_end = 0;
#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
            if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
                using Q = std::conditional_t<std::is_same_v<T, int8_t>, qint8, quint8>;
                using QV = vec::Vectorized<Q>;
                c = 0;
                for (; c + QV::size() <= width; c += QV::size()) {
                    typename QV::float_vec_return_type normalized;
                    for (size_t part = 0; part < normalized.size(); ++part)
                        normalized[part] = V::loadu(exponentials.data() + c + part * lanes) * V(normalization);
                    QV::quantize(normalized, 1.f, static_cast<int32_t>(zp), inverse).store(y + base + c);
                }
                output_end = c;
            }
#endif
            for (c = output_end; c < width; ++c)
                y[base + c] = quant::encode<T>(exponentials[c] * normalization, inverse, zp);
        } else {
            float sum = 0.f;
            for (int64_t c = 0; c < width; ++c)
                sum += ::expf(static_cast<float>(static_cast<int64_t>(x[base + c * inner]) - maximum) * scale);
            for (int64_t c = 0; c < width; ++c) {
                const float value = ::expf(static_cast<float>(static_cast<int64_t>(x[base + c * inner]) - maximum) * scale) / sum;
                y[base + c * inner] = quant::encode<T>(value, inverse, zp);
            }
        }
    }
}

void softmax_impl(const void* x, void* y, DType dtype, int64_t width, int64_t inner,
        int64_t rows, int64_t zp, float scale, float inverse) {
    parallel::parallel_for(0, rows, 1, [&](int64_t first, int64_t last) {
        switch (dtype) {
            case DType::QInt8: softmax_rows(static_cast<const int8_t*>(x), static_cast<int8_t*>(y), width, inner, first, last, zp, scale, inverse); break;
            case DType::QUInt8: softmax_rows(static_cast<const uint8_t*>(x), static_cast<uint8_t*>(y), width, inner, first, last, zp, scale, inverse); break;
            case DType::QInt32: softmax_rows(static_cast<const int32_t*>(x), static_cast<int32_t*>(y), width, inner, first, last, zp, scale, inverse); break;
            default: break;
        }
    });
}

float rowwise_loss(const float* x, int64_t size, float lo, float hi, int bits) {
    lo = static_cast<float>(Half(lo));
    const float range = hi - lo, qmax = static_cast<float>((1 << bits) - 1);
    const float scale = range == 0.f ? 1.f : static_cast<float>(Half(range / qmax));
    const float inverse = scale == 0.f ? 1.f : 1.f / scale;
    float sum = 0.f;
    for (int64_t i = 0; i < size; ++i) {
        const float code = ::fmaxf(0.f, ::fminf(qmax, ::nearbyintf((x[i] - lo) * inverse)));
        const float error = x[i] - (code * scale + lo);
        sum += error * error;
    }
    return ::sqrtf(sum);
}

// The width of a row's codes is decided before any row is looked at, so it
// arrives as a template argument: the packing loops then divide a column count
// by a constant rather than by a register, and a row of whole bytes carries
// none of the shifting that only a sub-byte width needs.
template<int Bits>
void rowwise_pack_rows(const float* input, const float* ranges, uint8_t* output,
        int64_t rows, int64_t columns, int64_t width, int64_t bins,
        bool optimized, float ratio) {
    using V = vec::Vectorized<float>;
    constexpr int lanes = V::size();
    constexpr int values_per_byte = 8 / Bits;
    // Rounding cannot change while the rows are being packed, so it is read once
    // for the whole call instead of on every row.
    const bool nearest = std::fegetround() == FE_TONEAREST;
    parallel::parallel_for(0, rows, 1, [&](int64_t first, int64_t last) {
        for (int64_t row = first; row < last; ++row) {
            const float* src = input + row * columns;
            uint8_t* dst = output + row * width;
            float lo, hi;
            if (ranges && !optimized) {
                lo = ranges[2 * row]; hi = ranges[2 * row + 1];
            } else if (nearest && columns >= lanes) {
                // A single accumulator waits for its own previous step, so the row
                // is folded by four independent chains that are merged at the end.
                // A bound only moves toward the value that overtakes it, which is
                // what a compare-and-select does in one instruction; the form that
                // also carries a NaN through the result costs two more per step to
                // preserve a value that the cross-lane fold below drops anyway.
                const float infinity = std::numeric_limits<float>::infinity();
                V low_a(infinity), low_b(infinity), low_c(infinity), low_d(infinity);
                V high_a(-infinity), high_b(-infinity), high_c(-infinity), high_d(-infinity);
                int64_t col = 0;
                for (; col + 4 * lanes <= columns; col += 4 * lanes) {
                    const V a = V::loadu(src + col), b = V::loadu(src + col + lanes);
                    const V c = V::loadu(src + col + 2 * lanes), d = V::loadu(src + col + 3 * lanes);
                    low_a = vec::clamp_max(low_a, a); high_a = vec::clamp_min(high_a, a);
                    low_b = vec::clamp_max(low_b, b); high_b = vec::clamp_min(high_b, b);
                    low_c = vec::clamp_max(low_c, c); high_c = vec::clamp_min(high_c, c);
                    low_d = vec::clamp_max(low_d, d); high_d = vec::clamp_min(high_d, d);
                }
                for (; col + lanes <= columns; col += lanes) {
                    const V values = V::loadu(src + col);
                    low_a = vec::clamp_max(low_a, values);
                    high_a = vec::clamp_min(high_a, values);
                }
                const V minimum = vec::clamp_max(vec::clamp_max(low_a, low_b), vec::clamp_max(low_c, low_d));
                const V maximum = vec::clamp_min(vec::clamp_min(high_a, high_b), vec::clamp_min(high_c, high_d));
                // The scale of a row is derived from these two values, so a store
                // followed by eight lane reads would put a memory round trip in
                // front of every code of the row. A narrow row offers no other
                // work to fill that gap, so the fold stays in registers.
                lo = minimum.reduce_min();
                hi = maximum.reduce_max();
                for (; col < columns; ++col) { lo = std::min(lo, src[col]); hi = std::max(hi, src[col]); }
            } else {
                lo = src[0]; hi = src[0];
                for (int64_t col = 1; col < columns; ++col) {
                    lo = ::fminf(lo, src[col]); hi = ::fmaxf(hi, src[col]);
                }
            }
            if (optimized && Bits != 8 && hi > lo) {
                const float step = (hi - lo) / static_cast<float>(bins);
                const float threshold = static_cast<float>(bins) * (1.f - ratio) * step;
                float best = rowwise_loss(src, columns, lo, hi, Bits), current = best;
                float left = lo, right = hi;
                while (left + threshold < right) {
                    const float loss_left = rowwise_loss(src, columns, left + step, right, Bits);
                    const float loss_right = rowwise_loss(src, columns, left, right - step, Bits);
                    if (current < loss_left && current < loss_right && current < best) {
                        best = current; lo = left; hi = right;
                    }
                    if (loss_left < loss_right) {
                        if (left + step == left) break;
                        left += step; current = loss_left;
                    } else {
                        if (right - step == right) break;
                        right -= step; current = loss_right;
                    }
                }
            }
            float scale, inverse;
            if (Bits == 8) {
                scale = (hi - lo) / 255.f;
                inverse = 255.f / (hi - lo + 1.e-8f);
            } else {
                lo = static_cast<float>(Half(lo));
                scale = hi == lo ? 1.f : static_cast<float>(Half((hi - lo) / ((1 << Bits) - 1)));
                inverse = scale == 0.f ? 1.f : 1.f / scale;
                if (scale == 0.f || std::isinf(inverse)) scale = inverse = 1.f;
            }
            int64_t col = 0;
            if (Bits == 8 && nearest) {
                // One byte per value asks for no decision per lane: a whole vector
                // of codes leaves through a single byte store instead of being
                // shifted into place one lane at a time.
                using QV = vec::Vectorized<quint8>;
                for (; col + QV::size() <= columns; col += QV::size()) {
                    typename QV::float_vec_return_type shifted;
                    for (size_t part = 0; part < shifted.size(); ++part)
                        shifted[part] = (V::loadu(src + col + part * lanes) - V(lo)) * V(inverse);
                    QV::quantize(shifted, 1.f, 0, 1.f).store(dst + col);
                }
            }
            if (nearest && lanes >= values_per_byte && lanes % values_per_byte == 0) {
                alignas(64) float codes[lanes];
#if defined(CPU_CAPABILITY_AVX2)
                if (Bits == 4) {
                    for (; col + 2 * lanes <= columns; col += 2 * lanes)
                        pack_row_chunk<4>(src, col, lo, inverse, dst + col / 2);
                } else if (Bits == 2) {
                    for (; col + 2 * lanes <= columns; col += 2 * lanes)
                        pack_row_chunk<2>(src, col, lo, inverse, dst + col / 4);
                }
#endif
                for (; col + lanes <= columns; col += lanes) {
                    V values = ((V::loadu(src + col) - V(lo)) * V(inverse)).round();
                    values = vec::clamp_max(vec::clamp_min(values, V(0.f)), V(static_cast<float>((1 << Bits) - 1)));
                    values.store(codes);
                    for (int lane = 0; lane < lanes; lane += values_per_byte) {
                        uint8_t packed = 0;
                        for (int v = 0; v < values_per_byte; ++v)
                            packed |= static_cast<uint8_t>(codes[lane + v]) << (v * Bits);
                        dst[(col + lane) / values_per_byte] = packed;
                    }
                }
            }
            for (; col < columns; col += values_per_byte) {
                uint8_t packed = 0;
                for (int v = 0; v < values_per_byte; ++v) {
                    const float code = ::nearbyintf((src[col + v] - lo) * inverse);
                    const uint8_t quantized = static_cast<uint8_t>(::fmaxf(0.f, ::fminf((1 << Bits) - 1, code)));
                    packed |= quantized << (v * Bits);
                }
                dst[col / values_per_byte] = packed;
            }
            if (Bits == 8) {
                quant::put_float(dst + width - 8, scale); quant::put_float(dst + width - 4, lo);
            } else {
                quant::put_half(dst + width - 4, scale); quant::put_half(dst + width - 2, lo);
            }
        }
    });
}

void rowwise_pack_impl(const float* input, const float* ranges, uint8_t* output,
        int64_t rows, int64_t columns, int64_t width, int64_t bins, int bits,
        bool optimized, float ratio) {
    switch (bits) {
        case 8:
            rowwise_pack_rows<8>(input, ranges, output, rows, columns, width, bins, optimized, ratio);
            break;
        case 4:
            rowwise_pack_rows<4>(input, ranges, output, rows, columns, width, bins, optimized, ratio);
            break;
        default:
            rowwise_pack_rows<2>(input, ranges, output, rows, columns, width, bins, optimized, ratio);
            break;
    }
}

void fp16_linear_impl(const float* x, const Half* w, const float* bias, float* y,
        int64_t first, int64_t last, int64_t n, int64_t k, bool relu) {
    using FV = vec::Vectorized<float>;
    using HV = vec::Vectorized<Half>;
    constexpr int width = FV::size();
    static_assert(HV::size() == 2 * width);
    for (int64_t i = first; i < last; ++i) {
        const float* row = x + i / n * k; const Half* weight = w + i % n * k;
        FV sum0(0.f), sum1(0.f); int64_t j = 0;
        for (; j + HV::size() <= k; j += HV::size()) {
            const auto values = vec::convert<float,2>(vec::VectorizedN<Half,1>(HV::loadu(weight + j)));
            sum0 = vec::fmadd(FV::loadu(row + j), values[0], sum0);
            sum1 = vec::fmadd(FV::loadu(row + j + width), values[1], sum1);
        }
        alignas(64) float lanes[width]; (sum0 + sum1).store(lanes);
        float sum = 0.f;
        for (int lane = 0; lane < width; ++lane) sum += lanes[lane];
        for (; j < k; ++j) sum += row[j] * static_cast<float>(weight[j]);
        if (bias) sum += bias[i % n];
        y[i] = relu ? std::fmax(sum, 0.f) : sum;
    }
}

// A packed row keeps its codes in bytes, nibbles or pairs of bits, so a block of
// them can be widened straight into float lanes: a byte is read as it stands, a
// nibble stream splits into the two halves of its bytes, and a bit stream into the
// four shifts of its words.
template <int Bits>
inline vec::Vectorized<float> row_codes(const uint8_t* row, int64_t first) {
    using FV = vec::Vectorized<float>;
    constexpr int lanes = FV::size();
#if defined(CPU_CAPABILITY_AVX2)
    if constexpr (Bits == 8) {
        const auto bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + first));
        return FV(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes)));
    } else if constexpr (Bits == 4) {
        int32_t stream_words;
        std::memcpy(&stream_words, row + first / 2, sizeof(stream_words));
        const auto stream = _mm_cvtsi32_si128(stream_words);
        const auto low = _mm_and_si128(stream, _mm_set1_epi8(0x0F));
        const auto high = _mm_and_si128(_mm_srli_epi16(stream, 4), _mm_set1_epi8(0x0F));
        return FV(_mm256_cvtepi32_ps(
            _mm256_cvtepu8_epi32(_mm_unpacklo_epi8(low, high))));
    } else {
        int32_t stream_words;
        std::memcpy(&stream_words, row + first / 4, sizeof(stream_words));
        const auto stream = _mm_cvtsi32_si128(stream_words);
        const auto mask = _mm_set1_epi8(0x03);
        const auto pair = _mm_unpacklo_epi8(_mm_and_si128(stream, mask),
            _mm_and_si128(_mm_srli_epi16(stream, 2), mask));
        const auto quad = _mm_unpacklo_epi8(_mm_and_si128(_mm_srli_epi16(stream, 4), mask),
            _mm_and_si128(_mm_srli_epi16(stream, 6), mask));
        return FV(_mm256_cvtepi32_ps(
            _mm256_cvtepu8_epi32(_mm_unpacklo_epi16(pair, quad))));
    }
#else
    alignas(64) float codes[lanes];
    for (int c = 0; c < lanes; ++c) codes[c] = static_cast<float>(quant::row_code(row, first + c, Bits));
    return FV::loadu(codes);
#endif
}

// Two codes share a byte, so sixteen bytes carry thirty-two of them and one split
// serves four vectors: taking a byte apart into its low and its high nibble gives
// every code a byte of its own, and interleaving the two streams puts the codes
// back in the order of their columns because the low nibble of a byte is its
// earlier column.
template <int Groups>
inline void nibble_codes(const uint8_t* row, int64_t first, vec::Vectorized<float>* codes) {
    using FV = vec::Vectorized<float>;
    constexpr int lanes = FV::size();
    for (int g = 0; g < Groups; ++g) {
#if defined(CPU_CAPABILITY_AVX2)
        const auto stream = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(row + (first + g * 4 * lanes) / 2));
        const auto low = _mm_and_si128(stream, _mm_set1_epi8(0x0F));
        const auto high = _mm_and_si128(_mm_srli_epi16(stream, 4), _mm_set1_epi8(0x0F));
        const auto lead = _mm_unpacklo_epi8(low, high);
        const auto after = _mm_unpackhi_epi8(low, high);
        codes[g * 4 + 0] = FV(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lead)));
        codes[g * 4 + 1] = FV(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lead, 8))));
        codes[g * 4 + 2] = FV(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(after)));
        codes[g * 4 + 3] = FV(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(after, 8))));
#else
        float unpacked[4 * lanes];
        for (int c = 0; c < 4 * lanes; ++c) {
            unpacked[c] = static_cast<float>(
                quant::row_code(row, first + g * 4 * lanes + c, 4));
        }
        for (int t = 0; t < 4; ++t) codes[g * 4 + t] = FV::loadu(unpacked + t * lanes);
#endif
    }
}

// A run of blocks, one per column vector, holds its sums in registers while the
// rows of the bag pass through: a block count fixed here is what keeps the
// accumulators out of memory, and every column adds its own rows in order
// whichever way the blocks are gathered.
template <int Bits, int Blocks>
__attribute__((always_inline)) inline void hold_blocks(const uint8_t* const* rows,
        const float* gradients, const float* shifts, int held, int64_t columns, int64_t col,
        float* y, int64_t bag, bool carried) {
    using FV = vec::Vectorized<float>;
    constexpr int lanes = FV::size();
    FV sums[Blocks];
    for (int t = 0; t < Blocks; ++t)
        sums[t] = carried ? FV::loadu(y + bag * columns + col + t * lanes) : FV(0.f);
    for (int k = 0; k < held; ++k) {
        const FV gradient(gradients[k]), shift(shifts[k]);
        if constexpr (Bits == 4 && Blocks >= 4) {
            FV codes[Blocks];
            nibble_codes<Blocks / 4>(rows[k], col, codes);
            for (int t = 0; t < Blocks; ++t)
                sums[t] = vec::fmadd(codes[t], gradient, sums[t] + shift);
        } else {
            for (int t = 0; t < Blocks; ++t)
                sums[t] = vec::fmadd(row_codes<Bits>(rows[k], col + t * lanes), gradient, sums[t] + shift);
        }
    }
    for (int t = 0; t < Blocks; ++t) sums[t].store(y + bag * columns + col + t * lanes);
}

template <int Bits>
void embedding_rows(const uint8_t* w, const int64_t* indices, const int64_t* offsets,
        const int64_t* mapping, const float* weights, float* y, int64_t first_bag, int64_t last_bag,
        int64_t bags, int64_t count, int64_t fixed, int64_t columns, int64_t width, bool embedding) {
    using FV = vec::Vectorized<float>;
    constexpr int lanes = FV::size();
    constexpr int held_rows = 32;
    // A row's grid and weight hold for every column of it, and reading them apart
    // from the codes costs four byte loads each, so a run of rows is resolved once
    // into arrays that the column walk only reads.
    const uint8_t* rows[held_rows];
    float gradients[held_rows], shifts[held_rows];
    alignas(64) float codes[lanes]{};
    for (int64_t bag = first_bag; bag < last_bag; ++bag) {
        const int64_t first = embedding ? bag : (offsets ? offsets[bag] : bag * fixed);
        const int64_t last = embedding ? bag + 1 : (offsets ? (bag + 1 < bags ? offsets[bag+1] : count) : first + fixed);
        if (first >= last) {
            // A bag that gathers no row still owns an output row, and it holds zeros.
            for (int64_t col = 0; col < columns; col += lanes)
                FV(0.f).store(y + bag * columns + col,
                              static_cast<int>(std::min<int64_t>(lanes, columns - col)));
            continue;
        }
        for (int64_t start = first; start < last; start += held_rows) {
            const int64_t stop = std::min(start + held_rows, last);
            const bool carried = start > first;
            int held = 0;
            for (int64_t j = start; j < stop; ++j) {
                const int64_t row_index = mapping ? mapping[indices[j]] : indices[j];
                if (row_index == -1) continue;
                const uint8_t* row = w + row_index * width;
                float scale, bias;
                quant::row_grid(row, width, Bits, scale, bias);
                const float weight = weights ? weights[j] : 1.f;
                rows[held] = row;
                gradients[held] = scale * weight;
                shifts[held] = bias * weight;
                ++held;
            }
            int64_t col = 0;
            for (; col + 8 * lanes <= columns; col += 8 * lanes)
                hold_blocks<Bits, 8>(rows, gradients, shifts, held, columns, col, y, bag, carried);
            for (; col + 4 * lanes <= columns; col += 4 * lanes)
                hold_blocks<Bits, 4>(rows, gradients, shifts, held, columns, col, y, bag, carried);
            for (; col + 2 * lanes <= columns; col += 2 * lanes)
                hold_blocks<Bits, 2>(rows, gradients, shifts, held, columns, col, y, bag, carried);
            for (; col + lanes <= columns; col += lanes)
                hold_blocks<Bits, 1>(rows, gradients, shifts, held, columns, col, y, bag, carried);
            if (col < columns) {
                const int take = static_cast<int>(columns - col);
                float tail[lanes]{};
                if (carried) for (int c = 0; c < take; ++c) tail[c] = y[bag * columns + col + c];
                FV sum = FV::loadu(tail);
                for (int k = 0; k < held; ++k) {
                    for (int c = 0; c < take; ++c) codes[c] = static_cast<float>(quant::row_code(rows[k], col + c, Bits));
                    sum = vec::fmadd(FV::loadu(codes), FV(gradients[k]), sum + FV(shifts[k]));
                }
                sum.store(y + bag * columns + col, take);
            }
        }
    }
}

void embedding_impl(const uint8_t* w, const int64_t* indices, const int64_t* offsets,
        const int64_t* mapping, const float* weights, float* y, int64_t first_bag, int64_t last_bag,
        int64_t bags, int64_t count, int64_t fixed, int64_t columns, int64_t width, int64_t bits, bool embedding) {
    if (bits == 8) embedding_rows<8>(w, indices, offsets, mapping, weights, y, first_bag, last_bag,
        bags, count, fixed, columns, width, embedding);
    else if (bits == 4) embedding_rows<4>(w, indices, offsets, mapping, weights, y, first_bag, last_bag,
        bags, count, fixed, columns, width, embedding);
    else embedding_rows<2>(w, indices, offsets, mapping, weights, y, first_bag, last_bag,
        bags, count, fixed, columns, width, embedding);
}

// Both affine grids hold for the whole call, so each of the 256 input codes has
// exactly one output code. Deciding them up front with the double to float path
// fixes the half way cases a float vector round could land on either side of,
// and turns every element of every row into one lookup.
void fill_cat_code_table(int8_t* table, int64_t input_zero_point, double input_scale,
        double output_scale, int64_t output_zero_point, bool relu) {
    const float inverse = 1.f / static_cast<float>(output_scale);
    const int8_t zero = static_cast<int8_t>(output_zero_point);
    for (int64_t i = 0; i < 256; ++i) {
        const int8_t code = static_cast<int8_t>(i - 128);
        const float value = static_cast<float>(
            (static_cast<double>(code) - input_zero_point) * input_scale);
        const float rounded = std::nearbyintf(value * inverse) +
            static_cast<float>(output_zero_point);
        int8_t entry = static_cast<int8_t>(std::clamp(
            static_cast<int64_t>(rounded), int64_t(-128), int64_t(127)));
        if (relu) entry = std::max(entry, zero);
        table[i] = entry;
    }
}

void cat_rows_impl(const int8_t* input, int8_t* output, int64_t rows,
        int64_t chunk, int64_t width, int64_t offset, double input_scale,
        int64_t input_zero_point, double output_scale, int64_t output_zero_point,
        bool same_qparams, bool relu) {
    using QV = vec::Vectorized<qint8>;
    const int8_t zero = static_cast<int8_t>(output_zero_point);
    const int64_t vector_end = chunk - chunk % QV::size();
    alignas(64) int8_t table[256];
    fill_cat_code_table(table, input_zero_point, input_scale, output_scale,
                        output_zero_point, relu);
    parallel::parallel_for(0, rows, 1, [&](int64_t first, int64_t last) {
        for (int64_t row = first; row < last; ++row) {
            const int8_t* src = input + row * chunk;
            int8_t* dst = output + row * width + offset;
            if (same_qparams) {
                int64_t col = 0;
                if (relu) {
                    const QV minimum{qint8(zero)};
                    for (; col < vector_end; col += QV::size())
                        QV::loadu(src + col).relu(minimum).store(dst + col);
                }
                if (!relu) {
                    std::memcpy(dst, src, static_cast<size_t>(chunk));
                } else {
                    for (; col < chunk; ++col) dst[col] = std::max(src[col], zero);
                }
                continue;
            }
            for (int64_t column = 0; column < chunk; ++column)
                dst[column] = table[static_cast<int64_t>(src[column]) + 128];
        }
    });
}

}
}

#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(quantized_subbyte_pack_stub, &subbyte_pack_impl);
REGISTER_DISPATCH(quantized_subbyte_unpack_stub, &subbyte_unpack_impl);
REGISTER_DISPATCH(quantized_unsigned_pack_stub, &unsigned_pack_impl);
REGISTER_DISPATCH(quantized_recurrent_step_stub, &recurrent_step_impl);
REGISTER_DISPATCH(quantized_dynamic_dot_stub, &dynamic_dot_impl);
REGISTER_DISPATCH(quantized_dynamic_block_stub, &dynamic_block_impl);
REGISTER_DISPATCH(quantized_linear_epilogue_stub, &linear_epilogue_impl);
REGISTER_DISPATCH(quantized_int4_pack_stub, &int4_pack_impl);
REGISTER_DISPATCH(quantized_int4_mm_stub, &int4_mm_impl);
REGISTER_DISPATCH(quantized_batch_norm_stub, &batch_norm_impl);
REGISTER_DISPATCH(quantized_bilinear_stub, &bilinear_impl);
REGISTER_DISPATCH(quantized_clip_stub, &clip_impl);
REGISTER_DISPATCH(quantized_prelu_stub, &prelu_impl);
REGISTER_DISPATCH(quantized_float_prelu_stub, &float_prelu_impl);
REGISTER_DISPATCH(quantized_sigmoid_stub, &word_activation_codes<false>);
REGISTER_DISPATCH(quantized_tanh_stub, &word_activation_codes<true>);
REGISTER_DISPATCH(quantized_moment_stub, &moment_impl);
REGISTER_DISPATCH(quantized_spatial_mean_stub, &spatial_mean_impl);
REGISTER_DISPATCH(quantized_pool_stub, &pool_impl);
REGISTER_DISPATCH(quantized_lookup_stub, &lookup_impl);
REGISTER_DISPATCH(quantized_softmax_stub, &softmax_impl);
REGISTER_DISPATCH(quantized_rowwise_pack_stub, &rowwise_pack_impl);
REGISTER_DISPATCH(quantized_binary_stub, &binary_impl);
REGISTER_DISPATCH(quantized_embedding_stub, &embedding_impl);
REGISTER_DISPATCH(quantized_fp16_linear_stub, &fp16_linear_impl);
REGISTER_DISPATCH(quantized_cat_rows_stub, &cat_rows_impl);
REGISTER_DISPATCH(quantized_normalize_stub, &normalize_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(quantized_subbyte_pack_stub, &subbyte_pack_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_subbyte_unpack_stub, &subbyte_unpack_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_unsigned_pack_stub, &unsigned_pack_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_recurrent_step_stub, &recurrent_step_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_dynamic_dot_stub, &dynamic_dot_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_dynamic_block_stub, &dynamic_block_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_linear_epilogue_stub, &linear_epilogue_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_int4_pack_stub, &int4_pack_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_int4_mm_stub, &int4_mm_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_batch_norm_stub, &batch_norm_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_bilinear_stub, &bilinear_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_clip_stub, &clip_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_prelu_stub, &prelu_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_float_prelu_stub, &float_prelu_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_sigmoid_stub, &word_activation_codes<false>);
ALSO_REGISTER_AVX512_DISPATCH(quantized_tanh_stub, &word_activation_codes<true>);
ALSO_REGISTER_AVX512_DISPATCH(quantized_moment_stub, &moment_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_spatial_mean_stub, &spatial_mean_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_pool_stub, &pool_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_lookup_stub, &lookup_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_softmax_stub, &softmax_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_rowwise_pack_stub, &rowwise_pack_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_binary_stub, &binary_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_embedding_stub, &embedding_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_fp16_linear_stub, &fp16_linear_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_cat_rows_stub, &cat_rows_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_normalize_stub, &normalize_impl);
#endif
}
