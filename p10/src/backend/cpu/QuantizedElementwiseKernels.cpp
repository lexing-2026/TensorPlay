#include "cpu/QuantizedElementwiseKernels.h"
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
        V::blendv(lo,vec::minimum(hi,vec::maximum(lo,code)),valid).store(codes);
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
            if (relu) value = vec::maximum(value, zero);
            const DV codes = vec::minimum(maximum, vec::maximum(minimum, (value / output_scale).round() + output_zero));
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
                value=mode==0 ? vec::maximum(value,zero) : value.tanh();
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

template<class T, class X = T>
void int4_mm_codes(const X* x, const uint8_t* packed, const T* params, T* output,
        int64_t m, int64_t n, int64_t k, int64_t group) {
    using FV = vec::Vectorized<float>;
    constexpr int lanes = FV::size(), columns = int4_columns;
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
                            if constexpr (std::is_same_v<T,float>) {
                                a=FV::loadu(params+index); b=FV::loadu(params+index+lanes);
                            } else {
                                using TV=vec::Vectorized<T>;
                                const auto values=vec::convert<float,2>(vec::VectorizedN<T,1>(TV::loadu(params+index)));
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
                        values=_mm512_and_si512(_mm512_srli_epi32(values,shift),_mm512_set1_epi32(15));
                        code_vector=FV(_mm512_cvtepi32_ps(_mm512_sub_epi32(values,_mm512_set1_epi32(8))));
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
            for (int r=0;r<rows;++r) for (int c=0;c<vectors;++c) {
                alignas(64) float values[lanes]; acc[r][c].store(values);
                for (int lane=0;lane<lanes && c*lanes+lane<width;++lane)
                    output[(row+r)*n+first+c*lanes+lane]=static_cast<T>(values[lane]);
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

void int4_mm_impl(const void* x, const uint8_t* packed, const void* params, void* output,
        DType dtype, int64_t m, int64_t n, int64_t k, int64_t group) {
    switch (dtype) {
        case DType::Float32: int4_mm_codes(static_cast<const float*>(x),packed,static_cast<const float*>(params),static_cast<float*>(output),m,n,k,group); break;
        case DType::Float16: int4_mm_codes(static_cast<const Half*>(x),packed,static_cast<const Half*>(params),static_cast<Half*>(output),m,n,k,group); break;
        case DType::BFloat16: {
            const auto* input = static_cast<const BFloat16*>(x);
            if (m < 8) {
                int4_mm_codes(input,packed,static_cast<const BFloat16*>(params),static_cast<BFloat16*>(output),m,n,k,group);
                break;
            }
            // Convert each activation once, then reuse the full-precision
            // values across output tiles without changing accumulation order.
            thread_local std::vector<float> real;
            real.resize(m * k);
            using V = vec::Vectorized<BFloat16>;
            int64_t i = 0;
            for (; i + V::size() <= m * k; i += V::size()) {
                const auto values = vec::convert<float,2>(vec::VectorizedN<BFloat16,1>(V::loadu(input + i)));
                values[0].store(real.data() + i);
                values[1].store(real.data() + i + vec::Vectorized<float>::size());
            }
            for (; i < m * k; ++i) real[i] = static_cast<float>(input[i]);
            int4_mm_codes<BFloat16,float>(real.data(),packed,static_cast<const BFloat16*>(params),
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
            if (relu) values[j] = vec::maximum(values[j], zero);
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
        for (; i + V::size() <= size; i += V::size()) V::loadu(x + i).relu(lower).store(y + i);
        if (i < size) V::loadu(x + i, size - i).relu(lower).store(y + i, size - i);
    } else {
        for (; i + V::size() <= size; i += V::size()) V::loadu(x + i).relu6(lower, upper).store(y + i);
        if (i < size) V::loadu(x + i, size - i).relu6(lower, upper).store(y + i, size - i);
    }
}

template<class Q>
void moment_codes(const void* input, void* output, int64_t count, int64_t rows,
                  double scale, int64_t zp, bool deviation, double correction) {
    using T = typename Q::underlying;
    using QV = vec::Vectorized<Q>;
    using IV = vec::Vectorized<qint32>;
    using FV = vec::Vectorized<float>;
    const auto* x = static_cast<const T*>(input);
    auto* y = static_cast<T*>(output);
    for (int64_t row = 0; row < rows; ++row) {
        const auto* values = x + row * count;
        int64_t sum = 0, integer_square_sum = 0;
        double square_sum = 0.;
        int64_t i = 0;
        if constexpr (sizeof(T) == 1) {
            const QV zero(Q(static_cast<T>(0)));
            while (i + QV::size() <= count) {
                const int64_t stop = std::min<int64_t>(i + 32768, count - count % QV::size());
                typename QV::int_vec_return_type totals, squares;
                for (size_t j = 0; j < totals.size(); ++j) { totals[j] = IV(qint32(0)); squares[j] = IV(qint32(0)); }
                for (; i < stop; i += QV::size()) {
                    const auto codes = QV::loadu(values + i).widening_subtract(zero);
                    for (size_t j = 0; j < totals.size(); ++j) {
                        totals[j] = totals[j] + codes[j];
                        if (deviation) squares[j] = squares[j] + codes[j] * codes[j];
                    }
                }
                alignas(64) int32_t lanes[IV::size()];
                for (size_t j = 0; j < totals.size(); ++j) {
                    totals[j].store(lanes);
                    for (const auto value : lanes) sum += value;
                    if (deviation) {
                        squares[j].store(lanes);
                        for (const auto value : lanes) integer_square_sum += value;
                    }
                }
            }
            for (; i < count; ++i) { sum += values[i]; integer_square_sum += static_cast<int64_t>(values[i]) * values[i]; }
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
                vec::minimum(hi, vec::maximum(lo, (vec::convert<float>(sums[v]) * inverse + zero).round())).store(codes + v * lanes);
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

void pool_impl(const void* input, void* output, int64_t first, int64_t last, DType dtype,
               const quant::PoolGeometry& g) {
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
    int64_t i = 0;
    for (; i + V::size() <= size; i += V::size()) {
        const V a = V::loadu(x + i);
        const V b = scalar_weight ? V(Q(w[0])) : V::loadu(w + i);
        auto positive = a.maximum(input_zero).widening_subtract(input_zero);
        const auto negative = a.minimum(input_zero).widening_subtract(input_zero);
        const auto slope = b.widening_subtract(weight_zero);
        for (size_t j = 0; j < positive.size(); ++j)
            positive[j] = positive[j] * unit + negative[j] * slope[j];
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
template<class Q>
void binary_codes(const void* aa, const void* bb, void* yy, int64_t size,
        double as, int64_t az, double bs, int64_t bz, double ys, int64_t yz, int op, bool relu) {
    using T = typename Q::underlying;
    using QV = vec::Vectorized<Q>;
    using FV = vec::Vectorized<float>;
    const T* a = static_cast<const T*>(aa); const T* b = static_cast<const T*>(bb); T* y = static_cast<T*>(yy);
    const FV asc(as), azp(az), bsc(bs), bzp(bz), zero(0.f);
    const FV lower((std::numeric_limits<T>::lowest() - yz) * static_cast<float>(ys));
    const FV upper((std::numeric_limits<T>::max() - yz) * static_cast<float>(ys));
    int64_t i = 0;
    for (; i + QV::size() <= size; i += QV::size()) {
        auto av = QV::loadu(a + i).dequantize(asc, azp);
        auto bv = QV::loadu(b + i).dequantize(bsc, bzp);
        for (size_t j = 0; j < av.size(); ++j) {
            switch (op) {
                case 0: av[j] = av[j] + bv[j]; break;
                case 1: av[j] = av[j] - bv[j]; break;
                case 2: av[j] = av[j] * bv[j]; break;
                default: av[j] = av[j] / bv[j]; break;
            }
            if (relu) av[j] = vec::maximum(av[j], zero);
            av[j] = vec::minimum(vec::maximum(av[j], lower), upper);
        }
        QV::quantize(av, ys, yz, 1.f / static_cast<float>(ys)).store(y + i);
    }
    for (; i < size; ++i) {
        const float av = static_cast<float>(static_cast<int64_t>(a[i]) - az) * static_cast<float>(as);
        const float bv = static_cast<float>(static_cast<int64_t>(b[i]) - bz) * static_cast<float>(bs);
        float v;
        switch (op) { case 0: v = av + bv; break; case 1: v = av - bv; break; case 2: v = av * bv; break; default: v = av / bv; }
        if (relu) v = std::fmax(v, 0.f);
        y[i] = quant::encode<T>(v, 1.f / static_cast<float>(ys), yz);
    }
}

void binary_impl(const void* a, const void* b, void* y, int64_t size, DType dtype,
        double as, int64_t az, double bs, int64_t bz, double ys, int64_t yz, int op, bool relu) {
    if (dtype == DType::QInt8) binary_codes<qint8>(a,b,y,size,as,az,bs,bz,ys,yz,op,relu);
    else if (dtype == DType::QUInt8) binary_codes<quint8>(a,b,y,size,as,az,bs,bz,ys,yz,op,relu);
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

void embedding_impl(const uint8_t* w, const int64_t* indices, const int64_t* offsets,
        const int64_t* mapping, const float* weights, float* y, int64_t first_bag, int64_t last_bag,
        int64_t bags, int64_t count, int64_t fixed, int64_t columns, int64_t width, int64_t bits, bool embedding) {
    using FV = vec::Vectorized<float>;
    constexpr int lanes = FV::size();
    for (int64_t bag = first_bag; bag < last_bag; ++bag) {
        const int64_t first = embedding ? bag : (offsets ? offsets[bag] : bag * fixed);
        const int64_t last = embedding ? bag + 1 : (offsets ? (bag + 1 < bags ? offsets[bag+1] : count) : first + fixed);
        for (int64_t col = 0; col < columns; col += lanes) {
            FV sum(0.f); const int take = static_cast<int>(std::min<int64_t>(lanes, columns - col));
            for (int64_t j = first; j < last; ++j) {
                const int64_t row_index = mapping ? mapping[indices[j]] : indices[j];
                if (row_index == -1) continue;
                const uint8_t* row = w + row_index * width; float scale, bias;
                quant::row_grid(row, width, bits, scale, bias);
                const float weight = weights ? weights[j] : 1.f;
                alignas(64) float codes[lanes]{};
                for (int c = 0; c < take; ++c) codes[c] = quant::row_code(row, col + c, bits);
                sum = vec::fmadd(FV::loadu(codes), FV(scale * weight), sum + FV(bias * weight));
            }
            sum.store(y + bag * columns + col, take);
        }
    }
}

}
}

#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(quantized_subbyte_pack_stub, &subbyte_pack_impl);
REGISTER_DISPATCH(quantized_subbyte_unpack_stub, &subbyte_unpack_impl);
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
REGISTER_DISPATCH(quantized_sigmoid_stub, &word_activation_codes<false>);
REGISTER_DISPATCH(quantized_tanh_stub, &word_activation_codes<true>);
REGISTER_DISPATCH(quantized_moment_stub, &moment_impl);
REGISTER_DISPATCH(quantized_pool_stub, &pool_impl);
REGISTER_DISPATCH(quantized_lookup_stub, &lookup_impl);
REGISTER_DISPATCH(quantized_binary_stub, &binary_impl);
REGISTER_DISPATCH(quantized_embedding_stub, &embedding_impl);
REGISTER_DISPATCH(quantized_fp16_linear_stub, &fp16_linear_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(quantized_subbyte_pack_stub, &subbyte_pack_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_subbyte_unpack_stub, &subbyte_unpack_impl);
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
ALSO_REGISTER_AVX512_DISPATCH(quantized_sigmoid_stub, &word_activation_codes<false>);
ALSO_REGISTER_AVX512_DISPATCH(quantized_tanh_stub, &word_activation_codes<true>);
ALSO_REGISTER_AVX512_DISPATCH(quantized_moment_stub, &moment_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_pool_stub, &pool_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_lookup_stub, &lookup_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_binary_stub, &binary_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_embedding_stub, &embedding_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_fp16_linear_stub, &fp16_linear_impl);
#endif
}
