#include "cpu/QuantizedElementwiseKernels.h"
#include "cpu/vec/vec.h"
#include "cpu/vec/vec_convert.h"
#include "QuantElementwise.h"
#include "QuantRowwise.h"
#include "Parallel.h"
#ifndef CPU_CAPABILITY_AVX512
#include "cpu/vec/vec256/vec256_qint.h"
#endif

namespace tensorplay::cpu {
inline namespace CPU_CAPABILITY {
namespace {
constexpr int int4_columns = vec::Vectorized<float>::size() * 4;

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

template<class T>
void int4_mm_codes(const T* x, const uint8_t* packed, const T* params, T* output,
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
        case DType::BFloat16: int4_mm_codes(static_cast<const BFloat16*>(x),packed,static_cast<const BFloat16*>(params),static_cast<BFloat16*>(output),m,n,k,group); break;
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
REGISTER_DISPATCH(quantized_recurrent_step_stub, &recurrent_step_impl);
REGISTER_DISPATCH(quantized_dynamic_dot_stub, &dynamic_dot_impl);
REGISTER_DISPATCH(quantized_dynamic_block_stub, &dynamic_block_impl);
REGISTER_DISPATCH(quantized_int4_pack_stub, &int4_pack_impl);
REGISTER_DISPATCH(quantized_int4_mm_stub, &int4_mm_impl);
REGISTER_DISPATCH(quantized_batch_norm_stub, &batch_norm_impl);
REGISTER_DISPATCH(quantized_bilinear_stub, &bilinear_impl);
REGISTER_DISPATCH(quantized_clip_stub, &clip_impl);
REGISTER_DISPATCH(quantized_lookup_stub, &lookup_impl);
REGISTER_DISPATCH(quantized_binary_stub, &binary_impl);
REGISTER_DISPATCH(quantized_embedding_stub, &embedding_impl);
REGISTER_DISPATCH(quantized_fp16_linear_stub, &fp16_linear_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(quantized_recurrent_step_stub, &recurrent_step_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_dynamic_dot_stub, &dynamic_dot_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_dynamic_block_stub, &dynamic_block_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_int4_pack_stub, &int4_pack_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_int4_mm_stub, &int4_mm_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_batch_norm_stub, &batch_norm_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_bilinear_stub, &bilinear_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_clip_stub, &clip_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_lookup_stub, &lookup_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_binary_stub, &binary_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_embedding_stub, &embedding_impl);
ALSO_REGISTER_AVX512_DISPATCH(quantized_fp16_linear_stub, &fp16_linear_impl);
#endif
}
