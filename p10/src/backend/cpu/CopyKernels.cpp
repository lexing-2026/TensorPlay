#include "Tensor.h"
#include "Dispatcher.h"
#include "Parallel.h"
#include "Scalar.h"
#include "TypePromotion.h"
#include "Utils.h"
#include "SparseKernels.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <type_traits>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#ifdef USE_CUDA
#include "CUDARuntime.h"
#include <cuda_runtime.h>
#endif

namespace tensorplay {
namespace cpu {

Tensor to_kernel(Tensor& self, DType dtype, bool non_blocking, bool copy) {
    if (self.dtype() == dtype) {
        return copy ? detail::contiguous_clone(self) : self;
    }
    // Create new tensor
    Tensor result(static_cast<std::vector<int64_t>>(self.shape()), dtype, self.device());
    result.copy_(self);
    return result;
}

// Parallel ND strided copy, mirroring the throughput of ATen's
// TensorIterator-driven copy for permuted/expanded operands: dims are visited
// in destination-stride order so the innermost loop writes consecutive
// memory (a unit-stride source there degrades into memcpy), size-1 dims are
// dropped, and chunks beyond the grain threshold are spread across the
// intra-op thread pool.  A single remaining dim is sliced directly so large
// strided vectors parallelize too.
// Contiguous dtype-conversion cast (defined below the SIMD helpers);
// forward-declared here because parallel_strided_copy's inner range copy
// dispatches to it.
template <typename dst_t, typename src_t>
void cast_contiguous(dst_t* dst, const src_t* src, int64_t len);

template <typename self_t, typename src_t>
void parallel_strided_copy(self_t* dst, const std::vector<int64_t>& dst_strides_in,
                           const src_t* src, const std::vector<int64_t>& src_strides_in,
                           const std::vector<int64_t>& sizes) {
    const int ndim = static_cast<int>(sizes.size());
    std::vector<int64_t> o_sizes, o_dstr, o_sstr;
    o_sizes.reserve(ndim);
    o_dstr.reserve(ndim);
    o_sstr.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
        if (sizes[i] == 1) continue;  // stride irrelevant; contributes one element
        if (sizes[i] == 0) return;    // nothing to copy
        o_sizes.push_back(sizes[i]);
        o_dstr.push_back(dst_strides_in[i]);
        o_sstr.push_back(src_strides_in[i]);
    }
    const int m = static_cast<int>(o_sizes.size());
    if (m == 0) {
        *dst = cast_value<self_t>(src[0]);
        return;
    }

    // Largest destination stride first, so the last dim varies fastest in
    // destination memory; ties break toward the larger source stride.
    std::vector<int> order(m);
    for (int i = 0; i < m; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (o_dstr[a] != o_dstr[b]) return o_dstr[a] > o_dstr[b];
        return o_sstr[a] > o_sstr[b];
    });
    std::vector<int64_t> c_sizes(m), c_dstr(m), c_sstr(m);
    for (int i = 0; i < m; ++i) {
        c_sizes[i] = o_sizes[order[i]];
        c_dstr[i] = o_dstr[order[i]];
        c_sstr[i] = o_sstr[order[i]];
    }

    const int64_t inner_len = c_sizes[m - 1];
    const int64_t d_is = c_dstr[m - 1];
    const int64_t s_is = c_sstr[m - 1];

    auto copy_range = [&](int64_t dst_off, int64_t src_off, int64_t len) {
        if (d_is == 1 && s_is == 1) {
            if constexpr (std::is_same_v<self_t, src_t>) {
                std::memcpy(dst + dst_off, src + src_off,
                            static_cast<size_t>(len) * sizeof(self_t));
            } else {
                cast_contiguous(dst + dst_off, src + src_off, len);
            }
        } else {
            for (int64_t i = 0; i < len; ++i)
                dst[dst_off + i * d_is] = cast_value<self_t>(src[src_off + i * s_is]);
        }
    };

    if (m == 1) {
        parallel::parallel_for(0, inner_len, parallel::GRAIN_SIZE,
                               [&](int64_t b, int64_t e) {
                                   copy_range(b * d_is, b * s_is, e - b);
                               });
        return;
    }

    const int od = m - 1;
    const int64_t outer =
        std::accumulate(c_sizes.begin(), c_sizes.begin() + od, int64_t{1},
                        std::multiplies<int64_t>{});
    const int64_t grain =
        std::max<int64_t>(1, parallel::GRAIN_SIZE / std::max<int64_t>(inner_len, 1));
    std::vector<int64_t> outer_sizes(c_sizes.begin(), c_sizes.end() - 1);
    std::vector<int64_t> outer_dstr(c_dstr.begin(), c_dstr.end() - 1);
    std::vector<int64_t> outer_sstr(c_sstr.begin(), c_sstr.end() - 1);

    parallel::parallel_for(0, outer, grain, [&]([[maybe_unused]] int64_t b, int64_t e) {
        // Decode the mixed-radix index once, then advance incrementally.
        std::vector<int64_t> idx(od, 0);
        int64_t d_off = 0, s_off = 0;
        int64_t rest = b;
        for (int i = od - 1; i >= 0; --i) {
            const int64_t q = rest / outer_sizes[i];
            idx[i] = rest - q * outer_sizes[i];
            rest = q;
            d_off += idx[i] * outer_dstr[i];
            s_off += idx[i] * outer_sstr[i];
        }
        for (int64_t o = b; o < e; ++o) {
            copy_range(d_off, s_off, inner_len);
            for (int i = od - 1; i >= 0; --i) {  // odometer increment
                ++idx[i];
                d_off += outer_dstr[i];
                s_off += outer_sstr[i];
                if (idx[i] < outer_sizes[i]) break;
                d_off -= outer_dstr[i] * idx[i];  // wrap this digit
                s_off -= outer_sstr[i] * idx[i];
                idx[i] = 0;
            }
        }
    });
}

// Helper for dynamic dispatch
template <typename T>
struct TypeTag { using type = T; };

template <typename F>
void dispatch_dtype(DType dtype, F&& callback) {
    #define DISPATCH_CASE(ctype, name) \
    case DType::name: { \
        callback(TypeTag<ctype>{}); \
        return; \
    }

    switch (dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_FP8(DISPATCH_CASE)
        default:
            throw std::runtime_error("Unsupported dtype in dispatch");
    }
    #undef DISPATCH_CASE
}

// ---------------------------------------------------------------------------
// Vectorized floating-point casts for contiguous ranges.
//
// The autocast weight-cast path (fp32 <-> fp16/bf16) is memory-bound; scalar
// static_cast loops leave 4-8x bandwidth on the table.  F16C covers half,
// AVX512-BF16 (vcvtneps2bf16) covers bfloat16 with round-to-nearest-even --
// bit-identical to BFloat16.h's scalar float_to_bfloat16_bits.  Per-function
// target attributes keep the rest of the TU at baseline -march.
// ---------------------------------------------------------------------------
#if defined(__x86_64__)
__attribute__((target("avx2,f16c")))
static void f32_to_f16_simd(const float* src, uint16_t* dst, int64_t n) {
    constexpr int k = 8;
    int64_t i = 0;
    for (; i + k <= n; i += k) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), h);
    }
    for (; i < n; ++i)
        dst[i] = cast_value<Half>(src[i]).x;
}

__attribute__((target("avx2,f16c")))
static void f16_to_f32_simd(const uint16_t* src, float* dst, int64_t n) {
    constexpr int k = 8;
    int64_t i = 0;
    for (; i + k <= n; i += k) {
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(h));
    }
    for (; i < n; ++i)
        dst[i] = cast_value<float>(Half(src[i], Half::from_bits()));
}

__attribute__((target("avx512bf16,avx512f")))
static void f32_to_bf16_avx512(const float* src, uint16_t* dst, int64_t n) {
    constexpr int k = 16;
    int64_t i = 0;
    for (; i + k <= n; i += k) {
        __m512 v = _mm512_loadu_ps(src + i);
        __m256i bh = (__m256i)_mm512_cvtneps_pbh(v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), bh);
    }
    for (; i < n; ++i)
        dst[i] = cast_value<BFloat16>(src[i]).x;
}

__attribute__((target("avx2,ssse3")))
static void f32_to_bf16_avx2_rne(const float* src, uint16_t* dst, int64_t n) {
    // Round-to-nearest-even via the integer trick, matching
    // detail::float_to_bfloat16_bits lane-wise; shuffle-based compaction.
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i bias = _mm256_set1_epi32(0x7FFF);
    const __m128i pick = _mm_set_epi8(
        char(-1), char(-1), char(-1), char(-1),
        char(-1), char(-1), char(-1), char(-1),
        char(13), char(12), char(9), char(8),
        char(5), char(4), char(1), char(0));
    constexpr int k = 8;
    int64_t i = 0;
    for (; i + k <= n; i += k) {
        __m256i u = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        __m256i r = _mm256_and_si256(_mm256_srli_epi32(u, 16), one);
        r = _mm256_add_epi32(_mm256_add_epi32(u, r), bias);
        __m256i out = _mm256_srli_epi32(r, 16);
        __m128i lo = _mm_shuffle_epi8(_mm256_castsi256_si128(out), pick);
        __m128i hi = _mm_shuffle_epi8(_mm256_extracti128_si256(out, 1), pick);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                         _mm_unpacklo_epi64(lo, hi));
    }
    for (; i < n; ++i)
        dst[i] = cast_value<BFloat16>(src[i]).x;
}

__attribute__((target("avx2")))
static void bf16_to_f32_simd(const uint16_t* src, float* dst, int64_t n) {
    constexpr int k = 8;
    int64_t i = 0;
    for (; i + k <= n; i += k) {
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
        _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(w));
    }
    for (; i < n; ++i)
        dst[i] = cast_value<float>(BFloat16(src[i], BFloat16::from_bits()));
}

inline bool cpu_has_f16c() {
    static const bool ok = __builtin_cpu_supports("f16c") != 0;
    return ok;
}
inline bool cpu_has_bf16() {
    static const bool ok = __builtin_cpu_supports("avx512bf16") != 0;
    return ok;
}
#endif // __x86_64__

// Contiguous dtype-conversion cast; SIMD for the autocast-critical fp32
// <-> fp16/bf16 pairs, scalar static_cast otherwise.
template <typename dst_t, typename src_t>
void cast_contiguous(dst_t* dst, const src_t* src, int64_t len) {
#if defined(__x86_64__)
    if constexpr (std::is_same_v<dst_t, Half> && std::is_same_v<src_t, float>) {
        if (cpu_has_f16c()) { f32_to_f16_simd(src, &dst->x, len); return; }
    } else if constexpr (std::is_same_v<dst_t, float> && std::is_same_v<src_t, Half>) {
        if (cpu_has_f16c()) { f16_to_f32_simd(&src->x, dst, len); return; }
    } else if constexpr (std::is_same_v<dst_t, BFloat16> && std::is_same_v<src_t, float>) {
        if (cpu_has_bf16()) { f32_to_bf16_avx512(src, &dst->x, len); return; }
        if (cpu_has_f16c() || __builtin_cpu_supports("avx2")) {
            f32_to_bf16_avx2_rne(src, &dst->x, len); return;
        }
    } else if constexpr (std::is_same_v<dst_t, float> && std::is_same_v<src_t, BFloat16>) {
        if (__builtin_cpu_supports("avx2")) { bf16_to_f32_simd(&src->x, dst, len); return; }
    }
#endif
    for (int64_t i = 0; i < len; ++i)
        dst[i] = cast_value<dst_t>(src[i]);
}



Tensor& copy_kernel(Tensor& self, const Tensor& src, bool non_blocking) {
    if (!self.device().is_cpu()) {
        throw std::runtime_error("copy_kernel (CPU) called with non-CPU destination");
    }

    // ATen parity (Copy.cpp:280): nothing to copy; also keeps NULL data_ptrs
    // of empty tensors away from memcpy/cudaMemcpyAsync.
    if (self.numel() == 0) {
        return self;
    }

    if (src.device().is_cuda()) {
#ifdef USE_CUDA
        cuda::CUDAGuard device_guard(src.device());
        auto stream = cuda::getCurrentCUDAStream(static_cast<int>(src.device().index()));

        Tensor src_ready = src;
        // Row-major contiguity (not is_contiguous(), which channels-last
        // tensors also report): flat cudaMemcpy only applies to canonical
        // row-major layouts.
        const auto cuda_rm_strides = SizesAndStrides::compute_contiguous_strides(
            static_cast<std::vector<int64_t>>(src.shape()));
        const bool src_row_major =
            static_cast<std::vector<int64_t>>(src.strides()) == cuda_rm_strides;
        if (!src_row_major || src.dtype() != self.dtype()) {
            src_ready = Tensor(static_cast<std::vector<int64_t>>(src.shape()), self.dtype(), src.device());
            src_ready.copy_(src);
        }

        const size_t nbytes = self.numel() * self.itemsize();
        const auto self_rm_strides = SizesAndStrides::compute_contiguous_strides(
            static_cast<std::vector<int64_t>>(self.shape()));
        if (static_cast<std::vector<int64_t>>(self.strides()) == self_rm_strides) {
            cuda::checkCuda(cudaMemcpyAsync(self.data_ptr(), src_ready.data_ptr(), nbytes,
                                            cudaMemcpyDeviceToHost, stream.stream()),
                            "cudaMemcpyAsync (D2H)");
            if (non_blocking && self.is_pinned()) {
                cuda::recordPinnedStream(
                    self.unsafeGetTensorImpl()->storage().data(), stream);
            } else {
                // Ordinary CPU storage may be consumed immediately after
                // copy_ returns, so the host-visible result must be complete.
                stream.synchronize();
            }
        } else {
            Tensor host_contiguous(static_cast<std::vector<int64_t>>(self.shape()),
                                   self.dtype(), Device(DeviceType::CPU));
            cuda::checkCuda(cudaMemcpyAsync(host_contiguous.data_ptr(), src_ready.data_ptr(), nbytes,
                                            cudaMemcpyDeviceToHost, stream.stream()),
                            "cudaMemcpyAsync (D2H staging)");
            stream.synchronize();
            self.copy_(host_contiguous);
        }
        return self;
#else
        throw std::runtime_error("CUDA source but USE_CUDA not enabled");
#endif
    }
    
    if (!src.device().is_cpu()) {
        throw std::runtime_error("copy_kernel only supports CPU or CUDA source");
    }
    
    dispatch_dtype(self.dtype(), [&](auto self_tag) {
        using self_t = typename decltype(self_tag)::type;
        
        dispatch_dtype(src.dtype(), [&](auto src_tag) {
            using src_t = typename decltype(src_tag)::type;
            
            // Fast path for contiguous same-shape copies: memcpy when the
            // dtype matches, otherwise the vectorized cast (fp32 <-> fp16 /
            // bf16 hits F16C / AVX512-BF16).  Skips the ND-iteration setup;
            // large casts still spread across the intra-op pool.
            // is_contiguous() alone is not enough: channels-last tensors
            // report contiguous (single layout flag), yet a flat memcpy/cast
            // between row-major and channels-last layouts would corrupt the
            // data.  Require canonical row-major strides on both sides.
            const auto rm_strides = SizesAndStrides::compute_contiguous_strides(
                static_cast<std::vector<int64_t>>(self.shape()));
            if (self.shape() == src.shape() &&
                static_cast<std::vector<int64_t>>(self.strides()) == rm_strides &&
                static_cast<std::vector<int64_t>>(src.strides()) == rm_strides) {
                if (std::is_same_v<self_t, src_t>) {
                    size_t nbytes = self.numel() * self.itemsize();
                    std::memcpy(self.data_ptr(), src.data_ptr(), nbytes);
                    return;
                }
                const int64_t n = self.numel();
                auto* d = self.data_ptr<self_t>();
                const auto* s = src.data_ptr<src_t>();
                if (n > 1 && n >= parallel::GRAIN_SIZE) {
                    parallel::parallel_for(0, n, parallel::GRAIN_SIZE,
                        [&](int64_t b, int64_t e) {
                            cast_contiguous(d + b, s + b, e - b);
                        });
                    return;
                }
                cast_contiguous(d, s, n);
                return;
            }

            parallel_strided_copy(self.data_ptr<self_t>(), self.strides(),
                                  src.data_ptr<src_t>(), src.strides(),
                                  static_cast<std::vector<int64_t>>(self.shape()));
        });
    });
    
    return self;
}



Tensor masked_select_cpu(const Tensor& self, const Tensor& mask) {
    // 1. Broadcast shapes
    std::vector<int64_t> broadcast_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(mask.shape()));
    
    Tensor self_expanded = self.expand(broadcast_shape);
    Tensor mask_expanded = mask.expand(broadcast_shape);
    
    // 2. Make contiguous for simple iteration
    Tensor self_contig = self_expanded.contiguous();
    Tensor mask_contig = mask_expanded.contiguous();
    
    int64_t numel = self_contig.numel();
    const uint8_t* mask_ptr = nullptr;
    
    // Handle mask dtype
    if (mask_contig.dtype() == DType::Bool || mask_contig.dtype() == DType::UInt8) {
        mask_ptr = mask_contig.data_ptr<uint8_t>();
    } else {
        TP_THROW(TypeError, "masked_select: mask must be Bool or Byte");
    }
    
    // 3. Count true elements
    int64_t true_count = 0;
    for (int64_t i = 0; i < numel; ++i) {
        if (mask_ptr[i]) true_count++;
    }
    
    // 4. Allocate result
    Tensor result = Tensor::empty({true_count}, self.dtype(), self.device());
    
    // 5. Fill result
    dispatch_dtype(self.dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        const T* src = self_contig.data_ptr<T>();
        T* dst = result.data_ptr<T>();
        
        int64_t idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            if (mask_ptr[i]) {
                dst[idx++] = src[i];
            }
        }
    });
    
    return result;
}

Tensor embedding_cpu(const Tensor& weight, const Tensor& indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
    // 1. Check inputs
    if (indices.dtype() != DType::Int64 && indices.dtype() != DType::Int32) {
        TP_THROW(TypeError, "embedding: indices must be Int64 or Int32");
    }
    
    // 2. Calculate output shape
    // Output shape = indices.shape + weight.shape[1:]
    std::vector<int64_t> out_shape = static_cast<std::vector<int64_t>>(indices.shape());
    std::vector<int64_t> weight_shape = static_cast<std::vector<int64_t>>(weight.shape());
    
    if (weight.dim() == 0) {
        TP_THROW(RuntimeError, "embedding: weight must be at least 1-dim");
    }
    
    for (size_t i = 1; i < weight_shape.size(); ++i) {
        out_shape.push_back(weight_shape[i]);
    }
    
    // 3. Allocate output
    Tensor output = Tensor::empty(out_shape, weight.dtype(), weight.device());
    
    // 4. Copy data
    // Flatten indices for iteration
    int64_t num_indices = indices.numel();
    int64_t row_size = 1;
    for (size_t i = 1; i < weight_shape.size(); ++i) row_size *= weight_shape[i];
    int64_t weight_size_0 = weight.size(0);
    
    // Contiguous access optimization
    Tensor indices_contig = indices.contiguous();
    Tensor weight_contig = weight.contiguous();
    
    dispatch_dtype(weight.dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        const T* weight_data = weight_contig.data_ptr<T>();
        T* out_data = output.data_ptr<T>();
        
        // Handle indices type
        if (indices.dtype() == DType::Int64) {
            const int64_t* idx_data = indices_contig.data_ptr<int64_t>();
            for (int64_t i = 0; i < num_indices; ++i) {
                int64_t idx = idx_data[i];
                if (idx < 0) idx += weight_size_0;
                if (idx < 0 || idx >= weight_size_0) {
                    TP_THROW(IndexError, "embedding: index out of range");
                }
                
                // Copy row
                std::memcpy(out_data + i * row_size, weight_data + idx * row_size, row_size * sizeof(T));
            }
        } else { // Int32
            const int32_t* idx_data = indices_contig.data_ptr<int32_t>();
            for (int64_t i = 0; i < num_indices; ++i) {
                int64_t idx = static_cast<int64_t>(idx_data[i]);
                if (idx < 0) idx += weight_size_0;
                if (idx < 0 || idx >= weight_size_0) {
                    TP_THROW(IndexError, "embedding: index out of range");
                }
                
                std::memcpy(out_data + i * row_size, weight_data + idx * row_size, row_size * sizeof(T));
            }
        }
    });
    
    return output;
}

Tensor embedding_dense_backward_cpu(const Tensor& grad_output, const Tensor& indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
    // 1. Check inputs
    if (indices.dtype() != DType::Int64 && indices.dtype() != DType::Int32) {
        TP_THROW(TypeError, "embedding_dense_backward: indices must be Int64 or Int32");
    }

    // 2. Allocate grad_weight
    std::vector<int64_t> grad_weight_shape;
    grad_weight_shape.push_back(num_weights);
    int64_t weight_dims = grad_output.dim() - indices.dim();
    for (int i = 0; i < weight_dims; ++i) {
        grad_weight_shape.push_back(grad_output.size(indices.dim() + i));
    }
    
    Tensor grad_weight = Tensor::zeros(grad_weight_shape, grad_output.dtype(), grad_output.device());
    
    // 3. Accumulate gradients
    int64_t num_indices = indices.numel();
    int64_t grad_numel = grad_output.numel();
    int64_t row_size = num_indices > 0 ? grad_numel / num_indices : 0;
    
    if (num_indices == 0) return grad_weight;

    Tensor indices_contig = indices.contiguous();
    Tensor grad_output_contig = grad_output.contiguous();
    
    dispatch_dtype(grad_output.dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_same_v<T, bool>) {
             TP_THROW(RuntimeError, "embedding_dense_backward: grad_output cannot be Bool");
        } else {
            const T* grad_data = grad_output_contig.data_ptr<T>();
            T* weight_grad_data = grad_weight.data_ptr<T>();
            
            // Handle indices type
            if (indices.dtype() == DType::Int64) {
                const int64_t* idx_data = indices_contig.data_ptr<int64_t>();
                for (int64_t i = 0; i < num_indices; ++i) {
                    int64_t idx = idx_data[i];
                    if (idx == padding_idx) continue;
                    if (idx < 0) idx += num_weights;
                    if (idx < 0 || idx >= num_weights) {
                         TP_THROW(IndexError, "embedding_dense_backward: index out of range");
                    }
                    
                    // Add row: weight_grad[idx] += grad[i]
                    T* dst_row = weight_grad_data + idx * row_size;
                    const T* src_row = grad_data + i * row_size;
                    for (int64_t j = 0; j < row_size; ++j) {
                        dst_row[j] += src_row[j];
                    }
                }
            } else { // Int32
                 const int32_t* idx_data = indices_contig.data_ptr<int32_t>();
                for (int64_t i = 0; i < num_indices; ++i) {
                    int64_t idx = static_cast<int64_t>(idx_data[i]);
                    if (idx == padding_idx) continue;
                    if (idx < 0) idx += num_weights;
                    if (idx < 0 || idx >= num_weights) {
                         TP_THROW(IndexError, "embedding_dense_backward: index out of range");
                    }
                    
                    // Add row
                    T* dst_row = weight_grad_data + idx * row_size;
                    const T* src_row = grad_data + i * row_size;
                    for (int64_t j = 0; j < row_size; ++j) {
                        dst_row[j] += src_row[j];
                    }
                }
            }
        }
    });
    
    return grad_weight;
}

Tensor embedding_backward_cpu(const Tensor& grad_output, const Tensor& indices,
                              int64_t num_weights, int64_t padding_idx,
                              bool scale_grad_by_freq, bool sparse) {
    if (sparse) {
        return embedding_sparse_backward_cpu(grad_output, indices, num_weights,
                                             padding_idx, scale_grad_by_freq);
    }
    return embedding_dense_backward_cpu(grad_output, indices, num_weights,
                                       padding_idx, scale_grad_by_freq);
}

TENSORPLAY_LIBRARY_IMPL(CPU, CopyKernels) {
    m.impl("to", to_kernel);
    m.impl("masked_select", masked_select_cpu);
    m.impl("copy_", copy_kernel);
    m.impl("sparse_coo_tensor", sparse_coo_tensor_cpu);
    m.impl("sparse_mask", sparse_mask_cpu);
    m.impl("to_dense", to_dense_sparse_cpu);
    m.impl("to_sparse", to_sparse_coo_cpu);
    m.impl("to_sparse_csr", to_sparse_csr_cpu);
    m.impl("_nnz", sparse_nnz_cpu);
    m.impl("sparse_mm", sparse_mm_cpu);
    m.impl("sparse_sum", sparse_sum_cpu);
    m.impl("sparse_add", sparse_add_cpu);
    m.impl("sparse_mul", sparse_mul_cpu);
    m.impl("spdiags", spdiags_cpu);
    m.impl("embedding", embedding_cpu);
    m.impl("embedding_dense_backward", embedding_dense_backward_cpu);
    m.impl("embedding_sparse_backward", embedding_sparse_backward_cpu);
    m.impl("embedding_backward", embedding_backward_cpu);
}

} // namespace cpu
} // namespace tensorplay
