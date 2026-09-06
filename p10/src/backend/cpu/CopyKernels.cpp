#include "Tensor.h"
#include "Dispatcher.h"
#include "Parallel.h"
#include "Scalar.h"
#include "TypePromotion.h"
#include "Utils.h"
#include "SparseKernels.h"
#include <algorithm>
#include <cmath>
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

#ifdef USE_VULKAN
#include "VulkanRuntime.h"
#include "backend/vulkan/api/Tensor.h"
#include "backend/vulkan/ops/Convert.h"
#include "backend/vulkan/ops/Copy.h"
#endif

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

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

template <typename T>
void transpose_copy_tiled(T* dst, const T* src, int64_t rows, int64_t cols) {
    constexpr int64_t block = 60;
    const int64_t tile_rows = (rows + block - 1) / block;
    const int64_t tile_cols = (cols + block - 1) / block;
    parallel::parallel_for(0, tile_rows * tile_cols, 1,
        [=](int64_t begin, int64_t end) {
            alignas(64) T tile[block * block];
            for (int64_t tile_id = begin; tile_id < end; ++tile_id) {
                const int64_t row0 = (tile_id / tile_cols) * block;
                const int64_t col0 = (tile_id % tile_cols) * block;
                const int64_t nr = std::min(block, rows - row0);
                const int64_t nc = std::min(block, cols - col0);
                for (int64_t r = 0; r < nr; ++r) {
                    for (int64_t c = 0; c < nc; ++c) {
                        tile[r * block + c] = src[row0 + r + (col0 + c) * rows];
                    }
                }
                for (int64_t r = 0; r < nr; ++r) {
                    std::memcpy(dst + (row0 + r) * cols + col0,
                                tile + r * block,
                                static_cast<size_t>(nc) * sizeof(T));
                }
            }
        });
}

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

    // Quantized dtypes share their underlying integer code type: copies and
    // casts move the raw codes, quantizer metadata rides on the impl.
    switch (dtype) {
        case DType::QInt8: {
            callback(TypeTag<int8_t>{});
            return;
        }
        case DType::QUInt8: {
            callback(TypeTag<uint8_t>{});
            return;
        }
        case DType::QInt32: {
            callback(TypeTag<int32_t>{});
            return;
        }
        default:
            break;
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

#ifdef USE_VULKAN
    if (src.device().is_vulkan()) {
        tensorplay::vulkan::api::vTensor v_src =
            tensorplay::vulkan::ops::convert(src);
        tensorplay::vulkan::ops::transfer_vulkan_to_cpu_impl(v_src, self);
        return self;
    }
#endif

    if (!src.device().is_cpu()) {
        throw std::runtime_error("copy_kernel only supports CPU, CUDA or Vulkan source");
    }

    if (self.dtype() == src.dtype() && self.dim() == 2 && self.numel() >= 60 * 60 &&
        self.shape() == src.shape()) {
        const int64_t rows = self.size(0);
        const int64_t cols = self.size(1);
        const auto dst_strides = static_cast<std::vector<int64_t>>(self.strides());
        const auto src_strides = static_cast<std::vector<int64_t>>(src.strides());
        if (dst_strides.size() == 2 && src_strides.size() == 2 &&
            dst_strides[0] == cols && dst_strides[1] == 1 &&
            src_strides[0] == 1 && src_strides[1] == rows) {
            dispatch_dtype(self.dtype(), [&](auto tag) {
                using T = typename decltype(tag)::type;
                transpose_copy_tiled(self.data_ptr<T>(), src.data_ptr<T>(), rows, cols);
            });
            return self;
        }
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

namespace {

// Element count targeted per parallel chunk in the copy/accumulate loops. Each
// chunk touches roughly 16k elements (64 KB of fp32), large enough to amortize
// thread-pool dispatch while still filling every core.
constexpr int64_t kEmbeddingGrainElements = 16384;

inline int64_t embedding_grain_indices(int64_t row_size) {
  return std::max<int64_t>(1, kEmbeddingGrainElements / std::max<int64_t>(row_size, 1));
}

// Work threshold (in elements) above which the dense backward switches from a
// single-threaded accumulation to the sorted-segment parallel path.
constexpr int64_t kEmbeddingSortedBackwardElements = 1 << 18;

template <typename T>
void embedding_renorm_rows(
        Tensor& weight, const Tensor& indices, double max_norm,
        double norm_type) {
    const Tensor indices_contiguous = indices.contiguous();
    const int64_t rows = weight.size(0);
    const int64_t columns = weight.size(1);
    const int64_t row_stride = weight.stride(0);
    const int64_t column_stride = weight.stride(1);
    T* const data = weight.data_ptr<T>();
    std::vector<uint8_t> seen(static_cast<size_t>(rows), 0);
    std::vector<int64_t> selected_rows;
    selected_rows.reserve(static_cast<size_t>(indices.numel()));

    auto renorm_row = [&](int64_t row) {
        if (seen[static_cast<size_t>(row)] != 0) return;
        seen[static_cast<size_t>(row)] = 1;
        T* const row_data = data + row * row_stride;

        double norm = 0.0;
        if (std::isinf(norm_type)) {
            for (int64_t column = 0; column < columns; ++column) {
                norm = std::max(
                    norm,
                    std::abs(static_cast<double>(row_data[column * column_stride])));
            }
        } else {
            double sum = 0.0;
            for (int64_t column = 0; column < columns; ++column) {
                const double value = static_cast<double>(
                    row_data[column * column_stride]);
                sum += std::pow(std::abs(value), norm_type);
            }
            norm = std::pow(sum, 1.0 / norm_type);
        }

        if (norm > max_norm) {
            const double scale = max_norm / (norm + 1e-7);
            for (int64_t column = 0; column < columns; ++column) {
                const int64_t offset = column * column_stride;
                row_data[offset] = static_cast<T>(
                    static_cast<double>(row_data[offset]) * scale);
            }
        }
    };

    if (indices.dtype() == DType::Int64) {
        const int64_t* const index_data = indices_contiguous.data_ptr<int64_t>();
        for (int64_t i = 0; i < indices.numel(); ++i) {
            int64_t row = index_data[i];
            if (row < 0 || row >= rows) {
                TP_THROW(IndexError, "embedding_renorm_: index out of range");
            }
            selected_rows.push_back(row);
        }
    } else {
        const int32_t* const index_data = indices_contiguous.data_ptr<int32_t>();
        for (int64_t i = 0; i < indices.numel(); ++i) {
            int64_t row = static_cast<int64_t>(index_data[i]);
            if (row < 0 || row >= rows) {
                TP_THROW(IndexError, "embedding_renorm_: index out of range");
            }
            selected_rows.push_back(row);
        }
    }
    for (const int64_t row : selected_rows) renorm_row(row);
}

} // namespace

Tensor& embedding_renorm_cpu(
        Tensor& weight, const Tensor& indices, double max_norm,
        double norm_type) {
    if (weight.device().type() != DeviceType::CPU ||
        indices.device().type() != DeviceType::CPU) {
        TP_THROW(RuntimeError, "embedding_renorm_: CPU tensors are required");
    }
    if (weight.device() != indices.device()) {
        TP_THROW(RuntimeError,
                 "embedding_renorm_: weight and indices must be on the same device");
    }
    if (weight.dim() != 2) {
        TP_THROW(RuntimeError, "embedding_renorm_: weight must be 2-D");
    }
    if (indices.dtype() != DType::Int64 && indices.dtype() != DType::Int32) {
        TP_THROW(TypeError, "embedding_renorm_: indices must be Int64 or Int32");
    }
    if (!(max_norm > 0.0)) {
        TP_THROW(ValueError, "embedding_renorm_: max_norm must be positive");
    }
    if (!(norm_type > 0.0)) {
        TP_THROW(ValueError, "embedding_renorm_: norm_type must be positive");
    }

    switch (weight.dtype()) {
#define RENORM_CASE(ctype, name) \
        case DType::name: \
            embedding_renorm_rows<ctype>(weight, indices, max_norm, norm_type); \
            return weight;
        RENORM_CASE(float, Float32)
        RENORM_CASE(double, Float64)
        RENORM_CASE(Half, Float16)
        RENORM_CASE(BFloat16, BFloat16)
#undef RENORM_CASE
        default:
            TP_THROW(TypeError, "embedding_renorm_: weight must have a floating dtype");
    }
}

Tensor embedding_cpu(const Tensor& weight, const Tensor& indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
    (void)padding_idx;
    (void)scale_grad_by_freq;
    (void)sparse;
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
    int64_t num_indices = indices.numel();
    int64_t row_size = 1;
    for (size_t i = 1; i < weight_shape.size(); ++i) row_size *= weight_shape[i];
    int64_t weight_size_0 = weight.size(0);
    if (num_indices == 0 || row_size == 0) return output;

    Tensor indices_contig = indices.contiguous();
    Tensor weight_contig = weight.contiguous();

    // Validate every lookup before the copy pass so the parallel region stays
    // exception-free. Negative indices wrap around from the end of the table.
    bool out_of_range = false;
    if (indices.dtype() == DType::Int64) {
        const int64_t* idx_data = indices_contig.data_ptr<int64_t>();
        parallel_for(0, num_indices, 32768, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                int64_t idx = idx_data[i];
                if (idx < 0) idx += weight_size_0;
                if (idx < 0 || idx >= weight_size_0) out_of_range = true;
            }
        });
    } else {
        const int32_t* idx_data = indices_contig.data_ptr<int32_t>();
        parallel_for(0, num_indices, 32768, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                int64_t idx = static_cast<int64_t>(idx_data[i]);
                if (idx < 0) idx += weight_size_0;
                if (idx < 0 || idx >= weight_size_0) out_of_range = true;
            }
        });
    }
    if (out_of_range) {
        TP_THROW(IndexError, "embedding: index out of range");
    }

    dispatch_dtype(weight.dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        const T* weight_data = weight_contig.data_ptr<T>();
        T* out_data = output.data_ptr<T>();
        const int64_t grain = embedding_grain_indices(row_size);

        if (indices.dtype() == DType::Int64) {
            const int64_t* idx_data = indices_contig.data_ptr<int64_t>();
            parallel_for(0, num_indices, grain, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    int64_t idx = idx_data[i];
                    if (idx < 0) idx += weight_size_0;
                    std::memcpy(out_data + i * row_size,
                                weight_data + idx * row_size,
                                row_size * sizeof(T));
                }
            });
        } else { // Int32
            const int32_t* idx_data = indices_contig.data_ptr<int32_t>();
            parallel_for(0, num_indices, grain, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    int64_t idx = static_cast<int64_t>(idx_data[i]);
                    if (idx < 0) idx += weight_size_0;
                    std::memcpy(out_data + i * row_size,
                                weight_data + idx * row_size,
                                row_size * sizeof(T));
                }
            });
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
        if constexpr (is_float8_v<T>) {
            TP_THROW(TypeError, "embedding_dense_backward: grad_output dtype is not supported");
        } else if constexpr (std::is_same_v<T, bool>) {
            TP_THROW(RuntimeError, "embedding_dense_backward: grad_output cannot be Bool");
        } else {
            const T* grad_data = grad_output_contig.data_ptr<T>();
            T* weight_grad_data = grad_weight.data_ptr<T>();

            const bool large = num_indices * row_size >= kEmbeddingSortedBackwardElements;
            if (large || scale_grad_by_freq) {
                // Sorted-segment accumulation: sorting (value, original lookup
                // position) pairs groups every duplicate index into one
                // contiguous run, so each weight row is written by exactly one
                // thread -- no write conflicts, and the per-row occurrence
                // count needed by scale_grad_by_freq falls out of the run
                // length for free. Rows absent from the lookup batch stay at
                // their zero initialization.
                std::vector<std::pair<int64_t, int64_t>> sorted(
                    static_cast<size_t>(num_indices));
                bool out_of_range = false;
                if (indices.dtype() == DType::Int64) {
                    const int64_t* idx_data = indices_contig.data_ptr<int64_t>();
                    parallel_for(0, num_indices, 32768, [&](int64_t begin, int64_t end) {
                        for (int64_t i = begin; i < end; ++i) {
                            int64_t idx = idx_data[i];
                            if (idx == padding_idx) {
                                sorted[i] = {-1, i};
                                continue;
                            }
                            if (idx < 0) idx += num_weights;
                            if (idx < 0 || idx >= num_weights) out_of_range = true;
                            sorted[i] = {idx, i};
                        }
                    });
                } else { // Int32
                    const int32_t* idx_data = indices_contig.data_ptr<int32_t>();
                    parallel_for(0, num_indices, 32768, [&](int64_t begin, int64_t end) {
                        for (int64_t i = begin; i < end; ++i) {
                            int64_t idx = static_cast<int64_t>(idx_data[i]);
                            if (idx == padding_idx) {
                                sorted[i] = {-1, i};
                                continue;
                            }
                            if (idx < 0) idx += num_weights;
                            if (idx < 0 || idx >= num_weights) out_of_range = true;
                            sorted[i] = {idx, i};
                        }
                    });
                }
                if (out_of_range) {
                    TP_THROW(IndexError, "embedding_dense_backward: index out of range");
                }

                std::sort(sorted.begin(), sorted.end(),
                          [](const std::pair<int64_t, int64_t>& a,
                             const std::pair<int64_t, int64_t>& b) {
                              return a.first < b.first;
                          });

                // Record the [begin, end) span of every run of equal indices.
                std::vector<int64_t> seg_row, seg_begin, seg_end;
                for (int64_t i = 0; i < num_indices;) {
                    if (sorted[i].first < 0) { ++i; continue; }
                    const int64_t w = sorted[i].first;
                    int64_t j = i + 1;
                    while (j < num_indices && sorted[j].first == w) ++j;
                    seg_row.push_back(w);
                    seg_begin.push_back(i);
                    seg_end.push_back(j);
                    i = j;
                }

                if (!seg_row.empty()) {
                    const int64_t n_segments = static_cast<int64_t>(seg_row.size());
                    const int64_t grain = embedding_grain_indices(row_size);
                    parallel_for(0, n_segments, grain, [&](int64_t begin, int64_t end) {
                        for (int64_t k = begin; k < end; ++k) {
                            const int64_t w = seg_row[k];
                            const int64_t first = seg_begin[k];
                            const int64_t last = seg_end[k];
                            T* dst = weight_grad_data + w * row_size;
                            const T* src = grad_data + sorted[first].second * row_size;
                            for (int64_t j = 0; j < row_size; ++j) dst[j] = src[j];
                            for (int64_t p = first + 1; p < last; ++p) {
                                const T* s = grad_data + sorted[p].second * row_size;
                                for (int64_t j = 0; j < row_size; ++j) dst[j] += s[j];
                            }
                            if constexpr (std::is_floating_point_v<T>) {
                                if (scale_grad_by_freq) {
                                    const T inv = static_cast<T>(1) /
                                                  static_cast<T>(last - first);
                                    for (int64_t j = 0; j < row_size; ++j) dst[j] *= inv;
                                }
                            }
                        }
                    });
                }
            } else {
                // Small-batch path: a single pass over the lookups without the
                // sort overhead. Duplicate rows accumulate in lookup order.
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
    m.impl("masked_select", masked_select_cpu);
    m.impl("copy_", copy_kernel);
    m.impl("sparse_coo_tensor", sparse_coo_tensor_cpu);
    m.impl("sparse_mask", sparse_mask_cpu);
    m.impl("to_dense", to_dense_sparse_cpu);
    m.impl("to_sparse", to_sparse_coo_cpu);
    m.impl("to_sparse_csr", to_sparse_csr_cpu);
    m.impl("_nnz", sparse_nnz_cpu);
    m.impl("sparse_mm", sparse_mm_cpu);
    m.impl("smm", smm_cpu);
    m.impl("sparse_sum", sparse_sum_cpu);
    m.impl("sparse_add", sparse_add_cpu);
    m.impl("sparse_mul", sparse_mul_cpu);
    m.impl("spdiags", spdiags_cpu);
    m.impl("_spdiags", spdiags_cpu);
    m.impl("_sparse_sum", _sparse_sum_cpu);
    m.impl("_sparse_sum.dtype", _sparse_sum_dtype_cpu);
    m.impl("_sparse_sum.dim", _sparse_sum_dim_cpu_2);
    m.impl("_sparse_sum.dim_dtype", _sparse_sum_dim_dtype_cpu);
    m.impl("_sparse_sum_backward", _sparse_sum_backward_cpu);
    m.impl("native_norm", native_norm_cpu);
    m.impl("native_norm.ScalarOpt_dim_dtype", native_norm_dim_cpu);
    m.impl("embedding", embedding_cpu);
    m.impl("embedding_renorm_", embedding_renorm_cpu);
    m.impl("embedding_dense_backward", embedding_dense_backward_cpu);
    m.impl("embedding_sparse_backward", embedding_sparse_backward_cpu);
    m.impl("embedding_backward", embedding_backward_cpu);
}

} // namespace cpu
} // namespace tensorplay
