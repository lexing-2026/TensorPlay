#include "Tensor.h"
#include "TypePromotion.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Parallel.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "Utils.h"
#include "OneDNNContext.h"
#include "GradMode.h"
#include "LinearAlgebraNames.h"
#include "Complex.h"
#include <vector>
#include <cmath>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <functional>
#include <cstdlib>
#include <optional>

// Dispatcher-level primitives for the linear composite (defined in
// TPXOpsGenerated.cpp; declared locally -- same pattern as Einsum.cpp).
// Dispatcher-level entry points come from the generated TPXOpsGenerated.h
// (included at the top).

// MKL dispatches AMD/Zen hosts to its generic kernel clones -- measured
// ~100x off tuned throughput on Zen4 (512^3 f32 GEMM: 291ms vs ~11ms).
// The documented debug knob doubles as the standard HPC workaround: force
// the closest Intel-tuned ISA family.  Runs once at library load, BEFORE
// the first MKL call initializes its CPU detection; an explicitly exported
// MKL_DEBUG_CPU_LIST always wins (no overwrite).
// GCC/Clang only: inline-asm CPUID, __builtin_cpu_supports, and setenv are
// all absent from MSVC, whose MKL builds ship their own dispatch tuning.
#if defined(USE_MKL) && (defined(__GNUC__) || defined(__clang__))
namespace {
struct MklAmdKernelWorkaround {
    MklAmdKernelWorkaround() {
        // __builtin_cpu_is only knows a vendor whitelist; read CPUID directly.
        unsigned __tp_ebx = 0, __tp_ecx = 0, __tp_edx = 0;
        const bool amd = ([&]() {
#if defined(__x86_64__) || defined(__i386__)
            unsigned leaf0_eax = 0;
            __asm__ volatile("cpuid"
                             : "=a"(leaf0_eax), "=b"(__tp_ebx),
                               "=c"(__tp_ecx), "=d"(__tp_edx)
                             : "a"(0));
            return __tp_ebx == 0x68747541u &&   // "Auth"
                   __tp_ecx == 0x444d4163u &&   // "cAMD"
                   __tp_edx == 0x69746e65u;     // "enti"
#else
            return false;
#endif
        })();
        if (!amd) return;
        if (!__builtin_cpu_supports("avx512f") &&
            !__builtin_cpu_supports("avx2")) return;
        if (std::getenv("MKL_DEBUG_CPU_LIST") != nullptr) return;
        setenv("MKL_DEBUG_CPU_LIST",
               __builtin_cpu_supports("avx512f") ? "SKX" : "CLX", 0);
    }
};
static MklAmdKernelWorkaround g_mkl_amd_workaround;
} // namespace
#endif // USE_MKL && (GCC || Clang)
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>


#ifdef USE_MKL
#include <mkl.h>
#elif defined(USE_BLAS)
#include <cblas.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tensorplay {
namespace cpu {

// Native F.linear CPU kernel; defined below addmm_kernel (its fast path).
Tensor linear_kernel(const Tensor& input, const Tensor& weight,
                     const std::optional<Tensor>& bias_opt);

using namespace tensorplay::parallel;

namespace {

void check_cpu_matmul_dtype(DType dtype) {
    switch (dtype) {
        case DType::Bool:
        case DType::UInt16:
        case DType::UInt32:
        case DType::UInt64:
            // rejection reads "addmm_impl_cpu_" not implemented for 'Bool'.
            TP_THROW(NotImplementedError, "\"addmm_impl_cpu_\" not implemented for '",
                     pretty_dtype_name(dtype), "'");
        case DType::ComplexHalf:
        case DType::BComplex32:
            TP_THROW(NotImplementedError, "\"addmm_impl_cpu_\" not implemented for '",
                     pretty_dtype_name(dtype), "'");
        default:
            return;
    }
}

// Naive matrix multiplication implementation (Optimized Loop Order M-K-N)
void gemm_naive(int64_t M, int64_t N, int64_t K, float alpha, const float* A, int64_t lda, const float* B, int64_t ldb, float beta, float* C, int64_t ldc) {
    // Scale C by beta first
    if (beta == 0.0f) {
        parallel_for(0, M, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            std::memset(C + m * ldc, 0, N * sizeof(float));
        }
        });
    } else if (beta != 1.0f) {
        parallel_for(0, M, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                C[m * ldc + n] *= beta;
            }
        }
        });
    }

    // Accumulate alpha * A * B
    // Loop order M-K-N optimizes cache access for RowMajor matrices B and C
    parallel_for(0, M, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = alpha * A[m * lda + k];
            // Vectorization friendly inner loop
            for (int64_t n = 0; n < N; ++n) {
                C[m * ldc + n] += a_val * B[k * ldb + n];
            }
        }
    }
    });
}

// Generic strided GEMM used by matmul for dtypes that do not have a BLAS or
// oneDNN fast path.  The inputs are deliberately addressed through strides so
// batched matmul can consume transposed and broadcast-selected matrix views
// without flattening an expanded (zero-stride) tensor.
template <typename T>
struct MatmulAccumType {
    using type = T;
};

// Integral matmul returns the input dtype.  Accumulate in unsigned arithmetic
// wide enough to make the two's-complement wrap explicit instead of relying on
template <> struct MatmulAccumType<int8_t> { using type = uint32_t; };
template <> struct MatmulAccumType<int16_t> { using type = uint32_t; };
template <> struct MatmulAccumType<int32_t> { using type = uint64_t; };
#if defined(__SIZEOF_INT128__)
template <> struct MatmulAccumType<int64_t> { using type = unsigned __int128; };
#else
// MSVC on x86-64 has no __int128; the two's-complement wrap of uint64_t
// still yields the low 64 bits the integer matmul keeps.
template <> struct MatmulAccumType<int64_t> { using type = uint64_t; };
#endif
template <> struct MatmulAccumType<uint8_t> { using type = uint32_t; };
template <> struct MatmulAccumType<uint16_t> { using type = uint32_t; };
template <> struct MatmulAccumType<uint32_t> { using type = uint64_t; };
#if defined(__SIZEOF_INT128__)
template <> struct MatmulAccumType<uint64_t> { using type = unsigned __int128; };
#else
template <> struct MatmulAccumType<uint64_t> { using type = uint64_t; };
#endif
template <> struct MatmulAccumType<Half> { using type = float; };
template <> struct MatmulAccumType<BFloat16> { using type = float; };
template <> struct MatmulAccumType<complex<Half>> { using type = complex<float>; };
template <> struct MatmulAccumType<complex<BFloat16>> { using type = complex<float>; };

template <typename T>
typename MatmulAccumType<T>::type matmul_to_accum(const T& value) {
    return static_cast<typename MatmulAccumType<T>::type>(value);
}

inline complex<float> matmul_to_accum(const complex<Half>& value) {
    return {static_cast<float>(value.real()), static_cast<float>(value.imag())};
}

inline complex<float> matmul_to_accum(const complex<BFloat16>& value) {
    return {static_cast<float>(value.real()), static_cast<float>(value.imag())};
}

template <typename T>
void gemm_strided(
    int64_t M, int64_t N, int64_t K,
    const T* A, int64_t a_stride0, int64_t a_stride1,
    const T* B, int64_t b_stride0, int64_t b_stride1,
    T* C, int64_t c_stride0, int64_t c_stride1) {
    using AccT = typename MatmulAccumType<T>::type;
    parallel_for(0, M, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                AccT acc{};
                for (int64_t k = 0; k < K; ++k) {
                    acc += matmul_to_accum(A[m * a_stride0 + k * a_stride1]) *
                           matmul_to_accum(B[k * b_stride0 + n * b_stride1]);
                }
                C[m * c_stride0 + n * c_stride1] = static_cast<T>(acc);
            }
        }
    });
}

void gemm_strided_dispatch(
    const Tensor& self, const Tensor& other, Tensor& result,
    int64_t M, int64_t N, int64_t K) {
    if (self.dtype() == DType::Bool) {
        TP_THROW(NotImplementedError, "matmul: Bool is not supported on CPU");
    }

#define MATMUL_CASE(ctype, name) \
    case DType::name: \
        gemm_strided<ctype>(M, N, K, \
            self.data_ptr<ctype>(), self.stride(0), self.stride(1), \
            other.data_ptr<ctype>(), other.stride(0), other.stride(1), \
            result.data_ptr<ctype>(), result.stride(0), result.stride(1)); \
        return;

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(MATMUL_CASE)
        case DType::ComplexHalf:
            gemm_strided<complex<Half>>(
                M, N, K, self.data_ptr<complex<Half>>(), self.stride(0), self.stride(1),
                other.data_ptr<complex<Half>>(), other.stride(0), other.stride(1),
                result.data_ptr<complex<Half>>(), result.stride(0), result.stride(1));
            return;
        case DType::ComplexFloat:
            gemm_strided<complex<float>>(
                M, N, K, self.data_ptr<complex<float>>(), self.stride(0), self.stride(1),
                other.data_ptr<complex<float>>(), other.stride(0), other.stride(1),
                result.data_ptr<complex<float>>(), result.stride(0), result.stride(1));
            return;
        case DType::ComplexDouble:
            gemm_strided<complex<double>>(
                M, N, K, self.data_ptr<complex<double>>(), self.stride(0), self.stride(1),
                other.data_ptr<complex<double>>(), other.stride(0), other.stride(1),
                result.data_ptr<complex<double>>(), result.stride(0), result.stride(1));
            return;
        case DType::BComplex32:
            gemm_strided<complex<BFloat16>>(
                M, N, K, self.data_ptr<complex<BFloat16>>(), self.stride(0), self.stride(1),
                other.data_ptr<complex<BFloat16>>(), other.stride(0), other.stride(1),
                result.data_ptr<complex<BFloat16>>(), result.stride(0), result.stride(1));
            return;
        default:
            TP_THROW(NotImplementedError, "matmul: unsupported dtype on CPU");
    }
#undef MATMUL_CASE
}

// The generic TensorIterator binary path is deliberately dtype-polymorphic,
// but its per-element BF16/F16 conversion is too expensive for the GEMM
// epilogue in Muon's Newton-Schulz loop.  Keep this narrow helper local to
// addmm: one pass over the already-contiguous result, with the same
// round-to-dtype store semantics as the public pointwise operation.
template <typename T>
void low_precision_gemm_epilogue(Tensor& result, const Tensor& seed,
                                 bool has_seed, float alpha, float beta) {
    T* dst = result.data_ptr<T>();
    const T* src = has_seed ? seed.data_ptr<T>() : nullptr;
    const int64_t n = result.numel();
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        if (has_seed) {
            if (alpha == 1.0f) {
                for (int64_t i = begin; i < end; ++i) {
                    dst[i] = static_cast<T>(static_cast<float>(dst[i]) +
                                             beta * static_cast<float>(src[i]));
                }
            } else {
                for (int64_t i = begin; i < end; ++i) {
                    dst[i] = static_cast<T>(alpha * static_cast<float>(dst[i]) +
                                             beta * static_cast<float>(src[i]));
                }
            }
        } else if (alpha != 1.0f) {
            for (int64_t i = begin; i < end; ++i) {
                dst[i] = static_cast<T>(alpha * static_cast<float>(dst[i]));
            }
        }
    });
}

#ifdef USE_ONEDNN
// oneDNN primitive cache -- building a matmul::primitive_desc triggers JIT
// kernel selection costing microseconds, which dominates tiny GEMMs (the
// same reason the cache avoids repeated primitive construction.  Keyed by
// dims+strides+dtype of
// all three operands; bounded to keep pathological shape variety honest.
using namespace dnnl;

namespace {
std::mutex g_mm_pd_cache_mutex;
struct VecKeyHash {
    size_t operator()(const std::vector<int64_t>& v) const {
        size_t h = 1469598103934665603ull;
        for (const auto x : v) {
            h ^= static_cast<size_t>(x) + 0x9e3779b97f4a7c15ull +
                 (h << 6) + (h >> 2);
        }
        return h;
    }
};

std::unordered_map<std::vector<int64_t>,
    std::pair<std::unique_ptr<matmul::primitive_desc>,
              std::shared_ptr<matmul>>, VecKeyHash>* g_mm_pd_cache =
    nullptr;

// Returns the shared, ready-to-execute primitive for the key.  Caching the
// constructed primitive (not just its descriptor) removes the per-call
// primitive re-wrap and refcount churn that dominated per-node dispatch
// overhead once pd selection was already cached; primitives are immutable
// after construction and safe to share under DNNL_ENABLE_CONCURRENT_EXEC.
// `accum_sum_beta` bakes the addmm-style sum post-op into the primitive:
// execution then computes prim_result + beta * dst_initial_contents.
const matmul& cached_matmul_prim(
        engine& eng, const memory::desc& s, const memory::desc& w,
        const memory::desc& d, std::vector<int64_t> key,
        std::optional<float> accum_sum_beta = std::nullopt,
        std::optional<float> output_alpha = std::nullopt) {
    std::lock_guard<std::mutex> lk(g_mm_pd_cache_mutex);
    if (!g_mm_pd_cache) {
        g_mm_pd_cache =
            new std::unordered_map<std::vector<int64_t>,
                std::pair<std::unique_ptr<matmul::primitive_desc>,
                          std::shared_ptr<matmul>>, VecKeyHash>();
    }
    auto& cache = *g_mm_pd_cache;
    if (accum_sum_beta) {
        key.push_back(1);
        float f = *accum_sum_beta;
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        key.push_back(static_cast<int64_t>(bits));
    } else {
        key.push_back(0);
        key.push_back(0);
    }
    if (output_alpha) {
        key.push_back(1);
        float f = *output_alpha;
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        key.push_back(static_cast<int64_t>(bits));
    } else {
        key.push_back(0);
        key.push_back(0);
    }
    auto it = cache.find(key);
    if (it != cache.end() && it->second.first) return *it->second.second;
    if (cache.size() >= 1024) cache.clear();  // simple bound
    primitive_attr attr;
    post_ops po;
    bool has_post_ops = false;
    if (output_alpha && *output_alpha != 1.0f) {
        // Linear is applied to the GEMM result before the sum post-op, giving
        // alpha * (mat1 @ mat2) + beta * seed in one low-precision primitive.
        po.append_eltwise(algorithm::eltwise_linear, *output_alpha, 0.0f);
        has_post_ops = true;
    }
    if (accum_sum_beta && *accum_sum_beta != 0.0f) {
        po.append_sum(*accum_sum_beta);
        has_post_ops = true;
    }
    if (has_post_ops) {
        attr.set_post_ops(po);
    }
    auto pd = std::make_unique<matmul::primitive_desc>(eng, s, w, d, attr);
    auto prim = std::make_shared<matmul>(*pd);
    auto entry = std::make_pair(std::move(pd), std::move(prim));
    const matmul& ref = *entry.second;
    cache.emplace(std::move(key), std::move(entry));
    return ref;
}

std::vector<int64_t> pd_key(const memory::desc& s, const memory::desc& w,
                            const memory::desc& d) {
    std::vector<int64_t> key;
    key.reserve(24);
    for (const auto v : s.get_dims()) key.push_back(v);
    for (const auto v : s.get_strides()) key.push_back(v);
    for (const auto v : w.get_dims()) key.push_back(v);
    for (const auto v : w.get_strides()) key.push_back(v);
    for (const auto v : d.get_dims()) key.push_back(v);
    for (const auto v : d.get_strides()) key.push_back(v);
    key.push_back(static_cast<int64_t>(s.get_data_type()));
    return key;
}
} // namespace

// oneDNN natively accelerates bf16/f16 matmul (f32 accumulate) -- routing
// only f32 here left autocast GEMMs on a scalar fallback ~40x slower than
inline memory::data_type onednn_matmul_dt(DType t) {
    switch (t) {
        case DType::Float32:  return memory::data_type::f32;
        case DType::BFloat16: return memory::data_type::bf16;
        case DType::Float16:  return memory::data_type::f16;
        default:              return memory::data_type::f32;
    }
}

inline bool onednn_matmul_dtype_ok(DType t) {
    return t == DType::Float32 || t == DType::BFloat16 || t == DType::Float16;
}

// Keep the layout tag visible to oneDNN for the two dense layouts it can
// consume without a reorder.  An explicit-stride descriptor is semantically
// operand as `ba`; preserving that tag lets the primitive select its packed
// transposed-weight kernel (important for Muon's X @ X.T loop).
inline memory::desc onednn_2d_desc(const memory::dims& dims,
                                    memory::data_type dt,
                                    int64_t stride0, int64_t stride1) {
    if (dims.size() == 2) {
        if (stride0 == dims[1] && stride1 == 1) {
            return memory::desc(dims, dt, memory::format_tag::ab);
        }
        if (stride0 == 1 && stride1 == dims[0]) {
            return memory::desc(dims, dt, memory::format_tag::ba);
        }
    }
    return memory::desc(dims, dt, {stride0, stride1});
}

// Fused bias/beta GEMM epilogue: executes mat1 @ mat2 accumulating into a
// freshly allocated dst pre-seeded with `seed` (the broadcast-add operand),
// using the sum post-op so the final value is dst_initial * sum_scale +
// mat1@mat2 -- addmm's exact definition when alpha == 1, without ever
// materializing a separate mm temporary or a standalone elementwise add
// pass.  Weight strides are consumed natively (transposed views stay views).
std::optional<Tensor> addmm_onednn(const Tensor& input, const Tensor& mat1,
                                   const Tensor& mat2, double beta,
                                   double alpha,
                                   const Tensor& seed) {
    const auto dt_in = mat1.dtype();
    if (!onednn_matmul_dtype_ok(dt_in)) return std::nullopt;
    if (dt_in != mat2.dtype() || dt_in != input.dtype()) return std::nullopt;

    const int64_t M = mat1.size(0);
    const int64_t N = mat2.size(1);
    const bool has_seed = seed.defined();

    try {
        auto& engine = OneDNNContext::get_engine();
        auto& stream = OneDNNContext::get_stream();
        const memory::data_type dt = onednn_matmul_dt(dt_in);

        // Source/destination live in the freshly seeded buffer; the kernel
        // writes dst in place on top of the broadcast copy.
        Tensor src_contig =
            mat1.is_contiguous() ? mat1 : mat1.contiguous();

        memory::dims src_dims = {M, static_cast<int64_t>(mat1.size(1))};
        memory::dims src_strides = {src_contig.stride(0), src_contig.stride(1)};
        auto src_md = onednn_2d_desc(src_dims, dt,
                                     src_strides[0], src_strides[1]);

        memory::dims weights_dims = {static_cast<int64_t>(mat2.size(0)), N};
        memory::dims weights_strides = {mat2.stride(0), mat2.stride(1)};
        auto weights_md = onednn_2d_desc(weights_dims, dt,
                                         weights_strides[0], weights_strides[1]);

        memory::dims dst_dims = {M, N};
        memory::dims dst_strides = {N, 1};
        auto dst_md = onednn_2d_desc(dst_dims, dt,
                                     dst_strides[0], dst_strides[1]);

        auto key = pd_key(src_md, weights_md, dst_md);
        std::optional<float> sum_scale;
        if (has_seed && beta != 0.0) sum_scale = static_cast<float>(beta);
        const std::optional<float> output_alpha = static_cast<float>(alpha);
        const auto& prim = cached_matmul_prim(engine, src_md, weights_md,
                                              dst_md, std::move(key),
                                              sum_scale, output_alpha);

        auto src_mem = memory(src_md, engine, src_contig.data_ptr());
        auto wei_mem = memory(weights_md, engine, mat2.data_ptr());
        void* dst_ptr;
        Tensor dst_holder;
        if (has_seed) {
            // seed was materialized contiguous {M,N} by the caller
            dst_holder = seed;
            dst_ptr = dst_holder.data_ptr();
        } else {
            dst_holder = Tensor::empty({M, N}, dt_in, mat1.device());
            dst_ptr = dst_holder.data_ptr();
        }
        auto dst_mem = memory(dst_md, engine, dst_ptr);

        prim.execute(stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_WEIGHTS, wei_mem},
            {DNNL_ARG_DST, dst_mem}
        });
        stream.wait();
        return dst_holder;
    } catch (dnnl::error& e) {
        return std::nullopt;
    }
}


bool mm_onednn(const Tensor& self, const Tensor& mat2, Tensor& result) {
    if (!OneDNNContext::is_enabled()) return false;
    if (!onednn_matmul_dtype_ok(self.dtype()) || self.dtype() != mat2.dtype() ||
        result.dtype() != self.dtype()) return false;

    // Dimensions
    int64_t M = self.size(0);
    int64_t K = self.size(1);
    int64_t N = mat2.size(1);

    try {
        auto& engine = OneDNNContext::get_engine();
        auto& stream = OneDNNContext::get_stream();

        const memory::data_type dt = onednn_matmul_dt(self.dtype());

        // Memory descriptors with explicit strides
        memory::dims src_dims = {M, K};
        memory::dims src_strides = {self.stride(0), self.stride(1)};
        auto src_md = onednn_2d_desc(src_dims, dt,
                                     src_strides[0], src_strides[1]);

        memory::dims weights_dims = {K, N};
        memory::dims weights_strides = {mat2.stride(0), mat2.stride(1)};
        auto weights_md = onednn_2d_desc(weights_dims, dt,
                                         weights_strides[0], weights_strides[1]);

        memory::dims dst_dims = {M, N};
        memory::dims dst_strides = {result.stride(0), result.stride(1)};
        auto dst_md = onednn_2d_desc(dst_dims, dt,
                                     dst_strides[0], dst_strides[1]);

        // Create memories sharing data pointers
        auto src_mem = memory(src_md, engine, self.data_ptr());
        auto weights_mem = memory(weights_md, engine, mat2.data_ptr());
        auto dst_mem = memory(dst_md, engine, result.data_ptr());

        // Cached primitive (JIT selection happens once per shape)
        const auto& matmul_prim = cached_matmul_prim(
            engine, src_md, weights_md, dst_md,
            pd_key(src_md, weights_md, dst_md));

        matmul_prim.execute(stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_WEIGHTS, weights_mem},
            {DNNL_ARG_DST, dst_mem}
        });

        stream.wait();
        return true;
    } catch (dnnl::error& e) {
        return false;
    }
}

bool matmul_onednn(const Tensor& src, const Tensor& weights, Tensor& dst) {
    if (!OneDNNContext::is_enabled()) return false;
    if (!onednn_matmul_dtype_ok(src.dtype()) || src.dtype() != weights.dtype() ||
        dst.dtype() != src.dtype()) return false;

    try {
        auto& engine = OneDNNContext::get_engine();
        auto& stream = OneDNNContext::get_stream();

        // Convert shapes and strides to memory::dims
        memory::dims src_dims = static_cast<std::vector<int64_t>>(src.shape());
        memory::dims src_strides = static_cast<std::vector<int64_t>>(src.strides());
        const memory::data_type mdt = onednn_matmul_dt(src.dtype());
        auto src_md = memory::desc(src_dims, mdt, src_strides);

        memory::dims weights_dims = static_cast<std::vector<int64_t>>(weights.shape());
        memory::dims weights_strides = static_cast<std::vector<int64_t>>(weights.strides());
        auto weights_md = memory::desc(weights_dims, mdt, weights_strides);

        memory::dims dst_dims = static_cast<std::vector<int64_t>>(dst.shape());
        memory::dims dst_strides = static_cast<std::vector<int64_t>>(dst.strides());
        auto dst_md = memory::desc(dst_dims, mdt, dst_strides);

        // Create memories sharing data pointers
        auto src_mem = memory(src_md, engine, src.data_ptr());
        auto weights_mem = memory(weights_md, engine, weights.data_ptr());
        auto dst_mem = memory(dst_md, engine, dst.data_ptr());

        // Cached primitive (JIT selection happens once per shape)
        const auto& matmul_prim = cached_matmul_prim(
            engine, src_md, weights_md, dst_md,
            pd_key(src_md, weights_md, dst_md));

        matmul_prim.execute(stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_WEIGHTS, weights_mem},
            {DNNL_ARG_DST, dst_mem}
        });

        stream.wait();
        return true;
    } catch (dnnl::error& e) {
        return false;
    }
}
#endif

} // anonymous namespace

// Core GEMM writing into a caller-owned contiguous {M,N} result.  Shared by
// mm_kernel (fresh alloc) and the batched-matmul loops (output slice views)
// so the broadcast path never pays temp-alloc + copy_ per slice.
static void mm_into_impl(const Tensor& self_p, const Tensor& mat2_p,
                         Tensor& result) {
    const int64_t M = self_p.size(0);
    const int64_t K = self_p.size(1);
    const int64_t N = mat2_p.size(1);
    if (K == 0) {
        // oneDNN and some BLAS implementations leave C untouched for a
        // beta*C + 0 and beta is zero for mm.
        result.fill_(Scalar(0));
        return;
    }

    if (self_p.dtype() == DType::Float16) {
        // fp16 matmul drops to a reference kernel (measured ~800x slower than
        // (its shgemm fallback converts the same way); do the same.
        Tensor a32 = self_p.to(DType::Float32);
        Tensor b32 = mat2_p.to(DType::Float32);
        Tensor r32 = Tensor::empty({M, N}, DType::Float32, result.device());
        mm_into_impl(a32, b32, r32);
        result.copy_(r32);
        return;
    }

    // Small GEMMs go straight to the (ISA-tuned via MKL_DEBUG_CPU_LIST) BLAS
    // call: oneDNN's pd-cache lookup + JIT launch + threaded-runner sync cost
    // more than the GEMM itself below ~8k MACs (Zen4: 2.8us vs 7.7us), and
    // still runs ~2x over MKL sgemm through the 32^3-64^3 band, while at
    // >=96^3 oneDNN wins again.  The crossover is shape/system dependent, so
    // the threshold is an env knob (TP_MM_ONEDNN_MIN_MACS) with a measured
    // default.
    static const int64_t onednn_min_macs = [] {
        if (const char* e = std::getenv("TP_MM_ONEDNN_MIN_MACS")) {
            char* end = nullptr;
            const long long v = std::strtoll(e, &end, 10);
            if (end != e && *end == '\0' && v >= 0) return static_cast<int64_t>(v);
        }
        return static_cast<int64_t>(524288);
    }();
    #ifdef USE_ONEDNN
    // Low-precision GEMM always prefers oneDNN (native bf16/f16 kernels with
    // f32 accumulate); the scalar gemm_strided fallback is never competitive.
    // f32 keeps the small-shape MKL shortcut via the MACs threshold.
    const bool mm_low_precision = self_p.dtype() != DType::Float32;
    // Decode is dominated by M=1 (and occasionally N=1) skinny GEMMs.
    // oneDNN's primitive/runner path is slower than the already-linked BLAS
    // kernel for these shapes; keep oneDNN for low precision and fat FP32
    // matrices where its JIT path is beneficial.
    const bool skinny_fp32 = self_p.dtype() == DType::Float32 && (M == 1 || N == 1);
    if ((mm_low_precision || (!skinny_fp32 && (M * K * N) >= onednn_min_macs)) &&
        mm_onednn(self_p, mat2_p, result)) {
        return;
    }
    #endif

    if (self_p.dtype() == DType::Float32 && mat2_p.dtype() == DType::Float32) {
        
        bool transA = false;
        bool transB = false;
        int64_t lda = 0;
        int64_t ldb = 0;
        
        // Helper to check for transposed layout (stride(0)=1, stride(1)=size(0))
        // This corresponds to a Column-Major layout of a matrix of size (size(0), size(1))
        // or a Transposed view of a Row-Major matrix.
        auto is_transposed = [](const Tensor& t) {
            return t.stride(0) == 1 && t.stride(1) == t.size(0);
        };
        
        Tensor a_input = self_p;
        if (self_p.is_contiguous()) {
            lda = K;
        } else if (is_transposed(self_p)) {
            transA = true;
            lda = M; 
        } else {
            a_input = detail::contiguous_clone(self_p);
            lda = K;
        }
        
        Tensor b_input = mat2_p;
        if (mat2_p.is_contiguous()) {
            ldb = N;
        } else if (is_transposed(mat2_p)) {
            transB = true;
            ldb = K;
        } else {
            b_input = detail::contiguous_clone(mat2_p);
            ldb = N;
        }
        
        const float* A = a_input.data_ptr<float>();
        const float* B = b_input.data_ptr<float>();
        float* C = result.data_ptr<float>();
        
        #if defined(USE_MKL) || defined(USE_BLAS)
            CBLAS_TRANSPOSE TransA = transA ? CblasTrans : CblasNoTrans;
            CBLAS_TRANSPOSE TransB = transB ? CblasTrans : CblasNoTrans;
            
            #ifdef USE_MKL
            cblas_sgemm(CblasRowMajor, TransA, TransB, 
                        M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, N);
            #else
            cblas_sgemm(CblasRowMajor, TransA, TransB, 
                        (int)M, (int)N, (int)K, 1.0f, A, (int)lda, B, (int)ldb, 0.0f, C, (int)N);
            #endif
            
        #else
            // Fallback to naive (no transpose support in naive yet, force clone)
            if (transA || transB) {
                 Tensor a_contig = self_p.contiguous();
                 Tensor b_contig = mat2_p.contiguous();
                 gemm_naive(M, N, K, 1.0f, a_contig.data_ptr<float>(), K, b_contig.data_ptr<float>(), N, 0.0f, C, N);
            } else {
                 gemm_naive(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
            }
        #endif
        return;
    }

    if (self_p.dtype() == DType::Float64 && mat2_p.dtype() == DType::Float64) {
#ifdef USE_MKL
        bool transA = false;
        bool transB = false;
        int64_t lda = 0;
        int64_t ldb = 0;
        auto is_transposed = [](const Tensor& t) {
            return t.stride(0) == 1 && t.stride(1) == t.size(0);
        };
        Tensor a_input = self_p;
        if (self_p.is_contiguous()) {
            lda = K;
        } else if (is_transposed(self_p)) {
            transA = true;
            lda = M;
        } else {
            a_input = detail::contiguous_clone(self_p);
            lda = K;
        }
        Tensor b_input = mat2_p;
        if (mat2_p.is_contiguous()) {
            ldb = N;
        } else if (is_transposed(mat2_p)) {
            transB = true;
            ldb = K;
        } else {
            b_input = detail::contiguous_clone(mat2_p);
            ldb = N;
        }
        cblas_dgemm(CblasRowMajor,
                    transA ? CblasTrans : CblasNoTrans,
                    transB ? CblasTrans : CblasNoTrans,
                    M, N, K, 1.0, a_input.data_ptr<double>(), lda,
                    b_input.data_ptr<double>(), ldb, 0.0,
                    result.data_ptr<double>(), N);
        return;
#elif defined(USE_BLAS)
        // LP64 reference CBLAS: narrow the sizes.
        bool transA = false;
        bool transB = false;
        int64_t lda = K;
        int64_t ldb = N;
        Tensor a_input = self_p.contiguous();
        Tensor b_input = mat2_p.contiguous();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)M, (int)N, (int)K, 1.0,
                    a_input.data_ptr<double>(), (int)lda,
                    b_input.data_ptr<double>(), (int)ldb, 0.0,
                    result.data_ptr<double>(), (int)N);
        return;
#endif
    }
    gemm_strided_dispatch(self_p, mat2_p, result, M, N, K);
}

Tensor mm_kernel(const Tensor& self, const Tensor& mat2) {
    if (self.dim() != 2) TP_THROW(RuntimeError, "self must be a matrix");
    if (mat2.dim() != 2) TP_THROW(RuntimeError, "mat2 must be a matrix");
    if (self.size(1) != mat2.size(0)) {
        TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (", self.size(0), "x", self.size(1),
                 " and ", mat2.size(0), "x", mat2.size(1), ")");
    }

    // dtype.  This is intentionally stricter than elementwise promotion.
    if (self.dtype() != mat2.dtype()) {
        TP_THROW(RuntimeError, "expected m1 and m2 to have the same dtype, but got: ",
                 c10_style_dtype_name(self.dtype()), " != ", c10_style_dtype_name(mat2.dtype()));
    }
    check_cpu_matmul_dtype(self.dtype());
    Tensor result = Tensor::empty({self.size(0), mat2.size(1)}, self.dtype(), self.device());
    mm_into_impl(self, mat2, result);
    return result;
}

// reports mismatches through its expand wording.  Returns a (possibly
// zero-stride) view; callers materialize with clone() before mutating.
Tensor expand_gemm_input(const Tensor& input, const std::vector<int64_t>& target) {
    const int64_t td = target.size();
    const int64_t sd = input.dim();
    if (sd > td) {
        TP_THROW(RuntimeError, "expand: the number of sizes provided (", td,
                 ") must be greater or equal to the number of dimensions in the tensor (", sd, ")");
    }

    std::vector<int64_t> src(td, 1);
    std::vector<int64_t> src_strides(td, 0);
    for (int64_t i = 0; i < sd; ++i) {
        src[td - 1 - i] = input.size(sd - 1 - i);
        src_strides[td - 1 - i] = input.stride(sd - 1 - i);
    }

    // The offending-size text shows the tensor's real sizes, not the
    // left-padded broadcast view.
    for (int64_t k = td - 1; k >= 0; --k) {
        if (src[k] != 1 && src[k] != target[k]) {
            std::string tgt = "[";
            std::string own = "[";
            const auto own_sizes = static_cast<std::vector<int64_t>>(input.shape());
            for (size_t d = 0; d < td; ++d) {
                if (d) { tgt += ", "; }
                tgt += std::to_string(target[d]);
            }
            for (size_t d = 0; d < own_sizes.size(); ++d) {
                if (d) { own += ", "; }
                own += std::to_string(own_sizes[d]);
            }
            tgt += "]";
            own += "]";
            TP_THROW(RuntimeError, "The expanded size of the tensor (", target[k],
                     ") must match the existing size (", src[k], ") at non-singleton dimension ",
                     k, ".  Target sizes: ", tgt, ".  Tensor sizes: ", own);
        }
    }

    std::vector<int64_t> out_strides(td);
    for (int64_t k = 0; k < td; ++k) {
        out_strides[k] = (src[k] == 1 && target[k] != 1) ? 0 : src_strides[k];
    }
    return input.as_strided(target, out_strides);
}

Tensor addmm_kernel(const Tensor& input, const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
    if (mat1.dim() != 2 || mat2.dim() != 2) TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (",
        mat1.dim(), "D and ", mat2.dim(), "D)");
    if (mat1.size(1) != mat2.size(0)) {
        TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (", mat1.size(0), "x", mat1.size(1),
                 " and ", mat2.size(0), "x", mat2.size(1), ")");
    }
    // self-vs-mat2 first, then mat1-vs-mat2 (LinearAlgebra.cpp:185-186).
    if (input.dtype() != mat2.dtype()) {
        TP_THROW(RuntimeError, "self and mat2 must have the same dtype, but got ",
                 pretty_dtype_name(input.dtype()), " and ", pretty_dtype_name(mat2.dtype()));
    }
    if (mat1.dtype() != mat2.dtype()) {
        TP_THROW(RuntimeError, "mat1 and mat2 must have the same dtype, but got ",
                 pretty_dtype_name(mat1.dtype()), " and ", pretty_dtype_name(mat2.dtype()));
    }

    int64_t M = mat1.size(0);
    int64_t N = mat2.size(1);
    double alpha_v = alpha.toDouble();
    double beta_v = beta.toDouble();

    // out = beta * input + alpha * (mat1 @ mat2)
    //
    // Fused-seed strategy: materialize the broadcast of `input` ONCE as the
    // destination buffer, then run a single accumulating GEMM over it --
    // BLAS carries alpha/beta natively; oneDNN bakes beta into the sum
    // post-op.  This removes the standalone mm temporary allocation and the
    // full-MxN pointwise add pass the previous mm_kernel()+add() composite
    // paid on every call.
    const bool has_seed = beta_v != 0.0;
    Tensor seed;
    if (has_seed) {
        // Left unscaled: beta is applied by the GEMM accumulate itself
        // (BLAS beta param / oneDNN sum scale), saving a mul_ pass here.
        seed = detail::contiguous_clone(expand_gemm_input(input, {M, N}));
    }

#ifdef USE_ONEDNN
    {
        const int64_t macs = M * N * mat1.size(1);
        // Same routing policy as mm_into_impl: oneDNN owns low-precision and
        // fat FP32 shapes; small FP32 goes to the already-linked BLAS call.
        static const int64_t onednn_min_macs = [] {
            if (const char* e = std::getenv("TP_MM_ONEDNN_MIN_MACS")) {
                char* end = nullptr;
                const long long v = std::strtoll(e, &end, 10);
                if (end != e && *end == '\0' && v >= 0)
                    return static_cast<int64_t>(v);
            }
            return static_cast<int64_t>(524288);
        }();
        const bool low_precision = mat1.dtype() != DType::Float32;
        const bool skinny_f32 = !low_precision && (M == 1 || N == 1);
        if (low_precision || (!skinny_f32 && macs >= onednn_min_macs)) {
        if (low_precision) {
                // Muon's Newton-Schulz update uses both alpha and beta.  Let
                // oneDNN fuse the linear alpha post-op and the beta sum into
                // the low-precision matmul; this avoids a scalar conversion
                // pass over the full BF16/F16 result.
                if (auto fused = addmm_onednn(
                        input, mat1, mat2, beta_v, alpha_v, seed))
                    return *fused;
            }
            if (alpha_v == 1.0) {
                // oneDNN's sum post-op is unexpectedly expensive for BF16/F16
                // addmm on this CPU (several times the standalone GEMM).  The
                // low-precision split above avoids it; retain the fused
                // primitive for FP32 and unusual layouts as a fallback.
                if (auto fused = addmm_onednn(
                        input, mat1, mat2, beta_v, alpha_v, seed))
                    return *fused;
            }
        }
    }
#endif

    if (mat1.dtype() == DType::Float32 && mat2.dtype() == DType::Float32 &&
        input.dtype() == DType::Float32) {
#if defined(USE_MKL) || defined(USE_BLAS)
        bool transA = false;
        bool transB = false;
        int64_t lda = 0;
        int64_t ldb = 0;
        auto is_transposed = [](const Tensor& t) {
            return t.stride(0) == 1 && t.stride(1) == t.size(0);
        };
        Tensor a_input = mat1;
        if (mat1.is_contiguous()) {
            lda = mat1.size(1);
        } else if (is_transposed(mat1)) {
            transA = true;
            lda = M;
        } else {
            a_input = detail::contiguous_clone(mat1);
            lda = mat1.size(1);
        }
        Tensor b_input = mat2;
        if (mat2.is_contiguous()) {
            ldb = N;
        } else if (is_transposed(mat2)) {
            transB = true;
            ldb = mat1.size(1);
        } else {
            b_input = detail::contiguous_clone(mat2);
            ldb = N;
        }

        Tensor result = has_seed ? seed : Tensor::empty({M, N}, mat1.dtype(), mat1.device());
        float* C = result.data_ptr<float>();
        const float fa = static_cast<float>(alpha_v);
        const float fb = static_cast<float>(beta_v);
        #ifdef USE_MKL
        cblas_sgemm(CblasRowMajor,
                    transA ? CblasTrans : CblasNoTrans,
                    transB ? CblasTrans : CblasNoTrans,
                    M, N, mat1.size(1), fa, a_input.data_ptr<float>(), lda,
                    b_input.data_ptr<float>(), ldb, fb, C, N);
        #else
        cblas_sgemm(CblasRowMajor,
                    transA ? CblasTrans : CblasNoTrans,
                    transB ? CblasTrans : CblasNoTrans,
                    (int)M, (int)N, (int)mat1.size(1), fa,
                    a_input.data_ptr<float>(), (int)lda,
                    b_input.data_ptr<float>(), (int)ldb, fb, C, (int)N);
        #endif
        return result;
#endif  // USE_MKL || USE_BLAS
    }

    if (mat1.dtype() == DType::Float64 && mat2.dtype() == DType::Float64 &&
        input.dtype() == DType::Float64) {
#ifdef USE_MKL
        bool transA = false;
        bool transB = false;
        int64_t lda = 0;
        int64_t ldb = 0;
        auto is_transposed = [](const Tensor& t) {
            return t.stride(0) == 1 && t.stride(1) == t.size(0);
        };
        Tensor a_input = mat1;
        if (mat1.is_contiguous()) {
            lda = mat1.size(1);
        } else if (is_transposed(mat1)) {
            transA = true;
            lda = M;
        } else {
            a_input = detail::contiguous_clone(mat1);
            lda = mat1.size(1);
        }
        Tensor b_input = mat2;
        if (mat2.is_contiguous()) {
            ldb = N;
        } else if (is_transposed(mat2)) {
            transB = true;
            ldb = mat1.size(1);
        } else {
            b_input = detail::contiguous_clone(mat2);
            ldb = N;
        }

        Tensor result = has_seed ? seed : Tensor::empty({M, N}, mat1.dtype(), mat1.device());
        double* C = result.data_ptr<double>();
        cblas_dgemm(CblasRowMajor,
                    transA ? CblasTrans : CblasNoTrans,
                    transB ? CblasTrans : CblasNoTrans,
                    M, N, mat1.size(1), alpha_v, a_input.data_ptr<double>(), lda,
                    b_input.data_ptr<double>(), ldb, beta_v, C, N);
        return result;
#endif
    }

    // Generic tail for dtypes without an accelerated accumulator (int8 etc.)
    // and builds without BLAS: decompose exactly as before.  A fresh seed is
    // materialized here because a failing accelerated route may have left
    // the previous buffer's contents unspecified.
    Tensor result = Tensor::empty({M, N}, mat1.dtype(), mat1.device());
    if (has_seed) {
        // seed was already expanded to {M,N} (broadcast copy of bias) by the
        // caller above; copy its contents in one shot instead of re-expanding.
        result.copy_(seed);
        if (beta_v != 1.0) result.mul_(beta);
    }
    Tensor mm = mm_kernel(mat1, mat2);
    if (alpha_v != 1.0) {
        mm = mm.mul(alpha_v);
    }
    result = result.add(mm, 1.0);
    return result;
}

std::vector<int64_t> decode_batch_index(
    int64_t linear_index, const std::vector<int64_t>& batch_shape) {
    std::vector<int64_t> index(batch_shape.size(), 0);
    for (int64_t dim = static_cast<int64_t>(batch_shape.size()) - 1; dim >= 0; --dim) {
        const int64_t size = batch_shape[dim];
        index[dim] = size == 0 ? 0 : linear_index % size;
        if (size != 0) linear_index /= size;
    }
    return index;
}

// Select one matrix from a possibly broadcast batch.  Dimensions are removed
// from high to low so the original dimension numbers remain valid.
Tensor select_batch_matrix(
    const Tensor& input,
    const std::vector<int64_t>& input_batch_shape,
    const std::vector<int64_t>& output_batch_shape,
    const std::vector<int64_t>& output_index) {
    Tensor matrix = input;
    const int64_t padding = static_cast<int64_t>(output_batch_shape.size()) -
                            static_cast<int64_t>(input_batch_shape.size());
    for (int64_t dim = static_cast<int64_t>(input_batch_shape.size()) - 1; dim >= 0; --dim) {
        const int64_t output_dim = padding + dim;
        const int64_t index = input_batch_shape[dim] == 1 ? 0 : output_index[output_dim];
        matrix = matrix.select(dim, index);
    }
    return matrix;
}

Tensor select_output_matrix(
    const Tensor& output,
    const std::vector<int64_t>& output_index) {
    Tensor matrix = output;
    for (int64_t dim = static_cast<int64_t>(output_index.size()) - 1; dim >= 0; --dim) {
        matrix = matrix.select(dim, output_index[dim]);
    }
    return matrix;
}

Tensor matmul_batched_2d(
    const Tensor& self, const Tensor& other,
    const std::vector<int64_t>& self_batch_shape,
    const std::vector<int64_t>& other_batch_shape) {
    if (self.dim() < 2 || other.dim() < 2) {
        TP_THROW(RuntimeError, "matmul: internal operands must be at least 2D");
    }
    const int64_t M = self.size(self.dim() - 2);
    const int64_t K = self.size(self.dim() - 1);
    if (K != other.size(other.dim() - 2)) {
        TP_THROW(RuntimeError, "matmul: size mismatch, got ", K, " and ", other.size(other.dim() - 2));
    }
    const int64_t N = other.size(other.dim() - 1);
    const std::vector<int64_t> batch_shape = broadcast_shapes(self_batch_shape, other_batch_shape);

    std::vector<int64_t> result_shape = batch_shape;
    result_shape.push_back(M);
    result_shape.push_back(N);
    Tensor result = Tensor::empty(result_shape, self.dtype(), self.device());

    int64_t batch_size = 1;
    for (const int64_t size : batch_shape) {
        if (size == 0) return result;
        if (batch_size > std::numeric_limits<int64_t>::max() / size) {
            TP_THROW(RuntimeError, "matmul: output batch size overflow");
        }
        batch_size *= size;
    }

    if (K == 0) {
        return result.fill_(Scalar(0));
    }

    // Preserve the oneDNN N-D fast path when neither operand needs an
    // implicit broadcast/padding. Expanded views can have zero strides,
    // which oneDNN rightfully rejects.
#ifdef USE_ONEDNN
    if (self_batch_shape == batch_shape && other_batch_shape == batch_shape &&
        self.dim() == static_cast<int64_t>(batch_shape.size()) + 2 &&
        other.dim() == static_cast<int64_t>(batch_shape.size()) + 2 &&
        matmul_onednn(self, other, result)) {
        return result;
    }
#endif

    // Small or thin slices: per-call BLAS thread fan-up dominates, so spread
    // gets from a single batched-BLAS call.  Large, fat slices already fill
    // the machine on their own and stay on the serial path below.
    const int64_t slice_flops = M * N * K;
    const int64_t threads = parallel::get_num_threads();
    const bool batch_parallel =
        batch_size > 4 && threads > 1 && batch_size >= threads &&
        (slice_flops <= 131072 || M == 1 || N == 1);

    if (batch_parallel) {
        const int64_t grain = std::max<int64_t>(1, batch_size / (threads * 4));
        parallel::parallel_for(0, batch_size, grain, [&](int64_t begin, int64_t end) {
            for (int64_t linear = begin; linear < end; ++linear) {
                const std::vector<int64_t> output_index = decode_batch_index(linear, batch_shape);
                Tensor self_matrix = select_batch_matrix(self, self_batch_shape, batch_shape, output_index);
                Tensor other_matrix = select_batch_matrix(other, other_batch_shape, batch_shape, output_index);
                Tensor out_matrix = select_output_matrix(result, output_index);
                mm_into_impl(self_matrix, other_matrix, out_matrix);
            }
        });
        return result;
    }

    for (int64_t linear = 0; linear < batch_size; ++linear) {
        const std::vector<int64_t> output_index = decode_batch_index(linear, batch_shape);
        Tensor self_matrix = select_batch_matrix(self, self_batch_shape, batch_shape, output_index);
        Tensor other_matrix = select_batch_matrix(other, other_batch_shape, batch_shape, output_index);
        Tensor out_matrix = select_output_matrix(result, output_index);
        mm_into_impl(self_matrix, other_matrix, out_matrix);
    }
    return result;
}

Tensor matmul_kernel(const Tensor& self, const Tensor& other) {
    const int64_t dim1 = self.dim();
    const int64_t dim2 = other.dim();
    if (self.dim() < 1 || other.dim() < 1) {
        TP_THROW(RuntimeError, "matmul(): input operands must be at least 1D");
    }

    // vector/batch dimensions into matrix rows before reporting mismatches,
    // so the reported shapes depend on operand ranks:
    //   vec @ mat      -> "mat1 and mat2 shapes cannot be multiplied (1xN and KxM)"
    //   mat @ vec      -> "size mismatch, got input (M), mat (MxK), vec (V)"
    //   batched @ mat  -> batch*M folded into rows of mat1
    //   * @ batched    -> "Expected size for first two dimensions of batch2 tensor ..."
    const auto self_shape = static_cast<std::vector<int64_t>>(self.shape());
    const auto other_shape = static_cast<std::vector<int64_t>>(other.shape());

    // (per-slice select views + temp result + copy_ back) that otherwise
    // costs ~3us on top of the GEMM itself.
    if (dim1 == 2 && dim2 == 2) {
        return mm_kernel(self, other);
    }

    if (dim1 == 1 && dim2 == 1) {
        if (self.size(0) != other.size(0)) {
            TP_THROW(RuntimeError, "inconsistent tensor size, expected tensor [", self.size(0),
                     "] and src [", other.size(0),
                     "] to have the same number of elements, but got ", self.size(0), " and ",
                     other.size(0), " elements respectively");
        }
    } else if (dim1 == 1) {
        const int64_t k = self.size(0);
        const int64_t other_k = other_shape[other_shape.size() - 2];
        if (k != other_k) {
            TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (1x", k,
                     " and ", other_k, "x", other_shape.back(), ")");
        }
    } else if (dim2 == 1) {
        int64_t folded_m = self_shape[self_shape.size() - 2];
        for (size_t i = 0; i + 2 < self_shape.size(); ++i) folded_m *= self_shape[i];
        const int64_t k = self_shape.back();
        if (k != other.size(0)) {
            TP_THROW(RuntimeError, "size mismatch, got input (", folded_m, "), mat (", folded_m,
                     "x", k, "), vec (", other.size(0), ")");
        }
    } else {
        const int64_t k = self_shape.back();
        const int64_t other_k = other_shape[other_shape.size() - 2];
        if (k != other_k) {
            if (dim2 == 2) {
                int64_t folded_m = self_shape[self_shape.size() - 2];
                for (size_t i = 0; i + 2 < self_shape.size(); ++i) folded_m *= self_shape[i];
                TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (", folded_m, "x", k,
                         " and ", other_k, "x", other_shape.back(), ")");
            } else {
                std::vector<int64_t> self_batch(self_shape.begin(), self_shape.end() - 2);
                std::vector<int64_t> other_batch(other_shape.begin(), other_shape.end() - 2);
                const std::vector<int64_t> batch = broadcast_shapes(self_batch, other_batch);
                int64_t prod_batch = 1;
                for (const int64_t s : batch) prod_batch *= s;
                TP_THROW(RuntimeError, "Expected size for first two dimensions of batch2 tensor to be: [",
                         prod_batch, ", ", k, "] but got: [", prod_batch, ", ", other_k, "].");
            }
        }
    }

    if (self.dtype() != other.dtype()) {
        TP_THROW(RuntimeError, "expected m1 and m2 to have the same dtype, but got: ",
                 c10_style_dtype_name(self.dtype()), " != ", c10_style_dtype_name(other.dtype()));
    }
    check_cpu_matmul_dtype(self.dtype());
    const Tensor& self_p = self;
    const Tensor& other_p = other;

    if (dim1 == 1 && dim2 == 1) {
        if (self_p.size(0) != other_p.size(0)) {
            TP_THROW(RuntimeError, "matmul: size mismatch, got ", self_p.size(0), " and ", other_p.size(0));
        }
        Tensor result = matmul_batched_2d(
            self_p.unsqueeze(0), other_p.unsqueeze(1), {}, {});
        return result.squeeze(0).squeeze(0);
    }

    if (dim1 == 1) {
        std::vector<int64_t> other_batch_shape(other_shape.begin(), other_shape.end() - 2);
        Tensor result = matmul_batched_2d(
            self_p.unsqueeze(0), other_p, {}, other_batch_shape);
        return result.squeeze(-2);
    }

    if (dim2 == 1) {
        std::vector<int64_t> self_batch_shape(self_shape.begin(), self_shape.end() - 2);
        Tensor result = matmul_batched_2d(
            self_p, other_p.unsqueeze(-1), self_batch_shape, {});
        return result.squeeze(-1);
    }

    std::vector<int64_t> self_batch_shape(self_shape.begin(), self_shape.end() - 2);
    std::vector<int64_t> other_batch_shape(other_shape.begin(), other_shape.end() - 2);
    return matmul_batched_2d(self_p, other_p, self_batch_shape, other_batch_shape);
}

Tensor& matmul_out_kernel(const Tensor& self, const Tensor& other, Tensor& out) {
    if (out.device() != self.device()) {
        TP_THROW(DeviceMismatchError, "matmul: out tensor must be on the same device as the inputs");
    }
    if (out.dtype() != self.dtype()) {
        TP_THROW(RuntimeError, "Expected out tensor to have dtype ",
                 static_cast<int>(self.dtype()), ", but got ",
                 static_cast<int>(out.dtype()));
    }
    if (GradMode::is_enabled() &&
        (self.requires_grad() || other.requires_grad() || out.requires_grad())) {
        TP_THROW(RuntimeError,
                 "matmul(): functions with out=... arguments don't support automatic differentiation, "
                 "but one of the arguments requires grad.");
    }

    Tensor result = matmul_kernel(self, other);
    if (out.shape() == result.shape()) {
        out.copy_(result);
    } else {
        // Out variants resize a write-only destination to the inferred shape.
        // Replacing metadata also gives the resized tensor fresh contiguous
        // storage, matching TensorIterator's write-only resize contract.
        out.unsafeGetTensorImpl()->copy_metadata_from(*result.unsafeGetTensorImpl());
    }
    return out;
}

namespace {

Tensor transpose_last_two_view(const Tensor& input) {
    if (input.dim() < 2) {
        TP_THROW(RuntimeError, "matmul backward: expected a matrix operand");
    }
    std::vector<int64_t> sizes = static_cast<std::vector<int64_t>>(input.shape());
    std::vector<int64_t> strides = input.strides();
    std::swap(sizes[sizes.size() - 2], sizes[sizes.size() - 1]);
    std::swap(strides[strides.size() - 2], strides[strides.size() - 1]);
    return input.as_strided(sizes, strides);
}

template <typename Component>
Tensor conjugate_contiguous_cpu(const Tensor& input) {
    Tensor result = Tensor::empty(
        static_cast<std::vector<int64_t>>(input.shape()), input.dtype(), input.device());
    const auto* src = input.data_ptr<complex<Component>>();
    auto* dst = result.data_ptr<complex<Component>>();
    parallel_for(0, input.numel(), GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i)
            dst[i] = complex<Component>(src[i].real(), -src[i].imag());
    });
    return result;
}

Tensor adjoint_last_two_cpu(const Tensor& input) {
    Tensor transposed = transpose_last_two_view(input);
    if (input.dtype() == DType::ComplexFloat) {
        return conjugate_contiguous_cpu<float>(transposed.contiguous());
    }
    if (input.dtype() == DType::ComplexDouble) {
        return conjugate_contiguous_cpu<double>(transposed.contiguous());
    }
    return transposed;
}

template <typename T>
void sum_to_shape_recursive(
    const T* src, const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& src_strides,
    T* dst, const std::vector<int64_t>& dst_shape,
    const std::vector<int64_t>& dst_strides,
    int64_t dim, int64_t src_offset, int64_t dst_offset) {
    if (dim == static_cast<int64_t>(src_shape.size())) {
        dst[dst_offset] = static_cast<T>(
            matmul_to_accum(dst[dst_offset]) + matmul_to_accum(src[src_offset]));
        return;
    }

    const int64_t leading = static_cast<int64_t>(src_shape.size()) -
                            static_cast<int64_t>(dst_shape.size());
    const int64_t dst_dim = dim - leading;
    const bool reduce = dst_dim < 0 || dst_shape[dst_dim] == 1;
    const int64_t dst_stride = (dst_dim < 0 || reduce) ? 0 : dst_strides[dst_dim];
    for (int64_t i = 0; i < src_shape[dim]; ++i) {
        sum_to_shape_recursive(
            src, src_shape, src_strides, dst, dst_shape, dst_strides,
            dim + 1, src_offset + i * src_strides[dim],
            dst_offset + (reduce ? 0 : i * dst_stride));
    }
}

Tensor sum_to_shape_cpu(const Tensor& input, const std::vector<int64_t>& target_shape) {
    const auto source_shape = static_cast<std::vector<int64_t>>(input.shape());
    if (target_shape.size() > source_shape.size()) {
        TP_THROW(RuntimeError, "matmul backward: target rank exceeds source rank");
    }
    const int64_t leading = static_cast<int64_t>(source_shape.size()) -
                            static_cast<int64_t>(target_shape.size());
    for (size_t i = 0; i < target_shape.size(); ++i) {
        const int64_t source_dim = source_shape[leading + static_cast<int64_t>(i)];
        if (target_shape[i] != 1 && target_shape[i] != source_dim) {
            TP_THROW(RuntimeError, "matmul backward: incompatible gradient shape");
        }
    }

    if (source_shape == target_shape) return input;
    Tensor result = Tensor::zeros(target_shape, input.dtype(), input.device());

#define SUM_TO_CASE(ctype, name) \
    case DType::name: \
        sum_to_shape_recursive<ctype>( \
            input.data_ptr<ctype>(), source_shape, input.strides(), \
            result.data_ptr<ctype>(), target_shape, result.strides(), 0, 0, 0); \
        return result;

    switch (input.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(SUM_TO_CASE)
        case DType::ComplexHalf:
            sum_to_shape_recursive<complex<Half>>(
                input.data_ptr<complex<Half>>(), source_shape, input.strides(),
                result.data_ptr<complex<Half>>(), target_shape, result.strides(), 0, 0, 0);
            return result;
        case DType::ComplexFloat:
            sum_to_shape_recursive<complex<float>>(
                input.data_ptr<complex<float>>(), source_shape, input.strides(),
                result.data_ptr<complex<float>>(), target_shape, result.strides(), 0, 0, 0);
            return result;
        case DType::ComplexDouble:
            sum_to_shape_recursive<complex<double>>(
                input.data_ptr<complex<double>>(), source_shape, input.strides(),
                result.data_ptr<complex<double>>(), target_shape, result.strides(), 0, 0, 0);
            return result;
        case DType::BComplex32:
            sum_to_shape_recursive<complex<BFloat16>>(
                input.data_ptr<complex<BFloat16>>(), source_shape, input.strides(),
                result.data_ptr<complex<BFloat16>>(), target_shape, result.strides(), 0, 0, 0);
            return result;
        default:
            TP_THROW(NotImplementedError, "matmul backward: unsupported dtype");
    }
#undef SUM_TO_CASE
}

struct MatmulBackwardInputs {
    Tensor self_matrix;
    Tensor other_matrix;
    Tensor grad_matrix;
    std::vector<int64_t> self_shape;
    std::vector<int64_t> other_shape;
    bool self_vector = false;
    bool other_vector = false;
};

MatmulBackwardInputs normalize_matmul_backward_inputs(
    const Tensor& grad_output, const Tensor& self, const Tensor& other) {
    MatmulBackwardInputs normalized;
    normalized.self_vector = self.dim() == 1;
    normalized.other_vector = other.dim() == 1;
    normalized.self_shape = static_cast<std::vector<int64_t>>(self.shape());
    normalized.other_shape = static_cast<std::vector<int64_t>>(other.shape());
    normalized.self_matrix = normalized.self_vector ? self.unsqueeze(0) : self;
    normalized.other_matrix = normalized.other_vector ? other.unsqueeze(-1) : other;
    normalized.grad_matrix = grad_output;

    if (normalized.self_vector && normalized.other_vector) {
        normalized.grad_matrix = grad_output.unsqueeze(0).unsqueeze(0);
    } else if (normalized.self_vector) {
        normalized.grad_matrix = grad_output.unsqueeze(-2);
    } else if (normalized.other_vector) {
        normalized.grad_matrix = grad_output.unsqueeze(-1);
    }
    return normalized;
}

} // namespace

Tensor matmul_backward_self_kernel(
    const Tensor& grad_output, const Tensor& self, const Tensor& other) {
    const MatmulBackwardInputs normalized =
        normalize_matmul_backward_inputs(grad_output, self, other);
    Tensor grad = matmul_kernel(
        normalized.grad_matrix,
        adjoint_last_two_cpu(normalized.other_matrix));
    std::vector<int64_t> target_shape = static_cast<std::vector<int64_t>>(
        normalized.self_matrix.shape());
    grad = sum_to_shape_cpu(grad, target_shape);
    if (normalized.self_vector) grad = grad.squeeze(0);
    if (grad.dtype() != self.dtype()) grad = grad.to(self.dtype());
    return grad;
}

Tensor matmul_backward_other_kernel(
    const Tensor& grad_output, const Tensor& self, const Tensor& other) {
    const MatmulBackwardInputs normalized =
        normalize_matmul_backward_inputs(grad_output, self, other);
    Tensor grad = matmul_kernel(
        adjoint_last_two_cpu(normalized.self_matrix),
        normalized.grad_matrix);
    std::vector<int64_t> target_shape = static_cast<std::vector<int64_t>>(
        normalized.other_matrix.shape());
    grad = sum_to_shape_cpu(grad, target_shape);
    if (normalized.other_vector) grad = grad.squeeze(-1);
    if (grad.dtype() != other.dtype()) grad = grad.to(other.dtype());
    return grad;
}

namespace {

// True when every batch item is a row-major or transposed-view matrix, so a
// per-slice GEMM can consume the stack through strides without a copy.
bool batch_items_contiguous_or_transposed(const Tensor& t) {
    if (t.dim() != 3) return false;
    const auto sizes = static_cast<std::vector<int64_t>>(t.shape());
    const auto strides = t.strides();
    // Per-slice row-major (column stride 1, row stride covering the inner
    // extent) or its transposed-storage mirror (row stride 1, column stride
    // covering the outer extent).  Both must hold strictly: a size-1 inner
    // dimension with arbitrary strides can satisfy the stride pattern while
    // violating CBLAS's lead-dimension validation.
    return (strides[2] == 1 && strides[1] >= sizes[2]) ||
           (strides[1] == 1 && strides[2] >= sizes[1]);
}

template <typename T>
void bgemm_batch(const Tensor& batch1, const Tensor& batch2, Tensor& result,
                 double alpha, double beta);

Tensor bmm_kernel(const Tensor& self, const Tensor& batch2) {
    if (self.dim() != 3) TP_THROW(RuntimeError, "batch1 must be a 3D tensor");
    if (batch2.dim() != 3) TP_THROW(RuntimeError, "batch2 must be a 3D tensor");
    // (B, M, K) @ (B, K, N): batch2's leading dims must match [B, K].
    if (self.size(0) != batch2.size(0) || self.size(2) != batch2.size(1)) {
        TP_THROW(RuntimeError, "Expected size for first two dimensions of batch2 tensor to be: [",
                 self.size(0), ", ", self.size(2), "] but got: [", batch2.size(0), ", ",
                 batch2.size(1), "].");
    }
    if (self.dtype() != batch2.dtype()) {
        TP_THROW(RuntimeError, "expected scalar type ", pretty_dtype_name(self.dtype()),
                 " but found ", pretty_dtype_name(batch2.dtype()));
    }
    check_cpu_matmul_dtype(self.dtype());
    // Small/thin slices keep the parallel batched-2d path; fat contiguous
    // stacks go through the fused batched GEMM without view-select round
    // trips.
    if ((self.dtype() == DType::Float32 || self.dtype() == DType::Float64) &&
        batch_items_contiguous_or_transposed(self) &&
        batch_items_contiguous_or_transposed(batch2)) {
        const int64_t B = self.size(0), M = self.size(1), N = batch2.size(2);
        const int64_t macs = B * self.size(1) * self.size(2) * N;
        if (macs >= 4096 || (M == 1 || N == 1 || self.size(2) == 1)) {
            Tensor result = Tensor::empty({B, M, N}, self.dtype(), self.device());
            if (self.dtype() == DType::Float32) {
                bgemm_batch<float>(self, batch2, result, 1.0, 0.0);
            } else {
                bgemm_batch<double>(self, batch2, result, 1.0, 0.0);
            }
            return result;
        }
    }
    return matmul_batched_2d(self, batch2, {self.size(0)}, {batch2.size(0)});
}

// One batched GEMM over per-slice raw pointers (batch dimension folded out):
// result[b] = alpha * op(A[b]) @ op(B[b]) + beta * result[b].  Operands must
// satisfy batch_items_contiguous_or_transposed and result must be
// contiguous; f32/f64 only (low precision keeps the opmath batched path).
template <typename T>
void bgemm_batch(const Tensor& batch1, const Tensor& batch2, Tensor& result,
                 double alpha, double beta) {
    const int64_t B = result.size(0);
    const int64_t M = result.size(1), N = result.size(2);
    const int64_t K = batch1.size(2);
    const auto s1 = batch1.strides();
    const auto s2 = batch2.strides();
    const auto sr = result.strides();
    // Row-major CBLAS arguments; the per-slice matrix strides carry any
    // transposed views directly.  Each orientation carries a lead-dimension
    // requirement: NoTrans needs the row stride to cover the operand's inner
    // extent, Trans needs the column stride to cover the outer extent.  When
    // both strides are 1 (a degenerate inner dimension) either flag is
    // semantically fine but only one satisfies CBLAS validation, so pick by
    // the requirement instead of by stride pattern.
    auto pick_a = [&](bool& trans, int64_t& ld) {
        if (s1[2] == 1 && s1[1] >= K) { trans = false; ld = s1[1]; return; }
        trans = true; ld = s1[2];
    };
    auto pick_b = [&](bool& trans, int64_t& ld) {
        if (s2[2] == 1 && s2[1] >= N) { trans = false; ld = s2[1]; return; }
        trans = true; ld = s2[2];
    };
    bool trans_a; int64_t lda;
    bool trans_b; int64_t ldb;
    pick_a(trans_a, lda);
    pick_b(trans_b, ldb);
    const T* A = batch1.data_ptr<T>();
    const T* Bp = batch2.data_ptr<T>();
    T* C = result.data_ptr<T>();
    const T alpha_t = static_cast<T>(alpha);
    const T beta_t = static_cast<T>(beta);
#ifdef USE_MKL
    // The grouped interface uses MKL's native index type.  Keep the
    // per-slice path for oversized tensors and for nonzero beta, whose
    // historical first-slice/next-slice semantics cannot be represented by
    // one grouped call.
    const auto mkl_int_fits = [](int64_t value) {
        return value >= static_cast<int64_t>(std::numeric_limits<MKL_INT>::min()) &&
               value <= static_cast<int64_t>(std::numeric_limits<MKL_INT>::max());
    };
    if (B > 1 && beta_t == T(0) &&
        mkl_int_fits(B) && mkl_int_fits(M) && mkl_int_fits(N) &&
        mkl_int_fits(K) && mkl_int_fits(lda) && mkl_int_fits(ldb) &&
        mkl_int_fits(sr[1])) {
        std::vector<const T*> a_ptrs(static_cast<size_t>(B));
        std::vector<const T*> b_ptrs(static_cast<size_t>(B));
        std::vector<T*> c_ptrs(static_cast<size_t>(B));
        for (int64_t bi = 0; bi < B; ++bi) {
            a_ptrs[static_cast<size_t>(bi)] = A + bi * s1[0];
            b_ptrs[static_cast<size_t>(bi)] = Bp + bi * s2[0];
            c_ptrs[static_cast<size_t>(bi)] = C + bi * sr[0];
        }
        const CBLAS_TRANSPOSE TA = trans_a ? CblasTrans : CblasNoTrans;
        const CBLAS_TRANSPOSE TB = trans_b ? CblasTrans : CblasNoTrans;
        const MKL_INT m = static_cast<MKL_INT>(M);
        const MKL_INT n = static_cast<MKL_INT>(N);
        const MKL_INT k = static_cast<MKL_INT>(K);
        const MKL_INT lda_mkl = static_cast<MKL_INT>(lda);
        const MKL_INT ldb_mkl = static_cast<MKL_INT>(ldb);
        const MKL_INT ldc_mkl = static_cast<MKL_INT>(sr[1]);
        const MKL_INT group_count = 1;
        const MKL_INT group_size = static_cast<MKL_INT>(B);
        if constexpr (std::is_same_v<T, float>) {
            cblas_sgemm_batch(
                CblasRowMajor, &TA, &TB, &m, &n, &k, &alpha_t,
                a_ptrs.data(), &lda_mkl, b_ptrs.data(), &ldb_mkl,
                &beta_t, c_ptrs.data(), &ldc_mkl, group_count, &group_size);
        } else {
            cblas_dgemm_batch(
                CblasRowMajor, &TA, &TB, &m, &n, &k, &alpha_t,
                a_ptrs.data(), &lda_mkl, b_ptrs.data(), &ldb_mkl,
                &beta_t, c_ptrs.data(), &ldc_mkl, group_count, &group_size);
        }
        return;
    }
#endif
    for (int64_t bi = 0; bi < B; ++bi) {
        const int64_t a_off = bi * s1[0];
        const int64_t b_off = bi * s2[0];
#if defined(USE_MKL) || defined(USE_BLAS)
        // beta == 0 must stay zero for every slice (bmm); only a nonzero
        // beta turns into an accumulate chain (baddbmm).
        const T beta_i = (bi == 0 || beta_t == T(0)) ? beta_t : T(1);
        const int64_t ldc = sr[1];
        const CBLAS_TRANSPOSE TA = trans_a ? CblasTrans : CblasNoTrans;
        const CBLAS_TRANSPOSE TB = trans_b ? CblasTrans : CblasNoTrans;
        if constexpr (std::is_same_v<T, float>) {
            cblas_sgemm(CblasRowMajor, TA, TB, (int)M, (int)N, (int)K,
                        alpha_t, A + a_off, (int)lda, Bp + b_off, (int)ldb,
                        beta_i, C + bi * sr[0], (int)ldc);
        } else {
            cblas_dgemm(CblasRowMajor, TA, TB, (int)M, (int)N, (int)K,
                        alpha_t, A + a_off, (int)lda, Bp + b_off, (int)ldb,
                        beta_i, C + bi * sr[0], (int)ldc);
        }
#else
        (void)alpha_t; (void)beta_t;
        gemm_strided<T>(M, N, K,
                        A + a_off, trans_a ? 1 : lda, trans_a ? lda : 1,
                        Bp + b_off, trans_b ? 1 : ldb, trans_b ? ldb : 1,
                        C + bi * sr[0], sr[1], sr[2]);
#endif
    }
}

}  // namespace

Tensor baddbmm_kernel(const Tensor& input, const Tensor& batch1, const Tensor& batch2,
                      Scalar beta, Scalar alpha) {
    // against `input`, so a wrong K shows up as an expand error.
    if (batch1.dim() != 3) TP_THROW(RuntimeError, "batch1 must be a 3D tensor");
    if (batch2.dim() != 3) TP_THROW(RuntimeError, "batch2 must be a 3D tensor");
    if (batch1.size(0) != batch2.size(0) || batch1.size(2) != batch2.size(1)) {
        TP_THROW(RuntimeError, "Expected size for first two dimensions of batch2 tensor to be: [",
                 batch1.size(0), ", ", batch1.size(2), "] but got: [", batch2.size(0), ", ",
                 batch2.size(1), "].");
    }
    if (input.dtype() != batch1.dtype() || batch1.dtype() != batch2.dtype()) {
        TP_THROW(RuntimeError, "Input dtypes must be the same, got: input ",
                 c10_style_dtype_name(input.dtype()), ", batch1: ",
                 c10_style_dtype_name(batch1.dtype()), ", batch2: ",
                 c10_style_dtype_name(batch2.dtype()));
    }
    check_cpu_matmul_dtype(batch1.dtype());

    const int64_t B = batch1.size(0);
    const int64_t M = batch1.size(1);
    const int64_t N = batch2.size(2);
    const double beta_v = beta.toDouble();
    const double alpha_v = alpha.toDouble();

    const std::vector<int64_t> target{B, M, N};

#if defined(USE_MKL) || defined(USE_BLAS)
    // Fused batched GEMM: seed materializes the broadcast of `input` once,
    // then a single GEMM chain per slice applies beta/alpha natively -- no
    // standalone (B, M, N) product and no pointwise add pass.
    const bool fused_ok =
        (batch1.dtype() == DType::Float32 || batch1.dtype() == DType::Float64) &&
        batch_items_contiguous_or_transposed(batch1) &&
        batch_items_contiguous_or_transposed(batch2);
    if (fused_ok) {
        // GEMM always runs with beta = 0 into a fresh buffer (MKL's
        // beta != 0 kernel selection is markedly slower here); the seed
        // folds in through one pointwise pass afterwards.
        Tensor result = Tensor::empty(target, batch1.dtype(), batch1.device());
        if (B == 0 || M == 0 || N == 0) {
            if (beta_v != 0.0 && B > 0 && M > 0 && N > 0) {
                result.copy_(detail::contiguous_clone(expand_gemm_input(input, target)));
                if (beta_v != 1.0) result.mul_(beta);
            } else if (beta_v != 0.0 && (M == 0 || N == 0) && batch1.size(2) != 0) {
                // (B, M, 0) x (B, 0, N) contraction over an empty axis: the
                // product contributes nothing, beta * seed remains.
            }
            return result;
        }
        if (batch1.size(2) == 0) {
            // Empty contraction: the product contributes nothing.
            result.copy_(detail::contiguous_clone(expand_gemm_input(input, target)));
            if (beta_v != 1.0) result.mul_(beta);
            return result;
        }
        Tensor product = matmul_batched_2d(batch1, batch2, {B}, {B});
        if (alpha_v != 1.0) product.mul_(alpha);
        return beta_v != 0.0 ? product.add_(input, beta) : product;
    }
#endif

    Tensor result;
    if (beta_v == 0.0) {
        result = Tensor::empty(target, batch1.dtype(), batch1.device());
    } else {
        result = detail::contiguous_clone(expand_gemm_input(input, target));
        if (beta_v != 1.0) result.mul_(beta);
    }

    Tensor product = bmm_kernel(batch1, batch2);
    if (alpha_v != 1.0) product = product.mul(alpha_v);
    return result.add(product, 1.0);
}

// Row-range matrix-vector product over tp's own thread pool.  Handles both
// row-major and transposed-view layouts; used for small/medium mv workloads
// where BLAS-pool wake latency outweighs its kernel efficiency.
template <typename acc_t>
Tensor gemv_rows(const Tensor& self, const Tensor& vec) {
    const int64_t M = self.size(0), K = self.size(1);
    const acc_t* base = self.data_ptr<acc_t>();
    const int64_t s_m = self.stride(0), s_k = self.stride(1);
    const acc_t* x = vec.data_ptr<acc_t>();
    Tensor result = Tensor::zeros({M}, self.dtype(), self.device());
    acc_t* y = result.data_ptr<acc_t>();
    const int64_t threads = parallel::get_num_threads();
    if (M * K <= 2048) {
        // Tiny: thread hand-off would dominate; a plain serial loop runs in
        // low single-digit microseconds, matching BLAS hot-path latency.
        for (int64_t m = 0; m < M; ++m) {
            const acc_t* row = base + m * s_m;
            acc_t acc{};
            for (int64_t k = 0; k < K; ++k) {
                acc += row[k * s_k] * x[k];
            }
            y[m] = acc;
        }
        return result;
    }
    if (s_k == 1) {
        // Row-major: each thread owns whole rows (sequential k, unit stride).
        const int64_t grain = std::max<int64_t>(1, M / (threads * 4));
        parallel::parallel_for(0, M, grain, [&](int64_t begin, int64_t end) {
            for (int64_t m = begin; m < end; ++m) {
                const acc_t* row = base + m * s_m;
                acc_t acc{};
                for (int64_t k = 0; k < K; ++k) {
                    acc += row[k] * x[k];
                }
                y[m] = acc;
            }
        });
    } else {
        // Transposed view: walk columns so A reads at unit stride and the
        // accumulator vector stays hot in cache.
        const int64_t grain = std::max<int64_t>(1, K / (threads * 4));
        parallel::parallel_for(0, K, grain, [&](int64_t begin, int64_t end) {
            for (int64_t k = begin; k < end; ++k) {
                const acc_t xk = x[k];
                const acc_t* col = base + k * s_k;
                for (int64_t m = 0; m < M; ++m) {
                    y[m] += col[m * s_m] * xk;
                }
            }
        });
    }
    return result;
}

Tensor parallel_gemv(const Tensor& self, const Tensor& vec) {
    if (self.dtype() == DType::Float32) return gemv_rows<float>(self, vec);
    if (self.dtype() == DType::Float64) return gemv_rows<double>(self, vec);
    TP_THROW(RuntimeError, "parallel_gemv: unsupported dtype");
}

Tensor mv_kernel(const Tensor& self, const Tensor& vec) {
    if (self.dim() != 2 || vec.dim() != 1) {
        TP_THROW(RuntimeError, "vector + matrix @ vector expected, got ", 1, ", ",
                 self.dim(), ", ", vec.dim());
    }
    if (self.size(1) != vec.size(0)) {
        TP_THROW(RuntimeError, "size mismatch, got input (", self.size(0), "), mat (",
                 self.size(0), "x", self.size(1), "), vec (", vec.size(0), ")");
    }
    if (self.dtype() != vec.dtype()) {
        // The leading value is addmv's accumulator slot; via mv it echoes vec.
        TP_THROW(RuntimeError, "addmv input tensors must have the same dtype, but got ",
                 pretty_dtype_name(vec.dtype()), ", ", pretty_dtype_name(self.dtype()), ", and ",
                 pretty_dtype_name(vec.dtype()));
    }
    check_cpu_matmul_dtype(self.dtype());

    // bmm-shaped wrapper's extra alloc/copy round-trip.  Both row-major and
    // transposed-view layouts are accepted without a copy, following
    // mm_kernel's stride policy; anything else falls through to the generic
    // batched path below.
#if defined(USE_MKL) || defined(USE_BLAS)
    if (self.dtype() == DType::Float32 || self.dtype() == DType::Float64) {
        const bool is_f32 = self.dtype() == DType::Float32;
        const int64_t M = self.size(0), K = self.size(1);
        const bool row_major = self.is_contiguous();
        const bool col_major_view = self.stride(0) == 1 && self.stride(1) == M;
        if (M != 0 && K != 0 && (row_major || col_major_view) && vec.is_contiguous()) {
            // Mid-size workloads run the row-range reduction on tp's own
            // always-warm intra-op pool: external-BLAS pool wake/sleep jitter
            // makes repeated small sgemv calls bimodal (fast when hot,
            // >100us after the workers park), which dominates real
            // einsum/attention workloads.  Tiny sizes stay serial (thread
            // hand-off would dominate); large ones go to BLAS, whose tuned
            // kernels win there.
            const int64_t mv_elems = M * K;
            if (mv_elems <= 2048) {
                // Tiny: serial self-kernel avoids every dispatch/hand-off
                // cost and runs in low single-digit microseconds.
                return parallel_gemv(self, vec);
            }
            // runtime), which is healthy now that both stacks share gomp.
            // Express both layouts in straight Fortran col-major terms (what
            // the BLAS underneath validates): a row-major MxK matrix reads as
            // a col-major KxM operand needing transposition; a transposed
            // view's buffer already is the col-major MxK matrix.
            Tensor result = Tensor::empty({M}, self.dtype(), self.device());
            const CBLAS_TRANSPOSE trans = row_major ? CblasTrans : CblasNoTrans;
            const int64_t cm_m = row_major ? K : M;   // col-major rows
            const int64_t cm_n = row_major ? M : K;   // col-major cols
            const int64_t lda = row_major ? K : M;
            if (is_f32) {
                cblas_sgemv(CblasColMajor, trans, (int)cm_m, (int)cm_n, 1.0f,
                            self.data_ptr<float>(), (int)lda,
                            vec.data_ptr<float>(), 1, 0.0f,
                            result.data_ptr<float>(), 1);
            } else {
                cblas_dgemv(CblasColMajor, trans, (int)cm_m, (int)cm_n, 1.0,
                            self.data_ptr<double>(), (int)lda,
                            vec.data_ptr<double>(), 1, 0.0,
                            result.data_ptr<double>(), 1);
            }
            return result;
        }
        if ((M == 0 || K == 0)) {
            return Tensor::zeros({M}, self.dtype(), self.device());
        }
    }
#endif
    return matmul_batched_2d(self, vec.unsqueeze(-1), {}, {}).squeeze(-1);
}

Tensor dot_kernel(const Tensor& self, const Tensor& other) {
    if (self.dim() != 1 || other.dim() != 1) {
        TP_THROW(RuntimeError, "1D tensors expected, but got ", self.dim(), "D and ",
                 other.dim(), "D tensors");
    }
    if (self.size(0) != other.size(0)) {
        TP_THROW(RuntimeError, "inconsistent tensor size, expected tensor [", self.size(0),
                 "] and src [", other.size(0),
                 "] to have the same number of elements, but got ", self.size(0), " and ",
                 other.size(0), " elements respectively");
    }
    if (self.dtype() != other.dtype()) {
        TP_THROW(RuntimeError, "dot : expected both vectors to have same dtype, but found ",
                 pretty_dtype_name(self.dtype()), " and ", pretty_dtype_name(other.dtype()));
    }

    const int64_t n = self.numel();
    Tensor result = Tensor::empty({}, self.dtype(), self.device());
    const auto accumulate = [&](auto&& body) {
        using Acc = decltype(body(int64_t{0}));
        Acc total{};
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            Acc part{};
            for (int64_t i = begin; i < end; ++i) part += body(i);
            // Serial combine keeps the reduction deterministic.
            static std::mutex m;
            std::lock_guard<std::mutex> lock(m);
            total += part;
        });
        return total;
    };

#define DOT_CASE(ctype, name) \
    case DType::name: { \
        if constexpr (std::is_same_v<ctype, bool>) { \
            TP_THROW(NotImplementedError, "\"dot\" not implemented for 'Bool'"); \
        } else { \
            const ctype* a = self.data_ptr<ctype>(); \
            const ctype* b = other.data_ptr<ctype>(); \
            ctype out = static_cast<ctype>(accumulate([&](int64_t i) { \
                return matmul_to_accum(a[i]) * matmul_to_accum(b[i]); })); \
            result.data_ptr<ctype>()[0] = out; \
            return result; \
        } \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(DOT_CASE)
        case DType::ComplexHalf:
        case DType::ComplexFloat:
        case DType::ComplexDouble:
        case DType::BComplex32:
            // Complex dot does not conjugate (that is vdot).
            switch (self.dtype()) {
                case DType::ComplexHalf: {
                    using c = complex<Half>;
                    const c* a = self.data_ptr<c>();
                    const c* b = other.data_ptr<c>();
                    complex<float> out{};
                    for (int64_t i = 0; i < n; ++i)
                        out += matmul_to_accum(a[i]) * matmul_to_accum(b[i]);
                    result.data_ptr<c>()[0] = static_cast<c>(out);
                    return result;
                }
                case DType::ComplexFloat: {
                    using c = complex<float>;
                    const c* a = self.data_ptr<c>();
                    const c* b = other.data_ptr<c>();
                    c out{};
                    for (int64_t i = 0; i < n; ++i) out += a[i] * b[i];
                    result.data_ptr<c>()[0] = out;
                    return result;
                }
                case DType::ComplexDouble: {
                    using c = complex<double>;
                    const c* a = self.data_ptr<c>();
                    const c* b = other.data_ptr<c>();
                    c out{};
                    for (int64_t i = 0; i < n; ++i) out += a[i] * b[i];
                    result.data_ptr<c>()[0] = out;
                    return result;
                }
                default: {
                    using c = complex<BFloat16>;
                    const c* a = self.data_ptr<c>();
                    const c* b = other.data_ptr<c>();
                    complex<float> out{};
                    for (int64_t i = 0; i < n; ++i)
                        out += matmul_to_accum(a[i]) * matmul_to_accum(b[i]);
                    result.data_ptr<c>()[0] = static_cast<c>(out);
                    return result;
                }
            }
        default:
            TP_THROW(NotImplementedError, "\"dot\" not implemented for '",
                     pretty_dtype_name(self.dtype()), "'");
    }
#undef DOT_CASE
}

Tensor inner_kernel(const Tensor& self, const Tensor& other) {
    // promotion); otherwise this is tensordot over the last dimension.
    if (self.dim() == 0 || other.dim() == 0) {
        return self * other;
    }
    const int64_t n = self.size(-1);
    if (other.size(-1) != n) {
        TP_THROW(RuntimeError, "inner() the last dimension must match on both input tensors but got shapes ",
                 Size(static_cast<std::vector<int64_t>>(self.shape())).toString(), " and ",
                 Size(static_cast<std::vector<int64_t>>(other.shape())).toString());
    }
    if (self.dim() == 1 && other.dim() == 1) {
        return dot_kernel(self, other);
    }

    // Contract the last dimension: flatten all leading dims to rows and run
    // one (batched) GEMM against the transposed partner, then restore shape.
    Tensor a = self.reshape({-1, n});
    Tensor b = other.reshape({-1, n});
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i + 1 < self.shape().size(); ++i) out_shape.push_back(self.shape()[i]);
    for (size_t i = 0; i + 1 < other.shape().size(); ++i) out_shape.push_back(other.shape()[i]);
    Tensor product = matmul_kernel(a, transpose_last_two_view(b));
    return product.reshape(out_shape);
}

Tensor outer_kernel(const Tensor& self, const Tensor& vec2) {
    if (self.dim() != 1 || vec2.dim() != 1) {
        TP_THROW(RuntimeError, "1D tensors expected, but got ", self.dim(), "D and ",
                 vec2.dim(), "D tensors");
    }
    if (self.dtype() != vec2.dtype()) {
        TP_THROW(RuntimeError, "expected scalar type ", pretty_dtype_name(self.dtype()),
                 " but found ", pretty_dtype_name(vec2.dtype()));
    }
    return self.unsqueeze(1).mul(vec2.unsqueeze(0));
}

Tensor inner_backward_self_kernel(const Tensor& grad_output, const Tensor& self, const Tensor& other) {
    if (self.dim() == 0 || other.dim() == 0) {
        return grad_output * other;
    }
    const int64_t n = std::max<int64_t>(self.size(-1), 1);
    const int64_t prod_a = std::max<int64_t>(self.numel() / n, 1);
    const int64_t prod_b = std::max<int64_t>(other.numel() / n, 1);
    // inner(A, B) == A2 @ B2^T with A2=(prod_a, N), B2=(prod_b, N), so
    // dA2 = grad2 @ B2 -- exactly matmul_backward_self on the flattened pair.
    Tensor grad2 = grad_output.reshape({prod_a, prod_b});
    Tensor other2 = other.reshape({-1, n});
    Tensor da2 = matmul_kernel(grad2, other2);
    Tensor grad = sum_to_shape_cpu(da2, static_cast<std::vector<int64_t>>(self.shape()));
    return grad.reshape(static_cast<std::vector<int64_t>>(self.shape()));
}

Tensor inner_backward_other_kernel(const Tensor& grad_output, const Tensor& self, const Tensor& other) {
    if (self.dim() == 0 || other.dim() == 0) {
        return grad_output * self;
    }
    const int64_t n = std::max<int64_t>(self.size(-1), 1);
    const int64_t prod_a = std::max<int64_t>(self.numel() / n, 1);
    const int64_t prod_b = std::max<int64_t>(other.numel() / n, 1);
    // dB2^T = A2^T @ grad2 -- exactly matmul_backward_other on the flat pair.
    Tensor grad2 = grad_output.reshape({prod_a, prod_b});
    Tensor self2 = self.reshape({-1, n});
    Tensor db2t = matmul_kernel(transpose_last_two_view(self2), grad2);
    Tensor db2 = transpose_last_two_view(db2t);
    Tensor grad = sum_to_shape_cpu(db2, static_cast<std::vector<int64_t>>(other.shape()));
    return grad.reshape(static_cast<std::vector<int64_t>>(other.shape()));
}

TENSORPLAY_LIBRARY_IMPL(CPU, LinearAlgebraKernels) {
    m.impl("mm", mm_kernel);
    m.impl("matmul", matmul_kernel);
    m.impl("matmul.out", matmul_out_kernel);
    m.impl("matmul_backward_self", matmul_backward_self_kernel);
    m.impl("matmul_backward_other", matmul_backward_other_kernel);
    m.impl("addmm", addmm_kernel);
    m.impl("linear", linear_kernel);
    m.impl("bmm", bmm_kernel);
    m.impl("baddbmm", baddbmm_kernel);
    m.impl("mv", mv_kernel);
    m.impl("dot", dot_kernel);
    m.impl("inner", inner_kernel);
    m.impl("inner_backward_self", inner_backward_self_kernel);
    m.impl("inner_backward_other", inner_backward_other_kernel);
    m.impl("outer", outer_kernel);
}

// F.linear under Composite (einsum/gradient precedent): expressed through
// dispatcher primitives so autograd records inner nodes on every backend,
// while python callers drop a per-call addmm+t+add python-composite tax.
// faster" route).  Everything composes into one seeded-GEMM addmm on the
// flattened 2-D view, so the dispatcher records a single LinearBackward node
// instead of the composite's matmul/add/t chain.  weight.t() is a raw
// as_strided view -- no dispatch, no clone; its transposed layout is consumed
// natively by both the BLAS and the oneDNN routes of addmm_kernel.
Tensor linear_kernel(const Tensor& input, const Tensor& weight,
                     const std::optional<Tensor>& bias_opt) {
    const int64_t input_dim = input.dim();
    if (input_dim == 0 || weight.dim() == 0) {
        TP_THROW(RuntimeError,
                 "both arguments to linear need to be at least 1D, but they are ",
                 input_dim, "D and ", weight.dim(), "D");
    }
    if (weight.dim() != 2) {
        TP_THROW(RuntimeError, "linear(): weight must be 2D (out_features, in_features), got ",
                 weight.dim(), "D");
    }

    auto flipped_weight_view = [&] {
        std::vector<int64_t> sizes =
            static_cast<std::vector<int64_t>>(weight.shape());
        std::vector<int64_t> strides = weight.strides();
        std::swap(sizes[sizes.size() - 2], sizes[sizes.size() - 1]);
        std::swap(strides[strides.size() - 2], strides[strides.size() - 1]);
        return weight.as_strided(sizes, strides);
    };
    Tensor wt = flipped_weight_view();

    if (input_dim == 1) {
        Tensor row = input.as_strided({1, input.size(0)},
                                      {input.size(0), input.stride(0)});
        Tensor out = bias_opt.has_value()
            ? addmm_kernel(*bias_opt, row, wt, Scalar(1), Scalar(1))
            : mm_kernel(row, wt);
        return out.as_strided({wt.size(1)}, {1});
    }

    Tensor in_flat = input;
    if (input_dim > 2) {
        in_flat = input.is_contiguous() ? input : input.contiguous();
        in_flat = in_flat.as_strided({in_flat.numel() / weight.size(1),
                                      weight.size(1)},
                                     {weight.size(1), 1});
    } else if (!input.is_contiguous()) {
        in_flat = input.contiguous();
    }

    Tensor out = bias_opt.has_value()
        ? addmm_kernel(*bias_opt, in_flat, wt, Scalar(1), Scalar(1))
        : mm_kernel(in_flat, wt);

    if (input_dim == 2) return out;
    auto result_sizes =
        static_cast<std::vector<int64_t>>(input.shape());
    result_sizes[result_sizes.size() - 1] = weight.size(0);
    std::vector<int64_t> result_strides(result_sizes.size(), 1);
    for (int64_t d = static_cast<int64_t>(result_sizes.size()) - 2; d >= 0; --d)
        result_strides[d] = result_strides[d + 1] * result_sizes[d + 1];
    return out.as_strided(result_sizes, result_strides);
}

Tensor linear_composite(const Tensor& input, const Tensor& weight,
                        const std::optional<Tensor>& bias_opt) {
    if (input.dim() == 0 || weight.dim() == 0) {
        TP_THROW(RuntimeError,
                 "both arguments to linear need to be at least 1D, but they are ",
                 input.dim(), "D and ", weight.dim(), "D");
    }
    if (weight.dim() != 2) {
        TP_THROW(RuntimeError, "linear(): weight must be 2D (out_features, in_features), got ",
                 weight.dim(), "D");
    }
    Tensor out = tpx::ops::matmul(input, tpx::ops::transpose(weight, 0, 1));
    if (bias_opt.has_value() && bias_opt->defined())
        out = tpx::ops::add(out, *bias_opt, Scalar(1));
    return out;
}

TENSORPLAY_LIBRARY_IMPL(Composite, LinearAlgebraComposite) {
    m.impl("linear", linear_composite);
}

} // namespace cpu
} // namespace tensorplay
