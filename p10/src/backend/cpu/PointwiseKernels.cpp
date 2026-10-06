#include "Tensor.h"
#include "Complex.h"
#include "Dispatcher.h"
#include "Utils.h"
#include "ErrorReporting.h"
#include "TensorIteratorOps.h"
#include "TypePromotion.h"
#include "OneDNNContext.h"
#include "Allocator.h"
#include "OutWrite.h"
#include "Parallel.h"
#include "cpu/VecUnary.h"
#include "cpu/ComplexUnary.h"
#include "cpu/VecComplex.h"
#include "cpu/ActivationUnaryKernels.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>
#include <type_traits>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

// DispatchStub instances for the tiered activation kernels; defined once here
// because the tier objects (TP_CPU_KERNEL_SRCS) each compile their own copy
// and would collide at link time.
namespace tensorplay { namespace cpu {
DEFINE_DISPATCH(sigmoid_f32_stub);
DEFINE_DISPATCH(silu_f32_stub);
DEFINE_DISPATCH(sigmoid_f64_stub);
DEFINE_DISPATCH(silu_f64_stub);
}} // namespace tensorplay::cpu

#ifdef USE_ONEDNN
#include "dnnl.hpp"
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tensorplay {
namespace cpu {
using namespace tensorplay::parallel;

namespace ops = tensorplay::tpx::ops;

// Reused from ArithmeticKernels.cpp so the binary kernel below keeps
// broadcasting, promotion and complex support without a second loop.
Tensor sub_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha);
Tensor mul_kernel(const Tensor& self, const Tensor& other);

// --- Unary Kernels ---

#ifdef USE_ONEDNN
void onednn_eltwise(const Tensor& src, Tensor& dst, dnnl::algorithm algo, float alpha = 0.0f, float beta = 0.0f) {
    auto& engine = OneDNNContext::get_engine();
    auto& stream = OneDNNContext::get_stream();

    // Create memory descriptors
    dnnl::memory::dims dims;
    for(auto d : src.shape()) dims.push_back(d);
    
    dnnl::memory::dims strides;
    for(auto s : src.strides()) strides.push_back(s);
    
    auto md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, strides);

    // Create primitive descriptor directly
    auto pd = dnnl::eltwise_forward::primitive_desc(
        engine,
        dnnl::prop_kind::forward_inference,
        algo,
        md,
        md,
        alpha,
        beta);
    
    auto src_mem = dnnl::memory(md, engine, src.data_ptr());
    // If inplace, dst is src
    auto dst_mem = (src.data_ptr() == dst.data_ptr()) ? src_mem : dnnl::memory(md, engine, dst.data_ptr());

    dnnl::eltwise_forward(pd).execute(stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dst_mem}
    });
    stream.wait();
}
#endif

namespace {
// Only f32/f64 have vector kernels.  Templating the dispatch (instead of
// if-constexpr at the macro site) matters: the switch below expands with a
// concrete ctype inside this non-template function, and a discarded
// constexpr branch there would still be fully type-checked.
template <typename T>
inline void vec_run(vecunary::VOp op, const vecunary::VParams& prm,
                    const T* src, T* dst, int64_t begin, int64_t end) {
    if constexpr (std::is_same_v<T, float>) {
        vecunary::run_f32(op, prm, src, dst, begin, end);
    } else if constexpr (std::is_same_v<T, double>) {
        vecunary::run_f64(op, prm, src, dst, begin, end);
    }
}
} // namespace

// Elementwise kernels use a finer grain than the global default: with the
// spinning intraop pool a chunk handoff costs ~1-2us, so splitting small-ish
// tensors across all workers wins far more than the handoff costs.
constexpr int64_t kUnaryGrain = 8192;

// Elementwise compute is storage-order agnostic, so a channels-last input is
// processed in its own physical order and the result keeps that layout
// instead of being repacked row-major.
static bool pointwise_keep_channels_last(const Tensor& t) {
    return t.dim() == 4 && t.is_contiguous(MemoryFormat::ChannelsLast);
}

static Tensor empty_like_in_input_order(const Tensor& t, DType dt) {
    const auto sizes = static_cast<std::vector<int64_t>>(t.shape());
    Tensor result = Tensor::empty(sizes, dt, t.device());
    if (pointwise_keep_channels_last(t)) {
        result = result.as_strided(sizes, get_channels_last_strides(sizes), 0);
    } else if (!t.is_contiguous() && t.dim() > 1) {
        // A dense permuted view (e.g. transpose) keeps its layout in the
        // output: the elementwise pass then runs over unit-stride runs on
        // both sides instead of gathering one cache line per element.
        const std::vector<int64_t> t_strides = t.strides();
        if (SizesAndStrides::is_non_overlapping_and_dense(sizes, t_strides)) {
            result = result.as_strided(sizes, t_strides, 0);
        }
    }
    return result;
}

// True when source and result share one storage-order traversal: identical
// strides that tile memory exactly (contiguous, channels-last, or a dense
// permuted view).  Element i in the source's storage then maps to element i
// in the result's storage, so the op can run as one flat pass over data_ptr
// instead of per-run addressing.
static bool flat_layout_pair(const Tensor& t, const Tensor& result) {
    if (t.is_contiguous() && result.is_contiguous()) return true;
    const std::vector<int64_t> ts = t.strides();
    if (ts != static_cast<std::vector<int64_t>>(result.strides())) return false;
    const auto sizes = static_cast<std::vector<int64_t>>(t.shape());
    return SizesAndStrides::is_non_overlapping_and_dense(sizes, ts);
}

// Elementwise work over a strided source without a materialization copy.
// The result carries the input's layout (dense permuted views keep their
// strides), so the innermost dim -- unit stride on the result side -- is
// chosen as:
//   * a dim that is unit-stride on the source too, when long enough -- the
//     body then sees adjacent runs on both sides and can vectorize (slices,
//     batch offsets, channels-last rows, transposed views);
//   * otherwise the last non-degenerate dim, with the body reading the
//     source at a constant byte step (layouts with holes, e.g. strided
//     slices, whose output is plain contiguous).
// body(src_run, src_step, dst_run, len) must handle any len.  Returns false
// only for a 0-dim input, where the caller's flat path is trivial.
template <class Body>
static bool strided_unary_loop(const Tensor& self, Tensor& result,
                               int64_t min_run, Body&& body) {
    const int64_t ndim = self.dim();
    if (ndim == 0) return false;
    const std::vector<int64_t> sizes = static_cast<std::vector<int64_t>>(self.shape());
    const std::vector<int64_t> src_strides = self.strides();

    // Destination strides come from the result itself: plain contiguous
    // except where the input's dense permuted layout was preserved.
    const std::vector<int64_t> dst_strides = result.strides();

    int64_t inner_dim = -1;
    for (int64_t d = 0; d < ndim; ++d) {
        if (sizes[d] > 1 && src_strides[d] == 1 && dst_strides[d] == 1 &&
            sizes[d] >= min_run) {
            inner_dim = d;
            break;
        }
    }
    if (inner_dim < 0) {
        for (int64_t d = ndim - 1; d >= 0; --d) {
            if (sizes[d] > 1) {
                inner_dim = d;
                break;
            }
        }
        if (inner_dim < 0) return false;  // scalar tensor
        // Gather reads pull one cache line per element when the inner step
        // is large.  A 2D layout still prefers that cost onto the blocked
        // copy + vector pass; deeper permutes have no such blocked copy, so
        // the direct gather wins there regardless of the step.
        const int64_t step_bytes =
            src_strides[inner_dim] * static_cast<int64_t>(self.itemsize());
        if (ndim == 2 && step_bytes > 256) return false;
    }
    const int64_t inner_size = sizes[inner_dim];
    const int64_t src_inner_step =
        src_strides[inner_dim] * static_cast<int64_t>(self.itemsize());

    // Outer dims, outermost first (descending source stride) so the
    // fastest-varying outer step takes the shortest source jumps.  Offsets
    // are byte-based: the two tensors may have different item sizes.
    struct OuterDim { int64_t size, src_step, dst_step; };
    std::vector<OuterDim> dims;
    dims.reserve(ndim);
    for (int64_t d = 0; d < ndim; ++d) {
        if (d == inner_dim || sizes[d] <= 1) continue;
        dims.push_back({sizes[d],
                        src_strides[d] * static_cast<int64_t>(self.itemsize()),
                        dst_strides[d] * static_cast<int64_t>(result.itemsize())});
    }
    std::sort(dims.begin(), dims.end(),
              [](const OuterDim& a, const OuterDim& b) { return a.src_step > b.src_step; });

    int64_t outer_count = 1;
    for (const OuterDim& od : dims) outer_count *= od.size;
    const int64_t grain_iters = std::max<int64_t>(
        1, kUnaryGrain / std::max<int64_t>(inner_size, 1));

    const char* src_base = static_cast<const char*>(self.data_ptr());
    char* dst_base = static_cast<char*>(result.data_ptr());
    const int ndims = static_cast<int>(dims.size());

    parallel_for(0, outer_count, grain_iters, [&](int64_t begin, int64_t end) {
        std::vector<int64_t> idx(ndims, 0);
        int64_t src_off = 0, dst_off = 0;
        int64_t linear = begin;
        for (int j = ndims - 1; j >= 0; --j) {
            const int64_t v = linear % dims[j].size;
            linear /= dims[j].size;
            idx[j] = v;
            src_off += v * dims[j].src_step;
            dst_off += v * dims[j].dst_step;
        }
        for (int64_t o = begin; o < end; ++o) {
            body(src_base + src_off, src_inner_step, dst_base + dst_off, inner_size);
            for (int j = ndims - 1; j >= 0; --j) {
                if (++idx[j] < dims[j].size) {
                    src_off += dims[j].src_step;
                    dst_off += dims[j].dst_step;
                    break;
                }
                // idx[j] wraps size-1 -> 0: its contribution (size-1)*step
                // leaves the sum.
                idx[j] = 0;
                src_off -= dims[j].src_step * (dims[j].size - 1);
                dst_off -= dims[j].dst_step * (dims[j].size - 1);
            }
        }
    });
    return true;
}

// Helper for operations that preserve dtype (e.g. abs, neg, square).
// vec_op selects the AVX2 fast path (see cpu/VecUnary.h) for float/double;
// the scalar lambda stays as the fallback for other dtypes and non-AVX2 hosts.
template<typename Func>
Tensor unary_op_kernel(const Tensor& self, Func func,
                       vecunary::VOp vec_op = vecunary::VOp::None,
                       vecunary::VParams vec_prm = {}) {
    Tensor result = empty_like_in_input_order(self, self.dtype());
    int64_t n = self.numel();

    // Vector fast paths exist only for f32/f64; other dtypes take the
    // scalar-lambda fallback and must never instantiate the vec calls.
    const bool vec_ok = vecunary::vec_ready() && vec_op != vecunary::VOp::None
        && (self.dtype() == DType::Float32 || self.dtype() == DType::Float64);
    const bool contiguous_path = flat_layout_pair(self, result);

    #define OP_CASE(ctype, name) \
    case DType::name: { \
        ctype* dst = result.data_ptr<ctype>(); \
        if (!contiguous_path) { \
            const bool strided = strided_unary_loop( \
                self, result, /*min_run=*/vec_ok ? 8 : 1, \
                [&](const char* sp, int64_t sstep, char* dp, int64_t len) { \
                    ctype* d = reinterpret_cast<ctype*>(dp); \
                    if (sstep == static_cast<int64_t>(sizeof(ctype))) { \
                        const ctype* s = reinterpret_cast<const ctype*>(sp); \
                        if (vec_ok) { \
                            vec_run(vec_op, vec_prm, s, d, 0, len); \
                        } else { \
                            for (int64_t i = 0; i < len; ++i) d[i] = func(s[i]); \
                        } \
                    } else { \
                        for (int64_t i = 0; i < len; ++i) { \
                            ctype v; \
                            std::memcpy(&v, sp + i * sstep, sizeof(ctype)); \
                            d[i] = func(v); \
                        } \
                    } \
                }); \
            if (strided) break; \
        } \
        Tensor self_contig = contiguous_path ? self : self.contiguous(); \
        const ctype* src = self_contig.data_ptr<ctype>(); \
        if (vec_ok) { \
            parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) { \
            vec_run(vec_op, vec_prm, src, dst, begin, end); \
            }); \
            break; \
        } \
        parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) { \
        for(int64_t i = begin; i < end; ++i) dst[i] = func(src[i]); \
        }); \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "Unsupported dtype");
    }
    #undef OP_CASE

    return result;
}

// Helper for operations that promote integer to float (e.g. sin, cos, exp).
// vec_op selects the AVX2 fast path (see cpu/VecUnary.h) for float/double and
// the widen-compute-narrow paths for half/bfloat16; the scalar lambda remains
// as fallback.
template<typename Func>
Tensor unary_float_op_kernel(const Tensor& self, Func func,
                             vecunary::VOp vec_op = vecunary::VOp::None,
                             vecunary::VParams vec_prm = {}) {
    DType out_dtype = self.dtype();
    if (isIntegralType(out_dtype)) {
        out_dtype = DType::Float32;
    }
    Tensor result = empty_like_in_input_order(self, out_dtype);
    int64_t n = self.numel();

    // Vector fast paths cover f32/f64 plus the widen-compute-narrow f16/bf16
    // kernels; integral inputs stay on the scalar-lambda fallback.
    const bool vec_ok = vecunary::vec_ready() && vec_op != vecunary::VOp::None
        && (self.dtype() == DType::Float32 || self.dtype() == DType::Float64
            || self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16);
    const bool contiguous_path = flat_layout_pair(self, result);

    if (isIntegralType(self.dtype())) {
        // Input int, Output float
        #define INT_CASE(ctype, name) \
        case DType::name: { \
            float* dst = result.data_ptr<float>(); \
            if (!contiguous_path) { \
                const bool strided = strided_unary_loop( \
                    self, result, /*min_run=*/1, \
                    [&](const char* sp, int64_t sstep, char* dp, int64_t len) { \
                        float* d = reinterpret_cast<float*>(dp); \
                        for (int64_t i = 0; i < len; ++i) { \
                            ctype v; \
                            std::memcpy(&v, sp + i * sstep, sizeof(ctype)); \
                            d[i] = static_cast<float>(func(static_cast<float>(v))); \
                        } \
                    }); \
                if (strided) break; \
            } \
            Tensor self_contig = contiguous_path ? self : self.contiguous(); \
            const ctype* src = self_contig.data_ptr<ctype>(); \
            parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) { \
            for(int64_t i = begin; i < end; ++i) dst[i] = static_cast<float>(func(static_cast<float>(src[i]))); \
            }); \
            break; \
        }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(INT_CASE) // This macro covers floats too, but we filtered with if
            default: TP_THROW(TypeError, "Unsupported dtype");
        }
        #undef INT_CASE
    } else if (self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16) {
        int64_t n = self.numel();
        if (self.dtype() == DType::Float16) {
            Half* dst = result.data_ptr<Half>();
            if (!contiguous_path) {
                const bool strided = strided_unary_loop(
                    self, result, /*min_run=*/(vec_ok && vecunary::f16c_available()) ? 8 : 1,
                    [&](const char* sp, int64_t sstep, char* dp, int64_t len) {
                        Half* d = reinterpret_cast<Half*>(dp);
                        if (sstep == 2 && vec_ok && vecunary::f16c_available()) {
                            vecunary::run_f16(vec_op, vec_prm,
                                              reinterpret_cast<const uint16_t*>(sp),
                                              reinterpret_cast<uint16_t*>(d), 0, len);
                        } else {
                            for (int64_t i = 0; i < len; ++i) {
                                Half h;
                                std::memcpy(&h, sp + i * sstep, sizeof(Half));
                                d[i] = static_cast<Half>(func(static_cast<float>(h)));
                            }
                        }
                    });
                if (strided) return result;
            }
            const Half* src = contiguous_path
                ? self.data_ptr<Half>() : self.contiguous().data_ptr<Half>();
            if (vec_ok && vecunary::f16c_available()) {
                parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                vecunary::run_f16(vec_op, vec_prm,
                                  reinterpret_cast<const uint16_t*>(src),
                                  reinterpret_cast<uint16_t*>(dst), begin, end);
                });
            } else {
                parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                for(int64_t i = begin; i < end; ++i) dst[i] = static_cast<Half>(func(static_cast<float>(src[i])));
                });
            }
        } else {
            BFloat16* dst = result.data_ptr<BFloat16>();
            if (!contiguous_path) {
                const bool strided = strided_unary_loop(
                    self, result, /*min_run=*/vec_ok ? 8 : 1,
                    [&](const char* sp, int64_t sstep, char* dp, int64_t len) {
                        BFloat16* d = reinterpret_cast<BFloat16*>(dp);
                        if (sstep == 2 && vec_ok) {
                            vecunary::run_bf16(vec_op, vec_prm,
                                               reinterpret_cast<const uint16_t*>(sp),
                                               reinterpret_cast<uint16_t*>(d), 0, len);
                        } else {
                            for (int64_t i = 0; i < len; ++i) {
                                BFloat16 h;
                                std::memcpy(&h, sp + i * sstep, sizeof(BFloat16));
                                d[i] = static_cast<BFloat16>(func(static_cast<float>(h)));
                            }
                        }
                    });
                if (strided) return result;
            }
            const BFloat16* src = contiguous_path
                ? self.data_ptr<BFloat16>() : self.contiguous().data_ptr<BFloat16>();
            if (vec_ok) {
                parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                vecunary::run_bf16(vec_op, vec_prm,
                                   reinterpret_cast<const uint16_t*>(src),
                                   reinterpret_cast<uint16_t*>(dst), begin, end);
                });
            } else {
                parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                for(int64_t i = begin; i < end; ++i) dst[i] = static_cast<BFloat16>(func(static_cast<float>(src[i])));
                });
            }
        }
    } else {
        // Input float, Output float
        #define FLOAT_CASE(ctype, name) \
        case DType::name: { \
            ctype* dst = result.data_ptr<ctype>(); \
            if (!contiguous_path) { \
                const bool strided = strided_unary_loop( \
                    self, result, /*min_run=*/vec_ok ? 8 : 1, \
                    [&](const char* sp, int64_t sstep, char* dp, int64_t len) { \
                        ctype* d = reinterpret_cast<ctype*>(dp); \
                        if (sstep == static_cast<int64_t>(sizeof(ctype))) { \
                            const ctype* s = reinterpret_cast<const ctype*>(sp); \
                            for (int64_t i = 0; i < len; ++i) d[i] = func(s[i]); \
                        } else { \
                            for (int64_t i = 0; i < len; ++i) { \
                                ctype v; \
                                std::memcpy(&v, sp + i * sstep, sizeof(ctype)); \
                                d[i] = func(v); \
                            } \
                        } \
                    }); \
                if (strided) break; \
            } \
            Tensor self_contig = contiguous_path ? self : self.contiguous(); \
            const ctype* src = self_contig.data_ptr<ctype>(); \
            parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) { \
            for(int64_t i = begin; i < end; ++i) dst[i] = func(src[i]); \
            }); \
            break; \
        }
        switch (self.dtype()) {
            case DType::Float32: {
                 float* dst = result.data_ptr<float>();
                 if (!contiguous_path) {
                     const bool strided = strided_unary_loop(
                         self, result, /*min_run=*/vec_ok ? 8 : 1,
                         [&](const char* sp, int64_t sstep, char* dp, int64_t len) {
                             float* d = reinterpret_cast<float*>(dp);
                             if (sstep == 4) {
                                 const float* s = reinterpret_cast<const float*>(sp);
                                 if (vec_ok) {
                                     vecunary::run_f32(vec_op, vec_prm, s, d, 0, len);
                                 } else {
                                     for (int64_t i = 0; i < len; ++i) d[i] = func(s[i]);
                                 }
                             } else {
                                 for (int64_t i = 0; i < len; ++i) {
                                     float v;
                                     std::memcpy(&v, sp + i * sstep, sizeof(float));
                                     d[i] = func(v);
                                 }
                             }
                         });
                     if (strided) break;
                 }
                 Tensor self_contig = contiguous_path ? self : self.contiguous();
                 const float* src = self_contig.data_ptr<float>();
                 if (vec_ok) {
                     parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) { \
                     vecunary::run_f32(vec_op, vec_prm, src, dst, begin, end);
                     });
                     break;
                 }
                 parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) { \
                 for(int64_t i = begin; i < end; ++i) dst[i] = func(src[i]);
                 });
                 break;
            }
            case DType::Float64: {
                 double* dst = result.data_ptr<double>();
                 if (!contiguous_path) {
                     const bool strided = strided_unary_loop(
                         self, result, /*min_run=*/vec_ok ? 8 : 1,
                         [&](const char* sp, int64_t sstep, char* dp, int64_t len) {
                             double* d = reinterpret_cast<double*>(dp);
                             if (sstep == 8) {
                                 const double* s = reinterpret_cast<const double*>(sp);
                                 if (vec_ok) {
                                     vecunary::run_f64(vec_op, vec_prm, s, d, 0, len);
                                 } else {
                                     for (int64_t i = 0; i < len; ++i) d[i] = func(s[i]);
                                 }
                             } else {
                                 for (int64_t i = 0; i < len; ++i) {
                                     double v;
                                     std::memcpy(&v, sp + i * sstep, sizeof(double));
                                     d[i] = func(v);
                                 }
                             }
                         });
                     if (strided) break;
                 }
                 Tensor self_contig = contiguous_path ? self : self.contiguous();
                 const double* src = self_contig.data_ptr<double>();
                 if (vec_ok) {
                     parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) { \
                     vecunary::run_f64(vec_op, vec_prm, src, dst, begin, end);
                     });
                     break;
                 }
                 parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) { \
                 for(int64_t i = begin; i < end; ++i) dst[i] = func(src[i]);
                 });
                 break;
            }
            default: TP_THROW(TypeError, "Unsupported dtype (expected float)");
        }
        #undef FLOAT_CASE
    }

    return result;
}

// corresponding real dtype (hypot(re, im)).
Tensor complex_abs_kernel(const Tensor& self) {
    DType out_dtype = toRealValueType(self.dtype());
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), out_dtype, self.device());
    const int64_t n = self.numel();
    Tensor self_contig = self.contiguous();

    switch (self.dtype()) {
        case DType::ComplexFloat: {
            using c_t = tensorplay::complex<float>;
            const c_t* src = reinterpret_cast<const c_t*>(self_contig.data_ptr());
            float* dst = result.data_ptr<float>();
            parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    dst[i] = std::hypot(src[i].real(), src[i].imag());
                }
            });
            break;
        }
        case DType::ComplexDouble: {
            using c_t = tensorplay::complex<double>;
            const c_t* src = reinterpret_cast<const c_t*>(self_contig.data_ptr());
            double* dst = result.data_ptr<double>();
            parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    dst[i] = std::hypot(src[i].real(), src[i].imag());
                }
            });
            break;
        }
        case DType::ComplexHalf:
        case DType::BComplex32: {
            // Reduced complexes compute the magnitude in float32.
            if (self.dtype() == DType::ComplexHalf) {
                const tensorplay::complex<Half>* src =
                    reinterpret_cast<const tensorplay::complex<Half>*>(self_contig.data_ptr());
                Half* dst = result.data_ptr<Half>();
                parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                        float re = static_cast<float>(src[i].real());
                        float im = static_cast<float>(src[i].imag());
                        dst[i] = static_cast<Half>(std::hypot(re, im));
                    }
                });
            } else {
                const tensorplay::complex<BFloat16>* src =
                    reinterpret_cast<const tensorplay::complex<BFloat16>*>(self_contig.data_ptr());
                BFloat16* dst = result.data_ptr<BFloat16>();
                parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                        float re = static_cast<float>(src[i].real());
                        float im = static_cast<float>(src[i].imag());
                        dst[i] = static_cast<BFloat16>(std::hypot(re, im));
                    }
                });
            }
            break;
        }
        default: TP_THROW(TypeError, "complex abs: unsupported dtype");
    }
    return result;
}

// Implementations

Tensor abs_kernel(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        if ((self.dtype() == DType::ComplexFloat ||
             self.dtype() == DType::ComplexDouble) &&
            self.is_contiguous() && self.numel() > 0 &&
            veccomplex::avx2_available()) {
            Tensor out = Tensor::empty(
                static_cast<std::vector<int64_t>>(self.shape()),
                toRealValueType(self.dtype()), self.device());
            if (veccomplex::try_abs(self.data_ptr(), out.data_ptr(),
                                    self.numel(), self.dtype()))
                return out;
        }
        return complex_abs_kernel(self);
    }
    return unary_op_kernel(self, [](auto x) {
        using T = decltype(x);
        if constexpr (std::is_unsigned_v<T>) {
            return x;
        } else {
            return std::abs(x);
        }
    }, vecunary::VOp::Abs);
}

// Vectorized complex unary driver — defined below the float kernels; declared
// here because neg/square route through it.
template <typename F>
static Tensor cplx_unary_vec(const Tensor& self, veccomplex::Op op, F fb);

Tensor neg_kernel(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        return cplx_unary_vec(self, veccomplex::Op::Neg,
                              [](auto x) { return -x; });
    }
    return unary_op_kernel(self, [](auto x) {
        if constexpr (std::is_same_v<decltype(x), bool>) {
             return x; // neg(bool) in same dtype is weird, just return x to avoid warning
        } else {
             return -x;
        }
    }, vecunary::VOp::Neg);
}

Tensor square_kernel(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        return cplx_unary_vec(self, veccomplex::Op::Square,
                              [](auto x) { return x * x; });
    }
    return unary_op_kernel(self, [](auto x) { return x * x; }, vecunary::VOp::Square);
}

Tensor sign_kernel(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        return complex_unary_op_kernel(self, [](auto x) {
            using C = decltype(x);
            using T = typename C::value_type;
            const T m = std::abs(x);
            if (m == T(0)) return C(T(0), T(0));
            return x / C(m, T(0));
        });
    }
    return unary_op_kernel(self, [](auto x) {
        if constexpr (std::is_same_v<decltype(x), bool>) {
            return x ? 1 : 0;
        } else {
            using ctype = decltype(x);
            if (x > ctype(0)) return static_cast<ctype>(1);
            if (x < ctype(0)) return static_cast<ctype>(-1);
            return static_cast<ctype>(0);
        }
    }, vecunary::VOp::Sign);
}

Tensor floor_kernel(const Tensor& self) {
    if (isIntegralType(self.dtype())) return self.clone();
    return unary_op_kernel(self, [](auto x) { return std::floor(x); }, vecunary::VOp::Floor);
}

Tensor ceil_kernel(const Tensor& self) {
    if (isIntegralType(self.dtype())) return self.clone();
    return unary_op_kernel(self, [](auto x) { return std::ceil(x); }, vecunary::VOp::Ceil);
}

Tensor round_kernel(const Tensor& self) {
    if (isIntegralType(self.dtype())) return self.clone();
    return unary_op_kernel(self, [](auto x) { return std::nearbyint(x); }, vecunary::VOp::Round);
}

// Float ops
//
// the complex dtypes (see docs/source/complex_numbers.md).  Complex inputs
// route through complex_unary_op_kernel with local complex math or the
// formulas above; real dtypes keep the vectorized paths.


// Vectorized complex unary fast path (cpu/VecComplex.h): contiguous
// Complex{Float,Double} inputs run the AVX2+libmvec cores; reduced complexes,
// non-contiguous input and pre-AVX2 hosts keep the scalar driver below.
template <typename F>
static Tensor cplx_unary_vec(const Tensor& self, veccomplex::Op op, F fb) {
    if ((self.dtype() == DType::ComplexFloat ||
         self.dtype() == DType::ComplexDouble) &&
        self.is_contiguous() && self.numel() > 0 &&
        veccomplex::unary_supported(op) && veccomplex::avx2_available()) {
        Tensor out = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), self.dtype(),
            self.device());
        if (veccomplex::try_unary(self.data_ptr(), out.data_ptr(),
                                  self.numel(), self.dtype(), op))
            return out;
    }
    return complex_unary_op_kernel(self, fb);
}

Tensor acos_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Acos,
                              [](auto x) { return tensorplay::acos(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::acos(x); }, vecunary::VOp::Acos);
}
Tensor acosh_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Acosh,
                              [](auto x) { return tensorplay::acosh(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::acosh(x); }, vecunary::VOp::Acosh);
}
Tensor asin_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Asin,
                              [](auto x) { return tensorplay::asin(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::asin(x); }, vecunary::VOp::Asin);
}
Tensor asinh_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Asinh,
                              [](auto x) { return tensorplay::asinh(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::asinh(x); }, vecunary::VOp::Asinh);
}
Tensor atan_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Atan,
                              [](auto x) { return tensorplay::atan(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::atan(x); }, vecunary::VOp::Atan);
}
Tensor atanh_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Atanh,
                              [](auto x) { return tensorplay::atanh(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::atanh(x); }, vecunary::VOp::Atanh);
}
Tensor cos_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Cos,
                              [](auto x) { return tensorplay::cos(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::cos(x); }, vecunary::VOp::Cos);
}
Tensor cosh_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Cosh,
                              [](auto x) { return tensorplay::cosh(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::cosh(x); }, vecunary::VOp::Cosh);
}
Tensor sin_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Sin,
                              [](auto x) { return tensorplay::sin(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::sin(x); }, vecunary::VOp::Sin);
}
Tensor& sin_out_cpu(const Tensor& self, Tensor& out) {
    write_out(out, sin_kernel(self));
    return out;
}
Tensor sinh_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Sinh,
                              [](auto x) { return tensorplay::sinh(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::sinh(x); }, vecunary::VOp::Sinh);
}
Tensor tan_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Tan,
                              [](auto x) { return tensorplay::tan(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::tan(x); }, vecunary::VOp::Tan);
}
Tensor tanh_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Tanh,
                              [](auto x) { return tensorplay::tanh(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::tanh(x); }, vecunary::VOp::Tanh);
}
Tensor sec_kernel_impl(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return complex_unary_op_kernel(self, [](auto x) {
            using T = decltype(x);
            return static_cast<T>(1) / tensorplay::cos(x);
        });
    return unary_float_op_kernel(self, [](auto x) { return 1 / std::cos(x); });
}
Tensor csc_kernel_impl(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return complex_unary_op_kernel(self, [](auto x) {
            using T = decltype(x);
            return static_cast<T>(1) / tensorplay::sin(x);
        });
    return unary_float_op_kernel(self, [](auto x) { return 1 / std::sin(x); });
}
Tensor cot_kernel_impl(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return complex_unary_op_kernel(self, [](auto x) {
            using T = decltype(x);
            return static_cast<T>(1) / tensorplay::tan(x);
        });
    return unary_float_op_kernel(self, [](auto x) { return 1 / std::tan(x); });
}
Tensor tanhshrink_kernel_impl(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return complex_unary_op_kernel(self, [](auto x) { return x - tensorplay::tanh(x); });
    return unary_float_op_kernel(self, [](auto x) { return x - std::tanh(x); });
}
Tensor squared_difference_kernel_impl(const Tensor& self, const Tensor& other) {
    Tensor d = sub_kernel(self, other, Scalar(1));
    return mul_kernel(d, d);
}
Tensor exp_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Exp,
                              [](auto x) { return tensorplay::exp(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::exp(x); }, vecunary::VOp::Exp);
}
Tensor expm1_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Expm1,
                              [](auto x) { return cx_expm1(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::expm1(x); }, vecunary::VOp::Expm1);
}
Tensor erf_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::erf(x); }, vecunary::VOp::Erf); }
Tensor erfc_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::erfc(x); }, vecunary::VOp::Erfc); }
Tensor log_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Log,
                              [](auto x) { return tensorplay::log(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::log(x); }, vecunary::VOp::Log);
}
Tensor log10_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Log10,
                              [](auto x) { return tensorplay::log10(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::log10(x); }, vecunary::VOp::Log10);
}
Tensor log1p_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Log1p,
                              [](auto x) { return cx_log1p(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::log1p(x); }, vecunary::VOp::Log1p);
}
Tensor log2_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Log2,
                              [](auto x) { return cx_log2(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::log2(x); }, vecunary::VOp::Log2);
}
Tensor lgamma_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::lgamma(x); }, vecunary::VOp::Lgamma); }
Tensor sqrt_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Sqrt,
                              [](auto x) { return tensorplay::sqrt(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::sqrt(x); }, vecunary::VOp::Sqrt);
}
Tensor rsqrt_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Rsqrt,
                              [](auto x) { return cx_rsqrt(x);  });
    return unary_float_op_kernel(self, [](auto x) { using T = decltype(x); return static_cast<T>(1) / std::sqrt(x); }, vecunary::VOp::Rsqrt);
}
Tensor sigmoid_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Sigmoid,
                              [](auto x) { return cx_sigmoid(x);  });
    // Contiguous f32/f64 goes through the tier-compiled stub (three-tier
    // build, compile-time ISA selection); everything else -- non-contiguous
    // (normalized by clone below), f16/bf16 widen-compute-narrow, integral
    // promotion -- stays on the generic runtime-dispatched path.
    const DType dt = self.dtype();
    if (self.is_contiguous() &&
        (dt == DType::Float32 || dt == DType::Float64)) {
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), dt, self.device());
        const int64_t n = self.numel();
        if (dt == DType::Float32) {
            const float* src = self.data_ptr<float>();
            float* dst = result.data_ptr<float>();
            parallel_for(0, n, kUnaryGrain, [&](int64_t b, int64_t e) {
                sigmoid_f32_stub(DeviceType::CPU, src + b, dst + b, e - b);
            });
        } else {
            const double* src = self.data_ptr<double>();
            double* dst = result.data_ptr<double>();
            parallel_for(0, n, kUnaryGrain, [&](int64_t b, int64_t e) {
                sigmoid_f64_stub(DeviceType::CPU, src + b, dst + b, e - b);
            });
        }
        return result;
    }
    return unary_float_op_kernel(self, [](auto x) { using T = decltype(x); return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x)); }, vecunary::VOp::Sigmoid);
}

Tensor frac_kernel(const Tensor& self) {
    if (isIntegralType(self.dtype())) {
        TP_THROW(NotImplementedError, "frac is not implemented for integral tensors");
    }
    return unary_op_kernel(self, [](auto x) { return x - std::trunc(x); }, vecunary::VOp::Frac);
}

Tensor trunc_kernel(const Tensor& self) {
    if (isIntegralType(self.dtype())) return self.clone();
    return unary_op_kernel(self, [](auto x) { return std::trunc(x); }, vecunary::VOp::Trunc);
}

Tensor relu_kernel(const Tensor& self) {
    // oneDNN eltwise rejected here (was: numel >= 4096): primitive
    // construction costs ~5us per call and measured *slower* than the native
    // kernels rather than oneDNN for ReLU.

    // Vectorized path for contiguous Float32 (see cpu/VecUnary.h).  The old
    // __AVX512F__/__AVX2__ blocks here were dead code: this TU compiles
    // without ISA flags, so dispatch goes through VecUnary's per-function
    // target attributes instead.  Non-AVX2 hosts fall through to the generic
    // kernel below.
    if (vecunary::vec_ready() && self.dtype() == DType::Float32 && self.is_contiguous()) {
         Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
         int64_t n = self.numel();
        const float* src = self.data_ptr<float>();
        float* dst = result.data_ptr<float>();
        if (n <= kUnaryGrain) {
            vecunary::run_f32(vecunary::VOp::Relu, vecunary::VParams{}, src, dst, 0, n);
        } else {
            parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                vecunary::run_f32(vecunary::VOp::Relu, vecunary::VParams{}, src, dst, begin, end);
            });
        }
         return result;
    }

    return unary_op_kernel(self, [](auto x) {
        using T = decltype(x);
        if constexpr (std::is_unsigned_v<T>) {
            return x;
        } else {
            return x < static_cast<T>(0) ? static_cast<T>(0) : x;
        }
    });
}

Tensor& relu_inplace_kernel(Tensor& self) {
    // Keep OneDNN for out-of-place ReLU, but use the SIMD/scalar path here.
    // Vectorized path for contiguous Float32 (see cpu/VecUnary.h); the old
    // __AVX512F__/__AVX2__ blocks were dead code in this TU and the #else arm
    // was a serial full-tensor loop.  Non-AVX2 hosts fall through to the
    // parallel generic branch below.
    if (vecunary::vec_ready() && self.dtype() == DType::Float32 && self.is_contiguous()) {
        int64_t n = self.numel();
        float* data = self.data_ptr<float>();
        if (n <= kUnaryGrain) {
            vecunary::run_f32(vecunary::VOp::Relu, vecunary::VParams{}, data, data, 0, n);
        } else {
            parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                vecunary::run_f32(vecunary::VOp::Relu, vecunary::VParams{}, data, data, begin, end);
            });
        }
         return self;
    }

    // Generic fallback
    int64_t n = self.numel();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        ctype* data = self.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
        for(int64_t i = begin; i < end; ++i) { \
            if constexpr (!std::is_unsigned_v<ctype>) { \
                data[i] = data[i] < static_cast<ctype>(0) ? static_cast<ctype>(0) : data[i]; \
            } \
        } \
        }); \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: {
             // Debug info
             std::cerr << "Unsupported dtype: " << (int)self.dtype() << " for relu_inplace" << std::endl;
             TP_THROW(TypeError, "Unsupported dtype");
        }
    }
    #undef OP_CASE
    
    return self;
}

// Defined below; used by the public gelu entry points above them.
Tensor gelu_tanh_impl(const Tensor& self);
Tensor gelu_backward_impl(const Tensor& grad_output, const Tensor& self, const std::string& approximate);

Tensor gelu_kernel(const Tensor& self, const std::string& approximate) {
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))); tanh approximation from
    if (approximate == "tanh") {
        return gelu_tanh_impl(self);
    } else if (approximate != "none") {
        TP_THROW(ValueError, "approximate argument must be either none or tanh, but got " + approximate);
    }
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        constexpr T kAlpha = static_cast<T>(0.70710678118654752440); // M_SQRT1_2
        return static_cast<T>(0.5) * x * (static_cast<T>(1) + std::erf(x * kAlpha));
    }, vecunary::VOp::GeluNone);
}

Tensor gelu_backward_kernel(const Tensor& grad_output, const Tensor& self, const std::string& approximate) {
    return gelu_backward_impl(grad_output, self, approximate);
}

Tensor silu_kernel(const Tensor& self) {
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    const DType dt = self.dtype();
    if (self.is_contiguous() &&
        (dt == DType::Float32 || dt == DType::Float64)) {
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), dt, self.device());
        const int64_t n = self.numel();
        if (dt == DType::Float32) {
            const float* src = self.data_ptr<float>();
            float* dst = result.data_ptr<float>();
            parallel_for(0, n, kUnaryGrain, [&](int64_t b, int64_t e) {
                silu_f32_stub(DeviceType::CPU, src + b, dst + b, e - b);
            });
        } else {
            const double* src = self.data_ptr<double>();
            double* dst = result.data_ptr<double>();
            parallel_for(0, n, kUnaryGrain, [&](int64_t b, int64_t e) {
                silu_f64_stub(DeviceType::CPU, src + b, dst + b, e - b);
            });
        }
        return result;
    }
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        return x / (static_cast<T>(1) + std::exp(-x));
    }, vecunary::VOp::Silu);
}

// Fused gated activation primitives.  These belong with the existing SiLU
// native/GatedLinearUnit.cpp rather than introducing an LLM-specific kernel
// bucket.  The packed form follows the decoder convention [gate | up].
namespace {

inline void check_silu_mul_inputs(const Tensor& gate, const Tensor& up,
                                  const char* op) {
    if (gate.device() != up.device()) {
        TP_THROW(DeviceMismatchError, op,
                 ": gate and up must be on the same device");
    }
    if (gate.shape() != up.shape()) {
        TP_THROW(RuntimeError, op, ": gate and up must have the same shape");
    }
    if (gate.dtype() != up.dtype()) {
        TP_THROW(RuntimeError, op, ": gate and up must have the same dtype");
    }
    if (!isFloatingType(gate.dtype())) {
        TP_THROW(NotImplementedError, op,
                 ": only floating point dtypes are supported");
    }
}

template <typename T, typename Acc>
void silu_mul_loop(const T* gate, const T* up, T* output, int64_t n) {
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const Acc x = static_cast<Acc>(gate[i]);
            const Acc y = static_cast<Acc>(up[i]);
            const Acc sigmoid = Acc(1) / (Acc(1) + std::exp(-x));
            output[i] = static_cast<T>(x * sigmoid * y);
        }
    });
}

template <typename T, typename Acc>
void silu_and_mul_loop(const T* input, T* output, int64_t n,
                       int64_t half_width) {
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t row = i / half_width;
            const int64_t col = i - row * half_width;
            const int64_t base = row * (2 * half_width);
            const Acc gate = static_cast<Acc>(input[base + col]);
            const Acc up = static_cast<Acc>(input[base + half_width + col]);
            const Acc sigmoid = Acc(1) / (Acc(1) + std::exp(-gate));
            output[i] = static_cast<T>(gate * sigmoid * up);
        }
    });
}

template <typename T>
Tensor silu_mul_typed(const Tensor& gate, const Tensor& up) {
    Tensor gate_c = gate.is_contiguous() ? gate : gate.contiguous();
    Tensor up_c = up.is_contiguous() ? up : up.contiguous();
    Tensor output = Tensor::empty(
        static_cast<std::vector<int64_t>>(gate_c.shape()), gate_c.dtype(),
        gate_c.device());
    using Acc = std::conditional_t<std::is_same_v<T, double>, double, float>;
    silu_mul_loop<T, Acc>(gate_c.data_ptr<T>(), up_c.data_ptr<T>(),
                          output.data_ptr<T>(), gate_c.numel());
    return output;
}

template <typename T>
Tensor silu_and_mul_typed(const Tensor& input) {
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    std::vector<int64_t> output_shape =
        static_cast<std::vector<int64_t>>(input_c.shape());
    const int64_t packed_width = output_shape.back();
    output_shape.back() = packed_width / 2;
    Tensor output = Tensor::empty(output_shape, input_c.dtype(),
                                  input_c.device());
    using Acc = std::conditional_t<std::is_same_v<T, double>, double, float>;
    silu_and_mul_loop<T, Acc>(input_c.data_ptr<T>(), output.data_ptr<T>(),
                              input_c.numel() / 2, packed_width / 2);
    return output;
}

} // namespace

Tensor silu_mul_cpu(const Tensor& gate, const Tensor& up) {
    check_silu_mul_inputs(gate, up, "silu_mul");
    switch (gate.dtype()) {
        case DType::Float32:
            return silu_mul_typed<float>(gate, up);
        case DType::Float64:
            return silu_mul_typed<double>(gate, up);
        case DType::Float16:
            return silu_mul_typed<Half>(gate, up);
        case DType::BFloat16:
            return silu_mul_typed<BFloat16>(gate, up);
        default:
            TP_THROW(NotImplementedError, "silu_mul: unsupported dtype");
    }
}

Tensor fused_swiglu_cpu(const Tensor& gate, const Tensor& up) {
    return silu_mul_cpu(gate, up);
}

Tensor silu_and_mul_cpu(const Tensor& input) {
    if (input.dim() < 1) {
        TP_THROW(RuntimeError,
                 "silu_and_mul: input must have at least one dimension");
    }
    const int64_t width = input.size(-1);
    if ((width & 1) != 0) {
        TP_THROW(RuntimeError,
                 "silu_and_mul: the packed last dimension must be even");
    }
    if (!isFloatingType(input.dtype())) {
        TP_THROW(NotImplementedError,
                 "silu_and_mul: only floating point dtypes are supported");
    }
    switch (input.dtype()) {
        case DType::Float32:
            return silu_and_mul_typed<float>(input);
        case DType::Float64:
            return silu_and_mul_typed<double>(input);
        case DType::Float16:
            return silu_and_mul_typed<Half>(input);
        case DType::BFloat16:
            return silu_and_mul_typed<BFloat16>(input);
        default:
            TP_THROW(NotImplementedError, "silu_and_mul: unsupported dtype");
    }
}

// ---------------------------------------------------------------------------
//     (hardsigmoid_kernel, hardtanh_backward_kernel, hardswish_kernel,
//      leaky_relu_kernel)
//     (scalar_gelu_approximated_with_tanh)
//     (get_scalar_elu_elementwise_func)
//     (GeluBackwardCUDAKernelImpl — the reference backward formulas)
//     (MishBackwardCUDAKernelImpl)
//     (SoftplusBackwardCUDAKernelImpl)
// ---------------------------------------------------------------------------
// The type an activation computes in: double for double, float for the rest
// (reduced-precision inputs widen to float, as their vector paths do).
template <typename T>
using activation_math_t = std::conditional_t<std::is_same_v<T, double>, double, float>;

template<typename Func>
Tensor activation_backward_kernel(const Tensor& grad_output, const Tensor& self, Func func) {
    DType out_dtype = grad_output.dtype();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()), out_dtype, grad_output.device());
    int64_t n = grad_output.numel();
    if (n == 0) return result;

    Tensor grad_contig = grad_output.contiguous();
    Tensor self_contig = self.contiguous();

    #define BACKWARD_CASE(ctype, name) \
    case DType::name: { \
        const ctype* dy = grad_contig.data_ptr<ctype>(); \
        const ctype* x = self_contig.data_ptr<ctype>(); \
        ctype* dst = result.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) { \
                using M = activation_math_t<ctype>; \
                dst[i] = static_cast<ctype>(func(static_cast<M>(dy[i]), static_cast<M>(x[i]))); \
            } \
        }); \
        break; \
    }
    switch (out_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(BACKWARD_CASE)
        default: TP_THROW(TypeError, "Unsupported dtype for activation backward");
    }
    #undef BACKWARD_CASE
    return result;
}

static inline float gelu_none_scalar(float x) {
    constexpr float kAlpha = 0.70710678118654752440f; // M_SQRT1_2
    return x * 0.5f * (1.0f + std::erf(x * kAlpha));
}
static inline float gelu_tanh_scalar(float x) {
    constexpr float kBeta = 1.41421356237309504880f * 1.12837916709551257390f * 0.5f; // M_SQRT2 * M_2_SQRTPI * 0.5
    constexpr float kKappa = 0.044715f;
    float x_cube = x * x * x;
    float inner = kBeta * (x + kKappa * x_cube);
    return 0.5f * x * (1.0f + std::tanh(inner));
}
static inline float gelu_backward_none_scalar(float dy, float x) {
    //   kAlpha = M_SQRT1_2; kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5
    //   cdf = 0.5*(1+erf(x*kAlpha)); pdf = kBeta*exp(-x*x*0.5); return dy*(cdf + x*pdf);
    constexpr float kAlpha = 0.70710678118654752440f;
    constexpr float kBeta = 1.12837916709551257390f * 0.70710678118654752440f * 0.5f;
    float cdf = 0.5f * (1.0f + std::erf(x * kAlpha));
    float pdf = kBeta * std::exp(x * x * -0.5f);
    return dy * (cdf + x * pdf);
}
static inline double gelu_tanh_scalar(double x) {
    constexpr double kBeta = 1.41421356237309504880 * 1.12837916709551257390 * 0.5;
    constexpr double kKappa = 0.044715;
    double inner = kBeta * (x + kKappa * x * x * x);
    return 0.5 * x * (1.0 + std::tanh(inner));
}
static inline double gelu_backward_none_scalar(double dy, double x) {
    constexpr double kAlpha = 0.70710678118654752440;
    constexpr double kBeta = 1.12837916709551257390 * 0.70710678118654752440 * 0.5;
    double cdf = 0.5 * (1.0 + std::erf(x * kAlpha));
    double pdf = kBeta * std::exp(x * x * -0.5);
    return dy * (cdf + x * pdf);
}
static inline double gelu_backward_tanh_scalar(double dy, double x) {
    constexpr double kBeta = 1.41421356237309504880 * 1.12837916709551257390 * 0.5;
    constexpr double kKappa = 0.044715;
    double x_sq = x * x;
    double inner = kBeta * (x + kKappa * x_sq * x);
    double tanh_inner = std::tanh(inner);
    double right_derivative =
        0.5 * x * (1.0 - tanh_inner * tanh_inner) * kBeta * (1.0 + 3.0 * kKappa * x_sq);
    return dy * (0.5 * (1.0 + tanh_inner) + right_derivative);
}
static inline float gelu_backward_tanh_scalar(float dy, float x) {
    constexpr float kBeta = 1.41421356237309504880f * 1.12837916709551257390f * 0.5f;
    constexpr float kKappa = 0.044715f;
    float x_sq = x * x;
    float x_cube = x_sq * x;
    float inner = kBeta * (x + kKappa * x_cube);
    float tanh_inner = std::tanh(inner);
    float left = 0.5f * x;
    float right = 1.0f + tanh_inner;
    float left_derivative = 0.5f * right;
    float tanh_derivative = 1.0f - tanh_inner * tanh_inner;
    float inner_derivative = kBeta * (1.0f + 3.0f * kKappa * x_sq);
    float right_derivative = left * tanh_derivative * inner_derivative;
    return dy * (left_derivative + right_derivative);
}

Tensor gelu_tanh_impl(const Tensor& self) {
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        return static_cast<T>(gelu_tanh_scalar(static_cast<activation_math_t<T>>(x)));
    }, vecunary::VOp::GeluTanh);
}

Tensor gelu_backward_impl(const Tensor& grad_output, const Tensor& self, const std::string& approximate) {
    if (approximate == "none") {
        return activation_backward_kernel(grad_output, self,
            [](auto dy, auto x) { return gelu_backward_none_scalar(dy, x); });
    } else if (approximate == "tanh") {
        return activation_backward_kernel(grad_output, self,
            [](auto dy, auto x) { return gelu_backward_tanh_scalar(dy, x); });
    }
    TP_THROW(ValueError, "approximate argument must be either none or tanh, but got " + approximate);
}

Tensor hardtanh_kernel_impl(const Tensor& self, const Scalar& min_val, const Scalar& max_val) {
    vecunary::VParams prm;
    prm.p0 = min_val.toDouble();
    prm.p1 = max_val.toDouble();
    return unary_float_op_kernel(self, [min_val, max_val](auto x) {
        using T = decltype(x);
        T lo = static_cast<T>(min_val.toDouble());
        T hi = static_cast<T>(max_val.toDouble());
        return x < lo ? lo : (x > hi ? hi : x);
    }, vecunary::VOp::Hardtanh, prm);
}

Tensor hardtanh_backward_kernel_impl(const Tensor& grad_output, const Tensor& self, const Scalar& min_val, const Scalar& max_val) {
    double lo = min_val.toDouble();
    double hi = max_val.toDouble();
    return activation_backward_kernel(grad_output, self,
        [lo, hi](auto dy, auto x) { return (x <= lo || x >= hi) ? decltype(dy)(0) : dy; });
}

Tensor relu6_kernel_impl(const Tensor& self) {
    return hardtanh_kernel_impl(self, Scalar(0.0), Scalar(6.0));
}

Tensor hardswish_kernel_impl(const Tensor& self) {
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        T xf = static_cast<T>(static_cast<activation_math_t<T>>(x));
        T clamped = (xf + T(3) < T(0)) ? T(0) : (xf + T(3) > T(6)) ? T(6) : xf + T(3);
        return xf * clamped / T(6);
    }, vecunary::VOp::Hardswish);
}

Tensor hardswish_backward_kernel_impl(const Tensor& grad_output, const Tensor& self) {
    //   d/dx [x * relu6(x + 3) / 6]:
    //   x <= -3 -> 0 ; -3 < x < 3 -> dy * (x/3 + 0.5) ; x >= 3 -> dy
    return activation_backward_kernel(grad_output, self,
        [](auto dy, auto x) {
            using M = decltype(x);
            if (x <= M(-3)) return M(0);
            if (x < M(3)) return dy * (x / M(3) + M(0.5));
            return dy;
        });
}

Tensor silu_backward_kernel_impl(const Tensor& grad_output, const Tensor& self) {
    //   sigmoid = 1 / (1 + exp(-x)); dy * sigmoid * (1 + x * (1 - sigmoid))
    return activation_backward_kernel(grad_output, self,
        [](auto dy, auto x) {
            using M = decltype(x);
            const M s = M(1) / (M(1) + std::exp(-x));
            return dy * s * (M(1) + x * (M(1) - s));
        });
}

Tensor hardsigmoid_kernel_impl(const Tensor& self) {
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        T xf = static_cast<T>(static_cast<activation_math_t<T>>(x));
        T v = xf + T(3);
        v = v < T(0) ? T(0) : (v > T(6) ? T(6) : v);
        return v / T(6);
    }, vecunary::VOp::Hardsigmoid);
}

Tensor hardsigmoid_backward_kernel_impl(const Tensor& grad_output, const Tensor& self) {
    //   d/dx [relu6(x + 3) / 6]: dy / 6 strictly inside (-3, 3), else 0
    return activation_backward_kernel(grad_output, self,
        [](auto dy, auto x) {
            using M = decltype(x);
            if (x <= M(-3) || x >= M(3)) return M(0);
            return dy / M(6);
        });
}

Tensor leaky_relu_kernel_impl(const Tensor& self, const Scalar& negative_slope) {
    double slope = negative_slope.toDouble();
    vecunary::VParams prm;
    prm.p0 = slope;
    return unary_float_op_kernel(self, [slope](auto x) {
        using T = decltype(x);
        T xf = static_cast<T>(static_cast<activation_math_t<T>>(x));
        return xf < T(0) ? static_cast<T>(slope) * xf : xf;
    }, vecunary::VOp::LeakyRelu, prm);
}

Tensor leaky_relu_backward_kernel_impl(const Tensor& grad_output, const Tensor& self, const Scalar& negative_slope, bool self_is_result) {
    (void)self_is_result; // out-of-place call always receives the input itself
    double slope = negative_slope.toDouble();
    return activation_backward_kernel(grad_output, self,
        [slope](auto dy, auto x) {
            using M = decltype(x);
            return x > M(0) ? dy : dy * static_cast<M>(slope);
        });
}

Tensor elu_kernel_impl(const Tensor& self, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale) {
    //   a < 0 ? expm1(a * input_scale) * negcoef : a * poscoef
    double negcoef = alpha.toDouble() * scale.toDouble();
    double poscoef = scale.toDouble();
    double negiptcoef = input_scale.toDouble();
    vecunary::VParams prm; // p0=alpha*scale, p1=scale, p2=input_scale
    prm.p0 = negcoef;
    prm.p1 = poscoef;
    prm.p2 = negiptcoef;
    return unary_float_op_kernel(self, [negcoef, poscoef, negiptcoef](auto x) {
        using T = decltype(x);
        using M = activation_math_t<T>;
        T a = static_cast<T>(static_cast<M>(x));
        return a < T(0)
            ? static_cast<T>(std::expm1(static_cast<M>(a) * static_cast<M>(negiptcoef)) * static_cast<M>(negcoef))
            : a * static_cast<T>(poscoef);
    }, vecunary::VOp::Elu, prm);
}

Tensor elu_backward_kernel_impl(const Tensor& grad_output, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale, bool is_result, const Tensor& self_or_result) {
    //   is_result: b <= 0 ? a*negiptcoef*(b + negcoef) : a*poscoef
    //   else:      b <= 0 ? a*negiptcoef*negcoef*exp(b*negiptcoef) : a*poscoef
    double negcoef = alpha.toDouble() * scale.toDouble();
    double poscoef = scale.toDouble();
    double negiptcoef = input_scale.toDouble();
    return activation_backward_kernel(grad_output, self_or_result,
        [negcoef, poscoef, negiptcoef, is_result](auto dy, auto b) {
            using M = decltype(b);
            return b <= M(0)
                ? (is_result
                      ? dy * static_cast<M>(negiptcoef) * (b + static_cast<M>(negcoef))
                      : dy * static_cast<M>(negiptcoef) * static_cast<M>(negcoef) * std::exp(b * static_cast<M>(negiptcoef)))
                : dy * static_cast<M>(poscoef);
        });
}

Tensor mish_kernel_impl(const Tensor& self) {
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        T xf = static_cast<T>(static_cast<activation_math_t<T>>(x));
        T sp = std::log(T(1) + std::exp(xf));
        return xf * std::tanh(sp);
    }, vecunary::VOp::Mish);
}

Tensor mish_backward_kernel_impl(const Tensor& grad_output, const Tensor& self) {
    //   sp = log1p(exp(x)); tanh_sp = tanh(sp); sech2 = 1 - tanh_sp^2
    //   return dy * (tanh_sp + x * sech2 * sigmoid(x))
    return activation_backward_kernel(grad_output, self,
        [](auto dy, auto x) {
            using M = decltype(x);
            M sp = std::log1p(std::exp(x));
            M tanh_sp = std::tanh(sp);
            M sech2 = M(1) - tanh_sp * tanh_sp;
            M gsp = M(1) / (M(1) + std::exp(-x));
            return dy * (tanh_sp + x * sech2 * gsp);
        });
}

Tensor selu_kernel_impl(const Tensor& self) {
    //   lambda_ = 1.0507009873554804934193349852946
    //   alpha_  = 1.6732632423543772848170429916717
    constexpr double lambda_ = 1.0507009873554804934193349852946;
    constexpr double alpha_ = 1.6732632423543772848170429916717;
    return unary_float_op_kernel(self, [lambda_, alpha_](auto x) {
        using T = decltype(x);
        T a = static_cast<T>(static_cast<activation_math_t<T>>(x));
        return a > T(0) ? a * static_cast<T>(lambda_)
                        : static_cast<T>(alpha_ * lambda_) * std::expm1(a);
    }, vecunary::VOp::Selu);
}

Tensor celu_kernel_impl(const Tensor& self, Scalar alpha) {
    double a = alpha.toDouble();
    vecunary::VParams prm;
    prm.p0 = a;
    return unary_float_op_kernel(self, [a](auto x) {
        using T = decltype(x);
        T af = static_cast<T>(static_cast<activation_math_t<T>>(x));
        return af > T(0) ? af : static_cast<T>(a) * (std::expm1(af / static_cast<T>(a)));
    }, vecunary::VOp::Celu, prm);
}

Tensor softplus_kernel_impl(const Tensor& self, const Scalar& beta, const Scalar& threshold) {
    //   beta_in * a > threshold ? a : log1p(exp(beta_in * a)) / beta_in
    double beta_in = beta.toDouble();
    double threshold_in = threshold.toDouble();
    vecunary::VParams prm; // p0=beta_in, p1=threshold_in
    prm.p0 = beta_in;
    prm.p1 = threshold_in;
    return unary_float_op_kernel(self, [beta_in, threshold_in](auto x) {
        using T = decltype(x);
        using M = activation_math_t<T>;
        T a = static_cast<T>(static_cast<M>(x));
        T beta_in_t = static_cast<T>(beta_in);
        return a * beta_in_t > static_cast<T>(threshold_in)
            ? a
            : static_cast<T>(std::log1p(std::exp(static_cast<M>(a * beta_in_t))) / beta_in);
    }, vecunary::VOp::Softplus, prm);
}

Tensor softplus_backward_kernel_impl(const Tensor& grad_output, const Tensor& self, const Scalar& beta, const Scalar& threshold) {
    //   beta_in * a > threshold ? dy : dy * sigmoid(beta_in * a)
    double beta_in = beta.toDouble();
    double threshold_in = threshold.toDouble();
    return activation_backward_kernel(grad_output, self,
        [beta_in, threshold_in](auto dy, auto a) {
            using M = decltype(a);
            return a * static_cast<M>(beta_in) > static_cast<M>(threshold_in)
                ? dy
                : dy * (M(1) / (M(1) + std::exp(-a * static_cast<M>(beta_in))));
        });
}

// ---------------------------------------------------------------------------
// log_sigmoid_cpu_kernel): out = min(x, 0) - log1p(exp(-|x|)).  The branch
// split keeps exp() bounded for both large-positive and large-negative inputs.
// ---------------------------------------------------------------------------
Tensor log_sigmoid_kernel_impl(const Tensor& self) {
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        T z = std::min(x, static_cast<T>(0));
        return static_cast<T>(z - std::log1p(std::exp(-std::abs(x))));
    });
}

Tensor log_sigmoid_backward_kernel_impl(const Tensor& grad_output, const Tensor& self) {
    //   grad * sigmoid(-x), branch-split so exp() never overflows:
    //     x >= 0: grad * exp(-x) / (1 + exp(-x))
    //     x <  0: grad / (1 + exp(x))
    // Computed in the storage dtype (f16/bf16 widen to float opmath) so that
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()),
                                  grad_output.dtype(), grad_output.device());
    const int64_t n = grad_output.numel();
    if (n == 0) return result;
    const Tensor gc = grad_output.contiguous();
    const Tensor sc = self.contiguous();
    #define LSIG_BWD_CASE(ctype, name) \
    case DType::name: { \
        const ctype* gp = gc.data_ptr<ctype>(); \
        const ctype* xp = sc.data_ptr<ctype>(); \
        ctype* yp = result.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) { \
                const ctype dy = gp[i]; \
                const ctype x = xp[i]; \
                yp[i] = x >= ctype(0) \
                    ? dy * (std::exp(-x) / (ctype(1) + std::exp(-x))) \
                    : dy / (ctype(1) + std::exp(x)); \
            } \
        }); \
        break; \
    }
    switch (grad_output.dtype()) {
        LSIG_BWD_CASE(float, Float32)
        LSIG_BWD_CASE(double, Float64)
        case DType::Float16:
        case DType::BFloat16: {
            if (grad_output.dtype() == DType::Float16) {
                const Half* gp = gc.data_ptr<Half>();
                const Half* xp = sc.data_ptr<Half>();
                Half* yp = result.data_ptr<Half>();
                parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                        const float dy = static_cast<float>(gp[i]);
                        const float x = static_cast<float>(xp[i]);
                        const float v = x >= 0.0f
                            ? dy * (std::exp(-x) / (1.0f + std::exp(-x)))
                            : dy / (1.0f + std::exp(x));
                        yp[i] = Half(v);
                    }
                });
            } else {
                const BFloat16* gp = gc.data_ptr<BFloat16>();
                const BFloat16* xp = sc.data_ptr<BFloat16>();
                BFloat16* yp = result.data_ptr<BFloat16>();
                parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                        const float dy = static_cast<float>(gp[i]);
                        const float x = static_cast<float>(xp[i]);
                        const float v = x >= 0.0f
                            ? dy * (std::exp(-x) / (1.0f + std::exp(-x)))
                            : dy / (1.0f + std::exp(x));
                        yp[i] = BFloat16(v);
                    }
                });
            }
            break;
        }
        default: TP_THROW(TypeError, "Unsupported dtype for log_sigmoid_backward");
    }
    #undef LSIG_BWD_CASE
    return result;
}

// ---------------------------------------------------------------------------
// negative elements by the (caller-provided) noise tensor, eval is leaky_relu
// TensorPlay kernels consume the noise the caller generated (nn.functional.rrelu
// draws it with rand), which keeps the kernel deterministic and RNG-free.
// ---------------------------------------------------------------------------
template <typename Func>
static Tensor binary_float_kernel(const Tensor& a, const Tensor& b, Func func) {
    if (a.shape() != b.shape())
        TP_THROW(RuntimeError, "rrelu_with_noise: expected noise to have the same shape as input");
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(a.shape()), a.dtype(), a.device());
    int64_t n = a.numel();
    if (n == 0) return result;
    Tensor ac = a.contiguous();
    Tensor bc = b.contiguous();
    #define RRELU_BIN_CASE(ctype, name) \
    case DType::name: { \
        const ctype* ap = ac.data_ptr<ctype>(); \
        const ctype* bp = bc.data_ptr<ctype>(); \
        ctype* yp = result.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) { \
                using M = activation_math_t<ctype>; \
                yp[i] = static_cast<ctype>(func(static_cast<M>(ap[i]), static_cast<M>(bp[i]))); \
            } \
        }); \
        break; \
    }
    switch (a.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(RRELU_BIN_CASE)
        default: TP_THROW(TypeError, "Unsupported dtype for rrelu_with_noise");
    }
    #undef RRELU_BIN_CASE
    return result;
}

Tensor rrelu_with_noise_backward_kernel_impl(const Tensor& grad_output, const Tensor& self, const Tensor& noise, const Scalar& lower, const Scalar& upper, bool training, bool self_is_result) {
    const double mean_slope = (lower.toDouble() + upper.toDouble()) / 2.0;
    // Training: the forward recorded each slope in noise (1 for positive
    // inputs), so the gradient is grad * noise.
    if (training) {
        if (grad_output.shape() != self.shape() || grad_output.shape() != noise.shape())
            TP_THROW(RuntimeError, "rrelu_with_noise_backward: shape mismatch");
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()),
                                      grad_output.dtype(), grad_output.device());
        const int64_t n = grad_output.numel();
        if (n == 0) return result;
        const Tensor gc = grad_output.contiguous();
        const Tensor nc = noise.contiguous();
        #define RRELU_TERN_CASE(ctype, name) \
        case DType::name: { \
            const ctype* gp = gc.data_ptr<ctype>(); \
            const ctype* np = nc.data_ptr<ctype>(); \
            ctype* yp = result.data_ptr<ctype>(); \
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
                for (int64_t i = begin; i < end; ++i) { \
                    using M = activation_math_t<ctype>; \
                    yp[i] = static_cast<ctype>( \
                        static_cast<M>(gp[i]) * static_cast<M>(np[i])); \
                } \
            }); \
            break; \
        }
        switch (grad_output.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(RRELU_TERN_CASE)
            default: TP_THROW(TypeError, "Unsupported dtype for rrelu_with_noise_backward");
        }
        #undef RRELU_TERN_CASE
        return result;
    }
    (void)self_is_result; // result > 0 iff self > 0 for a positive slope.
    // The leaky slope applies at zero as well (x > 0 passes through).
    return binary_float_kernel(grad_output, self, [mean_slope](auto dy, auto x) {
        using M = decltype(x);
        return x > M(0) ? dy : dy * static_cast<M>(mean_slope);
    });
}

Tensor pow_scalar_kernel(const Tensor& self, const Scalar& exponent) {
    if (self.dtype() == DType::Bool) TP_THROW(TypeError, "pow is not supported for bool tensors");
    if (isComplexType(self.dtype()) || exponent.isComplex()) {
        // scalar both produce complex results.  Negative integer exponents
        // are fine over complex.
        DType base_dt = isComplexType(self.dtype())
            ? self.dtype()
            : (isFloatingType(self.dtype()) ? toComplexType(self.dtype())
                                            : DType::ComplexFloat);
        DType result_dtype = promoteTypes(base_dt,
            isComplexType(exponent.dtype()) ? exponent.dtype() : toComplexType(base_dt));
        Tensor base = self.to(result_dtype);
        if (!isComplexType(exponent.dtype())) {
            double ev = exponent.toDouble();
            if (ev == 0.5) return sqrt_kernel(base);
            if (ev == -0.5) return rsqrt_kernel(base);
            if (ev == 1.0) return base.clone();
            if (ev == 2.0) return square_kernel(base);
            if (ev == 3.0) return complex_unary_op_kernel(base, [](auto x) { return x * x * x; });
            return complex_unary_op_kernel(base, [ev](auto x) {
                using V = typename decltype(x)::value_type;
                return tensorplay::pow(x, static_cast<V>(ev));
            });
        }
        // Complex exponent: promote the scalar into the result dtype once.
        if (result_dtype == DType::ComplexDouble) {
            auto e = exponent.to<tensorplay::complex<double>>();
            return complex_unary_op_kernel(base, [e](auto x) {
                using V = typename decltype(x)::value_type;
                return tensorplay::pow(
                    x, tensorplay::complex<V>(static_cast<V>(e.real()), static_cast<V>(e.imag())));
            });
        }
        auto e = exponent.to<tensorplay::complex<float>>();
        return complex_unary_op_kernel(base, [e](auto x) {
            using V = typename decltype(x)::value_type;
            return tensorplay::pow(
                x, tensorplay::complex<V>(static_cast<V>(e.real()), static_cast<V>(e.imag())));
        });
    }
    if (isIntegralType(self.dtype()) && exponent.isIntegral() && exponent.to<int64_t>() < 0) {
        TP_THROW(RuntimeError, "Integers to negative integer powers are not allowed.");
    }
    if (exponent.isFloatingPoint()) {
        double exp_val = exponent.toDouble();
        if (exp_val == 0.5 && self.dtype() != DType::Float64) return sqrt_kernel(self);
        if (exp_val == -0.5 && self.dtype() != DType::Float64) return rsqrt_kernel(self);
        if (exp_val == 1.0) return self.clone();
        if (exp_val == 2.0) return square_kernel(self);
        if (exp_val == 3.0) {
            return unary_float_op_kernel(self, [](auto x) { using T = decltype(x); return x * x * x; });
        }
        return unary_float_op_kernel(self, [exp_val](auto x) { using T = decltype(x); return std::pow(x, static_cast<T>(exp_val)); });
    } else {
        int64_t exp_val = exponent.to<int64_t>();
        if (exp_val < 0) {
             return unary_float_op_kernel(self, [exp_val](auto x) { using T = decltype(x); return std::pow(x, static_cast<T>(static_cast<double>(exp_val))); });
        }
        // Small non-negative integer exponents reduce to cheap elementwise
        // forms instead of the per-element square-and-multiply loop.
        if (exp_val == 0) {
            Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
            return result.fill_(Scalar(1));
        }
        if (exp_val == 1) return self.clone();
        if (exp_val == 2) return square_kernel(self);
        if (exp_val == 3) {
            return unary_op_kernel(self, [](auto x) { using T = decltype(x); return x * x * x; });
        }
        return unary_op_kernel(self, [exp_val](auto x) {
             using T = decltype(x);
             T base = x;
             T acc = static_cast<T>(1);
             int64_t e = exp_val;
             while (e > 0) {
                 if (e & 1) acc = acc * base;
                 e >>= 1;
                 if (e) base = base * base;
             }
             return acc;
        });
    }
}



Tensor angle_kernel(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        if ((self.dtype() == DType::ComplexFloat ||
             self.dtype() == DType::ComplexDouble) &&
            self.is_contiguous() && self.numel() > 0 &&
            veccomplex::avx2_available()) {
            Tensor out = Tensor::empty(
                static_cast<std::vector<int64_t>>(self.shape()),
                toRealValueType(self.dtype()), self.device());
            if (veccomplex::try_angle(self.data_ptr(), out.data_ptr(),
                                      self.numel(), self.dtype()))
                return out;
        }
        DType out_dtype = toRealValueType(self.dtype());
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), out_dtype, self.device());
        const int64_t n = self.numel();
        Tensor self_contig = self.contiguous();
        switch (self.dtype()) {
            case DType::ComplexHalf:
            case DType::BComplex32: {
                // Reduced complexes compute in float32.
                if (self.dtype() == DType::ComplexHalf) {
                    const tensorplay::complex<Half>* src =
                        reinterpret_cast<const tensorplay::complex<Half>*>(self_contig.data_ptr());
                    Half* dst = result.data_ptr<Half>();
                    parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                        for (int64_t i = begin; i < end; ++i) {
                            dst[i] = static_cast<Half>(std::atan2(
                                static_cast<float>(src[i].imag()),
                                static_cast<float>(src[i].real())));
                        }
                    });
                } else {
                    const tensorplay::complex<BFloat16>* src =
                        reinterpret_cast<const tensorplay::complex<BFloat16>*>(self_contig.data_ptr());
                    BFloat16* dst = result.data_ptr<BFloat16>();
                    parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                        for (int64_t i = begin; i < end; ++i) {
                            dst[i] = static_cast<BFloat16>(std::atan2(
                                static_cast<float>(src[i].imag()),
                                static_cast<float>(src[i].real())));
                        }
                    });
                }
                break;
            }
            case DType::ComplexFloat: {
                using c_t = tensorplay::complex<float>;
                const c_t* src = reinterpret_cast<const c_t*>(self_contig.data_ptr());
                float* dst = result.data_ptr<float>();
                parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                        dst[i] = std::atan2(src[i].imag(), src[i].real());
                    }
                });
                break;
            }
            case DType::ComplexDouble: {
                using c_t = tensorplay::complex<double>;
                const c_t* src = reinterpret_cast<const c_t*>(self_contig.data_ptr());
                double* dst = result.data_ptr<double>();
                parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                        dst[i] = std::atan2(src[i].imag(), src[i].real());
                    }
                });
                break;
            }
            default: TP_THROW(TypeError, "angle: unsupported dtype");
        }
        return result;
    }
    // For real numbers, angle is 0 if >=0, pi if <0
    return unary_float_op_kernel(self, [](auto x) {
        if (x >= 0) return 0.0;
        return 3.14159265358979323846;
    });
}

// --- Binary/Ternary Kernels ---

#if defined(__x86_64__)
namespace {
__attribute__((target("avx512f")))
void clamp_f32_avx512(const float* src, float* dst, int64_t n,
                      float lo, float hi) {
    const __m512 vlo = _mm512_set1_ps(lo);
    const __m512 vhi = _mm512_set1_ps(hi);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        v = _mm512_max_ps(vlo, v);
        v = _mm512_min_ps(vhi, v);
        _mm512_storeu_ps(dst + i, v);
    }
    for (; i < n; ++i) {
        float v = src[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        dst[i] = v;
    }
}

__attribute__((target("avx512f")))
void clamp_f64_avx512(const double* src, double* dst, int64_t n,
                      double lo, double hi) {
    const __m512d vlo = _mm512_set1_pd(lo);
    const __m512d vhi = _mm512_set1_pd(hi);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d v = _mm512_loadu_pd(src + i);
        v = _mm512_max_pd(vlo, v);
        v = _mm512_min_pd(vhi, v);
        _mm512_storeu_pd(dst + i, v);
    }
    for (; i < n; ++i) {
        double v = src[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        dst[i] = v;
    }
}
} // namespace
#endif

Tensor clamp_kernel(const Tensor& self, const std::optional<Scalar>& min, const std::optional<Scalar>& max) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    Tensor self_contig = self.contiguous();

#if defined(__x86_64__)
    if (vecunary::avx512_available() &&
        (self.dtype() == DType::Float32 || self.dtype() == DType::Float64)) {
        const double lo = min.has_value()
            ? min->toDouble() : -std::numeric_limits<double>::infinity();
        const double hi = max.has_value()
            ? max->toDouble() : std::numeric_limits<double>::infinity();
        if (self.dtype() == DType::Float32) {
            const float* src = self_contig.data_ptr<float>();
            float* dst = result.data_ptr<float>();
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                clamp_f32_avx512(src + begin, dst + begin, end - begin,
                                 static_cast<float>(lo), static_cast<float>(hi));
            });
        } else {
            const double* src = self_contig.data_ptr<double>();
            double* dst = result.data_ptr<double>();
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                clamp_f64_avx512(src + begin, dst + begin, end - begin,
                                 lo, hi);
            });
        }
        return result;
    }
#endif

    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* src = self_contig.data_ptr<ctype>(); \
        ctype* dst = result.data_ptr<ctype>(); \
        ctype min_val = min.has_value() ? min->to<ctype>() : std::numeric_limits<ctype>::lowest(); \
        ctype max_val = max.has_value() ? max->to<ctype>() : std::numeric_limits<ctype>::max(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
        for(int64_t i=begin; i<end; ++i) { \
            ctype val = src[i]; \
            if (min.has_value() && val < min_val) val = min_val; \
            if (max.has_value() && val > max_val) val = max_val; \
            dst[i] = val; \
        } \
        }); \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "Unsupported dtype");
    }
    #undef OP_CASE
    return result;
}

// clamp(self, bound, nullopt); delegate to the same kernel here.
Tensor clamp_min_kernel(const Tensor& self, const Scalar& min) {
    return clamp_kernel(self, min, std::nullopt);
}
Tensor clamp_max_kernel(const Tensor& self, const Scalar& max) {
    return clamp_kernel(self, std::nullopt, max);
}
Tensor& clamp_min__kernel(Tensor& self, const Scalar& min) {
    self.copy_(clamp_kernel(self, min, std::nullopt));
    return self;
}
Tensor& clamp_max__kernel(Tensor& self, const Scalar& max) {
    self.copy_(clamp_kernel(self, std::nullopt, max));
    return self;
}

// Helper for clamp backward
Tensor clamp_backward_kernel(const Tensor& grad_output, const Tensor& self, const std::optional<Scalar>& min, const std::optional<Scalar>& max) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()), grad_output.dtype(), grad_output.device());
    int64_t n = grad_output.numel();
    
    Tensor self_contig = self.contiguous();
    Tensor grad_contig = grad_output.contiguous();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* src = self_contig.data_ptr<ctype>(); \
        const ctype* grad = grad_contig.data_ptr<ctype>(); \
        ctype* dst = result.data_ptr<ctype>(); \
        ctype min_val = min.has_value() ? min->to<ctype>() : std::numeric_limits<ctype>::lowest(); \
        ctype max_val = max.has_value() ? max->to<ctype>() : std::numeric_limits<ctype>::max(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
        for(int64_t i = begin; i < end; ++i) { \
            ctype val = src[i]; \
            if ((min.has_value() && val < min_val) || (max.has_value() && val > max_val)) { \
                dst[i] = 0; \
            } else { \
                dst[i] = grad[i]; \
            } \
        } \
        }); \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "Unsupported dtype");
    }
    #undef OP_CASE
    
    return result;
}

Tensor threshold_backward_kernel(const Tensor& grad_output, const Tensor& output, const Scalar& threshold) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()), grad_output.dtype(), grad_output.device());
    int64_t n = grad_output.numel();
    
    Tensor output_contig = output.contiguous();
    Tensor grad_contig = grad_output.contiguous();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* src = output_contig.data_ptr<ctype>(); \
        const ctype* grad = grad_contig.data_ptr<ctype>(); \
        ctype* dst = result.data_ptr<ctype>(); \
        ctype thresh = threshold.to<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
        for(int64_t i = begin; i < end; ++i) { \
            if (src[i] <= thresh) { \
                dst[i] = 0; \
            } else { \
                dst[i] = grad[i]; \
            } \
        } \
        }); \
        break; \
    }

    switch (output.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "Unsupported dtype");
    }
    #undef OP_CASE
    
    return result;
}

// (max pass, exp+sum pass, write pass) instead of materializing 5 temporaries.
// Fast path: contiguous input, reduction over last dim. Fallback: composition.

// ---------------------------------------------------------------------------
// Row-wise softmax helpers.  Per-row passes are hand-vectorized with
// per-function target attributes and selected at runtime (AVX-512 -> AVX2 ->
// scalar), keeping exp on the libmvec vector ABI instead of a scalar libm
// call per element.
// ---------------------------------------------------------------------------
namespace softmax_row {

#if defined(__x86_64__)

#define TP_SOFTMAX_ROW_F32(fn_suffix, vec_t, width, exp_fn, mm_add, mm_sub, mm_mul, mm_max, mm_load, mm_store, mm_set1, mm_zero) \
__attribute__((target("avx512f")))                                       \
inline float row_max_##fn_suffix(const float* x, int64_t n) {            \
    vec_t m = mm_set1(x[0]);                                             \
    int64_t i = 1;                                                       \
    for (; i + width <= n; i += width)                                   \
        m = mm_max(mm_load(x + i), m);                                   \
    alignas(64) float buf[width];                                        \
    mm_store(buf, m);                                                    \
    float best = buf[0];                                                 \
    for (int64_t k = 1; k < width; ++k) best = std::max(best, buf[k]);   \
    for (; i < n; ++i) best = std::max(best, x[i]);                      \
    return best;                                                         \
}                                                                        \
__attribute__((target("avx512f")))                                       \
inline float row_expsum_##fn_suffix(const float* x, float m, float* out, int64_t n) { \
    vec_t vm = mm_set1(m);                                               \
    vec_t a0 = mm_zero(), a1 = mm_zero(), a2 = mm_zero(), a3 = mm_zero();\
    int64_t i = 0;                                                       \
    for (; i + 4 * width <= n; i += 4 * width) {                         \
        vec_t e0 = exp_fn(mm_sub(mm_load(x + i), vm));                   \
        vec_t e1 = exp_fn(mm_sub(mm_load(x + i + width), vm));           \
        vec_t e2 = exp_fn(mm_sub(mm_load(x + i + 2 * width), vm));       \
        vec_t e3 = exp_fn(mm_sub(mm_load(x + i + 3 * width), vm));       \
        mm_store(out + i, e0);                                           \
        mm_store(out + i + width, e1);                                   \
        mm_store(out + i + 2 * width, e2);                               \
        mm_store(out + i + 3 * width, e3);                               \
        a0 = mm_add(a0, e0);                                             \
        a1 = mm_add(a1, e1);                                             \
        a2 = mm_add(a2, e2);                                             \
        a3 = mm_add(a3, e3);                                             \
    }                                                                    \
    vec_t acc = mm_add(mm_add(a0, a1), mm_add(a2, a3));                  \
    alignas(64) float buf[width];                                        \
    mm_store(buf, acc);                                                  \
    float s = 0.f;                                                       \
    for (int64_t k = 0; k < width; ++k) s += buf[k];                     \
    for (; i < n; ++i) {                                                 \
        float e = std::exp(x[i] - m);                                    \
        out[i] = e;                                                      \
        s += e;                                                          \
    }                                                                    \
    return s;                                                            \
}                                                                        \
__attribute__((target("avx512f")))                                       \
inline void row_scale_##fn_suffix(float* out, float inv, int64_t n) {    \
    vec_t vi = mm_set1(inv);                                             \
    int64_t i = 0;                                                       \
    for (; i + width <= n; i += width)                                   \
        mm_store(out + i, mm_mul(mm_load(out + i), vi));                 \
    for (; i < n; ++i) out[i] *= inv;                                    \
}                                                                        \
__attribute__((target("avx512f")))                                       \
inline void row_logmode_##fn_suffix(const float* x, float* out, float m, float lse, int64_t n) { \
    vec_t vm = mm_set1(m), vl = mm_set1(lse);                            \
    int64_t i = 0;                                                       \
    for (; i + width <= n; i += width)                                   \
        mm_store(out + i, mm_sub(mm_sub(mm_load(x + i), vm), vl));       \
    for (; i < n; ++i) out[i] = (x[i] - m) - lse;                        \
}

TP_SOFTMAX_ROW_F32(f32_512, __m512, 16, tensorplay::tpsleef::exp,
                   _mm512_add_ps, _mm512_sub_ps, _mm512_mul_ps, _mm512_max_ps,
                   _mm512_loadu_ps, _mm512_storeu_ps, _mm512_set1_ps, _mm512_setzero_ps)
TP_SOFTMAX_ROW_F32(f32_256, __m256, 8, tensorplay::tpsleef::exp,
                   _mm256_add_ps, _mm256_sub_ps, _mm256_mul_ps, _mm256_max_ps,
                   _mm256_loadu_ps, _mm256_storeu_ps, _mm256_set1_ps, _mm256_setzero_ps)
#undef TP_SOFTMAX_ROW_F32

#define TP_SOFTMAX_ROW_F64(fn_suffix, vec_t, width, exp_fn, mm_add, mm_sub, mm_mul, mm_max, mm_load, mm_store, mm_set1, mm_zero) \
__attribute__((target("avx512f")))                                       \
inline double row_max_##fn_suffix(const double* x, int64_t n) {          \
    vec_t m = mm_set1(x[0]);                                             \
    int64_t i = 1;                                                       \
    for (; i + width <= n; i += width)                                   \
        m = mm_max(mm_load(x + i), m);                                   \
    alignas(64) double buf[width];                                       \
    mm_store(buf, m);                                                    \
    double best = buf[0];                                                \
    for (int64_t k = 1; k < width; ++k) best = std::max(best, buf[k]);   \
    for (; i < n; ++i) best = std::max(best, x[i]);                      \
    return best;                                                         \
}                                                                        \
__attribute__((target("avx512f")))                                       \
inline double row_expsum_##fn_suffix(const double* x, double m, double* out, int64_t n) { \
    vec_t vm = mm_set1(m);                                               \
    vec_t a0 = mm_zero(), a1 = mm_zero(), a2 = mm_zero(), a3 = mm_zero();\
    int64_t i = 0;                                                       \
    for (; i + 4 * width <= n; i += 4 * width) {                         \
        vec_t e0 = exp_fn(mm_sub(mm_load(x + i), vm));                   \
        vec_t e1 = exp_fn(mm_sub(mm_load(x + i + width), vm));           \
        vec_t e2 = exp_fn(mm_sub(mm_load(x + i + 2 * width), vm));       \
        vec_t e3 = exp_fn(mm_sub(mm_load(x + i + 3 * width), vm));       \
        mm_store(out + i, e0);                                           \
        mm_store(out + i + width, e1);                                   \
        mm_store(out + i + 2 * width, e2);                               \
        mm_store(out + i + 3 * width, e3);                               \
        a0 = mm_add(a0, e0);                                             \
        a1 = mm_add(a1, e1);                                             \
        a2 = mm_add(a2, e2);                                             \
        a3 = mm_add(a3, e3);                                             \
    }                                                                    \
    vec_t acc = mm_add(mm_add(a0, a1), mm_add(a2, a3));                  \
    alignas(64) double buf[width];                                       \
    mm_store(buf, acc);                                                  \
    double s = 0.0;                                                      \
    for (int64_t k = 0; k < width; ++k) s += buf[k];                     \
    for (; i < n; ++i) {                                                 \
        double e = std::exp(x[i] - m);                                   \
        out[i] = e;                                                      \
        s += e;                                                          \
    }                                                                    \
    return s;                                                            \
}                                                                        \
__attribute__((target("avx512f")))                                       \
inline void row_scale_##fn_suffix(double* out, double inv, int64_t n) {  \
    vec_t vi = mm_set1(inv);                                             \
    int64_t i = 0;                                                       \
    for (; i + width <= n; i += width)                                   \
        mm_store(out + i, mm_mul(mm_load(out + i), vi));                 \
    for (; i < n; ++i) out[i] *= inv;                                    \
}                                                                        \
__attribute__((target("avx512f")))                                       \
inline void row_logmode_##fn_suffix(const double* x, double* out, double m, double lse, int64_t n) { \
    vec_t vm = mm_set1(m), vl = mm_set1(lse);                            \
    int64_t i = 0;                                                       \
    for (; i + width <= n; i += width)                                   \
        mm_store(out + i, mm_sub(mm_sub(mm_load(x + i), vm), vl));       \
    for (; i < n; ++i) out[i] = (x[i] - m) - lse;                        \
}

TP_SOFTMAX_ROW_F64(f64_512, __m512d, 8, tensorplay::tpsleef::exp,
                   _mm512_add_pd, _mm512_sub_pd, _mm512_mul_pd, _mm512_max_pd,
                   _mm512_loadu_pd, _mm512_storeu_pd, _mm512_set1_pd, _mm512_setzero_pd)
TP_SOFTMAX_ROW_F64(f64_256, __m256d, 4, tensorplay::tpsleef::exp,
                   _mm256_add_pd, _mm256_sub_pd, _mm256_mul_pd, _mm256_max_pd,
                   _mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd, _mm256_setzero_pd)
#undef TP_SOFTMAX_ROW_F64

#endif  // __x86_64__

}  // namespace softmax_row

#if defined(__x86_64__)
// Vectorized strided-softmax unit: softmax along the strided dim, lanes across
// the contiguous inner dimension.  One unit = one (outer, lane-block) pair;
// the running max/sum vectors live in registers while sweeping k, so loads and
// stores along the inner dim are coalesced.
template <bool LogMode>
struct SoftmaxStridedCtx {
    const float* in_f32;
    float* out_f32;
    const double* in_f64;
    double* out_f64;
    int64_t size, stride, nblk;
};

template <bool LogMode>
__attribute__((target("avx512f")))
static void softmax_strided_f32_512_unit(const SoftmaxStridedCtx<LogMode>& c, int64_t u) {
    const int64_t a = u / c.nblk;
    const int64_t iv0 = (u % c.nblk) * 16;
    const float* base = c.in_f32 + a * c.size * c.stride + iv0;
    float* obase = c.out_f32 + a * c.size * c.stride + iv0;
    const __m512 vinf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    __m512 vm = vinf;
    for (int64_t k = 0; k < c.size; ++k)
        vm = _mm512_max_ps(vm, _mm512_loadu_ps(base + k * c.stride));
    __m512 vs = _mm512_setzero_ps();
    for (int64_t k = 0; k < c.size; ++k) {
        __m512 e = tensorplay::tpsleef::exp(
            _mm512_sub_ps(_mm512_loadu_ps(base + k * c.stride), vm));
        vs = _mm512_add_ps(vs, e);
        _mm512_storeu_ps(obase + k * c.stride, e);
    }
    if constexpr (LogMode) {
        alignas(64) float sb[16];
        _mm512_storeu_ps(sb, vs);
        for (int64_t l = 0; l < 16; ++l) sb[l] = std::log(sb[l]);
        const __m512 vlse = _mm512_loadu_ps(sb);
        for (int64_t k = 0; k < c.size; ++k) {
            __m512 v = _mm512_sub_ps(_mm512_loadu_ps(base + k * c.stride), vm);
            _mm512_storeu_ps(obase + k * c.stride, _mm512_sub_ps(v, vlse));
        }
    } else {
        const __m512 vinv = _mm512_div_ps(_mm512_set1_ps(1.0f), vs);
        for (int64_t k = 0; k < c.size; ++k) {
            _mm512_storeu_ps(obase + k * c.stride,
                             _mm512_mul_ps(_mm512_loadu_ps(obase + k * c.stride), vinv));
        }
    }
}

template <bool LogMode>
__attribute__((target("avx512f")))
static void softmax_strided_f64_512_unit(const SoftmaxStridedCtx<LogMode>& c, int64_t u) {
    const int64_t a = u / c.nblk;
    const int64_t iv0 = (u % c.nblk) * 8;
    const double* base = c.in_f64 + a * c.size * c.stride + iv0;
    double* obase = c.out_f64 + a * c.size * c.stride + iv0;
    __m512d vm = _mm512_set1_pd(-std::numeric_limits<double>::infinity());
    for (int64_t k = 0; k < c.size; ++k)
        vm = _mm512_max_pd(vm, _mm512_loadu_pd(base + k * c.stride));
    __m512d vs = _mm512_setzero_pd();
    for (int64_t k = 0; k < c.size; ++k) {
        __m512d e = tensorplay::tpsleef::exp(
            _mm512_sub_pd(_mm512_loadu_pd(base + k * c.stride), vm));
        vs = _mm512_add_pd(vs, e);
        _mm512_storeu_pd(obase + k * c.stride, e);
    }
    if constexpr (LogMode) {
        alignas(64) double sb[8];
        _mm512_storeu_pd(sb, vs);
        for (int64_t l = 0; l < 8; ++l) sb[l] = std::log(sb[l]);
        const __m512d vlse = _mm512_loadu_pd(sb);
        for (int64_t k = 0; k < c.size; ++k) {
            __m512d v = _mm512_sub_pd(_mm512_loadu_pd(base + k * c.stride), vm);
            _mm512_storeu_pd(obase + k * c.stride, _mm512_sub_pd(v, vlse));
        }
    } else {
        const __m512d vinv = _mm512_div_pd(_mm512_set1_pd(1.0), vs);
        for (int64_t k = 0; k < c.size; ++k) {
            _mm512_storeu_pd(obase + k * c.stride,
                             _mm512_mul_pd(_mm512_loadu_pd(obase + k * c.stride), vinv));
        }
    }
}
#endif  // __x86_64__

template <bool LogMode>
static Tensor softmax_fused_kernel_impl(const Tensor& self, int64_t dim, DType out_dtype) {
    // A 0-dim input holds one normalization unit: the softmax dim does not
    // exist on the scalar shape, so the value normalizes as a one-element
    // row (softmax answers 1, log-softmax 0) and the scalar shape returns.
    if (self.dim() == 0) {
        const int64_t d = dim < 0 ? dim + 1 : dim;
        if (d != 0) {
            TP_THROW(IndexError, format_dim_range(1, dim));
        }
        Tensor row = self.to(out_dtype).view({1});
        return softmax_fused_kernel_impl<LogMode>(row, 0, out_dtype).view({});
    }
    Tensor input = self.to(out_dtype);
    int64_t d = dim < 0 ? dim + input.dim() : dim;
    if (d < 0 || d >= input.dim()) {
        TP_THROW(IndexError, format_dim_range(input.dim(), dim));
    }

    bool innermost = input.is_contiguous() && (d == input.dim() - 1);
    if (!innermost && input.is_contiguous()) {
        // Strided softmax over contiguous rows of length `size` with stride
        // `stride` (the softmax dim is not last).  One streaming pass per row,
        // no transpose materialization.
        const int64_t size = input.size(d);
        const int64_t stride = input.stride(d);
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(input.shape()), out_dtype, input.device());

        // Iterate rows as (a, b) with a in [0, before), b in [0, after);
        // row data lives at a*(size*stride) + b + k*stride.
        int64_t before = 1, after = 1;
        for (int64_t i = 0; i < d; ++i) before *= input.size(i);
        for (int64_t i = d + 1; i < input.dim(); ++i) after *= input.size(i);

        #define TP_SOFTMAX_STRIDED_ROW(ctype)                              \
            const ctype* src = in + a * size * stride + b;                 \
            ctype* dst = out + a * size * stride + b;                      \
            ctype m = src[0];                                              \
            for (int64_t k = 1; k < size; ++k) m = std::max(m, src[k * stride]); \
            ctype sum = ctype(0);                                          \
            for (int64_t k = 0; k < size; ++k) {                           \
                ctype e = std::exp(src[k * stride] - m);                   \
                dst[k * stride] = e;                                       \
                sum += e;                                                  \
            }                                                              \
            if constexpr (LogMode) {                                       \
                ctype lse = std::log(sum);                                 \
                for (int64_t k = 0; k < size; ++k)                         \
                    dst[k * stride] = (src[k * stride] - m) - lse;         \
            } else {                                                       \
                ctype inv = ctype(1) / sum;                                \
                for (int64_t k = 0; k < size; ++k) dst[k * stride] *= inv; \
            }

#if defined(__x86_64__)
        SoftmaxStridedCtx<LogMode> sctx{
            out_dtype == DType::Float32 ? input.data_ptr<float>() : nullptr,
            out_dtype == DType::Float32 ? result.data_ptr<float>() : nullptr,
            out_dtype == DType::Float64 ? input.data_ptr<double>() : nullptr,
            out_dtype == DType::Float64 ? result.data_ptr<double>() : nullptr,
            size, stride, 0};
        auto run_f32_512 = [&](int64_t ub, int64_t ue) {
            for (int64_t u = ub; u < ue; ++u) softmax_strided_f32_512_unit<LogMode>(sctx, u);
        };
        auto run_f64_512 = [&](int64_t ub, int64_t ue) {
            for (int64_t u = ub; u < ue; ++u) softmax_strided_f64_512_unit<LogMode>(sctx, u);
        };
#endif

        if (out_dtype == DType::Float32) {
#if defined(__x86_64__)
            if (vecunary::avx512_available() && after >= 16 && after % 16 == 0) {
                sctx.nblk = after / 16;
                parallel_for(0, before * sctx.nblk, 1, run_f32_512);
                return result;
            }
            const float* in = sctx.in_f32;
            float* out = sctx.out_f32;
#else
            const float* in = input.data_ptr<float>();
            float* out = result.data_ptr<float>();
#endif
            parallel_for(0, before * after, 1, [&](int64_t rb, int64_t re) {
                for (int64_t r = rb; r < re; ++r) {
                    const int64_t a = r / after;
                    const int64_t b = r % after;
                    TP_SOFTMAX_STRIDED_ROW(float)
                }
            });
        } else {
#if defined(__x86_64__)
            if (vecunary::avx512_available() && after >= 8 && after % 8 == 0) {
                sctx.nblk = after / 8;
                parallel_for(0, before * sctx.nblk, 1, run_f64_512);
                return result;
            }
            const double* in = sctx.in_f64;
            double* out = sctx.out_f64;
#else
            const double* in = input.data_ptr<double>();
            double* out = result.data_ptr<double>();
#endif
            parallel_for(0, before * after, 1, [&](int64_t rb, int64_t re) {
                for (int64_t r = rb; r < re; ++r) {
                    const int64_t a = r / after;
                    const int64_t b = r % after;
                    TP_SOFTMAX_STRIDED_ROW(double)
                }
            });
        }
        #undef TP_SOFTMAX_STRIDED_ROW
        return result;
    }
    if (!innermost) {
        // Non-contiguous input: fall back to the transpose path.
        Tensor t = input.transpose(d, -1);
        if (!t.is_contiguous()) t = t.contiguous();
        Tensor result = softmax_fused_kernel_impl<LogMode>(t, t.dim() - 1, out_dtype);
        return result.transpose(d, -1).contiguous();
    }

    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(input.shape()), out_dtype, input.device());
    int64_t rows = input.numel() / input.size(-1);
    int64_t size = input.size(-1);

    #define SOFTMAX_SCALAR_ROW(ctype) \
        const ctype* row = in + r * size; \
        ctype* orow = out + r * size; \
        ctype m = row[0]; \
        for (int64_t j = 1; j < size; ++j) m = std::max(m, row[j]); \
        ctype sum = ctype(0); \
        for (int64_t j = 0; j < size; ++j) { \
            ctype e = std::exp(row[j] - m); \
            orow[j] = e; \
            sum += e; \
        } \
        if constexpr (LogMode) { \
            ctype lse = std::log(sum); \
            for (int64_t j = 0; j < size; ++j) orow[j] = (row[j] - m) - lse; \
        } else { \
            ctype inv = ctype(1) / sum; \
            for (int64_t j = 0; j < size; ++j) orow[j] *= inv; \
        }

#if defined(__x86_64__)
    #define SOFTMAX_ROW_VEC(ns, suffix, ctype, row_, orow_) \
        { \
            ctype m = softmax_row::row_max_##suffix(row_, size); \
            ctype sum = softmax_row::row_expsum_##suffix(row_, m, orow_, size); \
            if constexpr (LogMode) { \
                softmax_row::row_logmode_##suffix(row_, orow_, m, std::log(sum), size); \
            } else { \
                softmax_row::row_scale_##suffix(orow_, ctype(1) / sum, size); \
            } \
        }
#else
    // No AVX vector rows on this architecture: the CASE macro's use512 /
    // use256 probes return false, so only the scalar row body is ever taken.
    // Bare block, not do/while: the call sites carry no trailing semicolon.
    #define SOFTMAX_ROW_VEC(ns, suffix, ctype, row_, orow_) \
        { (void)row_; (void)orow_; }
#endif

    #define SOFTMAX_CASE(ctype, name, s512, s256, t512, t256) \
    case DType::name: { \
        const ctype* in = input.data_ptr<ctype>(); \
        ctype* out = result.data_ptr<ctype>(); \
        const bool use512 = vecunary::avx512_available(); \
        const bool use256 = vecunary::avx2_available(); \
        parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) { \
            for (int64_t r = begin; r < end; ++r) { \
                const ctype* row = in + r * size; \
                ctype* orow = out + r * size; \
                if (use512 && size >= t512) { \
                    SOFTMAX_ROW_VEC(softmax_row, s512, ctype, row, orow) \
                } else if (use256 && size >= t256) { \
                    SOFTMAX_ROW_VEC(softmax_row, s256, ctype, row, orow) \
                } else { \
                    SOFTMAX_SCALAR_ROW(ctype) \
                } \
            } \
        }); \
        break; \
    }
    switch (out_dtype) {
        SOFTMAX_CASE(float, Float32, f32_512, f32_256, 64, 32)
        SOFTMAX_CASE(double, Float64, f64_512, f64_256, 32, 16)
        default: TP_THROW(TypeError, "softmax: unsupported dtype");
    }
    #undef SOFTMAX_CASE
    #undef SOFTMAX_ROW_VEC
    #undef SOFTMAX_SCALAR_ROW
    return result;
}

Tensor softmax_kernel(const Tensor& self, int64_t dim, DType dtype) {
    DType out_dtype = (dtype != DType::Undefined) ? dtype : self.dtype();
    if (isIntegralType(out_dtype)) out_dtype = DType::Float32;
    if (isReducedFloatingType(out_dtype)) {
        return softmax_fused_kernel_impl<false>(self, dim, DType::Float32).to(out_dtype);
    }
    return softmax_fused_kernel_impl<false>(self, dim, out_dtype);
}

Tensor log_softmax_kernel(const Tensor& self, int64_t dim, DType dtype) {
    DType out_dtype = (dtype != DType::Undefined) ? dtype : self.dtype();
    if (isIntegralType(out_dtype)) out_dtype = DType::Float32;
    if (isReducedFloatingType(out_dtype)) {
        return softmax_fused_kernel_impl<true>(self, dim, DType::Float32).to(out_dtype);
    }
    return softmax_fused_kernel_impl<true>(self, dim, out_dtype);
}

// Dtype for an elementwise op between a tensor and a 0-dim operand created
// from a Python scalar: the placeholder side does not widen the result, it
// adopts the tensor side's dtype. A floating scalar acting on an integer
// tensor still promotes to Float32, a complex scalar to ComplexFloat.
inline DType wrapped_scalar_pair_dtype(DType tensor_dtype, DType scalar_dtype) {
    if (isFloatingOrComplexType(tensor_dtype)) return tensor_dtype;
    if (isComplexType(scalar_dtype)) return DType::ComplexFloat;
    if (isFloatingType(scalar_dtype)) return DType::Float32;
    return tensor_dtype;
}

#if defined(__x86_64__)
namespace {

// Runtime ISA gates; defined with the other x86 helpers further below.
bool pointwise_cpu_has_avx512();
bool pointwise_cpu_has_avx2();

// Vector power over dense same-shape operands.  libmvec-class u10 entries
// evaluate the whole vector at once; the scalar tail falls back to std::pow.
__attribute__((target("avx2,fma")))
void pow_tensor_f32_avx2(const float* base, const float* exp, float* out,
                         int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i,
                         tpsleef::pow(_mm256_loadu_ps(base + i),
                                      _mm256_loadu_ps(exp + i)));
    }
    for (; i < n; ++i) {
        out[i] = std::pow(base[i], exp[i]);
    }
}

__attribute__((target("avx2,fma")))
void pow_tensor_f64_avx2(const double* base, const double* exp, double* out,
                         int64_t n) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        _mm256_storeu_pd(out + i,
                         tpsleef::pow(_mm256_loadu_pd(base + i),
                                      _mm256_loadu_pd(exp + i)));
    }
    for (; i < n; ++i) {
        out[i] = std::pow(base[i], exp[i]);
    }
}

__attribute__((target("avx512f,fma")))
void pow_tensor_f32_avx512(const float* base, const float* exp, float* out,
                           int64_t n) {
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i,
                         tpsleef::pow(_mm512_loadu_ps(base + i),
                                      _mm512_loadu_ps(exp + i)));
    }
    for (; i < n; ++i) {
        out[i] = std::pow(base[i], exp[i]);
    }
}

__attribute__((target("avx512f,fma")))
void pow_tensor_f64_avx512(const double* base, const double* exp, double* out,
                           int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm512_storeu_pd(out + i,
                         tpsleef::pow(_mm512_loadu_pd(base + i),
                                      _mm512_loadu_pd(exp + i)));
    }
    for (; i < n; ++i) {
        out[i] = std::pow(base[i], exp[i]);
    }
}

template <typename T>
void pow_vec_contiguous(const T* base, const T* exp, T* out, int64_t n) {
#if defined(__x86_64__)
    if constexpr (std::is_same_v<T, float>) {
        if (pointwise_cpu_has_avx512()) {
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
                pow_tensor_f32_avx512(base + b, exp + b, out + b, e - b);
            });
            return;
        }
        if (pointwise_cpu_has_avx2()) {
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
                pow_tensor_f32_avx2(base + b, exp + b, out + b, e - b);
            });
            return;
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (pointwise_cpu_has_avx512()) {
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
                pow_tensor_f64_avx512(base + b, exp + b, out + b, e - b);
            });
            return;
        }
        if (pointwise_cpu_has_avx2()) {
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
                pow_tensor_f64_avx2(base + b, exp + b, out + b, e - b);
            });
            return;
        }
    }
#endif
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            out[i] = std::pow(base[i], exp[i]);
        }
    });
}

} // namespace
#endif

// Helper for pow (Tensor, Tensor)
Tensor pow_tensor_tensor_kernel(const Tensor& self, const Tensor& exponent) {
    std::vector<int64_t> out_shape = broadcast_shapes(self.sizes(), exponent.sizes());
    const bool self_wrapped =
        self.dim() == 0 && self.unsafeGetTensorImpl()->is_wrapped_number();
    const bool exp_wrapped =
        exponent.dim() == 0 && exponent.unsafeGetTensorImpl()->is_wrapped_number();
    DType result_dtype = promoteTypes(self.dtype(), exponent.dtype());
    if (self_wrapped != exp_wrapped) {
        result_dtype = self_wrapped
            ? wrapped_scalar_pair_dtype(exponent.dtype(), self.dtype())
            : wrapped_scalar_pair_dtype(self.dtype(), exponent.dtype());
    }

    Tensor result = Tensor::empty(out_shape, result_dtype, self.device());

    Tensor self_c = (self.dtype() == result_dtype) ? self : self.to(result_dtype);
    Tensor exp_c = (exponent.dtype() == result_dtype) ? exponent : exponent.to(result_dtype);

    // Single vectorized pass when both operands are dense, share one shape
    // and stay in a vector-capable float dtype; anything else (broadcasting,
    // promotion, other dtypes) keeps the generic path below.
#if defined(__x86_64__)
    if (!isComplexType(result_dtype) &&
        (result_dtype == DType::Float32 || result_dtype == DType::Float64) &&
        self_c.dtype() == exp_c.dtype() &&
        self_c.is_contiguous() && exp_c.is_contiguous() &&
        self_c.sizes() == exp_c.sizes()) {
        const int64_t n = self_c.numel();
        if (result_dtype == DType::Float32) {
            pow_vec_contiguous<float>(self_c.data_ptr<float>(),
                                      exp_c.data_ptr<float>(),
                                      result.data_ptr<float>(), n);
        } else {
            pow_vec_contiguous<double>(self_c.data_ptr<double>(),
                                       exp_c.data_ptr<double>(),
                                       result.data_ptr<double>(), n);
        }
        return result;
    }
#endif

    if (isComplexType(result_dtype)) {
        // Reduced-width values compute in full precision before narrowing.
        if (result_dtype == DType::ComplexHalf || result_dtype == DType::BComplex32) {
            return pow_tensor_tensor_kernel(self.to(DType::ComplexFloat),
                                            exponent.to(DType::ComplexFloat))
                .to(result_dtype);
        }
        ti_apply_arith(result, self_c, exp_c,
            [](auto b, auto e) {
                using B = decltype(b);
                if constexpr (is_complex_type_v<B>) {
                    using V = typename B::value_type;
                    if constexpr (std::is_same_v<V, float> || std::is_same_v<V, double>) {
                        return tensorplay::pow(b, e);
                    } else {
                        using F = tensorplay::complex<float>;
                        const auto r =
                            tensorplay::pow(
                                F(static_cast<float>(b.real()), static_cast<float>(b.imag())),
                                F(static_cast<float>(e.real()), static_cast<float>(e.imag())));
                        return B(static_cast<V>(r.real()), static_cast<V>(r.imag()));
                    }
                } else {
                    return static_cast<B>(std::pow(static_cast<double>(b), static_cast<double>(e)));
                }
            });
        return result;
    }

    ti_apply_binary(result, self_c, exp_c,
        [](auto b, auto e) { return static_cast<decltype(b)>(std::pow(static_cast<double>(b), static_cast<double>(e))); });
    return result;
}

// Scalar-base power: a base of 1 short-circuits to ones, anything else
// wraps the scalar as a 0-dim tensor and reuses the Tensor_Tensor kernel.
Tensor pow_scalar_tensor_kernel(const Scalar& base, const Tensor& exponent) {
    const DType result_dtype = ops::result_type(base, exponent);
    if (!base.isComplex() && base.toDouble() == 1.0) {
        return Tensor::ones(static_cast<std::vector<int64_t>>(exponent.shape()),
                            result_dtype, exponent.device());
    }
    Tensor base_t = Tensor::full({}, base, result_dtype, exponent.device());
    Tensor exponent_cast = exponent.dtype() == result_dtype
        ? exponent : exponent.to(result_dtype);
    return pow_tensor_tensor_kernel(base_t, exponent_cast);
}

// Lerp implementations using composition
template <typename T, typename W>
inline T lerp_scalar_value(T self, T end, W weight) {
    using compute_t = std::conditional_t<
        std::is_same_v<T, double>, double, float>;
    const compute_t s = static_cast<compute_t>(self);
    const compute_t e = static_cast<compute_t>(end);
    const compute_t w = static_cast<compute_t>(weight);
    const compute_t value = std::abs(w) < compute_t(0.5)
        ? s + w * (e - s)
        : e - (e - s) * (compute_t(1) - w);
    return static_cast<T>(value);
}

inline bool lerp_same_shape(const Tensor& self, const Tensor& end) {
    if (self.dim() != end.dim()) return false;
    for (int64_t d = 0; d < self.dim(); ++d) {
        if (self.size(d) != end.size(d)) return false;
    }
    return true;
}

#if defined(__x86_64__)
namespace {

inline bool pointwise_cpu_has_avx512() {
    static const bool ok = __builtin_cpu_supports("avx512f") != 0 &&
                           __builtin_cpu_supports("fma") != 0;
    return ok;
}

inline bool pointwise_cpu_has_avx512_bf16() {
    static const bool ok = pointwise_cpu_has_avx512() &&
                           __builtin_cpu_supports("avx512bf16") != 0;
    return ok;
}

inline bool pointwise_cpu_has_avx2() {
    static const bool ok = __builtin_cpu_supports("avx2") != 0 &&
                           __builtin_cpu_supports("fma") != 0;
    return ok;
}

inline bool pointwise_cpu_has_avx512dq() {
    static const bool ok = pointwise_cpu_has_avx512() &&
                           __builtin_cpu_supports("avx512dq") != 0;
    return ok;
}

__attribute__((target("avx512f,fma")))
void lerp_f32_avx512(const float* self, const float* end, float* result,
                     int64_t n, float weight) {
    const __m512 w = _mm512_set1_ps(weight);
    const __m512 coeff = std::abs(weight) < 0.5f
        ? w : _mm512_sub_ps(w, _mm512_set1_ps(1.0f));
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 s = _mm512_loadu_ps(self + i);
        const __m512 e = _mm512_loadu_ps(end + i);
        const __m512 b = std::abs(weight) < 0.5f ? s : e;
        _mm512_storeu_ps(result + i,
                         _mm512_fmadd_ps(coeff, _mm512_sub_ps(e, s), b));
    }
    for (; i < n; ++i) {
        result[i] = lerp_scalar_value(self[i], end[i], weight);
    }
}

__attribute__((target("avx512f,fma")))
void lerp_f64_avx512(const double* self, const double* end, double* result,
                     int64_t n, double weight) {
    const __m512d w = _mm512_set1_pd(weight);
    const __m512d coeff = std::abs(weight) < 0.5
        ? w : _mm512_sub_pd(w, _mm512_set1_pd(1.0));
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512d s = _mm512_loadu_pd(self + i);
        const __m512d e = _mm512_loadu_pd(end + i);
        const __m512d b = std::abs(weight) < 0.5 ? s : e;
        _mm512_storeu_pd(result + i,
                         _mm512_fmadd_pd(coeff, _mm512_sub_pd(e, s), b));
    }
    for (; i < n; ++i) {
        result[i] = lerp_scalar_value(self[i], end[i], weight);
    }
}

__attribute__((target("avx512f,avx512bf16,fma")))
void lerp_bf16_avx512(const uint16_t* self, const uint16_t* end,
                      uint16_t* result, int64_t n, float weight) {
    const __m512 w = _mm512_set1_ps(weight);
    const bool small = std::abs(weight) < 0.5f;
    const __m512 coeff = small ? w : _mm512_sub_ps(w, _mm512_set1_ps(1.0f));
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256i sr = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(self + i));
        const __m256i er = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(end + i));
        const __m512 s = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(sr), 16));
        const __m512 e = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(er), 16));
        const __m512 base = small ? s : e;
        const __m512 out = _mm512_fmadd_ps(coeff, _mm512_sub_ps(e, s), base);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + i),
                            (__m256i)_mm512_cvtneps_pbh(out));
    }
    for (; i < n; ++i) {
        const float s = detail::bfloat16_to_float_bits(self[i]);
        const float e = detail::bfloat16_to_float_bits(end[i]);
        result[i] = detail::float_to_bfloat16_bits(
            lerp_scalar_value(s, e, weight));
    }
}

__attribute__((target("avx2,fma")))
void lerp_f32_avx2(const float* self, const float* end, float* result,
                   int64_t n, float weight) {
    const __m256 w = _mm256_set1_ps(weight);
    const __m256 coeff = std::abs(weight) < 0.5f
        ? w : _mm256_sub_ps(w, _mm256_set1_ps(1.0f));
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 s = _mm256_loadu_ps(self + i);
        const __m256 e = _mm256_loadu_ps(end + i);
        const __m256 b = std::abs(weight) < 0.5f ? s : e;
        _mm256_storeu_ps(result + i,
                         _mm256_fmadd_ps(coeff, _mm256_sub_ps(e, s), b));
    }
    for (; i < n; ++i) {
        result[i] = lerp_scalar_value(self[i], end[i], weight);
    }
}

__attribute__((target("avx2,fma")))
void lerp_f64_avx2(const double* self, const double* end, double* result,
                   int64_t n, double weight) {
    const __m256d w = _mm256_set1_pd(weight);
    const __m256d coeff = std::abs(weight) < 0.5
        ? w : _mm256_sub_pd(w, _mm256_set1_pd(1.0));
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d s = _mm256_loadu_pd(self + i);
        const __m256d e = _mm256_loadu_pd(end + i);
        const __m256d b = std::abs(weight) < 0.5 ? s : e;
        _mm256_storeu_pd(result + i,
                         _mm256_fmadd_pd(coeff, _mm256_sub_pd(e, s), b));
    }
    for (; i < n; ++i) {
        result[i] = lerp_scalar_value(self[i], end[i], weight);
    }
}

// Tensor-weight lerp: the weight varies per lane, so the small-weight
// selection and the base operand are blended per lane instead of hoisted.
__attribute__((target("avx2,fma")))
void lerp_tensor_f32_avx2(const float* self, const float* end,
                          const float* weight, float* result, int64_t n) {
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 abs_mask = _mm256_set1_ps(-0.0f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 s = _mm256_loadu_ps(self + i);
        const __m256 e = _mm256_loadu_ps(end + i);
        const __m256 w = _mm256_loadu_ps(weight + i);
        const __m256 small = _mm256_cmp_ps(
            _mm256_andnot_ps(abs_mask, w), half, _CMP_LT_OQ);
        const __m256 coeff = _mm256_blendv_ps(_mm256_sub_ps(w, one), w, small);
        const __m256 base = _mm256_blendv_ps(e, s, small);
        _mm256_storeu_ps(result + i,
                         _mm256_fmadd_ps(coeff, _mm256_sub_ps(e, s), base));
    }
    for (; i < n; ++i) {
        result[i] = lerp_scalar_value(self[i], end[i], weight[i]);
    }
}

__attribute__((target("avx2,fma")))
void lerp_tensor_f64_avx2(const double* self, const double* end,
                          const double* weight, double* result, int64_t n) {
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d abs_mask = _mm256_set1_pd(-0.0);
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d s = _mm256_loadu_pd(self + i);
        const __m256d e = _mm256_loadu_pd(end + i);
        const __m256d w = _mm256_loadu_pd(weight + i);
        const __m256d small = _mm256_cmp_pd(
            _mm256_andnot_pd(abs_mask, w), half, _CMP_LT_OQ);
        const __m256d coeff = _mm256_blendv_pd(_mm256_sub_pd(w, one), w, small);
        const __m256d base = _mm256_blendv_pd(e, s, small);
        _mm256_storeu_pd(result + i,
                         _mm256_fmadd_pd(coeff, _mm256_sub_pd(e, s), base));
    }
    for (; i < n; ++i) {
        result[i] = lerp_scalar_value(self[i], end[i], weight[i]);
    }
}

__attribute__((target("avx512f,avx512dq,fma")))
void lerp_tensor_f32_avx512(const float* self, const float* end,
                            const float* weight, float* result, int64_t n) {
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 abs_mask = _mm512_set1_ps(-0.0f);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 s = _mm512_loadu_ps(self + i);
        const __m512 e = _mm512_loadu_ps(end + i);
        const __m512 w = _mm512_loadu_ps(weight + i);
        const __mmask16 small = _mm512_cmp_ps_mask(
            _mm512_xor_ps(abs_mask, w), half, _CMP_LT_OQ);
        const __m512 coeff = _mm512_mask_blend_ps(
            small, _mm512_sub_ps(w, one), w);
        const __m512 base = _mm512_mask_blend_ps(small, e, s);
        _mm512_storeu_ps(result + i,
                         _mm512_fmadd_ps(coeff, _mm512_sub_ps(e, s), base));
    }
    for (; i < n; ++i) {
        result[i] = lerp_scalar_value(self[i], end[i], weight[i]);
    }
}

__attribute__((target("avx512f,avx512dq,fma")))
void lerp_tensor_f64_avx512(const double* self, const double* end,
                            const double* weight, double* result, int64_t n) {
    const __m512d half = _mm512_set1_pd(0.5);
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d abs_mask = _mm512_set1_pd(-0.0);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512d s = _mm512_loadu_pd(self + i);
        const __m512d e = _mm512_loadu_pd(end + i);
        const __m512d w = _mm512_loadu_pd(weight + i);
        const __mmask8 small = _mm512_cmp_pd_mask(
            _mm512_xor_pd(abs_mask, w), half, _CMP_LT_OQ);
        const __m512d coeff = _mm512_mask_blend_pd(
            small, _mm512_sub_pd(w, one), w);
        const __m512d base = _mm512_mask_blend_pd(small, e, s);
        _mm512_storeu_pd(result + i,
                         _mm512_fmadd_pd(coeff, _mm512_sub_pd(e, s), base));
    }
    for (; i < n; ++i) {
        result[i] = lerp_scalar_value(self[i], end[i], weight[i]);
    }
}

template <typename T>
void lerp_tensor_contiguous(const T* self, const T* end, const T* weight,
                            T* result, int64_t n) {
#if defined(__x86_64__)
    if constexpr (std::is_same_v<T, float>) {
        if (pointwise_cpu_has_avx512dq()) {
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t finish) {
                lerp_tensor_f32_avx512(self + begin, end + begin,
                                       weight + begin, result + begin,
                                       finish - begin);
            });
            return;
        }
        if (pointwise_cpu_has_avx2()) {
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t finish) {
                lerp_tensor_f32_avx2(self + begin, end + begin,
                                     weight + begin, result + begin,
                                     finish - begin);
            });
            return;
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (pointwise_cpu_has_avx512dq()) {
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t finish) {
                lerp_tensor_f64_avx512(self + begin, end + begin,
                                       weight + begin, result + begin,
                                       finish - begin);
            });
            return;
        }
        if (pointwise_cpu_has_avx2()) {
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t finish) {
                lerp_tensor_f64_avx2(self + begin, end + begin,
                                     weight + begin, result + begin,
                                     finish - begin);
            });
            return;
        }
    }
#endif
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t finish) {
        for (int64_t i = begin; i < finish; ++i) {
            result[i] = lerp_scalar_value(self[i], end[i], weight[i]);
        }
    });
}

// Single-pass fused lerp for three same-shape dense f32/f64 operands;
// anything else keeps the general broadcasting/promotion path.
bool lerp_tensor_fast(const Tensor& self, const Tensor& end,
                      const Tensor& weight, Tensor& result) {
    if (!self.is_contiguous() || !end.is_contiguous() ||
        !weight.is_contiguous()) {
        return false;
    }
    if (!lerp_same_shape(self, end) || !lerp_same_shape(self, weight)) {
        return false;
    }
    const DType dt = self.dtype();
    if (end.dtype() != dt || weight.dtype() != dt ||
        (dt != DType::Float32 && dt != DType::Float64)) {
        return false;
    }
    result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                           dt, self.device());
    const int64_t n = self.numel();
    if (dt == DType::Float32) {
        lerp_tensor_contiguous<float>(self.data_ptr<float>(),
                                      end.data_ptr<float>(),
                                      weight.data_ptr<float>(),
                                      result.data_ptr<float>(), n);
    } else {
        lerp_tensor_contiguous<double>(self.data_ptr<double>(),
                                       end.data_ptr<double>(),
                                       weight.data_ptr<double>(),
                                       result.data_ptr<double>(), n);
    }
    return true;
}

} // namespace
#endif

template <typename T, typename W>
void lerp_scalar_contiguous(const T* self, const T* end, T* result,
                            int64_t n, W weight) {
#if defined(__x86_64__)
    if constexpr (std::is_same_v<T, float>) {
        if (pointwise_cpu_has_avx512()) {
            const float w = static_cast<float>(weight);
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t finish) {
                lerp_f32_avx512(self + begin, end + begin, result + begin,
                                finish - begin, w);
            });
            return;
        }
        if (pointwise_cpu_has_avx2()) {
            const float w = static_cast<float>(weight);
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t finish) {
                lerp_f32_avx2(self + begin, end + begin, result + begin,
                              finish - begin, w);
            });
            return;
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (pointwise_cpu_has_avx512()) {
            const double w = static_cast<double>(weight);
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t finish) {
                lerp_f64_avx512(self + begin, end + begin, result + begin,
                                finish - begin, w);
            });
            return;
        }
        if (pointwise_cpu_has_avx2()) {
            const double w = static_cast<double>(weight);
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t finish) {
                lerp_f64_avx2(self + begin, end + begin, result + begin,
                              finish - begin, w);
            });
            return;
        }
    } else if constexpr (std::is_same_v<T, BFloat16>) {
        if (pointwise_cpu_has_avx512_bf16()) {
            const auto* s = reinterpret_cast<const uint16_t*>(self);
            const auto* e = reinterpret_cast<const uint16_t*>(end);
            auto* r = reinterpret_cast<uint16_t*>(result);
            const float w = static_cast<float>(weight);
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t finish) {
                lerp_bf16_avx512(s + begin, e + begin, r + begin,
                                 finish - begin, w);
            });
            return;
        }
    }
#endif
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t finish) {
        for (int64_t i = begin; i < finish; ++i) {
            result[i] = lerp_scalar_value(self[i], end[i], weight);
        }
    });
}

// TensorIterator's generic lerp composition is several full-tensor passes
// for reduced floating types.  Muon calls scalar-weight lerp twice per
// both operands are already dense and have the same dtype.  The general
// broadcasting/promotion path below remains the semantic fallback.
template <typename W>
bool lerp_scalar_fast(const Tensor& self, const Tensor& end, Tensor& result,
                      W weight) {
    if (!self.is_contiguous() || !end.is_contiguous() ||
        self.dtype() != end.dtype() || !lerp_same_shape(self, end)) {
        return false;
    }
    const int64_t n = self.numel();
    switch (self.dtype()) {
        case DType::Float16:
            lerp_scalar_contiguous(self.data_ptr<Half>(), end.data_ptr<Half>(),
                                   result.data_ptr<Half>(), n,
                                   static_cast<float>(weight));
            return true;
        case DType::BFloat16:
            lerp_scalar_contiguous(self.data_ptr<BFloat16>(),
                                   end.data_ptr<BFloat16>(),
                                   result.data_ptr<BFloat16>(), n,
                                   static_cast<float>(weight));
            return true;
        case DType::Float32:
            lerp_scalar_contiguous(self.data_ptr<float>(), end.data_ptr<float>(),
                                   result.data_ptr<float>(), n,
                                   static_cast<float>(weight));
            return true;
        case DType::Float64:
            lerp_scalar_contiguous(self.data_ptr<double>(),
                                   end.data_ptr<double>(),
                                   result.data_ptr<double>(), n,
                                   static_cast<double>(weight));
            return true;
        default:
            return false;
    }
}

Tensor lerp_tensor_kernel(const Tensor& self, const Tensor& end, const Tensor& weight) {
    DType common_dtype = promoteTypes(self.dtype(), end.dtype());
    common_dtype = promoteTypes(common_dtype, weight.dtype());
    if (isIntegralType(common_dtype)) common_dtype = DType::Float32;

    // Single pass over three dense same-shape operands; the generic
    // composition below spends several full-tensor passes and temporaries.
    if (self.dtype() == common_dtype && end.dtype() == common_dtype &&
        weight.dtype() == common_dtype) {
        Tensor result;
        if (lerp_tensor_fast(self, end, weight, result)) {
            return result;
        }
    }

    // result = self + weight * (end - self)
    // Ensure all operands are cast to common_dtype
    Tensor s = self.to(common_dtype);
    Tensor e = end.to(common_dtype);
    Tensor w = weight.to(common_dtype);

    return s + w * (e - s);
}

Tensor lerp_scalar_kernel(const Tensor& self, const Tensor& end, const Scalar& weight) {
    if (self.dtype() == end.dtype() && lerp_same_shape(self, end) &&
        self.is_contiguous() && end.is_contiguous() &&
        (self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16 ||
         self.dtype() == DType::Float32 || self.dtype() == DType::Float64)) {
        // Native lerp keeps the reduced floating output dtype even though its
        // arithmetic is performed in float32.  This also avoids the three
        // temporary tensors used by the generic composition above.
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                                      self.dtype(), self.device());
        if (lerp_scalar_fast(self, end, result, weight.to<double>())) {
            return result;
        }
    }

    DType common_dtype = promoteTypes(self.dtype(), end.dtype());
    if (weight.isFloatingPoint()) common_dtype = promoteTypes(common_dtype, DType::Float32);
    if (isIntegralType(common_dtype)) common_dtype = DType::Float32;

    Tensor s = self.to(common_dtype);
    Tensor e = end.to(common_dtype);

    double w = weight.toDouble();
    if (std::abs(w) < 0.5) {
        return s + weight * (e - s);
    }
    return e - (e - s) * (1.0 - w);
}

Tensor& lerp_scalar_inplace_kernel(Tensor& self, const Tensor& end, const Scalar& weight) {
    if (self.dtype() == end.dtype() && lerp_same_shape(self, end) &&
        self.is_contiguous() && end.is_contiguous() &&
        (self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16 ||
         self.dtype() == DType::Float32 || self.dtype() == DType::Float64)) {
        if (lerp_scalar_fast(self, end, self, weight.to<double>())) {
            return self;
        }
    }
    self.copy_(lerp_scalar_kernel(self, end, weight));
    return self;
}

Tensor& lerp_tensor_inplace_kernel(Tensor& self, const Tensor& end, const Tensor& weight) {
    self.copy_(lerp_tensor_kernel(self, end, weight));
    return self;
}

Tensor& abs_inplace_kernel(Tensor& self) {
    self.copy_(abs_kernel(self));
    return self;
}

Tensor& neg_inplace_kernel(Tensor& self) {
    self.copy_(neg_kernel(self));
    return self;
}

Tensor& sqrt_inplace_kernel(Tensor& self) {
    self.copy_(sqrt_kernel(self));
    return self;
}

Tensor& rsqrt_inplace_kernel(Tensor& self) {
    self.copy_(rsqrt_kernel(self));
    return self;
}

TENSORPLAY_LIBRARY_IMPL(CPU, PointwiseKernels) {
    m.impl("abs", abs_kernel);
    m.impl("neg", neg_kernel);
    m.impl("square", square_kernel);
    m.impl("sign", sign_kernel);
    m.impl("floor", floor_kernel);
    m.impl("ceil", ceil_kernel);
    m.impl("round", round_kernel);
    m.impl("acos", acos_kernel);
    m.impl("acosh", acosh_kernel);
    m.impl("asin", asin_kernel);
    m.impl("asinh", asinh_kernel);
    m.impl("atan", atan_kernel);
    m.impl("atanh", atanh_kernel);
    m.impl("cos", cos_kernel);
    m.impl("cosh", cosh_kernel);
    m.impl("sin", sin_kernel);
    m.impl("sin.out", sin_out_cpu);
    m.impl("sinh", sinh_kernel);
    m.impl("tan", tan_kernel);
    m.impl("tanh", tanh_kernel);
    m.impl("exp", exp_kernel);
    m.impl("expm1", expm1_kernel);
    m.impl("erf", erf_kernel);
    m.impl("erfc", erfc_kernel);
    m.impl("log", log_kernel);
    m.impl("log10", log10_kernel);
    m.impl("log1p", log1p_kernel);
    m.impl("log2", log2_kernel);
    m.impl("lgamma", lgamma_kernel);
    m.impl("sqrt", sqrt_kernel);
    m.impl("rsqrt", rsqrt_kernel);
    m.impl("frac", frac_kernel);
    m.impl("trunc", trunc_kernel);
    m.impl("sigmoid", sigmoid_kernel);
    m.impl("relu", relu_kernel);
    m.impl("relu_", relu_inplace_kernel);
    m.impl("gelu", gelu_kernel);
    m.impl("gelu_backward", gelu_backward_kernel);
    m.impl("silu", silu_kernel);
    m.impl("silu_backward", silu_backward_kernel_impl);
    m.impl("silu_mul", silu_mul_cpu);
    m.impl("fused_swiglu", fused_swiglu_cpu);
    m.impl("silu_and_mul", silu_and_mul_cpu);
    m.impl("hardtanh", hardtanh_kernel_impl);
    m.impl("hardtanh_backward", hardtanh_backward_kernel_impl);
    m.impl("relu6", relu6_kernel_impl);
    m.impl("hardswish", hardswish_kernel_impl);
    m.impl("hardswish_backward", hardswish_backward_kernel_impl);
    m.impl("sec", sec_kernel_impl);
    m.impl("csc", csc_kernel_impl);
    m.impl("cot", cot_kernel_impl);
    m.impl("tanhshrink", tanhshrink_kernel_impl);
    m.impl("squared_difference", squared_difference_kernel_impl);
    m.impl("swish", silu_kernel);
    m.impl("hardsigmoid", hardsigmoid_kernel_impl);
    m.impl("hardsigmoid_backward", hardsigmoid_backward_kernel_impl);
    m.impl("leaky_relu", leaky_relu_kernel_impl);
    m.impl("leaky_relu_backward", leaky_relu_backward_kernel_impl);
    m.impl("elu", elu_kernel_impl);
    m.impl("elu_backward", elu_backward_kernel_impl);
    m.impl("mish", mish_kernel_impl);
    m.impl("mish_backward", mish_backward_kernel_impl);
    m.impl("softplus", softplus_kernel_impl);
    m.impl("softplus_backward", softplus_backward_kernel_impl);
    m.impl("log_sigmoid", log_sigmoid_kernel_impl);
    m.impl("log_sigmoid_backward", log_sigmoid_backward_kernel_impl);
    m.impl("rrelu_with_noise_backward", rrelu_with_noise_backward_kernel_impl);
    m.impl("pow.Tensor_Scalar", pow_scalar_kernel);
    m.impl("angle", angle_kernel);
    m.impl("clamp", clamp_kernel);
    m.impl("clamp_min", clamp_min_kernel);
    m.impl("clamp_max", clamp_max_kernel);
    m.impl("clamp_min_", clamp_min__kernel);
    m.impl("clamp_max_", clamp_max__kernel);
    m.impl("clamp_backward", clamp_backward_kernel);
    m.impl("threshold_backward", threshold_backward_kernel);
    m.impl("softmax", softmax_kernel);
    m.impl("log_softmax", log_softmax_kernel);
    m.impl("pow.Tensor_Tensor", pow_tensor_tensor_kernel);
    m.impl("pow.Scalar", pow_scalar_tensor_kernel);
    m.impl("lerp", lerp_scalar_kernel);
    m.impl("lerp.Tensor", lerp_tensor_kernel);
    m.impl("lerp_.Scalar", lerp_scalar_inplace_kernel);
    m.impl("lerp_.Tensor", lerp_tensor_inplace_kernel);
    m.impl("abs_", abs_inplace_kernel);
    m.impl("neg_", neg_inplace_kernel);
    m.impl("sqrt_", sqrt_inplace_kernel);
    m.impl("rsqrt_", rsqrt_inplace_kernel);
}

} // namespace cpu
} // namespace tensorplay
