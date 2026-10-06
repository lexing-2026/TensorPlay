// BLAS-level products outside the matmul family proper: addbmm, addmv,
// addr, vdot.
//
// Row-major operands go to the CBLAS row-major entry points directly;
// transposed views are consumed without a copy by flipping the operation
// flag.  Half/BFloat16 accumulate in float (opmath), matching the reduction
// contract used across the other GEMM-family kernels.

#include "Tensor.h"
#include "TypePromotion.h"
#include "Scalar.h"
#include "Exception.h"
#include "Parallel.h"
#include "Utils.h"
#include "Complex.h"
#include "cpu/BlockGemm.h"
#include "../BlasComposite.h"

#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <mutex>

// The generated block-product kernels of the convolution library, where the
// build has them.
#ifdef USE_ONEDNN
#include "oneapi/dnnl/dnnl_config.h"
#if defined(DNNL_EXPERIMENTAL_UKERNEL) && (defined(__x86_64__) || defined(_M_X64))
#include "oneapi/dnnl/dnnl_ukernel.hpp"
#define TP_BLOCK_GEMM_UKERNEL 1
#endif
#endif
#ifndef TP_BLOCK_GEMM_UKERNEL
#define TP_BLOCK_GEMM_UKERNEL 0
#endif

#ifdef USE_MKL
#include <mkl.h>
#elif defined(USE_BLAS)
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

namespace {

// Whether all three operands are floating -- what the kernels here serve,
// mixed precisions included; complex, whole-number and truth-value operands
// are composed from the products.
bool all_floating(const Tensor& self, const Tensor& a, const Tensor& b) {
    return isFloatingType(self.dtype()) && isFloatingType(a.dtype()) && isFloatingType(b.dtype());
}

bool is_cplx(DType d) {
    return d == DType::ComplexFloat || d == DType::ComplexDouble;
}

// Opmath: float accumulate for half/bf16 storage, native type otherwise.
template <typename T> struct OpMath { using type = T; };
template <> struct OpMath<Half> { using type = float; };
template <> struct OpMath<BFloat16> { using type = float; };

// y = alpha * mat @ x + beta * self_b, computed row-wise in opmath
// precision.  Used by addmv for half/bf16 inputs and in builds without a
// BLAS.
template <typename T>
void addmv_rows(const Tensor& mat, const Tensor& vec, const Tensor& self_b,
                Tensor& out, double alpha, double beta) {
    using M = typename OpMath<T>::type;
    const T* mp = mat.data_ptr<T>();
    const T* vp = vec.data_ptr<T>();
    const T* sp = self_b.data_ptr<T>();
    T* op = out.data_ptr<T>();
    const int64_t m = out.numel(), k = vec.numel();
    const M av = static_cast<M>(alpha), bv = static_cast<M>(beta);
    parallel_for(0, m, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const T* row = mp + i * k;
            M acc{};
            for (int64_t j = 0; j < k; ++j) {
                acc += static_cast<M>(row[j]) * static_cast<M>(vp[j]);
            }
            op[i] = static_cast<T>(bv * static_cast<M>(sp[i]) + av * acc);
        }
    });
}

// work += batch1[bi] @ batch2[bi] accumulated in opmath precision, walking
// the product in M-K-N order so both factor reads stay row-major.
template <typename T, typename Acc>
void bmm_accumulate(const Tensor& batch1, const Tensor& batch2, int64_t bi,
                    Tensor& work, int64_t n, int64_t p, int64_t m) {
    const T* A = batch1.data_ptr<T>() + bi * n * p;
    const T* B = batch2.data_ptr<T>() + bi * p * m;
    Acc* W = work.data_ptr<Acc>();
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            Acc* wrow = W + i * m;
            for (int64_t k = 0; k < p; ++k) {
                const Acc a_val = static_cast<Acc>(A[i * p + k]);
                if (a_val == Acc(0)) continue;
                const T* brow = B + k * m;
                for (int64_t j = 0; j < m; ++j) {
                    wrow[j] += a_val * static_cast<Acc>(brow[j]);
                }
            }
        }
    });
}

// out = beta * self_b + alpha * work, evaluated in double like the scalar
// epilogues of the other low-precision GEMM paths.  Acc is the workspace
// element type: float for half/bfloat16/float32 reductions, double for
// float64.
template <typename T, typename Acc>
void addbmm_epilogue(Tensor& out, const Tensor& self_acc, const Tensor& work,
                     double beta, double alpha) {
    const T* sp = self_acc.data_ptr<T>();
    const Acc* wp = work.data_ptr<Acc>();
    T* op = out.data_ptr<T>();
    const int64_t total = out.numel();
    parallel_for(0, total, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            op[i] = static_cast<T>(beta * static_cast<double>(sp[i]) +
                                   alpha * static_cast<double>(wp[i]));
        }
    });
}
}  // namespace

// ---------------------------------------------------------------------------
// addmv: beta * self + alpha * (mat @ vec)
// ---------------------------------------------------------------------------

Tensor addmv_cpu(const Tensor& self, const Tensor& mat, const Tensor& vec,
                 const Scalar& beta, const Scalar& alpha) {
    if (!all_floating(self, mat, vec)) {
        return blas_composite::addmv(self, mat, vec, beta, alpha);
    }
    if (mat.dim() != 2) TP_THROW(RuntimeError, "addmv: mat must be a matrix");
    if (vec.dim() != 1) TP_THROW(RuntimeError, "addmv: vec must be a vector");
    const int64_t m = mat.size(0), k = mat.size(1);
    if (vec.numel() != k)
        TP_THROW(RuntimeError, "addmv: both args should have matching shapes");
    const DType dt = promoteTypes(promoteTypes(mat.dtype(), vec.dtype()), self.dtype());
    const DType cdt = (dt == DType::Float64) ? DType::Float64 : DType::Float32;
    const double alpha_v = alpha.toDouble();
    const double beta_v = beta.toDouble();

#if defined(USE_MKL) || defined(USE_BLAS)
    if (dt == cdt) {
        // Native GEMV.  y is seeded with the broadcast self and beta is
        // applied by the call itself; beta == 0 leaves y unread, so the seed
        // copy is skipped and the output buffer stays uninitialized.  Every
        // operand is read at the compute dtype, whatever precision it came in.
        Tensor result = beta_v != 0.0
            ? detail::contiguous_clone((self.dtype() == dt ? self : self.to(dt)).expand({m}))
            : Tensor::empty({m}, dt, mat.device());
        Tensor xc = vec.dtype() == dt ? vec : vec.to(dt);
        if (!xc.is_contiguous()) xc = detail::contiguous_clone(xc);
        Tensor a_input = mat.dtype() == dt ? mat : mat.to(dt);
        int64_t lda = k;
        bool trans = false;
        if (a_input.is_contiguous()) {
            lda = k;
        } else if (a_input.stride(0) == 1 && a_input.stride(1) == m) {
            trans = true;
            lda = m;
        } else {
            a_input = detail::contiguous_clone(a_input);
            lda = k;
        }
        if (dt == DType::Float32) {
            cblas_sgemv(CblasRowMajor, trans ? CblasTrans : CblasNoTrans,
                        static_cast<int>(m), static_cast<int>(k),
                        static_cast<float>(alpha_v),
                        a_input.data_ptr<float>(), static_cast<int>(lda),
                        xc.data_ptr<float>(), 1,
                        static_cast<float>(beta_v),
                        result.data_ptr<float>(), 1);
        } else {
            cblas_dgemv(CblasRowMajor, trans ? CblasTrans : CblasNoTrans,
                        static_cast<int>(m), static_cast<int>(k),
                        alpha_v,
                        a_input.data_ptr<double>(), static_cast<int>(lda),
                        xc.data_ptr<double>(), 1,
                        beta_v, result.data_ptr<double>(), 1);
        }
        return result;
    }
#endif

    // Half/BFloat16 (and no-BLAS builds): upcast to the compute dtype and
    // run the row-wise opmath reduction.
    Tensor mc = mat.contiguous().to(cdt);
    Tensor vc = vec.contiguous().to(cdt);
    Tensor self_b = self.expand({m}).contiguous().to(cdt);
    Tensor out = Tensor::empty({m}, dt, mat.device());
    if (cdt == DType::Float32) {
        addmv_rows<float>(mc, vc, self_b, out, alpha_v, beta_v);
    } else {
        addmv_rows<double>(mc, vc, self_b, out, alpha_v, beta_v);
    }
    return out;
}

// ---------------------------------------------------------------------------
// addbmm: beta * self + alpha * sum_i batch1[i] @ batch2[i]
// ---------------------------------------------------------------------------

Tensor addbmm_cpu(const Tensor& self, const Tensor& batch1, const Tensor& batch2,
                  const Scalar& beta_arg, const Scalar& alpha) {
    Scalar beta = beta_arg;
    if (!all_floating(self, batch1, batch2)) {
        return blas_composite::addbmm(self, batch1, batch2, beta, alpha);
    }
    if (batch1.dim() != 3) TP_THROW(RuntimeError, "batch1 must be a 3D tensor");
    if (batch2.dim() != 3) TP_THROW(RuntimeError, "batch2 must be a 3D tensor");
    if (batch1.size(0) != batch2.size(0) || batch1.size(2) != batch2.size(1)) {
        TP_THROW(RuntimeError, "Incompatible matrix sizes for bmm (",
                 batch1.size(1), "x", batch1.size(2), " and ",
                 batch2.size(1), "x", batch2.size(2), ")");
    }
    const int64_t b = batch1.size(0), n = batch1.size(1);
    const int64_t p = batch1.size(2), m = batch2.size(2);
    const DType dt = promoteTypes(batch1.dtype(), batch2.dtype());
    const double beta_v = beta.toDouble();
    const double alpha_v = alpha.toDouble();

#if defined(USE_MKL) || defined(USE_BLAS)
    if (dt == DType::Float32 || dt == DType::Float64) {
        // One accumulating GEMM chain over the broadcast self: the first
        // call carries beta, subsequent calls accumulate with beta = 1, so
        // the cross-batch sum never allocates an intermediate product.
        Tensor result = dt == self.dtype()
            ? detail::contiguous_clone(self.expand({n, m}))
            : detail::contiguous_clone(self.expand({n, m}).to(dt));
        Tensor b1 = batch1.dtype() == dt ? batch1.contiguous()
                                         : batch1.to(dt).contiguous();
        Tensor b2 = batch2.dtype() == dt ? batch2.contiguous()
                                         : batch2.to(dt).contiguous();
        for (int64_t bi = 0; bi < b; ++bi) {
            const double beta_i = bi == 0 ? beta_v : 1.0;
            if (dt == DType::Float32) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<int>(n), static_cast<int>(m),
                            static_cast<int>(p), static_cast<float>(alpha_v),
                            b1.data_ptr<float>() + bi * n * p, static_cast<int>(p),
                            b2.data_ptr<float>() + bi * p * m, static_cast<int>(m),
                            static_cast<float>(beta_i),
                            result.data_ptr<float>(), static_cast<int>(m));
            } else {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<int>(n), static_cast<int>(m),
                            static_cast<int>(p), alpha_v,
                            b1.data_ptr<double>() + bi * n * p, static_cast<int>(p),
                            b2.data_ptr<double>() + bi * p * m, static_cast<int>(m),
                            beta_i, result.data_ptr<double>(), static_cast<int>(m));
            }
        }
        return result;
    }
#endif

    // Half/BFloat16 (and no-BLAS builds): accumulate the cross-batch sum in
    // a workspace held at the reduction's opmath precision (float, or double
    // for float64 inputs -- the workspace dtype must track the accumulator,
    // or the accumulate pass writes past the allocation), then apply
    // beta/alpha in one epilogue pass.  Both factors are converted to the
    // compute dtype up front so the accumulate pass reads them at one width.
    const DType work_dt = dt == DType::Float64 ? DType::Float64 : DType::Float32;
    Tensor work = Tensor::zeros({n, m}, work_dt, self.device());
    Tensor b1 = batch1.dtype() == dt ? batch1.contiguous()
                                     : batch1.to(dt).contiguous();
    Tensor b2 = batch2.dtype() == dt ? batch2.contiguous()
                                     : batch2.to(dt).contiguous();
    for (int64_t bi = 0; bi < b; ++bi) {
        switch (b1.dtype()) {
            case DType::Float32:
                bmm_accumulate<float, float>(b1, b2, bi, work, n, p, m);
                break;
            case DType::Float64:
                bmm_accumulate<double, double>(b1, b2, bi, work, n, p, m);
                break;
            case DType::Float16:
                bmm_accumulate<Half, float>(b1, b2, bi, work, n, p, m);
                break;
            case DType::BFloat16:
                bmm_accumulate<BFloat16, float>(b1, b2, bi, work, n, p, m);
                break;
            default:
                TP_THROW(TypeError, "addbmm: unsupported dtype");
        }
    }
    // self stays in its own dtype when it matches the output; the epilogue
    // reads it as T and promotes to double per element.
    Tensor self_acc = self.dtype() == dt
        ? detail::contiguous_clone(self.expand({n, m}))
        : detail::contiguous_clone(self.expand({n, m}).to(dt));
    Tensor out = Tensor::empty({n, m}, dt, self.device());
    if (dt == DType::Float32) {
        addbmm_epilogue<float, float>(out, self_acc, work, beta_v, alpha_v);
    } else if (dt == DType::Float64) {
        addbmm_epilogue<double, double>(out, self_acc, work, beta_v, alpha_v);
    } else if (dt == DType::Float16) {
        addbmm_epilogue<Half, float>(out, self_acc, work, beta_v, alpha_v);
    } else if (dt == DType::BFloat16) {
        addbmm_epilogue<BFloat16, float>(out, self_acc, work, beta_v, alpha_v);
    } else {
        TP_THROW(TypeError, "addbmm: unsupported dtype");
    }
    return out;
}

// ---------------------------------------------------------------------------
// addr: beta * self + alpha * vec1 (outer) vec2
// ---------------------------------------------------------------------------

Tensor addr_cpu(const Tensor& self, const Tensor& vec1, const Tensor& vec2,
                const Scalar& beta, const Scalar& alpha) {
    if (!all_floating(self, vec1, vec2)) {
        return blas_composite::addr(self, vec1, vec2, beta, alpha);
    }
    const int64_t m = vec1.numel(), k = vec2.numel();
    const DType dt = promoteTypes(promoteTypes(vec1.dtype(), vec2.dtype()), self.dtype());
    blas_composite::check_scalar(dt, beta, "beta");
    blas_composite::check_scalar(dt, alpha, "alpha");
    const DType cdt = (dt == DType::Float64) ? DType::Float64 : DType::Float32;
    const Tensor v1 = vec1.contiguous().to(cdt);
    const Tensor v2 = vec2.contiguous().to(cdt);
    const Tensor self_b = self.expand({m, k}).contiguous().to(cdt);
    const Tensor out = Tensor::empty({m, k}, dt, self.device());
    const double beta_d = beta.toDouble(), alpha_d = alpha.toDouble();
#define TP_ADDR_ACC(ctype, name_, acct)                                        \
    case DType::name_: {                                                       \
        const acct* a = v1.data_ptr<acct>();                                   \
        const acct* bv = v2.data_ptr<acct>();                                  \
        const acct* sp = self_b.data_ptr<acct>();                              \
        ctype* dp = out.data_ptr<ctype>();                                     \
        parallel_for(0, m, GRAIN_SIZE, [&](int64_t begin, int64_t end) {       \
            for (int64_t i = begin; i < end; ++i) {                            \
                for (int64_t j = 0; j < k; ++j) {                              \
                    dp[i * k + j] = static_cast<ctype>(                        \
                        beta_d * sp[i * k + j] + alpha_d * a[i] * bv[j]);      \
                }                                                              \
            }                                                                  \
        });                                                                    \
        break;                                                                 \
    }
    switch (dt) {
        TP_ADDR_ACC(float, Float32, float)
        TP_ADDR_ACC(double, Float64, double)
        TP_ADDR_ACC(BFloat16, BFloat16, float)
        TP_ADDR_ACC(Half, Float16, float)
        default: TP_THROW(TypeError, "addr: unsupported dtype");
    }
#undef TP_ADDR_ACC
    return out;
}

// ---------------------------------------------------------------------------
// vdot: conj(a) . b over the flattened operands
// ---------------------------------------------------------------------------

Tensor vdot_cpu(const Tensor& a_in, const Tensor& b_in) {
    Tensor a = a_in.contiguous().reshape({a_in.numel()});
    Tensor b = b_in.contiguous().reshape({b_in.numel()});
    if (a.numel() != b.numel()) TP_THROW(RuntimeError, "vdot: sizes don't match");
    const int64_t n = a.numel();
    const DType dt = a_in.dtype();
    // The BLAS entries need matching operand dtypes; a silently promotes b.
    if (b.dtype() != dt) b = b.to(dt).contiguous();

    if (is_cplx(dt)) {
#if defined(USE_MKL)
        // Conjugating dot products from the BLAS: single pass, no copies.
        Tensor result = Tensor::empty({}, dt, a.device());
        if (dt == DType::ComplexFloat) {
            complex<float> out{};
            cblas_cdotc_sub(static_cast<int>(n), a.data_ptr<complex<float>>(), 1,
                            b.data_ptr<complex<float>>(), 1, &out);
            result.data_ptr<complex<float>>()[0] = out;
        } else {
            complex<double> out{};
            cblas_zdotc_sub(static_cast<int>(n), a.data_ptr<complex<double>>(), 1,
                            b.data_ptr<complex<double>>(), 1, &out);
            result.data_ptr<complex<double>>()[0] = out;
        }
        return result;
#else
        if (dt == DType::ComplexFloat) {
            const complex<float>* ap = a.data_ptr<complex<float>>();
            const complex<float>* bp = b.data_ptr<complex<float>>();
            complex<double> acc = 0;
            for (int64_t i = 0; i < n; ++i) {
                const complex<float> conjugate(ap[i].real(), -ap[i].imag());
                acc += static_cast<complex<double>>(conjugate) *
                       static_cast<complex<double>>(bp[i]);
            }
            return Tensor::full({}, Scalar(complex<float>(
                                     static_cast<float>(acc.real()),
                                     static_cast<float>(acc.imag()))),
                                dt, a.device());
        }
        const complex<double>* ap = a.data_ptr<complex<double>>();
        const complex<double>* bp = b.data_ptr<complex<double>>();
        complex<double> acc = 0;
        for (int64_t i = 0; i < n; ++i) {
            const complex<double> conjugate(ap[i].real(), -ap[i].imag());
            acc += conjugate * bp[i];
        }
        return Tensor::full({}, Scalar(acc), dt, a.device());
#endif
    }

    Tensor result = Tensor::empty({}, dt == DType::Float64 ? DType::Float64 : DType::Float32,
                                  a.device());
#if defined(USE_MKL)
    if (dt == DType::Float32) {
        // Double-precision accumulation over single-precision operands.
        result.data_ptr<float>()[0] = static_cast<float>(
            cblas_dsdot(static_cast<int>(n), a.data_ptr<float>(), 1,
                        b.data_ptr<float>(), 1));
        return result;
    }
    if (dt == DType::Float64) {
        result.data_ptr<double>()[0] =
            cblas_ddot(static_cast<int>(n), a.data_ptr<double>(), 1,
                       b.data_ptr<double>(), 1);
        return result;
    }
#endif
    // Half/BFloat16 (and non-MKL builds): parallel partial sums in double,
    // combined serially so the reduction stays deterministic.
    Tensor a32;
    Tensor b32;
    if (dt != DType::Float64) {
        a32 = a.to(DType::Float32);
        b32 = b.to(DType::Float32);
    }
    const bool is64 = dt == DType::Float64;
    double total = 0;
    {
        // Per-thread partial sums combined serially: no lock on the workers,
        // and the fixed lane order keeps the sum deterministic.
        const int nthreads = std::max(1, get_num_threads());
        std::vector<double> partials(static_cast<size_t>(nthreads), 0.0);
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            double part = 0;
            if (is64) {
                const double* ap = a.data_ptr<double>();
                const double* bp = b.data_ptr<double>();
                for (int64_t i = begin; i < end; ++i) part += ap[i] * bp[i];
            } else {
                const float* ap = a32.data_ptr<float>();
                const float* bp = b32.data_ptr<float>();
                for (int64_t i = begin; i < end; ++i) part += static_cast<double>(ap[i]) * bp[i];
            }
            partials[static_cast<size_t>(get_thread_num())] += part;
        });
        for (double part : partials) total += part;
    }
    if (dt == DType::Float64) {
        result.data_ptr<double>()[0] = total;
        return result;
    }
    result.data_ptr<float>()[0] = static_cast<float>(total);
    return result;
}

TENSORPLAY_LIBRARY_IMPL(CPU, Blas) {
    m.impl("addbmm", addbmm_cpu);
    m.impl("addmv", addmv_cpu);
    m.impl("addr", addr_cpu);
    m.impl("vdot", vdot_cpu);
}

}  // namespace cpu

namespace {

#if TP_BLOCK_GEMM_UKERNEL
// A product shape the processor has a generated kernel for.  A block product is
// called once per block of a much larger computation, with the same few shapes
// over and over, so the kernel for a shape is generated once and kept: what a
// library call spends per call on deciding how to compute a shape is spent here
// once per shape instead.
struct BlockGemmKey {
    int64_t M, N, K, ld_a, ld_b, ld_c;
    bool add_C;
    bool operator==(const BlockGemmKey& o) const {
        return M == o.M && N == o.N && K == o.K && ld_a == o.ld_a &&
               ld_b == o.ld_b && ld_c == o.ld_c && add_C == o.add_C;
    }
};

struct BlockGemmKeyHash {
    size_t operator()(const BlockGemmKey& k) const {
        size_t h = std::hash<int64_t>()(k.M);
        for (int64_t v : {k.N, k.K, k.ld_a, k.ld_b, k.ld_c}) {
            h = std::hash<int64_t>()(v) ^ (h << 1);
        }
        return std::hash<bool>()(k.add_C) ^ (h << 1);
    }
};

struct BlockGemmKernel {
    dnnl::ukernel::brgemm brg;
    std::vector<uint8_t> scratchpad;
    std::vector<std::pair<dnnl::memory::dim, dnnl::memory::dim>> offsets{{0, 0}};
};

// Whether this processor has the generated float kernel, which it does from
// the 256-bit vector tier up; below that the library product is the one used.
bool block_gemm_ukernel_available() {
    static const bool available = [] {
        try {
            return dnnl::get_effective_cpu_isa() >= dnnl::cpu_isa::avx2 &&
                   dnnl::ukernel::brgemm::get_B_pack_type(
                       dnnl::memory::data_type::f32,
                       dnnl::memory::data_type::f32) ==
                       dnnl::ukernel::pack_type::no_trans;
        } catch (...) {
            return false;
        }
    }();
    return available;
}

// The kernel for this shape on the calling thread, or null when none can be
// made.  The cache is the thread's own: a kernel is executed on the thread
// that made it ready, and readying one is a per-thread state of the unit.
BlockGemmKernel* block_gemm_kernel(const BlockGemmKey& key) {
    thread_local std::unordered_map<BlockGemmKey,
                                    std::unique_ptr<BlockGemmKernel>,
                                    BlockGemmKeyHash>
        cache;
    thread_local BlockGemmKernel* current = nullptr;
    auto it = cache.find(key);
    if (it == cache.end()) {
        std::unique_ptr<BlockGemmKernel> made;
        try {
            auto kernel = std::make_unique<BlockGemmKernel>();
            kernel->brg = dnnl::ukernel::brgemm(
                key.M, key.N, key.K, /*batch_size=*/1, key.ld_a, key.ld_b,
                key.ld_c, dnnl::memory::data_type::f32,
                dnnl::memory::data_type::f32, dnnl::memory::data_type::f32,
                /*allow_empty=*/true);
            if (kernel->brg) {
                kernel->brg.set_add_C(key.add_C);
                kernel->brg.finalize();
                kernel->scratchpad.resize(kernel->brg.get_scratchpad_size());
                kernel->brg.generate();
                made = std::move(kernel);
            }
        } catch (...) {
            made.reset();
        }
        it = cache.emplace(key, std::move(made)).first;
    }
    BlockGemmKernel* kernel = it->second.get();
    if (kernel != nullptr && kernel != current) {
        kernel->brg.set_hw_context();
        current = kernel;
    }
    return kernel;
}
#endif

// C (M x N, row stride ld_c) = A @ B (+ C), every operand in float.
void block_gemm_float(int64_t M, int64_t N, int64_t K, int64_t ld_a,
                      int64_t ld_b, int64_t ld_c, bool add_C, const float* A,
                      const float* B, float* C) {
    if (M <= 0 || N <= 0) return;
#if TP_BLOCK_GEMM_UKERNEL
    if (K > 0 && block_gemm_ukernel_available()) {
        if (BlockGemmKernel* kernel =
                block_gemm_kernel({M, N, K, ld_a, ld_b, ld_c, add_C})) {
            kernel->brg.execute(A, B, kernel->offsets, C,
                                kernel->scratchpad.data());
            return;
        }
    }
#endif
#if defined(USE_MKL) || defined(USE_BLAS)
    if (K > 0) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(M), static_cast<int>(N),
                    static_cast<int>(K), 1.0f, A, static_cast<int>(ld_a), B,
                    static_cast<int>(ld_b), add_C ? 1.0f : 0.0f, C,
                    static_cast<int>(ld_c));
        return;
    }
#endif
    for (int64_t i = 0; i < M; ++i) {
        float* c_row = C + i * ld_c;
        if (!add_C) {
            for (int64_t j = 0; j < N; ++j) c_row[j] = 0.0f;
        }
        for (int64_t p = 0; p < K; ++p) {
            const float a = A[i * ld_a + p];
            const float* b_row = B + p * ld_b;
            for (int64_t j = 0; j < N; ++j) c_row[j] += a * b_row[j];
        }
    }
}

// The same product for sixteen-bit operands: both blocks are widened to float
// first, which is the precision the product is defined in, and then multiplied
// as float blocks.
template <typename T>
void block_gemm_widened(int64_t M, int64_t N, int64_t K, int64_t ld_a,
                        int64_t ld_b, int64_t ld_c, bool add_C, const T* A,
                        const T* B, float* C) {
    thread_local std::vector<float> a_wide;
    thread_local std::vector<float> b_wide;
    a_wide.resize(static_cast<size_t>(std::max<int64_t>(M * K, 0)));
    b_wide.resize(static_cast<size_t>(std::max<int64_t>(K * N, 0)));
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t p = 0; p < K; ++p) {
            a_wide[i * K + p] = static_cast<float>(A[i * ld_a + p]);
        }
    }
    for (int64_t p = 0; p < K; ++p) {
        for (int64_t j = 0; j < N; ++j) {
            b_wide[p * N + j] = static_cast<float>(B[p * ld_b + j]);
        }
    }
    block_gemm_float(M, N, K, K, N, ld_c, add_C, a_wide.data(), b_wide.data(), C);
}

}  // namespace

void brgemm(int64_t M, int64_t N, int64_t K, int64_t ld_a, int64_t ld_b,
            int64_t ld_c, bool add_C, const float* A, const float* B, float* C,
            bool is_vnni) {
    TP_CHECK(!is_vnni, "a float block product has no pair-interleaved form");
    block_gemm_float(M, N, K, ld_a, ld_b, ld_c, add_C, A, B, C);
}

void brgemm(int64_t M, int64_t N, int64_t K, int64_t ld_a, int64_t ld_b,
            int64_t ld_c, bool add_C, const BFloat16* A, const BFloat16* B,
            float* C, bool is_vnni) {
    TP_CHECK(!is_vnni,
             "a pair-interleaved bfloat16 block needs a matrix unit this "
             "library does not drive");
    block_gemm_widened(M, N, K, ld_a, ld_b, ld_c, add_C, A, B, C);
}

void brgemm(int64_t M, int64_t N, int64_t K, int64_t ld_a, int64_t ld_b,
            int64_t ld_c, bool add_C, const Half* A, const Half* B, float* C,
            bool is_vnni) {
    TP_CHECK(!is_vnni,
             "a pair-interleaved half block needs a matrix unit this library "
             "does not drive");
    block_gemm_widened(M, N, K, ld_a, ld_b, ld_c, add_C, A, B, C);
}

bool could_pack(DType dtype) {
    (void)dtype;
    return false;
}

void brgemm_release(bool is_vnni) { (void)is_vnni; }

}  // namespace tensorplay
