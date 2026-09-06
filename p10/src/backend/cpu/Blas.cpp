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

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <mutex>

#ifdef USE_MKL
#include <mkl.h>
#elif defined(USE_BLAS)
#include <cblas.h>
#endif

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

namespace {

void require_float(const Tensor& t, const char* who) {
    if (!isFloatingType(t.dtype()))
        TP_THROW(TypeError, who, ": only floating-point tensors are supported");
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
// epilogues of the other low-precision GEMM paths.
template <typename T>
void addbmm_epilogue(Tensor& out, const Tensor& self_acc, const Tensor& work,
                     double beta, double alpha) {
    const T* sp = self_acc.data_ptr<T>();
    const float* wp = work.data_ptr<float>();
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
                 Scalar beta, Scalar alpha) {
    require_float(mat, "addmv");
    require_float(vec, "addmv");
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
        // copy is skipped and the output buffer stays uninitialized.
        Tensor result = beta_v != 0.0
            ? detail::contiguous_clone(self.expand({m}))
            : Tensor::empty({m}, dt, mat.device());
        Tensor xc = vec.is_contiguous() ? vec : detail::contiguous_clone(vec);
        Tensor a_input = mat;
        int64_t lda = k;
        bool trans = false;
        if (mat.is_contiguous()) {
            lda = k;
        } else if (mat.stride(0) == 1 && mat.stride(1) == m) {
            trans = true;
            lda = m;
        } else {
            a_input = detail::contiguous_clone(mat);
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
                  Scalar beta, Scalar alpha) {
    require_float(batch1, "addbmm");
    require_float(batch2, "addbmm");
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
    // a float workspace, then apply beta/alpha in one epilogue pass.
    Tensor work = Tensor::zeros({n, m}, DType::Float32, self.device());
    Tensor b1 = batch1.contiguous();
    Tensor b2 = batch2.contiguous();
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
        addbmm_epilogue<float>(out, self_acc, work, beta_v, alpha_v);
    } else if (dt == DType::Float64) {
        addbmm_epilogue<double>(out, self_acc, work, beta_v, alpha_v);
    } else if (dt == DType::Float16) {
        addbmm_epilogue<Half>(out, self_acc, work, beta_v, alpha_v);
    } else if (dt == DType::BFloat16) {
        addbmm_epilogue<BFloat16>(out, self_acc, work, beta_v, alpha_v);
    } else {
        TP_THROW(TypeError, "addbmm: unsupported dtype");
    }
    return out;
}

// ---------------------------------------------------------------------------
// addr: beta * self + alpha * vec1 (outer) vec2
// ---------------------------------------------------------------------------

Tensor addr_cpu(const Tensor& self, const Tensor& vec1, const Tensor& vec2,
                Scalar beta, Scalar alpha) {
    require_float(vec1, "addr");
    require_float(vec2, "addr");
    const int64_t m = vec1.numel(), k = vec2.numel();
    const DType dt = promoteTypes(promoteTypes(vec1.dtype(), vec2.dtype()), self.dtype());
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
        static std::mutex combine_mutex;
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
            std::lock_guard<std::mutex> lock(combine_mutex);
            total += part;
        });
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
}  // namespace tensorplay
