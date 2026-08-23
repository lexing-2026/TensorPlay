#include "ForwardKernels.h"
#include "Exception.h"

#include <cmath>

namespace tensorplay {
namespace cpu {
namespace {

enum class UKind { Neg, Exp, Log, Sin, Cos, Sqrt, Tanh, Sigmoid, Relu };
enum class BKind { Add, Sub, Mul, Div, Pow };

template <typename T>
struct TypeTag { using type = T; };

// Forward kernels support the two compute dtypes natively; narrower floats
// must be promoted by the caller (the DualTensor layer does this).
DType require_compute_dtype(const Tensor& t, const char* op) {
    if (t.dtype() == DType::Float32 || t.dtype() == DType::Float64) {
        return t.dtype();
    }
    TP_THROW(TypeError,
             std::string(op) + ": forward-mode AD kernels require Float32/Float64 tensors");
}

template <typename T>
void unary_loop(UKind kind, const T* a, const T* da, T* r, T* dr, int64_t n) {
    switch (kind) {
        case UKind::Neg:
            for (int64_t i = 0; i < n; ++i) { r[i] = -a[i]; dr[i] = -da[i]; }
            return;
        case UKind::Exp:
            for (int64_t i = 0; i < n; ++i) {
                r[i] = std::exp(a[i]); dr[i] = r[i] * da[i];
            }
            return;
        case UKind::Log:
            for (int64_t i = 0; i < n; ++i) {
                r[i] = std::log(a[i]); dr[i] = da[i] / a[i];
            }
            return;
        case UKind::Sin:
            for (int64_t i = 0; i < n; ++i) {
                r[i] = std::sin(a[i]); dr[i] = std::cos(a[i]) * da[i];
            }
            return;
        case UKind::Cos:
            for (int64_t i = 0; i < n; ++i) {
                r[i] = std::cos(a[i]); dr[i] = -std::sin(a[i]) * da[i];
            }
            return;
        case UKind::Sqrt:
            for (int64_t i = 0; i < n; ++i) {
                r[i] = std::sqrt(a[i]); dr[i] = da[i] / (T(2) * r[i]);
            }
            return;
        case UKind::Tanh:
            for (int64_t i = 0; i < n; ++i) {
                r[i] = std::tanh(a[i]);
                dr[i] = (T(1) - r[i] * r[i]) * da[i];
            }
            return;
        case UKind::Sigmoid:
            for (int64_t i = 0; i < n; ++i) {
                r[i] = T(1) / (T(1) + std::exp(-a[i]));
                dr[i] = r[i] * (T(1) - r[i]) * da[i];
            }
            return;
        case UKind::Relu:
            for (int64_t i = 0; i < n; ++i) {
                r[i] = a[i] > T(0) ? a[i] : T(0);
                dr[i] = a[i] > T(0) ? da[i] : T(0);
            }
            return;
    }
}

template <typename T>
void binary_loop(BKind kind, const T* a, const T* da, const T* b, const T* db,
                 T* r, T* dr, int64_t n) {
    switch (kind) {
        case BKind::Add:
            for (int64_t i = 0; i < n; ++i) {
                r[i] = a[i] + b[i]; dr[i] = da[i] + db[i];
            }
            return;
        case BKind::Sub:
            for (int64_t i = 0; i < n; ++i) {
                r[i] = a[i] - b[i]; dr[i] = da[i] - db[i];
            }
            return;
        case BKind::Mul:
            for (int64_t i = 0; i < n; ++i) {
                r[i] = a[i] * b[i]; dr[i] = da[i] * b[i] + a[i] * db[i];
            }
            return;
        case BKind::Div:
            for (int64_t i = 0; i < n; ++i) {
                r[i] = a[i] / b[i]; dr[i] = (da[i] - r[i] * db[i]) / b[i];
            }
            return;
        case BKind::Pow:
            for (int64_t i = 0; i < n; ++i) {
                if (!(a[i] > T(0))) {
                    TP_THROW(NotImplementedError,
                             "forward_pow: requires a strictly positive base");
                }
                r[i] = std::pow(a[i], b[i]);
                dr[i] = r[i] * (db[i] * std::log(a[i]) + b[i] * da[i] / a[i]);
            }
            return;
    }
}

template <typename T>
std::tuple<Tensor, Tensor> run_unary(const Tensor& primal, const Tensor& tangent,
                                     UKind kind) {
    Tensor a = primal.contiguous();
    Tensor da = tangent.contiguous();
    if (da.shape() != a.shape()) {
        TP_THROW(RuntimeError,
                 "forward AD: tangent must match the primal tensor's shape");
    }
    Tensor r = Tensor::empty(a.shape(), a.dtype(), a.device());
    Tensor dr = Tensor::empty(a.shape(), a.dtype(), a.device());
    const int64_t n = a.numel();
    unary_loop<T>(kind, a.data_ptr<T>(), da.data_ptr<T>(),
                  r.data_ptr<T>(), dr.data_ptr<T>(), n);
    return {r, dr};
}

template <typename T>
std::tuple<Tensor, Tensor> run_binary(const Tensor& pa, const Tensor& ta,
                                      const Tensor& pb, const Tensor& tb,
                                      BKind kind) {
    Tensor a = pa.contiguous();
    Tensor b = pb.contiguous();
    Tensor da = ta.contiguous();
    Tensor db = tb.contiguous();
    if (a.shape() != b.shape() || da.shape() != a.shape() ||
        db.shape() != b.shape()) {
        TP_THROW(RuntimeError,
                 "forward AD: binary ops require matching operand/tangent shapes");
    }
    Tensor r = Tensor::empty(a.shape(), a.dtype(), a.device());
    Tensor dr = Tensor::empty(a.shape(), a.dtype(), a.device());
    const int64_t n = a.numel();
    binary_loop<T>(kind, a.data_ptr<T>(), da.data_ptr<T>(),
                   b.data_ptr<T>(), db.data_ptr<T>(),
                   r.data_ptr<T>(), dr.data_ptr<T>(), n);
    return {r, dr};
}

#define TP_FORWARD_UNARY(name, kind)                                        \
    std::tuple<Tensor, Tensor> name(const Tensor& a, const Tensor& da) {    \
        const DType dtype = require_compute_dtype(a, #name);                \
        if (dtype == DType::Float64) return run_unary<double>(a, da, kind); \
        return run_unary<float>(a, da, kind);                               \
    }

#define TP_FORWARD_BINARY(name, kind)                                          \
    std::tuple<Tensor, Tensor> name(const Tensor& a, const Tensor& da,         \
                                    const Tensor& b, const Tensor& db) {       \
        const DType dtype = require_compute_dtype(a, #name);                   \
        if (dtype != require_compute_dtype(b, #name)) {                        \
            TP_THROW(TypeError, #name ": operands must share one dtype");      \
        }                                                                      \
        if (dtype == DType::Float64) return run_binary<double>(a, da, b, db, kind); \
        return run_binary<float>(a, da, b, db, kind);                          \
    }

} // namespace

TP_FORWARD_UNARY(forward_neg_cpu, UKind::Neg)
TP_FORWARD_UNARY(forward_exp_cpu, UKind::Exp)
TP_FORWARD_UNARY(forward_log_cpu, UKind::Log)
TP_FORWARD_UNARY(forward_sin_cpu, UKind::Sin)
TP_FORWARD_UNARY(forward_cos_cpu, UKind::Cos)
TP_FORWARD_UNARY(forward_sqrt_cpu, UKind::Sqrt)
TP_FORWARD_UNARY(forward_tanh_cpu, UKind::Tanh)
TP_FORWARD_UNARY(forward_sigmoid_cpu, UKind::Sigmoid)
TP_FORWARD_UNARY(forward_relu_cpu, UKind::Relu)

TP_FORWARD_BINARY(forward_add_cpu, BKind::Add)
TP_FORWARD_BINARY(forward_sub_cpu, BKind::Sub)
TP_FORWARD_BINARY(forward_mul_cpu, BKind::Mul)
TP_FORWARD_BINARY(forward_div_cpu, BKind::Div)
TP_FORWARD_BINARY(forward_pow_cpu, BKind::Pow)

std::tuple<Tensor, Tensor> forward_mm_cpu(const Tensor& pa, const Tensor& ta,
                                          const Tensor& pb, const Tensor& tb) {
    require_compute_dtype(pa, "forward_mm");
    require_compute_dtype(pb, "forward_mm");
    Tensor a = pa.contiguous();
    Tensor b = pb.contiguous();
    Tensor da = ta.contiguous();
    Tensor db = tb.contiguous();
    if (a.dim() != 2 || b.dim() != 2 || a.size(1) != b.size(0)) {
        TP_THROW(RuntimeError, "forward_mm(): expects 2-D operands with a shared inner dimension");
    }
    if (da.shape() != a.shape() || db.shape() != b.shape()) {
        TP_THROW(RuntimeError, "forward_mm(): tangents must match their primals' shapes");
    }
    const int64_t M = a.size(0), K = a.size(1), N = b.size(1);
    Tensor r = Tensor::zeros({M, N}, a.dtype(), a.device());
    Tensor dr = Tensor::zeros({M, N}, a.dtype(), a.device());

    auto mm_body = [&](auto tag) {
        using T = typename decltype(tag)::type;
        const T* A = a.data_ptr<T>();
        const T* B = b.data_ptr<T>();
        const T* DA = da.data_ptr<T>();
        const T* DB = db.data_ptr<T>();
        T* R = r.data_ptr<T>();
        T* DR = dr.data_ptr<T>();
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                T acc_r = T(0), acc_dr = T(0);
                for (int64_t k = 0; k < K; ++k) {
                    const T av = A[i * K + k], bv = B[k * N + j];
                    acc_r += av * bv;
                    acc_dr += DA[i * K + k] * bv + av * DB[k * N + j];
                }
                R[i * N + j] = acc_r;
                DR[i * N + j] = acc_dr;
            }
        }
    };
    if (pa.dtype() == DType::Float64 || pb.dtype() == DType::Float64) {
        mm_body(TypeTag<double>{});
    } else {
        mm_body(TypeTag<float>{});
    }
    return {r, dr};
}

TENSORPLAY_LIBRARY_IMPL(CPU, ForwardKernels) {
    m.impl("forward_neg", forward_neg_cpu);
    m.impl("forward_exp", forward_exp_cpu);
    m.impl("forward_log", forward_log_cpu);
    m.impl("forward_sin", forward_sin_cpu);
    m.impl("forward_cos", forward_cos_cpu);
    m.impl("forward_sqrt", forward_sqrt_cpu);
    m.impl("forward_tanh", forward_tanh_cpu);
    m.impl("forward_sigmoid", forward_sigmoid_cpu);
    m.impl("forward_relu", forward_relu_cpu);
    m.impl("forward_add", forward_add_cpu);
    m.impl("forward_sub", forward_sub_cpu);
    m.impl("forward_mul", forward_mul_cpu);
    m.impl("forward_div", forward_div_cpu);
    m.impl("forward_pow", forward_pow_cpu);
    m.impl("forward_mm", forward_mm_cpu);
}

} // namespace cpu
} // namespace tensorplay
