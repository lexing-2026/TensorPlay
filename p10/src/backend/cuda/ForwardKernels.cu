#include "ForwardKernels.h"
#include "CUDARuntime.h"
#include "Exception.h"

#include <cuda_runtime.h>

namespace tensorplay {
namespace cuda {
namespace {

enum class UKind { Neg, Exp, Log, Sin, Cos, Sqrt, Tanh, Sigmoid, Relu };
enum class BKind { Add, Sub, Mul, Div, Pow };

template <typename T>
__global__ void forward_unary_kernel(
    int64_t n, int kind,
    const T* a, const T* da, T* r, T* dr) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const T x = a[i];
    const T g = da[i];
    T v, d;
    switch (static_cast<UKind>(kind)) {
        case UKind::Neg:     v = -x; d = -g; break;
        case UKind::Exp:     v = ::exp(x); d = v * g; break;
        case UKind::Log:     v = ::log(x); d = g / x; break;
        case UKind::Sin:     v = ::sin(x); d = ::cos(x) * g; break;
        case UKind::Cos:     v = ::cos(x); d = -::sin(x) * g; break;
        case UKind::Sqrt:    v = ::sqrt(x); d = g / (T(2) * v); break;
        case UKind::Tanh:    v = ::tanh(x); d = (T(1) - v * v) * g; break;
        case UKind::Sigmoid: v = T(1) / (T(1) + ::exp(-x)); d = v * (T(1) - v) * g; break;
        case UKind::Relu:    v = x > T(0) ? x : T(0); d = x > T(0) ? g : T(0); break;
        default:             v = x; d = g; break;
    }
    r[i] = v;
    dr[i] = d;
}

template <typename T>
__global__ void forward_binary_kernel(
    int64_t n, int kind,
    const T* a, const T* da, const T* b, const T* db, T* r, T* dr) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const T x = a[i], y = b[i];
    const T gx = da[i], gy = db[i];
    T v, d;
    switch (static_cast<BKind>(kind)) {
        case BKind::Add: v = x + y; d = gx + gy; break;
        case BKind::Sub: v = x - y; d = gx - gy; break;
        case BKind::Mul: v = x * y; d = gx * y + x * gy; break;
        case BKind::Div: v = x / y; d = (gx - v * gy) / y; break;
        case BKind::Pow:
            v = ::pow(x, y);
            d = v * (gy * ::log(x) + y * gx / x);
            break;
        default: v = x; d = gx; break;
    }
    r[i] = v;
    dr[i] = d;
}

template <typename T>
__global__ void forward_mm_kernel(
    int64_t M, int64_t K, int64_t N,
    const T* A, const T* DA, const T* B, const T* DB, T* R, T* DR) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= M * N) return;
    const int64_t i = linear / N;
    const int64_t j = linear % N;
    T acc_r = T(0), acc_d = T(0);
    for (int64_t k = 0; k < K; ++k) {
        const T av = A[i * K + k], bv = B[k * N + j];
        acc_r += av * bv;
        acc_d += DA[i * K + k] * bv + av * DB[k * N + j];
    }
    R[linear] = acc_r;
    DR[linear] = acc_d;
}

DType require_compute_dtype(const Tensor& t, const char* op) {
    if (t.dtype() == DType::Float32 || t.dtype() == DType::Float64) {
        return t.dtype();
    }
    TP_THROW(TypeError,
             std::string(op) + ": forward-mode AD kernels require Float32/Float64 tensors");
}

Tensor contiguous_checked(const Tensor& t, const char* op) {
    require_compute_dtype(t, op);
    return t.contiguous();
}

std::tuple<Tensor, Tensor> run_unary(
    const Tensor& primal, const Tensor& tangent,
    int kind, const char* op) {
    Tensor a = contiguous_checked(primal, op);
    Tensor da = tangent.contiguous();
    if (da.shape() != a.shape()) {
        TP_THROW(RuntimeError,
                 "forward AD: tangent must match the primal tensor's shape");
    }
    Tensor r = Tensor::empty(a.shape(), a.dtype(), a.device());
    Tensor dr = Tensor::empty(a.shape(), a.dtype(), a.device());
    const int64_t n = a.numel();
    if (n == 0) return {r, dr};
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    constexpr int threads = 128;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    if (a.dtype() == DType::Float64) {
        forward_unary_kernel<double><<<blocks, threads, 0, stream>>>(
            n, kind, a.data_ptr<double>(), da.data_ptr<double>(),
            r.data_ptr<double>(), dr.data_ptr<double>());
    } else {
        forward_unary_kernel<float><<<blocks, threads, 0, stream>>>(
            n, kind, a.data_ptr<float>(), da.data_ptr<float>(),
            r.data_ptr<float>(), dr.data_ptr<float>());
    }
    checkCuda(cudaGetLastError(), op);
    return {r, dr};
}

std::tuple<Tensor, Tensor> run_binary(
    const Tensor& pa, const Tensor& ta,
    const Tensor& pb, const Tensor& tb,
    int kind, const char* op) {
    Tensor a = contiguous_checked(pa, op);
    Tensor b = contiguous_checked(pb, op);
    if (a.dtype() != b.dtype()) {
        TP_THROW(TypeError, "forward AD: operands must share one dtype");
    }
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
    if (n == 0) return {r, dr};
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    constexpr int threads = 128;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    if (a.dtype() == DType::Float64) {
        forward_binary_kernel<double><<<blocks, threads, 0, stream>>>(
            n, kind, a.data_ptr<double>(), da.data_ptr<double>(),
            b.data_ptr<double>(), db.data_ptr<double>(),
            r.data_ptr<double>(), dr.data_ptr<double>());
    } else {
        forward_binary_kernel<float><<<blocks, threads, 0, stream>>>(
            n, kind, a.data_ptr<float>(), da.data_ptr<float>(),
            b.data_ptr<float>(), db.data_ptr<float>(),
            r.data_ptr<float>(), dr.data_ptr<float>());
    }
    checkCuda(cudaGetLastError(), op);
    return {r, dr};
}

#define TP_FORWARD_UNARY_CUDA(name, ukind)                                  \
    std::tuple<Tensor, Tensor> name(const Tensor& a, const Tensor& da) {    \
        return run_unary(a, da, static_cast<int>(UKind::ukind), #name);     \
    }

#define TP_FORWARD_BINARY_CUDA(name, bkind)                                 \
    std::tuple<Tensor, Tensor> name(const Tensor& a, const Tensor& da,      \
                                    const Tensor& b, const Tensor& db) {    \
        return run_binary(a, da, b, db, static_cast<int>(BKind::bkind), #name); \
    }

} // namespace

TP_FORWARD_UNARY_CUDA(forward_neg_cuda, Neg)
TP_FORWARD_UNARY_CUDA(forward_exp_cuda, Exp)
TP_FORWARD_UNARY_CUDA(forward_log_cuda, Log)
TP_FORWARD_UNARY_CUDA(forward_sin_cuda, Sin)
TP_FORWARD_UNARY_CUDA(forward_cos_cuda, Cos)
TP_FORWARD_UNARY_CUDA(forward_sqrt_cuda, Sqrt)
TP_FORWARD_UNARY_CUDA(forward_tanh_cuda, Tanh)
TP_FORWARD_UNARY_CUDA(forward_sigmoid_cuda, Sigmoid)
TP_FORWARD_UNARY_CUDA(forward_relu_cuda, Relu)

TP_FORWARD_BINARY_CUDA(forward_add_cuda, Add)
TP_FORWARD_BINARY_CUDA(forward_sub_cuda, Sub)
TP_FORWARD_BINARY_CUDA(forward_mul_cuda, Mul)
TP_FORWARD_BINARY_CUDA(forward_div_cuda, Div)
TP_FORWARD_BINARY_CUDA(forward_pow_cuda, Pow)

std::tuple<Tensor, Tensor> forward_mm_cuda(const Tensor& pa, const Tensor& ta,
                                           const Tensor& pb, const Tensor& tb) {
    Tensor a = contiguous_checked(pa, "forward_mm");
    Tensor b = contiguous_checked(pb, "forward_mm");
    if (a.dtype() != b.dtype()) {
        TP_THROW(TypeError, "forward_mm(): operands must share one dtype");
    }
    Tensor da = ta.contiguous();
    Tensor db = tb.contiguous();
    if (a.dim() != 2 || b.dim() != 2 || a.size(1) != b.size(0)) {
        TP_THROW(RuntimeError,
                 "forward_mm(): expects 2-D operands with a shared inner dimension");
    }
    if (da.shape() != a.shape() || db.shape() != b.shape()) {
        TP_THROW(RuntimeError, "forward_mm(): tangents must match their primals' shapes");
    }
    const int64_t M = a.size(0), K = a.size(1), N = b.size(1);
    Tensor r = Tensor::zeros({M, N}, a.dtype(), a.device());
    Tensor dr = Tensor::zeros({M, N}, a.dtype(), a.device());
    const int64_t total = M * N;
    if (total == 0) return {r, dr};
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    constexpr int threads = 128;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    if (a.dtype() == DType::Float64) {
        forward_mm_kernel<double><<<blocks, threads, 0, stream>>>(
            M, K, N, a.data_ptr<double>(), da.data_ptr<double>(),
            b.data_ptr<double>(), db.data_ptr<double>(),
            r.data_ptr<double>(), dr.data_ptr<double>());
    } else {
        forward_mm_kernel<float><<<blocks, threads, 0, stream>>>(
            M, K, N, a.data_ptr<float>(), da.data_ptr<float>(),
            b.data_ptr<float>(), db.data_ptr<float>(),
            r.data_ptr<float>(), dr.data_ptr<float>());
    }
    checkCuda(cudaGetLastError(), "forward_mm");
    return {r, dr};
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ForwardKernels) {
    m.impl("forward_neg", forward_neg_cuda);
    m.impl("forward_exp", forward_exp_cuda);
    m.impl("forward_log", forward_log_cuda);
    m.impl("forward_sin", forward_sin_cuda);
    m.impl("forward_cos", forward_cos_cuda);
    m.impl("forward_sqrt", forward_sqrt_cuda);
    m.impl("forward_tanh", forward_tanh_cuda);
    m.impl("forward_sigmoid", forward_sigmoid_cuda);
    m.impl("forward_relu", forward_relu_cuda);
    m.impl("forward_add", forward_add_cuda);
    m.impl("forward_sub", forward_sub_cuda);
    m.impl("forward_mul", forward_mul_cuda);
    m.impl("forward_div", forward_div_cuda);
    m.impl("forward_pow", forward_pow_cuda);
    m.impl("forward_mm", forward_mm_cuda);
}

} // namespace cuda
} // namespace tensorplay
