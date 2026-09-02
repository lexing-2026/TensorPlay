// nn loss operators - CUDA kernels for the reduction-parameterized family
// (mse / nll / nll2d / smooth_l1 / huber / bce and their backwards).
//
// Conventions match the sibling loss files: elementwise (or
// one-thread-per-row) kernels write per-element losses into a Float64
// buffer, an atomicAdd grid reduction sums it, and the scalar finalization
// happens on the host.  Gradients are one thread per element (or per row)
// writing into a zeroed result; mean reduction rescales by the contributing
// count.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"

#include <cuda_runtime.h>

#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

constexpr int kThreads = 256;

inline dim3 loss_grid(int64_t work) {
    return dim3(static_cast<unsigned>((work + kThreads - 1) / kThreads));
}

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

__global__ void atomic_sum_kernel(int64_t n, const double* in, double* total) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) atomicAdd(total, in[i]);
}

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

double host_sum_f64(const Tensor& elems, int64_t n) {
    Tensor total = Tensor::zeros({1}, DType::Float64, elems.device());
    if (n > 0) {
        atomic_sum_kernel<<<loss_grid(n), kThreads, 0,
                            getCurrentCUDAStream().stream()>>>(
            n, elems.data_ptr<double>(), total.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    double h = 0;
    CUDA_CHECK(cudaMemcpy(&h, total.data_ptr<double>(), sizeof(double),
                          cudaMemcpyDeviceToHost));
    return h;
}

std::pair<Tensor, Tensor> pair_f64_dev(const Tensor& a, const Tensor& b) {
    const Tensor ac = a.is_contiguous() ? a : a.contiguous();
    const Tensor bc = b.is_contiguous() ? b : b.contiguous();
    Tensor ae = ac;
    Tensor be = bc;
    if (ac.shape() != bc.shape()) {
        const std::vector<int64_t> bs = [&] {
            std::vector<int64_t> out(static_cast<std::vector<int64_t>>(ac.shape()).size());
            const auto& as = static_cast<std::vector<int64_t>>(ac.shape());
            const auto& bsv = static_cast<std::vector<int64_t>>(bc.shape());
            const size_t n = std::max(as.size(), bsv.size());
            std::vector<int64_t> ra(n - as.size(), 1), rb(n - bsv.size(), 1);
            ra.insert(ra.end(), as.begin(), as.end());
            rb.insert(rb.end(), bsv.begin(), bsv.end());
            for (size_t i = 0; i < n; ++i) out[i] = std::max(ra[i], rb[i]);
            return out;
        }();
        ae = ac.expand(bs);
        be = bc.expand(bs);
    }
    return {ae.to(DType::Float64), be.to(DType::Float64)};
}

// ---------------------------------------------------------------------------
// elementwise loss / gradient kernels
// ---------------------------------------------------------------------------

__global__ void mse_elem_kernel(int64_t n, const double* x, const double* t,
                                double* o) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st) {
        const double d = x[i] - t[i];
        o[i] = d * d;
    }
}

__global__ void mse_grad_kernel(int64_t n, const double* x, const double* t,
                                const double* g, double norm, double* o) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st) {
        const double d = x[i] - t[i];
        o[i] = norm * 2.0 * d * g[i];
    }
}

__global__ void smooth_l1_elem_kernel(int64_t n, const double* x, const double* t,
                                      double beta, double* o) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st) {
        const double d = ::fabs(x[i] - t[i]);
        // quadratic inside beta, linear outside (the two branches agree at
        // the boundary, so the closed comparison either side is fine)
        o[i] = d < beta ? 0.5 * d * d / beta : d - 0.5 * beta;
    }
}

__global__ void smooth_l1_grad_kernel(int64_t n, const double* x, const double* t,
                                      const double* g, double beta, double norm,
                                      double* o) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st) {
        const double d = x[i] - t[i];
        const double ad = ::fabs(d);
        const double local = ad <= beta ? d / beta
                                        : (d > 0.0 ? 1.0 : (d < 0.0 ? -1.0 : 0.0));
        o[i] = norm * local * g[i];
    }
}

__global__ void huber_elem_kernel(int64_t n, const double* x, const double* t,
                                  double delta, double* o) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st) {
        const double d = ::fabs(x[i] - t[i]);
        o[i] = d < delta ? 0.5 * d * d : delta * (d - 0.5 * delta);
    }
}

__global__ void huber_grad_kernel(int64_t n, const double* x, const double* t,
                                  const double* g, double delta, double norm,
                                  double* o) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st) {
        const double d = x[i] - t[i];
        const double ad = ::fabs(d);
        const double local = ad <= delta ? d : delta * (d > 0.0 ? 1.0 : -1.0);
        o[i] = norm * local * g[i];
    }
}

// ---------------------------------------------------------------------------
// binary_cross_entropy elementwise kernels
// ---------------------------------------------------------------------------

__global__ void bce_elem_kernel(int64_t n, const double* x, const double* t,
                                const double* w, bool has_w, double* o) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st) {
        // (t-1) log(1-x) - t log(x), floored at -100 like the elementwise cap
        const double lx = ::fmax(::log(x[i]), -100.0);
        const double l1x = ::fmax(::log(1.0 - x[i]), -100.0);
        double v = (t[i] - 1.0) * l1x - t[i] * lx;
        o[i] = has_w ? w[i] * v : v;
    }
}

__global__ void bce_grad_kernel(int64_t n, const double* x, const double* t,
                                const double* g, const double* w, bool has_w,
                                double norm, double* o) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st) {
        const double denom = ::fmax(x[i] * (1.0 - x[i]), 1e-12);
        double v = g[i] * (x[i] - t[i]) / denom;
        if (has_w) v *= w[i];
        o[i] = norm * v;
    }
}

// ---------------------------------------------------------------------------
// nll forward / backward (row form and 2-D spatial form)
// ---------------------------------------------------------------------------

// One thread per batch row gathers the target class score; optional
// per-class weights multiply.  Writes per-row loss and per-row weight so the
// host reduction can honour both reduction modes and ignore_index.
__global__ void nll_row_kernel(int64_t n, int64_t C, const double* x,
                               const int64_t* tgt, const double* w, bool has_w,
                               int64_t ignore, double* loss, double* wout) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st) {
        const int64_t t = tgt[i];
        if (t == ignore) {
            loss[i] = 0.0;
            wout[i] = 0.0;
        } else {
            const double wi = has_w ? w[t] : 1.0;
            loss[i] = -x[i * C + t] * wi;
            wout[i] = wi;
        }
    }
}

// One thread per spatial position; input row (n*C+t)*HW + pos.
__global__ void nll2d_row_kernel(int64_t rows, int64_t C, int64_t HW,
                                 const double* x, const int64_t* tgt,
                                 const double* w, bool has_w, int64_t ignore,
                                 double* loss, double* wout) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < rows; i += st) {
        const int64_t t = tgt[i];
        if (t == ignore) {
            loss[i] = 0.0;
            wout[i] = 0.0;
        } else {
            const int64_t n = i / HW;
            const int64_t pos = i % HW;
            const double wi = has_w ? w[t] : 1.0;
            loss[i] = -x[(n * C + t) * HW + pos] * wi;
            wout[i] = wi;
        }
    }
}

// reduction == 0: gradient scatters -w * g_i to the target slot.
__global__ void nll_grad_none_kernel(int64_t n, int64_t C, const double* g,
                                     const int64_t* tgt, const double* w,
                                     bool has_w, int64_t ignore, double* gi) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st) {
        const int64_t t = tgt[i];
        if (t == ignore || t < 0 || t >= C) continue;
        const double wi = has_w ? w[t] : 1.0;
        gi[i * C + t] = -wi * g[i];
    }
}

__global__ void nll2d_grad_none_kernel(int64_t rows, int64_t C, int64_t HW,
                                      const double* g, const int64_t* tgt,
                                      const double* w, bool has_w,
                                      int64_t ignore, double* gi) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < rows; i += st) {
        const int64_t t = tgt[i];
        if (t == ignore || t < 0 || t >= C) continue;
        const int64_t n = i / HW;
        const int64_t pos = i % HW;
        const double wi = has_w ? w[t] : 1.0;
        gi[(n * C + t) * HW + pos] = -wi * g[i];
    }
}

// Scalar-output modes: every valid row contributes -w * g / tw.
__global__ void nll_grad_scalar_kernel(int64_t n, int64_t C, const double g,
                                       const int64_t* tgt, const double* w,
                                       bool has_w, int64_t ignore, double tw,
                                       double* gi) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const double gg = tw > 0 ? g / tw : g;
    for (; i < n; i += st) {
        const int64_t t = tgt[i];
        if (t == ignore || t < 0 || t >= C) continue;
        const double wi = has_w ? w[t] : 1.0;
        gi[i * C + t] = -wi * gg;
    }
}

__global__ void nll2d_grad_scalar_kernel(int64_t rows, int64_t C, int64_t HW,
                                         const double g, const int64_t* tgt,
                                         const double* w, bool has_w,
                                         int64_t ignore, double tw, double* gi) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const double gg = tw > 0 ? g / tw : g;
    for (; i < rows; i += st) {
        const int64_t t = tgt[i];
        if (t == ignore || t < 0 || t >= C) continue;
        const int64_t n = i / HW;
        const int64_t pos = i % HW;
        const double wi = has_w ? w[t] : 1.0;
        gi[(n * C + t) * HW + pos] = -wi * gg;
    }
}

// Sum of a small device buffer (row weights) into a host scalar.

inline bool is_loss_float(DType t) {
    return t == DType::Float32 || t == DType::Float64 ||
           t == DType::Float16 || t == DType::BFloat16;
}

inline DType out_scalar_dtype(DType t) {
    return t == DType::Float64 ? DType::Float64 : DType::Float32;
}

Tensor f64_dev(const Tensor& t) {
    const Tensor c = t.is_contiguous() ? t : t.contiguous();
    return c.dtype() == DType::Float64 ? c : c.to(DType::Float64);
}

}  // anonymous namespace

// ===========================================================================
// mse_loss family
// ===========================================================================

Tensor mse_loss_cuda(const Tensor& input, const Tensor& target,
                     int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2) {
        TP_THROW(ValueError, "Invalid reduction mode");
    }
    auto pr = pair_f64_dev(input, target);
    Tensor elems = Tensor::empty(shape_of(pr.first), DType::Float64,
                                  input.device());
    const int64_t n = elems.numel();
    if (n) {
        mse_elem_kernel<<<loss_grid(n), kThreads, 0,
                          getCurrentCUDAStream().stream()>>>(
            n, pr.first.data_ptr<double>(), pr.second.data_ptr<double>(),
            elems.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    if (reduction == 0) {
        return elems.to(input.dtype() == DType::Float64 ? DType::Float64
                                                       : DType::Float32)
            .to(input.dtype());
    }
    const double total = host_sum_f64(elems, n);
    const double v = reduction == 1 && n ? total / n : total;
    return Tensor::full({}, Scalar(v), out_scalar_dtype(input.dtype()),
                         input.device())
        .to(input.dtype());
}

Tensor mse_loss_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& target, int64_t reduction) {
    auto pr = pair_f64_dev(input, target);
    Tensor g = f64_dev(grad_output);
    if (g.shape() != pr.first.shape() && reduction != 0) {
        g = g.expand(shape_of(pr.first));
    }
    Tensor out = Tensor::empty(shape_of(pr.first), DType::Float64,
                                input.device());
    const int64_t n = out.numel();
    if (n) {
        const double norm = reduction == 1 ? 1.0 / static_cast<double>(n) : 1.0;
        mse_grad_kernel<<<loss_grid(n), kThreads, 0,
                          getCurrentCUDAStream().stream()>>>(
            n, pr.first.data_ptr<double>(), pr.second.data_ptr<double>(),
            g.data_ptr<double>(), norm, out.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    return out.to(input.dtype());
}

// ===========================================================================
// smooth_l1_loss family
// ===========================================================================

Tensor smooth_l1_loss_cuda(const Tensor& input, const Tensor& target,
                           int64_t reduction, double beta) {
    if (beta < 0) {
        TP_THROW(ValueError,
                 "smooth_l1_loss does not support negative values for beta.");
    }
    auto pr = pair_f64_dev(input, target);
    Tensor elems = Tensor::empty(shape_of(pr.first), DType::Float64,
                                  input.device());
    const int64_t n = elems.numel();
    if (n) {
        smooth_l1_elem_kernel<<<loss_grid(n), kThreads, 0,
                                getCurrentCUDAStream().stream()>>>(
            n, pr.first.data_ptr<double>(), pr.second.data_ptr<double>(),
            beta, elems.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    if (reduction == 0) return elems.to(input.dtype());
    const double total = host_sum_f64(elems, n);
    const double v = reduction == 1 && n ? total / n : total;
    return Tensor::full({}, Scalar(v), out_scalar_dtype(input.dtype()),
                         input.device())
        .to(input.dtype());
}

Tensor smooth_l1_loss_backward_cuda(const Tensor& grad_output,
                                    const Tensor& input, const Tensor& target,
                                    int64_t reduction, double beta) {
    auto pr = pair_f64_dev(input, target);
    Tensor g = f64_dev(grad_output);
    if (g.shape() != pr.first.shape() && reduction != 0) {
        g = g.expand(shape_of(pr.first));
    }
    Tensor out = Tensor::empty(shape_of(pr.first), DType::Float64,
                               input.device());
    const int64_t n = out.numel();
    if (n) {
        const double norm = reduction == 1 ? 1.0 / static_cast<double>(n) : 1.0;
        smooth_l1_grad_kernel<<<loss_grid(n), kThreads, 0,
                                getCurrentCUDAStream().stream()>>>(
            n, pr.first.data_ptr<double>(), pr.second.data_ptr<double>(),
            g.data_ptr<double>(), beta, norm, out.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    return out.to(input.dtype());
}

// ===========================================================================
// huber_loss family
// ===========================================================================

Tensor huber_loss_cuda(const Tensor& input, const Tensor& target,
                       int64_t reduction, double delta) {
    if (delta <= 0) {
        TP_THROW(ValueError,
                 "huber_loss does not support non-positive values for delta.");
    }
    auto pr = pair_f64_dev(input, target);
    Tensor elems = Tensor::empty(shape_of(pr.first), DType::Float64,
                                  input.device());
    const int64_t n = elems.numel();
    if (n) {
        huber_elem_kernel<<<loss_grid(n), kThreads, 0,
                            getCurrentCUDAStream().stream()>>>(
            n, pr.first.data_ptr<double>(), pr.second.data_ptr<double>(),
            delta, elems.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    if (reduction == 0) return elems.to(input.dtype());
    const double total = host_sum_f64(elems, n);
    const double v = reduction == 1 && n ? total / n : total;
    return Tensor::full({}, Scalar(v), out_scalar_dtype(input.dtype()),
                         input.device())
        .to(input.dtype());
}

Tensor huber_loss_backward_cuda(const Tensor& grad_output,
                                const Tensor& input, const Tensor& target,
                                int64_t reduction, double delta) {
    auto pr = pair_f64_dev(input, target);
    Tensor g = f64_dev(grad_output);
    if (g.shape() != pr.first.shape() && reduction != 0) {
        g = g.expand(shape_of(pr.first));
    }
    Tensor out = Tensor::empty(shape_of(pr.first), DType::Float64,
                               input.device());
    const int64_t n = out.numel();
    if (n) {
        const double norm = reduction == 1 ? 1.0 / static_cast<double>(n) : 1.0;
        huber_grad_kernel<<<loss_grid(n), kThreads, 0,
                            getCurrentCUDAStream().stream()>>>(
            n, pr.first.data_ptr<double>(), pr.second.data_ptr<double>(),
            g.data_ptr<double>(), delta, norm, out.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    return out.to(input.dtype());
}

// ===========================================================================
// binary_cross_entropy family
// ===========================================================================

namespace {

// Validates the 0/1 domain on the host by summing out-of-range indicators.
void bce_check_01_cuda(const Tensor& t, const char* what) {
    if (t.numel() == 0) return;
    Tensor bad = (t.lt(Scalar(0.0)) + t.gt(Scalar(1.0))).sum();
    if (bad.item().to<double>() > 0) {
        TP_THROW(RuntimeError, std::string("all elements of ") + what +
                 " should be between 0 and 1");
    }
}

}  // namespace

Tensor binary_cross_entropy_cuda(const Tensor& input, const Tensor& target,
                                 const std::optional<Tensor>& weight_opt,
                                 int64_t reduction) {
    if (!is_loss_float(input.dtype())) {
        TP_THROW(NotImplementedError,
                 "binary_cross_entropy CUDA supports floating dtypes only");
    }
    bce_check_01_cuda(input, "input");
    bce_check_01_cuda(target, "target");
    auto pr = pair_f64_dev(input, target);
    const bool has_w = weight_opt.has_value() && weight_opt->defined();
    Tensor w = has_w ? f64_dev(weight_opt->expand(shape_of(pr.first)))
                     : pr.first;  // dummy pointer when absent
    Tensor elems = Tensor::empty(shape_of(pr.first), DType::Float64,
                                  input.device());
    const int64_t n = elems.numel();
    if (n) {
        bce_elem_kernel<<<loss_grid(n), kThreads, 0,
                          getCurrentCUDAStream().stream()>>>(
            n, pr.first.data_ptr<double>(), pr.second.data_ptr<double>(),
            w.data_ptr<double>(), has_w, elems.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    if (reduction == 0) return elems.to(input.dtype());
    const double total = host_sum_f64(elems, n);
    const double v = reduction == 1 && n ? total / n : total;
    return Tensor::full({}, Scalar(v), out_scalar_dtype(input.dtype()),
                         input.device())
        .to(input.dtype());
}

Tensor binary_cross_entropy_backward_cuda(const Tensor& grad_output,
                                          const Tensor& input,
                                          const Tensor& target,
                                          const std::optional<Tensor>& weight_opt,
                                          int64_t reduction) {
    auto pr = pair_f64_dev(input, target);
    Tensor g = f64_dev(grad_output);
    if (g.shape() != pr.first.shape() && reduction != 0) {
        g = g.expand(shape_of(pr.first));
    }
    const bool has_w = weight_opt.has_value() && weight_opt->defined();
    Tensor w = has_w ? f64_dev(weight_opt->expand(shape_of(pr.first)))
                     : pr.first;
    Tensor out = Tensor::empty(shape_of(pr.first), DType::Float64,
                               input.device());
    const int64_t n = out.numel();
    if (n) {
        const double norm = reduction == 1 ? 1.0 / static_cast<double>(n) : 1.0;
        bce_grad_kernel<<<loss_grid(n), kThreads, 0,
                          getCurrentCUDAStream().stream()>>>(
            n, pr.first.data_ptr<double>(), pr.second.data_ptr<double>(),
            g.data_ptr<double>(), w.data_ptr<double>(), has_w, norm,
            out.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    return out.to(input.dtype());
}

// ===========================================================================
// nll_loss family (row form)
// ===========================================================================

std::tuple<Tensor, Tensor> nll_loss_cuda(const Tensor& input,
                                         const Tensor& target,
                                         const std::optional<Tensor>& weight_opt,
                                         int64_t reduction, int64_t ignore_index) {
    if (input.dim() != 2) {
        TP_THROW(RuntimeError, "nll_loss: expected a 2-D input (N, C), got ",
                 input.dim(), " dimensions");
    }
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    if (!is_loss_float(input.dtype())) {
        TP_THROW(NotImplementedError,
                 "nll_loss CUDA supports floating dtypes only");
    }
    if (target.dtype() != DType::Int64) {
        TP_THROW(RuntimeError, "nll_loss: target must have dtype Int64");
    }
    const Tensor x = f64_dev(input);
    const Tensor tgt = target.is_contiguous() ? target : target.contiguous();
    const bool has_w = weight_opt.has_value() && weight_opt->defined();
    Tensor w = has_w ? f64_dev(*weight_opt) : Tensor();
    Tensor loss_rows = Tensor::empty({N}, DType::Float64, input.device());
    Tensor wrows = Tensor::empty({N}, DType::Float64, input.device());
    if (N) {
        nll_row_kernel<<<loss_grid(N), kThreads, 0,
                         getCurrentCUDAStream().stream()>>>(
            N, C, x.data_ptr<double>(), tgt.data_ptr<int64_t>(),
            has_w ? w.data_ptr<double>() : nullptr, has_w, ignore_index,
            loss_rows.data_ptr<double>(), wrows.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    const double wtotal = host_sum_f64(wrows, N);
    const DType tw_dt =
        input.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    Tensor total_weight =
        Tensor::full({}, Scalar(wtotal), tw_dt, input.device());
    if (reduction == 0) {
        return {loss_rows.to(input.dtype()), total_weight};
    }
    const double total = host_sum_f64(loss_rows, N);
    const double v = reduction == 1 && wtotal > 0 ? total / wtotal : total;
    return {Tensor::full({}, Scalar(v), out_scalar_dtype(input.dtype()),
                         input.device())
                .to(input.dtype()),
            total_weight};
}

Tensor nll_loss_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& target,
                              const std::optional<Tensor>& weight_opt,
                              int64_t reduction, int64_t ignore_index,
                              const Tensor& total_weight) {
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const Tensor x = f64_dev(input);
    const Tensor tgt = target.is_contiguous() ? target : target.contiguous();
    const bool has_w = weight_opt.has_value() && weight_opt->defined();
    Tensor w = has_w ? f64_dev(*weight_opt) : Tensor();
    Tensor grad_input = Tensor::zeros(
        {N, C}, DType::Float64, input.device());
    if (N) {
        const auto stream = getCurrentCUDAStream().stream();
        if (reduction == 0) {
            Tensor g = f64_dev(grad_output);
            nll_grad_none_kernel<<<loss_grid(N), kThreads, 0, stream>>>(
                N, C, g.data_ptr<double>(), tgt.data_ptr<int64_t>(),
                has_w ? w.data_ptr<double>() : nullptr, has_w, ignore_index,
                grad_input.data_ptr<double>());
        } else {
            const double g = grad_output.item().to<double>();
            double tw = total_weight.defined()
                            ? total_weight.item().to<double>()
                            : 0.0;
            // mean normalizer: total_weight when provided, otherwise the
            // count of non-ignored rows
            if (tw == 0) {
                Tensor valid = tgt.ne(Scalar(static_cast<double>(ignore_index)))
                                   .to(DType::Float64)
                                   .sum();
                tw = valid.item().to<double>();
            }
            nll_grad_scalar_kernel<<<loss_grid(N), kThreads, 0, stream>>>(
                N, C, g, tgt.data_ptr<int64_t>(),
                has_w ? w.data_ptr<double>() : nullptr, has_w, ignore_index, tw,
                grad_input.data_ptr<double>());
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return grad_input.to(input.dtype());
}

// ===========================================================================
// nll_loss2d family (spatial form)
// ===========================================================================

std::tuple<Tensor, Tensor> nll_loss2d_cuda(const Tensor& input,
                                           const Tensor& target,
                                           const std::optional<Tensor>& weight_opt,
                                           int64_t reduction,
                                           int64_t ignore_index) {
    if (input.dim() != 4) {
        TP_THROW(RuntimeError, "nll_loss2d: Expected 4D input");
    }
    if (target.dim() != 3) {
        TP_THROW(RuntimeError, "nll_loss2d: Expected 3D target");
    }
    const int64_t N = input.size(0), C = input.size(1), H = input.size(2),
                  W = input.size(3);
    if (target.size(0) != N || target.size(1) != H || target.size(2) != W) {
        TP_THROW(RuntimeError,
                 "nll_loss2d: target shape must match input spatial dims");
    }
    if (!is_loss_float(input.dtype())) {
        TP_THROW(NotImplementedError,
                 "nll_loss2d CUDA supports floating dtypes only");
    }
    const int64_t rows = N * H * W;
    const Tensor x = f64_dev(input);
    const Tensor tgt = target.is_contiguous() ? target : target.contiguous();
    const bool has_w = weight_opt.has_value() && weight_opt->defined();
    Tensor w = has_w ? f64_dev(*weight_opt) : Tensor();
    Tensor loss_rows = Tensor::empty({rows}, DType::Float64, input.device());
    Tensor wrows = Tensor::empty({rows}, DType::Float64, input.device());
    if (rows) {
        nll2d_row_kernel<<<loss_grid(rows), kThreads, 0,
                           getCurrentCUDAStream().stream()>>>(
            rows, C, H * W, x.data_ptr<double>(), tgt.data_ptr<int64_t>(),
            has_w ? w.data_ptr<double>() : nullptr, has_w, ignore_index,
            loss_rows.data_ptr<double>(), wrows.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    const double wtotal = host_sum_f64(wrows, rows);
    const DType tw_dt =
        input.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    Tensor total_weight =
        Tensor::full({}, Scalar(wtotal), tw_dt, input.device());
    if (reduction == 0) {
        return {loss_rows.to(input.dtype())
                    .reshape({N, H, W}),
                total_weight};
    }
    const double total = host_sum_f64(loss_rows, rows);
    const double v = reduction == 1 && wtotal > 0 ? total / wtotal : total;
    return {Tensor::full({}, Scalar(v), out_scalar_dtype(input.dtype()),
                         input.device())
                .to(input.dtype()),
            total_weight};
}

Tensor nll_loss2d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                               const Tensor& target,
                               const std::optional<Tensor>& weight_opt,
                               int64_t reduction, int64_t ignore_index,
                               const Tensor& total_weight) {
    const int64_t N = input.size(0), C = input.size(1), H = input.size(2),
                  W = input.size(3);
    const int64_t rows = N * H * W;
    const Tensor x = f64_dev(input);
    const Tensor tgt = target.is_contiguous() ? target : target.contiguous();
    const bool has_w = weight_opt.has_value() && weight_opt->defined();
    Tensor w = has_w ? f64_dev(*weight_opt) : Tensor();
    Tensor grad_input =
        Tensor::zeros({N, C, H, W}, DType::Float64, input.device());
    if (rows) {
        const auto stream = getCurrentCUDAStream().stream();
        if (reduction == 0) {
            Tensor g = f64_dev(grad_output);
            nll2d_grad_none_kernel<<<loss_grid(rows), kThreads, 0, stream>>>(
                rows, C, H * W, g.data_ptr<double>(), tgt.data_ptr<int64_t>(),
                has_w ? w.data_ptr<double>() : nullptr, has_w, ignore_index,
                grad_input.data_ptr<double>());
        } else {
            const double g = grad_output.item().to<double>();
            double tw = total_weight.defined() ? total_weight.item().to<double>()
                                               : 0.0;
            if (tw == 0) {
                Tensor valid = tgt.ne(Scalar(static_cast<double>(ignore_index)))
                                   .to(DType::Float64)
                                   .sum();
                tw = valid.item().to<double>();
            }
            nll2d_grad_scalar_kernel<<<loss_grid(rows), kThreads, 0, stream>>>(
                rows, C, H * W, g, tgt.data_ptr<int64_t>(),
                has_w ? w.data_ptr<double>() : nullptr, has_w, ignore_index, tw,
                grad_input.data_ptr<double>());
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return grad_input.to(input.dtype());
}

// ===========================================================================
// registration
// ===========================================================================

TENSORPLAY_LIBRARY_IMPL(CUDA, LossFillKernels) {
    m.impl("mse_loss", mse_loss_cuda);
    m.impl("mse_loss_backward", mse_loss_backward_cuda);
    m.impl("smooth_l1_loss", smooth_l1_loss_cuda);
    m.impl("smooth_l1_loss_backward", smooth_l1_loss_backward_cuda);
    m.impl("huber_loss", huber_loss_cuda);
    m.impl("huber_loss_backward", huber_loss_backward_cuda);
    m.impl("binary_cross_entropy", binary_cross_entropy_cuda);
    m.impl("binary_cross_entropy_backward", binary_cross_entropy_backward_cuda);
    m.impl("nll_loss", nll_loss_cuda);
    m.impl("nll_loss_backward", nll_loss_backward_cuda);
    m.impl("nll_loss2d", nll_loss2d_cuda);
    m.impl("nll_loss2d_backward", nll_loss2d_backward_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
