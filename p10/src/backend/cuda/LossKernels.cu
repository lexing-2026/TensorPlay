// nn loss operators - CUDA kernels (native device implementations).
//
// Mean reduction pattern: an elementwise (or one-thread-per-row) kernel
// writes per-element/per-row losses into a Float64 buffer; an atomicAdd
// reduction sums it; the scalar mean is finalized on the host after one
// l1/smooth_l1/huber/kl_div/bce/bce_with_logits/cosine_embedding/
// hinge_embedding/margin_ranking/soft_margin/triplet_margin/poisson_nll/
// multilabel_soft_margin. (multi_margin_loss / multilabel_margin_loss live
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"

#include <cuda_runtime.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace tensorplay {
namespace cuda {

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

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

Tensor mean_from_elems(const Tensor& elems, int64_t n, DType dt, const Device& dev) {
    Tensor total = Tensor::zeros({1}, DType::Float64, dev);
    if (n > 0) {
        auto stream = getCurrentCUDAStream().stream();
        atomic_sum_kernel<<<loss_grid(n), kThreads, 0, stream>>>(
            n, elems.data_ptr<double>(), total.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    double h = 0;
    CUDA_CHECK(cudaMemcpy(&h, total.data_ptr<double>(), sizeof(double),
                          cudaMemcpyDeviceToHost));
    return Tensor::full({}, Scalar(n ? h / n : 0.0),
                        dt == DType::Float64 ? DType::Float64 : DType::Float32, dev);
}

std::pair<Tensor, Tensor> pair_f64_dev(const Tensor& a, const Tensor& b) {
    std::vector<int64_t> shape = broadcast_shapes(shape_of(a), shape_of(b));
    return {a.expand(shape).contiguous().to(DType::Float64),
            b.expand(shape).contiguous().to(DType::Float64)};
}

Tensor expand_f64_dev(const Tensor& a, const std::vector<int64_t>& shape) {
    return a.expand(shape).contiguous().to(DType::Float64);
}

// ---------------------------------------------------------------------------
// elementwise kernels (one thread per pair element)
// ---------------------------------------------------------------------------

__global__ void l1_elem_kernel(int64_t n, const double* a, const double* b, double* o) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st) o[i] = ::fabs(a[i] - b[i]);
}

__global__ void kl_div_elem_kernel(int64_t n, const double* x, const double* t, double* o) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st)
        o[i] = t[i] > 0 ? t[i] * (::log(t[i]) - x[i]) : 0.0;
}

__device__ inline double dsp(double y) {
    return ::fmax(y, 0.0) + ::log1p(::exp(-::fabs(y)));
}

__global__ void bce_with_logits_elem_kernel(int64_t n, const double* x, const double* t,
                                            const double* w, const double* pw,
                                            bool has_w, bool has_pw, double* o) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st) {
        double wi = has_w ? w[i] : 1.0;
        double pi = has_pw ? pw[i] : 1.0;
        o[i] = wi * (pi * t[i] * dsp(-x[i]) + (1.0 - t[i]) * dsp(x[i]));
    }
}

__global__ void hinge_emb_elem_kernel(int64_t n, const double* x, const double* t,
                                      double margin, double* o) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st)
        o[i] = (t[i] == 1.0) ? x[i] : ::fmax(0.0, margin - x[i]);
}

__global__ void margin_rank_elem_kernel(int64_t n, const double* a, const double* b,
                                        const double* g, double margin, double* o) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st)
        o[i] = ::fmax(0.0, margin - g[i] * (a[i] - b[i]));
}

__global__ void soft_margin_elem_kernel(int64_t n, const double* x, const double* t,
                                        double* o) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st) o[i] = dsp(-t[i] * x[i]);
}

__global__ void poisson_elem_kernel(int64_t n, const double* x, const double* z,
                                    bool log_input, bool full, double eps, double* o) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st) {
        double l2 = log_input ? (::exp(x[i]) - z[i] * x[i])
                              : (x[i] - z[i] * ::log(::exp(x[i]) + eps));
        if (full && z[i] > 0) l2 += z[i] * ::log(z[i]) - ::lgamma(z[i] + 1.0);
        o[i] = l2;
    }
}

// ---------------------------------------------------------------------------
// row-wise kernels (one thread per row)
// ---------------------------------------------------------------------------

__global__ void cosine_row_kernel(int64_t N, int64_t D, const double* a, const double* b,
                                  const double* g, double margin, double* o) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < N; i += st) {
        double dot = 0, na = 0, nb = 0;
        for (int64_t j = 0; j < D; ++j) {
            dot += a[i * D + j] * b[i * D + j];
            na += a[i * D + j] * a[i * D + j];
            nb += b[i * D + j] * b[i * D + j];
        }
        double cosv = dot / (::sqrt(na) * ::sqrt(nb) + 1e-12);
        o[i] = (g[i] == 1.0) ? 1.0 - cosv : ::fmax(0.0, cosv - margin);
    }
}

__global__ void triplet_row_kernel(int64_t N, int64_t D, const double* a, const double* p,
                                   const double* nn, double margin, double pw, double* o) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < N; i += st) {
        auto dist = [&](const double* u, const double* v) {
            if (pw == std::numeric_limits<double>::infinity()) {
                double mx = 0;
                for (int64_t j = 0; j < D; ++j) mx = ::fmax(mx, ::fabs(u[j] - v[j]));
                return mx;
            }
            double s2 = 0;
            for (int64_t j = 0; j < D; ++j) s2 += ::pow(::fabs(u[j] - v[j]), pw);
            return ::pow(s2, 1.0 / pw);
        };
        o[i] = ::fmax(0.0, dist(a + i * D, p + i * D) -
                               dist(a + i * D, nn + i * D) + margin);
    }
}

__global__ void mlsm_row_kernel(int64_t N, int64_t C, const double* x, const double* t,
                                double* o) {
    // -1/C sum_c [t*logsig(x) + (1-t)*logsig(-x)]; logsig(u) = -dsp(-u)
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < N; i += st) {
        double row = 0;
        for (int64_t c = 0; c < C; ++c) {
            double xv = x[i * C + c], tv = t[i * C + c];
            row += tv * -dsp(-xv) + (1.0 - tv) * -dsp(xv);
        }
        o[i] = -row / C;
    }
}

} // anonymous namespace

// ===========================================================================
// Public entry points
// ===========================================================================

Tensor l1_loss_cuda(const Tensor& input, const Tensor& target) {
    auto pr = pair_f64_dev(input, target);
    Tensor elems = Tensor::empty(shape_of(pr.first), DType::Float64, input.device());
    int64_t n = elems.numel();
    if (n) {
        auto stream = getCurrentCUDAStream().stream();
        l1_elem_kernel<<<loss_grid(n), kThreads, 0, stream>>>(
            n, pr.first.data_ptr<double>(), pr.second.data_ptr<double>(),
            elems.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    return mean_from_elems(elems, n, input.dtype(), input.device());
}

Tensor kl_div_cuda(const Tensor& input, const Tensor& target) {
    auto pr = pair_f64_dev(input, target);
    Tensor elems = Tensor::empty(shape_of(pr.first), DType::Float64, input.device());
    int64_t n = elems.numel();
    if (n) {
        auto stream = getCurrentCUDAStream().stream();
        kl_div_elem_kernel<<<loss_grid(n), kThreads, 0, stream>>>(
            n, pr.first.data_ptr<double>(), pr.second.data_ptr<double>(),
            elems.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    return mean_from_elems(elems, n, input.dtype(), input.device());
}

Tensor binary_cross_entropy_with_logits_cuda(const Tensor& self, const Tensor& target,
                                             const std::optional<Tensor>& weight_opt,
                                             const std::optional<Tensor>& pos_weight_opt) {
    Tensor weight = weight_opt.value_or(Tensor());
    Tensor pos_weight = pos_weight_opt.value_or(Tensor());
    auto pr = pair_f64_dev(self, target);
    bool has_w = weight.defined() && weight.numel() > 0;
    bool has_pw = pos_weight.defined() && pos_weight.numel() > 0;
    Tensor w = has_w ? expand_f64_dev(weight, shape_of(pr.first))
                     : pr.first;  // dummy pointer when absent
    Tensor pw = has_pw ? expand_f64_dev(pos_weight, shape_of(pr.first))
                       : pr.first;
    Tensor elems = Tensor::empty(shape_of(pr.first), DType::Float64, self.device());
    int64_t n = elems.numel();
    if (n) {
        auto stream = getCurrentCUDAStream().stream();
        bce_with_logits_elem_kernel<<<loss_grid(n), kThreads, 0, stream>>>(
            n, pr.first.data_ptr<double>(), pr.second.data_ptr<double>(),
            w.data_ptr<double>(), pw.data_ptr<double>(), has_w, has_pw,
            elems.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    return mean_from_elems(elems, n, self.dtype(), self.device());
}

Tensor hinge_embedding_loss_cuda(const Tensor& input, const Tensor& target, Scalar margin) {
    auto pr = pair_f64_dev(input, target);
    Tensor elems = Tensor::empty(shape_of(pr.first), DType::Float64, input.device());
    int64_t n = elems.numel();
    if (n) {
        auto stream = getCurrentCUDAStream().stream();
        hinge_emb_elem_kernel<<<loss_grid(n), kThreads, 0, stream>>>(
            n, pr.first.data_ptr<double>(), pr.second.data_ptr<double>(), margin.toDouble(),
            elems.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    return mean_from_elems(elems, n, input.dtype(), input.device());
}

Tensor margin_ranking_loss_cuda(const Tensor& input1, const Tensor& input2,
                                const Tensor& target, Scalar margin) {
    auto pr = pair_f64_dev(input1, input2);
    Tensor tg = expand_f64_dev(target, shape_of(pr.first));
    Tensor elems = Tensor::empty(shape_of(pr.first), DType::Float64, input1.device());
    int64_t n = elems.numel();
    if (n) {
        auto stream = getCurrentCUDAStream().stream();
        margin_rank_elem_kernel<<<loss_grid(n), kThreads, 0, stream>>>(
            n, pr.first.data_ptr<double>(), pr.second.data_ptr<double>(),
            tg.data_ptr<double>(), margin.toDouble(), elems.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    return mean_from_elems(elems, n, input1.dtype(), input1.device());
}

Tensor soft_margin_loss_cuda(const Tensor& input, const Tensor& target) {
    auto pr = pair_f64_dev(input, target);
    Tensor elems = Tensor::empty(shape_of(pr.first), DType::Float64, input.device());
    int64_t n = elems.numel();
    if (n) {
        auto stream = getCurrentCUDAStream().stream();
        soft_margin_elem_kernel<<<loss_grid(n), kThreads, 0, stream>>>(
            n, pr.first.data_ptr<double>(), pr.second.data_ptr<double>(),
            elems.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    return mean_from_elems(elems, n, input.dtype(), input.device());
}

Tensor poisson_nll_loss_cuda(const Tensor& input, const Tensor& target, bool log_input,
                             bool full, double eps) {
    auto pr = pair_f64_dev(input, target);
    Tensor elems = Tensor::empty(shape_of(pr.first), DType::Float64, input.device());
    int64_t n = elems.numel();
    if (n) {
        auto stream = getCurrentCUDAStream().stream();
        poisson_elem_kernel<<<loss_grid(n), kThreads, 0, stream>>>(
            n, pr.first.data_ptr<double>(), pr.second.data_ptr<double>(),
            log_input, full, eps, elems.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    return mean_from_elems(elems, n, input.dtype(), input.device());
}

Tensor cosine_embedding_loss_cuda(const Tensor& x1, const Tensor& x2, const Tensor& target,
                                  Scalar margin) {
    int64_t N = x1.size(0), D = x1.size(1);
    Tensor a = x1.contiguous().to(DType::Float64);
    Tensor b = x2.contiguous().to(DType::Float64);
    Tensor tg = target.contiguous().to(DType::Float64);
    if (tg.dim() == 0) tg = tg.expand({N}).contiguous();
    Tensor elems = Tensor::empty({N}, DType::Float64, x1.device());
    if (N) {
        auto stream = getCurrentCUDAStream().stream();
        cosine_row_kernel<<<loss_grid(N), kThreads, 0, stream>>>(
            N, D, a.data_ptr<double>(), b.data_ptr<double>(), tg.data_ptr<double>(),
            margin.toDouble(), elems.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    return mean_from_elems(elems, N, x1.dtype(), x1.device());
}

Tensor triplet_margin_loss_cuda(const Tensor& anchor, const Tensor& positive,
                                const Tensor& negative, Scalar margin, double p) {
    int64_t N = anchor.size(0), D = anchor.size(1);
    Tensor a = anchor.contiguous().to(DType::Float64);
    Tensor pp = positive.contiguous().to(DType::Float64);
    Tensor nn = negative.contiguous().to(DType::Float64);
    Tensor elems = Tensor::empty({N}, DType::Float64, anchor.device());
    if (N) {
        auto stream = getCurrentCUDAStream().stream();
        triplet_row_kernel<<<loss_grid(N), kThreads, 0, stream>>>(
            N, D, a.data_ptr<double>(), pp.data_ptr<double>(), nn.data_ptr<double>(),
            margin.toDouble(), p, elems.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    return mean_from_elems(elems, N, anchor.dtype(), anchor.device());
}

Tensor multilabel_soft_margin_loss_cuda(const Tensor& input, const Tensor& target) {
    int64_t N = input.size(0), C = input.size(1);
    auto pr = pair_f64_dev(input, target);
    Tensor elems = Tensor::empty({N}, DType::Float64, input.device());
    if (N) {
        auto stream = getCurrentCUDAStream().stream();
        mlsm_row_kernel<<<loss_grid(N), kThreads, 0, stream>>>(
            N, C, pr.first.data_ptr<double>(), pr.second.data_ptr<double>(),
            elems.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    return mean_from_elems(elems, N, input.dtype(), input.device());
}

TENSORPLAY_LIBRARY_IMPL(CUDA, LossKernels) {
    m.impl("l1_loss", l1_loss_cuda);
    m.impl("kl_div", kl_div_cuda);
    m.impl("binary_cross_entropy_with_logits", binary_cross_entropy_with_logits_cuda);
    m.impl("cosine_embedding_loss", cosine_embedding_loss_cuda);
    m.impl("hinge_embedding_loss", hinge_embedding_loss_cuda);
    m.impl("margin_ranking_loss", margin_ranking_loss_cuda);
    m.impl("soft_margin_loss", soft_margin_loss_cuda);
    m.impl("triplet_margin_loss", triplet_margin_loss_cuda);
    m.impl("poisson_nll_loss", poisson_nll_loss_cuda);
    m.impl("multilabel_soft_margin_loss", multilabel_soft_margin_loss_cuda);
}

} // namespace cuda
} // namespace tensorplay
