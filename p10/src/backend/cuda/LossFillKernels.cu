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
#include "CUDALoops.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "Atomic.cuh"
#include "OutWrite.h"

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

template <typename T>
__global__ void atomic_sum_kernel_t(int64_t n, const T* in, T* total) {
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

// Device-side finalize for reduced losses: writes total * scale into a
// 0-dim result so the scalar never crosses the PCIe bus in the training
// loop (`loss.backward()` does not need the host-side value).
template <typename T>
__global__ void finalize_loss_scalar_kernel(const T* total, T* out, double scale) {
    out[0] = static_cast<T>(total[0] * scale);
}

// ---------------------------------------------------------------------------
// Squared-difference reduction
//
// The reduced mean needs one scalar out of the whole tensor.  Folding every
// element into a single address costs one contended atomic per element, which
// on a batch-sized loss dominates the launch itself.  These two stages trade
// that for a tree: each block reduces a fixed slice and writes one partial,
// then a single block folds the partials.  Both stages accumulate in the loss
// carry type, and the elementwise difference is folded into the first stage so
// no per-element buffer is written or read back.
//
// The elements are read one at a time on purpose.  A packed load would need
// the base address and the element count to be a multiple of four, and a loss
// may sit on any offset of any shape; the scalar path reads the same bytes in
// fully coalesced 32-wide transactions either way.

// Upper bound on the first stage's block count, chosen so the partial buffer
// stays small enough for the second stage to reduce in a single block.
constexpr int64_t kLossReduceMaxBlocks = 1024;
constexpr int kLossReduceBlock = 256;

template <typename T>
__global__ void loss_sq_diff_partial_kernel(int64_t n, const T* x, const T* t,
                                            T* partials) {
    __shared__ T warp_partials[kLossReduceBlock / 32];
    const int64_t stride = static_cast<int64_t>(kLossReduceBlock) * gridDim.x;
    T acc = T(0);
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * kLossReduceBlock +
                     threadIdx.x;
         i < n; i += stride) {
        const T d = x[i] - t[i];
        acc += d * d;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffffffffffull, acc, offset);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_partials[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        T total = (lane < kLossReduceBlock / 32) ? warp_partials[lane] : T(0);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffffffffffffull, total, offset);
        }
        if (lane == 0) partials[blockIdx.x] = total;
    }
}

template <typename T>
__global__ void loss_finish_partials_kernel(int64_t count, const T* partials,
                                            T* out, double scale) {
    __shared__ T warp_partials[kLossReduceBlock / 32];
    T acc = T(0);
    for (int64_t i = threadIdx.x; i < count; i += kLossReduceBlock) {
        acc += partials[i];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffffffffffull, acc, offset);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_partials[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        T total = (lane < kLossReduceBlock / 32) ? warp_partials[lane] : T(0);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffffffffffffull, total, offset);
        }
        if (lane == 0) out[0] = static_cast<T>(total * static_cast<T>(scale));
    }
}

// Host-side reduction for the loss families that still derive their scalar
// on the CPU (kept for the smooth_l1 / huber / bce rows).
double host_sum_f64(const Tensor& elems, int64_t n) {
    Tensor total = Tensor::zeros({1}, DType::Float64, elems.device());
    if (n > 0) {
        atomic_sum_kernel_t<double><<<loss_grid(n), kThreads, 0,
                            getCurrentCUDAStream().stream()>>>(
            n, elems.data_ptr<double>(), total.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    double h = 0;
    CUDA_CHECK(cudaMemcpy(&h, total.data_ptr<double>(), sizeof(double),
                          cudaMemcpyDeviceToHost));
    return h;
}

// Reduction carry type for a loss input: half/float accumulate in float,
// everything else (double, integer inputs) keeps its existing double path.
inline DType loss_accumulate_dtype(DType t) {
    return (t == DType::Float32 || t == DType::Float16 || t == DType::BFloat16)
        ? DType::Float32
        : DType::Float64;
}

std::pair<Tensor, Tensor> pair_dev(const Tensor& a, const Tensor& b,
                                   DType target) {
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
        ae = ac.expand(bs).contiguous();
        be = bc.expand(bs).contiguous();
    }
    return {ae.to(target), be.to(target)};
}

std::pair<Tensor, Tensor> pair_f64_dev(const Tensor& a, const Tensor& b) {
    return pair_dev(a, b, DType::Float64);
}

// ---------------------------------------------------------------------------
// elementwise loss / gradient kernels
// ---------------------------------------------------------------------------

template <typename T>
__global__ void mse_grad_kernel_t(int64_t n, const T* x, const T* t,
                                  const T* g, T norm, T* o) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st) {
        const T d = x[i] - t[i];
        o[i] = norm * 2.0 * d * g[i];
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
//
// Everything runs in the input's own dtype: the forward gathers one score
// per row, so rewriting the whole matrix through a wider accumulator only
// burns bandwidth, and the reduced scalar stays on the device so a training
// step never waits on a host round trip.
// ---------------------------------------------------------------------------

// One thread per batch row gathers the target class score; optional
// per-class weights multiply.  Out-of-range targets contribute nothing,
// which is the same guard the backward scatter applies.
template <typename T>
__global__ void nll_row_kernel(int64_t n, int64_t C, const T* x,
                               const int64_t* tgt, const T* w, bool has_w,
                               int64_t ignore, T* loss) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st) {
        const int64_t t = tgt[i];
        if (t == ignore || t < 0 || t >= C) {
            loss[i] = T(0);
        } else {
            const T wi = has_w ? w[t] : T(1);
            loss[i] = -x[i * C + t] * wi;
        }
    }
}

// Stage one of the reduced forward: each block folds a strided slice of
// rows into a (loss, weight) pair with a fixed tree order, so repeated
// calls land on the same value.
template <typename T, typename A>
__global__ void nll_partial_kernel(int64_t n, int64_t C, const T* x,
                                   const int64_t* tgt, const T* w, bool has_w,
                                   int64_t ignore, A* part_l, A* part_w) {
    __shared__ A sh_l[kThreads];
    __shared__ A sh_w[kThreads];
    const int tid = static_cast<int>(threadIdx.x);
    A sl = A(0), sw = A(0);
    const int64_t st = static_cast<int64_t>(gridDim.x) * kThreads;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + tid;
         i < n; i += st) {
        const int64_t t = tgt[i];
        if (t != ignore && t >= 0 && t < C) {
            const A wi = has_w ? static_cast<A>(w[t]) : A(1);
            sl -= static_cast<A>(x[i * C + t]) * wi;
            sw += wi;
        }
    }
    sh_l[tid] = sl;
    sh_w[tid] = sw;
    __syncthreads();
#pragma unroll 1
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_l[tid] += sh_l[tid + stride];
            sh_w[tid] += sh_w[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        part_l[blockIdx.x] = sh_l[0];
        part_w[blockIdx.x] = sh_w[0];
    }
}

// Stage two: one block merges the per-block partials and writes the final
// scalar pair.  A mean over zero contributing weight divides zero by zero,
// which is the NaN the unreduced-empty case must produce.
template <typename A, typename T>
__global__ void nll_finalize_kernel(int blocks, const A* part_l,
                                    const A* part_w, bool size_average,
                                    T* out, T* total_w) {
    __shared__ A sh_l[kThreads];
    __shared__ A sh_w[kThreads];
    const int tid = static_cast<int>(threadIdx.x);
    A sl = A(0), sw = A(0);
    for (int b = tid; b < blocks; b += kThreads) {
        sl += part_l[b];
        sw += part_w[b];
    }
    sh_l[tid] = sl;
    sh_w[tid] = sw;
    __syncthreads();
#pragma unroll 1
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_l[tid] += sh_l[tid + stride];
            sh_w[tid] += sh_w[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        total_w[0] = static_cast<T>(sh_w[0]);
        out[0] = static_cast<T>(size_average ? sh_l[0] / sh_w[0] : sh_l[0]);
    }
}

// reduction == 0: per-row gradients; the caller zero-fills and this scatter
// writes -w * g_i into each valid target slot.
template <typename T>
__global__ void nll_grad_none_kernel(int64_t n, int64_t C, const T* g,
                                     const int64_t* tgt, const T* w,
                                     bool has_w, int64_t ignore, T* gi) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += st) {
        const int64_t t = tgt[i];
        if (t == ignore || t < 0 || t >= C) continue;
        const T wi = has_w ? w[t] : T(1);
        gi[i * C + t] = -wi * g[i];
    }
}

// Scalar-output modes: every valid row contributes -w * g (/ tw); both
// scalars stay on the device, so no sync is needed to read them.
template <typename T>
__global__ void nll_grad_scalar_kernel(int64_t n, int64_t C, const T* g,
                                       const int64_t* tgt, const T* w,
                                       bool has_w, int64_t ignore,
                                       const T* tw, bool size_average,
                                       T* gi) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const T gv = *g;
    const T gg = size_average ? gv / *tw : gv;
    for (; i < n; i += st) {
        const int64_t t = tgt[i];
        if (t == ignore || t < 0 || t >= C) continue;
        const T wi = has_w ? w[t] : T(1);
        gi[i * C + t] = -wi * gg;
    }
}

template <typename T>
__global__ void nll2d_row_kernel_t(int64_t rows, int64_t C, int64_t HW,
                                   const T* x, const int64_t* tgt,
                                   const T* w, bool has_w, int64_t ignore,
                                   T* loss) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < rows; i += st) {
        const int64_t t = tgt[i];
        if (t == ignore || t < 0 || t >= C) {
            loss[i] = T(0);
        } else {
            const int64_t n = i / HW;
            const int64_t pos = i % HW;
            const T wi = has_w ? w[t] : T(1);
            loss[i] = -x[(n * C + t) * HW + pos] * wi;
        }
    }
}

template <typename T, typename A>
__global__ void nll2d_partial_kernel(int64_t rows, int64_t C, int64_t HW,
                                     const T* x, const int64_t* tgt,
                                     const T* w, bool has_w, int64_t ignore,
                                     A* part_l, A* part_w) {
    __shared__ A sh_l[kThreads];
    __shared__ A sh_w[kThreads];
    const int tid = static_cast<int>(threadIdx.x);
    A sl = A(0), sw = A(0);
    const int64_t st = static_cast<int64_t>(gridDim.x) * kThreads;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + tid;
         i < rows; i += st) {
        const int64_t t = tgt[i];
        if (t != ignore && t >= 0 && t < C) {
            const int64_t n = i / HW;
            const int64_t pos = i % HW;
            const A wi = has_w ? static_cast<A>(w[t]) : A(1);
            sl -= static_cast<A>(x[(n * C + t) * HW + pos]) * wi;
            sw += wi;
        }
    }
    sh_l[tid] = sl;
    sh_w[tid] = sw;
    __syncthreads();
#pragma unroll 1
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_l[tid] += sh_l[tid + stride];
            sh_w[tid] += sh_w[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        part_l[blockIdx.x] = sh_l[0];
        part_w[blockIdx.x] = sh_w[0];
    }
}

template <typename T>
__global__ void nll2d_grad_none_kernel(int64_t rows, int64_t C, int64_t HW,
                                       const T* g, const int64_t* tgt,
                                       const T* w, bool has_w,
                                       int64_t ignore, T* gi) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < rows; i += st) {
        const int64_t t = tgt[i];
        if (t == ignore || t < 0 || t >= C) continue;
        const int64_t n = i / HW;
        const int64_t pos = i % HW;
        const T wi = has_w ? w[t] : T(1);
        gi[(n * C + t) * HW + pos] = -wi * g[i];
    }
}

template <typename T>
__global__ void nll2d_grad_scalar_kernel(int64_t rows, int64_t C, int64_t HW,
                                         const T* g, const int64_t* tgt,
                                         const T* w, bool has_w,
                                         int64_t ignore, const T* tw,
                                         bool size_average, T* gi) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t st = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const T gv = *g;
    const T gg = size_average ? gv / *tw : gv;
    for (; i < rows; i += st) {
        const int64_t t = tgt[i];
        if (t == ignore || t < 0 || t >= C) continue;
        const int64_t n = i / HW;
        const int64_t pos = i % HW;
        const T wi = has_w ? w[t] : T(1);
        gi[(n * C + t) * HW + pos] = -wi * gg;
    }
}

// Reduced forward driver: cap the partial grid so stage two's merge stays
// tiny and its order fixed regardless of batch size.
constexpr int64_t kNllMaxPartials = 1024;

template <typename T, typename A>
void nll_forward_reduced(int64_t rows, int64_t C, int64_t HW, const T* x,
                         const int64_t* tgt, const T* w, bool has_w,
                         int64_t ignore, bool size_average, T* out,
                         T* total_w, cudaStream_t stream) {
    const int64_t want = (rows + kThreads - 1) / kThreads;
    const unsigned blocks =
        static_cast<unsigned>(want < kNllMaxPartials ? want : kNllMaxPartials);
    Tensor partials = Tensor::empty(
        {2 * static_cast<int64_t>(blocks)},
        std::is_same<A, double>::value ? DType::Float64 : DType::Float32,
        Device(DeviceType::CUDA));
    A* part_l = partials.data_ptr<A>();
    A* part_w = part_l + blocks;
    if (HW == 1) {
        nll_partial_kernel<T, A><<<blocks, kThreads, 0, stream>>>(
            rows, C, x, tgt, w, has_w, ignore, part_l, part_w);
    } else {
        nll2d_partial_kernel<T, A><<<blocks, kThreads, 0, stream>>>(
            rows, C, HW, x, tgt, w, has_w, ignore, part_l, part_w);
    }
    nll_finalize_kernel<A, T><<<1, kThreads, 0, stream>>>(
        static_cast<int>(blocks), part_l, part_w, size_average, out, total_w);
    CUDA_CHECK(cudaGetLastError());
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
    const DType acc = loss_accumulate_dtype(input.dtype());
    auto pr = pair_dev(input, target, acc);
    const int64_t n = pr.first.numel();
    if (reduction != 0) {
        // A reduced loss wants one number, so the squared difference is never
        // written out: the first stage folds it into a per-block partial and
        // the second folds the partials.  A mean divides by the element count
        // and a sum does not, which is the whole difference between the two.
        const double scale =
            (reduction == 1 && n) ? 1.0 / static_cast<double>(n) : 1.0;
        Tensor result = Tensor::empty({}, acc, input.device());
        if (!n) {
            // Nothing to fold, so the answer is the zero the accumulator would
            // have held.
            Tensor total = Tensor::zeros({1}, acc, input.device());
            if (acc == DType::Float64) {
                finalize_loss_scalar_kernel<double><<<1, 1, 0,
                    getCurrentCUDAStream().stream()>>>(
                    total.data_ptr<double>(), result.data_ptr<double>(), scale);
            } else {
                finalize_loss_scalar_kernel<float><<<1, 1, 0,
                    getCurrentCUDAStream().stream()>>>(
                    total.data_ptr<float>(), result.data_ptr<float>(), scale);
            }
            CUDA_CHECK(cudaGetLastError());
            return result.to(input.dtype());
        }
        {
            const int64_t blocks = std::min<int64_t>(
                (n + kLossReduceBlock - 1) / kLossReduceBlock,
                kLossReduceMaxBlocks);
            Tensor partials = Tensor::empty({blocks}, acc, input.device());
            if (acc == DType::Float64) {
                loss_sq_diff_partial_kernel<double>
                    <<<static_cast<unsigned>(blocks), kLossReduceBlock, 0,
                       getCurrentCUDAStream().stream()>>>(
                        n, pr.first.data_ptr<double>(),
                        pr.second.data_ptr<double>(),
                        partials.data_ptr<double>());
                CUDA_CHECK(cudaGetLastError());
                loss_finish_partials_kernel<double>
                    <<<1, kLossReduceBlock, 0,
                       getCurrentCUDAStream().stream()>>>(
                        blocks, partials.data_ptr<double>(),
                        result.data_ptr<double>(), scale);
            } else {
                loss_sq_diff_partial_kernel<float>
                    <<<static_cast<unsigned>(blocks), kLossReduceBlock, 0,
                       getCurrentCUDAStream().stream()>>>(
                        n, pr.first.data_ptr<float>(), pr.second.data_ptr<float>(),
                        partials.data_ptr<float>());
                CUDA_CHECK(cudaGetLastError());
                loss_finish_partials_kernel<float>
                    <<<1, kLossReduceBlock, 0,
                       getCurrentCUDAStream().stream()>>>(
                        blocks, partials.data_ptr<float>(),
                        result.data_ptr<float>(), scale);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        return result.to(input.dtype());
    }
    Tensor elems = Tensor::empty(shape_of(pr.first), acc, input.device());
    if (n) {
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(elems)
            .add_const_input(pr.first)
            .add_const_input(pr.second)
            .build();
        if (acc == DType::Float64) {
            gpu_kernel(iter, [] __host__ __device__(double x, double t) -> double {
                const double d = x - t;
                return d * d;
            });
        } else {
            gpu_kernel(iter, [] __host__ __device__(float x, float t) -> float {
                const float d = x - t;
                return d * d;
            });
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return elems.to(input.dtype());
}

Tensor mse_loss_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& target, int64_t reduction) {
    const DType acc = loss_accumulate_dtype(input.dtype());
    auto pr = pair_dev(input, target, acc);
    // The kernels read grad linearly: materialize its broadcast (the
    // incoming grad may be a stride-0 expand view from the autograd graph).
    const Tensor g = (grad_output.shape() == pr.first.shape()
                          ? grad_output
                          : grad_output.expand(shape_of(pr.first)))
        .contiguous()
        .to(acc);
    Tensor out = Tensor::empty(shape_of(pr.first), acc, input.device());
    const int64_t n = out.numel();
    if (n) {
        if (acc == DType::Float64) {
            const double norm =
                reduction == 1 ? 1.0 / static_cast<double>(n) : 1.0;
            mse_grad_kernel_t<double><<<loss_grid(n), kThreads, 0,
                                      getCurrentCUDAStream().stream()>>>(
                n, pr.first.data_ptr<double>(), pr.second.data_ptr<double>(),
                g.data_ptr<double>(), norm, out.data_ptr<double>());
        } else {
            const float norm =
                reduction == 1 ? 1.0f / static_cast<float>(n) : 1.0f;
            mse_grad_kernel_t<float><<<loss_grid(n), kThreads, 0,
                                     getCurrentCUDAStream().stream()>>>(
                n, pr.first.data_ptr<float>(), pr.second.data_ptr<float>(),
                g.data_ptr<float>(), norm, out.data_ptr<float>());
        }
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
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(elems)
            .add_const_input(pr.first)
            .add_const_input(pr.second)
            .build();
        gpu_kernel(iter, [beta] __host__ __device__(double x, double t) -> double {
            const double d = ::fabs(x - t);
            return d < beta ? 0.5 * d * d / beta : d - 0.5 * beta;
        });
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
    // The kernels read grad linearly: materialize its broadcast.
    const Tensor g = grad_output.shape() == pr.first.shape()
        ? f64_dev(grad_output)
        : f64_dev(grad_output.expand(shape_of(pr.first)));
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
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(elems)
            .add_const_input(pr.first)
            .add_const_input(pr.second)
            .build();
        gpu_kernel(iter, [delta] __host__ __device__(double x, double t) -> double {
            const double d = ::fabs(x - t);
            return d < delta ? 0.5 * d * d : delta * (d - 0.5 * delta);
        });
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
    // The kernels read grad linearly: materialize its broadcast.
    const Tensor g = grad_output.shape() == pr.first.shape()
        ? f64_dev(grad_output)
        : f64_dev(grad_output.expand(shape_of(pr.first)));
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
    // The kernels read grad linearly: materialize its broadcast.
    const Tensor g = grad_output.shape() == pr.first.shape()
        ? f64_dev(grad_output)
        : f64_dev(grad_output.expand(shape_of(pr.first)));
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
    const Tensor x = input.is_contiguous() ? input : input.contiguous();
    const Tensor tgt = target.is_contiguous() ? target : target.contiguous();
    const bool has_w = weight_opt.has_value() && weight_opt->defined();
    // Weights meet only the gathered slots; keeping them in the input dtype
    // lets the kernels run without a second copy of the class matrix.
    const Tensor w = has_w ? weight_opt->contiguous().to(input.dtype())
                           : Tensor();
    const auto stream = getCurrentCUDAStream().stream();

    if (reduction == 0) {
        // Unreduced batched losses carry no normalizer.
        Tensor loss_rows = Tensor::empty({N}, input.dtype(), input.device());
        if (N) {
            switch (input.dtype()) {
                case DType::Float64:
                    nll_row_kernel<double><<<loss_grid(N), kThreads, 0, stream>>>(
                        N, C, x.data_ptr<double>(), tgt.data_ptr<int64_t>(),
                        has_w ? w.data_ptr<double>() : nullptr, has_w,
                        ignore_index, loss_rows.data_ptr<double>());
                    break;
                case DType::Float32:
                    nll_row_kernel<float><<<loss_grid(N), kThreads, 0, stream>>>(
                        N, C, x.data_ptr<float>(), tgt.data_ptr<int64_t>(),
                        has_w ? w.data_ptr<float>() : nullptr, has_w,
                        ignore_index, loss_rows.data_ptr<float>());
                    break;
                case DType::Float16:
                    nll_row_kernel<Half><<<loss_grid(N), kThreads, 0, stream>>>(
                        N, C, x.data_ptr<Half>(), tgt.data_ptr<int64_t>(),
                        has_w ? w.data_ptr<Half>() : nullptr, has_w,
                        ignore_index, loss_rows.data_ptr<Half>());
                    break;
                case DType::BFloat16:
                    nll_row_kernel<BFloat16><<<loss_grid(N), kThreads, 0, stream>>>(
                        N, C, x.data_ptr<BFloat16>(), tgt.data_ptr<int64_t>(),
                        has_w ? w.data_ptr<BFloat16>() : nullptr, has_w,
                        ignore_index, loss_rows.data_ptr<BFloat16>());
                    break;
                default:
                    TP_THROW(NotImplementedError,
                             "nll_loss CUDA supports floating dtypes only");
            }
            CUDA_CHECK(cudaGetLastError());
        }
        return {loss_rows,
                Tensor::full({}, Scalar(0.0), input.dtype(), input.device())};
    }

    Tensor out = Tensor::empty({}, input.dtype(), input.device());
    Tensor total_weight = Tensor::empty({}, input.dtype(), input.device());
    if (N) {
        switch (input.dtype()) {
            case DType::Float64:
                nll_forward_reduced<double, double>(
                    N, C, 1, x.data_ptr<double>(), tgt.data_ptr<int64_t>(),
                    has_w ? w.data_ptr<double>() : nullptr, has_w,
                    ignore_index, reduction == 1, out.data_ptr<double>(),
                    total_weight.data_ptr<double>(), stream);
                break;
            case DType::Float32:
                nll_forward_reduced<float, float>(
                    N, C, 1, x.data_ptr<float>(), tgt.data_ptr<int64_t>(),
                    has_w ? w.data_ptr<float>() : nullptr, has_w,
                    ignore_index, reduction == 1, out.data_ptr<float>(),
                    total_weight.data_ptr<float>(), stream);
                break;
            case DType::Float16:
                nll_forward_reduced<Half, float>(
                    N, C, 1, x.data_ptr<Half>(), tgt.data_ptr<int64_t>(),
                    has_w ? w.data_ptr<Half>() : nullptr, has_w,
                    ignore_index, reduction == 1, out.data_ptr<Half>(),
                    total_weight.data_ptr<Half>(), stream);
                break;
            case DType::BFloat16:
                nll_forward_reduced<BFloat16, float>(
                    N, C, 1, x.data_ptr<BFloat16>(), tgt.data_ptr<int64_t>(),
                    has_w ? w.data_ptr<BFloat16>() : nullptr, has_w,
                    ignore_index, reduction == 1, out.data_ptr<BFloat16>(),
                    total_weight.data_ptr<BFloat16>(), stream);
                break;
            default:
                TP_THROW(NotImplementedError,
                         "nll_loss CUDA supports floating dtypes only");
        }
    } else {
        // Empty batch: zero contributing weight; a mean becomes NaN.
        switch (input.dtype()) {
            case DType::Float64:
                nll_finalize_kernel<double, double><<<1, kThreads, 0, stream>>>(
                    0, nullptr, nullptr, reduction == 1,
                    out.data_ptr<double>(), total_weight.data_ptr<double>());
                break;
            case DType::Float32:
                nll_finalize_kernel<float, float><<<1, kThreads, 0, stream>>>(
                    0, nullptr, nullptr, reduction == 1,
                    out.data_ptr<float>(), total_weight.data_ptr<float>());
                break;
            case DType::Float16:
                nll_finalize_kernel<float, Half><<<1, kThreads, 0, stream>>>(
                    0, nullptr, nullptr, reduction == 1,
                    out.data_ptr<Half>(), total_weight.data_ptr<Half>());
                break;
            case DType::BFloat16:
                nll_finalize_kernel<float, BFloat16><<<1, kThreads, 0, stream>>>(
                    0, nullptr, nullptr, reduction == 1,
                    out.data_ptr<BFloat16>(), total_weight.data_ptr<BFloat16>());
                break;
            default:
                TP_THROW(NotImplementedError,
                         "nll_loss CUDA supports floating dtypes only");
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return {out, total_weight};
}

Tensor nll_loss_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& target,
                              const std::optional<Tensor>& weight_opt,
                              int64_t reduction, int64_t ignore_index,
                              const Tensor& total_weight) {
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    TP_CHECK(grad_output.dtype() == input.dtype(),
             "nll_loss_backward: grad_output dtype must match input dtype");
    const Tensor tgt = target.is_contiguous() ? target : target.contiguous();
    const bool has_w = weight_opt.has_value() && weight_opt->defined();
    const Tensor w = has_w ? weight_opt->contiguous().to(input.dtype())
                           : Tensor();
    Tensor grad_input = Tensor::zeros({N, C}, input.dtype(), input.device());
    if (N) {
        const auto stream = getCurrentCUDAStream().stream();
        const Tensor g = grad_output.is_contiguous()
            ? grad_output : grad_output.contiguous();
        #define TP_NLL_BWD_LAUNCH(SCALAR_T)                                \
            do {                                                           \
                if (reduction == 0) {                                      \
                    nll_grad_none_kernel<SCALAR_T><<<loss_grid(N), kThreads, 0, stream>>>( \
                        N, C, g.data_ptr<SCALAR_T>(),                      \
                        tgt.data_ptr<int64_t>(),                           \
                        has_w ? w.data_ptr<SCALAR_T>() : nullptr, has_w,   \
                        ignore_index, grad_input.data_ptr<SCALAR_T>());    \
                } else {                                                   \
                    TP_CHECK(grad_output.numel() == 1,                     \
                             "nll_loss_backward: expected grad_output to be a single element tensor"); \
                    TP_CHECK(total_weight.numel() == 1,                    \
                             "nll_loss_backward: expected total_weight to be a single element tensor"); \
                    nll_grad_scalar_kernel<SCALAR_T><<<loss_grid(N), kThreads, 0, stream>>>( \
                        N, C, g.data_ptr<SCALAR_T>(),                      \
                        tgt.data_ptr<int64_t>(),                           \
                        has_w ? w.data_ptr<SCALAR_T>() : nullptr, has_w,   \
                        ignore_index,                                      \
                        total_weight.data_ptr<SCALAR_T>(), reduction == 1, \
                        grad_input.data_ptr<SCALAR_T>());                  \
                }                                                          \
            } while (0)
        switch (input.dtype()) {
            case DType::Float64: TP_NLL_BWD_LAUNCH(double); break;
            case DType::Float32: TP_NLL_BWD_LAUNCH(float); break;
            case DType::Float16: TP_NLL_BWD_LAUNCH(Half); break;
            case DType::BFloat16: TP_NLL_BWD_LAUNCH(BFloat16); break;
            default:
                TP_THROW(NotImplementedError,
                         "nll_loss_backward CUDA supports floating dtypes only");
        }
        #undef TP_NLL_BWD_LAUNCH
        CUDA_CHECK(cudaGetLastError());
    }
    return grad_input;
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
    const Tensor x = input.is_contiguous() ? input : input.contiguous();
    const Tensor tgt = target.is_contiguous() ? target : target.contiguous();
    const bool has_w = weight_opt.has_value() && weight_opt->defined();
    const Tensor w = has_w ? weight_opt->contiguous().to(input.dtype())
                           : Tensor();
    const auto stream = getCurrentCUDAStream().stream();

    if (reduction == 0) {
        // Unreduced losses carry no normalizer: total_weight stays zero.
        Tensor loss_rows =
            Tensor::empty({rows}, input.dtype(), input.device());
        if (rows) {
            switch (input.dtype()) {
                case DType::Float64:
                    nll2d_row_kernel_t<double><<<loss_grid(rows), kThreads, 0, stream>>>(
                        rows, C, H * W, x.data_ptr<double>(),
                        tgt.data_ptr<int64_t>(),
                        has_w ? w.data_ptr<double>() : nullptr, has_w,
                        ignore_index, loss_rows.data_ptr<double>());
                    break;
                case DType::Float32:
                    nll2d_row_kernel_t<float><<<loss_grid(rows), kThreads, 0, stream>>>(
                        rows, C, H * W, x.data_ptr<float>(),
                        tgt.data_ptr<int64_t>(),
                        has_w ? w.data_ptr<float>() : nullptr, has_w,
                        ignore_index, loss_rows.data_ptr<float>());
                    break;
                case DType::Float16:
                    nll2d_row_kernel_t<Half><<<loss_grid(rows), kThreads, 0, stream>>>(
                        rows, C, H * W, x.data_ptr<Half>(),
                        tgt.data_ptr<int64_t>(),
                        has_w ? w.data_ptr<Half>() : nullptr, has_w,
                        ignore_index, loss_rows.data_ptr<Half>());
                    break;
                case DType::BFloat16:
                    nll2d_row_kernel_t<BFloat16><<<loss_grid(rows), kThreads, 0, stream>>>(
                        rows, C, H * W, x.data_ptr<BFloat16>(),
                        tgt.data_ptr<int64_t>(),
                        has_w ? w.data_ptr<BFloat16>() : nullptr, has_w,
                        ignore_index, loss_rows.data_ptr<BFloat16>());
                    break;
                default:
                    TP_THROW(NotImplementedError,
                             "nll_loss2d CUDA supports floating dtypes only");
            }
            CUDA_CHECK(cudaGetLastError());
        }
        return {loss_rows.reshape({N, H, W}),
                Tensor::full({}, Scalar(0.0), input.dtype(), input.device())};
    }

    Tensor out = Tensor::empty({}, input.dtype(), input.device());
    Tensor total_weight = Tensor::empty({}, input.dtype(), input.device());
    if (rows) {
        switch (input.dtype()) {
            case DType::Float64:
                nll_forward_reduced<double, double>(
                    rows, C, H * W, x.data_ptr<double>(),
                    tgt.data_ptr<int64_t>(),
                    has_w ? w.data_ptr<double>() : nullptr, has_w,
                    ignore_index, reduction == 1, out.data_ptr<double>(),
                    total_weight.data_ptr<double>(), stream);
                break;
            case DType::Float32:
                nll_forward_reduced<float, float>(
                    rows, C, H * W, x.data_ptr<float>(),
                    tgt.data_ptr<int64_t>(),
                    has_w ? w.data_ptr<float>() : nullptr, has_w,
                    ignore_index, reduction == 1, out.data_ptr<float>(),
                    total_weight.data_ptr<float>(), stream);
                break;
            case DType::Float16:
                nll_forward_reduced<Half, float>(
                    rows, C, H * W, x.data_ptr<Half>(),
                    tgt.data_ptr<int64_t>(),
                    has_w ? w.data_ptr<Half>() : nullptr, has_w,
                    ignore_index, reduction == 1, out.data_ptr<Half>(),
                    total_weight.data_ptr<Half>(), stream);
                break;
            case DType::BFloat16:
                nll_forward_reduced<BFloat16, float>(
                    rows, C, H * W, x.data_ptr<BFloat16>(),
                    tgt.data_ptr<int64_t>(),
                    has_w ? w.data_ptr<BFloat16>() : nullptr, has_w,
                    ignore_index, reduction == 1, out.data_ptr<BFloat16>(),
                    total_weight.data_ptr<BFloat16>(), stream);
                break;
            default:
                TP_THROW(NotImplementedError,
                         "nll_loss2d CUDA supports floating dtypes only");
        }
    } else {
        // Empty spatial input: zero contributing weight; a mean is NaN.
        switch (input.dtype()) {
            case DType::Float64:
                nll_finalize_kernel<double, double><<<1, kThreads, 0, stream>>>(
                    0, nullptr, nullptr, reduction == 1,
                    out.data_ptr<double>(), total_weight.data_ptr<double>());
                break;
            case DType::Float32:
                nll_finalize_kernel<float, float><<<1, kThreads, 0, stream>>>(
                    0, nullptr, nullptr, reduction == 1,
                    out.data_ptr<float>(), total_weight.data_ptr<float>());
                break;
            case DType::Float16:
                nll_finalize_kernel<float, Half><<<1, kThreads, 0, stream>>>(
                    0, nullptr, nullptr, reduction == 1,
                    out.data_ptr<Half>(), total_weight.data_ptr<Half>());
                break;
            case DType::BFloat16:
                nll_finalize_kernel<float, BFloat16><<<1, kThreads, 0, stream>>>(
                    0, nullptr, nullptr, reduction == 1,
                    out.data_ptr<BFloat16>(), total_weight.data_ptr<BFloat16>());
                break;
            default:
                TP_THROW(NotImplementedError,
                         "nll_loss2d CUDA supports floating dtypes only");
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return {out, total_weight};
}

Tensor nll_loss2d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                               const Tensor& target,
                               const std::optional<Tensor>& weight_opt,
                               int64_t reduction, int64_t ignore_index,
                               const Tensor& total_weight) {
    const int64_t N = input.size(0), C = input.size(1), H = input.size(2),
                  W = input.size(3);
    const int64_t rows = N * H * W;
    TP_CHECK(grad_output.dtype() == input.dtype(),
             "nll_loss2d_backward: grad_output dtype must match input dtype");
    const Tensor tgt = target.is_contiguous() ? target : target.contiguous();
    const bool has_w = weight_opt.has_value() && weight_opt->defined();
    const Tensor w = has_w ? weight_opt->contiguous().to(input.dtype())
                           : Tensor();
    Tensor grad_input =
        Tensor::zeros({N, C, H, W}, input.dtype(), input.device());
    if (rows) {
        const auto stream = getCurrentCUDAStream().stream();
        const Tensor g = grad_output.is_contiguous()
            ? grad_output : grad_output.contiguous();
        #define TP_NLL2D_BWD_LAUNCH(SCALAR_T)                                  \
            do {                                                               \
                if (reduction == 0) {                                          \
                    nll2d_grad_none_kernel<SCALAR_T><<<loss_grid(rows), kThreads, 0, stream>>>( \
                        rows, C, H * W, g.data_ptr<SCALAR_T>(),                \
                        tgt.data_ptr<int64_t>(),                               \
                        has_w ? w.data_ptr<SCALAR_T>() : nullptr, has_w,       \
                        ignore_index, grad_input.data_ptr<SCALAR_T>());        \
                } else {                                                       \
                    TP_CHECK(grad_output.numel() == 1,                         \
                             "nll_loss2d_backward: expected grad_output to be a single element tensor"); \
                    TP_CHECK(total_weight.numel() == 1,                        \
                             "nll_loss2d_backward: expected total_weight to be a single element tensor"); \
                    nll2d_grad_scalar_kernel<SCALAR_T><<<loss_grid(rows), kThreads, 0, stream>>>( \
                        rows, C, H * W, g.data_ptr<SCALAR_T>(),                \
                        tgt.data_ptr<int64_t>(),                               \
                        has_w ? w.data_ptr<SCALAR_T>() : nullptr, has_w,       \
                        ignore_index,                                          \
                        total_weight.data_ptr<SCALAR_T>(), reduction == 1,     \
                        grad_input.data_ptr<SCALAR_T>());                      \
                }                                                              \
            } while (0)
        switch (input.dtype()) {
            case DType::Float64: TP_NLL2D_BWD_LAUNCH(double); break;
            case DType::Float32: TP_NLL2D_BWD_LAUNCH(float); break;
            case DType::Float16: TP_NLL2D_BWD_LAUNCH(Half); break;
            case DType::BFloat16: TP_NLL2D_BWD_LAUNCH(BFloat16); break;
            default:
                TP_THROW(NotImplementedError,
                         "nll_loss2d_backward CUDA supports floating dtypes only");
        }
        #undef TP_NLL2D_BWD_LAUNCH
        CUDA_CHECK(cudaGetLastError());
    }
    return grad_input;
}

// ===========================================================================
// registration
// ===========================================================================

Tensor& interop_smooth_l1_loss_backward_grad_input_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& target,
              int64_t reduction, double beta, Tensor& grad_input) {
        write_out(grad_input, smooth_l1_loss_backward_cuda(grad_output, input, target,
                                                  reduction, beta));
        return grad_input;
    
}

Tensor& interop_binary_cross_entropy_out_cuda(const Tensor& input, const Tensor& target,
              const std::optional<Tensor>& weight, int64_t reduction, Tensor& out) {
        write_out(out, binary_cross_entropy_cuda(input, target, weight, reduction));
        return out;
    
}

Tensor& interop_binary_cross_entropy_backward_grad_input_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& target,
              const std::optional<Tensor>& weight, int64_t reduction,
              Tensor& grad_input) {
        write_out(grad_input, binary_cross_entropy_backward_cuda(grad_output, input, target,
                                                        weight, reduction));
        return grad_input;
    
}

Tensor& interop_nll_loss_backward_grad_input_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& target,
              const std::optional<Tensor>& weight, int64_t reduction,
              int64_t ignore_index, const Tensor& total_weight,
              Tensor& grad_input) {
        write_out(grad_input, nll_loss_backward_cuda(grad_output, input, target, weight,
                                            reduction, ignore_index, total_weight));
        return grad_input;
    
}

Tensor& interop_nll_loss2d_backward_grad_input_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& target,
              const std::optional<Tensor>& weight, int64_t reduction,
              int64_t ignore_index, const Tensor& total_weight,
              Tensor& grad_input) {
        write_out(grad_input, nll_loss2d_backward_cuda(grad_output, input, target, weight,
                                              reduction, ignore_index, total_weight));
        return grad_input;
    
}

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

    // out-variants: run the value kernel, then transfer into the caller's
    // buffer (grad_input for backward spellings).
    m.impl("smooth_l1_loss_backward.grad_input", interop_smooth_l1_loss_backward_grad_input_cuda);
    m.impl("binary_cross_entropy.out", interop_binary_cross_entropy_out_cuda);
    m.impl("binary_cross_entropy_backward.grad_input", interop_binary_cross_entropy_backward_grad_input_cuda);
    m.impl("nll_loss_backward.grad_input", interop_nll_loss_backward_grad_input_cuda);
    m.impl("nll_loss2d_backward.grad_input", interop_nll_loss2d_backward_grad_input_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
