// Fused RNN cell kernels — port of
// (_thnn_fused_lstm_cell_cuda / _thnn_fused_gru_cell_cuda and their
// backward impls).  TensorInfo addressing is replaced by flat contiguous
// pointers: gates are row-major (N, G) with G = 4*H (LSTM) / 3*H (GRU),
// states are (N, H), biases are (G,) or absent.

#include "RNNCudaKernels.h"

#include "CUDAContext.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include <cuda_runtime.h>

namespace tensorplay {
namespace cuda {
namespace rnn {
namespace {

#define CUDA_CHECK(condition)                                    \
    do {                                                         \
        cudaError_t error = condition;                           \
        if (error != cudaSuccess) {                              \
            TP_THROW(RuntimeError, std::string("CUDA Error: ") + \
                                       cudaGetErrorString(error)); \
        }                                                        \
    } while (0)

template <typename T>
struct AccTraits {
    using type = T;
};
template <>
struct AccTraits<tensorplay::Half> {
    using type = float;
};
template <>
struct AccTraits<tensorplay::BFloat16> {
    using type = float;
};

template <typename T>
__device__ inline typename AccTraits<T>::type ldv(const T* p) {
    return static_cast<typename AccTraits<T>::type>(*p);
}
template <typename T>
__device__ inline void stv(T* p, typename AccTraits<T>::type v) {
    *p = static_cast<T>(v);
}

template <typename M>
__device__ inline M msigmoid(M x) {
    return M(1) / (M(1) + ::exp(-x));
}

constexpr int64_t kThreads = 256;

inline dim3 make_grid(int64_t n) {
    int64_t g = (n + kThreads - 1) / kThreads;
    return dim3(static_cast<unsigned int>(g));
}

// ---------------------------------------------------------------------------
// LSTM cell forward
// ---------------------------------------------------------------------------
template <typename T>
__global__ void lstm_cell_forward_kernel(
        int64_t total, int64_t hsz,
        const T* input_gates, const T* hidden_gates,
        const T* bias1, const T* bias2,
        const T* cx, T* hy, T* cy, T* workspace) {
    using M = typename AccTraits<T>::type;
    for (int64_t li = blockIdx.x * blockDim.x + threadIdx.x;
         li < total;
         li += gridDim.x * blockDim.x) {
        const int64_t off = (li / hsz) * 4 * hsz + li % hsz;

        const M iig = ldv(input_gates + off);
        const M ifg = ldv(input_gates + off + hsz);
        const M icg = ldv(input_gates + off + 2 * hsz);
        const M iog = ldv(input_gates + off + 3 * hsz);

        const M hig = ldv(hidden_gates + off);
        const M hfg = ldv(hidden_gates + off + hsz);
        const M hcg = ldv(hidden_gates + off + 2 * hsz);
        const M hog = ldv(hidden_gates + off + 3 * hsz);

        const bool has_bias = bias1 != nullptr;
        const int64_t b = li % hsz;
        M b1i = 0, b1f = 0, b1c = 0, b1o = 0;
        M b2i = 0, b2f = 0, b2c = 0, b2o = 0;
        if (has_bias) {
            b1i = ldv(bias1 + b);            b1f = ldv(bias1 + b + hsz);
            b1c = ldv(bias1 + b + 2 * hsz);  b1o = ldv(bias1 + b + 3 * hsz);
            b2i = ldv(bias2 + b);            b2f = ldv(bias2 + b + hsz);
            b2c = ldv(bias2 + b + 2 * hsz);  b2o = ldv(bias2 + b + 3 * hsz);
        }

        const M ig = msigmoid(iig + hig + b1i + b2i);
        const M fg = msigmoid(ifg + hfg + b1f + b2f);
        const M cg = ::tanh(icg + hcg + b1c + b2c);
        const M og = msigmoid(iog + hog + b1o + b2o);

        const M cv = fg * ldv(cx + li) + ig * cg;
        stv(cy + li, cv);
        stv(hy + li, og * ::tanh(cv));

        // Saved for backward: gate activations in workspace.
        stv(workspace + off, ig);
        stv(workspace + off + hsz, fg);
        stv(workspace + off + 2 * hsz, cg);
        stv(workspace + off + 3 * hsz, og);
    }
}

// ---------------------------------------------------------------------------
// LSTM cell backward
// ---------------------------------------------------------------------------
template <typename T>
__global__ void lstm_cell_backward_kernel(
        int64_t total, int64_t hsz,
        const T* grad_hy, const T* grad_cy,
        const T* cx, const T* cy, const T* workspace,
        T* grad_gates, T* grad_cx) {
    using M = typename AccTraits<T>::type;
    for (int64_t li = blockIdx.x * blockDim.x + threadIdx.x;
         li < total;
         li += gridDim.x * blockDim.x) {
        const int64_t off = (li / hsz) * 4 * hsz + li % hsz;

        const M ig = ldv(workspace + off);
        const M fg = ldv(workspace + off + hsz);
        const M cg = ldv(workspace + off + 2 * hsz);
        const M og = ldv(workspace + off + 3 * hsz);

        const M cxv = ldv(cx + li);
        const M cyv = ldv(cy + li);

        const M go = grad_hy != nullptr ? ldv(grad_hy + li) : M(0);
        const M goc = grad_cy != nullptr ? ldv(grad_cy + li) : M(0);

        M gcx = ::tanh(cyv);

        M gog = go * gcx;
        gcx = go * og * (M(1) - gcx * gcx) + goc;

        M gig = gcx * cg;
        M gfg = gcx * cxv;
        M gcg = gcx * ig;

        gcx = gcx * fg;

        gig = gig * (M(1) - ig) * ig;
        gfg = gfg * (M(1) - fg) * fg;
        gcg = gcg * (M(1) - cg * cg);
        gog = gog * (M(1) - og) * og;

        stv(grad_gates + off, gig);
        stv(grad_gates + off + hsz, gfg);
        stv(grad_gates + off + 2 * hsz, gcg);
        stv(grad_gates + off + 3 * hsz, gog);
        stv(grad_cx + li, gcx);
    }
}

// ---------------------------------------------------------------------------
// GRU cell forward
// ---------------------------------------------------------------------------
template <typename T>
__global__ void gru_cell_forward_kernel(
        int64_t total, int64_t hsz,
        const T* input_gates, const T* hidden_gates,
        const T* bias1, const T* bias2,
        const T* hx, T* hy, T* workspace) {
    using M = typename AccTraits<T>::type;
    for (int64_t li = blockIdx.x * blockDim.x + threadIdx.x;
         li < total;
         li += gridDim.x * blockDim.x) {
        int64_t off = (li / hsz) * 3 * hsz + li % hsz;

        const M ir = ldv(input_gates + off);
        const M ii = ldv(input_gates + off + hsz);
        const M in = ldv(input_gates + off + 2 * hsz);
        const M hr = ldv(hidden_gates + off);
        const M hi = ldv(hidden_gates + off + hsz);
        const M hn = ldv(hidden_gates + off + 2 * hsz);

        const M hxv = ldv(hx + li);

        const bool has_bias = bias1 != nullptr;
        const int64_t b = li % hsz;
        M b1r = 0, b1i = 0, b1n = 0, b2r = 0, b2i = 0, b2n = 0;
        if (has_bias) {
            b1r = ldv(bias1 + b);            b1i = ldv(bias1 + b + hsz);
            b1n = ldv(bias1 + b + 2 * hsz);
            b2r = ldv(bias2 + b);            b2i = ldv(bias2 + b + hsz);
            b2n = ldv(bias2 + b + 2 * hsz);
        }

        const int64_t woff = (li / hsz) * 5 * hsz + li % hsz;

        const M rg = msigmoid(ir + hr + b1r + b2r);
        const M ig = msigmoid(ii + hi + b1i + b2i);
        M ng = in + b1n + rg * (hn + b2n);
        ng = ::tanh(ng);
        stv(hy + li, ng + ig * (hxv - ng));

        // Workspace layout (5H): rg, ig, ng, hx, hn+b_hn.
        stv(workspace + woff, rg);
        stv(workspace + woff + hsz, ig);
        stv(workspace + woff + 2 * hsz, ng);
        stv(workspace + woff + 3 * hsz, hxv);
        stv(workspace + woff + 4 * hsz, hn + b2n);
    }
}

// ---------------------------------------------------------------------------
// GRU cell backward
// ---------------------------------------------------------------------------
template <typename T>
__global__ void gru_cell_backward_kernel(
        int64_t total, int64_t hsz,
        const T* grad_hy, const T* workspace,
        T* grad_input_gates, T* grad_hidden_gates, T* grad_hx) {
    using M = typename AccTraits<T>::type;
    for (int64_t li = blockIdx.x * blockDim.x + threadIdx.x;
         li < total;
         li += gridDim.x * blockDim.x) {
        const int64_t woff = (li / hsz) * 5 * hsz + li % hsz;

        const M rg = ldv(workspace + woff);
        const M ig = ldv(workspace + woff + hsz);
        const M ng = ldv(workspace + woff + 2 * hsz);
        const M hx = ldv(workspace + woff + 3 * hsz);
        const M hn = ldv(workspace + woff + 4 * hsz);

        const M go = ldv(grad_hy + li);

        const int64_t off = (li / hsz) * 3 * hsz + li % hsz;

        const M gig = go * (hx - ng) * (M(1) - ig) * ig;
        const M ghx = go * ig;
        const M gin = go * (M(1) - ig) * (M(1) - ng * ng);
        const M ghn = gin * rg;
        const M grg = gin * hn * (M(1) - rg) * rg;

        stv(grad_input_gates + off, grg);
        stv(grad_input_gates + off + hsz, gig);
        stv(grad_input_gates + off + 2 * hsz, gin);

        stv(grad_hidden_gates + off, grg);
        stv(grad_hidden_gates + off + hsz, gig);
        stv(grad_hidden_gates + off + 2 * hsz, ghn);
        stv(grad_hx + li, ghx);
    }
}

inline Tensor cont(const Tensor& t) { return t.is_contiguous() ? t : t.contiguous(); }

// AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16) equivalent).
#define TP_RNN_DISPATCH(FN)                                             \
    switch (dtype) {                                                    \
        case DType::Float64: FN(double{}); break;                       \
        case DType::Float16: FN(tensorplay::Half{}); break;             \
        case DType::BFloat16: FN(tensorplay::BFloat16{}); break;        \
        default: FN(float{}); break;                                    \
    }

} // namespace

std::tuple<Tensor, Tensor, Tensor> fused_lstm_cell(
        const Tensor& input_gates, const Tensor& hidden_gates,
        const Tensor& cx, const Tensor& input_bias, const Tensor& hidden_bias) {
    const DType dtype = input_gates.dtype();
    Tensor ig = cont(input_gates);
    Tensor hg = cont(hidden_gates);
    Tensor c = cont(cx);
    const bool has_bias = input_bias.defined() && input_bias.numel() > 0 &&
                           hidden_bias.defined() && hidden_bias.numel() > 0;
    Tensor b1 = has_bias ? cont(input_bias) : Tensor();
    Tensor b2 = has_bias ? cont(hidden_bias) : Tensor();

    const int64_t N = c.size(0);
    const int64_t H = c.size(1);
    const int64_t total = N * H;

    Tensor hy = Tensor::empty({N, H}, dtype, c.device());
    Tensor cy = Tensor::empty_like(hy, DType::Undefined, hy.device());
    Tensor workspace = Tensor::empty({N, 4 * H}, dtype, c.device());

    auto launch = [&](auto tag) -> void {
        using T = decltype(tag);
        const T* b1p = has_bias ? b1.data_ptr<T>() : nullptr;
        const T* b2p = has_bias ? b2.data_ptr<T>() : nullptr;
        lstm_cell_forward_kernel<T><<<make_grid(total), kThreads, 0,
                                     getCurrentCUDAStream().stream()>>>(
            total, H, ig.data_ptr<T>(), hg.data_ptr<T>(), b1p, b2p,
            c.data_ptr<T>(), hy.data_ptr<T>(), cy.data_ptr<T>(),
            workspace.data_ptr<T>());
        CUDA_CHECK(cudaGetLastError());
    };
    TP_RNN_DISPATCH(launch)
    return {hy, cy, workspace};
}

std::tuple<Tensor, Tensor> fused_gru_cell(
        const Tensor& input_gates, const Tensor& hidden_gates,
        const Tensor& hx, const Tensor& input_bias, const Tensor& hidden_bias) {
    const DType dtype = input_gates.dtype();
    Tensor ig = cont(input_gates);
    Tensor hg = cont(hidden_gates);
    Tensor h = cont(hx);
    const bool has_bias = input_bias.defined() && input_bias.numel() > 0 &&
                           hidden_bias.defined() && hidden_bias.numel() > 0;
    Tensor b1 = has_bias ? cont(input_bias) : Tensor();
    Tensor b2 = has_bias ? cont(hidden_bias) : Tensor();

    const int64_t N = h.size(0);
    const int64_t H = h.size(1);
    const int64_t total = N * H;

    Tensor hy = Tensor::empty_like(h, DType::Undefined, h.device());
    Tensor workspace = Tensor::empty({N, 5 * H}, dtype, h.device());

    auto launch = [&](auto tag) -> void {
        using T = decltype(tag);
        const T* b1p = has_bias ? b1.data_ptr<T>() : nullptr;
        const T* b2p = has_bias ? b2.data_ptr<T>() : nullptr;
        gru_cell_forward_kernel<T><<<make_grid(total), kThreads, 0,
                                    getCurrentCUDAStream().stream()>>>(
            total, H, ig.data_ptr<T>(), hg.data_ptr<T>(), b1p, b2p,
            h.data_ptr<T>(), hy.data_ptr<T>(), workspace.data_ptr<T>());
        CUDA_CHECK(cudaGetLastError());
    };
    TP_RNN_DISPATCH(launch)
    return {hy, workspace};
}

std::tuple<Tensor, Tensor, Tensor> fused_lstm_cell_backward_impl(
        const Tensor& grad_hy, const Tensor& grad_cy,
        const Tensor& cx, const Tensor& cy, const Tensor& workspace) {
    const DType dtype = workspace.dtype();
    const bool has_hy = grad_hy.numel() > 0;
    const bool has_cy = grad_cy.numel() > 0;
    if (!has_hy && !has_cy) {
        TP_THROW(RuntimeError, "_thnn_fused_lstm_cell_backward: both gradients undefined");
    }
    Tensor c = cont(cx);
    Tensor y = cont(cy);
    Tensor ws = cont(workspace);
    Tensor gh = has_hy ? cont(grad_hy) : Tensor();
    Tensor gc = has_cy ? cont(grad_cy) : Tensor();

    const int64_t N = c.size(0);
    const int64_t H = c.size(1);
    const int64_t total = N * H;

    Tensor grad_gates = Tensor::empty_like(ws, DType::Undefined, ws.device());
    Tensor grad_cx_out = Tensor::empty_like(c, DType::Undefined, c.device());

    auto launch = [&](auto tag) -> void {
        using T = decltype(tag);
        const T* ghp = has_hy ? gh.data_ptr<T>() : nullptr;
        const T* gcp = has_cy ? gc.data_ptr<T>() : nullptr;
        lstm_cell_backward_kernel<T><<<make_grid(total), kThreads, 0,
                                      getCurrentCUDAStream().stream()>>>(
            total, H, ghp, gcp, c.data_ptr<T>(), y.data_ptr<T>(),
            ws.data_ptr<T>(), grad_gates.data_ptr<T>(),
            grad_cx_out.data_ptr<T>());
        CUDA_CHECK(cudaGetLastError());
    };
    TP_RNN_DISPATCH(launch)

    Tensor zero;
    return {grad_gates, grad_cx_out, zero};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> fused_gru_cell_backward(
        const Tensor& grad_hy, const Tensor& workspace) {
    const DType dtype = workspace.dtype();
    Tensor gh = cont(grad_hy);
    Tensor ws = cont(workspace);

    const int64_t N = ws.size(0);
    const int64_t H = ws.size(1) / 5;
    const int64_t total = N * H;

    Tensor grad_ig = Tensor::empty({N, 3 * H}, dtype, ws.device());
    Tensor grad_hg = Tensor::empty({N, 3 * H}, dtype, ws.device());
    Tensor grad_hx = Tensor::empty_like(gh, DType::Undefined, gh.device());

    auto launch = [&](auto tag) -> void {
        using T = decltype(tag);
        gru_cell_backward_kernel<T><<<make_grid(total), kThreads, 0,
                                     getCurrentCUDAStream().stream()>>>(
            total, H, gh.data_ptr<T>(), ws.data_ptr<T>(),
            grad_ig.data_ptr<T>(), grad_hg.data_ptr<T>(),
            grad_hx.data_ptr<T>());
        CUDA_CHECK(cudaGetLastError());
    };
    TP_RNN_DISPATCH(launch)

    Tensor zero;
    // at the composite level (RNN.cpp _thnn_fused_gru_cell_backward).
    return {grad_ig, grad_hg, grad_hx, zero, zero};
}

} // namespace rnn
} // namespace cuda
} // namespace tensorplay
