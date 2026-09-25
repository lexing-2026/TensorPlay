// Fused RNN cell kernels: forward/backward LSTM and GRU cell updates in a
// single elementwise pass.  Gates are row-major (N, G) with G = 4*H (LSTM) /
// 3*H (GRU), states are (N, H), biases are (G,) or absent.

#include "RNNCudaKernels.h"

#include "CUDAContext.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "GradMode.h"
#include <cuda_runtime.h>
#include <vector>
#include "tensorplay/ops/TensorRedispatchGenerated.h"
#include "CudaDispatchHelpers.cuh"

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
        TP_THROW(RuntimeError, "lstm cell backward: both gradients undefined");
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
    // consumed at the composite level (fused gru cell backward).
    return {grad_ig, grad_hg, grad_hx, zero, zero};
}

} // namespace rnn

// ---------------------------------------------------------------------------
// Sequence runner: layers of (LSTMCell / GRUCell / SimpleCell) driving the
// fused-cell primitives above, one sequence-wide input-side GEMM per
// layer+direction and a per-timestep hidden-side GEMM + cell update.
// ---------------------------------------------------------------------------

namespace {

inline Tensor rnn_row_gates(const Tensor& x2d, const Tensor& w,
                            const Tensor& b, int64_t t, int64_t N) {
    // Gates for timestep t: mm(x[t], w^T) + b  ((N,G)).
    Tensor g = x2d.narrow(0, t * N, N).mm(w.t());
    if (b.numel() > 0) g = g + b;
    return g;
}

// The rnn autograd wrapper attaches the RNN backward nodes to the outputs and
// those replay the forward themselves (RNNBackward.h); the graph the per-op
// wrappers would build inside rnn_cuda_impl is never consumed.  Suppress it.
struct RnnForwardNoGrad {
    RnnForwardNoGrad() : prev_(GradMode::is_enabled()) { GradMode::set_enabled(false); }
    ~RnnForwardNoGrad() { GradMode::set_enabled(prev_); }
    bool prev_;
};

static std::tuple<Tensor, Tensor, Tensor> rnn_cuda_impl(
    int kind,  // 0=lstm, 1=gru, 2=tanh, 3=relu
    const Tensor& input, const std::vector<Tensor>& hx,
    const std::vector<Tensor>& params, bool has_biases, int64_t num_layers,
    bool bidirectional, bool batch_first,
    double dropout_p, bool training) {
    RnnForwardNoGrad no_grad_guard;
    using tensorplay::cuda::rnn::fused_gru_cell;
    using tensorplay::cuda::rnn::fused_lstm_cell;

    Tensor x = batch_first ? input.transpose(0, 1).contiguous() : input.contiguous();
    const int64_t T = x.size(0), N = x.size(1);
    if (hx.empty()) TP_THROW(RuntimeError, "rnn: hx required");
    // two hidden states"); an undersized hx would read past the vector.
    if (kind == 0 && hx.size() != 2) TP_THROW(RuntimeError, "lstm expects two hidden states");
    const int64_t L = num_layers;
    const int64_t dirs = bidirectional ? 2 : 1;
    const int64_t H = hx[0].size(-1);
    const DType dt = x.dtype();

    Tensor hn_out = Tensor::zeros({L * dirs, N, H}, hx[0].dtype(), x.device());
    Tensor cn_out = kind == 0
        ? Tensor::zeros({L * dirs, N, H}, hx[0].dtype(), x.device())
        : Tensor();

    size_t ppi = 0;  // params cursor: per layer/direction w_ih, w_hh[, b_ih, b_hh]
    auto param_at = [&](void) -> const Tensor& {
        return params.at(ppi++);
    };

    for (int64_t layer = 0; layer < L; ++layer) {
        // Per-direction outputs written through Tensor::select views (narrow
        // must not be used as an assignment target), concatenated along the
        // feature dim afterwards.
        std::vector<Tensor> dir_outs;
        for (int64_t dir = 0; dir < dirs; ++dir) {
            const int64_t state_idx = layer * dirs + dir;
            Tensor h = hx[0].select(0, state_idx).contiguous();
            Tensor c = kind == 0 ? hx[1].select(0, state_idx).contiguous() : h;

            const Tensor& w_ih = param_at();
            const Tensor& w_hh = param_at();
            Tensor b_ih, b_hh;
            if (has_biases) {
                b_ih = param_at();
                b_hh = param_at();
                if (!(b_ih.numel() > 0)) b_ih = Tensor();
                if (!(b_hh.numel() > 0)) b_hh = Tensor();
            }

            // Input-side gates for the whole sequence in one GEMM:
            // (T*N, feat) @ (feat, G)^T -> (T*N, G).
            Tensor x2d = x.reshape({T * N, x.size(2)});
            Tensor in_gates = x2d.mm(w_ih.t());
            if (b_ih.numel() > 0) in_gates = in_gates + b_ih;
            const int64_t G = in_gates.size(1);

            Tensor dir_out = Tensor::zeros({T, N, H}, dt, x.device());
            for (int64_t t = 0; t < T; ++t) {
                const int64_t tt = dir == 0 ? t : (T - 1 - t);
                Tensor ig_row = in_gates.narrow(0, tt * N, N);
                Tensor hg_row = h.mm(w_hh.t());
                if (b_hh.numel() > 0) hg_row = hg_row + b_hh;

                if (kind == 0) {
                    auto r = fused_lstm_cell(ig_row, hg_row, c, Tensor(), Tensor());
                    h = std::get<0>(r);
                    c = std::get<1>(r);
                } else if (kind == 1) {
                    auto r = fused_gru_cell(ig_row, hg_row, h, Tensor(), Tensor());
                    h = std::get<0>(r);
                } else {
                    Tensor gates = ig_row + hg_row;
                    h = (kind == 2) ? gates.tanh() : gates.relu();
                }

                dir_out.select(0, tt).copy_(h);
                hn_out.select(0, state_idx).copy_(h);
                if (kind == 0) cn_out.select(0, state_idx).copy_(c);
            }
            dir_outs.push_back(dir_out);
        }
        Tensor layer_out;
        if (dirs == 1) {
            layer_out = dir_outs[0];
        } else {
            layer_out = Tensor::cat({dir_outs[0], dir_outs[1]}, 2);
        }
        x = layer_out;
        // Dropout applies to every layer's output except the last one.
        if (dropout_p != 0 && training && layer < L - 1) {
            x = ::tensorplay::detail::redispatch_dropout_function(x, dropout_p, true);
        }
    }
    Tensor y = batch_first ? x.transpose(0, 1).contiguous() : x;
    return {y, hn_out, cn_out};
}

}  // namespace

std::tuple<Tensor, Tensor, Tensor> lstm_cuda(const Tensor& input,
                                             const std::vector<Tensor>& hx,
                                             const std::vector<Tensor>& params,
                                             bool has_biases, int64_t num_layers,
                                             double dropout_p, bool training,
                                             bool bidirectional, bool batch_first) {
    return rnn_cuda_impl(0, input, hx, params, has_biases, num_layers,
                         bidirectional, batch_first, dropout_p, training);
}
std::tuple<Tensor, Tensor> gru_cuda(const Tensor& input, const std::vector<Tensor>& hx,
                                    const std::vector<Tensor>& params, bool has_biases,
                                    int64_t num_layers, double dropout_p, bool training,
                                    bool bidirectional, bool batch_first) {
    auto r = rnn_cuda_impl(1, input, hx, params, has_biases, num_layers,
                           bidirectional, batch_first, dropout_p, training);
    return {std::get<0>(r), std::get<1>(r)};
}
std::tuple<Tensor, Tensor> rnn_relu_cuda(const Tensor& input, const std::vector<Tensor>& hx,
                                         const std::vector<Tensor>& params, bool has_biases,
                                         int64_t num_layers, double dropout_p, bool training,
                                         bool bidirectional, bool batch_first) {
    auto r = rnn_cuda_impl(3, input, hx, params, has_biases, num_layers,
                           bidirectional, batch_first, dropout_p, training);
    return {std::get<0>(r), std::get<1>(r)};
}
std::tuple<Tensor, Tensor> rnn_tanh_cuda(const Tensor& input, const std::vector<Tensor>& hx,
                                         const std::vector<Tensor>& params, bool has_biases,
                                         int64_t num_layers, double dropout_p, bool training,
                                         bool bidirectional, bool batch_first) {
    auto r = rnn_cuda_impl(2, input, hx, params, has_biases, num_layers,
                           bidirectional, batch_first, dropout_p, training);
    return {std::get<0>(r), std::get<1>(r)};
}

TENSORPLAY_LIBRARY_IMPL(CUDA, RnnSequence) {
    m.impl("lstm", lstm_cuda);
    m.impl("gru", gru_cuda);
    m.impl("rnn_relu", rnn_relu_cuda);
    m.impl("rnn_tanh", rnn_tanh_cuda);
}

namespace {


Tensor gate_slice(const Tensor& gates, int64_t idx, int64_t total_gates) {
    const int64_t span = gates.size(-1) / total_gates;
    return gates.narrow(-1, idx * span, span);
}


std::tuple<Tensor, Tensor> interop__thnn_fused_gru_cell_cuda(
        const Tensor& input_gates, const Tensor& hidden_gates,
        const Tensor& hx, const std::optional<Tensor>& input_bias,
        const std::optional<Tensor>& hidden_bias) {
    Tensor gi = input_gates;
    Tensor gh = hidden_gates;
    if (input_bias.has_value() && input_bias->defined()) {
        gi = ops::add(gi, *input_bias);
    }
    if (hidden_bias.has_value() && hidden_bias->defined()) {
        gh = ops::add(gh, *hidden_bias);
    }
    Tensor r = ops::sigmoid(ops::add(gate_slice(gi, 0, 3), gate_slice(gh, 0, 3)));
    Tensor z = ops::sigmoid(ops::add(gate_slice(gi, 1, 3), gate_slice(gh, 1, 3)));
    Tensor n = ops::tanh(ops::add(gate_slice(gi, 2, 3),
                                  ops::mul(r, gate_slice(gh, 2, 3))));
    Tensor hy = ops::add(n, ops::mul(z, ops::sub(hx, n)));
    // Workspace layout: [r, z, n, hx, gh_n + b_n].
    Tensor workspace = ops::cat(
        {r, z, n, hx, gate_slice(gh, 2, 3)}, -1);
    return std::make_tuple(hy, workspace);
}


// ---------------------------------------------------------------------------
// _thnn_fused_lstm_cell: gates run along the last dimension in
// [input, forget, cell, output] order.
//
// cy = f * cx + i * c, hy = o * tanh(cy).  The workspace saves [i, f, c, o].
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor, Tensor> interop__thnn_fused_lstm_cell_cuda(
        const Tensor& input_gates, const Tensor& hidden_gates,
        const Tensor& cx, const std::optional<Tensor>& input_bias,
        const std::optional<Tensor>& hidden_bias) {
    Tensor gi = input_gates;
    Tensor gh = hidden_gates;
    if (input_bias.has_value() && input_bias->defined()) {
        gi = ops::add(gi, *input_bias);
    }
    if (hidden_bias.has_value() && hidden_bias->defined()) {
        gh = ops::add(gh, *hidden_bias);
    }
    Tensor i = ops::sigmoid(ops::add(gate_slice(gi, 0, 4), gate_slice(gh, 0, 4)));
    Tensor f = ops::sigmoid(ops::add(gate_slice(gi, 1, 4), gate_slice(gh, 1, 4)));
    Tensor c = ops::tanh(ops::add(gate_slice(gi, 2, 4), gate_slice(gh, 2, 4)));
    Tensor o = ops::sigmoid(ops::add(gate_slice(gi, 3, 4), gate_slice(gh, 3, 4)));
    Tensor cy = ops::add(ops::mul(f, cx), ops::mul(i, c));
    Tensor hy = ops::mul(o, ops::tanh(cy));
    Tensor workspace = ops::cat({i, f, c, o}, -1);
    return std::make_tuple(hy, cy, workspace);
}


std::tuple<Tensor, Tensor, Tensor> interop__thnn_fused_lstm_cell_backward_impl_cuda(
        const std::optional<Tensor>& grad_hy,
        const std::optional<Tensor>& grad_cy, const Tensor& cx,
        const Tensor& cy, const Tensor& workspace, bool has_bias) {
    const bool has_ghy = grad_hy.has_value() && grad_hy->defined();
    const bool has_gcy = grad_cy.has_value() && grad_cy->defined();
    if (!has_ghy && !has_gcy) {
        return std::tuple<Tensor, Tensor, Tensor>();
    }
    Tensor i = gate_slice(workspace, 0, 4);
    Tensor f = gate_slice(workspace, 1, 4);
    Tensor c = gate_slice(workspace, 2, 4);
    Tensor o = gate_slice(workspace, 3, 4);
    Tensor tanh_cy = ops::tanh(cy);
    Tensor go = has_ghy ? *grad_hy : Tensor();
    Tensor goc = has_gcy ? *grad_cy : Tensor();
    // hy = o * tanh(cy); cy = f * cx + i * c.
    Tensor gog = has_ghy ? ops::mul(go, tanh_cy) : Tensor();
    // gcx accumulates the total cell gradient before the forget gate.
    Tensor gcx;
    if (has_ghy) {
        Tensor tanh_cy_sq = ops::mul(tanh_cy, tanh_cy);
        gcx = ops::add(ops::mul(go, ops::mul(o,
                                  ops::sub(ops::ones_like(tanh_cy_sq),
                                           tanh_cy_sq))),
                       has_gcy ? goc : ops::zeros_like(cy));
    } else {
        gcx = goc;
    }
    Tensor gig = ops::mul(gcx, c);
    Tensor gfg = ops::mul(gcx, cx);
    Tensor gcg = ops::mul(gcx, i);
    Tensor grad_cx = ops::mul(gcx, f);
    gig = ops::mul(gig, ops::mul(ops::sub(ops::ones_like(i), i), i));
    gfg = ops::mul(gfg, ops::mul(ops::sub(ops::ones_like(f), f), f));
    gcg = ops::mul(gcg, ops::sub(ops::ones_like(ops::mul(c, c)), ops::mul(c, c)));
    if (has_ghy) {
        gog = ops::mul(gog, ops::mul(ops::sub(ops::ones_like(o), o), o));
    } else {
        gog = ops::zeros_like(o);
    }
    Tensor grad_gates = ops::cat({gig, gfg, gcg, gog}, -1);
    Tensor grad_bias;
    if (has_bias) {
        std::vector<int64_t> reduce_dims;
        for (int64_t d = 0; d < grad_gates.dim() - 1; ++d) {
            reduce_dims.push_back(d);
        }
        grad_bias = ops::sum(grad_gates, reduce_dims, false);
    }
    return std::make_tuple(grad_gates, grad_cx, grad_bias);
}


std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
interop__thnn_fused_gru_cell_backward_cuda(const Tensor& grad_hy,
                                           const Tensor& workspace,
                                           bool has_bias) {
    // Workspace slices (per hidden unit): r, z, n, hx, hgn.
    Tensor r = gate_slice(workspace, 0, 5);
    Tensor z = gate_slice(workspace, 1, 5);
    Tensor n = gate_slice(workspace, 2, 5);
    Tensor hx = gate_slice(workspace, 3, 5);
    Tensor hgn = gate_slice(workspace, 4, 5);
    Tensor go = grad_hy;
    // hy = n + z * (hx - n):
    //   dz = go * (hx - n) * z * (1 - z)
    //   dn = go * (1 - z) * (1 - n^2)
    //   dhx = go * z
    //   dr = dn * hgn * r * (1 - r)
    // gate grads: input [dr, dz, dn]; hidden [dr, dz, r * dn].
    Tensor sig_z = ops::mul(z, ops::sub(ops::ones_like(z), z));
    Tensor sig_r = ops::mul(r, ops::sub(ops::ones_like(r), r));
    Tensor gz = ops::mul(ops::mul(go, ops::sub(hx, n)), sig_z);
    Tensor gn = ops::mul(ops::mul(go, ops::sub(ops::ones_like(z), z)),
                         ops::sub(ops::ones_like(ops::mul(n, n)), ops::mul(n, n)));
    Tensor gr = ops::mul(ops::mul(gn, hgn), sig_r);
    Tensor grad_hx = ops::mul(go, z);
    Tensor grad_input_gates = ops::cat({gr, gz, gn}, -1);
    Tensor grad_hidden_gates = ops::cat({gr, gz, ops::mul(r, gn)}, -1);
    Tensor grad_input_bias;
    Tensor grad_hidden_bias;
    if (has_bias) {
        // Bias gradients sum over every batch axis.
        std::vector<int64_t> reduce_dims;
        for (int64_t d = 0; d < grad_input_gates.dim() - 1; ++d) {
            reduce_dims.push_back(d);
        }
        grad_input_bias = ops::sum(grad_input_gates, reduce_dims, false);
        grad_hidden_bias = ops::sum(grad_hidden_gates, reduce_dims, false);
    }
    return std::make_tuple(grad_input_gates, grad_hidden_gates, grad_hx,
                           grad_input_bias, grad_hidden_bias);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, RnnInterop) {
    // fused rnn cells
    m.impl("_thnn_fused_gru_cell", interop__thnn_fused_gru_cell_cuda);
    m.impl("_thnn_fused_gru_cell_backward", interop__thnn_fused_gru_cell_backward_cuda);
    m.impl("_thnn_fused_lstm_cell", interop__thnn_fused_lstm_cell_cuda);
    m.impl("_thnn_fused_lstm_cell_backward_impl", interop__thnn_fused_lstm_cell_backward_impl_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
