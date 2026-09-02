#pragma once
// Fused CPU RNN cells -- the CUDA fused-cell kernels' CPU twin:
// forward/backward LSTM and GRU cell updates in one elementwise pass
//
// updates; TensorPlay's decomposed loop instead dispatched ~15-20 ops (each a
// dispatch + allocation + potential parallel region) per timestep.  These
// kernels collapse the same math into one parallel_for over N*H elements, which
// stays below GRAIN_SIZE for typical shapes and therefore runs serially on the
// calling thread -- the low-overhead regime the decomposition was missing.
//
// Gate math is bit-for-bit the decomposed rnn_impl / RNNBackward.h formulas
// (biases folded identically), so numerics are unchanged.  fp16/bf16
// accumulate in float (opmath), matching the CUDA AccTraits.
//
// Shared by rnn_impl (forward) and RNNBackward.h (replay + sweep).  Callers
// gate on dtype (fp32/fp64) and, for the backward, on GradMode (create_graph
// must keep the recordable decomposed path).

#include "Tensor.h"
#include "Parallel.h"
#include "Half.h"
#include "BFloat16.h"
#include "DType.h"

#include <cmath>
#include <tuple>

namespace tensorplay {
namespace rnn_cpu {

template <typename T> struct Acc { using type = T; };
template <> struct Acc<Half> { using type = float; };
template <> struct Acc<BFloat16> { using type = float; };

template <typename M> inline M sig(M x) { return M(1) / (M(1) + std::exp(-x)); }

// --- forward cells (used by rnn_impl) ---------------------------------------

// LSTM: ig/hg are (N,4H) with b_ih/b_hh already folded in by the caller.
template <typename T>
inline std::tuple<Tensor, Tensor> lstm_cell(const Tensor& ig, const Tensor& hg,
                                            const Tensor& cx) {
    using M = typename Acc<T>::type;
    const int64_t N = ig.size(0), H = ig.size(1) / 4;
    Tensor hy = Tensor::empty({N, H}, ig.dtype(), ig.device());
    Tensor cy = Tensor::empty({N, H}, ig.dtype(), ig.device());
    const T* igp = ig.data_ptr<T>();
    const T* hgp = hg.data_ptr<T>();
    const T* cxp = cx.data_ptr<T>();
    T* hyp = hy.data_ptr<T>();
    T* cyp = cy.data_ptr<T>();
    parallel::parallel_for(0, N * H, parallel::GRAIN_SIZE, [&](int64_t b, int64_t e) {
        for (int64_t li = b; li < e; ++li) {
            const int64_t off = (li / H) * 4 * H + li % H;
            const M i = sig<M>(M(igp[off]) + M(hgp[off]));
            const M f = sig<M>(M(igp[off + H]) + M(hgp[off + H]));
            const M g = std::tanh(M(igp[off + 2 * H]) + M(hgp[off + 2 * H]));
            const M o = sig<M>(M(igp[off + 3 * H]) + M(hgp[off + 3 * H]));
            const M cv = f * M(cxp[li]) + i * g;
            cyp[li] = T(cv);
            hyp[li] = T(o * std::tanh(cv));
        }
    });
    return {hy, cy};
}

// GRU: ig is (N,3H) with b_ih folded in; hg is (N,3H) WITHOUT b_hh (caller
// adds the three b_hh segments per-gate, reproduced here).
template <typename T>
inline Tensor gru_cell(const Tensor& ig, const Tensor& hg, const Tensor& hx,
                       const Tensor& b_hh) {
    using M = typename Acc<T>::type;
    const int64_t N = ig.size(0), H = ig.size(1) / 3;
    Tensor hy = Tensor::empty({N, H}, ig.dtype(), ig.device());
    const T* igp = ig.data_ptr<T>();
    const T* hgp = hg.data_ptr<T>();
    const T* hxp = hx.data_ptr<T>();
    const T* bp = b_hh.defined() ? b_hh.data_ptr<T>() : nullptr;
    T* hyp = hy.data_ptr<T>();
    parallel::parallel_for(0, N * H, parallel::GRAIN_SIZE, [&](int64_t b, int64_t e) {
        for (int64_t li = b; li < e; ++li) {
            const int64_t col = li % H;
            const int64_t off = (li / H) * 3 * H + col;
            const M br = bp ? M(bp[col]) : M(0);
            const M bz = bp ? M(bp[H + col]) : M(0);
            const M bn = bp ? M(bp[2 * H + col]) : M(0);
            const M r = sig<M>(M(igp[off]) + M(hgp[off]) + br);
            const M z = sig<M>(M(igp[off + H]) + M(hgp[off + H]) + bz);
            const M hn = M(hgp[off + 2 * H]) + bn;
            const M n = std::tanh(M(igp[off + 2 * H]) + r * hn);
            hyp[li] = T(n + z * (M(hxp[li]) - n));
        }
    });
    return hy;
}

// --- replay cells (forward + gate activations for the backward sweep) -------

// LSTM replay: returns (hy, cy, gates) with gates=(N,4H)=[i,f,g,o].
template <typename T>
inline std::tuple<Tensor, Tensor, Tensor> lstm_cell_replay(
        const Tensor& ig, const Tensor& hg, const Tensor& cx) {
    using M = typename Acc<T>::type;
    const int64_t N = ig.size(0), H = ig.size(1) / 4;
    Tensor hy = Tensor::empty({N, H}, ig.dtype(), ig.device());
    Tensor cy = Tensor::empty({N, H}, ig.dtype(), ig.device());
    Tensor gates = Tensor::empty({N, 4 * H}, ig.dtype(), ig.device());
    const T* igp = ig.data_ptr<T>();
    const T* hgp = hg.data_ptr<T>();
    const T* cxp = cx.data_ptr<T>();
    T* hyp = hy.data_ptr<T>();
    T* cyp = cy.data_ptr<T>();
    T* gp = gates.data_ptr<T>();
    parallel::parallel_for(0, N * H, parallel::GRAIN_SIZE, [&](int64_t b, int64_t e) {
        for (int64_t li = b; li < e; ++li) {
            const int64_t off = (li / H) * 4 * H + li % H;
            const M i = sig<M>(M(igp[off]) + M(hgp[off]));
            const M f = sig<M>(M(igp[off + H]) + M(hgp[off + H]));
            const M g = std::tanh(M(igp[off + 2 * H]) + M(hgp[off + 2 * H]));
            const M o = sig<M>(M(igp[off + 3 * H]) + M(hgp[off + 3 * H]));
            const M cv = f * M(cxp[li]) + i * g;
            cyp[li] = T(cv);
            hyp[li] = T(o * std::tanh(cv));
            gp[off] = T(i);
            gp[off + H] = T(f);
            gp[off + 2 * H] = T(g);
            gp[off + 3 * H] = T(o);
        }
    });
    return {hy, cy, gates};
}

// GRU replay: returns (hy, gates, hn_lin) with gates=(N,3H)=[r,z,n].
template <typename T>
inline std::tuple<Tensor, Tensor, Tensor> gru_cell_replay(
        const Tensor& ig, const Tensor& hg, const Tensor& hx, const Tensor& b_hh) {
    using M = typename Acc<T>::type;
    const int64_t N = ig.size(0), H = ig.size(1) / 3;
    Tensor hy = Tensor::empty({N, H}, ig.dtype(), ig.device());
    Tensor gates = Tensor::empty({N, 3 * H}, ig.dtype(), ig.device());
    Tensor hn_lin = Tensor::empty({N, H}, ig.dtype(), ig.device());
    const T* igp = ig.data_ptr<T>();
    const T* hgp = hg.data_ptr<T>();
    const T* hxp = hx.data_ptr<T>();
    const T* bp = b_hh.defined() ? b_hh.data_ptr<T>() : nullptr;
    T* hyp = hy.data_ptr<T>();
    T* gp = gates.data_ptr<T>();
    T* hnp = hn_lin.data_ptr<T>();
    parallel::parallel_for(0, N * H, parallel::GRAIN_SIZE, [&](int64_t b, int64_t e) {
        for (int64_t li = b; li < e; ++li) {
            const int64_t col = li % H;
            const int64_t off = (li / H) * 3 * H + col;
            const M br = bp ? M(bp[col]) : M(0);
            const M bz = bp ? M(bp[H + col]) : M(0);
            const M bn = bp ? M(bp[2 * H + col]) : M(0);
            const M r = sig<M>(M(igp[off]) + M(hgp[off]) + br);
            const M z = sig<M>(M(igp[off + H]) + M(hgp[off + H]) + bz);
            const M hn = M(hgp[off + 2 * H]) + bn;
            const M n = std::tanh(M(igp[off + 2 * H]) + r * hn);
            hyp[li] = T(n + z * (M(hxp[li]) - n));
            gp[off] = T(r);
            gp[off + H] = T(z);
            gp[off + 2 * H] = T(n);
            hnp[li] = T(hn);
        }
    });
    return {hy, gates, hn_lin};
}

// --- backward cells (consume replay gates, produce gate gradients) ----------

// LSTM backward: returns (dpre (N,4H), dc_next (N,H)).
// Matches RNNBackward.h decomposed sweep / CUDA lstm_cell_backward_kernel.
template <typename T>
inline std::tuple<Tensor, Tensor> lstm_cell_backward(
        const Tensor& dh, const Tensor& dc_in, const Tensor& c,
        const Tensor& c_prev, const Tensor& gates) {
    using M = typename Acc<T>::type;
    const int64_t N = dh.size(0), H = dh.size(1);
    Tensor dpre = Tensor::empty({N, 4 * H}, dh.dtype(), dh.device());
    Tensor dc_next = Tensor::empty({N, H}, dh.dtype(), dh.device());
    const T* dhp = dh.data_ptr<T>();
    const T* dcp = dc_in.data_ptr<T>();
    const T* cp = c.data_ptr<T>();
    const T* cpp = c_prev.data_ptr<T>();
    const T* gp = gates.data_ptr<T>();
    T* dp = dpre.data_ptr<T>();
    T* dcn = dc_next.data_ptr<T>();
    parallel::parallel_for(0, N * H, parallel::GRAIN_SIZE, [&](int64_t b, int64_t e) {
        for (int64_t li = b; li < e; ++li) {
            const int64_t off = (li / H) * 4 * H + li % H;
            const M i = M(gp[off]);
            const M f = M(gp[off + H]);
            const M g = M(gp[off + 2 * H]);
            const M o = M(gp[off + 3 * H]);
            const M cxv = M(cpp[li]);
            const M cyv = M(cp[li]);
            const M go = M(dhp[li]);
            const M goc = M(dcp[li]);
            const M tanh_c = std::tanh(cyv);
            const M dc = go * o * (M(1) - tanh_c * tanh_c) + goc;
            const M gig = dc * g;
            const M gfg = dc * cxv;
            const M gcg = dc * i;
            const M gog = go * tanh_c;
            dcn[li] = T(dc * f);
            dp[off] = T(gig * (M(1) - i) * i);
            dp[off + H] = T(gfg * (M(1) - f) * f);
            dp[off + 2 * H] = T(gcg * (M(1) - g * g));
            dp[off + 3 * H] = T(gog * (M(1) - o) * o);
        }
    });
    return {dpre, dc_next};
}

// GRU backward: returns (dpre (N,3H), dhid (N,3H), ghx (N,H)).
// Matches RNNBackward.h decomposed sweep / CUDA gru_cell_backward_kernel.
template <typename T>
inline std::tuple<Tensor, Tensor, Tensor> gru_cell_backward(
        const Tensor& dh, const Tensor& h_prev, const Tensor& gates,
        const Tensor& hn_lin) {
    using M = typename Acc<T>::type;
    const int64_t N = dh.size(0), H = dh.size(1);
    Tensor dpre = Tensor::empty({N, 3 * H}, dh.dtype(), dh.device());
    Tensor dhid = Tensor::empty({N, 3 * H}, dh.dtype(), dh.device());
    Tensor ghx = Tensor::empty({N, H}, dh.dtype(), dh.device());
    const T* dhp = dh.data_ptr<T>();
    const T* hpp = h_prev.data_ptr<T>();
    const T* gp = gates.data_ptr<T>();
    const T* hnp = hn_lin.data_ptr<T>();
    T* dp = dpre.data_ptr<T>();
    T* dhidp = dhid.data_ptr<T>();
    T* gh = ghx.data_ptr<T>();
    parallel::parallel_for(0, N * H, parallel::GRAIN_SIZE, [&](int64_t b, int64_t e) {
        for (int64_t li = b; li < e; ++li) {
            const int64_t col = li % H;
            const int64_t off = (li / H) * 3 * H + col;
            const M r = M(gp[off]);
            const M z = M(gp[off + H]);
            const M n = M(gp[off + 2 * H]);
            const M hn = M(hnp[li]);
            const M hxv = M(hpp[li]);
            const M go = M(dhp[li]);
            const M gig = go * (hxv - n) * (M(1) - z) * z;
            const M ghxv = go * z;
            const M gin = go * (M(1) - z) * (M(1) - n * n);
            const M ghn = gin * r;
            const M grg = gin * hn * (M(1) - r) * r;
            dp[off] = T(grg);
            dp[off + H] = T(gig);
            dp[off + 2 * H] = T(gin);
            dhidp[off] = T(grg);
            dhidp[off + H] = T(gig);
            dhidp[off + 2 * H] = T(ghn);
            gh[li] = T(ghxv);
        }
    });
    return {dpre, dhid, ghx};
}

} // namespace rnn_cpu
} // namespace tensorplay
