// Sequence-level RNN forward (lstm/gru/rnn_relu/rnn_tanh).
//
// Layout contract: params carry [w_ih, w_hh, b_ih?, b_hh?] per layer and
// direction; LSTM gates run in i, f, g, o order and GRU in r, z, n order.
// The CPU fast path runs the whole sequence through one fused oneDNN
// primitive per layer+direction; the decomposed loop below it covers the
// remaining dtypes and builds without oneDNN.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Utils.h"
#include "Exception.h"
#include "Parallel.h"
#include "GradMode.h"
#include "RNNCpuKernels.h"
#include "RNNOneDNN.h"
#include "OneDNNContext.h"
#include "tensorplay/ops/TensorRedispatchGenerated.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <optional>
#include <tuple>
#include <utility>

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

// RNN cells ([w_ih, w_hh, b_ih?, b_hh?] per layer and direction)
// ---------------------------------------------------------------------------

// cat_kernel/stack_kernel are defined in ViewKernels.cpp at tensorplay::cpu
// scope (no header).  Declare them here (in tensorplay::cpu, outside the
// anonymous namespace below) so the RNN code links against those definitions.
Tensor cat_kernel(const std::vector<Tensor>& tensors, int64_t dim);
Tensor stack_kernel(const std::vector<Tensor>& tensors, int64_t dim);

namespace {

// Fetch one parameter from the params list with bounds checking.
const Tensor& param_at(const std::vector<Tensor>& params, size_t idx, bool has_biases) {
    (void)has_biases;
    if (idx >= params.size()) TP_THROW(RuntimeError, "rnn: missing parameter ", idx);
    return params[idx];
}

// The rnn autograd wrapper attaches RnnTanhBackward & co. to the outputs and
// those nodes replay the forward themselves (RNNBackward.h); the graph the
// per-op wrappers would build inside rnn_impl is never consumed.  Suppress it
// so the sequence loop does not pay node/saved-variable overhead per op.
struct RnnForwardNoGrad {
    RnnForwardNoGrad() : prev_(GradMode::is_enabled()) { GradMode::set_enabled(false); }
    ~RnnForwardNoGrad() { GradMode::set_enabled(prev_); }
    bool prev_;
};

#ifdef USE_ONEDNN
// ---------------------------------------------------------------------------
// LSTM) and bias = b_ih + b_hh (_shuffle_bias).  Weight dims follow oneDNN
// rnn_pd.hpp: weights_layer=[L,D,SLC,G,DHC], weights_iter=[L,D,SIC,G,DHC],
// view (no physical repack) and let oneDNN reorder into its packed layout.
// The forward uses forward_inference (no workspace).  The fused oneDNN
// backward (onednn_lstm_backward below) re-runs forward_training to
// regenerate its own workspace, so none is threaded through autograd.

namespace onednn_lstm_detail {

using dnnl::memory;
using dnnl::prop_kind;
using dnnl::rnn_direction;

struct VecKeyHash {
    size_t operator()(const std::vector<int64_t>& v) const {
        size_t h = 1469598103934665603ULL;
        for (auto x : v) { h ^= (size_t)x; h *= 1099511628211ULL; }
        return h;
    }
};

inline memory::desc w_layer_view_md(int64_t in, int64_t H) {
    // (4H, in) row-major -> [1,1,in,4,H]; elem (i,g,o) at w[(g*H+o)*in + i].
    return memory::desc({1, 1, in, 4, H}, memory::data_type::f32,
                        {4 * H * in, 4 * H * in, 1, H * in, in});
}
inline memory::desc w_iter_view_md(int64_t H) {
    // (4H, H) row-major -> [1,1,H,4,H]; elem (sic,g,o) at w[(g*H+o)*H + sic].
    return memory::desc({1, 1, H, 4, H}, memory::data_type::f32,
                        {4 * H * H, 4 * H * H, 1, H * H, H});
}
inline memory::desc bias_view_md(int64_t H) {
    // (4H,) -> [1,1,4,H]; elem (g,o) at b[g*H+o].
    return memory::desc({1, 1, 4, H}, memory::data_type::f32,
                        {4 * H, 4 * H, H, 1});
}

struct PrimEntry {
    std::unique_ptr<dnnl::lstm_forward::primitive_desc> pd;
    std::shared_ptr<dnnl::lstm_forward> prim;
};
static std::unordered_map<std::vector<int64_t>, PrimEntry, VecKeyHash>* g_lstm_cache =
    nullptr;

const dnnl::lstm_forward& cached_lstm_prim(
        dnnl::engine& eng, int64_t T, int64_t N, int64_t F, int64_t H,
        bool reverse, bool has_bias, const PrimEntry** out_entry) {
    if (!g_lstm_cache)
        g_lstm_cache = new std::unordered_map<std::vector<int64_t>, PrimEntry,
                                              VecKeyHash>();
    std::vector<int64_t> key = {T, N, F, H, reverse ? 1 : 0, has_bias ? 1 : 0};
    auto it = g_lstm_cache->find(key);
    if (it != g_lstm_cache->end()) {
        if (out_entry) *out_entry = &it->second;
        return *it->second.prim;
    }
    if (g_lstm_cache->size() >= 256) g_lstm_cache->clear();
    const memory::data_type dt = memory::data_type::f32;
    auto src_layer = memory::desc({T, N, F}, dt, memory::format_tag::tnc);
    auto src_iter  = memory::desc({1, 1, N, H}, dt, memory::format_tag::ldnc);
    auto dst_layer = memory::desc({T, N, H}, dt, memory::format_tag::tnc);
    auto dst_iter  = memory::desc({1, 1, N, H}, dt, memory::format_tag::ldnc);
    auto w_layer   = memory::desc({1, 1, F, 4, H}, dt, memory::format_tag::any);
    auto w_iter    = memory::desc({1, 1, H, 4, H}, dt, memory::format_tag::any);
    auto bias      = has_bias ? memory::desc({1, 1, 4, H}, dt, memory::format_tag::any)
                              : memory::desc();
    auto dir = reverse ? rnn_direction::unidirectional_right2left
                       : rnn_direction::unidirectional_left2right;
    auto pd = std::make_unique<dnnl::lstm_forward::primitive_desc>(
        eng, prop_kind::forward_inference, dir, src_layer, src_iter, src_iter,
        w_layer, w_iter, memory::desc(), memory::desc(), bias, dst_layer,
        dst_iter, dst_iter);
    auto prim = std::make_shared<dnnl::lstm_forward>(*pd);
    PrimEntry entry{std::move(pd), std::move(prim)};
    auto ins = g_lstm_cache->emplace(std::move(key), std::move(entry));
    const PrimEntry* eptr = &ins.first->second;
    if (out_entry) *out_entry = eptr;
    return *eptr->prim;
}

// Reorder a user tensor (described by user_md, wrapping data_ptr) into the
// primitive's chosen layout target_md; if identical, wrap in place.
inline memory prepare_weight(dnnl::engine& eng, dnnl::stream& stream,
                             const Tensor& t, const memory::desc& user_md,
                             const memory::desc& target_md) {
    if (user_md == target_md) return memory(target_md, eng, t.data_ptr());
    memory src(user_md, eng, t.data_ptr());
    memory dst(target_md, eng);
    dnnl::reorder(src, dst).execute(stream, src, dst);
    return dst;
}

// Inverse of prepare_weight: reorder a primitive-produced diff (packed layout
// src_md) back into a freshly allocated user-layout tensor described by
// user_md.  Returns the destination tensor owning the data.
inline Tensor unpack_diff(dnnl::engine& eng, dnnl::stream& stream,
                          memory& src_mem, const memory::desc& user_md,
                          const std::vector<int64_t>& user_shape) {
    Tensor dst_t = Tensor::empty(user_shape, DType::Float32, Device(DeviceType::CPU));
    memory dst(user_md, eng, dst_t.data_ptr());
    dnnl::reorder(src_mem, dst).execute(stream, src_mem, dst);
    return dst_t;
}

// Backward primitive bundle: a forward_training pd/prim (to regenerate the
// workspace + forward outputs during the backward replay) plus the fused
// lstm_backward pd/prim built from it.  Keyed like the forward cache.
struct BwdEntry {
    std::unique_ptr<dnnl::lstm_forward::primitive_desc> fwd_pd;
    std::shared_ptr<dnnl::lstm_forward> fwd_prim;
    std::unique_ptr<dnnl::lstm_backward::primitive_desc> bwd_pd;
    std::shared_ptr<dnnl::lstm_backward> bwd_prim;
};
static std::unordered_map<std::vector<int64_t>, BwdEntry, VecKeyHash>* g_lstm_bwd_cache =
    nullptr;

const BwdEntry& cached_lstm_bwd(
        dnnl::engine& eng, int64_t T, int64_t N, int64_t F, int64_t H,
        bool reverse, bool has_bias) {
    if (!g_lstm_bwd_cache)
        g_lstm_bwd_cache = new std::unordered_map<std::vector<int64_t>, BwdEntry,
                                                  VecKeyHash>();
    std::vector<int64_t> key = {T, N, F, H, reverse ? 1 : 0, has_bias ? 1 : 0};
    auto it = g_lstm_bwd_cache->find(key);
    if (it != g_lstm_bwd_cache->end()) return it->second;
    if (g_lstm_bwd_cache->size() >= 256) g_lstm_bwd_cache->clear();

    const memory::data_type dt = memory::data_type::f32;
    auto src_layer = memory::desc({T, N, F}, dt, memory::format_tag::tnc);
    auto src_iter  = memory::desc({1, 1, N, H}, dt, memory::format_tag::ldnc);
    auto dst_layer = memory::desc({T, N, H}, dt, memory::format_tag::tnc);
    auto dst_iter  = memory::desc({1, 1, N, H}, dt, memory::format_tag::ldnc);
    auto w_layer   = memory::desc({1, 1, F, 4, H}, dt, memory::format_tag::any);
    auto w_iter    = memory::desc({1, 1, H, 4, H}, dt, memory::format_tag::any);
    auto bias      = has_bias ? memory::desc({1, 1, 4, H}, dt, memory::format_tag::any)
                              : memory::desc();
    auto dir = reverse ? rnn_direction::unidirectional_right2left
                       : rnn_direction::unidirectional_left2right;

    // forward_training pd (produces a workspace the backward consumes).
    auto fwd_pd = std::make_unique<dnnl::lstm_forward::primitive_desc>(
        eng, prop_kind::forward_training, dir, src_layer, src_iter, src_iter,
        w_layer, w_iter, memory::desc(), memory::desc(), bias, dst_layer,
        dst_iter, dst_iter);

    // diff descriptors: activations in plain tnc/ldnc; diff weights/bias left
    // as any so oneDNN picks its layout, reordered back to user layout after
    // execution (unpack_diff).
    auto diff_src_layer = memory::desc({T, N, F}, dt, memory::format_tag::tnc);
    auto diff_src_iter  = memory::desc({1, 1, N, H}, dt, memory::format_tag::ldnc);
    auto diff_dst_layer = memory::desc({T, N, H}, dt, memory::format_tag::tnc);
    auto diff_dst_iter  = memory::desc({1, 1, N, H}, dt, memory::format_tag::ldnc);
    auto diff_w_layer   = memory::desc({1, 1, F, 4, H}, dt, memory::format_tag::any);
    auto diff_w_iter    = memory::desc({1, 1, H, 4, H}, dt, memory::format_tag::any);
    auto diff_bias      = has_bias ? memory::desc({1, 1, 4, H}, dt, memory::format_tag::any)
                                   : memory::desc();

    auto bwd_pd = std::make_unique<dnnl::lstm_backward::primitive_desc>(
        eng, prop_kind::backward, dir, src_layer, src_iter, src_iter,
        w_layer, w_iter, bias, dst_layer, dst_iter, dst_iter,
        diff_src_layer, diff_src_iter, diff_src_iter,
        diff_w_layer, diff_w_iter, diff_bias,
        diff_dst_layer, diff_dst_iter, diff_dst_iter, *fwd_pd);

    BwdEntry entry;
    entry.fwd_pd = std::move(fwd_pd);
    entry.fwd_prim = std::make_shared<dnnl::lstm_forward>(*entry.fwd_pd);
    entry.bwd_pd = std::move(bwd_pd);
    entry.bwd_prim = std::make_shared<dnnl::lstm_backward>(*entry.bwd_pd);
    auto ins = g_lstm_bwd_cache->emplace(std::move(key), std::move(entry));
    return ins.first->second;
}

} // namespace onednn_lstm_detail

// Sequence-level oneDNN LSTM forward.  Returns nullopt to fall back to the
// decomposed loop when oneDNN is unavailable or the primitive cannot be built.
static std::optional<std::tuple<Tensor, Tensor, Tensor>> onednn_lstm_forward(
        const Tensor& input, const std::vector<Tensor>& hx,
        const std::vector<Tensor>& params, bool has_biases, int64_t num_layers,
        bool bidirectional, bool batch_first) {
    using namespace onednn_lstm_detail;
    if (input.dtype() != DType::Float32) return std::nullopt;
    if (!OneDNNContext::is_available() || !OneDNNContext::is_enabled())
        return std::nullopt;

    Tensor x = batch_first ? input.transpose(0, 1).contiguous()
                           : input.contiguous();
    const int64_t T = x.size(0), N = x.size(1);
    const int64_t L = num_layers;
    const int64_t dirs = bidirectional ? 2 : 1;
    const int64_t H = hx[0].size(-1);

    try {
        auto& eng = OneDNNContext::get_engine();
        auto& stream = OneDNNContext::get_stream();
        const memory::data_type dt = memory::data_type::f32;

        Tensor hstate = hx[0].contiguous();
        Tensor cstate = hx[1].contiguous();
        std::vector<Tensor> hn_rows(L * dirs), cn_rows(L * dirs);

        for (int64_t layer = 0; layer < L; ++layer) {
            std::vector<Tensor> dir_outs(dirs);
            for (int64_t dir = 0; dir < dirs; ++dir) {
                const int64_t si = layer * dirs + dir;
                const int64_t pbase = si * (2 + (has_biases ? 2 : 0));
                const Tensor& w_ih = params[pbase];
                const Tensor& w_hh = params[pbase + 1];
                const int64_t F = x.size(2);

                Tensor bias;
                if (has_biases) {
                    const Tensor& b_ih = params[pbase + 2];
                    const Tensor& b_hh = params[pbase + 3];
                    bias = b_ih.add(b_hh).contiguous();
                }

                const bool reverse = (dir == 1);
                const PrimEntry* entry = nullptr;
                const auto& prim = cached_lstm_prim(eng, T, N, F, H, reverse,
                                                    has_biases, &entry);
                const auto& pd = *entry->pd;

                Tensor hx_l = hstate.select(0, si).contiguous();
                Tensor cx_l = cstate.select(0, si).contiguous();

                auto src_layer_md = memory::desc({T, N, F}, dt, memory::format_tag::tnc);
                auto src_iter_md  = memory::desc({1, 1, N, H}, dt, memory::format_tag::ldnc);
                auto dst_layer_md = memory::desc({T, N, H}, dt, memory::format_tag::tnc);
                auto dst_iter_md  = memory::desc({1, 1, N, H}, dt, memory::format_tag::ldnc);

                memory w_layer_mem = prepare_weight(eng, stream, w_ih,
                    w_layer_view_md(F, H), pd.weights_layer_desc());
                memory w_iter_mem = prepare_weight(eng, stream, w_hh,
                    w_iter_view_md(H), pd.weights_iter_desc());
                memory bias_mem;
                if (has_biases)
                    bias_mem = prepare_weight(eng, stream, bias,
                        bias_view_md(H), pd.bias_desc());

                Tensor out = Tensor::empty({T, N, H}, DType::Float32, input.device());
                Tensor hy = Tensor::empty({N, H}, DType::Float32, input.device());
                Tensor cy = Tensor::empty({N, H}, DType::Float32, input.device());

                memory src_layer_mem(src_layer_md, eng, x.data_ptr());
                memory src_iter_mem(src_iter_md, eng, hx_l.data_ptr());
                memory src_iter_c_mem(src_iter_md, eng, cx_l.data_ptr());
                memory dst_layer_mem(dst_layer_md, eng, out.data_ptr());
                memory dst_iter_mem(dst_iter_md, eng, hy.data_ptr());
                memory dst_iter_c_mem(dst_iter_md, eng, cy.data_ptr());

                std::unordered_map<int, memory> args = {
                    {DNNL_ARG_SRC_LAYER, src_layer_mem},
                    {DNNL_ARG_SRC_ITER, src_iter_mem},
                    {DNNL_ARG_SRC_ITER_C, src_iter_c_mem},
                    {DNNL_ARG_WEIGHTS_LAYER, w_layer_mem},
                    {DNNL_ARG_WEIGHTS_ITER, w_iter_mem},
                    {DNNL_ARG_DST_LAYER, dst_layer_mem},
                    {DNNL_ARG_DST_ITER, dst_iter_mem},
                    {DNNL_ARG_DST_ITER_C, dst_iter_c_mem},
                };
                if (has_biases) args.insert({DNNL_ARG_BIAS, bias_mem});
                prim.execute(stream, args);
                stream.wait();

                dir_outs[dir] = out;
                hn_rows[si] = hy;
                cn_rows[si] = cy;
            }
            x = dirs == 1 ? dir_outs[0]
                          : cat_kernel({dir_outs[0], dir_outs[1]}, 2).contiguous();
        }
        Tensor y = batch_first ? x.transpose(0, 1).contiguous() : x;
        Tensor hn_out = stack_kernel(hn_rows, 0);
        Tensor cn_out = stack_kernel(cn_rows, 0);
        return std::make_tuple(y, hn_out, cn_out);
    } catch (...) {
        return std::nullopt;
    }
}
#endif // USE_ONEDNN

} // anonymous namespace

#ifdef USE_ONEDNN
// Sequence-level oneDNN LSTM backward (externally linked; declared in
// RNNOneDNN.h, called from tpx/include/RNNBackward.h).  One fused
// lstm_backward primitive per layer+direction, consuming the forward
// workspace.  The autograd node only saves (input, hx, params), so the
// workspace is regenerated by re-running the oneDNN forward
// (forward_training) per layer+direction, then the fused backward sweeps
// top layer down.  Returns nullopt to fall back to the decomposed replay
// when oneDNN is unavailable or the case is unsupported.
std::optional<std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>>>
onednn_lstm_backward(const Tensor& grad_y_in, const Tensor& grad_hy_in,
                     const Tensor& grad_cy_in, const Tensor& input,
                     const std::vector<Tensor>& hx,
                     const std::vector<Tensor>& params, bool has_biases,
                     int64_t num_layers, bool bidirectional, bool batch_first) {
    using namespace onednn_lstm_detail;
    if (input.dtype() != DType::Float32) return std::nullopt;
    if (!OneDNNContext::is_available() || !OneDNNContext::is_enabled())
        return std::nullopt;
    if (hx.size() != 2) return std::nullopt;
    if (!grad_y_in.defined()) return std::nullopt;

    Tensor x = batch_first ? input.transpose(0, 1).contiguous()
                           : input.contiguous();
    const int64_t T = x.size(0), N = x.size(1);
    const int64_t L = num_layers;
    const int64_t dirs = bidirectional ? 2 : 1;
    const int64_t H = hx[0].size(-1);

    Tensor grad_y = batch_first ? grad_y_in.transpose(0, 1).contiguous()
                                : grad_y_in.contiguous();

    try {
        auto& eng = OneDNNContext::get_engine();
        auto& stream = OneDNNContext::get_stream();
        const memory::data_type dt = memory::data_type::f32;

        Tensor hstate = hx[0].contiguous();
        Tensor cstate = hx[1].contiguous();

        // --- Phase 1: re-run forward_training to capture per-layer inputs,
        // forward outputs and workspaces. ---
        struct DirFwd {
            Tensor src_layer;    // (T,N,F) layer input
            Tensor dst_layer;    // (T,N,H)
            Tensor dst_iter;     // (N,H)
            Tensor dst_iter_c;   // (N,H)
            memory workspace;    // opaque oneDNN workspace
        };
        std::vector<std::vector<DirFwd>> fwd(L, std::vector<DirFwd>(dirs));

        Tensor xcur = x;
        for (int64_t layer = 0; layer < L; ++layer) {
            std::vector<Tensor> dir_outs(dirs);
            for (int64_t dir = 0; dir < dirs; ++dir) {
                const int64_t si = layer * dirs + dir;
                const int64_t pbase = si * (2 + (has_biases ? 2 : 0));
                const Tensor& w_ih = params[pbase];
                const Tensor& w_hh = params[pbase + 1];
                const int64_t F = xcur.size(2);
                Tensor bias;
                if (has_biases)
                    bias = params[pbase + 2].add(params[pbase + 3]).contiguous();
                const bool reverse = (dir == 1);

                const BwdEntry& entry = cached_lstm_bwd(eng, T, N, F, H,
                                                        reverse, has_biases);
                const auto& fpd = *entry.fwd_pd;

                Tensor hx_l = hstate.select(0, si).contiguous();
                Tensor cx_l = cstate.select(0, si).contiguous();

                auto src_layer_md = memory::desc({T, N, F}, dt, memory::format_tag::tnc);
                auto src_iter_md  = memory::desc({1, 1, N, H}, dt, memory::format_tag::ldnc);
                auto dst_layer_md = memory::desc({T, N, H}, dt, memory::format_tag::tnc);
                auto dst_iter_md  = memory::desc({1, 1, N, H}, dt, memory::format_tag::ldnc);

                memory w_layer_mem = prepare_weight(eng, stream, w_ih,
                    w_layer_view_md(F, H), fpd.weights_layer_desc());
                memory w_iter_mem = prepare_weight(eng, stream, w_hh,
                    w_iter_view_md(H), fpd.weights_iter_desc());
                memory bias_mem;
                if (has_biases)
                    bias_mem = prepare_weight(eng, stream, bias,
                        bias_view_md(H), fpd.bias_desc());

                DirFwd& df = fwd[layer][dir];
                df.src_layer = xcur;
                df.dst_layer = Tensor::empty({T, N, H}, DType::Float32, input.device());
                df.dst_iter = Tensor::empty({N, H}, DType::Float32, input.device());
                df.dst_iter_c = Tensor::empty({N, H}, DType::Float32, input.device());
                df.workspace = memory(fpd.workspace_desc(), eng);

                memory src_layer_mem(src_layer_md, eng, xcur.data_ptr());
                memory src_iter_mem(src_iter_md, eng, hx_l.data_ptr());
                memory src_iter_c_mem(src_iter_md, eng, cx_l.data_ptr());
                memory dst_layer_mem(dst_layer_md, eng, df.dst_layer.data_ptr());
                memory dst_iter_mem(dst_iter_md, eng, df.dst_iter.data_ptr());
                memory dst_iter_c_mem(dst_iter_md, eng, df.dst_iter_c.data_ptr());

                std::unordered_map<int, memory> args = {
                    {DNNL_ARG_SRC_LAYER, src_layer_mem},
                    {DNNL_ARG_SRC_ITER, src_iter_mem},
                    {DNNL_ARG_SRC_ITER_C, src_iter_c_mem},
                    {DNNL_ARG_WEIGHTS_LAYER, w_layer_mem},
                    {DNNL_ARG_WEIGHTS_ITER, w_iter_mem},
                    {DNNL_ARG_DST_LAYER, dst_layer_mem},
                    {DNNL_ARG_DST_ITER, dst_iter_mem},
                    {DNNL_ARG_DST_ITER_C, dst_iter_c_mem},
                    {DNNL_ARG_WORKSPACE, df.workspace},
                };
                if (has_biases) args.insert({DNNL_ARG_BIAS, bias_mem});
                entry.fwd_prim->execute(stream, args);
                stream.wait();

                dir_outs[dir] = df.dst_layer;
            }
            xcur = dirs == 1 ? dir_outs[0]
                             : cat_kernel({dir_outs[0], dir_outs[1]}, 2).contiguous();
        }

        // --- Phase 2: fused backward, top layer down. ---
        std::vector<Tensor> grad_params(params.size());
        std::vector<Tensor> grad_hx_list(L * dirs), grad_cx_list(L * dirs);

        Tensor grad_cur = grad_y;
        for (int64_t layer = L - 1; layer >= 0; --layer) {
            const int64_t F = fwd[layer][0].src_layer.size(2);
            Tensor grad_layer_input = Tensor::zeros({T, N, F}, DType::Float32,
                                                    input.device());
            for (int64_t dir = dirs - 1; dir >= 0; --dir) {
                const int64_t si = layer * dirs + dir;
                const int64_t pbase = si * (2 + (has_biases ? 2 : 0));
                const Tensor& w_ih = params[pbase];
                const Tensor& w_hh = params[pbase + 1];
                Tensor bias;
                if (has_biases)
                    bias = params[pbase + 2].add(params[pbase + 3]).contiguous();
                const bool reverse = (dir == 1);

                const BwdEntry& entry = cached_lstm_bwd(eng, T, N, F, H,
                                                        reverse, has_biases);
                const auto& bpd = *entry.bwd_pd;
                const DirFwd& df = fwd[layer][dir];

                Tensor hx_l = hstate.select(0, si).contiguous();
                Tensor cx_l = cstate.select(0, si).contiguous();

                auto src_layer_md = memory::desc({T, N, F}, dt, memory::format_tag::tnc);
                auto src_iter_md  = memory::desc({1, 1, N, H}, dt, memory::format_tag::ldnc);
                auto dst_layer_md = memory::desc({T, N, H}, dt, memory::format_tag::tnc);
                auto dst_iter_md  = memory::desc({1, 1, N, H}, dt, memory::format_tag::ldnc);

                // Pack against the BACKWARD pd's resolved layouts -- the
                // bwd pd may pick different weight formats than the fwd pd
                // (the fused primitive validates none of this at execute).
                memory w_layer_mem = prepare_weight(eng, stream, w_ih,
                    w_layer_view_md(F, H), bpd.weights_layer_desc());
                memory w_iter_mem = prepare_weight(eng, stream, w_hh,
                    w_iter_view_md(H), bpd.weights_iter_desc());
                memory bias_mem;
                if (has_biases)
                    bias_mem = prepare_weight(eng, stream, bias,
                        bias_view_md(H), bpd.bias_desc());

                Tensor gy_dir = dirs == 1
                    ? grad_cur
                    : grad_cur.narrow(2, dir * H, H).contiguous();
                Tensor ghy_si = grad_hy_in.defined()
                    ? grad_hy_in.select(0, si).contiguous()
                    : Tensor::zeros({N, H}, DType::Float32, input.device());
                Tensor gcy_si = grad_cy_in.defined()
                    ? grad_cy_in.select(0, si).contiguous()
                    : Tensor::zeros({N, H}, DType::Float32, input.device());

                Tensor diff_src_layer = Tensor::empty({T, N, F}, DType::Float32, input.device());
                Tensor diff_src_iter = Tensor::empty({N, H}, DType::Float32, input.device());
                Tensor diff_src_iter_c = Tensor::empty({N, H}, DType::Float32, input.device());

                memory diff_src_layer_mem(bpd.diff_src_layer_desc(), eng, diff_src_layer.data_ptr());
                memory diff_src_iter_mem(bpd.diff_src_iter_desc(), eng, diff_src_iter.data_ptr());
                memory diff_src_iter_c_mem(bpd.diff_src_iter_c_desc(), eng, diff_src_iter_c.data_ptr());
                memory diff_dst_layer_mem(bpd.diff_dst_layer_desc(), eng, gy_dir.data_ptr());
                memory diff_dst_iter_mem(bpd.diff_dst_iter_desc(), eng, ghy_si.data_ptr());
                memory diff_dst_iter_c_mem(bpd.diff_dst_iter_c_desc(), eng, gcy_si.data_ptr());

                // The fused backward accumulates (beta=1 GEMM / += reduction)
                // into the diff weights and diff bias buffers -- oneDNN only
                // self-initializes them when the pd is built with the
                // overwrite flag, which the C++ API never sets.  Back every
                // diff memory with a zero-filled tensor so each execution
                // starts from a clean slate; the buffers must stay alive
                // until after execution and the unpack reorders below.
                std::vector<Tensor> diff_bufs;
                auto zero_backed_mem = [&](const memory::desc& md) {
                    Tensor buf = Tensor::zeros({md.get_size() / 4},
                                               DType::Float32,
                                               Device(DeviceType::CPU));
                    memory m(md, eng, buf.data_ptr());
                    diff_bufs.push_back(std::move(buf));
                    return m;
                };
                memory diff_w_layer_mem =
                    zero_backed_mem(bpd.diff_weights_layer_desc());
                memory diff_w_iter_mem =
                    zero_backed_mem(bpd.diff_weights_iter_desc());
                memory diff_bias_mem;
                if (has_biases)
                    diff_bias_mem = zero_backed_mem(bpd.diff_bias_desc());

                std::unordered_map<int, memory> args = {
                    {DNNL_ARG_SRC_LAYER, memory(src_layer_md, eng, df.src_layer.data_ptr())},
                    {DNNL_ARG_SRC_ITER, memory(src_iter_md, eng, hx_l.data_ptr())},
                    {DNNL_ARG_SRC_ITER_C, memory(src_iter_md, eng, cx_l.data_ptr())},
                    {DNNL_ARG_WEIGHTS_LAYER, w_layer_mem},
                    {DNNL_ARG_WEIGHTS_ITER, w_iter_mem},
                    {DNNL_ARG_DST_LAYER, memory(dst_layer_md, eng, df.dst_layer.data_ptr())},
                    {DNNL_ARG_DST_ITER, memory(dst_iter_md, eng, df.dst_iter.data_ptr())},
                    {DNNL_ARG_DST_ITER_C, memory(dst_iter_md, eng, df.dst_iter_c.data_ptr())},
                    {DNNL_ARG_WORKSPACE, df.workspace},
                    {DNNL_ARG_DIFF_SRC_LAYER, diff_src_layer_mem},
                    {DNNL_ARG_DIFF_SRC_ITER, diff_src_iter_mem},
                    {DNNL_ARG_DIFF_SRC_ITER_C, diff_src_iter_c_mem},
                    {DNNL_ARG_DIFF_WEIGHTS_LAYER, diff_w_layer_mem},
                    {DNNL_ARG_DIFF_WEIGHTS_ITER, diff_w_iter_mem},
                    {DNNL_ARG_DIFF_DST_LAYER, diff_dst_layer_mem},
                    {DNNL_ARG_DIFF_DST_ITER, diff_dst_iter_mem},
                    {DNNL_ARG_DIFF_DST_ITER_C, diff_dst_iter_c_mem},
                };
                if (has_biases) {
                    args.insert({DNNL_ARG_BIAS, bias_mem});
                    args.insert({DNNL_ARG_DIFF_BIAS, diff_bias_mem});
                }
                entry.bwd_prim->execute(stream, args);
                stream.wait();

                grad_params[pbase] = unpack_diff(eng, stream, diff_w_layer_mem,
                    w_layer_view_md(F, H), {4 * H, F});
                grad_params[pbase + 1] = unpack_diff(eng, stream, diff_w_iter_mem,
                    w_iter_view_md(H), {4 * H, H});
                if (has_biases) {
                    Tensor db = unpack_diff(eng, stream, diff_bias_mem,
                        bias_view_md(H), {4 * H});
                    grad_params[pbase + 2] = db;
                    grad_params[pbase + 3] = db.clone();
                }

                grad_layer_input = grad_layer_input.add(diff_src_layer);
                grad_hx_list[si] = diff_src_iter;
                grad_cx_list[si] = diff_src_iter_c;
            }
            grad_cur = grad_layer_input;
        }

        Tensor grad_input = batch_first ? grad_cur.transpose(0, 1).contiguous()
                                        : grad_cur;
        std::vector<Tensor> hx_grads;
        hx_grads.push_back(stack_kernel(grad_hx_list, 0));
        hx_grads.push_back(stack_kernel(grad_cx_list, 0));
        return std::make_tuple(grad_input, hx_grads, grad_params);
    } catch (...) {
        return std::nullopt;
    }
}
#else
std::optional<std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>>>
onednn_lstm_backward(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                     const std::vector<Tensor>&, const std::vector<Tensor>&,
                     bool, int64_t, bool, bool) {
    return std::nullopt;
}
#endif // USE_ONEDNN

static std::tuple<Tensor, Tensor, Tensor> rnn_impl(
    int kind,  // 0=lstm, 1=gru, 2=tanh, 3=relu
    const Tensor& input, const std::vector<Tensor>& hx,
    const std::vector<Tensor>& params, bool has_biases, int64_t num_layers,
    bool bidirectional, bool batch_first) {
    RnnForwardNoGrad no_grad_guard;
    // (FullLayer::operator() runs one sequence-wide linear_ih GEMM, then
    // the element kernels widen to fp32 internally (opmath semantics).
    const DType dt = input.dtype();
    if (dt != DType::Float32 && dt != DType::Float64 &&
        dt != DType::Float16 && dt != DType::BFloat16) {
        TP_THROW(RuntimeError, "rnn: only Float16/BFloat16/Float32/Float64 inputs are supported");
    }
    Tensor x = batch_first ? input.transpose(0, 1).contiguous() : input.contiguous();
    const int64_t T = x.size(0), N = x.size(1);
    if (hx.empty()) TP_THROW(RuntimeError, "rnn: hx required");
    if (kind == 0 && hx.size() != 2) TP_THROW(RuntimeError, "lstm expects two hidden states");
#ifdef USE_ONEDNN
    // that fast path.  Falls back to the decomposed loop if unavailable.
    if (kind == 0) {
        if (auto r = onednn_lstm_forward(input, hx, params, has_biases,
                                         num_layers, bidirectional, batch_first))
            return *r;
    }
#endif
    const int64_t L = num_layers;
    const int64_t dirs = bidirectional ? 2 : 1;
    const int64_t H = hx[0].size(-1);

    Tensor hn_out = Tensor::zeros({L * dirs, N, H}, hx[0].dtype(), input.device());
    Tensor cn_out = kind == 0
        ? Tensor::zeros({L * dirs, N, H}, hx[0].dtype(), input.device())
        : Tensor();

    size_t ppi = 0;  // params cursor: per layer/direction w_ih, w_hh[, b_ih, b_hh]
    auto param_at = [&](void) -> const Tensor& {
        if (ppi >= params.size()) TP_THROW(RuntimeError, "rnn: missing parameter ", ppi);
        return params[ppi++];
    };

    for (int64_t layer = 0; layer < L; ++layer) {
        // Per-direction sequence outputs concatenated along the feature dim;
        // writes go through Tensor::slice/select views (narrow is a copying
        // op on this backend and must not be used as an assignment target).
        std::vector<Tensor> dir_outs;
        for (int64_t dir = 0; dir < dirs; ++dir) {
            const int64_t state_idx = layer * dirs + dir;
            Tensor h = hx[0].select(0, state_idx).contiguous();
            Tensor c = kind == 0 ? hx[1].select(0, state_idx).contiguous() : h;
            Tensor dir_out = Tensor::zeros({T, N, H}, dt, x.device());

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
            // (T*N, feat) @ (feat, G)^T + b_ih  ->  (T*N, G).
            Tensor x2d = x.reshape({T * N, x.size(2)});
            Tensor in_gates = x2d.mm(w_ih.t());
            if (b_ih.defined()) in_gates = in_gates.add(b_ih);
            const int64_t G = in_gates.size(1);

            const Tensor w_hh_t = w_hh.t();  // (H, G)

            for (int64_t t = 0; t < T; ++t) {
                const int64_t tt = dir == 0 ? t : (T - 1 - t);
                const Tensor ig = in_gates.narrow(0, tt * N, N);   // (N, G)
                Tensor hg = h.mm(w_hh_t);                          // (N, G)
                // lstm / simple cells fold b_hh linearly into every gate;
                // gru handles the three bias segments separately below.
                if (kind != 1 && b_hh.defined()) hg = hg.add(b_hh);

                // Fused-cell fast path (fp32/fp64): one kernel per step instead
                // of the decomposed op sequence below.  Bit-for-bit same math.
                const bool fused = (dt == DType::Float32 || dt == DType::Float64);

                if (kind == 0) {
                    if (fused) {
                        if (dt == DType::Float32) {
                            auto r = rnn_cpu::lstm_cell<float>(ig, hg, c);
                            h = std::get<0>(r); c = std::get<1>(r);
                        } else {
                            auto r = rnn_cpu::lstm_cell<double>(ig, hg, c);
                            h = std::get<0>(r); c = std::get<1>(r);
                        }
                    } else {
                        auto gate = [&](int64_t off, Tensor (*fn)(const Tensor&)) -> Tensor {
                            return fn(ig.narrow(1, off, H).add(hg.narrow(1, off, H)));
                        };
                        Tensor i_ = gate(0, [](const Tensor& v) { return v.sigmoid(); });
                        Tensor f_ = gate(H, [](const Tensor& v) { return v.sigmoid(); });
                        Tensor g_ = gate(2 * H, [](const Tensor& v) { return v.tanh(); });
                        Tensor o_ = gate(3 * H, [](const Tensor& v) { return v.sigmoid(); });
                        c = f_.mul(c).add(i_.mul(g_));
                        h = o_.mul(c.tanh());
                    }
                } else if (kind == 1) {
                    //   r = sigmoid(ir + hr), z = sigmoid(iz + hz)
                    //   n = tanh(in + r * (hn + b_hn))
                    //   h' = (1 - z) * n + z * h
                    if (fused) {
                        h = (dt == DType::Float32)
                            ? rnn_cpu::gru_cell<float>(ig, hg, h, b_hh)
                            : rnn_cpu::gru_cell<double>(ig, hg, h, b_hh);
                    } else {
                        Tensor b_r, b_z, b_n;
                        if (b_hh.defined()) {
                            b_r = b_hh.narrow(0, 0, H);
                            b_z = b_hh.narrow(0, H, H);
                            b_n = b_hh.narrow(0, 2 * H, H);
                        }
                        Tensor r_ = ig.narrow(1, 0, H)
                                        .add(b_r.defined() ? b_r.add(hg.narrow(1, 0, H))
                                                           : hg.narrow(1, 0, H))
                                        .sigmoid();
                        Tensor z_ = ig.narrow(1, H, H)
                                        .add(b_z.defined() ? b_z.add(hg.narrow(1, H, H))
                                                           : hg.narrow(1, H, H))
                                        .sigmoid();
                        Tensor hn_lin = hg.narrow(1, 2 * H, H);
                        if (b_n.defined()) hn_lin = hn_lin.add(b_n);
                        Tensor n_ = ig.narrow(1, 2 * H, H).add(r_.mul(hn_lin)).tanh();
                        Tensor one_minus_z = z_.neg().add(Scalar(1));
                        h = one_minus_z.mul(n_).add(z_.mul(h));
                    }
                } else {
                    Tensor pre = ig.add(hg);
                    h = (kind == 2) ? pre.tanh() : pre.relu();
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
            layer_out = cat_kernel({dir_outs[0], dir_outs[1]}, 2);
        }
        x = layer_out;
    }
    Tensor y = batch_first ? x.transpose(0, 1).contiguous() : x;
    return {y, hn_out, cn_out};
}

std::tuple<Tensor, Tensor> rnn_relu_cpu(const Tensor& input, const std::vector<Tensor>& hx,
                                        const std::vector<Tensor>& params, bool has_biases,
                                        int64_t num_layers, float dropout_p, bool training,
                                        bool bidirectional, bool batch_first) {
    (void)dropout_p; (void)training;
    auto r = rnn_impl(3, input, hx, params, has_biases, num_layers, bidirectional, batch_first);
    return {std::get<0>(r), std::get<1>(r)};
}
std::tuple<Tensor, Tensor> rnn_tanh_cpu(const Tensor& input, const std::vector<Tensor>& hx,
                                        const std::vector<Tensor>& params, bool has_biases,
                                        int64_t num_layers, float dropout_p, bool training,
                                        bool bidirectional, bool batch_first) {
    (void)dropout_p; (void)training;
    auto r = rnn_impl(2, input, hx, params, has_biases, num_layers, bidirectional, batch_first);
    return {std::get<0>(r), std::get<1>(r)};
}
std::tuple<Tensor, Tensor> gru_cpu(const Tensor& input, const std::vector<Tensor>& hx,
                                   const std::vector<Tensor>& params, bool has_biases,
                                   int64_t num_layers, float dropout_p, bool training,
                                   bool bidirectional, bool batch_first) {
    (void)dropout_p; (void)training;
    auto r = rnn_impl(1, input, hx, params, has_biases, num_layers, bidirectional, batch_first);
    return {std::get<0>(r), std::get<1>(r)};
}
std::tuple<Tensor, Tensor, Tensor> lstm_cpu(const Tensor& input, const std::vector<Tensor>& hx,
                                            const std::vector<Tensor>& params, bool has_biases,
                                            int64_t num_layers, float dropout_p, bool training,
                                            bool bidirectional, bool batch_first) {
    (void)dropout_p; (void)training;
    return rnn_impl(0, input, hx, params, has_biases, num_layers, bidirectional, batch_first);
}


TENSORPLAY_LIBRARY_IMPL(CPU, RnnSequence) {
    m.impl("lstm", lstm_cpu);
    m.impl("gru", gru_cpu);
    m.impl("rnn_relu", rnn_relu_cpu);
    m.impl("rnn_tanh", rnn_tanh_cpu);
}

}  // namespace cpu
}  // namespace tensorplay
