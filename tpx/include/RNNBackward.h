#pragma once
// Native sequence-level RNN backward (lstm / gru / rnn_tanh / rnn_relu).
//
// ATen provides no CPU sequence backward (torch rides mkldnn/cudnn there);
// this is a replay-based port of the cell math in aten/src/ATen/native/
// RNN.cpp + cuda/RNN.cu, structured like ATen's lstm_backward: replay the
// forward per layer to recover hidden states and gates, then step backwards
// through time accumulating gate gradients, and reduce the weight gradients
// with one GEMM per direction.  Every step is a dispatched recordable op, so
// the same code runs on CPU and CUDA and create_graph records through it.

#include "Node.h"
#include "Autograd.h"
#include "SavedVariable.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <tuple>
#include <vector>

namespace tensorplay {
namespace tpx {

namespace rnn_bwd_detail {

inline Tensor narrow_gate(const Tensor& gates, int64_t off, int64_t h) {
    return ops::narrow(gates, 1, off, h);
}

inline Tensor sig_back(const Tensor& dpost, const Tensor& sig) {
    return dpost * sig * (sig * (-1) + 1);
}

inline Tensor tanh_back(const Tensor& dpost, const Tensor& th) {
    return dpost * (th * th * (-1) + 1);
}

struct DirReplay {
    Tensor h0, c0;                       // initial states (N, H)
    std::vector<Tensor> h_hist;          // h after each cell step (cell order)
    std::vector<Tensor> c_hist;          // lstm: c after each cell step
    std::vector<Tensor> gates_hist;      // post-activation gates per step:
                                         // lstm i,f,g,o / gru r,z,n / vanilla -
    std::vector<Tensor> hn_lin_hist;     // gru: hidden n-part incl. bias
    Tensor in_gates;                     // (T*N, G) input-side preactivations
};

struct LayerReplay {
    Tensor input;                        // (T, N, F) layer input
    std::vector<DirReplay> dirs;
};

// Replay TP's rnn_impl forward exactly (same gate math, same param cursor),
// recording the states the backward needs.
inline std::vector<LayerReplay> replay(
        int kind, const Tensor& x0, const std::vector<Tensor>& hx,
        const std::vector<Tensor>& params, bool has_biases, int64_t L,
        bool bidirectional, int64_t T, int64_t N) {
    const int64_t dirs = bidirectional ? 2 : 1;
    const int64_t H = hx[0].size(-1);
    std::vector<LayerReplay> layers;
    Tensor x = x0;
    size_t ppi = 0;
    for (int64_t layer = 0; layer < L; ++layer) {
        LayerReplay lr;
        lr.input = x;
        for (int64_t dir = 0; dir < dirs; ++dir) {
            const int64_t si = layer * dirs + dir;
            DirReplay dr;
            dr.h0 = ops::contiguous(ops::select(hx[0], 0, si));
            dr.c0 = kind == 0 ? ops::contiguous(ops::select(hx[1], 0, si))
                              : Tensor();
            const Tensor& w_ih = params.at(ppi);
            const Tensor& w_hh = params.at(ppi + 1);
            Tensor b_ih, b_hh;
            if (has_biases) {
                b_ih = params.at(ppi + 2);
                b_hh = params.at(ppi + 3);
                if (b_ih.numel() == 0) b_ih = Tensor();
                if (b_hh.numel() == 0) b_hh = Tensor();
            }
            ppi += 2 + (has_biases ? 2 : 0);

            const int64_t F = x.size(2);
            const Tensor x2d = ops::reshape(x, {T * N, F});
            Tensor in_gates = ops::mm(x2d, ops::t(w_ih));
            if (b_ih.defined()) in_gates = ops::add(in_gates, b_ih);
            dr.in_gates = in_gates;
            const int64_t G = in_gates.size(1);
            const Tensor w_hh_t = ops::t(w_hh);

            Tensor h = dr.h0, c = dr.c0;
            Tensor b_r, b_z, b_n;
            if (kind == 1 && b_hh.defined()) {
                b_r = ops::narrow(b_hh, 0, 0, H);
                b_z = ops::narrow(b_hh, 0, H, H);
                b_n = ops::narrow(b_hh, 0, 2 * H, H);
            }
            for (int64_t t = 0; t < T; ++t) {
                const int64_t tt = dir == 0 ? t : (T - 1 - t);
                const Tensor ig = ops::narrow(in_gates, 0, tt * N, N);
                Tensor hg = ops::mm(h, w_hh_t);
                if (kind != 1 && b_hh.defined()) hg = ops::add(hg, b_hh);
                if (kind == 0) {
                    Tensor i_ = ops::sigmoid(narrow_gate(ig, 0, H) + narrow_gate(hg, 0, H));
                    Tensor f_ = ops::sigmoid(narrow_gate(ig, H, H) + narrow_gate(hg, H, H));
                    Tensor g_ = ops::tanh(narrow_gate(ig, 2 * H, H) + narrow_gate(hg, 2 * H, H));
                    Tensor o_ = ops::sigmoid(narrow_gate(ig, 3 * H, H) + narrow_gate(hg, 3 * H, H));
                    c = f_ * c + i_ * g_;
                    h = o_ * ops::tanh(c);
                    dr.h_hist.push_back(h);
                    dr.c_hist.push_back(c);
                    dr.gates_hist.push_back(ops::cat({i_, f_, g_, o_}, 1));
                } else if (kind == 1) {
                    Tensor r_ = ops::sigmoid(
                        narrow_gate(ig, 0, H) +
                        (b_r.defined() ? b_r + narrow_gate(hg, 0, H)
                                       : narrow_gate(hg, 0, H)));
                    Tensor z_ = ops::sigmoid(
                        narrow_gate(ig, H, H) +
                        (b_z.defined() ? b_z + narrow_gate(hg, H, H)
                                       : narrow_gate(hg, H, H)));
                    Tensor hn_lin = narrow_gate(hg, 2 * H, H);
                    if (b_n.defined()) hn_lin = hn_lin + b_n;
                    Tensor n_ = ops::tanh(narrow_gate(ig, 2 * H, H) + r_ * hn_lin);
                    h = (z_ * (-1) + 1) * n_ + z_ * h;
                    dr.h_hist.push_back(h);
                    dr.gates_hist.push_back(ops::cat({r_, z_, n_}, 1));
                    dr.hn_lin_hist.push_back(hn_lin);
                } else {
                    Tensor pre = ig + hg;
                    h = kind == 2 ? ops::tanh(pre) : ops::relu(pre);
                    dr.h_hist.push_back(h);
                }
            }
            (void)G;
            lr.dirs.push_back(std::move(dr));
        }
        // Layer output (matches rnn_impl): concat directions on feature dim.
        auto dir_out = [&](int64_t dir) -> Tensor {
            // Reconstruct the time-ordered output from the cell-order history.
            std::vector<Tensor> rows(T);
            for (int64_t t = 0; t < T; ++t) {
                const int64_t tt = dir == 0 ? t : (T - 1 - t);
                rows[tt] = lr.dirs[dir].h_hist[t];
            }
            return ops::stack(rows, 0);
        };
        x = dirs == 1 ? dir_out(0) : ops::cat({dir_out(0), dir_out(1)}, 2);
        layers.push_back(std::move(lr));
    }
    return layers;
}

} // namespace rnn_bwd_detail

// Returns (grad_input, grad_hx, grad_cx, grad_params).
// kind: 0=lstm, 1=gru, 2=rnn_tanh, 3=rnn_relu (matches rnn_impl).
inline std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>>
rnn_backward_impl(int kind, const Tensor& grad_y_in, const Tensor& grad_hy_in,
                  const Tensor& grad_cy_in, const Tensor& input,
                  const std::vector<Tensor>& hx,
                  const std::vector<Tensor>& params, bool has_biases,
                  int64_t num_layers, bool bidirectional, bool batch_first) {
    using namespace rnn_bwd_detail;
    Tensor x0 = batch_first ? ops::contiguous(ops::transpose(input, 0, 1))
                            : ops::contiguous(input);
    const int64_t T = x0.size(0), N = x0.size(1);
    const int64_t L = num_layers;
    const int64_t dirs = bidirectional ? 2 : 1;
    const int64_t H = hx[0].size(-1);
    const DType dt = input.dtype();

    Tensor grad_y = batch_first
        ? ops::contiguous(ops::transpose(grad_y_in, 0, 1))
        : ops::contiguous(grad_y_in);

    const auto layers = replay(kind, x0, hx, params, has_biases, L,
                               bidirectional, T, N);

    std::vector<Tensor> grad_params(params.size());
    std::vector<Tensor> grad_hx_list(L * dirs), grad_cx_list;
    if (kind == 0) grad_cx_list.resize(L * dirs);

    Tensor grad_cur = grad_y;
    for (int64_t layer = L - 1; layer >= 0; --layer) {
        const LayerReplay& lr = layers[layer];
        const int64_t F = lr.input.size(2);
        const Tensor x2d = ops::reshape(lr.input, {T * N, F});
        Tensor grad_layer_input = ops::zeros_like(lr.input);

        for (int64_t dir = dirs - 1; dir >= 0; --dir) {
            const int64_t si = layer * dirs + dir;
            const DirReplay& dr = lr.dirs[dir];
            const int64_t pbase = si * (2 + (has_biases ? 2 : 0));
            const Tensor& w_ih = params[pbase];
            const Tensor& w_hh = params[pbase + 1];
            Tensor b_hh;
            if (has_biases) {
                b_hh = params[pbase + 3];
                if (b_hh.numel() == 0) b_hh = Tensor();
            }

            const Tensor gy_dir = dirs == 1
                ? grad_cur
                : ops::narrow(grad_cur, 2, dir * H, H);

            Tensor dh_next = grad_hy_in.defined()
                ? ops::select(grad_hy_in, 0, si)
                : ops::zeros({N, H}, dt, input.device());
            Tensor dc_next = (kind == 0 && grad_cy_in.defined())
                ? ops::select(grad_cy_in, 0, si)
                : ops::zeros({N, H}, dt, input.device());

            const Tensor w_hh_t = ops::t(w_hh);
            Tensor b_r, b_z, b_n;
            if (kind == 1 && b_hh.defined()) {
                b_r = ops::narrow(b_hh, 0, 0, H);
                b_z = ops::narrow(b_hh, 0, H, H);
                b_n = ops::narrow(b_hh, 0, 2 * H, H);
            }

            // Gate-gradient matrices: g_rows is assembled in TIME order (it
            // pairs with x2d, whose rows are time-ordered, for the w_ih
            // GEMM); hg/hprev rows stay in cell order (they only pair with
            // each other for the w_hh GEMM).
            std::vector<Tensor> g_rows(T), hg_rows(T), hprev_rows(T), dx_rows(T);
            for (int64_t t = T - 1; t >= 0; --t) {
                const int64_t tt = dir == 0 ? t : (T - 1 - t);
                const Tensor gy_t = ops::select(gy_dir, 0, tt);
                const Tensor dh = gy_t + dh_next;
                const Tensor h_prev = t == 0 ? dr.h0 : dr.h_hist[t - 1];
                const Tensor ig = ops::narrow(dr.in_gates, 0, tt * N, N);
                Tensor dpre, dhid;  // input-side gate grad, hidden-side gate grad

                if (kind == 0) {
                    const Tensor c = dr.c_hist[t];
                    const Tensor c_prev = t == 0 ? dr.c0 : dr.c_hist[t - 1];
                    const Tensor gates = dr.gates_hist[t];
                    const Tensor i_ = ops::narrow(gates, 1, 0, H);
                    const Tensor f_ = ops::narrow(gates, 1, H, H);
                    const Tensor g_ = ops::narrow(gates, 1, 2 * H, H);
                    const Tensor o_ = ops::narrow(gates, 1, 3 * H, H);
                    const Tensor tanh_c = ops::tanh(c);
                    const Tensor dc = dc_next + dh * o_ * (tanh_c * tanh_c * (-1) + 1);
                    const Tensor dpre_i = sig_back(dc * g_, i_);
                    const Tensor dpre_f = sig_back(dc * c_prev, f_);
                    const Tensor dpre_g = tanh_back(dc * i_, g_);
                    const Tensor dpre_o = sig_back(dh * tanh_c, o_);
                    dc_next = dc * f_;
                    dpre = ops::cat({dpre_i, dpre_f, dpre_g, dpre_o}, 1);
                    dhid = dpre;
                    dh_next = ops::mm(dhid, w_hh);
                } else if (kind == 1) {
                    const Tensor gates = dr.gates_hist[t];
                    const Tensor r_ = ops::narrow(gates, 1, 0, H);
                    const Tensor z_ = ops::narrow(gates, 1, H, H);
                    const Tensor n_ = ops::narrow(gates, 1, 2 * H, H);
                    const Tensor hn_lin = dr.hn_lin_hist[t];
                    const Tensor dz = dh * (h_prev - n_);
                    const Tensor dn = dh * (z_ * (-1) + 1);
                    const Tensor dpre_n = tanh_back(dn, n_);
                    const Tensor dr_ = dpre_n * hn_lin;
                    const Tensor dhn = dpre_n * r_;
                    const Tensor dpre_r = sig_back(dr_, r_);
                    const Tensor dpre_z = sig_back(dz, z_);
                    dpre = ops::cat({dpre_r, dpre_z, dpre_n}, 1);
                    dhid = ops::cat({dpre_r, dpre_z, dhn}, 1);
                    dh_next = dh * z_ + ops::mm(dhid, w_hh);
                } else {
                    dpre = kind == 2 ? tanh_back(dh, dr.h_hist[t])
                                     : dh * ops::gt(dr.h_hist[t], Scalar(0));
                    dhid = dpre;
                    dh_next = ops::mm(dhid, w_hh);
                }

                g_rows[tt] = dpre;
                hg_rows[t] = dhid;
                hprev_rows[t] = h_prev;
                dx_rows[tt] = ops::mm(dpre, w_ih);
                (void)ig;
            }

            const Tensor G2d = ops::reshape(ops::stack(g_rows, 0), {T * N, -1});
            const Tensor HG2d = ops::reshape(ops::stack(hg_rows, 0), {T * N, -1});
            const Tensor Hprev2d = ops::reshape(ops::stack(hprev_rows, 0), {T * N, H});

            grad_params[pbase] = ops::mm(ops::t(G2d), x2d);
            grad_params[pbase + 1] = ops::mm(ops::t(HG2d), Hprev2d);
            if (has_biases) {
                grad_params[pbase + 2] = ops::sum(G2d, {0});
                grad_params[pbase + 3] = ops::sum(HG2d, {0});
            }
            grad_hx_list[si] = dh_next;
            if (kind == 0) grad_cx_list[si] = dc_next;

            const Tensor dx = ops::stack(dx_rows, 0);  // (T, N, F) time order
            grad_layer_input = grad_layer_input + dx;
        }
        grad_cur = grad_layer_input;
    }

    Tensor grad_input = batch_first
        ? ops::contiguous(ops::transpose(grad_cur, 0, 1))
        : grad_cur;
    Tensor grad_hx = ops::stack(grad_hx_list, 0);
    std::vector<Tensor> hx_grads{grad_hx};
    std::vector<Tensor> cx_grads;
    if (kind == 0) {
        hx_grads.push_back(ops::stack(grad_cx_list, 0));
    }
    return {grad_input, hx_grads, grad_params};
}

// ---------------------------------------------------------------------------
// Backward nodes.  apply() outputs align positionally with the edges
// collected at record time: input, hx elements, params elements.
// ---------------------------------------------------------------------------

namespace rnn_bwd_detail {

class RnnNodeBase : public Node {
public:
    RnnNodeBase(int kind, Tensor input, std::vector<Tensor> hx,
                std::vector<Tensor> params, bool has_biases,
                int64_t num_layers, double dropout_p, bool training,
                bool bidirectional, bool batch_first)
        : kind_(kind), input_(std::move(input)), has_biases_(has_biases),
          num_layers_(num_layers), dropout_p_(dropout_p), training_(training),
          bidirectional_(bidirectional), batch_first_(batch_first) {
        hx_.reserve(hx.size());
        for (auto& t : hx) hx_.emplace_back(std::move(t));
        params_.reserve(params.size());
        for (auto& t : params) params_.emplace_back(std::move(t));
    }

    // Gradient inputs: grad_output, grad_hidden and (lstm only) grad_cell,
    // delivered at input slots 0..n by the engine.
    size_t num_inputs() const override { return kind_ == 0 ? 3 : 2; }

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) {
            variable_list undef(1 + hx_.size() + params_.size());
            return undef;
        }
        std::vector<Tensor> hx, params;
        hx.reserve(hx_.size());
        for (auto& sv : hx_) hx.push_back(sv.unpack());
        params.reserve(params_.size());
        for (auto& sv : params_) params.push_back(sv.unpack());
        const Tensor grad_y = inputs[0];
        const Tensor grad_hy = inputs.size() > 1 ? inputs[1] : Tensor();
        const Tensor grad_cy = inputs.size() > 2 ? inputs[2] : Tensor();
        auto [gx, ghx, gparams] = rnn_backward_impl(
            kind_, grad_y, grad_hy, grad_cy, input_.unpack(), hx, params,
            has_biases_, num_layers_, bidirectional_, batch_first_);
        variable_list grads;
        grads.reserve(1 + hx_.size() + params_.size());
        grads.push_back(std::move(gx));
        for (auto& g : ghx) grads.push_back(std::move(g));
        for (auto& g : gparams) grads.push_back(std::move(g));
        return grads;
    }

    void release_variables() override {
        Node::release_variables();
        input_.reset_data();
        for (auto& sv : hx_) sv.reset_data();
        for (auto& sv : params_) sv.reset_data();
    }

protected:
    int kind_;
    SavedVariable input_;
    std::vector<SavedVariable> hx_;
    std::vector<SavedVariable> params_;
    bool has_biases_;
    int64_t num_layers_;
    double dropout_p_;
    bool training_;
    bool bidirectional_;
    bool batch_first_;
};

} // namespace rnn_bwd_detail

struct LstmBackward : public rnn_bwd_detail::RnnNodeBase {
    LstmBackward(Tensor input, std::vector<Tensor> hx,
                 std::vector<Tensor> params, bool has_biases,
                 int64_t num_layers, double dropout_p, bool training,
                 bool bidirectional, bool batch_first)
        : RnnNodeBase(0, std::move(input), std::move(hx), std::move(params),
                      has_biases, num_layers, dropout_p, training,
                      bidirectional, batch_first) {}
};

struct GruBackward : public rnn_bwd_detail::RnnNodeBase {
    GruBackward(Tensor input, std::vector<Tensor> hx,
                std::vector<Tensor> params, bool has_biases,
                int64_t num_layers, double dropout_p, bool training,
                bool bidirectional, bool batch_first)
        : RnnNodeBase(1, std::move(input), std::move(hx), std::move(params),
                      has_biases, num_layers, dropout_p, training,
                      bidirectional, batch_first) {}
};

struct RnnTanhBackward : public rnn_bwd_detail::RnnNodeBase {
    RnnTanhBackward(Tensor input, std::vector<Tensor> hx,
                    std::vector<Tensor> params, bool has_biases,
                    int64_t num_layers, double dropout_p, bool training,
                    bool bidirectional, bool batch_first)
        : RnnNodeBase(2, std::move(input), std::move(hx), std::move(params),
                      has_biases, num_layers, dropout_p, training,
                      bidirectional, batch_first) {}
};

struct RnnReluBackward : public rnn_bwd_detail::RnnNodeBase {
    RnnReluBackward(Tensor input, std::vector<Tensor> hx,
                    std::vector<Tensor> params, bool has_biases,
                    int64_t num_layers, double dropout_p, bool training,
                    bool bidirectional, bool batch_first)
        : RnnNodeBase(3, std::move(input), std::move(hx), std::move(params),
                      has_biases, num_layers, dropout_p, training,
                      bidirectional, batch_first) {}
};

} // namespace tpx
} // namespace tensorplay
