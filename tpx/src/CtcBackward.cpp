#include "CtcBackward.h"
#include "GradMode.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace tensorplay {
namespace tpx {
namespace {

constexpr double neginf = -std::numeric_limits<double>::infinity();

Tensor constant(const Tensor& like, int64_t length, double value) {
    return Tensor::full({length}, Scalar(value), like.dtype(), like.device());
}

Tensor shift(const Tensor& row, int64_t amount, double fill) {
    const int64_t count = std::min<int64_t>(std::abs(amount), row.size(0));
    Tensor padding = constant(row, count, fill);
    return amount > 0
        ? ops::cat({padding, ops::narrow(row, 0, 0, row.size(0) - count)}, 0)
        : ops::cat({ops::narrow(row, 0, count, row.size(0) - count), padding}, 0);
}

// Unreachable states stay at -inf and have zero derivatives. Replace their
// zero exponent sum before taking log so reverse mode never evaluates 0/0.
Tensor log_sum(const std::vector<Tensor>& terms) {
    Tensor stacked = ops::stack(terms, 0);
    Tensor pivot = ops::amax(stacked, {0});
    pivot = ops::where(ops::eq(pivot, Scalar(neginf)), Scalar(0), pivot);
    Tensor total = ops::sum(ops::exp(ops::sub(stacked, pivot)), {0});
    Tensor absent = ops::eq(total, Scalar(0));
    Tensor safe = ops::where(absent, Scalar(1), total);
    return ops::where(absent, Scalar(neginf), ops::add(ops::log(safe), pivot));
}

Tensor weights(const Tensor& term, const Tensor& total) {
    return ops::exp(ops::sub(term, ops::where(ops::eq(total, Scalar(neginf)), Scalar(0), total)));
}

struct Sequence {
    int64_t length;
    Tensor labels, skip_forward, skip_backward, initial, terminal;
};

std::vector<Sequence> sequences(const Tensor& lp, const Tensor& targets,
                                const Tensor& input_lengths, const Tensor& target_lengths,
                                int64_t blank) {
    Tensor tg = targets.to(Device(DeviceType::CPU), DType::Int64).contiguous();
    Tensor il = input_lengths.to(Device(DeviceType::CPU), DType::Int64).contiguous();
    Tensor tl = target_lengths.to(Device(DeviceType::CPU), DType::Int64).contiguous();
    const auto* target_data = tg.data_ptr<int64_t>();
    std::vector<Sequence> result;
    int64_t offset = 0;
    for (int64_t b = 0; b < lp.size(1); ++b) {
        const int64_t target_length = tl.data_ptr<int64_t>()[b];
        const int64_t states = target_length * 2 + 1;
        if (targets.dim() == 2) offset = b * tg.size(1);
        std::vector<int64_t> labels(static_cast<size_t>(states), blank);
        for (int64_t s = 1; s < states; s += 2)
            labels[static_cast<size_t>(s)] = target_data[offset + s / 2];
        std::vector<double> forward(static_cast<size_t>(states), neginf);
        std::vector<double> backward = forward, initial = forward, terminal = forward;
        initial[0] = 0;
        if (states > 1) initial[1] = 0;
        terminal[static_cast<size_t>(states - 1)] = 0;
        if (states > 1) terminal[static_cast<size_t>(states - 2)] = 0;
        for (int64_t s = 2; s < states; ++s) {
            if (labels[static_cast<size_t>(s)] != labels[static_cast<size_t>(s - 2)]) {
                forward[static_cast<size_t>(s)] = 0;
                backward[static_cast<size_t>(s - 2)] = 0;
            }
        }
        result.push_back({il.data_ptr<int64_t>()[b],
            Tensor::tensor(labels, DType::Int64).to(lp.device()),
            Tensor::tensor(forward, lp.dtype()).to(lp.device()),
            Tensor::tensor(backward, lp.dtype()).to(lp.device()),
            Tensor::tensor(initial, lp.dtype()).to(lp.device()),
            Tensor::tensor(terminal, lp.dtype()).to(lp.device())});
        if (targets.dim() == 1) offset += target_length;
    }
    return result;
}

std::vector<Tensor> alphas(const Tensor& emissions, const Sequence& seq) {
    std::vector<Tensor> rows;
    if (seq.length == 0) return rows;
    rows.push_back(ops::add(ops::select(emissions, 0, 0), seq.initial));
    for (int64_t t = 1; t < seq.length; ++t) {
        const Tensor& prev = rows.back();
        rows.push_back(ops::add(ops::select(emissions, 0, t), log_sum({prev,
            shift(prev, 1, neginf), ops::add(shift(prev, 2, neginf), seq.skip_forward)})));
    }
    return rows;
}

std::vector<Tensor> betas(const Tensor& emissions, const Sequence& seq) {
    std::vector<Tensor> rows(static_cast<size_t>(seq.length));
    if (seq.length == 0) return rows;
    rows.back() = ops::add(ops::select(emissions, 0, seq.length - 1), seq.terminal);
    for (int64_t t = seq.length - 2; t >= 0; --t) {
        const Tensor& next = rows[static_cast<size_t>(t + 1)];
        rows[static_cast<size_t>(t)] = ops::add(ops::select(emissions, 0, t), log_sum({next,
            shift(next, -1, neginf), ops::add(shift(next, -2, neginf), seq.skip_backward)}));
    }
    return rows;
}

Tensor likelihood(const std::vector<Tensor>& alpha) {
    const Tensor& last = alpha.back();
    std::vector<Tensor> terms{ops::select(last, 0, last.size(0) - 1)};
    if (last.size(0) > 1) terms.push_back(ops::select(last, 0, last.size(0) - 2));
    return ops::neg(log_sum(terms));
}

Tensor class_sum(const Tensor& state_values, const Tensor& labels, const Tensor& frame) {
    return ops::index_add(ops::zeros_like(frame), 0, labels, state_values);
}

Tensor pad_frames(std::vector<Tensor> rows, const Tensor& lp) {
    while (static_cast<int64_t>(rows.size()) < lp.size(0))
        rows.push_back(ops::mul(ops::select(lp, 0, static_cast<int64_t>(rows.size())), Scalar(0)));
    return ops::stack(rows, 0);
}

// Recompute saved statistics with history when recording a loss backward.
// Its derivative includes both halves of the dynamic program and the
// normalization likelihood, while retaining the exp(log_probs) term.
Tensor recorded_backward(const Tensor& go, const Tensor& lp,
                          const std::vector<Sequence>& seqs, bool zero_infinity) {
    std::vector<Tensor> batches;
    for (int64_t b = 0; b < lp.size(1); ++b) {
        const auto& seq = seqs[static_cast<size_t>(b)];
        Tensor batch = ops::select(lp, 1, b);
        Tensor emissions = ops::index_select(batch, 1, seq.labels);
        auto alpha = alphas(emissions, seq);
        auto beta = betas(emissions, seq);
        std::vector<Tensor> rows;
        if (seq.length > 0) {
            Tensor nll = likelihood(alpha);
            Tensor ignored = ops::eq(nll, Scalar(std::numeric_limits<double>::infinity()));
            Tensor normalizer = zero_infinity ? ops::where(ignored, Scalar(0), nll) : nll;
            for (int64_t t = 0; t < seq.length; ++t) {
                Tensor frame = ops::select(batch, 0, t);
                Tensor post = ops::exp(ops::sub(ops::add(
                    ops::add(alpha[static_cast<size_t>(t)], beta[static_cast<size_t>(t)]), normalizer),
                    ops::select(emissions, 0, t)));
                Tensor row = ops::mul(ops::sub(ops::exp(frame), class_sum(post, seq.labels, frame)),
                                      ops::select(go, 0, b));
                if (zero_infinity) row = ops::where(ignored, Scalar(0), row);
                rows.push_back(row);
            }
        }
        batches.push_back(pad_frames(std::move(rows), batch));
    }
    return batches.empty() ? ops::zeros_like(lp) : ops::stack(batches, 1);
}

// Reverse the alpha recurrence for gradients arriving at the saved states.
Tensor alpha_backward(const Tensor& adj, const Tensor& lp,
                       const std::vector<Sequence>& seqs) {
    std::vector<Tensor> batches;
    for (int64_t b = 0; b < lp.size(1); ++b) {
        const auto& seq = seqs[static_cast<size_t>(b)];
        Tensor batch = ops::select(lp, 1, b);
        Tensor emissions = ops::index_select(batch, 1, seq.labels);
        auto alpha = alphas(emissions, seq);
        Tensor carry = constant(lp, seq.labels.numel(), 0);
        std::vector<Tensor> rows(static_cast<size_t>(seq.length));
        for (int64_t t = seq.length - 1; t >= 0; --t) {
            Tensor incoming = ops::add(carry, ops::narrow(ops::select(ops::select(adj, 0, b), 0, t),
                                                           0, 0, seq.labels.numel()));
            // Unreachable states are constants, including their emissions.
            incoming = ops::where(ops::eq(alpha[static_cast<size_t>(t)], Scalar(neginf)), Scalar(0), incoming);
            rows[static_cast<size_t>(t)] = class_sum(incoming, seq.labels, ops::select(batch, 0, t));
            if (t > 0) {
                Tensor a = alpha[static_cast<size_t>(t - 1)];
                Tensor a1 = shift(a, 1, neginf);
                Tensor a2 = ops::add(shift(a, 2, neginf), seq.skip_forward);
                Tensor total = log_sum({a, a1, a2});
                carry = ops::add(ops::add(ops::mul(incoming, weights(a, total)),
                    shift(ops::mul(incoming, weights(a1, total)), -1, 0)),
                    shift(ops::mul(incoming, weights(a2, total)), -2, 0));
            }
        }
        batches.push_back(pad_frames(std::move(rows), batch));
    }
    return batches.empty() ? ops::zeros_like(lp) : ops::stack(batches, 1);
}

variable_list backward_adjoints(const Tensor& adj, const Tensor& go, const Tensor& lp,
                                const Tensor& nll, const Tensor& alpha,
                                const std::vector<Sequence>& seqs, bool zero_infinity) {
    std::vector<Tensor> dgo, dlp, dnll, dalpha;
    for (int64_t b = 0; b < lp.size(1); ++b) {
        const auto& seq = seqs[static_cast<size_t>(b)];
        Tensor batch = ops::select(lp, 1, b);
        Tensor emissions = ops::index_select(batch, 1, seq.labels);
        Tensor incoming = ops::select(adj, 1, b);
        Tensor gr = ops::select(go, 0, b), normalizer = ops::select(nll, 0, b);
        Tensor ignored = ops::eq(normalizer, Scalar(std::numeric_limits<double>::infinity()));
        if (zero_infinity) normalizer = ops::where(ignored, Scalar(0), normalizer);
        auto beta = betas(emissions, seq);
        Tensor carry = constant(lp, seq.labels.numel(), 0);
        Tensor dg = ops::mul(gr, Scalar(0)), dn = ops::mul(normalizer, Scalar(0));
        std::vector<Tensor> rows, alpha_rows;
        for (int64_t t = 0; t < seq.length; ++t) {
            Tensor frame = ops::select(batch, 0, t), v = ops::select(incoming, 0, t);
            Tensor a = ops::narrow(ops::select(ops::select(alpha, 0, b), 0, t), 0, 0, seq.labels.numel());
            Tensor post = ops::exp(ops::sub(ops::add(ops::add(a, beta[static_cast<size_t>(t)]), normalizer),
                                           ops::select(emissions, 0, t)));
            Tensor h = ops::neg(ops::mul(ops::mul(post, ops::index_select(v, 0, seq.labels)), gr));
            Tensor dg_frame = ops::sum(ops::mul(v, ops::sub(ops::exp(frame), class_sum(post, seq.labels, frame))));
            if (zero_infinity) {
                h = ops::where(ignored, Scalar(0), h);
                dg_frame = ops::where(ignored, Scalar(0), dg_frame);
            }
            dg = ops::add(dg, dg_frame);
            dn = ops::add(dn, ops::sum(h));
            Tensor dbeta = ops::add(h, carry);
            Tensor direct = ops::mul(ops::mul(v, ops::exp(frame)), gr);
            if (zero_infinity) direct = ops::where(ignored, Scalar(0), direct);
            rows.push_back(ops::add(direct, class_sum(ops::sub(dbeta, h), seq.labels, frame)));
            alpha_rows.push_back(ops::cat({h, constant(lp, alpha.size(2) - h.size(0), 0)}, 0));
            if (t + 1 < seq.length) {
                Tensor next = beta[static_cast<size_t>(t + 1)];
                Tensor b1 = shift(next, -1, neginf);
                Tensor b2 = ops::add(shift(next, -2, neginf), seq.skip_backward);
                Tensor total = log_sum({next, b1, b2});
                carry = ops::add(ops::add(ops::mul(dbeta, weights(next, total)),
                    shift(ops::mul(dbeta, weights(b1, total)), 1, 0)),
                    shift(ops::mul(dbeta, weights(b2, total)), 2, 0));
            }
        }
        dgo.push_back(dg);
        dnll.push_back(dn);
        dlp.push_back(pad_frames(std::move(rows), batch));
        while (static_cast<int64_t>(alpha_rows.size()) < lp.size(0))
            alpha_rows.push_back(constant(lp, alpha.size(2), 0));
        dalpha.push_back(ops::stack(alpha_rows, 0));
    }
    if (dlp.empty()) return {ops::zeros_like(go), ops::zeros_like(lp), Tensor(), Tensor(), Tensor(),
                             ops::zeros_like(nll), ops::zeros_like(alpha)};
    return {ops::stack(dgo, 0), ops::stack(dlp, 1), Tensor(), Tensor(), Tensor(),
             ops::stack(dnll, 0), ops::stack(dalpha, 0)};
}

}  // namespace

CtcLossBackward::CtcLossBackward(Tensor lp, Tensor targets, Tensor il, Tensor tl,
    int64_t blank, bool zero_infinity, Tensor nll, Tensor alpha)
    : log_probs_(lp), targets_(targets), input_lengths_(il), target_lengths_(tl),
      nll_(nll, true), alpha_(alpha, true), blank_(blank), zero_infinity_(zero_infinity) {}

variable_list CtcLossBackward::apply(variable_list&& inputs) {
    variable_list result(4);
    if (!should_compute_output(0)) return result;
    Tensor lp = log_probs_.unpack(), targets = targets_.unpack();
    Tensor il = input_lengths_.unpack(), tl = target_lengths_.unpack();
    Tensor nll = nll_.unpack_output(shared_from_this(), 0);
    Tensor alpha = alpha_.unpack_output(shared_from_this(), 1);
    Tensor go = inputs.size() > 0 ? inputs[0] : Tensor();
    Tensor ga = inputs.size() > 1 ? inputs[1] : Tensor();
    if (go.defined()) {
        result[0] = GradMode::is_enabled()
            ? recorded_backward(go, lp, sequences(lp, targets, il, tl, blank_), zero_infinity_)
            : ops::_ctc_loss_backward(go, lp, targets, il, tl, nll, alpha, blank_, zero_infinity_);
    }
    if (ga.defined()) {
        Tensor extra = alpha_backward(ga, lp, sequences(lp, targets, il, tl, blank_));
        result[0] = result[0].defined() ? ops::add(result[0], extra) : extra;
    }
    return result;
}

void CtcLossBackward::release_variables() {
    Node::release_variables();
    log_probs_.reset_data(); targets_.reset_data(); input_lengths_.reset_data();
    target_lengths_.reset_data(); nll_.reset_data(); alpha_.reset_data();
}

CtcLossBackwardBackward::CtcLossBackwardBackward(Tensor go, Tensor lp, Tensor targets,
    Tensor il, Tensor tl, Tensor nll, Tensor alpha, int64_t blank, bool zero_infinity)
    : grad_output_(go), log_probs_(lp), targets_(targets), input_lengths_(il), target_lengths_(tl),
      nll_(nll), alpha_(alpha), blank_(blank), zero_infinity_(zero_infinity) {}

variable_list CtcLossBackwardBackward::apply(variable_list&& inputs) {
    if (inputs.empty() || !inputs[0].defined()) return variable_list(7);
    Tensor lp = log_probs_.unpack(), targets = targets_.unpack();
    Tensor il = input_lengths_.unpack(), tl = target_lengths_.unpack();
    auto result = backward_adjoints(inputs[0], grad_output_.unpack(), lp,
        nll_.unpack(), alpha_.unpack(), sequences(lp, targets, il, tl, blank_), zero_infinity_);
    for (size_t i = 0; i < result.size(); ++i)
        if (!should_compute_output(i)) result[i] = Tensor();
    return result;
}

void CtcLossBackwardBackward::release_variables() {
    Node::release_variables();
    grad_output_.reset_data(); log_probs_.reset_data(); targets_.reset_data();
    input_lengths_.reset_data(); target_lengths_.reset_data(); nll_.reset_data(); alpha_.reset_data();
}

}  // namespace tpx
}  // namespace tensorplay
