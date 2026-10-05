#pragma once
// Second derivatives through the convolution and normalization backward
// kernels.
//
// Convolution.  Every convolution is a bilinear map F(x, w), and its backward
// kernels are the two transposes: grad_input = F_x^T g, grad_weight = F_w^T g,
// grad_bias = sum of g over everything but the channel dimension.  Pairing a
// kernel's result with an incoming gradient gg and moving the transpose to the
// other side gives the derivatives:
//     <gg, F_x^T g> = <g, F(gg, w)>   ->  d/dg = F(gg, w),  d/dw = F_w^T(g | input gg)
//     <gg, F_w^T g> = <g, F(x, gg)>   ->  d/dg = F(x, gg),  d/dx = F_x^T(g | weight gg)
// so a second derivative is again a convolution or a convolution backward,
// and a third pass records as well.  The transposed convolutions use the same
// identities with the transposed flag set.
//
// Normalization.  Every layer, batch, instance and group norm normalizes rows
// of the input (a row is the set of elements sharing statistics).  With the
// normalized value xh = (x - mean) r, r = (var + eps)^-1/2, and a = gamma g,
// the input gradient is gi = T(a) where, per row,
//     T(u) = r (u - mean(u) - xh mean(u xh))
// is a symmetric linear map; grad_weight = sum(g xh) and grad_bias = sum(g)
// over the elements that share a parameter.  For incoming gradients
// (ggi, ggw, ggb):
//     d/dg     = gamma T(ggi) + ggw xh + ggb
//     d/dgamma = sum(g T(ggi))
//     d/dx     = -(r / n) (sum(ggi gi) xh + T(sv a + sa ggi)) + T(ggw g)
// with sv = sum(ggi xh), sa = sum(a xh) per row and n the row size; the row
// statistics are recomputed from the input with differentiable operations.
// With fixed statistics (eval mode) xh = (x - running mean) r is affine in x:
//     d/dg = gamma r ggi + ggw xh + ggb,  d/dgamma = sum(g r ggi),
//     d/dx = r ggw g.

#include "Autograd.h"
#include "GradMode.h"
#include "Node.h"
#include "SavedVariable.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace tpx {

namespace conv_bwd_detail {

// Spatial rank of a convolution weight (out, in / groups, spatial...).
inline int64_t spatial_rank(const Tensor& weight) { return weight.dim() - 2; }

// A stride / padding / dilation list as one entry per spatial dim; an empty
// list takes `fallback` and a single entry repeats.
inline std::vector<int64_t> per_dim(const std::vector<int64_t>& v, int64_t nd, int64_t fallback) {
    if (static_cast<int64_t>(v.size()) == nd) return v;
    if (v.empty()) return std::vector<int64_t>(static_cast<size_t>(nd), fallback);
    if (v.size() == 1) return std::vector<int64_t>(static_cast<size_t>(nd), v[0]);
    return v;
}

struct Params {
    std::vector<int64_t> stride, padding, dilation, output_padding;
    bool transposed;
    int64_t groups;
};

inline Params make_params(const Tensor& weight, const std::vector<int64_t>& stride,
                          const std::vector<int64_t>& padding,
                          const std::vector<int64_t>& dilation, bool transposed,
                          const std::vector<int64_t>& output_padding, int64_t groups) {
    const int64_t nd = spatial_rank(weight);
    return {per_dim(stride, nd, 1), per_dim(padding, nd, 0), per_dim(dilation, nd, 1),
            per_dim(output_padding, nd, 0), transposed, groups};
}

// F(x, w): the forward map the backward kernels transpose.
inline Tensor forward(const Tensor& x, const Tensor& w, const Params& p) {
    return ops::convolution(x, w, std::nullopt, p.stride, p.padding, p.dilation, p.transposed,
                            p.output_padding, p.groups);
}

// F_x^T g with weight `w`.
inline Tensor input_grad(const Tensor& g, const Tensor& x, const Tensor& w, const Params& p) {
    return std::get<0>(ops::convolution_backward(g, x, w, std::nullopt, p.stride, p.padding,
                                                 p.dilation, p.transposed, p.output_padding,
                                                 p.groups, {true, false, false}));
}

// F_w^T g with input `x`.
inline Tensor weight_grad(const Tensor& g, const Tensor& x, const Tensor& w, const Params& p) {
    return std::get<1>(ops::convolution_backward(g, x, w, std::nullopt, p.stride, p.padding,
                                                 p.dilation, p.transposed, p.output_padding,
                                                 p.groups, {false, true, false}));
}

// The bias gradient is a sum over everything but dim 1; its transpose repeats
// the incoming per-channel gradient over all of them.
inline Tensor bias_repeat(const Tensor& gg, const Tensor& like) {
    std::vector<int64_t> shape(static_cast<size_t>(like.dim()), 1);
    shape[1] = -1;
    return ops::expand(ops::reshape(gg, shape), like.shape());
}

}  // namespace conv_bwd_detail

// Single-output kernels (the conv*_grad_input / _grad_weight / _grad_bias
// family): helpers for the formulas in derivatives.yaml.

// d grad_input / d grad_output: the forward map applied to the incoming gradient.
inline Tensor conv_grad_input_wrt_grad_output(const Tensor& gg, const Tensor& weight,
                                              const std::vector<int64_t>& stride,
                                              const std::vector<int64_t>& padding,
                                              const std::vector<int64_t>& dilation,
                                              bool transposed,
                                              const std::vector<int64_t>& output_padding,
                                              int64_t groups) {
    const auto p = conv_bwd_detail::make_params(weight, stride, padding, dilation, transposed,
                                                output_padding, groups);
    return conv_bwd_detail::forward(gg, weight, p);
}

// d grad_input / d weight: the weight gradient with the incoming gradient as input.
inline Tensor conv_grad_input_wrt_weight(const Tensor& gg, const Tensor& grad_output,
                                         const Tensor& weight,
                                         const std::vector<int64_t>& stride,
                                         const std::vector<int64_t>& padding,
                                         const std::vector<int64_t>& dilation, bool transposed,
                                         const std::vector<int64_t>& output_padding,
                                         int64_t groups) {
    const auto p = conv_bwd_detail::make_params(weight, stride, padding, dilation, transposed,
                                                output_padding, groups);
    return conv_bwd_detail::weight_grad(grad_output, gg, weight, p);
}

// d grad_weight / d grad_output: the forward map with the incoming gradient as weight.
inline Tensor conv_grad_weight_wrt_grad_output(const Tensor& gg, const Tensor& input,
                                               const std::vector<int64_t>& stride,
                                               const std::vector<int64_t>& padding,
                                               const std::vector<int64_t>& dilation,
                                               bool transposed,
                                               const std::vector<int64_t>& output_padding,
                                               int64_t groups) {
    const auto p = conv_bwd_detail::make_params(gg, stride, padding, dilation, transposed,
                                                output_padding, groups);
    return conv_bwd_detail::forward(input, gg, p);
}

// d grad_weight / d input: the input gradient with the incoming gradient as weight.
inline Tensor conv_grad_weight_wrt_input(const Tensor& gg, const Tensor& grad_output,
                                         const Tensor& input,
                                         const std::vector<int64_t>& stride,
                                         const std::vector<int64_t>& padding,
                                         const std::vector<int64_t>& dilation, bool transposed,
                                         const std::vector<int64_t>& output_padding,
                                         int64_t groups) {
    const auto p = conv_bwd_detail::make_params(gg, stride, padding, dilation, transposed,
                                                output_padding, groups);
    return conv_bwd_detail::input_grad(grad_output, input, gg, p);
}

// d grad_bias / d grad_output.
inline Tensor conv_grad_bias_wrt_grad_output(const Tensor& gg, const Tensor& grad_output) {
    return conv_bwd_detail::bias_repeat(gg, grad_output);
}

namespace conv_bwd_detail {

// The three-output convolution backward: shared by both spellings.
inline variable_list convolution_double_backward(
        Node& node, const variable_list& inputs, const Tensor& grad_output, const Tensor& input,
        const Tensor& weight, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation,
        bool transposed, const std::vector<int64_t>& output_padding, int64_t groups) {
    const Tensor ggi = inputs.size() > 0 ? inputs[0] : Tensor();
    const Tensor ggw = inputs.size() > 1 ? inputs[1] : Tensor();
    const Tensor ggb = inputs.size() > 2 ? inputs[2] : Tensor();
    variable_list grads(3);
    if (!ggi.defined() && !ggw.defined() && !ggb.defined()) return grads;
    if (!grad_output.defined() || !input.defined() || !weight.defined()) return grads;
    const Params p = make_params(weight, stride, padding, dilation, transposed, output_padding,
                                 groups);
    if (node.should_compute_output(0)) {
        Tensor g;
        auto accumulate = [&g](Tensor t) { g = g.defined() ? ops::add(g, t) : std::move(t); };
        if (ggi.defined()) accumulate(forward(ggi, weight, p));
        if (ggw.defined()) accumulate(forward(input, ggw, p));
        if (ggb.defined()) accumulate(bias_repeat(ggb, grad_output));
        grads[0] = g;
    }
    if (node.should_compute_output(1) && ggw.defined()) {
        grads[1] = input_grad(grad_output, input, ggw, p);
    }
    if (node.should_compute_output(2) && ggi.defined()) {
        grads[2] = weight_grad(grad_output, ggi, weight, p);
    }
    return grads;
}

}  // namespace conv_bwd_detail

struct ConvolutionBackwardBackward : public Node {
    SavedVariable grad_output_;
    SavedVariable input_;
    SavedVariable weight_;
    std::vector<int64_t> stride_;
    std::vector<int64_t> padding_;
    std::vector<int64_t> dilation_;
    bool transposed_;
    std::vector<int64_t> output_padding_;
    int64_t groups_;

    ConvolutionBackwardBackward(Tensor grad_output, Tensor input, Tensor weight,
                                std::vector<int64_t> stride, std::vector<int64_t> padding,
                                std::vector<int64_t> dilation, bool transposed,
                                std::vector<int64_t> output_padding, int64_t groups)
        : grad_output_(std::move(grad_output)), input_(std::move(input)),
          weight_(std::move(weight)), stride_(std::move(stride)), padding_(std::move(padding)),
          dilation_(std::move(dilation)), transposed_(transposed),
          output_padding_(std::move(output_padding)), groups_(groups) {}

    size_t num_inputs() const override { return 3; }

    variable_list apply(variable_list&& inputs) override {
        return conv_bwd_detail::convolution_double_backward(
            *this, inputs, grad_output_.unpack(), input_.unpack(), weight_.unpack(), stride_,
            padding_, dilation_, transposed_, output_padding_, groups_);
    }

    void release_variables() override {
        Node::release_variables();
        grad_output_.reset_data();
        input_.reset_data();
        weight_.reset_data();
    }
};

// The overrideable spelling carries no bias sizes; the maths is the same.
struct ConvolutionBackwardOverrideableBackward : public ConvolutionBackwardBackward {
    using ConvolutionBackwardBackward::ConvolutionBackwardBackward;
};

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

namespace norm_bwd_detail {

// How a normalization lays its rows out.  The input is viewed as `shape`;
// `row_dims` are the dims of one row; the affine parameter is viewed as
// `param_view` to broadcast onto it and parameter gradients sum over
// `param_dims` and take the shape `param_shape`.
struct Layout {
    std::vector<int64_t> shape;
    std::vector<int64_t> row_dims;
    std::vector<int64_t> param_view;
    std::vector<int64_t> param_dims;
    std::vector<int64_t> param_shape;
};

// Channels at dim 1 (batch and instance norm): the parameter is a [C] vector.
inline Layout channel_layout(const Tensor& x, bool rows_over_batch) {
    Layout l;
    l.shape = x.shape();
    const int64_t nd = x.dim();
    l.param_view.assign(static_cast<size_t>(nd), 1);
    l.param_view[1] = x.size(1);
    l.param_shape = {x.size(1)};
    for (int64_t d = 0; d < nd; ++d) {
        if (d != 1) l.param_dims.push_back(d);
        if (rows_over_batch ? d != 1 : d >= 2) l.row_dims.push_back(d);
    }
    return l;
}

// Rows over the trailing normalized_shape dims; the parameter has that shape.
inline Layout layer_layout(const Tensor& x, const std::vector<int64_t>& normalized_shape) {
    int64_t inner = 1;
    for (int64_t s : normalized_shape) inner *= s;
    const int64_t outer = inner == 0 ? 0 : x.numel() / inner;
    Layout l;
    l.shape = {outer, inner};
    l.row_dims = {1};
    l.param_view = {1, inner};
    l.param_dims = {0};
    l.param_shape = normalized_shape;
    return l;
}

// Rows over a group of channels and all spatial positions.
inline Layout group_layout(const Tensor& x, int64_t groups) {
    const int64_t n = x.size(0);
    const int64_t c = x.size(1);
    int64_t spatial = 1;
    for (int64_t d = 2; d < x.dim(); ++d) spatial *= x.size(d);
    Layout l;
    l.shape = {n, groups, c / groups, spatial};
    l.row_dims = {2, 3};
    l.param_view = {1, groups, c / groups, 1};
    l.param_dims = {0, 3};
    l.param_shape = {c};
    return l;
}

inline Tensor to_view(const Tensor& t, const Layout& l) { return ops::reshape(t, l.shape); }

inline Tensor param_broadcast(const Tensor& p, const Layout& l) {
    return ops::reshape(p, l.param_view);
}

// Sum a view-shaped tensor down to the parameter's shape.
inline Tensor param_sum(const Tensor& t, const Layout& l) {
    return ops::reshape(ops::sum(t, l.param_dims, false), l.param_shape);
}

inline bool has(const std::optional<Tensor>& t) { return t.has_value() && t->defined(); }

// The normalized rows and their reciprocal deviation r, recomputed from the
// input.
struct Rows {
    Tensor xh;
    Tensor r;
};

inline Rows batch_rows(const Tensor& x, const Layout& l, double eps) {
    const Tensor centered = ops::sub(x, ops::mean(x, l.row_dims, true));
    const Tensor var = ops::mean(ops::square(centered), l.row_dims, true);
    const Tensor r = ops::rsqrt(ops::add(var, Scalar(eps)));
    return {ops::mul(centered, r), r};
}

// The forward saved the reciprocal deviation but not the epsilon it used; the
// epsilon that reproduces it is read back (as a constant) so the statistics
// stay differentiable in the input.
inline Rows saved_rstd_rows(const Tensor& x, const Layout& l, const Tensor& rstd) {
    const Tensor centered = ops::sub(x, ops::mean(x, l.row_dims, true));
    const Tensor var = ops::mean(ops::square(centered), l.row_dims, true);
    std::vector<int64_t> keep = l.shape;
    for (int64_t d : l.row_dims) keep[static_cast<size_t>(d)] = 1;
    const Tensor rs = ops::reshape(ops::to(rstd, x.dtype()), keep);
    const Tensor eps = ops::detach(ops::sub(ops::pow(rs, Scalar(-2.0)), var));
    const Tensor r = ops::rsqrt(ops::add(var, eps));
    return {ops::mul(centered, r), r};
}

// Running statistics broadcast to the view, as fixed constants.
inline Rows fixed_rows(const Tensor& x, const Layout& l, const Tensor& running_mean,
                       const Tensor& running_var, double eps) {
    const Tensor mean = param_broadcast(ops::to(running_mean, x.dtype()), l);
    const Tensor r = ops::rsqrt(ops::add(param_broadcast(ops::to(running_var, x.dtype()), l),
                                         Scalar(eps)));
    return {ops::mul(ops::sub(x, mean), r), r};
}

struct Grads {
    Tensor grad_output;
    Tensor input;
    Tensor weight;
};

// The row-statistics case; see the derivation at the top of the file.
inline Grads row_double_backward(const Tensor& g_in, const Tensor& x_in,
                                 const std::optional<Tensor>& weight, const Tensor& ggi_in,
                                 const Tensor& ggw, const Tensor& ggb, const Layout& l,
                                 const Rows& rows, bool fixed_stats, bool need_g, bool need_x,
                                 bool need_w) {
    const Tensor g = to_view(g_in, l);
    const Tensor ggi = ggi_in.defined() ? to_view(ggi_in, l) : Tensor();
    Tensor gamma;
    if (has(weight)) gamma = param_broadcast(ops::to(*weight, g.dtype()), l);
    const Tensor& xh = rows.xh;
    const Tensor& r = rows.r;
    const int64_t n = x_in.numel() / std::max<int64_t>(r.numel(), 1);
    Grads out;

    // T(u) = r (u - mean(u) - xh mean(u xh)), the row projection of the kernel.
    auto project = [&](const Tensor& u) {
        const Tensor centered = ops::sub(u, ops::mean(u, l.row_dims, true));
        const Tensor along = ops::mul(xh, ops::mean(ops::mul(u, xh), l.row_dims, true));
        return ops::mul(ops::sub(centered, along), r);
    };

    Tensor tg;  // the input-gradient map applied to ggi
    if (ggi.defined()) {
        tg = fixed_stats ? ops::mul(ggi, r) : project(ggi);
    }
    Tensor ggw_b = ggw.defined() ? param_broadcast(ops::to(ggw, g.dtype()), l) : Tensor();

    if (need_g) {
        Tensor acc;
        auto add = [&acc](Tensor t) { acc = acc.defined() ? ops::add(acc, t) : std::move(t); };
        if (tg.defined()) add(gamma.defined() ? ops::mul(tg, gamma) : tg);
        if (ggw_b.defined()) add(ops::mul(ggw_b, xh));
        if (ggb.defined()) add(ops::expand(param_broadcast(ops::to(ggb, g.dtype()), l), l.shape));
        if (acc.defined()) out.grad_output = ops::reshape(acc, g_in.shape());
    }
    if (need_w && has(weight) && tg.defined()) {
        out.weight = ops::to(param_sum(ops::mul(g, tg), l), weight->dtype());
    }
    if (need_x) {
        Tensor acc;
        auto add = [&acc](Tensor t) { acc = acc.defined() ? ops::add(acc, t) : std::move(t); };
        if (fixed_stats) {
            if (ggw_b.defined()) add(ops::mul(ops::mul(ggw_b, g), r));
        } else {
            if (ggi.defined()) {
                const Tensor a = gamma.defined() ? ops::mul(g, gamma) : g;
                const Tensor gi = project(a);
                const Tensor along_rows = ops::sum(ops::mul(ggi, gi), l.row_dims, true);
                const Tensor sv = ops::sum(ops::mul(ggi, xh), l.row_dims, true);
                const Tensor sa = ops::sum(ops::mul(a, xh), l.row_dims, true);
                const Tensor mixed = project(ops::add(ops::mul(a, sv), ops::mul(ggi, sa)));
                const Tensor inner = ops::add(ops::mul(xh, along_rows), mixed);
                add(ops::neg(ops::div(ops::mul(inner, r), Scalar(static_cast<double>(n)))));
            }
            if (ggw_b.defined()) add(project(ops::mul(ggw_b, g)));
        }
        if (acc.defined()) out.input = ops::reshape(acc, x_in.shape());
    }
    return out;
}

}  // namespace norm_bwd_detail

// The shared shape of the norm backward nodes: the incoming gradients are
// (input, weight, bias), the node's edges follow the schema's tensor
// arguments, and `slot` places each of grad_output / input / weight among
// them.
struct NormBackwardBackwardBase : public Node {
    static variable_list place(size_t count, size_t g_slot, size_t x_slot, size_t w_slot,
                               const norm_bwd_detail::Grads& grads) {
        variable_list out(count);
        out[g_slot] = grads.grad_output;
        out[x_slot] = grads.input;
        out[w_slot] = grads.weight;
        return out;
    }
};

struct BatchNormBackwardBackward : public NormBackwardBackwardBase {
    SavedVariable grad_output_;
    SavedVariable input_;
    SavedVariable weight_;
    SavedVariable running_mean_;
    SavedVariable running_var_;
    bool training_;
    double eps_;

    BatchNormBackwardBackward(Tensor grad_output, Tensor input, std::optional<Tensor> weight,
                              std::optional<Tensor> running_mean,
                              std::optional<Tensor> running_var, bool training, double eps)
        : grad_output_(std::move(grad_output)), input_(std::move(input)),
          weight_(norm_bwd_detail::has(weight) ? *weight : Tensor()),
          running_mean_(norm_bwd_detail::has(running_mean) ? *running_mean : Tensor()),
          running_var_(norm_bwd_detail::has(running_var) ? *running_var : Tensor()),
          training_(training), eps_(eps) {}

    size_t num_inputs() const override { return 3; }

    variable_list apply(variable_list&& inputs) override {
        using namespace norm_bwd_detail;
        const Tensor ggi = inputs.size() > 0 ? inputs[0] : Tensor();
        const Tensor ggw = inputs.size() > 1 ? inputs[1] : Tensor();
        const Tensor ggb = inputs.size() > 2 ? inputs[2] : Tensor();
        if (!ggi.defined() && !ggw.defined() && !ggb.defined()) return variable_list(5);
        const Tensor g = grad_output_.unpack();
        const Tensor x = input_.unpack();
        const Tensor w = weight_.unpack();
        if (!g.defined() || !x.defined()) return variable_list(5);
        const Layout l = channel_layout(x, true);
        const Tensor xv = to_view(x, l);
        Rows rows;
        if (training_) {
            rows = batch_rows(xv, l, eps_);
        } else {
            rows = fixed_rows(xv, l, running_mean_.unpack(), running_var_.unpack(), eps_);
        }
        std::optional<Tensor> weight = w.defined() ? std::optional<Tensor>(w) : std::nullopt;
        const Grads grads = row_double_backward(
            g, x, weight, ggi, ggw, ggb, l, rows, !training_, should_compute_output(0),
            should_compute_output(1), should_compute_output(2));
        // Edges: grad_output, input, weight, running_mean, running_var.
        return place(5, 0, 1, 2, grads);
    }

    void release_variables() override {
        Node::release_variables();
        grad_output_.reset_data();
        input_.reset_data();
        weight_.reset_data();
        running_mean_.reset_data();
        running_var_.reset_data();
    }
};

struct InstanceNormBackwardBackward : public NormBackwardBackwardBase {
    SavedVariable grad_output_;
    SavedVariable input_;
    SavedVariable weight_;
    SavedVariable running_mean_;
    SavedVariable running_var_;
    bool use_input_stats_;
    double eps_;

    InstanceNormBackwardBackward(Tensor grad_output, Tensor input, std::optional<Tensor> weight,
                                 std::optional<Tensor> bias, std::optional<Tensor> running_mean,
                                 std::optional<Tensor> running_var, bool use_input_stats,
                                 double eps)
        : grad_output_(std::move(grad_output)), input_(std::move(input)),
          weight_(norm_bwd_detail::has(weight) ? *weight : Tensor()),
          running_mean_(norm_bwd_detail::has(running_mean) ? *running_mean : Tensor()),
          running_var_(norm_bwd_detail::has(running_var) ? *running_var : Tensor()),
          use_input_stats_(use_input_stats), eps_(eps) {}

    size_t num_inputs() const override { return 3; }

    variable_list apply(variable_list&& inputs) override {
        using namespace norm_bwd_detail;
        const Tensor ggi = inputs.size() > 0 ? inputs[0] : Tensor();
        const Tensor ggw = inputs.size() > 1 ? inputs[1] : Tensor();
        const Tensor ggb = inputs.size() > 2 ? inputs[2] : Tensor();
        // Edges: grad_output, input, weight, bias, running_mean, running_var.
        if (!ggi.defined() && !ggw.defined() && !ggb.defined()) return variable_list(6);
        const Tensor g = grad_output_.unpack();
        const Tensor x = input_.unpack();
        const Tensor w = weight_.unpack();
        if (!g.defined() || !x.defined()) return variable_list(6);
        const Layout l = channel_layout(x, false);
        const Tensor xv = to_view(x, l);
        Rows rows;
        if (use_input_stats_) {
            rows = batch_rows(xv, l, eps_);
        } else {
            rows = fixed_rows(xv, l, running_mean_.unpack(), running_var_.unpack(), eps_);
        }
        std::optional<Tensor> weight = w.defined() ? std::optional<Tensor>(w) : std::nullopt;
        const Grads grads = row_double_backward(
            g, x, weight, ggi, ggw, ggb, l, rows, !use_input_stats_, should_compute_output(0),
            should_compute_output(1), should_compute_output(2));
        return place(6, 0, 1, 2, grads);
    }

    void release_variables() override {
        Node::release_variables();
        grad_output_.reset_data();
        input_.reset_data();
        weight_.reset_data();
        running_mean_.reset_data();
        running_var_.reset_data();
    }
};

struct GroupNormBackwardBackward : public NormBackwardBackwardBase {
    SavedVariable grad_output_;
    SavedVariable input_;
    int64_t num_groups_;
    SavedVariable weight_;
    double eps_;

    GroupNormBackwardBackward(Tensor grad_output, Tensor input, int64_t num_groups,
                              std::optional<Tensor> weight, std::optional<Tensor> bias,
                              double eps)
        : grad_output_(std::move(grad_output)), input_(std::move(input)),
          num_groups_(num_groups),
          weight_(norm_bwd_detail::has(weight) ? *weight : Tensor()), eps_(eps) {}

    size_t num_inputs() const override { return 3; }

    variable_list apply(variable_list&& inputs) override {
        using namespace norm_bwd_detail;
        const Tensor ggi = inputs.size() > 0 ? inputs[0] : Tensor();
        const Tensor ggw = inputs.size() > 1 ? inputs[1] : Tensor();
        const Tensor ggb = inputs.size() > 2 ? inputs[2] : Tensor();
        // Edges: grad_output, input, weight, bias.
        if (!ggi.defined() && !ggw.defined() && !ggb.defined()) return variable_list(4);
        const Tensor g = grad_output_.unpack();
        const Tensor x = input_.unpack();
        const Tensor w = weight_.unpack();
        if (!g.defined() || !x.defined()) return variable_list(4);
        const Layout l = group_layout(x, num_groups_);
        const Rows rows = batch_rows(to_view(x, l), l, eps_);
        std::optional<Tensor> weight = w.defined() ? std::optional<Tensor>(w) : std::nullopt;
        const Grads grads = row_double_backward(
            g, x, weight, ggi, ggw, ggb, l, rows, false, should_compute_output(0),
            should_compute_output(1), should_compute_output(2));
        return place(4, 0, 1, 2, grads);
    }

    void release_variables() override {
        Node::release_variables();
        grad_output_.reset_data();
        input_.reset_data();
        weight_.reset_data();
    }
};

struct NativeGroupNormBackwardBackward : public NormBackwardBackwardBase {
    SavedVariable grad_out_;
    SavedVariable input_;
    SavedVariable mean_;
    SavedVariable rstd_;
    SavedVariable weight_;
    int64_t group_;

    NativeGroupNormBackwardBackward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd,
                                    std::optional<Tensor> weight, int64_t N, int64_t C,
                                    int64_t HxW, int64_t group)
        : grad_out_(std::move(grad_out)), input_(std::move(input)), mean_(std::move(mean)),
          rstd_(std::move(rstd)), weight_(norm_bwd_detail::has(weight) ? *weight : Tensor()),
          group_(group) {
        (void)N;
        (void)C;
        (void)HxW;
    }

    size_t num_inputs() const override { return 3; }

    variable_list apply(variable_list&& inputs) override {
        using namespace norm_bwd_detail;
        const Tensor ggi = inputs.size() > 0 ? inputs[0] : Tensor();
        const Tensor ggw = inputs.size() > 1 ? inputs[1] : Tensor();
        const Tensor ggb = inputs.size() > 2 ? inputs[2] : Tensor();
        // Edges: grad_out, input, mean, rstd, weight.
        if (!ggi.defined() && !ggw.defined() && !ggb.defined()) return variable_list(5);
        const Tensor g = grad_out_.unpack();
        const Tensor x = input_.unpack();
        const Tensor rstd = rstd_.unpack();
        const Tensor w = weight_.unpack();
        if (!g.defined() || !x.defined() || !rstd.defined()) return variable_list(5);
        const Layout l = group_layout(x, group_);
        const Rows rows = saved_rstd_rows(to_view(x, l), l, rstd);
        std::optional<Tensor> weight = w.defined() ? std::optional<Tensor>(w) : std::nullopt;
        const Grads grads = row_double_backward(
            g, x, weight, ggi, ggw, ggb, l, rows, false, should_compute_output(0),
            should_compute_output(1), should_compute_output(4));
        return place(5, 0, 1, 4, grads);
    }

    void release_variables() override {
        Node::release_variables();
        grad_out_.reset_data();
        input_.reset_data();
        mean_.reset_data();
        rstd_.reset_data();
        weight_.reset_data();
    }
};

struct NativeLayerNormBackwardBackward : public NormBackwardBackwardBase {
    SavedVariable grad_out_;
    SavedVariable input_;
    std::vector<int64_t> normalized_shape_;
    SavedVariable mean_;
    SavedVariable rstd_;
    SavedVariable weight_;

    NativeLayerNormBackwardBackward(Tensor grad_out, Tensor input,
                                    std::vector<int64_t> normalized_shape, Tensor mean,
                                    Tensor rstd, std::optional<Tensor> weight,
                                    std::optional<Tensor> bias)
        : grad_out_(std::move(grad_out)), input_(std::move(input)),
          normalized_shape_(std::move(normalized_shape)), mean_(std::move(mean)),
          rstd_(std::move(rstd)), weight_(norm_bwd_detail::has(weight) ? *weight : Tensor()) {
        (void)bias;
    }

    size_t num_inputs() const override { return 3; }

    variable_list apply(variable_list&& inputs) override {
        using namespace norm_bwd_detail;
        const Tensor ggi = inputs.size() > 0 ? inputs[0] : Tensor();
        const Tensor ggw = inputs.size() > 1 ? inputs[1] : Tensor();
        const Tensor ggb = inputs.size() > 2 ? inputs[2] : Tensor();
        // Edges: grad_out, input, mean, rstd, weight, bias.
        if (!ggi.defined() && !ggw.defined() && !ggb.defined()) return variable_list(6);
        const Tensor g = grad_out_.unpack();
        const Tensor x = input_.unpack();
        const Tensor rstd = rstd_.unpack();
        const Tensor w = weight_.unpack();
        if (!g.defined() || !x.defined() || !rstd.defined()) return variable_list(6);
        const Layout l = layer_layout(x, normalized_shape_);
        const Rows rows = saved_rstd_rows(to_view(x, l), l, rstd);
        std::optional<Tensor> weight = w.defined() ? std::optional<Tensor>(w) : std::nullopt;
        const Grads grads = row_double_backward(
            g, x, weight, ggi, ggw, ggb, l, rows, false, should_compute_output(0),
            should_compute_output(1), should_compute_output(4));
        return place(6, 0, 1, 4, grads);
    }

    void release_variables() override {
        Node::release_variables();
        grad_out_.reset_data();
        input_.reset_data();
        mean_.reset_data();
        rstd_.reset_data();
        weight_.reset_data();
    }
};

}  // namespace tpx
}  // namespace tensorplay
