#pragma once
// Second derivatives through the grid sampler backward kernels.
//
// grid_sampler_{2,3}d_backward(grad_output, input, grid) returns
// (grad_input, grad_grid).  grad_input is linear in grad_output and reads
// nothing of input; grad_grid is bilinear in grad_output and input and, for
// a bicubic kernel, curved in grid.  Given the incoming gradients of the two
// outputs (ggI, ggGrid) the node below returns the gradients with respect to
// grad_output, input and grid.  Every term is written with differentiable
// operations, so a further pass records as well.

#include "Autograd.h"
#include "GradMode.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <array>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace tpx {
namespace grid_bwd_detail {

enum : int64_t { kBilinear = 0, kNearest = 1, kBicubic = 2 };
enum : int64_t { kZeros = 0, kBorder = 1, kReflection = 2 };

inline Tensor as_dtype(const Tensor& mask, DType dtype) { return mask.to(dtype); }

inline Tensor add_defined(const Tensor& a, const Tensor& b) {
    return a.defined() ? ops::add(a, b) : b;
}

// The unnormalized source coordinate of one grid component and its slope
// d(source)/d(grid), padding included: border clamps (flat outside the
// interior), reflection folds (the slope changes sign with each fold).
inline std::pair<Tensor, Tensor> source_coords(const Tensor& coord, int64_t size,
                                               int64_t padding, bool align_corners) {
    const double scale = static_cast<double>(align_corners ? size - 1 : size) / 2.0;
    Tensor ix = ops::mul(ops::add(coord, 1.0), scale);
    if (!align_corners) ix = ops::sub(ix, 0.5);
    const DType dt = ix.dtype();
    const double hi = static_cast<double>(size - 1);
    Tensor slope;
    if (padding == kZeros) {
        slope = ops::ones_like(ix);
    } else if (padding == kBorder) {
        slope = as_dtype(ops::logical_and(ops::gt(ix, 0.0), ops::lt(ix, hi)), dt);
        ix = ops::clamp(ix, Scalar(0.0), Scalar(hi));
    } else {
        const double twice_low = align_corners ? 0.0 : -1.0;
        const double twice_high = static_cast<double>(2 * size) - (align_corners ? 2.0 : 1.0);
        if (twice_high <= twice_low) {
            Tensor z = ops::zeros_like(ix);
            return {z, z};
        }
        const double low = twice_low / 2.0;
        const double span = (twice_high - twice_low) / 2.0;
        Tensor shifted = ops::sub(ix, low);
        Tensor dist = ops::abs(shifted);
        Tensor side = ops::sub(ops::mul(as_dtype(ops::ge(shifted, 0.0), dt), 2.0), 1.0);
        Tensor even = ops::lt(ops::fmod(ops::floor(ops::div(dist, span)), 2.0), 0.5);
        Tensor flip = ops::sub(ops::mul(as_dtype(even, dt), 2.0), 1.0);
        Tensor rem = ops::fmod(dist, span);
        Tensor folded = ops::where(even, ops::add(rem, low), ops::add(ops::rsub(rem, span), low));
        Tensor inside = as_dtype(ops::logical_and(ops::gt(folded, 0.0), ops::lt(folded, hi)), dt);
        slope = ops::mul(ops::mul(side, flip), inside);
        ix = ops::clamp(folded, Scalar(0.0), Scalar(hi));
    }
    return {ix, ops::mul(slope, scale)};
}

// A bicubic tap index brought inside [0, size) the way the forward reads it.
inline Tensor bound_tap(const Tensor& idx, int64_t size, int64_t padding, bool align_corners) {
    if (padding != kReflection) return ops::clamp(idx, Scalar(0), Scalar(size - 1));
    if (size <= 1) return ops::zeros_like(idx);
    const double span = static_cast<double>(align_corners ? size - 1 : size);
    const double low = align_corners ? 0.0 : -0.5;
    Tensor dist = ops::abs(ops::sub(idx.to(DType::Float64), low));
    Tensor even = ops::lt(ops::fmod(ops::floor(ops::div(dist, span)), 2.0), 0.5);
    Tensor rem = ops::fmod(dist, span);
    Tensor folded = ops::where(even, rem, ops::rsub(rem, span));
    return ops::clamp(ops::round(ops::add(folded, low)), Scalar(0.0),
                      Scalar(static_cast<double>(size - 1)))
        .to(DType::Int64);
}

// Flat (spatial) indices of K taps per output position, ready to gather from
// or scatter into an (N, C, prod(spatial)) view.
inline Tensor flat_taps(const Tensor& flat, int64_t N, int64_t C) {
    const int64_t taps = flat.numel() / N;
    return ops::expand(ops::reshape(flat, {N, 1, taps}), {N, C, taps}).contiguous();
}

// Zero for taps outside the input when padding with zeros.
inline Tensor in_bounds(const std::vector<std::pair<Tensor, int64_t>>& axes) {
    Tensor mask;
    for (const auto& [idx, size] : axes) {
        Tensor m = ops::logical_and(ops::ge(idx, 0), ops::lt(idx, size));
        mask = mask.defined() ? ops::logical_and(mask, m) : m;
    }
    return mask;
}

// Gathers K taps per output position: (N, C, *out, K).
inline Tensor gather_taps(const Tensor& input, const Tensor& flat, const Tensor& mask,
                          std::vector<int64_t> out_shape) {
    const int64_t N = input.size(0), C = input.size(1);
    const int64_t spatial = input.numel() / std::max<int64_t>(N * C, 1);
    Tensor taps = ops::gather(ops::reshape(input, {N, C, spatial}), 2, flat_taps(flat, N, C));
    out_shape.insert(out_shape.begin(), {N, C});
    taps = ops::reshape(taps, out_shape);
    if (mask.defined()) taps = ops::mul(taps, as_dtype(ops::unsqueeze(mask, 1), taps.dtype()));
    return taps;
}

// The adjoint of gather_taps: values (N, C, *out) weighted per tap by
// weights (N, *out, K), summed into the input shape.
inline Tensor scatter_taps(const Tensor& values, const Tensor& weights, const Tensor& flat,
                           const Tensor& mask, const std::vector<int64_t>& input_shape) {
    const int64_t N = values.size(0), C = values.size(1);
    int64_t spatial = 1;
    for (size_t d = 2; d < input_shape.size(); ++d) spatial *= input_shape[d];
    Tensor weighted = ops::mul(ops::unsqueeze(values, -1), ops::unsqueeze(weights, 1));
    if (mask.defined()) {
        weighted = ops::mul(weighted, as_dtype(ops::unsqueeze(mask, 1), weighted.dtype()));
    }
    const int64_t taps = weighted.numel() / std::max<int64_t>(N * C, 1);
    Tensor out = ops::zeros({N, C, spatial}, values.dtype(), values.device());
    out = ops::scatter_add(out, 2, flat_taps(flat, N, C), ops::reshape(weighted, {N, C, taps}));
    return ops::reshape(out, input_shape);
}

// sum over taps of values (N, C, *out, K) times basis (N, *out, K).
inline Tensor tap_sum(const Tensor& values, const Tensor& basis) {
    return ops::sum(ops::mul(values, ops::unsqueeze(basis, 1)), std::vector<int64_t>{-1});
}

// [..., j * K + i] = b[..., j] * a[..., i]
inline Tensor outer_last(const Tensor& a, const Tensor& b) {
    std::vector<int64_t> shape = a.sizes();
    shape.back() *= b.size(-1);
    return ops::reshape(ops::mul(ops::unsqueeze(b, -1), ops::unsqueeze(a, -2)), shape);
}

inline Tensor stack_last(const std::vector<Tensor>& parts) { return ops::stack(parts, -1); }

// The cubic convolution kernel (a = -0.75) on the two outer and two inner
// taps, and its first and second derivatives in the tap distance.
constexpr double kCubicA = -0.75;
inline Tensor cubic_outer(const Tensor& t) {
    const double a = kCubicA;
    return ops::sub(ops::mul(ops::add(ops::mul(ops::sub(ops::mul(t, a), 5 * a), t), 8 * a), t), 4 * a);
}
inline Tensor cubic_inner(const Tensor& t) {
    const double a = kCubicA;
    return ops::add(ops::mul(ops::sub(ops::mul(t, a + 2), a + 3), ops::square(t)), 1.0);
}
inline Tensor dcubic_outer(const Tensor& t) {
    const double a = kCubicA;
    return ops::add(ops::mul(ops::sub(ops::mul(t, 3 * a), 10 * a), t), 8 * a);
}
inline Tensor dcubic_inner(const Tensor& t) {
    const double a = kCubicA;
    return ops::mul(ops::sub(ops::mul(t, 3 * (a + 2)), 2 * (a + 3)), t);
}
inline Tensor d2cubic_outer(const Tensor& t) {
    const double a = kCubicA;
    return ops::sub(ops::mul(t, 6 * a), 10 * a);
}
inline Tensor d2cubic_inner(const Tensor& t) {
    const double a = kCubicA;
    return ops::sub(ops::mul(t, 6 * (a + 2)), 2 * (a + 3));
}

// Weights of the four taps at offsets -1, 0, 1, 2 from floor(x), with f the
// fractional part, and their first and second derivatives in x.
struct CubicWeights {
    Tensor w, dw, d2w;
};
inline CubicWeights cubic_weights(const Tensor& f) {
    Tensor f1 = ops::add(f, 1.0), f2 = ops::rsub(f, 1.0), f3 = ops::rsub(f, 2.0);
    return {stack_last({cubic_outer(f1), cubic_inner(f), cubic_inner(f2), cubic_outer(f3)}),
            stack_last({dcubic_outer(f1), dcubic_inner(f), ops::neg(dcubic_inner(f2)),
                        ops::neg(dcubic_outer(f3))}),
            stack_last({d2cubic_outer(f1), d2cubic_inner(f), d2cubic_inner(f2),
                        d2cubic_outer(f3)})};
}

inline Tensor floor_index(const Tensor& x) { return ops::floor(x).to(DType::Int64); }
inline Tensor frac(const Tensor& x) { return ops::sub(x, ops::floor(x)); }

}  // namespace grid_bwd_detail

// d(grid_sampler_2d_backward) applied to (ggI, ggGrid): the gradients with
// respect to (grad_output, input, grid).
inline std::array<Tensor, 3> grid_sampler_2d_double_backward(
        const Tensor& ggI, const Tensor& ggGrid, const Tensor& grad_output, const Tensor& input,
        const Tensor& grid, int64_t interpolation, int64_t padding, bool align_corners,
        std::array<bool, 3> mask) {
    using namespace grid_bwd_detail;
    Tensor d_grad_output, d_input, d_grid;
    // grad_input = sample^T(grad_output): sampling ggI is its derivative in
    // grad_output, and ggI standing in for input gives the one in grid.
    if (mask[0] && ggI.defined()) {
        d_grad_output = ops::grid_sampler_2d(ggI, grid, interpolation, padding, align_corners);
    }
    if (mask[2] && ggI.defined()) {
        d_grid = std::get<1>(ops::grid_sampler_2d_backward(
            grad_output, ggI, grid, interpolation, padding, align_corners, {false, true}));
    }
    // A nearest kernel is flat in the grid: grad_grid is zero.
    if (!ggGrid.defined() || interpolation == kNearest) return {d_grad_output, d_input, d_grid};

    const int64_t N = input.size(0), H = input.size(2), W = input.size(3);
    const int64_t Ho = grid.size(1), Wo = grid.size(2);
    const std::vector<int64_t> input_shape = input.sizes();
    Tensor gx = ops::select(grid, -1, 0), gy = ops::select(grid, -1, 1);

    if (interpolation == kBilinear) {
        auto [ix, sx] = source_coords(gx, W, padding, align_corners);
        auto [iy, sy] = source_coords(gy, H, padding, align_corners);
        Tensor x0 = floor_index(ix), y0 = floor_index(iy);
        Tensor fx = frac(ix), fy = frac(iy);
        Tensor x1 = ops::add(x0, 1), y1 = ops::add(y0, 1);
        // Taps nw, ne, sw, se.
        Tensor h_idx = stack_last({y0, y0, y1, y1}), w_idx = stack_last({x0, x1, x0, x1});
        Tensor flat = ops::add(ops::mul(ops::clamp(h_idx, Scalar(0), Scalar(H - 1)), W),
                               ops::clamp(w_idx, Scalar(0), Scalar(W - 1)));
        Tensor oob = padding == kZeros ? in_bounds({{h_idx, H}, {w_idx, W}}) : Tensor();
        Tensor taps = gather_taps(input, flat, oob, {Ho, Wo, 4});
        Tensor gg_x = ops::mul(ops::select(ggGrid, -1, 0), sx);
        Tensor gg_y = ops::mul(ops::select(ggGrid, -1, 1), sy);
        Tensor ox = ops::rsub(fx, 1.0), oy = ops::rsub(fy, 1.0);
        Tensor dw_dx = stack_last({ops::neg(oy), oy, ops::neg(fy), fy});
        Tensor dw_dy = stack_last({ops::neg(ox), ops::neg(fx), ox, fx});
        if (mask[0]) {
            d_grad_output = add_defined(
                d_grad_output, ops::add(ops::mul(tap_sum(taps, dw_dx), ops::unsqueeze(gg_x, 1)),
                                        ops::mul(tap_sum(taps, dw_dy), ops::unsqueeze(gg_y, 1))));
        }
        if (mask[1]) {
            Tensor w = ops::add(ops::mul(ops::unsqueeze(gg_x, -1), dw_dx),
                                ops::mul(ops::unsqueeze(gg_y, -1), dw_dy));
            d_input = scatter_taps(grad_output, w, flat, oob, input_shape);
        }
        if (mask[2]) {
            // Bilinear weights are linear in each coordinate: only the mixed
            // second derivative survives, with tap signs (+, -, -, +).
            Tensor one = ops::ones_like(fx);
            Tensor signs = stack_last({one, ops::neg(one), ops::neg(one), one});
            Tensor cross = ops::sum(ops::mul(grad_output, tap_sum(taps, signs)),
                                    std::vector<int64_t>{1});
            d_grid = add_defined(d_grid, stack_last({ops::mul(ops::mul(sx, gg_y), cross),
                                                     ops::mul(ops::mul(sy, gg_x), cross)}));
        }
    } else if (interpolation == kBicubic) {
        // The bicubic kernels read raw unnormalized coordinates and pad only
        // when fetching each tap, so the slope is the unnormalize scale.
        const double x_scale = static_cast<double>(align_corners ? W - 1 : W) / 2.0;
        const double y_scale = static_cast<double>(align_corners ? H - 1 : H) / 2.0;
        Tensor x = ops::mul(ops::add(gx, 1.0), x_scale);
        Tensor y = ops::mul(ops::add(gy, 1.0), y_scale);
        if (!align_corners) {
            x = ops::sub(x, 0.5);
            y = ops::sub(y, 0.5);
        }
        Tensor x0 = floor_index(x), y0 = floor_index(y);
        CubicWeights cx = cubic_weights(frac(x)), cy = cubic_weights(frac(y));
        Tensor offs = ops::arange(-1, 3, 1, DType::Int64, x0.device());
        Tensor xt = ops::add(ops::unsqueeze(x0, -1), offs);
        Tensor yt = ops::add(ops::unsqueeze(y0, -1), offs);
        // Tap k = j * 4 + i reads row yt[j], column xt[i].
        Tensor w_idx = ops::reshape(ops::expand(ops::unsqueeze(xt, -2), {N, Ho, Wo, 4, 4}),
                                    {N, Ho, Wo, 16});
        Tensor h_idx = ops::reshape(ops::expand(ops::unsqueeze(yt, -1), {N, Ho, Wo, 4, 4}),
                                    {N, Ho, Wo, 16});
        Tensor flat = ops::add(ops::mul(bound_tap(h_idx, H, padding, align_corners), W),
                               bound_tap(w_idx, W, padding, align_corners));
        Tensor oob = padding == kZeros ? in_bounds({{h_idx, H}, {w_idx, W}}) : Tensor();
        Tensor taps = gather_taps(input, flat, oob, {Ho, Wo, 16});
        Tensor gg_x = ops::mul(ops::select(ggGrid, -1, 0), x_scale);
        Tensor gg_y = ops::mul(ops::select(ggGrid, -1, 1), y_scale);
        Tensor b_dx = outer_last(cx.dw, cy.w), b_dy = outer_last(cx.w, cy.dw);
        if (mask[0]) {
            d_grad_output = add_defined(
                d_grad_output, ops::add(ops::mul(tap_sum(taps, b_dx), ops::unsqueeze(gg_x, 1)),
                                        ops::mul(tap_sum(taps, b_dy), ops::unsqueeze(gg_y, 1))));
        }
        if (mask[1]) {
            Tensor w = ops::add(ops::mul(ops::unsqueeze(gg_x, -1), b_dx),
                                ops::mul(ops::unsqueeze(gg_y, -1), b_dy));
            d_input = scatter_taps(grad_output, w, flat, oob, input_shape);
        }
        if (mask[2]) {
            // <grad_output, tap> per tap, against the second derivatives of
            // the separable weights.
            Tensor dots = ops::sum(ops::mul(ops::unsqueeze(grad_output, -1), taps),
                                   std::vector<int64_t>{1});
            auto along = [&](const Tensor& a, const Tensor& b) {
                return ops::sum(ops::mul(dots, outer_last(a, b)), std::vector<int64_t>{-1});
            };
            Tensor dxx = along(cx.d2w, cy.w), dyy = along(cx.w, cy.d2w), dxy = along(cx.dw, cy.dw);
            Tensor d_x = ops::mul(ops::add(ops::mul(gg_x, dxx), ops::mul(gg_y, dxy)), x_scale);
            Tensor d_y = ops::mul(ops::add(ops::mul(gg_x, dxy), ops::mul(gg_y, dyy)), y_scale);
            d_grid = add_defined(d_grid, stack_last({d_x, d_y}));
        }
    } else {
        TP_THROW(NotImplementedError, "grid_sampler_2d double backward: unknown interpolation mode ",
                 interpolation);
    }
    return {d_grad_output, d_input, d_grid};
}

// The 3-D twin for nearest and trilinear sampling.
inline std::array<Tensor, 3> grid_sampler_3d_double_backward(
        const Tensor& ggI, const Tensor& ggGrid, const Tensor& grad_output, const Tensor& input,
        const Tensor& grid, int64_t interpolation, int64_t padding, bool align_corners,
        std::array<bool, 3> mask) {
    using namespace grid_bwd_detail;
    Tensor d_grad_output, d_input, d_grid;
    if (mask[0] && ggI.defined()) {
        d_grad_output = ops::grid_sampler_3d(ggI, grid, interpolation, padding, align_corners);
    }
    if (mask[2] && ggI.defined()) {
        d_grid = std::get<1>(ops::grid_sampler_3d_backward(
            grad_output, ggI, grid, interpolation, padding, align_corners, {false, true}));
    }
    if (!ggGrid.defined() || interpolation == kNearest) return {d_grad_output, d_input, d_grid};
    TP_CHECK(interpolation == kBilinear,
             "grid_sampler_3d double backward: unknown interpolation mode ", interpolation);

    const int64_t D = input.size(2), H = input.size(3), W = input.size(4);
    const int64_t Do = grid.size(1), Ho = grid.size(2), Wo = grid.size(3);
    const std::vector<int64_t> input_shape = input.sizes();
    auto [ix, sx] = source_coords(ops::select(grid, -1, 0), W, padding, align_corners);
    auto [iy, sy] = source_coords(ops::select(grid, -1, 1), H, padding, align_corners);
    auto [iz, sz] = source_coords(ops::select(grid, -1, 2), D, padding, align_corners);
    Tensor x0 = floor_index(ix), y0 = floor_index(iy), z0 = floor_index(iz);
    Tensor x1 = ops::add(x0, 1), y1 = ops::add(y0, 1), z1 = ops::add(z0, 1);
    Tensor fx = frac(ix), fy = frac(iy), fz = frac(iz);
    Tensor ox = ops::rsub(fx, 1.0), oy = ops::rsub(fy, 1.0), oz = ops::rsub(fz, 1.0);
    // Taps tnw, tne, tsw, tse, bnw, bne, bsw, bse.
    Tensor d_idx = stack_last({z0, z0, z0, z0, z1, z1, z1, z1});
    Tensor h_idx = stack_last({y0, y0, y1, y1, y0, y0, y1, y1});
    Tensor w_idx = stack_last({x0, x1, x0, x1, x0, x1, x0, x1});
    Tensor flat = ops::add(
        ops::mul(ops::add(ops::mul(ops::clamp(d_idx, Scalar(0), Scalar(D - 1)), H),
                          ops::clamp(h_idx, Scalar(0), Scalar(H - 1))),
                 W),
        ops::clamp(w_idx, Scalar(0), Scalar(W - 1)));
    Tensor oob = padding == kZeros ? in_bounds({{d_idx, D}, {h_idx, H}, {w_idx, W}}) : Tensor();
    Tensor taps = gather_taps(input, flat, oob, {Do, Ho, Wo, 8});
    Tensor gg_x = ops::mul(ops::select(ggGrid, -1, 0), sx);
    Tensor gg_y = ops::mul(ops::select(ggGrid, -1, 1), sy);
    Tensor gg_z = ops::mul(ops::select(ggGrid, -1, 2), sz);
    auto m = [](const Tensor& a, const Tensor& b) { return ops::mul(a, b); };
    auto n = [](const Tensor& a) { return ops::neg(a); };
    Tensor dw_dx = stack_last({n(m(oy, oz)), m(oy, oz), n(m(fy, oz)), m(fy, oz),
                               n(m(oy, fz)), m(oy, fz), n(m(fy, fz)), m(fy, fz)});
    Tensor dw_dy = stack_last({n(m(ox, oz)), n(m(fx, oz)), m(ox, oz), m(fx, oz),
                               n(m(ox, fz)), n(m(fx, fz)), m(ox, fz), m(fx, fz)});
    Tensor dw_dz = stack_last({n(m(ox, oy)), n(m(fx, oy)), n(m(ox, fy)), n(m(fx, fy)),
                               m(ox, oy), m(fx, oy), m(ox, fy), m(fx, fy)});
    if (mask[0]) {
        Tensor c = ops::add(ops::add(ops::mul(tap_sum(taps, dw_dx), ops::unsqueeze(gg_x, 1)),
                                     ops::mul(tap_sum(taps, dw_dy), ops::unsqueeze(gg_y, 1))),
                            ops::mul(tap_sum(taps, dw_dz), ops::unsqueeze(gg_z, 1)));
        d_grad_output = add_defined(d_grad_output, c);
    }
    if (mask[1]) {
        Tensor w = ops::add(ops::add(ops::mul(ops::unsqueeze(gg_x, -1), dw_dx),
                                     ops::mul(ops::unsqueeze(gg_y, -1), dw_dy)),
                            ops::mul(ops::unsqueeze(gg_z, -1), dw_dz));
        d_input = scatter_taps(grad_output, w, flat, oob, input_shape);
    }
    if (mask[2]) {
        // Trilinear weights are linear in each coordinate: the mixed second
        // derivatives d2/dxdy, d2/dxdz and d2/dydz remain.
        Tensor d2_xy = stack_last({oz, n(oz), n(oz), oz, fz, n(fz), n(fz), fz});
        Tensor d2_xz = stack_last({oy, n(oy), fy, n(fy), n(oy), oy, n(fy), fy});
        Tensor d2_yz = stack_last({ox, fx, n(ox), n(fx), n(ox), n(fx), ox, fx});
        auto dot = [&](const Tensor& basis) {
            return ops::sum(ops::mul(grad_output, tap_sum(taps, basis)), std::vector<int64_t>{1});
        };
        Tensor xy = dot(d2_xy), xz = dot(d2_xz), yz = dot(d2_yz);
        Tensor d_x = ops::mul(ops::add(ops::mul(gg_y, xy), ops::mul(gg_z, xz)), sx);
        Tensor d_y = ops::mul(ops::add(ops::mul(gg_x, xy), ops::mul(gg_z, yz)), sy);
        Tensor d_z = ops::mul(ops::add(ops::mul(gg_x, xz), ops::mul(gg_y, yz)), sz);
        d_grid = add_defined(d_grid, stack_last({d_x, d_y, d_z}));
    }
    return {d_grad_output, d_input, d_grid};
}

// The second-derivative nodes of the two backward kernels.  Incoming:
// gradients of (grad_input, grad_grid); outgoing: gradients of
// (grad_output, input, grid).
template <int Rank>
struct GridSamplerBackwardBackwardBase : public Node {
    SavedVariable grad_output_;
    SavedVariable input_;
    SavedVariable grid_;
    int64_t interpolation_;
    int64_t padding_;
    bool align_corners_;

    GridSamplerBackwardBackwardBase(Tensor grad_output, Tensor input, Tensor grid,
                                    int64_t interpolation, int64_t padding, bool align_corners)
        : grad_output_(std::move(grad_output)), input_(std::move(input)), grid_(std::move(grid)),
          interpolation_(interpolation), padding_(padding), align_corners_(align_corners) {}

    size_t num_inputs() const override { return 2; }

    variable_list apply(variable_list&& inputs) override {
        const std::array<bool, 3> mask = {should_compute_output(0), should_compute_output(1),
                                          should_compute_output(2)};
        const Tensor ggI = inputs.size() > 0 ? inputs[0] : Tensor();
        const Tensor ggGrid = inputs.size() > 1 ? inputs[1] : Tensor();
        std::array<Tensor, 3> out;
        if constexpr (Rank == 2) {
            out = grid_sampler_2d_double_backward(ggI, ggGrid, grad_output_.unpack(),
                                                  input_.unpack(), grid_.unpack(), interpolation_,
                                                  padding_, align_corners_, mask);
        } else {
            out = grid_sampler_3d_double_backward(ggI, ggGrid, grad_output_.unpack(),
                                                  input_.unpack(), grid_.unpack(), interpolation_,
                                                  padding_, align_corners_, mask);
        }
        return variable_list(out.begin(), out.end());
    }

    void release_variables() override {
        Node::release_variables();
        grad_output_.reset_data();
        input_.reset_data();
        grid_.reset_data();
    }
};

struct GridSampler2dBackwardBackward : public GridSamplerBackwardBackwardBase<2> {
    using GridSamplerBackwardBackwardBase<2>::GridSamplerBackwardBackwardBase;
};
struct GridSampler3dBackwardBackward : public GridSamplerBackwardBackwardBase<3> {
    using GridSamplerBackwardBackwardBase<3>::GridSamplerBackwardBackwardBase;
};

}  // namespace tpx
}  // namespace tensorplay
