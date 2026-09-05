#include "Dispatcher.h"
#include "Exception.h"
#include "Tensor.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cstdint>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

Tensor linspace_from_neg_one(const Tensor& grid, int64_t num_steps,
                             bool align_corners) {
    if (num_steps <= 1) {
        return ops::full({}, Scalar(0), grid.dtype(), grid.device());
    }
    Tensor range = ops::linspace(Scalar(-1), Scalar(1), num_steps,
                                 grid.dtype(), grid.device());
    if (!align_corners) {
        range = ops::div(ops::mul(range, Scalar(num_steps - 1)),
                          Scalar(num_steps));
    }
    return range;
}

Tensor make_base_grid_4d(const Tensor& theta, int64_t n, int64_t c,
                         int64_t h, int64_t w, bool align_corners) {
    (void)c;
    Tensor base_grid({n, h, w, 3}, theta.dtype(), theta.device());
    Tensor x = ops::select(base_grid, -1, 0);
    Tensor y = ops::select(base_grid, -1, 1);
    ops::copy_(x, ops::expand(
        ops::view(linspace_from_neg_one(theta, w, align_corners), {1, 1, w}),
        {n, h, w}));
    ops::copy_(y, ops::expand(
        ops::view(linspace_from_neg_one(theta, h, align_corners), {1, h, 1}),
        {n, h, w}));
    Tensor w_col = ops::select(base_grid, -1, 2);
    ops::fill_(w_col, Scalar(1));
    return base_grid;
}

Tensor make_base_grid_5d(const Tensor& theta, int64_t n, int64_t c,
                         int64_t d, int64_t h, int64_t w,
                         bool align_corners) {
    (void)c;
    Tensor base_grid({n, d, h, w, 4}, theta.dtype(), theta.device());
    Tensor x = ops::select(base_grid, -1, 0);
    Tensor y = ops::select(base_grid, -1, 1);
    Tensor z = ops::select(base_grid, -1, 2);
    ops::copy_(x, ops::expand(
        ops::view(linspace_from_neg_one(theta, w, align_corners),
                  {1, 1, 1, w}),
        {n, d, h, w}));
    ops::copy_(y, ops::expand(
        ops::view(linspace_from_neg_one(theta, h, align_corners),
                  {1, 1, h, 1}),
        {n, d, h, w}));
    ops::copy_(z, ops::expand(
        ops::view(linspace_from_neg_one(theta, d, align_corners),
                  {1, d, 1, 1}),
        {n, d, h, w}));
    Tensor d_col = ops::select(base_grid, -1, 3);
    ops::fill_(d_col, Scalar(1));
    return base_grid;
}

Tensor affine_grid_generator_4d(const Tensor& theta, int64_t n, int64_t c,
                                int64_t h, int64_t w, bool align_corners) {
    Tensor base_grid = make_base_grid_4d(theta, n, c, h, w, align_corners);
    Tensor grid = ops::bmm(ops::view(base_grid, {n, h * w, 3}),
                           ops::transpose(theta, 1, 2));
    return ops::view(grid, {n, h, w, 2});
}

Tensor affine_grid_generator_5d(const Tensor& theta, int64_t n, int64_t c,
                                int64_t d, int64_t h, int64_t w,
                                bool align_corners) {
    Tensor base_grid = make_base_grid_5d(theta, n, c, d, h, w, align_corners);
    Tensor grid = ops::bmm(ops::view(base_grid, {n, d * h * w, 4}),
                           ops::transpose(theta, 1, 2));
    return ops::view(grid, {n, d, h, w, 3});
}

Tensor affine_grid_generator_composite(const Tensor& theta,
                                       const std::vector<int64_t>& size,
                                       bool align_corners) {
    TP_CHECK(size.size() == 4 || size.size() == 5,
             "AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.");
    if (size.size() == 4) {
        return affine_grid_generator_4d(theta, size[0], size[1], size[2],
                                        size[3], align_corners);
    }
    return affine_grid_generator_5d(theta, size[0], size[1], size[2], size[3],
                                    size[4], align_corners);
}

Tensor affine_grid_generator_backward_4d(const Tensor& grad, int64_t n,
                                         int64_t c, int64_t h, int64_t w,
                                         bool align_corners) {
    Tensor base_grid = make_base_grid_4d(grad, n, c, h, w, align_corners);
    TP_CHECK((grad.shape() == std::vector<int64_t>{n, h, w, 2}),
             "affine_grid_generator_backward: invalid gradient shape");
    Tensor grad_theta = ops::bmm(
        ops::transpose(ops::view(base_grid, {n, h * w, 3}), 1, 2),
        ops::view(grad, {n, h * w, 2}));
    return ops::transpose(grad_theta, 1, 2);
}

Tensor affine_grid_generator_backward_5d(const Tensor& grad, int64_t n,
                                         int64_t c, int64_t d, int64_t h,
                                         int64_t w, bool align_corners) {
    Tensor base_grid = make_base_grid_5d(grad, n, c, d, h, w, align_corners);
    TP_CHECK((grad.shape() == std::vector<int64_t>{n, d, h, w, 3}),
             "affine_grid_generator_backward: invalid gradient shape");
    Tensor grad_theta = ops::bmm(
        ops::transpose(ops::view(base_grid, {n, d * h * w, 4}), 1, 2),
        ops::view(grad, {n, d * h * w, 3}));
    return ops::transpose(grad_theta, 1, 2);
}

Tensor affine_grid_generator_backward_composite(
    const Tensor& grad, const std::vector<int64_t>& size, bool align_corners) {
    TP_CHECK(size.size() == 4 || size.size() == 5,
             "AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.");
    if (size.size() == 4) {
        return affine_grid_generator_backward_4d(
            grad, size[0], size[1], size[2], size[3], align_corners);
    }
    return affine_grid_generator_backward_5d(
        grad, size[0], size[1], size[2], size[3], size[4], align_corners);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(Composite, AffineGridComposite) {
    m.impl("affine_grid_generator", affine_grid_generator_composite);
    m.impl("affine_grid_generator_backward",
           affine_grid_generator_backward_composite);
}

} // namespace composite
} // namespace tensorplay
