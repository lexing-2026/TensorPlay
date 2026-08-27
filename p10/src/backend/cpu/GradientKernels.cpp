// torch.gradient (ATen Gradient.cpp / numpy semantics).
//
// Upstream registers gradient.* with no dispatch section -- the default
// CompositeImplicitAutograd: forward is a pure composition of differentiable
// primitives, autograd is derived from the inner calls and no derivatives.yaml
// entry exists.  Every returned tensor keeps the grad_fn its inner calls
// recorded, so backward AND double-backward work exactly like torch
// (torch 2.13 exposes a Cat -> Div -> Sub -> Slice chain, never a fused
// GradientBackward node).  We mirror that by registering one device-generic
// composition under the backend-neutral Composite key;
// OperatorHandle::getKernel falls through to it for every dense backend
// (p10/include/Dispatcher.h) and, since the op has no generated autograd
// wrapper, nothing stands between the caller and the inner recording.
//
// Edge formulas (numpy.gradient / ATen np_gradient):
//   edge_order selects the BORDER accuracy only -- eo1 one-sided chord
//   slopes, eo2 3-point Lagrange stencils.  The interior ALWAYS uses the
//   second-order central (3-point quadratic-fit) weights
//   wl=-hr/(hl(hl+hr)), wm=(hr-hl)/(hl*hr), wr=hl/(hr(hl+hr)), which reduce
//   to (f[k+1]-f[k-1])/(2h) under uniform spacing.
//
// The kernel never inspects device pointers: every primitive routes through
// the Dispatcher, so CPU/CUDA (and future dense backends) share this code.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"

#include <optional>
#include <vector>

namespace tensorplay {
namespace gradientops {

namespace {

int64_t grad_wrap_dim(int64_t dim, int64_t ndim) {
    if (dim < -ndim || dim >= ndim) {
        TP_THROW(IndexError,
                 "Dimension out of range (expected to be in range of [",
                 -ndim, ", ", ndim - 1, "], but got ", dim, ")");
    }
    return dim < 0 ? dim + ndim : dim;
}

// Reshape a 1-D weight vector so its only non-unit axis sits at `d`
// (broadcast-ready against slices of a tensor with `ndim` axes).
Tensor weight_along(const Tensor& w1d, int64_t d, int64_t ndim) {
    std::vector<int64_t> view(static_cast<size_t>(ndim), 1);
    view[static_cast<size_t>(d)] = w1d.numel();
    return w1d.reshape(view);
}

} // namespace

// Derivative of `src` along axis `d`.  coord carries either the per-position
// coordinates (numel == n) or the uniform step (numel == 1).
Tensor gradient_axis(const Tensor& src, int64_t d, const Tensor& coord,
                     int64_t edge_order) {
    const int64_t n = src.size(d);
    if (n <= edge_order) {
        TP_THROW(RuntimeError,
                 "torch.gradient expected each dimension size to be at least "
                 "edge_order+1");
    }
    const Tensor c = coord.to(src.dtype());

    // 1-wide border views keep every expression shape-generic through
    // broadcasting.
    const auto seg = [&](int64_t start) { return src.narrow(d, start, 1); };

    if (c.numel() == 1) {
        // Uniform spacing h (possibly different per axis).
        const Scalar h = c.item();
        const double hd = h.toDouble();
        const Scalar two_h = Scalar(2.0 * hd);
        // interior k = 1 .. n-2: (f[k+1]-f[k-1])/(2h)
        Tensor gi = src.narrow(d, 2, n - 2).sub(src.narrow(d, 0, n - 2))
                        .div(two_h);
        if (edge_order == 1) {
            Tensor gl = seg(1).sub(seg(0)).div(h);
            Tensor gr = seg(n - 1).sub(seg(n - 2)).div(h);
            return Tensor::cat({gl, gi, gr}, d);
        }
        // edge_order == 2 borders: (-3 f0 + 4 f1 - f2)/(2h),
        //                          (3 a - 4 b + c)/(2h)
        Tensor gl = seg(1).mul(4).sub(seg(0).mul(3)).sub(seg(2)).div(two_h);
        Tensor gr = seg(n - 1).mul(3).sub(seg(n - 2).mul(4)).add(seg(n - 3))
                        .div(two_h);
        return Tensor::cat({gl, gi, gr}, d);
    }

    if (c.numel() != n) {
        TP_THROW(RuntimeError,
                 "torch.gradient expected one coordinate per position along "
                 "the differentiated dimension (got ",
                 c.numel(), " coordinates for ", n, " positions)");
    }

    // Non-uniform coordinates.  All weights stay tensors so that gradients
    // wrt the coordinates themselves would flow exactly like torch's.
    // Interior (k = 1 .. n-2), quadratic fit through (k-1, k, k+1) -- this
    // is numpy/torch's second-order central difference, used for BOTH edge
    // orders; edge_order only changes the borders below.
    const Tensor hl = c.narrow(0, 1, n - 2).sub(c.narrow(0, 0, n - 2));
    const Tensor hr = c.narrow(0, 2, n - 2).sub(c.narrow(0, 1, n - 2));
    const Tensor hsum = hl.add(hr);
    Tensor wl = hr.neg().div(hl.mul(hsum));
    Tensor wm = hr.sub(hl).div(hl.mul(hr));
    Tensor wr = hl.div(hr.mul(hsum));
    Tensor gi = src.narrow(d, 0, n - 2).mul(weight_along(wl, d, src.dim()))
                    .add(src.narrow(d, 1, n - 2)
                             .mul(weight_along(wm, d, src.dim())))
                    .add(src.narrow(d, 2, n - 2)
                             .mul(weight_along(wr, d, src.dim())));

    Tensor gl;
    Tensor gr;
    if (edge_order == 1) {
        gl = seg(1).sub(seg(0))
                .div(c.narrow(0, 1, 1).sub(c.narrow(0, 0, 1)));
        gr = seg(n - 1).sub(seg(n - 2))
                .div(c.narrow(0, n - 1, 1).sub(c.narrow(0, n - 2, 1)));
        return Tensor::cat({gl, gi, gr}, d);
    }

    // edge_order == 2 borders.  Left on (x0, x1, x2), h1 = x1-x0,
    // h2 = x2-x1:
    //   w = (-(2h1+h2)/(h1(h1+h2)), (h1+h2)/(h1 h2), -h1/(h2(h1+h2)))
    const Tensor h1 = c.narrow(0, 1, 1).sub(c.narrow(0, 0, 1));
    const Tensor h2 = c.narrow(0, 2, 1).sub(c.narrow(0, 1, 1));
    const Tensor lhsum = h1.add(h2);
    gl = seg(0).mul(lhsum.add(h1).neg().div(h1.mul(lhsum)))
             .add(seg(1).mul(lhsum.div(h1.mul(h2))))
             .add(seg(2).mul(h1.neg().div(h2.mul(lhsum))));

    // Right border on (x_{n-3}, x_{n-2}, x_{n-1}); Lagrange derivative at
    // x_{n-1} with g1 = x_{n-2}-x_{n-3}, g2 = x_{n-1}-x_{n-2}:
    //   v = (g2/(g1(g1+g2)), -(g1+g2)/(g1 g2), (g1+2g2)/(g2(g1+g2)))
    const Tensor g1 = c.narrow(0, n - 2, 1).sub(c.narrow(0, n - 3, 1));
    const Tensor g2 = c.narrow(0, n - 1, 1).sub(c.narrow(0, n - 2, 1));
    gr = seg(n - 3).mul(g2.div(g1.mul(g1.add(g2))))
             .add(seg(n - 2).mul(g1.add(g2).neg().div(g1.mul(g2))))
             .add(seg(n - 1).mul(g2.mul(2).add(g1).div(g2.mul(g1.add(g2)))));
    return Tensor::cat({gl, gi, gr}, d);
}

// Dispatcher entry: schema is
//   gradient(Tensor self, Tensor[] spacing=[], int[] dim=[], int edge_order=1)
//     -> Tensor[]
// Empty `dim` differentiates along every axis; empty `spacing` means unit
// step for every requested axis; otherwise both lists are parallel (the
// Python wrapper guarantees the pairing).  Integral / bool inputs promote to
// Float32 exactly like torch's result_type(self, 1.0) behavior.
std::vector<Tensor> gradient_composite(const Tensor& self,
                                       const std::vector<Tensor>& spacing,
                                       const std::vector<int64_t>& dims,
                                       int64_t edge_order) {
    if (edge_order != 1 && edge_order != 2) {
        TP_THROW(RuntimeError,
                 "torch.gradient only supports edge_order=1 and edge_order=2.");
    }
    if (!spacing.empty() && spacing.size() != dims.size()) {
        TP_THROW(RuntimeError,
                 "torch.gradient expected one spacing per differentiated "
                 "dimension");
    }
    const int64_t ndim = self.dim();
    std::vector<int64_t> axes;
    axes.reserve(dims.empty() ? static_cast<size_t>(ndim) : dims.size());
    if (dims.empty()) {
        for (int64_t i = 0; i < ndim; ++i) axes.push_back(i);
    } else {
        for (int64_t d : dims) axes.push_back(grad_wrap_dim(d, ndim));
    }

    Tensor work = self;
    if (!isFloatingType(work.dtype()) && !isComplexType(work.dtype())) {
        work = work.to(DType::Float32);
    }

    std::vector<Tensor> outs;
    outs.reserve(axes.size());
    for (size_t i = 0; i < axes.size(); ++i) {
        Tensor coord;
        if (spacing.empty()) {
            // Unit step for every axis.
            coord = Tensor::full({1}, 1, work.dtype(), work.device());
        } else if (spacing.size() == 1) {
            // A single spacing broadcasts across every requested axis
            // (torch's scalar / one-tensor spacing form).
            coord = spacing[0];
        } else {
            coord = spacing[i];
        }
        outs.push_back(gradient_axis(work, axes[i], coord, edge_order));
    }
    return outs;
}

TENSORPLAY_LIBRARY_IMPL(Composite, GradientKernels) {
    m.impl("gradient", gradient_composite);
}

} // namespace gradientops
} // namespace tensorplay

