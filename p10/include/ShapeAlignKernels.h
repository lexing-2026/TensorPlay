#pragma once

// Backend-neutral implementations registered under the Composite dispatch
// key. Every function composes its primitives through generated Tensor
// members, so each inner call routes through the dispatcher with device and
// autograd keys.
//
// Per-device registrations can override these implementations when an
// operation has specialized backend code.
//
// repeat() is deliberately absent: it carries real per-device code
// (cpu/ShapeAlignKernels.cpp gather + cuda/ShapeAlignKernels.cu twin).

#include "Tensor.h"

#include <vector>

namespace tensorplay {
namespace shapeops {

// zero strides over broadcast dims (-1 infers the input size).
Tensor tpsa_expand(const Tensor& self, const std::vector<int64_t>& size, bool implicit);
Tensor tpsa_expand_as(const Tensor& self, const Tensor& other);
Tensor tpsa_broadcast_to(const Tensor& self, const std::vector<int64_t>& size);

// defers to repeat.
Tensor tpsa_tile(const Tensor& self, const std::vector<int64_t>& dims);

// column_stack: promote inputs with atleast_Nd, then cat along the axis.
Tensor tpsa_hstack(const std::vector<Tensor>& tensors);
Tensor& tpsa_hstack_out(const std::vector<Tensor>& tensors, Tensor& out);
Tensor tpsa_vstack(const std::vector<Tensor>& tensors);
Tensor& tpsa_vstack_out(const std::vector<Tensor>& tensors, Tensor& out);
Tensor tpsa_dstack(const std::vector<Tensor>& tensors);
Tensor& tpsa_dstack_out(const std::vector<Tensor>& tensors, Tensor& out);
Tensor tpsa_row_stack(const std::vector<Tensor>& tensors);
Tensor& tpsa_row_stack_out(const std::vector<Tensor>& tensors, Tensor& out);
Tensor tpsa_column_stack(const std::vector<Tensor>& tensors);
Tensor& tpsa_column_stack_out(const std::vector<Tensor>& tensors, Tensor& out);

// _tensor_split_indices; hsplit/vsplit/dsplit are fixed-dim aliases.
std::vector<Tensor> tpsa_tensor_split_sections(const Tensor& self, int64_t sections, int64_t dim);
std::vector<Tensor> tpsa_tensor_split_indices(const Tensor& self,
                                              const std::vector<int64_t>& indices,
                                              int64_t dim);
std::vector<Tensor> tpsa_tensor_split_tensor(const Tensor& self,
                                             const Tensor& tensor_indices_or_sections,
                                             int64_t dim);
std::vector<Tensor> tpsa_hsplit_int(const Tensor& self, int64_t sections);
std::vector<Tensor> tpsa_hsplit_array(const Tensor& self, const std::vector<int64_t>& indices);
std::vector<Tensor> tpsa_vsplit_int(const Tensor& self, int64_t sections);
std::vector<Tensor> tpsa_vsplit_array(const Tensor& self, const std::vector<int64_t>& indices);
std::vector<Tensor> tpsa_dsplit_int(const Tensor& self, int64_t sections);
std::vector<Tensor> tpsa_dsplit_array(const Tensor& self, const std::vector<int64_t>& indices);

Tensor tpsa_atleast_1d(const Tensor& self);
std::vector<Tensor> tpsa_atleast_1d_seq(const std::vector<Tensor>& tensors);
Tensor tpsa_atleast_2d(const Tensor& self);
std::vector<Tensor> tpsa_atleast_2d_seq(const std::vector<Tensor>& tensors);
Tensor tpsa_atleast_3d(const Tensor& self);
std::vector<Tensor> tpsa_atleast_3d_seq(const std::vector<Tensor>& tensors);

Tensor tpsa_flatten(const Tensor& self, int64_t start_dim, int64_t end_dim);
Tensor tpsa_unflatten(const Tensor& self, int64_t dim, const std::vector<int64_t>& sizes);
Tensor tpsa_ravel(const Tensor& self);

// moveaxis / swapaxes / swapdims -- moveaxis is a movedim alias;
// swapaxes/swapdims are transpose aliases (numpy names).
Tensor tpsa_moveaxis_intlist(const Tensor& self, const std::vector<int64_t>& source,
                             const std::vector<int64_t>& destination);
Tensor tpsa_moveaxis_int(const Tensor& self, int64_t source, int64_t destination);
Tensor tpsa_swapaxes(const Tensor& self, int64_t axis0, int64_t axis1);
Tensor tpsa_swapdims(const Tensor& self, int64_t dim0, int64_t dim1);

// argwhere -- nonzero's (nnz, ndim) layout.
Tensor tpsa_argwhere(const Tensor& self);

// Equal and allclose use one device-generic composition here.
bool tpsa_equal(const Tensor& self, const Tensor& other);
bool tpsa_allclose(const Tensor& self, const Tensor& other, double rtol, double atol,
                   bool equal_nan);

Tensor tpsa_fill_scalar(const Tensor& self, Scalar value);
Tensor tpsa_fill_tensor(const Tensor& self, const Tensor& value);

} // namespace shapeops
} // namespace tensorplay
