#pragma once

// Shape & view alignment composite entry points (torch native-parity batch).
//
// These are the backend-neutral implementations registered under the
// Composite dispatch key -- TensorPlay's analog of upstream's
// CompositeExplicitAutograd / CompositeImplicitAutograd mappings in
// native_functions.yaml, whose kernels live in ATen's backend-neutral
// native/*.cpp (TensorShape.cpp, TensorTransformations.cpp, Fill.cpp) and
// register through build/aten/src/ATen/RegisterCompositeExplicitAutograd.cpp.
// Every function composes its primitives through generated Tensor members,
// so each inner call routes through the Dispatcher (device + autograd keys)
// exactly like an at:: call inside an ATen composite; a dense backend may
// still override any of these per key (upstream does this for repeat on MPS).
//
// repeat() is deliberately absent: it carries real per-device code
// (cpu/ShapeAlignKernels.cpp gather + cuda/ShapeAlignKernels.cu twin).

#include "Tensor.h"

#include <vector>

namespace tensorplay {
namespace shapeops {

// expand family -- ATen TensorShape.cpp expand(): as_strided view with
// zero strides over broadcast dims (-1 infers the input size).
Tensor tpsa_expand(const Tensor& self, const std::vector<int64_t>& size, bool implicit);
Tensor tpsa_expand_as(const Tensor& self, const Tensor& other);
Tensor tpsa_broadcast_to(const Tensor& self, const std::vector<int64_t>& size);

// tile -- ATen TensorShape.cpp tile(): prepends unit dims to short reps and
// defers to repeat.
Tensor tpsa_tile(const Tensor& self, const std::vector<int64_t>& dims);

// stacking family -- ATen TensorShape.cpp hstack/vstack/dstack/row_stack/
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

// tensor_split & friends -- ATen TensorShape.cpp tensor_split_sections /
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

// atleast_Nd -- ATen TensorTransformations.cpp.
Tensor tpsa_atleast_1d(const Tensor& self);
std::vector<Tensor> tpsa_atleast_1d_seq(const std::vector<Tensor>& tensors);
Tensor tpsa_atleast_2d(const Tensor& self);
std::vector<Tensor> tpsa_atleast_2d_seq(const std::vector<Tensor>& tensors);
Tensor tpsa_atleast_3d(const Tensor& self);
std::vector<Tensor> tpsa_atleast_3d_seq(const std::vector<Tensor>& tensors);

// flatten / unflatten / ravel -- ATen TensorShape.cpp.
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

// equal / allclose -- compose eq/all/isclose exactly like ATen's native impls
// (TensorCompare.cpp allclose = isclose(self,...).all().item<uint8_t>());
// upstream registers equal with real CPU/CUDA kernels where p10 keeps the
// single device-generic composition for both keys.
bool tpsa_equal(const Tensor& self, const Tensor& other);
bool tpsa_allclose(const Tensor& self, const Tensor& other, double rtol, double atol,
                   bool equal_nan);

// fill -- ATen Fill.cpp: full_like preserves dtype/device identically.
Tensor tpsa_fill_scalar(const Tensor& self, Scalar value);
Tensor tpsa_fill_tensor(const Tensor& self, const Tensor& value);

} // namespace shapeops
} // namespace tensorplay
