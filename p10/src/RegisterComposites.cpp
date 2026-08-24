// Registration of backend-neutral composite kernels -- TensorPlay's analog of
// the generated build/aten/src/ATen/RegisterCompositeExplicitAutograd.cpp.
//
// native_functions.yaml maps this batch to CompositeExplicitAutograd
// (expand/allclose/fill/repeat) or to no dispatch section, i.e. the default
// CompositeImplicitAutograd (broadcast_to/tensor_split/tile/unflatten and the
// stacking/split/atleast/flatten/moveaxis aliases); upstream `equal` carries
// real per-backend kernels where p10 serves both dense keys from one
// device-generic composition.  The kernels themselves live beside their ATen
// counterparts in backend/cpu/ShapeAlignKernels.cpp; repeat() keeps explicit
// CPU/CUDA registrations (its gather has real device code) and is therefore
// not listed here.

#include "ShapeAlignKernels.h"
#include "Dispatcher.h"

namespace tensorplay {

TENSORPLAY_LIBRARY_IMPL(Composite, ShapeAlignComposites) {
    using namespace shapeops;

    // expand family
    m.impl("expand", tpsa_expand);
    m.impl("expand_as", tpsa_expand_as);
    m.impl("broadcast_to", tpsa_broadcast_to);
    m.impl("tile", tpsa_tile);

    // stacking family
    m.impl("hstack", tpsa_hstack);
    m.impl("hstack.out", tpsa_hstack_out);
    m.impl("vstack", tpsa_vstack);
    m.impl("vstack.out", tpsa_vstack_out);
    m.impl("dstack", tpsa_dstack);
    m.impl("dstack.out", tpsa_dstack_out);
    m.impl("row_stack", tpsa_row_stack);
    m.impl("row_stack.out", tpsa_row_stack_out);
    m.impl("column_stack", tpsa_column_stack);
    m.impl("column_stack.out", tpsa_column_stack_out);

    // tensor_split & split aliases
    m.impl("tensor_split.sections", tpsa_tensor_split_sections);
    m.impl("tensor_split.indices", tpsa_tensor_split_indices);
    m.impl("tensor_split.tensor_indices_or_sections", tpsa_tensor_split_tensor);
    m.impl("hsplit.int", tpsa_hsplit_int);
    m.impl("hsplit.array", tpsa_hsplit_array);
    m.impl("vsplit.int", tpsa_vsplit_int);
    m.impl("vsplit.array", tpsa_vsplit_array);
    m.impl("dsplit.int", tpsa_dsplit_int);
    m.impl("dsplit.array", tpsa_dsplit_array);

    // atleast_Nd
    m.impl("atleast_1d", tpsa_atleast_1d);
    m.impl("atleast_1d.Sequence", tpsa_atleast_1d_seq);
    m.impl("atleast_2d", tpsa_atleast_2d);
    m.impl("atleast_2d.Sequence", tpsa_atleast_2d_seq);
    m.impl("atleast_3d", tpsa_atleast_3d);
    m.impl("atleast_3d.Sequence", tpsa_atleast_3d_seq);

    // flatten / unflatten / ravel
    m.impl("flatten.using_ints", tpsa_flatten);
    m.impl("unflatten.int", tpsa_unflatten);
    m.impl("ravel", tpsa_ravel);

    // moveaxis / swapaxes / swapdims
    m.impl("moveaxis.intlist", tpsa_moveaxis_intlist);
    m.impl("moveaxis.int", tpsa_moveaxis_int);
    m.impl("swapaxes", tpsa_swapaxes);
    m.impl("swapdims", tpsa_swapdims);

    // argwhere / equal / allclose
    m.impl("argwhere", tpsa_argwhere);
    m.impl("equal", tpsa_equal);
    m.impl("allclose", tpsa_allclose);

    // fill family
    m.impl("fill.Scalar", tpsa_fill_scalar);
    m.impl("fill.Tensor", tpsa_fill_tensor);
}

} // namespace tensorplay
