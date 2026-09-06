// Registration of backend-neutral composite kernels.
//
// These helpers build their results from dispatcher-visible tensor
// operations. Operations with explicit CPU/CUDA implementations are
// registered in their backend translation units.

#include "ShapeAlignKernels.h"
#include "Dispatcher.h"
#include "MemoryFormat.h"
#include "Tensor.h"

#include <optional>
#include <vector>

namespace tensorplay {

namespace {

int64_t layout_of(const Tensor& tensor) {
    if (!tensor.is_sparse()) return 2;
    return tensor.unsafeGetTensorImpl()->sparse_layout();
}

Tensor pin_if_requested(Tensor value, std::optional<bool> pin_memory) {
    if (pin_memory.has_value() && *pin_memory) {
        return value.pin_memory();
    }
    return value;
}

Tensor to_copy_composite(
    const Tensor& self,
    std::optional<DType> dtype,
    std::optional<int64_t> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<int64_t> memory_format) {
    if (layout.has_value() && *layout != layout_of(self)) {
        TP_THROW(NotImplementedError,
                 "to(): converting to a different layout is not supported");
    }

    const DType target_dtype = dtype.value_or(self.dtype());
    const Device target_device = device.value_or(self.device());
    const auto format = static_cast<MemoryFormat>(
        memory_format.value_or(static_cast<int64_t>(MemoryFormat::Preserve)));
    if (format != MemoryFormat::Contiguous &&
        format != MemoryFormat::Preserve &&
        format != MemoryFormat::ChannelsLast &&
        format != MemoryFormat::ChannelsLast3d) {
        TP_THROW(ValueError, "to(): invalid memory format");
    }

    if (self.is_sparse()) {
        if (format != MemoryFormat::Preserve) {
            TP_THROW(RuntimeError,
                     "to(): sparse tensors only support Preserve memory format");
        }
        if (self.is_sparse_compressed()) {
            return Tensor::make_sparse_compressed_tensor(
                pin_if_requested(
                    self._crow_indices().to(target_device, non_blocking, true),
                    pin_memory),
                pin_if_requested(
                    self._col_indices().to(target_device, non_blocking, true),
                    pin_memory),
                pin_if_requested(
                    self._values().to(target_device, target_dtype, non_blocking, true),
                    pin_memory),
                static_cast<std::vector<int64_t>>(self.shape()),
                self.unsafeGetTensorImpl()->sparse_layout(),
                self.sparse_blocksize());
        }
        return Tensor::make_sparse_coo_tensor(
            pin_if_requested(
                self._indices().to(target_device, non_blocking, true), pin_memory),
            pin_if_requested(
                self._values().to(target_device, target_dtype, non_blocking, true),
                pin_memory),
            static_cast<std::vector<int64_t>>(self.shape()), self.is_coalesced());
    }

    const auto sizes = static_cast<std::vector<int64_t>>(self.shape());
    std::vector<int64_t> strides;
    if (format == MemoryFormat::Preserve) {
        const auto source_strides = static_cast<std::vector<int64_t>>(self.strides());
        if (SizesAndStrides::is_non_overlapping_and_dense(sizes, source_strides)) {
            strides = source_strides;
        } else {
            strides = SizesAndStrides::compute_contiguous_strides(sizes);
        }
    } else if (format == MemoryFormat::Contiguous) {
        strides = SizesAndStrides::compute_contiguous_strides(sizes);
    } else {
        const int64_t expected_rank =
            format == MemoryFormat::ChannelsLast ? 4 : 5;
        if (self.dim() != expected_rank) {
            TP_THROW(RuntimeError,
                     "to(): memory format requires rank ", expected_rank,
                     " but got rank ", self.dim());
        }
        strides = get_strides_for(sizes, format);
    }

    Storage storage(static_cast<size_t>(self.numel()) * elementSize(target_dtype),
                    getAllocator(target_device.type()), target_device);
    Tensor result(std::move(storage), sizes, strides, target_dtype);
    result.copy_(self, non_blocking);
    if (pin_memory.has_value() && *pin_memory) {
        result = result.pin_memory();
    }
    return result;
}

bool to_will_alias(
    const Tensor& self,
    std::optional<DType> dtype,
    std::optional<int64_t> layout,
    std::optional<Device> device,
    bool copy,
    std::optional<int64_t> memory_format) {
    if (layout.has_value() && *layout != layout_of(self)) return false;
    if (dtype.has_value() && *dtype != self.dtype()) return false;
    if (device.has_value() && *device != self.device()) return false;
    if (copy) return false;
    if (memory_format.has_value() &&
        (*memory_format < static_cast<int64_t>(MemoryFormat::Contiguous) ||
         *memory_format > static_cast<int64_t>(MemoryFormat::ChannelsLast3d))) {
        return false;
    }
    if (!memory_format.has_value() ||
        *memory_format == static_cast<int64_t>(MemoryFormat::Preserve)) {
        return true;
    }
    return self.is_contiguous(static_cast<MemoryFormat>(*memory_format));
}

Tensor to_impl_composite(
    const Tensor& self,
    std::optional<DType> dtype,
    std::optional<int64_t> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    bool copy,
    std::optional<int64_t> memory_format) {
    if (to_will_alias(self, dtype, layout, device, copy, memory_format)) {
        return self;
    }
    return to_copy_composite(
        self, dtype, layout, device, pin_memory, non_blocking, memory_format);
}

Tensor to_dtype_layout_composite(
    const Tensor& self,
    std::optional<DType> dtype,
    std::optional<int64_t> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    bool copy,
    std::optional<int64_t> memory_format) {
    return to_impl_composite(self, dtype, layout, device, pin_memory,
                             non_blocking, copy, memory_format);
}

// Reduces a broadcast-expanded tensor back to its pre-broadcast shape, the
// reduction contract used by autograd to shrink a gradient onto its source:
// dimensions the target does not track sum away, every surviving dimension
// pinned to 1 sums with keepdim so the rank survives, and the result is
// reshaped to the exact target shape.
Tensor sum_to_size_composite(const Tensor& self,
                             const std::vector<int64_t>& size) {
    const auto self_sizes = static_cast<std::vector<int64_t>>(self.shape());
    if (self_sizes == size) {
        return self;
    }
    if (self_sizes.size() < size.size()) {
        TP_THROW(RuntimeError,
                 "sum_to_size(): target rank ", size.size(),
                 " exceeds the input rank ", self_sizes.size());
    }
    Tensor result = self;
    const int64_t ndim = static_cast<int64_t>(self_sizes.size());
    const int64_t target_ndim = static_cast<int64_t>(size.size());
    for (int64_t i = 0; i < ndim; ++i) {
        const int64_t target = i - (ndim - target_ndim);
        if ((target < 0 || size[target] == 1) && result.size(i) != 1) {
            result = result.sum({i}, /*keepdim=*/true);
        }
    }
    return result.reshape(size);
}

Tensor to_device_composite(
    const Tensor& self,
    Device device,
    DType dtype,
    bool non_blocking,
    bool copy,
    std::optional<int64_t> memory_format) {
    return to_impl_composite(self, dtype, std::nullopt, device, std::nullopt,
                             non_blocking, copy, memory_format);
}

Tensor to_dtype_composite(
    const Tensor& self,
    DType dtype,
    bool non_blocking,
    bool copy,
    std::optional<int64_t> memory_format) {
    return to_impl_composite(self, dtype, std::nullopt, std::nullopt,
                             std::nullopt, non_blocking, copy, memory_format);
}

Tensor to_other_composite(
    const Tensor& self,
    const Tensor& other,
    bool non_blocking,
    bool copy,
    std::optional<int64_t> memory_format) {
    return to_impl_composite(self, other.dtype(), layout_of(other), other.device(),
                             std::nullopt, non_blocking, copy, memory_format);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(Composite, ShapeAlignComposites) {
    m.impl("_to_copy", to_copy_composite);
    m.impl("to.dtype_layout", to_dtype_layout_composite);
    m.impl("to.device", to_device_composite);
    m.impl("to.dtype", to_dtype_composite);
    m.impl("to.other", to_other_composite);
    m.impl("sum_to_size", sum_to_size_composite);

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
