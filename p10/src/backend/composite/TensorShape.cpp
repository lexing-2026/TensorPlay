// Composite kernels: reshape_as / unsafe_chunk / unsafe_split.
// differ from chunk/split by zeroing the results' version counters (an
// autograd-internal concern); the view structure is identical.

#include "CompositeCommon.h"
#include "SetStorage.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

Tensor reshape_as_native(const Tensor& self, const Tensor& other) {
    return ops::reshape(self, static_cast<std::vector<int64_t>>(other.shape()));
}

std::vector<Tensor> unsafe_chunk_native(const Tensor& self, int64_t chunks,
                                        int64_t dim) {
    if (self.dim() == 0) {
        TP_THROW(RuntimeError, "chunk expects at least a 1-dimensional tensor");
    }
    if (chunks <= 0) {
        TP_THROW(RuntimeError,
                 "chunk expects chunks to be a positive integer");
    }
    const int64_t d = wrap_dim(dim, self.dim());
    const int64_t dim_size = self.size(d);
    const int64_t split_size = (dim_size + chunks - 1) / chunks;
    if (split_size == 0 && dim_size == 0) {
        std::vector<Tensor> pieces;
        pieces.reserve(static_cast<size_t>(chunks));
        for (int64_t i = 0; i < chunks; ++i) {
            pieces.push_back(ops::slice(self, d, 0, 0, 1));
        }
        return pieces;
    }
    return ops::split(self, split_size, d);
}

std::vector<Tensor> unsafe_split_native(const Tensor& self,
                                        int64_t split_size, int64_t dim) {
    return ops::split(self, split_size, dim);
}

Tensor fliplr_native(const Tensor& self) {
    return ops::flip(self, {-1});
}

Tensor flipud_native(const Tensor& self) {
    return ops::flip(self, {0});
}

Tensor& resize_as__native(Tensor& self, const Tensor& other,
                          std::optional<int64_t> memory_format) {
    const auto sizes = static_cast<std::vector<int64_t>>(other.shape());
    ops::resize_(self, sizes);
    if (!memory_format.has_value()) {
        return self;
    }

    auto format = static_cast<MemoryFormat>(*memory_format);
    if (format == MemoryFormat::Preserve) {
        format = other.memory_format();
        if (format != MemoryFormat::ChannelsLast &&
            format != MemoryFormat::ChannelsLast3d) {
            format = MemoryFormat::Contiguous;
        }
    }
    if (format != MemoryFormat::Contiguous &&
        format != MemoryFormat::ChannelsLast &&
        format != MemoryFormat::ChannelsLast3d) {
        TP_THROW(ValueError, "resize_as_: invalid memory format");
    }
    if (format == MemoryFormat::ChannelsLast && sizes.size() != 4) {
        TP_THROW(RuntimeError,
                 "resize_as_: channels-last format requires rank 4");
    }
    if (format == MemoryFormat::ChannelsLast3d && sizes.size() != 5) {
        TP_THROW(RuntimeError,
                 "resize_as_: channels-last-3d format requires rank 5");
    }
    self.unsafeGetTensorImpl()->set_sizes_and_strides(
        sizes, get_strides_for(sizes, format));
    return self;
}

Tensor& set__source_Tensor_storage_offset_native(
    Tensor& self, const Tensor& source, int64_t storage_offset,
    const std::vector<int64_t>& size,
    const std::vector<int64_t>& stride) {
    return native::set_tensor_storage_offset_native(
        self, source, storage_offset, size, stride);
}

TENSORPLAY_LIBRARY_IMPL(Composite, TensorShapeComposite) {
    m.impl("reshape_as", reshape_as_native);
    m.impl("unsafe_chunk", unsafe_chunk_native);
    m.impl("unsafe_split.Tensor", unsafe_split_native);
    m.impl("fliplr", fliplr_native);
    m.impl("flipud", flipud_native);
    m.impl("resize_as_", resize_as__native);
    m.impl("set_.source_Tensor_storage_offset",
           set__source_Tensor_storage_offset_native);
}

} // namespace composite
} // namespace tensorplay
