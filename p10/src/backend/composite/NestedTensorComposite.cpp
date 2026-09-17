// Backend-neutral nested (ragged) tensor kernels.
//
// Representation: a nested tensor keeps one flat buffer plus per-constituent
// metadata mounted on the tensor impl:
//   nested_sizes    [B, R] Int64 -- row i is the shape of constituent i;
//   nested_strides  [B, R] Int64 -- row-major strides implied by that shape;
//   storage_offsets [B]    Int64 -- element offset of each constituent's
//                                   first element inside the flat buffer.
// The tensor reports dim() == R + 1 and numel() == the sum of the row
// volumes. A single dense size or stride does not exist, so the matching
// accessors reject nested input in favour of the per-constituent metadata.
//
// Every kernel here is a rewrite onto dispatcher-visible primitives (cat /
// stack / full / narrow / view / select / copy_ / amax / to), so construction
// and padding run on whichever device holds the buffer. The metadata tensors
// are built on the same device as the buffer.

#include "CompositeCommon.h"
#include "Tensor.h"
#include "Exception.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tensorplay {
namespace composite {

namespace {

constexpr int64_t kStridedLayout = 0;

auto require_nested_state(const Tensor& self, const char* op) {
    if (!self.defined() || !self.impl()->is_nested()) {
        TP_THROW(RuntimeError,
                 std::string(op) + ": expected a nested tensor");
    }
    return self.impl()->nested_state();
}

// Row-major strides of one shape row: the last axis advances by one element
// and each earlier axis advances by its extent times the stride of the axis
// after it.
std::vector<int64_t> row_major_strides(const int64_t* sizes, int64_t rank) {
    std::vector<int64_t> strides(rank > 0 ? static_cast<size_t>(rank) : 0, 1);
    for (int64_t j = rank - 2; j >= 0; --j) {
        strides[static_cast<size_t>(j)] =
            strides[static_cast<size_t>(j + 1)] * sizes[j + 1];
    }
    return strides;
}

int64_t row_volume(const int64_t* sizes, int64_t rank) {
    int64_t vol = 1;
    for (int64_t j = 0; j < rank; ++j) vol *= sizes[j];
    return vol;
}

Tensor nested_from_tensor_list(
    const std::vector<Tensor>& list,
    std::optional<DType> dtype,
    std::optional<int64_t> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
    if (list.empty()) {
        TP_THROW(RuntimeError,
                 "_nested_tensor_from_tensor_list(): expected a non-empty "
                 "list of tensors");
    }
    if (layout.has_value() && *layout != kStridedLayout) {
        TP_THROW(RuntimeError,
                 "_nested_tensor_from_tensor_list(): only the strided layout "
                 "is supported, got layout " + std::to_string(*layout));
    }
    const int64_t rank = list[0].dim();
    for (size_t i = 1; i < list.size(); ++i) {
        const int64_t d = list[i].dim();
        if (d != rank) {
            TP_THROW(RuntimeError,
                     "_nested_tensor_from_tensor_list(): all constituents "
                     "must have the same rank; got rank " +
                         std::to_string(d) + " at index " + std::to_string(i) +
                         " but rank " + std::to_string(rank) + " at index 0");
        }
    }

    const Device target_device = device.value_or(list[0].device());
    const DType target_dtype = dtype.value_or(list[0].dtype());

    // Flatten the constituents in row-major order and concatenate; the packed
    // result is the backing buffer, so constituent i starts at the running
    // element count of the rows before it.
    std::vector<Tensor> flat_parts;
    flat_parts.reserve(list.size());
    for (const Tensor& t : list) {
        if (t.is_nested()) {
            TP_THROW(RuntimeError,
                     "_nested_tensor_from_tensor_list(): constituents must be "
                     "dense tensors");
        }
        Tensor part = t;
        if (part.device() != target_device || part.dtype() != target_dtype) {
            part = part.to(target_device, target_dtype);
        }
        flat_parts.push_back(part.reshape({part.numel()}).contiguous());
    }
    Tensor buffer = Tensor::cat(flat_parts, 0);
    if (pin_memory.has_value() && *pin_memory) {
        buffer = buffer.pin_memory();
    }

    const int64_t batch = static_cast<int64_t>(list.size());
    std::vector<int64_t> sizes_data(static_cast<size_t>(batch * rank), 0);
    std::vector<int64_t> strides_data(static_cast<size_t>(batch * rank), 0);
    std::vector<int64_t> offsets_data(static_cast<size_t>(batch), 0);
    int64_t running = 0;
    for (int64_t i = 0; i < batch; ++i) {
        for (int64_t j = 0; j < rank; ++j) {
            sizes_data[static_cast<size_t>(i * rank + j)] = list[i].size(j);
        }
        const auto row_strides =
            row_major_strides(sizes_data.data() + i * rank, rank);
        for (int64_t j = 0; j < rank; ++j) {
            strides_data[static_cast<size_t>(i * rank + j)] = row_strides[static_cast<size_t>(j)];
        }
        offsets_data[static_cast<size_t>(i)] = running;
        running += row_volume(sizes_data.data() + i * rank, rank);
    }

    // The impl keeps the flat 1-D geometry; the nested state carries the
    // per-constituent view of the same storage. The size/stride tables are
    // mounted as [batch, rank] grids.
    Tensor out(buffer.impl()->storage(), {buffer.numel()}, {1}, buffer.dtype(),
               0);
    out.impl()->set_nested_state(
        Tensor::tensor<int64_t>(sizes_data, DType::Int64)
            .to(target_device)
            .reshape({batch, rank})
            .impl(),
        Tensor::tensor<int64_t>(strides_data, DType::Int64)
            .to(target_device)
            .reshape({batch, rank})
            .impl(),
        Tensor::tensor<int64_t>(offsets_data, DType::Int64)
            .to(target_device)
            .impl());
    return out;
}

Tensor nested_size_accessor(const Tensor& self) {
    const auto st = require_nested_state(self, "_nested_tensor_size");
    return Tensor(st->nested_sizes);
}

Tensor nested_strides_accessor(const Tensor& self) {
    const auto st = require_nested_state(self, "_nested_tensor_strides");
    return Tensor(st->nested_strides);
}

Tensor nested_storage_offsets_accessor(const Tensor& self) {
    const auto st = require_nested_state(self, "_nested_tensor_storage_offsets");
    return Tensor(st->storage_offsets);
}

Tensor to_padded_tensor(
    const Tensor& self, double padding,
    std::optional<std::vector<int64_t>> output_size) {
    const auto st = require_nested_state(self, "to_padded_tensor");
    const Tensor sizes(st->nested_sizes);
    const Tensor offsets_t(st->storage_offsets);

    const int64_t batch = sizes.size(0);
    const int64_t rank = sizes.size(1);

    // Plain dense aliases: the flat buffer and its metadata are ordinary
    // tensors, so the primitives below never see the nested wrapper. The
    // host loop reads metadata through CPU copies (no-op for CPU buffers).
    Tensor flat(self.impl()->storage(), {self.impl()->numel()}, {1},
                self.dtype(),
                static_cast<size_t>(self.impl()->storage_offset()));
    const Device host = Device(DeviceType::CPU);
    Tensor offsets_flat = offsets_t.contiguous().to(host);

    // Padded extents: axis d (d >= 1) spans the largest extent any
    // constituent reports for that axis.
    std::vector<int64_t> extents(static_cast<size_t>(rank), 0);
    if (batch > 0 && rank > 0) {
        Tensor row_max =
            Tensor::amax(sizes, std::vector<int64_t>{0}).contiguous().to(host);
        const int64_t* mp = row_max.data_ptr<int64_t>();
        for (int64_t j = 0; j < rank; ++j) {
            extents[static_cast<size_t>(j)] = mp[j];
        }
    }

    std::vector<int64_t> out_shape;
    out_shape.push_back(batch);
    out_shape.insert(out_shape.end(), extents.begin(), extents.end());
    if (output_size.has_value()) {
        const auto& target = *output_size;
        if (static_cast<int64_t>(target.size()) != rank + 1) {
            TP_THROW(RuntimeError,
                     "to_padded_tensor(): output_size must have rank " +
                         std::to_string(rank + 1) + ", got " +
                         std::to_string(target.size()));
        }
        for (int64_t d = 0; d <= rank; ++d) {
            if (target[static_cast<size_t>(d)] < out_shape[static_cast<size_t>(d)]) {
                TP_THROW(RuntimeError,
                         "to_padded_tensor(): output_size must not truncate "
                         "the padded extents; dimension " + std::to_string(d) +
                             " needs at least " +
                             std::to_string(out_shape[static_cast<size_t>(d)]));
            }
        }
        out_shape = target;
    }

    Tensor out = Tensor::full(out_shape, Scalar(padding), self.dtype(),
                              self.device());
    if (batch == 0 || rank == 0) {
        return out;
    }

    const Tensor sizes_flat = sizes.contiguous().to(host);
    const int64_t* sp = sizes_flat.data_ptr<int64_t>();
    const int64_t* op = offsets_flat.data_ptr<int64_t>();
    for (int64_t i = 0; i < batch; ++i) {
        const int64_t* row = sp + i * rank;
        const int64_t vol = row_volume(row, rank);
        if (vol == 0) {
            continue;
        }
        std::vector<int64_t> row_shape;
        row_shape.reserve(static_cast<size_t>(rank));
        for (int64_t j = 0; j < rank; ++j) {
            row_shape.push_back(row[j]);
        }
        Tensor src = flat.narrow(0, op[i], vol).view(row_shape);
        // Constituents sit at the leading corner of their padded slice;
        // narrow the destination down to the row extent so the copy is
        // shape-exact and the trailing pad entries keep their fill value.
        Tensor dst = out.select(0, i);
        for (int64_t j = 0; j < rank; ++j) {
            dst = dst.narrow(j, 0, row[j]);
        }
        dst.copy_(src);
    }
    return out;
}

// Mount pre-computed metadata onto an existing buffer without copying: the
// result shares the buffer's storage and reports the given per-constituent
// shapes, strides and offsets. Callers guarantee the metadata describes the
// buffer faithfully (packed rows, row-major strides, or explicit blanks).
Tensor nested_view_from_buffer(
    const Tensor& buffer, const Tensor& nested_size,
    const Tensor& nested_strides, const Tensor& offsets) {
    if (nested_size.dim() != 2 || nested_strides.dim() != 2) {
        TP_THROW(RuntimeError,
                 "_nested_view_from_buffer(): nested_size and nested_strides "
                 "must be 2-D [batch, rank] tables");
    }
    if (nested_size.size(0) != nested_strides.size(0) ||
        offsets.dim() != 1 || offsets.size(0) != nested_size.size(0)) {
        TP_THROW(RuntimeError,
                 "_nested_view_from_buffer(): metadata must agree on the "
                 "number of constituents");
    }
    if (nested_size.dtype() != DType::Int64 ||
        nested_strides.dtype() != DType::Int64 ||
        offsets.dtype() != DType::Int64) {
        TP_THROW(RuntimeError,
                 "_nested_view_from_buffer(): metadata tensors must hold "
                 "64-bit integers");
    }
    Tensor out(buffer.impl()->storage(), {buffer.impl()->numel()}, {1},
               buffer.dtype(),
               static_cast<size_t>(buffer.impl()->storage_offset()));
    out.impl()->set_nested_state(nested_size.contiguous().impl(),
                                 nested_strides.contiguous().impl(),
                                 offsets.contiguous().impl());
    return out;
}

// Reads the impl flag directly: the generated Tensor member routes back
// through the dispatcher, which would re-enter this kernel.
bool is_nested_op(const Tensor& self) {
    return self.impl() && self.impl()->is_nested();
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(Composite, NestedTensorKernels) {
    m.impl("_nested_tensor_from_tensor_list", nested_from_tensor_list);
    m.impl("_nested_view_from_buffer", nested_view_from_buffer);
    m.impl("_nested_tensor_size", nested_size_accessor);
    m.impl("_nested_tensor_strides", nested_strides_accessor);
    m.impl("_nested_tensor_storage_offsets", nested_storage_offsets_accessor);
    m.impl("to_padded_tensor", to_padded_tensor);
    m.impl("nested_to_padded_tensor", to_padded_tensor);
    m.impl("is_nested", is_nested_op);
}

} // namespace composite
} // namespace tensorplay
