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

#include <algorithm>
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
    const std::optional<std::vector<int64_t>>& output_size) {
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

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

// The values are a plain 1-D dense alias of the backing buffer; the
// per-constituent view lives entirely in the metadata tables.
Tensor nested_get_values(const Tensor& self) {
    require_nested_state(self, "_nested_get_values");
    return Tensor(self.impl()->storage(), {self.impl()->numel()}, {1},
                  self.dtype(),
                  static_cast<size_t>(self.impl()->storage_offset()));
}

Tensor nested_get_values_copy(const Tensor& self) {
    return nested_get_values(self).clone();
}

Tensor nested_get_offsets(const Tensor& self) {
    const auto st = require_nested_state(self, "_nested_get_offsets");
    return Tensor(st->storage_offsets);
}

// In this representation the ragged axis is the first column of the size
// table: row i holds [len_i, trailing extents...].
Tensor nested_get_lengths(const Tensor& self) {
    const auto st = require_nested_state(self, "_nested_get_lengths");
    const Tensor sizes(st->nested_sizes);
    if (sizes.size(1) == 0) {
        return Tensor::zeros({sizes.size(0)}, DType::Int64, sizes.device());
    }
    return sizes.select(1, 0).contiguous();
}

int64_t nested_get_ragged_idx(const Tensor& self) {
    require_nested_state(self, "_nested_get_ragged_idx");
    // The ragged axis sits at logical dim 1, i.e. column 0 of the size table.
    return 0;
}

// Sequence-length statistics over the ragged axis, read through a host copy
// (the tables are tiny and may live beside a non-CPU buffer).
Tensor nested_seqlen_stat(const Tensor& self, const char* op, bool want_min) {
    const auto st = require_nested_state(self, op);
    const Tensor sizes(st->nested_sizes);
    const int64_t batch = sizes.size(0);
    if (batch == 0 || sizes.size(1) == 0) {
        return Tensor::empty({0}, DType::Int64, sizes.device());
    }
    Tensor lens = sizes.select(1, 0).contiguous().to(Device(DeviceType::CPU));
    const int64_t* p = lens.data_ptr<int64_t>();
    int64_t acc = p[0];
    for (int64_t i = 1; i < batch; ++i) {
        acc = want_min ? std::min(acc, p[i]) : std::max(acc, p[i]);
    }
    return Tensor::tensor<int64_t>({acc}, DType::Int64).to(sizes.device());
}

Tensor nested_get_min_seqlen(const Tensor& self) {
    return nested_seqlen_stat(self, "_nested_get_min_seqlen", true);
}

Tensor nested_get_max_seqlen(const Tensor& self) {
    return nested_seqlen_stat(self, "_nested_get_max_seqlen", false);
}

// An empty tensor that carries the device and dtype for the jagged view
// entry points, which take it as an opaque example argument.
Tensor nested_get_jagged_dummy(const Tensor& any) {
    return Tensor::empty({0}, any.dtype(), any.device());
}

// Row-major strides per size row plus the packed start offset of each row:
// the last axis advances by one element and each earlier axis by its extent
// times the stride of the axis after it; constituent i starts at the total
// volume of the rows before it.
std::tuple<Tensor, Tensor> nested_compute_contiguous_strides_offsets(
    const Tensor& nested_size) {
    if (nested_size.dim() != 2) {
        TP_THROW(RuntimeError,
                 "_nested_compute_contiguous_strides_offsets(): expected a "
                 "2-D [batch, rank] size table");
    }
    const Device host = Device(DeviceType::CPU);
    Tensor flat = nested_size.contiguous().to(host);
    const int64_t batch = flat.size(0), rank = flat.size(1);
    const int64_t* p = flat.data_ptr<int64_t>();
    std::vector<int64_t> strides_data(static_cast<size_t>(batch * rank), 1);
    std::vector<int64_t> offsets_data(static_cast<size_t>(batch), 0);
    int64_t running = 0;
    for (int64_t i = 0; i < batch; ++i) {
        const auto row = row_major_strides(p + i * rank, rank);
        for (int64_t j = 0; j < rank; ++j) {
            strides_data[static_cast<size_t>(i * rank + j)] = row[static_cast<size_t>(j)];
        }
        offsets_data[static_cast<size_t>(i)] = running;
        running += row_volume(p + i * rank, rank);
    }
    const Device dev = nested_size.device();
    return std::make_tuple(
        Tensor::tensor<int64_t>(strides_data, DType::Int64).to(dev)
            .reshape({batch, rank}),
        Tensor::tensor<int64_t>(offsets_data, DType::Int64).to(dev));
}

// ---------------------------------------------------------------------------
// Construction from padded buffers and masks
// ---------------------------------------------------------------------------

// Shared mount for the padded/jagged entry points: buffer stays dense and
// row i keeps its trailing extents; metadata tables are built on the buffer
// device.  `rows` holds per-constituent sizes, `element_offsets` the element
// position of each row's first entry inside the flat buffer.
Tensor mount_rows(const Tensor& buffer,
                  const std::vector<std::vector<int64_t>>& rows,
                  const std::vector<int64_t>& element_offsets) {
    const int64_t batch = static_cast<int64_t>(rows.size());
    const int64_t rank = batch > 0 ? static_cast<int64_t>(rows[0].size()) : 0;
    std::vector<int64_t> strides_data(static_cast<size_t>(batch * rank), 1);
    std::vector<int64_t> sizes_flat;
    sizes_flat.reserve(static_cast<size_t>(batch * rank));
    for (int64_t i = 0; i < batch; ++i) {
        const auto row = row_major_strides(rows[static_cast<size_t>(i)].data(), rank);
        for (int64_t j = 0; j < rank; ++j) {
            strides_data[static_cast<size_t>(i * rank + j)] = row[static_cast<size_t>(j)];
        }
        sizes_flat.insert(sizes_flat.end(),
                          rows[static_cast<size_t>(i)].begin(),
                          rows[static_cast<size_t>(i)].end());
    }
    const Device dev = buffer.device();
    Tensor out(buffer.impl()->storage(), {buffer.impl()->numel()}, {1},
               buffer.dtype(),
               static_cast<size_t>(buffer.impl()->storage_offset()));
    out.impl()->set_nested_state(
        Tensor::tensor<int64_t>(sizes_flat, DType::Int64).to(dev)
            .reshape({batch, rank})
            .impl(),
        Tensor::tensor<int64_t>(strides_data, DType::Int64).to(dev)
            .reshape({batch, rank})
            .impl(),
        Tensor::tensor<int64_t>(element_offsets, DType::Int64).to(dev)
            .impl());
    return out;
}

// A padded batch [N, L, D] where each row keeps only its first len_i
// positions: row i starts at i * L * D inside the flat buffer.
Tensor nested_from_padded(const Tensor& padded, const Tensor& nested_size,
                          bool fuse_transform_0213) {
    (void)fuse_transform_0213;  // the 0213 transpose shortcut only affects an
                                // internal layout this representation lacks
    if (padded.dim() != 3) {
        TP_THROW(RuntimeError,
                 "_nested_from_padded(): expected a 3-D [N, L, D] padded "
                 "batch, got dim " + std::to_string(padded.dim()));
    }
    if (nested_size.dim() != 2 || nested_size.size(1) != 2 ||
        nested_size.size(0) != padded.size(0)) {
        TP_THROW(RuntimeError,
                 "_nested_from_padded(): expected a [N, 2] size table whose "
                 "rows are [len_i, D]");
    }
    const Device host = Device(DeviceType::CPU);
    Tensor flat = nested_size.contiguous().to(host);
    const int64_t n = padded.size(0), padded_len = padded.size(1);
    const int64_t d = padded.size(2);
    const int64_t* p = flat.data_ptr<int64_t>();
    std::vector<std::vector<int64_t>> rows;
    std::vector<int64_t> element_offsets;
    rows.reserve(static_cast<size_t>(n));
    element_offsets.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const int64_t len = p[i * 2];
        if (len < 0 || len > padded_len) {
            TP_THROW(RuntimeError,
                     "_nested_from_padded(): length " + std::to_string(len) +
                         " at row " + std::to_string(i) +
                         " exceeds the padded length " +
                         std::to_string(padded_len));
        }
        if (p[i * 2 + 1] != d) {
            TP_THROW(RuntimeError,
                     "_nested_from_padded(): size table column 1 must equal "
                     "the padded last dimension");
        }
        rows.push_back({len, d});
        element_offsets.push_back(i * padded_len * d);
    }
    return mount_rows(padded, rows, element_offsets);
}

// A padded batch [B, L, ...] plus cumulative ragged ends: row i keeps its
// first lengths[i] = offsets[i+1] - offsets[i] positions along axis 1, and
// the ragged axis is logical dim 1 (the only placement this layout carries).
Tensor nested_from_padded_tensor(
    const Tensor& padded, const Tensor& offsets, const Tensor& dummy,
    int64_t ragged_idx, const std::optional<Tensor>& min_seqlen_opt, const std::optional<Tensor>& max_seqlen_opt,
    std::optional<int64_t> sum_s) {
    const Tensor min_seqlen = min_seqlen_opt.has_value() ? *min_seqlen_opt : Tensor();
    const Tensor max_seqlen = max_seqlen_opt.has_value() ? *max_seqlen_opt : Tensor();
    (void)dummy;
    (void)min_seqlen;
    (void)max_seqlen;
    (void)sum_s;
    if (ragged_idx != 1) {
        TP_THROW(RuntimeError,
                 "_nested_from_padded_tensor(): only ragged_idx=1 is "
                 "supported, got " + std::to_string(ragged_idx));
    }
    if (padded.dim() < 2 || offsets.dim() != 1 ||
        offsets.size(0) != padded.size(0) + 1) {
        TP_THROW(RuntimeError,
                 "_nested_from_padded_tensor(): expected a [B, L, ...] padded "
                 "batch and B+1 cumulative offsets");
    }
    const Device host = Device(DeviceType::CPU);
    Tensor off = offsets.contiguous().to(host);
    const int64_t* op = off.data_ptr<int64_t>();
    const int64_t padded_len = padded.size(1);
    int64_t trail = 1;
    for (int64_t j = 2; j < padded.dim(); ++j) trail *= padded.size(j);
    std::vector<std::vector<int64_t>> rows;
    std::vector<int64_t> element_offsets;
    for (int64_t i = 0; i < padded.size(0); ++i) {
        const int64_t len = op[i + 1] - op[i];
        if (len < 0 || len > padded_len) {
            TP_THROW(RuntimeError,
                     "_nested_from_padded_tensor(): length " +
                         std::to_string(len) + " at row " +
                         std::to_string(i) + " exceeds the padded length " +
                         std::to_string(padded_len));
        }
        std::vector<int64_t> row;
        row.push_back(len);
        for (int64_t j = 2; j < padded.dim(); ++j) row.push_back(padded.size(j));
        rows.push_back(row);
        element_offsets.push_back(i * padded_len * trail);
    }
    return mount_rows(padded, rows, element_offsets);
}

// A jagged values batch [sum_S, ...] plus cumulative offsets: row i is the
// slice values[offsets[i]:offsets[i+1]], so its element offset is
// offsets[i] scaled by the trailing volume.
Tensor nested_view_from_jagged(const Tensor& values, const Tensor& offsets,
                               const Tensor& dummy, const Tensor& lengths,
                               int64_t ragged_idx, const Tensor& min_seqlen,
                               const Tensor& max_seqlen, bool copy) {
    (void)dummy;
    (void)min_seqlen;
    (void)max_seqlen;
    if (ragged_idx != 1) {
        TP_THROW(RuntimeError,
                 "_nested_view_from_jagged(): only ragged_idx=1 is supported, "
                 "got " + std::to_string(ragged_idx));
    }
    if (values.dim() < 1 || offsets.dim() != 1 || offsets.size(0) < 1) {
        TP_THROW(RuntimeError,
                 "_nested_view_from_jagged(): expected values of rank >= 1 "
                 "and a 1-D offsets tensor");
    }
    const Device host = Device(DeviceType::CPU);
    Tensor off = offsets.contiguous().to(host);
    const int64_t* op = off.data_ptr<int64_t>();
    const int64_t batch = off.size(0) - 1;
    if (op[0] != 0) {
        TP_THROW(RuntimeError,
                 "_nested_view_from_jagged(): offsets must start at zero");
    }
    int64_t trail = 1;
    for (int64_t j = 1; j < values.dim(); ++j) trail *= values.size(j);
    if (op[batch] * trail > values.numel()) {
        TP_THROW(RuntimeError,
                 "_nested_view_from_jagged(): offsets reach past the values "
                 "buffer");
    }
    std::vector<std::vector<int64_t>> rows;
    std::vector<int64_t> element_offsets;
    for (int64_t i = 0; i < batch; ++i) {
        const int64_t len = op[i + 1] - op[i];
        if (len < 0) {
            TP_THROW(RuntimeError,
                     "_nested_view_from_jagged(): offsets must be "
                     "non-decreasing");
        }
        std::vector<int64_t> row;
        row.push_back(len);
        for (int64_t j = 1; j < values.dim(); ++j) row.push_back(values.size(j));
        rows.push_back(row);
        element_offsets.push_back(op[i] * trail);
    }
    Tensor buffer = copy ? values.clone() : values;
    return mount_rows(buffer, rows, element_offsets);
}

// A padding mask [N, L] over a batch [N, L, D]: the per-row true count is the
// constituent length, and a left-aligned mask (true prefix, then false) is
// required so each row's kept slice is a contiguous prefix.
Tensor mask_row_lengths(const Tensor& t, const Tensor& mask,
                        const char* op, bool check) {
    if (mask.dtype() != DType::Bool) {
        TP_THROW(RuntimeError,
                 std::string(op) + ": expected a Bool mask");
    }
    if (mask.dim() != 2 || t.dim() != 3 || t.size(0) != mask.size(0) ||
        t.size(1) != mask.size(1)) {
        TP_THROW(RuntimeError,
                 std::string(op) +
                     ": expected a 3-D [N, L, D] input and a 2-D [N, L] mask "
                     "matching its first two dimensions");
    }
    const int64_t n = t.size(0), seq_len = t.size(1);
    // Per-row length: the count of true entries.  The first false position
    // (with one sentinel false appended) is the prefix length; the two agree
    // exactly when the mask is left-aligned without gaps.
    Tensor counts = mask.to(DType::Int64).cumsum(1).select(1, seq_len - 1);
    Tensor first_false = Tensor::cat(
                            {mask, Tensor::zeros({n, 1}, DType::Bool,
                                                 mask.device())},
                            1)
                            .to(DType::Int64)
                            .argmin(1);
    if (check) {
        Tensor counts_host = counts.to(DType::Int64).contiguous()
                                 .to(Device(DeviceType::CPU));
        Tensor nums_host = first_false.contiguous().to(Device(DeviceType::CPU));
        const int64_t* cp = counts_host.data_ptr<int64_t>();
        const int64_t* np = nums_host.data_ptr<int64_t>();
        for (int64_t i = 0; i < n; ++i) {
            if (cp[i] != np[i]) {
                TP_THROW(RuntimeError,
                         std::string(op) +
                             ": the mask must be left-aligned without gaps");
            }
        }
    }
    return counts.to(DType::Int64).reshape({n, 1});
}

Tensor nested_from_mask(const Tensor& t, const Tensor& mask,
                        bool mask_check) {
    Tensor counts = mask_row_lengths(t, mask, "_nested_tensor_from_mask",
                                     mask_check);
    const int64_t d = t.size(2);
    Tensor d_col = Tensor::full({t.size(0), 1}, Scalar(d), DType::Int64,
                                counts.device());
    Tensor table = Tensor::cat({counts, d_col}, 1);
    return nested_from_padded(t, table, false);
}

bool nested_from_mask_left_aligned(const Tensor& t, const Tensor& mask) {
    try {
        mask_row_lengths(t, mask, "_nested_tensor_from_mask_left_aligned",
                         true);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// ---------------------------------------------------------------------------
// Autograd support kernels
// ---------------------------------------------------------------------------

// Gradient of a sum over the ragged (last metadata) axis: each gradient
// element repeats across that axis's extent in the corresponding row.  The
// gradient arrives as a nested tensor whose size table matches the input's
// with the summed axis removed (or sized 1 under keepdim).
Tensor nested_sum_backward(const Tensor& grad, const Tensor& self,
                           const std::optional<std::vector<int64_t>>& dims,
                           bool keepdim) {
    (void)dims;
    (void)keepdim;
    const auto self_st = require_nested_state(self, "_nested_sum_backward");
    const auto grad_st = require_nested_state(grad, "_nested_sum_backward");
    const Tensor grad_sizes(grad_st->nested_sizes);
    const Tensor self_sizes(self_st->nested_sizes);
    const Device host = Device(DeviceType::CPU);
    const int64_t batch = self_sizes.size(0);
    const int64_t rank = self_sizes.size(1);

    Tensor grad_sizes_host = grad_sizes.contiguous().to(host);
    Tensor self_sizes_host = self_sizes.contiguous().to(host);
    const int64_t* gp = grad_sizes_host.data_ptr<int64_t>();
    const int64_t* sp = self_sizes_host.data_ptr<int64_t>();

    Tensor out_flat =
        Tensor::zeros({self.impl()->numel()}, self.dtype(), self.device());
    Tensor grad_flat = nested_get_values(grad);
    // The gradient's table drops (or shrinks) the summed axis, so its rows
    // are indexed with the gradient's own rank.
    const int64_t grad_rank = grad_sizes.size(1);
    int64_t grad_cursor = 0, self_cursor = 0;
    for (int64_t i = 0; i < batch; ++i) {
        int64_t segments = 1;
        for (int64_t j = 0; j < grad_rank; ++j) {
            segments *= gp[i * grad_rank + j];
        }
        const int64_t seg_len = sp[i * rank + rank - 1];
        const int64_t span = segments * seg_len;
        if (span == 0) {
            continue;
        }
        Tensor chunk = grad_flat.narrow(0, grad_cursor, segments)
                           .view({segments, 1})
                           .expand({segments, seg_len})
                           .reshape({span});
        out_flat.narrow(0, self_cursor, span).copy_(chunk);
        grad_cursor += segments;
        self_cursor += span;
    }

    return nested_view_from_buffer(
        out_flat, Tensor(self_st->nested_sizes), Tensor(self_st->nested_strides),
        Tensor(self_st->storage_offsets));
}

// Gradient of a select: the buffer starts at zero and the selected position
// in each row receives the gradient slice.  dim 0 picks one constituent;
// deeper dims pick the same axis position in every row.  The generated
// dispatch surface passes the symbolic index as a plain integer.
Tensor nested_select_backward(const Tensor& grad, const Tensor& self,
                              int64_t dim, int64_t index) {
    const auto st = require_nested_state(self, "_nested_select_backward");
    if (index < 0) {
        TP_THROW(RuntimeError,
                 "_nested_select_backward(): expected a non-negative index");
    }

    Tensor self_flat = nested_get_values(self);
    Tensor out_flat =
        Tensor::zeros({self.impl()->numel()}, self.dtype(), self.device());
    Tensor out = nested_view_from_buffer(
        out_flat, Tensor(st->nested_sizes), Tensor(st->nested_strides),
        Tensor(st->storage_offsets));

    const Device host = Device(DeviceType::CPU);
    Tensor sizes_host = Tensor(st->nested_sizes).contiguous().to(host);
    Tensor offsets_host = Tensor(st->storage_offsets).contiguous().to(host);
    const int64_t batch = sizes_host.size(0), rank = sizes_host.size(1);
    const int64_t* sp = sizes_host.data_ptr<int64_t>();
    const int64_t* op = offsets_host.data_ptr<int64_t>();

    if (dim == 0) {
        if (index >= batch) {
            TP_THROW(IndexError,
                     "_nested_select_backward(): constituent index out of "
                     "range");
        }
        const int64_t vol = row_volume(sp + index * rank, rank);
        if (vol == 0) {
            return out;
        }
        std::vector<int64_t> row_shape;
        for (int64_t j = 0; j < rank; ++j) {
            row_shape.push_back(sp[index * rank + j]);
        }
        out_flat.narrow(0, op[index], vol)
            .view(row_shape)
            .copy_(grad.reshape(row_shape));
        return out;
    }

    if (dim < 1 || dim > rank) {
        TP_THROW(RuntimeError,
                 "_nested_select_backward(): dim out of range");
    }
    for (int64_t i = 0; i < batch; ++i) {
        if (index >= sp[i * rank + dim - 1]) {
            TP_THROW(IndexError,
                     "_nested_select_backward(): index out of range for row " +
                         std::to_string(i));
        }
        const int64_t vol = row_volume(sp + i * rank, rank);
        if (vol == 0) {
            continue;
        }
        std::vector<int64_t> row_shape;
        for (int64_t j = 0; j < rank; ++j) {
            row_shape.push_back(sp[i * rank + j]);
        }
        Tensor dst = out_flat.narrow(0, op[i], vol).view(row_shape)
                         .select(dim - 1, index);
        dst.copy_(grad.select(0, i));
    }
    return out;
}

// Softmax over each row's ragged (last) axis, numerically stabilized by the
// per-row maximum.  The query argument carries the expected leading shape
// for the caller; the reduction itself depends only on the scores.
Tensor nested_softmax_with_shape(const Tensor& scores, const Tensor& query) {
    (void)query;
    const auto st =
        require_nested_state(scores, "_nested_tensor_softmax_with_shape");
    Tensor flat = nested_get_values(scores);
    Tensor out_flat = Tensor::empty({scores.impl()->numel()}, scores.dtype(),
                                    scores.device());

    const Device host = Device(DeviceType::CPU);
    Tensor sizes_host = Tensor(st->nested_sizes).contiguous().to(host);
    Tensor offsets_host = Tensor(st->storage_offsets).contiguous().to(host);
    const int64_t batch = sizes_host.size(0), rank = sizes_host.size(1);
    const int64_t* sp = sizes_host.data_ptr<int64_t>();
    const int64_t* op = offsets_host.data_ptr<int64_t>();

    for (int64_t i = 0; i < batch; ++i) {
        const int64_t vol = row_volume(sp + i * rank, rank);
        if (vol == 0) {
            continue;
        }
        std::vector<int64_t> row_shape;
        for (int64_t j = 0; j < rank; ++j) {
            row_shape.push_back(sp[i * rank + j]);
        }
        Tensor row = flat.narrow(0, op[i], vol).view(row_shape);
        Tensor shifted = row - Tensor::amax(row, {rank - 1}, true);
        Tensor e = shifted.exp();
        e = e / Tensor::sum(e, {rank - 1}, true);
        out_flat.narrow(0, op[i], vol).copy_(e.reshape({vol}));
    }
    return nested_view_from_buffer(
        out_flat, Tensor(st->nested_sizes), Tensor(st->nested_strides),
        Tensor(st->storage_offsets));
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

    m.impl("_nested_get_values", nested_get_values);
    m.impl("_nested_get_values_copy", nested_get_values_copy);
    m.impl("_nested_get_offsets", nested_get_offsets);
    m.impl("_nested_get_lengths", nested_get_lengths);
    m.impl("_nested_get_ragged_idx", nested_get_ragged_idx);
    m.impl("_nested_get_min_seqlen", nested_get_min_seqlen);
    m.impl("_nested_get_max_seqlen", nested_get_max_seqlen);
    m.impl("_nested_get_jagged_dummy", nested_get_jagged_dummy);
    m.impl("_nested_compute_contiguous_strides_offsets",
           nested_compute_contiguous_strides_offsets);

    m.impl("_nested_from_padded", nested_from_padded);
    m.impl("_nested_from_padded_tensor", nested_from_padded_tensor);
    m.impl("_nested_view_from_jagged",
           [](const Tensor& values, const Tensor& offsets, const Tensor& dummy,
              const std::optional<Tensor>& lengths, int64_t ragged_idx,
              const std::optional<Tensor>& min_seqlen,
              const std::optional<Tensor>& max_seqlen) {
               return nested_view_from_jagged(
                   values, offsets, dummy, lengths.value_or(Tensor()), ragged_idx,
                   min_seqlen.value_or(Tensor()), max_seqlen.value_or(Tensor()), false);
           });
    m.impl("_nested_view_from_jagged_copy",
           [](const Tensor& values, const Tensor& offsets, const Tensor& dummy,
              const std::optional<Tensor>& lengths, int64_t ragged_idx,
              const std::optional<Tensor>& min_seqlen,
              const std::optional<Tensor>& max_seqlen) {
               return nested_view_from_jagged(
                   values, offsets, dummy, lengths.value_or(Tensor()), ragged_idx,
                   min_seqlen.value_or(Tensor()), max_seqlen.value_or(Tensor()), true);
           });
    m.impl("_nested_tensor_from_mask", nested_from_mask);
    m.impl("_nested_tensor_from_mask_left_aligned",
           nested_from_mask_left_aligned);

    m.impl("_nested_sum_backward", nested_sum_backward);
    m.impl("_nested_select_backward", nested_select_backward);
    m.impl("_nested_tensor_softmax_with_shape", nested_softmax_with_shape);
}

} // namespace composite
} // namespace tensorplay
