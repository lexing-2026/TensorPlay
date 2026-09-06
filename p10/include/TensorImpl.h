#pragma once

#include <array>
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <stdexcept>
#include <random>
#include <cstring>
#include <utility>
#include "Macros.h"
#include "DType.h"
#include "Device.h"
#include "DispatchKey.h"
#include "Dispatcher.h"
#include "InferenceMode.h"
#include "MemoryFormat.h"
#include "VariableVersion.h"
#include "Storage.h"
#include "SizesAndStrides.h"
#include "AutogradMetaBase.h"

namespace tensorplay {

class Tensor;
class Quantizer;

class P10_API TensorImpl {
private:
    size_t storage_offset_;
    SizesAndStrides sizes_and_strides_;
    DType dtype_;
    Device device_;
    tensorplay::VariableVersion version_counter_{!InferenceMode::is_enabled()};
    bool inference_tensor_ = InferenceMode::is_enabled();

    bool is_contiguous_;
    // Layout tag: Contiguous, ChannelsLast (NHWC) or ChannelsLast3d (NDHWC).
    // Never Preserve on a live tensor. Replaces the old is_channels_last_
    // bool so the full format (including 3-D) is representable.
    MemoryFormat memory_format_ = MemoryFormat::Contiguous;

    // Autograd extension point.
    // Never copied by TensorImpl copy operations: copies start without
    std::shared_ptr<AutogradMetaBase> autograd_meta_;

    // View identity for operations that alias the input storage, so detach_()
    // can reject views created by shallow_copy_and_detach.
    bool is_view_ = false;

    // True when the tensor was auto-wrapped from a C++/Python number
    // ('t + 2' wraps 2).  Wrapped numbers participate in the result type
    // computation only when no plain tensor operand is present, so integer
    // literals combine with floating tensors instead of demoting them.
    bool is_wrapped_number_ = false;

    // Opaque pointer to OneDNN memory descriptor (std::shared_ptr<dnnl::memory::desc>)
    // std::shared_ptr<void> onednn_md_;
    
    Storage storage_;

    struct SharedState {
        std::shared_ptr<void> onednn_md;
        std::shared_ptr<void> onednn_memory_cache; // Cache for OneDNN memory object (reordered)
    };
    std::shared_ptr<SharedState> shared_state_;

    // COO/CSR sparse tensors have no dense storage of their own.  Keep the
    // sparse metadata next to the dense metadata, split between logical sizes
    // and the storage implementation.  The
    // component tensors are shared handles, so ordinary Tensor copies
    // (shape [sparse_dim, nnz]); CSR (2D only) stores row pointers in
    // `crow` (shape [rows+1]) and column coordinates in `col` (shape [nnz]).
    // CSC stores column pointers in `crow` and row coordinates in `col`.
    // BSR/BSC additionally store a 3-D values tensor of blocks and record
    // the two block sizes in `blocksize`.
    struct SparseState {
        int layout = 0;  // 0 = SparseCOO, 1 = SparseCSR, 2 = SparseCSC,
                         // 3 = SparseBSR, 4 = SparseBSC
        std::shared_ptr<TensorImpl> indices;
        std::shared_ptr<TensorImpl> values;
        std::shared_ptr<TensorImpl> crow;
        std::shared_ptr<TensorImpl> col;
        std::array<int64_t, 2> blocksize = {0, 0};
        std::vector<int64_t> sparse_sizes;
        bool coalesced = false;
    };
    std::shared_ptr<SparseState> sparse_state_;

    // A transform wrapper keeps the physical value separate from its public
    // logical metadata. The wrapper is immutable with respect to storage;
    // batching rules create a new wrapper for each result.
    std::shared_ptr<TensorImpl> transform_value_;
    int64_t transform_batch_dim_ = -1;
    int64_t transform_level_ = -1;

    // Quantized tensors carry their affine parameters (scheme, scale,
    // zero point, per-channel tables) in an immutable, shared quantizer.
    // Plain integer tensors over the same storage keep this null.
    std::shared_ptr<Quantizer> quantizer_;

public:
    static constexpr int kSparseCOOLayout = 0;
    static constexpr int kSparseCSRLayout = 1;
    static constexpr int kSparseCSCLayout = 2;
    static constexpr int kSparseBSRLayout = 3;
    static constexpr int kSparseBSCLayout = 4;
    TensorImpl();
    TensorImpl(const std::vector<int64_t>& sizes, DType dtype, const Device& device = Device());
    // Sparse COO tensors need logical sizes and dtype/device metadata without
    // allocating a dense backing store for the full logical shape.
    TensorImpl(const std::vector<int64_t>& sizes, DType dtype,
               const Device& device, bool allocate_storage);
    TensorImpl(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, DType dtype, const Device& device = Device());
    TensorImpl(Storage storage, const std::vector<int64_t>& sizes, DType dtype, size_t storage_offset = 0);
    TensorImpl(Storage storage, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, DType dtype, size_t storage_offset = 0);
    TensorImpl(std::shared_ptr<TensorImpl> transform_value,
               const std::vector<int64_t>& sizes,
               const std::vector<int64_t>& strides);
    
    // Copy/Move
    TensorImpl(const TensorImpl& other);
    TensorImpl(TensorImpl&& other) noexcept;
    TensorImpl& operator=(const TensorImpl& other);
    TensorImpl& operator=(TensorImpl&& other) noexcept;
    
    ~TensorImpl() = default;

    // Storage
    const Storage& storage() const { return storage_; }
    size_t storage_offset() const { return storage_offset_; }
    void set_storage(Storage storage) { storage_ = std::move(storage); }
    void clear_storage() { storage_ = Storage(); }
    void set_storage_offset(size_t offset) { storage_offset_ = offset; }
    bool has_storage() const { return storage_.defined(); }
    
    // Metadata
    const SizesAndStrides& sizes_and_strides() const { return sizes_and_strides_; }
    IntArrayRef sizes() const { return sizes_and_strides_.sizes(); }
    IntArrayRef strides() const { return sizes_and_strides_.strides(); }
    int64_t size(size_t dim) const { return sizes_and_strides_.size(dim); }
    int64_t stride(size_t dim) const { return sizes_and_strides_.stride(dim); }
    size_t dim() const { return sizes_and_strides_.dim(); }
    int64_t numel() const { return sizes_and_strides_.numel(); }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }

    // The dispatch key set describing this tensor and its backend.
    DispatchKeySet key_set() const {
        DispatchKey backend = computeDispatchKey(device_);
        DispatchKeySet ks;
        ks.add(backend);
        if (is_batched()) {
            ks.add(toVmapKey(backend));
            return ks;
        }
        if (!inference_tensor_) {
            ks.add(toAutogradKey(backend));
        }
        return ks;
    }

    bool is_inference() const { return inference_tensor_; }

    bool is_batched() const { return transform_value_ != nullptr; }
    int64_t batch_dim() const { return transform_batch_dim_; }
    int64_t batch_level() const { return transform_level_; }
    int64_t batch_size() const {
        return is_batched() ? transform_value_->size(static_cast<size_t>(transform_batch_dim_)) : 0;
    }
    std::shared_ptr<TensorImpl> transform_value_impl() const { return transform_value_; }
    void set_transform_value(std::shared_ptr<TensorImpl> value,
                             int64_t batch_dim,
                             int64_t level) {
        transform_value_ = std::move(value);
        transform_batch_dim_ = batch_dim;
        transform_level_ = level;
    }
    void clear_transform_value() {
        transform_value_.reset();
        transform_batch_dim_ = -1;
        transform_level_ = -1;
    }
    
    size_t itemsize() const { return elementSize(dtype_); }
    bool is_contiguous() const { return is_contiguous_; }

    MemoryFormat memory_format() const {
        return memory_format_ == MemoryFormat::Preserve ? MemoryFormat::Contiguous
                                                        : memory_format_;
    }
    void set_memory_format(MemoryFormat format) {
        if (format == MemoryFormat::Preserve) format = MemoryFormat::Contiguous;
        memory_format_ = format;
        is_contiguous_ = is_contiguous_in(format);
    }
    bool is_channels_last() const { return memory_format_ == MemoryFormat::ChannelsLast; }
    bool is_channels_last_2d() const { return memory_format_ == MemoryFormat::ChannelsLast; }
    bool is_channels_last_3d() const { return memory_format_ == MemoryFormat::ChannelsLast3d; }

    // View identity: true when this tensor was created by a view op at the
    bool is_view() const { return is_view_; }
    void set_is_view(bool v) { is_view_ = v; }

    // True when this tensor was auto-wrapped from a C++ or Python number.
    bool is_wrapped_number() const { return is_wrapped_number_; }

    // Marks a tensor as a wrapped number; only meaningful for 0-dim tensors.
    void set_wrapped_number(bool value) {
        TP_CHECK(dim() == 0, "wrapped numbers must be 0-dim tensors");
        is_wrapped_number_ = value;
    }

    // Stride-set equality against `format`'s canonical layout; singleton
    // dimensions are not special-cased here.
    bool is_contiguous_in(MemoryFormat format) const {
        const auto sz = sizes_and_strides_.sizes().vec();
        const auto st = sizes_and_strides_.strides().vec();
        switch (format) {
            case MemoryFormat::Preserve:
                return is_contiguous_;
            case MemoryFormat::ChannelsLast:
            case MemoryFormat::ChannelsLast3d:
                return st == get_channels_last_strides(sz);
            default:
                return is_contiguous_;
        }
    }

    bool is_sparse() const { return sparse_state_ != nullptr; }
    int sparse_layout() const { return sparse_state_ ? sparse_state_->layout : kSparseCOOLayout; }
    // Any of the compressed layouts (CSR/CSC/BSR/BSC).
    bool is_sparse_compressed() const {
        return sparse_state_ && sparse_state_->layout != kSparseCOOLayout;
    }
    // Row-compressed (CSR/BSR): the compressed axis enumerates rows.
    bool is_sparse_row_compressed() const {
        return sparse_state_ && (sparse_state_->layout == kSparseCSRLayout ||
                                 sparse_state_->layout == kSparseBSRLayout);
    }
    bool is_sparse_csr() const {
        return sparse_state_ && sparse_state_->layout == kSparseCSRLayout;
    }
    bool is_sparse_csc() const {
        return sparse_state_ && sparse_state_->layout == kSparseCSCLayout;
    }
    bool is_sparse_bsr() const {
        return sparse_state_ && sparse_state_->layout == kSparseBSRLayout;
    }
    bool is_sparse_bsc() const {
        return sparse_state_ && sparse_state_->layout == kSparseBSCLayout;
    }
    // Block sizes for BSR/BSC; {0, 0} for the other layouts.
    std::array<int64_t, 2> sparse_blocksize() const {
        return sparse_state_ ? sparse_state_->blocksize
                             : std::array<int64_t, 2>{0, 0};
    }
    bool is_coalesced() const { return sparse_state_ && sparse_state_->coalesced; }
    std::shared_ptr<TensorImpl> sparse_indices_impl() const {
        return (sparse_state_ && sparse_state_->layout == kSparseCOOLayout)
                   ? sparse_state_->indices : nullptr;
    }
    std::shared_ptr<TensorImpl> sparse_values_impl() const {
        return sparse_state_ ? sparse_state_->values : nullptr;
    }
    // Compressed-axis pointers (crow for CSR/BSR, ccol for CSC/BSC).
    std::shared_ptr<TensorImpl> sparse_crow_impl() const {
        return (sparse_state_ && sparse_state_->layout != kSparseCOOLayout)
                   ? sparse_state_->crow : nullptr;
    }
    // Plain-axis coordinates (col for CSR/BSR, row for CSC/BSC).
    std::shared_ptr<TensorImpl> sparse_col_impl() const {
        return (sparse_state_ && sparse_state_->layout != kSparseCOOLayout)
                   ? sparse_state_->col : nullptr;
    }
    const std::vector<int64_t>& sparse_sizes() const {
        static const std::vector<int64_t> empty;
        return sparse_state_ ? sparse_state_->sparse_sizes : empty;
    }
    void set_sparse_state(std::shared_ptr<TensorImpl> indices,
                          std::shared_ptr<TensorImpl> values,
                          std::vector<int64_t> sparse_sizes,
                          bool coalesced) {
        sparse_state_ = std::make_shared<SparseState>();
        sparse_state_->layout = kSparseCOOLayout;
        sparse_state_->indices = std::move(indices);
        sparse_state_->values = std::move(values);
        sparse_state_->sparse_sizes = std::move(sparse_sizes);
        sparse_state_->coalesced = coalesced;
        is_contiguous_ = false;
        storage_offset_ = 0;
        clear_storage();
    }

    // CSR layout constructor (2D only): `crow` has rows+1 entries, `col` and
    // `values` have one entry per stored element.
    void set_sparse_csr_state(std::shared_ptr<TensorImpl> crow,
                              std::shared_ptr<TensorImpl> col,
                              std::shared_ptr<TensorImpl> values,
                              std::vector<int64_t> dense_sizes) {
        set_sparse_compressed_state(std::move(crow), std::move(col),
                                    std::move(values), std::move(dense_sizes),
                                    kSparseCSRLayout, {0, 0});
    }

    // Generic compressed-layout constructor (CSR/CSC/BSR/BSC).  `crow`
    // receives the compressed-axis pointers, `col` the plain-axis
    // coordinates; `blocksize` is meaningful for the blocked layouts.
    void set_sparse_compressed_state(std::shared_ptr<TensorImpl> crow,
                                     std::shared_ptr<TensorImpl> col,
                                     std::shared_ptr<TensorImpl> values,
                                     std::vector<int64_t> dense_sizes,
                                     int layout,
                                     std::array<int64_t, 2> blocksize) {
        TP_CHECK(crow != nullptr && col != nullptr && values != nullptr,
                 "set_sparse_compressed_state: crow/col/values must be defined");
        TP_CHECK(layout == kSparseCSRLayout || layout == kSparseCSCLayout ||
                     layout == kSparseBSRLayout || layout == kSparseBSCLayout,
                 "set_sparse_compressed_state: invalid compressed layout");
        sparse_state_ = std::make_shared<SparseState>();
        sparse_state_->layout = layout;
        sparse_state_->crow = std::move(crow);
        sparse_state_->col = std::move(col);
        sparse_state_->values = std::move(values);
        sparse_state_->blocksize =
            (layout == kSparseBSRLayout || layout == kSparseBSCLayout)
                ? blocksize : std::array<int64_t, 2>{0, 0};
        sparse_state_->sparse_sizes = std::move(dense_sizes);
        sparse_state_->coalesced = true;  // compressed layouts are canonical
        is_contiguous_ = false;
        storage_offset_ = 0;
        clear_storage();
    }

    // the counter with their base via share_version_counter().
    uint32_t version() const {
        if (!version_counter_.is_enabled()) {
            throw std::runtime_error("Inference tensors do not track version counter.");
        }
        return version_counter_.current_version();
    }
    void bump_version() {
        if (inference_tensor_ && !InferenceMode::is_enabled()) {
            throw std::runtime_error(
                "In-place update to an inference tensor outside inference mode is not allowed.");
        }
        version_counter_.bump();
    }
    // Zero the counter (used when a fresh tensor is materialized by an
    // internal copy such as clone(), so the result starts unmutated).
    void reset_version() { version_counter_.reset(); }
    void set_version_counter(const VariableVersion& vc) {
        if (inference_tensor_ && vc.is_enabled()) {
            throw std::runtime_error("Cannot set a version counter on an inference tensor.");
        }
        version_counter_ = vc;
    }
    const VariableVersion& version_counter() const { return version_counter_; }
    // Make this tensor's version counter alias `other`'s (view semantics).
    void share_version_counter(const TensorImpl& other) {
        inference_tensor_ = other.inference_tensor_;
        version_counter_ = other.inference_tensor_
            ? VariableVersion(false)
            : other.version_counter_;
    }
    
    // Copy metadata (storage, sizes, strides, dtype, device) from another TensorImpl
    // preserving autograd_meta
    void copy_metadata_from(const TensorImpl& other);
    
    // Access to raw data pointer (typed)
    template<typename T>
    T* data() const {
        if (!has_storage()) return nullptr;
        void* base_ptr = storage_.data();
#ifdef USE_CUDA
        if (device_.is_cuda()) cuda::recordStream(base_ptr, device_);
#endif
        return static_cast<T*>(base_ptr) + storage_offset_;
    }
    
    void* data() const {
        if (!has_storage()) return nullptr;
        void* base_ptr = storage_.data();
#ifdef USE_CUDA
        if (device_.is_cuda()) cuda::recordStream(base_ptr, device_);
#endif
        size_t elem_size = elementSize(dtype_);
        return static_cast<char*>(base_ptr) + storage_offset_ * elem_size;
    }

    void set_onednn_md(std::shared_ptr<void> md) {
        if (!shared_state_) shared_state_ = std::make_shared<SharedState>();
        shared_state_->onednn_md = std::move(md);
    }
    std::shared_ptr<void> get_onednn_md() const {
        return shared_state_ ? shared_state_->onednn_md : nullptr;
    }
    bool has_onednn_md() const {
        return shared_state_ && shared_state_->onednn_md != nullptr;
    }

    void set_onednn_memory_cache(std::shared_ptr<void> mem) {
        if (!shared_state_) shared_state_ = std::make_shared<SharedState>();
        shared_state_->onednn_memory_cache = std::move(mem);
    }
    std::shared_ptr<void> get_onednn_memory_cache() const {
        return shared_state_ ? shared_state_->onednn_memory_cache : nullptr;
    }
    bool has_onednn_memory_cache() const {
        return shared_state_ && shared_state_->onednn_memory_cache != nullptr;
    }

    void set_autograd_meta(std::shared_ptr<AutogradMetaBase> meta) { autograd_meta_ = std::move(meta); }
    AutogradMetaBase* autograd_meta() const { return autograd_meta_.get(); }
    bool has_autograd_meta() const { return autograd_meta_ != nullptr; }
    std::shared_ptr<AutogradMetaBase> autograd_meta_shared() const { return autograd_meta_; }
    void set_requires_grad(bool requires_grad);

    // Quantizer metadata: present exactly on quantized tensors.  The
    // quantizer is shared between views/copies and is immutable.
    void set_quantizer(std::shared_ptr<Quantizer> q) { quantizer_ = std::move(q); }
    std::shared_ptr<Quantizer> quantizer() const { return quantizer_; }
    bool has_quantizer() const { return quantizer_ != nullptr; }

    // resize_ support: adopt new logical sizes with fresh contiguous strides.
    // Storage growth is the caller's job (kernels grow it first so the old
    // contents stay readable in place).
    void set_sizes_contiguous(const std::vector<int64_t>& new_sizes) {
        sizes_and_strides_.resize(new_sizes);
        is_contiguous_ = true;
        memory_format_ = MemoryFormat::Contiguous;
    }

    // Adopt explicit sizes/strides (used by contiguous(memory_format) and
    // other layout-materializing paths); layout flags are recomputed.
    void set_sizes_and_strides(const std::vector<int64_t>& new_sizes,
                               const std::vector<int64_t>& new_strides) {
        sizes_and_strides_.set_sizes_and_strides(new_sizes, new_strides);
        is_contiguous_ =
            new_strides == SizesAndStrides::compute_contiguous_strides(new_sizes);
        if (is_contiguous_) {
            memory_format_ = MemoryFormat::Contiguous;
        } else if (new_strides == get_channels_last_strides(new_sizes)) {
            memory_format_ = new_sizes.size() == 5 ? MemoryFormat::ChannelsLast3d
                                                   : MemoryFormat::ChannelsLast;
        }
    }
    
    // Share storage state from another TensorImpl
    void share_storage_from(const TensorImpl& other) {
        storage_ = other.storage_;
        shared_state_ = other.shared_state_;
    }
};

} // namespace tensorplay
