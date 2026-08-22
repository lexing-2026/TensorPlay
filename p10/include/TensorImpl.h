#pragma once

#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <stdexcept>
#include <random>
#include <cstring>
#include "Macros.h"
#include "DType.h"
#include "Device.h"
#include "DispatchKey.h"
#include "Dispatcher.h"
#include "VariableVersion.h"
#include "Storage.h"
#include "SizesAndStrides.h"
#include "AutogradMetaBase.h"

namespace tensorplay {

class Tensor;

class P10_API TensorImpl {
private:
    size_t storage_offset_;
    SizesAndStrides sizes_and_strides_;
    DType dtype_;
    Device device_;
    tensorplay::VariableVersion version_counter_;

    bool is_contiguous_;
    bool is_channels_last_;

    // Autograd extension point (mirrors c10::TensorImpl::autograd_meta_).
    // Never copied by TensorImpl copy operations: copies start without
    // autograd metadata, matching PyTorch semantics.
    std::shared_ptr<AutogradMetaBase> autograd_meta_;

    // Opaque pointer to OneDNN memory descriptor (std::shared_ptr<dnnl::memory::desc>)
    // std::shared_ptr<void> onednn_md_;
    
    struct SharedState {
        Storage storage;
        std::shared_ptr<void> onednn_md;
        std::shared_ptr<void> onednn_memory_cache; // Cache for OneDNN memory object (reordered)
    };
    std::shared_ptr<SharedState> shared_state_;

    // COO sparse tensors have no dense storage of their own.  Keep the COO
    // metadata next to the dense metadata, mirroring TensorImpl's split
    // between logical sizes and the storage implementation.  The component
    // tensors are shared handles, so ordinary Tensor copies preserve the
    // aliasing rules of torch.sparse_coo_tensor(indices, values, ...).
    struct SparseState {
        std::shared_ptr<TensorImpl> indices;
        std::shared_ptr<TensorImpl> values;
        std::vector<int64_t> sparse_sizes;
        bool coalesced = false;
    };
    std::shared_ptr<SparseState> sparse_state_;

public:
    TensorImpl();
    TensorImpl(const std::vector<int64_t>& sizes, DType dtype, const Device& device = Device());
    // Sparse COO tensors need logical sizes and dtype/device metadata without
    // allocating a dense backing store for the full logical shape.
    TensorImpl(const std::vector<int64_t>& sizes, DType dtype,
               const Device& device, bool allocate_storage);
    TensorImpl(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, DType dtype, const Device& device = Device());
    TensorImpl(Storage storage, const std::vector<int64_t>& sizes, DType dtype, size_t storage_offset = 0);
    TensorImpl(Storage storage, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, DType dtype, size_t storage_offset = 0);
    
    // Copy/Move
    TensorImpl(const TensorImpl& other);
    TensorImpl(TensorImpl&& other) noexcept;
    TensorImpl& operator=(const TensorImpl& other);
    TensorImpl& operator=(TensorImpl&& other) noexcept;
    
    ~TensorImpl() = default;

    // Storage
    const Storage& storage() const { return shared_state_->storage; }
    size_t storage_offset() const { return storage_offset_; }
    void set_storage(Storage storage) { shared_state_->storage = storage; }
    void clear_storage() { shared_state_->storage = Storage(); }
    void set_storage_offset(size_t offset) { storage_offset_ = offset; }
    bool has_storage() const { return shared_state_ && shared_state_->storage.defined(); }
    
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

    // The dispatch key set describing this tensor: its backend key plus the
    // matching autograd key (mirrors c10::TensorImpl::key_set_, which always
    // carries an Autograd key for dense tensors; the autograd kernel itself
    // decides whether to record based on GradMode/requires_grad).
    DispatchKeySet key_set() const {
        DispatchKey backend = computeDispatchKey(device_);
        DispatchKeySet ks;
        ks.add(backend);
        ks.add(toAutogradKey(backend));
        return ks;
    }
    
    size_t itemsize() const { return elementSize(dtype_); }
    bool is_contiguous() const { return is_contiguous_; }

    bool is_sparse() const { return sparse_state_ != nullptr; }
    bool is_coalesced() const { return sparse_state_ && sparse_state_->coalesced; }
    std::shared_ptr<TensorImpl> sparse_indices_impl() const {
        return sparse_state_ ? sparse_state_->indices : nullptr;
    }
    std::shared_ptr<TensorImpl> sparse_values_impl() const {
        return sparse_state_ ? sparse_state_->values : nullptr;
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
        sparse_state_->indices = std::move(indices);
        sparse_state_->values = std::move(values);
        sparse_state_->sparse_sizes = std::move(sparse_sizes);
        sparse_state_->coalesced = coalesced;
        is_contiguous_ = false;
        storage_offset_ = 0;
        clear_storage();
    }

    // Version counter access (PyTorch-style mutation tracking). Views share
    // the counter with their base via share_version_counter().
    uint32_t version() const { return version_counter_.current_version(); }
    void bump_version() { version_counter_.bump(); }
    // Zero the counter (used when a fresh tensor is materialized by an
    // internal copy such as clone(), so the result starts unmutated).
    void reset_version() { version_counter_.reset(); }
    const VariableVersion& version_counter() const { return version_counter_; }
    // Make this tensor's version counter alias `other`'s (view semantics).
    void share_version_counter(const TensorImpl& other) { version_counter_ = other.version_counter_; }
    
    // Copy metadata (storage, sizes, strides, dtype, device) from another TensorImpl
    // preserving autograd_meta
    void copy_metadata_from(const TensorImpl& other);
    
    // Access to raw data pointer (typed)
    template<typename T>
    T* data() const {
        if (!has_storage()) return nullptr;
        void* base_ptr = shared_state_->storage.data();
#ifdef USE_CUDA
        if (device_.is_cuda()) cuda::recordStream(base_ptr, device_);
#endif
        return static_cast<T*>(base_ptr) + storage_offset_;
    }
    
    void* data() const {
        if (!has_storage()) return nullptr;
        void* base_ptr = shared_state_->storage.data();
#ifdef USE_CUDA
        if (device_.is_cuda()) cuda::recordStream(base_ptr, device_);
#endif
        size_t elem_size = elementSize(dtype_);
        return static_cast<char*>(base_ptr) + storage_offset_ * elem_size;
    }

    void set_onednn_md(std::shared_ptr<void> md) { shared_state_->onednn_md = md; }
    std::shared_ptr<void> get_onednn_md() const { return shared_state_->onednn_md; }
    bool has_onednn_md() const { return shared_state_->onednn_md != nullptr; }

    void set_onednn_memory_cache(std::shared_ptr<void> mem) { shared_state_->onednn_memory_cache = mem; }
    std::shared_ptr<void> get_onednn_memory_cache() const { return shared_state_->onednn_memory_cache; }
    bool has_onednn_memory_cache() const { return shared_state_->onednn_memory_cache != nullptr; }

    void set_autograd_meta(std::shared_ptr<AutogradMetaBase> meta) { autograd_meta_ = std::move(meta); }
    AutogradMetaBase* autograd_meta() const { return autograd_meta_.get(); }
    bool has_autograd_meta() const { return autograd_meta_ != nullptr; }
    std::shared_ptr<AutogradMetaBase> autograd_meta_shared() const { return autograd_meta_; }
    
    // Share storage state from another TensorImpl
    void share_storage_from(const TensorImpl& other) {
        shared_state_ = other.shared_state_;
    }
};

} // namespace tensorplay
