#include "TensorImpl.h"
#include "Allocator.h"
#include "Storage.h"
// #include "AutogradMetaInterface.h"
#include "Tensor.h"
#include <iostream>

namespace tensorplay {

const char* toString(MemoryFormat format) {
    switch (format) {
        case MemoryFormat::Contiguous: return "Contiguous";
        case MemoryFormat::Preserve: return "Preserve";
        case MemoryFormat::ChannelsLast: return "ChannelsLast";
        case MemoryFormat::ChannelsLast3d: return "ChannelsLast3d";
    }
    return "Unknown";
}

TensorImpl::TensorImpl()
    : storage_offset_(0), dtype_(DType::Float32), device_(DeviceType::CPU),
      is_contiguous_(true) {
    shared_state_ = std::make_shared<SharedState>();
}

TensorImpl::TensorImpl(const std::vector<int64_t>& sizes, DType dtype, const Device& device)
    : TensorImpl(sizes, dtype, device, true) {}

TensorImpl::TensorImpl(const std::vector<int64_t>& sizes, DType dtype,
                       const Device& device, bool allocate_storage)
    : storage_offset_(0), sizes_and_strides_(sizes), dtype_(dtype), device_(device),
      is_contiguous_(true) {
    
    shared_state_ = std::make_shared<SharedState>();
    int64_t num_elements = sizes_and_strides_.numel();
    if (allocate_storage && num_elements > 0) {
        size_t total_bytes = static_cast<size_t>(num_elements) * elementSize(dtype);
        Allocator* allocator = getAllocator(device.type());
        shared_state_->storage = Storage(total_bytes, allocator, device);
        device_ = shared_state_->storage.device();
    }
}

TensorImpl::TensorImpl(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, DType dtype, const Device& device)
    : storage_offset_(0), sizes_and_strides_(sizes, strides), dtype_(dtype), device_(device),
      is_contiguous_(false) {
    
    shared_state_ = std::make_shared<SharedState>();
    is_contiguous_ = sizes_and_strides_.is_contiguous();
    
    int64_t num_elements = sizes_and_strides_.numel();
    if (num_elements > 0) {
        size_t total_bytes = static_cast<size_t>(num_elements) * elementSize(dtype);
        Allocator* allocator = getAllocator(device.type());
        shared_state_->storage = Storage(total_bytes, allocator, device);
        device_ = shared_state_->storage.device();
    }
}

TensorImpl::TensorImpl(Storage storage, const std::vector<int64_t>& sizes, DType dtype, size_t storage_offset)
    : storage_offset_(storage_offset), sizes_and_strides_(sizes), dtype_(dtype),
      device_(storage.device()), is_contiguous_(true) {
    
    shared_state_ = std::make_shared<SharedState>();
    shared_state_->storage = storage;
    is_contiguous_ = sizes_and_strides_.is_contiguous();
}

TensorImpl::TensorImpl(Storage storage, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, DType dtype, size_t storage_offset)
    : storage_offset_(storage_offset), sizes_and_strides_(sizes, strides), dtype_(dtype),
      device_(storage.device()), is_contiguous_(false) {
      
    shared_state_ = std::make_shared<SharedState>();
    shared_state_->storage = storage;
    is_contiguous_ = sizes_and_strides_.is_contiguous();
}

// Copy does not carry autograd metadata: copies start fresh, matching
// PyTorch (autograd metadata is attached by the autograd layer, never copied).
TensorImpl::TensorImpl(const TensorImpl& other)
    : storage_offset_(other.storage_offset_),
      sizes_and_strides_(other.sizes_and_strides_),
      dtype_(other.dtype_),
      device_(other.device_),
      version_counter_(other.version_counter_),
      is_contiguous_(other.is_contiguous_),
      memory_format_(other.memory_format_),
      shared_state_(other.shared_state_),
      sparse_state_(other.sparse_state_) {}
TensorImpl::TensorImpl(TensorImpl&& other) noexcept = default;
TensorImpl& TensorImpl::operator=(const TensorImpl& other) = default;
TensorImpl& TensorImpl::operator=(TensorImpl&& other) noexcept = default;

/*
void TensorImpl::set_requires_grad(bool requires_grad) {
    if (requires_grad) {
        if (!autograd_meta_) {
            AutogradMetaFactory* factory = GetAutogradMetaFactory();
            if (factory) {
                autograd_meta_ = factory->make();
                autograd_meta_->set_requires_grad(true);
            } else {
                // If no factory, we can't enable autograd. 
                // In PyTorch this might throw or use default. 
                // For now, let's warn or throw.
                std::cerr << "Warning: AutogradMetaFactory not registered. cannot set requires_grad=true" << std::endl;
            }
        } else {
            autograd_meta_->set_requires_grad(true);
        }
    } else {
        if (autograd_meta_) {
            autograd_meta_->set_requires_grad(false);
        }
    }
}
*/

/*
void TensorImpl::retain_grad() {
    if (!autograd_meta_) {
        AutogradMetaFactory* factory = GetAutogradMetaFactory();
        if (factory) {
            autograd_meta_ = factory->make();
        } else {
             // Handle error
             return;
        }
    }
    autograd_meta_->set_retain_grad(true);
}

void TensorImpl::set_grad_fn(std::shared_ptr<Node> grad_fn) {
    if (!autograd_meta_) {
        AutogradMetaFactory* factory = GetAutogradMetaFactory();
        if (factory) {
            autograd_meta_ = factory->make();
        } else {
             return;
        }
    }
    autograd_meta_->set_grad_fn(std::move(grad_fn));
    if (autograd_meta_->grad_fn()) {
        autograd_meta_->set_requires_grad(true);
    }
}

std::shared_ptr<Node> TensorImpl::grad_fn() const {
    if (autograd_meta_) {
        return autograd_meta_->grad_fn();
    }
    return nullptr;
}

void TensorImpl::set_autograd_meta(std::shared_ptr<AutogradMetaInterface> autograd_meta) {
    autograd_meta_ = std::move(autograd_meta);
}
*/

void TensorImpl::copy_metadata_from(const TensorImpl& other) {
    // storage_ = other.storage_; // Replaced by shared_state_
    shared_state_ = other.shared_state_;
    storage_offset_ = other.storage_offset_;
    sizes_and_strides_ = other.sizes_and_strides_;
    dtype_ = other.dtype_;
    device_ = other.device_;
    is_contiguous_ = other.is_contiguous_;
    memory_format_ = other.memory_format_;
    sparse_state_ = other.sparse_state_;
    // onednn_md_ = other.onednn_md_; // Replaced by shared_state_
}

/*
Tensor TensorImpl::grad() const {
    if (autograd_meta_) {
        return autograd_meta_->grad();
    }
    return Tensor();
}

void TensorImpl::set_grad(const Tensor& grad) {
    if (!autograd_meta_) {
         AutogradMetaFactory* factory = GetAutogradMetaFactory();
         if (factory) {
             autograd_meta_ = factory->make();
         } else {
             return;
         }
    }
    autograd_meta_->set_grad(grad);
}
*/

} // namespace tensorplay
