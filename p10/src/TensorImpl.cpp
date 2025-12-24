#include "TensorImpl.h"
#include "Allocator.h"
#include "Storage.h"
// #include "AutogradMetaInterface.h"
#include "Tensor.h"
#include <iostream>

namespace tensorplay {

TensorImpl::TensorImpl() 
    : storage_offset_(0), dtype_(DType::Float32), device_(DeviceType::CPU),
      is_contiguous_(true), is_channels_last_(false) {
    shared_state_ = std::make_shared<SharedState>();
}

TensorImpl::TensorImpl(const std::vector<int64_t>& sizes, DType dtype, const Device& device)
    : storage_offset_(0), sizes_and_strides_(sizes), dtype_(dtype), device_(device),
      is_contiguous_(true), is_channels_last_(false) {
    
    shared_state_ = std::make_shared<SharedState>();
    int64_t num_elements = sizes_and_strides_.numel();
    if (num_elements > 0) {
        size_t total_bytes = static_cast<size_t>(num_elements) * elementSize(dtype);
        Allocator* allocator = getAllocator(device.type());
        shared_state_->storage = Storage(total_bytes, allocator);
    }
}

TensorImpl::TensorImpl(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, DType dtype, const Device& device)
    : storage_offset_(0), sizes_and_strides_(sizes, strides), dtype_(dtype), device_(device),
      is_contiguous_(false), is_channels_last_(false) {
    
    shared_state_ = std::make_shared<SharedState>();
    is_contiguous_ = sizes_and_strides_.is_contiguous();
    
    int64_t num_elements = sizes_and_strides_.numel();
    if (num_elements > 0) {
        size_t total_bytes = static_cast<size_t>(num_elements) * elementSize(dtype);
        Allocator* allocator = getAllocator(device.type());
        shared_state_->storage = Storage(total_bytes, allocator);
    }
}

TensorImpl::TensorImpl(Storage storage, const std::vector<int64_t>& sizes, DType dtype, size_t storage_offset)
    : storage_offset_(storage_offset), sizes_and_strides_(sizes), dtype_(dtype),
      device_(storage.device()), is_contiguous_(true), is_channels_last_(false) {
    
    shared_state_ = std::make_shared<SharedState>();
    shared_state_->storage = storage;
    is_contiguous_ = sizes_and_strides_.is_contiguous();
}

TensorImpl::TensorImpl(Storage storage, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, DType dtype, size_t storage_offset)
    : storage_offset_(storage_offset), sizes_and_strides_(sizes, strides), dtype_(dtype),
      device_(storage.device()), is_contiguous_(false), is_channels_last_(false) {
      
    shared_state_ = std::make_shared<SharedState>();
    shared_state_->storage = storage;
    is_contiguous_ = sizes_and_strides_.is_contiguous();
}

TensorImpl::TensorImpl(const TensorImpl& other) = default;
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
    is_channels_last_ = other.is_channels_last_;
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
