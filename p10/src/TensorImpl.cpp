#include "TensorImpl.h"
#include "Allocator.h"
#include "Storage.h"
#include "Tensor.h"

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
    if (allocate_storage) {
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
    size_t total_bytes = static_cast<size_t>(num_elements) * elementSize(dtype);
    Allocator* allocator = getAllocator(device.type());
    shared_state_->storage = Storage(total_bytes, allocator, device);
    device_ = shared_state_->storage.device();
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

TensorImpl::TensorImpl(std::shared_ptr<TensorImpl> transform_value,
                       const std::vector<int64_t>& sizes,
                       const std::vector<int64_t>& strides)
    : storage_offset_(0), sizes_and_strides_(sizes, strides),
      dtype_(transform_value ? transform_value->dtype() : DType::Undefined),
      device_(transform_value ? transform_value->device() : Device(DeviceType::CPU)),
      is_contiguous_(sizes_and_strides_.is_contiguous()),
      transform_value_(std::move(transform_value)) {
    shared_state_ = std::make_shared<SharedState>();
}

// Copy does not carry autograd metadata: copies start fresh, matching
TensorImpl::TensorImpl(const TensorImpl& other)
    : storage_offset_(other.storage_offset_),
      sizes_and_strides_(other.sizes_and_strides_),
      dtype_(other.dtype_),
      device_(other.device_),
      version_counter_(other.version_counter_),
      inference_tensor_(other.inference_tensor_),
      is_contiguous_(other.is_contiguous_),
      memory_format_(other.memory_format_),
      shared_state_(other.shared_state_),
      sparse_state_(other.sparse_state_),
      transform_value_(other.transform_value_),
      transform_batch_dim_(other.transform_batch_dim_),
      transform_level_(other.transform_level_) {}
TensorImpl::TensorImpl(TensorImpl&& other) noexcept = default;
TensorImpl& TensorImpl::operator=(const TensorImpl& other) = default;
TensorImpl& TensorImpl::operator=(TensorImpl&& other) noexcept = default;

void TensorImpl::set_requires_grad(bool requires_grad) {
    if (requires_grad && inference_tensor_ && !InferenceMode::is_enabled()) {
        throw std::runtime_error(
            "Setting requires_grad=True on an inference tensor outside inference mode is not allowed.");
    }
    if (autograd_meta_) {
        autograd_meta_->set_requires_grad(requires_grad);
    }
}

void TensorImpl::copy_metadata_from(const TensorImpl& other) {
    shared_state_ = other.shared_state_;
    storage_offset_ = other.storage_offset_;
    sizes_and_strides_ = other.sizes_and_strides_;
    dtype_ = other.dtype_;
    device_ = other.device_;
    is_contiguous_ = other.is_contiguous_;
    memory_format_ = other.memory_format_;
    sparse_state_ = other.sparse_state_;
    transform_value_ = other.transform_value_;
    transform_batch_dim_ = other.transform_batch_dim_;
    transform_level_ = other.transform_level_;
}

} // namespace tensorplay
