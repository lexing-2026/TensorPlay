#ifdef USE_VULKAN

#include "Tensor.h"
#include "Utils.h"

#include <algorithm>
#include <numeric>

namespace tensorplay {
namespace vulkan {

namespace {

/*
 * Calculates the strides of a contiguous tensor.
 */
std::vector<int64_t> calc_contiguous_strides(
    const std::vector<int64_t>& sizes) {
  int64_t ndim = static_cast<int64_t>(sizes.size());
  std::vector<int64_t> strides(ndim);

  int64_t running_product = 1;
  if (ndim >= 1) {
    strides.at(ndim - 1) = running_product;
    for (int i = static_cast<int>(sizes.size()) - 2; i >= 0; --i) {
      running_product *= sizes.at(i + 1);
      strides.at(i) = running_product;
    }
  }

  return strides;
}

std::vector<int64_t> calc_channels_last_strides(
    const std::vector<int64_t>& sizes) {
  std::vector<int64_t> strides(sizes.size());

  switch (sizes.size()) {
    case 4:
      strides.at(1) = 1;
      strides.at(3) = sizes.at(1);
      strides.at(2) = strides.at(3) * sizes.at(3);
      strides.at(0) = strides.at(2) * sizes.at(2);
      return strides;
    case 3:
      strides.at(0) = 1;
      strides.at(2) = sizes.at(0);
      strides.at(1) = strides.at(2) * sizes.at(2);
      return strides;
    default:
      VK_THROW("ChannelsLast format only available for 3 <= ndim <= 4!");
  }

  return strides;
}

/*
 * Calculates the strides of a tensor based on the sizes and memory format.
 * Note that strides are only valid for vTensors that are backed by buffer
 * storage; if texture storage is used then the strides are invalid and set
 * to zeros.
 */
std::vector<int64_t> calc_strides(
    const std::vector<int64_t>& sizes,
    const api::GPUMemoryLayout memory_layout,
    const api::StorageType storage_type) {
  switch (storage_type) {
    case api::StorageType::BUFFER:
      switch (memory_layout) {
        case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
          return calc_contiguous_strides(sizes);
        case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
          return calc_channels_last_strides(sizes);
        default:
          VK_THROW("Invalid memory format used to create vTensor!");
      }
      break;
    case api::StorageType::TEXTURE_3D:
      return std::vector<int64_t>(sizes.size());
    default:
      VK_THROW("Invalid storage type used to create vTensor!");
  }
}

/*
 * When stored on the GPU, one dimension will be aligned to the next multiple
 * of 4 in order to take advantage of vec4 data types. The dimension that is
 * packed is denoted by the GPUMemoryLayout. This function adjusts one of
 * the dimensions based on the desired memory format and storage type and
 * returns a sizes array describing the dimensions of the memory used to
 * store the tensor data on the GPU.
 */
std::vector<int64_t> calc_gpu_sizes(
    const std::vector<int64_t>& sizes,
    const api::GPUMemoryLayout memory_layout,
    const api::StorageType storage_type) {
  VK_CHECK_COND(storage_type != api::StorageType::UNKNOWN);

  std::vector<int64_t> gpu_sizes;
  if (storage_type == api::StorageType::BUFFER) {
    gpu_sizes.resize(sizes.size());
    for (size_t i = 0; i < sizes.size(); i++) {
      gpu_sizes.at(i) = sizes.at(i);
    }
  }
  // For texture storage, tensors are typically stored using 3D image
  // textures. Batches are stacked along the depth dimension. To represent
  // the physical 3 dimensionality of the image texture (with concatenated
  // batches) GPU sizes will be fixed to 4 dimensions when using texture
  // storage.
  else {
    VK_CHECK_COND(
        sizes.size() >= 0 && sizes.size() <= 4,
        "Texture storage only valid for 0 <= ndim <= 4, received: ",
        sizes.size());

    gpu_sizes.resize(4);
    gpu_sizes.at(0) = api::utils::val_at(-4, sizes);
    gpu_sizes.at(1) = api::utils::val_at(-3, sizes);
    gpu_sizes.at(2) = api::utils::val_at(-2, sizes);
    gpu_sizes.at(3) = api::utils::val_at(-1, sizes);
  }

  size_t ndim = gpu_sizes.size();
  switch (memory_layout) {
    case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
      if (ndim >= 1) {
        gpu_sizes.at(ndim - 1) =
            api::utils::align_up(
                static_cast<uint32_t>(api::utils::val_at(-1, sizes)), 4u);
      }
      break;

    case api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED:
      if (ndim >= 2) {
        gpu_sizes.at(ndim - 2) =
            api::utils::align_up(
                static_cast<uint32_t>(api::utils::val_at(-2, sizes)), 4u);
      }
      break;

    case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
      if (ndim >= 3) {
        gpu_sizes.at(ndim - 3) =
            api::utils::align_up(
                static_cast<uint32_t>(api::utils::val_at(-3, sizes)), 4u);
      }
      break;
  }

  return gpu_sizes;
}

/*
 * Creates a uvec3 denoting the extents of the image texture that will be
 * created to store a tensor of a given size.
 */
api::utils::uvec3 create_image_extents(
    const std::vector<int64_t>& gpu_sizes,
    const api::StorageType storage_type,
    const api::GPUMemoryLayout memory_layout) {
  size_t ndim = gpu_sizes.size();

  if (storage_type == api::StorageType::BUFFER) {
    // image extents do not apply to buffer storage
    return {0u, 0u, 0u};
  } else {
    VK_CHECK_COND(
        ndim >= 1 && ndim <= 4,
        "Texture storage only valid for 1 <= ndim <= 4!");

    using namespace api::utils;
    uint32_t width = safe_downcast_to_u32(val_at(-1, gpu_sizes));
    uint32_t height = safe_downcast_to_u32(val_at(-2, gpu_sizes));
    uint32_t channels = safe_downcast_to_u32(val_at(-3, gpu_sizes));
    uint32_t batch = safe_downcast_to_u32(val_at(-4, gpu_sizes));

    switch (memory_layout) {
      case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
        VK_CHECK_COND(width % 4 == 0, "Channels must be divisible by 4!");
        width /= 4;
        break;
      case api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED:
        VK_CHECK_COND(height % 4 == 0, "Channels must be divisible by 4!");
        height /= 4;
        break;
      case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
        VK_CHECK_COND(channels % 4 == 0, "Channels must be divisible by 4!");
        channels /= 4;
        break;
      default:
        VK_THROW("Invalid memory format used!");
    }

    return {width, height, batch * channels};
  }
}

api::UniformParamsBuffer make_metadata_uniform(
    api::Context* const context,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    const api::StorageType storage_type) {
  if (storage_type != api::StorageType::BUFFER) {
    return api::UniformParamsBuffer();
  }

  api::vTensor::BufferMetadata metadata{
      api::utils::make_whcn_uvec4(sizes),
      api::utils::make_whcn_uvec4(strides),
      api::utils::safe_downcast_to_u32(sizes.size()),
      api::utils::safe_downcast_to_u32(
          static_cast<int64_t>(api::utils::multiply_integers(sizes))),
  };

  return api::UniformParamsBuffer(context, metadata);
}

} // namespace

namespace api {

//
// vTensorStorage
//

static VulkanImage allocate_image(
    api::Context* const context_ptr,
    api::utils::uvec3& extents,
    const api::StorageType storage_type,
    const VkFormat image_format,
    const bool allocate_memory) {
  api::Adapter* adapter_ptr = context_ptr->adapter_ptr();

  VK_CHECK_COND(
      (extents[0u] > 0u) && (extents[1u] > 0u) && (extents[2u] > 0u),
      "Vulkan image texture extents must be greater than 0!");

  const bool allow_transfer = true;

  // Texture reads go through combined image samplers, so every image carries
  // a cached sampler matching its sampling properties.
  const api::ImageSampler::Properties sampler_props{
      VK_FILTER_NEAREST, // filter
      VK_SAMPLER_MIPMAP_MODE_NEAREST, // mipmap_mode
      VK_SAMPLER_ADDRESS_MODE_REPEAT, // address_mode:
      VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, // border_color
  };
  VkSampler sampler =
      adapter_ptr->sampler_cache().retrieve(sampler_props);

  VulkanImage image = adapter_ptr->vma().create_image(
      {
          extents[0u], // width
          extents[1u], // height
          extents[2u], // depth
      },
      image_format,
      VK_IMAGE_TYPE_3D,
      VK_IMAGE_VIEW_TYPE_3D,
      sampler_props,
      sampler,
      allow_transfer,
      allocate_memory);

  return image;
}

vTensorStorage::vTensorStorage(
    api::Context* context,
    const api::StorageType storage_type,
    const api::GPUMemoryLayout gpu_memory_layout,
    const std::vector<int64_t>& sizes,
    const DType dtype,
    const bool allocate_memory)
    : context_(context),
      storage_type_(storage_type),
      gpu_memory_layout_(gpu_memory_layout),
      buffer_length_{},
      image_{},
      buffer_{},
      last_access_{} {
  verify();

  if (api::StorageType::BUFFER == storage_type) {
    buffer_length_ = static_cast<int64_t>(utils::multiply_integers(sizes));

    buffer_ = context_->adapter_ptr()->vma().create_storage_buffer(
        buffer_length_ * tensorplay::elementSize(dtype),
        !context_->adapter_ptr()->has_unified_memory(),
        allocate_memory);
  } else if (api::StorageType::TEXTURE_3D == storage_type) {
    extents_ = create_image_extents(sizes, storage_type, gpu_memory_layout);

    image_ = allocate_image(
        context_, extents_, storage_type, to_vkformat(dtype), allocate_memory);
  } else {
    VK_THROW("Not implemented!");
  }
}

vTensorStorage::~vTensorStorage() {
  try {
    flush();
  } catch (...) {
  }
}

void vTensorStorage::flush() {
  // Non-empty resources must be released via the context clearlist so that
  // they stay alive until all in-flight GPU work referencing them has
  // completed.  Destruction happens on the next flush.
  if (context_) {
    api::VulkanImage image = std::move(image_);
    context_->register_image_cleanup(image);

    api::VulkanBuffer buffer = std::move(buffer_);
    context_->register_buffer_cleanup(buffer);
  }
}

void vTensorStorage::transition(
    api::PipelineBarrier& pipeline_barrier,
    const api::PipelineStageFlags stage,
    const api::MemoryAccessFlags access) {
  verify();

  const bool read = access & api::MemoryAccessType::READ;
  const bool write = access & api::MemoryAccessType::WRITE;

  if (api::StorageType::TEXTURE_3D == storage_type_) {
    // Images track their layout through the barrier system: every access
    // transitions to (and re-asserts) VK_IMAGE_LAYOUT_GENERAL, matching the
    // layout the descriptor bindings advertise.  Barriers between dependent
    // dispatches are recorded the same way as buffers.
    const bool is_write_only = write && !read;

    if (write) {
      if (last_access_.stage != api::PipelineStage::NO_STAGE) {
        pipeline_barrier.stage.src |= api::vk_stage(last_access_.stage);
        pipeline_barrier.stage.dst |= api::vk_stage(stage);

        pipeline_barrier.images.emplace_back(
            api::vk_access(last_access_.stage, last_access_.access),
            api::vk_access(stage, access),
            VK_IMAGE_LAYOUT_GENERAL,
            is_write_only ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_GENERAL,
            image_);
      }

      image_.set_layout(VK_IMAGE_LAYOUT_GENERAL);
      last_access_ = LastAccess{stage, access};
    } else if (read) {
      if (last_access_.access & api::MemoryAccessType::WRITE) {
        // RAW dependency: a read must not overtake the previous write.
        pipeline_barrier.stage.src |= api::vk_stage(last_access_.stage);
        pipeline_barrier.stage.dst |= api::vk_stage(stage);

        pipeline_barrier.images.emplace_back(
            api::vk_access(last_access_.stage, last_access_.access),
            api::vk_access(stage, access),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_GENERAL,
            image_);
      }

      image_.set_layout(VK_IMAGE_LAYOUT_GENERAL);

      if (last_access_.stage == api::PipelineStage::NO_STAGE) {
        last_access_ = LastAccess{stage, access};
      } else {
        // Preserve the pending-write hazard for future writes while
        // allowing this read to proceed.
        LastAccess merged = last_access_;
        merged.stage = stage;
        merged.access =
            (last_access_.access & api::MemoryAccessType::WRITE) |
            api::MemoryAccessType::READ;
        last_access_ = merged;
      }
    }

    return;
  }

  // Buffer storage barrier insertion.
  if (write) {
    if (last_access_.stage != api::PipelineStage::NO_STAGE) {
      pipeline_barrier.stage.src |= api::vk_stage(last_access_.stage);
      pipeline_barrier.stage.dst |= api::vk_stage(stage);

      pipeline_barrier.buffers.emplace_back(
          api::vk_access(last_access_.stage, last_access_.access),
          api::vk_access(stage, access),
          buffer_);
    }

    last_access_ = LastAccess{stage, access};
  } else if (read) {
    if (last_access_.access & api::MemoryAccessType::WRITE) {
      pipeline_barrier.stage.src |= api::vk_stage(last_access_.stage);
      pipeline_barrier.stage.dst |= api::vk_stage(stage);

      pipeline_barrier.buffers.emplace_back(
          api::vk_access(last_access_.stage, last_access_.access),
          api::vk_access(stage, access),
          buffer_);
    }

    if (last_access_.stage == api::PipelineStage::NO_STAGE) {
      last_access_ = LastAccess{stage, access};
    } else {
      LastAccess merged = last_access_;
      merged.stage = stage;
      merged.access =
          (last_access_.access & api::MemoryAccessType::WRITE) |
          api::MemoryAccessType::READ;
      last_access_ = merged;
    }
  }
}

void vTensorStorage::verify() const {
  if (!context_) {
    VK_THROW("Vulkan vTensorStorage: context is not set!");
  }
}

void vTensorStorage::discard_and_reallocate(
    const std::vector<int64_t>& gpu_sizes,
    const api::GPUMemoryLayout gpu_memory_layout,
    const DType dtype) {
  verify();

  flush();

  if (api::StorageType::BUFFER == storage_type_) {
    buffer_length_ = static_cast<int64_t>(utils::multiply_integers(gpu_sizes));

    buffer_ = context_->adapter_ptr()->vma().create_storage_buffer(
        buffer_length_ * tensorplay::elementSize(dtype),
        !context_->adapter_ptr()->has_unified_memory());
  } else if (api::StorageType::TEXTURE_3D == storage_type_) {
    extents_ = create_image_extents(gpu_sizes, storage_type_, gpu_memory_layout);

    image_ = allocate_image(
        context_, extents_, storage_type_, to_vkformat(dtype), true);
  } else {
    VK_THROW("Not implemented!");
  }

  last_access_ = LastAccess{};
}

//
// vTensor
//

vTensor::vTensor(
    api::Context* context,
    const std::vector<int64_t>& sizes,
    const DType dtype,
    const api::StorageType storage_type,
    const api::GPUMemoryLayout memory_layout,
    const bool allocate_memory)
    : dtype_(dtype),
      memory_layout_(memory_layout),
      // Calculate sizes and strides
      sizes_(sizes.begin(), sizes.end()),
      strides_{calc_strides(sizes, memory_layout_, storage_type)},
      gpu_sizes_{calc_gpu_sizes(sizes, memory_layout_, storage_type)},
      gpu_strides_{calc_strides(gpu_sizes_, memory_layout_, storage_type)},
      virtual_extents_(
          create_image_extents(gpu_sizes_, storage_type, memory_layout)),
      // Utility Uniform Buffers that can be passed to shaders as arguments
      metadata_uniform_(),
      cpu_sizes_uniform_(nullptr),
      gpu_sizes_uniform_(nullptr),
      extents_uniform_(nullptr),
      // Construct Tensor storage
      view_(std::make_shared<vTensorStorage>(
          context,
          storage_type,
          memory_layout_,
          gpu_sizes_,
          dtype_,
          allocate_memory)) {}

vTensor::vTensor(
    api::Context* context,
    const std::vector<int64_t>& sizes,
    const DType dtype,
    vTensorStorage& existing_storage,
    const api::GPUMemoryLayout memory_layout)
    : dtype_(dtype),
      memory_layout_(memory_layout),
      // Calculate sizes and strides
      sizes_(sizes.begin(), sizes.end()),
      strides_{
          calc_strides(sizes, memory_layout_, existing_storage.storage_type_)},
      gpu_sizes_{
          calc_gpu_sizes(sizes, memory_layout_, existing_storage.storage_type_)},
      gpu_strides_{calc_strides(
          gpu_sizes_, memory_layout_, existing_storage.storage_type_)},
      virtual_extents_(create_image_extents(
          gpu_sizes_, existing_storage.storage_type_, memory_layout)),
      // Utility Uniform Buffers that can be passed to shaders as arguments
      metadata_uniform_(),
      cpu_sizes_uniform_(nullptr),
      gpu_sizes_uniform_(nullptr),
      extents_uniform_(nullptr),
      // Reference the caller-owned storage without owning it
      view_(&existing_storage, [](vTensorStorage*) {}) {}

api::VulkanImage& vTensor::image(
    api::PipelineBarrier& pipeline_barrier,
    const api::PipelineStageFlags stage) const& {
  view_->transition(pipeline_barrier, stage, api::MemoryAccessType::READ);
  return view_->image_;
}

api::VulkanImage& vTensor::image(
    api::PipelineBarrier& pipeline_barrier,
    const api::PipelineStageFlags stage,
    const api::MemoryAccessFlags access) & {
  view_->transition(pipeline_barrier, stage, access);
  return view_->image_;
}

api::VulkanBuffer& vTensor::buffer(
    api::PipelineBarrier& pipeline_barrier,
    const api::PipelineStageFlags stage) const& {
  view_->transition(pipeline_barrier, stage, api::MemoryAccessType::READ);
  return view_->buffer_;
}

api::VulkanBuffer& vTensor::buffer(
    api::PipelineBarrier& pipeline_barrier,
    const api::PipelineStageFlags stage,
    const api::MemoryAccessFlags access) & {
  view_->transition(pipeline_barrier, stage, access);
  return view_->buffer_;
}

api::VulkanBuffer& vTensor::buffer_metadata() {
  if (!metadata_uniform_.buffer()) {
    metadata_uniform_ = make_metadata_uniform(
        view_->context_, gpu_sizes_, gpu_strides_, storage_type());
  }
  return metadata_uniform_.buffer();
}

std::shared_ptr<api::UniformParamsBuffer> vTensor::cpu_sizes_ubo() {
  if (!cpu_sizes_uniform_) {
    cpu_sizes_uniform_.reset(new api::UniformParamsBuffer(
        view_->context_, api::utils::make_whcn_ivec4(sizes_)));
  }
  return cpu_sizes_uniform_;
}

std::shared_ptr<api::UniformParamsBuffer> vTensor::gpu_sizes_ubo() {
  if (!gpu_sizes_uniform_) {
    gpu_sizes_uniform_.reset(new api::UniformParamsBuffer(
        view_->context_, api::utils::make_whcn_ivec4(gpu_sizes_)));
  }
  return gpu_sizes_uniform_;
}

std::shared_ptr<api::UniformParamsBuffer> vTensor::extents_ubo() {
  if (!extents_uniform_) {
    extents_uniform_.reset(new api::UniformParamsBuffer(
        view_->context_,
        api::utils::uvec4{
            view_->extents_[0u],
            view_->extents_[1u],
            view_->extents_[2u],
            1u}));
  }
  return extents_uniform_;
}

vTensor::BufferMetadata vTensor::get_cpu_buffer_metadata() const {
  return {
      api::utils::make_whcn_uvec4(sizes_),
      api::utils::make_whcn_uvec4(strides_),
      api::utils::safe_downcast_to_u32(sizes_.size()),
      api::utils::safe_downcast_to_u32(
          static_cast<int64_t>(api::utils::multiply_integers(sizes_))),
  };
}

VmaAllocationCreateInfo vTensor::get_allocation_create_info() const {
  switch (storage_type()) {
    case api::StorageType::BUFFER:
      return view_->buffer_.allocation_create_info();
    case api::StorageType::TEXTURE_3D:
      return view_->image_.allocation_create_info();
    case api::StorageType::UNKNOWN:
      break;
  }
  return {};
}

VkMemoryRequirements vTensor::get_memory_requirements() const {
  switch (storage_type()) {
    case api::StorageType::BUFFER:
      return view_->buffer_.get_memory_requirements();
    case api::StorageType::TEXTURE_3D:
      return view_->image_.get_memory_requirements();
    case api::StorageType::UNKNOWN:
      break;
  }
  return {};
}

void vTensor::bind_allocation(const api::MemoryAllocation& allocation) {
  switch (storage_type()) {
    case api::StorageType::BUFFER:
      view_->buffer_.bind_allocation(allocation);
      break;
    case api::StorageType::TEXTURE_3D:
      view_->image_.bind_allocation(allocation);
      break;
    case api::StorageType::UNKNOWN:
      break;
  }
}

void vTensor::update_size_metadata(const std::vector<int64_t>& new_sizes) {
  sizes_ = new_sizes;
  gpu_sizes_ = calc_gpu_sizes(sizes_, memory_layout_, storage_type());
  virtual_extents_ =
      create_image_extents(gpu_sizes_, storage_type(), memory_layout_);

  strides_ = calc_strides(sizes_, memory_layout_, storage_type());
  gpu_strides_ = calc_strides(gpu_sizes_, memory_layout_, storage_type());

  if (cpu_sizes_uniform_) {
    cpu_sizes_uniform_->update(api::utils::make_whcn_ivec4(sizes_));
  }

  if (gpu_sizes_uniform_) {
    gpu_sizes_uniform_->update(api::utils::make_whcn_ivec4(gpu_sizes_));
  }

  if (extents_uniform_) {
    extents_uniform_->update(api::utils::uvec4{
        virtual_extents_[0u],
        virtual_extents_[1u],
        virtual_extents_[2u],
        1u});
  }
}

void vTensor::reallocate(const std::vector<int64_t>& new_sizes) {
  update_size_metadata(new_sizes);
  view_->discard_and_reallocate(
      calc_gpu_sizes(new_sizes, memory_layout_, storage_type()),
      memory_layout_,
      dtype_);
}

void vTensor::virtual_resize(const std::vector<int64_t>& new_sizes) {
  update_size_metadata(new_sizes);
  if (storage_type() == api::StorageType::BUFFER) {
    if (gpu_nbytes() > view_->buffer_.mem_size()) {
      VK_THROW(
          "Cannot virtual_resize a vTensor with sizes that require a larger "
          "buffer! reallocate() should be used instead.");
    }
  } else {
    bool valid_resize = true;
    if (virtual_extents_[0u] > view_->extents_[0u]) {
      valid_resize = false;
    }
    if (virtual_extents_[1u] > view_->extents_[1u]) {
      valid_resize = false;
    }
    if (virtual_extents_[2u] > view_->extents_[2u]) {
      valid_resize = false;
    }

    if (!valid_resize) {
      VK_THROW(
          "Cannot virtual_resize a vTensor with sizes that require a larger "
          "image texture! reallocate() should be used instead.");
    }
  }
}

} // namespace api
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
