#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Factory.h"

#include <Utils.h>

#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

void validate_float_operand(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32,
      std::string("Vulkan ") + name + " supports Float32 tensors only");
  TP_CHECK(
      t.dim() <= 4,
      std::string("Vulkan ") + name + " supports up to 4d tensors");
}

struct CompareBlock final {
  ivec4 out_sizes;
  int c_depth;
  int fill;
};

// Elementwise comparison driver.  Both operands are already shaped (the
// scalar variants fold into full tensors in the public kernels).
Tensor compare_impl(
    const Tensor& self,
    const Tensor& other,
    const char* kernel_name,
    const char* name) {
  validate_float_operand(self, name);
  validate_float_operand(other, name);
  TP_CHECK(
      self.shape() == other.shape(),
      std::string("Vulkan ") + name +
          " requires equal-shaped operands (broadcast at the caller)");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);
  api::vTensor v_other = convert(other);
  api::vTensor v_output{context, v_input.sizes(), DType::Bool};

  const struct CompareBlock block{
      make_whcn_ivec4(v_output.sizes()),
      c_depth_of(v_output.sizes()),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL_FROM_STR(kernel_name), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor fold_scalar(const Tensor& self, Scalar other) {
  return full_kernel(
      static_cast<std::vector<int64_t>>(self.shape()),
      other,
      DType::Float32,
      self.device(),
      false);
}

//
// The comparison shaders read float planes only.  Integer and boolean
// payloads compare on the host over a staged copy: the values stream out,
// the fold runs per element, and the Bool mask uploads.  The result is
// device-agnostic in value order and answers the same Bool mask the
// shaders produce.
//
template <typename T, typename Pred>
Tensor compare_host(const Tensor& self, const Tensor& other, Pred pred) {
  const Tensor a = self.to(Device(DeviceType::CPU)).contiguous();
  const Tensor b = other.to(Device(DeviceType::CPU)).contiguous();
  const int64_t n = a.numel();
  Tensor out = Tensor::empty(
      static_cast<std::vector<int64_t>>(a.shape()), DType::Bool,
      Device(DeviceType::CPU));
  const T* lhs = static_cast<const T*>(a.impl()->storage().data());
  const T* rhs = static_cast<const T*>(b.impl()->storage().data());
  bool* mask = static_cast<bool*>(out.impl()->storage().data());
  for (int64_t i = 0; i < n; ++i) {
    mask[i] = pred(lhs[i], rhs[i]);
  }
  return out.to(self.device());
}

const char* validate_compare_pair(const Tensor& self, const Tensor& other) {
  if (self.dtype() != DType::Float32 || other.dtype() != DType::Float32) {
    TP_CHECK(
        self.dtype() == other.dtype(),
        "Vulkan comparison: mixed non-float operand dtypes require Float32");
    return self.dtype() == DType::Int32   ? "i"
           : self.dtype() == DType::Bool  ? "b"
           : self.dtype() == DType::Int8  ? "c"
           : self.dtype() == DType::UInt8 ? "C"
                                          : "";
  }
  return nullptr;
}

} // namespace

Tensor eq_tensor_kernel(const Tensor& self, const Tensor& other) {
  if (const char* tag = validate_compare_pair(self, other)) {
    if (tag[0] == 'i') {
      return compare_host<int32_t>(self, other,
                                   [](int32_t x, int32_t y) { return x == y; });
    }
    if (tag[0] == 'b') {
      return compare_host<bool>(self, other,
                                [](bool x, bool y) { return x == y; });
    }
    if (tag[0] == 'c') {
      return compare_host<int8_t>(self, other,
                                  [](int8_t x, int8_t y) { return x == y; });
    }
    if (tag[0] == 'C') {
      return compare_host<uint8_t>(self, other,
                                   [](uint8_t x, uint8_t y) { return x == y; });
    }
  }
  return compare_impl(self, other, "eq", "eq");
}
Tensor ne_tensor_kernel(const Tensor& self, const Tensor& other) {
  if (const char* tag = validate_compare_pair(self, other)) {
    if (tag[0] == 'i') {
      return compare_host<int32_t>(self, other,
                                   [](int32_t x, int32_t y) { return x != y; });
    }
    if (tag[0] == 'b') {
      return compare_host<bool>(self, other,
                                [](bool x, bool y) { return x != y; });
    }
    if (tag[0] == 'c') {
      return compare_host<int8_t>(self, other,
                                  [](int8_t x, int8_t y) { return x != y; });
    }
    if (tag[0] == 'C') {
      return compare_host<uint8_t>(self, other,
                                   [](uint8_t x, uint8_t y) { return x != y; });
    }
  }
  return compare_impl(self, other, "ne", "ne");
}
Tensor lt_tensor_kernel(const Tensor& self, const Tensor& other) {
  if (const char* tag = validate_compare_pair(self, other)) {
    if (tag[0] == 'i') {
      return compare_host<int32_t>(self, other,
                                   [](int32_t x, int32_t y) { return x < y; });
    }
    if (tag[0] == 'c') {
      return compare_host<int8_t>(self, other,
                                  [](int8_t x, int8_t y) { return x < y; });
    }
    if (tag[0] == 'C') {
      return compare_host<uint8_t>(self, other,
                                   [](uint8_t x, uint8_t y) { return x < y; });
    }
    if (tag[0] == 'b') {
      return compare_host<bool>(self, other,
                                [](bool x, bool y) { return x < y; });
    }
  }
  return compare_impl(self, other, "lt", "lt");
}
Tensor le_tensor_kernel(const Tensor& self, const Tensor& other) {
  if (const char* tag = validate_compare_pair(self, other)) {
    if (tag[0] == 'i') {
      return compare_host<int32_t>(self, other,
                                   [](int32_t x, int32_t y) { return x <= y; });
    }
    if (tag[0] == 'c') {
      return compare_host<int8_t>(self, other,
                                  [](int8_t x, int8_t y) { return x <= y; });
    }
    if (tag[0] == 'C') {
      return compare_host<uint8_t>(self, other,
                                   [](uint8_t x, uint8_t y) { return x <= y; });
    }
    if (tag[0] == 'b') {
      return compare_host<bool>(self, other,
                                [](bool x, bool y) { return x <= y; });
    }
  }
  return compare_impl(self, other, "le", "le");
}
Tensor gt_tensor_kernel(const Tensor& self, const Tensor& other) {
  if (const char* tag = validate_compare_pair(self, other)) {
    if (tag[0] == 'i') {
      return compare_host<int32_t>(self, other,
                                   [](int32_t x, int32_t y) { return x > y; });
    }
    if (tag[0] == 'c') {
      return compare_host<int8_t>(self, other,
                                  [](int8_t x, int8_t y) { return x > y; });
    }
    if (tag[0] == 'C') {
      return compare_host<uint8_t>(self, other,
                                   [](uint8_t x, uint8_t y) { return x > y; });
    }
    if (tag[0] == 'b') {
      return compare_host<bool>(self, other,
                                [](bool x, bool y) { return x > y; });
    }
  }
  return compare_impl(self, other, "gt", "gt");
}
Tensor ge_tensor_kernel(const Tensor& self, const Tensor& other) {
  if (const char* tag = validate_compare_pair(self, other)) {
    if (tag[0] == 'i') {
      return compare_host<int32_t>(self, other,
                                   [](int32_t x, int32_t y) { return x >= y; });
    }
    if (tag[0] == 'c') {
      return compare_host<int8_t>(self, other,
                                  [](int8_t x, int8_t y) { return x >= y; });
    }
    if (tag[0] == 'C') {
      return compare_host<uint8_t>(self, other,
                                   [](uint8_t x, uint8_t y) { return x >= y; });
    }
    if (tag[0] == 'b') {
      return compare_host<bool>(self, other,
                                [](bool x, bool y) { return x >= y; });
    }
  }
  return compare_impl(self, other, "ge", "ge");
}

// Scalar comparisons follow the weak-scalar rule the CPU kernels apply:
// the scalar folds into the self dtype (never widening an Int32 tensor to
// Float32), so the tensor form sees one dtype on both sides.
Tensor eq_scalar_kernel(const Tensor& self, Scalar other) {
  if (self.dtype() != DType::Float32) {
    return eq_tensor_kernel(
        self, full_kernel({}, other, self.dtype(), self.device(), false)
                  .expand(static_cast<std::vector<int64_t>>(self.shape())));
  }
  return eq_tensor_kernel(self, fold_scalar(self, other));
}
Tensor ne_scalar_kernel(const Tensor& self, Scalar other) {
  if (self.dtype() != DType::Float32) {
    return ne_tensor_kernel(
        self, full_kernel({}, other, self.dtype(), self.device(), false)
                  .expand(static_cast<std::vector<int64_t>>(self.shape())));
  }
  return ne_tensor_kernel(self, fold_scalar(self, other));
}
Tensor lt_scalar_kernel(const Tensor& self, Scalar other) {
  if (self.dtype() != DType::Float32) {
    return lt_tensor_kernel(
        self, full_kernel({}, other, self.dtype(), self.device(), false)
                  .expand(static_cast<std::vector<int64_t>>(self.shape())));
  }
  return lt_tensor_kernel(self, fold_scalar(self, other));
}
Tensor le_scalar_kernel(const Tensor& self, Scalar other) {
  if (self.dtype() != DType::Float32) {
    return le_tensor_kernel(
        self, full_kernel({}, other, self.dtype(), self.device(), false)
                  .expand(static_cast<std::vector<int64_t>>(self.shape())));
  }
  return le_tensor_kernel(self, fold_scalar(self, other));
}
Tensor gt_scalar_kernel(const Tensor& self, Scalar other) {
  if (self.dtype() != DType::Float32) {
    return gt_tensor_kernel(
        self, full_kernel({}, other, self.dtype(), self.device(), false)
                  .expand(static_cast<std::vector<int64_t>>(self.shape())));
  }
  return gt_tensor_kernel(self, fold_scalar(self, other));
}
Tensor ge_scalar_kernel(const Tensor& self, Scalar other) {
  if (self.dtype() != DType::Float32) {
    return ge_tensor_kernel(
        self, full_kernel({}, other, self.dtype(), self.device(), false)
                  .expand(static_cast<std::vector<int64_t>>(self.shape())));
  }
  return ge_tensor_kernel(self, fold_scalar(self, other));
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, ComparisonKernels) {
  m.impl("eq.Tensor", &tensorplay::vulkan::ops::eq_tensor_kernel);
  m.impl("ne.Tensor", &tensorplay::vulkan::ops::ne_tensor_kernel);
  m.impl("lt.Tensor", &tensorplay::vulkan::ops::lt_tensor_kernel);
  m.impl("le.Tensor", &tensorplay::vulkan::ops::le_tensor_kernel);
  m.impl("gt.Tensor", &tensorplay::vulkan::ops::gt_tensor_kernel);
  m.impl("ge.Tensor", &tensorplay::vulkan::ops::ge_tensor_kernel);
  m.impl("eq.Scalar", &tensorplay::vulkan::ops::eq_scalar_kernel);
  m.impl("ne.Scalar", &tensorplay::vulkan::ops::ne_scalar_kernel);
  m.impl("lt.Scalar", &tensorplay::vulkan::ops::lt_scalar_kernel);
  m.impl("le.Scalar", &tensorplay::vulkan::ops::le_scalar_kernel);
  m.impl("gt.Scalar", &tensorplay::vulkan::ops::gt_scalar_kernel);
  m.impl("ge.Scalar", &tensorplay::vulkan::ops::ge_scalar_kernel);
}

#endif /* USE_VULKAN */
