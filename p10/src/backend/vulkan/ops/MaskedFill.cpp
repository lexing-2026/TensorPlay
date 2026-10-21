#ifdef USE_VULKAN

#include "Common.h"
#include "Convert.h"
#include "Factory.h"

namespace tensorplay {
namespace vulkan {
namespace ops {

Tensor where_kernel(const Tensor& condition, const Tensor& self, const Tensor& other);

namespace {

void validate_payload(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32 || t.dtype() == DType::Float16 ||
          t.dtype() == DType::Int32 || t.dtype() == DType::Int64,
      std::string("Vulkan ") + name +
          " supports Float32, Float16, Int32 and Int64 tensors only");
  TP_CHECK(
      t.dim() <= 4, std::string("Vulkan ") + name + " supports up to 4d tensors");
}

/*
 * The walk behind a selection answers for both public forms: a masked lane
 * takes the fill and every other lane keeps the payload, which is the same
 * choice between two branches.  The value-tensor overload only accepts a
 * 0-dimensional value here, so it contributes its single element and joins the
 * scalar form.
 *
 * The fill rides a plane of one in the payload's own width, which the walk
 * repeats, so a mask narrower than the input broadcasts without either the
 * mask or the fill being materialized at the output shape, and one selection
 * implementation covers every element type instead of a per-width copy.
 */
Tensor masked_fill_impl(
    const Tensor& self,
    const Tensor& mask,
    Scalar value,
    const char* name) {
  validate_payload(self, name);
  TP_CHECK(
      mask.dtype() == DType::Bool,
      std::string("Vulkan ") + name + " expects a Bool mask");
  TP_CHECK(
      mask.dim() <= 4,
      std::string("Vulkan ") + name + " supports up to 4d masks");

  const Tensor mask_on_device =
      mask.device().is_vulkan() ? mask : mask.to(self.device());
  // A number stored into a payload of a given width is read into that width
  // first, so the plane holds what the masked lanes take.
  const Tensor held =
      full_kernel({1}, value, self.dtype(), self.device(), false);
  return where_kernel(mask_on_device, held, self);
}

} // namespace

Tensor masked_fill_scalar_kernel(
    const Tensor& self, const Tensor& mask, const Scalar& value) {
  return masked_fill_impl(self, mask, value, "masked_fill");
}

Tensor masked_fill_tensor_kernel(
    const Tensor& self, const Tensor& mask, const Tensor& value) {
  TP_CHECK(
      value.dim() == 0,
      "Vulkan masked_fill.Tensor: only a 0-dimensional value tensor is "
      "supported");
  return masked_fill_impl(self, mask, value.item(), "masked_fill.Tensor");
}

Tensor& masked_fill_scalar_inplace_kernel(
    Tensor& self, const Tensor& mask, const Scalar& value) {
  self.copy_(masked_fill_scalar_kernel(self, mask, value));
  return self;
}

Tensor& masked_fill_tensor_inplace_kernel(
    Tensor& self, const Tensor& mask, const Tensor& value) {
  self.copy_(masked_fill_tensor_kernel(self, mask, value));
  return self;
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, MaskedFillKernels) {
  m.impl("masked_fill", &tensorplay::vulkan::ops::masked_fill_scalar_kernel);
  m.impl(
      "masked_fill.Tensor",
      &tensorplay::vulkan::ops::masked_fill_tensor_kernel);
  m.impl(
      "masked_fill_",
      &tensorplay::vulkan::ops::masked_fill_scalar_inplace_kernel);
  m.impl(
      "masked_fill_.Tensor",
      &tensorplay::vulkan::ops::masked_fill_tensor_inplace_kernel);
}

#endif /* USE_VULKAN */
