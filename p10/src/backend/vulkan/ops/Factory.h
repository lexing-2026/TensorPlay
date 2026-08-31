#pragma once

#ifdef USE_VULKAN

#include "Common.h"

namespace tensorplay {
namespace vulkan {
namespace ops {

Tensor empty_kernel(
    const std::vector<int64_t>& size,
    DType dtype,
    Device device,
    bool pin_memory);

Tensor zeros_kernel(
    const std::vector<int64_t>& size,
    DType dtype,
    Device device,
    bool pin_memory);

Tensor ones_kernel(
    const std::vector<int64_t>& size,
    DType dtype,
    Device device,
    bool pin_memory);

Tensor full_kernel(
    const std::vector<int64_t>& size,
    Scalar fill_value,
    DType dtype,
    Device device,
    bool pin_memory);

Tensor empty_like_kernel(
    const Tensor& self,
    DType dtype,
    std::optional<Device> device);

Tensor zeros_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device);

Tensor ones_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device);

Tensor full_like_kernel(
    const Tensor& self,
    Scalar fill_value,
    DType dtype,
    std::optional<Device> device);

Tensor& fill_kernel(Tensor& self, Scalar value);

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
