#pragma once

#include "Tensor.h"
#include "Scalar.h"

namespace tensorplay {

// Materializes a Scalar as a 0-dim tensor carrying the scalar's own dtype,
// on CPU by default or on the requested device otherwise (the dispatcher
// routes a mixed-device op by the first tensor argument, so reflected scalar
// ops pass the tensor operand's device here).
inline Tensor scalar_to_tensor(const Scalar& s, Device device = Device(DeviceType::CPU)) {
  return Tensor::full({}, s, s.type(), device);
}

namespace native {

// A wrapped scalar tensor participates in the result type computation only
// when no plain tensor operand is present: 'float32_tensor + 2' stays
// float32, while '2 + 3' promotes to the default float dtype.  The marker is
// read by the iterator's common dtype computation through
// TensorImpl::is_wrapped_number.
inline Tensor wrapped_scalar_tensor(
    const Scalar& scalar,
    const Device device = Device(DeviceType::CPU)) {
  auto tensor = scalar_to_tensor(scalar, device);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

}  // namespace native
}  // namespace tensorplay
