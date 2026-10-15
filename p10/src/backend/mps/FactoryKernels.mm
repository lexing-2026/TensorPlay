#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Context.h"
#include "Exception.h"

#include "backend/mps/MPSDevice.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

namespace tensorplay {
namespace mps {

// ---------------------------------------------------------------------------
// Factory kernels for the MPS backend.
//
// The MPS allocator hands out shared-storage (unified-memory) MTLBuffers whose
// `contents` pointer is host-visible, so `data_ptr()` is a valid CPU mapping
// of the GPU memory.  Factory fills and the scalar fill path below write
// through that mapping; the GPU observes the writes on next command buffer
// execution.  This keeps the skeleton backend free of per-op Metal compute
// shaders while remaining correct on Apple Silicon.
// ---------------------------------------------------------------------------

namespace {

DType resolve_factory_dtype(const std::optional<DType>& dtype) {
  return (dtype.has_value() && *dtype != DType::Undefined)
             ? *dtype
             : globalContext().defaultDType();
}

DType infer_full_dtype(const Scalar& fill_value) {
  if (fill_value.isBoolean()) return DType::Bool;
  if (fill_value.isIntegral(false)) return DType::Int64;
  if (fill_value.isComplex()) {
    return globalContext().defaultDType() == DType::Float64
               ? DType::ComplexDouble
               : DType::ComplexFloat;
  }
  return globalContext().defaultDType();
}

} // namespace

Tensor& fill_kernel(Tensor& self, const Scalar& value) {
  if (self.numel() == 0) return self;
  if (isQuantizedType(self.dtype())) {
    TP_THROW(NotImplementedError, "fill_ not implemented for quantized MPS tensors");
  }
  // Shared-memory buffers are host-visible; fill in place through the mapping.
  if (self.is_contiguous()) {
#define OP_CASE(ctype, name)                                             \
  case DType::name: {                                                    \
    const ctype val = value.to<ctype>();                                 \
    ctype* data = self.data_ptr<ctype>();                                \
    std::fill(data, data + self.numel(), val);                           \
    break;                                                               \
  }
    switch (self.dtype()) {
      TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(OP_CASE)
      default:
        TP_THROW(NotImplementedError,
                 "fill_ not implemented for this dtype on MPS");
    }
#undef OP_CASE
    return self;
  }
  // Strided path: only contiguous tensors are expected in the skeleton
  // backend; non-contiguous views fall back to a flat walk over the storage.
  void* data = self.data_ptr();
  switch (self.dtype()) {
    case DType::Float32: {
      const float val = value.to<float>();
      float* p = static_cast<float*>(data);
      for (int64_t i = 0; i < self.numel(); ++i) p[i] = val;
      break;
    }
    case DType::Float64: {
      const double val = value.to<double>();
      double* p = static_cast<double*>(data);
      for (int64_t i = 0; i < self.numel(); ++i) p[i] = val;
      break;
    }
    case DType::Int64: {
      const int64_t val = value.to<int64_t>();
      int64_t* p = static_cast<int64_t*>(data);
      for (int64_t i = 0; i < self.numel(); ++i) p[i] = val;
      break;
    }
    default:
      TP_THROW(NotImplementedError,
               "fill_ not implemented for non-contiguous tensors of this "
               "dtype on MPS");
  }
  return self;
}

Tensor& zero_inplace_kernel(Tensor& self) {
  if (self.numel() == 0) return self;
  void* data = self.data_ptr();
  if (self.is_contiguous()) {
    std::memset(data, 0, static_cast<size_t>(self.numel()) * self.itemsize());
    return self;
  }
  switch (self.dtype()) {
    case DType::Float32: {
      float* p = static_cast<float*>(data);
      for (int64_t i = 0; i < self.numel(); ++i) p[i] = 0.0f;
      break;
    }
    case DType::Float64: {
      double* p = static_cast<double*>(data);
      for (int64_t i = 0; i < self.numel(); ++i) p[i] = 0.0;
      break;
    }
    case DType::Int64: {
      int64_t* p = static_cast<int64_t*>(data);
      for (int64_t i = 0; i < self.numel(); ++i) p[i] = 0;
      break;
    }
    default:
      TP_THROW(NotImplementedError,
               "zero_ not implemented for non-contiguous tensors of this "
               "dtype on MPS");
  }
  return self;
}

Tensor zeros_kernel(const std::vector<int64_t>& size, DType dtype,
                    Device device, bool pin_memory) {
  if (pin_memory) {
    TP_THROW(RuntimeError, "pin_memory is only valid for CPU tensors");
  }
  Tensor t(size, dtype, device);
  if (t.numel() > 0) {
    std::memset(t.data_ptr(), 0,
                static_cast<size_t>(t.numel()) * t.itemsize());
  }
  return t;
}

Tensor ones_kernel(const std::vector<int64_t>& size, DType dtype,
                   Device device, bool pin_memory) {
  if (pin_memory) {
    TP_THROW(RuntimeError, "pin_memory is only valid for CPU tensors");
  }
  Tensor t(size, dtype, device);
  fill_kernel(t, 1);
  return t;
}

Tensor empty_kernel(const std::vector<int64_t>& size, DType dtype,
                    Device device, bool pin_memory) {
  if (pin_memory) {
    TP_THROW(RuntimeError, "pin_memory is only valid for CPU tensors");
  }
  return Tensor(size, dtype, device);
}

Tensor full_kernel(const std::vector<int64_t>& size, Scalar fill_value,
                   DType dtype, Device device, bool pin_memory) {
  if (pin_memory) {
    TP_THROW(RuntimeError, "pin_memory is only valid for CPU tensors");
  }
  if (dtype == DType::Undefined) {
    dtype = infer_full_dtype(fill_value);
  }
  Tensor t(size, dtype, device);
  fill_kernel(t, fill_value);
  return t;
}

Tensor zeros_like_kernel(const Tensor& self, DType dtype,
                         std::optional<Device> device) {
  if (dtype == DType::Undefined) dtype = self.dtype();
  Device dev = device.has_value() ? *device : self.device();
  return zeros_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype,
                      dev, false);
}

Tensor ones_like_kernel(const Tensor& self, DType dtype,
                        std::optional<Device> device) {
  if (dtype == DType::Undefined) dtype = self.dtype();
  Device dev = device.has_value() ? *device : self.device();
  return ones_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype,
                     dev, false);
}

Tensor empty_like_kernel(const Tensor& self, DType dtype,
                         std::optional<Device> device) {
  if (dtype == DType::Undefined) dtype = self.dtype();
  Device dev = device.has_value() ? *device : self.device();
  if (self.impl()->has_quantizer()) {
    TP_THROW(NotImplementedError,
             "empty_like not implemented for quantized MPS tensors");
  }
  return empty_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype,
                      dev, false);
}

Tensor full_like_kernel(const Tensor& self, const Scalar& fill_value,
                        DType dtype, std::optional<Device> device) {
  if (dtype == DType::Undefined) dtype = self.dtype();
  Device dev = device.has_value() ? *device : self.device();
  Tensor t = empty_kernel(static_cast<std::vector<int64_t>>(self.shape()),
                          dtype, dev, false);
  fill_kernel(t, fill_value);
  return t;
}

// ---------------------------------------------------------------------------
// Range factories.  These write through the host-visible shared-memory
// mapping of the MPS buffers, which is valid on unified-memory Apple
// silicon and keeps the factories free of per-op Metal kernels.
// ---------------------------------------------------------------------------

namespace {

template <typename ctype>
void fill_host_sequence(ctype* data, int64_t len,
                          const std::function<ctype(int64_t)>& value_at) {
  for (int64_t i = 0; i < len; ++i) {
    data[i] = value_at(i);
  }
}

#define MPS_RANGE_CASE(ctype, name, DATA, LEN, VALUE_AT)        \
  case DType::name:                                             \
    fill_host_sequence<ctype>((ctype*)(DATA), (LEN), (VALUE_AT)); \
    break;

} // namespace

Tensor eye_kernel(int64_t n, int64_t m, DType dtype, Device device) {
  if (m < 0) m = n;
  Tensor t = Tensor({n, m}, dtype, device);
  if (n * m > 0) {
    std::memset(t.data_ptr(), 0, static_cast<size_t>(n) * m * t.itemsize());
  }
  const int64_t min_dim = std::min(n, m);
  if (min_dim == 0) return t;
#define EYE_CASE(ctype, name)                                          \
  case DType::name: {                                                  \
    ctype* data = t.data_ptr<ctype>();                                 \
    for (int64_t i = 0; i < min_dim; ++i) data[i * m + i] = ctype(1);  \
    break;                                                             \
  }
  switch (dtype) {
    TENSORPLAY_FORALL_SCALAR_TYPES(EYE_CASE)
    default:
      TP_THROW(NotImplementedError, "eye: unsupported dtype on MPS");
  }
#undef EYE_CASE
  return t;
}

Tensor arange_start_step_kernel(Scalar start, Scalar end, Scalar step,
                                DType dtype, Device device) {
  const double s_d = start.toDouble();
  const double e_d = end.toDouble();
  const double st_d = step.toDouble();
  if (st_d == 0) {
    TP_THROW(RuntimeError, "step must be nonzero");
  }
  int64_t len = static_cast<int64_t>(std::ceil((e_d - s_d) / st_d));
  if (len < 0) len = 0;

  if (dtype == DType::Undefined) {
    if (start.isFloatingPoint() || end.isFloatingPoint() ||
        step.isFloatingPoint()) {
      dtype = globalContext().defaultDType();
    } else {
      dtype = DType::Int64;
    }
  }
  Tensor t({len}, dtype, device);
  if (len == 0) return t;

  const double st = st_d;
  const double s = s_d;
  switch (dtype) {
    MPS_RANGE_CASE(float, Float32, t.data_ptr(), len,
                   [&](int64_t i) { return static_cast<float>(s + i * st); })
    MPS_RANGE_CASE(double, Float64, t.data_ptr(), len,
                   [&](int64_t i) { return s + i * st; })
    MPS_RANGE_CASE(int64_t, Int64, t.data_ptr(), len,
                   [&](int64_t i) {
                     return static_cast<int64_t>(s + i * st);
                   })
    MPS_RANGE_CASE(int32_t, Int32, t.data_ptr(), len,
                   [&](int64_t i) {
                     return static_cast<int32_t>(s + i * st);
                   })
    MPS_RANGE_CASE(int16_t, Int16, t.data_ptr(), len,
                   [&](int64_t i) {
                     return static_cast<int16_t>(s + i * st);
                   })
    MPS_RANGE_CASE(int8_t, Int8, t.data_ptr(), len,
                   [&](int64_t i) { return static_cast<int8_t>(s + i * st); })
    MPS_RANGE_CASE(uint8_t, UInt8, t.data_ptr(), len,
                   [&](int64_t i) {
                     return static_cast<uint8_t>(s + i * st);
                   })
    MPS_RANGE_CASE(Half, Float16, t.data_ptr(), len,
                   [&](int64_t i) {
                     return Half(static_cast<float>(s + i * st));
                   })
    MPS_RANGE_CASE(BFloat16, BFloat16, t.data_ptr(), len,
                   [&](int64_t i) {
                     return BFloat16(static_cast<float>(s + i * st));
                   })
    default:
      TP_THROW(NotImplementedError,
               "arange does not support this dtype on MPS");
  }
  return t;
}

Tensor arange_kernel(Scalar end, DType dtype, Device device) {
  return arange_start_step_kernel(0, end, 1, dtype, device);
}

Tensor linspace_kernel(Scalar start, Scalar end, int64_t steps, DType dtype,
                       Device device) {
  if (steps < 0) {
    TP_THROW(RuntimeError, "number of steps must be non-negative");
  }
  Tensor t({steps}, dtype, device);
  if (steps == 0) return t;
  if (steps == 1) {
    fill_kernel(t, start);
    return t;
  }
  const double s = start.toDouble();
  const double e = end.toDouble();
  const double step = (e - s) / static_cast<double>(steps - 1);
  switch (dtype) {
    MPS_RANGE_CASE(float, Float32, t.data_ptr(), steps,
                   [&](int64_t i) { return static_cast<float>(s + i * step); })
    MPS_RANGE_CASE(double, Float64, t.data_ptr(), steps,
                   [&](int64_t i) { return s + i * step; })
    MPS_RANGE_CASE(int64_t, Int64, t.data_ptr(), steps,
                   [&](int64_t i) {
                     return static_cast<int64_t>(s + i * step);
                   })
    MPS_RANGE_CASE(int32_t, Int32, t.data_ptr(), steps,
                   [&](int64_t i) {
                     return static_cast<int32_t>(s + i * step);
                   })
    MPS_RANGE_CASE(Half, Float16, t.data_ptr(), steps,
                   [&](int64_t i) {
                     return Half(static_cast<float>(s + i * step));
                   })
    MPS_RANGE_CASE(BFloat16, BFloat16, t.data_ptr(), steps,
                   [&](int64_t i) {
                     return BFloat16(static_cast<float>(s + i * step));
                   })
    default:
      TP_THROW(NotImplementedError,
               "linspace does not support this dtype on MPS");
  }
  return t;
}

Tensor logspace_kernel(Scalar start, Scalar end, int64_t steps, double base,
                       DType dtype, Device device) {
  if (steps < 0) {
    TP_THROW(RuntimeError, "number of steps must be non-negative");
  }
  Tensor t({steps}, dtype, device);
  if (steps == 0) return t;
  if (steps == 1) {
    fill_kernel(t, Scalar(std::pow(base, start.toDouble())));
    return t;
  }
  const double s = start.toDouble();
  const double e = end.toDouble();
  const double step = (e - s) / static_cast<double>(steps - 1);
  switch (dtype) {
    MPS_RANGE_CASE(float, Float32, t.data_ptr(), steps,
                   [&](int64_t i) {
                     return static_cast<float>(
                         std::pow(base, s + i * step));
                   })
    MPS_RANGE_CASE(double, Float64, t.data_ptr(), steps,
                   [&](int64_t i) { return std::pow(base, s + i * step); })
    MPS_RANGE_CASE(Half, Float16, t.data_ptr(), steps,
                   [&](int64_t i) {
                     return Half(static_cast<float>(
                         std::pow(base, s + i * step)));
                   })
    MPS_RANGE_CASE(BFloat16, BFloat16, t.data_ptr(), steps,
                   [&](int64_t i) {
                     return BFloat16(static_cast<float>(
                         std::pow(base, s + i * step)));
                   })
    default:
      TP_THROW(NotImplementedError,
               "logspace does not support this dtype on MPS");
  }
  return t;
}

namespace {

Tensor empty_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                  std::optional<Device> device, bool pin_memory) {
  return empty_kernel(size, resolve_factory_dtype(dtype),
                      device.value_or(Device(DeviceType::MPS)), pin_memory);
}

Tensor empty_memory_format_stub(const std::vector<int64_t>& size,
                                std::optional<DType> dtype,
                                std::optional<int64_t> layout,
                                std::optional<Device> device,
                                std::optional<bool> pin_memory,
                                std::optional<int64_t> memory_format) {
  if (layout.has_value() && *layout != 5) {
    TP_THROW(NotImplementedError,
             "empty is only implemented for strided (dense) layout tensors");
  }
  const DType dt = resolve_factory_dtype(dtype);
  Tensor out = empty_kernel(size, dt,
                            device.value_or(Device(DeviceType::MPS)),
                            pin_memory.value_or(false));
  const auto format = static_cast<MemoryFormat>(
      memory_format.value_or(static_cast<int64_t>(MemoryFormat::Contiguous)));
  TP_CHECK(format != MemoryFormat::Preserve,
           "empty: preserve memory format is unsupported");
  if (format == MemoryFormat::ChannelsLast ||
      format == MemoryFormat::ChannelsLast3d) {
    out = out.as_strided(size, get_channels_last_strides(size));
  }
  return out;
}

Tensor zeros_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                  std::optional<Device> device, bool pin_memory) {
  return zeros_kernel(size, resolve_factory_dtype(dtype),
                      device.value_or(Device(DeviceType::MPS)), pin_memory);
}

Tensor ones_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                 std::optional<Device> device, bool pin_memory) {
  return ones_kernel(size, resolve_factory_dtype(dtype),
                     device.value_or(Device(DeviceType::MPS)), pin_memory);
}

Tensor full_stub(const std::vector<int64_t>& size, const Scalar& fill_value,
                 DType dtype, std::optional<Device> device, bool pin_memory) {
  return full_kernel(size, fill_value, dtype,
                     device.value_or(Device(DeviceType::MPS)), pin_memory);
}

Tensor eye_stub(int64_t n, int64_t m, DType dtype,
                std::optional<Device> device) {
  return eye_kernel(n, m, dtype,
                    device.value_or(Device(DeviceType::MPS)));
}

Tensor arange_start_step_stub(const Scalar& start, const Scalar& end,
                              const Scalar& step, DType dtype,
                              std::optional<Device> device) {
  return arange_start_step_kernel(start, end, step, dtype,
                                  device.value_or(Device(DeviceType::MPS)));
}

Tensor arange_end_stub(const Scalar& end, DType dtype,
                       std::optional<Device> device) {
  return arange_kernel(end, dtype,
                       device.value_or(Device(DeviceType::MPS)));
}

Tensor linspace_stub(const Scalar& start, const Scalar& end, int64_t steps,
                     DType dtype, std::optional<Device> device) {
  return linspace_kernel(start, end, steps, dtype,
                         device.value_or(Device(DeviceType::MPS)));
}

Tensor logspace_stub(const Scalar& start, const Scalar& end, int64_t steps,
                     double base, DType dtype,
                     std::optional<Device> device) {
  return logspace_kernel(start, end, steps, base, dtype,
                         device.value_or(Device(DeviceType::MPS)));
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(MPS, FactoryKernels) {
  m.impl("zeros", zeros_stub);
  m.impl("ones", ones_stub);
  m.impl("full", full_stub);
  m.impl("empty", empty_stub);
  m.impl("empty.memory_format", empty_memory_format_stub);
  m.impl("fill_.Scalar", fill_kernel);
  m.impl("zero_", zero_inplace_kernel);
  m.impl("empty_like", empty_like_kernel);
  m.impl("zeros_like", zeros_like_kernel);
  m.impl("ones_like", ones_like_kernel);
  m.impl("full_like", full_like_kernel);
  m.impl("eye", eye_stub);
  m.impl("arange", arange_start_step_stub);
  m.impl("arange.end", arange_end_stub);
  m.impl("linspace", linspace_stub);
  m.impl("logspace", logspace_stub);
}

} // namespace mps
} // namespace tensorplay