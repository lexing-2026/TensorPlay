#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Context.h"
#include "Exception.h"
#include "Generator.h"
#include "DistributionsHelper.h"
#include "DistributionDispatch.h"
#include "Quantizer.h"

#include "backend/mps/MPSAllocator.h"
#include "backend/mps/MPSDevice.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSStream.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "tensorplay/ops/TPXOpsGenerated.h"

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

// GPU constant fill for a contiguous, offset-0 tensor.
template <typename T>
void fill_shader_impl(Tensor& self, T value) {
  const uint32_t numel = static_cast<uint32_t>(self.numel());
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      @try {
        id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
        const std::string kernel = "fill_" + scalarToMetalTypeString(self.dtype());
        id<MTLComputePipelineState> PSO =
            MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
                kernel);
        [computeEncoder setComputePipelineState:PSO];
        mtl_setArgs(computeEncoder, self, value, numel);
        mtl_dispatch1DJob(computeEncoder, PSO, static_cast<NSUInteger>(numel));
        mpsStream->endKernelCoalescing();
      } @catch (NSException* e) {
        std::string reason =
            [e reason] ? std::string([[e reason] UTF8String])
                       : std::string("unknown Metal exception");
        throw tensorplay::RuntimeError(
            {__FILE__, __func__, __LINE__},
            tensorplay::detail::format_msg("MPS fill shader failed: ", reason));
      }
    }
  });
}

} // namespace

Tensor& fill_kernel(Tensor& self, const Scalar& value) {
  if (self.numel() == 0) return self;
  if (isQuantizedType(self.dtype())) {
    // Per-tensor affine quantized tensors store round(value/scale)+zero_point
    // codes; other schemes are not supported on MPS yet.
    if (!self.impl()->has_quantizer() ||
        self.impl()->quantizer()->qscheme() != kPerTensorAffine) {
      TP_THROW(NotImplementedError,
               "fill_ for non-per-tensor quantized MPS tensors is not "
               "implemented");
    }
    if (!self.is_contiguous()) {
      TP_THROW(NotImplementedError,
               "fill_ for non-contiguous quantized MPS tensors is not "
               "implemented");
    }
    const double scale = self.impl()->quantizer()->scale();
    const int64_t zp = self.impl()->quantizer()->zero_point();
    const float v = static_cast<float>(value.toDouble());
    const float inverse = 1.0f / static_cast<float>(scale);
    const bool is_nan = std::isnan(v);
    void* data = self.data_ptr();
    switch (self.dtype()) {
      case DType::QInt8: {
        const int64_t code =
            is_nan ? INT8_MAX
                   : std::clamp(static_cast<int64_t>(std::nearbyint(v * inverse)) + zp,
                                static_cast<int64_t>(INT8_MIN),
                                static_cast<int64_t>(INT8_MAX));
        std::fill(static_cast<int8_t*>(data),
                  static_cast<int8_t*>(data) + self.numel(),
                  static_cast<int8_t>(code));
        break;
      }
      case DType::QUInt8: {
        const int64_t code =
            is_nan ? UINT8_MAX
                   : std::clamp(static_cast<int64_t>(std::nearbyint(v * inverse)) + zp,
                                static_cast<int64_t>(0),
                                static_cast<int64_t>(UINT8_MAX));
        std::fill(static_cast<uint8_t*>(data),
                  static_cast<uint8_t*>(data) + self.numel(),
                  static_cast<uint8_t>(code));
        break;
      }
      case DType::QInt32: {
        const int64_t code =
            is_nan ? INT32_MIN
                   : std::clamp(static_cast<int64_t>(std::nearbyint(v * inverse)) + zp,
                                static_cast<int64_t>(INT32_MIN),
                                static_cast<int64_t>(INT32_MAX));
        std::fill(static_cast<int32_t*>(data),
                  static_cast<int32_t*>(data) + self.numel(),
                  static_cast<int32_t>(code));
        break;
      }
      default:
        TP_THROW(NotImplementedError,
                 "fill_ not implemented for this quantized dtype on MPS");
    }
    return self;
  }
  // Contiguous offset-0 tensors are filled on the GPU by the Fill shader
  // (float/half/bfloat/int*/uint8/bool); double and complex dtypes fall back
  // to the host path since the shaders do not target them.
  if (self.is_contiguous() && self.storage_offset() == 0) {
    switch (self.dtype()) {
      case DType::Float32:
        fill_shader_impl<float>(self, value.to<float>());
        return self;
      case DType::Float16:
        fill_shader_impl<Half>(self, value.to<Half>());
        return self;
      case DType::BFloat16:
        fill_shader_impl<BFloat16>(self, value.to<BFloat16>());
        return self;
      case DType::Int64:
        fill_shader_impl<int64_t>(self, value.to<int64_t>());
        return self;
      case DType::Int32:
        fill_shader_impl<int32_t>(self, value.to<int32_t>());
        return self;
      case DType::Int16:
        fill_shader_impl<int16_t>(self, value.to<int16_t>());
        return self;
      case DType::Int8:
        fill_shader_impl<int8_t>(self, value.to<int8_t>());
        return self;
      case DType::UInt8:
        fill_shader_impl<uint8_t>(self, value.to<uint8_t>());
        return self;
      case DType::Bool:
        fill_shader_impl<bool>(self, value.to<bool>());
        return self;
      default:
        break;
    }
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
  if (self.is_contiguous() && self.storage_offset() == 0) {
    switch (self.dtype()) {
      case DType::Float32:
        fill_shader_impl<float>(self, 0.0f);
        return self;
      case DType::Float16:
        fill_shader_impl<Half>(self, Half(0.0f));
        return self;
      case DType::BFloat16:
        fill_shader_impl<BFloat16>(self, BFloat16(0.0f));
        return self;
      case DType::Int64:
        fill_shader_impl<int64_t>(self, int64_t(0));
        return self;
      case DType::Int32:
        fill_shader_impl<int32_t>(self, int32_t(0));
        return self;
      case DType::Int16:
        fill_shader_impl<int16_t>(self, int16_t(0));
        return self;
      case DType::Int8:
        fill_shader_impl<int8_t>(self, int8_t(0));
        return self;
      case DType::UInt8:
        fill_shader_impl<uint8_t>(self, uint8_t(0));
        return self;
      case DType::Bool:
        fill_shader_impl<bool>(self, false);
        return self;
      default:
        break;
    }
  }
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
  Tensor out = empty_kernel(static_cast<std::vector<int64_t>>(self.shape()),
                            dtype, dev, false);
  if (self.impl()->has_quantizer()) {
    const QuantizerPtr src_q = self.impl()->quantizer();
    switch (src_q->qscheme()) {
      case kPerTensorAffine:
        out.impl()->set_quantizer(make_per_tensor_affine_quantizer(
            src_q->scale(), src_q->zero_point(), dtype));
        break;
      case kPerChannelAffine:
      case kPerChannelAffineFloatQParams: {
        Tensor scales = src_q->scales();
        Tensor zero_points = src_q->zero_points();
        if (scales.device() != dev) scales = scales.to(dev);
        if (zero_points.device() != dev) zero_points = zero_points.to(dev);
        out.impl()->set_quantizer(make_per_channel_affine_quantizer(
            scales, zero_points, src_q->axis(), dtype));
        break;
      }
      default:
        TP_THROW(ValueError,
                 "empty_like: unsupported quantization scheme on MPS");
    }
  }
  return out;
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
  if (n * m == 0) return t;

  const std::string suffix = scalarToMetalTypeString(dtype);
  const bool use_shader =
      suffix == "float" || suffix == "half" || suffix == "bfloat" ||
      suffix == "long" || suffix == "int" || suffix == "short" ||
      suffix == "char" || suffix == "uchar" || suffix == "bool";
  if (use_shader) {
    const uint32_t rows = static_cast<uint32_t>(n);
    const uint32_t cols = static_cast<uint32_t>(m);
    MPSStream* mpsStream = getCurrentMPSStream();
    dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
        id<MTLComputePipelineState> PSO =
            MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
                "eye_" + suffix);
        [computeEncoder setComputePipelineState:PSO];
        mtl_setArgs(computeEncoder, t, rows, cols);
        mtl_dispatch1DJob(computeEncoder, PSO,
                          static_cast<NSUInteger>(n * m));
        mpsStream->endKernelCoalescing();
      }
    });
    return t;
  }

  // Host fallback for dtypes without a shader (double and complex).
  const int64_t min_dim = std::min(n, m);
  std::memset(t.data_ptr(), 0, static_cast<size_t>(n) * m * t.itemsize());
#define EYE_CASE(ctype, name)                                            case DType::name: {                                                      ctype* data = t.data_ptr<ctype>();                                     for (int64_t i = 0; i < min_dim; ++i) data[i * m + i] = ctype(1);      break;                                                               }
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

  // Float64 keeps the host path (the shaders do not target double).
  if (dtype == DType::Float64) {
    double* data = t.data_ptr<double>();
    for (int64_t i = 0; i < len; ++i) {
      data[i] = s_d + i * st_d;
    }
    return t;
  }

  const std::string suffix = scalarToMetalTypeString(dtype);
  if (suffix != "float" && suffix != "half" && suffix != "bfloat" &&
      suffix != "long" && suffix != "int" && suffix != "short" &&
      suffix != "char" && suffix != "uchar") {
    TP_THROW(NotImplementedError, "arange does not support this dtype on MPS");
  }
  const double s = s_d;
  const double st = st_d;
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> PSO =
          MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
              "arange_" + suffix);
      [computeEncoder setComputePipelineState:PSO];
      mtl_setArgs(computeEncoder, t, static_cast<float>(s),
                  static_cast<float>(st), static_cast<uint32_t>(len));
      mtl_dispatch1DJob(computeEncoder, PSO, static_cast<NSUInteger>(len));
      mpsStream->endKernelCoalescing();
    }
  });
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

  // Float64 keeps the host path (the shaders do not target double).
  if (dtype == DType::Float64) {
    double* data = t.data_ptr<double>();
    for (int64_t i = 0; i < steps; ++i) {
      data[i] = s + i * step;
    }
    return t;
  }
  const std::string suffix = scalarToMetalTypeString(dtype);
  if (suffix != "float" && suffix != "half" && suffix != "bfloat" &&
      suffix != "long" && suffix != "int") {
    TP_THROW(NotImplementedError, "linspace does not support this dtype on MPS");
  }
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> PSO =
          MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
              "linspace_" + suffix);
      [computeEncoder setComputePipelineState:PSO];
      mtl_setArgs(computeEncoder, t, static_cast<float>(s),
                  static_cast<float>(step), static_cast<uint32_t>(steps));
      mtl_dispatch1DJob(computeEncoder, PSO, static_cast<NSUInteger>(steps));
      mpsStream->endKernelCoalescing();
    }
  });
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

  // Float64 keeps the host path (the shaders do not target double).
  if (dtype == DType::Float64) {
    double* data = t.data_ptr<double>();
    for (int64_t i = 0; i < steps; ++i) {
      data[i] = std::pow(base, s + i * step);
    }
    return t;
  }
  const std::string suffix = scalarToMetalTypeString(dtype);
  if (suffix != "float" && suffix != "half" && suffix != "bfloat") {
    TP_THROW(NotImplementedError, "logspace does not support this dtype on MPS");
  }
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> PSO =
          MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
              "logspace_" + suffix);
      [computeEncoder setComputePipelineState:PSO];
      mtl_setArgs(computeEncoder, t, static_cast<float>(base),
                  static_cast<float>(s), static_cast<float>(step),
                  static_cast<uint32_t>(steps));
      mtl_dispatch1DJob(computeEncoder, PSO, static_cast<NSUInteger>(steps));
      mpsStream->endKernelCoalescing();
    }
  });
  return t;
}

// Random factories.  Samples are drawn on the host through the shared-memory
// mapping of the MPS buffers (valid on unified-memory Apple silicon), using
// the same generator and distributions as the CPU backend.
// ---------------------------------------------------------------------------

// Random factories.  The uniforms / normals are generated on the GPU by the
// Distributions.metal shaders; the host only draws a single 32-bit seed per
// call from the default generator.  Float64 keeps the host path because the
// MPS shaders do not target double precision.
Tensor rand_kernel(const std::vector<int64_t>& size, DType dtype,
                   Device device) {
  Tensor t = Tensor(size, dtype, device);
  const int64_t n = t.numel();
  if (n == 0) {
    return t;
  }
  if (dtype == DType::Float64) {
    double* data = t.data_ptr<double>();
    auto& gen = default_generator();
    uniform_real_distribution<double> dist(0.0, 1.0);
    for (int64_t i = 0; i < n; ++i) {
      data[i] = dist(&gen);
    }
    return t;
  }
  const std::string suffix = scalarToMetalTypeString(dtype);
  if (suffix != "float" && suffix != "half" && suffix != "bfloat") {
    TP_THROW(NotImplementedError, "rand does not support this dtype on MPS");
  }
  auto& gen = default_generator();
  const uint32_t seed = gen.random();
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> PSO =
          MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
              "rand_uniform_" + suffix);
      [computeEncoder setComputePipelineState:PSO];
      mtl_setArgs(computeEncoder, t, seed, static_cast<uint32_t>(n));
      mtl_dispatch1DJob(computeEncoder, PSO, static_cast<NSUInteger>(n));
      mpsStream->endKernelCoalescing();
    }
  });
  return t;
}

Tensor randn_kernel(const std::vector<int64_t>& size, DType dtype,
                    Device device) {
  Tensor t = Tensor(size, dtype, device);
  const int64_t n = t.numel();
  if (n == 0) {
    return t;
  }
  if (dtype == DType::Float64) {
    double* data = t.data_ptr<double>();
    auto& gen = default_generator();
    normal_distribution<double> dist(0.0, 1.0);
    for (int64_t i = 0; i < n; ++i) {
      data[i] = dist(&gen);
    }
    return t;
  }
  const std::string suffix = scalarToMetalTypeString(dtype);
  if (suffix != "float" && suffix != "half" && suffix != "bfloat") {
    TP_THROW(NotImplementedError, "randn does not support this dtype on MPS");
  }
  auto& gen = default_generator();
  const uint32_t seed = gen.random();
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> PSO =
          MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
              "rand_normal_" + suffix);
      [computeEncoder setComputePipelineState:PSO];
      mtl_setArgs(computeEncoder, t, seed, static_cast<uint32_t>(n));
      mtl_dispatch1DJob(computeEncoder, PSO, static_cast<NSUInteger>(n));
      mpsStream->endKernelCoalescing();
    }
  });
  return t;
}

Tensor randint_kernel(int64_t low, int64_t high,
                      const std::vector<int64_t>& size, DType dtype,
                      Device device) {
  TP_THROW_IF(high <= low, RuntimeError,
              "randint expects 'from' to be less than 'to', but got from=",
              low, " >= to=", high);
  distribution::check_random_from_to_bounds(low, high, dtype);
  Tensor t = Tensor(size, dtype, device);
  const int64_t n = t.numel();
  if (n == 0) {
    return t;
  }
  const uint64_t range =
      static_cast<uint64_t>(high) - static_cast<uint64_t>(low);
  const std::string suffix = scalarToMetalTypeString(dtype);
  const bool use_shader =
      (suffix == "long" || suffix == "int" || suffix == "short" ||
       suffix == "char" || suffix == "uchar") &&
      range <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  const int64_t base = low;
  if (use_shader) {
    auto& gen = default_generator();
    const uint32_t seed = gen.random();
    MPSStream* mpsStream = getCurrentMPSStream();
    dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
        id<MTLComputePipelineState> PSO =
            MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
                "randint_" + suffix);
        [computeEncoder setComputePipelineState:PSO];
        mtl_setArgs(computeEncoder, t, base, range, seed,
                    static_cast<uint32_t>(n));
        mtl_dispatch1DJob(computeEncoder, PSO, static_cast<NSUInteger>(n));
        mpsStream->endKernelCoalescing();
      }
    });
    return t;
  }
  // Host fallback (large int64 ranges or unsupported dtypes).
  auto& gen = default_generator();
  distribution::dispatch_dtype(dtype, [&](auto tag) {
    using scalar_t = decltype(tag);
    scalar_t* data = t.data_ptr<scalar_t>();
    uniform_int_from_to_distribution<scalar_t> dist(range, base);
    for (int64_t i = 0; i < n; ++i) {
      data[i] = dist(&gen);
    }
  });
  return t;
}

Tensor randperm_kernel(int64_t n, DType dtype, Device device) {
  if (n < 0) {
    TP_THROW(RuntimeError, "randperm(): n must be non-negative, got ", n);
  }
  if (n == 0) {
    return Tensor({0}, dtype, device);
  }
  // GPU path: draw n random keys with the rand shader, then argsort them to
  // obtain a uniform random permutation.
  namespace ops = tpx::ops;
  Tensor keys = rand_kernel({n}, DType::Float32, device);
  Tensor perm = ops::argsort(keys, 0, /*descending=*/false);
  return dtype == DType::Int64 ? perm : perm.to(dtype);
}

Tensor rand_like_kernel(const Tensor& self, DType dtype,
                        std::optional<Device> device) {
  if (dtype == DType::Undefined) dtype = self.dtype();
  Device dev = device.has_value() ? *device : self.device();
  return rand_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype,
                     dev);
}

Tensor randn_like_kernel(const Tensor& self, DType dtype,
                         std::optional<Device> device) {
  if (dtype == DType::Undefined) dtype = self.dtype();
  Device dev = device.has_value() ? *device : self.device();
  return randn_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype,
                      dev);
}

Tensor randint_like_kernel(const Tensor& self, int64_t low, int64_t high,
                           DType dtype, std::optional<Device> device) {
  if (dtype == DType::Undefined) dtype = self.dtype();
  Device dev = device.has_value() ? *device : self.device();
  return randint_kernel(low, high,
                        static_cast<std::vector<int64_t>>(self.shape()),
                        dtype, dev);
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

Tensor rand_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                 std::optional<Device> device) {
  return rand_kernel(size, resolve_factory_dtype(dtype),
                     device.value_or(Device(DeviceType::MPS)));
}

Tensor randn_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                  std::optional<Device> device) {
  return randn_kernel(size, resolve_factory_dtype(dtype),
                      device.value_or(Device(DeviceType::MPS)));
}

Tensor randint_stub(int64_t low, int64_t high,
                    const std::vector<int64_t>& size, DType dtype,
                    std::optional<Device> device) {
  return randint_kernel(low, high, size, dtype,
                        device.value_or(Device(DeviceType::MPS)));
}

Tensor randperm_stub(int64_t n, DType dtype,
                     std::optional<Device> device) {
  return randperm_kernel(n, dtype,
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
  m.impl("rand", rand_stub);
  m.impl("randn", randn_stub);
  m.impl("randint", randint_stub);
  m.impl("randperm", randperm_stub);
  m.impl("rand_like", rand_like_kernel);
  m.impl("randn_like", randn_like_kernel);
  m.impl("randint_like", randint_like_kernel);
}

} // namespace mps
} // namespace tensorplay