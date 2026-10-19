#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "OperationUtils.h"
#include "Quantizer.h"

#include "backend/mps/MPSAllocator.h"
#include "backend/mps/MPSDevice.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSStream.h"

#include <Metal/Metal.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace tensorplay {
namespace mps {

// Defined in BlasKernels.mm.
Tensor mm_kernel(const Tensor& self, const Tensor& mat2);

// The bundled Metal shader library (Quantized.metal).
static auto& lib = MetalShaderLibrary::getBundledLibrary();

// ---------------------------------------------------------------------------
// Weight-only quantization (int4/int8, Metal shader based) and per-tensor /
// per-channel affine quantization.
//
// The int4/int8 packed matmuls run as Metal compute shaders:
// C = A @ B^T * scales, with groupwise int4 dequant.
// ---------------------------------------------------------------------------

namespace {

// Maps a quantized dtype to its underlying integer storage dtype.
DType underlying_storage(DType dtype) {
  switch (dtype) {
    case DType::QInt8:
      return DType::Int8;
    case DType::QUInt8:
      return DType::UInt8;
    case DType::QInt32:
      return DType::Int32;
    default:
      TP_THROW(TypeError, "unsupported quantized dtype ", toString(dtype));
  }
}

std::pair<int64_t, int64_t> quantized_range(DType dtype) {
  switch (dtype) {
    case DType::QInt8:
      return {-128, 127};
    case DType::QUInt8:
      return {0, 255};
    case DType::QInt32:
      return {std::numeric_limits<int32_t>::min(),
              std::numeric_limits<int32_t>::max()};
    default:
      TP_THROW(TypeError, "unsupported quantized dtype ", toString(dtype));
  }
}

// Host-side per-tensor quantization of a contiguous Float32 MPS tensor.
// `value` must already be checked against the quantized range.
template <typename code_t>
void quantize_host(const float* input, code_t* codes, int64_t numel,
                 double scale, int64_t zero_point, int64_t quant_min,
                 int64_t quant_max) {
  const float inv_scale = 1.0f / static_cast<float>(scale);
  for (int64_t i = 0; i < numel; ++i) {
    const int64_t rounded =
        static_cast<int64_t>(zero_point) +
        static_cast<int64_t>(std::nearbyint(input[i] * inv_scale));
    codes[i] = static_cast<code_t>(
        std::min<int64_t>(quant_max, std::max<int64_t>(quant_min, rounded)));
  }
}

} // namespace

Tensor weight_int8pack_mm_kernel(const Tensor& self, const Tensor& mat2,
                                  const Tensor& scales) {
  if (!self.device().is_mps() || !mat2.device().is_mps() ||
      !scales.device().is_mps()) {
    TP_THROW(RuntimeError,
             "_weight_int8pack_mm(): tensors must be on the MPS device");
  }
  if (self.dim() != 2 || mat2.dim() != 2) {
    TP_THROW(RuntimeError,
             "_weight_int8pack_mm(): expected 2-D input and weight");
  }
  if (self.size(1) != mat2.size(1)) {
    TP_THROW(RuntimeError,
             "_weight_int8pack_mm(): weight size(1) must match input size(1)");
  }
  if (scales.dim() != 1 || scales.size(0) != mat2.size(0)) {
    TP_THROW(RuntimeError,
             "_weight_int8pack_mm(): scales must be 1-D with one entry per "
             "weight row");
  }
  const DType input_dt = self.dtype();
  if (input_dt != DType::Float32 && input_dt != DType::Float16 &&
      input_dt != DType::BFloat16) {
    TP_THROW(RuntimeError,
             "_weight_int8pack_mm(): input must be float16, bfloat16 or "
             "float32");
  }
  if (mat2.dtype() != DType::Int8) {
    TP_THROW(RuntimeError,
             "_weight_int8pack_mm(): weight must be an int8 tensor");
  }
  const int64_t M = self.size(0);
  const int64_t N = mat2.size(0);
  const int64_t K = self.size(1);
  Tensor output = Tensor({M, N}, input_dt, self.device());
  if (output.numel() == 0) {
    return output;
  }
  if (N % 32 != 0 || K % 32 != 0) {
    // The packed shader only handles 32-aligned N/K; for other shapes compute
    // the equivalent (A @ B.T) * scales on the GPU with MPSGraph instead.
    Tensor B_f = mat2.to(input_dt).contiguous();
    Tensor B_t = B_f.transpose(0, 1);
    Tensor C = mm_kernel(self, B_t);
    Tensor S = scales.to(input_dt).view({1, N});
    return C * S;
  }
  Tensor A = materializeForGraph(self);
  Tensor B = materializeForGraph(mat2);
  Tensor S = materializeForGraph(scales);

  MPSStream* mpsStream = getCurrentMPSStream();
  std::array<uint32_t, 3> sizes = {
      static_cast<uint32_t>(M), static_cast<uint32_t>(K),
      static_cast<uint32_t>(N)};
  run_serialized(mpsStream->queue(), ^{
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      const std::string kernel = (M < 12)
          ? "int8pack_mv_" + scalarToMetalTypeString(input_dt)
          : "large_m_int8pack_mm_" + scalarToMetalTypeString(input_dt);
      id<MTLComputePipelineState> PSO =
          lib.getPipelineStateForFunc(kernel);
      [computeEncoder setComputePipelineState:PSO];
      mtl_setArgs(computeEncoder, A, B, S, output, sizes);
      if (M < 12) {
        [computeEncoder setThreadgroupMemoryLength:32 atIndex:0];
        [computeEncoder dispatchThreadgroups:MTLSizeMake((N + 7) / 8, M, 1)
                       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else {
        [computeEncoder setThreadgroupMemoryLength:12288 atIndex:0];
        [computeEncoder dispatchThreadgroups:MTLSizeMake((M + 31) / 32,
                                                        (N + 63) / 64, 1)
                       threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      }
      mpsStream->endKernelCoalescing();
    }
  });
  return output;
}

// Packs an int4 weight (uint8, N x K/2) into the int32 4-D format consumed by
// _weight_int4pack_mm.  Runs the `weight_to_int4pack` Metal shader.
Tensor convert_weight_to_int4pack_kernel(const Tensor& in,
                                         int64_t innerKTiles) {
  if (in.dim() != 2) {
    TP_THROW(RuntimeError,
             "_convert_weight_to_int4pack(): expect weight to be 2D tensor");
  }
  if (in.dtype() != DType::UInt8) {
    TP_THROW(RuntimeError,
             "_convert_weight_to_int4pack(): expect weight to be kByte");
  }
  if (innerKTiles != 2 && innerKTiles != 4 && innerKTiles != 8) {
    TP_THROW(RuntimeError,
             "_convert_weight_to_int4pack(): innerKTiles need to be 2, 4, or "
             "8, got ",
             innerKTiles);
  }
  const Tensor weight = in.is_contiguous() ? in : in.contiguous();
  const int64_t N = weight.size(0);
  const int64_t Kdiv2 = weight.size(1);
  const int64_t K = Kdiv2 * 2;

  Tensor weight_packed =
      Tensor::empty({N / 8, K / (16 * innerKTiles), 32, innerKTiles / 2},
                  DType::Int32, in.device());
  MPSStream* mpsStream = getCurrentMPSStream();
  std::array<uint32_t, 2> sizes = {
      static_cast<uint32_t>(N), static_cast<uint32_t>(Kdiv2 / 4)};
  run_serialized(mpsStream->queue(), ^{
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> PSO =
          lib.getPipelineStateForFunc("weight_to_int4pack");
      [computeEncoder setComputePipelineState:PSO];
      mtl_setArgs(computeEncoder, weight, weight_packed, sizes);
      [computeEncoder dispatchThreads:MTLSizeMake(N, Kdiv2 / 4, 1)
               threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      mpsStream->endKernelCoalescing();
    }
  });
  return weight_packed;
}

// Groupwise int4 packed matmul: C = A @ dequant(B)^T.  B is the int32 4-D
// packed weight; runs the `int4pack_mm_<group>_<dtype>` Metal shader.
Tensor weight_int4pack_mm_kernel(const Tensor& self, const Tensor& mat2,
                                  int64_t qGroupSize,
                                  const Tensor& qScaleAndZeros) {
  const int64_t M = self.size(0);
  const int64_t N = mat2.size(0) * 8;
  const int64_t K = self.size(1);
  const DType input_dt = self.dtype();
  if (input_dt != DType::BFloat16 && input_dt != DType::Float16 &&
      input_dt != DType::Float32) {
    TP_THROW(RuntimeError,
             "_weight_int4pack_mm(): expect A to be either 32-bit or 16-bit "
             "float tensor");
  }
  if (!self.is_contiguous() || self.dim() != 2) {
    TP_THROW(RuntimeError,
             "_weight_int4pack_mm(): expect A to be contiguous 2D tensor");
  }
  if (mat2.dtype() != DType::Int32 || !mat2.is_contiguous() ||
      mat2.dim() != 4) {
    TP_THROW(RuntimeError,
             "_weight_int4pack_mm(): expect B to be contiguous int32 4D "
             "tensor");
  }
  if (qGroupSize != 32 && qGroupSize != 64 && qGroupSize != 128 &&
      qGroupSize != 256) {
    TP_THROW(RuntimeError,
             "_weight_int4pack_mm(): expect qGroupSize to be 32, 64, 128 or "
             "256, got ",
             qGroupSize);
  }
  if (qScaleAndZeros.dim() != 3 || qScaleAndZeros.size(1) != N ||
      qScaleAndZeros.size(2) != 2) {
    TP_THROW(RuntimeError,
             "_weight_int4pack_mm(): expect qScaleAndZeros to be 3d tensor "
             "with sizes [:,",
             N, ", 2]");
  }

  Tensor output = Tensor({M, N}, input_dt, self.device());
  if (output.numel() == 0) {
    return output;
  }
  Tensor A = materializeForGraph(self);
  Tensor B = materializeForGraph(mat2);
  Tensor Q = materializeForGraph(qScaleAndZeros);

  MPSStream* mpsStream = getCurrentMPSStream();
  std::array<uint32_t, 3> sizes = {
      static_cast<uint32_t>(M), static_cast<uint32_t>(K),
      static_cast<uint32_t>(N)};
  run_serialized(mpsStream->queue(), ^{
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      const std::string kernel = "int4pack_mm_" +
          std::to_string(qGroupSize) + "_" +
          scalarToMetalTypeString(input_dt);
      id<MTLComputePipelineState> PSO = lib.getPipelineStateForFunc(kernel);
      [computeEncoder setComputePipelineState:PSO];
      mtl_setArgs(computeEncoder, A, B, Q, output, sizes);
      [computeEncoder dispatchThreads:MTLSizeMake(N / 4 * 32, 1, M)
               threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      mpsStream->endKernelCoalescing();
    }
  });
  return output;
}

Tensor quantize_per_tensor_kernel(const Tensor& self, double scale,
                                   int64_t zero_point, DType dtype) {
  if (self.dtype() != DType::Float32) {
    TP_THROW(TypeError, "quantize(): expected a Float32 tensor, got ",
             toString(self.dtype()));
  }
  if (!self.device().is_mps()) {
    TP_THROW(RuntimeError, "quantize(): input must be on the MPS device");
  }
  const Tensor input = self.is_contiguous() ? self : self.contiguous();
  const std::vector<int64_t> shape =
      static_cast<std::vector<int64_t>>(self.shape());
  const int64_t numel = input.numel();
  const auto range = quantized_range(dtype);
  const DType storage_dt = underlying_storage(dtype);
  // The loops below read and write the shared-memory mappings.
  synchronizeHostAccess();

  switch (dtype) {
    case DType::QInt8: {
      Tensor codes = Tensor::empty(shape, DType::Int8, self.device());
      quantize_host<int8_t>(input.data_ptr<float>(), codes.data_ptr<int8_t>(),
                           numel, scale, zero_point, range.first, range.second);
      return quantized::make_qtensor(
          codes, make_per_tensor_affine_quantizer(scale, zero_point, dtype),
          dtype);
    }
    case DType::QUInt8: {
      Tensor codes = Tensor::empty(shape, DType::UInt8, self.device());
      quantize_host<uint8_t>(input.data_ptr<float>(), codes.data_ptr<uint8_t>(),
                             numel, scale, zero_point, range.first,
                             range.second);
      return quantized::make_qtensor(
          codes, make_per_tensor_affine_quantizer(scale, zero_point, dtype),
          dtype);
    }
    case DType::QInt32: {
      Tensor codes = Tensor::empty(shape, DType::Int32, self.device());
      quantize_host<int32_t>(input.data_ptr<float>(),
                             codes.data_ptr<int32_t>(), numel, scale,
                             zero_point, range.first, range.second);
      return quantized::make_qtensor(
          codes, make_per_tensor_affine_quantizer(scale, zero_point, dtype),
          dtype);
    }
    default:
      TP_THROW(TypeError, "quantize(): unsupported quantized dtype ",
               toString(dtype));
  }
}

Tensor make_per_tensor_quantized_tensor_kernel(const Tensor& self,
                                               double scale,
                                               int64_t zero_point) {
  if (!self.device().is_mps()) {
    TP_THROW(RuntimeError,
             "_make_per_tensor_quantized_tensor(): input must be on the MPS "
             "device");
  }
  const DType qdt = [&]() {
    switch (self.dtype()) {
      case DType::Int8:
        return DType::QInt8;
      case DType::UInt8:
        return DType::QUInt8;
      case DType::Int32:
        return DType::QInt32;
      default:
        TP_THROW(TypeError,
                 "_make_per_tensor_quantized_tensor(): expected an "
                 "Int8/UInt8/Int32 code tensor, got ",
                 toString(self.dtype()));
    }
  }();
  return quantized::make_qtensor(
      self.clone(), make_per_tensor_affine_quantizer(scale, zero_point, qdt),
      qdt);
}

// ---------------------------------------------------------------------------
// Per-channel affine quantization: each channel along `axis` uses its own
// scale/zero_point.
// ---------------------------------------------------------------------------

namespace {

int64_t channel_of(int64_t linear_index, int64_t stride_on_axis) {
  return stride_on_axis == 0 ? 0 : linear_index / stride_on_axis;
}

} // namespace

Tensor quantize_per_channel_kernel(const Tensor& self, const Tensor& scales,
                                   const Tensor& zero_points, int64_t axis,
                                   DType dtype) {
  if (self.dtype() != DType::Float32) {
    TP_THROW(TypeError, "quantize(): expected a Float32 tensor, got ",
             toString(self.dtype()));
  }
  if (!self.device().is_mps() || !scales.device().is_mps() ||
      !zero_points.device().is_mps()) {
    TP_THROW(RuntimeError,
             "quantize_per_channel(): tensors must be on the MPS device");
  }
  if (scales.dim() != 1 || zero_points.dim() != 1 ||
      scales.numel() != zero_points.numel()) {
    TP_THROW(ValueError,
             "quantize_per_channel(): scales and zero_points must be 1-D "
             "tensors with equal sizes");
  }
  if (axis < 0) axis += self.dim();
  if (axis < 0 || axis >= self.dim()) {
    TP_THROW(ValueError, "quantize_per_channel(): axis out of range");
  }
  const int64_t channels = scales.size(0);
  if (channels != self.size(axis)) {
    TP_THROW(ValueError,
             "quantize_per_channel(): scales size must match the quantized "
             "dimension");
  }
  const auto range = quantized_range(dtype);
  const DType storage_dt = underlying_storage(dtype);
  const Tensor input = self.is_contiguous() ? self : self.contiguous();
  const int64_t numel = input.numel();
  const std::vector<int64_t> shape =
      static_cast<std::vector<int64_t>>(self.shape());

  int64_t stride_on_axis = 1;
  for (int64_t d = axis + 1; d < self.dim(); ++d) {
    stride_on_axis *= self.size(d);
  }

  Tensor sc = scales.to(DType::Float64).contiguous();
  const double* sc_ptr = sc.data_ptr<double>();
  const bool float_zp = isFloatingType(zero_points.dtype());
  Tensor zpf;
  Tensor zpi;
  const float* zpf_ptr = nullptr;
  const int64_t* zpi_ptr = nullptr;
  if (float_zp) {
    zpf = zero_points.to(DType::Float32).contiguous();
    zpf_ptr = zpf.data_ptr<float>();
  } else {
    zpi = zero_points.to(DType::Int64).contiguous();
    zpi_ptr = zpi.data_ptr<int64_t>();
  }

  // The mappings taken here and below are only sound once the conversions that
  // filled them have run.
  synchronizeHostAccess();
  const float* in = input.data_ptr<float>();
  switch (dtype) {
    case DType::QInt8: {
      Tensor codes = Tensor::empty(shape, DType::Int8, self.device());
      int8_t* out = codes.data_ptr<int8_t>();
      for (int64_t i = 0; i < numel; ++i) {
        const int64_t c = channel_of(i, stride_on_axis) % channels;
        const double inv_scale =
            1.0f / static_cast<float>(sc_ptr[c]);
        const double rounded = float_zp
            ? std::nearbyint(in[i] * inv_scale + zpf_ptr[c])
            : static_cast<double>(zpi_ptr[c]) +
                  std::nearbyint(in[i] * inv_scale);
        out[i] = static_cast<int8_t>(std::min<double>(
            static_cast<double>(range.second),
            std::max<double>(static_cast<double>(range.first), rounded)));
      }
      return quantized::make_qtensor(
          codes, make_per_channel_affine_quantizer(scales, zero_points, axis,
                                                    dtype),
          dtype);
    }
    case DType::QUInt8: {
      Tensor codes = Tensor::empty(shape, DType::UInt8, self.device());
      uint8_t* out = codes.data_ptr<uint8_t>();
      for (int64_t i = 0; i < numel; ++i) {
        const int64_t c = channel_of(i, stride_on_axis) % channels;
        const double inv_scale =
            1.0f / static_cast<float>(sc_ptr[c]);
        const double rounded = float_zp
            ? std::nearbyint(in[i] * inv_scale + zpf_ptr[c])
            : static_cast<double>(zpi_ptr[c]) +
                  std::nearbyint(in[i] * inv_scale);
        out[i] = static_cast<uint8_t>(std::min<double>(
            static_cast<double>(range.second),
            std::max<double>(static_cast<double>(range.first), rounded)));
      }
      return quantized::make_qtensor(
          codes, make_per_channel_affine_quantizer(scales, zero_points, axis,
                                                    dtype),
          dtype);
    }
    case DType::QInt32: {
      Tensor codes = Tensor::empty(shape, DType::Int32, self.device());
      int32_t* out = codes.data_ptr<int32_t>();
      for (int64_t i = 0; i < numel; ++i) {
        const int64_t c = channel_of(i, stride_on_axis) % channels;
        const double inv_scale =
            1.0f / static_cast<float>(sc_ptr[c]);
        const double rounded = float_zp
            ? std::nearbyint(in[i] * inv_scale + zpf_ptr[c])
            : static_cast<double>(zpi_ptr[c]) +
                  std::nearbyint(in[i] * inv_scale);
        out[i] = static_cast<int32_t>(std::min<double>(
            static_cast<double>(range.second),
            std::max<double>(static_cast<double>(range.first), rounded)));
      }
      return quantized::make_qtensor(
          codes, make_per_channel_affine_quantizer(scales, zero_points, axis,
                                                    dtype),
          dtype);
    }
    default:
      TP_THROW(TypeError, "quantize(): unsupported quantized dtype ",
               toString(dtype));
  }
}

Tensor make_per_channel_quantized_tensor_kernel(const Tensor& self,
                                                const Tensor& scale,
                                                const Tensor& zero_point,
                                                int64_t axis) {
  if (!self.device().is_mps() || !scale.device().is_mps() ||
      !zero_point.device().is_mps()) {
    TP_THROW(RuntimeError,
             "_make_per_channel_quantized_tensor(): input must be on the MPS "
             "device");
  }
  if (scale.dim() != 1 || zero_point.shape() != scale.shape()) {
    TP_THROW(ValueError,
             "_make_per_channel_quantized_tensor(): scales/zero_points must "
             "be 1-D with equal sizes");
  }
  if (axis < 0) axis += self.dim();
  if (axis < 0 || axis >= self.dim()) {
    TP_THROW(ValueError,
             "_make_per_channel_quantized_tensor(): axis out of range");
  }
  if (scale.size(0) != self.size(axis)) {
    TP_THROW(ValueError,
             "_make_per_channel_quantized_tensor(): scales size must match "
             "the quantized dimension");
  }
  const DType qdt = [&]() {
    switch (self.dtype()) {
      case DType::Int8:
        return DType::QInt8;
      case DType::UInt8:
        return DType::QUInt8;
      case DType::Int32:
        return DType::QInt32;
      default:
        TP_THROW(TypeError,
                 "_make_per_channel_quantized_tensor(): expected an "
                 "Int8/UInt8/Int32 code tensor, got ",
                 toString(self.dtype()));
    }
  }();
  return quantized::make_qtensor(
      self.clone(),
      make_per_channel_affine_quantizer(scale, zero_point, axis, qdt), qdt);
}

// Host-side dequantization through the shared-memory mapping.  Supports
// per-tensor and per-channel affine quantizers.
Tensor dequantize_self_kernel(const Tensor& self) {
  if (!self.device().is_mps()) {
    TP_THROW(RuntimeError, "dequantize(): input must be on the MPS device");
  }
  if (!quantized::is_quantized(self)) {
    return self.to(DType::Float32);
  }
  const auto q = quantized::quantizer_of(self);
  const Tensor input = self.is_contiguous() ? self : self.contiguous();
  const std::vector<int64_t> shape =
      static_cast<std::vector<int64_t>>(self.shape());
  Tensor out = Tensor::empty(shape, DType::Float32, self.device());
  const int64_t numel = input.numel();
  if (numel == 0) {
    return out;
  }
  synchronizeHostAccess();

  if (isPerTensorQScheme(q->qscheme())) {
    const double scale = q->scale();
    const int64_t zp = q->zero_point();
    float* o = out.data_ptr<float>();
    switch (self.dtype()) {
      case DType::QInt8: {
        const int8_t* p = input.data_ptr<int8_t>();
        for (int64_t i = 0; i < numel; ++i) {
          o[i] = static_cast<float>(
              (static_cast<double>(p[i]) - static_cast<double>(zp)) * scale);
        }
        break;
      }
      case DType::QUInt8: {
        const uint8_t* p = input.data_ptr<uint8_t>();
        for (int64_t i = 0; i < numel; ++i) {
          o[i] = static_cast<float>(
              (static_cast<double>(p[i]) - static_cast<double>(zp)) * scale);
        }
        break;
      }
      case DType::QInt32: {
        const int32_t* p = input.data_ptr<int32_t>();
        for (int64_t i = 0; i < numel; ++i) {
          o[i] = static_cast<float>(
              (static_cast<double>(p[i]) - static_cast<double>(zp)) * scale);
        }
        break;
      }
      default:
        TP_THROW(TypeError, "dequantize(): unsupported quantized dtype ",
                 toString(self.dtype()));
    }
    return out;
  }

  // Per-channel affine: each channel along `axis` has its own scale/zp.
  int64_t axis = q->axis();
  if (axis < 0) axis += self.dim();
  if (axis < 0 || axis >= self.dim()) {
    TP_THROW(ValueError, "dequantize(): quantized axis out of range");
  }
  const Tensor scales = q->scales().to(DType::Float64)
                            .to(Device(DeviceType::CPU)).contiguous();
  const Tensor zero_points = q->zero_points().to(DType::Int64)
                                  .to(Device(DeviceType::CPU)).contiguous();
  const double* scale_data = scales.data_ptr<double>();
  const int64_t* zp_data = zero_points.data_ptr<int64_t>();
  const std::vector<int64_t> strides =
      static_cast<std::vector<int64_t>>(input.strides());
  const int64_t axis_stride = (axis < static_cast<int64_t>(strides.size()))
                                  ? strides[axis]
                                  : 1;
  const int64_t axis_size = self.size(axis);

  float* o = out.data_ptr<float>();
  switch (self.dtype()) {
    case DType::QInt8: {
      const int8_t* p = input.data_ptr<int8_t>();
      for (int64_t i = 0; i < numel; ++i) {
        const int64_t c = axis_stride ? (i / axis_stride) % axis_size : 0;
        o[i] = static_cast<float>(
            (static_cast<double>(p[i]) - static_cast<double>(zp_data[c])) *
            scale_data[c]);
      }
      break;
    }
    case DType::QUInt8: {
      const uint8_t* p = input.data_ptr<uint8_t>();
      for (int64_t i = 0; i < numel; ++i) {
        const int64_t c = axis_stride ? (i / axis_stride) % axis_size : 0;
        o[i] = static_cast<float>(
            (static_cast<double>(p[i]) - static_cast<double>(zp_data[c])) *
            scale_data[c]);
      }
      break;
    }
    case DType::QInt32: {
      const int32_t* p = input.data_ptr<int32_t>();
      for (int64_t i = 0; i < numel; ++i) {
        const int64_t c = axis_stride ? (i / axis_stride) % axis_size : 0;
        o[i] = static_cast<float>(
            (static_cast<double>(p[i]) - static_cast<double>(zp_data[c])) *
            scale_data[c]);
      }
      break;
    }
    default:
      TP_THROW(TypeError, "dequantize(): unsupported quantized dtype ",
               toString(self.dtype()));
  }
  return out;
}

Tensor int_repr_kernel(const Tensor& self) {
    quantized::require_quantized(self, "int_repr");
    return quantized::strip_quantizer(self).clone();
}

TENSORPLAY_LIBRARY_IMPL(MPS, QuantizedKernels) {
  m.impl("_weight_int8pack_mm", weight_int8pack_mm_kernel);
  m.impl("int_repr", int_repr_kernel);
  m.impl("_convert_weight_to_int4pack", convert_weight_to_int4pack_kernel);
  m.impl("_weight_int4pack_mm", weight_int4pack_mm_kernel);
  m.impl("quantize_per_tensor", quantize_per_tensor_kernel);
  m.impl("quantize_per_channel", quantize_per_channel_kernel);
  m.impl("_make_per_tensor_quantized_tensor",
         make_per_tensor_quantized_tensor_kernel);
  m.impl("_make_per_channel_quantized_tensor",
         make_per_channel_quantized_tensor_kernel);
  m.impl("dequantize.self", dequantize_self_kernel);
}

} // namespace mps
} // namespace tensorplay