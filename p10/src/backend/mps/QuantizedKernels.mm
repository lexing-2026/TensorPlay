#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "OperationUtils.h"
#include "Quantizer.h"

#include "backend/mps/MPSDevice.h"

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <limits>
#include <vector>

namespace tensorplay {
namespace mps {

// ---------------------------------------------------------------------------
// Weight-only quantization (int8) and per-tensor affine quantization.
//
// The quantized factory/dequantize kernels write through the host-visible
// shared-memory mapping of the MPS buffers (valid on unified-memory Apple
// silicon), matching the CPU kernel semantics bit-for-bit.  The int8 packed
// matmul runs as an MPSGraph: C = A @ B^T * scales.
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

  const std::vector<int64_t> a_shape = {self.size(0), self.size(1)};
  const std::vector<int64_t> b_shape = {mat2.size(0), mat2.size(1)};
  const std::vector<int64_t> s_shape = {mat2.size(0)};
  const std::vector<int64_t> out_shape = {self.size(0), mat2.size(0)};
  Tensor output = Tensor(out_shape, input_dt, self.device());
  if (output.numel() == 0) {
    return output;
  }

  Tensor self_c = materializeForGraph(self);
  Tensor mat2_c = materializeForGraph(mat2);
  Tensor scales_c = materializeForGraph(scales);

  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* aShape = getMPSShape(a_shape);
    MPSShape* bShape = getMPSShape(b_shape);
    MPSShape* sShape = getMPSShape(s_shape);
    MPSShape* outShape = getMPSShape(out_shape);
    const MPSDataType a_dt = getMPSDataType(input_dt);
    const MPSDataType b_dt = getMPSDataType(mat2_c.dtype());
    const MPSDataType s_dt = getMPSDataType(scales_c.dtype());
    const MPSDataType out_dt = getMPSDataType(output.dtype());

    MPSGraphTensor* at = [graph placeholderWithShape:aShape
                                           dataType:a_dt
                                                name:@"a"];
    MPSGraphTensor* bt = [graph placeholderWithShape:bShape
                                           dataType:b_dt
                                                name:@"b"];
    MPSGraphTensor* st = [graph placeholderWithShape:sShape
                                           dataType:s_dt
                                                name:@"scales"];
    MPSGraphTensor* bCast = [graph castTensor:bt toType:a_dt name:@"cast_b"];
    MPSGraphTensor* bT = [graph transposeTensor:bCast
                                      dimension:1
                                   withDimension:0
                                            name:nil];
    MPSGraphTensor* mm = [graph matrixMultiplicationWithPrimaryTensor:at
                                                      secondaryTensor:bT
                                                                 name:nil];
    MPSGraphTensor* scalesCast = [graph castTensor:st toType:a_dt
                                              name:@"cast_scales"];
    MPSGraphTensor* result =
        [graph multiplicationWithPrimaryTensor:mm
                                 secondaryTensor:scalesCast
                                            name:nil];
    if ([result dataType] != out_dt) {
      result = [graph castTensor:result toType:out_dt name:@"cast_out"];
    }

    id<MTLBuffer> aBuf = mtlBufferOf(self_c);
    id<MTLBuffer> bBuf = mtlBufferOf(mat2_c);
    id<MTLBuffer> sBuf = mtlBufferOf(scales_c);
    id<MTLBuffer> outBuf = mtlBufferOf(output);
    MPSGraphTensorData* aData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:aBuf
                                                shape:aShape
                                             dataType:a_dt] autorelease];
    MPSGraphTensorData* bData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:bBuf
                                                shape:bShape
                                             dataType:b_dt] autorelease];
    MPSGraphTensorData* sData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:sBuf
                                                shape:sShape
                                             dataType:s_dt] autorelease];
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:outShape
                                             dataType:out_dt] autorelease];

    NSDictionary* feeds = @{at : aData, bt : bData, st : sData};
    NSDictionary* results = @{result : outData};
    getCurrentMPSStream()->executeMPSGraph(graph, feeds, results,
                                          SyncType::COMMIT_AND_WAIT);
  }
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

TENSORPLAY_LIBRARY_IMPL(MPS, QuantizedKernels) {
  m.impl("_weight_int8pack_mm", weight_int8pack_mm_kernel);
  m.impl("quantize_per_tensor", quantize_per_tensor_kernel);
  m.impl("_make_per_tensor_quantized_tensor",
         make_per_tensor_quantized_tensor_kernel);
  m.impl("dequantize.self", dequantize_self_kernel);
}

} // namespace mps
} // namespace tensorplay