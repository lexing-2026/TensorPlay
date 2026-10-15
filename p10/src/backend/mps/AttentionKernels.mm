#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "OperationUtils.h"

#include "backend/mps/MPSDevice.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSStream.h"
#include "backend/mps/kernels/common.h"

#include <Metal/Metal.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "tensorplay/ops/TPXOpsGenerated.h"

namespace tensorplay {
namespace mps {

namespace ops = tpx::ops;

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

// Expands K/V heads to match Q heads for grouped-query attention by tiling
// along the head dimension.
Tensor expand_kv_heads(const Tensor& kv, int64_t q_heads) {
  const int64_t kv_heads = kv.size(1);
  if (kv_heads == q_heads) {
    return kv;
  }
  if (q_heads % kv_heads != 0) {
    TP_THROW(RuntimeError,
             "scaled_dot_product_attention: GQA requires q_heads % kv_heads "
             "== 0");
  }
  const int64_t factor = q_heads / kv_heads;
  std::vector<int64_t> repeats(kv.dim(), 1);
  repeats[1] = factor;
  return ops::repeat(kv, repeats);
}

// Builds an additive mask tensor for the attention scores.
//  - bool masks: True attends, False -> -inf.
//  - float masks are added directly.
//  - is_causal adds an upper-triangular -inf mask (top-left aligned).
Tensor build_attn_mask(const std::optional<Tensor>& attn_mask, bool is_causal,
                       int64_t L, int64_t S, DType dtype,
                       const Device& device) {
  Tensor mask;
  if (is_causal) {
    Tensor row = ops::arange(Scalar(L), DType::Int32, device);       // [L]
    Tensor col = ops::arange(Scalar(S), DType::Int32, device);       // [S]
    Tensor col_v = ops::view(col, {1, S});
    Tensor row_v = ops::view(row, {L, 1});
    Tensor causal = ops::ge(col_v, row_v);                    // [L, S]
    mask = ops::where(causal, Tensor::full({}, 0.0, dtype, device),
                      Tensor::full({}, kNegInf, dtype, device));
  }
  if (attn_mask.has_value() && attn_mask->defined()) {
    Tensor am = *attn_mask;
    if (am.dtype() == DType::Bool) {
      Tensor additive = ops::where(
          am, Tensor::full({}, 0.0, dtype, am.device()),
          Tensor::full({}, kNegInf, dtype, am.device()));
      mask = mask.defined() ? ops::add(mask, additive.to(dtype)) : additive;
    } else {
      Tensor cast = am.dtype() == dtype ? am : am.to(dtype);
      mask = mask.defined() ? ops::add(mask, cast) : cast;
    }
  }
  return mask.defined() ? mask : Tensor();
}

// One-pass decode-phase attention shader (`sdpa_vector_*`).  Only used for
// short query sequences with no external mask; returns nullopt to let the
// caller fall back to the MPSGraph path.
std::optional<Tensor> sdpa_decode_shader(const Tensor& q_in,
                                         const Tensor& k_in,
                                         const Tensor& v_in, bool is_causal,
                                         double scale_factor,
                                         int64_t gqa_factor,
                                         bool unsqueezed) {
  const int64_t B = q_in.size(0);
  const int64_t H = q_in.size(1);
  const int64_t L = q_in.size(2);
  const int64_t E = q_in.size(3);
  const int64_t S = k_in.size(2);
  if (L > 32) {
    return std::nullopt;
  }
  if (E != 64 && E != 96 && E != 128 && E != 256) {
    return std::nullopt;
  }
  const DType dt = q_in.dtype();
  const std::string dtype = scalarToMetalTypeString(dt);
  if (dtype != "float" && dtype != "half" && dtype != "bfloat") {
    return std::nullopt;
  }
  if (v_in.size(1) != k_in.size(1)) {
    return std::nullopt;
  }

  Tensor q = materializeForGraph(q_in);
  Tensor k = materializeForGraph(k_in);
  Tensor v = materializeForGraph(v_in);

  std::vector<int64_t> out_shape = {B, H, L, E};
  Tensor output = Tensor::empty(out_shape, dt, q.device());
  if (output.numel() == 0) {
    return output;
  }

  const std::string kernel = "sdpa_vector_" + dtype + "_" +
      std::to_string(E) + "_" + std::to_string(E) +
      (is_causal ? "_masknone_causal" : "_masknone");
  const uint32_t N = static_cast<uint32_t>(S);
  const uint32_t gqa_u = static_cast<uint32_t>(gqa_factor);
  const p10::metal::vec3<uint32_t> head_strides{
      static_cast<uint32_t>(q.stride(1)),
      static_cast<uint32_t>(k.stride(1)),
      static_cast<uint32_t>(v.stride(1))};
  const p10::metal::vec3<uint32_t> seq_strides{
      static_cast<uint32_t>(q.stride(2)),
      static_cast<uint32_t>(k.stride(2)),
      static_cast<uint32_t>(v.stride(2))};
  const std::array<uint32_t, 4> mask_strides{0, 0, 0, 0};
  const std::array<uint32_t, 4> batch_strides{
      static_cast<uint32_t>(q.stride(0)),
      static_cast<uint32_t>(k.stride(0)),
      static_cast<uint32_t>(v.stride(0)),
      static_cast<uint32_t>(H)};
  const float scale_f = static_cast<float>(scale_factor);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> PSO =
          MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
              kernel);
      [computeEncoder setComputePipelineState:PSO];
      unsigned idx = 0;
      [computeEncoder setBuffer:mtlBufferOf(q) offset:0 atIndex:idx++];
      [computeEncoder setBuffer:mtlBufferOf(k) offset:0 atIndex:idx++];
      [computeEncoder setBuffer:mtlBufferOf(v) offset:0 atIndex:idx++];
      [computeEncoder setBuffer:mtlBufferOf(output) offset:0 atIndex:idx++];
      [computeEncoder setBytes:&gqa_u length:sizeof(uint32_t) atIndex:idx++];
      [computeEncoder setBytes:&N length:sizeof(uint32_t) atIndex:idx++];
      [computeEncoder setBytes:&head_strides length:sizeof(head_strides)
                        atIndex:idx++];
      [computeEncoder setBytes:&seq_strides length:sizeof(seq_strides)
                        atIndex:idx++];
      [computeEncoder setBytes:&scale_f length:sizeof(float) atIndex:idx++];
      [computeEncoder setBuffer:nil offset:0 atIndex:idx++];
      [computeEncoder setBytes:&mask_strides length:sizeof(mask_strides)
                        atIndex:idx++];
      [computeEncoder setBytes:&batch_strides length:sizeof(batch_strides)
                        atIndex:idx++];

      MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(B * H),
                                 static_cast<NSUInteger>(L), 1);
      MTLSize threads = MTLSizeMake(32, 1, 1);
      [computeEncoder dispatchThreadgroups:grid
                     threadsPerThreadgroup:threads];
      mpsStream->endKernelCoalescing();
    }
  });

  if (unsqueezed) {
    output = ops::squeeze(output, 1);
  }
  return output;
}

// Host-side counterparts of the shader parameter structs (PrefillAttention.h).
struct PrefillAttnParamsHost {
  int B;
  int H;
  int D;
  int qL;
  int kL;
  int gqa_factor;
  float scale;
  float softcapping;
  int64_t Q_strides[3];
  int64_t K_strides[3];
  int64_t V_strides[3];
  int64_t O_strides[3];
};

struct PrefillAttnMaskParamsHost {
  int64_t M_strides[4];
};

struct PrefillBlockShape {
  int BQ;
  int BK;
  int WM;
  int WN;
};

PrefillBlockShape prefill_block_shape_for_head_dim(int64_t head_dim) {
  switch (head_dim) {
    case 256:
      return {16, 8, 2, 1};
    case 128:
      return {32, 16, 4, 1};
    case 96:
    case 80:
    case 72:
    case 64:
    case 32:
      return {32, 32, 4, 1};
    default:
      TP_THROW(RuntimeError,
               "scaled_dot_product_attention: unsupported head_dim for "
               "prefill attention: ",
               head_dim);
  }
}

bool prefill_attention_supports_head_dim(int64_t head_dim) {
  switch (head_dim) {
    case 32:
    case 64:
    case 72:
    case 80:
    case 96:
    case 128:
    case 256:
      return true;
    default:
      return false;
  }
}

std::string prefill_dtype_str(DType dt) {
  switch (dt) {
    case DType::Float32:
      return "float32";
    case DType::Float16:
      return "float16";
    case DType::BFloat16:
      return "bfloat16";
    default:
      TP_THROW(RuntimeError, "unsupported dtype for prefill attention");
  }
}

// Flash-attention prefill (long query).  Returns nullopt when the inputs do
// not fit the shader so the caller falls back to the MPSGraph path.
std::optional<Tensor> sdpa_prefill_shader(const Tensor& q_in,
                                          const Tensor& k_in,
                                          const Tensor& v_in,
                                          const std::optional<Tensor>& mask_,
                                          bool is_causal,
                                          double scale_factor,
                                          int64_t gqa_factor,
                                          bool unsqueezed) {
  const int64_t B = q_in.size(0);
  const int64_t H = q_in.size(1);
  const int64_t qL = q_in.size(2);
  const int64_t kL = k_in.size(2);
  const int64_t D = q_in.size(3);
  const DType dt = q_in.dtype();
  if (!prefill_attention_supports_head_dim(D)) {
    return std::nullopt;
  }
  if (dt != DType::Float32 && dt != DType::Float16 &&
      dt != DType::BFloat16) {
    return std::nullopt;
  }
  if (v_in.size(1) != k_in.size(1)) {
    return std::nullopt;
  }

  Tensor q = materializeForGraph(q_in);
  Tensor k = materializeForGraph(k_in);
  Tensor v = materializeForGraph(v_in);
  if (q.stride(-1) != 1 || k.stride(-1) != 1 || v.stride(-1) != 1) {
    return std::nullopt;
  }

  const bool has_mask = mask_.has_value() && mask_->defined();
  Tensor mask;
  if (has_mask) {
    Tensor m = *mask_;
    if (m.dtype() == DType::Bool || m.dtype() == dt) {
      // Accept [L,S] and broadcast to [1,1,L,S].
      if (m.dim() == 2) {
        m = ops::view(m, {1, 1, m.size(0), m.size(1)});
      }
      if (m.dim() != 4) {
        return std::nullopt;
      }
      mask = m.is_contiguous() ? m : m.contiguous();
    } else {
      return std::nullopt;
    }
  }

  const PrefillBlockShape shape = prefill_block_shape_for_head_dim(D);
  const std::string dtype_str = prefill_dtype_str(dt);
  std::string mask_dtype_str = dtype_str;
  if (has_mask && mask.dtype() == DType::Bool) {
    mask_dtype_str = "bool_";
  }
  const std::string kname = "prefill_attention_" + dtype_str + "_bq" +
      std::to_string(shape.BQ) + "_bk" + std::to_string(shape.BK) + "_bd" +
      std::to_string(D) + "_wm" + std::to_string(shape.WM) + "_wn" +
      std::to_string(shape.WN) + "_hm" + (has_mask ? "1" : "0") + "_dc" +
      (is_causal ? "1" : "0") + "_mask" + mask_dtype_str;

  std::vector<int64_t> out_shape = static_cast<std::vector<int64_t>>(q.shape());
  Tensor output = Tensor::empty(out_shape, dt, q.device());
  if (output.numel() == 0) {
    return output;
  }

  PrefillAttnParamsHost params{};
  params.B = static_cast<int>(B);
  params.H = static_cast<int>(H);
  params.D = static_cast<int>(D);
  params.qL = static_cast<int>(qL);
  params.kL = static_cast<int>(kL);
  params.gqa_factor = static_cast<int>(gqa_factor);
  params.scale = static_cast<float>(scale_factor);
  params.softcapping = 1.0f;
  params.Q_strides[0] = q.stride(0);
  params.Q_strides[1] = q.stride(1);
  params.Q_strides[2] = q.stride(2);
  params.K_strides[0] = k.stride(0);
  params.K_strides[1] = k.stride(1);
  params.K_strides[2] = k.stride(2);
  params.V_strides[0] = v.stride(0);
  params.V_strides[1] = v.stride(1);
  params.V_strides[2] = v.stride(2);
  params.O_strides[0] = output.stride(0);
  params.O_strides[1] = output.stride(1);
  params.O_strides[2] = output.stride(2);

  PrefillAttnMaskParamsHost mask_params{};
  if (has_mask) {
    mask_params.M_strides[0] = mask.stride(0);
    mask_params.M_strides[1] = mask.stride(1);
    mask_params.M_strides[2] = mask.stride(2);
    mask_params.M_strides[3] = mask.stride(3);
  }

  const int64_t nQ = (qL + shape.BQ - 1) / shape.BQ;
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> PSO =
          MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
              kname);
      [computeEncoder setComputePipelineState:PSO];
      mtl_setArgs(computeEncoder, q, k, v, output, params);
      if (has_mask) {
        mtl_setArgs<5>(computeEncoder, mask_params, mask);
      } else {
        // Dummy bindings to satisfy buffer slots 5/6 when HAS_MASK=false.
        mtl_setArgs<5>(computeEncoder, uint32_t(0), uint32_t(0));
      }
      MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(nQ),
                                     static_cast<NSUInteger>(H),
                                     static_cast<NSUInteger>(B));
      MTLSize threadgroupSize =
          MTLSizeMake(32, static_cast<NSUInteger>(shape.WM),
                      static_cast<NSUInteger>(shape.WN));
      [computeEncoder dispatchThreadgroups:gridSize
                     threadsPerThreadgroup:threadgroupSize];
      mpsStream->endKernelCoalescing();
    }
  });

  if (unsqueezed) {
    output = ops::squeeze(output, 1);
  }
  return output;
}

} // namespace

// MPSGraph-based scaled dot product attention:
//   probs = softmax((Q @ K^T) * scale + mask)
//   out   = probs @ V
// The attention probabilities are produced by the first MPSGraph; when
// dropout_p > 0 they pass through the GPU dropout shader before the second
// MPSGraph multiplies them with V.
Tensor scaled_dot_product_attention_kernel(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const std::optional<Tensor>& attn_mask, double dropout_p, bool is_causal,
    const std::optional<double>& scale, bool enable_gqa) {
  if (!query.device().is_mps() || !key.device().is_mps() ||
      !value.device().is_mps()) {
    TP_THROW(RuntimeError,
             "scaled_dot_product_attention: tensors must be on the MPS "
             "device");
  }

  Tensor q = query;
  Tensor k = key;
  Tensor v = value;
  const int64_t q_ndim = q.dim();
  bool unsqueezed = false;
  if (q_ndim == 3) {
    q = ops::unsqueeze(q, 1);
    k = ops::unsqueeze(k, 1);
    v = ops::unsqueeze(v, 1);
    unsqueezed = true;
  } else if (q_ndim != 4) {
    TP_THROW(RuntimeError,
             "scaled_dot_product_attention: expected 3D or 4D query");
  }

  const int64_t B = q.size(0);
  const int64_t H = q.size(1);
  const int64_t L = q.size(2);
  const int64_t E = q.size(3);
  const int64_t S = k.size(2);
  if (k.size(3) != E || v.size(3) != E) {
    TP_THROW(RuntimeError,
             "scaled_dot_product_attention: head dims must match");
  }
  if (v.size(1) != k.size(1)) {
    TP_THROW(RuntimeError,
             "scaled_dot_product_attention: key/value head counts must match");
  }

  const int64_t kH = k.size(1);
  int64_t gqa_factor = 1;
  if (kH != H) {
    if (!enable_gqa) {
      TP_THROW(RuntimeError,
               "scaled_dot_product_attention: key/value head count must match "
               "query heads (enable GQA with enable_gqa=True)");
    }
    if (H % kH != 0) {
      TP_THROW(RuntimeError,
               "scaled_dot_product_attention: GQA requires q_heads % "
               "kv_heads == 0");
    }
    gqa_factor = H / kH;
  }

  const DType dt = q.dtype();
  if (dt != DType::Float32 && dt != DType::Float16 &&
      dt != DType::BFloat16) {
    TP_THROW(RuntimeError,
             "scaled_dot_product_attention: MPS supports float32/float16/"
             "bfloat16");
  }

  const double scale_factor =
      scale.has_value() ? *scale : 1.0 / std::sqrt(static_cast<double>(E));

  // Fast decode path: short query, no external mask -> one-pass shader.
  if ((!attn_mask.has_value() || !attn_mask->defined()) && dropout_p == 0) {
    auto decoded = sdpa_decode_shader(q, k, v, is_causal, scale_factor,
                                      gqa_factor, unsqueezed);
    if (decoded.has_value()) {
      return *decoded;
    }
  }

  // Flash-attention prefill path (long query / any mask).
  {
    auto prefilled = sdpa_prefill_shader(q, k, v, attn_mask, is_causal,
                                         scale_factor, gqa_factor,
                                         unsqueezed);
    if (prefilled.has_value()) {
      return *prefilled;
    }
  }

  // MPSGraph path requires matched head counts; expand K/V for GQA.
  if (gqa_factor > 1) {
    k = expand_kv_heads(k, H);
    v = expand_kv_heads(v, H);
  }
  Tensor mask =
      build_attn_mask(attn_mask, is_causal, L, S, dt, q.device());

  Tensor q_c = q.is_contiguous() ? q : q.contiguous();
  Tensor k_c = k.is_contiguous() ? k : k.contiguous();
  Tensor v_c = v.is_contiguous() ? v : v.contiguous();

  std::vector<int64_t> probs_shape = {B, H, L, S};
  std::vector<int64_t> out_shape = {B, H, L, E};
  Tensor output = Tensor::empty(out_shape, dt, q_c.device());
  if (output.numel() == 0) {
    if (unsqueezed) {
      output = ops::squeeze(output, 1);
    }
    return output;
  }

  // Pass 1: attention probabilities.
  Tensor probs = Tensor::empty(probs_shape, dt, q_c.device());
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* qShape = getMPSShape(static_cast<std::vector<int64_t>>(q_c.shape()));
    MPSShape* kShape = getMPSShape(static_cast<std::vector<int64_t>>(k_c.shape()));
    MPSShape* probsShape = getMPSShape(probs_shape);
    const MPSDataType mtl_dt = getMPSDataType(dt);

    MPSGraphTensor* qt = [graph placeholderWithShape:qShape
                                            dataType:mtl_dt
                                                 name:@"q"];
    MPSGraphTensor* kt = [graph placeholderWithShape:kShape
                                            dataType:mtl_dt
                                                 name:@"k"];
    MPSGraphTensor* kT = [graph transposeTensor:kt
                                      dimension:2
                                   withDimension:3
                                            name:@"kT"];
    MPSGraphTensor* scores =
        [graph matrixMultiplicationWithPrimaryTensor:qt
                                       secondaryTensor:kT
                                                  name:@"scores"];
    MPSGraphTensor* scaleTensor =
        [graph constantWithScalar:scale_factor dataType:mtl_dt];
    scores = [graph multiplicationWithPrimaryTensor:scores
                                     secondaryTensor:scaleTensor
                                                name:@"scaled"];
    if (mask.defined()) {
      MPSGraphTensor* mt = [graph placeholderWithShape:getMPSShape(
          static_cast<std::vector<int64_t>>(mask.shape()))
                                             dataType:mtl_dt
                                                  name:@"mask"];
      scores = [graph additionWithPrimaryTensor:scores
                                 secondaryTensor:mt
                                            name:@"masked"];
    }
    MPSGraphTensor* probsT = [graph softMaxWithTensor:scores
                                                 axis:3
                                                 name:@"probs"];

    id<MTLBuffer> qBuf = mtlBufferOf(q_c);
    id<MTLBuffer> kBuf = mtlBufferOf(k_c);
    id<MTLBuffer> probsBuf = mtlBufferOf(probs);
    MPSGraphTensorData* qData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:qBuf
                                                shape:qShape
                                             dataType:mtl_dt] autorelease];
    MPSGraphTensorData* kData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:kBuf
                                                shape:kShape
                                             dataType:mtl_dt] autorelease];
    MPSGraphTensorData* probsData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:probsBuf
                                                shape:probsShape
                                             dataType:mtl_dt] autorelease];

    NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
    feeds[qt] = qData;
    feeds[kt] = kData;
    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[probsT] = probsData;
    if (mask.defined()) {
      Tensor mask_c = mask.is_contiguous() ? mask : mask.contiguous();
      id<MTLBuffer> maskBuf = mtlBufferOf(mask_c);
      MPSGraphTensorData* maskData =
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:maskBuf
                                                  shape:getMPSShape(
                                                      static_cast<std::vector<int64_t>>(mask_c.shape()))
                                               dataType:mtl_dt] autorelease];
      feeds[mt] = maskData;
    }
    getCurrentMPSStream()->executeMPSGraph(graph, feeds, results,
                                      SyncType::COMMIT_AND_WAIT);
  }

  Tensor probs_in = probs;
  if (dropout_p > 0) {
    auto [dropped, dropout_mask] = ops::native_dropout(probs, dropout_p, true);
    (void)dropout_mask;
    probs_in = dropped;
  }

  // Pass 2: output = probs @ V.
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* pShape = getMPSShape(static_cast<std::vector<int64_t>>(probs_in.shape()));
    MPSShape* vShape = getMPSShape(static_cast<std::vector<int64_t>>(v_c.shape()));
    MPSShape* outShape = getMPSShape(out_shape);
    const MPSDataType mtl_dt = getMPSDataType(dt);

    MPSGraphTensor* pt = [graph placeholderWithShape:pShape
                                            dataType:mtl_dt
                                                 name:@"probs"];
    MPSGraphTensor* vt = [graph placeholderWithShape:vShape
                                            dataType:mtl_dt
                                                 name:@"v"];
    MPSGraphTensor* result =
        [graph matrixMultiplicationWithPrimaryTensor:pt
                                       secondaryTensor:vt
                                                  name:@"out"];

    id<MTLBuffer> pBuf = mtlBufferOf(probs_in);
    id<MTLBuffer> vBuf = mtlBufferOf(v_c);
    id<MTLBuffer> outBuf = mtlBufferOf(output);
    MPSGraphTensorData* pData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:pBuf
                                                shape:pShape
                                             dataType:mtl_dt] autorelease];
    MPSGraphTensorData* vData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:vBuf
                                                shape:vShape
                                             dataType:mtl_dt] autorelease];
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:outShape
                                             dataType:mtl_dt] autorelease];

    NSDictionary* feeds = @{pt : pData, vt : vData};
    NSDictionary* results = @{result : outData};
    getCurrentMPSStream()->executeMPSGraph(graph, feeds, results,
                                      SyncType::COMMIT_AND_WAIT);
  }

  if (unsqueezed) {
    output = ops::squeeze(output, 1);
  }
  return output;
}

TENSORPLAY_LIBRARY_IMPL(MPS, AttentionKernels) {
  m.impl("scaled_dot_product_attention", scaled_dot_product_attention_kernel);
}

} // namespace mps
} // namespace tensorplay