#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "OperationUtils.h"

#include "backend/mps/MPSDevice.h"
#include "backend/mps/MPSStream.h"

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <optional>
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

} // namespace

// MPSGraph-based scaled dot product attention:
//   scores = (Q @ K^T) * scale + mask
//   probs  = softmax(scores, dim=-1)
//   out    = probs @ V
// Runs entirely as an MPSGraph on the GPU.  dropout_p > 0 is not fused yet.
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
  if (dropout_p != 0.0) {
    TP_THROW(NotImplementedError,
             "scaled_dot_product_attention: MPS fused path supports "
             "dropout_p == 0 only");
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

  if (enable_gqa && k.size(1) != H) {
    k = expand_kv_heads(k, H);
    v = expand_kv_heads(v, H);
  } else if (k.size(1) != H || v.size(1) != H) {
    TP_THROW(RuntimeError,
             "scaled_dot_product_attention: key/value head count must match "
             "query heads (enable GQA with enable_gqa=True)");
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

  Tensor mask =
      build_attn_mask(attn_mask, is_causal, L, S, dt, q.device());

  Tensor q_c = q.is_contiguous() ? q : q.contiguous();
  Tensor k_c = k.is_contiguous() ? k : k.contiguous();
  Tensor v_c = v.is_contiguous() ? v : v.contiguous();
  std::vector<int64_t> out_shape = {B, H, L, E};
  Tensor output = Tensor::empty(out_shape, dt, q_c.device());
  if (output.numel() == 0) {
    if (unsqueezed) {
      output = ops::squeeze(output, 1);
    }
    return output;
  }

  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* qShape = getMPSShape(static_cast<std::vector<int64_t>>(q_c.shape()));
    MPSShape* kShape = getMPSShape(static_cast<std::vector<int64_t>>(k_c.shape()));
    MPSShape* vShape = getMPSShape(static_cast<std::vector<int64_t>>(v_c.shape()));
    MPSShape* outShape = getMPSShape(out_shape);
    const MPSDataType mtl_dt = getMPSDataType(dt);

    MPSGraphTensor* qt = [graph placeholderWithShape:qShape
                                            dataType:mtl_dt
                                                 name:@"q"];
    MPSGraphTensor* kt = [graph placeholderWithShape:kShape
                                            dataType:mtl_dt
                                                 name:@"k"];
    MPSGraphTensor* vt = [graph placeholderWithShape:vShape
                                            dataType:mtl_dt
                                                 name:@"v"];
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
    MPSGraphTensor* probs = [graph softMaxWithTensor:scores
                                                axis:3
                                                name:@"probs"];
    MPSGraphTensor* result =
        [graph matrixMultiplicationWithPrimaryTensor:probs
                                       secondaryTensor:vt
                                                  name:@"out"];

    id<MTLBuffer> qBuf = mtlBufferOf(q_c);
    id<MTLBuffer> kBuf = mtlBufferOf(k_c);
    id<MTLBuffer> vBuf = mtlBufferOf(v_c);
    id<MTLBuffer> outBuf = mtlBufferOf(output);
    MPSGraphTensorData* qData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:qBuf
                                                shape:qShape
                                             dataType:mtl_dt] autorelease];
    MPSGraphTensorData* kData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:kBuf
                                                shape:kShape
                                             dataType:mtl_dt] autorelease];
    MPSGraphTensorData* vData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:vBuf
                                                shape:vShape
                                             dataType:mtl_dt] autorelease];
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:outShape
                                             dataType:mtl_dt] autorelease];

    NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
    feeds[qt] = qData;
    feeds[kt] = kData;
    feeds[vt] = vData;
    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[result] = outData;
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