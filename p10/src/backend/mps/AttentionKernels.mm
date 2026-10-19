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

// Repeats each K/V head for its group of query heads.
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
  return ops::repeat_interleave(kv, factor, 1, std::nullopt);
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
    Tensor causal = ops::le(col_v, row_v);                    // [L, S]
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
                                         const Tensor& v_in,
                                         const std::optional<Tensor>& attn_mask,
                                         bool is_causal,
                                         double scale_factor,
                                         int64_t gqa_factor,
                                         bool unsqueezed) {
  const int64_t B = q_in.size(0);
  const int64_t H = q_in.size(1);
  const int64_t L = q_in.size(2);
  const int64_t E = q_in.size(3);
  const int64_t S = k_in.size(2);
  if (L > 8) {
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

  bool has_mask = false;
  bool mask_is_bool = false;
  Tensor mask;
  if (attn_mask.has_value() && attn_mask->defined()) {
    Tensor m = *attn_mask;
    if (m.dtype() == DType::Bool) {
      mask_is_bool = true;
    } else if (m.dtype() != dt) {
      return std::nullopt;
    }
    if (m.dim() == 2) {
      m = ops::view(m, {1, 1, m.size(0), m.size(1)});
    } else if (m.dim() != 4) {
      return std::nullopt;
    }
    mask = m.is_contiguous() ? m : m.contiguous();
    has_mask = true;
  }

  std::vector<int64_t> out_shape = {B, H, L, E};
  Tensor output = Tensor::empty(out_shape, dt, q.device());
  if (output.numel() == 0) {
    return output;
  }

  const std::string kernel =
      "sdpa_vector_" + dtype + "_" + std::to_string(E) + "_" +
      std::to_string(E) +
      (has_mask
           ? (mask_is_bool ? "_maskbool" : "_mask" + dtype)
           : (is_causal ? "_masknone_causal" : "_masknone"));
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
  std::array<uint32_t, 4> mask_strides{0, 0, 0, 0};
  if (has_mask) {
    mask_strides = {static_cast<uint32_t>(mask.size(3) > 1 ? mask.stride(3) : 0),
                    static_cast<uint32_t>(mask.size(2) > 1 ? mask.stride(2) : 0),
                    static_cast<uint32_t>(mask.size(1) > 1 ? mask.stride(1) : 0),
                    static_cast<uint32_t>(mask.size(0) > 1 ? mask.stride(0) : 0)};
  }
  const std::array<uint32_t, 4> batch_strides{
      static_cast<uint32_t>(q.stride(0)),
      static_cast<uint32_t>(k.stride(0)),
      static_cast<uint32_t>(v.stride(0)),
      static_cast<uint32_t>(H)};
  const float scale_f = static_cast<float>(scale_factor);

  MPSStream* mpsStream = getCurrentMPSStream();
  run_serialized(mpsStream->queue(), ^() {
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
      if (has_mask) {
        [computeEncoder setBuffer:mtlBufferOf(mask) offset:0 atIndex:idx++];
      } else {
        [computeEncoder setBuffer:nil offset:0 atIndex:idx++];
      }
      [computeEncoder setBytes:&mask_strides length:sizeof(mask_strides)
                        atIndex:idx++];
      [computeEncoder setBytes:&batch_strides length:sizeof(batch_strides)
                        atIndex:idx++];

      MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(B * H),
                                 static_cast<NSUInteger>(L), 1);
      MTLSize threads = MTLSizeMake(1024, 1, 1);
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

// Two-pass decode attention for long key sequences: pass 1 splits the K
// loop into 32 blocks writing partial sums/maxs; pass 2 aggregates them.
std::optional<Tensor> sdpa_decode_2pass_shader(
    const Tensor& q_in, const Tensor& k_in, const Tensor& v_in,
    const std::optional<Tensor>& attn_mask, bool is_causal,
    double scale_factor, int64_t gqa_factor, bool unsqueezed) {
  const int64_t B = q_in.size(0);
  const int64_t H = q_in.size(1);
  const int64_t L = q_in.size(2);
  const int64_t E = q_in.size(3);
  const int64_t S = k_in.size(2);
  if (L > 8) {
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

  bool has_mask = false;
  bool mask_is_bool = false;
  Tensor mask;
  if (attn_mask.has_value() && attn_mask->defined()) {
    Tensor m = *attn_mask;
    if (m.dtype() == DType::Bool) {
      mask_is_bool = true;
    } else if (m.dtype() != dt) {
      return std::nullopt;
    }
    if (m.dim() == 2) {
      m = ops::view(m, {1, 1, m.size(0), m.size(1)});
    } else if (m.dim() != 4) {
      return std::nullopt;
    }
    mask = m.is_contiguous() ? m : m.contiguous();
    has_mask = true;
  }

  constexpr int64_t kBlocks = 32;
  std::vector<int64_t> out_shape = {B, H, L, E};
  std::vector<int64_t> partial_shape = {B, H, L, kBlocks, E};
  std::vector<int64_t> stats_shape = {B, H, L, kBlocks};
  Tensor output = Tensor::empty(out_shape, dt, q.device());
  if (output.numel() == 0) {
    return output;
  }
  Tensor intermediate = Tensor::empty(partial_shape, dt, q.device());
  Tensor sums = Tensor::empty(stats_shape, DType::Float32, q.device());
  Tensor maxs = Tensor::empty(stats_shape, DType::Float32, q.device());

  const std::string pass1_kernel =
      "sdpa_vector_2pass_1_" + dtype + "_" + std::to_string(E) + "_" +
      std::to_string(E) +
      (has_mask
           ? (mask_is_bool ? "_maskbool" : "_mask" + dtype)
           : (is_causal ? "_masknone_causal" : "_masknone"));
  const std::string pass2_kernel =
      "sdpa_vector_2pass_2_" + dtype + "_" + std::to_string(E);

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
  std::array<uint32_t, 4> mask_strides{0, 0, 0, 0};
  if (has_mask) {
    mask_strides = {static_cast<uint32_t>(mask.size(3) > 1 ? mask.stride(3) : 0),
                    static_cast<uint32_t>(mask.size(2) > 1 ? mask.stride(2) : 0),
                    static_cast<uint32_t>(mask.size(1) > 1 ? mask.stride(1) : 0),
                    static_cast<uint32_t>(mask.size(0) > 1 ? mask.stride(0) : 0)};
  }
  const std::array<uint32_t, 4> batch_strides{
      static_cast<uint32_t>(q.stride(0)),
      static_cast<uint32_t>(k.stride(0)),
      static_cast<uint32_t>(v.stride(0)),
      static_cast<uint32_t>(H)};
  const float scale_f = static_cast<float>(scale_factor);

  MPSStream* mpsStream = getCurrentMPSStream();
  run_serialized(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> pass1PSO =
          MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
              pass1_kernel);
      [computeEncoder setComputePipelineState:pass1PSO];
      unsigned idx = 0;
      [computeEncoder setBuffer:mtlBufferOf(q) offset:0 atIndex:idx++];
      [computeEncoder setBuffer:mtlBufferOf(k) offset:0 atIndex:idx++];
      [computeEncoder setBuffer:mtlBufferOf(v) offset:0 atIndex:idx++];
      [computeEncoder setBuffer:mtlBufferOf(intermediate) offset:0 atIndex:idx++];
      [computeEncoder setBuffer:mtlBufferOf(sums) offset:0 atIndex:idx++];
      [computeEncoder setBuffer:mtlBufferOf(maxs) offset:0 atIndex:idx++];
      [computeEncoder setBytes:&gqa_u length:sizeof(uint32_t) atIndex:idx++];
      [computeEncoder setBytes:&N length:sizeof(uint32_t) atIndex:idx++];
      [computeEncoder setBytes:&head_strides length:sizeof(head_strides)
                        atIndex:idx++];
      [computeEncoder setBytes:&seq_strides length:sizeof(seq_strides)
                        atIndex:idx++];
      [computeEncoder setBytes:&scale_f length:sizeof(float) atIndex:idx++];
      if (has_mask) {
        [computeEncoder setBuffer:mtlBufferOf(mask) offset:0 atIndex:idx++];
      } else {
        [computeEncoder setBuffer:nil offset:0 atIndex:idx++];
      }
      [computeEncoder setBytes:&mask_strides length:sizeof(mask_strides)
                        atIndex:idx++];
      [computeEncoder setBytes:&batch_strides length:sizeof(batch_strides)
                        atIndex:idx++];

      MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(B * H),
                                 static_cast<NSUInteger>(L),
                                 static_cast<NSUInteger>(kBlocks));
      MTLSize group = MTLSizeMake(256, 1, 1);
      [computeEncoder dispatchThreadgroups:grid
                     threadsPerThreadgroup:group];

      id<MTLComputePipelineState> pass2PSO =
          MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
              pass2_kernel);
      [computeEncoder setComputePipelineState:pass2PSO];
      [computeEncoder setBuffer:mtlBufferOf(intermediate) offset:0 atIndex:0];
      [computeEncoder setBuffer:mtlBufferOf(sums) offset:0 atIndex:1];
      [computeEncoder setBuffer:mtlBufferOf(maxs) offset:0 atIndex:2];
      [computeEncoder setBuffer:mtlBufferOf(output) offset:0 atIndex:3];
      MTLSize grid2 = MTLSizeMake(static_cast<NSUInteger>(B * H),
                                  static_cast<NSUInteger>(L), 1);
      MTLSize group2 = MTLSizeMake(1024, 1, 1);
      [computeEncoder dispatchThreadgroups:grid2
                     threadsPerThreadgroup:group2];
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
  run_serialized(mpsStream->queue(), ^() {
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
    std::optional<double> scale, bool enable_gqa) {
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

  // Kernel path selection:
  //  - qL <= 8, compatible mask, no causal+mask combo -> decode shaders
  //    (two-pass for long K, one-pass otherwise)
  //  - qL > 8, supported head dim, compatible mask -> prefill flash
  //  - otherwise MPSGraph
  const bool mask_compatible =
      !attn_mask.has_value() || !attn_mask->defined() ||
      attn_mask->dtype() == DType::Bool || attn_mask->dtype() == dt;
  const bool vector_head_supported =
      E == 64 || E == 96 || E == 128 || E == 256;
  const bool causal_with_mask =
      is_causal && attn_mask.has_value() && attn_mask->defined();
  const bool supports_vector =
      L <= 8 && mask_compatible && vector_head_supported && !causal_with_mask;
  if (supports_vector && dropout_p == 0) {
    const bool use_2pass = S >= 1024 || (kH < H && S >= 4096);
    auto decoded = use_2pass
        ? sdpa_decode_2pass_shader(q, k, v, attn_mask, is_causal,
                                scale_factor, gqa_factor, unsqueezed)
        : sdpa_decode_shader(q, k, v, attn_mask, is_causal,
                             scale_factor, gqa_factor, unsqueezed);
    if (decoded.has_value()) {
      return *decoded;
    }
  }

  // Flash-attention prefill path (long query / any mask).
  if (L > 8) {
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
  // The two products are described with a single leading dimension: the runtime
  // only compiles a batched matrix product over one batch axis, and merging the
  // batch and head axes leaves the element order untouched.
  const int64_t BH = B * H;
  const std::vector<int64_t> q3_shape = {BH, L, E};
  const std::vector<int64_t> kv3_shape = {BH, S, E};
  const std::vector<int64_t> probs3_shape = {BH, L, S};
  // A causal pattern is one (L, S) plane shared by every head, so it keeps a
  // leading dimension of one. Any other mask is broadcast over the heads first
  // so that its leading dimensions fold into the merged one.
  std::vector<int64_t> mask3_shape;
  Tensor mask_in;
  if (mask.defined()) {
    if (mask.dim() == 2) {
      mask_in = materializeForGraph(mask);
      mask3_shape = {1, L, S};
    } else {
      mask_in = materializeForGraph(mask.expand({B, H, L, S}));
      mask3_shape = probs3_shape;
    }
  }

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
    MPSShape* qShape = getMPSShape(q3_shape);
    MPSShape* kShape = getMPSShape(kv3_shape);
    MPSShape* probsShape = getMPSShape(probs3_shape);
    const MPSDataType mtl_dt = getMPSDataType(dt);

    MPSGraphTensor* qt = [graph placeholderWithShape:qShape
                                            dataType:mtl_dt
                                                 name:@"q"];
    MPSGraphTensor* kt = [graph placeholderWithShape:kShape
                                            dataType:mtl_dt
                                                 name:@"k"];
    MPSGraphTensor* kT = [graph transposeTensor:kt
                                      dimension:1
                                   withDimension:2
                                            name:@"kT"];
    MPSGraphTensor* scores =
        [graph matrixMultiplicationWithPrimaryTensor:qt
                                       secondaryTensor:kT
                                                  name:@"scores"];
    MPSGraphTensor* scaleTensor =
        [graph constantWithScalar:scale_factor
                            shape:getMPSShape({1})
                         dataType:mtl_dt];
    scores = [graph multiplicationWithPrimaryTensor:scores
                                     secondaryTensor:scaleTensor
                                                name:@"scaled"];
    MPSGraphTensor* mt = nil;
    if (mask_in.defined()) {
      mt = [graph placeholderWithShape:getMPSShape(mask3_shape)
                              dataType:mtl_dt
                                  name:@"mask"];
      scores = [graph additionWithPrimaryTensor:scores
                                 secondaryTensor:mt
                                            name:@"masked"];
    }
    MPSGraphTensor* probsT = [graph softMaxWithTensor:scores
                                                 axis:2
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
    if (mask_in.defined()) {
      id<MTLBuffer> maskBuf = mtlBufferOf(mask_in);
      MPSGraphTensorData* maskData =
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:maskBuf
                                                  shape:getMPSShape(mask3_shape)
                                               dataType:mtl_dt] autorelease];
      feeds[mt] = maskData;
    }
    getCurrentMPSStream()->executeMPSGraph(graph, feeds, results,
                                      SyncType::COMMIT_ADAPTIVE);
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
    MPSShape* pShape = getMPSShape(probs3_shape);
    MPSShape* vShape = getMPSShape(kv3_shape);
    MPSShape* outShape = getMPSShape({BH, L, E});
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
                                      SyncType::COMMIT_ADAPTIVE);
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
