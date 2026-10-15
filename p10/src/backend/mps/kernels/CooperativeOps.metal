// Cooperative-tensor kernels for M5+ (macOS 26.2+, Metal 4.0).
//
// These kernels use MetalPerformancePrimitives `mpp::tensor_ops::matmul2d`
// and `convolution2d`, which require the Metal 4.0 toolchain and the
// cooperative-tensor hardware found on Apple10+ GPUs.  The whole file is
// compiled out under older toolchains (`__METAL_VERSION__ < 400`), and the
// host dispatch is gated on `has_mpp()`, so existing MPSGraph-based paths
// remain the fallback on older machines.

#include <metal_stdlib>

#if __METAL_VERSION__ >= 400 && \
    __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_cooperative_tensor>
#include <metal_simdgroup>

using namespace metal;
using namespace mpp::tensor_ops;

// Dense GEMM C = A * B for row-major contiguous tensors, with an optional
// batch dimension (tgid.z).  One threadgroup computes a BM x BN output tile;
// the K dimension is dynamic and consumed by a single matmul2d run.
template <typename T, int BM, int BN, int NSG>
kernel void gemm_mpp(
    device T* a [[buffer(0)]],
    device T* b [[buffer(1)]],
    device T* c [[buffer(2)]],
    constant array<ulong2, 3>& strides [[buffer(3)]],
    constant uint3& sizes [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  const int m = int(sizes.x);
  const int k = int(sizes.y);
  const int n = int(sizes.z);
  const int row = int(tgid.y) * BM;
  const int col = int(tgid.x) * BN;
  const int b = int(tgid.z);
  if (b >= int(batch)) {
    return;
  }

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      BM,
      BN,
      static_cast<int>(dynamic_extent),
      false,
      false,
      false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroups<NSG>> op;

  // Contiguous row-major batch layout: A (M x K), B (K x N), C (M x N).
  device T* Ab = a + (int64_t)b * m * k;
  device T* Bb = b + (int64_t)b * k * n;
  device T* Cb = c + (int64_t)b * m * n;

  tensor<device T, dextents<int32_t, 2>, tensor_inline> mat1(
      Ab,
      dextents<int32_t, 2>(k, m),
      array<int32_t, 2>{int(strides[0].y), int(strides[0].x)});
  tensor<device T, dextents<int32_t, 2>, tensor_inline> mat2(
      Bb,
      dextents<int32_t, 2>(n, k),
      array<int32_t, 2>{int(strides[1].y), int(strides[1].x)});
  tensor<device T, dextents<int32_t, 2>, tensor_inline> output(
      Cb,
      dextents<int32_t, 2>(n, m),
      array<int32_t, 2>{int(strides[2].y), int(strides[2].x)});

  const bool inside = row + BM <= m && col + BN <= n;
  if (inside) {
    auto a_ = mat1.template slice<dynamic_extent, BM>(0, row);
    auto b_ = mat2.template slice<BN, dynamic_extent>(col, 0);
    auto c_ = output.template slice<BN, BM>(col, row);
    auto result = op.template get_destination_cooperative_tensor<
        decltype(a_),
        decltype(b_),
        float>();
    op.run(a_, b_, result);
    result.store(c_);
  } else {
    auto a_ = mat1.slice(0, row);
    auto b_ = mat2.slice(col, 0);
    auto c_ = output.slice(col, row);
    auto result = op.template get_destination_cooperative_tensor<
        decltype(a_),
        decltype(b_),
        float>();
    op.run(a_, b_, result);
    result.store(c_);
  }
}

template [[host_name("gemm_mpp_float_64_64_4")]]
kernel void gemm_mpp<float, 64, 64, 4>(
    device float* a [[buffer(0)]],
    device float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant array<ulong2, 3>& strides [[buffer(3)]],
    constant uint3& sizes [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]]);

// ---------------------------------------------------------------------------
// Cooperative-tensor convolution (conv2d).
// ---------------------------------------------------------------------------

struct ConvWeightPermuteParams {
  uint32_t output_channels;
  uint32_t input_channels_per_group;
  uint32_t kernel_height;
  uint32_t kernel_width;
  uint32_t output_channel_stride;
  uint32_t input_channel_stride;
  uint32_t height_stride;
  uint32_t width_stride;
};

struct Conv2DParams {
  int32_t N;
  int32_t C_in;
  int32_t C_out;
  int32_t H;
  int32_t W;
  int32_t outH;
  int32_t outW;
  int32_t kH;
  int32_t kW;
  int32_t sH;
  int32_t sW;
  int32_t padH;
  int32_t padW;
  int32_t dH;
  int32_t dW;
  int32_t C_in_per_group;
  int32_t C_out_per_group;
  bool has_bias;
};

// NCHW -> NHWC layout copy.  One thread per element; used to feed the NHWC
// activation view the cooperative convolution kernel expects.
template <typename T>
kernel void nchw_to_nhwc(
    device const T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& spatial [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]) {
  const uint n = tid.z;
  const uint c = tid.y;
  const uint s = tid.x;
  if (c >= channels || s >= spatial) {
    return;
  }
  const uint src_off = (n * channels + c) * spatial + s;
  const uint dst_off = (n * spatial + s) * channels + c;
  dst[dst_off] = src[src_off];
}

// OIHW -> HWIO weight copy.  Each thread streams one kernel-width row.
template <typename T>
kernel void conv_weight_to_hwio(
    device const T* source [[buffer(0)]],
    device T* destination [[buffer(1)]],
    constant ConvWeightPermuteParams& params [[buffer(2)]],
    uint3 position [[thread_position_in_grid]]) {
  const int output_channel = int(position.x);
  const int input_channel = int(position.y);
  const int kernel_height_index = int(position.z);
  device const T* source_row =
      source +
      output_channel * int(params.output_channel_stride) +
      input_channel * int(params.input_channel_stride) +
      kernel_height_index * int(params.height_stride);
  device T* destination_row =
      destination +
      (kernel_height_index * int(params.kernel_width) *
           int(params.input_channels_per_group) +
       input_channel) *
          int(params.output_channels) +
      output_channel;
  for (uint kernel_width_index = 0;
       kernel_width_index < params.kernel_width; ++kernel_width_index) {
    destination_row[kernel_width_index *
                    int(params.input_channels_per_group) *
                    int(params.output_channels)] =
        source_row[kernel_width_index * int(params.width_stride)];
  }
}

template <int X, int Y, int Z>
struct StaticInt3 {
  static constexpr constant int x = X;
  static constexpr constant int y = Y;
  static constexpr constant int z = Z;
};

// Cooperative 2D convolution.  The activation is NHWC, the weights are HWIO,
// and the output is written back in NCHW order.  Kernel size, stride, dilation
// and source-channel count are compile-time template parameters, so the host
// looks up a kernel name for the exact configuration and falls back to the
// MPSGraph path when that configuration is not instantiated.
template <
    typename T,
    typename Block,
    int NSG,
    typename Kernel,
    typename Stride,
    typename Dilation,
    typename Source,
    bool RELAXED,
    bool HAS_BIAS>
kernel void conv2d_mpp(
    device T* act [[buffer(0)]],
    device T* wts [[buffer(1)]],
    device T* dst [[buffer(2)]],
    constant Conv2DParams& p [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  constexpr int BO = Block::x, BW = Block::y, BH = Block::z;
  constexpr int KW = Kernel::x, KH = Kernel::y;
  constexpr int SX = Stride::x, SY = Stride::y;
  constexpr int DX = Dilation::x, DY = Dilation::y;
  constexpr int SRCC = Source::x;
  const int h_tiles = p.outH / BH + (p.outH % BH != 0);
  const int o_tiles = p.C_out / BO + (p.C_out % BO != 0);
  const int o_off = int(tgid.x) % o_tiles * BO;
  const int o_end = p.C_out;
  const int wo_off = int(tgid.y) * BW;
  const int ho_off = (int(tgid.z) % h_tiles) * BH;
  const int nb = int(tgid.z) / h_tiles;

  // HWIO weights: C_out-major view of the (H, W, C_in, C_out) buffer.
  tensor<device T, dextents<int32_t, 4>, tensor_inline> tWt(
      wts,
      dextents<int32_t, 4>(p.C_out, p.C_in, KW, KH));

  constexpr auto desc = convolution2d_descriptor(
      int4(BO, BW, BH, 1),
      int4(SRCC, 16384, 16384, 1),
      int2(KW, KH),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      int2(SX, SY),
      int2(DX, DY),
      1,
      RELAXED,
      convolution2d_descriptor::mode::multiply_accumulate);
  convolution2d<desc, execution_simdgroups<NSG>> op;
  op.set_offsets(int2(
      wo_off * SX - p.padW + ((KW - 1) * DX + 1) / 2,
      ho_off * SY - p.padH + ((KH - 1) * DY + 1) / 2));

  const int64_t plane = (int64_t)p.H * p.W * p.C_in;
  device T* actn = act + (int64_t)nb * plane;
  tensor<device T, dextents<int32_t, 4>, tensor_inline> tA(
      actn,
      dextents<int32_t, 4>(p.C_in, p.W, p.H, 1),
      array<int32_t, 4>{1, p.C_in, p.C_in * p.W, p.C_in * p.W * p.H});
  auto mA = tA.slice(0, 0, 0, 0);
  auto mW = tWt.slice(o_off, 0, 0, 0);

  auto cT = op.template get_destination_cooperative_tensor<
      decltype(mA),
      decltype(mW),
      float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    cT[i] = 0.0f;
  }

  op.run(mA, mW, cT);

  // NCHW output: iterate the cooperative tensor and write in-bounds lanes.
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    auto idx = cT.get_multidimensional_index(i);
    const int o = o_off + int(idx[0]);
    const int x = wo_off + int(idx[1]);
    const int y = ho_off + int(idx[2]);
    if (o >= o_end || x >= p.outW || y >= p.outH) {
      continue;
    }
    float v = cT[i];
    if constexpr (HAS_BIAS) {
      v += float(bias[o]);
    }
    const int64_t di =
        ((((int64_t)nb * p.C_out + o) * p.outH + y) * p.outW + x);
    dst[di] = T(v);
  }
}

#define INSTANTIATE_CONV2D_MPP(                                            \
    DT, RELAXED, BO, BW, BH, NSG, KD, KH, KW, SX, SY, DX, DY, CNAME,        \
    SRCC, BNAME, HAS_BIAS)                                                  \
  template [[host_name("conv2d_mpp_" #DT "_b" #BO "_w" #BW "_h" #BH "_s"   \
                       #NSG "_k" #KH #KW "_s" #SX #SY "_d" #DX #DY         \
                       "_c" #CNAME "_" #BNAME)]]                           \
  kernel void conv2d_mpp<                                                   \
      DT,                                                                   \
      StaticInt3<BO, BW, BH>,                                               \
      NSG,                                                                  \
      StaticInt3<KW, KH, 1>,                                                \
      StaticInt3<SX, SY, 1>,                                                \
      StaticInt3<DX, DY, 1>,                                                \
      StaticInt3<SRCC, 16384, 16384>,                                        \
      RELAXED,                                                             \
      HAS_BIAS>(                                                            \
      device DT * act [[buffer(0)]],                                        \
      device DT * wts [[buffer(1)]],                                        \
      device DT * dst [[buffer(2)]],                                        \
      constant Conv2DParams & p [[buffer(3)]],                              \
      device const DT * bias [[buffer(4)]],                                 \
      uint3 tgid [[threadgroup_position_in_grid]]);

#define INSTANTIATE_CONV2D_MPP_DTYPES(                                     \
    BO, BW, BH, NSG, KH, KW, SX, SY, DX, DY, CNAME, SRCC, BNAME, HAS_BIAS) \
  INSTANTIATE_CONV2D_MPP(                                                  \
      float, false, BO, BW, BH, NSG, 1, KH, KW, SX, SY, DX, DY, CNAME,      \
      SRCC, BNAME, HAS_BIAS)                                                \
  INSTANTIATE_CONV2D_MPP(                                                  \
      half, true, BO, BW, BH, NSG, 1, KH, KW, SX, SY, DX, DY, CNAME,        \
      SRCC, BNAME, HAS_BIAS);

#define INSTANTIATE_CONV2D_MPP_KERNEL(KH, KW, CNAME, SRCC)                 \
  INSTANTIATE_CONV2D_MPP_DTYPES(                                           \
      32, 8, 8, 2, KH, KW, 1, 1, 1, 1, CNAME, SRCC, "bias", true)          \
  INSTANTIATE_CONV2D_MPP_DTYPES(                                           \
      32, 8, 8, 2, KH, KW, 1, 1, 1, 1, CNAME, SRCC, "nobias", false);

INSTANTIATE_CONV2D_MPP_KERNEL(1, 1, "c1", 1)
INSTANTIATE_CONV2D_MPP_KERNEL(1, 1, "c3", 3)
INSTANTIATE_CONV2D_MPP_KERNEL(1, 1, "c16", 16)
INSTANTIATE_CONV2D_MPP_KERNEL(1, 1, "c32", 32)
INSTANTIATE_CONV2D_MPP_KERNEL(1, 1, "c64", 64)
INSTANTIATE_CONV2D_MPP_KERNEL(2, 2, "c1", 1)
INSTANTIATE_CONV2D_MPP_KERNEL(2, 2, "c3", 3)
INSTANTIATE_CONV2D_MPP_KERNEL(2, 2, "c16", 16)
INSTANTIATE_CONV2D_MPP_KERNEL(2, 2, "c32", 32)
INSTANTIATE_CONV2D_MPP_KERNEL(2, 2, "c64", 64)
INSTANTIATE_CONV2D_MPP_KERNEL(3, 3, "c1", 1)
INSTANTIATE_CONV2D_MPP_KERNEL(3, 3, "c3", 3)
INSTANTIATE_CONV2D_MPP_KERNEL(3, 3, "c16", 16)
INSTANTIATE_CONV2D_MPP_KERNEL(3, 3, "c32", 32)
INSTANTIATE_CONV2D_MPP_KERNEL(3, 3, "c64", 64)

#define REGISTER_CONV_WEIGHT_TO_HWIO(DT)                                   \
  template [[host_name("conv_weight_to_hwio_" #DT)]] kernel void           \
  conv_weight_to_hwio<DT>(                                                 \
      device const DT * source [[buffer(0)]],                               \
      device DT * destination [[buffer(1)]],                                \
      constant ConvWeightPermuteParams & params [[buffer(2)]],              \
      uint3 position [[thread_position_in_grid]]);

#define REGISTER_NCHW_TO_NHWC(DT)                                          \
  template [[host_name("nchw_to_nhwc_" #DT)]] kernel void                   \
  nchw_to_nhwc<DT>(                                                         \
      device const DT * src [[buffer(0)]],                                  \
      device DT * dst [[buffer(1)]],                                        \
      constant uint & channels [[buffer(2)]],                               \
      constant uint & spatial [[buffer(3)]],                                \
      uint3 tid [[thread_position_in_grid]]);

REGISTER_CONV_WEIGHT_TO_HWIO(float)
REGISTER_CONV_WEIGHT_TO_HWIO(half)
REGISTER_NCHW_TO_NHWC(float)
REGISTER_NCHW_TO_NHWC(half)

#endif // __METAL_VERSION__ >= 400 && MPP header present