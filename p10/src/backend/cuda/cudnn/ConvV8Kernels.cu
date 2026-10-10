#if defined(USE_CUDNN) && __has_include(<cudnn_frontend.h>) && \
    __has_include(<cudnn_frontend_find_plan.h>)
#define TP_HAS_CUDNN_FRONTEND 1
#include <cudnn_frontend.h>
#include <cudnn_frontend_find_plan.h>
namespace fe = cudnn_frontend;
#endif
#include "cudnn/ConvShared.h"
#include "Tensor.h"
#include "Convolution.h"
#include "Dispatcher.h"
#include "Context.h"
#include "Exception.h"
#include "CUDAContext.h"
#include "CUDARuntime.h"
#include "CUDNNUtils.h"
#include "CudaGemm.h"
#include "Allocator.h"
#include <vector>
#include <array>
#include <unordered_map>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <memory>
#include <cstdint>
#include <optional>
#include <algorithm>

namespace tensorplay {
namespace cuda {

// ---------------------------------------------------------------------------
// Gradient-bias reduction kernels for the float64 fallback path
// ---------------------------------------------------------------------------
template <typename InputT, typename AccT, typename OutputT>
__global__ void conv2d_grad_bias_reduce_kernel(
    const InputT* __restrict__ grad_output, int64_t batch, int64_t channels,
    int64_t spatial, OutputT* __restrict__ grad_bias) {
    const int64_t channel = static_cast<int64_t>(blockIdx.x);
    if (channel >= channels) return;

    const int64_t channel_stride = channels * spatial;
    AccT value = AccT(0);
    for (int64_t n = 0; n < batch; ++n) {
        const InputT* row = grad_output + n * channel_stride + channel * spatial;
        for (int64_t offset = threadIdx.x; offset < spatial;
             offset += static_cast<int64_t>(blockDim.x)) {
            value += static_cast<AccT>(row[offset]);
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
    }
    __shared__ AccT warp_values[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();

    if (warp == 0) {
        const int warp_count = (blockDim.x + 31) / 32;
        value = lane < warp_count ? warp_values[lane] : AccT(0);
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
        }
        if (lane == 0) grad_bias[channel] = static_cast<OutputT>(value);
    }
}

template <typename OutputT>
__global__ void conv2d_grad_bias_vec_half_kernel(
    const Half* __restrict__ grad_output, int64_t batch, int64_t channels,
    int64_t spatial, OutputT* __restrict__ grad_bias) {
    const int64_t channel = static_cast<int64_t>(blockIdx.x);
    if (channel >= channels) return;

    const int64_t channel_stride = channels * spatial;
    const int64_t vector_count = spatial / 8;
    float value = 0.0f;
    for (int64_t n = 0; n < batch; ++n) {
        const Half* row = grad_output + n * channel_stride + channel * spatial;
        const float4* row_vec = reinterpret_cast<const float4*>(row);
        for (int64_t offset = threadIdx.x; offset < vector_count;
             offset += static_cast<int64_t>(blockDim.x)) {
            const float4 packed = row_vec[offset];
            const __half2 h0 = *reinterpret_cast<const __half2*>(&packed.x);
            const __half2 h1 = *reinterpret_cast<const __half2*>(&packed.y);
            const __half2 h2 = *reinterpret_cast<const __half2*>(&packed.z);
            const __half2 h3 = *reinterpret_cast<const __half2*>(&packed.w);
            value += __half2float(h0.x) + __half2float(h0.y) +
                     __half2float(h1.x) + __half2float(h1.y) +
                     __half2float(h2.x) + __half2float(h2.y) +
                     __half2float(h3.x) + __half2float(h3.y);
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
    }
    __shared__ float warp_values[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();

    if (warp == 0) {
        const int warp_count = (blockDim.x + 31) / 32;
        value = lane < warp_count ? warp_values[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
        }
        if (lane == 0) grad_bias[channel] = static_cast<OutputT>(value);
    }
}

template <typename OutputT>
__global__ void conv2d_grad_bias_half2_kernel(
    const Half* __restrict__ grad_output, int64_t batch, int64_t channels,
    int64_t spatial, OutputT* __restrict__ grad_bias) {
    const int64_t channel = static_cast<int64_t>(blockIdx.x);
    if (channel >= channels) return;

    const int64_t channel_stride = channels * spatial;
    const int64_t pair_count = spatial / 2;
    float value = 0.0f;
    for (int64_t n = 0; n < batch; ++n) {
        const Half* row = grad_output + n * channel_stride + channel * spatial;
        const __half2* row_vec = reinterpret_cast<const __half2*>(row);
        for (int64_t offset = threadIdx.x; offset < pair_count;
             offset += static_cast<int64_t>(blockDim.x)) {
            const __half2 pair = row_vec[offset];
            value += __half2float(pair.x) + __half2float(pair.y);
        }
        if ((spatial & 1) != 0 && threadIdx.x == 0) {
            value += static_cast<float>(row[spatial - 1]);
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
    }
    __shared__ float warp_values[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();

    if (warp == 0) {
        const int warp_count = (blockDim.x + 31) / 32;
        value = lane < warp_count ? warp_values[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
        }
        if (lane == 0) grad_bias[channel] = static_cast<OutputT>(value);
    }
}

template <typename InputT, typename AccT, typename OutputT>
__global__ void conv2d_grad_bias_strided_kernel(
    const InputT* __restrict__ grad_output, int64_t batch, int64_t channels,
    int64_t height, int64_t width, int64_t stride_n, int64_t stride_c,
    int64_t stride_h, int64_t stride_w, OutputT* __restrict__ grad_bias) {
    const int64_t channel = static_cast<int64_t>(blockIdx.x);
    if (channel >= channels) return;

    const int64_t spatial = height * width;
    const int64_t samples = batch * spatial;
    AccT value = AccT(0);
    for (int64_t sample = threadIdx.x; sample < samples;
         sample += static_cast<int64_t>(blockDim.x)) {
        const int64_t n = sample / spatial;
        const int64_t position = sample - n * spatial;
        const int64_t h = position / width;
        const int64_t w = position - h * width;
        value += static_cast<AccT>(grad_output[
            n * stride_n + channel * stride_c + h * stride_h + w * stride_w]);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
    }
    __shared__ AccT warp_values[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();

    if (warp == 0) {
        const int warp_count = (blockDim.x + 31) / 32;
        value = lane < warp_count ? warp_values[lane] : AccT(0);
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
        }
        if (lane == 0) grad_bias[channel] = static_cast<OutputT>(value);
    }
}


static Tensor conv2d_cuda_impl(const Tensor& input, const Tensor& weight, const Tensor& bias, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, int64_t groups, bool fused_relu) {
    convolution::check_conv_shapes(input, weight, bias, groups, false);
    convolution::check_conv_geometry(input, weight, stride_arg, padding_arg,
                                     dilation_arg, "conv2d");
#if defined(USE_ROCM)
    // The DNN surface has no usable double-precision convolution on every
    // supported AMD target (fp64 code-object builds can fail at run time),
    // so fp64 tensors compute through the native im2col + GEMM path.
    if (input.dtype() == DType::Float64) {
        Tensor out = conv2d_slow_fp64(input, weight, bias, stride_arg, padding_arg,
                                      dilation_arg, groups);
        if (fused_relu) out = out.clamp_min(0.0);
        return out;
    }
#endif
#ifdef USE_CUDNN
#if defined(TP_HAS_CUDNN_FRONTEND)
    auto stride = expand_param_if_needed(stride_arg, 2, 1);
    auto padding = expand_param_if_needed(padding_arg, 2, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 2, 1);

    const int64_t N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    const int64_t K = weight.size(0), R = weight.size(2), S = weight.size(3);
    const int64_t OH = (H + 2 * padding[0] - dilation[0] * (R - 1) - 1) / stride[0] + 1;
    const int64_t OW = (W + 2 * padding[1] - dilation[1] * (S - 1) - 1) / stride[1] + 1;

    if (fused_relu && bias.defined() &&
        (input.dtype() == DType::Float32 || input.dtype() == DType::Float64)) {
        return conv2d_relu_cudnn(
            input, weight, bias, stride, padding, dilation, groups);
    }

    cudnnDataType_t dtype;
    if (input.dtype() == DType::Float32) dtype = CUDNN_DATA_FLOAT;
    else if (input.dtype() == DType::Float64) dtype = CUDNN_DATA_DOUBLE;
    else if (input.dtype() == DType::Float16) dtype = CUDNN_DATA_HALF;
    else if (input.dtype() == DType::BFloat16) dtype = CUDNN_DATA_BFLOAT16;
    else TP_THROW(NotImplementedError, "cuDNN: only float/double/half/bfloat16 supported");

    cudnnDataType_t compute = (dtype == CUDNN_DATA_DOUBLE) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    if (dtype == CUDNN_DATA_FLOAT && tensorplay::globalContext().allowTF32CuDNN()) {
        compute = CUDNN_DATA_FLOAT;
    }

    // Conv_v8 path builds descriptors from actual sizes and strides, so the
    // layout is part of the plan identity as well.
    struct ConvKey {
        cudnnDataType_t dtype;
        int64_t N, C, H, W, K, R, S, groups;
        int64_t ph, pw, sh, sw, dh, dw;
        int device;
        bool has_bias;
        bool fused_relu;
        bool autotune;
        // Numeric switches are part of the plan identity: flipping either one
        // changes which engines are legal, so a cached plan must not be reused.
        bool allow_tf32;
        bool deterministic;
        std::array<int64_t, 4> x_stride;
        std::array<int64_t, 4> w_stride;
        std::array<int64_t, 4> y_stride;
        bool operator==(const ConvKey& o) const {
            return dtype == o.dtype && N == o.N && C == o.C && H == o.H && W == o.W &&
                   K == o.K && R == o.R && S == o.S && groups == o.groups &&
                   ph == o.ph && pw == o.pw && sh == o.sh && sw == o.sw &&
                   dh == o.dh && dw == o.dw && device == o.device &&
                   has_bias == o.has_bias && fused_relu == o.fused_relu &&
                   allow_tf32 == o.allow_tf32 && deterministic == o.deterministic &&
                   autotune == o.autotune &&
                   x_stride == o.x_stride && w_stride == o.w_stride &&
                   y_stride == o.y_stride;
        }
    };
    struct ConvKeyHash {
        size_t operator()(const ConvKey& k) const {
            size_t h = std::hash<int64_t>{}(k.N);
            for (auto v : {k.C, k.H, k.W, k.K, k.R, k.S, k.groups, k.ph, k.pw, k.sh, k.sw, k.dh, k.dw})
                h = h * 1000003 ^ std::hash<int64_t>{}(v);
            h = h * 1000003 ^ std::hash<int>{}((int)k.dtype);
            h = h * 1000003 ^ std::hash<int>{}(k.device);
            h = h * 1000003 ^ std::hash<int>{}((int)k.fused_relu);
            h = h * 1000003 ^ std::hash<int>{}((int)k.autotune);
            for (auto v : k.x_stride) h = h * 1000003 ^ std::hash<int64_t>{}(v);
            for (auto v : k.w_stride) h = h * 1000003 ^ std::hash<int64_t>{}(v);
            for (auto v : k.y_stride) h = h * 1000003 ^ std::hash<int64_t>{}(v);
            return h;
        }
    };
    static std::unordered_map<ConvKey, std::shared_ptr<fe::ExecutionPlan>, ConvKeyHash> g_conv_plan_cache;
    static std::mutex g_conv_cache_mutex;

    // Byte alignment of a tensor's storage, as the engine selection is
    // sensitive to it: an engine that requires a wider alignment than the
    // one declared here is not offered.  Doubling stops at 32 bytes, which
    // is the widest alignment the engine configurations are built around.
    auto alignment_of = [](const void* ptr) -> int64_t {
        int64_t alignment = 1;
        auto address = reinterpret_cast<uintptr_t>(ptr);
        for (; alignment < 32; alignment *= 2) {
            if (address % static_cast<uintptr_t>(alignment * 2)) return alignment;
        }
        return alignment;
    };

    const bool use_channels_last = is_channels_last_4d(input);
    const bool post_bias_add =
        bias.defined() && bias.numel() != 0 && !fused_relu &&
        !use_channels_last &&
        (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
    const std::array<int64_t, 4> x_stride{
        input.stride(0), input.stride(1), input.stride(2), input.stride(3)};
    const std::array<int64_t, 4> w_stride{
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3)};
    const std::array<int64_t, 4> y_stride = use_channels_last
        ? channels_last_strides(K, OH, OW)
        : std::array<int64_t, 4>{K * OH * OW, OH * OW, OW, 1};
    Tensor out = empty_conv_output(
        N, K, OH, OW, input.dtype(), input.device(), use_channels_last);

    ConvKey key{dtype, N, C, H, W, K, R, S, groups,
                padding[0], padding[1], stride[0], stride[1], dilation[0], dilation[1],
                static_cast<int>(input.device().index()),
                bias.defined() && !post_bias_add, fused_relu, conv_autotune_enabled(),
                tensorplay::globalContext().allowTF32CuDNN(),
                tensorplay::globalContext().deterministicAlgorithms(),
                x_stride, w_stride, y_stride};

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();

    // Executes one plan against this call's tensors (used both for the real
    // compute and for timing candidate plans in benchmark mode).
    auto run_plan = [&](const fe::ExecutionPlan& p, void* ws_ptr, size_t ws_size) {
        if (key.has_bias) {
            void* data_ptrs[4] = {input.data_ptr(), weight.data_ptr(), bias.data_ptr(), out.data_ptr()};
            int64_t uids[4] = {'x', 'w', 'b', 'y'};
            auto variant_pack = fe::VariantPackBuilder()
                                    .setWorkspacePointer(ws_size ? ws_ptr : nullptr)
                                    .setDataPointers(4, data_ptrs)
                                    .setUids(4, uids)
                                    .build();
            CUDNN_CHECK(cudnnBackendExecute(handle, p.get_raw_desc(), variant_pack.get_raw_desc()));
        } else {
            void* data_ptrs[3] = {input.data_ptr(), weight.data_ptr(), out.data_ptr()};
            int64_t uids[3] = {'x', 'w', 'y'};
            auto variant_pack = fe::VariantPackBuilder()
                                    .setWorkspacePointer(ws_size ? ws_ptr : nullptr)
                                    .setDataPointers(3, data_ptrs)
                                    .setUids(3, uids)
                                    .build();
            CUDNN_CHECK(cudnnBackendExecute(handle, p.get_raw_desc(), variant_pack.get_raw_desc()));
        }
    };

    // The heuristic's fallback list: engines ranked without the instant
    // model, including the plain ones it leaves out.
    auto fallback_engine_configs = [](fe::OperationGraph& op_graph) {
        auto heuristics = fe::EngineHeuristicsBuilder()
                              .setOperationGraph(op_graph)
                              .setHeurMode(CUDNN_HEUR_MODE_FALLBACK)
                              .build();
        fe::EngineConfigList configs =
            heuristics.getEngineConfig(heuristics.getEngineConfigCount());
        return configs;
    };

    // Numerical-note based engine filter: an engine is unusable when it is
    // non-deterministic (in deterministic mode), when it down-converts the
    // inputs, or when it relies on tensor cores while TF32 is off for float32.
    auto usable_engine_configs = [](fe::EngineConfigList& from,
                                     bool deterministic,
                                     bool allow_tf32,
                                     cudnnDataType_t scalar_type) {
        fe::EngineConfigList kept;
        auto drop = [=](cudnnBackendDescriptor_t c) {
            if (deterministic &&
                fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(c)) {
                return true;
            }
            if (fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(c)) {
                return true;
            }
            if (scalar_type == CUDNN_DATA_FLOAT && !allow_tf32 &&
                fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(c)) {
                return true;
            }
            return false;
        };
        fe::filter(from, kept, drop);
        return kept;
    };

    // Picks an execution plan for the graph.  Normally the first heuristic
    // config is used; in benchmark mode several configs are timed instead.
    // Cap on how many candidate plans the timer profiles.
    constexpr uint64_t kConvBenchmarkPlanLimit = 10000;

    auto pick_plan = [&](fe::OperationGraph& op_graph) -> std::shared_ptr<fe::ExecutionPlan> {
        const bool autotune = conv_autotune_enabled();
        const bool deterministic = tensorplay::globalContext().deterministicAlgorithms();
        const bool allow_tf32 = tensorplay::globalContext().allowTF32CuDNN();
        auto heuristics = fe::EngineHeuristicsBuilder()
                              .setOperationGraph(op_graph)
                              .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                              .build();
        // Benchmarking times the candidates of the same heuristic list the
        // plain path ranks; switching to the fallback list instead loses the
        // leading engines and ends up measurably slower than just taking the
        // first entry.
        auto& engine_configs = heuristics.getEngineConfig(
            autotune ? heuristics.getEngineConfigCount() : 1);
        if (engine_configs.empty()) {
            TP_THROW(RuntimeError, "cuDNN: no engine configs for conv2d");
        }
        auto filtered = usable_engine_configs(engine_configs, deterministic,
                                              allow_tf32, dtype);
        if (filtered.empty()) {
            // The top configs are all unusable; fall back to the full
            // heuristic list before giving up on filtering.
            engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
            filtered = usable_engine_configs(engine_configs, deterministic,
                                             allow_tf32, dtype);
        }
        if (filtered.empty()) {
            // The instant list can hold tensor-core engines only; the fallback
            // list carries the plain ones a float32 call without TF32 needs.
            fe::EngineConfigList plain = fallback_engine_configs(op_graph);
            filtered = usable_engine_configs(plain, deterministic, allow_tf32, dtype);
        }
        if (filtered.empty()) filtered = engine_configs;
        if (!autotune || filtered.size() == 1) {
            for (auto& ec : filtered) {
                try {
                    return std::make_shared<fe::ExecutionPlan>(
                        fe::ExecutionPlanBuilder()
                            .setHandle(handle)
                            .setEngineConfig(ec)
                            .build());
                } catch (...) {
                    // Config cannot be realized on this device; try the next.
                }
            }
            TP_THROW(RuntimeError, "cuDNN: no executable engine configs for conv2d");
        }
        std::vector<std::shared_ptr<fe::ExecutionPlan>> candidates;
        for (auto& ec : filtered) {
            try {
                candidates.push_back(std::make_shared<fe::ExecutionPlan>(
                    fe::ExecutionPlanBuilder().setHandle(handle).setEngineConfig(ec).build()));
            } catch (...) {
                // Config failed to build an executable plan; skip it.
            }
        }
        if (candidates.empty()) {
            TP_THROW(RuntimeError, "cuDNN: no executable engine configs for conv2d");
        }
        if (candidates.size() == 1) return candidates[0];

        // Hand the candidates to the library timer, which warms each plan up
        // once, times a single run, and returns them ordered by that time.
        // A home-grown timing loop is not equivalent: with one sample per
        // candidate the ordering follows the sample, and a loop that keeps
        // only the minimum of a few back-to-back runs ranks candidates by a
        // different statistic than the one the sorted result is built from.
        int64_t max_ws = 0;
        for (auto& cand : candidates) {
            max_ws = std::max(max_ws, cand->getWorkspaceSize());
        }
        auto ws_buf = getAllocator(DeviceType::CUDA)->allocate(max_ws ? max_ws : 1);
        fe::executionPlans_t plans;
        plans.reserve(candidates.size());
        for (auto& cand : candidates) plans.push_back(std::move(*cand));
        std::optional<fe::VariantPack> variant_pack;
        if (key.has_bias) {
            void* data_ptrs[4] = {input.data_ptr(), weight.data_ptr(), bias.data_ptr(), out.data_ptr()};
            int64_t uids[4] = {'x', 'w', 'b', 'y'};
            variant_pack.emplace(fe::VariantPackBuilder()
                               .setWorkspacePointer(max_ws ? ws_buf.get() : nullptr)
                               .setDataPointers(4, data_ptrs)
                               .setUids(4, uids)
                               .build());
        } else {
            void* data_ptrs[3] = {input.data_ptr(), weight.data_ptr(), out.data_ptr()};
            int64_t uids[3] = {'x', 'w', 'y'};
            variant_pack.emplace(fe::VariantPackBuilder()
                               .setWorkspacePointer(max_ws ? ws_buf.get() : nullptr)
                               .setDataPointers(3, data_ptrs)
                               .setUids(3, uids)
                               .build());
        }
        auto timed = fe::time_sorted_plan<fe::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(
            handle, std::move(plans), *variant_pack, kConvBenchmarkPlanLimit);
        if (timed.empty()) {
            TP_THROW(RuntimeError, "cuDNN: no candidate plan could be timed");
        }
        return std::make_shared<fe::ExecutionPlan>(std::move(timed.front()));
    };

    const int64_t x_align = alignment_of(input.data_ptr());
    const int64_t w_align = alignment_of(weight.data_ptr());
    const int64_t y_align = alignment_of(out.data_ptr());
    const int64_t bias_align =
        bias.defined() ? alignment_of(bias.data_ptr()) : y_align;

    std::shared_ptr<fe::ExecutionPlan> plan;
    {
        std::lock_guard<std::mutex> lock(g_conv_cache_mutex);
        auto it = g_conv_plan_cache.find(key);
        if (it != g_conv_plan_cache.end()) {
            plan = it->second;
        } else {
            auto x_desc = fe::TensorBuilder()
                              .setDim(4, std::array<int64_t, 4>{N, C, H, W}.data())
                              .setStrides(4, x_stride.data())
                              .setId('x')
                              .setAlignment(x_align)
                              .setDataType(dtype)
                              .build();
            auto w_desc = fe::TensorBuilder()
                              .setDim(4, std::array<int64_t, 4>{K, C / groups, R, S}.data())
                              .setStrides(4, w_stride.data())
                              .setId('w')
                              .setAlignment(w_align)
                              .setDataType(dtype)
                              .build();
            auto y_desc = fe::TensorBuilder()
                              .setDim(4, std::array<int64_t, 4>{N, K, OH, OW}.data())
                              .setStrides(4, y_stride.data())
                              .setId('y')
                              .setAlignment(y_align)
                              .setDataType(dtype)
                              .build();

            int64_t pad[2] = {padding[0], padding[1]};
            int64_t strd[2] = {stride[0], stride[1]};
            int64_t dil[2] = {dilation[0], dilation[1]};

            auto conv_desc = fe::ConvDescBuilder()
                                 .setComputeType(compute)
                                 .setMathMode(CUDNN_CROSS_CORRELATION)
                                 .setSpatialDimCount(2)
                                 .setSpatialStride(2, strd)
                                 .setPrePadding(2, pad)
                                 .setPostPadding(2, pad)
                                 .setDilation(2, dil)
                                 .build();

            fe::Operation conv_op = fe::OperationBuilder(
                                        CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                                        .setxDesc(x_desc)
                                        .setwDesc(w_desc)
                                        .setyDesc(y_desc)
                                        .setcDesc(conv_desc)
                                        .build();

            std::shared_ptr<fe::ExecutionPlan> new_plan;
            if (key.has_bias) {
                // Conv output is a virtual tensor in compute precision ('C'),
                // the add op writes the final NCHW output ('y').
                auto conv_out_desc = fe::TensorBuilder()
                                         .setDim(4, std::array<int64_t, 4>{N, K, OH, OW}.data())
                                         .setStrides(4, y_stride.data())
                                         .setId('C')
                                         .setAlignment(y_align)
                                         .setDataType(compute)
                                         .setVirtual(true)
                                         .build();
                auto b_desc = fe::TensorBuilder()
                                  .setDim(4, std::array<int64_t, 4>{1, K, 1, 1}.data())
                                  .setStrides(4, std::array<int64_t, 4>{K, 1, 1, 1}.data())
                                  .setId('b')
                                  .setAlignment(bias_align)
                                  .setDataType(dtype)
                                  .build();
                auto bias_add_desc = fe::PointWiseDescBuilder()
                                         .setMode(CUDNN_POINTWISE_ADD)
                                         .setMathPrecision(compute)
                                         .build();
                auto conv_bias_op = fe::OperationBuilder(
                                        CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                                        .setxDesc(x_desc)
                                        .setwDesc(w_desc)
                                        .setyDesc(conv_out_desc)
                                        .setcDesc(conv_desc)
                                        .build();
                std::optional<fe::Tensor> bias_out_desc;
                if (key.fused_relu) {
                    bias_out_desc = fe::TensorBuilder()
                                        .setDim(4, std::array<int64_t, 4>{N, K, OH, OW}.data())
                                        .setStrides(4, y_stride.data())
                                        .setId('B')
                                        .setAlignment(y_align)
                                        .setDataType(compute)
                                        .setVirtual(true)
                                        .build();
                }
                auto bias_op = fe::OperationBuilder(
                                   CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                   .setxDesc(conv_bias_op.getOutputTensor())
                                   .setbDesc(b_desc)
                                   .setyDesc(key.fused_relu ? *bias_out_desc : y_desc)
                                   .setpwDesc(bias_add_desc)
                                   .build();
                std::shared_ptr<fe::OperationGraph> op_graph_ptr;
                if (key.fused_relu) {
                    auto relu_desc = fe::PointWiseDescBuilder()
                                         .setMode(CUDNN_POINTWISE_RELU_FWD)
                                         .setMathPrecision(compute)
                                         .build();
                    auto relu_op = fe::OperationBuilder(
                                       CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                       .setxDesc(bias_op.getOutputTensor())
                                       .setyDesc(y_desc)
                                       .setpwDesc(relu_desc)
                                       .build();
                    std::array<fe::Operation const*, 3> ops = {&conv_bias_op, &bias_op, &relu_op};
                    auto op_graph = std::make_shared<fe::OperationGraph>(
                        fe::OperationGraphBuilder()
                            .setHandle(handle)
                            .setOperationGraph(ops.size(), ops.data())
                            .build());
                    op_graph_ptr = std::move(op_graph);
                } else {
                    std::array<fe::Operation const*, 2> ops = {&conv_bias_op, &bias_op};
                    op_graph_ptr = std::make_shared<fe::OperationGraph>(
                        fe::OperationGraphBuilder()
                            .setHandle(handle)
                            .setOperationGraph(ops.size(), ops.data())
                            .build());
                }
                new_plan = pick_plan(*op_graph_ptr);
            } else if (!key.fused_relu) {
                std::array<fe::Operation const*, 1> ops = {&conv_op};
                auto op_graph = fe::OperationGraphBuilder()
                                    .setHandle(handle)
                                    .setOperationGraph(ops.size(), ops.data())
                                    .build();
                new_plan = pick_plan(op_graph);
            } else {
                auto conv_out_desc = fe::TensorBuilder()
                                         .setDim(4, std::array<int64_t, 4>{N, K, OH, OW}.data())
                                         .setStrides(4, y_stride.data())
                                         .setId('C')
                                         .setAlignment(y_align)
                                         .setDataType(compute)
                                         .setVirtual(true)
                                         .build();
                auto fused_conv_op = fe::OperationBuilder(
                                         CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                                         .setxDesc(x_desc)
                                         .setwDesc(w_desc)
                                         .setyDesc(conv_out_desc)
                                         .setcDesc(conv_desc)
                                         .build();
                auto relu_desc = fe::PointWiseDescBuilder()
                                     .setMode(CUDNN_POINTWISE_RELU_FWD)
                                     .setMathPrecision(compute)
                                     .build();
                auto relu_op = fe::OperationBuilder(
                                   CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                   .setxDesc(fused_conv_op.getOutputTensor())
                                   .setyDesc(y_desc)
                                   .setpwDesc(relu_desc)
                                   .build();
                std::array<fe::Operation const*, 2> ops = {&fused_conv_op, &relu_op};
                auto op_graph = fe::OperationGraphBuilder()
                                    .setHandle(handle)
                                    .setOperationGraph(ops.size(), ops.data())
                                    .build();
                new_plan = pick_plan(op_graph);
            }
            plan = new_plan;
            g_conv_plan_cache[key] = new_plan;
        }
    }

    size_t workspace_size = plan->getWorkspaceSize();
    auto workspace = getAllocator(DeviceType::CUDA)->allocate(workspace_size ? workspace_size : 1);

    run_plan(*plan, workspace.get(), workspace_size);

    if (post_bias_add) {
        Tensor bias_c = bias.is_contiguous() ? bias : bias.contiguous();
        Tensor bias_4d = bias_c.reshape({1, K, 1, 1});
        bool added = false;
        if ((out.dtype() == DType::Float16 || out.dtype() == DType::BFloat16) &&
            (bias_c.dtype() == DType::Float32 ||
             bias_c.dtype() == DType::Float16 ||
             bias_c.dtype() == DType::BFloat16)) {
            added = add_channel_broadcast_inplace_cuda(out, bias_4d);
        }
        if (!added) {
            auto bias_desc = get_cached_tensor_desc(bias_4d);
            auto out_desc = get_cached_tensor_desc(out);
            float alpha = 1.0f;
            float beta = 1.0f;
            CUDNN_CHECK(cudnnAddTensor(
                handle, &alpha, *bias_desc, bias_4d.data_ptr(), &beta,
                *out_desc, out.data_ptr()));
        }
    }

    return out;
#else
    return conv2d_cudnn_legacy(
        input, weight, bias, stride_arg, padding_arg, dilation_arg, groups,
        fused_relu);
#endif
#else
    TP_THROW(NotImplementedError, "conv2d_cuda requires cuDNN");
#endif
}

// Transposed convolution as a backward-data graph.  The engine list this
// formulation reaches holds transforms that the algorithm-enumeration surface
// never offers for the transposed shape, and among them the fastest choice is
// routinely a tiled-transform engine rather than the implicit gemm the
// enumeration ranks first.  Returns an undefined tensor when the case is not
// representable here, so the caller can fall back.
Tensor conv_transpose2d_cudnn_v8(const Tensor& input, const Tensor& weight,
                                 const Tensor& bias,
                                 const std::vector<int64_t>& stride,
                                 const std::vector<int64_t>& padding,
                                 const std::vector<int64_t>& output_padding,
                                 int64_t groups,
                                 const std::vector<int64_t>& dilation) {
#if defined(TP_HAS_CUDNN_FRONTEND)
    if (input.dim() != 4 || weight.dim() != 4) return Tensor();
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float16 &&
        input.dtype() != DType::BFloat16) {
        return Tensor();
    }
    if (bias.defined() && bias.numel() != 0 && bias.dtype() != input.dtype())
        return Tensor();
    // The conv descriptor of this frontend build carries no group count, so a
    // grouped transposed convolution has no graph form here.
    if (groups != 1) return Tensor();

    const int64_t N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    const int64_t K = weight.size(1) * groups;
    const int64_t R = weight.size(2), S = weight.size(3);
    // The output size carries the extra rows and columns that output_padding
    // asks for; the transform writes the whole extent, and those positions
    // receive the same contributions as any other border position.
    const int64_t OH = (H - 1) * stride[0] - 2 * padding[0] +
                       dilation[0] * (R - 1) + output_padding[0] + 1;
    const int64_t OW = (W - 1) * stride[1] - 2 * padding[1] +
                       dilation[1] * (S - 1) + output_padding[1] + 1;
    if (OH <= 0 || OW <= 0) return Tensor();

    cudnnDataType_t dtype;
    if (input.dtype() == DType::Float32) dtype = CUDNN_DATA_FLOAT;
    else if (input.dtype() == DType::Float16) dtype = CUDNN_DATA_HALF;
    else dtype = CUDNN_DATA_BFLOAT16;
    const cudnnDataType_t compute = CUDNN_DATA_FLOAT;

    auto alignment_of = [](const void* ptr) -> int64_t {
        int64_t alignment = 1;
        auto address = reinterpret_cast<uintptr_t>(ptr);
        for (; alignment < 32; alignment *= 2) {
            if (address % static_cast<uintptr_t>(alignment * 2)) return alignment;
        }
        return alignment;
    };

    const std::array<int64_t, 4> dy_stride{
        input.stride(0), input.stride(1), input.stride(2), input.stride(3)};
    const std::array<int64_t, 4> w_stride{
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3)};
    const std::array<int64_t, 4> dx_stride{K * OH * OW, OH * OW, OW, 1};
    Tensor out = Tensor::empty({N, K, OH, OW}, input.dtype(), input.device());

    struct TKey {
        cudnnDataType_t dtype;
        int64_t N, C, H, W, K, R, S, groups;
        int64_t ph, pw, sh, sw, dh, dw;
        int device;
        bool allow_tf32;
        std::array<int64_t, 4> dy_stride, w_stride, dx_stride;
        bool operator==(const TKey& o) const {
            return dtype == o.dtype && N == o.N && C == o.C && H == o.H && W == o.W &&
                   K == o.K && R == o.R && S == o.S && groups == o.groups &&
                   ph == o.ph && pw == o.pw && sh == o.sh && sw == o.sw &&
                   dh == o.dh && dw == o.dw && device == o.device &&
                   allow_tf32 == o.allow_tf32 && dy_stride == o.dy_stride &&
                   w_stride == o.w_stride &&
                   dx_stride == o.dx_stride;
        }
    };
    struct TKeyHash {
        size_t operator()(const TKey& k) const {
            size_t h = std::hash<int64_t>{}(k.N);
            for (auto v : {k.C, k.H, k.W, k.K, k.R, k.S, k.groups, k.ph, k.pw,
                           k.sh, k.sw, k.dh, k.dw})
                h = h * 1000003 ^ std::hash<int64_t>{}(v);
            h = h * 1000003 ^ std::hash<int>{}((int)k.dtype);
            h = h * 1000003 ^ std::hash<int>{}(k.device);
            h = h * 1000003 ^ std::hash<int>{}((int)k.allow_tf32);
            for (auto v : k.dy_stride) h = h * 1000003 ^ std::hash<int64_t>{}(v);
            for (auto v : k.w_stride) h = h * 1000003 ^ std::hash<int64_t>{}(v);
            for (auto v : k.dx_stride) h = h * 1000003 ^ std::hash<int64_t>{}(v);
            return h;
        }
    };
    static std::unordered_map<TKey, std::shared_ptr<fe::ExecutionPlan>, TKeyHash> g_t_cache;
    static std::mutex g_t_mutex;

    const bool allow_tf32 = tensorplay::globalContext().allowTF32CuDNN();
    const bool deterministic = tensorplay::globalContext().deterministicAlgorithms();
    Tensor bias_c;
    if (bias.defined() && bias.numel() != 0) {
        bias_c = bias.is_contiguous() ? bias : bias.contiguous();
        if (bias_c.numel() != K) return Tensor();
        bias_c = bias_c.reshape({1, K, 1, 1});
    }
    TKey key{dtype, N, C, H, W, K, R, S, groups,
             padding[0], padding[1], stride[0], stride[1], dilation[0], dilation[1],
             static_cast<int>(input.device().index()), allow_tf32,
             dy_stride, w_stride, dx_stride};
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();

    std::shared_ptr<fe::ExecutionPlan> plan;
    {
        std::lock_guard<std::mutex> guard(g_t_mutex);
        auto it = g_t_cache.find(key);
        if (it != g_t_cache.end()) {
            plan = it->second;
        } else try {
            const int64_t dy_align = alignment_of(input.data_ptr());
            const int64_t w_align = alignment_of(weight.data_ptr());
            const int64_t dx_align = alignment_of(out.data_ptr());
            auto dy_desc = fe::TensorBuilder()
                               .setDim(4, std::array<int64_t, 4>{N, C, H, W}.data())
                               .setStrides(4, dy_stride.data())
                               .setId('x')
                               .setAlignment(dy_align)
                               .setDataType(dtype)
                               .build();
            // Backward-data filters are laid out (C, K / groups, R, S), which
            // is exactly the transposed-convolution weight layout.
            auto w_desc = fe::TensorBuilder()
                               .setDim(4, std::array<int64_t, 4>{C, K, R, S}.data())
                               .setStrides(4, w_stride.data())
                               .setId('w')
                               .setAlignment(w_align)
                               .setDataType(dtype)
                               .build();
            auto dx_desc = fe::TensorBuilder()
                               .setDim(4, std::array<int64_t, 4>{N, K, OH, OW}.data())
                               .setStrides(4, dx_stride.data())
                               .setId('y')
                               .setAlignment(dx_align)
                               .setDataType(dtype)
                               .build();
            int64_t pad[2] = {padding[0], padding[1]};
            int64_t strd[2] = {stride[0], stride[1]};
            int64_t dil[2] = {dilation[0], dilation[1]};
            auto conv_desc = fe::ConvDescBuilder()
                                 .setComputeType(compute)
                                 .setMathMode(CUDNN_CROSS_CORRELATION)
                                 .setSpatialDimCount(2)
                                 .setSpatialStride(2, strd)
                                 .setPrePadding(2, pad)
                                 .setPostPadding(2, pad)
                                 .setDilation(2, dil)
                                 .build();
            // The bias is applied after the transform rather than inside the
            // graph: a second operation in the graph costs far more than the
            // single pass that adds the vector afterwards.
            auto bwd_op = fe::OperationBuilder(
                              CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
                              .setwDesc(w_desc)
                              .setdyDesc(dy_desc)
                              .setdxDesc(dx_desc)
                              .setcDesc(conv_desc)
                              .build();
            std::array<fe::Operation const*, 1> ops = {&bwd_op};
            auto op_graph = fe::OperationGraphBuilder()
                                .setHandle(handle)
                                .setOperationGraph(ops.size(), ops.data())
                                .build();
            auto heuristics = fe::EngineHeuristicsBuilder()
                                  .setOperationGraph(op_graph)
                                  .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                                  .build();
            auto engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
            auto drop = [=](cudnnBackendDescriptor_t c) {
                if (deterministic &&
                    fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(c)) {
                    return true;
                }
                if (fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(c)) {
                    return true;
                }
                if (dtype == CUDNN_DATA_FLOAT && !allow_tf32 &&
                    fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(c)) {
                    return true;
                }
                return false;
            };
            fe::EngineConfigList kept;
            fe::filter(engine_configs, kept, drop);
            if (kept.empty()) {
                // The instant list can hold tensor-core engines only; the
                // fallback list carries the plain ones float32 without TF32
                // needs.
                auto plain_heuristics = fe::EngineHeuristicsBuilder()
                                            .setOperationGraph(op_graph)
                                            .setHeurMode(CUDNN_HEUR_MODE_FALLBACK)
                                            .build();
                fe::EngineConfigList plain = plain_heuristics.getEngineConfig(
                    plain_heuristics.getEngineConfigCount());
                fe::filter(plain, kept, drop);
            }
            if (kept.empty()) kept = engine_configs;
            for (auto& ec : kept) {
                try {
                    plan = std::make_shared<fe::ExecutionPlan>(
                        fe::ExecutionPlanBuilder().setHandle(handle).setEngineConfig(ec).build());
                    break;
                } catch (...) {
                }
            }
            if (!plan) return Tensor();
            g_t_cache.emplace(key, plan);
        } catch (...) {
            // A formulation this library version refuses must not fail the
            // call; the caller has a second implementation to fall back on.
            return Tensor();
        }
    }

    const size_t workspace_size = plan->getWorkspaceSize();
    auto workspace = getAllocator(DeviceType::CUDA)->allocate(workspace_size ? workspace_size : 1);
    void* data_ptrs[3] = {input.data_ptr(), weight.data_ptr(), out.data_ptr()};
    int64_t uids[3] = {'x', 'w', 'y'};
    auto variant_pack = fe::VariantPackBuilder()
                            .setWorkspacePointer(workspace_size ? workspace.get() : nullptr)
                            .setDataPointers(3, data_ptrs)
                            .setUids(3, uids)
                            .build();
    try {
        CUDNN_CHECK(cudnnBackendExecute(handle, plan->get_raw_desc(),
                                       variant_pack.get_raw_desc()));
    } catch (...) {
        return Tensor();
    }
    if (bias_c.defined() && !add_channel_broadcast_inplace_cuda(out, bias_c)) {
        auto bias_desc = get_cached_tensor_desc(bias_c);
        auto out_desc = get_cached_tensor_desc(out);
        float one = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(handle, &one, *bias_desc, bias_c.data_ptr(),
                                   &one, *out_desc, out.data_ptr()));
    }
    return out;
#else
    return Tensor();
#endif
}

Tensor conv2d_grad_input_cudnn_v8(const Tensor& grad_output, const Tensor& input,
                                  const Tensor& weight,
                                  const std::vector<int64_t>& stride,
                                  const std::vector<int64_t>& padding,
                                  int64_t groups,
                                  const std::vector<int64_t>& dilation) {
#if defined(TP_HAS_CUDNN_FRONTEND)
    if (input.dim() != 4 || weight.dim() != 4 || grad_output.dim() != 4) {
        return Tensor();
    }
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float16 &&
        input.dtype() != DType::BFloat16) {
        return Tensor();
    }
    if (grad_output.dtype() != input.dtype() || weight.dtype() != input.dtype()) {
        return Tensor();
    }
    // The conv descriptor of this frontend build carries no group count, so
    // a grouped backward-data convolution has no graph form here.
    if (groups != 1) return Tensor();

    const int64_t N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    const int64_t K = weight.size(0);
    const int64_t R = weight.size(2), S = weight.size(3);
    const int64_t OH = grad_output.size(2), OW = grad_output.size(3);
    if (OH <= 0 || OW <= 0) return Tensor();

    cudnnDataType_t dtype;
    if (input.dtype() == DType::Float32) dtype = CUDNN_DATA_FLOAT;
    else if (input.dtype() == DType::Float16) dtype = CUDNN_DATA_HALF;
    else dtype = CUDNN_DATA_BFLOAT16;
    const cudnnDataType_t compute = CUDNN_DATA_FLOAT;

    auto alignment_of = [](const void* ptr) -> int64_t {
        int64_t alignment = 1;
        auto address = reinterpret_cast<uintptr_t>(ptr);
        for (; alignment < 32; alignment *= 2) {
            if (address % static_cast<uintptr_t>(alignment * 2)) return alignment;
        }
        return alignment;
    };

    const std::array<int64_t, 4> dy_stride{
        grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3)};
    const std::array<int64_t, 4> w_stride{
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3)};
    const bool use_channels_last =
        is_channels_last_4d(grad_output) || is_channels_last_4d(input);
    const std::array<int64_t, 4> dx_stride = use_channels_last
        ? channels_last_strides(C, H, W)
        : std::array<int64_t, 4>{C * H * W, H * W, W, 1};
    Tensor out = Tensor::empty({N, C, H, W}, input.dtype(), input.device());
    if (use_channels_last) {
        out = out.as_strided(
            {N, C, H, W},
            std::vector<int64_t>(dx_stride.begin(), dx_stride.end()));
    }

    struct BKey {
        cudnnDataType_t dtype;
        int64_t N, C, H, W, K, R, S;
        int64_t ph, pw, sh, sw, dh, dw;
        int device;
        bool allow_tf32;
        std::array<int64_t, 4> dy_stride, w_stride, dx_stride;
        bool operator==(const BKey& o) const {
            return dtype == o.dtype && N == o.N && C == o.C && H == o.H && W == o.W &&
                   K == o.K && R == o.R && S == o.S &&
                   ph == o.ph && pw == o.pw && sh == o.sh && sw == o.sw &&
                   dh == o.dh && dw == o.dw && device == o.device &&
                   allow_tf32 == o.allow_tf32 && dy_stride == o.dy_stride &&
                   w_stride == o.w_stride && dx_stride == o.dx_stride;
        }
    };
    struct BKeyHash {
        size_t operator()(const BKey& k) const {
            size_t h = std::hash<int64_t>{}(k.N);
            for (auto v : {k.C, k.H, k.W, k.K, k.R, k.S, k.ph, k.pw,
                           k.sh, k.sw, k.dh, k.dw})
                h = h * 1000003 ^ std::hash<int64_t>{}(v);
            h = h * 1000003 ^ std::hash<int>{}((int)k.dtype);
            h = h * 1000003 ^ std::hash<int>{}(k.device);
            h = h * 1000003 ^ std::hash<int>{}((int)k.allow_tf32);
            for (auto v : k.dy_stride) h = h * 1000003 ^ std::hash<int64_t>{}(v);
            for (auto v : k.w_stride) h = h * 1000003 ^ std::hash<int64_t>{}(v);
            for (auto v : k.dx_stride) h = h * 1000003 ^ std::hash<int64_t>{}(v);
            return h;
        }
    };
    static std::unordered_map<BKey, std::shared_ptr<fe::ExecutionPlan>, BKeyHash> g_bwd_cache;
    static std::mutex g_bwd_mutex;

    const bool allow_tf32 = tensorplay::globalContext().allowTF32CuDNN();
    const bool deterministic = tensorplay::globalContext().deterministicAlgorithms();

    BKey key{dtype, N, C, H, W, K, R, S,
             padding[0], padding[1], stride[0], stride[1], dilation[0], dilation[1],
             static_cast<int>(input.device().index()), allow_tf32,
             dy_stride, w_stride, dx_stride};
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();

    std::shared_ptr<fe::ExecutionPlan> plan;
    {
        std::lock_guard<std::mutex> guard(g_bwd_mutex);
        auto it = g_bwd_cache.find(key);
        if (it != g_bwd_cache.end()) {
            plan = it->second;
        } else try {
            const int64_t dy_align = alignment_of(grad_output.data_ptr());
            const int64_t w_align = alignment_of(weight.data_ptr());
            const int64_t dx_align = alignment_of(out.data_ptr());
            auto dy_desc = fe::TensorBuilder()
                               .setDim(4, std::array<int64_t, 4>{N, K, OH, OW}.data())
                               .setStrides(4, dy_stride.data())
                               .setId('x')
                               .setAlignment(dy_align)
                               .setDataType(dtype)
                               .build();
            // Backward-data filters are laid out (K, C / groups, R, S); with
            // groups == 1 this is the plain convolution weight layout.
            auto w_desc = fe::TensorBuilder()
                               .setDim(4, std::array<int64_t, 4>{K, C, R, S}.data())
                               .setStrides(4, w_stride.data())
                               .setId('w')
                               .setAlignment(w_align)
                               .setDataType(dtype)
                               .build();
            auto dx_desc = fe::TensorBuilder()
                               .setDim(4, std::array<int64_t, 4>{N, C, H, W}.data())
                               .setStrides(4, dx_stride.data())
                               .setId('y')
                               .setAlignment(dx_align)
                               .setDataType(dtype)
                               .build();
            int64_t pad[2] = {padding[0], padding[1]};
            int64_t strd[2] = {stride[0], stride[1]};
            int64_t dil[2] = {dilation[0], dilation[1]};
            auto conv_desc = fe::ConvDescBuilder()
                                 .setComputeType(compute)
                                 .setMathMode(CUDNN_CROSS_CORRELATION)
                                 .setSpatialDimCount(2)
                                 .setSpatialStride(2, strd)
                                 .setPrePadding(2, pad)
                                 .setPostPadding(2, pad)
                                 .setDilation(2, dil)
                                 .build();
            auto bwd_op = fe::OperationBuilder(
                              CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
                              .setwDesc(w_desc)
                              .setdyDesc(dy_desc)
                              .setdxDesc(dx_desc)
                              .setcDesc(conv_desc)
                              .build();
            std::array<fe::Operation const*, 1> ops = {&bwd_op};
            auto op_graph = fe::OperationGraphBuilder()
                                .setHandle(handle)
                                .setOperationGraph(ops.size(), ops.data())
                                .build();
            auto heuristics = fe::EngineHeuristicsBuilder()
                                  .setOperationGraph(op_graph)
                                  .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                                  .build();
            auto engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
            auto drop = [=](cudnnBackendDescriptor_t c) {
                if (deterministic &&
                    fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(c)) {
                    return true;
                }
                if (fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(c)) {
                    return true;
                }
                if (dtype == CUDNN_DATA_FLOAT && !allow_tf32 &&
                    fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(c)) {
                    return true;
                }
                return false;
            };
            fe::EngineConfigList kept;
            fe::filter(engine_configs, kept, drop);
            if (kept.empty()) {
                // The instant list can hold tensor-core engines only; the
                // fallback list carries the plain ones float32 without TF32
                // needs.
                auto plain_heuristics = fe::EngineHeuristicsBuilder()
                                            .setOperationGraph(op_graph)
                                            .setHeurMode(CUDNN_HEUR_MODE_FALLBACK)
                                            .build();
                fe::EngineConfigList plain = plain_heuristics.getEngineConfig(
                    plain_heuristics.getEngineConfigCount());
                fe::filter(plain, kept, drop);
            }
            if (kept.empty()) kept = engine_configs;
            for (auto& ec : kept) {
                try {
                    plan = std::make_shared<fe::ExecutionPlan>(
                        fe::ExecutionPlanBuilder().setHandle(handle).setEngineConfig(ec).build());
                    break;
                } catch (...) {
                }
            }
            if (!plan) return Tensor();
            g_bwd_cache.emplace(key, plan);
        } catch (...) {
            // A formulation this library version refuses must not fail the
            // call; the caller has a second implementation to fall back on.
            return Tensor();
        }
    }

    const size_t workspace_size = plan->getWorkspaceSize();
    auto workspace = getAllocator(DeviceType::CUDA)->allocate(workspace_size ? workspace_size : 1);
    void* data_ptrs[3] = {grad_output.data_ptr(), weight.data_ptr(), out.data_ptr()};
    int64_t uids[3] = {'x', 'w', 'y'};
    auto variant_pack = fe::VariantPackBuilder()
                            .setWorkspacePointer(workspace_size ? workspace.get() : nullptr)
                            .setDataPointers(3, data_ptrs)
                            .setUids(3, uids)
                            .build();
    try {
        CUDNN_CHECK(cudnnBackendExecute(handle, plan->get_raw_desc(),
                                       variant_pack.get_raw_desc()));
    } catch (...) {
        return Tensor();
    }
    return out;
#else
    return Tensor();
#endif
}

Tensor conv2d_grad_weight_cudnn_v8(const Tensor& grad_output, const Tensor& input,
                                   const Tensor& weight,
                                   const std::vector<int64_t>& stride,
                                   const std::vector<int64_t>& padding,
                                   int64_t groups,
                                   const std::vector<int64_t>& dilation) {
#if defined(TP_HAS_CUDNN_FRONTEND)
    if (input.dim() != 4 || weight.dim() != 4 || grad_output.dim() != 4) {
        return Tensor();
    }
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float16 &&
        input.dtype() != DType::BFloat16) {
        return Tensor();
    }
    if (grad_output.dtype() != input.dtype() || weight.dtype() != input.dtype()) {
        return Tensor();
    }
    // The conv descriptor of this frontend build carries no group count, so
    // a grouped backward-filter convolution has no graph form here.
    if (groups != 1) return Tensor();

    const int64_t N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    const int64_t K = weight.size(0);
    const int64_t R = weight.size(2), S = weight.size(3);
    const int64_t OH = grad_output.size(2), OW = grad_output.size(3);
    if (OH <= 0 || OW <= 0) return Tensor();

    cudnnDataType_t dtype;
    if (input.dtype() == DType::Float32) dtype = CUDNN_DATA_FLOAT;
    else if (input.dtype() == DType::Float16) dtype = CUDNN_DATA_HALF;
    else dtype = CUDNN_DATA_BFLOAT16;
    const cudnnDataType_t compute = CUDNN_DATA_FLOAT;

    auto alignment_of = [](const void* ptr) -> int64_t {
        int64_t alignment = 1;
        auto address = reinterpret_cast<uintptr_t>(ptr);
        for (; alignment < 32; alignment *= 2) {
            if (address % static_cast<uintptr_t>(alignment * 2)) return alignment;
        }
        return alignment;
    };

    const std::array<int64_t, 4> x_stride{
        input.stride(0), input.stride(1), input.stride(2), input.stride(3)};
    const std::array<int64_t, 4> dy_stride{
        grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3)};
    const std::array<int64_t, 4> dw_stride{C * R * S, R * S, S, 1};
    Tensor out = Tensor::empty({K, C, R, S}, input.dtype(), input.device());

    struct WKey {
        cudnnDataType_t dtype;
        int64_t N, C, H, W, K, R, S;
        int64_t ph, pw, sh, sw, dh, dw;
        int device;
        bool allow_tf32;
        std::array<int64_t, 4> x_stride, dy_stride, dw_stride;
        bool operator==(const WKey& o) const {
            return dtype == o.dtype && N == o.N && C == o.C && H == o.H && W == o.W &&
                   K == o.K && R == o.R && S == o.S &&
                   ph == o.ph && pw == o.pw && sh == o.sh && sw == o.sw &&
                   dh == o.dh && dw == o.dw && device == o.device &&
                   allow_tf32 == o.allow_tf32 && x_stride == o.x_stride &&
                   dy_stride == o.dy_stride && dw_stride == o.dw_stride;
        }
    };
    struct WKeyHash {
        size_t operator()(const WKey& k) const {
            size_t h = std::hash<int64_t>{}(k.N);
            for (auto v : {k.C, k.H, k.W, k.K, k.R, k.S, k.ph, k.pw,
                           k.sh, k.sw, k.dh, k.dw})
                h = h * 1000003 ^ std::hash<int64_t>{}(v);
            h = h * 1000003 ^ std::hash<int>{}((int)k.dtype);
            h = h * 1000003 ^ std::hash<int>{}(k.device);
            h = h * 1000003 ^ std::hash<int>{}((int)k.allow_tf32);
            for (auto v : k.x_stride) h = h * 1000003 ^ std::hash<int64_t>{}(v);
            for (auto v : k.dy_stride) h = h * 1000003 ^ std::hash<int64_t>{}(v);
            for (auto v : k.dw_stride) h = h * 1000003 ^ std::hash<int64_t>{}(v);
            return h;
        }
    };
    static std::unordered_map<WKey, std::shared_ptr<fe::ExecutionPlan>, WKeyHash> g_w_cache;
    static std::mutex g_w_mutex;

    const bool allow_tf32 = tensorplay::globalContext().allowTF32CuDNN();
    const bool deterministic = tensorplay::globalContext().deterministicAlgorithms();

    WKey key{dtype, N, C, H, W, K, R, S,
             padding[0], padding[1], stride[0], stride[1], dilation[0], dilation[1],
             static_cast<int>(input.device().index()), allow_tf32,
             x_stride, dy_stride, dw_stride};
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();

    std::shared_ptr<fe::ExecutionPlan> plan;
    {
        std::lock_guard<std::mutex> guard(g_w_mutex);
        auto it = g_w_cache.find(key);
        if (it != g_w_cache.end()) {
            plan = it->second;
        } else try {
            const int64_t x_align = alignment_of(input.data_ptr());
            const int64_t dy_align = alignment_of(grad_output.data_ptr());
            const int64_t dw_align = alignment_of(out.data_ptr());
            auto x_desc = fe::TensorBuilder()
                              .setDim(4, std::array<int64_t, 4>{N, C, H, W}.data())
                              .setStrides(4, x_stride.data())
                              .setId('x')
                              .setAlignment(x_align)
                              .setDataType(dtype)
                              .build();
            auto dy_desc = fe::TensorBuilder()
                               .setDim(4, std::array<int64_t, 4>{N, K, OH, OW}.data())
                               .setStrides(4, dy_stride.data())
                               .setId('y')
                               .setAlignment(dy_align)
                               .setDataType(dtype)
                               .build();
            // Backward-filter gradients are laid out (K, C / groups, R, S);
            // with groups == 1 this is the plain convolution weight layout.
            auto dw_desc = fe::TensorBuilder()
                               .setDim(4, std::array<int64_t, 4>{K, C, R, S}.data())
                               .setStrides(4, dw_stride.data())
                               .setId('w')
                               .setAlignment(dw_align)
                               .setDataType(dtype)
                               .build();
            int64_t pad[2] = {padding[0], padding[1]};
            int64_t strd[2] = {stride[0], stride[1]};
            int64_t dil[2] = {dilation[0], dilation[1]};
            auto conv_desc = fe::ConvDescBuilder()
                                 .setComputeType(compute)
                                 .setMathMode(CUDNN_CROSS_CORRELATION)
                                 .setSpatialDimCount(2)
                                 .setSpatialStride(2, strd)
                                 .setPrePadding(2, pad)
                                 .setPostPadding(2, pad)
                                 .setDilation(2, dil)
                                 .build();
            auto bwd_op = fe::OperationBuilder(
                              CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR)
                              .setxDesc(x_desc)
                              .setdyDesc(dy_desc)
                              .setdwDesc(dw_desc)
                              .setcDesc(conv_desc)
                              .build();
            std::array<fe::Operation const*, 1> ops = {&bwd_op};
            auto op_graph = fe::OperationGraphBuilder()
                                .setHandle(handle)
                                .setOperationGraph(ops.size(), ops.data())
                                .build();
            auto heuristics = fe::EngineHeuristicsBuilder()
                                  .setOperationGraph(op_graph)
                                  .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                                  .build();
            auto engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
            auto drop = [=](cudnnBackendDescriptor_t c) {
                if (deterministic &&
                    fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(c)) {
                    return true;
                }
                if (fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(c)) {
                    return true;
                }
                if (dtype == CUDNN_DATA_FLOAT && !allow_tf32 &&
                    fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(c)) {
                    return true;
                }
                return false;
            };
            fe::EngineConfigList kept;
            fe::filter(engine_configs, kept, drop);
            if (kept.empty()) {
                // The instant list can hold tensor-core engines only; the
                // fallback list carries the plain ones float32 without TF32
                // needs.
                auto plain_heuristics = fe::EngineHeuristicsBuilder()
                                            .setOperationGraph(op_graph)
                                            .setHeurMode(CUDNN_HEUR_MODE_FALLBACK)
                                            .build();
                fe::EngineConfigList plain = plain_heuristics.getEngineConfig(
                    plain_heuristics.getEngineConfigCount());
                fe::filter(plain, kept, drop);
            }
            if (kept.empty()) kept = engine_configs;
            for (auto& ec : kept) {
                try {
                    plan = std::make_shared<fe::ExecutionPlan>(
                        fe::ExecutionPlanBuilder().setHandle(handle).setEngineConfig(ec).build());
                    break;
                } catch (...) {
                }
            }
            if (!plan) return Tensor();
            g_w_cache.emplace(key, plan);
        } catch (...) {
            return Tensor();
        }
    }

    const size_t workspace_size = plan->getWorkspaceSize();
    auto workspace = getAllocator(DeviceType::CUDA)->allocate(workspace_size ? workspace_size : 1);
    void* data_ptrs[3] = {input.data_ptr(), grad_output.data_ptr(), out.data_ptr()};
    int64_t uids[3] = {'x', 'y', 'w'};
    auto variant_pack = fe::VariantPackBuilder()
                            .setWorkspacePointer(workspace_size ? workspace.get() : nullptr)
                            .setDataPointers(3, data_ptrs)
                            .setUids(3, uids)
                            .build();
    try {
        CUDNN_CHECK(cudnnBackendExecute(handle, plan->get_raw_desc(),
                                       variant_pack.get_raw_desc()));
    } catch (...) {
        return Tensor();
    }
    return out;
#else
    return Tensor();
#endif
}

Tensor conv2d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias,
                   const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                   const std::vector<int64_t>& dilation, int64_t groups) {
    // Reduced-precision ungrouped calls run in the channel-major order: the
    // row-major plan for them interleaves repacking kernels around the
    // convolution, while here the operands are repacked once and the result
    // comes back in the order the plan writes it.  A bias rides inside the
    // plan in that order instead of a separate pointwise kernel.  The
    // graph-only conversion is compiled out where the graph path is, since
    // the descriptor fallback spells no filter strides.
#if defined(TP_HAS_CUDNN_FRONTEND)
    if (groups == 1 && conv_operand_repackable(input, false) &&
        conv_operand_repackable(weight, true)) {
        return conv2d_cuda_impl(conv_to_channel_major(input),
                                conv_to_channel_major(weight), bias, stride,
                                padding, dilation, groups, false);
    }
#endif
    return conv2d_cuda_impl(input, weight, bias, stride, padding, dilation, groups, false);
}

// Keep the fused IR contract shared with CPU.  The cuDNN frontend plan owns
// the Conv(+bias)->ReLU graph, so this path launches one backend plan rather
// than a convolution followed by a separate pointwise kernel.
Tensor conv2d_relu_cuda(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt,
                        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                        const std::vector<int64_t>& dilation, int64_t groups) {
    const Tensor bias = bias_opt.has_value() ? *bias_opt : Tensor();
#if defined(TP_HAS_CUDNN_FRONTEND)
    if (groups == 1 && conv_operand_repackable(input, false) &&
        conv_operand_repackable(weight, true)) {
        return conv2d_cuda_impl(conv_to_channel_major(input),
                                conv_to_channel_major(weight), bias, stride,
                                padding, dilation, groups, true);
    }
#endif
    return conv2d_cuda_impl(input, weight, bias, stride, padding, dilation, groups, true);
}

Tensor conv2d_grad_input_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, int64_t groups) {
#if defined(USE_ROCM)
    if (input.dtype() == DType::Float64) {
        return conv2d_slow_fp64_grad_input(grad_output, input, weight, stride_arg,
                                           padding_arg, dilation_arg, groups);
    }
#endif
#ifdef USE_CUDNN
    auto stride = expand_param_if_needed(stride_arg, 2, 1);
    auto padding = expand_param_if_needed(padding_arg, 2, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 2, 1);

    // Backward grads can arrive as broadcast views (e.g. after .sum()); the
    // cuDNN kernels here read either a contiguous NCHW tensor or a
    // channels-last one, and anything else is copied to a contiguous form.
    const DType result_dtype = input.dtype();
    const bool grad_out_cl = is_channels_last_4d(grad_output);
    const bool input_cl = is_channels_last_4d(input);
    Tensor grad_output_c = (grad_output.is_contiguous() || grad_out_cl)
                               ? grad_output
                               : grad_output.contiguous();
    Tensor input_c = (input.is_contiguous() || input_cl)
                        ? input
                        : input.contiguous();
    // A channel-major filter is read as it lies by the graph path; the
    // descriptor path below renormalizes it.
    Tensor weight_c = (weight.is_contiguous() || is_channels_last_4d(weight))
                          ? weight
                          : weight.contiguous();

    // A row-major reduced-precision gradient runs the data gradient in the
    // channel-major order: the repacks are one kernel per operand, where the
    // row-major plan would pay for repacks around every engine read.  The
    // data gradient then comes back in that order, dense and readable
    // either way.  The graph path spells every operand's strides, so the
    // channel-major filter is read as it lies; the legacy descriptor path,
    // which spells no filter strides, renormalizes its operands below.
    if (groups == 1 && conv_operand_repackable(grad_output_c, false)) {
        grad_output_c = conv_to_channel_major(grad_output_c);
    }
    if (groups == 1 && is_channels_last_4d(grad_output_c) &&
        conv_operand_repackable(weight_c, true)) {
        weight_c = conv_to_channel_major(weight_c);
    }

#if defined(TP_HAS_CUDNN_FRONTEND)
    // The graph path is the primary route for the ungrouped reduced-precision
    // and float32 cases: its engine selection has no frequency-domain member,
    // so the backward-data gradient never pays for an FFT the legacy heuristic
    // picks.  The legacy descriptor path remains for every shape the graph
    // cannot express.
    Tensor v8 = conv2d_grad_input_cudnn_v8(
        grad_output_c, input_c, weight_c, stride, padding, groups, dilation);
    if (v8.defined()) {
        return v8.dtype() == result_dtype ? v8 : v8.to(result_dtype);
    }
#endif

    const DType compute_dtype = weight_c.dtype();
    if (input_c.dtype() != compute_dtype) input_c = input_c.to(compute_dtype);
    if (grad_output_c.dtype() != compute_dtype) {
        grad_output_c = grad_output_c.to(compute_dtype);
    }

    // The descriptor algorithms take every activation's order from the
    // filter descriptor, which is always spelled row-major; anything still
    // in the channel-major order here is repacked back so the descriptors
    // name the memory they are handed.
    if (is_channels_last_4d(grad_output_c)) grad_output_c = grad_output_c.contiguous();
    if (is_channels_last_4d(input_c)) input_c = input_c.contiguous();
    if (is_channels_last_4d(weight_c)) weight_c = weight_c.contiguous();

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();

    auto dx_desc = get_cached_tensor_desc(input_c); // gradient of input has same shape as input
    auto w_desc = get_cached_filter_desc(weight_c);
    auto dy_desc = get_cached_tensor_desc(grad_output_c);
    
    auto conv_desc = get_cached_conv_desc(
        (int)padding[0], (int)padding[1], (int)stride[0], (int)stride[1],
        (int)dilation[0], (int)dilation[1], (int)groups, input_c.dtype());
    
    Tensor grad_input = Tensor::empty_like(input_c, DType::Undefined, input_c.device());
    
    ConvBwdKey cache_key = make_conv_bwd_key(
        0, input_c, weight_c, grad_output_c, stride, padding, dilation, groups);
    cudnnConvolutionBwdDataAlgo_t algo;
    size_t workspace_size;
    {
        bool have = false;
        {
            std::lock_guard<std::mutex> lock(g_conv_bwd_cache_mutex);
            auto it = g_conv_bwd_algo_cache.find(cache_key);
            if (it != g_conv_bwd_algo_cache.end()) {
                algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(it->second.algorithm);
                workspace_size = it->second.workspace_size;
                have = true;
            }
        }
        if (!have) {
            ConvBwdAlgo entry;
            if (conv_autotune_enabled()) {
                entry = autotune_conv_bwd_data(handle, *w_desc, weight_c.data_ptr(),
                                               *dy_desc, grad_output_c.data_ptr(), *conv_desc,
                                               *dx_desc, grad_input.data_ptr(), input_c.device());
            } else {
                // Ask for the single best candidate rather than the whole list.
                // The list is in enumeration order, and the frequency-domain
                // algorithms sit early in it, so taking the first entry that
                // reports success out of the full list picks one of those over
                // the cheaper implicit-gemm ones.  The top candidate is what
                // the ranking is for; the full list is the fallback for when it
                // cannot be built or run, and that pass starts after it so the
                // candidate already tried is not tried twice.
                cudnnConvolutionBwdDataAlgoPerf_t perf_results[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
                auto usable = [&](const cudnnConvolutionBwdDataAlgoPerf_t& p) {
                    return p.status == CUDNN_STATUS_SUCCESS;
                };
                auto record = [&](int index) {
                    size_t ws = 0;
                    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
                        handle, *w_desc, *dy_desc, *conv_desc, *dx_desc,
                        perf_results[index].algo, &ws));
                    entry = ConvBwdAlgo{static_cast<int>(perf_results[index].algo), ws};
                };

                int returned_algo_count = 0;
                CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                    handle, *w_desc, *dy_desc, *conv_desc, *dx_desc,
                    1, &returned_algo_count, perf_results));
                bool found = false;
                if (returned_algo_count > 0 && usable(perf_results[0])) {
                    record(0);
                    found = true;
                }
                if (!found) {
                    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                        handle, *w_desc, *dy_desc, *conv_desc, *dx_desc,
                        CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT,
                        &returned_algo_count, perf_results));
                    for (int i = 0; i < returned_algo_count; ++i) {
                        if (usable(perf_results[i])) {
                            record(i);
                            found = true;
                            break;
                        }
                    }
                }
                if (!found) {
                    TP_THROW(RuntimeError, "cuDNN: no backward-data convolution algorithm");
                }
            }
            {
                std::lock_guard<std::mutex> lock(g_conv_bwd_cache_mutex);
                g_conv_bwd_algo_cache.emplace(cache_key, entry);
            }
            algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(entry.algorithm);
            workspace_size = entry.workspace_size;
        }
    }
    
    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        workspace_size, input_c.device());
    
    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void *alpha_p = &alpha, *beta_p = &beta;
    if (input_c.dtype() == DType::Float64) {
        alpha_p = &alpha_d; beta_p = &beta_d;
    }
    
    CUDNN_CHECK(cudnnConvolutionBackwardData(handle, alpha_p, *w_desc, weight_c.data_ptr(), *dy_desc, grad_output_c.data_ptr(), *conv_desc, algo, workspace.get(), workspace_size, beta_p, *dx_desc, grad_input.data_ptr()));
    
    return grad_input.dtype() == result_dtype ? grad_input : grad_input.to(result_dtype);
#else
    TP_THROW(NotImplementedError, "conv2d_grad_input_cuda requires cuDNN");
#endif
}

Tensor conv2d_grad_weight_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, int64_t groups) {
#if defined(USE_ROCM)
    if (input.dtype() == DType::Float64) {
        return conv2d_slow_fp64_grad_weight(grad_output, input, weight, stride_arg,
                                            padding_arg, dilation_arg, groups);
    }
#endif
#ifdef USE_CUDNN
    auto stride = expand_param_if_needed(stride_arg, 2, 1);
    auto padding = expand_param_if_needed(padding_arg, 2, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 2, 1);

    const bool grad_out_cl = is_channels_last_4d(grad_output);
    const bool input_cl = is_channels_last_4d(input);
    Tensor grad_output_c = (grad_output.is_contiguous() || grad_out_cl)
                               ? grad_output
                               : grad_output.contiguous();
    Tensor input_c = (input.is_contiguous() || input_cl)
                        ? input
                        : input.contiguous();
    // Only the filter's shape is read here; the descriptor path below
    // renormalizes a channel-major one.
    Tensor weight_c = (weight.is_contiguous() || is_channels_last_4d(weight))
                          ? weight
                          : weight.contiguous();

    // The filter gradient reads both operands; running it in the
    // channel-major order when they arrive row-major in reduced precision
    // trades the plans' own repacking for one kernel per operand.  The
    // gradient of the filter itself stays row-major either way.
    if (groups == 1 && conv_operand_repackable(grad_output_c, false) &&
        conv_operand_repackable(input_c, false)) {
        grad_output_c = conv_to_channel_major(grad_output_c);
        input_c = conv_to_channel_major(input_c);
    }

#if defined(TP_HAS_CUDNN_FRONTEND)
    // The graph path is the primary route for the ungrouped reduced-precision
    // and float32 cases: its engine selection has no frequency-domain member,
    // so the backward-filter gradient never pays for an FFT the legacy
    // heuristic picks.  The legacy descriptor path remains for every shape the
    // graph cannot express.
    Tensor v8 = conv2d_grad_weight_cudnn_v8(
        grad_output_c, input_c, weight_c, stride, padding, groups, dilation);
    if (v8.defined()) {
        return v8.dtype() == weight.dtype() ? v8 : v8.to(weight.dtype());
    }
#endif

    const DType compute_dtype = weight_c.dtype();
    if (input_c.dtype() != compute_dtype) input_c = input_c.to(compute_dtype);
    if (grad_output_c.dtype() != compute_dtype) {
        grad_output_c = grad_output_c.to(compute_dtype);
    }

    // The descriptor algorithms take every activation's order from the
    // filter descriptor, which is always spelled row-major; anything still
    // in the channel-major order here is repacked back so the descriptors
    // name the memory they are handed.
    if (is_channels_last_4d(grad_output_c)) grad_output_c = grad_output_c.contiguous();
    if (is_channels_last_4d(input_c)) input_c = input_c.contiguous();
    if (is_channels_last_4d(weight_c)) weight_c = weight_c.contiguous();

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();

    auto x_desc = get_cached_tensor_desc(input_c);
    auto dy_desc = get_cached_tensor_desc(grad_output_c);
    
    auto conv_desc = get_cached_conv_desc(
        (int)padding[0], (int)padding[1], (int)stride[0], (int)stride[1],
        (int)dilation[0], (int)dilation[1], (int)groups, input_c.dtype());
    
    Tensor grad_weight_compute = Tensor::empty_like(
        weight_c, compute_dtype, weight_c.device());
    auto dw_desc = get_cached_filter_desc(grad_weight_compute);
    
    ConvBwdKey cache_key = make_conv_bwd_key(
        1, input_c, weight_c, grad_output_c, stride, padding, dilation, groups);
    cudnnConvolutionBwdFilterAlgo_t algo;
    size_t workspace_size;
    {
        bool have = false;
        {
            std::lock_guard<std::mutex> lock(g_conv_bwd_cache_mutex);
            auto it = g_conv_bwd_algo_cache.find(cache_key);
            if (it != g_conv_bwd_algo_cache.end()) {
                algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(it->second.algorithm);
                workspace_size = it->second.workspace_size;
                have = true;
            }
        }
        if (!have) {
            ConvBwdAlgo entry;
            if (conv_autotune_enabled()) {
                entry = autotune_conv_bwd_filter(handle, *x_desc, input_c.data_ptr(),
                                                 *dy_desc, grad_output_c.data_ptr(), *conv_desc,
                                                 *dw_desc, grad_weight_compute.data_ptr(), input_c.device());
            } else {
                cudnnConvolutionBwdFilterAlgoPerf_t perf_results;
                int returned_algo_count = 0;
                CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                    handle, *x_desc, *dy_desc, *conv_desc, *dw_desc,
                    1, &returned_algo_count, &perf_results));
                if (returned_algo_count == 0) {
                    TP_THROW(RuntimeError, "cuDNN: no backward-filter convolution algorithm");
                }
                size_t ws_size = 0;
                CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                    handle, *x_desc, *dy_desc, *conv_desc, *dw_desc, perf_results.algo, &ws_size));
                entry = ConvBwdAlgo{static_cast<int>(perf_results.algo), ws_size};
            }
            {
                std::lock_guard<std::mutex> lock(g_conv_bwd_cache_mutex);
                g_conv_bwd_algo_cache.emplace(cache_key, entry);
            }
            algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(entry.algorithm);
            workspace_size = entry.workspace_size;
        }
    }
    
    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        workspace_size, input_c.device());
    
    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void *alpha_p = &alpha, *beta_p = &beta;
    if (input_c.dtype() == DType::Float64) {
        alpha_p = &alpha_d; beta_p = &beta_d;
    }
    
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle, alpha_p, *x_desc, input_c.data_ptr(), *dy_desc, grad_output_c.data_ptr(), *conv_desc, algo, workspace.get(), workspace_size, beta_p, *dw_desc, grad_weight_compute.data_ptr()));
    
    const DType result_dtype = weight.dtype();
    return grad_weight_compute.dtype() == result_dtype
        ? grad_weight_compute
        : grad_weight_compute.to(result_dtype);
#else
    TP_THROW(NotImplementedError, "conv2d_grad_weight_cuda requires cuDNN");
#endif
}

Tensor conv2d_grad_bias_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, int64_t groups) {
#if defined(USE_ROCM)
    if (input.dtype() == DType::Float64) {
        return conv2d_slow_fp64_grad_bias(grad_output, input, weight, stride, padding,
                                          dilation, groups);
    }
#endif
#ifdef USE_CUDNN
    Tensor grad_output_c = grad_output;
    const DType result_dtype = weight.dtype();
    const bool preserve_half =
        grad_output_c.dtype() == DType::Float16 &&
        (result_dtype == DType::Float16 || result_dtype == DType::Float32);
    if (!preserve_half && grad_output_c.dtype() != result_dtype) {
        grad_output_c = grad_output_c.to(result_dtype);
    }
    const int64_t batch = grad_output_c.size(0);
    const int64_t channels = grad_output_c.size(1);
    const int64_t height = grad_output_c.size(2);
    const int64_t width = grad_output_c.size(3);
    const int64_t spatial = height * width;
    // The bias gradient reduces over batch and spatial positions. The generic
    // reduction engine handles the strided layouts and the vectorization and
    // split decisions, so the dedicated kernels below only serve the
    // half-in/float-out mixed-width combination, which a same-width sum
    // cannot express without an extra full-tensor conversion.
    if (grad_output_c.dtype() == result_dtype) {
        return grad_output_c.sum(std::vector<int64_t>{0, 2, 3});
    }
    int threads = 32;
    while (threads < spatial && threads < 512) threads <<= 1;
    Tensor grad_bias = Tensor::empty({channels}, result_dtype, grad_output_c.device());
    const auto stream = getCurrentCUDAStream().stream();
    const bool strided = !grad_output_c.is_contiguous();
    if (strided && grad_output_c.dim() == 4) {
        const int64_t stride_n = grad_output_c.stride(0);
        const int64_t stride_c = grad_output_c.stride(1);
        const int64_t stride_h = grad_output_c.stride(2);
        const int64_t stride_w = grad_output_c.stride(3);
        switch (grad_output_c.dtype()) {
            case DType::Float16:
                if (result_dtype == DType::Float32) {
                    conv2d_grad_bias_strided_kernel<Half, float, float>
                        <<<static_cast<unsigned>(channels), threads, 0, stream>>>(
                            grad_output_c.data_ptr<Half>(), batch, channels, height, width,
                            stride_n, stride_c, stride_h, stride_w,
                            grad_bias.data_ptr<float>());
                } else {
                    conv2d_grad_bias_strided_kernel<Half, float, Half>
                        <<<static_cast<unsigned>(channels), threads, 0, stream>>>(
                            grad_output_c.data_ptr<Half>(), batch, channels, height, width,
                            stride_n, stride_c, stride_h, stride_w,
                            grad_bias.data_ptr<Half>());
                }
                break;
            case DType::BFloat16:
                conv2d_grad_bias_strided_kernel<BFloat16, float, BFloat16>
                    <<<static_cast<unsigned>(channels), threads, 0, stream>>>(
                        grad_output_c.data_ptr<BFloat16>(), batch, channels, height, width,
                        stride_n, stride_c, stride_h, stride_w,
                        grad_bias.data_ptr<BFloat16>());
                break;
            case DType::Float32:
                conv2d_grad_bias_strided_kernel<float, float, float>
                    <<<static_cast<unsigned>(channels), threads, 0, stream>>>(
                        grad_output_c.data_ptr<float>(), batch, channels, height, width,
                        stride_n, stride_c, stride_h, stride_w,
                        grad_bias.data_ptr<float>());
                break;
            case DType::Float64:
                conv2d_grad_bias_strided_kernel<double, double, double>
                    <<<static_cast<unsigned>(channels), threads, 0, stream>>>(
                        grad_output_c.data_ptr<double>(), batch, channels, height, width,
                        stride_n, stride_c, stride_h, stride_w,
                        grad_bias.data_ptr<double>());
                break;
            default:
                TP_THROW(NotImplementedError, "conv2d_grad_bias_cuda: unsupported dtype");
        }
        TP_CONV_CUDA_CHECK(cudaGetLastError());
        return grad_bias;
    }
    if (!grad_output_c.is_contiguous()) grad_output_c = grad_output_c.contiguous();
    switch (grad_output_c.dtype()) {
        case DType::Float16:
            if (result_dtype == DType::Float32) {
                const bool vectorizable =
                    (spatial % 8) == 0 &&
                    (reinterpret_cast<uintptr_t>(grad_output_c.data_ptr()) & 15u) == 0;
                if (vectorizable) {
                    const int64_t vector_count = spatial / 8;
                    int vector_threads = 32;
                    while (vector_threads < vector_count && vector_threads < 512) {
                        vector_threads <<= 1;
                    }
                    conv2d_grad_bias_vec_half_kernel<float>
                        <<<static_cast<unsigned>(channels), vector_threads, 0, stream>>>(
                            grad_output_c.data_ptr<Half>(), batch, channels, spatial,
                            grad_bias.data_ptr<float>());
                } else if ((spatial & 1) == 0 &&
                           (reinterpret_cast<uintptr_t>(grad_output_c.data_ptr()) & 3u) == 0) {
                    const int64_t pair_count = spatial / 2;
                    int pair_threads = 32;
                    while (pair_threads < pair_count && pair_threads < 512) {
                        pair_threads <<= 1;
                    }
                    conv2d_grad_bias_half2_kernel<float>
                        <<<static_cast<unsigned>(channels), pair_threads, 0, stream>>>(
                            grad_output_c.data_ptr<Half>(), batch, channels, spatial,
                            grad_bias.data_ptr<float>());
                } else {
                    conv2d_grad_bias_reduce_kernel<Half, float, float>
                        <<<static_cast<unsigned>(channels), threads, 0, stream>>>(
                            grad_output_c.data_ptr<Half>(), batch, channels, spatial,
                            grad_bias.data_ptr<float>());
                }
            } else {
                const bool vectorizable =
                    (spatial % 8) == 0 &&
                    (reinterpret_cast<uintptr_t>(grad_output_c.data_ptr()) & 15u) == 0;
                if (vectorizable) {
                    const int64_t vector_count = spatial / 8;
                    int vector_threads = 32;
                    while (vector_threads < vector_count && vector_threads < 512) {
                        vector_threads <<= 1;
                    }
                    conv2d_grad_bias_vec_half_kernel<Half>
                        <<<static_cast<unsigned>(channels), vector_threads, 0, stream>>>(
                            grad_output_c.data_ptr<Half>(), batch, channels, spatial,
                            grad_bias.data_ptr<Half>());
                } else if ((spatial & 1) == 0 &&
                           (reinterpret_cast<uintptr_t>(grad_output_c.data_ptr()) & 3u) == 0) {
                    const int64_t pair_count = spatial / 2;
                    int pair_threads = 32;
                    while (pair_threads < pair_count && pair_threads < 512) {
                        pair_threads <<= 1;
                    }
                    conv2d_grad_bias_half2_kernel<Half>
                        <<<static_cast<unsigned>(channels), pair_threads, 0, stream>>>(
                            grad_output_c.data_ptr<Half>(), batch, channels, spatial,
                            grad_bias.data_ptr<Half>());
                } else {
                    conv2d_grad_bias_reduce_kernel<Half, float, Half>
                        <<<static_cast<unsigned>(channels), threads, 0, stream>>>(
                            grad_output_c.data_ptr<Half>(), batch, channels, spatial,
                            grad_bias.data_ptr<Half>());
                }
            }
            break;
        case DType::BFloat16:
            conv2d_grad_bias_reduce_kernel<BFloat16, float, BFloat16>
                <<<static_cast<unsigned>(channels), threads, 0, stream>>>(
                    grad_output_c.data_ptr<BFloat16>(), batch, channels, spatial,
                    grad_bias.data_ptr<BFloat16>());
            break;
        case DType::Float32:
            conv2d_grad_bias_reduce_kernel<float, float, float>
                <<<static_cast<unsigned>(channels), threads, 0, stream>>>(
                    grad_output_c.data_ptr<float>(), batch, channels, spatial,
                    grad_bias.data_ptr<float>());
            break;
        case DType::Float64:
            conv2d_grad_bias_reduce_kernel<double, double, double>
                <<<static_cast<unsigned>(channels), threads, 0, stream>>>(
                    grad_output_c.data_ptr<double>(), batch, channels, spatial,
                    grad_bias.data_ptr<double>());
            break;
        default:
            TP_THROW(NotImplementedError, "conv2d_grad_bias_cuda: unsupported dtype");
    }
    TP_CONV_CUDA_CHECK(cudaGetLastError());
    return grad_bias;
#else
    TP_THROW(NotImplementedError, "conv2d_grad_bias_cuda requires cuDNN");
#endif
}

// =========================================================================
// =========================================================================

TENSORPLAY_LIBRARY_IMPL(CUDA, ConvKernels) {
    m.impl("conv2d", conv2d_cuda);
    m.impl("conv2d_relu", conv2d_relu_cuda);
    m.impl("conv2d_grad_input", conv2d_grad_input_cuda);
    m.impl("conv2d_grad_weight", conv2d_grad_weight_cuda);
    m.impl("conv2d_grad_bias", conv2d_grad_bias_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
