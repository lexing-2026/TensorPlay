#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "CUDAContext.h"
#include "CUDNNUtils.h"
#include "Allocator.h"
#include <vector>
#include <array>
#include <iostream>
#include <unordered_map>
#include <string>
#include <mutex>
#include <memory>

#ifdef USE_CUDNN
#include <cudnn.h>
#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;
#endif

namespace tensorplay {
namespace cuda {

#ifdef USE_CUDNN
Tensor& relu_inplace_kernel_cudnn(Tensor& self);
#endif

namespace {
    std::vector<int64_t> expand_param_if_needed(const std::vector<int64_t>& list, int64_t n, int64_t default_val) {
        if (list.empty()) return std::vector<int64_t>(n, default_val);
        if (list.size() == 1) return std::vector<int64_t>(n, list[0]);
        if (list.size() != n) TP_THROW(ValueError, "Parameter size mismatch");
        return list;
    }

    bool is_channels_last_4d(const Tensor& tensor) {
        if (tensor.dim() != 4) return false;
        const int64_t c = tensor.size(1);
        const int64_t h = tensor.size(2);
        const int64_t w = tensor.size(3);
        return tensor.stride(0) == c * h * w &&
               tensor.stride(1) == 1 &&
               tensor.stride(2) == w * c &&
               tensor.stride(3) == c;
    }

    std::array<int64_t, 4> channels_last_strides(
        int64_t c,
        int64_t h,
        int64_t w) {
        return {c * h * w, 1, w * c, c};
    }

    Tensor empty_conv_output(
        int64_t n,
        int64_t c,
        int64_t h,
        int64_t w,
        DType dtype,
        const Device& device,
        bool channels_last) {
        const std::vector<int64_t> shape{n, c, h, w};
        Tensor result = Tensor::empty(shape, dtype, device);
        if (!channels_last) return result;
        const auto strides = channels_last_strides(c, h, w);
        return result.as_strided(
            shape, std::vector<int64_t>(strides.begin(), strides.end()));
    }
}

#ifdef USE_CUDNN

// RAII Wrappers for cuDNN descriptors
struct TensorDesc {
    cudnnTensorDescriptor_t desc;
    TensorDesc() { CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc)); }
    ~TensorDesc() { cudnnDestroyTensorDescriptor(desc); }
    operator cudnnTensorDescriptor_t() const { return desc; }
    
    void set(const Tensor& t) {
        cudnnDataType_t dtype;
        if (t.dtype() == DType::Float32) dtype = CUDNN_DATA_FLOAT;
        else if (t.dtype() == DType::Float64) dtype = CUDNN_DATA_DOUBLE;
        else TP_THROW(NotImplementedError, "cuDNN: only float/double supported");
        
        int n = static_cast<int>(t.size(0));
        int c = static_cast<int>(t.size(1));
        int h = static_cast<int>(t.size(2));
        int w = static_cast<int>(t.size(3));
        
        // TensorPlay is NCHW by default
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, dtype, n, c, h, w));
    }
};

struct FilterDesc {
    cudnnFilterDescriptor_t desc;
    FilterDesc() { CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc)); }
    ~FilterDesc() { cudnnDestroyFilterDescriptor(desc); }
    operator cudnnFilterDescriptor_t() const { return desc; }
    
    void set(const Tensor& t) {
        cudnnDataType_t dtype;
        if (t.dtype() == DType::Float32) dtype = CUDNN_DATA_FLOAT;
        else if (t.dtype() == DType::Float64) dtype = CUDNN_DATA_DOUBLE;
        else TP_THROW(NotImplementedError, "cuDNN: only float/double supported");
        
        int k = static_cast<int>(t.size(0));
        int c = static_cast<int>(t.size(1));
        int h = static_cast<int>(t.size(2));
        int w = static_cast<int>(t.size(3));
        
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(desc, dtype, CUDNN_TENSOR_NCHW, k, c, h, w));
    }
};

struct ConvDesc {
    cudnnConvolutionDescriptor_t desc;
    ConvDesc() { CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc)); }
    ~ConvDesc() { cudnnDestroyConvolutionDescriptor(desc); }
    operator cudnnConvolutionDescriptor_t() const { return desc; }
    
    void set(int pad_h, int pad_w, int str_h, int str_w, int dil_h, int dil_w, int groups, DType dtype) {
        cudnnDataType_t computeType;
        if (dtype == DType::Float32) computeType = CUDNN_DATA_FLOAT;
        else if (dtype == DType::Float64) computeType = CUDNN_DATA_DOUBLE;
        else TP_THROW(NotImplementedError, "cuDNN: only float/double supported");
        
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(desc, pad_h, pad_w, str_h, str_w, dil_h, dil_w, CUDNN_CROSS_CORRELATION, computeType));
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(desc, groups));
    }
};

struct ActivationDesc {
    cudnnActivationDescriptor_t desc;
    ActivationDesc() { CUDNN_CHECK(cudnnCreateActivationDescriptor(&desc)); }
    ~ActivationDesc() { cudnnDestroyActivationDescriptor(desc); }
    operator cudnnActivationDescriptor_t() const { return desc; }

    void set_relu() {
        CUDNN_CHECK(cudnnSetActivationDescriptor(
            desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
    }
};

struct ConvFwdAlgo {
    cudnnConvolutionFwdAlgo_t algorithm;
    size_t workspace_size;
};

static std::unordered_map<std::string, ConvFwdAlgo> g_conv_fwd_algo_cache;
static std::mutex g_conv_fwd_cache_mutex;

std::string make_conv_fwd_cache_key(
    const Tensor& input,
    const Tensor& weight,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups,
    bool fused_relu) {
    std::string key = fused_relu ? "relu:" : "conv:";
    key += std::to_string(static_cast<int>(input.dtype()));
    key += ":" + std::to_string(input.device().index());
    for (int64_t dim : {input.size(0), input.size(1), input.size(2), input.size(3),
                        weight.size(0), weight.size(1), weight.size(2), weight.size(3),
                        stride[0], stride[1], padding[0], padding[1],
                        dilation[0], dilation[1], groups}) {
        key += ":" + std::to_string(dim);
    }
    return key;
}

// Backward convolution algorithm selection is shape- and dtype-dependent,
// but not iteration-dependent.  PyTorch's compiled path amortizes this
// decision; caching it here removes a cuDNN v7 heuristic query from every
// training convolution.
struct ConvBwdKey {
    int kind;  // 0 = grad-input, 1 = grad-weight
    int dtype;
    int device;
    std::array<int64_t, 4> input_shape;
    std::array<int64_t, 4> weight_shape;
    std::array<int64_t, 4> grad_shape;
    std::array<int64_t, 2> stride;
    std::array<int64_t, 2> padding;
    std::array<int64_t, 2> dilation;
    int64_t groups;

    bool operator==(const ConvBwdKey& other) const {
        return kind == other.kind && dtype == other.dtype && device == other.device &&
               input_shape == other.input_shape && weight_shape == other.weight_shape &&
               grad_shape == other.grad_shape && stride == other.stride &&
               padding == other.padding && dilation == other.dilation &&
               groups == other.groups;
    }
};

struct ConvBwdKeyHash {
    size_t operator()(const ConvBwdKey& key) const {
        size_t hash = 0;
        auto combine = [&hash](int64_t value) {
            hash = hash * 1000003U ^ std::hash<int64_t>{}(value);
        };
        combine(key.kind);
        combine(key.dtype);
        combine(key.device);
        for (auto value : key.input_shape) combine(value);
        for (auto value : key.weight_shape) combine(value);
        for (auto value : key.grad_shape) combine(value);
        for (auto value : key.stride) combine(value);
        for (auto value : key.padding) combine(value);
        for (auto value : key.dilation) combine(value);
        combine(key.groups);
        return hash;
    }
};

struct ConvBwdAlgo {
    int algorithm;
    size_t workspace_size;
};

static std::unordered_map<ConvBwdKey, ConvBwdAlgo, ConvBwdKeyHash> g_conv_bwd_algo_cache;
static std::mutex g_conv_bwd_cache_mutex;

ConvBwdKey make_conv_bwd_key(
    int kind,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& grad_output,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups) {
    return ConvBwdKey{
        kind,
        static_cast<int>(input.dtype()),
        static_cast<int>(input.device().index()),
        {input.size(0), input.size(1), input.size(2), input.size(3)},
        {weight.size(0), weight.size(1), weight.size(2), weight.size(3)},
        {grad_output.size(0), grad_output.size(1), grad_output.size(2), grad_output.size(3)},
        {stride[0], stride[1]},
        {padding[0], padding[1]},
        {dilation[0], dilation[1]},
        groups};
}

#endif

#ifdef USE_CUDNN
static Tensor conv2d_relu_cudnn(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const std::vector<int64_t>& stride_arg,
    const std::vector<int64_t>& padding_arg,
    const std::vector<int64_t>& dilation_arg,
    int64_t groups) {
    if (!bias.defined() ||
        (input.dtype() != DType::Float32 && input.dtype() != DType::Float64)) {
        return Tensor();
    }

    auto stride = expand_param_if_needed(stride_arg, 2, 1);
    auto padding = expand_param_if_needed(padding_arg, 2, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 2, 1);
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();
    Tensor bias_c = bias.is_contiguous() ? bias : bias.contiguous();

    const int64_t n = input_c.size(0);
    const int64_t k = weight_c.size(0);
    const int64_t h = input_c.size(2);
    const int64_t w = input_c.size(3);
    const int64_t r = weight_c.size(2);
    const int64_t s = weight_c.size(3);
    const int64_t oh = (h + 2 * padding[0] - dilation[0] * (r - 1) - 1) / stride[0] + 1;
    const int64_t ow = (w + 2 * padding[1] - dilation[1] * (s - 1) - 1) / stride[1] + 1;

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    TensorDesc x_desc; x_desc.set(input_c);
    FilterDesc w_desc; w_desc.set(weight_c);
    Tensor out = Tensor::empty({n, k, oh, ow}, input_c.dtype(), input_c.device());
    TensorDesc y_desc; y_desc.set(out);
    Tensor bias_4d = bias_c.reshape({1, k, 1, 1});
    TensorDesc bias_desc; bias_desc.set(bias_4d);
    ConvDesc conv_desc;
    conv_desc.set(
        static_cast<int>(padding[0]), static_cast<int>(padding[1]),
        static_cast<int>(stride[0]), static_cast<int>(stride[1]),
        static_cast<int>(dilation[0]), static_cast<int>(dilation[1]),
        static_cast<int>(groups), input_c.dtype());
    ActivationDesc activation_desc;
    activation_desc.set_relu();

    std::string cache_key = make_conv_fwd_cache_key(
        input_c, weight_c, stride, padding, dilation, groups, true);
    cudnnConvolutionFwdAlgo_t algorithm;
    size_t workspace_size;
    {
        std::lock_guard<std::mutex> lock(g_conv_fwd_cache_mutex);
        auto it = g_conv_fwd_algo_cache.find(cache_key);
        if (it == g_conv_fwd_algo_cache.end()) {
            cudnnConvolutionFwdAlgoPerf_t perf_results;
            int returned_algo_count = 0;
            CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
                handle, x_desc, w_desc, conv_desc, y_desc,
                1, &returned_algo_count, &perf_results));
            if (returned_algo_count == 0) {
                TP_THROW(RuntimeError, "cuDNN: no fused forward convolution algorithm");
            }
            algorithm = perf_results.algo;
            CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
                handle, x_desc, w_desc, conv_desc, y_desc,
                algorithm, &workspace_size));
            g_conv_fwd_algo_cache.emplace(
                cache_key, ConvFwdAlgo{algorithm, workspace_size});
        } else {
            algorithm = it->second.algorithm;
            workspace_size = it->second.workspace_size;
        }
    }

    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        workspace_size ? workspace_size : 1, input_c.device());
    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void* alpha_p = &alpha;
    void* beta_p = &beta;
    if (input_c.dtype() == DType::Float64) {
        alpha_p = &alpha_d;
        beta_p = &beta_d;
    }

    // alpha2 is zero, so z is not read; using y as its descriptor/pointer
    // keeps the legacy cuDNN API valid across cuDNN 8 and 9.
    CUDNN_CHECK(cudnnConvolutionBiasActivationForward(
        handle, alpha_p, x_desc, input_c.data_ptr(), w_desc, weight_c.data_ptr(),
        conv_desc, algorithm, workspace.get(), workspace_size, beta_p,
        y_desc, out.data_ptr(), bias_desc, bias_4d.data_ptr(), activation_desc,
        y_desc, out.data_ptr()));
    return out;
}
#endif

static Tensor conv2d_cuda_impl(const Tensor& input, const Tensor& weight, const Tensor& bias, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, int64_t groups, bool fused_relu) {
#ifdef USE_CUDNN
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

    // Compute type: FLOAT for fp32/half/bf16, DOUBLE for fp64 (matches torch).
    cudnnDataType_t compute = (dtype == CUDNN_DATA_DOUBLE) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;

    // Cache key: everything that determines the plan.  TorchInductor's
    // Conv_v8 path builds descriptors from actual sizes and strides, so the
    // layout is part of the plan identity as well.
    struct ConvKey {
        cudnnDataType_t dtype;
        int64_t N, C, H, W, K, R, S, groups;
        int64_t ph, pw, sh, sw, dh, dw;
        int device;
        bool has_bias;
        bool fused_relu;
        std::array<int64_t, 4> x_stride;
        std::array<int64_t, 4> w_stride;
        std::array<int64_t, 4> y_stride;
        bool operator==(const ConvKey& o) const {
            return dtype == o.dtype && N == o.N && C == o.C && H == o.H && W == o.W &&
                   K == o.K && R == o.R && S == o.S && groups == o.groups &&
                   ph == o.ph && pw == o.pw && sh == o.sh && sw == o.sw &&
                   dh == o.dh && dw == o.dw && device == o.device &&
                   has_bias == o.has_bias && fused_relu == o.fused_relu &&
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
            h = h * 1000003 ^ std::hash<int>{}((int)k.has_bias);
            h = h * 1000003 ^ std::hash<int>{}((int)k.fused_relu);
            for (auto v : k.x_stride) h = h * 1000003 ^ std::hash<int64_t>{}(v);
            for (auto v : k.w_stride) h = h * 1000003 ^ std::hash<int64_t>{}(v);
            for (auto v : k.y_stride) h = h * 1000003 ^ std::hash<int64_t>{}(v);
            return h;
        }
    };
    static std::unordered_map<ConvKey, std::shared_ptr<fe::ExecutionPlan>, ConvKeyHash> g_conv_plan_cache;
    static std::mutex g_conv_cache_mutex;

    const bool use_channels_last = is_channels_last_4d(input);
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
                bias.defined(), fused_relu, x_stride, w_stride, y_stride};

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();

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
                              .setAlignment(16)
                              .setDataType(dtype)
                              .build();
            auto w_desc = fe::TensorBuilder()
                              .setDim(4, std::array<int64_t, 4>{K, C / groups, R, S}.data())
                              .setStrides(4, w_stride.data())
                              .setId('w')
                              .setAlignment(16)
                              .setDataType(dtype)
                              .build();
            auto y_desc = fe::TensorBuilder()
                              .setDim(4, std::array<int64_t, 4>{N, K, OH, OW}.data())
                              .setStrides(4, y_stride.data())
                              .setId('y')
                              .setAlignment(16)
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
                                         .setAlignment(16)
                                         .setDataType(compute)
                                         .setVirtual(true)
                                         .build();
                auto b_desc = fe::TensorBuilder()
                                  .setDim(4, std::array<int64_t, 4>{1, K, 1, 1}.data())
                                  .setStrides(4, std::array<int64_t, 4>{K, 1, 1, 1}.data())
                                  .setId('b')
                                  .setAlignment(16)
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
                                        .setAlignment(16)
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
                auto heuristics = fe::EngineHeuristicsBuilder()
                                      .setOperationGraph(*op_graph_ptr)
                                      .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                                      .build();
                auto& engine_configs = heuristics.getEngineConfig(1);
                if (engine_configs.empty()) {
                    TP_THROW(RuntimeError, "cuDNN: no engine configs for conv2d");
                }
                new_plan = std::make_shared<fe::ExecutionPlan>(
                    fe::ExecutionPlanBuilder()
                        .setHandle(handle)
                        .setEngineConfig(engine_configs[0])
                        .build());
            } else if (!key.fused_relu) {
                std::array<fe::Operation const*, 1> ops = {&conv_op};
                auto op_graph = fe::OperationGraphBuilder()
                                    .setHandle(handle)
                                    .setOperationGraph(ops.size(), ops.data())
                                    .build();
                auto heuristics = fe::EngineHeuristicsBuilder()
                                      .setOperationGraph(op_graph)
                                      .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                                      .build();
                auto& engine_configs = heuristics.getEngineConfig(1);
                if (engine_configs.empty()) {
                    TP_THROW(RuntimeError, "cuDNN: no engine configs for conv2d");
                }
                new_plan = std::make_shared<fe::ExecutionPlan>(
                    fe::ExecutionPlanBuilder()
                        .setHandle(handle)
                        .setEngineConfig(engine_configs[0])
                        .build());
            } else {
                auto conv_out_desc = fe::TensorBuilder()
                                         .setDim(4, std::array<int64_t, 4>{N, K, OH, OW}.data())
                                         .setStrides(4, y_stride.data())
                                         .setId('C')
                                         .setAlignment(16)
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
                auto heuristics = fe::EngineHeuristicsBuilder()
                                      .setOperationGraph(op_graph)
                                      .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                                      .build();
                auto& engine_configs = heuristics.getEngineConfig(1);
                if (engine_configs.empty()) {
                    TP_THROW(RuntimeError, "cuDNN: no engine configs for fused conv2d_relu");
                }
                new_plan = std::make_shared<fe::ExecutionPlan>(
                    fe::ExecutionPlanBuilder()
                        .setHandle(handle)
                        .setEngineConfig(engine_configs[0])
                        .build());
            }
            plan = new_plan;
            g_conv_plan_cache[key] = new_plan;
        }
    }

    size_t workspace_size = plan->getWorkspaceSize();
    auto workspace = getAllocator(DeviceType::CUDA)->allocate(workspace_size ? workspace_size : 1);

    if (key.has_bias) {
        void* data_ptrs[4] = {input.data_ptr(), weight.data_ptr(), bias.data_ptr(), out.data_ptr()};
        int64_t uids[4] = {'x', 'w', 'b', 'y'};
        auto variant_pack = fe::VariantPackBuilder()
                                .setWorkspacePointer(workspace_size ? workspace.get() : nullptr)
                                .setDataPointers(4, data_ptrs)
                                .setUids(4, uids)
                                .build();
        CUDNN_CHECK(cudnnBackendExecute(handle, plan->get_raw_desc(), variant_pack.get_raw_desc()));
    } else {
        void* data_ptrs[3] = {input.data_ptr(), weight.data_ptr(), out.data_ptr()};
        int64_t uids[3] = {'x', 'w', 'y'};
        auto variant_pack = fe::VariantPackBuilder()
                                .setWorkspacePointer(workspace_size ? workspace.get() : nullptr)
                                .setDataPointers(3, data_ptrs)
                                .setUids(3, uids)
                                .build();
        CUDNN_CHECK(cudnnBackendExecute(handle, plan->get_raw_desc(), variant_pack.get_raw_desc()));
    }

    return out;
#else
    TP_THROW(NotImplementedError, "conv2d_cuda requires cuDNN");
#endif
}

Tensor conv2d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias,
                   const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                   const std::vector<int64_t>& dilation, int64_t groups) {
    return conv2d_cuda_impl(input, weight, bias, stride, padding, dilation, groups, false);
}

// Keep the fused IR contract shared with CPU.  The cuDNN frontend plan owns
// the Conv(+bias)->ReLU graph, so this path launches one backend plan rather
// than a convolution followed by a separate pointwise kernel.
Tensor conv2d_relu_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias,
                        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                        const std::vector<int64_t>& dilation, int64_t groups) {
    return conv2d_cuda_impl(input, weight, bias, stride, padding, dilation, groups, true);
}

Tensor conv2d_grad_input_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, int64_t groups) {
#ifdef USE_CUDNN
    auto stride = expand_param_if_needed(stride_arg, 2, 1);
    auto padding = expand_param_if_needed(padding_arg, 2, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 2, 1);

    // Backward grads can arrive as broadcast views (e.g. after .sum()); cuDNN
    // needs contiguous NCHW.
    Tensor grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();

    TensorDesc dx_desc; dx_desc.set(input_c); // gradient of input has same shape as input
    FilterDesc w_desc; w_desc.set(weight_c);
    TensorDesc dy_desc; dy_desc.set(grad_output_c);
    
    ConvDesc conv_desc;
    conv_desc.set((int)padding[0], (int)padding[1], (int)stride[0], (int)stride[1], (int)dilation[0], (int)dilation[1], (int)groups, input_c.dtype());
    
    Tensor grad_input = Tensor::empty_like(input_c, DType::Undefined, input_c.device());
    
    ConvBwdKey cache_key = make_conv_bwd_key(
        0, input_c, weight_c, grad_output_c, stride, padding, dilation, groups);
    cudnnConvolutionBwdDataAlgo_t algo;
    size_t workspace_size;
    {
        std::lock_guard<std::mutex> lock(g_conv_bwd_cache_mutex);
        auto it = g_conv_bwd_algo_cache.find(cache_key);
        if (it == g_conv_bwd_algo_cache.end()) {
            cudnnConvolutionBwdDataAlgoPerf_t perf_results;
            int returned_algo_count = 0;
            CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                handle, w_desc, dy_desc, conv_desc, dx_desc,
                1, &returned_algo_count, &perf_results));
            if (returned_algo_count == 0) {
                TP_THROW(RuntimeError, "cuDNN: no backward-data convolution algorithm");
            }
            algo = perf_results.algo;
            CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
                handle, w_desc, dy_desc, conv_desc, dx_desc, algo, &workspace_size));
            g_conv_bwd_algo_cache.emplace(
                cache_key, ConvBwdAlgo{static_cast<int>(algo), workspace_size});
        } else {
            algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(it->second.algorithm);
            workspace_size = it->second.workspace_size;
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
    
    CUDNN_CHECK(cudnnConvolutionBackwardData(handle, alpha_p, w_desc, weight_c.data_ptr(), dy_desc, grad_output_c.data_ptr(), conv_desc, algo, workspace.get(), workspace_size, beta_p, dx_desc, grad_input.data_ptr()));
    
    return grad_input;
#else
    TP_THROW(NotImplementedError, "conv2d_grad_input_cuda requires cuDNN");
#endif
}

Tensor conv2d_grad_weight_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, int64_t groups) {
#ifdef USE_CUDNN
    auto stride = expand_param_if_needed(stride_arg, 2, 1);
    auto padding = expand_param_if_needed(padding_arg, 2, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 2, 1);

    Tensor grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();
    
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    
    TensorDesc x_desc; x_desc.set(input_c);
    TensorDesc dy_desc; dy_desc.set(grad_output_c);
    FilterDesc dw_desc; dw_desc.set(weight_c); // grad_weight has same shape as weight
    
    ConvDesc conv_desc;
    conv_desc.set((int)padding[0], (int)padding[1], (int)stride[0], (int)stride[1], (int)dilation[0], (int)dilation[1], (int)groups, input_c.dtype());
    
    Tensor grad_weight = Tensor::empty_like(weight_c, DType::Undefined, weight_c.device());
    
    ConvBwdKey cache_key = make_conv_bwd_key(
        1, input_c, weight_c, grad_output_c, stride, padding, dilation, groups);
    cudnnConvolutionBwdFilterAlgo_t algo;
    size_t workspace_size;
    {
        std::lock_guard<std::mutex> lock(g_conv_bwd_cache_mutex);
        auto it = g_conv_bwd_algo_cache.find(cache_key);
        if (it == g_conv_bwd_algo_cache.end()) {
            cudnnConvolutionBwdFilterAlgoPerf_t perf_results;
            int returned_algo_count = 0;
            CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                handle, x_desc, dy_desc, conv_desc, dw_desc,
                1, &returned_algo_count, &perf_results));
            if (returned_algo_count == 0) {
                TP_THROW(RuntimeError, "cuDNN: no backward-filter convolution algorithm");
            }
            algo = perf_results.algo;
            CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                handle, x_desc, dy_desc, conv_desc, dw_desc, algo, &workspace_size));
            g_conv_bwd_algo_cache.emplace(
                cache_key, ConvBwdAlgo{static_cast<int>(algo), workspace_size});
        } else {
            algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(it->second.algorithm);
            workspace_size = it->second.workspace_size;
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
    
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle, alpha_p, x_desc, input_c.data_ptr(), dy_desc, grad_output_c.data_ptr(), conv_desc, algo, workspace.get(), workspace_size, beta_p, dw_desc, grad_weight.data_ptr()));
    
    return grad_weight;
#else
    TP_THROW(NotImplementedError, "conv2d_grad_weight_cuda requires cuDNN");
#endif
}

Tensor conv2d_grad_bias_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, int64_t groups) {
#ifdef USE_CUDNN
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    
    Tensor grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    
    TensorDesc dy_desc; dy_desc.set(grad_output_c);
    
    Tensor grad_bias = Tensor::empty({grad_output_c.size(1)}, grad_output_c.dtype(), grad_output_c.device());
    
    TensorDesc db_desc;
    Tensor grad_bias_reshaped = grad_bias.reshape({1, grad_bias.size(0), 1, 1});
    db_desc.set(grad_bias_reshaped);
    
    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void *alpha_p = &alpha, *beta_p = &beta;
    if (grad_output_c.dtype() == DType::Float64) {
        alpha_p = &alpha_d; beta_p = &beta_d;
    }
    
    CUDNN_CHECK(cudnnConvolutionBackwardBias(handle, alpha_p, dy_desc, grad_output_c.data_ptr(), beta_p, db_desc, grad_bias.data_ptr()));
    
    return grad_bias;
#else
    TP_THROW(NotImplementedError, "conv2d_grad_bias_cuda requires cuDNN");
#endif
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ConvKernels) {
    m.impl("conv2d", conv2d_cuda);
    m.impl("conv2d_relu", conv2d_relu_cuda);
    m.impl("conv2d_grad_input", conv2d_grad_input_cuda);
    m.impl("conv2d_grad_weight", conv2d_grad_weight_cuda);
    m.impl("conv2d_grad_bias", conv2d_grad_bias_cuda);
}

} // namespace cuda
} // namespace tensorplay
