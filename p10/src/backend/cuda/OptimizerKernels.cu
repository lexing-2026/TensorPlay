#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Torch-aligned multi-tensor apply primitives.
// Reference: aten/src/ATen/native/cuda/MultiTensorApply.cuh:16-63
//   kILP=4, kChunkSize=65536, kBlockSize=512,
//   is_aligned / load_store (aligned_vec_t → LDG.128/STG.128)
//   ForeachFunctors.cuh:109/165 — load_args/store_args with bounds checks.
// ---------------------------------------------------------------------------
constexpr int kOptILP = 4;
constexpr int64_t kFusedChunk = 65536;
constexpr int64_t kFusedBlock = 512;

template <typename T>
__device__ __forceinline__ bool opt_is_aligned(T* p) {
    return (reinterpret_cast<uintptr_t>(p) & ((kOptILP * sizeof(T)) - 1)) == 0;
}

template <typename T>
struct alignas(kOptILP * sizeof(T)) OptVec { T v[kOptILP]; };

template <typename T>
__device__ __forceinline__ void opt_load_store(
    T* dst, T* src, int64_t dst_off, int64_t src_off) {
    using LT = OptVec<T>;
    reinterpret_cast<LT*>(dst)[dst_off] = reinterpret_cast<const LT*>(src)[src_off];
}


namespace tensorplay {
namespace cuda {
namespace {

void validate_lists(const std::vector<Tensor>& params,
                   const std::vector<Tensor>& grads,
                   const std::vector<Tensor>& first_state,
                   const std::vector<Tensor>& second_state,
                   const std::vector<Tensor>& third_state,
                   const std::vector<int64_t>& steps,
                   bool require_first_state,
                   bool require_second_state,
                   bool require_third_state,
                   const char* op_name) {
    const auto count = params.size();
    if (grads.size() != count || first_state.size() != count ||
        second_state.size() != count || third_state.size() != count) {
        TP_THROW(ValueError, std::string(op_name) +
            ": tensor list sizes must match");
    }
    if (!steps.empty() && steps.size() != count) {
        TP_THROW(ValueError, std::string(op_name) +
            ": step list size must match parameter list");
    }
    if (count > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        TP_THROW(ValueError, std::string(op_name) +
            ": too many tensors for one CUDA grid");
    }

    const DType dtype = count ? params[0].dtype() : DType::Undefined;
    const Device device = count ? params[0].device() : Device(DeviceType::CUDA);
    for (size_t i = 0; i < count; ++i) {
        const Tensor& param = params[i];
        const Tensor& grad = grads[i];
        if (!param.defined() || !grad.defined()) {
            TP_THROW(ValueError, std::string(op_name) +
                ": parameters and gradients must be defined");
        }
        if (!param.is_contiguous() || !grad.is_contiguous() ||
            param.shape() != grad.shape() || param.dtype() != grad.dtype() ||
            param.dtype() != dtype || param.device() != device) {
            TP_THROW(NotImplementedError, std::string(op_name) +
                ": requires contiguous same-device parameter/gradient pairs with one dtype");
        }

        const Tensor* states[] = {&first_state[i], &second_state[i]};
        const bool required[] = {require_first_state, require_second_state};
        for (size_t state_index = 0; state_index < 2; ++state_index) {
            if (!required[state_index]) continue;
            const Tensor& state = *states[state_index];
            if (!state.defined() || !state.is_contiguous() ||
                state.shape() != param.shape() || state.dtype() != dtype ||
                state.device() != device) {
                TP_THROW(NotImplementedError, std::string(op_name) +
                    ": optimizer state must match its parameter layout");
            }
        }
        if (require_third_state) {
            const Tensor& state = third_state[i];
            if (!state.defined() || !state.is_contiguous() ||
                state.shape() != param.shape() || state.dtype() != dtype ||
                state.device() != device) {
                TP_THROW(NotImplementedError, std::string(op_name) +
                    ": AMSGrad state must match its parameter layout");
            }
        }
    }
}

template <typename T>
class DeviceArray {
public:
    DeviceArray(cudaStream_t stream, const std::vector<T>& values)
        : stream_(stream) {
        if (values.empty()) return;
        checkCuda(cudaMallocAsync(reinterpret_cast<void**>(&data_),
                                  values.size() * sizeof(T), stream_),
                  "cudaMallocAsync optimizer metadata");
        checkCuda(cudaMemcpyAsync(data_, values.data(),
                                  values.size() * sizeof(T),
                                  cudaMemcpyHostToDevice, stream_),
                  "cudaMemcpyAsync optimizer metadata");
    }

    ~DeviceArray() {
        if (data_) (void)cudaFreeAsync(data_, stream_);
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    T* data() const noexcept { return data_; }

private:
    cudaStream_t stream_ = nullptr;
    T* data_ = nullptr;
};

template <typename scalar_t>
__global__ void foreach_sgd_kernel(scalar_t* const* params,
                                   scalar_t* const* grads,
                                   scalar_t* const* momentum_buffers,
                                   const int64_t* numels,
                                   int64_t parameter_count,
                                   double lr,
                                   double momentum,
                                   double dampening,
                                   double weight_decay,
                                   int nesterov,
                                   int first_momentum_step) {
    const int64_t list_index = static_cast<int64_t>(blockIdx.x);
    if (list_index >= parameter_count) return;

    scalar_t* param = params[list_index];
    scalar_t* grad = grads[list_index];
    scalar_t* buffer = momentum_buffers[list_index];
    const scalar_t lr_value = static_cast<scalar_t>(lr);
    const scalar_t momentum_value = static_cast<scalar_t>(momentum);
    const scalar_t dampening_value = static_cast<scalar_t>(dampening);
    const scalar_t decay_value = static_cast<scalar_t>(weight_decay);

    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i < numels[list_index]; i += static_cast<int64_t>(blockDim.x)) {
        scalar_t update = grad[i];
        if (weight_decay != 0.0) update += decay_value * param[i];
        if (momentum != 0.0) {
            if (first_momentum_step) {
                buffer[i] = update;
                if (nesterov) update += momentum_value * buffer[i];
                else update = buffer[i];
            } else {
                buffer[i] = momentum_value * buffer[i] +
                    (scalar_t(1) - dampening_value) * update;
                if (nesterov) update += momentum_value * buffer[i];
                else update = buffer[i];
            }
        }
        param[i] -= lr_value * update;
    }
}

template <typename scalar_t>
__global__ void foreach_adam_kernel(scalar_t* const* params,
                                    scalar_t* const* grads,
                                    scalar_t* const* exp_avgs,
                                    scalar_t* const* exp_avg_sqs,
                                    scalar_t* const* max_exp_avg_sqs,
                                    const int64_t* numels,
                                    const double* step_sizes,
                                    const double* correction2_sqrts,
                                    int64_t parameter_count,
                                    double beta1,
                                    double beta2,
                                    double eps,
                                    double weight_decay,
                                    int amsgrad) {
    const int64_t list_index = static_cast<int64_t>(blockIdx.x);
    if (list_index >= parameter_count) return;

    scalar_t* param = params[list_index];
    scalar_t* grad = grads[list_index];
    scalar_t* exp_avg = exp_avgs[list_index];
    scalar_t* exp_avg_sq = exp_avg_sqs[list_index];
    scalar_t* max_exp_avg_sq = amsgrad ? max_exp_avg_sqs[list_index] : nullptr;
    // Bias corrections are identical for every element in this tensor.  They
    // are computed once on the host per parameter tensor instead of repeating
    // pow/sqrt in every CUDA thread.
    const scalar_t step_size = static_cast<scalar_t>(step_sizes[list_index]);
    const scalar_t correction2_sqrt = static_cast<scalar_t>(
        correction2_sqrts[list_index]);
    const scalar_t beta1_value = static_cast<scalar_t>(beta1);
    const scalar_t beta2_value = static_cast<scalar_t>(beta2);
    const scalar_t one_minus_beta1 = static_cast<scalar_t>(1.0 - beta1);
    const scalar_t one_minus_beta2 = static_cast<scalar_t>(1.0 - beta2);
    const scalar_t eps_value = static_cast<scalar_t>(eps);
    const scalar_t decay_value = static_cast<scalar_t>(weight_decay);

    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i < numels[list_index]; i += static_cast<int64_t>(blockDim.x)) {
        scalar_t g = grad[i];
        if (weight_decay != 0.0) g += decay_value * param[i];
        exp_avg[i] = beta1_value * exp_avg[i] + one_minus_beta1 * g;
        exp_avg_sq[i] = beta2_value * exp_avg_sq[i] + one_minus_beta2 * g * g;

        scalar_t second_moment = exp_avg_sq[i];
        if (amsgrad) {
            if (max_exp_avg_sq[i] < second_moment) {
                max_exp_avg_sq[i] = second_moment;
            }
            second_moment = max_exp_avg_sq[i];
        }
        const scalar_t denom = static_cast<scalar_t>(
            sqrt(static_cast<double>(second_moment)) /
            static_cast<double>(correction2_sqrt)) + eps_value;
        param[i] -= step_size * exp_avg[i] / denom;
    }
}

template <typename scalar_t>
void launch_sgd(const std::vector<Tensor>& params,
                const std::vector<Tensor>& grads,
                const std::vector<Tensor>& momentum_buffers,
                double lr,
                double momentum,
                double dampening,
                double weight_decay,
                bool nesterov,
                bool first_momentum_step) {
    const auto stream = getCurrentCUDAStream().stream();
    std::vector<scalar_t*> param_ptrs;
    std::vector<scalar_t*> grad_ptrs;
    std::vector<scalar_t*> buffer_ptrs;
    std::vector<int64_t> numels;
    param_ptrs.reserve(params.size());
    grad_ptrs.reserve(params.size());
    buffer_ptrs.reserve(params.size());
    numels.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        param_ptrs.push_back(params[i].data_ptr<scalar_t>());
        grad_ptrs.push_back(grads[i].data_ptr<scalar_t>());
        buffer_ptrs.push_back(momentum_buffers[i].defined()
            ? momentum_buffers[i].data_ptr<scalar_t>() : nullptr);
        numels.push_back(params[i].numel());
    }
    DeviceArray<scalar_t*> d_params(stream, param_ptrs);
    DeviceArray<scalar_t*> d_grads(stream, grad_ptrs);
    DeviceArray<scalar_t*> d_buffers(stream, buffer_ptrs);
    DeviceArray<int64_t> d_numels(stream, numels);

    foreach_sgd_kernel<scalar_t><<<static_cast<unsigned int>(params.size()), 256, 0, stream>>>(
        d_params.data(), d_grads.data(), d_buffers.data(), d_numels.data(),
                static_cast<int64_t>(params.size()), lr, momentum, dampening,
        weight_decay, nesterov ? 1 : 0, first_momentum_step ? 1 : 0);
    checkCuda(cudaGetLastError(), "_foreach_sgd kernel launch");
}

template <typename scalar_t>
void launch_adam(const std::vector<Tensor>& params,
                 const std::vector<Tensor>& grads,
                 const std::vector<Tensor>& exp_avgs,
                 const std::vector<Tensor>& exp_avg_sqs,
                 const std::vector<Tensor>& max_exp_avg_sqs,
                 const std::vector<int64_t>& steps,
                 double lr,
                 double beta1,
                 double beta2,
                 double eps,
                 double weight_decay,
                 bool amsgrad) {
    const auto stream = getCurrentCUDAStream().stream();
    std::vector<scalar_t*> param_ptrs;
    std::vector<scalar_t*> grad_ptrs;
    std::vector<scalar_t*> exp_avg_ptrs;
    std::vector<scalar_t*> exp_avg_sq_ptrs;
    std::vector<scalar_t*> max_exp_avg_sq_ptrs;
    std::vector<int64_t> numels;
    std::vector<double> step_sizes;
    std::vector<double> correction2_sqrts;
    param_ptrs.reserve(params.size());
    grad_ptrs.reserve(params.size());
    exp_avg_ptrs.reserve(params.size());
    exp_avg_sq_ptrs.reserve(params.size());
    max_exp_avg_sq_ptrs.reserve(params.size());
    numels.reserve(params.size());
    step_sizes.reserve(params.size());
    correction2_sqrts.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        param_ptrs.push_back(params[i].data_ptr<scalar_t>());
        grad_ptrs.push_back(grads[i].data_ptr<scalar_t>());
        exp_avg_ptrs.push_back(exp_avgs[i].data_ptr<scalar_t>());
        exp_avg_sq_ptrs.push_back(exp_avg_sqs[i].data_ptr<scalar_t>());
        max_exp_avg_sq_ptrs.push_back(amsgrad
            ? max_exp_avg_sqs[i].data_ptr<scalar_t>() : nullptr);
        numels.push_back(params[i].numel());
        const double bias_correction1 =
            1.0 - std::pow(beta1, static_cast<double>(steps[i]));
        const double bias_correction2 =
            1.0 - std::pow(beta2, static_cast<double>(steps[i]));
        step_sizes.push_back(lr / bias_correction1);
        correction2_sqrts.push_back(std::sqrt(bias_correction2));
    }
    DeviceArray<scalar_t*> d_params(stream, param_ptrs);
    DeviceArray<scalar_t*> d_grads(stream, grad_ptrs);
    DeviceArray<scalar_t*> d_exp_avgs(stream, exp_avg_ptrs);
    DeviceArray<scalar_t*> d_exp_avg_sqs(stream, exp_avg_sq_ptrs);
    DeviceArray<scalar_t*> d_max_exp_avg_sqs(stream, max_exp_avg_sq_ptrs);
    DeviceArray<int64_t> d_numels(stream, numels);
    DeviceArray<double> d_step_sizes(stream, step_sizes);
    DeviceArray<double> d_correction2_sqrts(stream, correction2_sqrts);

    foreach_adam_kernel<scalar_t><<<static_cast<unsigned int>(params.size()), 256, 0, stream>>>(
        d_params.data(), d_grads.data(), d_exp_avgs.data(), d_exp_avg_sqs.data(),
        d_max_exp_avg_sqs.data(), d_numels.data(), d_step_sizes.data(),
        d_correction2_sqrts.data(), static_cast<int64_t>(params.size()), beta1,
        beta2, eps, weight_decay, amsgrad ? 1 : 0);
    checkCuda(cudaGetLastError(), "_foreach_adam kernel launch");
}

void validate_fused_pairs(const std::vector<Tensor>& params,
                          const std::vector<Tensor>& grads,
                          const char* op_name) {
    if (params.size() != grads.size()) {
        TP_THROW(ValueError, std::string(op_name) +
            ": parameter and gradient lists must have the same length");
    }
    if (params.empty()) return;
    const DType dtype = params[0].dtype();
    if (dtype != DType::Float16 && dtype != DType::BFloat16 &&
        dtype != DType::Float32 && dtype != DType::Float64) {
        TP_THROW(NotImplementedError, std::string(op_name) +
            ": fused kernels support float16, bfloat16, float32, and float64");
    }
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& param = params[i];
        const Tensor& grad = grads[i];
        if (!param.defined() || !grad.defined()) {
            TP_THROW(ValueError, std::string(op_name) +
                ": parameters and gradients must be defined");
        }
        if (param.is_sparse() || grad.is_sparse() || isComplexType(param.dtype()) ||
            !param.is_contiguous() || !grad.is_contiguous() ||
            param.shape() != grad.shape() || param.dtype() != grad.dtype() ||
            param.dtype() != dtype || param.device() != Device(DeviceType::CUDA) ||
            grad.device() != Device(DeviceType::CUDA)) {
            TP_THROW(NotImplementedError, std::string(op_name) +
                ": requires contiguous CUDA tensors with matching floating dtype and shape");
        }
    }
}

void validate_fused_state(const std::vector<Tensor>& params,
                          const std::vector<Tensor>& state,
                          bool required,
                          const char* op_name) {
    if (!required && state.empty()) return;
    if (state.size() != params.size()) {
        TP_THROW(ValueError, std::string(op_name) +
            ": optimizer state list must match parameter list");
    }
    for (size_t i = 0; i < params.size(); ++i) {
        if (!state[i].defined() || !state[i].is_contiguous() ||
            state[i].shape() != params[i].shape() ||
            state[i].dtype() != params[i].dtype() ||
            state[i].device() != params[i].device()) {
            TP_THROW(NotImplementedError, std::string(op_name) +
                ": optimizer state must match its parameter layout");
        }
    }
}

void validate_fused_steps(const std::vector<Tensor>& params,
                          const std::vector<Tensor>& state_steps,
                          const char* op_name) {
    if (state_steps.size() != params.size()) {
        TP_THROW(ValueError, std::string(op_name) +
            ": state_steps must match parameter list");
    }
    for (const Tensor& step : state_steps) {
        if (!step.defined() || !step.is_contiguous() || step.numel() != 1 ||
            step.dtype() != DType::Float32 ||
            step.device() != Device(DeviceType::CUDA)) {
            TP_THROW(NotImplementedError, std::string(op_name) +
                ": state_steps must be singleton CUDA float32 tensors");
        }
    }
}

const float* optional_fused_float_ptr(const std::optional<Tensor>& value,
                                      const char* name) {
    if (!value.has_value() || !value->defined()) return nullptr;
    if (value->numel() != 1 || value->dtype() != DType::Float32 ||
        value->device() != Device(DeviceType::CUDA)) {
        TP_THROW(NotImplementedError, std::string(name) +
            " must be a singleton CUDA float32 tensor");
    }
    return value->data_ptr<float>();
}


// ---------------------------------------------------------------------------
// Dual-path fused optimizer kernels (torch fused_adam_utils.cuh:169-285).
// Same kernel handles both vectorized and scalar paths; each chunk chooses
// independently based on pointer alignment. This eliminates batch-level
// fallback for odd-sized tensors.
// ---------------------------------------------------------------------------

namespace {
__device__ __forceinline__ void opt_load_args(
    float r[][kOptILP], float** args, int depth_count,
    int64_t i_start, int64_t csz, int64_t n) {
#pragma unroll
    for (int ii = 0; ii < kOptILP; ++ii) {
        const int64_t i = i_start + threadIdx.x + ii * blockDim.x;
        for (int d = 0; d < depth_count; ++d) {
            r[d][ii] = 0.f;
            if (i < n && i < csz) r[d][ii] = args[d][i];
        }
    }
}
__device__ __forceinline__ void opt_store_args(
    float** dst, float r[][kOptILP], int depth_count,
    int64_t i_start, int64_t csz, int64_t n, int skip_grad, const float* gs) {
#pragma unroll
    for (int ii = 0; ii < kOptILP; ++ii) {
        const int64_t i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < csz) {
            for (int d = 0; d < depth_count; ++d) {
                if (d == 1 && !gs) continue;
                dst[d][i] = r[d][ii];
            }
        }
    }
}
} // anonymous namespace

template <typename lr_t, int DEPTH, bool ADAMW, bool AMSGRAD>
__global__ void __launch_bounds__(512, 2) fused_adam_kernel(
    float* const* params, float* const* grads,
    float* const* exp_avgs, float* const* exp_avg_sqs,
    float* const* max_exp_avg_sqs,
    float* const* state_steps,
    const int64_t* numels,
    const int32_t* b2t, const int64_t* b2c,
    int64_t chunk_size,
    const lr_t* tensor_lr, double scalar_lr,
    double beta1, double beta2, double weight_decay, double eps,
    int maximize,
    const float* grad_scale, const float* found_inf,
    int64_t parameter_count) {
    constexpr int P=0, G=1, M=2, V=3, N=4;
    const int64_t tloc = static_cast<int64_t>(b2t[blockIdx.x]);
    if (tloc >= parameter_count || (found_inf && *found_inf == 1.0f)) return;

    const float lr = tensor_lr ? static_cast<float>(tensor_lr[0]) : static_cast<float>(scalar_lr);
    const float step = state_steps[tloc][0];
    const float bc1 = 1.f - powf(beta1, step);
    const float bc2s = sqrtf(1.f - powf(beta2, step));
    const float ss = lr / bc1;
    const float b2v = static_cast<float>(beta2);
    const float omb1 = 1.f - static_cast<float>(beta1);
    const float omb2 = 1.f - b2v;

    float* args[DEPTH];
    args[P]=params[tloc]; args[G]=grads[tloc];
    args[M]=exp_avgs[tloc]; args[V]=exp_avg_sqs[tloc];
    if (DEPTH>4) args[N]=max_exp_avg_sqs[tloc];

    const int64_t cb = static_cast<int64_t>(b2c[blockIdx.x]) * chunk_size;
    const int64_t n = numels[tloc] - cb;
    for (int d=0;d<DEPTH;++d) args[d]+=cb;

    bool aligned=true;
    for (int d=0;d<DEPTH;++d) { if(!opt_is_aligned(args[d])) aligned=false; }

    float r[DEPTH][kOptILP];

    auto math_fn = [&]() {
#pragma unroll
        for (int ii=0; ii<kOptILP; ++ii) {
            float gv=r[G][ii], pv=r[P][ii];
            if(grad_scale){gv/=*grad_scale;}
            if(maximize)gv=-gv;
            if(ADAMW){pv*=(1.f-lr*static_cast<float>(weight_decay));}
            else if(weight_decay!=0.f){gv+=static_cast<float>(weight_decay)*pv;}
            float mv=r[M][ii];
            if(fabsf(omb1)<0.5f) mv+=omb1*(gv-mv);
            else mv=gv-(gv-mv)*(1.f-omb1);
            float vv=b2v*r[V][ii]+omb2*gv*gv;
            r[M][ii]=mv; r[V][ii]=vv;
            float sec=vv;
            if(AMSGRAD){sec=fmaxf(sec,r[N][ii]);r[N][ii]=sec;}
            float den=sqrtf(sec)/bc2s+static_cast<float>(eps);
            r[P][ii]=pv-ss*mv/den;
        }
    };

    if ((n%kOptILP==0)&&(chunk_size%kOptILP==0)&&aligned) {
        // FAST PATH: opt_load_store → LDG.128/STG.128
        for (int64_t is=threadIdx.x;
             is*kOptILP<n&&is*kOptILP<chunk_size;is+=blockDim.x) {
#pragma unroll
            for(int d=0;d<DEPTH;++d) opt_load_store(r[d],args[d],0,is);
            math_fn();
#pragma unroll
            for(int d=0;d<DEPTH;++d){
                if(d!=G||grad_scale)opt_load_store(args[d],r[d],is,0);
            }
        }
    } else {
        // SLOW PATH: scalar bounds-checked
        for (int64_t is=0;is<n&&is<chunk_size;
             is+=static_cast<int64_t>(blockDim.x)*kOptILP) {
            opt_load_args(r,args,DEPTH,is,chunk_size,n);
            math_fn();
            opt_store_args(args,r,DEPTH,is,chunk_size,n,!!grad_scale,grad_scale);
        }
    }
}

template <typename lr_t>
__global__ void __launch_bounds__(512, 2) fused_sgd_kernel(
    float* const* params, float* const* grads,
    float* const* momentum_buffers,
    const int64_t* numels,
    const int32_t* b2t, const int64_t* b2c,
    int64_t chunk_size,
    const lr_t* tensor_lr, double scalar_lr,
    double momentum, double dampening, double weight_decay,
    int nesterov, int maximize, int is_first_step,
    const float* grad_scale, const float* found_inf,
    int64_t parameter_count) {
    const int64_t tloc = static_cast<int64_t>(b2t[blockIdx.x]);
    if (tloc >= parameter_count || (found_inf && *found_inf == 1.0f)) return;
    const float lr = tensor_lr ? static_cast<float>(tensor_lr[0]) : static_cast<float>(scalar_lr);
    const float mom = static_cast<float>(momentum);
    const float damp = static_cast<float>(dampening);

    float* args[3];
    args[0]=params[tloc]; args[1]=grads[tloc]; args[2]=momentum_buffers[tloc];
    const int64_t cb = static_cast<int64_t>(b2c[blockIdx.x]) * chunk_size;
    const int64_t n = numels[tloc] - cb;
    for (int d=0;d<3;++d) args[d]+=cb;

    bool aligned=true;
    for (int d=0;d<3;++d) { if(!opt_is_aligned(args[d])) aligned=false; }

    float r[3][kOptILP];

    auto sgd_math = [&]() {
#pragma unroll
        for (int ii=0; ii<kOptILP; ++ii) {
            float gv=r[1][ii], pv=r[0][ii];
            if(grad_scale)gv/=*grad_scale;
            if(maximize)gv=-gv;
            if(weight_decay!=0.f)gv+=static_cast<float>(weight_decay)*pv;
            if(momentum!=0.f){
                float buf=is_first_step?gv:(mom*r[2][ii]+(1.f-damp)*gv);
                r[2][ii]=buf;
                gv=nesterov?(gv+mom*buf):buf;
            }
            r[0][ii]=pv-lr*gv;
        }
    };

    if ((n%kOptILP==0)&&(chunk_size%kOptILP==0)&&aligned) {
        for (int64_t is=threadIdx.x;
             is*kOptILP<n&&is*kOptILP<chunk_size;is+=blockDim.x) {
#pragma unroll
            for(int d=0;d<3;++d) opt_load_store(r[d],args[d],0,is);
            sgd_math();
#pragma unroll
            for(int d=0;d<3;++d){
                if(d!=1||grad_scale)opt_load_store(args[d],r[d],is,0);
            }
        }
    } else {
        for (int64_t is=0;is<n&&is<chunk_size;
             is+=static_cast<int64_t>(blockDim.x)*kOptILP) {
            opt_load_args(r,args,3,is,chunk_size,n);
            sgd_math();
            opt_store_args(args,r,3,is,chunk_size,n,!!grad_scale,grad_scale);
        }
    }
}

// ---------------------------------------------------------------------------
// Launchers: build chunk maps, upload metadata, dispatch to dual-path kernels.
// Float32-only fast path; other dtypes not yet ported to dual-path.
// ---------------------------------------------------------------------------

template <typename scalar_t, typename math_t, typename lr_t>
void launch_fused_sgd(const std::vector<Tensor>& params,
                      const std::vector<Tensor>& grads,
                      const std::vector<Tensor>& momentum_buffers,
                      double lr, double momentum, double dampening,
                      double weight_decay, bool nesterov, bool maximize,
                      bool is_first_step,
                      const std::optional<Tensor>& grad_scale,
                      const std::optional<Tensor>& found_inf,
                      const Tensor* tensor_lr) {
    const auto stream = getCurrentCUDAStream().stream();
    std::vector<float*> param_ptrs, grad_ptrs, buffer_ptrs;
    std::vector<int64_t> numels;
    for (size_t i = 0; i < params.size(); ++i) {
        param_ptrs.push_back(params[i].data_ptr<float>());
        grad_ptrs.push_back(grads[i].data_ptr<float>());
        buffer_ptrs.push_back(momentum_buffers.empty() ? nullptr
            : momentum_buffers[i].data_ptr<float>());
        numels.push_back(params[i].numel());
    }
    DeviceArray<float*> d_params(stream, param_ptrs);
    DeviceArray<float*> d_grads(stream, grad_ptrs);
    DeviceArray<float*> d_buffers(stream, buffer_ptrs);
    DeviceArray<int64_t> d_numels(stream, numels);
    std::vector<int32_t> b2t; std::vector<int64_t> b2c;
    for (size_t t = 0; t < numels.size(); ++t) {
        const int64_t pieces = (numels[t] + kFusedChunk - 1) / kFusedChunk;
        for (int64_t c = 0; c < pieces; ++c) { b2t.push_back(static_cast<int32_t>(t)); b2c.push_back(c); }
    }
    DeviceArray<int32_t> d_b2t(stream, b2t);
    DeviceArray<int64_t> d_b2c(stream, b2c);
    const lr_t* lr_ptr = tensor_lr ? tensor_lr->data_ptr<lr_t>() : nullptr;
    const float* scale_ptr = optional_fused_float_ptr(grad_scale, "grad_scale");
    const float* found_ptr = optional_fused_float_ptr(found_inf, "found_inf");
    const unsigned grid = static_cast<unsigned>(b2t.size());
    fused_sgd_kernel<lr_t><<<grid, kFusedBlock, 0, stream>>>(
        d_params.data(), d_grads.data(), d_buffers.data(),
        d_numels.data(), d_b2t.data(), d_b2c.data(), kFusedChunk,
        lr_ptr, lr, momentum, dampening, weight_decay, nesterov ? 1 : 0,
        maximize ? 1 : 0, is_first_step ? 1 : 0, scale_ptr, found_ptr,
        static_cast<int64_t>(params.size()));
    checkCuda(cudaGetLastError(), "_fused_sgd kernel launch");
}

template <typename scalar_t, typename math_t, typename lr_t>
void launch_fused_adam(const std::vector<Tensor>& params,
                       const std::vector<Tensor>& grads,
                       const std::vector<Tensor>& exp_avgs,
                       const std::vector<Tensor>& exp_avg_sqs,
                       const std::vector<Tensor>& max_exp_avg_sqs,
                       const std::vector<Tensor>& state_steps,
                       double lr, double beta1, double beta2,
                       double weight_decay, double eps,
                       bool amsgrad, bool maximize, bool adamw,
                       const std::optional<Tensor>& grad_scale,
                       const std::optional<Tensor>& found_inf,
                       const Tensor* tensor_lr) {
    const auto stream = getCurrentCUDAStream().stream();
    std::vector<float*> p_ptrs, g_ptrs, m_ptrs, v_ptrs, n_ptrs;
    std::vector<float*> step_ptrs;
    std::vector<int64_t> numels;
    for (size_t i = 0; i < params.size(); ++i) {
        p_ptrs.push_back(params[i].data_ptr<float>());
        g_ptrs.push_back(grads[i].data_ptr<float>());
        m_ptrs.push_back(exp_avgs[i].data_ptr<float>());
        v_ptrs.push_back(exp_avg_sqs[i].data_ptr<float>());
        n_ptrs.push_back(amsgrad ? max_exp_avg_sqs[i].data_ptr<float>() : nullptr);
        step_ptrs.push_back(state_steps[i].data_ptr<float>());
        numels.push_back(params[i].numel());
    }
    DeviceArray<float*> d_p(stream, p_ptrs);
    DeviceArray<float*> d_g(stream, g_ptrs);
    DeviceArray<float*> d_m(stream, m_ptrs);
    DeviceArray<float*> d_v(stream, v_ptrs);
    DeviceArray<float*> d_n(stream, n_ptrs);
    DeviceArray<float*> d_steps(stream, step_ptrs);
    DeviceArray<int64_t> d_numels(stream, numels);
    std::vector<int32_t> b2t; std::vector<int64_t> b2c;
    for (size_t t = 0; t < numels.size(); ++t) {
        const int64_t pieces = (numels[t] + kFusedChunk - 1) / kFusedChunk;
        for (int64_t c = 0; c < pieces; ++c) { b2t.push_back(static_cast<int32_t>(t)); b2c.push_back(c); }
    }
    DeviceArray<int32_t> d_b2t(stream, b2t);
    DeviceArray<int64_t> d_b2c(stream, b2c);
    const lr_t* lr_ptr = tensor_lr ? tensor_lr->data_ptr<lr_t>() : nullptr;
    const float* scale_ptr = optional_fused_float_ptr(grad_scale, "grad_scale");
    const float* found_ptr = optional_fused_float_ptr(found_inf, "found_inf");
    const unsigned grid = static_cast<unsigned>(b2t.size());

    if (amsgrad) {
        fused_adam_kernel<lr_t, 5, true, true><<<grid, kFusedBlock, 0, stream>>>(
            d_p.data(), d_g.data(), d_m.data(), d_v.data(), d_n.data(),
            d_steps.data(), d_numels.data(), d_b2t.data(), d_b2c.data(), kFusedChunk,
            lr_ptr, lr, beta1, beta2, weight_decay, eps,
            maximize ? 1 : 0, scale_ptr, found_ptr,
            static_cast<int64_t>(params.size()));
    } else {
        fused_adam_kernel<lr_t, 4, false, false><<<grid, kFusedBlock, 0, stream>>>(
            d_p.data(), d_g.data(), d_m.data(), d_v.data(), d_n.data(),
            d_steps.data(), d_numels.data(), d_b2t.data(), d_b2c.data(), kFusedChunk,
            lr_ptr, lr, beta1, beta2, weight_decay, eps,
            maximize ? 1 : 0, scale_ptr, found_ptr,
            static_cast<int64_t>(params.size()));
    }
    checkCuda(cudaGetLastError(), "_fused_adam kernel launch");
}


template <typename F>
void dispatch_fused_cuda_dtype(const std::vector<Tensor>& params,
                               const char* op_name,
                               F&& fn) {
    if (params.empty()) return;
    switch (params[0].dtype()) {
        case DType::Float32: fn.template operator()<float, float>(); break;
        default:
            TP_THROW(NotImplementedError, std::string(op_name) +
                ": dual-path optimizer currently supports Float32 only");
    }
}
template <typename F>
void dispatch_fused_cuda_lr(const Tensor* lr, F&& fn) {
    if (!lr) {
        fn.template operator()<double>();
    } else if (lr->dtype() == DType::Float32) {
        fn.template operator()<float>();
    } else if (lr->dtype() == DType::Float64) {
        fn.template operator()<double>();
    } else {
        TP_THROW(NotImplementedError, "fused optimizer Tensor lr must be float32 or float64");
    }
}


template <typename scalar_t, typename math_t, typename lr_t>
void launch_fused_adagrad(const std::vector<Tensor>& params,
                          const std::vector<Tensor>& grads,
                          const std::vector<Tensor>& state_sums,
                          const std::vector<Tensor>& state_steps,
                          double lr, double lr_decay, double weight_decay,
                          double eps, bool maximize,
                          const std::optional<Tensor>& grad_scale,
                          const std::optional<Tensor>& found_inf,
                          const Tensor* tensor_lr) {
    TP_THROW(NotImplementedError, "_fused_adagrad_ dual-path pending migration");
}


void fused_sgd_cuda_impl(std::vector<Tensor> params,
                         const std::vector<Tensor>& grads,
                         const std::vector<Tensor>& momentum_buffers,
                         double lr,
                         double momentum,
                         double dampening,
                         double weight_decay,
                         bool nesterov,
                         bool maximize,
                         bool is_first_step,
                         const std::optional<Tensor>& grad_scale,
                         const std::optional<Tensor>& found_inf,
                         const Tensor* tensor_lr) {
    validate_fused_pairs(params, grads, "_fused_sgd_");
    if (params.empty()) return;
    if (momentum == 0.0) {
        if (!momentum_buffers.empty()) {
            TP_THROW(ValueError, "_fused_sgd_: momentum buffer list must be empty when momentum is zero");
        }
    } else {
        validate_fused_state(params, momentum_buffers, true, "_fused_sgd_");
    }
    dispatch_fused_cuda_lr(tensor_lr, [&]<typename lr_t>() {
        dispatch_fused_cuda_dtype(params, "_fused_sgd_", [&]<typename scalar_t, typename math_t>() {
            launch_fused_sgd<scalar_t, math_t, lr_t>(params, grads,
                momentum_buffers, lr, momentum, dampening, weight_decay,
                nesterov, maximize, is_first_step, grad_scale, found_inf,
                tensor_lr);
        });
    });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void fused_adam_cuda_impl(std::vector<Tensor> params,
                          const std::vector<Tensor>& grads,
                          const std::vector<Tensor>& exp_avgs,
                          const std::vector<Tensor>& exp_avg_sqs,
                          const std::vector<Tensor>& max_exp_avg_sqs,
                          const std::vector<Tensor>& state_steps,
                          double lr,
                          double beta1,
                          double beta2,
                          double weight_decay,
                          double eps,
                          bool amsgrad,
                          bool maximize,
                          bool adamw,
                          const std::optional<Tensor>& grad_scale,
                          const std::optional<Tensor>& found_inf,
                          const Tensor* tensor_lr) {
    const char* op_name = adamw ? "_fused_adamw_" : "_fused_adam_";
    validate_fused_pairs(params, grads, op_name);
    if (params.empty()) return;
    validate_fused_state(params, exp_avgs, true, op_name);
    validate_fused_state(params, exp_avg_sqs, true, op_name);
    validate_fused_state(params, max_exp_avg_sqs, amsgrad, op_name);
    if (!amsgrad && !max_exp_avg_sqs.empty()) {
        TP_THROW(ValueError, std::string(op_name) +
            ": max_exp_avg_sqs must be empty when amsgrad is false");
    }
    validate_fused_steps(params, state_steps, op_name);
    dispatch_fused_cuda_lr(tensor_lr, [&]<typename lr_t>() {
        dispatch_fused_cuda_dtype(params, op_name, [&]<typename scalar_t, typename math_t>() {
            launch_fused_adam<scalar_t, math_t, lr_t>(params, grads, exp_avgs,
                exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2,
                weight_decay, eps, amsgrad, maximize, adamw, grad_scale,
                found_inf, tensor_lr);
        });
    });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void fused_adagrad_cuda_impl(std::vector<Tensor> params,
                             const std::vector<Tensor>& grads,
                             const std::vector<Tensor>& state_sums,
                             const std::vector<Tensor>& state_steps,
                             double lr,
                             double lr_decay,
                             double weight_decay,
                             double eps,
                             bool maximize,
                             const std::optional<Tensor>& grad_scale,
                             const std::optional<Tensor>& found_inf,
                             const Tensor* tensor_lr) {
    validate_fused_pairs(params, grads, "_fused_adagrad_");
    if (params.empty()) return;
    validate_fused_state(params, state_sums, true, "_fused_adagrad_");
    validate_fused_steps(params, state_steps, "_fused_adagrad_");
    dispatch_fused_cuda_lr(tensor_lr, [&]<typename lr_t>() {
        dispatch_fused_cuda_dtype(params, "_fused_adagrad_", [&]<typename scalar_t, typename math_t>() {
            launch_fused_adagrad<scalar_t, math_t, lr_t>(params, grads,
                state_sums, state_steps, lr, lr_decay, weight_decay, eps,
                maximize, grad_scale, found_inf, tensor_lr);
        });
    });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void fused_adam_cuda(std::vector<Tensor> params,
                     std::vector<Tensor> grads,
                     std::vector<Tensor> exp_avgs,
                     std::vector<Tensor> exp_avg_sqs,
                     std::vector<Tensor> max_exp_avg_sqs,
                     const std::vector<Tensor>& state_steps,
                     double lr, double beta1, double beta2, double weight_decay,
                     double eps, bool amsgrad, bool maximize,
                     const std::optional<Tensor>& grad_scale,
                     const std::optional<Tensor>& found_inf) {
    fused_adam_cuda_impl(std::move(params), grads, exp_avgs, exp_avg_sqs,
        max_exp_avg_sqs, state_steps, lr, beta1,
        beta2, weight_decay, eps, amsgrad,
        maximize, false, grad_scale, found_inf, nullptr);
}

void fused_adam_tensor_lr_cuda(std::vector<Tensor> params,
                               std::vector<Tensor> grads,
                               std::vector<Tensor> exp_avgs,
                               std::vector<Tensor> exp_avg_sqs,
                               std::vector<Tensor> max_exp_avg_sqs,
                               const std::vector<Tensor>& state_steps,
                               const Tensor& lr, double beta1, double beta2,
                               double weight_decay, double eps, bool amsgrad,
                               bool maximize,
                               const std::optional<Tensor>& grad_scale,
                               const std::optional<Tensor>& found_inf) {
    fused_adam_cuda_impl(std::move(params), grads, exp_avgs, exp_avg_sqs,
        max_exp_avg_sqs, state_steps, 0.0, beta1, beta2,
        weight_decay, eps, amsgrad, maximize, false,
        grad_scale, found_inf, &lr);
}

void fused_adamw_cuda(std::vector<Tensor> params,
                      std::vector<Tensor> grads,
                      std::vector<Tensor> exp_avgs,
                      std::vector<Tensor> exp_avg_sqs,
                      std::vector<Tensor> max_exp_avg_sqs,
                      const std::vector<Tensor>& state_steps,
                      double lr, double beta1, double beta2, double weight_decay,
                      double eps, bool amsgrad, bool maximize,
                      const std::optional<Tensor>& grad_scale,
                      const std::optional<Tensor>& found_inf) {
    fused_adam_cuda_impl(std::move(params), grads, exp_avgs, exp_avg_sqs,
        max_exp_avg_sqs, state_steps, lr, beta1,
        beta2, weight_decay, eps, amsgrad,
        maximize, true, grad_scale, found_inf, nullptr);
}

void fused_adamw_tensor_lr_cuda(std::vector<Tensor> params,
                                std::vector<Tensor> grads,
                                std::vector<Tensor> exp_avgs,
                                std::vector<Tensor> exp_avg_sqs,
                                std::vector<Tensor> max_exp_avg_sqs,
                                const std::vector<Tensor>& state_steps,
                                const Tensor& lr, double beta1, double beta2,
                                double weight_decay, double eps, bool amsgrad,
                                bool maximize,
                                const std::optional<Tensor>& grad_scale,
                                const std::optional<Tensor>& found_inf) {
    fused_adam_cuda_impl(std::move(params), grads, exp_avgs, exp_avg_sqs,
        max_exp_avg_sqs, state_steps, 0.0, beta1, beta2,
        weight_decay, eps, amsgrad, maximize, true,
        grad_scale, found_inf, &lr);
}

void fused_sgd_cuda(std::vector<Tensor> params,
                    std::vector<Tensor> grads,
                    std::vector<Tensor> momentum_buffers,
                    double weight_decay, double momentum, double lr,
                    double dampening, bool nesterov, bool maximize,
                    bool is_first_step, const std::optional<Tensor>& grad_scale,
                    const std::optional<Tensor>& found_inf) {
    fused_sgd_cuda_impl(std::move(params), grads, momentum_buffers,
        lr, momentum, dampening,
        weight_decay, nesterov, maximize, is_first_step,
        grad_scale, found_inf, nullptr);
}

void fused_sgd_tensor_lr_cuda(std::vector<Tensor> params,
                              std::vector<Tensor> grads,
                              std::vector<Tensor> momentum_buffers,
                              double weight_decay, double momentum,
                              const Tensor& lr, double dampening, bool nesterov,
                              bool maximize, bool is_first_step,
                              const std::optional<Tensor>& grad_scale,
                              const std::optional<Tensor>& found_inf) {
    fused_sgd_cuda_impl(std::move(params), grads, momentum_buffers, 0.0,
        momentum, dampening, weight_decay,
        nesterov, maximize, is_first_step, grad_scale, found_inf, &lr);
}

void fused_adagrad_cuda(std::vector<Tensor> params,
                        std::vector<Tensor> grads,
                        std::vector<Tensor> state_sums,
                        std::vector<Tensor> state_steps,
                        double lr, double lr_decay, double weight_decay,
                        double eps, bool maximize,
                        const std::optional<Tensor>& grad_scale,
                        const std::optional<Tensor>& found_inf) {
    fused_adagrad_cuda_impl(std::move(params), grads, state_sums, state_steps,
        lr, lr_decay, weight_decay,
        eps, maximize, grad_scale, found_inf, nullptr);
}

void fused_adagrad_tensor_lr_cuda(std::vector<Tensor> params,
                                  std::vector<Tensor> grads,
                                  std::vector<Tensor> state_sums,
                                  std::vector<Tensor> state_steps,
                                  const Tensor& lr, double lr_decay,
                                  double weight_decay, double eps, bool maximize,
                                  const std::optional<Tensor>& grad_scale,
                                  const std::optional<Tensor>& found_inf) {
    fused_adagrad_cuda_impl(std::move(params), grads, state_sums, state_steps,
        0.0, lr_decay, weight_decay, eps,
        maximize, grad_scale, found_inf, &lr);
}

} // namespace

std::vector<Tensor> foreach_sgd_cuda(const std::vector<Tensor>& params,
                                      const std::vector<Tensor>& grads,
                                      const std::vector<Tensor>& momentum_buffers,
                                      double lr,
                                      double momentum,
                                      double dampening,
                                      double weight_decay,
                                      bool nesterov,
                                      bool first_momentum_step) {
    std::vector<Tensor> empty_states(params.size());
    std::vector<int64_t> no_steps;
    validate_lists(params, grads, momentum_buffers, empty_states, empty_states,
                   no_steps, momentum != 0.0, false, false, "_foreach_sgd");
    if (params.empty()) return params;
    if (params[0].dtype() == DType::Float32) {
        launch_sgd<float>(params, grads, momentum_buffers, lr, momentum,
                          dampening, weight_decay, nesterov, first_momentum_step);
    } else if (params[0].dtype() == DType::Float64) {
        launch_sgd<double>(params, grads, momentum_buffers, lr, momentum,
                           dampening, weight_decay, nesterov, first_momentum_step);
    } else {
        TP_THROW(NotImplementedError,
                 "_foreach_sgd supports float32 and float64 CUDA tensors");
    }
    // Match PyTorch's in-place optimizer contract: the parameter version
    // changes immediately after the queued update, even though the CUDA
    // kernel itself executes asynchronously.
    for (const auto& param : params) {
        param.unsafeGetTensorImpl()->bump_version();
    }
    return params;
}

std::vector<Tensor> foreach_adam_cuda(const std::vector<Tensor>& params,
                                       const std::vector<Tensor>& grads,
                                       const std::vector<Tensor>& exp_avgs,
                                       const std::vector<Tensor>& exp_avg_sqs,
                                       const std::vector<Tensor>& max_exp_avg_sqs,
                                       const std::vector<int64_t>& steps,
                                       double lr,
                                       double beta1,
                                       double beta2,
                                       double eps,
                                       double weight_decay,
                                       bool amsgrad) {
    if (steps.size() != params.size()) {
        TP_THROW(ValueError, "_foreach_adam: step list size must match parameter list");
    }
    std::vector<Tensor> empty_states(params.size());
    validate_lists(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                   steps, true, true, amsgrad, "_foreach_adam");
    if (params.empty()) return params;
    if (params[0].dtype() == DType::Float32) {
        launch_adam<float>(params, grads, exp_avgs, exp_avg_sqs,
                           max_exp_avg_sqs, steps, lr, beta1, beta2, eps,
                           weight_decay, amsgrad);
    } else if (params[0].dtype() == DType::Float64) {
        launch_adam<double>(params, grads, exp_avgs, exp_avg_sqs,
                            max_exp_avg_sqs, steps, lr, beta1, beta2, eps,
                            weight_decay, amsgrad);
    } else {
        TP_THROW(NotImplementedError,
                 "_foreach_adam supports float32 and float64 CUDA tensors");
    }
    for (const auto& param : params) {
        param.unsafeGetTensorImpl()->bump_version();
    }
    return params;
}

namespace {

template <typename F>
std::vector<Tensor> foreach_map(const std::vector<Tensor>& self, F&& fn) {
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (const Tensor& value : self) result.push_back(fn(value));
    return result;
}

template <typename F>
std::vector<Tensor> foreach_map_inplace(std::vector<Tensor> self, F&& fn) {
    for (Tensor& value : self) fn(value);
    return self;
}

template <typename F>
std::vector<Tensor> foreach_map_pair(const std::vector<Tensor>& self,
                                     const std::vector<Tensor>& other, F&& fn) {
    if (self.size() != other.size()) {
        TP_THROW(ValueError, "foreach tensor list arguments must have the same length");
    }
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) result.push_back(fn(self[i], other[i]));
    return result;
}

template <typename F>
std::vector<Tensor> foreach_map_pair_inplace(std::vector<Tensor> self,
                                             const std::vector<Tensor>& other,
                                             F&& fn) {
    if (self.size() != other.size()) {
        TP_THROW(ValueError, "foreach tensor list arguments must have the same length");
    }
    for (size_t i = 0; i < self.size(); ++i) fn(self[i], other[i]);
    return self;
}

template <typename F>
std::vector<Tensor> foreach_map_scalars(const std::vector<Tensor>& self,
                                        const std::vector<Scalar>& scalars, F&& fn) {
    if (self.size() != scalars.size()) {
        TP_THROW(ValueError, "foreach tensor and scalar lists must have the same length");
    }
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) result.push_back(fn(self[i], scalars[i]));
    return result;
}

template <typename F>
std::vector<Tensor> foreach_map_scalars_inplace(std::vector<Tensor> self,
                                                const std::vector<Scalar>& scalars,
                                                F&& fn) {
    if (self.size() != scalars.size()) {
        TP_THROW(ValueError, "foreach tensor and scalar lists must have the same length");
    }
    for (size_t i = 0; i < self.size(); ++i) fn(self[i], scalars[i]);
    return self;
}

template <typename F>
std::vector<Tensor> foreach_map_ternary(const std::vector<Tensor>& self,
                                        const std::vector<Tensor>& tensor1,
                                        const std::vector<Tensor>& tensor2, F&& fn) {
    if (self.size() != tensor1.size() || self.size() != tensor2.size()) {
        TP_THROW(ValueError, "foreach ternary tensor lists must have the same length");
    }
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) result.push_back(fn(self[i], tensor1[i], tensor2[i]));
    return result;
}

template <typename F>
std::vector<Tensor> foreach_map_ternary_inplace(std::vector<Tensor> self,
                                                const std::vector<Tensor>& tensor1,
                                                const std::vector<Tensor>& tensor2, F&& fn) {
    if (self.size() != tensor1.size() || self.size() != tensor2.size()) {
        TP_THROW(ValueError, "foreach ternary tensor lists must have the same length");
    }
    for (size_t i = 0; i < self.size(); ++i) fn(self[i], tensor1[i], tensor2[i]);
    return self;
}

template <typename F>
std::vector<Tensor> foreach_map_ternary_scalar_lists(
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars,
        F&& fn) {
    if (self.size() != tensor1.size() || self.size() != tensor2.size() ||
        self.size() != scalars.size()) {
        TP_THROW(ValueError, "foreach ternary tensor/scalar lists must have the same length");
    }
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
        result.push_back(fn(self[i], tensor1[i], tensor2[i], scalars[i]));
    }
    return result;
}

template <typename F>
std::vector<Tensor> foreach_map_ternary_scalar_lists_inplace(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars,
        F&& fn) {
    if (self.size() != tensor1.size() || self.size() != tensor2.size() ||
        self.size() != scalars.size()) {
        TP_THROW(ValueError, "foreach ternary tensor/scalar lists must have the same length");
    }
    for (size_t i = 0; i < self.size(); ++i) {
        fn(self[i], tensor1[i], tensor2[i], scalars[i]);
    }
    return self;
}

template <typename F>
std::vector<Tensor> foreach_map_pair_scalars(
        const std::vector<Tensor>& self, const std::vector<Tensor>& other,
        const std::vector<Scalar>& scalars, F&& fn) {
    if (self.size() != other.size() || self.size() != scalars.size()) {
        TP_THROW(ValueError, "foreach tensor/scalar lists must have the same length");
    }
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
        result.push_back(fn(self[i], other[i], scalars[i]));
    }
    return result;
}

template <typename F>
std::vector<Tensor> foreach_map_pair_scalars_inplace(
        std::vector<Tensor> self, const std::vector<Tensor>& other,
        const std::vector<Scalar>& scalars, F&& fn) {
    if (self.size() != other.size() || self.size() != scalars.size()) {
        TP_THROW(ValueError, "foreach tensor/scalar lists must have the same length");
    }
    for (size_t i = 0; i < self.size(); ++i) {
        fn(self[i], other[i], scalars[i]);
    }
    return self;
}

#define DEFINE_FOREACH_ADD_SUB(NAME, METHOD) \
std::vector<Tensor> foreach_##NAME##_scalar_cuda(const std::vector<Tensor>& self, Scalar scalar) { \
    return foreach_map(self, [&](const Tensor& value) { return value.METHOD(scalar); }); \
} \
std::vector<Tensor> foreach_##NAME##_list_cuda(const std::vector<Tensor>& self, const std::vector<Tensor>& other, Scalar alpha) { \
    return foreach_map_pair(self, other, [&](const Tensor& value, const Tensor& rhs) { return value.METHOD(rhs, alpha); }); \
} \
std::vector<Tensor> foreach_##NAME##_scalar_list_cuda(const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) { \
    return foreach_map_scalars(self, scalars, [&](const Tensor& value, Scalar scalar) { return value.METHOD(scalar); }); \
} \
std::vector<Tensor> foreach_##NAME##_tensor_cuda(const std::vector<Tensor>& self, const Tensor& other, Scalar alpha) { \
    return foreach_map(self, [&](const Tensor& value) { return value.METHOD(other, alpha); }); \
} \
void foreach_##NAME##_scalar_inplace_cuda(std::vector<Tensor> self, Scalar scalar) { \
    foreach_map_inplace(self, [&](Tensor& value) { value.METHOD##_(scalar); }); \
} \
void foreach_##NAME##_list_inplace_cuda(std::vector<Tensor> self, const std::vector<Tensor>& other, Scalar alpha) { \
    foreach_map_pair_inplace(self, other, [&](Tensor& value, const Tensor& rhs) { value.METHOD##_(rhs, alpha); }); \
} \
void foreach_##NAME##_scalar_list_inplace_cuda(std::vector<Tensor> self, const std::vector<Scalar>& scalars) { \
    foreach_map_scalars_inplace(self, scalars, [&](Tensor& value, Scalar scalar) { value.METHOD##_(scalar); }); \
} \
void foreach_##NAME##_tensor_inplace_cuda(std::vector<Tensor> self, const Tensor& other, Scalar alpha) { \
    foreach_map_inplace(self, [&](Tensor& value) { value.METHOD##_(other, alpha); }); \
}

DEFINE_FOREACH_ADD_SUB(sub, sub)
#undef DEFINE_FOREACH_ADD_SUB

std::vector<Tensor> foreach_add_scalar_cuda(const std::vector<Tensor>& self, Scalar scalar) {
    return foreach_map(self, [&](const Tensor& value) { return value.add(scalar); });
}
std::vector<Tensor> foreach_add_list_cuda(const std::vector<Tensor>& self, const std::vector<Tensor>& other, Scalar alpha) {
    return foreach_map_pair(self, other, [&](const Tensor& value, const Tensor& rhs) { return value.add(rhs, alpha); });
}
std::vector<Tensor> foreach_add_scalar_list_cuda(const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {
    return foreach_map_scalars(self, scalars, [&](const Tensor& value, Scalar scalar) { return value.add(scalar); });
}
std::vector<Tensor> foreach_add_tensor_cuda(const std::vector<Tensor>& self, const Tensor& other, Scalar alpha) {
    return foreach_map(self, [&](const Tensor& value) { return value.add(other, alpha); });
}
void foreach_add_scalar_inplace_cuda(std::vector<Tensor> self, Scalar scalar) {
    foreach_map_inplace(self, [&](Tensor& value) { value.add_(scalar); });
}
void foreach_add_list_inplace_cuda(std::vector<Tensor> self, const std::vector<Tensor>& other, Scalar alpha) {
    foreach_map_pair_inplace(self, other, [&](Tensor& value, const Tensor& rhs) { value.add_(rhs, alpha); });
}
void foreach_add_scalar_list_inplace_cuda(std::vector<Tensor> self, const std::vector<Scalar>& scalars) {
    foreach_map_scalars_inplace(self, scalars, [&](Tensor& value, Scalar scalar) { value.add_(scalar); });
}
void foreach_add_tensor_inplace_cuda(std::vector<Tensor> self, const Tensor& other, Scalar alpha) {
    foreach_map_inplace(self, [&](Tensor& value) { value.add_(other, alpha); });
}

#define DEFINE_FOREACH_MUL_DIV(NAME, METHOD) \
std::vector<Tensor> foreach_##NAME##_scalar_cuda(const std::vector<Tensor>& self, Scalar scalar) { \
    return foreach_map(self, [&](const Tensor& value) { return value.METHOD(scalar); }); \
} \
std::vector<Tensor> foreach_##NAME##_list_cuda(const std::vector<Tensor>& self, const std::vector<Tensor>& other) { \
    return foreach_map_pair(self, other, [&](const Tensor& value, const Tensor& rhs) { return value.METHOD(rhs); }); \
} \
std::vector<Tensor> foreach_##NAME##_scalar_list_cuda(const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) { \
    return foreach_map_scalars(self, scalars, [&](const Tensor& value, Scalar scalar) { return value.METHOD(scalar); }); \
} \
std::vector<Tensor> foreach_##NAME##_tensor_cuda(const std::vector<Tensor>& self, const Tensor& other) { \
    return foreach_map(self, [&](const Tensor& value) { return value.METHOD(other); }); \
} \
void foreach_##NAME##_scalar_inplace_cuda(std::vector<Tensor> self, Scalar scalar) { \
    foreach_map_inplace(self, [&](Tensor& value) { value.METHOD##_(scalar); }); \
} \
void foreach_##NAME##_list_inplace_cuda(std::vector<Tensor> self, const std::vector<Tensor>& other) { \
    foreach_map_pair_inplace(self, other, [&](Tensor& value, const Tensor& rhs) { value.METHOD##_(rhs); }); \
} \
void foreach_##NAME##_scalar_list_inplace_cuda(std::vector<Tensor> self, const std::vector<Scalar>& scalars) { \
    foreach_map_scalars_inplace(self, scalars, [&](Tensor& value, Scalar scalar) { value.METHOD##_(scalar); }); \
} \
void foreach_##NAME##_tensor_inplace_cuda(std::vector<Tensor> self, const Tensor& other) { \
    foreach_map_inplace(self, [&](Tensor& value) { value.METHOD##_(other); }); \
}

DEFINE_FOREACH_MUL_DIV(mul, mul)
DEFINE_FOREACH_MUL_DIV(div, div)
#undef DEFINE_FOREACH_MUL_DIV

#define DEFINE_FOREACH_UNARY(NAME, METHOD) \
std::vector<Tensor> foreach_##NAME##_cuda(const std::vector<Tensor>& self) { \
    return foreach_map(self, [&](const Tensor& value) { return value.METHOD(); }); \
} \
void foreach_##NAME##_inplace_cuda(std::vector<Tensor> self) { \
    foreach_map_inplace(self, [&](Tensor& value) { value.copy_(value.METHOD()); }); \
}
DEFINE_FOREACH_UNARY(sqrt, sqrt)
DEFINE_FOREACH_UNARY(rsqrt, rsqrt)
DEFINE_FOREACH_UNARY(neg, neg)
DEFINE_FOREACH_UNARY(abs, abs)
DEFINE_FOREACH_UNARY(sign, sign)
#undef DEFINE_FOREACH_UNARY

std::vector<Tensor> foreach_reciprocal_cuda(const std::vector<Tensor>& self) {
    return foreach_map(self, [&](const Tensor& value) {
        return value.pow(Scalar(-1));
    });
}
void foreach_reciprocal_inplace_cuda(std::vector<Tensor> self) {
    foreach_map_inplace(self, [&](Tensor& value) {
        value.copy_(value.pow(Scalar(-1)));
    });
}

std::vector<Tensor> foreach_addcmul_scalar_cuda(const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1, const std::vector<Tensor>& tensor2, Scalar value) {
    return foreach_map_ternary(self, tensor1, tensor2, [&](const Tensor& x, const Tensor& a, const Tensor& b) { return x.addcmul(a, b, value); });
}
void foreach_addcmul_scalar_inplace_cuda(std::vector<Tensor> self, const std::vector<Tensor>& tensor1, const std::vector<Tensor>& tensor2, Scalar value) {
    foreach_map_ternary_inplace(self, tensor1, tensor2, [&](Tensor& x, const Tensor& a, const Tensor& b) { x.addcmul_(a, b, value); });
}
std::vector<Tensor> foreach_addcdiv_scalar_cuda(const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1, const std::vector<Tensor>& tensor2, Scalar value) {
    return foreach_map_ternary(self, tensor1, tensor2, [&](const Tensor& x, const Tensor& a, const Tensor& b) { return x.addcdiv(a, b, value); });
}
void foreach_addcdiv_scalar_inplace_cuda(std::vector<Tensor> self, const std::vector<Tensor>& tensor1, const std::vector<Tensor>& tensor2, Scalar value) {
    foreach_map_ternary_inplace(self, tensor1, tensor2, [&](Tensor& x, const Tensor& a, const Tensor& b) { x.addcdiv_(a, b, value); });
}

std::vector<Tensor> foreach_addcmul_scalar_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars) {
    return foreach_map_ternary_scalar_lists(self, tensor1, tensor2, scalars,
        [&](const Tensor& x, const Tensor& a, const Tensor& b, Scalar value) {
            return x.addcmul(a, b, value);
        });
}
void foreach_addcmul_scalar_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars) {
    foreach_map_ternary_scalar_lists_inplace(std::move(self), tensor1, tensor2, scalars,
        [&](Tensor& x, const Tensor& a, const Tensor& b, Scalar value) {
            x.addcmul_(a, b, value);
        });
}
std::vector<Tensor> foreach_addcmul_tensor_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const Tensor& scalars) {
    return foreach_addcmul_scalar_cuda(self, tensor1, tensor2, scalars.item());
}
void foreach_addcmul_tensor_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const Tensor& scalars) {
    foreach_addcmul_scalar_inplace_cuda(std::move(self), tensor1, tensor2, scalars.item());
}

std::vector<Tensor> foreach_addcdiv_scalar_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars) {
    return foreach_map_ternary_scalar_lists(self, tensor1, tensor2, scalars,
        [&](const Tensor& x, const Tensor& a, const Tensor& b, Scalar value) {
            return x.addcdiv(a, b, value);
        });
}
void foreach_addcdiv_scalar_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars) {
    foreach_map_ternary_scalar_lists_inplace(std::move(self), tensor1, tensor2, scalars,
        [&](Tensor& x, const Tensor& a, const Tensor& b, Scalar value) {
            x.addcdiv_(a, b, value);
        });
}
std::vector<Tensor> foreach_addcdiv_tensor_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const Tensor& scalars) {
    return foreach_addcdiv_scalar_cuda(self, tensor1, tensor2, scalars.item());
}
void foreach_addcdiv_tensor_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const Tensor& scalars) {
    foreach_addcdiv_scalar_inplace_cuda(std::move(self), tensor1, tensor2, scalars.item());
}

std::vector<Tensor> foreach_lerp_scalar_cuda(const std::vector<Tensor>& self, const std::vector<Tensor>& end, Scalar weight) {
    return foreach_map_pair(self, end, [&](const Tensor& x, const Tensor& y) { return x.lerp(y, weight); });
}
std::vector<Tensor> foreach_lerp_list_cuda(const std::vector<Tensor>& self, const std::vector<Tensor>& end, const std::vector<Tensor>& weight) {
    if (self.size() != end.size() || self.size() != weight.size()) TP_THROW(ValueError, "foreach lerp lists must have the same length");
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) result.push_back(self[i].lerp(end[i], weight[i]));
    return result;
}
void foreach_lerp_scalar_inplace_cuda(std::vector<Tensor> self, const std::vector<Tensor>& end, Scalar weight) {
    foreach_map_pair_inplace(self, end, [&](Tensor& x, const Tensor& y) { x.copy_(x.lerp(y, weight)); });
}
void foreach_lerp_list_inplace_cuda(std::vector<Tensor> self, const std::vector<Tensor>& end, const std::vector<Tensor>& weight) {
    if (self.size() != end.size() || self.size() != weight.size()) TP_THROW(ValueError, "foreach lerp lists must have the same length");
    for (size_t i = 0; i < self.size(); ++i) self[i].copy_(self[i].lerp(end[i], weight[i]));
}
std::vector<Tensor> foreach_lerp_scalar_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& end,
        const std::vector<Scalar>& weight) {
    return foreach_map_pair_scalars(self, end, weight,
        [&](const Tensor& x, const Tensor& y, Scalar w) { return x.lerp(y, w); });
}
void foreach_lerp_scalar_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& end,
        const std::vector<Scalar>& weight) {
    foreach_map_pair_scalars_inplace(std::move(self), end, weight,
        [&](Tensor& x, const Tensor& y, Scalar w) { x.copy_(x.lerp(y, w)); });
}

std::vector<Tensor> foreach_pow_scalar_cuda(const std::vector<Tensor>& self, Scalar exponent) {
    return foreach_map(self, [&](const Tensor& value) { return value.pow(exponent); });
}
std::vector<Tensor> foreach_pow_scalar_tensor_cuda(
        Scalar self, const std::vector<Tensor>& exponent) {
    return foreach_map(exponent, [&](const Tensor& value) {
        Tensor base = Tensor::full({}, self, value.dtype(), value.device());
        return base.pow(value);
    });
}
std::vector<Tensor> foreach_pow_list_cuda(const std::vector<Tensor>& self, const std::vector<Tensor>& exponent) {
    return foreach_map_pair(self, exponent, [&](const Tensor& value, const Tensor& rhs) { return value.pow(rhs); });
}
void foreach_pow_scalar_inplace_cuda(std::vector<Tensor> self, Scalar exponent) {
    foreach_map_inplace(self, [&](Tensor& value) { value.copy_(value.pow(exponent)); });
}
void foreach_pow_list_inplace_cuda(std::vector<Tensor> self, const std::vector<Tensor>& exponent) {
    foreach_map_pair_inplace(self, exponent, [&](Tensor& value, const Tensor& rhs) { value.copy_(value.pow(rhs)); });
}
std::vector<Tensor> foreach_pow_scalar_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Scalar>& exponent) {
    return foreach_map_scalars(self, exponent,
        [&](const Tensor& value, Scalar rhs) { return value.pow(rhs); });
}
void foreach_pow_scalar_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Scalar>& exponent) {
    foreach_map_scalars_inplace(std::move(self), exponent,
        [&](Tensor& value, Scalar rhs) { value.copy_(value.pow(rhs)); });
}

std::vector<Tensor> foreach_clamp_min_scalar_cuda(const std::vector<Tensor>& self, Scalar scalar) {
    return foreach_map(self, [&](const Tensor& value) { return value.clamp(scalar, std::nullopt); });
}
std::vector<Tensor> foreach_clamp_max_scalar_cuda(const std::vector<Tensor>& self, Scalar scalar) {
    return foreach_map(self, [&](const Tensor& value) { return value.clamp(std::nullopt, scalar); });
}
void foreach_clamp_min_scalar_inplace_cuda(std::vector<Tensor> self, Scalar scalar) {
    foreach_map_inplace(self, [&](Tensor& value) { value.copy_(value.clamp(scalar, std::nullopt)); });
}
void foreach_clamp_max_scalar_inplace_cuda(std::vector<Tensor> self, Scalar scalar) {
    foreach_map_inplace(self, [&](Tensor& value) { value.copy_(value.clamp(std::nullopt, scalar)); });
}
std::vector<Tensor> foreach_clamp_min_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& other) {
    return foreach_map_pair(self, other,
        [&](const Tensor& value, const Tensor& rhs) { return Tensor::maximum(value, rhs); });
}
void foreach_clamp_min_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& other) {
    foreach_map_pair_inplace(std::move(self), other,
        [&](Tensor& value, const Tensor& rhs) { value.copy_(Tensor::maximum(value, rhs)); });
}
std::vector<Tensor> foreach_clamp_max_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& other) {
    return foreach_map_pair(self, other,
        [&](const Tensor& value, const Tensor& rhs) { return Tensor::minimum(value, rhs); });
}
void foreach_clamp_max_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& other) {
    foreach_map_pair_inplace(std::move(self), other,
        [&](Tensor& value, const Tensor& rhs) { value.copy_(Tensor::minimum(value, rhs)); });
}
std::vector<Tensor> foreach_clamp_min_scalar_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {
    return foreach_map_scalars(self, scalars,
        [&](const Tensor& value, Scalar rhs) { return value.clamp(rhs, std::nullopt); });
}
void foreach_clamp_min_scalar_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Scalar>& scalars) {
    foreach_map_scalars_inplace(std::move(self), scalars,
        [&](Tensor& value, Scalar rhs) { value.copy_(value.clamp(rhs, std::nullopt)); });
}
std::vector<Tensor> foreach_clamp_max_scalar_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {
    return foreach_map_scalars(self, scalars,
        [&](const Tensor& value, Scalar rhs) { return value.clamp(std::nullopt, rhs); });
}
void foreach_clamp_max_scalar_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Scalar>& scalars) {
    foreach_map_scalars_inplace(std::move(self), scalars,
        [&](Tensor& value, Scalar rhs) { value.copy_(value.clamp(std::nullopt, rhs)); });
}
std::vector<Tensor> foreach_maximum_scalar_cuda(const std::vector<Tensor>& self, Scalar scalar) { return foreach_clamp_min_scalar_cuda(self, scalar); }
std::vector<Tensor> foreach_minimum_scalar_cuda(const std::vector<Tensor>& self, Scalar scalar) { return foreach_clamp_max_scalar_cuda(self, scalar); }
void foreach_maximum_scalar_inplace_cuda(std::vector<Tensor> self, Scalar scalar) { foreach_clamp_min_scalar_inplace_cuda(self, scalar); }
void foreach_minimum_scalar_inplace_cuda(std::vector<Tensor> self, Scalar scalar) { foreach_clamp_max_scalar_inplace_cuda(self, scalar); }
std::vector<Tensor> foreach_maximum_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& other) {
    return foreach_map_pair(self, other,
        [&](const Tensor& value, const Tensor& rhs) { return Tensor::maximum(value, rhs); });
}
void foreach_maximum_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& other) {
    foreach_map_pair_inplace(std::move(self), other,
        [&](Tensor& value, const Tensor& rhs) { value.copy_(Tensor::maximum(value, rhs)); });
}
std::vector<Tensor> foreach_maximum_scalar_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {
    return foreach_clamp_min_scalar_list_cuda(self, scalars);
}
void foreach_maximum_scalar_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Scalar>& scalars) {
    foreach_clamp_min_scalar_list_inplace_cuda(std::move(self), scalars);
}
std::vector<Tensor> foreach_minimum_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& other) {
    return foreach_map_pair(self, other,
        [&](const Tensor& value, const Tensor& rhs) { return Tensor::minimum(value, rhs); });
}
void foreach_minimum_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& other) {
    foreach_map_pair_inplace(std::move(self), other,
        [&](Tensor& value, const Tensor& rhs) { value.copy_(Tensor::minimum(value, rhs)); });
}
std::vector<Tensor> foreach_minimum_scalar_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {
    return foreach_clamp_max_scalar_list_cuda(self, scalars);
}
void foreach_minimum_scalar_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Scalar>& scalars) {
    foreach_clamp_max_scalar_list_inplace_cuda(std::move(self), scalars);
}
void foreach_copy_inplace_cuda(std::vector<Tensor> self, const std::vector<Tensor>& src, bool non_blocking) {
    foreach_map_pair_inplace(self, src, [&](Tensor& value, const Tensor& rhs) { value.copy_(rhs, non_blocking); });
}
void foreach_zero_inplace_cuda(std::vector<Tensor> self) {
    for (Tensor& value : self) value.zero_();
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, OptimizerKernels) {
    m.impl("_foreach_sgd", foreach_sgd_cuda);
    m.impl("_foreach_adam", foreach_adam_cuda);
    m.impl("_fused_adam_", fused_adam_cuda);
    m.impl("_fused_adam_.tensor_lr", fused_adam_tensor_lr_cuda);
    m.impl("_fused_adamw_", fused_adamw_cuda);
    m.impl("_fused_adamw_.tensor_lr", fused_adamw_tensor_lr_cuda);
    m.impl("_fused_sgd_", fused_sgd_cuda);
    m.impl("_fused_sgd_.tensor_lr", fused_sgd_tensor_lr_cuda);
    m.impl("_fused_adagrad_", fused_adagrad_cuda);
    m.impl("_fused_adagrad_.tensor_lr", fused_adagrad_tensor_lr_cuda);
    m.impl("_foreach_add.Scalar", foreach_add_scalar_cuda);
    m.impl("_foreach_add.List", foreach_add_list_cuda);
    m.impl("_foreach_add.ScalarList", foreach_add_scalar_list_cuda);
    m.impl("_foreach_add.Tensor", foreach_add_tensor_cuda);
    m.impl("_foreach_add_.Scalar", foreach_add_scalar_inplace_cuda);
    m.impl("_foreach_add_.List", foreach_add_list_inplace_cuda);
    m.impl("_foreach_add_.ScalarList", foreach_add_scalar_list_inplace_cuda);
    m.impl("_foreach_add_.Tensor", foreach_add_tensor_inplace_cuda);

#define REGISTER_FOREACH_BINARY(NAME) \
    m.impl("_foreach_" #NAME ".Scalar", foreach_##NAME##_scalar_cuda); \
    m.impl("_foreach_" #NAME ".List", foreach_##NAME##_list_cuda); \
    m.impl("_foreach_" #NAME ".ScalarList", foreach_##NAME##_scalar_list_cuda); \
    m.impl("_foreach_" #NAME ".Tensor", foreach_##NAME##_tensor_cuda); \
    m.impl("_foreach_" #NAME "_.Scalar", foreach_##NAME##_scalar_inplace_cuda); \
    m.impl("_foreach_" #NAME "_.List", foreach_##NAME##_list_inplace_cuda); \
    m.impl("_foreach_" #NAME "_.ScalarList", foreach_##NAME##_scalar_list_inplace_cuda); \
    m.impl("_foreach_" #NAME "_.Tensor", foreach_##NAME##_tensor_inplace_cuda);
    REGISTER_FOREACH_BINARY(sub)
    REGISTER_FOREACH_BINARY(mul)
    REGISTER_FOREACH_BINARY(div)
#undef REGISTER_FOREACH_BINARY

#define REGISTER_FOREACH_UNARY(NAME) \
    m.impl("_foreach_" #NAME, foreach_##NAME##_cuda); \
    m.impl("_foreach_" #NAME "_", foreach_##NAME##_inplace_cuda);
    REGISTER_FOREACH_UNARY(sqrt)
    REGISTER_FOREACH_UNARY(rsqrt)
    REGISTER_FOREACH_UNARY(neg)
    REGISTER_FOREACH_UNARY(abs)
    REGISTER_FOREACH_UNARY(reciprocal)
    REGISTER_FOREACH_UNARY(sign)
#undef REGISTER_FOREACH_UNARY

    m.impl("_foreach_addcmul.Scalar", foreach_addcmul_scalar_cuda);
    m.impl("_foreach_addcmul_.Scalar", foreach_addcmul_scalar_inplace_cuda);
    m.impl("_foreach_addcmul.ScalarList", foreach_addcmul_scalar_list_cuda);
    m.impl("_foreach_addcmul_.ScalarList", foreach_addcmul_scalar_list_inplace_cuda);
    m.impl("_foreach_addcmul.Tensor", foreach_addcmul_tensor_cuda);
    m.impl("_foreach_addcmul_.Tensor", foreach_addcmul_tensor_inplace_cuda);
    m.impl("_foreach_addcdiv.Scalar", foreach_addcdiv_scalar_cuda);
    m.impl("_foreach_addcdiv_.Scalar", foreach_addcdiv_scalar_inplace_cuda);
    m.impl("_foreach_addcdiv.ScalarList", foreach_addcdiv_scalar_list_cuda);
    m.impl("_foreach_addcdiv_.ScalarList", foreach_addcdiv_scalar_list_inplace_cuda);
    m.impl("_foreach_addcdiv.Tensor", foreach_addcdiv_tensor_cuda);
    m.impl("_foreach_addcdiv_.Tensor", foreach_addcdiv_tensor_inplace_cuda);
    m.impl("_foreach_lerp.Scalar", foreach_lerp_scalar_cuda);
    m.impl("_foreach_lerp.List", foreach_lerp_list_cuda);
    m.impl("_foreach_lerp_.Scalar", foreach_lerp_scalar_inplace_cuda);
    m.impl("_foreach_lerp_.List", foreach_lerp_list_inplace_cuda);
    m.impl("_foreach_lerp.ScalarList", foreach_lerp_scalar_list_cuda);
    m.impl("_foreach_lerp_.ScalarList", foreach_lerp_scalar_list_inplace_cuda);
    m.impl("_foreach_pow.Scalar", foreach_pow_scalar_cuda);
    m.impl("_foreach_pow.ScalarAndTensor", foreach_pow_scalar_tensor_cuda);
    m.impl("_foreach_pow.List", foreach_pow_list_cuda);
    m.impl("_foreach_pow_.Scalar", foreach_pow_scalar_inplace_cuda);
    m.impl("_foreach_pow_.List", foreach_pow_list_inplace_cuda);
    m.impl("_foreach_pow.ScalarList", foreach_pow_scalar_list_cuda);
    m.impl("_foreach_pow_.ScalarList", foreach_pow_scalar_list_inplace_cuda);
    m.impl("_foreach_clamp_min.Scalar", foreach_clamp_min_scalar_cuda);
    m.impl("_foreach_clamp_max.Scalar", foreach_clamp_max_scalar_cuda);
    m.impl("_foreach_clamp_min_.Scalar", foreach_clamp_min_scalar_inplace_cuda);
    m.impl("_foreach_clamp_max_.Scalar", foreach_clamp_max_scalar_inplace_cuda);
    m.impl("_foreach_clamp_min.List", foreach_clamp_min_list_cuda);
    m.impl("_foreach_clamp_min_.List", foreach_clamp_min_list_inplace_cuda);
    m.impl("_foreach_clamp_min.ScalarList", foreach_clamp_min_scalar_list_cuda);
    m.impl("_foreach_clamp_min_.ScalarList", foreach_clamp_min_scalar_list_inplace_cuda);
    m.impl("_foreach_clamp_max.List", foreach_clamp_max_list_cuda);
    m.impl("_foreach_clamp_max_.List", foreach_clamp_max_list_inplace_cuda);
    m.impl("_foreach_clamp_max.ScalarList", foreach_clamp_max_scalar_list_cuda);
    m.impl("_foreach_clamp_max_.ScalarList", foreach_clamp_max_scalar_list_inplace_cuda);
    m.impl("_foreach_maximum.Scalar", foreach_maximum_scalar_cuda);
    m.impl("_foreach_minimum.Scalar", foreach_minimum_scalar_cuda);
    m.impl("_foreach_maximum_.Scalar", foreach_maximum_scalar_inplace_cuda);
    m.impl("_foreach_minimum_.Scalar", foreach_minimum_scalar_inplace_cuda);
    m.impl("_foreach_maximum.List", foreach_maximum_list_cuda);
    m.impl("_foreach_maximum_.List", foreach_maximum_list_inplace_cuda);
    m.impl("_foreach_maximum.ScalarList", foreach_maximum_scalar_list_cuda);
    m.impl("_foreach_maximum_.ScalarList", foreach_maximum_scalar_list_inplace_cuda);
    m.impl("_foreach_minimum.List", foreach_minimum_list_cuda);
    m.impl("_foreach_minimum_.List", foreach_minimum_list_inplace_cuda);
    m.impl("_foreach_minimum.ScalarList", foreach_minimum_scalar_list_cuda);
    m.impl("_foreach_minimum_.ScalarList", foreach_minimum_scalar_list_inplace_cuda);
    m.impl("_foreach_copy_", foreach_copy_inplace_cuda);
    m.impl("_foreach_zero_", foreach_zero_inplace_cuda);
}

} // namespace cuda
} // namespace tensorplay
