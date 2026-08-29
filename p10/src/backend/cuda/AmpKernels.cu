#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"

#include <cuda_runtime.h>

#include <cmath>
#include <string>
#include <vector>

namespace tensorplay {
namespace cuda {
namespace {

template <typename T>
class DeviceArray {
public:
    DeviceArray(cudaStream_t stream, const std::vector<T>& values)
        : stream_(stream) {
        if (values.empty()) return;
        checkCuda(cudaMallocAsync(reinterpret_cast<void**>(&data_),
                                  values.size() * sizeof(T), stream_),
                  "cudaMallocAsync amp metadata");
        checkCuda(cudaMemcpyAsync(data_, values.data(),
                                  values.size() * sizeof(T),
                                  cudaMemcpyHostToDevice, stream_),
                  "cudaMemcpyAsync amp metadata");
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

// One block per grad tensor: scan the tensor for non-finite values; if any is
// found, raise the shared found_inf flag, otherwise rescale the tensor in
template <typename scalar_t>
__global__ void amp_non_finite_check_and_unscale_kernel(
    scalar_t* const* grads, const int64_t* numels, float* found_inf,
    float inv_scale) {
    const int64_t tensor_id = blockIdx.x;
    scalar_t* g = grads[tensor_id];
    const int64_t n = numels[tensor_id];

    bool local_non_finite = false;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = static_cast<float>(g[i]);
        if (isnan(v) || isinf(v)) {
            local_non_finite = true;
            break;
        }
    }
    __shared__ bool block_non_finite;
    if (threadIdx.x == 0) block_non_finite = false;
    __syncthreads();
    if (local_non_finite) block_non_finite = true;
    __syncthreads();

    if (block_non_finite) {
        if (threadIdx.x == 0) atomicExch(found_inf, 1.0f);
        return;
    }
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        g[i] = static_cast<scalar_t>(static_cast<float>(g[i]) * inv_scale);
    }
}

__global__ void amp_update_scale_kernel(
    float* scale, int32_t* growth_tracker, float found_inf,
    float growth_factor, float backoff_factor, int growth_interval) {
    // Single-element tensors: a single thread performs the update.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (found_inf > 0) {
            scale[0] = scale[0] * backoff_factor;
            growth_tracker[0] = 0;
        } else {
            growth_tracker[0] += 1;
            if (growth_tracker[0] == growth_interval) {
                scale[0] = scale[0] * growth_factor;
                growth_tracker[0] = 0;
            }
        }
    }
}

} // anonymous namespace

void _amp_foreach_non_finite_check_and_unscale_cuda(
    std::vector<Tensor> self, Tensor& found_inf, const Tensor& inv_scale) {
    if (self.empty()) return;
    const auto stream = getCurrentCUDAStream().stream();
    const float inv_scale_val = inv_scale.data_ptr<float>()[0];
    const int64_t count = static_cast<int64_t>(self.size());

    auto launch = [&](auto type_tag) {
        using scalar_t = decltype(type_tag);
        std::vector<scalar_t*> grad_ptrs;
        std::vector<int64_t> numels;
        grad_ptrs.reserve(self.size());
        numels.reserve(self.size());
        for (const auto& g : self) {
            grad_ptrs.push_back(g.data_ptr<scalar_t>());
            numels.push_back(g.numel());
        }
        DeviceArray<scalar_t*> d_grads(stream, grad_ptrs);
        DeviceArray<int64_t> d_numels(stream, numels);

        amp_non_finite_check_and_unscale_kernel<scalar_t>
            <<<static_cast<unsigned int>(count), 256, 0, stream>>>(
                d_grads.data(), d_numels.data(),
                found_inf.data_ptr<float>(), inv_scale_val);
        checkCuda(cudaGetLastError(),
                  "_amp_foreach_non_finite_check_and_unscale_ kernel launch");
    };

    switch (self[0].dtype()) {
        case DType::Float32: launch(float{}); break;
        case DType::Float64: launch(double{}); break;
        case DType::Float16: launch(Half{}); break;
        case DType::BFloat16: launch(BFloat16{}); break;
        default:
            TP_THROW(NotImplementedError,
                "_amp_foreach_non_finite_check_and_unscale_: unsupported dtype");
    }
}

Tensor& _amp_update_scale_cuda(
    Tensor& self, Tensor& growth_tracker, const Tensor& found_inf,
    double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval) {
    const auto stream = getCurrentCUDAStream().stream();
    amp_update_scale_kernel<<<1, 1, 0, stream>>>(
        self.data_ptr<float>(), growth_tracker.data_ptr<int32_t>(),
        found_inf.data_ptr<float>()[0], static_cast<float>(scale_growth_factor),
        static_cast<float>(scale_backoff_factor), static_cast<int>(growth_interval));
    checkCuda(cudaGetLastError(), "_amp_update_scale_ kernel launch");
    return self;
}

} // namespace cuda

TENSORPLAY_LIBRARY_IMPL(CUDA, AmpKernels) {
    m.impl("_amp_foreach_non_finite_check_and_unscale_",
           cuda::_amp_foreach_non_finite_check_and_unscale_cuda);
    m.impl("_amp_update_scale_", cuda::_amp_update_scale_cuda);
}

} // namespace tensorplay
