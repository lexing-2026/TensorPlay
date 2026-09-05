#include "Tensor.h"
#include "CUDARuntime.h"
#include "Dispatcher.h"
#include "CUDAGenerator.h"
#include "Generator.h"
#include "Exception.h"
#include "Utils.h"
#include "DistributionDispatch.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tensorplay {
namespace cuda {

// thread drives an independent curandStatePhilox4_32_10_t subsequence seeded
// with (seed, thread index, offset). The (seed, offset) pair comes from the
// host-side generator, which reserves counter values per launch, so results
// are independent of launch geometry and reproducible across runs. During a
// CUDA graph capture the pair instead lives in device memory owned by the
// capturing graph (PhiloxCudaState pointer mode) and is refreshed before each
// replay, so replays draw fresh randoms (see CUDAGenerator.h).

namespace {

// Unpacks PhiloxCudaState into the effective (seed, offset) this launch
// consumes; in pointer mode the device buffer is dereferenced at kernel
// execution time, both during capture and on every later replay.
__device__ inline void philox_unpack(const PhiloxCudaState& state,
                                     uint64_t* seed, uint64_t* offset) {
    if (state.captured) {
        *seed = *state.seed_dev;
        *offset = *state.offset_dev + state.offset_intragraph;
    } else {
        *seed = state.seed;
        *offset = state.offset;
    }
}

constexpr uint32_t kBlockSize = 256;
// curand device API consumes at most 4 counter values per call.
constexpr uint64_t kMaxGeneratorOffsetsPerCall = 4;

uint32_t deviceAttribute(cudaDeviceAttr attr) {
    int value = 0;
    int device_index = 0;
    cudaGetDevice(&device_index);
    cudaError_t error = cudaDeviceGetAttribute(&value, attr, device_index);
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("cudaDeviceGetAttribute failed: ") +
                 cudaGetErrorString(error));
    }
    return static_cast<uint32_t>(value);
}

// Utility function that calculates the proper philox_offset for the
std::tuple<uint64_t, dim3, dim3> calc_execution_policy(int64_t total_elements,
                                                       uint32_t unroll_factor) {
    const uint64_t numel = static_cast<uint64_t>(total_elements);
    const uint32_t block_size = kBlockSize;
    dim3 dim_block(block_size);
    dim3 grid(static_cast<uint32_t>((numel + block_size - 1) / block_size));
    const uint32_t blocks_per_sm =
        deviceAttribute(cudaDevAttrMaxThreadsPerMultiProcessor) / block_size;
    grid.x = std::min(deviceAttribute(cudaDevAttrMultiProcessorCount) * blocks_per_sm,
                      grid.x);
    // Number of randoms generated per thread, as philox counter increments.
    const uint64_t counter_offset =
        ((numel - 1) / (block_size * grid.x * unroll_factor) + 1) *
        kMaxGeneratorOffsetsPerCall;
    return std::make_tuple(counter_offset, grid, dim_block);
}

template <typename scalar_t, typename dist_return_t, int unroll_factor,
          typename dist_t, typename transform_t>
__global__ void distribution_elementwise_grid_stride_kernel(
        int64_t numel, PhiloxCudaState philox_args, scalar_t* out_data,
        dist_t dist_func, transform_t transform_func) {
    uint64_t seed;
    uint64_t offset;
    philox_unpack(philox_args, &seed, &offset);
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);
    const int64_t total_threads = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t rounded_size =
        ((numel - 1) / (total_threads * unroll_factor) + 1) * total_threads * unroll_factor;
    for (int64_t linear_index = idx; linear_index < rounded_size;
         linear_index += total_threads * unroll_factor) {
        auto rand = dist_func(&state);
        #pragma unroll
        for (int ii = 0; ii < unroll_factor; ii++) {
            int64_t li = linear_index + total_threads * ii;
            if (li < numel) {
                out_data[li] = transform_func((&rand.x)[ii]);
            }
        }
    }
}

template <typename scalar_t, typename dist_return_t, int unroll_factor,
          typename dist_t, typename transform_t>
void distribution_nullary_kernel(scalar_t* out_data, int64_t numel,
                                 dist_t dist_func, transform_t transform_func,
                                 std::optional<Generator> generator = std::nullopt) {
    if (numel == 0) return;
    auto policy = calc_execution_policy(numel, unroll_factor);
    const uint64_t counter_offset = std::get<0>(policy);
    const dim3 grid = std::get<1>(policy);
    const dim3 block = std::get<2>(policy);
    PhiloxCudaState philox_args;
    if (generator.has_value()) {
        philox_args.seed = generator->random64();
        philox_args.offset = 0;
    } else {
        philox_args = philox_cuda_state(counter_offset);
    }
    cudaStream_t stream = getCurrentCUDAStream().stream();
    distribution_elementwise_grid_stride_kernel<scalar_t, dist_return_t, unroll_factor>
        <<<grid, block, 0, stream>>>(numel, philox_args, out_data,
                                     dist_func, transform_func);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error));
    }
}


// In-place sampling writes through a contiguous buffer: non-contiguous
// destinations are filled via a contiguous temporary copied back with the
// strided copy kernel.  A stride-0 dimension with more than one element
// aliases the whole dimension, makes the draw order observable, and is
// rejected.
template <typename FillFn>
Tensor& fill_via_contiguous(Tensor& self, FillFn&& fill) {
    const auto sizes = static_cast<std::vector<int64_t>>(self.shape());
    const auto strides = self.strides();
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (strides[i] == 0 && sizes[i] > 1) {
            TP_THROW(RuntimeError,
                     "unsupported operation: more than one element of the written-to tensor "
                     "refers to a single memory location. Please clone() the tensor before "
                     "performing the operation.");
        }
    }
    Tensor tmp = Tensor::empty(sizes, self.dtype(), self.device());
    fill(tmp);
    self.copy_(tmp);
    return self;
}

} // namespace

Tensor rand_kernel_cuda(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t = Tensor::empty(size, dtype, device);
    int64_t n = t.numel();

    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [] __device__ (float rand) { return rand; });
    } else if (dtype == DType::Float64) {
        double* data = t.data_ptr<double>();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
            [] __device__ (double rand) { return rand; });
    } else if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        if (dtype == DType::Float16) {
            Half* data = t.data_ptr<Half>();
            distribution_nullary_kernel<Half, float4, 4>(
                data, n,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
                [] __device__ (float rand) { return static_cast<Half>(rand); });
        } else {
            BFloat16* data = t.data_ptr<BFloat16>();
            distribution_nullary_kernel<BFloat16, float4, 4>(
                data, n,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
                [] __device__ (float rand) { return static_cast<BFloat16>(rand); });
        }
    } else if (dtype == DType::ComplexFloat || dtype == DType::ComplexDouble) {
        // the interleaved component buffer as a real array.
        const int64_t comps = t.numel() * 2;
        if (dtype == DType::ComplexFloat) {
            float* raw = static_cast<float*>(t.data_ptr());
            distribution_nullary_kernel<float, float4, 4>(
                raw, comps,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
                [] __device__ (float v) { return v; });
        } else {
            double* raw = static_cast<double*>(t.data_ptr());
            distribution_nullary_kernel<double, double2, 2>(
                raw, comps,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
                [] __device__ (double v) { return v; });
        }
    } else {
         TP_THROW(NotImplementedError, "rand() only supports floating dtypes on CUDA for now");
    }
    return t;
}

Tensor randn_kernel_cuda(const std::vector<int64_t>& size, DType dtype, Device device,
                         std::optional<Generator> generator = std::nullopt) {
    Tensor t = Tensor::empty(size, dtype, device);
    int64_t n = t.numel();

    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
            [] __device__ (float rand) { return rand; }, std::move(generator));
    } else if (dtype == DType::Float64) {
        double* data = t.data_ptr<double>();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
            [] __device__ (double rand) { return rand; }, std::move(generator));
    } else if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        if (dtype == DType::Float16) {
            Half* data = t.data_ptr<Half>();
            distribution_nullary_kernel<Half, float4, 4>(
                data, n,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
                [] __device__ (float rand) { return static_cast<Half>(rand); },
                std::move(generator));
        } else {
            BFloat16* data = t.data_ptr<BFloat16>();
            distribution_nullary_kernel<BFloat16, float4, 4>(
                data, n,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
                [] __device__ (float rand) { return static_cast<BFloat16>(rand); },
                std::move(generator));
        }
    } else if (dtype == DType::ComplexFloat || dtype == DType::ComplexDouble) {
        // standard-normal factory is N(0, 1/sqrt(2)) per component.
        constexpr float kInvSqrt2f = 0.70710678118654752f;
        constexpr double kInvSqrt2 = 0.70710678118654752440;
        const int64_t comps = t.numel() * 2;
        if (dtype == DType::ComplexFloat) {
            float* raw = static_cast<float*>(t.data_ptr());
            distribution_nullary_kernel<float, float4, 4>(
                raw, comps,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
                [] __device__ (float v) { return v * kInvSqrt2f; },
                std::move(generator));
        } else {
            double* raw = static_cast<double*>(t.data_ptr());
            distribution_nullary_kernel<double, double2, 2>(
                raw, comps,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
                [] __device__ (double v) { return v * kInvSqrt2; },
                std::move(generator));
        }
    } else {
         TP_THROW(NotImplementedError, "randn() only supports floating dtypes on CUDA for now");
    }
    return t;
}

Tensor rand_like_kernel_cuda(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();

    Device target_device = device.has_value() ? *device : self.device();

    return rand_kernel_cuda(static_cast<std::vector<int64_t>>(self.shape()), dtype, target_device);
}

Tensor randn_like_kernel_cuda(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device target_device = device.has_value() ? *device : self.device();
    return randn_kernel_cuda(static_cast<std::vector<int64_t>>(self.shape()), dtype, target_device);
}

Tensor& uniform_kernel_cuda(Tensor& self, double from, double to,
                            std::optional<Generator> generator) {
    if (isComplexType(self.dtype())) {
        Tensor real_view = tpx::ops::view_as_real(self);
        uniform_kernel_cuda(real_view, from, to, std::move(generator));
        return self;
    }
    if (self.dtype() != DType::Float16 && self.dtype() != DType::BFloat16 &&
        self.dtype() != DType::Float32 && self.dtype() != DType::Float64) {
        TP_THROW(NotImplementedError, "uniform_() only supports floating dtypes on CUDA");
    }
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return uniform_kernel_cuda(t, from, to, std::move(generator));
        });
    }
    const double dtype_max = self.dtype() == DType::Float16
        ? distribution::fp_dtype_max<Half>()
        : self.dtype() == DType::BFloat16
            ? distribution::fp_dtype_max<BFloat16>()
            : self.dtype() == DType::Float32
                ? distribution::fp_dtype_max<float>()
                : distribution::fp_dtype_max<double>();
    const char* dtype_name = distribution::bounds_dtype_name(self.dtype());
    TP_THROW_IF(!(from >= -dtype_max && from <= dtype_max), RuntimeError,
                "from is out of bounds for ", dtype_name);
    TP_THROW_IF(!(to >= -dtype_max && to <= dtype_max), RuntimeError,
                "to is out of bounds for ", dtype_name);
    TP_THROW_IF(from > to, RuntimeError,
                "uniform_ expects to return a [from, to) range, but found from=",
                from, " > to=", to);
    TP_THROW_IF((to - from) > dtype_max,
                RuntimeError,
                "uniform_ expects to-from <= std::numeric_limits<",
                dtype_name,
                ">::max(), but found to=", to, " and from=", from,
                " which result in to-from to exceed the limit");
    from = std::clamp(from, -dtype_max, dtype_max);
    to = std::clamp(to, -dtype_max, dtype_max);

    if (self.numel() == 0) return self;
    int64_t n = self.numel();
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        const float lo = static_cast<float>(from);
        const float hi = static_cast<float>(to);
        const float range = hi - lo;
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [lo, hi, range] __device__ (float rand) {
                const float value = static_cast<float>(lo + range * rand);
                return value == hi ? lo : value;
            },
            std::move(generator));
    } else if (self.dtype() == DType::Float64) {
        double* data = self.data_ptr<double>();
        const double range = to - from;
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
            [from, to, range] __device__ (double rand) {
                const double value = from + range * rand;
                return value == to ? from : value;
            },
            std::move(generator));
    } else if (self.dtype() == DType::Float16) {
        Half* data = self.data_ptr<Half>();
        const Half lo = static_cast<Half>(from);
        const Half hi = static_cast<Half>(to);
        const float lo_value = static_cast<float>(lo);
        const float hi_value = static_cast<float>(hi);
        const float range = hi_value - lo_value;
        distribution_nullary_kernel<Half, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [lo, hi, lo_value, range] __device__ (float rand) {
                const Half value = static_cast<Half>(lo_value + range * rand);
                return value == hi ? lo : value;
            },
            std::move(generator));
    } else {
        BFloat16* data = self.data_ptr<BFloat16>();
        const BFloat16 lo = static_cast<BFloat16>(from);
        const BFloat16 hi = static_cast<BFloat16>(to);
        const float lo_value = static_cast<float>(lo);
        const float hi_value = static_cast<float>(hi);
        const float range = hi_value - lo_value;
        distribution_nullary_kernel<BFloat16, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [lo, hi, lo_value, range] __device__ (float rand) {
                const BFloat16 value = static_cast<BFloat16>(lo_value + range * rand);
                return value == hi ? lo : value;
            },
            std::move(generator));
    }
    return self;
}

Tensor& normal_kernel_cuda(Tensor& self, double mean, double std,
                           std::optional<Generator> generator) {
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return normal_kernel_cuda(t, mean, std, std::move(generator));
        });
    }
    int64_t n = self.numel();
    if (std < 0.0) {
        TP_THROW(RuntimeError, "normal expects std >= 0.0, but found std ", std);
    }
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        const float mu = static_cast<float>(mean);
        const float sigma = static_cast<float>(std);
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
            [mu, sigma] __device__ (float rand) { return mu + sigma * rand; },
            std::move(generator));
    } else if (self.dtype() == DType::Float64) {
        double* data = self.data_ptr<double>();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
            [mean, std] __device__ (double rand) { return mean + std * rand; },
            std::move(generator));
    } else if (self.dtype() == DType::Float16) {
        Half* data = self.data_ptr<Half>();
        const float mu = static_cast<float>(mean);
        const float sigma = static_cast<float>(std);
        distribution_nullary_kernel<Half, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
            [mu, sigma] __device__ (float rand) {
                return static_cast<Half>(mu + sigma * rand);
            },
            std::move(generator));
    } else if (self.dtype() == DType::BFloat16) {
        BFloat16* data = self.data_ptr<BFloat16>();
        const float mu = static_cast<float>(mean);
        const float sigma = static_cast<float>(std);
        distribution_nullary_kernel<BFloat16, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
            [mu, sigma] __device__ (float rand) {
                return static_cast<BFloat16>(mu + sigma * rand);
            },
            std::move(generator));
    } else {
        TP_THROW(NotImplementedError,
                 "normal_() only supports floating dtypes on CUDA");
    }
    return self;
}

// transformation::exponential CUDA branch
// (0, 1]; log(1) is 0 and the exponential distribution excludes 0, so values
// within epsilon/2 of 1 clamp their log to -epsilon/2.
Tensor& exponential_kernel_cuda(Tensor& self, double lambd) {
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return exponential_kernel_cuda(t, lambd);
        });
    }
    int64_t n = self.numel();
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        const float lambda = static_cast<float>(lambd);
        constexpr float kEps = std::numeric_limits<float>::epsilon();
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [lambda] __device__ (float val) {
                float log = val >= 1.f - kEps / 2 ? -kEps / 2 : __logf(val);
                return -1.f / lambda * log;
            });
    } else if (self.dtype() == DType::Float64) {
        double* data = self.data_ptr<double>();
        const double lambda = lambd;
        constexpr double kEps = std::numeric_limits<double>::epsilon();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
            [lambda] __device__ (double val) {
                double log = val >= 1. - kEps / 2 ? -kEps / 2 : ::log(val);
                return -1. / lambda * log;
            });
    } else if (self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16) {
        if (self.dtype() == DType::Float16) {
            Half* data = self.data_ptr<Half>();
            const float lambda = static_cast<float>(lambd);
            constexpr float kEps = std::numeric_limits<float>::epsilon();
            distribution_nullary_kernel<Half, float4, 4>(
                data, n,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
                [lambda] __device__ (float val) {
                    float log = val >= 1.f - kEps / 2 ? -kEps / 2 : __logf(val);
                    return static_cast<Half>(-1.f / lambda * log);
                });
        } else {
            BFloat16* data = self.data_ptr<BFloat16>();
            const float lambda = static_cast<float>(lambd);
            constexpr float kEps = std::numeric_limits<float>::epsilon();
            distribution_nullary_kernel<BFloat16, float4, 4>(
                data, n,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
                [lambda] __device__ (float val) {
                    float log = val >= 1.f - kEps / 2 ? -kEps / 2 : __logf(val);
                    return static_cast<BFloat16>(-1.f / lambda * log);
                });
        }
    } else {
        TP_THROW(NotImplementedError, "exponential_() only supports floating dtypes on CUDA for now");
    }
    return self;
}

// --- TransformationHelper.h): random_ / randint / geometric_ / -------------
// --- log_normal_ / cauchy_ / poisson / randperm ----------------------------

template <typename scalar_t>
void geometric_fill_float_cuda(scalar_t* data, int64_t n, float p) {
    distribution_nullary_kernel<scalar_t, float4, 4>(
        data, n,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
        [p] __device__ (float val) {
            return static_cast<scalar_t>(::ceilf(::logf(val) / ::log1pf(-p)));
        });
}

void geometric_fill_double_cuda(double* data, int64_t n, double p) {
    distribution_nullary_kernel<double, double2, 2>(
        data, n,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
        [p] __device__ (double val) {
            return ::ceil(::log(val) / ::log1p(-p));
        });
}

Tensor& geometric_kernel_cuda(Tensor& self, double p) {
    TP_THROW_IF(!(p > 0.0 && p < 1.0), RuntimeError,
                "geometric_ expects p to be in (0, 1), but got p=", p);
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return geometric_kernel_cuda(t, p);
        });
    }

    const int64_t n = self.numel();
    distribution::dispatch_dtype(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        if constexpr (std::is_same_v<scalar_t, double>) {
            geometric_fill_double_cuda(self.data_ptr<double>(), n, p);
        } else {
            geometric_fill_float_cuda<scalar_t>(
                self.data_ptr<scalar_t>(), n, static_cast<float>(p));
        }
    });
    return self;
}

Tensor& log_normal_kernel_cuda(Tensor& self, double mean, double std) {
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) { return log_normal_kernel_cuda(t, mean, std); });
    }
    int64_t n = self.numel();
    if (std <= 0.0) {
        TP_THROW(RuntimeError, "log_normal_ expects std > 0.0, but found std=", std);
    }
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        const float mu = static_cast<float>(mean);
        const float sigma = static_cast<float>(std);
        // transformation::log_normal: exp(normal_draw).
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
            [mu, sigma] __device__ (float rand) { return ::expf(mu + sigma * rand); });
    } else if (self.dtype() == DType::Float64) {
        double* data = self.data_ptr<double>();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
            [mean, std] __device__ (double rand) { return ::exp(mean + std * rand); });
    } else if (self.dtype() == DType::Float16) {
        Half* data = self.data_ptr<Half>();
        const float mu = static_cast<float>(mean);
        const float sigma = static_cast<float>(std);
        distribution_nullary_kernel<Half, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
            [mu, sigma] __device__ (float rand) {
                return static_cast<Half>(::expf(mu + sigma * rand));
            });
    } else if (self.dtype() == DType::BFloat16) {
        BFloat16* data = self.data_ptr<BFloat16>();
        const float mu = static_cast<float>(mean);
        const float sigma = static_cast<float>(std);
        distribution_nullary_kernel<BFloat16, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
            [mu, sigma] __device__ (float rand) {
                return static_cast<BFloat16>(::expf(mu + sigma * rand));
            });
    } else {
        TP_THROW(NotImplementedError, "log_normal_() only supports floating dtypes on CUDA");
    }
    return self;
}

Tensor& cauchy_kernel_cuda(Tensor& self, double median, double sigma) {
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) { return cauchy_kernel_cuda(t, median, sigma); });
    }
    int64_t n = self.numel();
    if (sigma <= 0.0) {
        TP_THROW(RuntimeError, "cauchy_ expects sigma > 0.0, but found sigma=", sigma);
    }
    constexpr double kPi = 3.14159265358979323846;
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        const float med = static_cast<float>(median);
        const float sig = static_cast<float>(sigma);
        constexpr float kEps = std::numeric_limits<float>::epsilon();
        // transformation::cauchy (float overload): clip val into
        // [eps, 1-eps] because tanf overflows at the open boundaries.
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [med, sig] __device__ (float val) {
                val = val > 1.f - kEps ? 1.f - kEps : val;
                val = val < kEps ? kEps : val;
                return med + sig * ::tanf(static_cast<float>(M_PI) * (val - 0.5f));
            });
    } else if (self.dtype() == DType::Float64) {
        double* data = self.data_ptr<double>();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
            [median, sigma] __device__ (double val) {
                return median + sigma * ::tan(kPi * (val - 0.5));
            });
    } else if (self.dtype() == DType::Float16) {
        Half* data = self.data_ptr<Half>();
        const float med = static_cast<float>(median);
        const float sig = static_cast<float>(sigma);
        constexpr float kEps = std::numeric_limits<float>::epsilon();
        distribution_nullary_kernel<Half, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [med, sig] __device__ (float val) {
                val = val > 1.f - kEps ? 1.f - kEps : val;
                val = val < kEps ? kEps : val;
                return static_cast<Half>(med + sig * ::tanf(
                    3.14159265358979323846f * (val - 0.5f)));
            });
    } else if (self.dtype() == DType::BFloat16) {
        BFloat16* data = self.data_ptr<BFloat16>();
        const float med = static_cast<float>(median);
        const float sig = static_cast<float>(sigma);
        constexpr float kEps = std::numeric_limits<float>::epsilon();
        distribution_nullary_kernel<BFloat16, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [med, sig] __device__ (float val) {
                val = val > 1.f - kEps ? 1.f - kEps : val;
                val = val < kEps ? kEps : val;
                return static_cast<BFloat16>(med + sig * ::tanf(
                    3.14159265358979323846f * (val - 0.5f)));
            });
    } else {
        TP_THROW(NotImplementedError, "cauchy_() only supports floating dtypes on CUDA");
    }
    return self;
}

template <typename scalar_t>
void launch_random_range_cuda(scalar_t* data, int64_t n,
                              uint64_t range, int64_t base) {
    if (range >= (1ULL << 28)) {
        distribution_nullary_kernel<scalar_t, ulonglong2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) {
                ulonglong2 random;
                uint4 words = curand4(state);
                random.x = (static_cast<uint64_t>(words.x) << 32) | words.y;
                random.y = (static_cast<uint64_t>(words.z) << 32) | words.w;
                return random;
            },
            [range, base] __device__ (uint64_t value) {
                return static_cast<scalar_t>(static_cast<int64_t>(
                    (value % range) + static_cast<uint64_t>(base)));
            });
    } else {
        distribution_nullary_kernel<scalar_t, uint4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) {
                return curand4(state);
            },
            [range, base] __device__ (unsigned int value) {
                return static_cast<scalar_t>(static_cast<int64_t>(
                    (static_cast<uint64_t>(value) % range) +
                    static_cast<uint64_t>(base)));
            });
    }
}

template <typename scalar_t>
void launch_random_full_range_cuda(scalar_t* data, int64_t n) {
    if constexpr (std::is_same_v<scalar_t, uint64_t>) {
        distribution_nullary_kernel<scalar_t, ulonglong2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) {
                ulonglong2 random;
                uint4 words = curand4(state);
                random.x = (static_cast<uint64_t>(words.x) << 32) | words.y;
                random.y = (static_cast<uint64_t>(words.z) << 32) | words.w;
                return random;
            },
            [] __device__ (uint64_t value) {
                return static_cast<scalar_t>(value);
            });
    } else {
        uint64_t range;
        if constexpr (std::is_same_v<scalar_t, int64_t>) {
            range = uint64_t{1} << 63;
        } else if constexpr (std::is_same_v<scalar_t, double>) {
            range = uint64_t{1} << 53;
        } else if constexpr (std::is_same_v<scalar_t, float>) {
            range = uint64_t{1} << 24;
        } else if constexpr (std::is_same_v<scalar_t, Half>) {
            range = uint64_t{1} << 11;
        } else if constexpr (std::is_same_v<scalar_t, BFloat16>) {
            range = uint64_t{1} << 8;
        } else if constexpr (std::is_same_v<scalar_t, bool>) {
            range = 2;
        } else {
            range = static_cast<uint64_t>(std::numeric_limits<scalar_t>::max()) + 1;
        }
        launch_random_range_cuda(data, n, range, 0);
    }
}

static void randint_fill_dispatch(Tensor& t, int64_t low, int64_t high) {
    int64_t n = t.numel();
    distribution::check_random_from_to_bounds(low, high, t.dtype());
    if (n == 0) return;
    const uint64_t range = static_cast<uint64_t>(high) -
        static_cast<uint64_t>(low);
    const int64_t base = low;
    distribution::dispatch_dtype(t.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        launch_random_range_cuda(t.data_ptr<scalar_t>(), n, range, base);
    });
}

Tensor randint_kernel_cuda(int64_t low, int64_t high, const std::vector<int64_t>& size,
                           DType dtype, Device device) {
    if (low >= high) {
        TP_THROW(RuntimeError, "randint(): low must be less than high");
    }
    Tensor t(size, dtype, device);
    randint_fill_dispatch(t, low, high);
    return t;
}

Tensor randint_like_kernel_cuda(const Tensor& self, int64_t low, int64_t high,
                                DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device target_device = device.has_value() ? *device : self.device();
    Tensor t = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), dtype, target_device);
    randint_fill_dispatch(t, low, high);
    return t;
}

Tensor& random_kernel_cuda(Tensor& self, int64_t low, int64_t high) {
    const bool full_range = (low == 0 && high == 0);
    if (!full_range && low >= high) {
        TP_THROW(RuntimeError, "random_ expects 'from' to be less than 'to', but got from=", low, " >= to=", high);
    }
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) { return random_kernel_cuda(t, low, high); });
    }
    int64_t n = self.numel();
    if (full_range) {
        if (n == 0) return self;
        distribution::dispatch_dtype(self.dtype(), [&](auto tag) {
            using scalar_t = decltype(tag);
            launch_random_full_range_cuda(self.data_ptr<scalar_t>(), n);
        });
    } else {
        randint_fill_dispatch(self, low, high);
    }
    return self;
}

// carries its own lambda and consumes a variable number of philox counters.
template <typename scalar_t>
__global__ void poisson_fill_impl(int64_t numel, PhiloxCudaState philox_args,
                                  const scalar_t* in_data, scalar_t* out_data) {
    uint64_t seed;
    uint64_t offset;
    philox_unpack(philox_args, &seed, &offset);
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);
    out_data[idx] = static_cast<scalar_t>(
        curand_poisson(&state, static_cast<double>(in_data[idx])));
}

Tensor poisson_kernel_cuda(const Tensor& self) {
    Tensor t(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    if (n == 0) return t;
    const Tensor input = self.is_contiguous() ? self : self.contiguous();
    if (!input.ge(Scalar(0)).all().item<bool>()) {
        TP_THROW(RuntimeError, "invalid Poisson rate, expected rate to be non-negative");
    }
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    // Each thread runs curand_poisson which consumes lambda-dependent counters;
    // reserve generously so concurrent calls never share counter slices.
    const uint64_t counter_offset = 16u *
        ((static_cast<uint64_t>(n) + threads * blocks - 1) / (threads * blocks) + 1) *
        kMaxGeneratorOffsetsPerCall;
    auto philox_args = philox_cuda_state(counter_offset);

    if (self.dtype() == DType::Float32) {
        poisson_fill_impl<float><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            n, philox_args, input.data_ptr<float>(), t.data_ptr<float>());
    } else if (self.dtype() == DType::Float64) {
        poisson_fill_impl<double><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            n, philox_args, input.data_ptr<double>(), t.data_ptr<double>());
    } else if (self.dtype() == DType::Float16) {
        poisson_fill_impl<Half><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            n, philox_args, input.data_ptr<Half>(), t.data_ptr<Half>());
    } else if (self.dtype() == DType::BFloat16) {
        poisson_fill_impl<BFloat16><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            n, philox_args, input.data_ptr<BFloat16>(), t.data_ptr<BFloat16>());
    } else {
        TP_THROW(NotImplementedError, "poisson() only supports floating dtypes on CUDA");
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA poisson Error: ") +
                    cudaGetErrorString(error));
    }
    return t;
}

// --- gamma / binomial / dirichlet sampling ----------------------------------
//
// Per-element samplers: each thread owns one output element and draws from
// its own philox subsequence with curand's scalar generators.  Draws per
// element are rejection-dependent, so counter reservation uses a generous
// multiple of the element count (same policy as curand_poisson above).

namespace {

// Digamma via the recurrence-to-10 plus asymptotic series.
__device__ inline float sampling_digamma(float x) {
    constexpr float PSI_10 = 2.25175258906672110764f;
    if (x == 0.0f) return INFINITY;
    float extra = 0.0f;
    if (x < 0.0f) {
        if (x == ::floorf(x)) return INFINITY;
        constexpr float PI_F = 3.14159265358979323846f;
        extra = -PI_F / ::tanf(PI_F * x);
        x = 1.0f - x;
    }
    float result = 0.0f;
    while (x < 10.0f) {
        result -= 1.0f / x;
        x += 1.0f;
    }
    if (x == 10.0f) return result + PSI_10 + extra;
    // Asymptotic expansion coefficients for the trigamma tail.
    const float A[7] = {8.33333333333333333333e-2f, -2.10927960927960927961e-2f,
                       7.57575757575757575758e-3f, -4.16666666666666666667e-3f,
                       3.96825396825396825397e-3f, -8.33333333333333333333e-3f,
                       8.33333333333333333333e-2f};
    float y = 0.0f;
    if (x < 1.0e17f) {
        const float z = 1.0f / (x * x);
        float term = A[6];
        for (int k = 5; k >= 0; --k) term = term * z + A[k];
        y = z * term;
    }
    return result + ::logf(x) - (0.5f / x) - y + extra;
}

// Marsaglia-Tsang standard gamma sampler for shape alpha > 0.  The
// accumulator type drives the curand draw widths: double state uses the
// *_double generators, float state the scalar ones.
template <typename scalar_t, typename acc_t>
__device__ inline acc_t sample_standard_gamma(
        curandStatePhilox4_32_10_t* state, acc_t alpha) {
    acc_t scale = 1.0;
    if (alpha < 1.0) {
        if (alpha == 0.0) return 0.0;
        acc_t u;
        if constexpr (std::is_same<acc_t, double>::value) {
            u = curand_uniform_double(state);
        } else {
            u = curand_uniform(state);
        }
        scale *= ::pow(1.0 - u, 1.0 / alpha);
        alpha += 1.0;
    }
    const acc_t d = alpha - 1.0 / 3.0;
    const acc_t c = 1.0 / ::sqrt(9.0 * d);
    for (;;) {
        acc_t x, y;
        do {
            if constexpr (std::is_same<acc_t, double>::value) {
                x = curand_normal_double(state);
            } else {
                x = curand_normal(state);
            }
            y = 1.0 + c * x;
        } while (y <= 0.0);
        const acc_t v = y * y * y;
        acc_t u;
        if constexpr (std::is_same<acc_t, double>::value) {
            u = 1.0 - curand_uniform_double(state);
        } else {
            u = 1.0 - curand_uniform(state);
        }
        const acc_t xx = x * x;
        if (u < 1.0 - 0.0331 * xx * xx) return scale * d * v;
        if (::log(u) < 0.5 * xx + d * (1.0 - v + ::log(v))) return scale * d * v;
    }
}

// Reparameterized gradient of a standard-gamma sample wrt its shape:
// -(d/dalpha cdf(x; alpha)) / pdf(x; alpha).
template <typename acc_t>
__device__ inline acc_t standard_gamma_grad_one(acc_t alpha, acc_t x) {
    if (x < 0.8) {
        acc_t numer = 1.0;
        acc_t denom = alpha;
        acc_t series1 = numer / denom;
        acc_t series2 = numer / (denom * denom);
        for (int i = 1; i <= 5; ++i) {
            numer *= -x / static_cast<acc_t>(i);
            denom += 1.0;
            series1 += numer / denom;
            series2 += numer / (denom * denom);
        }
        const acc_t pow_x_alpha = ::pow(x, alpha);
        const acc_t gamma_pdf = ::pow(x, alpha - 1.0) * ::exp(-x);
        const acc_t gamma_cdf = pow_x_alpha * series1;
        const acc_t gamma_cdf_alpha =
            (::log(x) - sampling_digamma(static_cast<float>(alpha))) *
                gamma_cdf -
            pow_x_alpha * series2;
        const acc_t result = -gamma_cdf_alpha / gamma_pdf;
        return isnan(result) ? 0.0 : result;
    }
    if (alpha > 8.0) {
        if (0.9 * alpha <= x && x <= 1.1 * alpha) {
            const acc_t numer_1 = 1 + 24 * alpha * (1 + 12 * alpha);
            const acc_t numer_2 = 1440 * (alpha * alpha) + 6 * x * (53 - 120 * x)
                - 65 * x * x / alpha + alpha * (107 + 3600 * x);
            const acc_t denom = 1244160 * (alpha * alpha) * (alpha * alpha);
            return numer_1 * numer_2 / denom;
        }
        const acc_t denom = ::sqrt(8 * alpha);
        const acc_t term2 = denom / (alpha - x);
        const acc_t term3 = ::pow(
            x - alpha - alpha * ::log(x / alpha), -1.5);
        const acc_t term23 = (x < alpha) ? term2 - term3 : term2 + term3;
        const acc_t term1 = ::log(x / alpha) * term23 -
            ::sqrt(2 / alpha) * (alpha + x) / ((alpha - x) * (alpha - x));
        const acc_t stirling = 1 + 1 / (12 * alpha) * (1 + 1 / (24 * alpha));
        const acc_t numer = x * term1;
        return -stirling * numer / denom;
    }
    // Bivariate rational approximation around the (log x/alpha, log alpha)
    // anchor for the remaining (alpha, x) region.
    const acc_t u = ::log(x / alpha);
    const acc_t v = ::log(alpha);
    static const acc_t coef_uv[3][8] = {
        {0.16009398, -0.094634809, 0.025146376, -0.0030648343,
         1, 0.32668115, 0.10406089, 0.0014179084},
        {0.53487893, 0.1298071, 0.065735949, -0.0015649758,
         0.16639465, 0.020070113, -0.0035938915, -0.00058392623},
        {0.040121004, -0.0065914022, -0.0026286047, -0.0013441777,
         0.017050642, -0.0021309326, 0.00085092367, -1.5247877e-07},
    };
    acc_t coef_v[8];
    for (int i = 0; i < 8; ++i) {
        coef_v[i] = coef_uv[0][i] + u * (coef_uv[1][i] + u * coef_uv[2][i]);
    }
    const acc_t p = coef_v[0] + v * (coef_v[1] + v * (coef_v[2] + v * coef_v[3]));
    const acc_t q = coef_v[4] + v * (coef_v[5] + v * (coef_v[6] + v * coef_v[7]));
    return ::exp(p / q);
}

template <typename scalar_t>
__global__ void standard_gamma_fill_impl(
        int64_t numel, PhiloxCudaState philox_args, const scalar_t* alpha,
        scalar_t* out) {
    uint64_t seed;
    uint64_t offset;
    philox_unpack(philox_args, &seed, &offset);
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);
    using acc_t = std::conditional_t<std::is_same<scalar_t, double>::value,
                                     double, float>;
    const acc_t a = static_cast<acc_t>(alpha[idx]);
    const acc_t sample = sample_standard_gamma<scalar_t, acc_t>(&state, a);
    const acc_t min_value = std::is_same<scalar_t, Half>::value
        ? static_cast<acc_t>(6.103515625e-05f)
        : std::numeric_limits<acc_t>::min();
    out[idx] = static_cast<scalar_t>(sample < min_value ? min_value : sample);
}

template <typename scalar_t>
__global__ void standard_gamma_grad_impl(
        int64_t numel, const scalar_t* alpha, const scalar_t* output,
        scalar_t* out) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    using acc_t = std::conditional_t<std::is_same<scalar_t, double>::value,
                                     double, float>;
    out[idx] = static_cast<scalar_t>(standard_gamma_grad_one<acc_t>(
        static_cast<acc_t>(alpha[idx]), static_cast<acc_t>(output[idx])));
}

// Binomial sampling: geometric-sum inversion for small count*prob, the
// transformed-rejection sampler (BTRS) otherwise.  The template parameter
// drives both the arithmetic width and the curand draw width.
template <typename scalar_t>
__device__ inline scalar_t sample_binomial_btrs(
        curandStatePhilox4_32_10_t* state, scalar_t count, scalar_t prob);

template <typename scalar_t>
__device__ inline scalar_t binomial_stirling_tail(scalar_t k);

template <typename scalar_t>
__device__ inline scalar_t sample_binomial_draw(
        curandStatePhilox4_32_10_t* state) {
    if constexpr (std::is_same<scalar_t, double>::value) {
        return curand_uniform_double(state);
    } else {
        return curand_uniform(state);
    }
}

template <typename scalar_t>
__device__ inline scalar_t sample_binomial_one(
        curandStatePhilox4_32_10_t* state, scalar_t count, scalar_t prob) {
    if (count <= 0.0 || prob <= 0.0) {
        return 0.0;
    }
    if (prob >= 1.0) {
        return count;
    }
    if (!(prob > 0.0)) {
        // NaN probability.
        return static_cast<scalar_t>(NAN);
    }
    if (prob > 0.5) {
        return count - sample_binomial_one(state, count, scalar_t(1) - prob);
    }
    if (count * prob >= 10.0) {
        return sample_binomial_btrs(state, count, prob);
    }
    // Inversion: draw geometric strides until their sum exceeds the count.
    scalar_t num_geom = 0;
    scalar_t geom_sum = 0;
    const scalar_t logprob = ::log1p(-prob);
    while (true) {
        const scalar_t u = sample_binomial_draw<scalar_t>(state);
        geom_sum += ::ceil(::log(u) / logprob);
        if (geom_sum > count) break;
        num_geom += 1;
    }
    return num_geom;
}

// Transformed rejection for Binomial(count, prob) with prob <= 0.5 and
// count*prob >= 10.
template <typename scalar_t>
__device__ inline scalar_t sample_binomial_btrs(
        curandStatePhilox4_32_10_t* state, scalar_t count, scalar_t prob) {
    const scalar_t stddev = ::sqrt(count * prob * (1 - prob));
    const scalar_t b = 1.15 + 2.53 * stddev;
    const scalar_t a = -0.0873 + 0.0248 * b + 0.01 * prob;
    const scalar_t c = count * prob + 0.5;
    const scalar_t v_r = 0.92 - 4.2 / b;
    const scalar_t r = prob / (1 - prob);
    const scalar_t alpha = (2.83 + 5.1 / b) * stddev;
    const scalar_t m = ::floor((count + 1) * prob);
    while (true) {
        const scalar_t u = sample_binomial_draw<scalar_t>(state) - 0.5;
        const scalar_t v = sample_binomial_draw<scalar_t>(state);
        const scalar_t us = 0.5 - ::fabs(u);
        const scalar_t k =
            ::floor((2 * a / us + b) * u + c);
        if (k < 0 || k > count) continue;
        if (us >= 0.07 && v <= v_r) return k;
        const scalar_t v_log =
            ::log(v * alpha / (a / (us * us) + b));
        const scalar_t upperbound =
            ((m + 0.5) * ::log((m + 1) / (r * (count - m + 1))) +
             (count + 1) * ::log((count - m + 1) / (count - k + 1)) +
             (k + 0.5) * ::log(r * (count - k + 1) / (k + 1)) +
             binomial_stirling_tail(m) + binomial_stirling_tail(count - m) -
             binomial_stirling_tail(k) - binomial_stirling_tail(count - k));
        if (v_log <= upperbound) return k;
    }
}

// Stirling series tail for the log-factorial correction in BTRS.
template <typename scalar_t>
__device__ inline scalar_t binomial_stirling_tail(scalar_t k) {
    const scalar_t kTailValues[10] = {
        0.0810614667953272, 0.0413406959554092, 0.0276779256849983,
        0.02079067210376509, 0.0166446911898211, 0.0138761288230707,
        0.0118967099458917, 0.0104112652619720, 0.00925546218271273,
        0.00833056343336287};
    if (k < 10) {
        return kTailValues[static_cast<int>(k)];
    }
    const scalar_t kp1sq = (k + 1) * (k + 1);
    return (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1);
}

template <typename scalar_t>
__global__ void binomial_fill_impl(int64_t numel, PhiloxCudaState philox_args,
                                   const scalar_t* count, const scalar_t* prob,
                                   scalar_t* out) {
    uint64_t seed;
    uint64_t offset;
    philox_unpack(philox_args, &seed, &offset);
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);
    using acc_t = std::conditional_t<std::is_same<scalar_t, double>::value,
                                     double, float>;
    out[idx] = static_cast<scalar_t>(sample_binomial_one<acc_t>(
        &state, static_cast<acc_t>(count[idx]),
        static_cast<acc_t>(prob[idx])));
}

// Dirichlet reparameterized gradient through the beta decomposition:
// grad wrt alpha of a category drawn as gamma(alpha)/sum(gammas).
template <typename scalar_t>
__device__ inline scalar_t dirichlet_grad_alpha_small(scalar_t x, scalar_t alpha, scalar_t beta);
template <typename scalar_t>
__device__ inline scalar_t dirichlet_grad_beta_small(scalar_t x, scalar_t alpha, scalar_t beta);
template <typename scalar_t>
__device__ inline scalar_t dirichlet_grad_alpha_mid(scalar_t x, scalar_t alpha, scalar_t beta);
template <typename scalar_t>
__device__ inline scalar_t dirichlet_grad_rational(scalar_t x, scalar_t alpha, scalar_t beta);

template <typename scalar_t>
__device__ inline scalar_t dirichlet_grad_one(scalar_t x, scalar_t alpha,
                                              scalar_t total) {
    const scalar_t beta = total - alpha;
    const scalar_t boundary = total * x * (1 - x);
    if (x <= 0.5 && boundary < 2.5) {
        return dirichlet_grad_alpha_small(x, alpha, beta);
    }
    if (x >= 0.5 && boundary < 0.75) {
        return -dirichlet_grad_beta_small(1 - x, beta, alpha);
    }
    if (alpha > 6 && beta > 6) {
        return dirichlet_grad_alpha_mid(x, alpha, beta);
    }
    return dirichlet_grad_rational(x, alpha, total);
}

template <typename scalar_t>
__device__ inline scalar_t dirichlet_grad_alpha_small(scalar_t x,
                                                      scalar_t alpha,
                                                      scalar_t beta) {
    const scalar_t factor = sampling_digamma(alpha) -
        sampling_digamma(alpha + beta) - ::log(x);
    scalar_t numer = 1;
    scalar_t series = numer / alpha * (factor + 1 / alpha);
    for (int i = 1; i <= 10; ++i) {
        const scalar_t casted_i = static_cast<scalar_t>(i);
        numer *= (casted_i - beta) * x / casted_i;
        const scalar_t denom = alpha + casted_i;
        series += numer / denom * (factor + 1 / denom);
    }
    const scalar_t result = -::pow(1 - x, 1 - beta) * series;
    return isnan(result) ? 0.0 : result;
}

template <typename scalar_t>
__device__ inline scalar_t dirichlet_grad_beta_small(scalar_t x,
                                                     scalar_t alpha,
                                                     scalar_t beta) {
    const scalar_t factor =
        sampling_digamma(alpha + beta) - sampling_digamma(beta);
    scalar_t numer = 1, betas = 1, dbetas = 0, series = factor / alpha;
    for (int i = 1; i <= 8; ++i) {
        const scalar_t casted_i = static_cast<scalar_t>(i);
        numer *= -x / casted_i;
        dbetas = dbetas * (beta - casted_i) + betas;
        betas = betas * (beta - casted_i);
        series += numer / (alpha + casted_i) * (dbetas + factor * betas);
    }
    const scalar_t result = -::pow(1 - x, 1 - beta) * series;
    return isnan(result) ? 0.0 : result;
}

template <typename scalar_t>
__device__ inline scalar_t dirichlet_grad_alpha_mid(scalar_t x, scalar_t alpha,
                                                    scalar_t beta) {
    const scalar_t total = alpha + beta;
    const scalar_t mean = alpha / total;
    const scalar_t std = ::sqrt(alpha * beta / (total + 1)) / total;
    if (mean - 0.1 * std <= x && x <= mean + 0.1 * std) {
        // Series around the density's mode to avoid the singularity.
        const scalar_t poly =
            47 * x * (beta * beta) * (beta * beta) + alpha * (
                (43 + 20 * (16 + 27 * beta) * x) * (beta * beta) * beta + alpha * (
                    3 * (59 + 180 * beta - 90 * x) * (beta * beta) + alpha * (
                        (453 + 1620 * beta * (1 - x) - 455 * x) * beta + alpha * (
                            8 * (1 - x) * (135 * beta - 11)))));
        const scalar_t prefactor_num =
            (1 + 12 * alpha) * (1 + 12 * beta) / (total * total);
        const scalar_t prefactor_den =
            12960 * alpha * alpha * alpha * beta * beta * (1 + 12 * total);
        return prefactor_num / (1 - x) * poly / prefactor_den;
    }
    const scalar_t prefactor = -x / ::sqrt(2 * alpha * beta / total);
    const scalar_t stirling =
        (1 + 1 / (12 * alpha) + 1 / (288 * alpha * alpha)) *
        (1 + 1 / (12 * beta) + 1 / (288 * beta * beta)) /
        (1 + 1 / (12 * total) + 1 / (288 * total * total));
    const scalar_t term1_num =
        2 * (alpha * alpha) * (x - 1) + alpha * beta * (x - 1) - x * (beta * beta);
    const scalar_t axbx = alpha * (x - 1) + beta * x;
    const scalar_t term1_den =
        ::sqrt(2 * alpha / beta) * ::pow(total, 1.5f) * axbx * axbx;
    const scalar_t term1 = term1_num / term1_den;
    const scalar_t term2 = 0.5f * ::log(alpha / (total * x));
    const scalar_t term3_num = ::sqrt(8 * alpha * beta / total);
    const scalar_t term3_den = beta * x + alpha * (x - 1);
    const scalar_t term3 = term3_num / term3_den;
    const scalar_t term4_base =
        beta * ::log(beta / (total * (1 - x))) +
        alpha * ::log(alpha / (total * x));
    const scalar_t term4 = ::pow(term4_base, -1.5f);
    const scalar_t term1234 =
        term1 + term2 * (term3 + (x < mean ? term4 : -term4));
    return stirling * prefactor * term1234;
}

template <typename scalar_t>
__device__ inline scalar_t dirichlet_grad_rational(scalar_t x, scalar_t alpha,
                                                   scalar_t total) {
    const scalar_t u = ::log(x);
    const scalar_t a = ::log(alpha) - u;
    const scalar_t b = ::log(total) - a;
    const scalar_t pow_u[3] = {1, u, u * u};
    const scalar_t pow_a[3] = {1, a, a * a};
    static const scalar_t c[2][3][3][4] = {
        {{{1.003668233, -0.01061107488, -0.0657888334, 0.01201642863},
          {0.6336835991, -0.3557432599, 0.05486251648, -0.001465281033},
          {-0.03276231906, 0.004474107445, 0.002429354597, -0.0001557569013}},
         {{0.221950385, -0.3187676331, 0.01799915743, 0.01074823814},
          {-0.2951249643, 0.06219954479, 0.01535556598, 0.001550077057},
          {0.02155310298, 0.004170831599, 0.001292462449, 6.976601077e-05}},
         {{-0.05980841433, 0.008441916499, 0.01085618172, 0.002319392565},
          {0.02911413504, 0.01400243777, -0.002721828457, 0.000751041181},
          {0.005900514878, -0.001936558688, -9.495446725e-06, 5.385558597e-05}}},
        {{{1, -0.02924021934, -0.04438342661, 0.007285809825},
          {0.6357567472, -0.3473456711, 0.05454656494, -0.002407477521},
          {-0.03301322327, 0.004845219414, 0.00231480583, -0.0002307248149}},
         {{0.5925320577, -0.1757678135, 0.01505928619, 0.000564515273},
          {0.1014815858, -0.06589186703, 0.01272886114, -0.0007316646956},
          {-0.007258481865, 0.001096195486, 0.0003934994223, -4.12701925e-05}},
         {{0.06469649321, -0.0236701437, 0.002902096474, -5.896963079e-05},
          {0.001925008108, -0.002869809258, 0.0008000589141, -6.063713228e-05},
          {-0.0003477407336, 6.959756487e-05, 1.097287507e-05, -1.650964693e-06}}},
    };
    scalar_t p = 0.0;
    scalar_t q = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const scalar_t ua = pow_u[i] * pow_a[j];
            p += ua * (c[0][i][j][0] +
                       b * (c[0][i][j][1] + b * (c[0][i][j][2] + b * c[0][i][j][3])));
            q += ua * (c[1][i][j][0] +
                       b * (c[1][i][j][1] + b * (c[1][i][j][2] + b * c[1][i][j][3])));
        }
    }
    const scalar_t approx =
        x * (sampling_digamma(total) - sampling_digamma(alpha)) /
        (total - alpha);
    return p / q * approx;
}

template <typename scalar_t>
__global__ void dirichlet_grad_impl(int64_t numel, const scalar_t* x,
                                    const scalar_t* alpha, const scalar_t* total,
                                    scalar_t* out) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    out[idx] = dirichlet_grad_one(x[idx], alpha[idx], total[idx]);
}

// Dirichlet sampling: independent standard gammas per category normalized by
// their row sum.
template <typename scalar_t, typename acc_t>
__global__ void dirichlet_normalize_impl(int64_t rows, int64_t cols,
                                        scalar_t* data) {
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    acc_t sum = 0;
    for (int64_t j = 0; j < cols; ++j) {
        sum += static_cast<acc_t>(data[row * cols + j]);
    }
    if (sum > 0) {
        for (int64_t j = 0; j < cols; ++j) {
            data[row * cols + j] = static_cast<scalar_t>(
                static_cast<acc_t>(data[row * cols + j]) / sum);
        }
    }
}

// Reserves a philox counter slice for a rejection-loop kernel; draws per
// element are unbounded in the worst case, so the reservation is a generous
// fixed multiple of the element count.
PhiloxCudaState sampling_philox_state(std::optional<Generator> generator,
                                      int64_t numel) {
    const uint64_t counter_offset = 64u * kMaxGeneratorOffsetsPerCall *
        ((static_cast<uint64_t>(numel) + 255) / 256 + 1);
    if (generator.has_value()) {
        PhiloxCudaState state;
        state.seed = generator->random64();
        state.offset = 0;
        return state;
    }
    return philox_cuda_state(counter_offset);
}

template <typename scalar_t>
void launch_standard_gamma_fill(const Tensor& alpha, Tensor& out,
                                std::optional<Generator> generator) {
    const int64_t numel = alpha.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    auto philox_args = sampling_philox_state(std::move(generator), numel);
    standard_gamma_fill_impl<scalar_t>
        <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            numel, philox_args, alpha.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>());
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA standard_gamma Error: ") +
                  cudaGetErrorString(error));
    }
}

}  // namespace

Tensor standard_gamma_kernel_cuda(const Tensor& self,
                                  std::optional<Generator> generator) {
    Tensor out(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(),
               self.device());
    if (out.numel() == 0) return out;
    if (!isFloatingType(self.dtype())) {
        TP_THROW(TypeError, "standard_gamma expects a floating dtype");
    }
    Tensor alpha_c = self.is_contiguous() ? self : self.contiguous();
    if (self.dtype() == DType::Float32) {
        launch_standard_gamma_fill<float>(alpha_c, out, std::move(generator));
    } else if (self.dtype() == DType::Float64) {
        launch_standard_gamma_fill<double>(alpha_c, out, std::move(generator));
    } else if (self.dtype() == DType::Float16) {
        launch_standard_gamma_fill<Half>(alpha_c, out, std::move(generator));
    } else if (self.dtype() == DType::BFloat16) {
        launch_standard_gamma_fill<BFloat16>(alpha_c, out, std::move(generator));
    } else {
        TP_THROW(NotImplementedError, "standard_gamma only supports floating dtypes on CUDA");
    }
    return out;
}

Tensor standard_gamma_grad_kernel_cuda(const Tensor& self,
                                       const Tensor& output) {
    if (self.device() != output.device()) {
        TP_THROW(DeviceMismatchError,
                 "standard_gamma_grad: inputs must be on the same device");
    }
    if (!isFloatingType(self.dtype())) {
        TP_THROW(TypeError, "standard_gamma_grad expects a floating dtype");
    }
    if (output.dtype() != self.dtype() || output.shape() != self.shape()) {
        TP_THROW(RuntimeError,
                 "standard_gamma_grad: input and sample must have matching dtype and shape");
    }
    Tensor out(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(),
               self.device());
    if (out.numel() == 0) return out;
    Tensor alpha_c = self.is_contiguous() ? self : self.contiguous();
    Tensor output_c = output.is_contiguous() ? output : output.contiguous();
    const int64_t numel = out.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    if (self.dtype() == DType::Float32) {
        standard_gamma_grad_impl<float>
            <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                numel, alpha_c.data_ptr<float>(), output_c.data_ptr<float>(),
                out.data_ptr<float>());
    } else if (self.dtype() == DType::Float64) {
        standard_gamma_grad_impl<double>
            <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                numel, alpha_c.data_ptr<double>(), output_c.data_ptr<double>(),
                out.data_ptr<double>());
    } else if (self.dtype() == DType::Float16) {
        standard_gamma_grad_impl<Half>
            <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                numel, alpha_c.data_ptr<Half>(), output_c.data_ptr<Half>(),
                out.data_ptr<Half>());
    } else if (self.dtype() == DType::BFloat16) {
        standard_gamma_grad_impl<BFloat16>
            <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                numel, alpha_c.data_ptr<BFloat16>(), output_c.data_ptr<BFloat16>(),
                out.data_ptr<BFloat16>());
    } else {
        TP_THROW(NotImplementedError,
                 "standard_gamma_grad only supports floating dtypes on CUDA");
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA standard_gamma_grad Error: ") +
                  cudaGetErrorString(error));
    }
    return out;
}

Tensor binomial_kernel_cuda(const Tensor& count, const Tensor& prob,
                            std::optional<Generator> generator) {
    if (count.device() != prob.device()) {
        TP_THROW(DeviceMismatchError,
                 "binomial: count and prob must be on the same device");
    }
    if (!isFloatingType(count.dtype()) || !isFloatingType(prob.dtype())) {
        TP_THROW(TypeError,
                 "binomial only supports floating-point dtypes for count and prob");
    }
    if (count.dtype() != prob.dtype()) {
        TP_THROW(RuntimeError, "Found dtype ", toString(prob.dtype()),
                 " but expected ", toString(count.dtype()));
    }
    const std::vector<int64_t> shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(count.shape()),
        static_cast<std::vector<int64_t>>(prob.shape()));
    Tensor out(shape, count.dtype(), count.device());
    if (out.numel() == 0) return out;
    // Align both operands onto the broadcast shape.
    auto expand_to = [](const Tensor& t, const std::vector<int64_t>& target) {
        Tensor e = t;
        std::vector<int64_t> padded(static_cast<size_t>(
                                        target.size() - t.dim()), 1);
        for (const auto s : t.shape()) padded.push_back(s);
        return e.reshape(padded)
            .expand(static_cast<std::vector<int64_t>>(target))
            .contiguous();
    };
    Tensor count_c = expand_to(count, shape);
    Tensor prob_c = expand_to(prob, shape);
    const int64_t numel = out.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    auto philox_args = sampling_philox_state(std::move(generator), numel);
    if (count.dtype() == DType::Float32) {
        binomial_fill_impl<float>
            <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                numel, philox_args, count_c.data_ptr<float>(),
                prob_c.data_ptr<float>(), out.data_ptr<float>());
    } else if (count.dtype() == DType::Float64) {
        binomial_fill_impl<double>
            <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                numel, philox_args, count_c.data_ptr<double>(),
                prob_c.data_ptr<double>(), out.data_ptr<double>());
    } else if (count.dtype() == DType::Float16) {
        binomial_fill_impl<Half>
            <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                numel, philox_args, count_c.data_ptr<Half>(),
                prob_c.data_ptr<Half>(), out.data_ptr<Half>());
    } else if (count.dtype() == DType::BFloat16) {
        binomial_fill_impl<BFloat16>
            <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                numel, philox_args, count_c.data_ptr<BFloat16>(),
                prob_c.data_ptr<BFloat16>(), out.data_ptr<BFloat16>());
    } else {
        TP_THROW(NotImplementedError, "binomial only supports floating dtypes on CUDA");
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA binomial Error: ") +
                  cudaGetErrorString(error));
    }
    return out;
}

Tensor sample_dirichlet_kernel_cuda(const Tensor& self,
                                    std::optional<Generator> generator) {
    // Independent gammas per category, then a row-wise normalization.
    Tensor gamma = standard_gamma_kernel_cuda(self, std::move(generator));
    const int64_t ndim = gamma.dim();
    TP_CHECK(ndim >= 1, "sample_dirichlet expects at least 1 dimension");
    if (gamma.numel() == 0 || gamma.size(-1) == 0) {
        return gamma;
    }
    Tensor flat = gamma.reshape(
        {gamma.numel() / gamma.size(-1), gamma.size(-1)}).contiguous();
    const int64_t rows = flat.size(0);
    const int64_t cols = flat.size(1);
    const int threads = 256;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    if (gamma.dtype() == DType::Float32) {
        dirichlet_normalize_impl<float, float>
            <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                rows, cols, flat.data_ptr<float>());
    } else if (gamma.dtype() == DType::Float64) {
        dirichlet_normalize_impl<double, double>
            <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                rows, cols, flat.data_ptr<double>());
    } else if (gamma.dtype() == DType::Float16) {
        dirichlet_normalize_impl<Half, float>
            <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                rows, cols, flat.data_ptr<Half>());
    } else if (gamma.dtype() == DType::BFloat16) {
        dirichlet_normalize_impl<BFloat16, float>
            <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                rows, cols, flat.data_ptr<BFloat16>());
    } else {
        TP_THROW(NotImplementedError,
                 "sample_dirichlet only supports floating dtypes on CUDA");
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA sample_dirichlet Error: ") +
                  cudaGetErrorString(error));
    }
    return flat.reshape(static_cast<std::vector<int64_t>>(gamma.shape()));
}

Tensor dirichlet_grad_kernel_cuda(const Tensor& x, const Tensor& alpha,
                                  const Tensor& total) {
    Tensor out(static_cast<std::vector<int64_t>>(x.shape()), x.dtype(),
               x.device());
    if (out.numel() == 0) return out;
    Tensor x_c = x.is_contiguous() ? x : x.contiguous();
    Tensor alpha_c = alpha.is_contiguous() ? alpha : alpha.contiguous();
    Tensor total_c = total.is_contiguous() ? total : total.contiguous();
    const int64_t numel = out.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    if (x.dtype() == DType::Float32) {
        dirichlet_grad_impl<float>
            <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                numel, x_c.data_ptr<float>(), alpha_c.data_ptr<float>(),
                total_c.data_ptr<float>(), out.data_ptr<float>());
    } else if (x.dtype() == DType::Float64) {
        dirichlet_grad_impl<double>
            <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                numel, x_c.data_ptr<double>(), alpha_c.data_ptr<double>(),
                total_c.data_ptr<double>(), out.data_ptr<double>());
    } else {
        TP_THROW(NotImplementedError,
                 "dirichlet_grad only supports Float32/Float64 on CUDA for now");
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA dirichlet_grad Error: ") +
                  cudaGetErrorString(error));
    }
    return out;
}

template <typename output_t, typename probability_t>
__global__ void bernoulli_tensor_fill_impl(
        int64_t numel, PhiloxCudaState philox_args,
        const probability_t* probability, output_t* output) {
    uint64_t seed;
    uint64_t offset;
    philox_unpack(philox_args, &seed, &offset);
    const int64_t thread_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t thread_count =
        static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t linear_index = thread_index;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, thread_index, offset, &state);
    for (; linear_index < numel; linear_index += thread_count * 4) {
        const float4 random = curand_uniform4(&state);
        const int64_t i0 = linear_index;
        const int64_t i1 = linear_index + thread_count;
        const int64_t i2 = linear_index + thread_count * 2;
        const int64_t i3 = linear_index + thread_count * 3;
        if (i0 < numel) {
            const double p = static_cast<double>(probability[i0]);
            assert(0.0 <= p && p <= 1.0);
            output[i0] = static_cast<output_t>(random.x <= p);
        }
        if (i1 < numel) {
            const double p = static_cast<double>(probability[i1]);
            assert(0.0 <= p && p <= 1.0);
            output[i1] = static_cast<output_t>(random.y <= p);
        }
        if (i2 < numel) {
            const double p = static_cast<double>(probability[i2]);
            assert(0.0 <= p && p <= 1.0);
            output[i2] = static_cast<output_t>(random.z <= p);
        }
        if (i3 < numel) {
            const double p = static_cast<double>(probability[i3]);
            assert(0.0 <= p && p <= 1.0);
            output[i3] = static_cast<output_t>(random.w <= p);
        }
    }
}

template <typename output_t>
__global__ void bernoulli_scalar_fill_impl(
        int64_t numel, PhiloxCudaState philox_args, double p,
        output_t* output) {
    uint64_t seed;
    uint64_t offset;
    philox_unpack(philox_args, &seed, &offset);
    const int64_t thread_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t thread_count =
        static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t linear_index = thread_index;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, thread_index, offset, &state);
    for (; linear_index < numel; linear_index += thread_count * 4) {
        const float4 random = curand_uniform4(&state);
        const int64_t i0 = linear_index;
        const int64_t i1 = linear_index + thread_count;
        const int64_t i2 = linear_index + thread_count * 2;
        const int64_t i3 = linear_index + thread_count * 3;
        if (i0 < numel) output[i0] = static_cast<output_t>(random.x <= p);
        if (i1 < numel) output[i1] = static_cast<output_t>(random.y <= p);
        if (i2 < numel) output[i2] = static_cast<output_t>(random.z <= p);
        if (i3 < numel) output[i3] = static_cast<output_t>(random.w <= p);
    }
}

PhiloxCudaState bernoulli_philox_state(
        std::optional<Generator> generator, uint64_t increment) {
    if (generator.has_value()) {
        PhiloxCudaState state;
        state.seed = generator->random64();
        state.offset = 0;
        return state;
    }
    return philox_cuda_state(increment);
}

template <typename output_t, typename probability_t>
void launch_bernoulli_tensor(const Tensor& probability, Tensor& output,
                             std::optional<Generator> generator) {
    const auto policy = calc_execution_policy(output.numel(), 4);
    const uint64_t counter_offset = std::get<0>(policy);
    const dim3 grid = std::get<1>(policy);
    const dim3 block = std::get<2>(policy);
    const PhiloxCudaState philox_args =
        bernoulli_philox_state(std::move(generator), counter_offset);
    bernoulli_tensor_fill_impl<output_t, probability_t>
        <<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            output.numel(), philox_args, probability.data_ptr<probability_t>(),
            output.data_ptr<output_t>());
}

template <typename probability_t>
void launch_bernoulli_tensor_for_output(
        const Tensor& probability, Tensor& output,
        std::optional<Generator> generator) {
#define TP_BERNOULLI_TENSOR_OUTPUT_CASE(ctype, name) \
    case DType::name: \
        launch_bernoulli_tensor<ctype, probability_t>( \
            probability, output, std::move(generator)); \
        break;
    switch (output.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_BERNOULLI_TENSOR_OUTPUT_CASE)
        default:
            TP_THROW(NotImplementedError,
                     "bernoulli_ output dtype is not supported");
    }
#undef TP_BERNOULLI_TENSOR_OUTPUT_CASE
}

Tensor prepare_bernoulli_probabilities(const Tensor& output,
                                       const Tensor& probabilities) {
    if (!isFloatingType(probabilities.dtype())) {
        TP_THROW(TypeError,
                 "bernoulli_ probability tensor must have a floating dtype");
    }
    const DType probability_dtype =
        output.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    Tensor probability = probabilities.dtype() == probability_dtype
        ? probabilities
        : probabilities.to(probability_dtype);
    if (probability.shape() != output.shape()) {
        probability = probability.expand(
            static_cast<std::vector<int64_t>>(output.shape()));
    }
    return probability.is_contiguous() ? probability : probability.contiguous();
}

Tensor& bernoulli_tensor_inplace_kernel_cuda(
        Tensor& self, const Tensor& probabilities,
        std::optional<Generator> generator) {
    if (self.device() != probabilities.device()) {
        TP_THROW(DeviceMismatchError,
                 "bernoulli_: probability tensor must be on the same device");
    }
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return bernoulli_tensor_inplace_kernel_cuda(
                t, probabilities, std::move(generator));
        });
    }

    Tensor probability = prepare_bernoulli_probabilities(self, probabilities);
    const Tensor valid_probabilities = Tensor::logical_and(
        probability.ge(Scalar(0)), probability.le(Scalar(1)));
    if (!valid_probabilities.all().item<bool>()) {
        TP_THROW(ValueError,
                 "bernoulli_ expects probability values in [0, 1]");
    }
    if (self.dtype() == DType::Float64) {
        launch_bernoulli_tensor_for_output<double>(
            probability, self, std::move(generator));
    } else {
        launch_bernoulli_tensor_for_output<float>(
            probability, self, std::move(generator));
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA bernoulli_ Error: ") +
                    cudaGetErrorString(error));
    }
    return self;
}

template <typename output_t>
void launch_bernoulli_scalar(Tensor& output, double p,
                             std::optional<Generator> generator) {
    const auto policy = calc_execution_policy(output.numel(), 4);
    const uint64_t counter_offset = std::get<0>(policy);
    const dim3 grid = std::get<1>(policy);
    const dim3 block = std::get<2>(policy);
    const PhiloxCudaState philox_args =
        bernoulli_philox_state(std::move(generator), counter_offset);
    bernoulli_scalar_fill_impl<output_t>
        <<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            output.numel(), philox_args, p, output.data_ptr<output_t>());
}

Tensor& bernoulli_scalar_inplace_kernel_cuda(
        Tensor& self, double p, std::optional<Generator> generator) {
    if (!(p >= 0.0 && p <= 1.0)) {
        TP_THROW(ValueError, "bernoulli_ expects p to be in [0, 1]");
    }
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return bernoulli_scalar_inplace_kernel_cuda(
                t, p, std::move(generator));
        });
    }
#define TP_BERNOULLI_SCALAR_OUTPUT_CASE(ctype, name) \
    case DType::name: \
        launch_bernoulli_scalar<ctype>(self, p, std::move(generator)); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_BERNOULLI_SCALAR_OUTPUT_CASE)
        default:
            TP_THROW(NotImplementedError,
                     "bernoulli_ output dtype is not supported");
    }
#undef TP_BERNOULLI_SCALAR_OUTPUT_CASE

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA bernoulli_ Error: ") +
                    cudaGetErrorString(error));
    }
    return self;
}

Tensor randperm_kernel_cuda(int64_t n, DType dtype, Device device) {
    if (dtype != DType::Int64 && dtype != DType::Int32) {
        TP_THROW(NotImplementedError, "randperm() only supports Int64/Int32 on CUDA");
    }
    Tensor idx(std::vector<int64_t>{n}, DType::Int64, device);
    if (n == 0) {
        if (dtype == DType::Int32) {
            Tensor out = Tensor::empty({n}, DType::Int32, device);
            return out;
        }
        return idx;
    }
    // Distinct-with-probability-1 fp32 keys; ties are astronomically unlikely.
    Tensor keys = Tensor::empty({n}, DType::Float32, device);
    float* data = keys.data_ptr<float>();
    distribution_nullary_kernel<float, float4, 4>(
        data, n,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
        [] __device__ (float val) { return val; });
    extern Tensor argsort_cuda(const Tensor& self, int64_t dim, bool descending);
    idx = argsort_cuda(keys, 0, false);
    if (dtype == DType::Int64) return idx;
    Tensor out = Tensor::empty({n}, DType::Int32, device);
    out.copy_(idx);
    return out;
}

// --- Stub-ABI adapters (schema-level optional<DType>/optional<Device>) ------
namespace {

Tensor rand_stub_cuda(const std::vector<int64_t>& size, std::optional<DType> dtype,
                      std::optional<Device> device) {
    return rand_kernel_cuda(size, dtype.value_or(DType::Float32),
                            device.value_or(Device(DeviceType::CUDA)));
}

Tensor randn_stub_cuda(const std::vector<int64_t>& size, std::optional<DType> dtype,
                       std::optional<Device> device) {
    return randn_kernel_cuda(size, dtype.value_or(DType::Float32),
                             device.value_or(Device(DeviceType::CUDA)));
}

Tensor randn_generator_stub_cuda(const std::vector<int64_t>& size,
                                 std::optional<Generator> generator,
                                 std::optional<DType> dtype,
                                 std::optional<int64_t> layout,
                                 std::optional<Device> device,
                                 std::optional<bool> pin_memory) {
    if (layout.has_value() && *layout != 2) {
        TP_THROW(NotImplementedError,
                 "randn is only implemented for strided (dense) layout tensors");
    }
    if (pin_memory.value_or(false)) {
        TP_THROW(RuntimeError,
                 "pin_memory is not valid for CUDA random tensor outputs");
    }
    return randn_kernel_cuda(size, dtype.value_or(DType::Float32),
                             device.value_or(Device(DeviceType::CUDA)),
                             std::move(generator));
}

Tensor randint_stub_cuda(int64_t low, int64_t high, const std::vector<int64_t>& size,
                         DType dtype, std::optional<Device> device) {
    return randint_kernel_cuda(low, high, size, dtype,
                               device.value_or(Device(DeviceType::CUDA)));
}

Tensor randperm_stub_cuda(int64_t n, DType dtype, std::optional<Device> device) {
    return randperm_kernel_cuda(n, dtype, device.value_or(Device(DeviceType::CUDA)));
}

Tensor bernoulli_kernel_cuda(const Tensor& self,
                             std::optional<Generator> generator) {
    Tensor out(static_cast<std::vector<int64_t>>(self.shape()),
               self.dtype(), self.device());
    bernoulli_tensor_inplace_kernel_cuda(out, self, std::move(generator));
    return out;
}

Tensor& bernoulli_out_kernel_cuda(const Tensor& self,
                                  std::optional<Generator> generator,
                                  Tensor& out) {
    if (self.device() != out.device()) {
        TP_THROW(DeviceMismatchError,
                 "bernoulli: output must be on the same device as input");
    }
    out.resize_(static_cast<std::vector<int64_t>>(self.shape()));
    return bernoulli_tensor_inplace_kernel_cuda(out, self,
                                                std::move(generator));
}

Tensor bernoulli_p_kernel_cuda(const Tensor& self, double p,
                               std::optional<Generator> generator) {
    Tensor out(static_cast<std::vector<int64_t>>(self.shape()),
               self.dtype(), self.device());
    bernoulli_scalar_inplace_kernel_cuda(out, p, std::move(generator));
    return out;
}

} // anonymous namespace

Tensor normal_broadcast_kernel_cuda(const Tensor& mean, const Tensor& std) {
    if (mean.device() != std.device()) {
        TP_THROW(DeviceMismatchError, "normal: mean and std must be on the same device");
    }
    if (mean.dtype() != std.dtype()) {
        TP_THROW(RuntimeError, "normal: mean and std must have the same dtype");
    }
    if (std.numel() > 0 && !std.ge(Scalar(0)).all().item<bool>()) {
        TP_THROW(RuntimeError, "normal: standard deviation must be non-negative");
    }

    std::vector<int64_t> shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(mean.shape()),
        static_cast<std::vector<int64_t>>(std.shape()));
    Tensor out(shape, mean.dtype(), mean.device());
    if (out.numel() == 0) return out;
    out.normal_(0.0, 1.0);
    out.mul_(std).add_(mean);
    return out;
}

/*
 * The registration table is kept at the end of the translation unit so every
 * overload has a complete schema-level function type before registration.
 */

TENSORPLAY_LIBRARY_IMPL(CUDA, RandomKernels) {
    m.impl("rand", rand_stub_cuda);
    m.impl("randn", randn_stub_cuda);
    m.impl("randn.generator", randn_generator_stub_cuda);
    m.impl("rand_like", rand_like_kernel_cuda);
    m.impl("randn_like", randn_like_kernel_cuda);
    m.impl("uniform_", uniform_kernel_cuda);
    m.impl("normal_", normal_kernel_cuda);
    m.impl("exponential_", exponential_kernel_cuda);
    m.impl("geometric_", geometric_kernel_cuda);
    m.impl("log_normal_", log_normal_kernel_cuda);
    m.impl("cauchy_", cauchy_kernel_cuda);
    m.impl("random_", random_kernel_cuda);
    m.impl("bernoulli", bernoulli_kernel_cuda);
    m.impl("bernoulli.out", bernoulli_out_kernel_cuda);
    m.impl("bernoulli.p", bernoulli_p_kernel_cuda);
    m.impl("bernoulli_.Tensor", bernoulli_tensor_inplace_kernel_cuda);
    m.impl("bernoulli_.float", bernoulli_scalar_inplace_kernel_cuda);
    m.impl("normal", normal_broadcast_kernel_cuda);
    m.impl("poisson", poisson_kernel_cuda);
    m.impl("_standard_gamma", standard_gamma_kernel_cuda);
    m.impl("_standard_gamma_grad", standard_gamma_grad_kernel_cuda);
    m.impl("binomial", binomial_kernel_cuda);
    m.impl("_sample_dirichlet", sample_dirichlet_kernel_cuda);
    m.impl("_dirichlet_grad", dirichlet_grad_kernel_cuda);
    m.impl("randint", randint_stub_cuda);
    m.impl("randint_like", randint_like_kernel_cuda);
    m.impl("randperm", randperm_stub_cuda);
}

} // namespace cuda
} // namespace tensorplay
