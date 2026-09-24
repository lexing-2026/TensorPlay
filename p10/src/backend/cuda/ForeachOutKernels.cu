#include "Exception.h"
#include "ForeachKernels.h"
#include "CUDARuntime.h"
#include "ForeachMultiTensor.cuh"

#include <algorithm>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {
// ---------------------------------------------------------------------------
// Remaining functional foreach operations and the _out variant family.
// The _out variants compute functionally, then copy each result into the
// matching output handle.
// ---------------------------------------------------------------------------

static void copy_foreach_out_cuda(std::vector<Tensor> result,
                                  std::vector<Tensor> out,
                                  const char* op_name) {
    if (result.size() != out.size()) {
        TP_THROW(ValueError, std::string(op_name) +
            ": output list must have the same length as the input list");
    }
    for (size_t i = 0; i < result.size(); ++i) {
        out[i].copy_(result[i]);
    }
}

#define DEFINE_FOREACH_EXTRA_UNARY(NAME) \
std::vector<Tensor> foreach_##NAME##_cuda(const std::vector<Tensor>& self) { \
    std::vector<Tensor> out; \
    out.reserve(self.size()); \
    for (const auto& value : self) out.push_back(value.NAME()); \
    return out; \
} \
void foreach_##NAME##_inplace_cuda(std::vector<Tensor> self) { \
    for (auto& value : self) value.copy_(value.NAME()); \
}
DEFINE_FOREACH_EXTRA_UNARY(acos)
DEFINE_FOREACH_EXTRA_UNARY(asin)
DEFINE_FOREACH_EXTRA_UNARY(atan)
DEFINE_FOREACH_EXTRA_UNARY(ceil)
DEFINE_FOREACH_EXTRA_UNARY(cos)
DEFINE_FOREACH_EXTRA_UNARY(cosh)
DEFINE_FOREACH_EXTRA_UNARY(erf)
DEFINE_FOREACH_EXTRA_UNARY(erfc)
DEFINE_FOREACH_EXTRA_UNARY(exp)
DEFINE_FOREACH_EXTRA_UNARY(expm1)
DEFINE_FOREACH_EXTRA_UNARY(floor)
DEFINE_FOREACH_EXTRA_UNARY(frac)
DEFINE_FOREACH_EXTRA_UNARY(lgamma)
DEFINE_FOREACH_EXTRA_UNARY(log)
DEFINE_FOREACH_EXTRA_UNARY(log10)
DEFINE_FOREACH_EXTRA_UNARY(log1p)
DEFINE_FOREACH_EXTRA_UNARY(log2)
DEFINE_FOREACH_EXTRA_UNARY(round)
DEFINE_FOREACH_EXTRA_UNARY(sigmoid)
DEFINE_FOREACH_EXTRA_UNARY(sin)
DEFINE_FOREACH_EXTRA_UNARY(sinh)
DEFINE_FOREACH_EXTRA_UNARY(tanh)
DEFINE_FOREACH_EXTRA_UNARY(tan)
DEFINE_FOREACH_EXTRA_UNARY(trunc)
#undef DEFINE_FOREACH_EXTRA_UNARY

std::vector<Tensor> foreach_max_cuda(const std::vector<Tensor>& self) {
    std::vector<Tensor> out;
    out.reserve(self.size());
    for (const auto& value : self) out.push_back(value.max());
    return out;
}

std::vector<Tensor> foreach_zero_cuda(const std::vector<Tensor>& self) {
    std::vector<Tensor> out;
    out.reserve(self.size());
    for (const auto& value : self) out.push_back(Tensor::zeros_like(value));
    return out;
}

std::vector<Tensor> foreach_clone_cuda(const std::vector<Tensor>& self,
                                       std::optional<int64_t> /*memory_format*/) {
    std::vector<Tensor> out;
    out.reserve(self.size());
    for (const auto& value : self) out.push_back(value.clone());
    return out;
}

std::vector<Tensor> foreach_copy_cuda(const std::vector<Tensor>& self,
                                      const std::vector<Tensor>& src,
                                      bool /*non_blocking*/) {
    if (self.size() != src.size()) {
        TP_THROW(ValueError, "_foreach_copy: list sizes must match");
    }
    std::vector<Tensor> out;
    out.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) out.push_back(src[i].clone());
    return out;
}

std::vector<Tensor> foreach_mm_cuda(const std::vector<Tensor>& self,
                                    const std::vector<Tensor>& mat2) {
    if (self.size() != mat2.size()) {
        TP_THROW(ValueError, "_foreach_mm: list sizes must match");
    }
    std::vector<Tensor> out;
    out.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) out.push_back(self[i].mm(mat2[i]));
    return out;
}

namespace {

constexpr int kNormMaxTensors = 128;
constexpr int kNormThreads = 256;
constexpr int64_t kNormChunkSize = 65536;
constexpr int64_t kNormMaxGridY = 65535;

__host__ __device__ inline int64_t foreach_norm_chunk_count(int64_t numel) {
    return numel / kNormChunkSize + (numel % kNormChunkSize != 0);
}

template <typename T, typename M>
struct ForeachNormMetadata {
    const T* inputs[kNormMaxTensors]{};
    T* outputs[kNormMaxTensors]{};
    int64_t numels[kNormMaxTensors]{};
    M* partials = nullptr;
    int32_t tensor_count = 0;
    int32_t max_chunks = 0;
};

template <typename T, typename M>
__global__ void foreach_norm_partial_kernel(
        ForeachNormMetadata<T, M> metadata) {
    const int32_t tensor = static_cast<int32_t>(blockIdx.x);
    const int32_t chunk = static_cast<int32_t>(blockIdx.y);
    if (tensor >= metadata.tensor_count || chunk >= metadata.max_chunks) return;

    const int64_t begin = static_cast<int64_t>(chunk) * kNormChunkSize;
    const int64_t count = metadata.numels[tensor];
    if (begin >= count) return;
    int64_t end = begin + kNormChunkSize;
    if (end > count) end = count;

    M value = M(0);
    for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const M x = static_cast<M>(metadata.inputs[tensor][i]);
        value += x * x;
    }

    __shared__ M values[kNormThreads];
    values[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) values[threadIdx.x] += values[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        metadata.partials[static_cast<int64_t>(tensor) * metadata.max_chunks + chunk] =
            values[0];
    }
}

template <typename T, typename M>
__global__ void foreach_norm_finalize_kernel(
        ForeachNormMetadata<T, M> metadata) {
    const int32_t tensor = static_cast<int32_t>(blockIdx.x);
    if (tensor >= metadata.tensor_count) return;

    const int64_t count = metadata.numels[tensor];
    const int32_t chunks = static_cast<int32_t>(foreach_norm_chunk_count(count));
    M value = M(0);
    for (int32_t chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
        value += metadata.partials[
            static_cast<int64_t>(tensor) * metadata.max_chunks + chunk];
    }

    __shared__ M values[kNormThreads];
    values[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) values[threadIdx.x] += values[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const M result = std::is_same_v<M, float>
            ? static_cast<M>(sqrtf(static_cast<float>(values[0])))
            : static_cast<M>(::sqrt(static_cast<double>(values[0])));
        metadata.outputs[tensor][0] = static_cast<T>(result);
    }
}

template <typename T, typename M>
void launch_foreach_norm(const std::vector<Tensor>& self,
                         std::vector<Tensor>& out) {
    const int64_t tensor_count = static_cast<int64_t>(self.size());
    const DType partial_dtype = self[0].dtype() == DType::Float64
        ? DType::Float64 : DType::Float32;
    OptionalCUDAGuard device_guard(self[0].device());
    const auto stream = getCurrentCUDAStream().stream();

    for (int64_t base = 0; base < tensor_count; base += kNormMaxTensors) {
        const int32_t count = static_cast<int32_t>(
            std::min<int64_t>(kNormMaxTensors, tensor_count - base));
        int32_t max_chunks = 1;
        for (int32_t i = 0; i < count; ++i) {
            const int64_t numel = self[base + i].numel();
            const int64_t chunks = foreach_norm_chunk_count(numel);
            max_chunks = std::max<int32_t>(max_chunks, static_cast<int32_t>(chunks));
        }

        Tensor partials = Tensor::empty(
            {static_cast<int64_t>(count) * max_chunks}, partial_dtype,
            self[base].device());
        ForeachNormMetadata<T, M> metadata{};
        metadata.partials = partials.data_ptr<M>();
        metadata.tensor_count = count;
        metadata.max_chunks = max_chunks;
        for (int32_t i = 0; i < count; ++i) {
            metadata.inputs[i] = self[base + i].data_ptr<T>();
            metadata.outputs[i] = out[base + i].data_ptr<T>();
            metadata.numels[i] = self[base + i].numel();
        }

        foreach_norm_partial_kernel<T, M><<<
            dim3(static_cast<unsigned>(count), static_cast<unsigned>(max_chunks)),
            kNormThreads, 0, stream>>>(metadata);
        checkCuda(cudaGetLastError(), "foreach norm partial kernel launch");
        foreach_norm_finalize_kernel<T, M><<<
            static_cast<unsigned>(count), kNormThreads, 0, stream>>>(metadata);
        checkCuda(cudaGetLastError(), "foreach norm finalize kernel launch");
    }
}

} // namespace

std::vector<Tensor> foreach_norm_cuda(const std::vector<Tensor>& self,
                                      const Scalar& ord,
                                      std::optional<DType> dtype) {
    std::vector<Tensor> out;
    out.reserve(self.size());
    bool fast_l2 = !self.empty() && !dtype.has_value() &&
        ord.toDouble() == 2.0 && foreach_mta::eligible_list(self);
    if (fast_l2) {
        for (const auto& value : self) {
            if (value.requires_grad() || value.numel() == 0 ||
                foreach_norm_chunk_count(value.numel()) > kNormMaxGridY) {
                fast_l2 = false;
                break;
            }
        }
    }
    if (fast_l2) {
        for (const auto& value : self) {
            out.push_back(Tensor::empty({}, value.dtype(), value.device()));
        }
        foreach_mta::dispatch_dtype(self[0].dtype(), [&]<typename T, typename M>() {
            launch_foreach_norm<T, M>(self, out);
        });
        return out;
    }
    for (const auto& value : self) {
        Tensor input = dtype.has_value() ? value.to(*dtype) : value;
        out.push_back(input.norm(ord.toDouble()));
    }
    return out;
}

std::vector<Tensor> foreach_powsum_cuda(const std::vector<Tensor>& self,
                                        const Scalar& ord,
                                        std::optional<DType> dtype) {
    std::vector<Tensor> out;
    out.reserve(self.size());
    for (const auto& value : self) {
        Tensor input = dtype.has_value() ? value.to(*dtype) : value;
        out.push_back(input.abs().pow(ord).sum());
    }
    return out;
}

#define DEFINE_FOREACH_UNARY_OUT_CUDA(NAME) \
void foreach_##NAME##_out_cuda(const std::vector<Tensor>& self, \
                               std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_cuda(self), std::move(out), \
                          "_foreach_" #NAME ".out"); \
}
DEFINE_FOREACH_UNARY_OUT_CUDA(sqrt)
DEFINE_FOREACH_UNARY_OUT_CUDA(rsqrt)
DEFINE_FOREACH_UNARY_OUT_CUDA(neg)
DEFINE_FOREACH_UNARY_OUT_CUDA(abs)
DEFINE_FOREACH_UNARY_OUT_CUDA(sign)
DEFINE_FOREACH_UNARY_OUT_CUDA(reciprocal)
DEFINE_FOREACH_UNARY_OUT_CUDA(acos)
DEFINE_FOREACH_UNARY_OUT_CUDA(asin)
DEFINE_FOREACH_UNARY_OUT_CUDA(atan)
DEFINE_FOREACH_UNARY_OUT_CUDA(ceil)
DEFINE_FOREACH_UNARY_OUT_CUDA(cos)
DEFINE_FOREACH_UNARY_OUT_CUDA(cosh)
DEFINE_FOREACH_UNARY_OUT_CUDA(erf)
DEFINE_FOREACH_UNARY_OUT_CUDA(erfc)
DEFINE_FOREACH_UNARY_OUT_CUDA(exp)
DEFINE_FOREACH_UNARY_OUT_CUDA(expm1)
DEFINE_FOREACH_UNARY_OUT_CUDA(floor)
DEFINE_FOREACH_UNARY_OUT_CUDA(frac)
DEFINE_FOREACH_UNARY_OUT_CUDA(lgamma)
DEFINE_FOREACH_UNARY_OUT_CUDA(log)
DEFINE_FOREACH_UNARY_OUT_CUDA(log10)
DEFINE_FOREACH_UNARY_OUT_CUDA(log1p)
DEFINE_FOREACH_UNARY_OUT_CUDA(log2)
DEFINE_FOREACH_UNARY_OUT_CUDA(round)
DEFINE_FOREACH_UNARY_OUT_CUDA(sigmoid)
DEFINE_FOREACH_UNARY_OUT_CUDA(sin)
DEFINE_FOREACH_UNARY_OUT_CUDA(sinh)
DEFINE_FOREACH_UNARY_OUT_CUDA(tan)
DEFINE_FOREACH_UNARY_OUT_CUDA(tanh)
DEFINE_FOREACH_UNARY_OUT_CUDA(trunc)
#undef DEFINE_FOREACH_UNARY_OUT_CUDA

#define DEFINE_FOREACH_ADDSUB_OUT_CUDA(NAME) \
void foreach_##NAME##_scalar_out_cuda(const std::vector<Tensor>& self, const Scalar& scalar, \
                                      std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_scalar_cuda(self, scalar), std::move(out), \
                          "_foreach_" #NAME ".Scalar_out"); \
} \
void foreach_##NAME##_list_out_cuda(const std::vector<Tensor>& self, \
                                    const std::vector<Tensor>& other, const Scalar& alpha, \
                                    std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_list_cuda(self, other, alpha), std::move(out), \
                          "_foreach_" #NAME ".List_out"); \
} \
void foreach_##NAME##_scalar_list_out_cuda(const std::vector<Tensor>& self, \
                                           const std::vector<Scalar>& scalars, \
                                           std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_scalar_list_cuda(self, scalars), std::move(out), \
                          "_foreach_" #NAME ".ScalarList_out"); \
} \
void foreach_##NAME##_tensor_out_cuda(const std::vector<Tensor>& self, const Tensor& other, \
                                      const Scalar& alpha, std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_tensor_cuda(self, other, alpha), std::move(out), \
                          "_foreach_" #NAME ".Tensor_out"); \
}
DEFINE_FOREACH_ADDSUB_OUT_CUDA(add)
DEFINE_FOREACH_ADDSUB_OUT_CUDA(sub)
#undef DEFINE_FOREACH_ADDSUB_OUT_CUDA

#define DEFINE_FOREACH_MULDIV_OUT_CUDA(NAME) \
void foreach_##NAME##_scalar_out_cuda(const std::vector<Tensor>& self, const Scalar& scalar, \
                                      std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_scalar_cuda(self, scalar), std::move(out), \
                          "_foreach_" #NAME ".Scalar_out"); \
} \
void foreach_##NAME##_list_out_cuda(const std::vector<Tensor>& self, \
                                    const std::vector<Tensor>& other, \
                                    std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_list_cuda(self, other), std::move(out), \
                          "_foreach_" #NAME ".List_out"); \
} \
void foreach_##NAME##_scalar_list_out_cuda(const std::vector<Tensor>& self, \
                                           const std::vector<Scalar>& scalars, \
                                           std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_scalar_list_cuda(self, scalars), std::move(out), \
                          "_foreach_" #NAME ".ScalarList_out"); \
} \
void foreach_##NAME##_tensor_out_cuda(const std::vector<Tensor>& self, const Tensor& other, \
                                      std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_tensor_cuda(self, other), std::move(out), \
                          "_foreach_" #NAME ".Tensor_out"); \
}
DEFINE_FOREACH_MULDIV_OUT_CUDA(mul)
DEFINE_FOREACH_MULDIV_OUT_CUDA(div)
#undef DEFINE_FOREACH_MULDIV_OUT_CUDA

#define DEFINE_FOREACH_CLAMP_OUT_CUDA(NAME) \
void foreach_##NAME##_scalar_out_cuda(const std::vector<Tensor>& self, const Scalar& scalar, \
                                      std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_scalar_cuda(self, scalar), std::move(out), \
                          "_foreach_" #NAME ".Scalar_out"); \
} \
void foreach_##NAME##_list_out_cuda(const std::vector<Tensor>& self, \
                                    const std::vector<Tensor>& other, \
                                    std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_list_cuda(self, other), std::move(out), \
                          "_foreach_" #NAME ".List_out"); \
} \
void foreach_##NAME##_scalar_list_out_cuda(const std::vector<Tensor>& self, \
                                           const std::vector<Scalar>& scalars, \
                                           std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_scalar_list_cuda(self, scalars), std::move(out), \
                          "_foreach_" #NAME ".ScalarList_out"); \
}
DEFINE_FOREACH_CLAMP_OUT_CUDA(clamp_max)
DEFINE_FOREACH_CLAMP_OUT_CUDA(clamp_min)
DEFINE_FOREACH_CLAMP_OUT_CUDA(maximum)
DEFINE_FOREACH_CLAMP_OUT_CUDA(minimum)
#undef DEFINE_FOREACH_CLAMP_OUT_CUDA

// lerp overloads have differing weight types; write them out explicitly.
void foreach_lerp_scalar_out_cuda(const std::vector<Tensor>& self,
                                  const std::vector<Tensor>& end,
                                  Scalar weight,
                                  std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_lerp_scalar_cuda(self, end, weight), std::move(out),
                          "_foreach_lerp.Scalar_out");
}
void foreach_lerp_list_out_cuda(const std::vector<Tensor>& self,
                                const std::vector<Tensor>& end,
                                const std::vector<Tensor>& weight,
                                std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_lerp_list_cuda(self, end, weight), std::move(out),
                          "_foreach_lerp.List_out");
}
void foreach_lerp_scalar_list_out_cuda(const std::vector<Tensor>& self,
                                       const std::vector<Tensor>& end,
                                       const std::vector<Scalar>& weights,
                                       std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_lerp_scalar_list_cuda(self, end, weights), std::move(out),
                          "_foreach_lerp.ScalarList_out");
}

void foreach_pow_scalar_out_cuda(const std::vector<Tensor>& self, Scalar exponent,
                                 std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_pow_scalar_cuda(self, exponent), std::move(out),
                          "_foreach_pow.Scalar_out");
}
void foreach_pow_list_out_cuda(const std::vector<Tensor>& self,
                               const std::vector<Tensor>& exponent,
                               std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_pow_list_cuda(self, exponent), std::move(out),
                          "_foreach_pow.List_out");
}
void foreach_pow_scalar_list_out_cuda(const std::vector<Tensor>& self,
                                      const std::vector<Scalar>& exponents,
                                      std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_pow_scalar_list_cuda(self, exponents), std::move(out),
                          "_foreach_pow.ScalarList_out");
}

void foreach_addcmul_scalar_out_cuda(const std::vector<Tensor>& self,
                                     const std::vector<Tensor>& tensor1,
                                     const std::vector<Tensor>& tensor2, const Scalar& value,
                                     std::vector<Tensor> out) {
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i)
        result.push_back(self[i].addcmul(tensor1[i], tensor2[i], value));
    copy_foreach_out_cuda(std::move(result), std::move(out), "_foreach_addcmul.Scalar_out");
}
void foreach_addcmul_scalar_list_out_cuda(const std::vector<Tensor>& self,
                                          const std::vector<Tensor>& tensor1,
                                          const std::vector<Tensor>& tensor2,
                                          const std::vector<Scalar>& scalars,
                                          std::vector<Tensor> out) {
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i)
        result.push_back(self[i].addcmul(tensor1[i], tensor2[i], scalars[i]));
    copy_foreach_out_cuda(std::move(result), std::move(out), "_foreach_addcmul.ScalarList_out");
}
void foreach_addcmul_tensor_out_cuda(const std::vector<Tensor>& self,
                                     const std::vector<Tensor>& tensor1,
                                     const std::vector<Tensor>& tensor2, const Scalar& value,
                                     std::vector<Tensor> out) {
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i)
        result.push_back(self[i].addcmul(tensor1[i], tensor2[i], value));
    copy_foreach_out_cuda(std::move(result), std::move(out), "_foreach_addcmul.Tensor_out");
}
void foreach_addcdiv_scalar_out_cuda(const std::vector<Tensor>& self,
                                     const std::vector<Tensor>& tensor1,
                                     const std::vector<Tensor>& tensor2, const Scalar& value,
                                     std::vector<Tensor> out) {
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i)
        result.push_back(self[i].addcdiv(tensor1[i], tensor2[i], value));
    copy_foreach_out_cuda(std::move(result), std::move(out), "_foreach_addcdiv.Scalar_out");
}
void foreach_addcdiv_scalar_list_out_cuda(const std::vector<Tensor>& self,
                                          const std::vector<Tensor>& tensor1,
                                          const std::vector<Tensor>& tensor2,
                                          const std::vector<Scalar>& scalars,
                                          std::vector<Tensor> out) {
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i)
        result.push_back(self[i].addcdiv(tensor1[i], tensor2[i], scalars[i]));
    copy_foreach_out_cuda(std::move(result), std::move(out), "_foreach_addcdiv.ScalarList_out");
}
void foreach_addcdiv_tensor_out_cuda(const std::vector<Tensor>& self,
                                     const std::vector<Tensor>& tensor1,
                                     const std::vector<Tensor>& tensor2, const Scalar& value,
                                     std::vector<Tensor> out) {
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i)
        result.push_back(self[i].addcdiv(tensor1[i], tensor2[i], value));
    copy_foreach_out_cuda(std::move(result), std::move(out), "_foreach_addcdiv.Tensor_out");
}

void foreach_max_out_cuda(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_max_cuda(self), std::move(out), "_foreach_max.out");
}
void foreach_zero_out_cuda(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_zero_cuda(self), std::move(out), "_foreach_zero.out");
}
void foreach_clone_out_cuda(const std::vector<Tensor>& self,
                            std::optional<int64_t> memory_format,
                            std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_clone_cuda(self, memory_format), std::move(out),
                          "_foreach_clone.out");
}
void foreach_copy_out_cuda(const std::vector<Tensor>& self,
                           const std::vector<Tensor>& src, bool non_blocking,
                           std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_copy_cuda(self, src, non_blocking), std::move(out),
                          "_foreach_copy.out");
}
void foreach_norm_out_cuda(const std::vector<Tensor>& self, Scalar ord,
                           std::optional<DType> dtype, std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_norm_cuda(self, ord, dtype), std::move(out),
                          "_foreach_norm.Scalar_out");
}
void foreach_powsum_out_cuda(const std::vector<Tensor>& self, Scalar ord,
                             std::optional<DType> dtype, std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_powsum_cuda(self, ord, dtype), std::move(out),
                          "_foreach_powsum.Scalar_out");
}

}  // namespace cuda
}  // namespace tensorplay
