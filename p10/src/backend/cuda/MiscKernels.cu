// Misc CUDA kernels: meshgrid / roll / diff / masked_fill / one_hot / glu.
//
// These mirror the CPU composites in cpu/MiscKernels.cpp; every primitive
// invoked (slice/view/expand/cat/sigmoid/where/eq) is itself dispatched to the
// device backend, matching ATen where these ops are composite functions:
//   aten/src/ATen/native/TensorShape.cpp        meshgrid()
//   aten/src/ATen/native/TensorTransformations.{h,cpp}  roll()/roll_common()
//   aten/src/ATen/native/ReduceOps.cpp          diff()/diff_helper()
//   aten/src/ATen/native/Onehot.cpp             one_hot()
//   aten/src/ATen/native/GatedLinearUnit.cpp    glu()/glu_backward()

#include "Tensor.h"
#include "Dispatcher.h"
#include "Utils.h"
#include "CUDAGenerator.h"
#include <curand_kernel.h>
#include <algorithm>
#include <tuple>

namespace tensorplay {
namespace cuda {

// Defined below the registration table.
Tensor& resize__cuda(Tensor& self, const std::vector<int64_t>& size);
std::tuple<Tensor, Tensor> native_dropout_cuda(const Tensor& input, double p);

// Defined in PointwiseKernels.cu.
Tensor eq_kernel_cuda(const Tensor& self, const Tensor& other);


// Defined in PointwiseKernels.cu.
Tensor where_cuda(const Tensor& condition, const Tensor& self, const Tensor& other);

namespace {

inline int64_t wrap_dim_local(int64_t dim, int64_t ndim) {
    const int64_t min_ = -ndim;
    const int64_t max_ = ndim - 1;
    if (dim < min_ || dim > max_) {
        TP_THROW(IndexError,
                 "Dimension out of range (expected to be in range of [" +
                     std::to_string(min_) + ", " + std::to_string(max_) + "], but got " +
                     std::to_string(dim) + ")");
    }
    if (dim < 0) dim += ndim;
    return dim;
}

} // anonymous namespace

static Tensor diff_helper(const Tensor& self, int64_t n, int64_t dim) {
    // ATen ReduceOps.cpp diff_helper.
    Tensor result = self;
    n = n > self.size(dim) ? self.size(dim) : n;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t out_len = result.size(dim) - 1;
        result = result.slice(dim, 1, out_len + 1) - result.slice(dim, 0, out_len);
    }
    return result;
}

Tensor diff_cuda(const Tensor& self, int64_t n, int64_t dim, const Tensor& prepend, const Tensor& append) {
    const int64_t d = wrap_dim_local(dim, self.dim());
    const bool has_prepend = prepend.defined();
    const bool has_append = append.defined();
    if ((!has_prepend && !has_append) || n == 0) return diff_helper(self, n, d);
    std::vector<Tensor> pieces;
    if (has_prepend) pieces.push_back(prepend);
    pieces.push_back(self);
    if (has_append) pieces.push_back(append);
    return diff_helper(Tensor::cat(pieces, d), n, d);
}

Tensor one_hot_cuda(const Tensor& self, int64_t num_classes) {
    // ATen Onehot.cpp functional branch: eq(self.unsqueeze(-1), arange)
    if (self.dtype() != DType::Int64) {
        TP_THROW(RuntimeError, "one_hot is only applicable to index tensor of type LongTensor.");
    }
    if (num_classes == -1) {
        if (self.numel() == 0) {
            TP_THROW(RuntimeError, "Can not infer total number of classes from empty tensor.");
        }
        num_classes = self.max().item().to<int64_t>() + 1;
    }
    Tensor index = Tensor::arange(Scalar(static_cast<int64_t>(0)), Scalar(num_classes),
                                  Scalar(static_cast<int64_t>(1)), DType::Int64, self.device());
    auto sizes = static_cast<std::vector<int64_t>>(self.shape());
    sizes.push_back(1);
    return eq_kernel_cuda(self.view(sizes), index).to(DType::Int64);
}

Tensor glu_cuda(const Tensor& self, int64_t dim) {
    // ATen GatedLinearUnit.cpp / cpu Activation.cpp glu_kernel.
    if (self.dim() == 0) TP_THROW(RuntimeError, "glu does not support 0-dimensional tensors");
    const int64_t d = wrap_dim_local(dim, self.dim());
    const int64_t nIn = self.size(d);
    if (nIn % 2 != 0) {
        TP_THROW(RuntimeError, "Halving dimension must be even, but dimension " + std::to_string(d) +
                                   " is size " + std::to_string(nIn));
    }
    const int64_t half = nIn / 2;
    Tensor firstHalf = self.slice(d, 0, half);
    Tensor secondHalf = self.slice(d, half, nIn);
    return firstHalf * secondHalf.sigmoid();
}

Tensor glu_backward_cuda(const Tensor& grad_output, const Tensor& self, int64_t dim) {
    // ATen GatedLinearUnit.cpp glu_backward_cpu_out semantics.
    if (self.dim() == 0) TP_THROW(RuntimeError, "glu does not support 0-dimensional tensors");
    const int64_t d = wrap_dim_local(dim, self.dim());
    const int64_t nIn = self.size(d);
    if (nIn % 2 != 0) {
        TP_THROW(RuntimeError, "Halving dimension must be even, but dimension " + std::to_string(d) +
                                   " is size " + std::to_string(nIn));
    }
    const int64_t inputSize = nIn / 2;
    Tensor firstHalf = self.slice(d, 0, inputSize);
    Tensor secondHalf = self.slice(d, inputSize, nIn);
    Tensor sig_second = secondHalf.sigmoid();
    Tensor grad_input_first = grad_output * sig_second;
    Tensor grad_input_second = grad_output * firstHalf * sig_second * (1 - sig_second);
    return Tensor::cat({grad_input_first, grad_input_second}, d);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, MiscKernels) {
    m.impl("diff", diff_cuda);
    m.impl("one_hot", one_hot_cuda);
    m.impl("glu", glu_cuda);
    m.impl("glu_backward", glu_backward_cuda);
    m.impl("resize_", resize__cuda);
    m.impl("native_dropout", native_dropout_cuda);
}

namespace {

constexpr uint32_t kDropoutBlockSize = 256;
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

// Grid-stride fused dropout: each thread draws curand uniforms and writes
// both the scaled output element and the bool keep-mask, mirroring the
// philox counter discipline of RandomKernels.cu (offsets reserved host-side
// via philox_engine_inputs so results are launch-geometry independent).
template <typename scalar_t>
__global__ void native_dropout_kernel(int64_t numel, uint64_t seed,
                                      uint64_t offset, float p, float scale,
                                      const scalar_t* in, scalar_t* out,
                                      bool* mask) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);

    const int64_t total_threads =
        static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t rounded_size =
        ((numel - 1) / (total_threads * 4) + 1) * total_threads * 4;
    for (int64_t linear_index = idx; linear_index < rounded_size;
         linear_index += total_threads * 4) {
        float4 rand = curand_uniform4(&state);
        const float rands[4] = {rand.x, rand.y, rand.z, rand.w};
        #pragma unroll
        for (int ii = 0; ii < 4; ii++) {
            const int64_t li = linear_index + total_threads * ii;
            if (li < numel) {
                const bool keep = rands[ii] >= p;
                mask[li] = keep;
                out[li] = keep ? static_cast<scalar_t>(static_cast<double>(in[li]) * scale)
                               : static_cast<scalar_t>(0.0);
            }
        }
    }
}

} // namespace

// Mirrors resize__cpu: grow storage in place preserving contents, shrink is
// logical-only. Allocation/copy go through the CUDA caching allocator and
// copyAllocationBytes, so no explicit memcpy or stream handling is needed.
Tensor& resize__cuda(Tensor& self, const std::vector<int64_t>& size) {
    auto* impl = self.unsafeGetTensorImpl().get();
    int64_t new_numel = 1;
    for (int64_t s : size) {
        if (s < 0) {
            TP_THROW(ValueError, "resize_: negative sizes are not allowed");
        }
        new_numel *= s;
    }
    const size_t new_bytes = static_cast<size_t>(new_numel) * impl->itemsize();
    if (!impl->has_storage()) {
        if (new_bytes > 0) {
            impl->set_storage(
                Storage(new_bytes, getAllocator(impl->device().type()), impl->device()));
        }
    } else if (new_bytes > impl->storage().nbytes()) {
        Storage storage = impl->storage();
        storage.set_nbytes(new_bytes);
    }
    impl->set_sizes_contiguous(size);
    return self;
}

std::tuple<Tensor, Tensor> native_dropout_cuda(const Tensor& input, double p) {
    if (p < 0 || p >= 1) {
        TP_THROW(ValueError, "native_dropout: p must be in [0, 1)");
    }
    Tensor mask(static_cast<std::vector<int64_t>>(input.shape()), DType::Bool,
                input.device());
    Tensor out(static_cast<std::vector<int64_t>>(input.shape()), input.dtype(),
               input.device());
    const int64_t n = input.numel();
    if (n == 0) return {std::move(out), std::move(mask)};
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64 &&
        input.dtype() != DType::Float16 && input.dtype() != DType::BFloat16) {
        TP_THROW(NotImplementedError,
                 "dropout is only supported on floating point tensors");
    }

    // Same counter-reservation policy as RandomKernels.cu with unroll=4:
    // results do not depend on how threads are laid out across the grid.
    dim3 block(kDropoutBlockSize);
    dim3 grid(static_cast<uint32_t>((n + kDropoutBlockSize - 1) / kDropoutBlockSize));
    grid.x = std::min(
        grid.x,
        static_cast<uint32_t>(
            deviceAttribute(cudaDevAttrMultiProcessorCount) *
            (deviceAttribute(cudaDevAttrMaxThreadsPerMultiProcessor) /
             kDropoutBlockSize)));
    const uint64_t counter_offset =
        ((static_cast<uint64_t>(n) - 1) / (kDropoutBlockSize * grid.x * 4) + 1) *
        kMaxGeneratorOffsetsPerCall;
    auto philox_args = philox_engine_inputs(counter_offset);

    const float pf = static_cast<float>(p);
    const float scale = static_cast<float>(1.0 / (1.0 - p));
    bool* mask_data = mask.data_ptr<bool>();

    auto launch = [&](auto type_tag) {
        using scalar_t = decltype(type_tag);
        native_dropout_kernel<scalar_t><<<grid, block>>>(
            n, philox_args.first, philox_args.second, pf, scale,
            input.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), mask_data);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            TP_THROW(RuntimeError,
                     std::string("CUDA Error: ") + cudaGetErrorString(error));
        }
    };

    switch (input.dtype()) {
        case DType::Float32: launch(0.0f); break;
        case DType::Float64: launch(0.0); break;
        case DType::Float16: launch(Half(0.0f)); break;
        case DType::BFloat16: launch(BFloat16(0.0f)); break;
        default: break;
    }
    return {std::move(out), std::move(mask)};
}

} // namespace cuda
} // namespace tensorplay
