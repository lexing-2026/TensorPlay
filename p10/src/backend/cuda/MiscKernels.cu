// Misc CUDA kernels: meshgrid / roll / diff / masked_fill / one_hot / glu.
//
// These follow the CPU composites in cpu/MiscKernels.cpp; every primitive
// invoked (slice/view/expand/cat/sigmoid/where/eq) is itself dispatched to the

#include "Tensor.h"
#include "CUDARuntime.h"
#include "CUDAReduce.cuh"
#include "Dispatcher.h"
#include "Utils.h"
#include "CUDAGenerator.h"
#include "CUDALoops.cuh"
#include <curand_kernel.h>
#include <algorithm>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cuda {

// Defined below the registration table.
Tensor& resize__cuda(Tensor& self, const std::vector<int64_t>& size);
std::tuple<Tensor, Tensor> native_dropout_cuda(const Tensor& input, double p,
                                             std::optional<bool> train);
Tensor native_dropout_backward_cuda(const Tensor& grad_output, const Tensor& mask, double scale);
Tensor native_dropout_double_backward_cuda(const Tensor& ggI, const Tensor& grad, const Tensor& mask, double scale);
std::tuple<Tensor, Tensor> native_alpha_dropout_cuda(const Tensor& input, double p);
Tensor alpha_dropout_backward_cuda(const Tensor& grad, const Tensor& mask, double p);
std::tuple<Tensor, Tensor> native_feature_dropout_cuda(const Tensor& input, double p);
Tensor feature_dropout_backward_cuda(const Tensor& grad, const Tensor& mask, double p);
Tensor trapezoid_cuda(const Tensor& y, const std::optional<Tensor>& x, const Scalar& dx, int64_t dim);
Tensor cumulative_trapezoid_cuda(const Tensor& y, const std::optional<Tensor>& x, const Scalar& dx, int64_t dim);
Tensor trapezoid_backward_cuda(const Tensor& grad, const std::optional<Tensor>& x,
                               const std::vector<int64_t>& ysizes, const Scalar& dx,
                               int64_t dim);
Tensor cumulative_trapezoid_backward_cuda(const Tensor& grad, const std::optional<Tensor>& x,
                                          const Scalar& dx, int64_t dim);
Tensor cov_cuda(const Tensor& self, int64_t correction,
                const std::optional<Tensor>& fweights_opt,
                const std::optional<Tensor>& aweights_opt);
Tensor corrcoef_cuda(const Tensor& self);
Tensor cov_backward_cuda(const Tensor& grad, const Tensor& self, int64_t correction,
                         const std::optional<Tensor>& fweights_opt,
                         const std::optional<Tensor>& aweights_opt);
Tensor corrcoef_backward_cuda(const Tensor& grad, const Tensor& self);
bool is_set_to_cuda(const Tensor& self, const Tensor& other);

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
    Tensor result = self;
    n = n > self.size(dim) ? self.size(dim) : n;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t out_len = result.size(dim) - 1;
        result = result.slice(dim, 1, out_len + 1) - result.slice(dim, 0, out_len);
    }
    return result;
}

Tensor diff_cuda(const Tensor& self, int64_t n, int64_t dim, const std::optional<Tensor>& prepend_opt, const std::optional<Tensor>& append_opt) {
    const Tensor prepend = prepend_opt.has_value() ? *prepend_opt : Tensor();
    const Tensor append = append_opt.has_value() ? *append_opt : Tensor();
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

namespace {

enum class HashMode { XorSum = 0 };

template <typename InputT>
struct XorHashOps {
    using acc_type = unsigned long long;

    __device__ static acc_type value_bits(InputT value) {
        if constexpr (std::is_same_v<InputT, double>) {
            return static_cast<acc_type>(__double_as_longlong(value));
        } else if constexpr (std::is_same_v<InputT, float> ||
                             std::is_same_v<InputT, Half> ||
                             std::is_same_v<InputT, BFloat16>) {
            return static_cast<acc_type>(__double_as_longlong(
                static_cast<double>(value)));
        } else {
            return static_cast<acc_type>(value);
        }
    }

    __device__ acc_type reduce(acc_type acc, InputT value, int64_t) const {
        return acc ^ value_bits(value);
    }
    __device__ acc_type combine(acc_type a, acc_type b) const {
        return a ^ b;
    }
    __device__ uint64_t project(acc_type value) const {
        return static_cast<uint64_t>(value);
    }
};

template <typename InputT>
void launch_hash_tensor_reduce(TensorIterator& iter) {
    const auto config = reduction::make_reduce_config<InputT,
                                                       unsigned long long,
                                                       uint64_t>(iter);
    XorHashOps<InputT> ops;
    if (config.input_vec_size == 8) {
        reduction::launch_reduce<InputT, unsigned long long, uint64_t,
                                 XorHashOps<InputT>, 4, 8>(
            iter, ops, 0ull);
    } else if (config.input_vec_size == 4) {
        reduction::launch_reduce<InputT, unsigned long long, uint64_t,
                                 XorHashOps<InputT>, 4, 4>(
            iter, ops, 0ull);
    } else {
        reduction::launch_reduce<InputT, unsigned long long, uint64_t,
                                 XorHashOps<InputT>, 4, 1>(
            iter, ops, 0ull);
    }
}

std::vector<int64_t> hash_output_shape(const Tensor& input,
                                       const std::vector<bool>& mask,
                                       bool keepdim) {
    std::vector<int64_t> shape;
    const auto input_shape = static_cast<std::vector<int64_t>>(input.shape());
    shape.reserve(input_shape.size());
    for (size_t dim = 0; dim < input_shape.size(); ++dim) {
        if (mask[dim]) {
            if (keepdim) shape.push_back(1);
        } else {
            shape.push_back(input_shape[dim]);
        }
    }
    return shape;
}

Tensor hash_reduction_view(const Tensor& result, const Tensor& input,
                           const std::vector<bool>& mask, bool keepdim) {
    if (input.dim() == 0) return result;
    const auto input_shape = static_cast<std::vector<int64_t>>(input.shape());
    const auto result_strides = static_cast<std::vector<int64_t>>(result.strides());
    std::vector<int64_t> strides(input_shape.size(), 0);
    size_t result_dim = 0;
    for (size_t dim = 0; dim < input_shape.size(); ++dim) {
        if (!mask[dim]) {
            strides[dim] = keepdim ? result_strides[dim]
                                   : result_strides[result_dim++];
        }
    }
    return result.as_strided(input_shape, strides, std::nullopt);
}

void hash_tensor_check_cuda(const Tensor& input,
                            const std::vector<int64_t>& dims,
                            int64_t mode) {
    if (mode != static_cast<int64_t>(HashMode::XorSum)) {
        TP_THROW(RuntimeError, "Unknown hash_tensor mode: ", mode);
    }
    if (input.numel() == 0) {
        TP_CHECK(!dims.empty(),
                 "hash_tensor: Expected reduction dim to be specified for "
                 "input.numel() == 0. Specify the reduction dim with the "
                 "'dim' argument.");
    }
    for (int64_t dim : dims) {
        if (input.dim() == 0) {
            TP_CHECK(dim == 0 || dim == -1,
                     "hash_tensor: Expected reduction dim -1 or 0 for scalar "
                     "but got ", dim);
        } else {
            const int64_t wrapped = dim < 0 ? dim + input.dim() : dim;
            TP_CHECK(wrapped >= 0 && wrapped < input.dim(),
                     "hash_tensor: dimension out of range");
            TP_CHECK(input.size(wrapped) != 0,
                     "hash_tensor: Expected reduction dim ", dim,
                     " to have non-zero size.");
        }
    }
}

void hash_tensor_into_cuda(const Tensor& input,
                           const std::vector<int64_t>& dims,
                           bool keepdim, int64_t mode, Tensor& result) {
    hash_tensor_check_cuda(input, dims, mode);
    if (result.dtype() != DType::UInt64) {
        TP_THROW(RuntimeError,
                 "hash_tensor: expected the result to have dtype uint64");
    }

    const int64_t ndim = input.dim();
    std::vector<bool> mask(static_cast<size_t>(ndim), false);
    std::vector<int64_t> wrapped_dims;
    if (dims.empty()) {
        wrapped_dims.reserve(static_cast<size_t>(ndim));
        for (int64_t dim = 0; dim < ndim; ++dim) wrapped_dims.push_back(dim);
    } else {
        wrapped_dims.reserve(dims.size());
        for (int64_t dim : dims) {
            wrapped_dims.push_back(dim < 0 ? dim + ndim : dim);
        }
    }
    for (int64_t dim : wrapped_dims) {
        if (dim >= 0 && dim < ndim) mask[static_cast<size_t>(dim)] = true;
    }

    if (input.numel() == 0) {
        result.fill_(Scalar(static_cast<uint64_t>(0)));
        return;
    }

    Tensor viewed = hash_reduction_view(result, input, mask, keepdim);
    TensorIterator iter = TensorIterator::reduce_op(viewed, input);
#define TP_HASH_CUDA_CASE(ctype, name) \
    case DType::name: \
        launch_hash_tensor_reduce<ctype>(iter); \
        break;
    switch (input.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_HASH_CUDA_CASE)
        default:
            TP_THROW(NotImplementedError,
                     "hash_tensor: unsupported input dtype");
    }
#undef TP_HASH_CUDA_CASE
}

Tensor hash_tensor_cuda(const Tensor& input, const std::vector<int64_t>& dims,
                        bool keepdim, int64_t mode) {
    hash_tensor_check_cuda(input, dims, mode);
    const int64_t ndim = input.dim();
    std::vector<bool> mask(static_cast<size_t>(ndim), false);
    if (dims.empty()) {
        for (int64_t dim = 0; dim < ndim; ++dim) mask[static_cast<size_t>(dim)] = true;
    } else {
        for (int64_t dim : dims) {
            const int64_t wrapped = dim < 0 ? dim + ndim : dim;
            if (wrapped >= 0 && wrapped < ndim) mask[static_cast<size_t>(wrapped)] = true;
        }
    }
    Tensor result = Tensor::empty(hash_output_shape(input, mask, keepdim),
                                  DType::UInt64, input.device());
    hash_tensor_into_cuda(input, dims, keepdim, mode, result);
    return result;
}

Tensor& hash_tensor_out_cuda(const Tensor& input,
                             const std::vector<int64_t>& dims,
                             bool keepdim, int64_t mode, Tensor& result) {
    if (!result.defined()) {
        result = hash_tensor_cuda(input, dims, keepdim, mode);
        return result;
    }
    hash_tensor_check_cuda(input, dims, mode);
    const int64_t ndim = input.dim();
    std::vector<bool> mask(static_cast<size_t>(ndim), false);
    if (dims.empty()) {
        for (int64_t dim = 0; dim < ndim; ++dim) mask[static_cast<size_t>(dim)] = true;
    } else {
        for (int64_t dim : dims) {
            const int64_t wrapped = dim < 0 ? dim + ndim : dim;
            if (wrapped >= 0 && wrapped < ndim) mask[static_cast<size_t>(wrapped)] = true;
        }
    }
    TP_CHECK(static_cast<std::vector<int64_t>>(result.shape()) ==
                 hash_output_shape(input, mask, keepdim),
             "hash_tensor.out: the result shape must match the input reduced "
             "over the requested dims");
    hash_tensor_into_cuda(input, dims, keepdim, mode, result);
    return result;
}

} // anonymous namespace

namespace {

// out[i] = a * sigmoid(b) with a and b read from the two halves of one axis.
__device__ __forceinline__ float glu_exp(float v) { return ::expf(v); }
__device__ __forceinline__ double glu_exp(double v) { return ::exp(v); }

template <typename T, typename acc_t, typename index_t>
__global__ void glu_fused_kernel(index_t n, index_t block, const T* src, T* out) {
    const index_t li = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (li >= n) return;
    const index_t a_off = li + (li / block) * block;
    const acc_t a = static_cast<acc_t>(src[a_off]);
    const acc_t b = static_cast<acc_t>(src[a_off + block]);
    const acc_t one = acc_t(1);
    out[li] = static_cast<T>(a * (one / (one + glu_exp(-b))));
}

// One pass for both halves of the gradient: the first half gets the gate
// gradient, the second half the value gradient, and both are read from the
// same pair of input offsets.
template <typename T, typename acc_t, typename index_t>
__global__ void glu_backward_fused_kernel(index_t n, index_t block,
                                          const T* grad, const T* src,
                                          T* out) {
    const index_t li = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (li >= n) return;
    const index_t a_off = li + (li / block) * block;
    const acc_t a = static_cast<acc_t>(src[a_off]);
    const acc_t b = static_cast<acc_t>(src[a_off + block]);
    const acc_t g = static_cast<acc_t>(grad[li]);
    const acc_t one = acc_t(1);
    const acc_t sig = one / (one + glu_exp(-b));
    out[a_off] = static_cast<T>(g * sig);
    out[a_off + block] = static_cast<T>(g * a * sig * (one - sig));
}

}  // namespace

Tensor glu_cuda(const Tensor& self, int64_t dim) {
    if (self.dim() == 0) TP_THROW(RuntimeError, "glu does not support 0-dimensional tensors");
    const int64_t d = wrap_dim_local(dim, self.dim());
    const int64_t nIn = self.size(d);
    if (nIn % 2 != 0) {
        TP_THROW(RuntimeError, "Halving dimension must be even, but dimension " + std::to_string(d) +
                                   " is size " + std::to_string(nIn));
    }
    const int64_t half = nIn / 2;
    // A contiguous input lets one pass do the whole thing: the output index is
    // the same offset in both halves, so each element is read once and written
    // once.  Splitting it into a sigmoid pass and a multiply pass moves the
    // data three times instead of once.
    if (self.is_contiguous() &&
        (self.dtype() == DType::Float32 || self.dtype() == DType::Float64 ||
         self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16)) {
        int64_t inner = 1;
        for (int64_t k = d + 1; k < self.dim(); ++k) inner *= self.size(k);
        // Distance between the two halves of one split axis, which is also the
        // length of an output row.  Splitting an inner axis interleaves the
        // halves, so the input offset of the first half is the output offset
        // plus one whole block for every preceding output row.
        const int64_t block_len = half * inner;
        auto out_shape = static_cast<std::vector<int64_t>>(self.shape());
        out_shape[static_cast<size_t>(d)] = half;
        Tensor out = Tensor::empty(out_shape, self.dtype(), self.device());
        const int64_t n = out.numel();
        if (n == 0) return out;
        const auto stream = getCurrentCUDAStream().stream();
        constexpr int64_t kGluThreads = 256;
        const unsigned blocks = static_cast<unsigned>((n + kGluThreads - 1) / kGluThreads);
        const dim3 block(kGluThreads);
#define TP_GLU(name_, ctype, acc_) \
    case DType::name_: { \
        const ctype* source = self.data_ptr<ctype>(); \
        ctype* dest = out.data_ptr<ctype>(); \
        if (n <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()) && \
            block_len <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) { \
            glu_fused_kernel<ctype, acc_, int32_t><<<blocks, block, 0, stream>>>( \
                static_cast<int32_t>(n), static_cast<int32_t>(block_len), \
                source, dest); \
        } else { \
            glu_fused_kernel<ctype, acc_, int64_t><<<blocks, block, 0, stream>>>( \
                n, block_len, source, dest); \
        } \
        break; \
    }
        switch (self.dtype()) {
            TP_GLU(Float32, float, float)
            TP_GLU(Float64, double, double)
            TP_GLU(Float16, Half, float)
            TP_GLU(BFloat16, BFloat16, float)
            default: break;
        }
#undef TP_GLU
        checkCuda(cudaGetLastError(), "glu fused kernel");
        return out;
    }
    Tensor firstHalf = self.slice(d, 0, half);
    Tensor secondHalf = self.slice(d, half, nIn);
    return firstHalf * secondHalf.sigmoid();
}

Tensor glu_backward_cuda(const Tensor& grad_output, const Tensor& self, int64_t dim) {
    if (self.dim() == 0) TP_THROW(RuntimeError, "glu does not support 0-dimensional tensors");
    const int64_t d = wrap_dim_local(dim, self.dim());
    const int64_t nIn = self.size(d);
    if (nIn % 2 != 0) {
        TP_THROW(RuntimeError, "Halving dimension must be even, but dimension " + std::to_string(d) +
                                   " is size " + std::to_string(nIn));
    }
    const int64_t inputSize = nIn / 2;
    if (self.is_contiguous() && grad_output.is_contiguous() &&
        (self.dtype() == DType::Float32 || self.dtype() == DType::Float64 ||
         self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16) &&
        grad_output.dtype() == self.dtype()) {
        int64_t inner = 1;
        for (int64_t k = d + 1; k < self.dim(); ++k) inner *= self.size(k);
        const int64_t block_len = inputSize * inner;
        Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                                   self.dtype(), self.device());
        const int64_t n = inputSize * inner *
                          (self.numel() / (nIn * inner));
        if (n == 0) return out;
        const auto stream = getCurrentCUDAStream().stream();
        constexpr int64_t kGluThreads = 256;
        const unsigned blocks = static_cast<unsigned>((n + kGluThreads - 1) / kGluThreads);
        const dim3 block(kGluThreads);
#define TP_GLU_BWD(name_, ctype, acc_) \
    case DType::name_: { \
        if (n <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()) && \
            block_len <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) { \
            glu_backward_fused_kernel<ctype, acc_, int32_t> \
                <<<blocks, block, 0, stream>>>( \
                    static_cast<int32_t>(n), static_cast<int32_t>(block_len), \
                    grad_output.data_ptr<ctype>(), self.data_ptr<ctype>(), \
                    out.data_ptr<ctype>()); \
        } else { \
            glu_backward_fused_kernel<ctype, acc_, int64_t> \
                <<<blocks, block, 0, stream>>>( \
                    n, block_len, grad_output.data_ptr<ctype>(), \
                    self.data_ptr<ctype>(), out.data_ptr<ctype>()); \
        } \
        break; \
    }
        switch (self.dtype()) {
            TP_GLU_BWD(Float32, float, float)
            TP_GLU_BWD(Float64, double, double)
            TP_GLU_BWD(Float16, Half, float)
            TP_GLU_BWD(BFloat16, BFloat16, float)
            default: break;
        }
#undef TP_GLU_BWD
        checkCuda(cudaGetLastError(), "glu backward fused kernel");
        return out;
    }
    Tensor firstHalf = self.slice(d, 0, inputSize);
    Tensor secondHalf = self.slice(d, inputSize, nIn);
    Tensor sig_second = secondHalf.sigmoid();
    Tensor grad_input_first = grad_output * sig_second;
    Tensor grad_input_second = grad_output * firstHalf * sig_second * (1 - sig_second);
    return Tensor::cat({grad_input_first, grad_input_second}, d);
}

bool is_set_to_cuda(const Tensor& self, const Tensor& other) {
    const auto self_impl = self.unsafeGetTensorImpl();
    const auto other_impl = other.unsafeGetTensorImpl();
    if (!self_impl || !other_impl || !self_impl->has_storage() ||
        !other_impl->has_storage()) {
        return false;
    }
    if (!self_impl->storage().is_same(other_impl->storage()) ||
        self_impl->storage_offset() != other_impl->storage_offset() ||
        self.dim() != other.dim()) {
        return false;
    }
    for (int64_t dim = 0; dim < self.dim(); ++dim) {
        if (self.size(dim) != other.size(dim) ||
            self.stride(dim) != other.stride(dim)) {
            return false;
        }
    }
    return true;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, MiscKernels) {
    m.impl("diff", diff_cuda);
    m.impl("one_hot", one_hot_cuda);
    m.impl("glu", glu_cuda);
    m.impl("glu_backward", glu_backward_cuda);
    m.impl("resize_", resize__cuda);
    m.impl("native_dropout", native_dropout_cuda);
    m.impl("native_dropout_backward", native_dropout_backward_cuda);
    m.impl("native_dropout_double_backward", native_dropout_double_backward_cuda);
    m.impl("native_alpha_dropout", native_alpha_dropout_cuda);
    m.impl("_alpha_dropout_backward", alpha_dropout_backward_cuda);
    m.impl("native_feature_dropout", native_feature_dropout_cuda);
    m.impl("_feature_dropout_backward", feature_dropout_backward_cuda);
    m.impl("trapezoid", trapezoid_cuda);
    m.impl("cumulative_trapezoid", cumulative_trapezoid_cuda);
    m.impl("_trapezoid_backward", trapezoid_backward_cuda);
    m.impl("_cumulative_trapezoid_backward", cumulative_trapezoid_backward_cuda);
    m.impl("cov", cov_cuda);
    m.impl("corrcoef", corrcoef_cuda);
    m.impl("_cov_backward", cov_backward_cuda);
    m.impl("_corrcoef_backward", corrcoef_backward_cuda);
    m.impl("hash_tensor", hash_tensor_cuda);
    m.impl("hash_tensor.out", hash_tensor_out_cuda);
    m.impl("is_set_to", is_set_to_cuda);
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
// both the scaled output element and the bool keep-mask, using one
// philox counter discipline of RandomKernels.cu (offsets reserved host-side
// so results are launch-geometry independent).  Under CUDA graph capture the
// (seed, offset) pair is read from the graph's device buffer instead and
// refreshed before each replay (see CUDAGenerator.h).
template <typename scalar_t>
__global__ void native_dropout_kernel(int64_t numel, PhiloxCudaState philox_args,
                                      float p, float scale,
                                      const scalar_t* in, scalar_t* out,
                                      bool* mask) {
    uint64_t seed;
    uint64_t offset;
    if (philox_args.captured) {
        seed = *philox_args.seed_dev;
        offset = *philox_args.offset_dev + philox_args.offset_intragraph;
    } else {
        seed = philox_args.seed;
        offset = philox_args.offset;
    }
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

// saved by the forward is bool, so mask.to(grad.dtype()) folds into the
// masked load instead of materializing a cast. One thread per element on a
// full grid (the SM-capped grid-stride shape of the forward measurably
// leaves bandwidth on the table here; the forward needs the cap for its
// philox counter discipline, the backward has no RNG). Math runs in float
// (double only for float64 inputs): consumer GPUs throttle fp64 ALU.
template <typename scalar_t>
__global__ void native_dropout_backward_kernel(int64_t numel, double scale,
                                               const scalar_t* grad,
                                               const bool* mask,
                                               scalar_t* out) {
    using acc_t = typename std::conditional<std::is_same<scalar_t, double>::value,
                                            double, float>::type;
    const acc_t s = static_cast<acc_t>(scale);
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_threads = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += total_threads) {
        out[i] = mask[i] ? static_cast<scalar_t>(static_cast<acc_t>(grad[i]) * s)
                         : static_cast<scalar_t>(0.0);
    }
}

// Half/bfloat16 fast path: 8 elements per thread (uint4 data + uint2 mask).
// Scalar half loads only reach ~840 GB/s; 16B loads hit the ~950 GB/s
// roofline measured for the fp32 kernel.
template <typename scalar_t>
__global__ void native_dropout_backward_kernel_vec8(int64_t nvec, float scale,
                                                    const uint4* grad,
                                                    const uint2* mask,
                                                    uint4* out) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_threads = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < nvec; i += total_threads) {
        union { uint4 u; scalar_t v[8]; } g, o;
        g.u = grad[i];
        const uint2 m = mask[i];
        const unsigned mb[2] = {m.x, m.y};
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const bool keep = (mb[k >> 2] >> ((k & 3) * 8)) & 0xff;
            o.v[k] = keep ? static_cast<scalar_t>(static_cast<float>(g.v[k]) * scale)
                          : static_cast<scalar_t>(0.0);
        }
        out[i] = o.u;
    }
}

} // namespace

// Grow storage in place while preserving contents; shrinking is logical-only.
// Allocation and copying go through the CUDA caching allocator and
// copyAllocationBytes, so no explicit memcpy or stream handling is needed.
Tensor& resize__cuda(Tensor& self, const std::vector<int64_t>& size) {
    auto* impl = self.unsafeGetTensorImpl().get();
    if (static_cast<std::vector<int64_t>>(impl->sizes()) == size) {
        return self;
    }

    bool has_zero = false;
    for (int64_t s : size) {
        if (s < 0) {
            TP_THROW(ValueError, "resize_: negative sizes are not allowed");
        }
        has_zero = has_zero || s == 0;
    }
    size_t new_numel = has_zero ? 0 : 1;
    for (int64_t s : size) {
        if (new_numel != 0 &&
            static_cast<size_t>(s) >
                std::numeric_limits<size_t>::max() / new_numel) {
            TP_THROW(RuntimeError, "resize_: requested size is too large");
        }
        new_numel *= static_cast<size_t>(s);
    }
    if (impl->itemsize() != 0 &&
        new_numel > std::numeric_limits<size_t>::max() / impl->itemsize()) {
        TP_THROW(RuntimeError, "resize_: requested storage is too large");
    }
    const size_t new_bytes = new_numel * impl->itemsize();
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

std::tuple<Tensor, Tensor> native_dropout_cuda(const Tensor& input, double p,
                                             std::optional<bool> train) {
    if (p < 0 || p > 1) {
        TP_THROW(ValueError, "native_dropout: p must be in [0, 1]");
    }
    // Not training is not a probability of zero: nothing is dropped, so
    // everything that was there is still there, and the mask says every
    // element was kept.
    if (train.has_value() && !train.value()) {
        return std::make_tuple(input.clone(),
                               Tensor::ones_like(input, DType::Bool));
    }
    if (p == 1) {
        // Every element is dropped, and the scale that would undo the
        // dropping is undefined; the weights are all zero and the mask says
        // so, which is what a backward reading the mask needs.
        return std::make_tuple(Tensor::zeros_like(input),
                               Tensor::zeros_like(input, DType::Bool));
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
    auto philox_args = philox_cuda_state(counter_offset);

    const float pf = static_cast<float>(p);
    const float scale = static_cast<float>(1.0 / (1.0 - p));
    bool* mask_data = mask.data_ptr<bool>();

    auto launch = [&](auto type_tag) {
        using scalar_t = decltype(type_tag);
        native_dropout_kernel<scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            n, philox_args, pf, scale,
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

// the bool mask saved by the forward. One thread per element on a full grid
// (capped at the hardware grid.x limit, grid-stride above that); half/bf16
// take the 8-wide vectorized path when size and alignment allow.
Tensor native_dropout_backward_cuda(const Tensor& grad_output, const Tensor& mask, double scale) {
    if (mask.dtype() != DType::Bool) {
        // Use the CPU composite (grad * mask.to(grad.dtype()) * scale);
        // the fused fast path below assumes the bool mask from native_dropout.
        return grad_output * mask.to(grad_output.dtype()) * scale;
    }
    Tensor out(static_cast<std::vector<int64_t>>(grad_output.shape()), grad_output.dtype(),
               grad_output.device());
    const int64_t n = grad_output.numel();
    if (n == 0) return out;
    if (grad_output.dtype() != DType::Float32 && grad_output.dtype() != DType::Float64 &&
        grad_output.dtype() != DType::Float16 && grad_output.dtype() != DType::BFloat16) {
        TP_THROW(NotImplementedError,
                 "dropout is only supported on floating point tensors");
    }
    Tensor grad = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor mask_c = mask.is_contiguous() ? mask : mask.contiguous();

    dim3 block(kDropoutBlockSize);
    auto launch = [&](auto type_tag) {
        using scalar_t = decltype(type_tag);
        dim3 grid(static_cast<uint32_t>(
            std::min<int64_t>((n + kDropoutBlockSize - 1) / kDropoutBlockSize,
                              std::numeric_limits<uint32_t>::max())));
        native_dropout_backward_kernel<scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            n, scale, grad.data_ptr<scalar_t>(), mask_c.data_ptr<bool>(),
            out.data_ptr<scalar_t>());
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            TP_THROW(RuntimeError,
                     std::string("CUDA Error: ") + cudaGetErrorString(error));
        }
    };
    auto launch_vec8 = [&](auto type_tag) {
        using scalar_t = decltype(type_tag);
        const int64_t nvec = n / 8;
        dim3 grid(static_cast<uint32_t>(
            std::min<int64_t>((nvec + kDropoutBlockSize - 1) / kDropoutBlockSize,
                              std::numeric_limits<uint32_t>::max())));
        native_dropout_backward_kernel_vec8<scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            nvec, static_cast<float>(scale),
            reinterpret_cast<const uint4*>(grad.data_ptr<scalar_t>()),
            reinterpret_cast<const uint2*>(mask_c.data_ptr<bool>()),
            reinterpret_cast<uint4*>(out.data_ptr<scalar_t>()));
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            TP_THROW(RuntimeError,
                     std::string("CUDA Error: ") + cudaGetErrorString(error));
        }
    };

    const bool vec8_ok =
        n % 8 == 0 &&
        (reinterpret_cast<uintptr_t>(grad.data_ptr()) & 0xf) == 0 &&
        (reinterpret_cast<uintptr_t>(mask_c.data_ptr()) & 0x7) == 0;
    switch (grad_output.dtype()) {
        case DType::Float32: launch(0.0f); break;
        case DType::Float64: launch(0.0); break;
        case DType::Float16:
            if (vec8_ok) { launch_vec8(Half(0.0f)); } else { launch(Half(0.0f)); }
            break;
        case DType::BFloat16:
            if (vec8_ok) { launch_vec8(BFloat16(0.0f)); } else { launch(BFloat16(0.0f)); }
            break;
        default: break;
    }
    return out;
}

// The backward scales the incoming gradient by the kept-mask and the scale.
// That product is linear in the incoming gradient, so differentiating it a
// second time multiplies the incoming gradient by the same two factors: the
// same arithmetic on the same kinds of mask, which is what the first backward
// already writes.
Tensor native_dropout_double_backward_cuda(const Tensor& ggI, const Tensor& grad, const Tensor& mask, double scale) {
    return native_dropout_backward_cuda(ggI, mask, scale);
}


// ---------------------------------------------------------------------------
// Alpha / feature dropout (CUDA) — same dispatcher-composite shape as the
// CPU side: bernoulli_ noise via the registered RNG kernel, affine math via
// dispatched mul/add. bernoulli_/mul/div redispatch to their CUDA kernels.
// ---------------------------------------------------------------------------

namespace {

constexpr double kAlphaDropoutAlphaCuda = 1.7580993408473766;

double alpha_dropout_scale_cuda(double p) {
    return 1.0 / std::sqrt(
                      (kAlphaDropoutAlphaCuda * kAlphaDropoutAlphaCuda * p + 1.0) *
                      (1.0 - p));
}

Tensor bernoulli_mask_cuda(const Tensor& input,
                           const std::vector<int64_t>& shape,
                           double keep_prob) {
    Tensor noise = Tensor::full(shape, keep_prob, DType::Float32,
                                input.device());
    noise.bernoulli_(keep_prob, std::nullopt);
    return noise;
}

} // anonymous namespace

std::tuple<Tensor, Tensor> native_alpha_dropout_cuda(const Tensor& input, double p) {
    if (p < 0 || p >= 1) {
        TP_THROW(ValueError, "alpha_dropout: p must be in [0, 1)");
    }
    Tensor mask = bernoulli_mask_cuda(
        input, static_cast<std::vector<int64_t>>(input.shape()), 1.0 - p);
    const double a = alpha_dropout_scale_cuda(p);
    Tensor out = mask.mul(input.mul(a).add(kAlphaDropoutAlphaCuda * a))
                    .add(kAlphaDropoutAlphaCuda * a * (p - 1.0));
    return {std::move(out), std::move(mask)};
}

Tensor alpha_dropout_backward_cuda(const Tensor& grad, const Tensor& mask,
                                   double p) {
    const double a = alpha_dropout_scale_cuda(p);
    return grad.mul(mask).mul(a);
}

std::tuple<Tensor, Tensor> native_feature_dropout_cuda(const Tensor& input, double p) {
    if (p < 0 || p >= 1) {
        TP_THROW(ValueError, "feature_dropout: p must be in [0, 1)");
    }
    if (input.dim() < 2) {
        TP_THROW(RuntimeError, "feature_dropout requires at least 2D input");
    }
    std::vector<int64_t> mask_shape =
        static_cast<std::vector<int64_t>>(input.shape());
    for (int64_t d = 2; d < input.dim(); ++d) mask_shape[d] = 1;
    Tensor mask = bernoulli_mask_cuda(input, mask_shape, 1.0 - p);
    Tensor out = input.mul(mask).div(1.0 - p);
    return {std::move(out), std::move(mask)};
}

Tensor feature_dropout_backward_cuda(const Tensor& grad, const Tensor& mask,
                                     double p) {
    return grad.mul(mask).div(1.0 - p);
}


// ---------------------------------------------------------------------------
// expressed as dispatcher composites (narrow/add/mul/sum|cumsum). x=None
// selects uniform spacing dx. Backward rebuilds the per-element weights:
//   sum form:   w = dx * [0.5, 1, ..., 1, 0.5]
//   x form:     w = 0.5 * ([x1-x0] ++ diff(x) ++ [x_{n-1}-x_{n-2}])
//   cumulative: grad_y[j] = w[j] * sum_{k >= j} grad[k]
// ---------------------------------------------------------------------------

namespace {

int64_t trapz_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(IndexError, "Dimension out of range");
    }
    return dim;
}

// Uniform-spacing weight vector for a length-n axis.
Tensor uniform_weights(int64_t n, double dx, const Tensor& like) {
    if (n < 2) {
        return Tensor::full({n}, dx, like.dtype(), like.device());
    }
    return Tensor::cat({Tensor::full({1}, 0.5 * dx, like.dtype(),
                                      like.device()),
                        Tensor::full({std::max<int64_t>(n - 2, 0)}, dx,
                                     like.dtype(), like.device()),
                        Tensor::full({1}, 0.5 * dx, like.dtype(),
                                     like.device())},
                       0);
}

// Coordinate-difference weights along `dim` of x:
//   w = 0.5 * ([seg_0] ++ (seg_i + seg_{i+1}) ++ [seg_{n-2}]), length n.
Tensor onesided_weights(const Tensor& x, int64_t d) {
    const int64_t n = x.size(d);
    Tensor segs = x.narrow(d, 1, n - 1).sub(x.narrow(d, 0, n - 1));
    if (n == 2) {
        Tensor half = segs.mul(0.5);
        return Tensor::cat({half, half}, d);
    }
    Tensor inner = segs.narrow(d, 0, n - 2).add(segs.narrow(d, 1, n - 2));
    return Tensor::cat({segs.narrow(d, 0, 1), inner,
                        segs.narrow(d, n - 2, 1)}, d).mul(0.5);
}

Tensor segment_widths(const Tensor& x, int64_t d) {
    const int64_t n = x.size(d);
    return x.narrow(d, 1, n - 1).sub(x.narrow(d, 0, n - 1));
}

} // anonymous namespace

Tensor trapezoid_cuda(const Tensor& y, const std::optional<Tensor>& x_opt, const Scalar& dx_s, int64_t dim) {
    const double dx = dx_s.toDouble();
    const Tensor x = x_opt.value_or(Tensor());
    const int64_t d = trapz_dim(dim, y.dim());
    const int64_t n = y.size(d);
    if (n < 2) return y.narrow(d, 0, 0).sum(std::vector<int64_t>{d}, false);
    Tensor avg = y.narrow(d, 0, n - 1).add(y.narrow(d, 1, n - 1)).mul(0.5);
    if (x.defined()) {
        bool was_1d = x.dim() == 1;
        Tensor xb = was_1d && y.dim() > 1
            ? [&] {
                  std::vector<int64_t> view(y.dim(), 1);
                  view[d] = x.size(0);
                  return x.reshape(view).expand(
                      static_cast<std::vector<int64_t>>(y.shape()));
              }()
            : x;
        avg = avg.mul(xb.narrow(d, 1, n - 1).sub(xb.narrow(d, 0, n - 1)));
    } else {
        avg = avg.mul(dx);
    }
    return avg.sum(std::vector<int64_t>{d}, false);
}

Tensor cumulative_trapezoid_cuda(const Tensor& y, const std::optional<Tensor>& x_opt, const Scalar& dx_s,
                                int64_t dim) {
    const double dx = dx_s.toDouble();
    const Tensor x = x_opt.value_or(Tensor());
    const int64_t d = trapz_dim(dim, y.dim());
    const int64_t n = y.size(d);
    if (n < 2) return y.narrow(d, 0, 0);
    Tensor avg = y.narrow(d, 0, n - 1).add(y.narrow(d, 1, n - 1)).mul(0.5);
    if (x.defined()) {
        bool was_1d = x.dim() == 1;
        Tensor xb = was_1d && y.dim() > 1
            ? [&] {
                  std::vector<int64_t> view(y.dim(), 1);
                  view[d] = x.size(0);
                  return x.reshape(view).expand(
                      static_cast<std::vector<int64_t>>(y.shape()));
              }()
            : x;
        avg = avg.mul(xb.narrow(d, 1, n - 1).sub(xb.narrow(d, 0, n - 1)));
    } else {
        avg = avg.mul(dx);
    }
    return avg.cumsum(d);
}

namespace {

// grad (shape minus the reduced dim) times weights viewed along dim.
Tensor apply_sum_weights(const Tensor& grad, int64_t d, const Tensor& w1d) {
    std::vector<int64_t> view(grad.dim() + 1, 1);
    view[d] = w1d.numel();
    return grad.unsqueeze(d).mul(w1d.reshape(view));
}

} // anonymous namespace

Tensor trapezoid_backward_cuda(const Tensor& grad, const std::optional<Tensor>& x_opt,
                               const std::vector<int64_t>& ysizes, const Scalar& dx_s,
                               int64_t dim) {
    const double dx = dx_s.toDouble();
    const Tensor x = x_opt.value_or(Tensor());
    const int64_t ndim = static_cast<int64_t>(ysizes.size());
    const int64_t d = trapz_dim(dim, ndim);
    const int64_t n = ysizes[d];
    if (n < 2) return Tensor::zeros(ysizes, grad.dtype(), grad.device());
    if (!x.defined()) {
        return apply_sum_weights(grad, d,
                                 uniform_weights(n, dx, grad).to(grad.dtype()));
    }
    if (x.dim() == 1) {
        return apply_sum_weights(
            grad, d, onesided_weights(x, 0).to(grad.dtype()));
    }
    Tensor weights = onesided_weights(x, d).to(grad.dtype());
    return grad.unsqueeze(d).mul(weights);
}

Tensor cumulative_trapezoid_backward_cuda(const Tensor& grad,
                                          const std::optional<Tensor>& x_opt,
                                          const Scalar& dx_s, int64_t dim) {
    // Same dual-suffix-sum structure as the CPU kernel.
    const Tensor x = x_opt.value_or(Tensor());
    const double dx = dx_s.toDouble();
    const int64_t d = wrap_dim_local(dim, grad.dim());
    const int64_t m = grad.size(d);
    if (m == 0) {
        auto output_shape = static_cast<std::vector<int64_t>>(grad.shape());
        output_shape[d] = x.defined() && x.dim() != 1
            ? x.size(d)
            : (x.defined() ? x.numel() : 1);
        return Tensor::zeros(output_shape, grad.dtype(), grad.device());
    }
    const std::vector<int64_t> dv{d};
    Tensor acc = Tensor::flip(Tensor::flip(grad, dv).cumsum(d), dv);

    Tensor seg_w;
    if (x.defined()) {
        seg_w = segment_widths(x, x.dim() == 1 ? 0 : d).to(grad.dtype());
    } else {
        seg_w = Tensor::full({m}, dx, grad.dtype(), grad.device());
    }
    std::vector<int64_t> wview(grad.dim(), 1);
    wview[d] = m;
    Tensor ws = x.defined() && x.dim() != 1
        ? acc.mul(seg_w)
        : acc.mul(seg_w.reshape(wview));
    auto tail_shape = static_cast<std::vector<int64_t>>(ws.shape());
    tail_shape[d] = 1;
    Tensor zero_tail = Tensor::zeros(tail_shape, grad.dtype(), grad.device());
    // point j gets segment j (as left end) and segment j-1 (as right end)
    Tensor term_a = Tensor::cat({ws, zero_tail}, d);
    Tensor term_b = Tensor::cat({zero_tail, ws}, d);
    return term_a.add(term_b).mul(0.5);
}

// ---------------------------------------------------------------------------
// native/Correlation.cpp; every primitive invoked is itself dispatched to
// the CUDA backend.
// ---------------------------------------------------------------------------

namespace {

bool cov_scalar_true_cuda(const Tensor& t) {
    if (t.dtype() == DType::Bool) return t.item().to<bool>();
    if (isIntegralType(t.dtype(), /*includeBool=*/false)) {
        return t.item().to<int64_t>() != 0;
    }
    return t.item().toDouble() != 0.0;
}

Tensor cov_scalar_long_cuda(int64_t v, const Tensor& like) {
    return Tensor::full({}, v, DType::Int64, like.device());
}

struct CovParts {
    Tensor in;
    Tensor w;
    Tensor wsum;
    Tensor fact;
    int64_t num_observations;
    bool had_fw;
    bool had_aw;
};

CovParts cov_parts_cuda(const Tensor& self, int64_t correction,
                        const std::optional<Tensor>& fweights_opt,
                        const std::optional<Tensor>& aweights_opt) {
    constexpr int64_t OBSERVATIONS_DIM = 1;
    CovParts p;
    p.had_fw = fweights_opt.has_value();
    p.had_aw = aweights_opt.has_value();

    if (self.dim() > 2) {
        TP_THROW(RuntimeError,
                 "cov(): expected input to have two or fewer dimensions but got "
                 "an input with " + std::to_string(self.dim()) + " dimensions");
    }
    TP_CHECK_NOT_IMPLEMENTED(self.dtype() != DType::Bool,
                             "cov(): bool dtype is not supported for input");

    Tensor in = self.dim() < 2 ? self.view({1, -1}) : self;
    p.num_observations = in.size(OBSERVATIONS_DIM);

    Tensor w;
    if (p.had_fw) {
        const Tensor& fwv = *fweights_opt;
        if (fwv.dim() > 1) {
            TP_THROW(RuntimeError,
                     "cov(): expected fweights to have one or fewer dimensions but got "
                     "fweights with " + std::to_string(fwv.dim()) + " dimensions");
        }
        TP_CHECK(isIntegralType(fwv.dtype(), /*includeBool=*/false),
                 "cov(): expected fweights to have integral dtype but got fweights with dtype ",
                 static_cast<int>(fwv.dtype()));
        TP_CHECK(fwv.numel() == p.num_observations,
                 "cov(): expected fweights to have the same numel as there are observations "
                 "in the input but got ", fwv.numel(), " != ", p.num_observations);
        TP_CHECK(p.num_observations == 0 || fwv.min().item().toDouble() >= 0.0,
                 "cov(): fweights cannot be negative");
        w = fwv;
    }

    if (p.had_aw) {
        const Tensor& aw = *aweights_opt;
        if (aw.dim() > 1) {
            TP_THROW(RuntimeError,
                     "cov(): expected aweights to have one or fewer dimensions but got "
                     "aweights with " + std::to_string(aw.dim()) + " dimensions");
        }
        TP_CHECK(isFloatingType(aw.dtype()),
                 "cov(): expected aweights to have floating point dtype but got "
                 "aweights with dtype ", static_cast<int>(aw.dtype()));
        TP_CHECK(aw.numel() == p.num_observations,
                 "cov(): expected aweights to have the same numel as there are observations "
                 "in the input but got ", aw.numel(), " != ", p.num_observations);
        TP_CHECK(p.num_observations == 0 || aw.min().item().toDouble() >= 0.0,
                 "cov(): aweights cannot be negative");
        // product of frequencies (fweights) and reliability weights (aweights)
        w = w.defined() ? w.mul(aw) : aw;
    }
    p.w = w;

    p.wsum = w.defined()
        ? w.sum()
        : cov_scalar_long_cuda(p.num_observations, in);

    TP_CHECK(!w.defined() || cov_scalar_true_cuda(p.wsum),
             "cov(): weights sum to zero, can't be normalized");

    const Tensor avg =
        (w.defined() ? in.mul(w) : in)
            .sum(std::vector<int64_t>{OBSERVATIONS_DIM})
            .div(p.wsum);

    if (!w.defined()) {
        p.fact = cov_scalar_long_cuda(p.num_observations - correction, in);
    } else if (correction == 0) {
        p.fact = p.wsum;
    } else if (!p.had_aw) {
        p.fact = p.wsum.sub(Scalar(correction));
    } else if (!p.had_fw && p.num_observations == 1 && correction == 1) {
        p.fact = cov_scalar_long_cuda(0, in);
    } else {
        p.fact = p.wsum.sub(w.mul(*aweights_opt).sum().mul(correction).div(p.wsum));
    }

    if (p.fact.item().toDouble() <= 0.0) {
        p.fact.zero_();
    }

    if (p.num_observations == 1 && p.had_fw != p.had_aw) {
        in.zero_();
        if (isIntegralType(in.dtype(), false)) {
            in = in.to(DType::Float32);
        }
        p.in = in;
    } else {
        p.in = in.sub(avg.unsqueeze(1));
    }
    return p;
}

Tensor cov_matrix_from_cuda(const CovParts& p) {
    Tensor c = Tensor::mm(p.in, (p.w.defined() ? p.in.mul(p.w) : p.in).t());
    return c.div(p.fact);
}

Tensor cov_apply_grad_cuda(const Tensor& H, const CovParts& p, const Tensor& like) {
    const int64_t n = p.num_observations;
    Tensor GM = H.add(H.t()).mm(p.in).div(p.fact);
    if (p.w.defined()) {
        const Tensor wrow = p.w.reshape({1, n});
        GM = GM.mul(wrow);
        Tensor rowsum = GM.sum(std::vector<int64_t>{1}, true);
        // rowsum is taken over the *weighted* G_M above
        return GM.sub(rowsum.mul(wrow).div(p.wsum))
            .reshape(static_cast<std::vector<int64_t>>(like.shape()));
    }
    Tensor rowsum = GM.sum(std::vector<int64_t>{1}, true);
    Tensor inv_n = Tensor::full({}, static_cast<int64_t>(n), DType::Int64,
                                GM.device());
    return GM.sub(rowsum.div(inv_n))
        .reshape(static_cast<std::vector<int64_t>>(like.shape()));
}

} // anonymous namespace

Tensor cov_cuda(const Tensor& self, int64_t correction,
                const std::optional<Tensor>& fweights_opt,
                const std::optional<Tensor>& aweights_opt) {
    CovParts p = cov_parts_cuda(self, correction, fweights_opt, aweights_opt);
    return cov_matrix_from_cuda(p).squeeze();
}

Tensor corrcoef_cuda(const Tensor& self) {
    if (self.dim() > 2) {
        TP_THROW(RuntimeError,
                 "corrcoef(): expected input to have two or fewer dimensions but got "
                 "an input with " + std::to_string(self.dim()) + " dimensions");
    }
    Tensor c = cov_cuda(self, 1, std::nullopt, std::nullopt);
    if (c.dim() == 0) {
        return c.div(c);
    }
    const Tensor d = c.diagonal();
    const Tensor stddev = d.sqrt();
    c = c.div(stddev.view({-1, 1}));
    c = c.div(stddev.view({1, -1}));
    return c.clamp(Scalar(-1.0), Scalar(1.0));
}

Tensor cov_backward_cuda(const Tensor& grad, const Tensor& self, int64_t correction,
                         const std::optional<Tensor>& fweights_opt,
                         const std::optional<Tensor>& aweights_opt) {
    CovParts p = cov_parts_cuda(self, correction, fweights_opt, aweights_opt);
    if (p.num_observations == 1 && p.had_fw != p.had_aw) {
        return Tensor::zeros(static_cast<std::vector<int64_t>>(self.shape()),
                             grad.dtype(), self.device());
    }
    const int64_t k = p.in.size(0);
    return cov_apply_grad_cuda(grad.reshape({k, k}), p, self);
}

Tensor corrcoef_backward_cuda(const Tensor& grad, const Tensor& self) {
    if (self.dim() > 2) {
        TP_THROW(RuntimeError,
                 "corrcoef(): expected input to have two or fewer dimensions but got "
                 "an input with " + std::to_string(self.dim()) + " dimensions");
    }
    CovParts p = cov_parts_cuda(self, 1, std::nullopt, std::nullopt);
    if (p.num_observations == 1 && p.had_fw != p.had_aw) {
        return Tensor::zeros(static_cast<std::vector<int64_t>>(self.shape()),
                             grad.dtype(), self.device());
    }
    Tensor C = cov_matrix_from_cuda(p);
    const int64_t k = C.size(0);
    if (k == 1 && C.size(1) == 1) {
        return Tensor::zeros(static_cast<std::vector<int64_t>>(self.shape()),
                             grad.dtype(), self.device());
    }
    Tensor H = grad.reshape({k, k});
    const Tensor s = C.diagonal().sqrt();
    const Tensor R = C.div(s.view({-1, 1})).div(s.view({1, -1}));
    Tensor Hp = Tensor::where(R.gt(-1.0), H, Scalar(0));
    Hp = Tensor::where(R.lt(1.0), Hp, Scalar(0));
    // R = D^-1 C D^-1 with D = diag(s):
    //   dL/dC = K + diag(-g_i / (2 s_i^2)),  K = Hp / (s s^T),
    //   g_i = sum_j K_ij C_ij + sum_j K_ji C_ij  (R is sensitive to s_i
    //   through both its row and its column; C symmetric keeps C_ij shared)
    Tensor K = Hp.div(s.view({-1, 1})).div(s.view({1, -1}));
    Tensor crow = K.mul(C).sum(std::vector<int64_t>{1}, false);
    Tensor ccol = K.transpose(0, 1).mul(C).sum(std::vector<int64_t>{1}, false);
    Tensor coef = crow.add(ccol).div(s.mul(2)).div(s);
    Tensor diagm = Tensor::eye(k, k, K.dtype(), K.device())
                       .mul(coef.reshape({1, k}));
    Tensor GC = K.sub(diagm);
    return cov_apply_grad_cuda(GC, p, self);
}



} // namespace cuda
} // namespace tensorplay
