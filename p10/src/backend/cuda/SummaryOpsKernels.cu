#include "ReduceKernels.cuh"
#include "CudaDispatchHelpers.cuh"

namespace tensorplay {
namespace cuda {

namespace {


template <typename T>
__device__ inline T reduce_empty_value() {
    if constexpr (std::is_same<T, float>::value ||
                  std::is_same<T, double>::value) {
        return static_cast<T>(std::numeric_limits<double>::quiet_NaN());
    } else if constexpr (std::is_same<T, Half>::value ||
                         std::is_same<T, BFloat16>::value) {
        return T(static_cast<float>(std::numeric_limits<float>::quiet_NaN()));
    } else {
        return std::numeric_limits<T>::lowest();
    }
}


// The selection kernels take their index type as a parameter: a radix pass
// carries counters, ballots and offsets in it, and the 64-bit form of all three
// is markedly more expensive than the 32-bit one.
template <typename T, typename index_t>
__global__ void nanmedian_select_flat_kernel(
        index_t n, const T* input, T* result) {
    __shared__ index_t radix_smem[32];
    __shared__ index_t nan_count;
    if (threadIdx.x == 0) nan_count = 0;
    __syncthreads();

    index_t local_nan_count = 0;
    for (index_t i = static_cast<index_t>(threadIdx.x); i < n;
         i += static_cast<index_t>(blockDim.x)) {
        local_nan_count += reduce_value_is_nan(input[i]) ? 1 : 0;
    }
    if (local_nan_count != 0) atomicAdd(&nan_count, local_nan_count);
    __syncthreads();

    const index_t valid = n - static_cast<index_t>(nan_count);
    if (valid == 0) {
        if (threadIdx.x == 0) result[0] = reduce_empty_value<T>();
        return;
    }
    const index_t k = (valid - 1) / 2 + 1;
    T median = static_cast<T>(0);
    topk_detail::topk_radix_select<T, index_t>(
        input, k, false, n, static_cast<index_t>(1), radix_smem, &median);
    if (threadIdx.x == 0) result[0] = median;
}


template <typename T, typename index_t>
__global__ void median_select_dim_kernel(
        index_t n_slices, index_t d_size, index_t inner, const T* input,
        T* values, int64_t* indices, bool ignore_nan) {
    const index_t si = static_cast<index_t>(blockIdx.x);
    if (si >= n_slices) return;
    __shared__ index_t radix_smem[32];
    __shared__ unsigned long long nan_count;
    __shared__ unsigned long long selected_index;
    if (threadIdx.x == 0) {
        nan_count = 0;
        selected_index = static_cast<unsigned long long>(d_size);
    }
    __syncthreads();

    const index_t outer_index = si / inner;
    const index_t inner_index = si % inner;
    const T* slice_input = input + outer_index * d_size * inner + inner_index;
    index_t local_nan_count = 0;
    for (index_t i = static_cast<index_t>(threadIdx.x); i < d_size;
         i += static_cast<index_t>(blockDim.x)) {
        local_nan_count += reduce_value_is_nan(slice_input[i * inner]) ? 1 : 0;
    }
    if (local_nan_count != 0) atomicAdd(&nan_count, local_nan_count);
    __syncthreads();

    const index_t valid = d_size - static_cast<index_t>(nan_count);
    if (ignore_nan && valid == 0) {
        if (threadIdx.x == 0) {
            values[si] = reduce_empty_value<T>();
            indices[si] = 0;
        }
        return;
    }
    const index_t k = !ignore_nan && nan_count != 0
        ? d_size
        : (valid - 1) / 2 + 1;
    T median = static_cast<T>(0);
    topk_detail::topk_radix_select<T, index_t>(
        slice_input, k, false, d_size, inner, radix_smem, &median);
    for (index_t i = static_cast<index_t>(threadIdx.x); i < d_size;
         i += static_cast<index_t>(blockDim.x)) {
        const T value = slice_input[i * inner];
        if (value == median ||
            (reduce_value_is_nan(value) && reduce_value_is_nan(median))) {
            atomicMin(&selected_index, static_cast<unsigned long long>(i));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        values[si] = median;
        indices[si] = static_cast<int64_t>(selected_index);
    }
}


template <typename T, typename index_t>
__global__ void kthvalue_select_kernel(
        index_t n_slices, index_t d_size, index_t inner, index_t k,
        const T* input, T* values, int64_t* indices) {
    const index_t si = static_cast<index_t>(blockIdx.x);
    if (si >= n_slices) return;
    __shared__ index_t radix_smem[32];
    __shared__ unsigned long long selected_index;
    if (threadIdx.x == 0) {
        selected_index = static_cast<unsigned long long>(d_size);
    }
    __syncthreads();

    const index_t outer_index = si / inner;
    const index_t inner_index = si % inner;
    const T* slice_input = input + outer_index * d_size * inner + inner_index;
    T selected = static_cast<T>(0);
    topk_detail::topk_radix_select<T, index_t>(
        slice_input, k, false, d_size, inner, radix_smem, &selected);
    for (index_t i = static_cast<index_t>(threadIdx.x); i < d_size;
         i += static_cast<index_t>(blockDim.x)) {
        const T value = slice_input[i * inner];
        if (value == selected ||
            (reduce_value_is_nan(value) && reduce_value_is_nan(selected))) {
            atomicMin(&selected_index, static_cast<unsigned long long>(i));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        values[si] = selected;
        indices[si] = static_cast<int64_t>(selected_index);
    }
}


template <typename T>
__device__ inline bool mode_value_equal(T lhs, T rhs) {
    return !(lhs < rhs) && !(rhs < lhs);
}


__global__ void mode_bool_kernel(
        int64_t n_slices, int64_t d_size, int64_t inner,
        const bool* input, bool* values, int64_t* indices) {
    const int64_t si = static_cast<int64_t>(blockIdx.x);
    if (si >= n_slices) return;
    __shared__ unsigned long long true_count;
    __shared__ unsigned long long selected_index;
    if (threadIdx.x == 0) {
        true_count = 0;
        selected_index = static_cast<unsigned long long>(d_size);
    }
    __syncthreads();

    const int64_t outer_index = si / inner;
    const int64_t inner_index = si % inner;
    const bool* slice_input = input + outer_index * d_size * inner + inner_index;
    for (uint64_t i = static_cast<uint64_t>(threadIdx.x);
         i < static_cast<uint64_t>(d_size);
         i += static_cast<uint64_t>(blockDim.x)) {
        if (slice_input[i * inner]) atomicAdd(&true_count, 1ull);
    }
    __syncthreads();

    const bool mode = true_count >
        static_cast<unsigned long long>(d_size) - true_count;
    for (uint64_t i = static_cast<uint64_t>(threadIdx.x);
         i < static_cast<uint64_t>(d_size);
         i += static_cast<uint64_t>(blockDim.x)) {
        if (slice_input[i * inner] == mode) {
            atomicMin(&selected_index, static_cast<unsigned long long>(i));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        values[si] = mode;
        indices[si] = static_cast<int64_t>(selected_index);
    }
}


template <typename T>
__device__ __forceinline__ bool mode_value_less(T lhs, T rhs) {
    const bool lhs_nan = reduce_value_is_nan(lhs);
    const bool rhs_nan = reduce_value_is_nan(rhs);
    if (lhs_nan != rhs_nan) return !lhs_nan;
    if (lhs_nan) return false;
    return lhs < rhs;
}


template <typename T>
__device__ __forceinline__ void mode_bitonic_swap(
        T& lhs, bool& lhs_valid, T& rhs, bool& rhs_valid, bool direction) {
    const bool should_swap =
        (mode_value_less(lhs, rhs) && lhs_valid) || !rhs_valid;
    if (should_swap == direction) {
        T value = lhs;
        lhs = rhs;
        rhs = value;
        const bool valid = lhs_valid;
        lhs_valid = rhs_valid;
        rhs_valid = valid;
    }
}


template <typename T, unsigned int Power2Size>
__device__ inline void mode_bitonic_sort(T* values, bool* valid) {
    for (unsigned int size = 2; size < Power2Size; size <<= 1) {
        const bool direction = (threadIdx.x & (size / 2)) != 0;
        for (unsigned int stride = size / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            const unsigned int position =
                2 * threadIdx.x - (threadIdx.x & (stride - 1));
            mode_bitonic_swap(
                values[position], valid[position],
                values[position + stride], valid[position + stride], direction);
        }
    }
    for (unsigned int stride = Power2Size / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        const unsigned int position =
            2 * threadIdx.x - (threadIdx.x & (stride - 1));
        mode_bitonic_swap(
            values[position], valid[position],
            values[position + stride], valid[position + stride], false);
    }
    __syncthreads();
}


template <typename T, unsigned int Power2Size>
__global__ void mode_fused_kernel(
        int64_t n_slices, int64_t d_size, int64_t inner,
        const T* input, T* values, int64_t* indices) {
    const int64_t si = static_cast<int64_t>(blockIdx.x);
    if (si >= n_slices) return;
    extern __shared__ unsigned char storage[];
    T* sorted = reinterpret_cast<T*>(storage);
    bool* valid = reinterpret_cast<bool*>(sorted + Power2Size);
    const int64_t outer_index = si / inner;
    const int64_t inner_index = si % inner;
    const T* slice_input = input + outer_index * d_size * inner + inner_index;

    const unsigned int second = blockDim.x + threadIdx.x;
    if (threadIdx.x < Power2Size) {
        valid[threadIdx.x] = threadIdx.x < static_cast<unsigned int>(d_size);
        sorted[threadIdx.x] = valid[threadIdx.x]
            ? slice_input[static_cast<int64_t>(threadIdx.x) * inner]
            : static_cast<T>(0);
    }
    if (second < Power2Size) {
        valid[second] = second < static_cast<unsigned int>(d_size);
        sorted[second] = valid[second]
            ? slice_input[static_cast<int64_t>(second) * inner]
            : static_cast<T>(0);
    }
    __syncthreads();
    mode_bitonic_sort<T, Power2Size>(sorted, valid);

    __shared__ T mode;
    __shared__ unsigned long long mode_index;
    if (threadIdx.x == 0) {
        int best_count = 0;
        int run_count = 0;
        unsigned int best_position = 0;
        for (unsigned int i = 0; i < static_cast<unsigned int>(d_size); ++i) {
            const bool same = i > 0 &&
                mode_value_equal(sorted[i], sorted[i - 1]);
            run_count = same ? run_count + 1 : 1;
            if (run_count > best_count) {
                best_count = run_count;
                best_position = i;
            }
        }
        mode = sorted[best_position];
        mode_index = static_cast<unsigned long long>(d_size);
    }
    __syncthreads();
    for (uint64_t i = static_cast<uint64_t>(threadIdx.x);
         i < static_cast<uint64_t>(d_size);
         i += static_cast<uint64_t>(blockDim.x)) {
        if (mode_value_equal(slice_input[i * inner], mode)) {
            atomicMin(&mode_index, static_cast<unsigned long long>(i));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        values[si] = mode;
        indices[si] = static_cast<int64_t>(mode_index);
    }
}


template <typename T>
void launch_mode_fused(
        int64_t n_slices, int64_t d_size, int64_t inner,
        const Tensor& input, Tensor& values, Tensor& indices) {
    const int64_t power = d_size <= 32 ? 32 : d_size <= 128 ? 128
                                      : d_size <= 1024 ? 1024 : 2048;
    const dim3 grid(static_cast<unsigned>(n_slices));
    auto stream = getCurrentCUDAStream().stream();
    switch (power) {
        case 32:
            mode_fused_kernel<T, 32><<<grid, 16, sizeof(T) * 32 + sizeof(bool) * 32, stream>>>(
                n_slices, d_size, inner, input.data_ptr<T>(),
                values.data_ptr<T>(), indices.data_ptr<int64_t>());
            break;
        case 128:
            mode_fused_kernel<T, 128><<<grid, 64, sizeof(T) * 128 + sizeof(bool) * 128, stream>>>(
                n_slices, d_size, inner, input.data_ptr<T>(),
                values.data_ptr<T>(), indices.data_ptr<int64_t>());
            break;
        case 1024:
            mode_fused_kernel<T, 1024><<<grid, 512, sizeof(T) * 1024 + sizeof(bool) * 1024, stream>>>(
                n_slices, d_size, inner, input.data_ptr<T>(),
                values.data_ptr<T>(), indices.data_ptr<int64_t>());
            break;
        default:
            mode_fused_kernel<T, 2048><<<grid, 1024, sizeof(T) * 2048 + sizeof(bool) * 2048, stream>>>(
                n_slices, d_size, inner, input.data_ptr<T>(),
                values.data_ptr<T>(), indices.data_ptr<int64_t>());
            break;
    }
}


template <typename T>
__global__ void mode_from_sorted_kernel(int64_t n_slices, int64_t d_size,
                                         int64_t inner, const T* sorted,
                                         const int64_t* sorted_indices,
                                         T* values, int64_t* indices) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t outer_index = si / inner;
        int64_t inner_index = si % inner;
        const T* source = sorted + outer_index * d_size * inner + inner_index;
        const int64_t* source_indices =
            sorted_indices + outer_index * d_size * inner + inner_index;
        T best_value = source[0];
        int64_t best_count = 0;
        int64_t best_index = source_indices[0];
        int64_t run_count = 0;
        int64_t run_index = source_indices[0];
        for (int64_t j = 0; j < d_size; ++j) {
            const T value = source[j * inner];
            const int64_t original_index = source_indices[j * inner];
            if (j > 0 && mode_value_equal(value, source[(j - 1) * inner])) {
                ++run_count;
                if (original_index < run_index) run_index = original_index;
            } else {
                run_count = 1;
                run_index = original_index;
            }
            if (run_count > best_count) {
                best_count = run_count;
                best_value = value;
                best_index = run_index;
            }
        }
        values[si] = best_value;
        indices[si] = best_index;
    }
}


Tensor nanmedian_cuda(const Tensor& self) {
    DType out_dt = isFloatingType(self.dtype()) ? self.dtype() : DType::Int64;
    DType work_dt = out_dt;
    if (isFloat8Type(work_dt)) work_dt = DType::Float32;
    if (self.numel() == 0) {
        Tensor result = Tensor::zeros({}, out_dt, self.device());
        if (isFloatingType(out_dt)) {
            return result.fill_(Scalar(std::numeric_limits<double>::quiet_NaN()));
        }
        return result.fill_(Scalar(std::numeric_limits<int64_t>::lowest()));
    }
    Tensor input = self.to(work_dt).contiguous().reshape({self.numel()});
    Tensor result = Tensor::empty({}, work_dt, self.device());
    auto stream = getCurrentCUDAStream().stream();
    const bool flat32 =
        input.numel() <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
#define TP_NANMEDIAN_FLAT_CASE(ctype, name_) \
    case DType::name_: \
        if (flat32) \
            nanmedian_select_flat_kernel<ctype, int32_t><<< \
                1, selection_threads(input.numel()), 0, stream>>>( \
                    static_cast<int32_t>(input.numel()), input.data_ptr<ctype>(), \
                    result.data_ptr<ctype>()); \
        else \
            nanmedian_select_flat_kernel<ctype, int64_t><<<1, selection_threads(input.numel()), 0, stream>>>( \
            input.numel(), input.data_ptr<ctype>(), result.data_ptr<ctype>()); \
        break;
    switch (work_dt) {
        TP_NANMEDIAN_FLAT_CASE(int64_t, Int64)
        TP_NANMEDIAN_FLAT_CASE(float, Float32)
        TP_NANMEDIAN_FLAT_CASE(double, Float64)
        TP_NANMEDIAN_FLAT_CASE(Half, Float16)
        TP_NANMEDIAN_FLAT_CASE(BFloat16, BFloat16)
        default: TP_THROW(TypeError, "nanmedian: unsupported dtype");
    }
#undef TP_NANMEDIAN_FLAT_CASE
    CUDA_CHECK(cudaGetLastError());
    return work_dt == out_dt ? result : result.to(out_dt);
}


std::tuple<Tensor, Tensor> nanmedian_dim_cuda(const Tensor& self, int64_t dim,
                                              bool keepdim) {
    const int64_t nd = self.dim();
    TP_CHECK(nd > 0,
             "nanmedian(): expects a tensor with at least one dimension");
    dim = wrap_dim(dim, nd);
    TP_CHECK(isFloatingType(self.dtype()),
             "nanmedian(): only floating point dtypes are supported");
    TP_CHECK(self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16 ||
                 self.dtype() == DType::Float32 || self.dtype() == DType::Float64,
             "nanmedian(): unsupported dtype ", toString(self.dtype()));
    Tensor input = self.contiguous();
    const int64_t d_size = input.size(dim);
    TP_CHECK(d_size > 0, "nanmedian(): Expected reduction dim ", dim,
             " to have non-zero size");
    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(shape_of(input), dim, outer, inner);
    std::vector<int64_t> out_shape = shape_of(input);
    out_shape[dim] = keepdim ? 1 : 0;
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    Tensor values = Tensor::empty(out_shape, input.dtype(), input.device());
    Tensor indices = Tensor::empty(out_shape, DType::Int64, input.device());
    const int64_t slices = outer * inner;
    if (slices == 0) return {values, indices};
    auto stream = getCurrentCUDAStream().stream();
    const bool sel32 = input.numel() <=
        static_cast<int64_t>(std::numeric_limits<int32_t>::max());
#define TP_NANMEDIAN_DIM_CASE(ctype, name_) \
    case DType::name_: \
        if (sel32) \
            median_select_dim_kernel<ctype, int32_t><<< \
                dim3(static_cast<unsigned>(slices)), selection_threads(d_size), \
                0, stream>>>( \
                static_cast<int32_t>(slices), static_cast<int32_t>(d_size), \
                static_cast<int32_t>(inner), input.data_ptr<ctype>(), \
                values.data_ptr<ctype>(), indices.data_ptr<int64_t>(), true); \
        else \
            median_select_dim_kernel<ctype, int64_t><<< \
            dim3(static_cast<unsigned>(slices)), selection_threads(d_size), 0, stream>>>( \
            slices, d_size, inner, input.data_ptr<ctype>(), values.data_ptr<ctype>(), \
            indices.data_ptr<int64_t>(), true); \
        break;
    switch (input.dtype()) {
        TP_NANMEDIAN_DIM_CASE(Half, Float16)
        TP_NANMEDIAN_DIM_CASE(BFloat16, BFloat16)
        TP_NANMEDIAN_DIM_CASE(float, Float32)
        TP_NANMEDIAN_DIM_CASE(double, Float64)
        default: TP_THROW(TypeError, "nanmedian: unsupported dtype");
    }
#undef TP_NANMEDIAN_DIM_CASE
    CUDA_CHECK(cudaGetLastError());
    return {values, indices};
}


std::tuple<Tensor, Tensor> mode_cuda(const Tensor& self, int64_t dim, bool keepdim) {
    int64_t nd = self.dim();
    if (nd == 0) {
        if (dim != 0 && dim != -1) {
            TP_THROW(IndexError,
                     "Dimension out of range for scalar mode input: ", dim);
        }
        Tensor values = Tensor::empty({}, self.dtype(), self.device());
        Tensor indices = Tensor::zeros({}, DType::Int64, self.device());
        values.copy_(self);
        return {values, indices};
    }
    dim = wrap_dim(dim, nd);
    Tensor input = self.contiguous();
    int64_t d_size = input.size(dim);
    TP_CHECK(d_size > 0,
             "mode: expected reduction dimension to have non-zero size");
    int64_t outer = 1, inner = 1;
    outer_inner(shape_of(input), dim, outer, inner);
    std::vector<int64_t> out_shape = shape_of(input);
    out_shape[dim] = keepdim ? 1 : 0;
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    Tensor values = Tensor::empty(out_shape, input.dtype(), input.device());
    Tensor indices = Tensor::empty(out_shape, DType::Int64, input.device());
    const int64_t slices = outer * inner;
    if (slices == 0) return {values, indices};
    if (input.dtype() == DType::Bool) {
        auto stream = getCurrentCUDAStream().stream();
        mode_bool_kernel<<<
            dim3(static_cast<unsigned>(slices)), selection_threads(d_size), 0, stream>>>(
            slices, d_size, inner, input.data_ptr<bool>(), values.data_ptr<bool>(),
            indices.data_ptr<int64_t>());
        CUDA_CHECK(cudaGetLastError());
        return {values, indices};
    }
    if (inner == 1 && d_size >= 2 && d_size <= 2048) {
        switch (input.dtype()) {
#define TP_MODE_FUSED_CASE(ctype, name_) \
            case DType::name_: \
                launch_mode_fused<ctype>( \
                    slices, d_size, inner, input, values, indices); \
                break;
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_MODE_FUSED_CASE)
#undef TP_MODE_FUSED_CASE
            default:
                TP_THROW(TypeError, "mode: unsupported dtype");
        }
        CUDA_CHECK(cudaGetLastError());
        return {values, indices};
    }
    auto sorted_result = sort_cuda(input, dim, false);
    Tensor sorted = std::get<0>(sorted_result);
    Tensor sorted_indices = std::get<1>(sorted_result);
    auto stream = getCurrentCUDAStream().stream();
    const bool sel32 = input.numel() <=
        static_cast<int64_t>(std::numeric_limits<int32_t>::max());
#define TP_MODE_DEVICE_CASE(ctype, name_) \
    case DType::name_: \
        mode_from_sorted_kernel<ctype><<<make_grid(slices), kThreads, 0, stream>>>( \
            slices, d_size, inner, sorted.data_ptr<ctype>(), \
            sorted_indices.data_ptr<int64_t>(), values.data_ptr<ctype>(), \
            indices.data_ptr<int64_t>()); \
        break;
    switch (input.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_MODE_DEVICE_CASE)
        default: TP_THROW(TypeError, "mode: unsupported dtype");
    }
#undef TP_MODE_DEVICE_CASE
    CUDA_CHECK(cudaGetLastError());
    return {values, indices};
}


std::tuple<Tensor, Tensor> kthvalue_cuda(const Tensor& self, int64_t k, int64_t dim,
                                         bool keepdim) {
    Tensor input = self.contiguous();
    int64_t nd = input.dim();
    if (nd == 0) {
        if (dim != 0 && dim != -1) {
            TP_THROW(IndexError,
                     "Dimension out of range for scalar kthvalue input: ", dim);
        }
        if (k != 1) {
            TP_THROW(RuntimeError,
                     "kthvalue(): selected number k out of range for dim 0");
        }
        Tensor values = Tensor::empty({}, input.dtype(), input.device());
        Tensor indices = Tensor::zeros({}, DType::Int64, input.device());
        values.copy_(input);
        return {values, indices};
    }
    dim = wrap_dim(dim, nd);
    int64_t d_size = input.size(dim);
    if (k < 1 || k > d_size)
        TP_THROW(RuntimeError, "kthvalue(): selected number k out of range for dim ", dim);
    std::vector<int64_t> out_shape = shape_of(input);
    out_shape[dim] = keepdim ? 1 : 0;
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    Tensor values_out = Tensor::empty(out_shape, input.dtype(), input.device());
    Tensor indices_out = Tensor::empty(out_shape, DType::Int64, input.device());
    if (input.numel() == 0) return {values_out, indices_out};
    if (input.dtype() == DType::Bool) {
        Tensor selected_values;
        Tensor selected_indices;
        std::tie(selected_values, selected_indices) =
            sort_cuda(input, dim, false);
        Tensor values = selected_values.select(dim, k - 1);
        Tensor indices = selected_indices.select(dim, k - 1);
        if (keepdim) {
            values = values.unsqueeze(dim);
            indices = indices.unsqueeze(dim);
        }
        return {values, indices};
    }

    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(shape_of(input), dim, outer, inner);
    const int64_t slices = outer * inner;
    auto stream = getCurrentCUDAStream().stream();
    const bool sel32 = input.numel() <=
        static_cast<int64_t>(std::numeric_limits<int32_t>::max());
#define TP_KTHVALUE_SELECT_CASE(ctype, name_) \
    case DType::name_: \
        if (sel32) \
            kthvalue_select_kernel<ctype, int32_t><<< \
                dim3(static_cast<unsigned>(slices)), selection_threads(d_size), \
                0, stream>>>( \
                static_cast<int32_t>(slices), static_cast<int32_t>(d_size), \
                static_cast<int32_t>(inner), static_cast<int32_t>(k), \
                input.data_ptr<ctype>(), values_out.data_ptr<ctype>(), \
                indices_out.data_ptr<int64_t>()); \
        else \
            kthvalue_select_kernel<ctype, int64_t><<< \
            dim3(static_cast<unsigned>(slices)), selection_threads(d_size), 0, stream>>>( \
            slices, d_size, inner, k, input.data_ptr<ctype>(), \
            values_out.data_ptr<ctype>(), indices_out.data_ptr<int64_t>()); \
        break;
    switch (input.dtype()) {
        TP_KTHVALUE_SELECT_CASE(uint8_t, UInt8)
        TP_KTHVALUE_SELECT_CASE(int8_t, Int8)
        TP_KTHVALUE_SELECT_CASE(int16_t, Int16)
        TP_KTHVALUE_SELECT_CASE(int32_t, Int32)
        TP_KTHVALUE_SELECT_CASE(int64_t, Int64)
        TP_KTHVALUE_SELECT_CASE(uint16_t, UInt16)
        TP_KTHVALUE_SELECT_CASE(uint32_t, UInt32)
        TP_KTHVALUE_SELECT_CASE(uint64_t, UInt64)
        TP_KTHVALUE_SELECT_CASE(Half, Float16)
        TP_KTHVALUE_SELECT_CASE(BFloat16, BFloat16)
        TP_KTHVALUE_SELECT_CASE(float, Float32)
        TP_KTHVALUE_SELECT_CASE(double, Float64)
#undef TP_KTHVALUE_SELECT_CASE
        default:
            TP_THROW(NotImplementedError, "kthvalue: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
    return {values_out, indices_out};
}


std::tuple<Tensor, Tensor> median_dim_cuda(const Tensor& self, int64_t dim,
                                           bool keepdim) {
    Tensor input = self.contiguous();
    const int64_t nd = input.dim();
    if (nd == 0) return kthvalue_cuda(input, 1, dim, keepdim);
    dim = wrap_dim(dim, nd);
    const int64_t d_size = input.size(dim);
    const int64_t k = (d_size + 1) / 2;
    const bool selection_supported =
        isIntegralType(input.dtype()) ||
        input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16 ||
        input.dtype() == DType::Float32 || input.dtype() == DType::Float64;
    if (!selection_supported) return kthvalue_cuda(input, k, dim, keepdim);

    std::vector<int64_t> out_shape = shape_of(input);
    out_shape[dim] = keepdim ? 1 : 0;
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    Tensor values = Tensor::empty(out_shape, input.dtype(), input.device());
    Tensor indices = Tensor::empty(out_shape, DType::Int64, input.device());
    if (input.numel() == 0) return {values, indices};

    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(shape_of(input), dim, outer, inner);
    const int64_t slices = outer * inner;
    auto stream = getCurrentCUDAStream().stream();
    const bool sel32 = input.numel() <=
        static_cast<int64_t>(std::numeric_limits<int32_t>::max());
#define TP_MEDIAN_DIM_CASE(ctype, name_) \
    case DType::name_: \
        if (sel32) \
            median_select_dim_kernel<ctype, int32_t><<< \
                dim3(static_cast<unsigned>(slices)), selection_threads(d_size), \
                0, stream>>>( \
                static_cast<int32_t>(slices), static_cast<int32_t>(d_size), \
                static_cast<int32_t>(inner), input.data_ptr<ctype>(), \
                values.data_ptr<ctype>(), \
                indices.data_ptr<int64_t>(), false); \
        else \
            median_select_dim_kernel<ctype, int64_t><<< \
            dim3(static_cast<unsigned>(slices)), selection_threads(d_size), 0, stream>>>( \
            slices, d_size, inner, input.data_ptr<ctype>(), values.data_ptr<ctype>(), \
            indices.data_ptr<int64_t>(), false); \
        break;
    switch (input.dtype()) {
        TP_MEDIAN_DIM_CASE(uint8_t, UInt8)
        TP_MEDIAN_DIM_CASE(int8_t, Int8)
        TP_MEDIAN_DIM_CASE(int16_t, Int16)
        TP_MEDIAN_DIM_CASE(int32_t, Int32)
        TP_MEDIAN_DIM_CASE(int64_t, Int64)
        TP_MEDIAN_DIM_CASE(uint16_t, UInt16)
        TP_MEDIAN_DIM_CASE(uint32_t, UInt32)
        TP_MEDIAN_DIM_CASE(uint64_t, UInt64)
        TP_MEDIAN_DIM_CASE(Half, Float16)
        TP_MEDIAN_DIM_CASE(BFloat16, BFloat16)
        TP_MEDIAN_DIM_CASE(float, Float32)
        TP_MEDIAN_DIM_CASE(double, Float64)
        default: return kthvalue_cuda(input, k, dim, keepdim);
    }
#undef TP_MEDIAN_DIM_CASE
    CUDA_CHECK(cudaGetLastError());
    return {values, indices};
}


std::tuple<Tensor, Tensor> interop_kthvalue_values_cuda(const Tensor& self, int64_t k, int64_t dim, bool keepdim,
              Tensor& values, Tensor& indices) {
        {
            auto __tp_result = kthvalue_cuda(self, k, dim, keepdim);
            write_out(values, std::get<0>(__tp_result));
            write_out(indices, std::get<1>(__tp_result));
        }
        return {values, indices};

}


std::tuple<Tensor, Tensor> interop_median_dim_values_cuda(
    const Tensor& self, int64_t dim, bool keepdim, Tensor& values,
    Tensor& indices) {
    {
        auto __tp_result = median_dim_cuda(self, dim, keepdim);
        write_out(values, std::get<0>(__tp_result));
        write_out(indices, std::get<1>(__tp_result));
    }
    return {values, indices};
}


std::tuple<Tensor, Tensor> interop_nanmedian_dim_values_cuda(
    const Tensor& self, int64_t dim, bool keepdim, Tensor& values,
    Tensor& indices) {
    {
        auto __tp_result = nanmedian_dim_cuda(self, dim, keepdim);
        write_out(values, std::get<0>(__tp_result));
        write_out(indices, std::get<1>(__tp_result));
    }
    return {values, indices};
}

Tensor logsumexp_cuda2(const Tensor& self, int64_t dim, bool keepdim) {
    if (!isFloatingType(self.dtype()) &&
        !isIntegralType(self.dtype(), true))
        TP_THROW(RuntimeError, "logsumexp(): Expected floating point type");
    int64_t nd = self.dim();
    if (nd == 0) {
        // A zero-dim tensor reduces along dim 0 or -1 as one value along one
        // axis, and one value is its own log-sum-exp.
        wrap_dim(dim, 1);
        return isIntegralType(self.dtype(), true)
            ? self.to(globalContext().defaultDType())
            : self.clone();
    }
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous();
    if (isIntegralType(sc.dtype(), true)) {
        sc = sc.to(globalContext().defaultDType());
    }
    const std::vector<int64_t> dims{dim};
    if (sc.numel() == 0) {
        return sum_dim_kernel(sc.exp(), dims, keepdim, sc.dtype()).log();
    }
    Tensor max_keep = amax_dim_kernel(sc, dims, true);
    const Scalar infinity(std::numeric_limits<double>::infinity());
    Tensor safe_max = Tensor::where(max_keep.abs().eq(infinity), Scalar(0), max_keep);
    Tensor summed = sum_dim_kernel((sc - safe_max).exp(), dims, keepdim, sc.dtype());
    Tensor result = summed.log();
    return result + (keepdim ? safe_max : safe_max.squeeze(dim));
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, SummaryOpsKernels) {
    m.impl("logsumexp", logsumexp_cuda2);
    m.impl("nanmedian", nanmedian_cuda);
    m.impl("nanmedian.dim", nanmedian_dim_cuda);
    m.impl("nanmedian.dim_values", interop_nanmedian_dim_values_cuda);
    m.impl("mode", mode_cuda);
    m.impl("kthvalue", kthvalue_cuda);
    m.impl("kthvalue.values", interop_kthvalue_values_cuda);
    m.impl("median.dim", median_dim_cuda);
    m.impl("median.dim_values", interop_median_dim_values_cuda);
}

namespace {


// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

int embedding_summary_multiprocessor_count() {
    static thread_local int cached_device = -1;
    static thread_local int cached_count = 1;
    const int device = currentDevice();
    if (device != cached_device) {
        int value = 0;
        if (cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount,
                                   device) != cudaSuccess) {
            value = 1;
        }
        cached_count = value > 0 ? value : 1;
        cached_device = device;
    }
    return cached_count;
}

// A range that collapsed to a single value is widened by one in the compute
// type the binning runs in, so the value still lands inside the histogram.
// Widening in double instead would move the edges the binning never sees.
void histc_expand_constant_range(DType dtype, double& lo, double& hi) {
    if (dtype == DType::Float64) {
        lo -= 1.0;
        hi += 1.0;
        return;
    }
    lo = static_cast<double>(static_cast<float>(lo) - 1.0f);
    hi = static_cast<double>(static_cast<float>(hi) + 1.0f);
}


// ---------------------------------------------------------------------------
// histc: bin edges are equally spaced over [min, max]; values outside the
// range are dropped, the rightmost edge is inclusive.
// ---------------------------------------------------------------------------

// Bin edges span [lo, hi] and the rightmost edge is inclusive, so a value at
// hi maps to the last bin.  NaN fails both bounds and drops.  The edge test
// runs in the input's own compute precision -- a double-precision edge test
// would send values sitting on a boundary to a different bin than the same
// input gets at single precision.
template <typename ComputeT>
__device__ __forceinline__ int histc_bin(ComputeT value, ComputeT lo,
                                        ComputeT hi, int bins) {
    if (!(value >= lo) || !(value <= hi)) return -1;
    int b = static_cast<int>((value - lo) * bins / (hi - lo));
    return b == bins ? bins - 1 : b;
}

// Staged histogram: every block keeps its own bin counters in shared memory and
// publishes them once at the end.  Straight to global memory, the atomics of a
// narrow histogram all land on the same handful of addresses and serialize --
// the shared copy turns those into per-block races that resolve in shared
// memory, and leaves one global atomic per bin per block.
template <typename InT, typename ComputeT, typename CountT>
__global__ void histc_shared_kernel(const InT* input, int64_t n, ComputeT lo,
                                    ComputeT hi, int bins,
                                    unsigned long long* counts) {
    extern __shared__ unsigned char histc_smem[];
    CountT* local = reinterpret_cast<CountT*>(histc_smem);
    for (int i = threadIdx.x; i < bins; i += blockDim.x) local[i] = CountT(0);
    __syncthreads();

    const int64_t stride = int64_t(gridDim.x) * blockDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        const int b = histc_bin(static_cast<ComputeT>(input[i]), lo, hi, bins);
        if (b >= 0) atomicAdd(local + b, CountT(1));
    }
    __syncthreads();

    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        const CountT v = local[i];
        if (v != CountT(0)) atomicAdd(counts + i, static_cast<unsigned long long>(v));
    }
}

// Fallback for bin counts that do not fit in shared memory: one pass over the
// input with every vote going straight to the output.
template <typename InT, typename ComputeT>
__global__ void histc_count_kernel(const InT* input, int64_t n, ComputeT lo,
                                   ComputeT hi, int bins,
                                   unsigned long long* counts) {
    const int64_t stride = int64_t(gridDim.x) * blockDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        const int b = histc_bin(static_cast<ComputeT>(input[i]), lo, hi, bins);
        if (b >= 0) atomicAdd(counts + b, 1ull);
    }
}


Tensor interop_histc_cuda(const Tensor& self, int64_t bins, const Scalar& min, const Scalar& max) {
    if (bins <= 0) TP_THROW(RuntimeError, "histc(): bins must be positive");
    // Integer inputs compute in double precision and report counts in the
    // input dtype; that keeps the CUDA contract wider than the CPU one.
    const bool promote_to_f64 = !isFloatingType(self.dtype());
    if (promote_to_f64 && !isIntegralType(self.dtype(), /*includeBool=*/false)) {
        TP_THROW(TypeError, "histc(): expected a floating-point tensor, got ",
                 toString(self.dtype()));
    }
    double lo = min.toDouble();
    double hi = max.toDouble();
    if (lo == hi && self.numel() > 0) {
        auto extrema = ops::aminmax(self);
        lo = std::get<0>(extrema).item().toDouble();
        hi = std::get<1>(extrema).item().toDouble();
    }
    if (lo == hi) {
        histc_expand_constant_range(self.dtype(), lo, hi);
    }
    if (!std::isfinite(lo) || !std::isfinite(hi)) {
        TP_THROW(RuntimeError, "histc: range of [", lo, ", ", hi,
                 "] is not finite");
    }
    if (!(lo < hi)) TP_THROW(RuntimeError, "histc: max must be larger than min");

    Tensor counts = Tensor::empty({bins}, DType::Int64, self.device());
    cudaStream_t stream = getCurrentCUDAStream().stream();
    const cudaError_t hist_memset = cudaMemsetAsync(
        counts.data_ptr(), 0, sizeof(int64_t) * static_cast<size_t>(bins),
        stream);
    if (hist_memset != cudaSuccess) {
        TP_THROW(RuntimeError,
                 std::string("CUDA Error: ") + cudaGetErrorString(hist_memset));
    }
    const Tensor flat = self.reshape({-1});
    const int64_t n = flat.numel();
    if (n > 0) {

        unsigned long long* counts_ptr =
            static_cast<unsigned long long*>(counts.data_ptr());
        const void* in_ptr = flat.data_ptr();
        // Staging costs one bin counter per bin per block and one global atomic
        // per bin per block at the end, so the grid is sized to keep that
        // publishing traffic a small fraction of the shared-memory votes: aim
        // for kHistcGlobalVoteRatio votes per published count, but never fewer
        // blocks than there are multiprocessors, and never so many that a block
        // runs dry before the next wave.
        constexpr int64_t kHistcGlobalVoteRatio = 64;
        constexpr int64_t kHistcMinBlocksPerSM = 4;
        const int threads = 256;
        const size_t shared_bytes = static_cast<size_t>(bins) * sizeof(unsigned int);
        const size_t smem_limit = 32 * 1024;
        const bool staged = shared_bytes <= smem_limit;
        const int sms = embedding_summary_multiprocessor_count();
        const int64_t ceiling_blocks = (n + threads - 1) / threads;
        int64_t grid = ceiling_blocks;
        if (staged) {
            // Every block publishes one atomic per bin, so the block count is
            // what decides how much of the traffic lands in global memory:
            // hold it to a small fraction of the votes, and let the floor keep
            // enough blocks resident to cover the machine.
            const int64_t publish_bound =
                n / (kHistcGlobalVoteRatio * static_cast<int64_t>(bins));
            grid = std::max<int64_t>(publish_bound,
                                     sms * kHistcMinBlocksPerSM);
            grid = std::min<int64_t>(grid, ceiling_blocks);
        } else {
            grid = std::min<int64_t>(grid, 4096);
        }
        const int grid_i = static_cast<int>(std::max<int64_t>(grid, 1));
        // Single-precision inputs bin in single precision, doubles in double.
        const bool wide = self.dtype() == DType::Float64;
        const float lo_f = static_cast<float>(lo);
        const float hi_f = static_cast<float>(hi);
#define TP_HISTC_CASE(ctype, name)                                            \
    case DType::name:                                                         \
        if (staged) {                                                         \
            if (wide) {                                                       \
                histc_shared_kernel<ctype, double, unsigned int>               \
                    <<<grid_i, threads, shared_bytes, stream>>>(                 \
                        static_cast<const ctype*>(in_ptr), n, lo, hi, bins,    \
                        counts_ptr);                                          \
            } else {                                                          \
                histc_shared_kernel<ctype, float, unsigned int>                \
                    <<<grid_i, threads, shared_bytes, stream>>>(                 \
                        static_cast<const ctype*>(in_ptr), n, lo_f, hi_f,      \
                        static_cast<int>(bins), counts_ptr);                  \
            }                                                                  \
        } else {                                                              \
            if (wide) {                                                       \
                histc_count_kernel<ctype, double><<<grid_i, threads, 0, stream>>>( \
                    static_cast<const ctype*>(in_ptr), n, lo, hi, bins,        \
                    counts_ptr);                                              \
            } else {                                                          \
                histc_count_kernel<ctype, float><<<grid_i, threads, 0, stream>>>(  \
                    static_cast<const ctype*>(in_ptr), n, lo_f, hi_f,          \
                    static_cast<int>(bins), counts_ptr);                      \
            }                                                                  \
        }                                                                      \
        break;
        switch (self.dtype()) {
            TP_HISTC_CASE(double, Float64)
            TP_HISTC_CASE(float, Float32)
            TP_HISTC_CASE(tensorplay::Half, Float16)
            TP_HISTC_CASE(tensorplay::BFloat16, BFloat16)
            TP_HISTC_CASE(int64_t, Int64)
            TP_HISTC_CASE(int32_t, Int32)
            TP_HISTC_CASE(int16_t, Int16)
            TP_HISTC_CASE(int8_t, Int8)
            TP_HISTC_CASE(uint8_t, UInt8)
            TP_HISTC_CASE(uint16_t, UInt16)
            TP_HISTC_CASE(uint32_t, UInt32)
            TP_HISTC_CASE(uint64_t, UInt64)
            default:
                TP_THROW(NotImplementedError,
                         "histc(): unsupported dtype '" +
                             std::string(toString(self.dtype())) + "'");
        }
#undef TP_HISTC_CASE
        const cudaError_t hist_err = cudaGetLastError();
        if (hist_err != cudaSuccess) {
            TP_THROW(RuntimeError,
                     std::string("CUDA Error: ") + cudaGetErrorString(hist_err));
        }
    }
    return counts.to(self.dtype());
}


Tensor& interop_histc_out_cuda(const Tensor& self, int64_t bins, const Scalar& min,
                               const Scalar& max, Tensor& out) {
    if (out.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "histc(): out tensor must have the same dtype as the input");
    }
    if (out.device() != self.device()) {
        TP_THROW(DeviceMismatchError,
                 "histc(): out tensor must be on the same device as the input");
    }
    Tensor r = interop_histc_cuda(self, bins, min, max);
    out.resize_(static_cast<std::vector<int64_t>>(r.shape()));
    out.copy_(r);
    return out;
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, SummaryOpsHistc) {
    // histogram / diagonal index builders
    m.impl("histc", interop_histc_cuda);
    m.impl("histc.out", interop_histc_out_cuda);
}

} // namespace cuda
} // namespace tensorplay
