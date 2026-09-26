#include "ReduceKernels.cuh"
#include "CudaDispatchHelpers.cuh"

namespace tensorplay {
namespace cuda {

namespace {


inline int64_t wrap_scan_dim(int64_t dim, int64_t ndim) {
    if (ndim == 0) {
        if (dim == -1 || dim == 0) return 0;
        TP_THROW(IndexError,
                 "Dimension out of range for a scalar tensor (expected -1 or 0, but got ",
                 dim, ")");
    }
    return wrap_dim(dim, ndim);
}


// The direction is a template parameter rather than a kernel argument: the
// comparison sits in the innermost loop of the shared-memory scan, where a
// runtime flag costs a select per step.
template <typename T, bool kIsMax>
__device__ inline void cum_extremum_update(const T lhs, T& rhs,
                                           int64_t lhs_idx, int64_t& rhs_idx) {
    const bool lhs_nan = reduce_value_is_nan(lhs);
    const bool rhs_nan = reduce_value_is_nan(rhs);
    if (!rhs_nan && (lhs_nan || (kIsMax ? lhs > rhs : lhs < rhs))) {
        rhs = lhs;
        rhs_idx = lhs_idx;
    }
}


template <typename T, bool kIsMax>
__global__ void cummaxmin_scan_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                      const T* in, T* vals, int64_t* idxs) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* s2p = in + o * d_size * inner + in2;
        T* vp = vals + o * d_size * inner + in2;
        int64_t* ip = idxs + o * d_size * inner + in2;
        T best = s2p[0];
        int64_t bi = 0;
        vp[0] = best;
        ip[0] = 0;
        for (int64_t j = 1; j < d_size; ++j) {
            const T current = s2p[j * inner];
            double cur = static_cast<double>(current);
            double b = static_cast<double>(best);
            if (cur != cur || (b == b &&
                (kIsMax ? cur >= b : cur <= b))) {
                best = current;
                bi = j;
            }
            vp[j * inner] = best;
            ip[j * inner] = bi;
        }
    }
}


inline int scan_log_num_threads_x(int64_t num_rows, int64_t row_size) {
    int log_num_threads_x = 0;
    int log_num_threads_y = 0;
    while ((int64_t{1} << log_num_threads_x) < row_size) ++log_num_threads_x;
    while ((int64_t{1} << log_num_threads_y) < num_rows) ++log_num_threads_y;
    return std::clamp((9 + log_num_threads_x - log_num_threads_y) / 2, 4, 9);
}


template <typename T, bool kIsMax>
__global__ void cummaxmin_innermost_scan_kernel(
    int64_t num_rows, int64_t row_size, const T* in, T* vals, int64_t* idxs,
    T init) {
    alignas(sizeof(double)) extern __shared__ char buffer[];
    T* value_buffer = reinterpret_cast<T*>(buffer);
    int64_t* index_buffer = reinterpret_cast<int64_t*>(
        value_buffer + 2 * blockDim.x * blockDim.y);
    T* row_values = value_buffer + 2 * blockDim.x * threadIdx.y;
    int64_t* row_indices = index_buffer + 2 * blockDim.x * threadIdx.y;
    const int64_t row_width = static_cast<int64_t>(blockDim.x) * 2;

    for (int64_t block_row = static_cast<int64_t>(blockIdx.x) * blockDim.y;
         block_row < num_rows;
         block_row += static_cast<int64_t>(blockDim.y) * gridDim.x) {
        const int64_t row = block_row + threadIdx.y;
        const bool row_exists = row < num_rows;
        const T* row_input = row_exists ? in + row * row_size : in;
        T* row_output = row_exists ? vals + row * row_size : vals;
        int64_t* row_output_indices = row_exists ? idxs + row * row_size : idxs;
        T block_total = init;
        int64_t block_index = 0;
        for (int64_t block_col = 0; block_col < row_size; block_col += row_width) {
            const int64_t col1 = block_col + threadIdx.x;
            const int64_t col2 = block_col + blockDim.x + threadIdx.x;
            row_values[threadIdx.x] = row_exists && col1 < row_size ? row_input[col1] : init;
            row_values[blockDim.x + threadIdx.x] = row_exists && col2 < row_size
                ? row_input[col2] : init;
            row_indices[threadIdx.x] = col1 < row_size ? col1 : 0;
            row_indices[blockDim.x + threadIdx.x] = col2 < row_size ? col2 : 0;
            if (row_exists && threadIdx.x == 0) {
                cum_extremum_update<T, kIsMax>(block_total, row_values[0],
                                               block_index, row_indices[0]);
            }
            __syncthreads();

            for (int stride = 1; stride <= blockDim.x; stride <<= 1) {
                const int base = (threadIdx.x / stride) * (2 * stride) + stride;
                const int target = base + (threadIdx.x % stride);
                const int source = base - 1;
                cum_extremum_update<T, kIsMax>(row_values[source], row_values[target],
                                               row_indices[source], row_indices[target]);
                __syncthreads();
            }

            if (row_exists && col1 < row_size) {
                row_output[col1] = row_values[threadIdx.x];
                row_output_indices[col1] = row_indices[threadIdx.x];
            }
            if (row_exists && col2 < row_size) {
                row_output[col2] = row_values[blockDim.x + threadIdx.x];
                row_output_indices[col2] = row_indices[blockDim.x + threadIdx.x];
            }
            block_total = row_values[row_width - 1];
            block_index = row_indices[row_width - 1];
            __syncthreads();
        }
    }
}


template <typename T>
inline T cum_extremum_identity(bool is_max) {
    if constexpr (std::is_same<T, bool>::value) {
        return is_max ? false : true;
    } else if constexpr (std::is_same<T, Half>::value ||
                         std::is_same<T, BFloat16>::value) {
        return T(static_cast<float>(is_max
            ? -std::numeric_limits<float>::infinity()
            : std::numeric_limits<float>::infinity()));
    } else if constexpr (std::is_floating_point<T>::value) {
        return static_cast<T>(is_max
            ? -std::numeric_limits<T>::infinity()
            : std::numeric_limits<T>::infinity());
    } else {
        return is_max ? std::numeric_limits<T>::lowest()
                      : std::numeric_limits<T>::max();
    }
}


template <typename T, bool kIsMax>
void launch_cummaxmin_innermost(const Tensor& input, Tensor& values, Tensor& indices,
                                int64_t num_rows, int64_t row_size,
                                cudaStream_t stream) {
    const int log_num_threads_x = scan_log_num_threads_x(num_rows, row_size);
    const int num_threads_x = 1 << log_num_threads_x;
    const int num_threads_y = 512 / num_threads_x;
    const dim3 block(static_cast<unsigned>(num_threads_x),
                     static_cast<unsigned>(num_threads_y));
    const dim3 grid(static_cast<unsigned>(std::min<int64_t>(
        (num_rows + num_threads_y - 1) / num_threads_y, 65535)));
    const size_t shared_bytes = 2 * 512 * (sizeof(T) + sizeof(int64_t));
    cummaxmin_innermost_scan_kernel<T, kIsMax><<<grid, block, shared_bytes, stream>>>(
        num_rows, row_size, input.data_ptr<T>(), values.data_ptr<T>(),
        indices.data_ptr<int64_t>(), cum_extremum_identity<T>(kIsMax));
}


void launch_cummaxmin_innermost(const Tensor& input, Tensor& values, Tensor& indices,
                                int64_t num_rows, int64_t row_size,
                                bool is_max, cudaStream_t stream) {
#define TP_CUM_INNER_CASE(ctype, name_) \
    case DType::name_: \
        if (is_max) { \
            launch_cummaxmin_innermost<ctype, true>( \
                input, values, indices, num_rows, row_size, stream); \
        } else { \
            launch_cummaxmin_innermost<ctype, false>( \
                input, values, indices, num_rows, row_size, stream); \
        } \
        break;
    switch (input.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_CUM_INNER_CASE)
        default: TP_THROW(TypeError, "cumulative extremum: unsupported dtype");
    }
#undef TP_CUM_INNER_CASE
}


std::tuple<Tensor, Tensor> cummax_cuda(const Tensor& self, int64_t dim) {
    int64_t nd = self.dim();
    dim = wrap_scan_dim(dim, nd);
    Tensor sc = self.contiguous();
    Tensor vals = Tensor::empty(shape_of(sc), sc.dtype(), sc.device());
    Tensor idxs = Tensor::empty(shape_of(sc), DType::Int64, sc.device());
    if (nd == 0) {
        vals.copy_(sc);
        idxs.fill_(Scalar(0));
        return {vals, idxs};
    }
    int64_t d_size = sc.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(shape_of(sc), dim, outer, inner);
    int64_t slices = outer * inner;
    if (slices > 0 && d_size > 0) {
        auto stream = getCurrentCUDAStream().stream();
        if (dim == nd - 1) {
            launch_cummaxmin_innermost(sc, vals, idxs, outer, d_size, true, stream);
        } else {
            dim3 grid = make_grid(slices), block(kThreads);
#define TP_CM(ctype, name_) \
    case DType::name_: \
        cummaxmin_scan_kernel<ctype, true><<<grid, block, 0, stream>>>( \
            slices, d_size, inner, sc.data_ptr<ctype>(), vals.data_ptr<ctype>(), \
            idxs.data_ptr<int64_t>()); \
        break;
        switch (sc.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_CM)
            default: TP_THROW(TypeError, "cummax: unsupported dtype");
        }
#undef TP_CM
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return {vals, idxs};
}

std::tuple<Tensor, Tensor> cummin_cuda(const Tensor& self, int64_t dim) {
    int64_t nd = self.dim();
    dim = wrap_scan_dim(dim, nd);
    Tensor sc = self.contiguous();
    Tensor vals = Tensor::empty(shape_of(sc), sc.dtype(), sc.device());
    Tensor idxs = Tensor::empty(shape_of(sc), DType::Int64, sc.device());
    if (nd == 0) {
        vals.copy_(sc);
        idxs.fill_(Scalar(0));
        return {vals, idxs};
    }
    int64_t d_size = sc.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(shape_of(sc), dim, outer, inner);
    int64_t slices = outer * inner;
    if (slices > 0 && d_size > 0) {
        auto stream = getCurrentCUDAStream().stream();
        if (dim == nd - 1) {
            launch_cummaxmin_innermost(sc, vals, idxs, outer, d_size, false, stream);
        } else {
            dim3 grid = make_grid(slices), block(kThreads);
#define TP_CMIN(ctype, name_) \
    case DType::name_: \
    cummaxmin_scan_kernel<ctype, false><<<grid, block, 0, stream>>>( \
            slices, d_size, inner, sc.data_ptr<ctype>(), vals.data_ptr<ctype>(), \
            idxs.data_ptr<int64_t>()); \
        break;
        switch (sc.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_CMIN)
            default: TP_THROW(TypeError, "cummin: unsupported dtype");
        }
#undef TP_CMIN
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return {vals, idxs};
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, CumminmaxKernels) {
    m.impl("cummax", cummax_cuda);
    m.impl("cummin", cummin_cuda);
}

namespace {


void interop__cummax_helper_cuda(const Tensor& self, Tensor& values,
                                 Tensor& indices, int64_t dim) {
    auto result = dispatch_cuda<std::tuple<Tensor, Tensor>>("cummax", self, dim);
    write_out(values, std::get<0>(result));
    write_out(indices, std::get<1>(result));
}


void interop__cummin_helper_cuda(const Tensor& self, Tensor& values,
                                 Tensor& indices, int64_t dim) {
    auto result = dispatch_cuda<std::tuple<Tensor, Tensor>>("cummin", self, dim);
    write_out(values, std::get<0>(result));
    write_out(indices, std::get<1>(result));
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, CumminmaxInterop) {
    // cummax/cummin out-style helpers
    m.impl("_cummax_helper", interop__cummax_helper_cuda);
    m.impl("_cummin_helper", interop__cummin_helper_cuda);
}

} // namespace cuda
} // namespace tensorplay
