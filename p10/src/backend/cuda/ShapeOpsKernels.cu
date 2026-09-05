// Tensor shape and layout operators - CUDA kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "Quantizer.h"
#include "CUDARuntime.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

namespace {

constexpr int kThreads = 256;

inline dim3 make_grid(int64_t work) {
    return dim3(static_cast<unsigned>((work + kThreads - 1) / kThreads));
}

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    // Dimension wrapping reports the original (unwrapped) value on error.
    const int64_t min = -ndim;
    const int64_t max = ndim - 1;
    if (dim < min || dim > max) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 min, ", ", max, "], but got ", dim, ")");
    }
    return dim < 0 ? dim + ndim : dim;
}

// Scalar wrapping: rank-0 accepts dims [-1, 0] (both wrap to 0).  Used by
// flip's dim-list conversion.
inline int64_t wrap_dim_scalar(int64_t dim, int64_t ndim) {
    return wrap_dim(dim, ndim == 0 ? 1 : ndim);
}

inline void outer_inner(const std::vector<int64_t>& shape, int64_t dim,
                        int64_t& outer, int64_t& inner) {
    outer = 1; inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) inner *= shape[i];
}

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

Tensor empty_transform_output(const Tensor& self) {
    const auto shape = shape_of(self);
    if (!isQuantizedType(self.dtype())) {
        return Tensor::empty(shape, self.dtype(), self.device());
    }
    quantized::require_quantized(self, "roll");
    Tensor codes = Tensor::empty(shape, underlying_storage_type(self.dtype()),
                                 self.device());
    return quantized::make_qtensor(codes, quantized::quantizer_of(self),
                                   self.dtype());
}

Tensor pack_i64(const std::vector<int64_t>& v, const Device& dev) {
    Tensor t = Tensor::empty({static_cast<int64_t>(std::max<size_t>(v.size(), 1))},
                             DType::Int64, dev);
    if (!v.empty())
        cudaMemcpy(t.data_ptr<int64_t>(), v.data(), v.size() * sizeof(int64_t),
                   cudaMemcpyHostToDevice);
    return t;
}


// Shape-op device kernels
__global__ void trace_batch_kernel(int64_t batch, int64_t rows, int64_t cols,
                                   const double* sp, double* dp) {
    int64_t bi = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; bi < batch; bi += stride) {
        double s2 = 0;
        int64_t d = rows < cols ? rows : cols;
        for (int64_t i = 0; i < d; ++i) s2 += sp[bi * rows * cols + i * cols + i];
        dp[bi] = s2;
    }
}

template <typename T>
__global__ void index_gather_kernel(int64_t n, const T* src, const int64_t* idx, T* out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = src[idx[i]];
}

template <typename T>
__global__ void diag_scatter_kernel(int64_t n, int64_t size, int64_t diagonal,
                                    const T* src, T* dst) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        int64_t r = diagonal >= 0 ? i : i - diagonal;
        int64_t c = diagonal >= 0 ? i + diagonal : i;
        dst[r * size + c] = src[i];
    }
}

template <typename T>
__global__ void row_copy_kernel(int64_t total_rows, int64_t inner,
                                const T* src, T* dst,
                                int64_t src_row_stride, int64_t dst_row_stride) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < total_rows * inner; i += stride) {
        int64_t r = i / inner, c = i % inner;
        dst[r * dst_row_stride + c] = src[r * src_row_stride + c];
    }
}

template <typename T>
__global__ void flip_map_kernel(int64_t n, int64_t nd, const T* src, T* dst,
                                const int64_t* sizes, const int64_t* flips,
                                const int64_t* out_strides) {
    int64_t li = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; li < n; li += stride) {
        int64_t r2 = li, src_off = 0, dst_off = 0, mult = 1;
        for (int64_t d2 = nd - 1; d2 >= 0; --d2) {
            int64_t c = r2 % sizes[d2];
            r2 /= sizes[d2];
            int64_t sc3 = flips[d2] ? (sizes[d2] - 1 - c) : c;
            src_off += sc3 * mult;
            dst_off += c * out_strides[d2];
            mult *= sizes[d2];
        }
        dst[dst_off] = src[src_off];
    }
}

template <typename T>
__global__ void roll_map_kernel(int64_t n, int64_t nd, const T* src, T* dst,
                                const int64_t* sizes, const int64_t* shifts) {
    int64_t li = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; li < n; li += stride) {
        int64_t r2 = li, src_off = 0, mult = 1;
        for (int64_t d2 = nd - 1; d2 >= 0; --d2) {
            int64_t c = r2 % sizes[d2];
            r2 /= sizes[d2];
            int64_t sc3 = c - shifts[d2];
            if (sc3 < 0) sc3 += sizes[d2];
            src_off += sc3 * mult;
            mult *= sizes[d2];
        }
        dst[li] = src[src_off];
    }
}

template <typename T>
__global__ void broadcast_map_kernel(int64_t n, int64_t nd, const T* src, T* dst,
                                     const int64_t* out_sizes,
                                     const int64_t* in_sizes_padded,
                                     const int64_t* in_strides_padded) {
    int64_t li = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; li < n; li += stride) {
        int64_t r2 = li, src_off = 0;
        for (int64_t d2 = nd - 1; d2 >= 0; --d2) {
            int64_t c = r2 % out_sizes[d2];
            r2 /= out_sizes[d2];
            src_off += (in_sizes_padded[d2] == 1 ? 0 : c) * in_strides_padded[d2];
        }
        dst[li] = src[src_off];
    }
}

} // anonymous namespace

// ===========================================================================
// Shape ops
// ===========================================================================

namespace {

__global__ void meshgrid_coord_kernel(int64_t total, size_t k, size_t j,
                                      const int64_t* sizes, int64_t* out) {
    int64_t li = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; li < total; li += stride) {
        int64_t r2 = li;
        int64_t coord = 0;
        for (size_t d2 = k; d2-- > 0;) {
            coord = r2 % sizes[d2];
            r2 /= sizes[d2];
            if (d2 == j) break;
        }
        out[li] = coord;
    }
}

template <typename T>
__global__ void pixel_shuffle_map_kernel(int64_t n, int64_t C, int64_t H, int64_t W,
                                         int64_t r, const T* src, T* dst) {
    int64_t li = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; li < n; li += stride) {
        int64_t w = li % W; int64_t rem = li / W;
        int64_t h = rem % H; rem /= H;
        int64_t c = rem % C; int64_t bn = rem / C;
        int64_t ih = h % r, iw = w % r;
        int64_t src_off = ((((bn * C * r * r) + c * r * r + ih * r + iw) * H) + h / r) * W + w / r;
        dst[li] = src[src_off];
    }
}

template <typename T>
__global__ void pixel_unshuffle_map_kernel(int64_t n, int64_t C, int64_t H, int64_t W,
                                           int64_t r, const T* src, T* dst) {
    // out: (N, C*r^2, H, W); in: (N, C, H*r, W*r)
    int64_t li = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t Wr = W * r;
    for (; li < n; li += stride) {
        int64_t w = li % W; int64_t rem = li / W;
        int64_t h = rem % H; rem /= H;
        int64_t cc = rem % (C * r * r); rem /= (C * r * r);
        int64_t bn = rem;
        int64_t c = cc / (r * r);
        int64_t ij = cc % (r * r);
        int64_t ih = ij / r, iw = ij % r;
        int64_t src_off = (((bn * C + c) * (H * r)) + h * r + ih) * Wr + w * r + iw;
        dst[li] = src[src_off];
    }
}

template <typename T>
__global__ void channel_shuffle_map_kernel(int64_t n, int64_t outer, int64_t C, int64_t inner,
                                           int64_t cg, const T* src, T* dst) {
    // li layout: (outer * C + c) * inner + tail; channel shuffle swaps the
    // group index and the within-group index.
    int64_t li = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; li < n; li += stride) {
        int64_t tail = li % inner;
        int64_t rest = li / inner;
        int64_t c = rest % C;
        int64_t o = rest / C;
        int64_t j = c / cg, gi = c % cg;
        int64_t src_c = gi * cg + j;
        dst[li] = src[(o * C + src_c) * inner + tail];
    }
}



} // anonymous namespace

Tensor trace_cuda(const Tensor& self) {
    if (self.dim() < 2) TP_THROW(RuntimeError, "trace: input must have at least 2 dimensions");
    int64_t rows = self.size(-2), cols = self.size(-1);
    int64_t batch = self.numel() / (rows * cols);
    std::vector<int64_t> out_shape = shape_of(self);
    out_shape.resize(out_shape.size() - 2);
    Tensor out64 = Tensor::zeros({std::max<int64_t>(batch, 1)}, DType::Float64, self.device());
    if (batch > 0) {
        Tensor sc = self.contiguous().to(DType::Float64);
        auto stream = getCurrentCUDAStream().stream();
        dim3 grid = make_grid(batch), block(kThreads);
        trace_batch_kernel<<<grid, block, 0, stream>>>(
            batch, rows, cols, sc.data_ptr<double>(), out64.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    return out64.reshape(out_shape).to(self.dtype());
}

Tensor diag_cuda(const Tensor& self, int64_t diagonal) {
    int64_t nd = self.dim();
    Tensor sc = self.contiguous();
    if (nd == 1) {
        int64_t n = sc.size(0);
        int64_t size = n + std::abs(diagonal);
        Tensor out = Tensor::zeros({size, size}, sc.dtype(), sc.device());
        auto stream = getCurrentCUDAStream().stream();
        dim3 grid = make_grid(std::max<int64_t>(n, 1)), block(kThreads);
#define TP_DGS(ctype, name_) \
    case DType::name_: \
        diag_scatter_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, size, diagonal, sc.data_ptr<ctype>(), out.data_ptr<ctype>()); \
        break;
        switch (sc.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_DGS)
            default: TP_THROW(TypeError, "diag: unsupported dtype");
        }
#undef TP_DGS
        CUDA_CHECK(cudaGetLastError());
        return out;
    }
    if (nd == 2) {
        int64_t rows = sc.size(0), cols = sc.size(1);
        std::vector<int64_t> idx;
        if (diagonal >= 0) {
            for (int64_t i = 0; i + diagonal < cols && i < rows; ++i)
                idx.push_back(i * cols + i + diagonal);
        } else {
            for (int64_t i = 0; i - diagonal < rows && i < cols; ++i)
                idx.push_back((i - diagonal) * cols + i);
        }
        Tensor out = Tensor::empty({static_cast<int64_t>(idx.size())}, sc.dtype(), sc.device());
        if (!idx.empty()) {
            Tensor d_idx = pack_i64(idx, sc.device());
            auto stream = getCurrentCUDAStream().stream();
            dim3 grid = make_grid(static_cast<int64_t>(idx.size())), block(kThreads);
#define TP_DGE(ctype, name_) \
    case DType::name_: \
        index_gather_kernel<ctype><<<grid, block, 0, stream>>>( \
            static_cast<int64_t>(idx.size()), sc.data_ptr<ctype>(), \
            d_idx.data_ptr<int64_t>(), out.data_ptr<ctype>()); \
        break;
            switch (sc.dtype()) {
                TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_DGE)
                default: TP_THROW(TypeError, "diag: unsupported dtype");
            }
#undef TP_DGE
            CUDA_CHECK(cudaGetLastError());
        }
        return out;
    }
    TP_THROW(RuntimeError, "diag: input must be 1-D or 2-D");
}

Tensor diag_embed_cuda(const Tensor& self, int64_t offset, int64_t dim1_, int64_t dim2_) {
    // Host-staged reference implementation for this infrequently used shape
    // transform.
    int64_t nDims = self.dim() + 1;
    int64_t dim1 = wrap_dim(dim1_, nDims);
    int64_t dim2 = wrap_dim(dim2_, nDims);
    if (dim1 == dim2) TP_THROW(RuntimeError, "diagonal dimensions cannot be identical");
    int64_t new_dim_len = std::abs(offset) + self.size(-1);
    std::vector<int64_t> sizes = shape_of(self);
    sizes.pop_back();
    sizes.insert(sizes.begin() + std::min(dim1, dim2), new_dim_len);
    sizes.insert(sizes.begin() + std::max(dim1, dim2), new_dim_len);
    Tensor rc = Tensor::zeros(sizes, self.dtype(), Device(DeviceType::CPU));
    Tensor sc = self.contiguous().to(Device(DeviceType::CPU));
    int64_t mid = std::max(dim1, dim2);
    int64_t lowdim = std::min(dim1, dim2);
    int64_t n = self.numel();
#define TP_DEW(ctype, name_) \
    case DType::name_: { \
        const ctype* s2 = sc.data_ptr<ctype>(); \
        ctype* d2 = rc.data_ptr<ctype>(); \
        for (int64_t li = 0; li < n; ++li) { \
            int64_t rem = li; \
            std::vector<int64_t> sc3(self.dim(), 0); \
            for (int64_t d3 = static_cast<int64_t>(self.dim()) - 1; d3 >= 0; --d3) { \
                sc3[d3] = rem % self.size(d3); rem /= self.size(d3); } \
            int64_t t = sc3.back(); \
            int64_t i = offset >= 0 ? t : t - offset; \
            int64_t j = offset >= 0 ? t + offset : t; \
            std::vector<int64_t> rc2(sizes.size(), 0); \
            int64_t sk = 0; \
            for (int64_t d3 = 0; d3 < static_cast<int64_t>(sizes.size()); ++d3) { \
                if (d3 == lowdim) rc2[d3] = i; \
                else if (d3 == mid) rc2[d3] = j; \
                else rc2[d3] = sc3[sk++]; } \
            int64_t lin = 0; \
            for (int64_t d3 = 0; d3 < static_cast<int64_t>(sizes.size()); ++d3) \
                lin = lin * sizes[d3] + rc2[d3]; \
            d2[lin] = s2[li]; \
        } \
        break; }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_DEW)
        default: TP_THROW(TypeError, "diag_embed: unsupported dtype");
    }
#undef TP_DEW
    return rc.to(self.device());
}

Tensor narrow_cuda(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
    if (self.dim() == 0) {
        TP_THROW(RuntimeError, "narrow() cannot be applied to a 0-dim tensor.");
    }
    if (length < 0) {
        TP_THROW(RuntimeError, "narrow(): length must be non-negative.");
    }
    dim = wrap_dim(dim, self.dim());
    const int64_t cur_size = self.size(dim);
    if (start < -cur_size || start > cur_size) {
        TP_THROW(IndexError, "start out of range (expected to be in range of [",
                 -cur_size, ", ", cur_size, "], but got ", start, ")");
    }
    if (start < 0) start += cur_size;
    if (start > cur_size - length) {
        TP_THROW(RuntimeError, "start (", start, ") + length (", length,
                 ") exceeds dimension size (", cur_size, ").");
    }
    return self.slice(dim, start, start + length, 1);
}

std::vector<Tensor> split_with_sizes_cuda(const Tensor& self, std::vector<int64_t> split_sizes,
                                          int64_t dim) {
    if (self.dim() == 0) {
        TP_THROW(RuntimeError, "split expects at least a 1-dimensional tensor");
    }
    const int64_t nd = self.dim();
    if (dim < -nd || dim >= nd) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 -nd, ", ", nd - 1, "], but got ", dim, ")");
    }
    if (dim < 0) dim += nd;
    const int64_t dim_size = self.size(dim);
    std::vector<Tensor> outs;
    outs.reserve(split_sizes.size());
    int64_t start = 0;
    for (const int64_t len : split_sizes) {
        if (len < 0) {
            TP_THROW(RuntimeError, "split_with_sizes expects split_sizes have only non-negative "
                     "entries, but got split_sizes=[", [&] {
                         std::string s;
                         for (size_t i = 0; i < split_sizes.size(); ++i) {
                             if (i) s += ", ";
                             s += std::to_string(split_sizes[i]);
                         }
                         return s;
                     }(), "]");
        }
        outs.push_back(self.slice(dim, start, start + len));
        start += len;
    }
    if (start != dim_size) {
        TP_THROW(RuntimeError, "split_with_sizes expects split_sizes to sum exactly to ",
                 dim_size, " (input tensor's size at dimension ", dim, "), but got split_sizes=[",
                 [&] {
                     std::string s;
                     for (size_t i = 0; i < split_sizes.size(); ++i) {
                         if (i) s += ", ";
                         s += std::to_string(split_sizes[i]);
                     }
                     return s;
                 }(), "]");
    }
    return outs;
}

std::vector<Tensor> tensor_split_cuda(const Tensor& self, int64_t sections, int64_t dim) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (sections <= 0) TP_THROW(RuntimeError, "tensor_split: number of sections must be larger than 0");
    int64_t size = self.size(dim);
    int64_t base = size / sections, rem = size % sections;
    std::vector<Tensor> outs;
    int64_t start = 0;
    for (int64_t i = 0; i < sections; ++i) {
        int64_t len = base + (i < rem ? 1 : 0);
        if (len > 0) outs.push_back(narrow_cuda(self, dim, start, len));
        else outs.emplace_back();
        start += len;
    }
    return outs;
}

Tensor flip_cuda(const Tensor& self, const std::vector<int64_t>& dims) {
    // TensorTransformations.cpp:36 flip: dim_list_to_bitset (WrapDimUtilsMulti.h)
    // wraps with wrap_scalar=true and rejects duplicate dims.
    int64_t nd = self.dim();
    std::vector<bool> seen(nd > 0 ? nd : 1, false);
    std::vector<int64_t> h_flips(nd, 0);
    for (auto d : dims) {
        int64_t w = wrap_dim_scalar(d, nd);
        if (nd > 0) {
            if (seen[w]) {
                TP_THROW(RuntimeError, "dim ", w,
                         " appears multiple times in the list of dims");
            }
            seen[w] = true;
            h_flips[w] = 1;
        }
    }
    Tensor sc = self.contiguous();
    Tensor out = detail::clone_impl(self);
    int64_t n = sc.numel();
    if (n == 0) return out;
    Tensor d_sizes = pack_i64(shape_of(sc), sc.device());
    Tensor d_flips = pack_i64(h_flips, sc.device());
    Tensor d_out_strides = pack_i64(
        static_cast<std::vector<int64_t>>(out.strides()), sc.device());
    auto stream = getCurrentCUDAStream().stream();
    dim3 grid = make_grid(n), block(kThreads);
#define TP_FL2(ctype, name_) \
    case DType::name_: \
        flip_map_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, nd, sc.data_ptr<ctype>(), out.data_ptr<ctype>(), \
            d_sizes.data_ptr<int64_t>(), d_flips.data_ptr<int64_t>(), \
            d_out_strides.data_ptr<int64_t>()); \
        break;
    switch (sc.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_FL2)
        TENSORPLAY_FORALL_QINT_TYPES(TP_FL2)
        default: TP_THROW(TypeError, "flip: unsupported dtype");
    }
#undef TP_FL2
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor roll_cuda(const Tensor& self, const std::vector<int64_t>& shifts, const std::vector<int64_t>& dims) {
    // TensorTransformations.cpp:110 roll + TensorTransformations.h roll_common.
    if (dims.size() != 1 || shifts.size() != 1) {
        if (shifts.empty()) TP_THROW(RuntimeError, "`shifts` required");
        if (dims.empty() && shifts.size() == 1) {
            // Flatten-roll: roll the flattened tensor and view back.
            Tensor flat = self.contiguous().reshape({self.numel()});
            Tensor rolled = roll_cuda(flat, {shifts[0]}, {0});
            return rolled.reshape(shape_of(self));
        }
        if (shifts.size() != dims.size()) {
            TP_THROW(RuntimeError, "shifts and dimensions must align. shifts: ",
                     shifts.size(), ", dims:", dims.size());
        }
        Tensor cur = self;
        for (size_t i = 0; i < dims.size(); ++i) {
            cur = roll_cuda(cur, {shifts[i]}, {dims[i]});
        }
        return cur;
    }
    // Avoid a div zero error below; empty input rolls to
    // itself.
    if (self.numel() == 0) return self.clone();
    const int64_t nd = self.dim();
    if (nd == 0) {
        // wrap_scalar=false rejects any dim.
        TP_THROW(IndexError, "Dimension specified as ", dims[0],
                 " but tensor has no dimensions");
    }
    const int64_t dim = wrap_dim(dims[0], nd);
    const int64_t size = self.size(dim);
    // roll_map_kernel reads src[c - sh] so a normalized positive shift keeps
    // the mapping identical to cat({narrow(start), narrow(0, start)}).
    std::vector<int64_t> sh(nd, 0);
    sh[dim] = ((shifts[0] % size) + size) % size;
    Tensor sc = self.contiguous();
    Tensor out = empty_transform_output(sc);
    int64_t n = sc.numel();
    Tensor d_sizes = pack_i64(shape_of(sc), sc.device());
    Tensor d_sh = pack_i64(sh, sc.device());
    auto stream = getCurrentCUDAStream().stream();
    dim3 grid = make_grid(n), block(kThreads);
#define TP_RL2(ctype, name_) \
    case DType::name_: \
        roll_map_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, nd, sc.data_ptr<ctype>(), out.data_ptr<ctype>(), \
            d_sizes.data_ptr<int64_t>(), d_sh.data_ptr<int64_t>()); \
        break;
    switch (sc.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_RL2)
        TENSORPLAY_FORALL_QINT_TYPES(TP_RL2)
        default: TP_THROW(TypeError, "roll: unsupported dtype");
    }
#undef TP_RL2
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor rot90_cuda(const Tensor& self, int64_t k, const std::vector<int64_t>& dims) {
    // TensorTransformations.cpp:127 rot90.
    const int64_t total_dims = self.dim();
    const int64_t total_rot_dims = static_cast<int64_t>(dims.size());
    if (total_rot_dims != 2) {
        TP_THROW(RuntimeError, "expected total rotation dims == 2, but got dims = ",
                 total_rot_dims);
    }
    if (total_dims < 2) {
        TP_THROW(RuntimeError, "expected total dims >= 2, but got total dims = ",
                 total_dims);
    }
    // Validate range first so out-of-range dims raise IndexError, then
    // normalize before checking for duplicates (e.g. [1, -1] on a 2D tensor).
    const int64_t dim0 = wrap_dim(dims[0], total_dims);
    const int64_t dim1 = wrap_dim(dims[1], total_dims);
    if (dim0 == dim1) {
        TP_THROW(RuntimeError, "expected rotation dims to be different, but got dim0 = ",
                 dims[0], " and dim1 = ", dims[1]);
    }
    // handle modulo with negative k
    k = (4 + (k % 4)) % 4;
    // transpose_ on the fresh flip result: a view with swapped sizes/strides.
    auto transpose_view = [](const Tensor& x, int64_t a, int64_t b) {
        std::vector<int64_t> sizes(x.dim()), strides(x.dim());
        for (int64_t i = 0; i < x.dim(); ++i) {
            sizes[i] = x.size(i);
            strides[i] = x.stride(i);
        }
        std::swap(sizes[a], sizes[b]);
        std::swap(strides[a], strides[b]);
        return x.as_strided(sizes, strides);
    };
    switch (k) {
        case 1: return transpose_view(flip_cuda(self, {dim1}), dim0, dim1);
        case 2: return flip_cuda(self, {dim0, dim1});
        case 3: return transpose_view(flip_cuda(self, {dim0}), dim0, dim1);
        default: return detail::contiguous_clone(self);
    }
}

std::vector<Tensor> meshgrid_cuda(const std::vector<Tensor>& tensors, const std::string& indexing) {
    // ij-indexing semantics.
    size_t k = tensors.size();
    if (k == 0) return {};
    std::vector<int64_t> sizes;
    sizes.reserve(k);
    for (auto& t : tensors) sizes.push_back(static_cast<int64_t>(t.numel()));
    int64_t total = 1;
    for (int64_t s2 : sizes) total *= s2;
    Tensor d_sizes = pack_i64(sizes, tensors[0].device());
    std::vector<Tensor> outs;
    auto stream = getCurrentCUDAStream().stream();
    for (size_t j = 0; j < k; ++j) {
        Tensor g = Tensor::empty(sizes, DType::Int64, tensors[0].device());
        if (total > 0) {
            dim3 grid = make_grid(total), block(kThreads);
            meshgrid_coord_kernel<<<grid, block, 0, stream>>>(
                total, k, j, d_sizes.data_ptr<int64_t>(), g.data_ptr<int64_t>());
            CUDA_CHECK(cudaGetLastError());
        }
        outs.push_back(g);
    }
    return outs;
}

std::vector<Tensor> broadcast_tensors_cuda(const std::vector<Tensor>& tensors) {
    // CompositeImplicit implementation of broadcast_tensors_cpu: device-generic
    // expand views; no CUDA-specific code required.
    std::vector<int64_t> shape;
    for (auto& t : tensors) shape = broadcast_shapes(shape, shape_of(t));
    std::vector<Tensor> outs;
    outs.reserve(tensors.size());
    for (auto& t : tensors) {
        std::vector<int64_t> ts = shape_of(t);
        outs.push_back(ts == shape ? t : t.expand(shape));
    }
    return outs;
}

Tensor block_diag_cuda(const std::vector<Tensor>& tensors) {
    // CompositeImplicit implementation of block_diag_cpu (device-generic members).
    if (tensors.empty()) return Tensor::empty({1, 0}, DType::Float32,
                                              Device(DeviceType::CUDA, currentDevice()));
    const Device& device = tensors[0].device();
    DType out_dtype = tensors[0].dtype();
    int64_t rows = 0, cols = 0;
    std::vector<Tensor> blocks2d;
    blocks2d.reserve(tensors.size());
    for (size_t idx = 0; idx < tensors.size(); ++idx) {
        const Tensor& t = tensors[idx];
        if (!(t.device() == device)) {
            TP_THROW(RuntimeError,
                     "block_diag: input tensors must all be on the same device.");
        }
        out_dtype = promoteTypes(out_dtype, t.dtype());
        const int64_t nd = t.dim();
        if (nd > 2) {
            TP_THROW(RuntimeError,
                     "block_diag: Input tensors must have 2 or fewer dimensions. Input ",
                     static_cast<int64_t>(idx), " has ", nd, " dimensions");
        }
        Tensor b2 = t;
        if (nd == 1) b2 = t.expand({1, t.size(0)});
        else if (nd == 0) b2 = t.expand({1, 1});
        blocks2d.push_back(b2);
        rows += b2.size(0);
        cols += b2.size(1);
    }
    Tensor out = Tensor::zeros({rows, cols}, out_dtype, device);
    int64_t off0 = 0, off1 = 0;
    for (const auto& b : blocks2d) {
        out.slice(0, off0, off0 + b.size(0))
           .slice(1, off1, off1 + b.size(1))
           .copy_(b);
        off0 += b.size(0);
        off1 += b.size(1);
    }
    return out;
}

Tensor pixel_shuffle_cuda(const Tensor& self, int64_t upscale_factor) {
    // PixelShuffle.cpp:23 semantics: (N, C*r^2, H, W) -> (N, C, H*r, W*r)
    if (self.dim() != 4) TP_THROW(RuntimeError, "pixel_shuffle expects 4D input");
    int64_t r = upscale_factor;
    int64_t N = self.size(0);
    int64_t C = self.size(1) / (r * r);
    int64_t H = self.size(2), W = self.size(3);
    if (C * r * r != self.size(1))
        TP_THROW(RuntimeError, "pixel_shuffle: channel dim must be divisible by r^2");
    Tensor out = Tensor::empty({N, C, H * r, W * r}, self.dtype(), self.device());
    Tensor sc = self.contiguous();
    int64_t n = out.numel();
    auto stream = getCurrentCUDAStream().stream();
    dim3 grid = make_grid(n), block(kThreads);
#define TP_PS(ctype, name_) \
    case DType::name_: \
        pixel_shuffle_map_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, C, H, W, r, sc.data_ptr<ctype>(), out.data_ptr<ctype>()); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_PS)
        default: TP_THROW(TypeError, "pixel_shuffle: unsupported dtype");
    }
#undef TP_PS
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor pixel_unshuffle_cuda(const Tensor& self, int64_t downscale_factor) {
    if (self.dim() != 4) TP_THROW(RuntimeError, "pixel_unshuffle expects 4D input");
    int64_t r = downscale_factor;
    int64_t N = self.size(0);
    int64_t C = self.size(1);
    int64_t H = self.size(2) / r, W = self.size(3) / r;
    if (H * r != self.size(2) || W * r != self.size(3))
        TP_THROW(RuntimeError, "pixel_unshuffle: spatial dims must be divisible by r");
    Tensor out = Tensor::empty({N, C * r * r, H, W}, self.dtype(), self.device());
    Tensor sc = self.contiguous();
    int64_t n = out.numel();
    auto stream = getCurrentCUDAStream().stream();
    dim3 grid = make_grid(n), block(kThreads);
#define TP_PU(ctype, name_) \
    case DType::name_: \
        pixel_unshuffle_map_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, C, H, W, r, sc.data_ptr<ctype>(), out.data_ptr<ctype>()); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_PU)
        default: TP_THROW(TypeError, "pixel_unshuffle: unsupported dtype");
    }
#undef TP_PU
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor channel_shuffle_cuda(const Tensor& self, int64_t groups) {
    // ChannelShuffle: view(N, g, C/g, ...) -> transpose(1,2). Channel dim is
    // dim 1 for >=2D input, dim 0 for 1-D input.
    if (self.dim() < 1) TP_THROW(RuntimeError, "channel_shuffle expects >= 1D input");
    int64_t cdim = self.dim() >= 2 ? 1 : 0;
    int64_t C = self.size(cdim);
    int64_t outer = 1;
    for (int64_t i = 0; i < cdim; ++i) outer *= self.size(i);
    int64_t inner = 1;
    for (int64_t i = cdim + 1; i < self.dim(); ++i) inner *= self.size(i);
    if (C % groups) TP_THROW(RuntimeError, "channel_shuffle: channel dim not divisible by groups");
    int64_t cg = C / groups;
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(shape_of(self), self.dtype(), self.device());
    int64_t n = self.numel();
    auto stream = getCurrentCUDAStream().stream();
    dim3 grid = make_grid(n), block(kThreads);
#define TP_CS(ctype, name_) \
    case DType::name_: \
        channel_shuffle_map_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, outer, C, inner, cg, sc.data_ptr<ctype>(), out.data_ptr<ctype>()); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_CS)
        default: TP_THROW(TypeError, "channel_shuffle: unsupported dtype");
    }
#undef TP_CS
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor unfold_cuda(const Tensor& self, int64_t dimension, int64_t size, int64_t step) {
    // unfold: an as_strided view.  wrap_scalar=true allows
    // dimension == 0 on 0-d tensors (max_size becomes 1).
    const int64_t nd = self.dim();
    dimension = wrap_dim_scalar(dimension, nd);

    std::vector<int64_t> sizes = shape_of(self);
    std::vector<int64_t> strides = self.strides();
    const int64_t max_size = nd == 0 ? 1 : sizes[dimension];
    if (size < 0) TP_THROW(RuntimeError, "size is ", size, " but must be >= 0");
    if (size > max_size) {
        TP_THROW(RuntimeError, "maximum size for tensor at dimension ", dimension,
                 " is ", max_size, " but size is ", size);
    }
    if (step <= 0) TP_THROW(RuntimeError, "step is ", step, " but must be > 0");
    sizes.push_back(size);
    strides.push_back(nd == 0 ? 1 : strides[dimension]);
    // The if handles the self.dim() == 0 case
    if (dimension < nd) {
        sizes[dimension] = (sizes[dimension] - size) / step + 1;
        strides[dimension] *= step;
    }
    return self.as_strided(sizes, strides);
}

namespace {

template <typename T>
__global__ void unfold_backward_gather_kernel(int64_t total, int64_t input_dim_size,
                                              int64_t count, int64_t inner,
                                              int64_t size, int64_t step,
                                              const T* grad, T* grad_input) {
    // Gather over grad_input elements (race-free); each element accumulates the
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; t < total; t += stride) {
        int64_t inner_idx = t % inner;
        int64_t rest = t / inner;
        int64_t idx_dim = rest % input_dim_size;
        int64_t outer_idx = rest / input_dim_size;
        int64_t left = (idx_dim > size) ? (idx_dim - size) / step : 0;
        if (!(left * step <= idx_dim && idx_dim < left * step + size)) ++left;
        int64_t right = idx_dim / step;
        if (right >= count) right = count - 1;
        T acc{};
        for (int64_t fold = left; fold <= right; ++fold) {
            int64_t j = idx_dim - fold * step;
            acc += grad[((outer_idx * count + fold) * inner + inner_idx) * size + j];
        }
        grad_input[t] = acc;
    }
}

} // anonymous namespace

Tensor unfold_backward_cuda(const Tensor& grad, const std::vector<int64_t>& input_sizes,
                            int64_t dim, int64_t size, int64_t step) {
    // accumulating where windows overlap (step < size).
    if (step <= 0) TP_THROW(RuntimeError, "step is ", step, " but must be > 0");
    Tensor grad_input = Tensor::zeros(input_sizes, grad.dtype(), grad.device());
    const int64_t nd = static_cast<int64_t>(input_sizes.size());
    if (nd == 0) {
        if (size > 0) grad_input.copy_(grad.select(0, 0));
        return grad_input;
    }
    dim = wrap_dim(dim, nd);
    const int64_t input_dim_size = input_sizes[dim];
    const int64_t count = grad.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(input_sizes, dim, outer, inner);
    Tensor gc = grad.contiguous();
    const int64_t total = outer * input_dim_size * inner;
    if (total == 0) return grad_input;
    auto stream = getCurrentCUDAStream().stream();
    dim3 grid = make_grid(total), block(kThreads);
#define TP_UFB(ctype, name_) \
    case DType::name_: \
        unfold_backward_gather_kernel<ctype><<<grid, block, 0, stream>>>( \
            total, input_dim_size, count, inner, size, step, \
            gc.data_ptr<ctype>(), grad_input.data_ptr<ctype>()); \
        break;
    switch (grad.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_UFB)
        default: TP_THROW(TypeError, "unfold_backward: unsupported dtype");
    }
#undef TP_UFB
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

namespace {

inline void check_scatter_source(const Tensor& target, const Tensor& src) {
    if (target.shape() != src.shape()) {
        TP_THROW(RuntimeError,
                 "expected src to have a size equal to the target slice");
    }
}

} // anonymous namespace

Tensor select_scatter_cuda(const Tensor& self, const Tensor& src, int64_t dim,
                           int64_t index) {
    dim = wrap_dim(dim, self.dim());
    Tensor output = detail::clone_impl(self);
    Tensor target = output.select(dim, index);
    check_scatter_source(target, src);
    target.copy_(src);
    return output;
}

Tensor slice_scatter_cuda(const Tensor& self, const Tensor& src, int64_t dim,
                          std::optional<int64_t> start,
                          std::optional<int64_t> end, int64_t step) {
    if (step <= 0) {
        TP_THROW(RuntimeError, "slice_scatter: step must be positive");
    }
    dim = wrap_dim(dim, self.dim());
    const int64_t length = self.size(dim);
    int64_t begin = start.value_or(0);
    int64_t finish = end.value_or(length);
    if (begin < 0) begin += length;
    if (finish < 0) finish += length;
    begin = std::max<int64_t>(0, std::min<int64_t>(begin, length));
    finish = std::max<int64_t>(0, std::min<int64_t>(finish, length));
    if (finish < begin) finish = begin;
    Tensor output = detail::clone_impl(self);
    Tensor target = output.slice(dim, begin, finish, step);
    check_scatter_source(target, src);
    target.copy_(src);
    return output;
}

Tensor diagonal_scatter_cuda(const Tensor& self, const Tensor& src,
                             int64_t offset, int64_t dim1, int64_t dim2) {
    Tensor output = detail::clone_impl(self);
    Tensor target = output.diagonal(offset, dim1, dim2);
    check_scatter_source(target, src);
    target.copy_(src);
    return output;
}


TENSORPLAY_LIBRARY_IMPL(CUDA, ShapeOpsKernels) {
    m.impl("trace", trace_cuda);
    m.impl("diag", diag_cuda);
    m.impl("diag_embed", diag_embed_cuda);
    m.impl("narrow", narrow_cuda);
    m.impl("split_with_sizes", split_with_sizes_cuda);
    m.impl("flip", flip_cuda);
    m.impl("roll", roll_cuda);
    m.impl("rot90", rot90_cuda);
    m.impl("meshgrid", meshgrid_cuda);
    m.impl("broadcast_tensors", broadcast_tensors_cuda);
    m.impl("block_diag", block_diag_cuda);
    m.impl("pixel_shuffle", pixel_shuffle_cuda);
    m.impl("pixel_unshuffle", pixel_unshuffle_cuda);
    m.impl("channel_shuffle", channel_shuffle_cuda);
    m.impl("unfold", unfold_cuda);
    m.impl("unfold_backward", unfold_backward_cuda);
    m.impl("select_scatter", select_scatter_cuda);
    m.impl("slice_scatter", slice_scatter_cuda);
    m.impl("diagonal_scatter", diagonal_scatter_cuda);
}

} // namespace cuda
} // namespace tensorplay
