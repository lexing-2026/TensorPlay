// Tensor shape and layout operators - CPU kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Utils.h"
#include "Exception.h"
#include "Parallel.h"
#include "Quantizer.h"
#include "TypePromotion.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cpu {
using namespace tensorplay::parallel;

namespace {

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

inline int64_t checked_pixel_factor(int64_t factor, const char* op) {
    if (factor <= 0) {
        TP_THROW(RuntimeError, op, " expects a positive factor, but got ", factor);
    }
    if (factor > std::numeric_limits<int64_t>::max() / factor) {
        TP_THROW(ValueError, op, ": factor is too large");
    }
    return factor * factor;
}

inline int64_t checked_pixel_extent(int64_t extent, int64_t factor,
                                    const char* op) {
    if (extent > std::numeric_limits<int64_t>::max() / factor) {
        TP_THROW(ValueError, op, ": output dimension is too large");
    }
    return extent * factor;
}

inline int64_t checked_diagonal_magnitude(int64_t offset, const char* op) {
    if (offset == std::numeric_limits<int64_t>::min()) {
        TP_THROW(ValueError, op, ": diagonal offset is too small");
    }
    return offset < 0 ? -offset : offset;
}

inline int64_t checked_diagonal_extent(int64_t base, int64_t offset,
                                       const char* op) {
    const int64_t magnitude = checked_diagonal_magnitude(offset, op);
    if (base > std::numeric_limits<int64_t>::max() - magnitude) {
        TP_THROW(ValueError, op, ": diagonal output dimension is too large");
    }
    return base + magnitude;
}

Tensor empty_transform_output(const Tensor& self) {
    const auto shape = static_cast<std::vector<int64_t>>(self.shape());
    if (!isQuantizedType(self.dtype())) {
        return Tensor::empty(shape, self.dtype(), self.device());
    }
    quantized::require_quantized(self, "roll");
    Tensor codes = Tensor::empty(shape, underlying_storage_type(self.dtype()),
                                 self.device());
    return quantized::make_qtensor(codes, quantized::quantizer_of(self),
                                   self.dtype());
}

template <typename T>
void trace_cpu_typed(const Tensor& self, Tensor& result) {
    const T* data = self.data_ptr<T>();
    const int64_t diagonal_size = std::min(self.size(0), self.size(1));
    const int64_t diagonal_stride = self.stride(0) + self.stride(1);
    if constexpr (std::is_integral_v<T>) {
        using AccT = std::conditional_t<std::is_unsigned_v<T>, uint64_t, int64_t>;
        AccT sum = 0;
        for (int64_t i = 0; i < diagonal_size; ++i)
            sum += static_cast<AccT>(data[i * diagonal_stride]);
        *result.data_ptr<int64_t>() = static_cast<int64_t>(sum);
    } else if constexpr (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
        float sum = 0.0f;
        for (int64_t i = 0; i < diagonal_size; ++i)
            sum += static_cast<float>(data[i * diagonal_stride]);
        *result.data_ptr<T>() = static_cast<T>(sum);
    } else {
        T sum{};
        for (int64_t i = 0; i < diagonal_size; ++i)
            sum += data[i * diagonal_stride];
        *result.data_ptr<T>() = sum;
    }
}


// ===========================================================================
// Shape ops
// ===========================================================================

Tensor trace_cpu(const Tensor& self) {
    if (self.dim() != 2) {
        TP_THROW(RuntimeError, "trace: expected a matrix, but got tensor with dim ", self.dim());
    }
    const DType out_dtype =
        isIntegralType(self.dtype(), true) ? DType::Int64 : self.dtype();
    Tensor result = Tensor::empty({}, out_dtype, self.device());
#define TP_TRACE_CASE(ctype, name_) \
    case DType::name_: \
        trace_cpu_typed<ctype>(self, result); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_TRACE_CASE)
        default: TP_THROW(TypeError, "trace: unsupported dtype");
    }
#undef TP_TRACE_CASE
    return result;
}

Tensor diag_cpu(const Tensor& self, int64_t diagonal) {
    int64_t nd = self.dim();
    Tensor sc = self.contiguous();
    if (nd == 1) {
        int64_t n = sc.size(0);
        int64_t size = checked_diagonal_extent(n, diagonal, "diag");
        Tensor outc = Tensor::zeros({size, size}, sc.dtype(), sc.device());
        switch (sc.dtype()) {
#define TP_DIAG_FILL(ctype, name_) \
    case DType::name_: { \
        const ctype* s = sc.data_ptr<ctype>(); \
        ctype* d = outc.data_ptr<ctype>(); \
        for (int64_t i = 0; i < n; ++i) { \
            int64_t r = diagonal >= 0 ? i : i - diagonal; \
            int64_t c = diagonal >= 0 ? i + diagonal : i; \
            d[r * size + c] = s[i]; \
        } \
        break; \
    }
            TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_DIAG_FILL)
#undef TP_DIAG_FILL
            default: TP_THROW(TypeError, "diag: unsupported dtype");
        }
        return outc;
    }
    if (nd == 2) {
        int64_t rows = sc.size(0), cols = sc.size(1);
        const int64_t offset_abs = checked_diagonal_magnitude(diagonal, "diag");
        const int64_t row_start = diagonal < 0 ? offset_abs : 0;
        const int64_t col_start = diagonal > 0 ? offset_abs : 0;
        const int64_t diagonal_size =
            row_start >= rows || col_start >= cols
                ? 0
                : std::min(rows - row_start, cols - col_start);
        std::vector<int64_t> idx;
        idx.reserve(static_cast<size_t>(diagonal_size));
        for (int64_t i = 0; i < diagonal_size; ++i)
            idx.push_back((row_start + i) * cols + col_start + i);
        Tensor out = Tensor::zeros({static_cast<int64_t>(idx.size())}, sc.dtype(), sc.device());
        switch (sc.dtype()) {
#define TP_DIAG_EX(ctype, name_) \
    case DType::name_: { \
        const ctype* s = sc.data_ptr<ctype>(); \
        ctype* d = out.data_ptr<ctype>(); \
        for (size_t k = 0; k < idx.size(); ++k) d[k] = s[idx[k]]; \
        break; \
    }
            TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_DIAG_EX)
#undef TP_DIAG_EX
            default: TP_THROW(TypeError, "diag: unsupported dtype");
        }
        return out;
    }
    TP_THROW(RuntimeError, "diag: input must be 1-D or 2-D");
}

Tensor diag_embed_cpu(const Tensor& self, int64_t offset, int64_t dim1_, int64_t dim2_) {
    int64_t nDims = self.dim() + 1;
    int64_t dim1 = wrap_dim(dim1_, nDims);
    int64_t dim2 = wrap_dim(dim2_, nDims);
    if (dim1 == dim2) TP_THROW(RuntimeError, "diagonal dimensions cannot be identical");
    int64_t new_dim_len =
        checked_diagonal_extent(self.size(-1), offset, "diag_embed");
    const Size self_shape = self.shape();
    std::vector<int64_t> sizes(self_shape.begin(), self_shape.end());
    sizes.pop_back();
    sizes.insert(sizes.begin() + std::min(dim1, dim2), new_dim_len);
    sizes.insert(sizes.begin() + std::max(dim1, dim2), new_dim_len);
    Tensor result = Tensor::zeros(sizes, self.dtype(), self.device());
    result.diagonal(offset, dim1, dim2).copy_(self);
    return result;
}

Tensor narrow_cpu(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
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

std::vector<Tensor> split_with_sizes_cpu(const Tensor& self, std::vector<int64_t> split_sizes, int64_t dim) {
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

std::vector<Tensor> tensor_split_cpu(const Tensor& self, int64_t sections, int64_t dim) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (sections <= 0) TP_THROW(RuntimeError, "tensor_split: number of sections must be larger than 0");
    int64_t size = self.size(dim);
    int64_t chunk_base = size / sections, chunk_rem = size % sections;
    std::vector<Tensor> outs;
    int64_t start = 0;
    for (int64_t i = 0; i < sections; ++i) {
        int64_t len = chunk_base + (i < chunk_rem ? 1 : 0);
        if (len > 0) outs.push_back(narrow_cpu(self, dim, start, len));
        else outs.emplace_back();
        start += len;
    }
    return outs;
}

Tensor flip_cpu(const Tensor& self, const std::vector<int64_t>& dims) {
    // flip: negative dims wrap; repeated dims are rejected.
    // wraps with wrap_scalar=true and rejects duplicate dims, then reverses
    // each listed dim.
    int64_t nd = self.dim();
    std::vector<bool> seen(nd > 0 ? nd : 1, false);
    std::vector<bool> flip_mask(nd, false);
    for (auto d : dims) {
        int64_t w = wrap_dim_scalar(d, nd);
        if (nd > 0) {
            if (seen[w]) {
                TP_THROW(RuntimeError, "dim ", w,
                         " appears multiple times in the list of dims");
            }
            seen[w] = true;
            flip_mask[w] = true;
        }
    }
    Tensor sc = self.contiguous();
    Tensor out = detail::clone_impl(self);
    const auto out_strides = static_cast<std::vector<int64_t>>(out.strides());
    int64_t n = sc.numel();
    auto worker = [&](int64_t b, int64_t e) {
        for (int64_t li = b; li < e; ++li) {
            // Decode output linear index and map flipped coordinates back to
            // the source offset.
            int64_t r2 = li, srco = 0, outo = 0, mult = 1;
            for (int64_t d2 = nd - 1; d2 >= 0; --d2) {
                int64_t c = r2 % sc.size(d2);
                r2 /= sc.size(d2);
                int64_t sc3 = flip_mask[d2] ? (sc.size(d2) - 1 - c) : c;
                srco += sc3 * mult;
                outo += c * out_strides[d2];
                mult *= sc.size(d2);
            }
            switch (sc.dtype()) {
#define TP_FLIP_W(ctype, name_) \
    case DType::name_: reinterpret_cast<ctype*>(out.data_ptr())[outo] = reinterpret_cast<const ctype*>(sc.data_ptr())[srco]; break;
                TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_FLIP_W)
                TENSORPLAY_FORALL_QINT_TYPES(TP_FLIP_W)
#undef TP_FLIP_W
                default: break;
            }
        }
    };
    parallel_for(0, n, GRAIN_SIZE, worker);
    return out;
}

Tensor roll_cpu(const Tensor& self, const std::vector<int64_t>& shifts, const std::vector<int64_t>& dims) {
    // roll: shifts wrap around each dimension; slices are concatenated.
    if (dims.size() != 1 || shifts.size() != 1) {
        if (shifts.empty()) TP_THROW(RuntimeError, "`shifts` required");
        if (dims.empty() && shifts.size() == 1) {
            // Flatten-roll: roll the flattened tensor and view back.
            Tensor flat = self.contiguous().reshape({self.numel()});
            Tensor rolled = roll_cpu(flat, {shifts[0]}, {0});
            return rolled.reshape(static_cast<std::vector<int64_t>>(self.shape()));
        }
        if (shifts.size() != dims.size()) {
            TP_THROW(RuntimeError, "shifts and dimensions must align. shifts: ",
                     shifts.size(), ", dims:", dims.size());
        }
        Tensor cur = self;
        for (size_t i = 0; i < dims.size(); ++i) {
            cur = roll_cpu(cur, {shifts[i]}, {dims[i]});
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
    const int64_t shift = ((shifts[0] % size) + size) % size;
    const int64_t start = (size - shift) % size;
    // Equivalent to cat({narrow(dim, start, size-start), narrow(dim, 0, start)}):
    // destination coord c along dim reads source coord (c + start) % size.
    Tensor sc = self.contiguous();
    Tensor out = empty_transform_output(sc);
    int64_t n = sc.numel();
    auto worker = [&](int64_t b, int64_t e) {
        for (int64_t li = b; li < e; ++li) {
            int64_t r2 = li, src = 0, mult = 1;
            for (int64_t d2 = nd - 1; d2 >= 0; --d2) {
                int64_t c = r2 % sc.size(d2);
                r2 /= sc.size(d2);
                int64_t sc3 = d2 == dim ? (c + start) % size : c;
                src += sc3 * mult;
                mult *= sc.size(d2);
            }
            switch (sc.dtype()) {
#define TP_ROLL_W(ctype, name_) case DType::name_: reinterpret_cast<ctype*>(out.data_ptr())[li] = reinterpret_cast<const ctype*>(sc.data_ptr())[src]; break;
                TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_ROLL_W)
                TENSORPLAY_FORALL_QINT_TYPES(TP_ROLL_W)
#undef TP_ROLL_W
                default: break;
            }
        }
    };
    parallel_for(0, n, GRAIN_SIZE, worker);
    return out;
}

Tensor rot90_cpu(const Tensor& self, int64_t k, const std::vector<int64_t>& dims) {
    // rot90: k quarter-turns in the plane spanned by the two axes.
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
        case 1: return transpose_view(flip_cpu(self, {dim1}), dim0, dim1);
        case 2: return flip_cpu(self, {dim0, dim1});
        case 3: return transpose_view(flip_cpu(self, {dim0}), dim0, dim1);
        default: return detail::contiguous_clone(self);
    }
}

std::vector<Tensor> meshgrid_cpu(const std::vector<Tensor>& tensors, const std::string& indexing) {
    size_t k = tensors.size();
    if (k == 0) {
        TP_THROW(RuntimeError, "meshgrid expects a non-empty TensorList");
    }
    if (indexing != "ij" && indexing != "xy") {
        TP_THROW(RuntimeError, "meshgrid: indexing must be 'ij' or 'xy', got " + indexing);
    }
    const Device& device = tensors[0].device();
    for (size_t i = 0; i < k; ++i) {
        const Tensor& t = tensors[i];
        if (!(t.device() == device)) {
            TP_THROW(RuntimeError, "meshgrid expects all tensors to have the same device");
        }
        if (t.dim() > 1) {
            TP_THROW(RuntimeError,
                     "meshgrid: expected 0-D or 1-D tensors");
        }
    }
    for (size_t i = 1; i < k; ++i) {
        if (tensors[i].dtype() != tensors[0].dtype()) {
            TP_THROW(RuntimeError, "meshgrid expects all tensors to have the same dtype");
        }
    }
    std::vector<Tensor> order(tensors.begin(), tensors.end());
    if (indexing == "xy" && k >= 2) {
        std::swap(order[0], order[1]);
    }
    std::vector<int64_t> sizes;
    sizes.reserve(k);
    for (const Tensor& t : order) sizes.push_back(t.numel());
    std::vector<Tensor> grids;
    grids.reserve(k);
    std::vector<int64_t> view_shape(k, 1);
    for (size_t i = 0; i < k; ++i) {
        view_shape[i] = sizes[i];
        grids.push_back(order[i].view(view_shape).expand(sizes));
        view_shape[i] = 1;
    }
    if (indexing == "xy" && k >= 2) {
        std::swap(grids[0], grids[1]);
    }
    return grids;
}

std::vector<Tensor> broadcast_tensors_cpu(const std::vector<Tensor>& tensors) {
    // common broadcast shape.  Returns stride-0 views; gradients flow through
    // the dispatcher expand op (sum-to-size backward).
    std::vector<int64_t> shape{};
    for (auto& t : tensors) {
        const Size t_shape = t.shape();
        std::vector<int64_t> ts(t_shape.begin(), t_shape.end());
        shape = broadcast_shapes(shape, ts);
    }
    std::vector<Tensor> outs;
    outs.reserve(tensors.size());
    for (auto& t : tensors) {
        const Size t_shape = t.shape();
        std::vector<int64_t> ts(t_shape.begin(), t_shape.end());
        if (ts == shape) { outs.push_back(t); continue; }
        outs.push_back(t.expand(shape));
    }
    return outs;
}


Tensor block_diag_cpu(const std::vector<Tensor>& tensors) {
    // 2-D rectangular blocks; result dtype = promoted inputs; empty call
    // yields a (1, 0) tensor.
    if (tensors.empty()) {
        return Tensor::empty(
            std::vector<int64_t>{1, 0},
            std::optional<DType>(DType::Float32),
            std::nullopt,
            false);
    }
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

Tensor pixel_shuffle_cpu(const Tensor& self, int64_t upscale_factor) {
    if (self.dim() < 3) {
        TP_THROW(RuntimeError, "pixel_shuffle expects input to have at least 3 dimensions, but got input with ",
                 self.dim(), " dimension(s)");
    }
    const int64_t factor_squared = checked_pixel_factor(upscale_factor, "pixel_shuffle");
    const int64_t input_channels = self.size(-3);
    if (input_channels % factor_squared != 0)
        TP_THROW(RuntimeError, "pixel_shuffle: channel dim must be divisible by r^2");
    const int64_t output_height =
        checked_pixel_extent(self.size(-2), upscale_factor, "pixel_shuffle");
    const int64_t output_width =
        checked_pixel_extent(self.size(-1), upscale_factor, "pixel_shuffle");
    std::vector<int64_t> output_shape =
        static_cast<std::vector<int64_t>>(self.shape());
    output_shape.resize(output_shape.size() - 3);
    output_shape.push_back(input_channels / factor_squared);
    output_shape.push_back(output_height);
    output_shape.push_back(output_width);
    Tensor out = Tensor::empty(output_shape, self.dtype(), self.device());
    if (out.numel() == 0) return out;
    Tensor sc = self.contiguous();
    int64_t r = upscale_factor;
    int64_t C = input_channels / factor_squared;
    int64_t H = self.size(-2), W = self.size(-1);
    int64_t n = self.numel();
    auto wk = [&](int64_t b, int64_t e) {
        for (int64_t li = b; li < e; ++li) {
            int64_t rem = li;
            int64_t w = rem % W; rem /= W;
            int64_t h = rem % H; rem /= H;
            int64_t c = rem % C; rem /= C;
            int64_t bn = rem;
            int64_t ih = h % r, iw = w % r;
            int64_t src = (((bn * (C * r * r) + c * r * r + ih * r + iw) * H + (h / r)) * W + (w / r));
            switch (self.dtype()) {
#define TP_PS_W(ctype, name_) case DType::name_: reinterpret_cast<ctype*>(out.data_ptr())[li] = reinterpret_cast<const ctype*>(sc.data_ptr())[src]; break;
                TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_PS_W)
#undef TP_PS_W
                default: break;
            }
        }
    };
    parallel_for(0, n, GRAIN_SIZE, wk);
    return out;
}

Tensor pixel_unshuffle_cpu(const Tensor& self, int64_t downscale_factor) {
    if (self.dim() < 3) {
        TP_THROW(RuntimeError, "pixel_unshuffle expects input to have at least 3 dimensions, but got input with ",
                 self.dim(), " dimension(s)");
    }
    const int64_t factor_squared =
        checked_pixel_factor(downscale_factor, "pixel_unshuffle");
    int64_t r = downscale_factor;
    int64_t C = self.size(-3);
    int64_t H = self.size(-2) / r, W = self.size(-1) / r;
    if (H * r != self.size(-2) || W * r != self.size(-1))
        TP_THROW(RuntimeError, "pixel_unshuffle: spatial dims must be divisible by r");
    if (C > std::numeric_limits<int64_t>::max() / factor_squared)
        TP_THROW(ValueError, "pixel_unshuffle: output channel dimension is too large");
    std::vector<int64_t> output_shape =
        static_cast<std::vector<int64_t>>(self.shape());
    output_shape.resize(output_shape.size() - 3);
    output_shape.push_back(C * factor_squared);
    output_shape.push_back(H);
    output_shape.push_back(W);
    Tensor out = Tensor::empty(output_shape, self.dtype(), self.device());
    if (out.numel() == 0) return out;
    Tensor sc = self.contiguous();
    int64_t n = out.numel();
    auto wk = [&](int64_t b, int64_t e) {
        for (int64_t li = b; li < e; ++li) {
            int64_t rem = li;
            int64_t w = rem % W; rem /= W;
            int64_t h = rem % H; rem /= H;
            int64_t cc = rem % (C * r * r); rem /= (C * r * r);
            int64_t bn = rem;
            int64_t c = cc / (r * r);
            int64_t ij = cc % (r * r);
            int64_t ih = ij / r, iw = ij % r;
            int64_t src = ((((bn * C + c) * (H * r) + h * r + ih) * (W * r)) + w * r + iw);
            switch (self.dtype()) {
#define TP_PU_W(ctype, name_) case DType::name_: reinterpret_cast<ctype*>(out.data_ptr())[li] = reinterpret_cast<const ctype*>(sc.data_ptr())[src]; break;
                TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_PU_W)
#undef TP_PU_W
                default: break;
            }
        }
    };
    parallel_for(0, n, GRAIN_SIZE, wk);
    return out;
}

Tensor channel_shuffle_cpu(const Tensor& self, int64_t groups) {
    if (self.dim() <= 2) {
        TP_THROW(RuntimeError, "channel_shuffle expects input with more than 2 dimensions");
    }
    if (groups <= 0) {
        TP_THROW(RuntimeError, "channel_shuffle: groups must be positive, but got ", groups);
    }
    int64_t C = self.size(1);
    int64_t outer = 1;   // product of dims before cdim
    for (int64_t i = 0; i < 1; ++i) outer *= self.size(i);
    int64_t inner = 1;   // product of dims after cdim
    for (int64_t i = 2; i < self.dim(); ++i) inner *= self.size(i);
    if (C % groups) TP_THROW(RuntimeError, "channel_shuffle: channel dim not divisible by groups");
    int64_t cg = C / groups;
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    if (n == 0) return out;
    auto wk = [&](int64_t b, int64_t e) {
        for (int64_t li = b; li < e; ++li) {
            // li layout: (outer * C + c) * inner + tail
            int64_t tail = li % inner;
            int64_t rest = li / inner;
            int64_t c = rest % C;
            int64_t o = rest / C;
            int64_t j = c / cg, gi = c % cg;
            int64_t src_c = gi * cg + j;
            int64_t src = ((o * C) + src_c) * inner + tail;
            switch (self.dtype()) {
#define TP_CS_W(ctype, name_) case DType::name_: reinterpret_cast<ctype*>(out.data_ptr())[li] = reinterpret_cast<const ctype*>(sc.data_ptr())[src]; break;
                TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_CS_W)
#undef TP_CS_W
                default: break;
            }
        }
    };
    parallel_for(0, n, GRAIN_SIZE, wk);
    return out;
}

Tensor unfold_cpu(const Tensor& self, int64_t dimension, int64_t size, int64_t step) {
    // unfold: an as_strided view.  wrap_scalar=true allows
    // dimension == 0 on 0-d tensors (max_size becomes 1).
    const int64_t nd = self.dim();
    dimension = wrap_dim_scalar(dimension, nd);

    std::vector<int64_t> sizes = static_cast<std::vector<int64_t>>(self.shape());
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

Tensor unfold_backward_cpu(const Tensor& grad, const std::vector<int64_t>& input_sizes,
                           int64_t dim, int64_t size, int64_t step) {
    // window's gradient back onto `dim`, accumulating where windows overlap
    // (step < size).  We gather over grad_input elements (race-free), which
    // degenerates to a plain copy when step >= size.
    if (step <= 0) TP_THROW(RuntimeError, "step is ", step, " but must be > 0");
    Tensor grad_input = Tensor::zeros(input_sizes, grad.dtype(), grad.device());
    const int64_t nd = static_cast<int64_t>(input_sizes.size());
    if (nd == 0) {
        // 0-d input: unfold appended a single axis; the lone element is hit once.
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
#define TP_UFB(ctype, name_) \
    case DType::name_: { \
        const ctype* gp = gc.data_ptr<ctype>(); \
        ctype* gip = grad_input.data_ptr<ctype>(); \
        parallel_for(0, total, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t t = b; t < e; ++t) { \
                int64_t inner_idx = t % inner; \
                int64_t rest = t / inner; \
                int64_t idx_dim = rest % input_dim_size; \
                int64_t outer_idx = rest / input_dim_size; \
                int64_t left = (idx_dim > size) ? (idx_dim - size) / step : 0; \
                if (!(left * step <= idx_dim && idx_dim < left * step + size)) ++left; \
                int64_t right = idx_dim / step; \
                if (right >= count) right = count - 1; \
                ctype acc{}; \
                for (int64_t fold = left; fold <= right; ++fold) { \
                    int64_t j = idx_dim - fold * step; \
                    acc += gp[((outer_idx * count + fold) * inner + inner_idx) * size + j]; \
                } \
                gip[t] = acc; \
            } \
        }); \
        break; \
    }
    switch (grad.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_UFB)
        default: TP_THROW(TypeError, "unfold_backward: unsupported dtype");
    }
#undef TP_UFB
    return grad_input;
}

inline void check_scatter_source(const Tensor& target, const Tensor& src) {
    if (target.shape() != src.shape()) {
        TP_THROW(RuntimeError,
                 "expected src to have a size equal to the target slice");
    }
}

Tensor select_scatter_cpu(const Tensor& self, const Tensor& src, int64_t dim,
                          int64_t index) {
    dim = wrap_dim(dim, self.dim());
    Tensor output = detail::clone_impl(self);
    Tensor target = output.select(dim, index);
    check_scatter_source(target, src);
    target.copy_(src);
    return output;
}

Tensor slice_scatter_cpu(const Tensor& self, const Tensor& src, int64_t dim,
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

Tensor diagonal_scatter_cpu(const Tensor& self, const Tensor& src,
                            int64_t offset, int64_t dim1, int64_t dim2) {
    Tensor output = detail::clone_impl(self);
    Tensor target = output.diagonal(offset, dim1, dim2);
    check_scatter_source(target, src);
    target.copy_(src);
    return output;
}


}  // namespace

TENSORPLAY_LIBRARY_IMPL(CPU, ShapeOpsKernels) {
    m.impl("trace", trace_cpu);
    m.impl("diag", diag_cpu);
    m.impl("diag_embed", diag_embed_cpu);
    m.impl("narrow", narrow_cpu);
    m.impl("split_with_sizes", split_with_sizes_cpu);
    m.impl("roll", roll_cpu);
    m.impl("flip", flip_cpu);
    m.impl("rot90", rot90_cpu);
    m.impl("meshgrid", meshgrid_cpu);
    m.impl("broadcast_tensors", broadcast_tensors_cpu);
    m.impl("block_diag", block_diag_cpu);
    m.impl("pixel_shuffle", pixel_shuffle_cpu);
    m.impl("pixel_unshuffle", pixel_unshuffle_cpu);
    m.impl("channel_shuffle", channel_shuffle_cpu);
    m.impl("unfold", unfold_cpu);
    m.impl("unfold_backward", unfold_backward_cpu);
    m.impl("select_scatter", select_scatter_cpu);
    m.impl("slice_scatter", slice_scatter_cpu);
    m.impl("diagonal_scatter", diagonal_scatter_cpu);
}

} // namespace cpu
} // namespace tensorplay
