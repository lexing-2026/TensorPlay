#include "SizesAndStrides.h"
#include "Exception.h"
#include <sstream>
#include <algorithm>

namespace tensorplay {

SizesAndStrides::SizesAndStrides(const std::vector<int64_t>& sizes) {
    set_extent(sizes.size());
    if (!sizes.empty()) {
        std::copy(sizes.begin(), sizes.end(), sizes_data());
    }
    auto strides = compute_contiguous_strides(sizes);
    if (!strides.empty()) {
        std::copy(strides.begin(), strides.end(), strides_data());
    }
}

SizesAndStrides::SizesAndStrides(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides) {
    if (sizes.size() != strides.size()) {
        TP_THROW(ValueError, "Sizes and strides must have the same length");
    }
    set_extent(sizes.size());
    if (!sizes.empty()) {
        std::copy(sizes.begin(), sizes.end(), sizes_data());
        std::copy(strides.begin(), strides.end(), strides_data());
    }
}

SizesAndStrides::SizesAndStrides(const SizesAndStrides& other)
    : size_(other.size_),
      inline_sizes_{},
      inline_strides_{},
      heap_sizes_(other.is_heap() ? std::unique_ptr<int64_t[]>(new int64_t[other.size_]) : nullptr),
      heap_strides_(other.is_heap() ? std::unique_ptr<int64_t[]>(new int64_t[other.size_]) : nullptr) {
    if (size_ == 0) return;
    if (is_heap()) {
        std::copy(other.sizes_data(), other.sizes_data() + size_, heap_sizes_.get());
        std::copy(other.strides_data(), other.strides_data() + size_, heap_strides_.get());
    } else {
        std::copy(other.inline_sizes_, other.inline_sizes_ + size_, inline_sizes_);
        std::copy(other.inline_strides_, other.inline_strides_ + size_, inline_strides_);
    }
}

SizesAndStrides::SizesAndStrides(SizesAndStrides&& other) noexcept
    : size_(other.size_),
      inline_sizes_{},
      inline_strides_{},
      heap_sizes_(std::move(other.heap_sizes_)),
      heap_strides_(std::move(other.heap_strides_)) {
    if (!is_heap() && size_ > 0) {
        std::copy(other.inline_sizes_, other.inline_sizes_ + size_, inline_sizes_);
        std::copy(other.inline_strides_, other.inline_strides_ + size_, inline_strides_);
    }
}

SizesAndStrides& SizesAndStrides::operator=(const SizesAndStrides& other) {
    if (this != &other) {
        SizesAndStrides tmp(other);
        swap(*this, tmp);
    }
    return *this;
}

SizesAndStrides& SizesAndStrides::operator=(SizesAndStrides&& other) noexcept {
    if (this != &other) {
        SizesAndStrides tmp(std::move(other));
        swap(*this, tmp);
    }
    return *this;
}

void SizesAndStrides::set_extent(size_t new_size) {
    const bool was_heap = is_heap();
    const bool will_be_heap = new_size > kInlineSize;
    if (will_be_heap) {
        if (!was_heap || size_ != new_size) {
            heap_sizes_.reset(new int64_t[new_size]);
            heap_strides_.reset(new int64_t[new_size]);
        }
    } else if (was_heap) {
        heap_sizes_.reset();
        heap_strides_.reset();
    }
    size_ = new_size;
}

void SizesAndStrides::resize(const std::vector<int64_t>& new_sizes) {
    set_extent(new_sizes.size());
    if (!new_sizes.empty()) {
        std::copy(new_sizes.begin(), new_sizes.end(), sizes_data());
    }
    auto strides = compute_contiguous_strides(new_sizes);
    if (!strides.empty()) {
        std::copy(strides.begin(), strides.end(), strides_data());
    }
}

void SizesAndStrides::set_sizes_and_strides(const std::vector<int64_t>& new_sizes, const std::vector<int64_t>& new_strides) {
    if (new_sizes.size() != new_strides.size()) {
        TP_THROW(ValueError, "Sizes and strides must have the same length");
    }

    set_extent(new_sizes.size());
    if (!new_sizes.empty()) {
        std::copy(new_sizes.begin(), new_sizes.end(), sizes_data());
        std::copy(new_strides.begin(), new_strides.end(), strides_data());
    }
}

void SizesAndStrides::set_size(size_t dim, int64_t new_size) {
    if (dim >= size_) {
        TP_THROW(IndexError, "Dimension out of range");
    }

    sizes_data()[dim] = new_size;
    // Note: We don't automatically recompute strides here to preserve custom strides
}

void SizesAndStrides::set_stride(size_t dim, int64_t new_stride) {
    if (dim >= size_) {
        TP_THROW(IndexError, "Dimension out of range");
    }

    strides_data()[dim] = new_stride;
}

int64_t SizesAndStrides::numel() const {
    if (size_ == 0) {
        // 0-dimensional tensor has 1 element (scalar)
        return 1;
    }

    int64_t result = 1;
    for (size_t i = 0; i < size_; ++i) {
        result *= sizes_data()[i];
    }

    return result;
}

bool SizesAndStrides::is_contiguous() const {
    if (size_ == 0) {
        return true;
    }

    // contiguous regardless of strides.
    for (size_t i = 0; i < size_; ++i) {
        if (sizes_data()[i] == 0) return true;
    }

    // Walk expected contiguous strides without materializing them. A
    int64_t expected = 1;
    for (size_t i = size_; i > 0; --i) {
        if (strides_data()[i - 1] != expected && sizes_data()[i - 1] != 1) {
            return false;
        }
        if (sizes_data()[i - 1] != 1) {
            expected *= sizes_data()[i - 1];
        }
    }

    return true;
}

std::vector<int64_t> SizesAndStrides::compute_contiguous_strides(const std::vector<int64_t>& sizes) {
    std::vector<int64_t> strides(sizes.size());

    if (sizes.empty()) {
        return strides;
    }

    // dim, never collapsing across a zero-sized dim -- stride[i] is
    // stride[i+1] * max(size[i+1], 1), so e.g. shape (2, 0) gets (1, 1).
    strides.back() = 1;
    for (int64_t i = static_cast<int64_t>(sizes.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * std::max<int64_t>(sizes[i + 1], 1);
    }

    return strides;
}

// 1. separate `oldshape` into chunks of dimensions that are contiguous within
//    each chunk, i.e. oldstride[i] == oldshape[i+1] * oldstride[i+1]
// 2. `newshape` must split into the same number of chunks, each with matching
//    numel.  Size-1 dims are skipped, so non-contiguous layouts that still
//    admit the view (same-shape views, subspace views) succeed.
std::optional<std::vector<int64_t>> SizesAndStrides::compute_view_strides(
    const std::vector<int64_t>& oldshape,
    const std::vector<int64_t>& oldstride,
    const std::vector<int64_t>& newshape) {
    if (oldshape.empty()) {
        return std::vector<int64_t>(newshape.size(), 1);
    }

    int64_t numel = 1;
    for (int64_t s : oldshape) numel *= s;

    // NOTE: stride is arbitrary in the numel() == 0 case; to match NumPy
    // behavior we copy the strides if the size matches, otherwise we use the
    // stride as if it were computed via resize.
    const bool zero_numel = (numel == 0);
    if (zero_numel && oldshape == newshape) {
        return oldstride;
    }

    std::vector<int64_t> newstride(newshape.size());
    if (zero_numel) {
        for (int64_t view_d = static_cast<int64_t>(newshape.size()) - 1; view_d >= 0; --view_d) {
            if (view_d == static_cast<int64_t>(newshape.size()) - 1) {
                newstride[view_d] = 1;
            } else {
                newstride[view_d] =
                    std::max<int64_t>(newshape[view_d + 1], 1) * newstride[view_d + 1];
            }
        }
        return newstride;
    }

    int64_t view_d = static_cast<int64_t>(newshape.size()) - 1;
    // stride for each subspace in the chunk
    int64_t chunk_base_stride = oldstride.back();
    // numel in current chunk
    int64_t tensor_numel = 1;
    int64_t view_numel = 1;
    for (int64_t tensor_d = static_cast<int64_t>(oldshape.size()) - 1; tensor_d >= 0; --tensor_d) {
        tensor_numel *= oldshape[tensor_d];
        // if end of tensor size chunk, check view
        if ((tensor_d == 0) ||
            (oldshape[tensor_d - 1] != 1 &&
             oldstride[tensor_d - 1] != tensor_numel * chunk_base_stride)) {
            while (view_d >= 0 &&
                   (view_numel < tensor_numel || newshape[view_d] == 1)) {
                newstride[view_d] = view_numel * chunk_base_stride;
                view_numel *= newshape[view_d];
                --view_d;
            }
            if (view_numel != tensor_numel) {
                return std::nullopt;
            }
            if (tensor_d > 0) {
                chunk_base_stride = oldstride[tensor_d - 1];
                tensor_numel = 1;
                view_numel = 1;
            }
        }
    }
    if (view_d != -1) {
        return std::nullopt;
    }
    return newstride;
}

namespace {
std::string shape_str(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}
} // namespace

std::vector<int64_t> SizesAndStrides::infer_size(const std::vector<int64_t>& shape, int64_t numel) {
    std::vector<int64_t> inferred = shape;
    int64_t newsize = 1;
    int64_t infer_dim = -1;
    for (size_t dim = 0; dim < inferred.size(); ++dim) {
        if (inferred[dim] == -1) {
            if (infer_dim != -1) TP_THROW(RuntimeError, "only one dimension can be inferred");
            infer_dim = static_cast<int64_t>(dim);
        } else {
            if (inferred[dim] < -1) {
                TP_THROW(RuntimeError, "invalid shape dimension " + std::to_string(inferred[dim]));
            }
            newsize *= inferred[dim];
        }
    }

    if (infer_dim != -1) {
        if (!((newsize > 0 && numel % newsize == 0) || numel == newsize)) {
            TP_THROW(RuntimeError, "shape '" + shape_str(shape) +
                     "' is invalid for input of size " + std::to_string(numel));
        }
        if (newsize == 0) {
            TP_THROW(RuntimeError, "cannot reshape tensor of 0 elements into shape " +
                     shape_str(shape) +
                     " because the unspecified dimension size -1 can be any value and is ambiguous");
        }
        inferred[infer_dim] = numel / newsize;
    } else if (numel != newsize) {
        TP_THROW(RuntimeError, "shape '" + shape_str(shape) +
                 "' is invalid for input of size " + std::to_string(numel));
    }
    return inferred;
}

// Sort dimensions by stride (size-0/1 dimensions sink to the end) and
// require each remaining dimension to pick up exactly the running product.
bool SizesAndStrides::is_non_overlapping_and_dense(
    const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides) {
    const int64_t dim = static_cast<int64_t>(sizes.size());
    if (dim == 1) {
        return sizes[0] < 2 || strides[0] == 1;
    }
    std::vector<int64_t> perm(dim);
    for (int64_t i = 0; i < dim; ++i) perm[static_cast<size_t>(i)] = i;
    std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
        if (sizes[static_cast<size_t>(a)] < 2) {
            return false;
        } else if (sizes[static_cast<size_t>(b)] < 2) {
            return true;
        }
        return strides[static_cast<size_t>(a)] < strides[static_cast<size_t>(b)];
    });
    int64_t require_stride = 1;
    for (int64_t i = 0; i < dim; ++i) {
        const int64_t size_perm_i = sizes[static_cast<size_t>(perm[static_cast<size_t>(i)])];
        if (size_perm_i < 2) {
            return true;
        }
        if (strides[static_cast<size_t>(perm[static_cast<size_t>(i)])] != require_stride) {
            return false;
        }
        require_stride *= size_perm_i;
    }
    return true;
}

std::string SizesAndStrides::toString() const {
    std::ostringstream oss;
    oss << "SizesAndStrides(sizes=[";
    for (size_t i = 0; i < size_; ++i) {
        if (i > 0) oss << ", ";
        oss << sizes_data()[i];
    }
    oss << "], strides=[";
    for (size_t i = 0; i < size_; ++i) {
        if (i > 0) oss << ", ";
        oss << strides_data()[i];
    }
    oss << "])";
    return oss.str();
}

} // namespace tensorplay
