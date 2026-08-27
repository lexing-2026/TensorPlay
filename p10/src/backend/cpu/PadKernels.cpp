#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "TypePromotion.h"
#include "Utils.h"
#include "Parallel.h"
#include <cstring>
#include <vector>
#include <algorithm>

namespace tensorplay {
namespace cpu {
using namespace tensorplay::parallel;

// Port of at::native::constant_pad_nd (aten/src/ATen/native/PadNd.cpp:29).
// Composite over narrow/fill_/copy_ so one body serves CPU and CUDA; negative
// pads crop the input through narrow, positive pads fill an output canvas.
Tensor constant_pad_nd_cpu(const Tensor& self, const std::vector<int64_t>& pad, Scalar value) {
    auto input_sizes = self.shape();
    int64_t l_inp = self.dim();
    int64_t l_pad = static_cast<int64_t>(pad.size()) / 2;

    if (pad.size() % 2 != 0) {
         TP_THROW(ValueError, "Length of pad must be even but instead it equals ", pad.size());
    }
    if (l_inp < l_pad) {
         TP_THROW(ValueError, "Length of pad should be no more than twice the number of "
                  "dimensions of the input. Pad length is ", pad.size(), " while the input has ",
                  l_inp, " dimensions.");
    }

    bool all_pads_non_positive = true;
    Tensor c_input = self;
    for (int64_t i = l_inp - l_pad; i < l_inp; ++i) {
        int64_t pad_idx = 2 * (l_inp - i - 1);
        if (pad[pad_idx] < 0) {
            c_input = c_input.slice(i, -pad[pad_idx], c_input.size(i));
        } else if (pad[pad_idx] != 0) {
            all_pads_non_positive = false;
        }
        if (pad[pad_idx + 1] < 0) {
            c_input = c_input.slice(i, 0, c_input.size(i) + pad[pad_idx + 1]);
        } else if (pad[pad_idx + 1] != 0) {
            all_pads_non_positive = false;
        }
    }

    // if none of the pads are positive we can optimize and just return the result
    // of calling .narrow() on the input
    if (all_pads_non_positive) {
        return c_input.clone();
    }

    std::vector<int64_t> new_shape;
    new_shape.reserve(l_inp - l_pad);
    for (int64_t i = 0; i < l_inp - l_pad; ++i) {
        new_shape.push_back(input_sizes[i]);
    }

    for (int64_t i = 0; i < l_pad; ++i) {
        size_t pad_idx = pad.size() - ((i + 1) * 2);
        int64_t new_dim = input_sizes[l_inp - l_pad + i] + pad[pad_idx] + pad[pad_idx + 1];
        if (new_dim < 0) {
            TP_THROW(ValueError, "The input size ", input_sizes[l_inp - l_pad + i],
                     ", plus negative padding ", pad[pad_idx], " and ", pad[pad_idx + 1],
                     " resulted in a negative output size, which is invalid. Check dimension ",
                     l_inp - l_pad + i, " of your input.");
        }
        new_shape.push_back(new_dim);
    }

    Tensor output = Tensor::empty(new_shape, self.dtype(), self.device());
    output.fill_(value);

    Tensor c_output = output;
    for (int64_t i = l_inp - l_pad; i < l_inp; ++i) {
        int64_t pad_idx = 2 * (l_inp - i - 1);
        if (pad[pad_idx] > 0) {
            c_output = c_output.slice(i, pad[pad_idx], c_output.size(i));
        }
        if (pad[pad_idx + 1] > 0) {
            c_output = c_output.slice(i, 0, c_output.size(i) - pad[pad_idx + 1]);
        }
    }
    c_output.copy_(c_input);

    return output;
}

// Port of torch::autograd::generated::constant_pad_nd_backward
// (torch/csrc/autograd/FunctionsManual.cpp): the co-gradient is
// constant_pad_nd(grad, -pad, 0), which uniformly zero-fills cropped regions
// and slices padded ones.
Tensor constant_pad_nd_backward_cpu(const Tensor& grad_output, const std::vector<int64_t>& pad) {
    std::vector<int64_t> negated_pad = pad;
    for (auto& p : negated_pad) p = -p;
    return constant_pad_nd_cpu(grad_output, negated_pad, 0);
}

// ===========================================================================
// Non-constant padding modes used by nn.Conv* padding_mode and F.pad
// (torch: reflection_pad1d/2d/3d, replication_pad1d/2d/3d, circular_pad1d/2d/3d;
// TensorPlay keeps the *_pad_nd spelling of its existing constant_pad_nd and
// handles any rank in one kernel).
// ===========================================================================
namespace {

// Maps a padded coordinate q back to a source coordinate for one padded dim.
// Reflection mirrors once (the op layer enforces torch's pad < dim-size
// rule), replication clamps to the edges, circular wraps modulo the dim
// size -- the same index math as aten's ReflectionPad / ReplicationPadding
// and the pad+roll composite used for circular.
enum class PadIndexMode { Reflect, Replicate, Circular };

inline int64_t pad_map_coord(int64_t q, int64_t left, int64_t size, PadIndexMode mode) {
    int64_t src = q - left;
    if (src >= 0 && src < size) return src;
    switch (mode) {
        case PadIndexMode::Reflect:
            if (src < 0) src = -src;
            else src = 2 * size - 2 - src;
            return src;
        case PadIndexMode::Replicate:
            return std::clamp(src, int64_t(0), size - 1);
        case PadIndexMode::Circular:
        default:
            return ((src % size) + size) % size;
    }
}

struct PadNdPlan {
    int64_t ndim = 0;
    int64_t k_padded = 0;              // number of padded dims (the last k)
    std::vector<int64_t> src_sizes;
    std::vector<int64_t> src_strides;  // contiguous strides of self
    std::vector<int64_t> dst_sizes;
    std::vector<int64_t> dst_strides;
    std::vector<int64_t> pad_pairs;    // torch order: [l(last), r(last), ...]
};

inline PadNdPlan pad_plan(const Tensor& self, const std::vector<int64_t>& pad) {
    PadNdPlan p;
    p.ndim = self.dim();
    p.pad_pairs = pad;
    p.k_padded = static_cast<int64_t>(pad.size()) / 2;
    p.src_sizes = self.shape();
    p.src_strides.assign(p.ndim, 1);
    for (int64_t d = p.ndim - 2; d >= 0; --d)
        p.src_strides[d] = p.src_strides[d + 1] * p.src_sizes[d + 1];
    p.dst_sizes = p.src_sizes;
    for (int64_t i = 0; i < p.k_padded; ++i)
        p.dst_sizes[p.ndim - 1 - i] += pad[2 * i] + pad[2 * i + 1];
    p.dst_strides.assign(p.ndim, 1);
    for (int64_t d = p.ndim - 2; d >= 0; --d)
        p.dst_strides[d] = p.dst_strides[d + 1] * p.dst_sizes[d + 1];
    return p;
}

template <typename T, PadIndexMode mode>
static void pad_gather_kernel(const T* src, T* dst, const PadNdPlan& p) {
    const int64_t total =
        p.dst_strides.empty() ? 1
                              : p.dst_strides.front() * p.dst_sizes.front();
    parallel_for(0, total, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
            int64_t rem = idx;
            int64_t src_off = 0;
            for (int64_t d = 0; d < p.ndim; ++d) {
                const int64_t q = rem / p.dst_strides[d];
                rem -= q * p.dst_strides[d];
                int64_t s = q;
                if (d >= p.ndim - p.k_padded) {
                    const int64_t pair = p.ndim - 1 - d;
                    s = pad_map_coord(q, p.pad_pairs[2 * pair], p.src_sizes[d], mode);
                }
                src_off += s * p.src_strides[d];
            }
            dst[idx] = src[src_off];
        }
    });
}

// Adjoint: iterate the padded positions of one "outer" (non-padded) prefix
// serially so every write to a given grad-input element stays on one thread.
template <typename T, PadIndexMode mode>
static void pad_scatter_kernel(const T* go, T* gi, const PadNdPlan& p) {
    const int64_t k_outer = p.ndim - p.k_padded;
    int64_t outer = 1;
    for (int64_t d = 0; d < k_outer; ++d) outer *= p.src_sizes[d];
    int64_t src_total = 1;
    for (int64_t d = 0; d < p.ndim; ++d) src_total *= p.src_sizes[d];
    std::memset(gi, 0, src_total * sizeof(T));
    // ``o`` is an ordinal over the OUTER product only (dims [0, k_outer)),
    // so it must be decoded with outer-only strides -- dividing by the
    // full-tensor strides would fold the padded dims' sizes into the outer
    // coordinates and misroute every non-trivial batch/channel slice.
    std::vector<int64_t> outer_strides(k_outer, 1);
    for (int64_t d = k_outer - 2; d >= 0; --d)
        outer_strides[d] = outer_strides[d + 1] * p.src_sizes[d + 1];
    parallel_for(0, outer, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t o = begin; o < end; ++o) {
            // decompose the outer prefix once; outer dims are unpadded so the
            // coordinates are identical in src and dst
            int64_t rem = o;
            int64_t src_outer = 0, dst_outer = 0;
            for (int64_t d = 0; d < k_outer; ++d) {
                const int64_t c = rem / outer_strides[d];
                rem -= c * outer_strides[d];
                src_outer += c * p.src_strides[d];
                dst_outer += c * p.dst_strides[d];
            }
            // walk every padded-coordinate combination (odometer, last dim
            // fastest); k_padded <= 3, so recomputing both offsets from the
            // coordinates each visit is cheap and keeps the loop simple.
            int64_t coord[3] = {0, 0, 0};
            while (true) {
                int64_t src_off = src_outer, dst_off = dst_outer;
                for (int64_t d = k_outer; d < p.ndim; ++d) {
                    const int64_t j = d - k_outer;
                    const int64_t pair = p.ndim - 1 - d;
                    src_off += pad_map_coord(coord[j], p.pad_pairs[2 * pair], p.src_sizes[d],
                                             mode) * p.src_strides[d];
                    dst_off += coord[j] * p.dst_strides[d];
                }
                gi[src_off] += go[dst_off];
                bool done = true;
                for (int64_t d = p.ndim - 1; d >= k_outer; --d) {
                    const int64_t j = d - k_outer;
                    if (++coord[j] < p.dst_sizes[d]) {
                        done = false;
                        break;
                    }
                    coord[j] = 0;
                }
                if (done) break;
            }
        }
    });
}

template <typename T>
void pad_scatter_dispatch(const T* go, T* gi, const PadNdPlan& p, PadIndexMode mode) {
    switch (mode) {
        case PadIndexMode::Reflect: pad_scatter_kernel<T, PadIndexMode::Reflect>(go, gi, p); break;
        case PadIndexMode::Replicate: pad_scatter_kernel<T, PadIndexMode::Replicate>(go, gi, p); break;
        case PadIndexMode::Circular: pad_scatter_kernel<T, PadIndexMode::Circular>(go, gi, p); break;
    }
}

template <typename T>
void pad_gather_dispatch(const T* src, T* dst, const PadNdPlan& p, PadIndexMode mode) {
    switch (mode) {
        case PadIndexMode::Reflect: pad_gather_kernel<T, PadIndexMode::Reflect>(src, dst, p); break;
        case PadIndexMode::Replicate: pad_gather_kernel<T, PadIndexMode::Replicate>(src, dst, p); break;
        case PadIndexMode::Circular: pad_gather_kernel<T, PadIndexMode::Circular>(src, dst, p); break;
    }
}

void pad_mode_check(const char* name, const Tensor& self, const std::vector<int64_t>& pad,
                    PadIndexMode mode) {
    if (pad.size() % 2 != 0)
        TP_THROW(ValueError, name, ": length of pad must be even but it equals ", pad.size());
    const int64_t k = static_cast<int64_t>(pad.size()) / 2;
    if (k > self.dim())
        TP_THROW(ValueError, name, ": padding length too large for input dimensionality");
    for (int64_t i = 0; i < k; ++i) {
        const int64_t dim = self.dim() - 1 - i;
        if (mode == PadIndexMode::Reflect) {
            // torch: "Padding size should be less than the corresponding
            // input dimension"
            if (pad[2 * i] >= self.size(dim) || pad[2 * i + 1] >= self.size(dim))
                TP_THROW(ValueError, name,
                         ": padding size should be less than the corresponding input dimension");
        } else if (mode == PadIndexMode::Circular) {
            // torch allows at most one wrap-around per side
            if (pad[2 * i] > self.size(dim) || pad[2 * i + 1] > self.size(dim))
                TP_THROW(ValueError, name,
                         ": padding value causes wrapping around more than once");
        }
    }
}

Tensor pad_mode_forward(const Tensor& self, const std::vector<int64_t>& pad, PadIndexMode mode) {
    Tensor src = self.contiguous();
    PadNdPlan p = pad_plan(src, pad);
    Tensor out = Tensor::empty(p.dst_sizes, src.dtype(), src.device());
    switch (src.dtype()) {
        case DType::Float32:
            pad_gather_dispatch<float>(src.data_ptr<float>(), out.data_ptr<float>(), p, mode);
            break;
        case DType::Float64:
            pad_gather_dispatch<double>(src.data_ptr<double>(), out.data_ptr<double>(), p, mode);
            break;
        case DType::Float16:
            pad_gather_dispatch<Half>(src.data_ptr<Half>(), out.data_ptr<Half>(), p, mode);
            break;
        case DType::BFloat16:
            pad_gather_dispatch<BFloat16>(src.data_ptr<BFloat16>(), out.data_ptr<BFloat16>(), p,
                                          mode);
            break;
        default:
            TP_THROW(NotImplementedError, "non-constant padding only supports floating dtypes");
    }
    return out;
}

Tensor pad_mode_backward(const Tensor& grad_output, const Tensor& self,
                         const std::vector<int64_t>& pad, PadIndexMode mode) {
    PadNdPlan p = pad_plan(self, pad);
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor gi = Tensor::empty(p.src_sizes, self.dtype(), self.device());
    switch (self.dtype()) {
        case DType::Float32:
            pad_scatter_dispatch<float>(go.data_ptr<float>(), gi.data_ptr<float>(), p, mode);
            break;
        case DType::Float64:
            pad_scatter_dispatch<double>(go.data_ptr<double>(), gi.data_ptr<double>(), p, mode);
            break;
        case DType::Float16:
            pad_scatter_dispatch<Half>(go.data_ptr<Half>(), gi.data_ptr<Half>(), p, mode);
            break;
        case DType::BFloat16:
            pad_scatter_dispatch<BFloat16>(go.data_ptr<BFloat16>(), gi.data_ptr<BFloat16>(), p,
                                           mode);
            break;
        default:
            TP_THROW(NotImplementedError, "non-constant padding only supports floating dtypes");
    }
    return gi;
}

} // namespace

Tensor reflection_pad_nd_cpu(const Tensor& self, const std::vector<int64_t>& pad) {
    pad_mode_check("reflection_pad_nd", self, pad, PadIndexMode::Reflect);
    return pad_mode_forward(self, pad, PadIndexMode::Reflect);
}

Tensor reflection_pad_nd_backward_cpu(const Tensor& grad_output, const Tensor& self,
                                      const std::vector<int64_t>& pad) {
    pad_mode_check("reflection_pad_nd_backward", self, pad, PadIndexMode::Reflect);
    return pad_mode_backward(grad_output, self, pad, PadIndexMode::Reflect);
}

Tensor replication_pad_nd_cpu(const Tensor& self, const std::vector<int64_t>& pad) {
    pad_mode_check("replication_pad_nd", self, pad, PadIndexMode::Replicate);
    return pad_mode_forward(self, pad, PadIndexMode::Replicate);
}

Tensor replication_pad_nd_backward_cpu(const Tensor& grad_output, const Tensor& self,
                                       const std::vector<int64_t>& pad) {
    pad_mode_check("replication_pad_nd_backward", self, pad, PadIndexMode::Replicate);
    return pad_mode_backward(grad_output, self, pad, PadIndexMode::Replicate);
}

Tensor circular_pad_nd_cpu(const Tensor& self, const std::vector<int64_t>& pad) {
    pad_mode_check("circular_pad_nd", self, pad, PadIndexMode::Circular);
    return pad_mode_forward(self, pad, PadIndexMode::Circular);
}

Tensor circular_pad_nd_backward_cpu(const Tensor& grad_output, const Tensor& self,
                                    const std::vector<int64_t>& pad) {
    pad_mode_check("circular_pad_nd_backward", self, pad, PadIndexMode::Circular);
    return pad_mode_backward(grad_output, self, pad, PadIndexMode::Circular);
}

TENSORPLAY_LIBRARY_IMPL(CPU, PadKernels) {
    m.impl("constant_pad_nd", constant_pad_nd_cpu);
    m.impl("constant_pad_nd_backward", constant_pad_nd_backward_cpu);
    m.impl("reflection_pad_nd", reflection_pad_nd_cpu);
    m.impl("reflection_pad_nd_backward", reflection_pad_nd_backward_cpu);
    m.impl("replication_pad_nd", replication_pad_nd_cpu);
    m.impl("replication_pad_nd_backward", replication_pad_nd_backward_cpu);
    m.impl("circular_pad_nd", circular_pad_nd_cpu);
    m.impl("circular_pad_nd_backward", circular_pad_nd_backward_cpu);
}

// constant_pad_nd is a composite over narrow/fill_/copy_ (PadNd.cpp), all of
// which dispatch by tensor device, so the CPU TU registers the same body for
// CUDA instead of carrying a duplicate kernel.
TENSORPLAY_LIBRARY_IMPL(CUDA, PadKernelsConstantComposite) {
    m.impl("constant_pad_nd", constant_pad_nd_cpu);
    m.impl("constant_pad_nd_backward", constant_pad_nd_backward_cpu);
}

} // namespace cpu
} // namespace tensorplay
