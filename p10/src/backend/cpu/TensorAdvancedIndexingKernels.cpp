// Advanced index selection operators - CPU kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#define TENSORPLAY_INDEXING_SKIP_TENSOR_MEMBERS
#include "TensorIndexing.h"
#undef TENSORPLAY_INDEXING_SKIP_TENSOR_MEMBERS
#include "Utils.h"
#include "Exception.h"
#include "AdvancedIndex.h"
#include "Context.h"
#include "IndexKernelUtils.h"

#include <atomic>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include <cstdint>
#include <optional>
#include <vector>

namespace tensorplay {
namespace cpu {

namespace {

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    const int64_t min = -ndim;
    const int64_t max = ndim - 1;
    if (dim < min || dim > max) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 min, ", ", max, "], but got ", dim, ")");
    }
    return dim < 0 ? dim + ndim : dim;
}

Tensor take_along_dim_cpu(const Tensor& self, const Tensor& indices, std::optional<int64_t> dim) {
    if (indices.dtype() != DType::Int64) {
        TP_THROW(TypeError, "take_along_dim: expected indices to have dtype Int64");
    }
    if (self.device() != indices.device()) {
        TP_THROW(DeviceMismatchError,
                 "take_along_dim: self and indices must be on the same device");
    }
    if (!dim.has_value()) {
        Tensor flat = self.view({-1});
        Tensor idx = indices.view({-1});
        return flat.gather(0, idx);
    }
    int64_t nd = self.dim();
    int64_t d = wrap_dim(*dim, nd);
    if (indices.dim() != nd) {
        TP_THROW(RuntimeError, "take_along_dim: indices must have the same number of dimensions as input");
    }
    // Broadcast both operands over every axis except d; along d only the
    // index extent matters (that is the gather length).
    std::vector<int64_t> target(nd);
    for (int64_t i = 0; i < nd; ++i) {
        if (i == d) { target[i] = indices.size(i); continue; }
        int64_t a = self.size(i), b = indices.size(i);
        if (a != b && a != 1 && b != 1) {
            TP_THROW(RuntimeError, "take_along_dim: input and indices must match on non-selected dimensions");
        }
        target[i] = std::max(a, b);
    }
    std::vector<int64_t> idx_target = target;
    std::vector<int64_t> self_target = target;
    self_target[d] = self.size(d);
    Tensor idx_b = indices.expand(idx_target).contiguous();
    Tensor self_b = self.expand(self_target).contiguous();
    idx_b = idx_b.remainder(Scalar(self_b.size(d)));
    return self_b.gather(d, idx_b);
}

// Index dimensions are represented by zero strides on the source view;
// each index tensor contributes its byte offset inside the iterator kernel.
Tensor index_cpu(const Tensor& self,
                 const std::vector<std::optional<Tensor>>& indices) {
    indexing::native::AdvancedIndex info(self, indices);
    Tensor output;
    TensorIteratorConfig config;
    config.set_check_mem_overlap(false).check_all_same_dtype(false)
        .add_output(output).add_const_input(info.source);
    for (const auto& index : info.indices) config.add_const_input(index);
    config.declare_static_dtype_and_device(info.source.dtype(), info.source.device());
    auto iter = config.build();
    const auto item_size = elementSize(self.dtype());
    cpu_index_kernel(iter, info.indexed_sizes, info.indexed_strides,
        [item_size](char* dst, char* src, int64_t offset) {
            std::memcpy(dst, src + offset, item_size);
        });
    return iter.output();
}


}  // namespace

namespace {

void assert_no_index_overlap(const Tensor& a, const Tensor& b) {
    if (a.unsafeGetTensorImpl() == b.unsafeGetTensorImpl()) {
        TP_THROW(RuntimeError, "unsupported operation: source and destination overlap");
    }
    if (a.numel() == 0 || b.numel() == 0) return;
    if (!SizesAndStrides::is_non_overlapping_and_dense(a.shape(), a.strides()) ||
        !SizesAndStrides::is_non_overlapping_and_dense(b.shape(), b.strides())) return;
    if (!a.unsafeGetTensorImpl()->storage().is_same(b.unsafeGetTensorImpl()->storage())) return;
    const auto a_begin = reinterpret_cast<uintptr_t>(a.data_ptr());
    const auto b_begin = reinterpret_cast<uintptr_t>(b.data_ptr());
    TP_CHECK(a_begin >= b_begin + b.numel() * elementSize(b.dtype()) ||
             b_begin >= a_begin + a.numel() * elementSize(a.dtype()),
             "unsupported operation: some elements of the input tensor and the written-to tensor refer to a single memory location");
}

template <typename T>
void index_put_kernel(TensorIterator& iter, const indexing::native::AdvancedIndex& info,
                      bool accumulate) {
    const bool deterministic = globalContext().deterministicAlgorithms();
    if (accumulate) {
        if constexpr (std::is_same_v<T, float>) {
            if (!deterministic && iter.numel() >= parallel::GRAIN_SIZE &&
                parallel::get_num_threads() > 1) {
                cpu_index_kernel(iter, info.indexed_sizes, info.indexed_strides,
                    [](char* dst, char* src, int64_t offset) {
                        // Feature-detect atomic_ref: toolchains whose standard
                        // library predates it fall back to a reinterpret cast
                        // of the target onto std::atomic<float>, which shares
                        // the representation the spin loop relies on.
#if defined(__cpp_lib_atomic_ref) && __cpp_lib_atomic_ref >= 201806L
                        std::atomic_ref<float> target(*reinterpret_cast<float*>(dst + offset));
#else
                        auto& target = *reinterpret_cast<std::atomic<float>*>(reinterpret_cast<float*>(dst + offset));
#endif
                        const float value = *reinterpret_cast<float*>(src);
                        float old = target.load();
                        while (!target.compare_exchange_weak(old, old + value)) {
#if defined(__x86_64__) || defined(__i386__)
                            _mm_pause();
#elif defined(__aarch64__)
                            __asm__ __volatile__("yield;" : : : "memory");
#endif
                        }
                    });
                return;
            }
        }
        cpu_index_kernel(iter, info.indexed_sizes, info.indexed_strides,
            [](char* dst, char* src, int64_t offset) {
                *reinterpret_cast<T*>(dst + offset) += *reinterpret_cast<T*>(src);
            }, true);
    } else {
        cpu_index_kernel(iter, info.indexed_sizes, info.indexed_strides,
            [](char* dst, char* src, int64_t offset) {
                *reinterpret_cast<T*>(dst + offset) = *reinterpret_cast<T*>(src);
            }, deterministic);
    }
}

} // namespace

Tensor& index_put_native_cpu(Tensor& self, const std::vector<Tensor>& indices,
                              const Tensor& values, bool accumulate) {
    TP_CHECK_INDEX(indices.size() <= static_cast<size_t>(self.dim()),
                   "too many indices for tensor of dimension ", self.dim());
    for (int64_t d = 0; d < self.dim(); ++d) {
        if (self.size(d) > 1 && self.strides()[d] == 0) {
            TP_WARN("Use of index_put_ on expanded tensors is deprecated; clone the tensor before writing");
            break;
        }
    }
    if (!accumulate && values.numel() == 1 && values.device().is_cpu()) {
        Tensor mask;
        int64_t consumed = 0;
        bool can_mask_fill = true;
        for (const auto& index : indices) {
            if (!index.defined()) {
                if (!mask.defined()) ++consumed;
            } else if ((index.dtype() != DType::Bool && index.dtype() != DType::UInt8) ||
                       index.device() != self.device() || mask.defined()) {
                can_mask_fill = false;
                break;
            } else {
                mask = index;
                for (int64_t d = 0; d < mask.dim(); ++d) {
                    TP_CHECK_INDEX(consumed + d < self.dim() &&
                                   mask.size(d) == self.size(consumed + d),
                                   "The shape of the mask does not match the indexed tensor");
                }
                consumed += mask.dim();
            }
        }
        if (can_mask_fill && mask.defined()) {
            for (int64_t d = consumed; d < self.dim(); ++d) mask = mask.unsqueeze(-1);
            return tpx::ops::masked_fill_(self, mask, values.item());
        }
    }
    Tensor value = values;
    if (value.device() != self.device() && value.numel() == 1 && value.dim() == 0) {
        value = value.to(self.device());
    }
    assert_no_index_overlap(self, values);
    std::vector<std::optional<Tensor>> optional_indices;
    for (const auto& index : indices) {
        if (index.defined()) {
            assert_no_index_overlap(self, index);
            optional_indices.emplace_back(index);
        } else optional_indices.emplace_back(std::nullopt);
    }
    indexing::native::AdvancedIndex info(self, optional_indices);
    TP_CHECK(value.dtype() == self.dtype(),
             "Index put requires the source and destination dtypes match");
    const auto shape = static_cast<std::vector<int64_t>>(info.source.shape());
    TP_CHECK(value.dim() <= static_cast<int64_t>(shape.size()),
             "shape mismatch: value tensor cannot be broadcast to indexing result");
    for (int64_t d = 0; d < value.dim(); ++d) {
        const auto target_dim = shape.size() - value.dim() + d;
        TP_CHECK(value.size(d) == 1 || value.size(d) == shape[target_dim],
                 "shape mismatch: value tensor cannot be broadcast to indexing result");
    }
    TensorIteratorConfig config;
    config.set_check_mem_overlap(false).resize_outputs(false)
        .check_all_same_dtype(false).add_output(info.source).add_const_input(value);
    for (const auto& index : info.indices) config.add_const_input(index);
    auto iter = config.build();
#define TP_NATIVE_INDEX_PUT(TYPE, NAME) \
    case DType::NAME: index_put_kernel<TYPE>(iter, info, accumulate); break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_NATIVE_INDEX_PUT)
        TP_NATIVE_INDEX_PUT(Float8_e4m3fn, Float8_e4m3fn)
        TP_NATIVE_INDEX_PUT(Float8_e5m2, Float8_e5m2)
        TP_NATIVE_INDEX_PUT(Float8_e4m3fnuz, Float8_e4m3fnuz)
        TP_NATIVE_INDEX_PUT(Float8_e5m2fnuz, Float8_e5m2fnuz)
        default: TP_THROW(TypeError, "index_put: unsupported dtype");
    }
#undef TP_NATIVE_INDEX_PUT
    return self;
}

TENSORPLAY_LIBRARY_IMPL(CPU, TensorAdvancedIndexingKernels) {
    m.impl("index.Tensor", index_cpu);
    m.impl("take_along_dim", take_along_dim_cpu);
}

} // namespace cpu
} // namespace tensorplay
