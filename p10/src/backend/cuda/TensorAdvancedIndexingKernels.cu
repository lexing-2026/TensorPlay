// Advanced index selection operators - CUDA kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "CUDARuntime.h"
#include "CUDALoops.cuh"
#include "AdvancedIndex.h"
#include "Context.h"
#include "OpMathType.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
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

Tensor gather_cuda(const Tensor& self, int64_t dim, const Tensor& index);

namespace {

constexpr int kAdvancedIndexMaxDims = 16;

template <int NumIndices, typename scalar_t>
void launch_advanced_index_kernel(const indexing::native::AdvancedIndex& info, Tensor& output) {
    static_assert(NumIndices > 0);
    TP_CHECK(static_cast<int>(output.dim()) <= kAdvancedIndexMaxDims,
             "index: tensor rank exceeds CUDA indexing limit");

    const int64_t rank = output.dim();
    const auto output_shape = static_cast<std::vector<int64_t>>(output.shape());
    std::array<std::vector<int64_t>, NumIndices + 1> byte_strides;
    byte_strides[0].resize(static_cast<size_t>(rank));
    for (int64_t d = 0; d < rank; ++d) {
        byte_strides[0][static_cast<size_t>(d)] =
            info.source.stride(d) * static_cast<int64_t>(sizeof(scalar_t));
    }
    for (int i = 0; i < NumIndices; ++i) {
        byte_strides[static_cast<size_t>(i + 1)].resize(static_cast<size_t>(rank));
        const Tensor& index = info.indices[static_cast<size_t>(i)];
        for (int64_t d = 0; d < rank; ++d) {
            // Size-one index dims broadcast over the output.
            byte_strides[static_cast<size_t>(i + 1)][static_cast<size_t>(d)] =
                index.size(d) == 1
                    ? 0
                    : index.stride(d) * static_cast<int64_t>(sizeof(int64_t));
        }
    }

    bool fast_offsets = output.numel() <=
        static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    for (int i = 0; i < NumIndices + 1; ++i) {
        for (int64_t stride : byte_strides[static_cast<size_t>(i)]) {
            fast_offsets = fast_offsets && stride >= 0 &&
                stride <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
        }
    }

    // Index offsets are byte offsets: address the index data as bytes.
    std::array<const char*, NumIndices> index_ptrs{};
    for (int i = 0; i < NumIndices; ++i) {
        index_ptrs[static_cast<size_t>(i)] = reinterpret_cast<const char*>(
            info.indices[static_cast<size_t>(i)].data_ptr<int64_t>());
    }
    const scalar_t* source_ptr = info.source.data_ptr<scalar_t>();
    std::array<int64_t, NumIndices> indexed_sizes{};
    std::array<int64_t, NumIndices> indexed_strides{};
    for (int i = 0; i < NumIndices; ++i) {
        indexed_sizes[static_cast<size_t>(i)] = info.indexed_sizes[static_cast<size_t>(i)];
        indexed_strides[static_cast<size_t>(i)] = info.indexed_strides[static_cast<size_t>(i)];
    }

    if (fast_offsets) {
        // The offset calculator walks dimensions innermost first.
        std::vector<int64_t> reversed_shape(output_shape.rbegin(), output_shape.rend());
        std::array<std::vector<int64_t>, NumIndices + 1> reversed_strides;
        std::array<const int64_t*, NumIndices + 1> reversed_ptrs{};
        for (int i = 0; i < NumIndices + 1; ++i) {
            const auto& forward = byte_strides[static_cast<size_t>(i)];
            reversed_strides[static_cast<size_t>(i)].assign(forward.rbegin(), forward.rend());
            reversed_ptrs[static_cast<size_t>(i)] =
                reversed_strides[static_cast<size_t>(i)].data();
        }
        OffsetCalculator<NumIndices + 1, uint32_t> offsets(
            static_cast<int>(rank), reversed_shape.data(), reversed_ptrs.data());
        gpu_kernel_with_index(output, [=] GPU_LAMBDA(int64_t linear_index) -> scalar_t {
            const auto byte_offsets = offsets.get(static_cast<uint32_t>(linear_index));
            int64_t source_offset = static_cast<int64_t>(byte_offsets[0]);
#pragma unroll
            for (int i = 0; i < NumIndices; ++i) {
                int64_t index = *reinterpret_cast<const int64_t*>(
                    index_ptrs[static_cast<size_t>(i)] + byte_offsets[static_cast<size_t>(i + 1)]);
                assert(index >= -indexed_sizes[static_cast<size_t>(i)] &&
                       index < indexed_sizes[static_cast<size_t>(i)]);
                if (index < 0) index += indexed_sizes[static_cast<size_t>(i)];
                source_offset += index * indexed_strides[static_cast<size_t>(i)];
            }
            return *reinterpret_cast<const scalar_t*>(
                reinterpret_cast<const char*>(source_ptr) + source_offset);
        });
        return;
    }

    std::array<int64_t, kAdvancedIndexMaxDims> shape{};
    std::array<int64_t, kAdvancedIndexMaxDims> source_byte_strides{};
    std::array<std::array<int64_t, kAdvancedIndexMaxDims>, NumIndices>
        index_byte_strides{};
    for (int64_t d = 0; d < rank; ++d) {
        shape[static_cast<size_t>(d)] = output_shape[static_cast<size_t>(d)];
        source_byte_strides[static_cast<size_t>(d)] = byte_strides[0][static_cast<size_t>(d)];
        for (int i = 0; i < NumIndices; ++i) {
            index_byte_strides[static_cast<size_t>(i)][static_cast<size_t>(d)] =
                byte_strides[static_cast<size_t>(i + 1)][static_cast<size_t>(d)];
        }
    }
    gpu_kernel_with_index(output, [=] GPU_LAMBDA(int64_t linear_index) -> scalar_t {
        int64_t remainder = linear_index;
        int64_t source_offset = 0;
        int64_t index_offsets[NumIndices] = {};
        for (int64_t d = rank - 1; d >= 0; --d) {
            const int64_t coordinate = remainder % shape[static_cast<size_t>(d)];
            remainder /= shape[static_cast<size_t>(d)];
            source_offset += coordinate * source_byte_strides[static_cast<size_t>(d)];
            for (int i = 0; i < NumIndices; ++i) {
                index_offsets[i] += coordinate *
                    index_byte_strides[static_cast<size_t>(i)][static_cast<size_t>(d)];
            }
        }
        for (int i = 0; i < NumIndices; ++i) {
            int64_t index = *reinterpret_cast<const int64_t*>(
                index_ptrs[static_cast<size_t>(i)] + index_offsets[i]);
            assert(index >= -indexed_sizes[static_cast<size_t>(i)] &&
                   index < indexed_sizes[static_cast<size_t>(i)]);
            if (index < 0) index += indexed_sizes[static_cast<size_t>(i)];
            source_offset += index * indexed_strides[static_cast<size_t>(i)];
        }
        return *reinterpret_cast<const scalar_t*>(
            reinterpret_cast<const char*>(source_ptr) + source_offset);
    });
}

template <int NumIndices>
void dispatch_advanced_index_kernel(const indexing::native::AdvancedIndex& info, Tensor& output) {
#define TP_ADVANCED_INDEX_CASE(ctype, name) \
    case DType::name: \
        launch_advanced_index_kernel<NumIndices, ctype>(info, output); \
        break;
    switch (info.source.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_ADVANCED_INDEX_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_ADVANCED_INDEX_CASE)
        TP_ADVANCED_INDEX_CASE(tensorplay::complex<Half>, ComplexHalf)
        TP_ADVANCED_INDEX_CASE(tensorplay::complex<float>, ComplexFloat)
        TP_ADVANCED_INDEX_CASE(tensorplay::complex<double>, ComplexDouble)
        TP_ADVANCED_INDEX_CASE(tensorplay::complex<BFloat16>, BComplex32)
        default:
            TP_THROW(TypeError, "index: unsupported dtype");
    }
#undef TP_ADVANCED_INDEX_CASE
}

Tensor index_cuda(const Tensor& self,
                  const std::vector<std::optional<Tensor>>& indices) {
    if (indices.empty()) {
        TP_THROW(IndexError, "index: at least one index must be provided");
    }
    indexing::native::AdvancedIndex info(self, indices);
    if (info.indices.empty()) return self;
    Tensor output = Tensor::empty(
        static_cast<std::vector<int64_t>>(info.source.shape()), self.dtype(),
        self.device());
    if (output.numel() == 0) return output;
    switch (info.indices.size()) {
        case 1: dispatch_advanced_index_kernel<1>(info, output); break;
        case 2: dispatch_advanced_index_kernel<2>(info, output); break;
        case 3: dispatch_advanced_index_kernel<3>(info, output); break;
        case 4: dispatch_advanced_index_kernel<4>(info, output); break;
        case 5: dispatch_advanced_index_kernel<5>(info, output); break;
        case 6: dispatch_advanced_index_kernel<6>(info, output); break;
        case 7: dispatch_advanced_index_kernel<7>(info, output); break;
        case 8: dispatch_advanced_index_kernel<8>(info, output); break;
        case 9: dispatch_advanced_index_kernel<9>(info, output); break;
        case 10: dispatch_advanced_index_kernel<10>(info, output); break;
        case 11: dispatch_advanced_index_kernel<11>(info, output); break;
        case 12: dispatch_advanced_index_kernel<12>(info, output); break;
        case 13: dispatch_advanced_index_kernel<13>(info, output); break;
        case 14: dispatch_advanced_index_kernel<14>(info, output); break;
        case 15: dispatch_advanced_index_kernel<15>(info, output); break;
        case 16: dispatch_advanced_index_kernel<16>(info, output); break;
        default:
            TP_THROW(IndexError, "index: too many advanced index tensors");
    }
    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ---------------------------------------------------------------------------
// _index_put_impl_
//
// Plain writes scatter the value straight through the advanced-index
// geometry (duplicate positions keep an unspecified writer).  Accumulation,
// and every write under deterministic algorithms, sorts the linearized
// index positions and lets one thread fold each run of duplicates in index
// order.
// ---------------------------------------------------------------------------

constexpr int kThreads = 256;

// Element storage of a given width: plain writes move bytes, so the dtype
// only matters through its size.
template <int Bytes> struct PutStorage;
template <> struct PutStorage<1> { using type = uint8_t; };
template <> struct PutStorage<2> { using type = uint16_t; };
template <> struct PutStorage<4> { using type = uint32_t; };
template <> struct PutStorage<8> { using type = uint64_t; };
struct alignas(16) PutStorage16 { uint64_t lo, hi; };
template <> struct PutStorage<16> { using type = PutStorage16; };

template <int NumIndices>
struct PutGeometry {
    int rank;
    int64_t shape[kAdvancedIndexMaxDims];
    int64_t dst_strides[kAdvancedIndexMaxDims];     // bytes
    int64_t value_strides[kAdvancedIndexMaxDims];   // bytes
    int64_t index_strides[NumIndices][kAdvancedIndexMaxDims];  // bytes
    const char* index_ptrs[NumIndices];
    int64_t indexed_sizes[NumIndices];
    int64_t indexed_strides[NumIndices];            // bytes
};

template <int NumIndices, typename storage_t>
__global__ void index_put_direct_kernel(int64_t numel,
                                        PutGeometry<NumIndices> g,
                                        char* dst, const char* value) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < numel; i += step) {
        int64_t remainder = i;
        int64_t dst_offset = 0;
        int64_t value_offset = 0;
        int64_t index_offsets[NumIndices] = {};
        for (int d = g.rank - 1; d >= 0; --d) {
            const int64_t coordinate = remainder % g.shape[d];
            remainder /= g.shape[d];
            dst_offset += coordinate * g.dst_strides[d];
            value_offset += coordinate * g.value_strides[d];
#pragma unroll
            for (int k = 0; k < NumIndices; ++k) {
                index_offsets[k] += coordinate * g.index_strides[k][d];
            }
        }
#pragma unroll
        for (int k = 0; k < NumIndices; ++k) {
            int64_t index = *reinterpret_cast<const int64_t*>(
                g.index_ptrs[k] + index_offsets[k]);
            assert(index >= -g.indexed_sizes[k] && index < g.indexed_sizes[k] &&
                   "index out of bounds");
            if (index < 0) index += g.indexed_sizes[k];
            dst_offset += index * g.indexed_strides[k];
        }
        *reinterpret_cast<storage_t*>(dst + dst_offset) =
            *reinterpret_cast<const storage_t*>(value + value_offset);
    }
}

template <int NumIndices>
void launch_index_put_direct(const indexing::native::AdvancedIndex& info,
                             const Tensor& value) {
    const Tensor& dst = info.source;
    const int64_t rank = dst.dim();
    TP_CHECK(rank <= kAdvancedIndexMaxDims,
             "index_put: tensor rank exceeds CUDA indexing limit");
    const int64_t element_size = static_cast<int64_t>(elementSize(dst.dtype()));
    const Tensor expanded_value =
        value.expand(static_cast<std::vector<int64_t>>(dst.shape()));
    PutGeometry<NumIndices> g{};
    g.rank = static_cast<int>(rank);
    for (int64_t d = 0; d < rank; ++d) {
        g.shape[d] = dst.size(d);
        g.dst_strides[d] = dst.stride(d) * element_size;
        g.value_strides[d] = expanded_value.stride(d) * element_size;
    }
    for (int k = 0; k < NumIndices; ++k) {
        const Tensor& index = info.indices[static_cast<size_t>(k)];
        const int64_t pad = rank - index.dim();
        for (int64_t d = 0; d < rank; ++d) {
            g.index_strides[k][d] = d < pad
                ? 0
                : (index.size(d - pad) == 1 ? 0 : index.stride(d - pad)) *
                      static_cast<int64_t>(sizeof(int64_t));
        }
        g.index_ptrs[k] = static_cast<const char*>(index.data_ptr());
        g.indexed_sizes[k] = info.indexed_sizes[static_cast<size_t>(k)];
        g.indexed_strides[k] = info.indexed_strides[static_cast<size_t>(k)];
    }
    const int64_t numel = dst.numel();
    const int64_t blocks = std::min<int64_t>((numel + kThreads - 1) / kThreads, 65535);
    auto stream = getCurrentCUDAStream().stream();
    char* dst_ptr = static_cast<char*>(dst.data_ptr());
    const char* value_ptr = static_cast<const char*>(expanded_value.data_ptr());
    switch (element_size) {
#define TP_PUT_WIDTH(BYTES) \
        case BYTES: \
            index_put_direct_kernel<NumIndices, typename PutStorage<BYTES>::type> \
                <<<blocks, kThreads, 0, stream>>>(numel, g, dst_ptr, value_ptr); \
            break;
        TP_PUT_WIDTH(1)
        TP_PUT_WIDTH(2)
        TP_PUT_WIDTH(4)
        TP_PUT_WIDTH(8)
        TP_PUT_WIDTH(16)
#undef TP_PUT_WIDTH
        default:
            TP_THROW(TypeError, "index_put: unsupported element size");
    }
    CUDA_CHECK(cudaGetLastError());
}

void index_put_direct(const indexing::native::AdvancedIndex& info,
                      const Tensor& value) {
    switch (info.indices.size()) {
#define TP_PUT_ARITY(N) case N: launch_index_put_direct<N>(info, value); break;
        TP_PUT_ARITY(1) TP_PUT_ARITY(2) TP_PUT_ARITY(3) TP_PUT_ARITY(4)
        TP_PUT_ARITY(5) TP_PUT_ARITY(6) TP_PUT_ARITY(7) TP_PUT_ARITY(8)
        TP_PUT_ARITY(9) TP_PUT_ARITY(10) TP_PUT_ARITY(11) TP_PUT_ARITY(12)
        TP_PUT_ARITY(13) TP_PUT_ARITY(14) TP_PUT_ARITY(15) TP_PUT_ARITY(16)
#undef TP_PUT_ARITY
        default:
            TP_THROW(IndexError, "index_put: too many advanced index tensors");
    }
}

// Accumulator of the sorted fold: reduced-width floating types widen to
// float, reduced-width complex to complex<float>.
template <typename T> struct PutAccumulate { using type = typename OpMathType<T>::type; };
template <> struct PutAccumulate<tensorplay::complex<Half>> { using type = tensorplay::complex<float>; };
template <> struct PutAccumulate<tensorplay::complex<BFloat16>> { using type = tensorplay::complex<float>; };

// One thread per (duplicate run start, leading element, slice element): the
// run's values fold in their original order, so the result does not depend
// on scheduling.  Value layout is (before, index position, slice).
template <typename scalar_t>
__global__ void index_put_sorted_kernel(int64_t num_indices,
                                        int64_t elements_before,
                                        int64_t slice_size,
                                        int64_t indexed_numel,
                                        const int64_t* sorted_keys,
                                        const int64_t* original_positions,
                                        const scalar_t* value,
                                        scalar_t* dst, bool accumulate) {
    using acc_t = typename PutAccumulate<scalar_t>::type;
    const int64_t total = num_indices * elements_before * slice_size;
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; t < total; t += step) {
        const int64_t s = t % slice_size;
        const int64_t rest = t / slice_size;
        const int64_t b = rest % elements_before;
        const int64_t i = rest / elements_before;
        const int64_t key = sorted_keys[i];
        if (i > 0 && sorted_keys[i - 1] == key) continue;
        scalar_t* out = dst + (b * indexed_numel + key) * slice_size + s;
        if (accumulate) {
            acc_t acc = static_cast<acc_t>(*out);
            for (int64_t j = i; j < num_indices && sorted_keys[j] == key; ++j) {
                const acc_t v = static_cast<acc_t>(
                    value[(b * num_indices + original_positions[j]) * slice_size + s]);
                if constexpr (std::is_same_v<acc_t, bool>) {
                    acc = acc || v;
                } else {
                    acc += v;
                }
            }
            *out = static_cast<scalar_t>(acc);
        } else {
            int64_t last = i;
            while (last + 1 < num_indices && sorted_keys[last + 1] == key) ++last;
            *out = value[(b * num_indices + original_positions[last]) * slice_size + s];
        }
    }
}

// Validates index ranges against the axis and wraps negative entries.
Tensor wrap_index_once(const Tensor& index, int64_t dim, int64_t dim_size,
                       bool check_range) {
    if (index.numel() != 0 && check_range) {
        const int64_t max_index = tpx::ops::max(index).item<int64_t>();
        const int64_t min_index = tpx::ops::min(index).item<int64_t>();
        TP_CHECK_INDEX(max_index < dim_size, "index ", max_index,
                       " is out of bounds for dimension ", dim, " with size ",
                       dim_size);
        TP_CHECK_INDEX(min_index >= -dim_size, "index ", min_index,
                       " is out of bounds for dimension ", dim, " with size ",
                       dim_size);
    }
    return tpx::ops::remainder(index, Scalar(dim_size));
}

void index_put_with_sort(Tensor& self,
                         const std::vector<std::optional<Tensor>>& indices,
                         const Tensor& value, bool accumulate, bool unsafe) {
    const bool self_contiguous = self.is_contiguous();
    Tensor self_ = self_contiguous ? self : self.contiguous();
    auto [src, expanded] =
        indexing::native::prepare_indices(self_, indices, /*ensure_same_device=*/true);

    // Linearize the indexed block: dims before it run over elements_before,
    // dims after it form contiguous slices of slice_size elements.
    int64_t elements_before = 1;
    int64_t slice_size = 1;
    int64_t indexed_numel = 1;
    int64_t dims_before = 0;
    int64_t dims_indexed = 0;
    bool seen = false;
    for (int64_t d = 0; d < src.dim(); ++d) {
        if (expanded[static_cast<size_t>(d)].defined()) {
            seen = true;
            ++dims_indexed;
            indexed_numel *= src.size(d);
        } else if (seen) {
            slice_size *= src.size(d);
        } else {
            elements_before *= src.size(d);
            ++dims_before;
        }
    }
    if (!seen) {
        tpx::ops::copy_(self, accumulate ? tpx::ops::add(self, value) : value);
        return;
    }
    Tensor linear;
    int64_t key_stride = indexed_numel;
    for (int64_t d = dims_before; d < dims_before + dims_indexed; ++d) {
        key_stride /= src.size(d);
        Tensor term = tpx::ops::mul(
            wrap_index_once(expanded[static_cast<size_t>(d)], d, src.size(d),
                            !unsafe),
            Scalar(key_stride));
        linear = linear.defined() ? tpx::ops::add(linear, term) : term;
    }

    std::vector<int64_t> values_shape =
        static_cast<std::vector<int64_t>>(src.shape());
    const auto index_shape = static_cast<std::vector<int64_t>>(linear.shape());
    values_shape.erase(values_shape.begin() + dims_before,
                       values_shape.begin() + dims_before + dims_indexed);
    values_shape.insert(values_shape.begin() + dims_before, index_shape.begin(),
                        index_shape.end());
    const Tensor expanded_value =
        tpx::ops::contiguous(value.expand(values_shape));
    const int64_t num_indices = linear.numel();

    if (num_indices > 0 && slice_size > 0 && elements_before > 0) {
        const bool permuted = !src.is_contiguous();
        Tensor src_ = permuted ? tpx::ops::contiguous(src) : src;
        const Tensor flat = tpx::ops::reshape(linear, {-1});
        auto [sorted_keys, original_positions] =
            tpx::ops::sort(flat, /*stable=*/true, 0, false);
        const int64_t total = num_indices * elements_before * slice_size;
        const int64_t blocks = std::min<int64_t>((total + kThreads - 1) / kThreads, 65535);
        auto stream = getCurrentCUDAStream().stream();
#define TP_PUT_SORTED(ctype, name) \
        case DType::name: \
            index_put_sorted_kernel<ctype><<<blocks, kThreads, 0, stream>>>( \
                num_indices, elements_before, slice_size, indexed_numel, \
                sorted_keys.data_ptr<int64_t>(), \
                original_positions.data_ptr<int64_t>(), \
                static_cast<const ctype*>(expanded_value.data_ptr()), \
                static_cast<ctype*>(src_.data_ptr()), accumulate); \
            break;
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_PUT_SORTED)
            TENSORPLAY_FORALL_FP8_TYPES(TP_PUT_SORTED)
            TP_PUT_SORTED(tensorplay::complex<Half>, ComplexHalf)
            TP_PUT_SORTED(tensorplay::complex<float>, ComplexFloat)
            TP_PUT_SORTED(tensorplay::complex<double>, ComplexDouble)
            TP_PUT_SORTED(tensorplay::complex<BFloat16>, BComplex32)
            default:
                TP_THROW(TypeError, "index_put: unsupported dtype");
        }
#undef TP_PUT_SORTED
        CUDA_CHECK(cudaGetLastError());
        if (permuted) tpx::ops::copy_(src, src_);
    }
    if (!self_contiguous) tpx::ops::copy_(self, self_);
}

} // namespace

Tensor& index_put_impl_cuda(Tensor& self,
                            const std::vector<std::optional<Tensor>>& indices,
                            const Tensor& values, bool accumulate, bool unsafe) {
    TP_CHECK_INDEX(indices.size() <= static_cast<size_t>(self.dim()),
                   "too many indices for tensor of dimension ", self.dim(),
                   " (got ", indices.size(), ")");
    for (int64_t d = 0; d < self.dim(); ++d) {
        if (self.size(d) > 1 && self.stride(d) == 0) {
            TP_WARN("Use of index_put_ on expanded tensors is deprecated. "
                    "Please clone() the tensor before performing this operation. "
                    "This also applies to advanced indexing e.g. tensor[indices] = tensor");
            break;
        }
    }
    if (!accumulate) {
        if (auto mask = indexing::native::can_dispatch_to_masked_fill(
                self, indices, values)) {
            return tpx::ops::masked_fill_(self, *mask, values.item());
        }
    }
    Tensor value = values;
    if (value.device() != self.device() && value.numel() == 1 && value.dim() == 0) {
        value = value.to(self.device());
    }
    TP_CHECK(value.device() == self.device(), "expected device ",
             self.device().toString(), " but got device ",
             value.device().toString(), " for value tensor");
    TP_CHECK(value.dtype() == self.dtype(),
             "Index put requires the source and destination dtypes match, got ",
             toString(self.dtype()), " for the destination and ",
             toString(value.dtype()), " for the source.");
    if (accumulate ||
        (globalContext().deterministicAlgorithms() && value.numel() > 1)) {
        index_put_with_sort(self, indices, value, accumulate, unsafe);
        return self;
    }
    indexing::native::AdvancedIndex info(self, indices);
    const auto shape = static_cast<std::vector<int64_t>>(info.source.shape());
    TP_CHECK(value.dim() <= static_cast<int64_t>(shape.size()),
             "shape mismatch: value tensor cannot be broadcast to indexing result");
    for (int64_t d = 0; d < value.dim(); ++d) {
        const auto target_dim = shape.size() - value.dim() + d;
        TP_CHECK(value.size(d) == 1 || value.size(d) == shape[target_dim],
                 "shape mismatch: value tensor cannot be broadcast to indexing result");
    }
    if (info.indices.empty()) {
        tpx::ops::copy_(self, value);
        return self;
    }
    if (info.source.numel() == 0) return self;
    index_put_direct(info, value);
    return self;
}

namespace {

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(RuntimeError, "Dimension out of range (expected to be in range of [",
                 -ndim, ", ", ndim - 1, "], but got ", dim - ndim, ")");
    }
    return dim;
}

Tensor take_along_dim_cuda(const Tensor& self, const Tensor& indices, std::optional<int64_t> dim) {
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
        return gather_cuda(flat, 0, idx);
    }
    int64_t nd = self.dim();
    int64_t d = wrap_dim(*dim, nd);
    if (indices.dim() != nd) {
        TP_THROW(RuntimeError, "take_along_dim: indices must have the same number of dimensions as input");
    }
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
    return gather_cuda(self_b, d, idx_b);
}


} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, TensorAdvancedIndexingKernels) {
    m.impl("index.Tensor", index_cuda);
    m.impl("take_along_dim", take_along_dim_cuda);
    m.impl("_index_put_impl_", index_put_impl_cuda);
}

} // namespace cuda
} // namespace tensorplay
