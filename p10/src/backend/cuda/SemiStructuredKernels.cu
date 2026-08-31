#include "SemiStructuredKernels.h"
#include "CUDARuntime.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace tensorplay {
namespace cuda {
namespace {

struct SemiConfig {
    int group_size;
    int keep;
    int groups_per_meta;
    DType meta_dtype;
};

SemiConfig semi_config(DType dtype) {
    switch (dtype) {
        case DType::Float16:
        case DType::BFloat16:
            return {4, 2, 4, DType::Int16};
        case DType::Float32:
            return {2, 1, 4, DType::Int16};
        case DType::Int8:
            return {4, 2, 8, DType::Int32};
        default:
            TP_THROW(NotImplementedError,
                     "semi-structured sparsity supports Float16, BFloat16, "
                     "Float32, and Int8");
    }
}

__device__ int64_t semi_meta_index_device(int64_t row, int64_t col,
                                          int64_t rows, int64_t meta_cols,
                                          int meta_itemsize) {
    const int64_t row_group = meta_itemsize == static_cast<int>(sizeof(int16_t))
        ? 32 : 16;
    const bool use_interleaved =
        rows >= row_group && rows % row_group == 0 && meta_cols >= 2 &&
        meta_cols % 2 == 0;
    if (!use_interleaved) return row * meta_cols + col;

    const int64_t interweave =
        meta_itemsize == static_cast<int>(sizeof(int16_t)) ? 4 : 2;
    int64_t dst_row = row / row_group * row_group +
                      (row % 8) * interweave + (row % row_group) / 8;
    int64_t dst_col = col;
    if ((dst_row % 2 == 0) && (dst_col % 2 == 1)) {
        ++dst_row;
        --dst_col;
    } else if ((dst_row % 2 == 1) && (dst_col % 2 == 0)) {
        --dst_row;
        ++dst_col;
    }
    return (dst_col / 2) * rows * 2 + dst_row * 2 + dst_col % 2;
}

void validate_dense_input(const Tensor& dense) {
    if (!dense.defined() || dense.is_sparse()) {
        TP_THROW(ValueError, "semi-structured compression expects a dense tensor");
    }
    if (dense.dim() != 2) {
        TP_THROW(ValueError, "semi-structured compression expects a 2-D tensor");
    }
    if (!dense.is_contiguous()) {
        TP_THROW(ValueError,
                 "semi-structured compression expects contiguous input");
    }
    const SemiConfig cfg = semi_config(dense.dtype());
    const int64_t meta_group =
        static_cast<int64_t>(cfg.group_size) * cfg.groups_per_meta;
    if (dense.size(1) < meta_group || dense.size(1) % meta_group != 0) {
        TP_THROW(ValueError,
                 "semi-structured compression requires a metadata-aligned column dimension");
    }
}

void validate_representation(const Tensor& packed, const Tensor& meta,
                             SemiConfig* cfg_out, int64_t* rows_out,
                             int64_t* cols_out) {
    if (!packed.defined() || !meta.defined() || packed.is_sparse() ||
        meta.is_sparse()) {
        TP_THROW(ValueError,
                 "semi-structured values and metadata must be dense tensors");
    }
    if (packed.dim() != 2 || meta.dim() != 2) {
        TP_THROW(ValueError,
                 "semi-structured values and metadata must be 2-D tensors");
    }
    if (packed.device() != meta.device()) {
        TP_THROW(ValueError,
                 "semi-structured values and metadata must share a device");
    }
    const SemiConfig cfg = semi_config(packed.dtype());
    const int64_t rows = packed.size(0);
    const int64_t cols = packed.size(1) * 2;
    const int64_t meta_group =
        static_cast<int64_t>(cfg.group_size) * cfg.groups_per_meta;
    if (meta.size(0) != rows || cols < meta_group || cols % meta_group != 0) {
        TP_THROW(ValueError, "invalid semi-structured representation shape");
    }
    const int64_t expected_meta_cols =
        cols / (cfg.group_size * cfg.groups_per_meta);
    if (meta.size(1) != expected_meta_cols || meta.dtype() != cfg.meta_dtype) {
        TP_THROW(ValueError, "invalid semi-structured metadata shape or dtype");
    }
    if (cfg_out) *cfg_out = cfg;
    if (rows_out) *rows_out = rows;
    if (cols_out) *cols_out = cols;
}

template <typename scalar_t, typename meta_t, int Group, int Keep,
          int GroupsPerMeta>
__global__ void semi_compress_kernel(int64_t work, int64_t cols,
                                     int64_t packed_cols, int64_t meta_cols,
                                     int64_t rows, const scalar_t* source,
                                     scalar_t* packed, meta_t* meta) {
    const int64_t linear =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= work) return;

    const int64_t row = linear / meta_cols;
    const int64_t meta_col = linear % meta_cols;
    uint32_t word = 0;
    for (int slot = 0; slot < GroupsPerMeta; ++slot) {
        const int64_t group = meta_col * GroupsPerMeta + slot;
        const int64_t base = row * cols + group * Group;
        int selected[Keep];
        bool used[Group];
        for (int pos = 0; pos < Group; ++pos) used[pos] = false;
        for (int rank = 0; rank < Keep; ++rank) {
            int best = -1;
            float best_score = -std::numeric_limits<float>::max();
            for (int pos = 0; pos < Group; ++pos) {
                if (used[pos]) continue;
                const float score = fabsf(static_cast<float>(source[base + pos]));
                if (best < 0 || score > best_score ||
                    (score == best_score && pos < best)) {
                    best = pos;
                    best_score = score;
                }
            }
            selected[rank] = best;
            used[best] = true;
        }
        for (int i = 0; i < Keep; ++i) {
            for (int j = i + 1; j < Keep; ++j) {
                if (selected[j] < selected[i]) {
                    const int tmp = selected[i];
                    selected[i] = selected[j];
                    selected[j] = tmp;
                }
            }
        }

        uint32_t code = 0;
        if constexpr (Group == 2) {
            code = selected[0] == 0 ? 4u : 14u;
        } else {
            code = static_cast<uint32_t>(selected[0]) |
                   (static_cast<uint32_t>(selected[1]) << 2);
        }
        word |= code << (4 * slot);
        for (int kept = 0; kept < Keep; ++kept) {
            packed[row * packed_cols + group * Keep + kept] =
                source[base + selected[kept]];
        }
    }
    const int64_t dst = semi_meta_index_device(
        row, meta_col, rows, meta_cols, sizeof(meta_t));
    meta[dst] = static_cast<meta_t>(word);
}

template <int Group, int Keep>
__device__ int semi_decoded_position(uint32_t code, int slot) {
    if constexpr (Group == 2) {
        return (code & 0x0fu) == 14u ? 1 : 0;
    } else {
        return slot == 0 ? static_cast<int>(code & 0x3u)
                         : static_cast<int>((code >> 2) & 0x3u);
    }
}

template <typename scalar_t, typename meta_t, int Group, int Keep,
          int GroupsPerMeta>
__global__ void semi_decompress_kernel(int64_t total, int64_t cols,
                                       int64_t packed_cols, int64_t rows,
                                       int64_t meta_cols,
                                       const scalar_t* packed,
                                       const meta_t* meta, scalar_t* dense) {
    const int64_t linear =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int64_t row = linear / cols;
    const int64_t col = linear % cols;
    const int64_t group = col / Group;
    const int pos = static_cast<int>(col % Group);
    const int64_t meta_col = group / GroupsPerMeta;
    const int slot = static_cast<int>(group % GroupsPerMeta);
    const uint32_t word = static_cast<uint32_t>(meta[
        semi_meta_index_device(row, meta_col, rows, meta_cols, sizeof(meta_t))]);
    const uint32_t code = (word >> (4 * slot)) & 0x0fu;
    int packed_slot = -1;
    for (int kept = 0; kept < Keep; ++kept) {
        if (semi_decoded_position<Group, Keep>(code, kept) == pos) {
            packed_slot = kept;
            break;
        }
    }
    dense[linear] = packed_slot < 0
        ? scalar_t(0)
        : packed[row * packed_cols + group * Keep + packed_slot];
}

template <typename scalar_t, typename meta_t, int Group, int Keep,
          int GroupsPerMeta>
__global__ void semi_mask_grad_kernel(int64_t total, int64_t cols,
                                      int64_t rows, int64_t meta_cols,
                                      const scalar_t* grad,
                                      const meta_t* meta, scalar_t* result) {
    const int64_t linear =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int64_t row = linear / cols;
    const int64_t col = linear % cols;
    const int64_t group = col / Group;
    const int pos = static_cast<int>(col % Group);
    const int64_t meta_col = group / GroupsPerMeta;
    const int slot = static_cast<int>(group % GroupsPerMeta);
    const uint32_t word = static_cast<uint32_t>(meta[
        semi_meta_index_device(row, meta_col, rows, meta_cols,
                               sizeof(meta_t))]);
    const uint32_t code = (word >> (4 * slot)) & 0x0fu;
    bool selected = false;
    for (int kept = 0; kept < Keep; ++kept) {
        if (semi_decoded_position<Group, Keep>(code, kept) == pos) {
            selected = true;
            break;
        }
    }
    result[linear] = selected ? grad[linear] : scalar_t(0);
}

template <typename scalar_t, typename meta_t, int Group, int Keep,
          int GroupsPerMeta>
__global__ void semi_gather_grad_kernel(int64_t total, int64_t cols,
                                        int64_t packed_cols, int64_t rows,
                                        int64_t meta_cols,
                                        const scalar_t* grad,
                                        const meta_t* meta, scalar_t* result) {
    const int64_t linear =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int64_t row = linear / packed_cols;
    const int64_t packed_col = linear % packed_cols;
    const int64_t group = packed_col / Keep;
    const int kept = static_cast<int>(packed_col % Keep);
    const int64_t meta_col = group / GroupsPerMeta;
    const int slot = static_cast<int>(group % GroupsPerMeta);
    const uint32_t word = static_cast<uint32_t>(meta[
        semi_meta_index_device(row, meta_col, rows, meta_cols,
                               sizeof(meta_t))]);
    const uint32_t code = (word >> (4 * slot)) & 0x0fu;
    const int pos = semi_decoded_position<Group, Keep>(code, kept);
    result[linear] = grad[row * cols + group * Group + pos];
}

template <typename scalar_t, typename meta_t, typename output_t,
          typename accum_t, int Group, int Keep, int GroupsPerMeta>
__global__ void semi_mm_kernel(int64_t total, int64_t cols,
                               int64_t packed_cols, int64_t rows,
                               int64_t meta_cols, int64_t out_cols,
                               const scalar_t* packed, const meta_t* meta,
                               const scalar_t* dense, const output_t* bias,
                               float alpha, float beta, output_t* out) {
    const int64_t linear =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int64_t row = linear / out_cols;
    const int64_t col = linear % out_cols;
    accum_t value = accum_t(0);
    for (int64_t group = 0; group < cols / Group; ++group) {
        const int64_t meta_col = group / GroupsPerMeta;
        const int slot = static_cast<int>(group % GroupsPerMeta);
        const uint32_t word = static_cast<uint32_t>(meta[
            semi_meta_index_device(row, meta_col, rows, meta_cols,
                                   sizeof(meta_t))]);
        const uint32_t code = (word >> (4 * slot)) & 0x0fu;
        for (int kept = 0; kept < Keep; ++kept) {
            const int pos = semi_decoded_position<Group, Keep>(code, kept);
            if (pos < Group) {
                const scalar_t a =
                    packed[row * packed_cols + group * Keep + kept];
                const scalar_t b = dense[(group * Group + pos) * out_cols + col];
                value += static_cast<accum_t>(a) * static_cast<accum_t>(b);
            }
        }
    }
    if (bias) {
        value = static_cast<accum_t>(value * alpha +
                                     static_cast<accum_t>(beta) *
                                     static_cast<accum_t>(bias[row]));
    } else {
        value = static_cast<accum_t>(value * alpha);
    }
    out[linear] = static_cast<output_t>(value);
}

template <typename scalar_t, typename meta_t, typename output_t,
          typename accum_t, int Group, int Keep, int GroupsPerMeta>
__global__ void semi_mm_right_kernel(int64_t total, int64_t sparse_rows,
                                     int64_t cols, int64_t packed_cols,
                                     int64_t meta_cols, int64_t out_cols,
                                     const scalar_t* dense,
                                     const scalar_t* packed,
                                     const meta_t* meta, output_t* out) {
    const int64_t linear =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int64_t row = linear / out_cols;
    const int64_t col = linear % out_cols;
    const int64_t group = col / Group;
    const int pos = static_cast<int>(col % Group);
    accum_t value = accum_t(0);
    for (int64_t sparse_row = 0; sparse_row < sparse_rows; ++sparse_row) {
        const int64_t meta_col = group / GroupsPerMeta;
        const int slot = static_cast<int>(group % GroupsPerMeta);
        const uint32_t word = static_cast<uint32_t>(meta[
            semi_meta_index_device(sparse_row, meta_col, sparse_rows,
                                   meta_cols, sizeof(meta_t))]);
        const uint32_t code = (word >> (4 * slot)) & 0x0fu;
        for (int kept = 0; kept < Keep; ++kept) {
            if (semi_decoded_position<Group, Keep>(code, kept) == pos) {
                const scalar_t a =
                    packed[sparse_row * packed_cols + group * Keep + kept];
                const scalar_t b = dense[row * sparse_rows + sparse_row];
                value += static_cast<accum_t>(b) * static_cast<accum_t>(a);
                break;
            }
        }
    }
    out[linear] = static_cast<output_t>(value);
}

template <typename scalar_t, typename meta_t, typename output_t,
          typename accum_t, int Group, int Keep, int GroupsPerMeta>
Tensor launch_mm_typed(const Tensor& packed, const Tensor& meta,
                       const Tensor& dense, DType output_dtype,
                       const Tensor* bias, float alpha, float beta) {
    const int64_t rows = packed.size(0);
    const int64_t packed_cols = packed.size(1);
    const int64_t cols = packed_cols * 2;
    const int64_t out_cols = dense.size(1);
    Tensor out = Tensor::empty({rows, out_cols}, output_dtype, packed.device());
    const int64_t total = rows * out_cols;
    if (total == 0) return out;
    constexpr int threads = 128;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    semi_mm_kernel<scalar_t, meta_t, output_t, accum_t, Group, Keep,
                   GroupsPerMeta><<<blocks, threads, 0,
                                     getCurrentCUDAStream().stream()>>>(
        total, cols, packed_cols, rows, meta.size(1), out_cols,
        packed.data_ptr<scalar_t>(), meta.data_ptr<meta_t>(),
        dense.data_ptr<scalar_t>(), bias ? bias->data_ptr<output_t>() : nullptr,
        alpha, beta, out.data_ptr<output_t>());
    checkCuda(cudaGetLastError(), "CUDA semi-structured matmul kernel");
    return out;
}

Tensor mm_dispatch(const Tensor& packed, const Tensor& meta,
                   const Tensor& dense, DType output_dtype,
                   const Tensor* bias, float alpha, float beta) {
    switch (packed.dtype()) {
        case DType::Float16:
            if (output_dtype == DType::Float16) {
                return launch_mm_typed<Half, int16_t, Half, float, 4, 2, 4>(
                    packed, meta, dense, output_dtype, bias, alpha, beta);
            }
            if (output_dtype == DType::Float32) {
                return launch_mm_typed<Half, int16_t, float, float, 4, 2, 4>(
                    packed, meta, dense, output_dtype, bias, alpha, beta);
            }
            break;
        case DType::BFloat16:
            if (output_dtype == DType::BFloat16) {
                return launch_mm_typed<BFloat16, int16_t, BFloat16, float,
                                       4, 2, 4>(packed, meta, dense,
                                                output_dtype, bias, alpha, beta);
            }
            if (output_dtype == DType::Float32) {
                return launch_mm_typed<BFloat16, int16_t, float, float, 4, 2,
                                       4>(packed, meta, dense, output_dtype,
                                          bias, alpha, beta);
            }
            break;
        case DType::Float32:
            if (output_dtype == DType::Float32) {
                return launch_mm_typed<float, int16_t, float, float, 2, 1, 4>(
                    packed, meta, dense, output_dtype, bias, alpha, beta);
            }
            break;
        case DType::Int8:
            if (output_dtype == DType::Int8) {
                return launch_mm_typed<int8_t, int32_t, int8_t, int32_t, 4, 2,
                                       8>(packed, meta, dense, output_dtype,
                                          bias, alpha, beta);
            }
            if (output_dtype == DType::Int32) {
                return launch_mm_typed<int8_t, int32_t, int32_t, int32_t, 4, 2,
                                       8>(packed, meta, dense, output_dtype,
                                          bias, alpha, beta);
            }
            break;
        default:
            break;
    }
    TP_THROW(NotImplementedError,
             "requested semi-structured output dtype is not supported");
}

template <typename scalar_t, typename meta_t, typename output_t,
          typename accum_t, int Group, int Keep, int GroupsPerMeta>
Tensor launch_mm_right_typed(const Tensor& dense, const Tensor& packed,
                             const Tensor& meta, DType output_dtype) {
    const int64_t left_rows = dense.size(0);
    const int64_t sparse_rows = packed.size(0);
    const int64_t packed_cols = packed.size(1);
    const int64_t cols = packed_cols * 2;
    const int64_t out_cols = cols;
    Tensor out = Tensor::empty({left_rows, out_cols}, output_dtype,
                               packed.device());
    const int64_t total = left_rows * out_cols;
    if (total == 0) return out;
    constexpr int threads = 128;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    semi_mm_right_kernel<scalar_t, meta_t, output_t, accum_t, Group, Keep,
                         GroupsPerMeta><<<blocks, threads, 0,
                                           getCurrentCUDAStream().stream()>>>(
        total, sparse_rows, cols, packed_cols, meta.size(1), out_cols,
        dense.data_ptr<scalar_t>(), packed.data_ptr<scalar_t>(),
        meta.data_ptr<meta_t>(), out.data_ptr<output_t>());
    checkCuda(cudaGetLastError(), "CUDA semi-structured right matmul kernel");
    return out;
}

Tensor mm_right_dispatch(const Tensor& dense, const Tensor& packed,
                         const Tensor& meta, DType output_dtype) {
    switch (packed.dtype()) {
        case DType::Float16:
            if (output_dtype == DType::Float16) {
                return launch_mm_right_typed<Half, int16_t, Half, float, 4, 2,
                                             4>(dense, packed, meta,
                                                output_dtype);
            }
            if (output_dtype == DType::Float32) {
                return launch_mm_right_typed<Half, int16_t, float, float, 4, 2,
                                             4>(dense, packed, meta,
                                                output_dtype);
            }
            break;
        case DType::BFloat16:
            if (output_dtype == DType::BFloat16) {
                return launch_mm_right_typed<BFloat16, int16_t, BFloat16,
                                             float, 4, 2, 4>(
                    dense, packed, meta, output_dtype);
            }
            if (output_dtype == DType::Float32) {
                return launch_mm_right_typed<BFloat16, int16_t, float, float,
                                             4, 2, 4>(dense, packed, meta,
                                                      output_dtype);
            }
            break;
        case DType::Float32:
            if (output_dtype == DType::Float32) {
                return launch_mm_right_typed<float, int16_t, float, float, 2,
                                             1, 4>(dense, packed, meta,
                                                    output_dtype);
            }
            break;
        case DType::Int8:
            if (output_dtype == DType::Int8) {
                return launch_mm_right_typed<int8_t, int32_t, int8_t, int32_t,
                                             4, 2, 8>(dense, packed, meta,
                                                       output_dtype);
            }
            if (output_dtype == DType::Int32) {
                return launch_mm_right_typed<int8_t, int32_t, int32_t, int32_t,
                                             4, 2, 8>(dense, packed, meta,
                                                       output_dtype);
            }
            break;
        default:
            break;
    }
    TP_THROW(NotImplementedError,
             "requested semi-structured output dtype is not supported");
}

template <typename scalar_t, typename meta_t, int Group, int Keep,
          int GroupsPerMeta>
Tensor launch_mask_grad_typed(const Tensor& grad, const Tensor& packed,
                              const Tensor& meta) {
    const int64_t rows = packed.size(0);
    const int64_t packed_cols = packed.size(1);
    const int64_t cols = packed_cols * 2;
    Tensor result = Tensor::empty({rows, cols}, packed.dtype(), packed.device());
    const int64_t total = rows * cols;
    if (total == 0) return result;
    constexpr int threads = 128;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    semi_mask_grad_kernel<scalar_t, meta_t, Group, Keep, GroupsPerMeta>
        <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            total, cols, rows, meta.size(1), grad.data_ptr<scalar_t>(),
            meta.data_ptr<meta_t>(), result.data_ptr<scalar_t>());
    checkCuda(cudaGetLastError(), "CUDA semi-structured gradient mask kernel");
    return result;
}

template <typename scalar_t, typename meta_t, int Group, int Keep,
          int GroupsPerMeta>
Tensor launch_gather_grad_typed(const Tensor& grad, const Tensor& packed,
                                const Tensor& meta) {
    const int64_t rows = packed.size(0);
    const int64_t packed_cols = packed.size(1);
    const int64_t cols = packed_cols * 2;
    Tensor result = Tensor::empty({rows, packed_cols}, packed.dtype(),
                                  packed.device());
    const int64_t total = rows * packed_cols;
    if (total == 0) return result;
    constexpr int threads = 128;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    semi_gather_grad_kernel<scalar_t, meta_t, Group, Keep, GroupsPerMeta>
        <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            total, cols, packed_cols, rows, meta.size(1),
            grad.data_ptr<scalar_t>(), meta.data_ptr<meta_t>(),
            result.data_ptr<scalar_t>());
    checkCuda(cudaGetLastError(), "CUDA semi-structured gradient gather kernel");
    return result;
}

void validate_grad_input(const Tensor& grad, const Tensor& packed,
                         int64_t rows, int64_t cols) {
    if (!grad.defined() || grad.is_sparse() || grad.dim() != 2 ||
        grad.size(0) != rows || grad.size(1) != cols ||
        grad.device() != packed.device() || grad.dtype() != packed.dtype()) {
        TP_THROW(ValueError,
                 "semi-structured gradient must match the dense logical shape and dtype");
    }
}

void validate_mm_inputs(const Tensor& packed, const Tensor& meta,
                        const Tensor& dense, SemiConfig* cfg, int64_t* rows,
                        int64_t* cols) {
    validate_representation(packed, meta, cfg, rows, cols);
    if (!dense.defined() || dense.is_sparse() || dense.dim() != 2) {
        TP_THROW(ValueError, "semi-structured matmul expects a dense 2-D matrix");
    }
    if (dense.device() != packed.device() || dense.dtype() != packed.dtype()) {
        TP_THROW(ValueError,
                 "semi-structured matmul operands must share device and dtype");
    }
    if (dense.size(0) != *cols) {
        TP_THROW(ValueError, "semi-structured matmul dimensions are incompatible");
    }
}

void validate_mm_right_inputs(const Tensor& dense, const Tensor& packed,
                             const Tensor& meta, SemiConfig* cfg,
                             int64_t* rows, int64_t* cols) {
    validate_representation(packed, meta, cfg, rows, cols);
    if (!dense.defined() || dense.is_sparse() || dense.dim() != 2) {
        TP_THROW(ValueError, "semi-structured matmul expects a dense 2-D matrix");
    }
    if (dense.device() != packed.device() || dense.dtype() != packed.dtype()) {
        TP_THROW(ValueError,
                 "semi-structured matmul operands must share device and dtype");
    }
    if (dense.size(1) != *rows) {
        TP_THROW(ValueError, "semi-structured matmul dimensions are incompatible");
    }
}

} // namespace

std::tuple<Tensor, Tensor> sparse_semi_structured_compress_cuda(
    const Tensor& dense) {
    validate_dense_input(dense);
    const SemiConfig cfg = semi_config(dense.dtype());
    const int64_t rows = dense.size(0);
    const int64_t cols = dense.size(1);
    Tensor packed = Tensor::empty({rows, cols / 2}, dense.dtype(), dense.device());
    Tensor meta = Tensor::empty(
        {rows, cols / (cfg.group_size * cfg.groups_per_meta)},
        cfg.meta_dtype, dense.device());
    const int64_t work = rows * meta.size(1);
    if (work == 0) return {packed, meta};
    constexpr int threads = 128;
    const int blocks = static_cast<int>((work + threads - 1) / threads);
    switch (dense.dtype()) {
        case DType::Float16:
            semi_compress_kernel<Half, int16_t, 4, 2, 4>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    work, cols, packed.size(1), meta.size(1), rows,
                    dense.data_ptr<Half>(), packed.data_ptr<Half>(),
                    meta.data_ptr<int16_t>());
            break;
        case DType::BFloat16:
            semi_compress_kernel<BFloat16, int16_t, 4, 2, 4>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    work, cols, packed.size(1), meta.size(1), rows,
                    dense.data_ptr<BFloat16>(), packed.data_ptr<BFloat16>(),
                    meta.data_ptr<int16_t>());
            break;
        case DType::Float32:
            semi_compress_kernel<float, int16_t, 2, 1, 4>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    work, cols, packed.size(1), meta.size(1), rows,
                    dense.data_ptr<float>(), packed.data_ptr<float>(),
                    meta.data_ptr<int16_t>());
            break;
        case DType::Int8:
            semi_compress_kernel<int8_t, int32_t, 4, 2, 8>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    work, cols, packed.size(1), meta.size(1), rows,
                    dense.data_ptr<int8_t>(), packed.data_ptr<int8_t>(),
                    meta.data_ptr<int32_t>());
            break;
        default:
            TP_THROW(NotImplementedError,
                     "requested semi-structured dtype is not supported");
    }
    checkCuda(cudaGetLastError(), "CUDA semi-structured compression kernel");
    return {packed, meta};
}

Tensor sparse_semi_structured_to_dense_cuda(const Tensor& packed,
                                            const Tensor& meta) {
    SemiConfig cfg{};
    int64_t rows = 0;
    int64_t cols = 0;
    validate_representation(packed, meta, &cfg, &rows, &cols);
    Tensor packed_contiguous =
        packed.is_contiguous() ? packed : packed.contiguous();
    Tensor meta_contiguous = meta.is_contiguous() ? meta : meta.contiguous();
    Tensor dense = Tensor::empty({rows, cols}, packed.dtype(), packed.device());
    const int64_t total = rows * cols;
    if (total == 0) return dense;
    constexpr int threads = 128;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    switch (packed.dtype()) {
        case DType::Float16:
            semi_decompress_kernel<Half, int16_t, 4, 2, 4>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    total, cols, packed_contiguous.size(1), rows,
                    meta_contiguous.size(1), packed_contiguous.data_ptr<Half>(),
                    meta_contiguous.data_ptr<int16_t>(), dense.data_ptr<Half>());
            break;
        case DType::BFloat16:
            semi_decompress_kernel<BFloat16, int16_t, 4, 2, 4>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    total, cols, packed_contiguous.size(1), rows,
                    meta_contiguous.size(1),
                    packed_contiguous.data_ptr<BFloat16>(),
                    meta_contiguous.data_ptr<int16_t>(),
                    dense.data_ptr<BFloat16>());
            break;
        case DType::Float32:
            semi_decompress_kernel<float, int16_t, 2, 1, 4>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    total, cols, packed_contiguous.size(1), rows,
                    meta_contiguous.size(1), packed_contiguous.data_ptr<float>(),
                    meta_contiguous.data_ptr<int16_t>(), dense.data_ptr<float>());
            break;
        case DType::Int8:
            semi_decompress_kernel<int8_t, int32_t, 4, 2, 8>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    total, cols, packed_contiguous.size(1), rows,
                    meta_contiguous.size(1), packed_contiguous.data_ptr<int8_t>(),
                    meta_contiguous.data_ptr<int32_t>(), dense.data_ptr<int8_t>());
            break;
        default:
            TP_THROW(NotImplementedError,
                     "requested semi-structured dtype is not supported");
    }
    checkCuda(cudaGetLastError(), "CUDA semi-structured decompression kernel");
    return dense;
}

Tensor sparse_semi_structured_mask_grad_cuda(
    const Tensor& grad, const Tensor& packed, const Tensor& meta) {
    SemiConfig cfg{};
    int64_t rows = 0;
    int64_t cols = 0;
    validate_representation(packed, meta, &cfg, &rows, &cols);
    validate_grad_input(grad, packed, rows, cols);
    Tensor grad_contiguous = grad.is_contiguous() ? grad : grad.contiguous();
    switch (packed.dtype()) {
        case DType::Float16:
            return launch_mask_grad_typed<Half, int16_t, 4, 2, 4>(
                grad_contiguous, packed, meta);
        case DType::BFloat16:
            return launch_mask_grad_typed<BFloat16, int16_t, 4, 2, 4>(
                grad_contiguous, packed, meta);
        case DType::Float32:
            return launch_mask_grad_typed<float, int16_t, 2, 1, 4>(
                grad_contiguous, packed, meta);
        case DType::Int8:
            return launch_mask_grad_typed<int8_t, int32_t, 4, 2, 8>(
                grad_contiguous, packed, meta);
        default:
            TP_THROW(NotImplementedError,
                     "requested semi-structured gradient dtype is not supported");
    }
}

Tensor sparse_semi_structured_gather_grad_cuda(
    const Tensor& grad, const Tensor& packed, const Tensor& meta) {
    SemiConfig cfg{};
    int64_t rows = 0;
    int64_t cols = 0;
    validate_representation(packed, meta, &cfg, &rows, &cols);
    validate_grad_input(grad, packed, rows, cols);
    Tensor grad_contiguous = grad.is_contiguous() ? grad : grad.contiguous();
    switch (packed.dtype()) {
        case DType::Float16:
            return launch_gather_grad_typed<Half, int16_t, 4, 2, 4>(
                grad_contiguous, packed, meta);
        case DType::BFloat16:
            return launch_gather_grad_typed<BFloat16, int16_t, 4, 2, 4>(
                grad_contiguous, packed, meta);
        case DType::Float32:
            return launch_gather_grad_typed<float, int16_t, 2, 1, 4>(
                grad_contiguous, packed, meta);
        case DType::Int8:
            return launch_gather_grad_typed<int8_t, int32_t, 4, 2, 8>(
                grad_contiguous, packed, meta);
        default:
            TP_THROW(NotImplementedError,
                     "requested semi-structured gradient dtype is not supported");
    }
}

Tensor sparse_semi_structured_mm_cuda(
    const Tensor& packed, const Tensor& meta, const Tensor& dense,
    std::optional<DType> out_dtype) {
    SemiConfig cfg{};
    int64_t rows = 0;
    int64_t cols = 0;
    validate_mm_inputs(packed, meta, dense, &cfg, &rows, &cols);
    Tensor packed_contiguous =
        packed.is_contiguous() ? packed : packed.contiguous();
    Tensor meta_contiguous = meta.is_contiguous() ? meta : meta.contiguous();
    Tensor dense_contiguous =
        dense.is_contiguous() ? dense : dense.contiguous();
    return mm_dispatch(packed_contiguous, meta_contiguous, dense_contiguous,
                       out_dtype.value_or(packed.dtype()), nullptr, 1.0f, 1.0f);
}

Tensor sparse_semi_structured_mm_right_cuda(
    const Tensor& dense, const Tensor& packed, const Tensor& meta,
    std::optional<DType> out_dtype) {
    SemiConfig cfg{};
    int64_t rows = 0;
    int64_t cols = 0;
    validate_mm_right_inputs(dense, packed, meta, &cfg, &rows, &cols);
    Tensor dense_contiguous =
        dense.is_contiguous() ? dense : dense.contiguous();
    Tensor packed_contiguous =
        packed.is_contiguous() ? packed : packed.contiguous();
    Tensor meta_contiguous = meta.is_contiguous() ? meta : meta.contiguous();
    return mm_right_dispatch(dense_contiguous, packed_contiguous,
                             meta_contiguous,
                             out_dtype.value_or(packed.dtype()));
}

Tensor sparse_semi_structured_addmm_cuda(
    const Tensor& input, const Tensor& packed, const Tensor& meta,
    const Tensor& dense, Scalar alpha, Scalar beta,
    std::optional<DType> out_dtype) {
    SemiConfig cfg{};
    int64_t rows = 0;
    int64_t cols = 0;
    validate_mm_inputs(packed, meta, dense, &cfg, &rows, &cols);
    const DType result_dtype = out_dtype.value_or(packed.dtype());
    const bool promoted_output =
        (packed.dtype() == DType::Float16 ||
         packed.dtype() == DType::BFloat16) && result_dtype == DType::Float32;
    const bool integer_output =
        packed.dtype() == DType::Int8 && result_dtype == DType::Int32;
    if (!input.defined() ||
        (result_dtype != packed.dtype() && !promoted_output && !integer_output) ||
        input.dim() != 1 ||
        input.size(0) != rows || input.dtype() != result_dtype ||
        input.device() != packed.device()) {
        TP_THROW(ValueError, "invalid semi-structured addmm bias or output dtype");
    }
    Tensor packed_contiguous =
        packed.is_contiguous() ? packed : packed.contiguous();
    Tensor meta_contiguous = meta.is_contiguous() ? meta : meta.contiguous();
    Tensor dense_contiguous =
        dense.is_contiguous() ? dense : dense.contiguous();
    Tensor input_contiguous =
        input.is_contiguous() ? input : input.contiguous();
    return mm_dispatch(packed_contiguous, meta_contiguous, dense_contiguous,
                       result_dtype, &input_contiguous,
                       static_cast<float>(alpha.toDouble()),
                       static_cast<float>(beta.toDouble()));
}

TENSORPLAY_LIBRARY_IMPL(CUDA, SemiStructuredKernels) {
    m.impl("_to_sparse_semi_structured", sparse_semi_structured_compress_cuda);
    m.impl("_sparse_semi_structured_to_dense",
           sparse_semi_structured_to_dense_cuda);
    m.impl("_sparse_semi_structured_mask_grad",
           sparse_semi_structured_mask_grad_cuda);
    m.impl("_sparse_semi_structured_gather_grad",
           sparse_semi_structured_gather_grad_cuda);
    m.impl("_sparse_semi_structured_mm", sparse_semi_structured_mm_cuda);
    m.impl("_sparse_semi_structured_mm_right",
           sparse_semi_structured_mm_right_cuda);
    m.impl("_sparse_semi_structured_addmm", sparse_semi_structured_addmm_cuda);
}

} // namespace cuda
} // namespace tensorplay
