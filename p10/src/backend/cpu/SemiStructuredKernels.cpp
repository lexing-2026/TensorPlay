#include "SemiStructuredKernels.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace tensorplay {
namespace cpu {
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

int64_t semi_meta_index(int64_t row, int64_t col, int64_t rows,
                        int64_t meta_cols, size_t meta_itemsize) {
    const int64_t row_group = meta_itemsize == sizeof(int16_t) ? 32 : 16;
    const bool use_interleaved =
        rows >= row_group && rows % row_group == 0 && meta_cols >= 2 &&
        meta_cols % 2 == 0;
    if (!use_interleaved) return row * meta_cols + col;

    const int64_t interweave = meta_itemsize == sizeof(int16_t) ? 4 : 2;
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
    SemiConfig cfg = semi_config(packed.dtype());
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

template <typename scalar_t>
float semi_score(scalar_t value) {
    return std::fabs(static_cast<float>(value));
}

template <int Group, int Keep, typename scalar_t>
std::array<int, Keep> choose_positions(const scalar_t* source, int64_t base) {
    std::array<int, Keep> selected{};
    std::array<bool, Group> used{};
    for (int rank = 0; rank < Keep; ++rank) {
        int best = -1;
        float best_score = -std::numeric_limits<float>::infinity();
        for (int pos = 0; pos < Group; ++pos) {
            if (used[pos]) continue;
            const float score = semi_score(source[base + pos]);
            if (best < 0 || score > best_score ||
                (score == best_score && pos < best)) {
                best = pos;
                best_score = score;
            }
        }
        selected[rank] = best;
        used[best] = true;
    }
    std::sort(selected.begin(), selected.end());
    return selected;
}

template <typename scalar_t, typename meta_t, int Group, int Keep,
          int GroupsPerMeta>
void compress_typed(const Tensor& dense, Tensor& packed, Tensor& meta) {
    const int64_t rows = dense.size(0);
    const int64_t cols = dense.size(1);
    const int64_t packed_cols = cols / 2;
    const int64_t meta_cols = cols / (Group * GroupsPerMeta);
    const scalar_t* source = dense.data_ptr<scalar_t>();
    scalar_t* packed_data = packed.data_ptr<scalar_t>();
    meta_t* meta_data = meta.data_ptr<meta_t>();

    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t meta_col = 0; meta_col < meta_cols; ++meta_col) {
            uint32_t word = 0;
            for (int slot = 0; slot < GroupsPerMeta; ++slot) {
                const int64_t group = meta_col * GroupsPerMeta + slot;
                const int64_t base = row * cols + group * Group;
                const auto selected = choose_positions<Group, Keep>(source, base);
                uint32_t code = 0;
                if constexpr (Group == 2) {
                    code = selected[0] == 0 ? 4u : 14u;
                } else {
                    code = static_cast<uint32_t>(selected[0]) |
                           (static_cast<uint32_t>(selected[1]) << 2);
                }
                word |= code << (4 * slot);
                for (int kept = 0; kept < Keep; ++kept) {
                    packed_data[row * packed_cols + group * Keep + kept] =
                        source[base + selected[kept]];
                }
            }
            const int64_t dst = semi_meta_index(
                row, meta_col, rows, meta_cols, sizeof(meta_t));
            meta_data[dst] = static_cast<meta_t>(word);
        }
    }
}

template <int Group, int Keep>
int decoded_position(uint32_t code, int slot) {
    if constexpr (Group == 2) {
        return (code & 0x0fu) == 14u ? 1 : 0;
    } else {
        return slot == 0 ? static_cast<int>(code & 0x3u)
                         : static_cast<int>((code >> 2) & 0x3u);
    }
}

template <typename scalar_t, typename meta_t, int Group, int Keep,
          int GroupsPerMeta>
void decompress_typed(const Tensor& packed, const Tensor& meta, Tensor& dense) {
    const int64_t rows = packed.size(0);
    const int64_t packed_cols = packed.size(1);
    const int64_t cols = packed_cols * 2;
    const int64_t meta_cols = meta.size(1);
    const scalar_t* packed_data = packed.data_ptr<scalar_t>();
    const meta_t* meta_data = meta.data_ptr<meta_t>();
    scalar_t* dense_data = dense.data_ptr<scalar_t>();

    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t group = 0; group < cols / Group; ++group) {
            const int64_t meta_col = group / GroupsPerMeta;
            const int slot = static_cast<int>(group % GroupsPerMeta);
            const uint32_t word = static_cast<uint32_t>(meta_data[
                semi_meta_index(row, meta_col, rows, meta_cols, sizeof(meta_t))]);
            const uint32_t code = (word >> (4 * slot)) & 0x0fu;
            for (int pos = 0; pos < Group; ++pos) {
                dense_data[row * cols + group * Group + pos] = scalar_t(0);
            }
            for (int kept = 0; kept < Keep; ++kept) {
                const int pos = decoded_position<Group, Keep>(code, kept);
                if (pos < Group) {
                    dense_data[row * cols + group * Group + pos] =
                        packed_data[row * packed_cols + group * Keep + kept];
                }
            }
        }
    }
}

template <typename scalar_t, typename meta_t, typename output_t,
          typename accum_t, int Group, int Keep, int GroupsPerMeta>
Tensor mm_typed(const Tensor& packed, const Tensor& meta, const Tensor& dense,
                DType output_dtype, const Tensor* bias, double alpha,
                double beta) {
    const int64_t rows = packed.size(0);
    const int64_t packed_cols = packed.size(1);
    const int64_t cols = packed_cols * 2;
    const int64_t out_cols = dense.size(1);
    Tensor out = Tensor::zeros({rows, out_cols}, output_dtype, packed.device());
    const scalar_t* packed_data = packed.data_ptr<scalar_t>();
    const meta_t* meta_data = meta.data_ptr<meta_t>();
    const scalar_t* dense_data = dense.data_ptr<scalar_t>();
    const output_t* bias_data = bias ? bias->data_ptr<output_t>() : nullptr;
    output_t* out_data = out.data_ptr<output_t>();
    const int64_t meta_cols = meta.size(1);

    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < out_cols; ++col) {
            accum_t value = accum_t(0);
            for (int64_t group = 0; group < cols / Group; ++group) {
                const int64_t meta_col = group / GroupsPerMeta;
                const int slot = static_cast<int>(group % GroupsPerMeta);
                const uint32_t word = static_cast<uint32_t>(meta_data[
                    semi_meta_index(row, meta_col, rows, meta_cols,
                                    sizeof(meta_t))]);
                const uint32_t code = (word >> (4 * slot)) & 0x0fu;
                for (int kept = 0; kept < Keep; ++kept) {
                    const int pos = decoded_position<Group, Keep>(code, kept);
                    if (pos < Group) {
                        const scalar_t a = packed_data[
                            row * packed_cols + group * Keep + kept];
                        const scalar_t b = dense_data[(group * Group + pos) *
                                                     out_cols + col];
                        value += static_cast<accum_t>(a) *
                                 static_cast<accum_t>(b);
                    }
                }
            }
            if (bias_data) {
                value = static_cast<accum_t>(value * alpha +
                                             static_cast<accum_t>(beta) *
                                             static_cast<accum_t>(bias_data[row]));
            }
            out_data[row * out_cols + col] = static_cast<output_t>(
                bias_data ? value : static_cast<accum_t>(value * alpha));
        }
    }
    return out;
}

template <typename scalar_t, typename meta_t, int Group, int Keep,
          int GroupsPerMeta>
void mask_grad_typed(const Tensor& grad, const Tensor& packed, const Tensor& meta,
                     Tensor& result) {
    const int64_t rows = packed.size(0);
    const int64_t packed_cols = packed.size(1);
    const int64_t cols = packed_cols * 2;
    const int64_t meta_cols = meta.size(1);
    const scalar_t* grad_data = grad.data_ptr<scalar_t>();
    const meta_t* meta_data = meta.data_ptr<meta_t>();
    scalar_t* result_data = result.data_ptr<scalar_t>();

    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t group = 0; group < cols / Group; ++group) {
            const int64_t meta_col = group / GroupsPerMeta;
            const int slot = static_cast<int>(group % GroupsPerMeta);
            const uint32_t word = static_cast<uint32_t>(meta_data[
                semi_meta_index(row, meta_col, rows, meta_cols,
                                sizeof(meta_t))]);
            const uint32_t code = (word >> (4 * slot)) & 0x0fu;
            for (int pos = 0; pos < Group; ++pos) {
                bool selected = false;
                for (int kept = 0; kept < Keep; ++kept) {
                    if (decoded_position<Group, Keep>(code, kept) == pos) {
                        selected = true;
                        break;
                    }
                }
                const int64_t offset = row * cols + group * Group + pos;
                result_data[offset] = selected ? grad_data[offset] : scalar_t(0);
            }
        }
    }
}

template <typename scalar_t, typename meta_t, int Group, int Keep,
          int GroupsPerMeta>
void gather_grad_typed(const Tensor& grad, const Tensor& packed,
                       const Tensor& meta, Tensor& result) {
    const int64_t rows = packed.size(0);
    const int64_t packed_cols = packed.size(1);
    const int64_t cols = packed_cols * 2;
    const int64_t meta_cols = meta.size(1);
    const scalar_t* grad_data = grad.data_ptr<scalar_t>();
    const meta_t* meta_data = meta.data_ptr<meta_t>();
    scalar_t* result_data = result.data_ptr<scalar_t>();

    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t group = 0; group < cols / Group; ++group) {
            const int64_t meta_col = group / GroupsPerMeta;
            const int slot = static_cast<int>(group % GroupsPerMeta);
            const uint32_t word = static_cast<uint32_t>(meta_data[
                semi_meta_index(row, meta_col, rows, meta_cols,
                                sizeof(meta_t))]);
            const uint32_t code = (word >> (4 * slot)) & 0x0fu;
            for (int kept = 0; kept < Keep; ++kept) {
                const int pos = decoded_position<Group, Keep>(code, kept);
                result_data[row * packed_cols + group * Keep + kept] =
                    grad_data[row * cols + group * Group + pos];
            }
        }
    }
}

template <typename scalar_t, typename meta_t, int Group, int Keep,
          int GroupsPerMeta>
Tensor mm_same_typed(const Tensor& packed, const Tensor& meta,
                     const Tensor& dense, const Tensor* bias, double alpha,
                     double beta) {
    using accum_t = std::conditional_t<std::is_same_v<scalar_t, int8_t>,
                                       int32_t, float>;
    return mm_typed<scalar_t, meta_t, scalar_t, accum_t, Group, Keep,
                    GroupsPerMeta>(packed, meta, dense, packed.dtype(), bias,
                                   alpha, beta);
}

template <typename scalar_t, typename meta_t, typename output_t,
          typename accum_t, int Group, int Keep, int GroupsPerMeta>
Tensor mm_right_typed(const Tensor& dense, const Tensor& packed,
                      const Tensor& meta, DType output_dtype) {
    const int64_t left_rows = dense.size(0);
    const int64_t sparse_rows = packed.size(0);
    const int64_t packed_cols = packed.size(1);
    const int64_t cols = packed_cols * 2;
    const int64_t meta_cols = meta.size(1);
    Tensor out = Tensor::zeros({left_rows, cols}, output_dtype, packed.device());
    const scalar_t* dense_data = dense.data_ptr<scalar_t>();
    const scalar_t* packed_data = packed.data_ptr<scalar_t>();
    const meta_t* meta_data = meta.data_ptr<meta_t>();
    output_t* out_data = out.data_ptr<output_t>();

    for (int64_t left_row = 0; left_row < left_rows; ++left_row) {
        for (int64_t col = 0; col < cols; ++col) {
            const int64_t group = col / Group;
            const int pos = static_cast<int>(col % Group);
            accum_t value = accum_t(0);
            for (int64_t sparse_row = 0; sparse_row < sparse_rows;
                 ++sparse_row) {
                const int64_t meta_col = group / GroupsPerMeta;
                const int slot = static_cast<int>(group % GroupsPerMeta);
                const uint32_t word = static_cast<uint32_t>(meta_data[
                    semi_meta_index(sparse_row, meta_col, sparse_rows,
                                    meta_cols, sizeof(meta_t))]);
                const uint32_t code = (word >> (4 * slot)) & 0x0fu;
                for (int kept = 0; kept < Keep; ++kept) {
                    if (decoded_position<Group, Keep>(code, kept) == pos) {
                        const scalar_t a = packed_data[
                            sparse_row * packed_cols + group * Keep + kept];
                        const scalar_t b =
                            dense_data[left_row * sparse_rows + sparse_row];
                        value += static_cast<accum_t>(b) *
                                 static_cast<accum_t>(a);
                        break;
                    }
                }
            }
            out_data[left_row * cols + col] = static_cast<output_t>(value);
        }
    }
    return out;
}

template <typename scalar_t, typename meta_t, int Group, int Keep,
          int GroupsPerMeta>
Tensor mm_right_same_typed(const Tensor& dense, const Tensor& packed,
                           const Tensor& meta, DType output_dtype) {
    using accum_t = std::conditional_t<std::is_same_v<scalar_t, int8_t>,
                                       int32_t, float>;
    return mm_right_typed<scalar_t, meta_t, scalar_t, accum_t, Group, Keep,
                          GroupsPerMeta>(dense, packed, meta, output_dtype);
}

template <typename scalar_t, typename meta_t, int Group, int Keep,
          int GroupsPerMeta>
Tensor mm_right_float_typed(const Tensor& dense, const Tensor& packed,
                            const Tensor& meta, DType output_dtype) {
    return mm_right_typed<scalar_t, meta_t, float, float, Group, Keep,
                          GroupsPerMeta>(dense, packed, meta, output_dtype);
}

template <typename scalar_t, typename meta_t, int Group, int Keep,
          int GroupsPerMeta>
Tensor mm_right_int32_typed(const Tensor& dense, const Tensor& packed,
                            const Tensor& meta, DType output_dtype) {
    return mm_right_typed<scalar_t, meta_t, int32_t, int32_t, Group, Keep,
                          GroupsPerMeta>(dense, packed, meta, output_dtype);
}

Tensor mm_dispatch(const Tensor& packed, const Tensor& meta,
                   const Tensor& dense, DType output_dtype,
                   const Tensor* bias, double alpha, double beta) {
    switch (packed.dtype()) {
        case DType::Float16:
            if (output_dtype == DType::Float16) {
                return mm_same_typed<Half, int16_t, 4, 2, 4>(
                    packed, meta, dense, bias, alpha, beta);
            }
            if (output_dtype == DType::Float32) {
                return mm_typed<Half, int16_t, float, float, 4, 2, 4>(
                    packed, meta, dense, output_dtype, bias, alpha, beta);
            }
            break;
        case DType::BFloat16:
            if (output_dtype == DType::BFloat16) {
                return mm_same_typed<BFloat16, int16_t, 4, 2, 4>(
                    packed, meta, dense, bias, alpha, beta);
            }
            if (output_dtype == DType::Float32) {
                return mm_typed<BFloat16, int16_t, float, float, 4, 2, 4>(
                    packed, meta, dense, output_dtype, bias, alpha, beta);
            }
            break;
        case DType::Float32:
            if (output_dtype == DType::Float32) {
                return mm_same_typed<float, int16_t, 2, 1, 4>(
                    packed, meta, dense, bias, alpha, beta);
            }
            break;
        case DType::Int8:
            if (output_dtype == DType::Int8) {
                return mm_same_typed<int8_t, int32_t, 4, 2, 8>(
                    packed, meta, dense, bias, alpha, beta);
            }
            if (output_dtype == DType::Int32) {
                return mm_typed<int8_t, int32_t, int32_t, int32_t, 4, 2, 8>(
                    packed, meta, dense, output_dtype, bias, alpha, beta);
            }
            break;
        default:
            break;
    }
    TP_THROW(NotImplementedError,
             "requested semi-structured output dtype is not supported");
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

std::tuple<Tensor, Tensor> sparse_semi_structured_compress_cpu(
    const Tensor& dense) {
    validate_dense_input(dense);
    const SemiConfig cfg = semi_config(dense.dtype());
    const int64_t rows = dense.size(0);
    const int64_t cols = dense.size(1);
    Tensor packed = Tensor::empty({rows, cols / 2}, dense.dtype(), dense.device());
    Tensor meta = Tensor::empty(
        {rows, cols / (cfg.group_size * cfg.groups_per_meta)},
        cfg.meta_dtype, dense.device());

    switch (dense.dtype()) {
        case DType::Float16:
            compress_typed<Half, int16_t, 4, 2, 4>(dense, packed, meta);
            break;
        case DType::BFloat16:
            compress_typed<BFloat16, int16_t, 4, 2, 4>(dense, packed, meta);
            break;
        case DType::Float32:
            compress_typed<float, int16_t, 2, 1, 4>(dense, packed, meta);
            break;
        case DType::Int8:
            compress_typed<int8_t, int32_t, 4, 2, 8>(dense, packed, meta);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "requested semi-structured dtype is not supported");
    }
    return {packed, meta};
}

Tensor sparse_semi_structured_to_dense_cpu(const Tensor& packed,
                                           const Tensor& meta) {
    SemiConfig cfg{};
    int64_t rows = 0;
    int64_t cols = 0;
    validate_representation(packed, meta, &cfg, &rows, &cols);
    Tensor packed_contiguous =
        packed.is_contiguous() ? packed : packed.contiguous();
    Tensor meta_contiguous = meta.is_contiguous() ? meta : meta.contiguous();
    Tensor dense = Tensor::zeros({rows, cols}, packed.dtype(), packed.device());

    switch (packed.dtype()) {
        case DType::Float16:
            decompress_typed<Half, int16_t, 4, 2, 4>(
                packed_contiguous, meta_contiguous, dense);
            break;
        case DType::BFloat16:
            decompress_typed<BFloat16, int16_t, 4, 2, 4>(
                packed_contiguous, meta_contiguous, dense);
            break;
        case DType::Float32:
            decompress_typed<float, int16_t, 2, 1, 4>(
                packed_contiguous, meta_contiguous, dense);
            break;
        case DType::Int8:
            decompress_typed<int8_t, int32_t, 4, 2, 8>(
                packed_contiguous, meta_contiguous, dense);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "requested semi-structured dtype is not supported");
    }
    return dense;
}

namespace {

void validate_grad_input(const Tensor& grad, const Tensor& packed,
                         int64_t rows, int64_t cols) {
    if (!grad.defined() || grad.is_sparse() || grad.dim() != 2 ||
        grad.size(0) != rows || grad.size(1) != cols ||
        grad.device() != packed.device() || grad.dtype() != packed.dtype()) {
        TP_THROW(ValueError,
                 "semi-structured gradient must match the dense logical shape and dtype");
    }
}

} // namespace

Tensor sparse_semi_structured_mask_grad_cpu(
    const Tensor& grad, const Tensor& packed, const Tensor& meta) {
    SemiConfig cfg{};
    int64_t rows = 0;
    int64_t cols = 0;
    validate_representation(packed, meta, &cfg, &rows, &cols);
    validate_grad_input(grad, packed, rows, cols);
    Tensor grad_contiguous = grad.is_contiguous() ? grad : grad.contiguous();
    Tensor result = Tensor::empty({rows, cols}, packed.dtype(), packed.device());
    switch (packed.dtype()) {
        case DType::Float16:
            mask_grad_typed<Half, int16_t, 4, 2, 4>(
                grad_contiguous, packed, meta, result);
            break;
        case DType::BFloat16:
            mask_grad_typed<BFloat16, int16_t, 4, 2, 4>(
                grad_contiguous, packed, meta, result);
            break;
        case DType::Float32:
            mask_grad_typed<float, int16_t, 2, 1, 4>(
                grad_contiguous, packed, meta, result);
            break;
        case DType::Int8:
            mask_grad_typed<int8_t, int32_t, 4, 2, 8>(
                grad_contiguous, packed, meta, result);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "requested semi-structured gradient dtype is not supported");
    }
    return result;
}

Tensor sparse_semi_structured_gather_grad_cpu(
    const Tensor& grad, const Tensor& packed, const Tensor& meta) {
    SemiConfig cfg{};
    int64_t rows = 0;
    int64_t cols = 0;
    validate_representation(packed, meta, &cfg, &rows, &cols);
    validate_grad_input(grad, packed, rows, cols);
    Tensor grad_contiguous = grad.is_contiguous() ? grad : grad.contiguous();
    Tensor result = Tensor::empty(
        {packed.size(0), packed.size(1)}, packed.dtype(), packed.device());
    switch (packed.dtype()) {
        case DType::Float16:
            gather_grad_typed<Half, int16_t, 4, 2, 4>(
                grad_contiguous, packed, meta, result);
            break;
        case DType::BFloat16:
            gather_grad_typed<BFloat16, int16_t, 4, 2, 4>(
                grad_contiguous, packed, meta, result);
            break;
        case DType::Float32:
            gather_grad_typed<float, int16_t, 2, 1, 4>(
                grad_contiguous, packed, meta, result);
            break;
        case DType::Int8:
            gather_grad_typed<int8_t, int32_t, 4, 2, 8>(
                grad_contiguous, packed, meta, result);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "requested semi-structured gradient dtype is not supported");
    }
    return result;
}

Tensor sparse_semi_structured_mm_cpu(
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
    const DType result_dtype = out_dtype.value_or(packed.dtype());
    return mm_dispatch(packed_contiguous, meta_contiguous, dense_contiguous,
                       result_dtype, nullptr, 1.0, 1.0);
}

Tensor sparse_semi_structured_mm_right_cpu(
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
    const DType result_dtype = out_dtype.value_or(packed.dtype());
    switch (packed.dtype()) {
        case DType::Float16:
            if (result_dtype == DType::Float16) {
                return mm_right_same_typed<Half, int16_t, 4, 2, 4>(
                    dense_contiguous, packed_contiguous, meta_contiguous,
                    result_dtype);
            }
            if (result_dtype == DType::Float32) {
                return mm_right_float_typed<Half, int16_t, 4, 2, 4>(
                    dense_contiguous, packed_contiguous, meta_contiguous,
                    result_dtype);
            }
            break;
        case DType::BFloat16:
            if (result_dtype == DType::BFloat16) {
                return mm_right_same_typed<BFloat16, int16_t, 4, 2, 4>(
                    dense_contiguous, packed_contiguous, meta_contiguous,
                    result_dtype);
            }
            if (result_dtype == DType::Float32) {
                return mm_right_float_typed<BFloat16, int16_t, 4, 2, 4>(
                    dense_contiguous, packed_contiguous, meta_contiguous,
                    result_dtype);
            }
            break;
        case DType::Float32:
            if (result_dtype == DType::Float32) {
                return mm_right_same_typed<float, int16_t, 2, 1, 4>(
                    dense_contiguous, packed_contiguous, meta_contiguous,
                    result_dtype);
            }
            break;
        case DType::Int8:
            if (result_dtype == DType::Int8) {
                return mm_right_same_typed<int8_t, int32_t, 4, 2, 8>(
                    dense_contiguous, packed_contiguous, meta_contiguous,
                    result_dtype);
            }
            if (result_dtype == DType::Int32) {
                return mm_right_int32_typed<int8_t, int32_t, 4, 2, 8>(
                    dense_contiguous, packed_contiguous, meta_contiguous,
                    result_dtype);
            }
            break;
        default:
            break;
    }
    TP_THROW(NotImplementedError,
             "requested semi-structured output dtype is not supported");
}

Tensor sparse_semi_structured_addmm_cpu(
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
                       result_dtype, &input_contiguous, alpha.toDouble(),
                       beta.toDouble());
}

TENSORPLAY_LIBRARY_IMPL(CPU, SemiStructuredKernels) {
    m.impl("_to_sparse_semi_structured", sparse_semi_structured_compress_cpu);
    m.impl("_sparse_semi_structured_to_dense",
           sparse_semi_structured_to_dense_cpu);
    m.impl("_sparse_semi_structured_mask_grad",
           sparse_semi_structured_mask_grad_cpu);
    m.impl("_sparse_semi_structured_gather_grad",
           sparse_semi_structured_gather_grad_cpu);
    m.impl("_sparse_semi_structured_mm", sparse_semi_structured_mm_cpu);
    m.impl("_sparse_semi_structured_mm_right",
           sparse_semi_structured_mm_right_cpu);
    m.impl("_sparse_semi_structured_addmm", sparse_semi_structured_addmm_cpu);
}

} // namespace cpu
} // namespace tensorplay
