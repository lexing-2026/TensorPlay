#pragma once

#include "Tensor.h"

namespace tensorplay {
namespace transform {
namespace batch {

Tensor unary(const char* op, const Tensor& input);
Tensor binary(const char* op, const Tensor& left, const Tensor& right);
Tensor binary_alpha(const char* op, const Tensor& left, const Tensor& right,
                    Scalar alpha);
Tensor scalar(const char* op, const Tensor& input, Scalar value);
Tensor scalar_alpha(const char* op, const Tensor& input, Scalar value,
                    Scalar alpha);
Tensor tensor_pow(const Tensor& left, const Tensor& right);
Tensor sum_all(const Tensor& input, DType dtype);
Tensor sum_dim(const Tensor& input, const std::vector<int64_t>& dims,
               bool keepdim, DType dtype);
Tensor view(const Tensor& input, const std::vector<int64_t>& shape);
Tensor permute(const Tensor& input, const std::vector<int64_t>& dims);
Tensor transpose(const Tensor& input, int64_t dim0, int64_t dim1);
Tensor movedim(const Tensor& input, const std::vector<int64_t>& source,
               const std::vector<int64_t>& destination);
Tensor reshape(const Tensor& input, const std::vector<int64_t>& shape);
Tensor expand(const Tensor& input, const std::vector<int64_t>& shape,
              bool implicit);
Tensor squeeze(const Tensor& input);
Tensor squeeze_dim(const Tensor& input, int64_t dim);
Tensor squeeze_dims(const Tensor& input, const std::vector<int64_t>& dims);
Tensor unsqueeze(const Tensor& input, int64_t dim);
Tensor contiguous(const Tensor& input, int64_t memory_format);
Tensor select(const Tensor& input, int64_t dim, int64_t index);
Tensor slice(const Tensor& input, int64_t dim,
             std::optional<int64_t> start, std::optional<int64_t> end,
             int64_t step);
Tensor narrow(const Tensor& input, int64_t dim, int64_t start, int64_t length);
Tensor index_select(const Tensor& input, int64_t dim, const Tensor& index);
Tensor cat(const std::vector<Tensor>& inputs, int64_t dim);
Tensor stack(const std::vector<Tensor>& inputs, int64_t dim);
Tensor mm(const Tensor& left, const Tensor& right);
Tensor matmul(const Tensor& left, const Tensor& right);
Tensor bmm(const Tensor& left, const Tensor& right);
Tensor rand(const std::vector<int64_t>& shape, std::optional<DType> dtype,
            std::optional<Device> device);
Tensor randn(const std::vector<int64_t>& shape, std::optional<DType> dtype,
             std::optional<Device> device);
Tensor randint(int64_t low, int64_t high, const std::vector<int64_t>& shape,
               DType dtype, std::optional<Device> device);
Tensor randperm(int64_t n, DType dtype, std::optional<Device> device);
Tensor rand_like(const Tensor& input, DType dtype,
                 std::optional<Device> device);
Tensor randint_like(const Tensor& input, int64_t low, int64_t high,
                    DType dtype, std::optional<Device> device);
Tensor randn_like(const Tensor& input, DType dtype,
                  std::optional<Device> device);

} // namespace batch
} // namespace transform
} // namespace tensorplay
