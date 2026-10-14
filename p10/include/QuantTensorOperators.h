#pragma once

#include "Tensor.h"
#include "Quantizer.h"
#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay::quantized {

P10_API Tensor storage_view(const Tensor& self);
P10_API Tensor with_quantizer(const Tensor& codes, const Tensor& self);
P10_API Tensor activation(const Tensor& self, int operation, double a = 0., double b = 1.);
P10_API Tensor& activation_out(const Tensor& self, Tensor& out, int operation, double a = 0., double b = 1.);
P10_API Tensor& relu_inplace(Tensor& self);
P10_API Tensor& fill(Tensor& self, const Scalar& value);
P10_API Tensor& masked_fill(Tensor& self, const Tensor& mask, const Scalar& value);
P10_API Tensor extrema(const Tensor& self, bool maximum);
P10_API std::tuple<Tensor, Tensor> extrema_dim(const Tensor& self, int64_t dim, bool keepdim, bool maximum);
P10_API Tensor moment(const Tensor& self, const std::vector<int64_t>& dims, bool keepdim,
                     bool standard_deviation, double correction = 1., DType dtype = DType::Undefined);
P10_API Tensor& index_put(Tensor& self, const std::vector<std::optional<Tensor>>& indices,
                         const Tensor& values, bool accumulate, bool unsafe);
P10_API Tensor& cat_out(const std::vector<Tensor>& tensors, int64_t dim, Tensor& out);
P10_API Tensor prelu(const Tensor& self, const Tensor& weight);
P10_API Tensor grid_sample(const Tensor& self, const Tensor& grid, int64_t interpolation,
                          int64_t padding, bool align_corners);

} // namespace tensorplay::quantized
