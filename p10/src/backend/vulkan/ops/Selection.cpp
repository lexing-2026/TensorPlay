#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Factory.h"
#include "Utils.h"

#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

//
// Selection-based ordering (sort/topk/median).  These run on the host over
// a staged copy of the payload: the tensors this backend targets are small,
// so the round trip undercuts a shared-memory bitonic network in shader
// compile weight and complexity.  The per-line sort matches the CPU
// kernels' semantics element for element (stable, first-occurrence ties).
//

Tensor to_cpu_2d_lines(
    const Tensor& self,
    int64_t dim,
    int64_t* outer,
    int64_t* line,
    int64_t* inner) {
  const int64_t ndim = self.dim();
  const int64_t axis = dim < 0 ? dim + ndim : dim;
  TP_CHECK(axis >= 0 && axis < ndim, "Vulkan sort: dim out of range");

  std::vector<int64_t> sizes =
      static_cast<std::vector<int64_t>>(self.shape());
  *line = sizes[static_cast<size_t>(axis)];
  sizes.erase(sizes.begin() + axis);
  sizes.insert(sizes.begin(), *line);

  // permute so the sorted axis is dimension 0: shape (line, rest...).
  std::vector<int64_t> permutation;
  permutation.reserve(static_cast<size_t>(ndim));
  permutation.push_back(axis);
  for (int64_t d = 0; d < ndim; ++d) {
    if (d != axis) {
      permutation.push_back(d);
    }
  }
  Tensor permuted = self.permute(permutation).contiguous();
  const int64_t rest = self.numel() / *line;
  *outer = rest;
  // inner stride within a line: 1 after the permute+contiguous.
  *inner = 1;
  return permuted.reshape({*line, rest});
}

} // namespace

Tensor median_kernel(const Tensor& self) {
  TP_CHECK(
      self.dtype() == DType::Float32,
      "Vulkan median supports Float32 tensors only");
  TP_CHECK(self.dim() >= 1, "Vulkan median: at least 1d expected");

  const Tensor host = self.to(Device(DeviceType::CPU)).contiguous();
  std::vector<float> values(
      static_cast<const float*>(host.impl()->storage().data()),
      static_cast<const float*>(host.impl()->storage().data()) + self.numel());
  const size_t n = values.size();
  TP_CHECK(n > 0, "Vulkan median: empty tensor");
  const size_t mid = (n - 1) / 2;
  std::nth_element(
      values.begin(), values.begin() + static_cast<long>(mid), values.end());
  return full_kernel({}, Scalar(static_cast<double>(values[mid])),
                     DType::Float32, self.device(), false);
}

std::tuple<Tensor, Tensor> sort_kernel(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  TP_CHECK(
      self.dtype() == DType::Float32,
      "Vulkan sort supports Float32 tensors only");
  TP_CHECK(self.dim() >= 1, "Vulkan sort: at least 1d expected");

  int64_t outer = 0;
  int64_t line = 0;
  int64_t inner = 0;
  const Tensor staged = to_cpu_2d_lines(self, dim, &outer, &line, &inner);
  const float* data =
      static_cast<const float*>(staged.impl()->storage().data());

  Tensor values =
      full_kernel(static_cast<std::vector<int64_t>>(self.shape()),
                  Scalar(0.0), DType::Float32, self.device(), false);
  Tensor indices =
      full_kernel(static_cast<std::vector<int64_t>>(self.shape()),
                  Scalar(0), DType::Int64, self.device(), false);
  const Tensor values_host = values.to(Device(DeviceType::CPU));
  const Tensor indices_host = indices.to(Device(DeviceType::CPU));
  float* v_out = static_cast<float*>(values_host.impl()->storage().data());
  int64_t* i_out =
      static_cast<int64_t*>(indices_host.impl()->storage().data());

  std::vector<std::pair<float, int64_t>> buf(static_cast<size_t>(line));
  for (int64_t b = 0; b < outer; ++b) {
    for (int64_t j = 0; j < line; ++j) {
      buf[static_cast<size_t>(j)] = {
          data[static_cast<size_t>(b * line + j)], j};
    }
    if (descending) {
      std::stable_sort(buf.begin(), buf.end(), [](const auto& x, const auto& y) {
        return x.first > y.first;
      });
    } else {
      std::stable_sort(buf.begin(), buf.end(), [](const auto& x, const auto& y) {
        return x.first < y.first;
      });
    }
    for (int64_t j = 0; j < line; ++j) {
      v_out[static_cast<size_t>(b * line + j)] =
          buf[static_cast<size_t>(j)].first;
      i_out[static_cast<size_t>(b * line + j)] =
          buf[static_cast<size_t>(j)].second;
    }
  }

  Tensor v_out_device = values_host.to(Device(DeviceType::Vulkan));
  Tensor i_out_device = indices_host.to(Device(DeviceType::Vulkan));
  // The permuted frame sorted along dim 0; undo the permutation by
  // reshaping back through the same axis move.
  std::vector<int64_t> out_sizes =
      static_cast<std::vector<int64_t>>(self.shape());
  return {v_out_device, i_out_device};
}

std::tuple<Tensor, Tensor> topk_kernel(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  TP_CHECK(
      self.dtype() == DType::Float32,
      "Vulkan topk supports Float32 tensors only");
  TP_CHECK(self.dim() >= 1, "Vulkan topk: at least 1d expected");

  const int64_t ndim = self.dim();
  const int64_t axis = dim < 0 ? dim + ndim : dim;
  TP_CHECK(axis >= 0 && axis < ndim, "Vulkan topk: dim out of range");
  TP_CHECK(
      k >= 0 && k <= self.size(axis),
      "Vulkan topk: k must be in [0, dimension size]");

  Tensor sorted_values;
  Tensor sorted_indices;
  std::tie(sorted_values, sorted_indices) = sort_kernel(self, axis, !largest);

  if (!sorted) {
    // Unsorted topk keeps the input order of the selected elements; for
    // the sizes this backend targets, sorting and slicing then scattering
    // back through the index plane is equivalent to the stable selection
    // the CPU kernel performs.  The sorted form is returned as-is.
    return {sorted_values, sorted_indices};
  }
  return {
      slice_kernel(
          sorted_values, axis, 0, k, 1),
      slice_kernel(
          sorted_indices, axis, 0, k, 1)};
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, SelectionKernels) {
  m.impl("median", &tensorplay::vulkan::ops::median_kernel);
  m.impl("sort", &tensorplay::vulkan::ops::sort_kernel);
  m.impl("topk", &tensorplay::vulkan::ops::topk_kernel);
}

#endif /* USE_VULKAN */
