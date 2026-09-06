#pragma once

//
// C++ tensor indexing.  An index expression such as
// `{None, "...", 0, true, Slice(1, None, 2), index_tensor}` resolves under
// the same rules the Python front end applies:
//
//   integer            -> select along the next axis
//   Slice(a, b, s)     -> narrow along the next axis
//   None               -> insert a length-one axis
//   "..." / Ellipsis   -> absorb the remaining unspecified axes
//   true / false       -> keep all / no elements along the next axis
//   index tensor       -> advanced gather; integer index tensors broadcast
//                         against each other, boolean masks select rows
//
// Basic indices apply first; the advanced index tensors then gather over
// the sliced payload.  When the advanced positions are adjacent the gather
// shape replaces them in place, otherwise it moves to the front.
//

#include "Tensor.h"
#include "Exception.h"
#include "Utils.h"

#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cstring>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

namespace tensorplay {
namespace indexing {

constexpr int64_t INDEX_MIN = std::numeric_limits<int64_t>::min();
constexpr int64_t INDEX_MAX = -(INDEX_MIN + 1);

enum class TensorIndexType { None, Ellipsis, Integer, Boolean, Slice, Tensor };

constexpr std::nullopt_t None = std::nullopt;

struct EllipsisIndexType final {
  EllipsisIndexType() = default;
};

inline constexpr EllipsisIndexType Ellipsis{};

struct Slice final {
 public:
  Slice(
      std::optional<int64_t> start_index = std::nullopt,
      std::optional<int64_t> stop_index = std::nullopt,
      std::optional<int64_t> step_index = std::nullopt) {
    step_ = step_index.has_value() ? *step_index : 1;
    TP_CHECK(step_ != 0, "slice step cannot be zero");

    start_ = start_index.has_value() ? *start_index
                                     : (step_ < 0 ? INDEX_MAX : 0);
    stop_ = stop_index.has_value() ? *stop_index
                                   : (step_ < 0 ? INDEX_MIN : INDEX_MAX);
  }

  inline int64_t start() const {
    return start_;
  }

  inline int64_t stop() const {
    return stop_;
  }

  inline int64_t step() const {
    return step_;
  }

 private:
  int64_t start_;
  int64_t stop_;
  int64_t step_;
};

//
// A single element of a C++ index list, holding one of: None, Ellipsis,
// an integer, a boolean, a Slice, or an index Tensor.
//
struct TensorIndex final {
  // Case 1: None
  TensorIndex(std::nullopt_t /*unused*/) : type_(TensorIndexType::None) {}

  // Case 2: Ellipsis or "..."
  TensorIndex(EllipsisIndexType /*unused*/) : type_(TensorIndexType::Ellipsis) {}
  TensorIndex(const char* str) : TensorIndex(Ellipsis) {
    TP_CHECK(
        strcmp(str, "...") == 0,
        "Expected \"...\" to represent an ellipsis index, but got \"",
        str,
        "\"");
  }

  // Case 3: integer value
  TensorIndex(int64_t integer) : integer_(integer), type_(TensorIndexType::Integer) {}
  TensorIndex(int integer) : TensorIndex(static_cast<int64_t>(integer)) {}

  // Case 4: boolean value
  template <class T, class = std::enable_if_t<std::is_same_v<bool, T>>>
  TensorIndex(T boolean) : boolean_(boolean), type_(TensorIndexType::Boolean) {}

  // Case 5: Slice
  TensorIndex(Slice slice) : slice_(std::move(slice)), type_(TensorIndexType::Slice) {}

  // Case 6: Tensor
  TensorIndex(Tensor tensor) : tensor_(std::move(tensor)), type_(TensorIndexType::Tensor) {}

  inline bool is_none() const {
    return type_ == TensorIndexType::None;
  }

  inline bool is_ellipsis() const {
    return type_ == TensorIndexType::Ellipsis;
  }

  inline bool is_integer() const {
    return type_ == TensorIndexType::Integer;
  }

  inline int64_t integer() const {
    return integer_;
  }

  inline bool is_boolean() const {
    return type_ == TensorIndexType::Boolean;
  }

  inline bool boolean() const {
    return boolean_;
  }

  inline bool is_slice() const {
    return type_ == TensorIndexType::Slice;
  }

  inline const Slice& slice() const {
    return slice_;
  }

  inline bool is_tensor() const {
    return type_ == TensorIndexType::Tensor;
  }

  inline const Tensor& tensor() const {
    return tensor_;
  }

 private:
  int64_t integer_ = 0;
  bool boolean_ = false;
  Slice slice_;
  Tensor tensor_;
  TensorIndexType type_;
};

inline std::ostream& operator<<(std::ostream& stream, const Slice& slice) {
  stream << slice.start() << ':' << slice.stop() << ':' << slice.step();
  return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const TensorIndex& tensor_index) {
  if (tensor_index.is_none()) {
    stream << "None";
  } else if (tensor_index.is_ellipsis()) {
    stream << "...";
  } else if (tensor_index.is_integer()) {
    stream << tensor_index.integer();
  } else if (tensor_index.is_boolean()) {
    stream << (tensor_index.boolean() ? "true" : "false");
  } else if (tensor_index.is_slice()) {
    stream << tensor_index.slice();
  } else if (tensor_index.is_tensor()) {
    stream << tensor_index.tensor();
  }
  return stream;
}

inline std::ostream& operator<<(
    std::ostream& stream,
    const std::vector<TensorIndex>& tensor_indices) {
  stream << '(';
  for (size_t i = 0; i < tensor_indices.size(); ++i) {
    stream << tensor_indices[i];
    if (i + 1 < tensor_indices.size()) stream << ", ";
  }
  stream << ')';
  return stream;
}

namespace impl {

inline Tensor applySlice(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t stop,
    int64_t step,
    bool disable_slice_optimization) {
  TP_CHECK(step > 0, "step must be greater than zero");

  // A slice that spans the whole axis aliases the input; dispatching the
  // narrow would produce an identical view anyway.  Callers that must
  // observe a fresh view (single-axis get-item) disable the shortcut.
  const std::vector<int64_t> sizes =
      static_cast<std::vector<int64_t>>(self.shape());
  if (!disable_slice_optimization && !sizes.empty() && start == 0 &&
      sizes[dim] <= stop && step == 1) {
    return self;
  }
  return self.slice(dim, start, stop, step);
}

inline Tensor applySelect(
    const Tensor& self,
    int64_t dim,
    int64_t index,
    int64_t real_dim) {
  if (self.dim() == 0) {
    TP_CHECK_INDEX(
        false,
        "invalid index of a 0-dim tensor. ",
        "Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number");
  }
  const int64_t size = self.size(dim);
  // Negative indices wrap from the end; -size is the first element and
  // -size - 1 is out of bounds.
  TP_CHECK_INDEX(
      size > index && (index >= 0 || size + index >= 0),
      "index ",
      index,
      " is out of bounds for dimension ",
      real_dim,
      " with size ",
      size);
  return self.select(dim, index);
}

// A boolean index adds one axis: true keeps it whole, false empties it.
inline Tensor boolToIndexingTensor(const Tensor& self, bool value) {
  if (value) {
    return Tensor::full({1}, Scalar(0), DType::Int64, self.device());
  }
  return Tensor::full({0}, Scalar(0), DType::Int64, self.device());
}

inline void recordTensorIndex(
    const Tensor& tensor,
    std::vector<Tensor>& out_indices,
    int64_t* dim_ptr) {
  if (out_indices.empty()) {
    out_indices.resize(static_cast<size_t>(*dim_ptr) + 1);
    out_indices[static_cast<size_t>(*dim_ptr)] = tensor;
  } else {
    out_indices.push_back(tensor);
  }
  // A boolean or byte mask spans one input axis per mask axis; any other
  // index tensor spans exactly one.
  if (tensor.dtype() == DType::UInt8 || tensor.dtype() == DType::Bool) {
    *dim_ptr += tensor.dim();
  } else {
    *dim_ptr += 1;
  }
}

// Count the indexed axes: everything except None and Ellipsis contributes,
// with masks counting one per mask axis.
inline int64_t count_specified_dimensions(
    const std::vector<TensorIndex>& indices) {
  int64_t count = 0;
  for (const auto& obj : indices) {
    if (obj.is_tensor()) {
      const Tensor& tensor = obj.tensor();
      if (tensor.dtype() == DType::UInt8 || tensor.dtype() == DType::Bool) {
        count += tensor.dim();
      } else {
        count++;
      }
    } else if (!obj.is_none() && !obj.is_ellipsis() && !obj.is_boolean()) {
      count++;
    }
  }
  return count;
}

} // namespace impl

// Boolean masks decode on the host: for every true lane the mask's own
// coordinates become one element in each axis's index vector.
inline std::vector<std::vector<int64_t>> decode_bool_mask(
    const Tensor& mask) {
  Tensor host_mask = mask;
  if (!host_mask.device().is_cpu()) {
    host_mask = host_mask.to(Device(DeviceType::CPU));
  }
  host_mask = host_mask.contiguous();
  const int64_t mask_dim = mask.dim();
  const int64_t mask_numel = host_mask.numel();
  const auto sizes = static_cast<std::vector<int64_t>>(mask.shape());

  std::vector<std::vector<int64_t>> coordinates(static_cast<size_t>(mask_dim));
  if (mask.dtype() == DType::Bool) {
    const bool* lanes = host_mask.data_ptr<bool>();
    for (int64_t linear = 0; linear < mask_numel; ++linear) {
      if (!lanes[linear]) continue;
      int64_t remainder = linear;
      std::vector<int64_t> current(static_cast<size_t>(mask_dim));
      for (int64_t d = mask_dim - 1; d >= 0; --d) {
        const int64_t size = sizes[static_cast<size_t>(d)];
        current[static_cast<size_t>(d)] = size == 0 ? 0 : remainder % size;
        if (size != 0) remainder /= size;
      }
      for (int64_t d = 0; d < mask_dim; ++d) {
        coordinates[static_cast<size_t>(d)].push_back(
            current[static_cast<size_t>(d)]);
      }
    }
  } else {
    const uint8_t* lanes = host_mask.data_ptr<uint8_t>();
    for (int64_t linear = 0; linear < mask_numel; ++linear) {
      if (!lanes[linear]) continue;
      int64_t remainder = linear;
      std::vector<int64_t> current(static_cast<size_t>(mask_dim));
      for (int64_t d = mask_dim - 1; d >= 0; --d) {
        const int64_t size = sizes[static_cast<size_t>(d)];
        current[static_cast<size_t>(d)] = size == 0 ? 0 : remainder % size;
        if (size != 0) remainder /= size;
      }
      for (int64_t d = 0; d < mask_dim; ++d) {
        coordinates[static_cast<size_t>(d)].push_back(
            current[static_cast<size_t>(d)]);
      }
    }
  }
  return coordinates;
}

namespace detail {

// One advanced index tensor after materialization: the axis it gathers
// over, the broadcast shape of the index, and the (host) index values in
// row-major order of that shape.
struct AdvancedComponent final {
  int64_t input_dim = -1;
  std::vector<int64_t> shape;
  std::vector<int64_t> values;
};

// Materializes one integer index tensor into an AdvancedComponent: values
// wrap for negative codes and bounds-check against their axis.
inline AdvancedComponent make_advanced_component(
    const Tensor& self,
    int64_t input_dim,
    const Tensor& raw_index) {
  TP_CHECK(
      isIntegralType(raw_index.dtype(), /*includeBool=*/false),
      "tensors used as indices must be long, int, short, byte or bool tensors");
  AdvancedComponent component;
  component.input_dim = input_dim;
  component.shape = static_cast<std::vector<int64_t>>(raw_index.shape());

  Tensor host_index = raw_index.to(DType::Int64);
  if (!host_index.device().is_cpu()) {
    host_index = host_index.to(Device(DeviceType::CPU));
  }
  host_index = host_index.contiguous();
  const int64_t dim_size = self.size(input_dim);
  const int64_t n = host_index.numel();
  component.values.resize(static_cast<size_t>(n));
  if (n > 0) {
    std::memcpy(
        component.values.data(),
        host_index.data_ptr<int64_t>(),
        static_cast<size_t>(n) * sizeof(int64_t));
  }
  for (int64_t& value : component.values) {
    if (value < 0) value += dim_size;
    TP_CHECK_INDEX(
        value >= 0 && value < dim_size,
        "index ",
        value,
        " is out of bounds for dimension ",
        input_dim,
        " with size ",
        dim_size);
  }
  return component;
}

// A boolean or byte mask fans out into one component per mask axis: for
// every true lane, the mask's own coordinates enter each axis's index
// vector element-wise.
inline std::vector<AdvancedComponent> make_mask_components(
    const Tensor& self,
    int64_t input_dim,
    const Tensor& mask) {
  const int64_t mask_dim = mask.dim();
  TP_CHECK_INDEX(
      mask_dim > 0 && input_dim + mask_dim <= self.dim(),
      "The shape of the mask does not match the indexed tensor");
  const auto self_sizes = static_cast<std::vector<int64_t>>(self.shape());
  for (int64_t d = 0; d < mask_dim; ++d) {
    TP_CHECK_INDEX(
        mask.size(d) == self_sizes[static_cast<size_t>(input_dim + d)],
        "The shape of the mask does not match the indexed tensor");
  }
  const std::vector<std::vector<int64_t>> coordinates = decode_bool_mask(mask);
  const int64_t count = coordinates.empty()
                            ? 0
                            : static_cast<int64_t>(coordinates[0].size());
  std::vector<AdvancedComponent> components;
  components.reserve(static_cast<size_t>(mask_dim));
  for (int64_t d = 0; d < mask_dim; ++d) {
    AdvancedComponent component;
    component.input_dim = input_dim + d;
    component.shape = {count};
    component.values = coordinates[static_cast<size_t>(d)];
    components.push_back(std::move(component));
  }
  return components;
}

inline int64_t checked_shape_numel(const std::vector<int64_t>& shape) {
  int64_t result = 1;
  for (const int64_t size : shape) {
    if (size < 0 ||
        (size != 0 && result > std::numeric_limits<int64_t>::max() / size)) {
      TP_THROW(RuntimeError, "invalid or overflowing indexed shape");
    }
    result *= size;
  }
  return result;
}

//
// The advanced gather plan: map every output element to one flat source
// offset.  `indexed` lists one component per indexed axis in axis order;
// every other axis of `self` copies through.  The gather shape sits in
// place when the indexed axes are adjacent, otherwise it moves to the
// front.
//
struct AdvancedPlan final {
  std::vector<int64_t> output_shape;
  std::vector<int64_t> linear_indices;
};

inline AdvancedPlan build_advanced_plan(
    const Tensor& self,
    const std::vector<AdvancedComponent>& components) {
  AdvancedPlan plan;
  const int64_t ndim = self.dim();

  std::vector<int64_t> advanced_shape;
  for (const auto& component : components) {
    if (advanced_shape.empty()) {
      advanced_shape = component.shape;
    } else {
      advanced_shape = broadcast_shapes(advanced_shape, component.shape);
    }
  }

  int64_t first_dim = ndim;
  int64_t last_dim = -1;
  for (const auto& component : components) {
    first_dim = std::min(first_dim, component.input_dim);
    last_dim = std::max(last_dim, component.input_dim);
  }
  const bool adjacent =
      components.empty() || (last_dim - first_dim + 1 ==
                             static_cast<int64_t>(components.size()));

  // Output axis assignment for every input axis: -1 for absorbed axes.
  std::vector<int64_t> axis_of_dim(static_cast<size_t>(ndim), -1);
  std::vector<const AdvancedComponent*> component_of_dim(
      static_cast<size_t>(ndim), nullptr);
  for (const auto& component : components) {
    component_of_dim[static_cast<size_t>(component.input_dim)] = &component;
  }

  const auto push_basic_axis = [&](int64_t d) {
    axis_of_dim[static_cast<size_t>(d)] =
        static_cast<int64_t>(plan.output_shape.size());
    plan.output_shape.push_back(self.size(d));
  };

  int64_t advanced_start = 0;
  if (adjacent) {
    for (int64_t d = 0; d < ndim; ++d) {
      if (d == first_dim) {
        advanced_start = static_cast<int64_t>(plan.output_shape.size());
        for (const int64_t size : advanced_shape) {
          plan.output_shape.push_back(size);
        }
      }
      if (component_of_dim[static_cast<size_t>(d)] == nullptr) {
        push_basic_axis(d);
      }
    }
  } else {
    for (const int64_t size : advanced_shape) {
      plan.output_shape.push_back(size);
    }
    for (int64_t d = 0; d < ndim; ++d) {
      if (component_of_dim[static_cast<size_t>(d)] == nullptr) {
        push_basic_axis(d);
      }
    }
  }

  const int64_t advanced_rank = static_cast<int64_t>(advanced_shape.size());
  const int64_t output_numel = checked_shape_numel(plan.output_shape);
  plan.linear_indices.resize(static_cast<size_t>(output_numel));

  // The broadcast value of one component at a given output coordinate: the
  // component's own rank aligns to the trailing advanced axes, and
  // singleton axes read coordinate zero.
  const auto advanced_value =
      [&](const AdvancedComponent& component,
          const std::vector<int64_t>& output_coords) {
        int64_t offset = 0;
        const int64_t component_rank =
            static_cast<int64_t>(component.shape.size());
        for (int64_t d = 0; d < component_rank; ++d) {
          const int64_t output_dim = advanced_start + advanced_rank -
                                     component_rank + d;
          const int64_t coordinate =
              component.shape[static_cast<size_t>(d)] == 1
                  ? 0
                  : output_coords[static_cast<size_t>(output_dim)];
          offset = offset * component.shape[static_cast<size_t>(d)] + coordinate;
        }
        return component.values[static_cast<size_t>(offset)];
      };

  std::vector<int64_t> output_coords(plan.output_shape.size(), 0);
  for (int64_t linear = 0; linear < output_numel; ++linear) {
    int64_t source_linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
      int64_t coordinate = 0;
      const AdvancedComponent* component =
          component_of_dim[static_cast<size_t>(d)];
      if (component != nullptr) {
        coordinate = advanced_value(*component, output_coords);
      } else {
        coordinate = output_coords[static_cast<size_t>(
            axis_of_dim[static_cast<size_t>(d)])];
      }
      source_linear = source_linear * self.size(d) + coordinate;
    }
    plan.linear_indices[static_cast<size_t>(linear)] = source_linear;

    int64_t remainder = linear;
    for (int64_t d = static_cast<int64_t>(plan.output_shape.size()) - 1;
         d >= 0;
         --d) {
      const int64_t size = plan.output_shape[static_cast<size_t>(d)];
      output_coords[static_cast<size_t>(d)] = size == 0 ? 0 : remainder % size;
      if (size != 0) remainder /= size;
    }
  }
  return plan;
}

// Flattens the payload into a row-major 1-D view and gathers with the
// precomputed offsets.
inline Tensor gather_linear(const Tensor& self, const AdvancedPlan& plan) {
  Tensor index = Tensor::tensor(plan.linear_indices, DType::Int64);
  if (self.device().is_vulkan()) {
    // The 8-byte code type has no texture format on the device; the
    // gather pipeline narrows the codes to the 4-byte store, so the index
    // rides the device in that width.
    index = index.to(DType::Int32).to(self.device());
  } else if (!self.device().is_cpu()) {
    index = index.to(self.device());
  }
  Tensor flat = self.reshape({self.numel()});
  Tensor selected = tpx::ops::index_select(flat, 0, index);
  return tpx::ops::reshape(selected, plan.output_shape);
}

} // namespace detail

// To match the scalar-assignment semantics of the element-wise set path:
// strip leading unit axes off the source before broadcasting it against
// the destination.
inline std::vector<int64_t> slicePrefix1sSize(
    const std::vector<int64_t>& sizes) {
  size_t first_non1 = sizes.size();
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] != 1) {
      first_non1 = i;
      break;
    }
  }
  return std::vector<int64_t>(sizes.begin() + static_cast<long>(first_non1),
                              sizes.end());
}

inline void copy_to(const Tensor& dst, const Tensor& src) {
  const auto dst_sizes =
      static_cast<std::vector<int64_t>>(dst.shape());
  const auto src_sizes =
      static_cast<std::vector<int64_t>>(src.shape());
  bool same_sizes = dst_sizes.size() == src_sizes.size();
  if (same_sizes) {
    for (size_t i = 0; i < dst_sizes.size(); ++i) {
      if (dst_sizes[i] != src_sizes[i]) {
        same_sizes = false;
        break;
      }
    }
  }
  if (same_sizes) {
    tpx::ops::copy_(const_cast<Tensor&>(dst), src);
    return;
  }
  if (src.dim() == 0) {
    tpx::ops::fill_(const_cast<Tensor&>(dst), src.item());
    return;
  }
  Tensor src_view = src.view(slicePrefix1sSize(src_sizes));
  Tensor expanded = tpx::ops::expand(src_view, dst_sizes);
  tpx::ops::copy_(const_cast<Tensor&>(dst), expanded);
}

inline Tensor handleDimInMultiDimIndexing(
    const Tensor& prev_dim_result,
    const Tensor& original_tensor,
    const TensorIndex& index,
    int64_t* dim_ptr,
    const int64_t* specified_dims_ptr,
    int64_t real_dim,
    std::vector<Tensor>& out_indices,
    bool disable_slice_optimization) {
  if (index.is_integer()) {
    return impl::applySelect(
        prev_dim_result, *dim_ptr, index.integer(), real_dim);
  } else if (index.is_slice()) {
    Tensor result = impl::applySlice(
        prev_dim_result,
        *dim_ptr,
        index.slice().start(),
        index.slice().stop(),
        index.slice().step(),
        disable_slice_optimization);
    (*dim_ptr)++;
    if (!out_indices.empty()) {
      out_indices.resize(out_indices.size() + 1);
    }
    return result;
  } else if (index.is_ellipsis()) {
    const int64_t ellipsis_ndims =
        original_tensor.dim() - *specified_dims_ptr;
    (*dim_ptr) += ellipsis_ndims;
    if (!out_indices.empty()) {
      out_indices.resize(out_indices.size() +
                         static_cast<size_t>(ellipsis_ndims));
    }
    return prev_dim_result;
  } else if (index.is_none()) {
    Tensor result = prev_dim_result.unsqueeze(*dim_ptr);
    (*dim_ptr)++;
    if (!out_indices.empty()) {
      out_indices.resize(out_indices.size() + 1);
    }
    return result;
  } else if (index.is_boolean()) {
    Tensor result = prev_dim_result.unsqueeze(*dim_ptr);
    impl::recordTensorIndex(
        impl::boolToIndexingTensor(result, index.boolean()),
        out_indices,
        dim_ptr);
    return result;
  } else if (index.is_tensor()) {
    Tensor result = prev_dim_result;
    const Tensor& tensor = index.tensor();
    if (tensor.dim() == 0 &&
        isIntegralType(tensor.dtype(), /*includeBool=*/true)) {
      if (tensor.dtype() != DType::UInt8 && tensor.dtype() != DType::Bool) {
        result = impl::applySelect(
            result, *dim_ptr, tensor.item().to<int64_t>(), real_dim);
      } else {
        result = result.unsqueeze(*dim_ptr);
        const bool flag = tensor.dtype() == DType::Bool
                              ? tensor.item().to<bool>()
                              : tensor.item().to<uint8_t>() != 0;
        impl::recordTensorIndex(
            impl::boolToIndexingTensor(result, flag), out_indices, dim_ptr);
      }
    } else {
      impl::recordTensorIndex(tensor, out_indices, dim_ptr);
    }
    return result;
  } else {
    TP_THROW(RuntimeError, "Invalid TensorIndex type");
  }
}

namespace impl {

inline Tensor applySlicing(
    const Tensor& self,
    const std::vector<TensorIndex>& indices,
    std::vector<Tensor>& out_indices,
    bool disable_slice_optimization) {
  int64_t dim = 0;
  const int64_t specified_dims = count_specified_dimensions(indices);

  TP_CHECK_INDEX(
      specified_dims <= self.dim(),
      "too many indices for tensor of dimension ",
      self.dim());

  Tensor result = self;
  for (size_t i = 0; i < indices.size(); ++i) {
    result = handleDimInMultiDimIndexing(
        /*prev_dim_result=*/result,
        /*original_tensor=*/self,
        /*index=*/indices[i],
        /*dim_ptr=*/&dim,
        /*specified_dims_ptr=*/&specified_dims,
        /*real_dim=*/static_cast<int64_t>(i),
        /*out_indices=*/out_indices,
        /*disable_slice_optimization=*/disable_slice_optimization);
  }
  return result;
}

} // namespace impl

// Expands one positional index entry into advanced components: a mask
// fans out into one component per mask axis, an integer index becomes a
// single component.  Returns the number of input axes consumed.
inline int64_t expand_index_entry(
    const Tensor& self,
    int64_t dim,
    const Tensor& entry,
    std::vector<detail::AdvancedComponent>& components) {
  if (entry.dtype() == DType::Bool || entry.dtype() == DType::UInt8) {
    std::vector<detail::AdvancedComponent> mask_components =
        detail::make_mask_components(self, dim, entry);
    const int64_t consumed = static_cast<int64_t>(mask_components.size());
    for (auto& component : mask_components) {
      components.push_back(std::move(component));
    }
    return consumed;
  }
  components.push_back(detail::make_advanced_component(self, dim, entry));
  return 1;
}

inline Tensor dispatch_index(const Tensor& self, std::vector<Tensor> indices) {
  while (!indices.empty() && !indices.back().defined()) {
    indices.pop_back();
  }
  if (indices.empty()) {
    return self;
  }

  // Entry k gathers over axis k; undefined trailing entries were dropped
  // above, so every remaining entry consumes its axis in order.
  const int64_t ndim = self.dim();
  std::vector<detail::AdvancedComponent> components;
  int64_t dim = 0;
  for (const auto& entry : indices) {
    TP_CHECK_INDEX(
        dim < ndim,
        "too many indices for tensor of dimension ",
        ndim);
    if (!entry.defined()) {
      dim += 1;
      continue;
    }
    dim += expand_index_entry(self, dim, entry, components);
  }
  if (components.empty()) {
    return self;
  }

  const detail::AdvancedPlan plan = detail::build_advanced_plan(self, components);
  return detail::gather_linear(self, plan);
}

inline Tensor& dispatch_index_put_(
    Tensor& self,
    std::vector<Tensor> indices,
    const Tensor& value,
    bool accumulate = false) {
  while (!indices.empty() && !indices.back().defined()) {
    indices.pop_back();
  }
  if (indices.empty()) {
    if (accumulate) {
      copy_to(self, tpx::ops::add(self, value));
    } else {
      copy_to(self, value);
    }
    return self;
  }

  const int64_t ndim = self.dim();
  std::vector<detail::AdvancedComponent> components;
  int64_t dim = 0;
  for (const auto& entry : indices) {
    TP_CHECK_INDEX(
        dim < ndim,
        "too many indices for tensor of dimension ",
        ndim);
    if (!entry.defined()) {
      dim += 1;
      continue;
    }
    dim += expand_index_entry(self, dim, entry, components);
  }
  if (components.empty()) {
    if (accumulate) {
      copy_to(self, tpx::ops::add(self, value));
    } else {
      copy_to(self, value);
    }
    return self;
  }

  const detail::AdvancedPlan plan = detail::build_advanced_plan(self, components);
  if (plan.linear_indices.empty()) {
    return self;
  }

  Tensor rhs = value;
  if (rhs.dtype() != self.dtype() || rhs.device() != self.device()) {
    rhs = rhs.to(self.device(), self.dtype());
  }
  rhs = rhs.view(slicePrefix1sSize(
            static_cast<std::vector<int64_t>>(rhs.shape())))
            .expand(plan.output_shape)
            .reshape({detail::checked_shape_numel(plan.output_shape)})
            .contiguous()
            .clone();

  // A single flat index tensor drives the linear writer; the host planner
  // produces Int64 offsets, and the backend's linear writer consumes them
  // on the destination's device.
  Tensor index = Tensor::tensor(plan.linear_indices, DType::Int64);
  if (!self.device().is_cpu()) {
    index = index.to(self.device());
  }
  Tensor target = self.is_contiguous() ? self : self.contiguous();
  Tensor flat_target = target.view({-1});
  tpx::ops::index_put_(
      flat_target, std::vector<Tensor>{index}, rhs, accumulate);
  if (!self.is_contiguous()) {
    tpx::ops::copy_(self, target);
  }
  return self;
}

//
// The get-item entry: basic indices resolve through select/slice/
// unsqueeze; any advanced index tensor finishes through dispatch_index.
//
inline Tensor get_item(
    const Tensor& self,
    const std::vector<TensorIndex>& indices) {
  // handle simple types: integers, slices, none, ellipsis, bool
  if (indices.size() == 1) {
    const TensorIndex& index = indices[0];
    if (index.is_integer()) {
      return impl::applySelect(self, 0, index.integer(), 0);
    } else if (index.is_slice()) {
      return impl::applySlice(
          self,
          0,
          index.slice().start(),
          index.slice().stop(),
          index.slice().step(),
          /*disable_slice_optimization=*/true);
    } else if (index.is_none()) {
      return self.unsqueeze(0);
    } else if (index.is_ellipsis()) {
      return tpx::ops::alias(self);
    } else if (index.is_boolean()) {
      Tensor result = self.unsqueeze(0);
      return dispatch_index(
          result,
          std::vector<Tensor>{impl::boolToIndexingTensor(result, index.boolean())});
    }
  }

  std::vector<Tensor> tensor_indices;
  Tensor sliced =
      impl::applySlicing(self, indices, tensor_indices,
                         /*disable_slice_optimization=*/false);
  if (tensor_indices.empty()) {
    return sliced;
  }
  return dispatch_index(sliced, std::move(tensor_indices));
}

// Scalar assignment materializes the value with the destination's dtype on
// the destination's device before the tensor path applies it.
inline Tensor scalarToTensor(
    const Scalar& v,
    DType dtype,
    const Device& device) {
  return tpx::ops::scalar_tensor(v, dtype, device);
}

inline void set_item(
    const Tensor& self,
    const std::vector<TensorIndex>& indices,
    const Tensor& value) {
  if (indices.size() == 1) {
    const TensorIndex& index = indices[0];
    if (index.is_boolean() && !index.boolean()) {
      // Assigning through a false boolean touches no element.
      return;
    } else if (index.is_ellipsis()) {
      copy_to(self, value);
      return;
    } else if (index.is_none() || (index.is_boolean() && index.boolean())) {
      copy_to(self.unsqueeze(0), value);
      return;
    } else if (index.is_integer()) {
      copy_to(impl::applySelect(self, 0, index.integer(), 0), value);
      return;
    } else if (index.is_slice()) {
      copy_to(
          impl::applySlice(
              self,
              0,
              index.slice().start(),
              index.slice().stop(),
              index.slice().step(),
              /*disable_slice_optimization=*/false),
          value);
      return;
    }
  }

  std::vector<Tensor> tensor_indices;
  Tensor sliced =
      impl::applySlicing(self, indices, tensor_indices,
                         /*disable_slice_optimization=*/false);
  if (tensor_indices.empty()) {
    copy_to(sliced, value);
    return;
  }
  dispatch_index_put_(sliced, std::move(tensor_indices), value);
}

inline void set_item(
    const Tensor& self,
    const std::vector<TensorIndex>& indices,
    const Scalar& v) {
  Tensor value = scalarToTensor(v, self.dtype(), self.device());
  set_item(self, indices, value);
}

} // namespace indexing

#ifndef TENSORPLAY_INDEXING_SKIP_TENSOR_MEMBERS
//
// Member definitions for the indexing surface declared on Tensor.  Kept
// out of Tensor.h so the core header does not pull the generated op
// front end.
//
inline Tensor Tensor::operator[](const Scalar& index) const {
  TP_CHECK_INDEX(
      index.isIntegral(/*includeBool=*/false),
      "Can only index tensors with integral scalars");
  return (*this)[index.to<int64_t>()];
}

inline Tensor Tensor::operator[](const Tensor& index) const {
  TP_CHECK_INDEX(index.defined(), "Can only index with tensors that are defined");
  TP_CHECK_INDEX(
      index.dim() == 0,
      "Can only index with tensors that are scalars (zero-dim)");
  return (*this)[index.item().to<int64_t>()];
}

inline Tensor Tensor::operator[](int64_t index) const {
  return select(0, index);
}

inline Tensor Tensor::index(
    const std::vector<indexing::TensorIndex>& indices) const {
  TP_CHECK(
      !indices.empty(),
      "Passing an empty index list to Tensor::index() is not valid syntax");
  return indexing::get_item(*this, indices);
}

inline Tensor Tensor::index(
    std::initializer_list<indexing::TensorIndex> indices) const {
  return index(std::vector<indexing::TensorIndex>(indices));
}

inline Tensor& Tensor::index_put_(
    const std::vector<indexing::TensorIndex>& indices,
    const Tensor& rhs) {
  TP_CHECK(
      !indices.empty(),
      "Passing an empty index list to Tensor::index_put_() is not valid syntax");
  indexing::set_item(*this, indices, rhs);
  return *this;
}

inline Tensor& Tensor::index_put_(
    const std::vector<indexing::TensorIndex>& indices,
    const Scalar& v) {
  TP_CHECK(
      !indices.empty(),
      "Passing an empty index list to Tensor::index_put_() is not valid syntax");
  indexing::set_item(*this, indices, v);
  return *this;
}

inline Tensor& Tensor::index_put_(
    std::initializer_list<indexing::TensorIndex> indices,
    const Tensor& rhs) {
  return index_put_(std::vector<indexing::TensorIndex>(indices), rhs);
}

inline Tensor& Tensor::index_put_(
    std::initializer_list<indexing::TensorIndex> indices,
    const Scalar& v) {
  return index_put_(std::vector<indexing::TensorIndex>(indices), v);
}
#endif // TENSORPLAY_INDEXING_SKIP_TENSOR_MEMBERS

} // namespace tensorplay
