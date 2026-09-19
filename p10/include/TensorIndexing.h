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
  return tpx::ops::slice(self, dim, start, stop, step);
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
  return tpx::ops::select(self, dim, index);
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
  if (src.dim() == 0 && src.device().type() == DeviceType::CPU) {
    tpx::ops::fill_(const_cast<Tensor&>(dst), src);
    return;
  }
  Tensor src_view = tpx::ops::view(src, slicePrefix1sSize(src_sizes));
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
    Tensor result = tpx::ops::unsqueeze(prev_dim_result, *dim_ptr);
    (*dim_ptr)++;
    if (!out_indices.empty()) {
      out_indices.resize(out_indices.size() + 1);
    }
    return result;
  } else if (index.is_boolean()) {
    Tensor result = tpx::ops::unsqueeze(prev_dim_result, *dim_ptr);
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
        result = tpx::ops::unsqueeze(result, *dim_ptr);
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

// Index lists as the operator contract spells them: undefined entries are
// dimensions taken whole.  Index tensors follow the indexed tensor's device.
inline std::vector<std::optional<Tensor>> typeConvertIndices(
    const Tensor& self,
    std::vector<Tensor>&& indices) {
  std::vector<std::optional<Tensor>> converted;
  converted.reserve(indices.size());
  for (auto& index : indices) {
    if (!index.defined()) {
      converted.emplace_back(std::nullopt);
    } else if (index.device() != self.device()) {
      converted.emplace_back(index.to(self.device()));
    } else {
      converted.emplace_back(std::move(index));
    }
  }
  return converted;
}

inline Tensor dispatch_index(const Tensor& self, std::vector<Tensor>&& indices) {
  return tpx::ops::index(self, typeConvertIndices(self, std::move(indices)));
}

inline Tensor& dispatch_index_put_(
    Tensor& self,
    std::vector<Tensor>&& indices,
    const Tensor& value) {
  return tpx::ops::index_put_(
      self, typeConvertIndices(self, std::move(indices)), value);
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
      return tpx::ops::unsqueeze(self, 0);
    } else if (index.is_ellipsis()) {
      return tpx::ops::alias(self);
    } else if (index.is_boolean()) {
      Tensor result = tpx::ops::unsqueeze(self, 0);
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
      copy_to(tpx::ops::unsqueeze(self, 0), value);
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
