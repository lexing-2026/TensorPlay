#include "TensorIterator.h"
#include "TensorIteratorInternal.h"
#include "Tensor.h"
#include "TypePromotion.h"
#include "Utils.h"
#include "irange.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <utility>

namespace tensorplay {

using DimMask = TensorIteratorBase::DimMask;
using PtrVector = TensorIteratorBase::PtrVector;
using loop2d_t = TensorIteratorBase::loop2d_t;
using StrideVector = TensorIteratorBase::StrideVector;

namespace {

inline void get_base_ptrs_impl(char** ptrs, const std::vector<OperandInfo>& operands) {
  std::transform(operands.begin(), operands.end(), ptrs, [](const OperandInfo& op) {
    return static_cast<char*>(op.data);
  });
}

inline void get_strides_impl(int64_t* strides, const std::vector<OperandInfo>& operands, int64_t ndim) {
  for (const auto dim : irange(ndim)) {
    for (const auto arg : irange(operands.size())) {
      *strides++ = operands[arg].stride_bytes[dim];
    }
  }
  // Always at least 2d strides to support 2d for_each loops
  if (ndim < 2) {
    auto ntensors = operands.size();
    std::fill_n(strides, (2 - ndim) * ntensors, 0);
  }
}

// Broadcasts a shape of size `a_size` against `b_size` (see torch
// at::infer_size): every dim must be equal or one of them 1; result takes the
// max.
DimVector infer_size(const DimVector& a, const DimVector& b) {
  size_t ndim = std::max(a.size(), b.size());
  DimVector result(ndim, 1);
  for (size_t i = 0; i < ndim; ++i) {
    int64_t dim = 1;
    if (i < a.size()) dim = std::max(dim, a[a.size() - 1 - i]);
    if (i < b.size()) dim = std::max(dim, b[b.size() - 1 - i]);
    result[ndim - 1 - i] = dim;
  }
  return result;
}

bool same_sizes(const Tensor& a, const Tensor& b) {
  return static_cast<std::vector<int64_t>>(a.shape())
      == static_cast<std::vector<int64_t>>(b.shape());
}

bool same_sizes(const Tensor& a, const DimVector& sizes) {
  return static_cast<std::vector<int64_t>>(a.shape()) == sizes;
}

bool tensors_share_storage(const Tensor& a, const Tensor& b) {
  auto impl_a = a.unsafeGetTensorImpl();
  auto impl_b = b.unsafeGetTensorImpl();
  if (!impl_a || !impl_b) {
    return false;
  }
  return impl_a->storage().defined() && impl_b->storage().defined()
      && impl_a->storage().data() == impl_b->storage().data()
      && impl_a->storage().nbytes() == impl_b->storage().nbytes();
}

// Simplified canCast: casting is always allowed between the supported CPU
// types; only undefined dtypes are rejected.
bool can_cast(ScalarType from, ScalarType to) {
  return from != ScalarType::Undefined && to != ScalarType::Undefined;
}

Tensor resize_output(const Tensor& t, const DimVector& sizes) {
  if (same_sizes(t, sizes)) {
    return t;
  }
  // Write-only resizing: allocate a fresh tensor. Only used by the
  // will_resize path, which never occurs for reductions.
  return Tensor::empty(sizes, t.dtype(), t.device());
}

} // namespace

void TensorIteratorBase::reorder_dimensions() {
  // Sort the dimensions based on strides in ascending order with reduced dims
  // at the front. NOTE: that this inverts the order of C-contiguous tensors.
  // strides[0] is the fastest moving dimension instead of strides[ndim - 1].

  perm_.resize(ndim());
  if (ndim() == 1) {
    perm_[0] = 0;
    return;
  }

  // initialize perm with n-1, n-2, ..., 1, 0
  std::iota(perm_.rbegin(), perm_.rend(), 0);

  // Reordering dimensions changes iteration order
  if (enforce_linear_iteration_) {
    permute_dimensions(perm_);
    return;
  }

  // returns 1 if the dim0 should come after dim1, -1 if dim0 should come
  // before dim1, and 0 if the comparison is ambiguous.
  auto should_swap = [&](size_t dim0, size_t dim1) {
    for (const auto arg : irange(ntensors())) {
      // ignore undefined or incorrectly sized tensors
      if (operands_[arg].stride_bytes.empty() || operands_[arg].will_resize) {
        continue;
      }
      int64_t stride0 = operands_[arg].stride_bytes[dim0];
      int64_t stride1 = operands_[arg].stride_bytes[dim1];
      if (is_reduction_ && operands_[arg].is_output) {
        // move reduced dimensions to the front
        // strides of reduced dimensions are always set to 0 by review_reduce_result
        if ((stride0 == 0) != (stride1 == 0)) {
          return stride1 == 0 ? 1 : -1;
        }
      }
      //move on to the next input if one of the dimensions is broadcasted
      if (stride0 == 0 || stride1 == 0) {
        continue;
      // it is important to return here only with strict comparisons, for equal strides we try to break the tie later
      // by comparing corresponding dimensions or if that does not work, moving on to the next tensor
      } else if (stride0 < stride1) {
        return -1;
      } else  if (stride0 > stride1) {
        return 1;
      } else { //equal strides, use dimensions themselves as the tie-breaker.
        //at this point, with zero strides out of the way, we are guaranteed that operand dimensions are equal to shape_
         auto t_dim0 = shape_[dim0];
         auto t_dim1 = shape_[dim1];
         //return only if dimensions should be swapped, otherwise move on to the next tensor
         if (t_dim0 > t_dim1) {
             return 1;
         }
      }
    }
    return 0;
  };

  // insertion sort with support for ambiguous comparisons
  for (const auto i : irange(1, ndim())) {
    int dim1 = i;
    for (int dim0 = i - 1; dim0 >= 0; dim0--) {
      int comparison = should_swap(perm_[dim0], perm_[dim1]);
      if (comparison > 0) {
        std::swap(perm_[dim0], perm_[dim1]);
        dim1 = dim0;
      } else if (comparison < 0) {
        break;
      }
    }
  }

  // perform re-ordering of shape and strides
  permute_dimensions(perm_);
}

// Computes a common dtype using type promotion
// See the [Common Dtype Computation] note
ScalarType TensorIteratorBase::compute_common_dtype() {
  ScalarType result = ScalarType::Undefined;
  for (const auto& op : operands_) {
    if (op.is_output) {
      continue;
    }
    if (result == ScalarType::Undefined) {
      result = op.target_dtype;
    } else {
      result = promoteTypes(result, op.target_dtype);
    }
  }
  common_dtype_ = result;
  TP_CHECK(common_dtype_ != ScalarType::Undefined, "undefined common dtype");
  return common_dtype_;
}

// Implements the behavior of the following flags:
//   - check_all_same_dtype_
//   - check_all_same_device_
//   - enforce_safe_casting_to_output_
//   - promote_inputs_to_common_dtype_
//   - cast_common_dtype_to_outputs_
//
// See their descriptions in TensorIterator.h for details.
void TensorIteratorBase::compute_types(const TensorIteratorConfig& config) {
  // Reviews operands (1/2)
  //   - validates that all input tensors are defined
  //   - computes common device
  //   - determines if there are undefined outputs
  //   - determines if there are different dtypes and attempts
  //       to quickly acquire a common dtype
  Device common_device = Device(DeviceType::CPU);
  common_dtype_ = ScalarType::Undefined;
  // NB: despite output_dtype's generic sounding name, it only is
  // used in a nontrivial way if check_all_same_dtype is true
  ScalarType output_dtype = ScalarType::Undefined;
  bool has_different_input_dtypes = false;
  bool has_different_output_dtypes = false;
  bool has_undefined_outputs = false;

  for (auto& op : operands_) {
    // Validates that all inputs have type information, and that
    //   if an output is missing type information that we can infer
    //   the device it should be allocated on.
    if (!op.is_type_defined()) {
      TP_CHECK(op.is_output, "Found type undefined input tensor!");

      if (config.static_dtype_.has_value()) {
        op.target_dtype = config.static_dtype_.value();
      } else {
        has_undefined_outputs = true;
      }

      if (config.static_device_.has_value()) {
        op.device = config.static_device_.value();
      } else {
        TP_CHECK(config.check_all_same_device_, "expected check_all_same_device");
      }

      if (has_undefined_outputs || !op.device.has_value()) {
        continue;
      }
    }

    // Validates input tensors are defined
    if (!op.tensor().defined()) {
      TP_CHECK(op.is_output, "Found undefined input tensor!");
      continue;
    }

    TP_CHECK(op.target_dtype == op.current_dtype, "dtype mismatch on operand")

    // Acquires the first non-CPU device (if any) as the common device
    if (common_device.is_cpu() && !op.tensor().device().is_cpu()) {
      common_device = op.tensor().device();
    }

    if (!op.is_output) {
      // Determines if there are varying input dtypes
      // NOTE: the common dtype is set to the first defined input dtype observed
      if (op.target_dtype != common_dtype_) {
        if (common_dtype_ == ScalarType::Undefined) {
          common_dtype_ = op.target_dtype;
        } else {
          has_different_input_dtypes = true;
        }
      }
    } else {  // op.is_output
      // Determines if there are varying output dtypes
      // NOTE: the output dtype is set to the first defined output dtype observed
      if (op.target_dtype != output_dtype) {
        if (output_dtype == ScalarType::Undefined) {
          output_dtype = op.target_dtype;
        } else {
          has_different_output_dtypes = true;
        }
      }
    }
  }

  // Checks that either the computation type is computable or unneeded
  TP_CHECK(
      !(has_different_input_dtypes && !config.promote_inputs_to_common_dtype_ &&
        (has_undefined_outputs || config.enforce_safe_casting_to_output_ ||
         config.cast_common_dtype_to_outputs_)),
      "no common dtype computable");

  // Checks that all inputs and defined outputs are the same dtype, if requested
  if (config.check_all_same_dtype_ &&
      (has_different_input_dtypes || has_different_output_dtypes ||
       (common_dtype_ != output_dtype && output_dtype != ScalarType::Undefined))) {
    // Throws an informative error message
    for (auto& op : operands_) {
      if (!op.tensor().defined()) {
        continue;
      }
      TP_CHECK(
          op.target_dtype == common_dtype_,
          "Found dtype ", static_cast<int>(op.target_dtype), " but expected ",
          static_cast<int>(common_dtype_));
    }
  }

  // Short-circuits if no additional work required
  if (!has_undefined_outputs && !config.check_all_same_device_ &&
      !config.promote_inputs_to_common_dtype_ && !config.cast_common_dtype_to_outputs_ &&
      !config.enforce_safe_casting_to_output_) {
    // Invalidates common_dtype_ if it could not be inferred
    common_dtype_ = has_different_input_dtypes ? ScalarType::Undefined : common_dtype_;
    return;
  }

  // Computes a common dtype, if needed
  if ((has_different_input_dtypes || all_ops_are_scalars_) && config.promote_inputs_to_common_dtype_) {
    common_dtype_ = compute_common_dtype();
  }

  // Promotes common dtype to the default float scalar type, if needed
  if (config.promote_integer_inputs_to_float_ &&
      isIntegralType(common_dtype_, /*includeBool=*/true)) {
    common_dtype_ = DType::Float32;
  }

  // Reviews operands (2/2)
  //   - sets metadata for undefined outputs
  //   - checks that all tensors are on the same device, if requested
  //   - checks that the common dtype can safely cast to each output, if requested
  //   - creates temporaries for CPU operations, if needed and requested
  common_device_ = common_device;
  for (auto& op : operands_) {
    bool is_type_defined = op.is_type_defined();
    bool is_device_defined = op.is_device_defined();

    if (!is_type_defined) {
      op.target_dtype = common_dtype_;
    }
    if (!is_device_defined) {
      op.device = common_device;
    }

    if (!is_type_defined && !is_device_defined) {
      continue;
    }

    // Skips undefined tensors
    if (!op.tensor().defined()) {
      continue;
    }

    // Checks all tensors are on the same device, if requested
    if (config.check_all_same_device_) {
      TP_CHECK(
          op.device.value_or(Device(DeviceType::CPU)) == common_device,
          "Expected all tensors to be on the same device, but found at least "
          "two devices");
    }

    // Checks safe casting, if requested
    if (config.enforce_safe_casting_to_output_ && op.is_output && op.current_dtype != common_dtype_) {
      TP_CHECK(
          can_cast(common_dtype_, op.current_dtype),
          "result type can't be cast to the desired output type");
    }

    // Creates temporaries for CPU operations, if needed and requested
    if (common_device.is_cpu()) {
      // Casts to outputs by creating temporaries of the correct dtype (if needed)
      if (config.cast_common_dtype_to_outputs_ && op.is_output && op.current_dtype != common_dtype_) {
        TP_CHECK(op.tensor().defined(), "output not defined");
        op.exchange_tensor(
            Tensor::empty_like(op.tensor(), common_dtype_, op.tensor().device()));
        op.current_dtype = common_dtype_;
        op.target_dtype = common_dtype_;
      }

      // Promotes inputs by creating temporaries of the correct dtype
      if (config.promote_inputs_to_common_dtype_ && !op.is_output && op.current_dtype != common_dtype_) {
        op.exchange_tensor(op.tensor().to(common_dtype_));
        op.current_dtype = common_dtype_;
        op.target_dtype = common_dtype_;
      }
    }
  }
}

StrideVector TensorIteratorBase::compatible_stride(int64_t element_size) const {
  auto stride = StrideVector();
  int64_t next_stride = element_size;
  for (const auto dim : irange(ndim())) {
    stride.push_back(next_stride);
    next_stride *= shape_[dim];
  }
  return stride;
}

DimVector TensorIteratorBase::invert_perm(const DimVector& input) const {
  // Invert the permutation caused by reorder_dimensions. This is not valid
  // after coalesce_dimensions is called.
  TP_CHECK(!has_coalesced_dimensions_, "cannot invert perm after coalescing");
  TP_CHECK(input.size() == perm_.size(), "perm size mismatch");
  auto res = DimVector(input.size()); //no initialization needed, every value in res should be written to.
  for (const auto dim : irange(ndim())) {
    res[perm_[dim]] = input[dim];
  }
  return res;
}

DimVector TensorIteratorBase::apply_perm_and_mul(const DimVector& input, int mul) const {
  TP_CHECK(!has_coalesced_dimensions_, "cannot apply perm after coalescing");
  TP_CHECK(input.size() == perm_.size(), "perm size mismatch");
  auto res = DimVector(input.size());
  for (const auto i : irange(perm_.size())) {
    res[i] = input[perm_[i]] * mul;
  }
  return res;
}

void TensorIteratorBase::allocate_or_resize_outputs() {
  // check if permutation is just an inverted order
  bool inverted = true;
  for (const auto j : irange(ndim())) {
    if (perm_[j] != ndim() - j - 1) {
      inverted = false;
      break;
    }
  }
  for (const auto i : irange(num_outputs_)) {
    auto& op = operands_[i];
    if (!op.tensor().defined() || op.will_resize) {
      TP_CHECK(op.is_type_defined(), "no type for operand", i);
      auto element_size = elementSize(op.target_dtype);
      op.stride_bytes = compatible_stride(static_cast<int64_t>(element_size));
      auto tensor_shape = invert_perm(shape_);
      if (inverted) {
        // can just return contiguous output
        // it is faster because it avoids allocating 0 size tensor and
        // resizing and restriding it
        set_output_raw_strided(i, tensor_shape, {}, op.target_dtype, op.options_device());
      } else {
        auto tensor_stride = invert_perm(op.stride_bytes);
        for (const auto dim : irange(ndim())) {
          tensor_stride[dim] /= static_cast<int64_t>(element_size);
        }
        set_output_raw_strided(i, tensor_shape, tensor_stride, op.target_dtype, op.options_device());
      }
      op.current_dtype = op.target_dtype;
    } else if (op.tensor().defined()) {
      // Even if we don't resize, we still need to tell set_output about
      // the output, so that we properly set guard
      set_output_raw_strided(
          i,
          static_cast<std::vector<int64_t>>(op.tensor().shape()),
          {},
          op.tensor().dtype(),
          op.tensor().device());
    }
  }
}

void TensorIteratorBase::coalesce_dimensions() {
  if (ndim() <= 1) {
    return;
  }


  // We can coalesce two adjacent dimensions if either dim has size 1 or if:
  // shape[n] * stride[n] == stride[n + 1].
  auto can_coalesce = [&](int dim0, int dim1) {
    auto shape0 = shape_[dim0];
    auto shape1 = shape_[dim1];
    if (shape0 == 1 || shape1 == 1) {
      return true;
    }
    for (const auto i : irange(ntensors())) {
      auto& stride = operands_[i].stride_bytes;
      if (shape0 * stride[dim0] != stride[dim1]) {
        return false;
      }
    }
    return true;
  };

  // replace each operands stride at dim0 with its stride at dim1
  auto replace_stride = [&](int dim0, int dim1) {
    for (const auto i : irange(ntensors())) {
      auto& stride = operands_[i].stride_bytes;
      stride[dim0] = stride[dim1];
    }
  };

  int prev_dim = 0;
  for (const auto dim : irange(1, ndim())) {
    if (can_coalesce(prev_dim, dim)) {
      if (shape_[prev_dim] == 1) {
        replace_stride(prev_dim, dim);
      }
      shape_[prev_dim] *= shape_[dim];
    } else {
      prev_dim++;
      if (prev_dim != dim) {
        replace_stride(prev_dim, dim);
        shape_[prev_dim] = shape_[dim];
      }
    }
  }

  shape_.resize(prev_dim + 1);
  for (const auto i : irange(ntensors())) {
    operands_[i].stride_bytes.resize(ndim());
  }
  has_coalesced_dimensions_ = true;

}

int64_t TensorIteratorBase::numel() const {
  int64_t numel = 1;
  for (int64_t size : shape_) {
    numel *= size;
  }
  return numel;
}

StrideVector TensorIteratorBase::get_dim_strides(int dim) const {
  auto dims = ndim();
  auto inner_strides = StrideVector();
  for (auto& op : operands_) {
    inner_strides.push_back(dims == 0 ? 0 : op.stride_bytes[dim]);
  }
  return inner_strides;
}

PtrVector TensorIteratorBase::get_base_ptrs() const {
  auto ptrs = PtrVector(ntensors());
  get_base_ptrs_impl(ptrs.data(), operands_);
  return ptrs;
}

bool TensorIteratorBase::is_dim_reduced(int dim) const {
  for (auto& op : operands_) {
    if (op.is_output && op.stride_bytes[dim] == 0 && shape_[dim] > 1) {
      return true;
    }
  }
  return false;
}

void TensorIteratorBase::permute_dimensions(const DimVector& perm) {
  TP_CHECK(perm.size() == static_cast<unsigned>(ndim()), "perm size mismatch");

  auto reorder = [perm](const DimVector& data) {
    auto res = DimVector(data.size(), 0);
    for (const auto i : irange(perm.size())) {
      res[i] = data[perm[i]];
    }
    return res;
  };

  // Update shape and strides
  shape_ = reorder(shape_);
  for (auto& op : operands_) {
    if (!op.stride_bytes.empty()) {
      op.stride_bytes = reorder(op.stride_bytes);
    }
  }
}

int64_t TensorIteratorBase::num_output_elements() const {
  int64_t elem = 1;
  for (const auto dim : irange(ndim())) {
    if (operands_[0].stride_bytes[dim] != 0 || shape_[dim] == 0)  {
      elem *= shape_[dim];
    }
  }
  return elem;
}

int TensorIteratorBase::num_reduce_dims() const {
  int count = 0;
  for (const auto dim : irange(ndim())) {
    if (operands_[0].stride_bytes[dim] == 0) {
      count++;
    }
  }
  return count;
}

void TensorIteratorBase::for_each(loop2d_t loop, int64_t grain_size) {
  int64_t numel = this->numel();
  if (numel == 0) {
    return;
  } else if (numel < grain_size || parallel::get_num_threads() == 1) {
    serial_for_each(loop, {0, numel});
    return;
  } else {
    parallel::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
      serial_for_each(loop, {begin, end});
    });
  }
}

StrideVector TensorIteratorBase::get_strides() const {
  const auto dim = ndim();
  StrideVector strides(static_cast<size_t>(std::max(dim, 2)) * ntensors());
  get_strides_impl(strides.data(), operands_, dim);
  return strides;
}

void TensorIteratorBase::serial_for_each(loop2d_t loop, Range range) const {
  if (range.size() == 0) {
    return;
  }

  const auto ntensors = this->ntensors();
  const auto ndim = this->ndim();

  std::vector<char*> ptrs(ntensors);
  std::vector<int64_t> strides(ntensors * static_cast<size_t>(std::max(ndim, 2)));

  get_base_ptrs_impl(ptrs.data(), operands_);
  get_strides_impl(strides.data(), operands_, ndim);
  internal::serial_for_each(shape_, strides, ptrs.data(), ptrs.size(), loop, range);
}

bool TensorIteratorBase::is_trivial_1d() const {
  // TODO: check for casting once it's supported
  return ndim() == 1;
}

bool TensorIteratorBase::is_contiguous() const {
  if (numel() == 1) {
    return true;
  }
  if (ndim() != 1) {
    return false;
  }
  return has_contiguous_first_dim();
}

bool TensorIteratorBase::is_scalar(int64_t arg) const {
  const auto& stride = operands_[arg].stride_bytes;
  for (const auto i : irange(ndim())) {
    if (stride[i] != 0 && shape_[i] != 1) {
      return false;
    }
  }
  return true;
}

bool TensorIteratorBase::is_cpu_scalar(int64_t arg) const {
  return is_scalar(arg) && device(arg).is_cpu();
}

void TensorIteratorBase::cast_outputs() {
  for (auto& op : operands_) {
    if (op.is_output && op.original_tensor().defined() &&
        op.original_tensor().dtype() != op.current_dtype) {
      auto& original_tensor = op.original_tensor();
      auto& tensor = op.tensor();
      if (!same_sizes(original_tensor, tensor)) {
        auto resized = Tensor::empty(
            static_cast<std::vector<int64_t>>(tensor.shape()),
            original_tensor.dtype(),
            original_tensor.device());
        resized.copy_(tensor);
        original_tensor.copy_(resized);
      } else {
        original_tensor.copy_(tensor);
      }
      op.restore_original_tensor();
    }
  }
}

void* TensorIteratorBase::data_ptr(int64_t arg) const {
  return operands_[arg].data;
}

void TensorIteratorBase::remove_operand(int64_t arg) {
  operands_.erase(operands_.begin() + arg);
}

void TensorIteratorBase::unsafe_replace_operand(int64_t arg, void* data) {
  operands_[arg].data = data;
}

void TensorIteratorBase::narrow(int dim, int64_t start, int64_t size) {
  TP_CHECK(dim < ndim() && size >= 1, "invalid narrow");
  shape_[dim] = size;
  view_offsets_[dim] += start;
  for (auto& op : operands_) {
    op.data = (static_cast<char*>(op.data)) + op.stride_bytes[dim] * start;
  }
  if (size == 1 && !is_reduction_) {
    coalesce_dimensions();
  }
}

void TensorIteratorBase::select_all_keeping_dim(int start_dim, const DimVector& indices) {
  TP_CHECK(start_dim <= ndim(), "invalid start_dim");
  for (const auto i : irange(start_dim, ndim())) {
    for (auto& op : operands_) {
      op.data = (static_cast<char*>(op.data)) + op.stride_bytes[i] * indices[i - start_dim];
    }
    shape_[i] = 1;
  }
}

TensorIterator TensorIterator::reduce_op(Tensor& out, const Tensor& a) {
  TP_CHECK(out.defined(), "output must be defined");
  return TensorIteratorConfig()
    .set_check_mem_overlap(false)
    .add_owned_output(out)
    .add_owned_const_input(a)
    .resize_outputs(false)
    .is_reduction(true)
    // TODO: not supporting casting to outputs is only really necessary for arg{min,max}
    .promote_inputs_to_common_dtype(true)
    .build();
}

TensorIterator TensorIterator::reduce_op(Tensor& out1, Tensor& out2, const Tensor& a) {
  TP_CHECK(out1.defined(), "output1 must be defined");
  TP_CHECK(out2.defined(), "output2 must be defined");
  TP_CHECK(
      a.device() == out1.device() && out1.device() == out2.device(),
      "reduce_op(): expected input and both outputs to be on same device");
  TP_CHECK(out1.dim() == out2.dim(), "reduce_op(): expected both outputs to have same number of dims");
  TP_CHECK(same_sizes(out1, out2), "reduce_op(): expected both outputs to have same sizes");
  TP_CHECK(
      static_cast<std::vector<int64_t>>(out1.strides())
          == static_cast<std::vector<int64_t>>(out2.strides()),
      "reduce_op(): expected both outputs to have same strides");
  return TensorIteratorConfig()
    .set_check_mem_overlap(false)
    .add_owned_output(out1)
    .add_owned_output(out2)
    .add_owned_const_input(a)
    .resize_outputs(false)
    .is_reduction(true)
    .check_all_same_dtype(false)
    .build();
}

void TensorIteratorBase::populate_operands(TensorIteratorConfig& config) {
  for (const auto idx : irange(config.tensors_.size())) {
    auto& tensor = config.tensors_[idx];
    operands_.emplace_back(tensor);
    operands_[idx].is_const = config.is_tensor_const(idx);
  }
  num_outputs_ = config.num_outputs_;
}

void TensorIteratorBase::mark_outputs() {
  // TODO: merge this into populate_operands
  for (const auto i : irange(num_outputs_)) {
    operands_[i].is_output = true;
    const auto& output = tensor(i);
    if (!output.defined()) continue;

    // check if output is also an input
    for (const auto arg : irange(num_outputs_, ntensors())) {
      const auto& input = tensor(arg);
      if (output.unsafeGetTensorImpl() == input.unsafeGetTensorImpl()) {
        operands_[i].is_read_write = true;
      }
    }
  }
}

void TensorIteratorBase::mark_resize_outputs(const TensorIteratorConfig& config) {
  // Outputs cannot be broadcasted. Check that the shape of the outputs matches
  // the inferred shape. There's an exception for write-only tensors to support
  // our legacy behavior that functions with `out=` arguments resize their
  // outputs.
  if (config.static_shape_.has_value()) {
    return;
  }
  for (const auto i : irange(num_outputs_)) {
    const auto& output = tensor(i);
    if (!output.defined()) {
      operands_[i].will_resize = true;
    }
    if (output.defined() && !same_sizes(output, shape_)) {
      if (config.resize_outputs_ && !operands_[i].is_read_write) {
        operands_[i].will_resize = true;
        continue;
      }
      // for reduction, output size does not match shape_, as output is reduced size, and shape_ is size of the input
      TP_CHECK(is_reduction_, "output with shape doesn't match the broadcast shape");
    }
  }
}

void TensorIteratorBase::compute_mem_overlaps(const TensorIteratorConfig& config) {
  if (!config.check_mem_overlap_) {
    return;
  }
  for (const auto i : irange(num_outputs_)) {
    const auto& output = tensor(i);
    if (!output.defined()) continue;
    for (const auto j : irange(num_outputs_, ntensors())) {
      const auto& input = tensor(j);
      if (output.unsafeGetTensorImpl() != input.unsafeGetTensorImpl()
          && tensors_share_storage(output, input)) {
        TP_CHECK(false, "unsupported memory overlap between output and input");
      }
    }
  }
}

void TensorIteratorBase::compute_shape(const TensorIteratorConfig& config) {
  if (config.static_shape_.has_value()) {
    shape_ = *config.static_shape_;
    return;
  }

  all_ops_same_shape_ = true;
  bool has_scalars = false;
  bool has_tensors = false;
  for (auto& op : operands_) {
    if (!op.tensor().defined()) continue;

    // For now, don't include output tensors when we're resizing outputs.
    // These shapes don't participate in shape computation.
    // This preserves the legacy behavior where torch.add(..., out=dst) resizes
    // the destination tensor.  If the output tensor is also an input, we'll
    // pick it up later in the operands.
    if (config.resize_outputs_ && op.is_output) continue;
    auto shape = static_cast<std::vector<int64_t>>(op.tensor().shape());
    if (shape.empty()) {
      has_scalars = true;
    } else {
      has_tensors = true;
    }
    if (has_scalars && has_tensors) {
      all_ops_same_shape_ = false;
    }
    if (shape_.empty()) {
      shape_ = shape;
    } else if (shape != shape_) {
      all_ops_same_shape_ = false;
      shape_ = infer_size(shape_, shape);
    }
  }
  all_ops_are_scalars_ = !has_tensors;
}

void TensorIteratorBase::compute_strides(const TensorIteratorConfig& config) {
  for (auto& op : operands_) {
    if (op.tensor().defined() && !op.will_resize) {
      auto original_shape = config.static_shape_.has_value()
          ? shape_
          : static_cast<std::vector<int64_t>>(op.tensor().shape());
      auto original_stride = op.tensor().strides();
      auto element_size_in_bytes = op.tensor().itemsize();
      auto offset = ndim() - original_shape.size();
      if (offset > 0)
          op.stride_bytes.resize(ndim(), 0);
      else
          op.stride_bytes.resize(ndim());
      for (const auto i : irange(original_shape.size())) {
        // see NOTE: [Computing output strides]
        if (original_shape[i] == 1 && shape_[offset + i] !=1) {
          op.stride_bytes[offset + i] = 0;
        } else {
          op.stride_bytes[offset + i] = original_stride[i] * element_size_in_bytes;
        }
      }
    }
  }
}

bool TensorIteratorBase::can_use_32bit_indexing() const {
  int64_t max_value = std::numeric_limits<int32_t>::max();
  if (numel() > max_value) {
    return false;
  }
  for (auto& op : operands_) {
    int64_t max_offset = 1;
    for (const auto dim : irange(ndim())) {
      max_offset += (shape_[dim] - 1) * op.stride_bytes[dim];
    }
    if (max_offset > max_value) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<TensorIterator> TensorIteratorBase::split(int dim) {
  TP_CHECK(dim >= 0 && dim < ndim() && shape()[dim] >= 2, "invalid split dim");
  auto copy = std::make_unique<TensorIterator>(*this);

  bool overlaps = is_dim_reduced(dim);
  auto copy_size = shape_[dim] / 2;
  auto this_size = shape_[dim] - copy_size;
  copy->narrow(dim, 0, copy_size);
  copy->final_output_ &= !overlaps;
  this->narrow(dim, copy_size, this_size);
  this->accumulate_ |= overlaps;

  return copy;
}

int TensorIteratorBase::get_dim_to_split() const {
  TP_CHECK(ndim() >= 1, "no dims to split");
  int64_t max_extent = -1;
  int dim_to_split = -1;
  for (int dim = ndim() - 1; dim >= 0; dim--) {
    const int64_t size = shape_[dim];
    if (size == 0) {
      continue;
    }
    for (auto& op : operands_) {
      // std::abs is necessary to handle some special cases where we support negative strides
      // see the CUDA backend of at::flip
      const int64_t extent = (size - 1) * std::abs(op.stride_bytes[dim]);
      if (extent > max_extent) {
        max_extent = extent;
        dim_to_split = dim;
      }
    }
  }
  TP_CHECK(max_extent >= 0, "invalid extent");
  return dim_to_split;
}

bool TensorIteratorBase::fast_set_up(const TensorIteratorConfig& config) {
  // This function tries to do a fast setup to avoid needless reordering of dimensions and tracking output strides
  // Return true if it can do fast setup or false otherwise
  // TODO enable fast handling for reductions
  FastSetupType setup_type = compute_fast_setup_type(config);
  if (setup_type == FastSetupType::NONE) {
    return false;
  }

  // allocate memory for output, memory format depends on setup_type
  switch (setup_type) {
    case FastSetupType::CONTIGUOUS:
      {
        for (const auto i : irange(num_outputs_)) {
          auto& op = operands_[i];
          if (!op.tensor().defined()) {
            TP_CHECK(op.is_type_defined(), "no type for operand", i);
          }
          set_output_raw_strided(i, shape_, {}, op.target_dtype, op.options_device());
        }
        break;
      }
    default:
      TP_CHECK(false, "Unsupported fast setup type");
  }
  //coalescing dimensions consists of collapsing dimensions to 1 (we are limited to contiguous no-broadcast cases here)
  if (ndim() > 1){
    has_coalesced_dimensions_ = true;
  }
  if (ndim() >= 1) {
    shape_[0] = numel();
    shape_.resize(1);
  }
  for (auto& op : operands_ ) {
    auto element_size_in_bytes = op.tensor().itemsize();
    op.stride_bytes.resize(ndim());
    if (ndim()>0) {
      op.stride_bytes[0] = element_size_in_bytes;
    }
  }
  return true;
}

FastSetupType TensorIteratorBase::compute_fast_setup_type(const TensorIteratorConfig& config) {
  if (is_reduction_ || !all_ops_same_shape_) {
    return FastSetupType::NONE;
  }

  // For linear iteration, only contiguous tensors can be coalesced
  // Fast setup of any other format requires changing iteration order
  if (enforce_linear_iteration_) {
    for (const auto& op : operands_) {
      if (op.tensor().defined() && !op.will_resize) {
        if (!op.tensor().is_contiguous()) {
          return FastSetupType::NONE;
        }
      }
    }
    return FastSetupType::CONTIGUOUS;
  }

  for (const auto& op : operands_) {
    if (op.tensor().defined() && !op.will_resize) {
      if (!op.tensor().is_contiguous()) {
        return FastSetupType::NONE;
      }
    }
  }
  return FastSetupType::CONTIGUOUS;
}

void TensorIteratorBase::build(TensorIteratorConfig& config) {
  // populate some persistent configuration fields
  is_reduction_ = config.is_reduction_;
  enforce_linear_iteration_ = config.enforce_linear_iteration_;

  // fill in operands_ based on configuration
  populate_operands(config);
  // set is_output and is_read_write flags on appropriate tensors
  mark_outputs();
  // Check that the outputs have no internal overlap
  // and do not share memory with inputs.
  compute_mem_overlaps(config);
  // compute the broadcasted shape
  compute_shape(config);
  // mark outputs for resizing if necessary
  mark_resize_outputs(config);
  // compute the result dtype and device
  compute_types(config);
  // try fast setup output tensor, if failed, fallback to normal setup
  if (!fast_set_up(config)) {
    // compute each tensor's stride after broadcasting
    compute_strides(config);
    // re-order dimensions to improve coalescing
    reorder_dimensions();
    // allocate the output tensor if it's not provided
    allocate_or_resize_outputs();
    // coalesce adjacent dimensions when possible
    coalesce_dimensions();
  }

  for (auto& op : operands_) {
    TP_CHECK(op.tensor().defined(), "operand not defined after build");
    op.data = op.tensor().data_ptr();
  }

  // zero out offsets
  // If the tensor is a scalar, we leave room for it
  // So index translations in reduction can access
  // a valid value for the offset
  int64_t ndim_offsets = (ndim() ? ndim() : 1);
  view_offsets_.assign(ndim_offsets, 0);
}

void TensorIteratorBase::set_output_raw_strided(
    int64_t output_idx,
    const DimVector& sizes,
    const DimVector& strides,
    DType dtype,
    Device device) {
  auto& op = operands_[output_idx];
  TP_CHECK(output_idx < num_outputs_, "output index out of bounds");
  if (!op.tensor().defined()) {
    if (strides.empty()) {
      op.set_tensor(Tensor::empty(sizes, dtype, device));
    } else {
      op.set_tensor(Tensor::empty(sizes, dtype, device).as_strided(sizes, strides));
    }
    op.current_dtype = op.target_dtype;
  } else if (op.will_resize) {
    auto resized = resize_output(op.tensor(), sizes);
    if (!strides.empty()) {
      resized = resized.as_strided(sizes, strides);
    }
    op.set_tensor(resized);
    op.current_dtype = op.target_dtype;
  }
}

} // namespace tensorplay
