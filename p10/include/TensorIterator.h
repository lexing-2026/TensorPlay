#pragma once

#include "Tensor.h"
#include "DType.h"
#include "Device.h"
#include "Exception.h"
#include "Parallel.h"
#include "irange.h"
#include <bitset>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensorplay {

using DimVector = std::vector<int64_t>;
using StrideVector = std::vector<int64_t>;

struct Range {
  int64_t begin;
  int64_t end;
  int64_t size() const { return end - begin; }
};

struct SplitUntil32Bit;
class TensorIteratorConfig;

struct OperandInfo {
  OperandInfo() = default;
  explicit OperandInfo(Tensor t) : tensor_(std::move(t)) {
    if (tensor_.defined()) {
      device = tensor_.device();
      target_dtype = tensor_.dtype();
      current_dtype = target_dtype;
    }
    validate();
  }
  OperandInfo(const OperandInfo&) = default;
  OperandInfo& operator=(const OperandInfo&) = default;
  OperandInfo(OperandInfo&&) noexcept = default;
  OperandInfo& operator=(OperandInfo&&) noexcept = default;
  ~OperandInfo() = default;

  /// The data pointer. This may be different from tensor->data_ptr() if the
  /// iterator is split.
  void* data = nullptr;

  /// Stride after broadcasting. The stride is in bytes, not number of elements.
  StrideVector stride_bytes;

  /// The desired device and type for the operand. For inputs, this specifies
  /// that the input should be converted to this type if necessary. For outputs,
  /// this specifies which type to allocate.
  std::optional<Device> device = std::nullopt;
  ScalarType target_dtype = ScalarType::Undefined;
  // Caches dtype of the tensor; updated when the tensor is replaced (e.g. by
  // type promotion or output allocation).
  ScalarType current_dtype = ScalarType::Undefined;

  bool is_device_defined() const { return device.has_value(); }
  bool is_type_defined() const { return target_dtype != ScalarType::Undefined; }
  DType options_dtype() const { return target_dtype; }
  Device options_device() const {
    return device.value_or(Device(DeviceType::CPU));
  }

  bool is_output = false;

  // will_resize is only for output tensors. When a defined output does not
  // match the computed broadcast shape, it is replaced by a freshly allocated
  // tensor of the right shape (write-only semantics).
  bool will_resize = false;

  bool is_read_write = false;

  bool is_const = false;

  void validate() {}

  const Tensor& tensor() const { return tensor_; }
  Tensor& tensor() { return tensor_; }
  const Tensor& original_tensor() const { return original_tensor_; }
  Tensor& original_tensor() { return original_tensor_; }

  // Set tensor to a new value, and store the old tensor value in
  // original_tensor. Should only ever be called once for the lifetime of an
  // operand.
  void exchange_tensor(Tensor new_tensor) {
    original_tensor_ = tensor_;
    tensor_ = std::move(new_tensor);
  }

  // Move original_tensor back into tensor; exchange_tensor must have been
  // called before.
  void restore_original_tensor() {
    tensor_ = std::move(original_tensor_);
    original_tensor_ = Tensor();
  }

  void set_tensor(Tensor new_tensor) {
    tensor_ = std::move(new_tensor);
  }

 private:
  Tensor tensor_;
  Tensor original_tensor_;
};

enum class FastSetupType : uint8_t {
  NONE,
  CONTIGUOUS
};

struct TensorIterator;

//   - tensorplay:: namespace, Tensor held by value (shared_ptr semantics)
//   - IntArrayRef -> std::vector<int64_t>, SmallVector -> std::vector
//   - function_ref -> std::function
//   - no meta tensors, no MemoryFormat (fast setup only allocates
//     contiguous), no TensorOptions (DType+Device passed directly)
struct TensorIteratorBase {
  using DimMask = std::bitset<64>;
  using PtrVector = std::vector<char*>;
  using StrideVector = std::vector<int64_t>;

  // The inner-loop function operates on the fastest moving dimension. It
  // implements element-wise operations in terms of 1-d strided tensors.
  //
  // Arguments:
  //  data: data pointers for each operand (length `ntensors`)
  //  strides: stride for each operand (length `ntensors` * ndim, at least 2d)
  //  size0: size of the inner loop (fastest moving dimension)
  //  size1: size of the outer loop
  using loop1d_t = std::function<void(
      char** data, const int64_t* strides, int64_t size)>;

  using loop2d_t = std::function<void(
      char** data, const int64_t* strides, int64_t size0, int64_t size1)>;

  using loop_subiter_t = std::function<void(TensorIteratorBase& subiter)>;

  void build(TensorIteratorConfig& /*config*/);

  void foreach_reduced_elt(loop_subiter_t loop, bool parallelize = true);

  int ndim() const { return static_cast<int>(shape_.size()); }
  const DimVector& shape() const { return shape_; }
  int64_t numel() const;
  int ntensors() const { return static_cast<int>(operands_.size()); }
  int noutputs() const { return num_outputs_; }
  int ninputs() const { return ntensors() - noutputs(); }
  const DimVector& view_offsets() const { return view_offsets_; }

  /// number of elements in the output operand. this is the same as numel() for
  /// operations that are not reductions.
  int64_t num_output_elements() const;

  /// number of reduced dimensions in a reduction operation
  int num_reduce_dims() const;

  /// 1-dimensional iteration and no buffering or type conversion
  bool is_trivial_1d() const;
  /// Reducible to 1-dimensional and all operands are contiguous
  bool is_contiguous() const;
  bool is_dim_reduced(int dim) const;

  /// Accessors for each operand
  const StrideVector& strides(int64_t arg) const {
    return operands_[arg].stride_bytes;
  }
  void* data_ptr(int64_t arg) const;
  ScalarType dtype(int64_t arg = 0) const {
    return operands_[arg].current_dtype;
  }
  ScalarType common_dtype() const {
    TP_CHECK(
        common_dtype_ != ScalarType::Undefined,
        "Queried for invalid common dtype!");
    return common_dtype_;
  }
  std::optional<ScalarType> maybe_common_dtype() const {
    return common_dtype_ == ScalarType::Undefined
        ? std::nullopt
        : std::optional<ScalarType>(common_dtype_);
  }
  ScalarType input_dtype(int64_t arg = 0) const {
    return operands_[num_outputs_ + arg].current_dtype;
  }
  Device device(int64_t arg = 0) const {
    return operands_[arg].device.value_or(Device(DeviceType::CPU));
  }
  DeviceType device_type(int64_t arg = 0) const { return device(arg).type(); }
  int64_t element_size(int64_t arg) const {
    return static_cast<int64_t>(elementSize(dtype(arg)));
  }
  bool is_scalar(int64_t arg) const;
  bool is_cpu_scalar(int64_t arg) const;

  const Tensor& tensor(int64_t arg) const { return operands_[arg].tensor(); }
  Tensor& tensor(int64_t arg) { return operands_[arg].tensor(); }
  const Tensor& output(int64_t arg = 0) const {
    TP_CHECK(arg < num_outputs_, "output index out of bounds");
    return tensor(arg);
  }
  Tensor& output(int64_t arg = 0) {
    TP_CHECK(arg < num_outputs_, "output index out of bounds");
    return tensor(arg);
  }
  const Tensor& output_base(int64_t arg = 0) const { return output(arg); }
  Tensor& output_base(int64_t arg = 0) { return output(arg); }
  const Tensor& input(int64_t arg = 0) const {
    TP_CHECK(
        arg >= 0 && arg < ntensors() - num_outputs_,
        "input index out of bounds");
    return tensor(num_outputs_ + arg);
  }

  // Copies from temporary outputs back to the original outputs
  // NOTE: only used on CPU
  void cast_outputs();

  /// Removes an operand from this iterator
  void remove_operand(int64_t arg);
  /// Shrinks an iterated dimension
  void narrow(int dim, int64_t start, int64_t size);
  /// Narrows every dim after and including `start_dim` to size one.
  void select_all_keeping_dim(int start_dim, const DimVector& starts);
  /// Replaces the data pointer for the operand at index `arg`.
  /// The new pointer should have the same sizes, strides and dtype as the
  /// original
  void unsafe_replace_operand(int64_t arg, void* data);

  /// Splits this TensorIterator into two iterators. Together they iterate over
  /// the entire operation. Used by `with_32bit_indexing()`.
  std::unique_ptr<TensorIterator> split(int dim);

  /// Returns the dimension with the largest extent: (size[dim]-1) * stride[dim]
  int get_dim_to_split() const;

  /// true if the stride computation can use 32-bit arithmetic. Used by GPU
  /// kernels
  bool can_use_32bit_indexing() const;

  /// An "iterable" object that recursively splits this iterator into
  /// sub-iterators that can use 32-bit indexing.
  SplitUntil32Bit with_32bit_indexing() const;

  /// If the kernel should accumulate into the output. Only relevant for CUDA
  /// reductions.
  bool should_accumulate() const { return accumulate_; }

  /// Whether this iterator produces the actual output,
  /// as opposed to something that will be accumulated further. Only relevant
  /// for CUDA reductions.
  bool is_final_output() const { return final_output_; }

  bool has_contiguous_first_dim() const {
    if (ndim() == 0) {
      return true;
    }
    for (const auto i : irange(ntensors())) {
      if (strides(i)[0] != element_size(i)) {
        return false;
      }
    }
    return true;
  }

  void set_output_raw_strided(
      int64_t output_idx,
      const DimVector& sizes,
      const DimVector& strides,
      DType dtype,
      Device device);

  /// Create a strides array for a Tensor with shape of this iterator. The
  /// parameter `element_size` specifies the size of Tensor's data type in
  /// bytes (e.g. `4` for `float`)
  StrideVector compatible_stride(int64_t element_size) const;

  /// Inverts the re-ordering done by reorder_dimensions. This can only be
  /// called *before* coalesce_dimensions() is called.
  DimVector invert_perm(const DimVector& input) const;

  /// Reapply same re-ordering as it is done by reorder_dimensions. This can
  /// only be called *before* coalesce_dimensions() is called.
  DimVector apply_perm_and_mul(const DimVector& input, int mul) const;

  /// Helper functions for CPU iteration
  StrideVector get_dim_strides(int dim) const;
  StrideVector get_strides() const;
  StrideVector get_inner_strides() const { return get_dim_strides(0); }
  PtrVector get_base_ptrs() const;

  void _unsafe_set_arg_strides(const int64_t arg, const StrideVector& strides) {
    operands_[arg].stride_bytes = strides;
  }
  void _unsafe_set_arg_data(const int64_t arg, void* data) {
    operands_[arg].data = data;
  }

  const OperandInfo& operand(int arg = 0) const { return operands_[arg]; }
  OperandInfo& operand(int arg = 0) { return operands_[arg]; }

  template <typename loop1d_t>
  auto loop_2d_from_1d(const loop1d_t& loop) {
    return
        [loop, ntensor = ntensors()](
            char** base, const int64_t* strides, int64_t size0, int64_t size1) {
          PtrVector data(base, base + ntensor);
          const int64_t* outer_strides = &strides[ntensor];
          for (const auto i : irange(size1)) {
            if (i > 0) {
              for (const auto arg : irange(ntensor)) {
                data[arg] += outer_strides[arg];
              }
            }
            loop(data.data(), strides, size0);
          }
        };
  }

  void for_each(loop1d_t loop, int64_t grain_size = parallel::GRAIN_SIZE) {
    for_each(loop_2d_from_1d(loop), grain_size);
  }

  void for_each(loop2d_t loop, int64_t grain_size = parallel::GRAIN_SIZE);

  void parallel_reduce(loop2d_t loop);

  void serial_for_each(loop1d_t loop, Range range) {
    serial_for_each(loop_2d_from_1d(loop), range);
  }

  void serial_for_each(loop2d_t loop, Range range) const;

 protected:
  // Mutable reference as it moves tensors out of TensorIteratorConfig
  void populate_operands(TensorIteratorConfig& /*config*/);
  void mark_outputs();
  void mark_resize_outputs(const TensorIteratorConfig& /*config*/);
  void compute_mem_overlaps(const TensorIteratorConfig& /*config*/);
  void compute_shape(const TensorIteratorConfig& /*config*/);
  void compute_strides(const TensorIteratorConfig& /*config*/);
  void reorder_dimensions();
  void permute_dimensions(const DimVector& perm);
  void compute_types(const TensorIteratorConfig& /*config*/);
  ScalarType compute_common_dtype();
  void allocate_or_resize_outputs();
  bool fast_set_up(const TensorIteratorConfig& /*config*/);
  FastSetupType compute_fast_setup_type(const TensorIteratorConfig& /*config*/);
  void coalesce_dimensions();

  /// Records the "computation" shape of the output tensor: the shape after
  /// dimension reordering and coalescing, i.e. the shape that actually
  /// matters for implementing the kernel.
  DimVector shape_;

  /// Temporarily records the permutation computed by reorder_dimensions.
  /// This permutation maps the computation output dimension (dim) to
  /// the original true output dimension (perm_[dim]). It is used by
  /// invert_perm to undo the permutation. After coalesce_dimensions is
  /// called, the permutation is no longer valid.
  DimVector perm_;

  /// Has coalesce_dimensions() (or any moral equivalent, e.g., fast_build())
  /// been called? This is SOLELY used to check validity of perm_.
  bool has_coalesced_dimensions_ = false;

  /// Whether iteration must be fixed. This disables dimension permuting and
  /// also changes how for_each divides work among threads.
  bool enforce_linear_iteration_ = false;

  /// The index offsets into the original tensors for each dimension.
  /// This is only non-zero when you narrow() a TensorIterator (e.g.,
  /// when you make sub-TensorIterators).
  DimVector view_offsets_;

  /// The operands of the TensorIterator: both the inputs and outputs. The
  /// outputs MUST come first in the operands_ list. There is always an
  /// operand for each output of the TensorIterator, even if TensorIterator
  /// will ultimately be responsible for allocating the output; in those
  /// cases, tensor is simply undefined (and will be populated later
  /// during build()).
  std::vector<OperandInfo> operands_;

  /// Number of outputs in operands_ (the length of the outputs prefix
  /// in operands_).
  int num_outputs_ = 0;

  /// Whether or not all operands have the same shape and are 1d+. Having all
  /// the same shape affects whether or not the iterator is eligible for fast
  /// setup.
  bool all_ops_same_shape_ = false;
  /// Whether or not all operands are 0d, this affects type promotion
  bool all_ops_are_scalars_ = false;

  /// The "computation" dtype of TensorIterator, specifying what the dtype
  /// we will do the internal computation in TensorIterator. Typically,
  /// this matches the dtype of the output tensors, but not always!
  ScalarType common_dtype_ = ScalarType::Undefined;

  /// The device of the computation. Currently always CPU.
  Device common_device_ = Device(DeviceType::CPU);

  /// Set by split(), see should_accumulate() and is_final_output()
  bool accumulate_ = false;
  bool final_output_ = true;

  // From TensorIteratorConfig
  bool is_reduction_ = false;

};

struct TensorIterator final : public TensorIteratorBase {
  TensorIterator() = default;
  // Slicing is OK, TensorIterator guaranteed NOT to have any fields
  TensorIterator(const TensorIteratorBase& iter) : TensorIteratorBase(iter) {}

  /// broadcast shape, inputs promoted to the common dtype.  Inputs may be
  /// arbitrarily strided; reorder/coalesce apply.
  static TensorIterator binary_op(Tensor& out, const Tensor& a, const Tensor& b);

  static TensorIterator reduce_op(Tensor& out, const Tensor& a);
  static TensorIterator reduce_op(
      Tensor& out1,
      Tensor& out2,
      const Tensor& a);
};

class TensorIteratorConfig final {
 public:
  friend struct TensorIteratorBase;
  friend struct TensorIterator;

  TensorIteratorConfig() = default;

  TensorIteratorConfig(const TensorIteratorConfig&) = delete;
  TensorIteratorConfig& operator=(const TensorIteratorConfig&) = delete;
  TensorIteratorConfig(TensorIteratorConfig&&) = default;
  TensorIteratorConfig& operator=(TensorIteratorConfig&&) = default;
  ~TensorIteratorConfig() = default;

  /// Construction. Tensor is a shared_ptr value type, so "borrowed" and
  /// "owned" storage are identical here.
  /// Important: the outputs have to be added before the inputs.
  TensorIteratorConfig& add_output(const Tensor& output) {
    return add_owned_output(output);
  }
  TensorIteratorConfig& add_input(const Tensor& input) {
    return add_owned_input(input);
  }
  TensorIteratorConfig& add_const_input(const Tensor& input) {
    return add_owned_const_input(input);
  }

  TensorIteratorConfig& add_owned_output(const Tensor& output) {
    tensors_.push_back(output);
    ++num_outputs_;
    return *this;
  }
  TensorIteratorConfig& add_owned_input(const Tensor& input) {
    tensors_.push_back(input);
    ++num_inputs_;
    return *this;
  }
  TensorIteratorConfig& add_owned_const_input(const Tensor& input) {
    tensors_.push_back(input);
    const_tensor_indices_.push_back(tensors_.size() - 1);
    ++num_inputs_;
    return *this;
  }

  TensorIteratorConfig& add_borrowed_output(const Tensor& output) {
    return add_owned_output(output);
  }
  TensorIteratorConfig& add_borrowed_input(const Tensor& input) {
    return add_owned_input(input);
  }
  TensorIteratorConfig& add_borrowed_const_input(const Tensor& input) {
    return add_owned_const_input(input);
  }

  // Sets the check_mem_overlap_ flag, which is true by default.
  // If true, inputs are checked for partial overlap with the outputs and
  // outputs are checked for internal overlap (e.g. broadcasted views). An error
  // is raised if unacceptable overlap is detected.
  TensorIteratorConfig& set_check_mem_overlap(bool check_mem_overlap) {
    check_mem_overlap_ = check_mem_overlap;
    return *this;
  }

  // Sets the check_all_same_dtype_ flag, which is true by default
  // If true, checks that all inputs and defined outputs have the same dtype
  // Setting either of promote_inputs_to_common_dtype_
  //   or cast_common_dtype_to_outputs_ to true will set
  //   check_all_same_dtype_ to false.
  TensorIteratorConfig& check_all_same_dtype(const bool _check_all_same_dtype) {
    check_all_same_dtype_ = _check_all_same_dtype;
    return *this;
  }

  // Sets the check_all_same_device_ flag, which is true by default
  // If true, all operands must be on the same device.
  TensorIteratorConfig& check_all_same_device(
      const bool _check_all_same_device) {
    check_all_same_device_ = _check_all_same_device;
    return *this;
  }

  // Sets the enforce_safe_casting_to_output_ flag, which is false by default
  // If true, the iterator's "common dtype" must be computable and
  // canCast(common dtype, output dtype) must be true for all outputs.
  TensorIteratorConfig& enforce_safe_casting_to_output(
      const bool _enforce_safe_casting_to_output) {
    enforce_safe_casting_to_output_ = _enforce_safe_casting_to_output;
    return *this;
  }

  // Sets the enforce_linear_iteration_ flag, which is false by default.
  // If true, iteration goes in the same order as a C-contiguous tensor
  // is laid out in memory. i.e. last dimension iterates fastest.
  TensorIteratorConfig& enforce_linear_iteration(
      const bool _enforce_linear_iteration = true) {
    enforce_linear_iteration_ = _enforce_linear_iteration;
    return *this;
  }

  // Sets the promote_inputs_to_common_dtype_ flag, which is false by default
  // If true, the iterator's "common dtype" is always computed (see the
  //   [Common Dtype Computation] note) and, on the CPU, temporary copies of
  //   the inputs in the common dtype are passed as the actual inputs to
  //   the operation.
  // Setting this flag to true sets check_all_same_dtype_ to false.
  TensorIteratorConfig& promote_inputs_to_common_dtype(
      const bool _promote_inputs_to_common_dtype) {
    promote_inputs_to_common_dtype_ = _promote_inputs_to_common_dtype;
    if (_promote_inputs_to_common_dtype) {
      check_all_same_dtype_ = false;
    }
    return *this;
  }

  // Sets the promote_integer_inputs_to_float_ flag, which is false by default
  // NOTE: If set to true, the promote_inputs_to_common_dtype_ must also be
  // true. If true, if the iterator's "common dtype" is an integral type
  // (including bool) then it is changed to the default float scalar type.
  TensorIteratorConfig& promote_integer_inputs_to_float(
      const bool _promote_integer_inputs_to_float) {
    promote_integer_inputs_to_float_ = _promote_integer_inputs_to_float;
    TP_CHECK(
        !promote_integer_inputs_to_float_ || promote_inputs_to_common_dtype_,
        "promote_integer_inputs_to_float requires promote_inputs_to_common_dtype");
    return *this;
  }

  TensorIteratorConfig& is_reduction(const bool _is_reduction) {
    is_reduction_ = _is_reduction;
    return *this;
  }

  TensorIteratorConfig& allow_cpu_scalars(const bool _allow_cpu_scalars) {
    allow_cpu_scalars_ = _allow_cpu_scalars;
    return *this;
  }

  // Sets the cast_common_dtype_to_outputs_ flag, which is false by default
  // If true, the iterator's "common dtype" must be computatable and, on the
  // CPU, temporary copies of the outputs are passed as the actual output to
  // the operation. These temporaries are then copied to the original outputs
  // after the operation is performed (see cast_outputs()).
  // Setting this flag to true sets check_all_same_dtype_ to false.
  TensorIteratorConfig& cast_common_dtype_to_outputs(
      const bool _cast_common_dtype_to_outputs) {
    cast_common_dtype_to_outputs_ = _cast_common_dtype_to_outputs;
    if (_cast_common_dtype_to_outputs) {
      check_all_same_dtype_ = false;
    }
    return *this;
  }

  TensorIteratorConfig& resize_outputs(bool resize_outputs) {
    resize_outputs_ = resize_outputs;
    return *this;
  }

  // Bypass output dtype/device computation and fix the dtype/device as
  // specified here.
  TensorIteratorConfig& declare_static_dtype_and_device(
      ScalarType dtype,
      Device device) {
    static_dtype_ = dtype;
    static_device_ = device;
    return *this;
  }
  TensorIteratorConfig& declare_static_dtype(ScalarType dtype) {
    static_dtype_ = dtype;
    return *this;
  }
  TensorIteratorConfig& declare_static_device(Device device) {
    static_device_ = device;
    return *this;
  }
  TensorIteratorConfig& declare_static_shape(const DimVector& shape) {
    static_shape_ = shape;
    return *this;
  }

  TensorIterator build() {
    TensorIterator iter;
    iter.build(*this);
    return iter;
  }

 private:
  bool is_tensor_const(size_t idx) const {
    return std::find(
               const_tensor_indices_.begin(), const_tensor_indices_.end(), idx)
        != const_tensor_indices_.end();
  }

  std::vector<Tensor> tensors_;
  int num_outputs_ = 0;
  int num_inputs_ = 0;

  std::vector<size_t> const_tensor_indices_;

  std::optional<DimVector> static_shape_ = std::nullopt;
  std::optional<ScalarType> static_dtype_ = std::nullopt;
  std::optional<Device> static_device_ = std::nullopt;
  bool check_mem_overlap_ = true;
  bool allow_cpu_scalars_ = false;
  bool is_reduction_ = false;
  bool resize_outputs_ = true;
  bool check_all_same_dtype_ = true;
  bool check_all_same_device_ = true;
  bool enforce_safe_casting_to_output_ = false;
  bool enforce_linear_iteration_ = false;
  bool promote_inputs_to_common_dtype_ = false;
  bool promote_integer_inputs_to_float_ = false;
  bool cast_common_dtype_to_outputs_ = false;
};

/// A container-like struct that acts as if it contains splits of a
/// TensorIterator that can use 32-bit indexing. Taken together the splits cover
/// the original TensorIterator.
struct SplitUntil32Bit {
  struct iterator {
    iterator() = default;
    iterator(const TensorIteratorBase& iter);
    iterator(iterator&&) = default;
    iterator& operator=(iterator&&) = default;
    ~iterator() = default;

    // Guaranteed to be a TensorIterator proper!
    TensorIterator& operator*() const;
    iterator& operator++();
    bool operator==(const iterator& other) const {
      // two iterators are equal if they are the same object or they're both
      // empty
      return this == &other || (vec.empty() && other.vec.empty());
    }
    // needed for C++11 range-based for loop
    bool operator!=(const iterator& other) const { return !(*this == other); }

    /// stack of TensorIterators to be split
    std::vector<std::unique_ptr<TensorIterator>> vec;
  };

  SplitUntil32Bit(const TensorIteratorBase& iter) : iter(iter) {}

  iterator begin() const;
  iterator end() const;

 private:
  const TensorIteratorBase& iter;
};

} // namespace tensorplay
