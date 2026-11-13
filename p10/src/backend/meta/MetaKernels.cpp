// Shape-only kernels for the meta device. Every kernel computes the sizes,
// dtype and device of its result and returns an unbacked tensor: meta storage
// reports the true byte count but hands out a null data pointer, so nothing
// here may read or write elements. Data-touching composite fallbacks are
// stopped by the TensorIterator meta guard; ops with neither a kernel here
// nor a metadata-only composite registration fail with a dispatcher miss.
//
// These kernels must never re-enter the dispatcher for their own op name: a
// generated Tensor method resolves through the dispatcher, so a kernel that
// called the same-named member would recurse without bound. Allocation goes
// through TensorImpl directly for the same reason.

#include "Tensor.h"
#include "TensorImpl.h"
#include "Convolution.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Utils.h"
#include "Context.h"
#include "Generator.h"
#include "Scalar.h"
#include "DType.h"
#include "TypePromotion.h"
#include "TypeProperties.h"

#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace meta {

namespace {

// Allocation that bypasses the dispatcher: the Tensor factory methods
// redispatch to the "empty" operator, which resolves back to this backend.
Tensor empty_meta(std::vector<int64_t> size, DType dtype, const Device& device) {
    return Tensor(make_intrusive<TensorImpl>(size, dtype, device));
}

Device resolve_device(const std::optional<Device>& device) {
    return device.has_value() ? *device : globalContext().defaultDevice();
}

DType resolve_factory_dtype(const std::optional<DType>& dtype) {
    return (dtype.has_value() && *dtype != DType::Undefined)
               ? *dtype
               : globalContext().defaultDType();
}

DType resolve_like_dtype(DType dtype, const Tensor& self) {
    return dtype == DType::Undefined ? self.dtype() : dtype;
}

Device resolve_like_device(const std::optional<Device>& device, const Tensor& self) {
    return device.has_value() ? *device : self.device();
}

std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

std::vector<int64_t> broadcast2(const Tensor& a, const Tensor& b) {
    return broadcast_shapes(shape_of(a), shape_of(b));
}

void check_all_meta(const Tensor& a, const Tensor& b) {
    if (!a.device().is_meta() || !b.device().is_meta()) {
        TP_THROW(DeviceMismatchError,
                 "expected both operands on the meta device but got ",
                 a.device().toString(), " and ", b.device().toString());
    }
}

// Result dtype for add/sub-style arithmetic: promote the operands, and a
// floating alpha lifts an integer pair to at least float.
DType add_like_dtype(DType self_dtype, DType other_dtype, const Scalar& alpha) {
    DType result = promoteTypes(self_dtype, other_dtype);
    if (alpha.isFloatingPoint() && !isFloatingType(result)) {
        result = promoteTypes(result, DType::Float32);
    }
    return result;
}

DType add_like_scalar_dtype(DType self_dtype, const Scalar& other, const Scalar& alpha) {
    return add_like_dtype(self_dtype, result_type(other, self_dtype), alpha);
}

// True division always falls back to float when the promoted pair is integral.
DType div_like_dtype(DType self_dtype, DType other_dtype) {
    DType result = promoteTypes(self_dtype, other_dtype);
    if (isIntegralType(result, /*include_bool=*/true)) {
        result = DType::Float32;
    }
    return result;
}

DType div_like_scalar_dtype(DType self_dtype, const Scalar& other) {
    return div_like_dtype(self_dtype, result_type(other, self_dtype));
}

// An in-place operand must fit the target: right-aligned, every source
// dimension is either 1 or matches the destination.
void check_inplace_fits(const Tensor& self, const Tensor& other) {
    const int64_t nd = self.dim();
    const int64_t od = other.dim();
    for (int64_t i = 1; i <= od; ++i) {
        const int64_t src = other.size(od - i);
        const int64_t dst = i <= nd ? self.size(nd - i) : 1;
        if (src != 1 && src != dst) {
            TP_THROW(RuntimeError, "the shape of the operand (",
                     other.size(od - i), " at position ", od - i,
                     ") cannot be applied in place to a destination of size ",
                     dst);
        }
    }
}

// ---------------------------------------------------------------------------
// Factories. Fill-style factories are empty allocations on this device: the
// fill value lives in element data, which meta tensors do not carry.
// ---------------------------------------------------------------------------

Tensor empty_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                  std::optional<Device> device, bool /*pin_memory*/) {
    return empty_meta(size, resolve_factory_dtype(dtype), resolve_device(device));
}

Tensor zeros_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                  std::optional<Device> device, bool /*pin_memory*/) {
    return empty_meta(size, resolve_factory_dtype(dtype), resolve_device(device));
}

Tensor ones_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                 std::optional<Device> device, bool /*pin_memory*/) {
    return empty_meta(size, resolve_factory_dtype(dtype), resolve_device(device));
}

Tensor full_stub(const std::vector<int64_t>& size, const Scalar& /*fill_value*/,
                 DType dtype, std::optional<Device> device, bool /*pin_memory*/) {
    return empty_meta(size, resolve_factory_dtype(dtype), resolve_device(device));
}

Tensor eye_stub(int64_t n, int64_t m, DType dtype, std::optional<Device> device) {
    return empty_meta({n, m < 0 ? n : m}, resolve_factory_dtype(dtype),
                      resolve_device(device));
}

// Sequence length: ceil((end - start) / step), zero when the range is empty
// in the step's direction.
int64_t arange_length(const Scalar& start, const Scalar& end, const Scalar& step) {
    if (step.toDouble() == 0.0) {
        TP_THROW(RuntimeError, "step must be nonzero");
    }
    const double s = start.toDouble();
    const double e = end.toDouble();
    const double st = step.toDouble();
    double len = std::ceil((e - s) / st);
    if (len < 0) len = 0;
    return static_cast<int64_t>(len);
}

DType arange_dtype(const Scalar& start, const Scalar& end, const Scalar& step,
                   DType dtype) {
    if (dtype != DType::Undefined) return dtype;
    if (start.isFloatingPoint() || end.isFloatingPoint() || step.isFloatingPoint()) {
        return globalContext().defaultDType();
    }
    return DType::Int64;
}

Tensor arange_start_step_stub(const Scalar& start, const Scalar& end,
                              const Scalar& step, DType dtype,
                              std::optional<Device> device) {
    const int64_t len = arange_length(start, end, step);
    return empty_meta({len}, arange_dtype(start, end, step, dtype),
                     resolve_device(device));
}

Tensor arange_end_stub(const Scalar& end, DType dtype, std::optional<Device> device) {
    return arange_start_step_stub(Scalar(0), end, Scalar(1), dtype, device);
}

Tensor linspace_stub(const Scalar& /*start*/, const Scalar& /*end*/, int64_t steps,
                     DType dtype, std::optional<Device> device) {
    return empty_meta({steps < 0 ? 0 : steps}, resolve_factory_dtype(dtype),
                      resolve_device(device));
}

Tensor logspace_stub(const Scalar& /*start*/, const Scalar& /*end*/, int64_t steps,
                     double /*base*/, DType dtype, std::optional<Device> device) {
    return empty_meta({steps < 0 ? 0 : steps}, resolve_factory_dtype(dtype),
                      resolve_device(device));
}

// Random factories sample values into element storage; only the shape is
// observable on this device.
Tensor rand_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                 std::optional<Device> device) {
    return empty_meta(size, resolve_factory_dtype(dtype), resolve_device(device));
}

Tensor randn_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                  std::optional<Device> device) {
    return empty_meta(size, resolve_factory_dtype(dtype), resolve_device(device));
}

Tensor randn_generator_stub(const std::vector<int64_t>& size,
                            std::optional<Generator> generator,
                            std::optional<DType> dtype,
                            std::optional<int64_t> layout,
                            std::optional<Device> device,
                            std::optional<bool> pin_memory) {
    (void)generator;
    (void)layout;
    (void)pin_memory;
    return empty_meta(size, resolve_factory_dtype(dtype), resolve_device(device));
}

Tensor randint_stub(int64_t /*low*/, int64_t /*high*/,
                    const std::vector<int64_t>& size, DType dtype,
                    std::optional<Device> device) {
    return empty_meta(size, resolve_factory_dtype(dtype), resolve_device(device));
}

Tensor randperm_stub(int64_t n, DType dtype, std::optional<Device> device) {
    return empty_meta({n}, resolve_factory_dtype(dtype), resolve_device(device));
}

Tensor empty_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    return empty_meta(shape_of(self), resolve_like_dtype(dtype, self),
                      resolve_like_device(device, self));
}

Tensor zeros_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    return empty_like_kernel(self, dtype, device);
}

Tensor ones_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    return empty_like_kernel(self, dtype, device);
}

Tensor full_like_kernel(const Tensor& self, const Scalar& /*fill_value*/, DType dtype,
                        std::optional<Device> device) {
    return empty_like_kernel(self, dtype, device);
}

Tensor rand_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    return empty_like_kernel(self, dtype, device);
}

Tensor randn_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    return empty_like_kernel(self, dtype, device);
}

Tensor randint_like_kernel(const Tensor& self, int64_t /*low*/, int64_t /*high*/,
                           DType dtype, std::optional<Device> device) {
    return empty_like_kernel(self, dtype, device);
}

// A copy is a value of its own: the same geometry and element type, and none
// of the source's memory.  When no arrangement is asked for, the source's is
// kept, which is what a copy that shares nothing still preserves.
Tensor clone_kernel(const Tensor& self, std::optional<int64_t> /*memory_format*/) {
    Tensor result = empty_meta(shape_of(self), self.dtype(), self.device());
    const std::vector<int64_t> strides = self.strides();
    if (strides != result.strides()) {
        result = result.as_strided(shape_of(self), strides, 0);
    }
    return result;
}

// A copy between devices reads the elements it moves, and a tensor on the
// meta device has none: it models shape and dtype only.  So a copy that would
// move one is refused where it is asked for, rather than being left to a
// kernel that would read a null pointer.  A copy that stays where the value
// is changes only its element type, and that is described like any other
// result.
[[noreturn]] void refuse_meta_move(const Device& device) {
    TP_THROW(RuntimeError,
             "cannot copy a tensor on the meta device to ", device.toString(),
             ": the meta device models shape and dtype only, so it holds no "
             "elements to move");
}

Tensor describe_copy(const Tensor& self, std::optional<DType> dtype,
                     std::optional<Device> device) {
    const Device target = device.value_or(self.device());
    if (target != self.device()) {
        refuse_meta_move(target);
    }
    return empty_meta(
        shape_of(self),
        resolve_like_dtype(dtype.value_or(DType::Undefined), self), self.device());
}

Tensor to_device_kernel(const Tensor& self, Device device, DType dtype,
                        bool /*non_blocking*/, bool /*copy*/,
                        std::optional<int64_t> /*memory_format*/) {
    return describe_copy(self, dtype, device);
}

Tensor to_other_kernel(const Tensor& self, const Tensor& other,
                       bool /*non_blocking*/, bool /*copy*/,
                       std::optional<int64_t> /*memory_format*/) {
    return describe_copy(self, other.dtype(), other.device());
}

Tensor to_copy_kernel(const Tensor& self, std::optional<DType> dtype,
                      std::optional<int64_t> /*layout*/,
                      std::optional<Device> device,
                      std::optional<bool> /*pin_memory*/, bool /*non_blocking*/,
                      std::optional<int64_t> /*memory_format*/) {
    return describe_copy(self, dtype, device);
}

Tensor to_dtype_layout_kernel(const Tensor& self, std::optional<DType> dtype,
                              std::optional<int64_t> /*layout*/,
                              std::optional<Device> device,
                              std::optional<bool> /*pin_memory*/,
                              bool /*non_blocking*/, bool /*copy*/,
                              std::optional<int64_t> /*memory_format*/) {
    return describe_copy(self, dtype, device);
}

// In-place writes change element data only; the metadata is already final.
Tensor& fill_kernel(Tensor& self, const Scalar& /*value*/) {
    return self;
}

Tensor& zero_kernel(Tensor& self) {
    return self;
}

// ---------------------------------------------------------------------------
// Pointwise arithmetic.
// ---------------------------------------------------------------------------

Tensor add_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha) {
    check_all_meta(self, other);
    return empty_meta(broadcast2(self, other),
                      add_like_dtype(self.dtype(), other.dtype(), alpha),
                      self.device());
}

Tensor add_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha) {
    return empty_meta(shape_of(self), add_like_scalar_dtype(self.dtype(), other, alpha),
                      self.device());
}

Tensor& add_out_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha,
                       Tensor& out) {
    Tensor expected = add_kernel(self, other, alpha);
    if (shape_of(out) != shape_of(expected) || out.dtype() != expected.dtype()) {
        TP_THROW(RuntimeError, "add.out: expected output with shape ",
                 expected.numel(), " and dtype ", toString(expected.dtype()),
                 " but got shape ", out.numel(), " and dtype ",
                 toString(out.dtype()));
    }
    return out;
}

Tensor& add_inplace_kernel(Tensor& self, const Tensor& other, const Scalar& alpha) {
    check_inplace_fits(self, other);
    return self;
}

Tensor& add_scalar_inplace_kernel(Tensor& self, const Scalar& other, const Scalar& alpha) {
    (void)other;
    (void)alpha;
    return self;
}

Tensor sub_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha) {
    check_all_meta(self, other);
    return empty_meta(broadcast2(self, other),
                      add_like_dtype(self.dtype(), other.dtype(), alpha),
                      self.device());
}

Tensor sub_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha) {
    return empty_meta(shape_of(self), add_like_scalar_dtype(self.dtype(), other, alpha),
                      self.device());
}

Tensor& sub_inplace_kernel(Tensor& self, const Tensor& other, const Scalar& alpha) {
    (void)alpha;
    check_inplace_fits(self, other);
    return self;
}

Tensor& sub_scalar_inplace_kernel(Tensor& self, const Scalar& other, const Scalar& alpha) {
    (void)other;
    (void)alpha;
    return self;
}

Tensor mul_kernel(const Tensor& self, const Tensor& other) {
    check_all_meta(self, other);
    return empty_meta(broadcast2(self, other),
                      promoteTypes(self.dtype(), other.dtype()), self.device());
}

Tensor mul_scalar_kernel(const Tensor& self, const Scalar& other) {
    return empty_meta(shape_of(self), result_type(other, self.dtype()),
                      self.device());
}

Tensor& mul_inplace_kernel(Tensor& self, const Tensor& other) {
    check_inplace_fits(self, other);
    return self;
}

Tensor& mul_scalar_inplace_kernel(Tensor& self, const Scalar& other) {
    (void)other;
    return self;
}

Tensor div_kernel(const Tensor& self, const Tensor& other) {
    check_all_meta(self, other);
    return empty_meta(broadcast2(self, other),
                      div_like_dtype(self.dtype(), other.dtype()), self.device());
}

Tensor div_scalar_kernel(const Tensor& self, const Scalar& other) {
    return empty_meta(shape_of(self), div_like_scalar_dtype(self.dtype(), other),
                      self.device());
}

Tensor& div_inplace_kernel(Tensor& self, const Tensor& other) {
    check_inplace_fits(self, other);
    return self;
}

Tensor& div_scalar_inplace_kernel(Tensor& self, const Scalar& other) {
    (void)other;
    return self;
}

// ---------------------------------------------------------------------------
// Pointwise unary. abs of a complex input produces the matching real value
// type; every other unary keeps dtype and shape.
// ---------------------------------------------------------------------------

Tensor abs_kernel(const Tensor& self) {
    DType out = isComplexType(self.dtype()) ? toRealValueType(self.dtype())
                                            : self.dtype();
    return empty_meta(shape_of(self), out, self.device());
}

Tensor neg_kernel(const Tensor& self) {
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor& neg_inplace_kernel(Tensor& self) {
    return self;
}

Tensor exp_kernel(const Tensor& self) {
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor log_kernel(const Tensor& self) {
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor sqrt_kernel(const Tensor& self) {
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor clamp_kernel(const Tensor& self, const std::optional<Scalar>& /*min*/,
                    const std::optional<Scalar>& /*max*/) {
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

// ---------------------------------------------------------------------------
// Comparisons and selection.
// ---------------------------------------------------------------------------

#define META_COMPARISON_KERNEL(NAME)                                          \
    Tensor NAME##_tensor_kernel(const Tensor& self, const Tensor& other) {     \
        check_all_meta(self, other);                                          \
        return empty_meta(broadcast2(self, other), DType::Bool, self.device());\
    }                                                                         \
    Tensor NAME##_scalar_kernel(const Tensor& self, const Scalar& /*other*/) {\
        return empty_meta(shape_of(self), DType::Bool, self.device());         \
    }

META_COMPARISON_KERNEL(eq)
META_COMPARISON_KERNEL(ne)
META_COMPARISON_KERNEL(lt)
META_COMPARISON_KERNEL(le)
META_COMPARISON_KERNEL(gt)
META_COMPARISON_KERNEL(ge)
#undef META_COMPARISON_KERNEL

Tensor where_cpu(const Tensor& condition, const Tensor& self, const Tensor& other) {
    const std::vector<int64_t> out =
        broadcast_shapes(broadcast2(condition, self), shape_of(other));
    return empty_meta(out, promoteTypes(self.dtype(), other.dtype()),
                      self.device());
}

Tensor where_scalar_self_cpu(const Tensor& condition, const Scalar& self,
                             const Tensor& other) {
    const std::vector<int64_t> out = broadcast2(condition, other);
    return empty_meta(out, result_type(self, other.dtype()), other.device());
}

Tensor where_scalar_other_cpu(const Tensor& condition, const Tensor& self,
                              const Scalar& other) {
    const std::vector<int64_t> out = broadcast2(condition, self);
    return empty_meta(out, result_type(other, self.dtype()), self.device());
}

// Two scalars select the selection dtype directly: complex wins, then
// float (widening a double to keep its precision), then Int64.
DType where_scalar_dtype(const Scalar& self, const Scalar& other) {
    if (self.isComplex() || other.isComplex()) {
        return promoteTypes(self.dtype(), other.dtype());
    }
    if (self.isFloatingPoint() || other.isFloatingPoint()) {
        return self.dtype() == DType::Float64 || other.dtype() == DType::Float64
                   ? DType::Float64
                   : DType::Float32;
    }
    return DType::Int64;
}

Tensor where_scalar_scalar_cpu(const Tensor& condition, const Scalar& self,
                               const Scalar& other) {
    return empty_meta(shape_of(condition), where_scalar_dtype(self, other),
                      condition.device());
}

Tensor maximum_cpu(const Tensor& self, const Tensor& other) {
    check_all_meta(self, other);
    return empty_meta(broadcast2(self, other),
                      promoteTypes(self.dtype(), other.dtype()), self.device());
}

Tensor minimum_cpu(const Tensor& self, const Tensor& other) {
    check_all_meta(self, other);
    return empty_meta(broadcast2(self, other),
                      promoteTypes(self.dtype(), other.dtype()), self.device());
}

// ---------------------------------------------------------------------------
// Reductions.
// ---------------------------------------------------------------------------

std::vector<int64_t> reduce_out_shape(const Tensor& self,
                                      const std::vector<int64_t>& dims,
                                      bool keepdim) {
    const int64_t nd = self.dim();
    std::vector<bool> reduced(static_cast<size_t>(nd), false);
    // An empty list names no dimension, and a reduction that names no
    // dimension is a reduction over all of them; a rank-0 operand has its one
    // virtual dimension reduced.
    if (dims.empty()) {
        if (nd == 0) {
            reduced.push_back(true);
        } else {
            for (int64_t i = 0; i < nd; ++i) {
                reduced[static_cast<size_t>(i)] = true;
            }
        }
    }
    for (int64_t d : dims) {
        int64_t dim = d;
        if (nd == 0) {
            // A 0-d tensor accepts -1 and 0 as its single (virtual) dimension.
            if (dim != 0 && dim != -1) {
                TP_THROW(IndexError, "reduction dimension ", dim,
                         " out of range for a rank-", nd, " tensor");
            }
            reduced.push_back(true);
            continue;
        }
        if (dim < 0) dim += nd;
        if (dim < 0 || dim >= nd) {
            TP_THROW(IndexError, "reduction dimension ", dim,
                     " out of range for a rank-", nd, " tensor");
        }
        reduced[static_cast<size_t>(dim)] = true;
    }
    std::vector<int64_t> out;
    out.reserve(reduced.size());
    for (size_t i = 0; i < reduced.size(); ++i) {
        if (!reduced[i]) {
            out.push_back(nd == 0 ? 0 : self.size(static_cast<int64_t>(i)));
        } else if (keepdim) {
            out.push_back(1);
        }
    }
    return out;
}

// Reductions widen narrow integers and booleans to Int64 unless a dtype is
// requested explicitly.
DType sum_like_dtype(const Tensor& self, DType dtype) {
    if (dtype != DType::Undefined) return dtype;
    if (isIntegralType(self.dtype(), /*include_bool=*/true)) return DType::Int64;
    return self.dtype();
}

// Means keep the input dtype for float and complex inputs and fall back to
// float for everything else.
DType mean_like_dtype(const Tensor& self, DType dtype) {
    if (dtype != DType::Undefined) return dtype;
    return isFloatingOrComplexType(self.dtype()) ? self.dtype() : DType::Float32;
}

Tensor sum_kernel(const Tensor& self, DType dtype) {
    return empty_meta({}, sum_like_dtype(self, dtype), self.device());
}

Tensor sum_dim_kernel(const Tensor& self, const std::vector<int64_t>& dims,
                      bool keepdim, DType dtype) {
    return empty_meta(reduce_out_shape(self, dims, keepdim),
                      sum_like_dtype(self, dtype), self.device());
}

Tensor mean_kernel(const Tensor& self, DType dtype) {
    return empty_meta({}, mean_like_dtype(self, dtype), self.device());
}

Tensor mean_dim_kernel(const Tensor& self, const std::vector<int64_t>& dims,
                       bool keepdim, DType dtype) {
    return empty_meta(reduce_out_shape(self, dims, keepdim),
                      mean_like_dtype(self, dtype), self.device());
}

Tensor prod_kernel(const Tensor& self, DType dtype) {
    return empty_meta({}, sum_like_dtype(self, dtype), self.device());
}

Tensor prod_dim_kernel(const Tensor& self, const std::vector<int64_t>& dims,
                       bool keepdim, DType dtype) {
    return empty_meta(reduce_out_shape(self, dims, keepdim),
                      sum_like_dtype(self, dtype), self.device());
}

// Index reductions return Int64 coordinates; a missing dim reduces
// everything to a scalar.
Tensor arg_kernel(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    if (!dim.has_value()) {
        return empty_meta({}, DType::Int64, self.device());
    }
    return empty_meta(reduce_out_shape(self, {*dim}, keepdim), DType::Int64,
                      self.device());
}

Tensor amax_cpu(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    return empty_meta(reduce_out_shape(self, dim, keepdim), self.dtype(),
                      self.device());
}

Tensor amin_cpu(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    return empty_meta(reduce_out_shape(self, dim, keepdim), self.dtype(),
                      self.device());
}

std::tuple<Tensor, Tensor> aminmax_cpu(const Tensor& self,
                                       const std::vector<int64_t>& dim,
                                       bool keepdim) {
    const auto out_shape = reduce_out_shape(self, dim, keepdim);
    return {empty_meta(out_shape, self.dtype(), self.device()),
            empty_meta(out_shape, self.dtype(), self.device())};
}

// ---------------------------------------------------------------------------
// Matrix products.
// ---------------------------------------------------------------------------

Tensor mm_kernel(const Tensor& self, const Tensor& mat2) {
    if (self.dim() != 2) TP_THROW(RuntimeError, "self must be a matrix");
    if (mat2.dim() != 2) TP_THROW(RuntimeError, "mat2 must be a matrix");
    if (self.size(1) != mat2.size(0)) {
        TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (",
                 self.size(0), "x", self.size(1), " and ", mat2.size(0), "x",
                 mat2.size(1), ")");
    }
    return empty_meta({self.size(0), mat2.size(1)},
                      promoteTypes(self.dtype(), mat2.dtype()), self.device());
}

// General N-D product shapes: vector operands fold to rows/columns, batch
// dimensions broadcast, and the matrix part follows mm.
Tensor matmul_kernel(const Tensor& self, const Tensor& other) {
    if (self.dim() < 1 || other.dim() < 1) {
        TP_THROW(RuntimeError, "matmul(): input operands must be at least 1D");
    }
    check_all_meta(self, other);
    if (self.dtype() != other.dtype()) {
        TP_THROW(RuntimeError, "expected m1 and m2 to have the same dtype, but got: ",
                 self.dtype(), " != ", other.dtype());
    }
    const std::vector<int64_t> a = shape_of(self);
    const std::vector<int64_t> b = shape_of(other);
    const int64_t dim1 = self.dim();
    const int64_t dim2 = other.dim();

    if (dim1 == 1 && dim2 == 1) {
        if (a[0] != b[0]) {
            TP_THROW(RuntimeError, "inconsistent tensor size, expected tensor [",
                     a[0], "] and src [", b[0],
                     "] to have the same number of elements");
        }
        return empty_meta({}, self.dtype(), self.device());
    }
    if (dim1 == 1) {
        const int64_t k = a[0];
        const int64_t other_k = b[b.size() - 2];
        if (k != other_k) {
            TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (1x", k,
                     " and ", other_k, "x", b.back(), ")");
        }
        std::vector<int64_t> out(b.begin(), b.end() - 2);
        out.push_back(b.back());
        return empty_meta(out, self.dtype(), self.device());
    }
    if (dim2 == 1) {
        const int64_t k = a.back();
        if (k != b[0]) {
            int64_t folded_m = a[a.size() - 2];
            for (size_t i = 0; i + 2 < a.size(); ++i) folded_m *= a[i];
            TP_THROW(RuntimeError, "size mismatch, got input (", folded_m,
                     "), mat (", folded_m, "x", k, "), vec (", b[0], ")");
        }
        std::vector<int64_t> out(a.begin(), a.end() - 2);
        out.push_back(a[a.size() - 2]);
        return empty_meta(out, self.dtype(), self.device());
    }
    const int64_t k = a.back();
    const int64_t other_k = b[b.size() - 2];
    if (k != other_k) {
        if (dim2 == 2) {
            int64_t folded_m = a[a.size() - 2];
            for (size_t i = 0; i + 2 < a.size(); ++i) folded_m *= a[i];
            TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (",
                     folded_m, "x", k, " and ", other_k, "x", b.back(), ")");
        }
        std::vector<int64_t> self_batch(a.begin(), a.end() - 2);
        std::vector<int64_t> other_batch(b.begin(), b.end() - 2);
        const std::vector<int64_t> batch = broadcast_shapes(self_batch, other_batch);
        int64_t prod_batch = 1;
        for (const int64_t s : batch) prod_batch *= s;
        TP_THROW(RuntimeError,
                 "Expected size for first two dimensions of batch2 tensor to be: [",
                 prod_batch, ", ", k, "] but got: [", prod_batch, ", ", other_k, "].");
    }
    std::vector<int64_t> out = broadcast_shapes(
        std::vector<int64_t>(a.begin(), a.end() - 2),
        std::vector<int64_t>(b.begin(), b.end() - 2));
    out.push_back(a[a.size() - 2]);
    out.push_back(b.back());
    return empty_meta(out, self.dtype(), self.device());
}

// Reshape only moves metadata; a -1 slot infers from the element count.
Tensor reshape_kernel(const Tensor& self, const std::vector<int64_t>& shape) {
    int64_t known = 1;
    int64_t infer = -1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == -1) {
            if (infer >= 0) {
                TP_THROW(RuntimeError, "only one dimension can be inferred");
            }
            infer = static_cast<int64_t>(i);
        } else {
            known *= shape[i];
        }
    }
    std::vector<int64_t> out = shape;
    if (infer >= 0) {
        if (known == 0 || self.numel() % known != 0) {
            TP_THROW(RuntimeError, "shape '", self.numel(),
                     "' is invalid for input of size ", self.numel());
        }
        out[static_cast<size_t>(infer)] = self.numel() / known;
    }
    int64_t total = 1;
    for (const int64_t d : out) total *= d;
    if (total != self.numel()) {
        TP_THROW(RuntimeError, "shape '", total,
                 "' is invalid for input of size ", self.numel());
    }
    return empty_meta(out, self.dtype(), self.device());
}

Tensor bmm_kernel(const Tensor& self, const Tensor& batch2) {
    if (self.dim() != 3) TP_THROW(RuntimeError, "self must be a 3D tensor");
    if (batch2.dim() != 3) TP_THROW(RuntimeError, "batch2 must be a 3D tensor");
    if (self.size(0) != batch2.size(0)) {
        TP_THROW(RuntimeError, "expected equal batch sizes but got ", self.size(0),
                 " and ", batch2.size(0));
    }
    if (self.size(2) != batch2.size(1)) {
        TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (",
                 self.size(1), "x", self.size(2), " and ", batch2.size(1), "x",
                 batch2.size(2), ")");
    }
    return empty_meta({self.size(0), self.size(1), batch2.size(2)},
                      promoteTypes(self.dtype(), batch2.dtype()), self.device());
}

// addmm broadcasts the bias input against the product of the two matrices.
Tensor addmm_kernel(const Tensor& input, const Tensor& mat1, const Tensor& mat2,
                    const Scalar& /*beta*/, const Scalar& alpha) {
    Tensor product = mm_kernel(mat1, mat2);
    const std::vector<int64_t> out = broadcast_shapes(shape_of(input),
                                                      shape_of(product));
    return empty_meta(out,
                      add_like_dtype(input.dtype(),
                                     promoteTypes(mat1.dtype(), mat2.dtype()), alpha),
                      product.device());
}

// ---------------------------------------------------------------------------
// Concatenation.
// ---------------------------------------------------------------------------

Tensor cat_kernel(const std::vector<Tensor>& tensors, int64_t dim) {
    if (tensors.empty()) {
        TP_THROW(ValueError, "cat(): expected a non-empty list of Tensors");
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].dim() == 0) {
            TP_THROW(RuntimeError, "zero-dimensional tensor (at position ", i,
                     ") cannot be concatenated");
        }
        if (tensors[i].device() != tensors[0].device()) {
            TP_THROW(DeviceMismatchError,
                     "cat(): all tensors must be on the same device");
        }
    }
    const int64_t nd = tensors[0].dim();
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd) {
        TP_THROW(IndexError, "cat(): dimension ", dim, " out of range");
    }

    DType out_dtype = tensors[0].dtype();
    int64_t size_at_dim = 0;
    for (const auto& t : tensors) {
        if (t.numel() == 0) continue;
        if (t.dim() != nd) {
            TP_THROW(RuntimeError, "cat(): tensors must have the same number of dims");
        }
        for (int64_t d = 0; d < nd; ++d) {
            if (d != dim && t.size(d) != tensors[0].size(d)) {
                TP_THROW(RuntimeError, "cat(): sizes of tensors must match except in ",
                         "dimension ", dim);
            }
        }
        size_at_dim += t.size(dim);
        out_dtype = promoteTypes(out_dtype, t.dtype());
    }

    std::vector<int64_t> out_shape = shape_of(tensors[0]);
    out_shape[static_cast<size_t>(dim)] = size_at_dim;
    return empty_meta(out_shape, out_dtype, tensors[0].device());
}

// ---------------------------------------------------------------------------
// View kernels. These ops are backend-dispatched, so the meta device needs
// its own registrations; they reuse the storage-sharing as_strided path,
// which is metadata-only and therefore device-independent.
// ---------------------------------------------------------------------------

Tensor transpose_kernel(const Tensor& self, int64_t dim0, int64_t dim1) {
    const int64_t ndim = self.dim();
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        TP_THROW(IndexError, "transpose: dimension out of range for a rank-",
                 ndim, " tensor");
    }
    std::vector<int64_t> sizes = shape_of(self);
    const std::vector<int64_t> strides = self.strides();
    std::vector<int64_t> new_strides(strides);
    std::swap(sizes[static_cast<size_t>(dim0)], sizes[static_cast<size_t>(dim1)]);
    std::swap(new_strides[static_cast<size_t>(dim0)], new_strides[static_cast<size_t>(dim1)]);
    return self.as_strided(sizes, new_strides,
                           static_cast<int64_t>(self.storage_offset()));
}

Tensor slice_kernel(const Tensor& self, int64_t dim,
                    std::optional<int64_t> start, std::optional<int64_t> end,
                    int64_t step) {
    return self.slice(dim, start.value_or(0),
                      end.value_or(std::numeric_limits<int64_t>::max()), step);
}

// ---------------------------------------------------------------------------
// Convolution. The result carries the weight's output channels and, per
// spatial dimension, the extent the direct-convolution formula gives for the
// padded input and the dilated kernel; the geometry is validated by the same
// shared rules the computing backends use, so a call they would reject is
// rejected here too.
// ---------------------------------------------------------------------------

Tensor conv2d_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     const std::vector<int64_t>& stride,
                     const std::vector<int64_t>& padding,
                     const std::vector<int64_t>& dilation, int64_t groups) {
    if (input.dim() != 4 || weight.dim() != 4) {
        TP_THROW(RuntimeError, "conv2d: Expected 4D input and weight");
    }
    convolution::check_conv_shapes(input, weight, bias, groups, /*transposed=*/false);
    convolution::check_conv_geometry(input, weight, stride, padding, dilation, "conv2d");
    const std::vector<int64_t> spatial =
        convolution::conv_output_spatial_sizes(input, weight, stride, padding, dilation);
    std::vector<int64_t> out{input.size(0), weight.size(0)};
    out.insert(out.end(), spatial.begin(), spatial.end());
    return empty_meta(out, input.dtype(), input.device());
}

// ---------------------------------------------------------------------------
// Normalization. Group norm normalizes per (sample, group) over the channels
// and the spatial extents together, so its result has the input's geometry
// and its per-group statistics hold one value per group.
// ---------------------------------------------------------------------------

// The statistics of a reduced-precision input are accumulated in single
// precision and come back as such; every other input keeps its own type.
DType group_norm_stat_dtype(const Tensor& input) {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        return DType::Float32;
    }
    return input.dtype();
}

Tensor group_norm_kernel(const Tensor& input, int64_t num_groups,
                         const std::optional<Tensor>& weight,
                         const std::optional<Tensor>& bias, double /*eps*/) {
    if (input.dim() < 2) {
        TP_THROW(RuntimeError, "group_norm requires at least 2 dims");
    }
    const int64_t channels = input.size(1);
    if (channels <= 0 || num_groups <= 0 || channels % num_groups != 0) {
        TP_THROW(RuntimeError,
                 "group_norm: num_channels must be divisible by num_groups");
    }
    (void)weight;
    (void)bias;
    return empty_meta(shape_of(input), input.dtype(), input.device());
}

std::tuple<Tensor, Tensor, Tensor> native_group_norm_kernel(
    const Tensor& input, const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias, int64_t N, int64_t C, int64_t HxW,
    int64_t group, double /*eps*/) {
    if (input.dim() < 2) {
        TP_THROW(RuntimeError, "native_group_norm requires at least 2 dims");
    }
    int64_t spatial = 1;
    for (int64_t dim = 2; dim < input.dim(); ++dim) {
        spatial *= input.size(dim);
    }
    if (N != input.size(0) || C != input.size(1) || HxW != spatial) {
        TP_THROW(RuntimeError, "native_group_norm dimensions do not match input");
    }
    if (C <= 0 || spatial <= 0 || group <= 0 || C % group != 0) {
        TP_THROW(RuntimeError, "native_group_norm has invalid group dimensions");
    }
    // One scale and one shift per channel, when either is given.
    const auto check_parameter = [C](const std::optional<Tensor>& parameter,
                                     const char* name) {
        if (!parameter.has_value() || !parameter->defined()) return;
        if (parameter->dim() != 1 || parameter->numel() != C) {
            TP_THROW(RuntimeError, std::string("native_group_norm ") + name +
                                     " must have one value per channel");
        }
    };
    check_parameter(weight, "weight");
    check_parameter(bias, "bias");
    const DType stat = group_norm_stat_dtype(input);
    return {empty_meta(shape_of(input), input.dtype(), input.device()),
            empty_meta({N, group}, stat, input.device()),
            empty_meta({N, group}, stat, input.device())};
}

// ---------------------------------------------------------------------------
// Activation and root pointwise operations. Every one of them maps an
// element to an element, so the result repeats the operand's geometry and
// type.
// ---------------------------------------------------------------------------

#define META_UNARY_KERNEL(NAME)                                               \
    Tensor NAME##_kernel(const Tensor& self) {                                \
        return empty_meta(shape_of(self), self.dtype(), self.device());       \
    }

META_UNARY_KERNEL(silu)
META_UNARY_KERNEL(rsqrt)
META_UNARY_KERNEL(cos)
META_UNARY_KERNEL(sin)
#undef META_UNARY_KERNEL

// ---------------------------------------------------------------------------
// Dropout. Dropping elements changes values, not the geometry, and the mask
// that says which were kept has the same geometry as a boolean tensor.
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor> native_dropout_kernel(const Tensor& input, double p,
                                                 std::optional<bool> /*train*/) {
    if (p < 0 || p > 1) {
        TP_THROW(ValueError, "native_dropout: p must be in [0, 1]");
    }
    return {empty_meta(shape_of(input), input.dtype(), input.device()),
            empty_meta(shape_of(input), DType::Bool, input.device())};
}

// ---------------------------------------------------------------------------
// Moments. The variance and the mean reduce over the same dimensions, so they
// share a shape and a type.
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor> var_mean_kernel(const Tensor& self,
                                           const std::vector<int64_t>& dims,
                                           bool /*unbiased*/, bool keepdim) {
    const std::vector<int64_t> out = reduce_out_shape(self, dims, keepdim);
    const DType dtype = mean_like_dtype(self, DType::Undefined);
    return {empty_meta(out, dtype, self.device()), empty_meta(out, dtype, self.device())};
}

// ---------------------------------------------------------------------------
// Attention. The result keeps the query's leading extents and takes its last
// one from the value; the attention weights themselves are summed out, so the
// log-sum-exp term drops that last extent and is reported in single precision.
// ---------------------------------------------------------------------------

std::vector<int64_t> attention_out_shape(const Tensor& query, const Tensor& value) {
    if (query.dim() < 1 || value.dim() < 1) {
        TP_THROW(ValueError, "scaled_dot_product_attention: query and value must "
                             "have at least one dimension");
    }
    std::vector<int64_t> out = shape_of(query);
    out.back() = value.size(-1);
    return out;
}

// What the attention the call describes requires of its operands: one head
// dimension read from both sides, a key and a value that describe the same
// weights, and head counts that differ only by whole groups when grouped
// attention is asked for.
void check_attention_shapes(const Tensor& query, const Tensor& key,
                            const Tensor& value, bool enable_gqa) {
    if (key.dim() < 1 || key.size(-1) != query.size(-1)) {
        TP_THROW(ValueError, "scaled_dot_product_attention: query and key head "
                             "dims must match");
    }
    if (shape_of(key) != shape_of(value)) {
        TP_THROW(ValueError, "scaled_dot_product_attention: key and value shapes "
                             "must match");
    }
    if (enable_gqa && query.dim() >= 3) {
        const int64_t heads = query.size(-3);
        const int64_t key_heads = key.size(-3);
        if (heads != key_heads && (key_heads == 0 || heads % key_heads != 0)) {
            TP_THROW(ValueError,
                     "scaled_dot_product_attention: enable_gqa requires the query "
                     "head count to be divisible by the key/value head count");
        }
    }
}

Tensor sdpa_kernel(const Tensor& query, const Tensor& key, const Tensor& value,
                   const std::optional<Tensor>& /*attn_mask*/, double /*dropout_p*/,
                   bool /*is_causal*/, std::optional<double> /*scale*/,
                   bool enable_gqa) {
    check_attention_shapes(query, key, value, enable_gqa);
    return empty_meta(attention_out_shape(query, value), query.dtype(),
                      query.device());
}

std::tuple<Tensor, Tensor> sdpa_with_lse_kernel(const Tensor& query,
                                                const Tensor& key,
                                                const Tensor& value,
                                                bool /*is_causal*/,
                                                int64_t /*impl*/) {
    check_attention_shapes(query, key, value, /*enable_gqa=*/false);
    std::vector<int64_t> out = attention_out_shape(query, value);
    std::vector<int64_t> lse(out.begin(), out.end() - 1);
    return {empty_meta(out, query.dtype(), query.device()),
            empty_meta(lse, DType::Float32, query.device())};
}

// ---------------------------------------------------------------------------
// Shapes that move extents around. These are views of their operand, so they
// go through the storage-sharing path and only the sizes and strides are
// computed here.
// ---------------------------------------------------------------------------

Tensor squeeze_kernel(const Tensor& self) {
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    for (int64_t i = 0; i < self.dim(); ++i) {
        if (self.size(i) == 1) continue;
        sizes.push_back(self.size(i));
        strides.push_back(self.stride(i));
    }
    return self.as_strided(sizes, strides,
                           static_cast<int64_t>(self.storage_offset()));
}

Tensor squeeze_dim_kernel(const Tensor& self, int64_t dim) {
    const int64_t ndim = self.dim();
    if (ndim == 0) {
        if (dim != 0 && dim != -1) {
            TP_THROW(IndexError, "Dimension out of range (expected to be in range "
                                 "of [-1, 0], but got ", dim, ")");
        }
        return self.as_strided(shape_of(self), self.strides(),
                               static_cast<int64_t>(self.storage_offset()));
    }
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 -ndim, ", ", ndim - 1, "], but got ", dim, ")");
    }
    if (self.size(dim) != 1) {
        return self.as_strided(shape_of(self), self.strides(),
                               static_cast<int64_t>(self.storage_offset()));
    }
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) continue;
        sizes.push_back(self.size(i));
        strides.push_back(self.stride(i));
    }
    return self.as_strided(sizes, strides,
                           static_cast<int64_t>(self.storage_offset()));
}

Tensor squeeze_dims_kernel(const Tensor& self, const std::vector<int64_t>& dims) {
    const int64_t ndim = self.dim();
    std::vector<bool> drop(static_cast<size_t>(ndim), false);
    std::vector<bool> seen(static_cast<size_t>(ndim > 0 ? ndim : 1), false);
    for (int64_t d : dims) {
        int64_t wrapped = d < 0 ? d + ndim : d;
        if (ndim <= 0 || wrapped < 0 || wrapped >= ndim) {
            TP_THROW(IndexError, "Dimension out of range (expected to be in range "
                                 "of [", -ndim, ", ", ndim - 1, "], but got ", d, ")");
        }
        if (seen[static_cast<size_t>(wrapped)]) {
            TP_THROW(RuntimeError, "dim ", wrapped,
                     " appears multiple times in the list of dims");
        }
        seen[static_cast<size_t>(wrapped)] = true;
        drop[static_cast<size_t>(wrapped)] = true;
    }
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    for (int64_t i = 0; i < ndim; ++i) {
        if (drop[static_cast<size_t>(i)] && self.size(i) == 1) continue;
        sizes.push_back(self.size(i));
        strides.push_back(self.stride(i));
    }
    return self.as_strided(sizes, strides,
                           static_cast<int64_t>(self.storage_offset()));
}

Tensor unsqueeze_kernel(const Tensor& self, int64_t dim) {
    const int64_t ndim = self.dim();
    if (dim < 0) dim += ndim + 1;
    if (dim < 0 || dim > ndim) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 -(ndim + 1), ", ", ndim, "], but got ", dim, ")");
    }
    std::vector<int64_t> sizes = shape_of(self);
    std::vector<int64_t> strides = self.strides();
    // The inserted axis spans nothing, so its stride is whatever the axis
    // after it is worth in memory; appending one has nothing after it.
    const int64_t stride = dim == ndim ? 1 : self.size(dim) * self.stride(dim);
    sizes.insert(sizes.begin() + dim, 1);
    strides.insert(strides.begin() + dim, stride);
    return self.as_strided(sizes, strides,
                           static_cast<int64_t>(self.storage_offset()));
}

std::vector<Tensor> chunk_kernel(const Tensor& self, int64_t chunks, int64_t dim) {
    if (self.dim() == 0) {
        TP_THROW(RuntimeError, "chunk expects at least a 1-dimensional tensor");
    }
    if (chunks <= 0) {
        TP_THROW(RuntimeError, "chunk expects `chunks` to be greater than 0, got: ",
                 chunks);
    }
    const int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(IndexError, "chunk(): dimension out of range");
    }
    const int64_t size = self.size(dim);
    const int64_t split = size / chunks + (size % chunks != 0 ? 1 : 0);
    std::vector<Tensor> out;
    if (split == 0) {
        // Splitting an empty dimension yields the requested number of empty
        // parts rather than none.
        out.reserve(static_cast<size_t>(chunks));
        for (int64_t i = 0; i < chunks; ++i) {
            out.push_back(self.slice(dim, 0, 0));
        }
        return out;
    }
    for (int64_t start = 0; start < size; start += split) {
        const int64_t end = start + split < size ? start + split : size;
        out.push_back(self.slice(dim, start, end));
    }
    return out;
}

// Padding extends the trailing dimensions; the pad vector lists the two sides
// of each of them, the last dimension first.  Negative sides narrow instead.
Tensor constant_pad_nd_kernel(const Tensor& self, const std::vector<int64_t>& pad,
                              const Scalar& /*value*/) {
    if (pad.size() % 2 != 0) {
        TP_THROW(ValueError, "Length of pad must be even but instead it equals ",
                 pad.size());
    }
    const int64_t rank = self.dim();
    const int64_t padded = static_cast<int64_t>(pad.size()) / 2;
    if (rank < padded) {
        TP_THROW(ValueError,
                 "Length of pad should be no more than twice the number of "
                 "dimensions of the input. Pad length is ", pad.size(),
                 " while the input has ", rank, " dimensions.");
    }
    std::vector<int64_t> out = shape_of(self);
    for (int64_t k = 0; k < padded; ++k) {
        const int64_t dim = rank - padded + k;
        const size_t pair = static_cast<size_t>(padded - 1 - k);
        const int64_t total = pad[2 * pair] + pad[2 * pair + 1];
        const int64_t extent = out[static_cast<size_t>(dim)] + total;
        if (extent < 0) {
            TP_THROW(RuntimeError,
                     "The input size ", out[static_cast<size_t>(dim)],
                     " plus negative padding ", total,
                     " resulted in a negative output size, which is invalid. "
                     "Check dimension ", dim, " of your input.");
        }
        out[static_cast<size_t>(dim)] = extent;
    }
    return empty_meta(out, self.dtype(), self.device());
}

// Resampling rearranges existing elements; only the requested output extent
// is observable, and the batch and channel extents pass through.
Tensor upsample_nearest2d_kernel(const Tensor& self,
                                 const std::vector<int64_t>& output_size,
                                 std::optional<double> /*scales_h*/,
                                 std::optional<double> /*scales_w*/) {
    if (self.dim() != 4) {
        TP_THROW(RuntimeError, "upsample_nearest2d: expected a 4-D input, got ",
                 self.dim(), "-D");
    }
    if (output_size.size() != 2) {
        TP_THROW(ValueError, "upsample_nearest2d: output_size must have 2 elements");
    }
    for (const int64_t size : output_size) {
        if (size <= 0) {
            TP_THROW(ValueError,
                     "upsample_nearest2d: output size must be positive");
        }
    }
    return empty_meta({self.size(0), self.size(1), output_size[0], output_size[1]},
                      self.dtype(), self.device());
}

// A tensor carries a quantizer or it does not; nothing about the values
// decides it, so the answer is available without reading any.
bool is_quantized_kernel(const Tensor& self) {
    return self.defined() && self.unsafeGetTensorImpl() != nullptr &&
           self.unsafeGetTensorImpl()->has_quantizer();
}

// ---------------------------------------------------------------------------
// Copy. A meta destination accepts any source and stays metadata-only; a
// real destination cannot be filled from a tensor that owns no data.
// ---------------------------------------------------------------------------

Tensor& copy_kernel(Tensor& self, const Tensor& src, bool /*non_blocking*/) {
    if (!self.device().is_meta()) {
        TP_THROW(NotImplementedError,
                 "cannot copy out of a meta tensor into ", self.device().toString(),
                 ": it holds no data");
    }
    if (!src.device().is_meta() && self.numel() != 0 &&
        shape_of(self) != shape_of(src)) {
        TP_THROW(RuntimeError, "copy_(): the source shape (", src.numel(),
                 " elements) cannot be copied into a destination of size ",
                 self.numel());
    }
    return self;
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaKernels) {
    // Factories.
    m.impl("empty", empty_stub);
    m.impl("zeros", zeros_stub);
    m.impl("ones", ones_stub);
    m.impl("full", full_stub);
    m.impl("arange", arange_start_step_stub);
    m.impl("arange.end", arange_end_stub);
    m.impl("eye", eye_stub);
    m.impl("linspace", linspace_stub);
    m.impl("logspace", logspace_stub);
    m.impl("rand", rand_stub);
    m.impl("randn", randn_stub);
    m.impl("randn.generator", randn_generator_stub);
    m.impl("randint", randint_stub);
    m.impl("randperm", randperm_stub);
    m.impl("empty_like", empty_like_kernel);
    m.impl("zeros_like", zeros_like_kernel);
    m.impl("ones_like", ones_like_kernel);
    m.impl("full_like", full_like_kernel);
    m.impl("rand_like", rand_like_kernel);
    m.impl("randn_like", randn_like_kernel);
    m.impl("randint_like", randint_like_kernel);
    m.impl("clone", clone_kernel);
    m.impl("to.device", to_device_kernel);
    m.impl("to.other", to_other_kernel);
    m.impl("_to_copy", to_copy_kernel);
    m.impl("to.dtype_layout", to_dtype_layout_kernel);
    m.impl("fill_.Scalar", fill_kernel);
    m.impl("zero_", zero_kernel);

    // Pointwise arithmetic.
    m.impl("add.Tensor", add_kernel);
    m.impl("add.Scalar", add_scalar_kernel);
    m.impl("add.out", add_out_kernel);
    m.impl("add_.Tensor", add_inplace_kernel);
    m.impl("add_.Scalar", add_scalar_inplace_kernel);
    m.impl("sub.Tensor", sub_kernel);
    m.impl("sub.Scalar", sub_scalar_kernel);
    m.impl("sub_.Tensor", sub_inplace_kernel);
    m.impl("sub_.Scalar", sub_scalar_inplace_kernel);
    m.impl("mul.Tensor", mul_kernel);
    m.impl("mul.Scalar", mul_scalar_kernel);
    m.impl("mul_.Tensor", mul_inplace_kernel);
    m.impl("mul_.Scalar", mul_scalar_inplace_kernel);
    m.impl("div.Tensor", div_kernel);
    m.impl("div.Scalar", div_scalar_kernel);
    m.impl("div_.Tensor", div_inplace_kernel);
    m.impl("div_.Scalar", div_scalar_inplace_kernel);

    // Pointwise unary.
    m.impl("abs", abs_kernel);
    m.impl("neg", neg_kernel);
    m.impl("neg_", neg_inplace_kernel);
    m.impl("exp", exp_kernel);
    m.impl("log", log_kernel);
    m.impl("sqrt", sqrt_kernel);
    m.impl("clamp", clamp_kernel);

    // Comparisons and selection.
    m.impl("eq.Tensor", eq_tensor_kernel);
    m.impl("eq.Scalar", eq_scalar_kernel);
    m.impl("ne.Tensor", ne_tensor_kernel);
    m.impl("ne.Scalar", ne_scalar_kernel);
    m.impl("lt.Tensor", lt_tensor_kernel);
    m.impl("lt.Scalar", lt_scalar_kernel);
    m.impl("le.Tensor", le_tensor_kernel);
    m.impl("le.Scalar", le_scalar_kernel);
    m.impl("gt.Tensor", gt_tensor_kernel);
    m.impl("gt.Scalar", gt_scalar_kernel);
    m.impl("ge.Tensor", ge_tensor_kernel);
    m.impl("ge.Scalar", ge_scalar_kernel);
    m.impl("where.self", where_cpu);
    m.impl("where.ScalarSelf", where_scalar_self_cpu);
    m.impl("where.ScalarOther", where_scalar_other_cpu);
    m.impl("where.Scalar", where_scalar_scalar_cpu);
    m.impl("maximum", maximum_cpu);
    m.impl("minimum", minimum_cpu);

    // Reductions.
    m.impl("sum", sum_kernel);
    m.impl("sum.dim_IntList", sum_dim_kernel);
    m.impl("mean", mean_kernel);
    m.impl("mean.dim", mean_dim_kernel);
    m.impl("prod", prod_kernel);
    m.impl("prod.dim_IntList", prod_dim_kernel);
    m.impl("argmax", arg_kernel);
    m.impl("argmin", arg_kernel);
    m.impl("amax", amax_cpu);
    m.impl("amin", amin_cpu);
    m.impl("aminmax", aminmax_cpu);

    // Matrix products and concatenation.
    m.impl("mm", mm_kernel);
    m.impl("matmul", matmul_kernel);
    m.impl("reshape", reshape_kernel);
    m.impl("bmm", bmm_kernel);
    m.impl("addmm", addmm_kernel);
    m.impl("cat", cat_kernel);

    // Views.
    m.impl("transpose", transpose_kernel);
    m.impl("slice.Tensor", slice_kernel);
    m.impl("squeeze", squeeze_kernel);
    m.impl("squeeze.dim", squeeze_dim_kernel);
    m.impl("squeeze.dims", squeeze_dims_kernel);
    m.impl("unsqueeze", unsqueeze_kernel);
    m.impl("chunk", chunk_kernel);

    // Convolution and normalization.
    m.impl("conv2d", conv2d_kernel);
    m.impl("group_norm", group_norm_kernel);
    m.impl("native_group_norm", native_group_norm_kernel);

    // Pointwise activations and roots.
    m.impl("silu", silu_kernel);
    m.impl("rsqrt", rsqrt_kernel);
    m.impl("cos", cos_kernel);
    m.impl("sin", sin_kernel);

    // Dropout and moments.
    m.impl("native_dropout", native_dropout_kernel);
    m.impl("var_mean", var_mean_kernel);

    // Attention.
    m.impl("scaled_dot_product_attention", sdpa_kernel);
    m.impl("_scaled_dot_product_attention_with_lse", sdpa_with_lse_kernel);

    // Padding, resampling and dtype queries.
    m.impl("constant_pad_nd", constant_pad_nd_kernel);
    m.impl("upsample_nearest2d", upsample_nearest2d_kernel);
    m.impl("is_quantized", is_quantized_kernel);

    // Copy.
    m.impl("copy_", copy_kernel);
}

} // namespace meta
} // namespace tensorplay
