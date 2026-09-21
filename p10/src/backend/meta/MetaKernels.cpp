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
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace meta {

namespace {

// Allocation that bypasses the dispatcher: the Tensor factory methods
// redispatch to the "empty" operator, which resolves back to this backend.
Tensor empty_meta(std::vector<int64_t> size, DType dtype, const Device& device) {
    return Tensor(std::make_shared<TensorImpl>(size, dtype, device));
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
    for (int64_t d : dims) {
        if (d < 0) d += nd;
        if (d < 0 || d >= nd) {
            TP_THROW(IndexError, "reduction dimension ", d, " out of range for a rank-",
                     nd, " tensor");
        }
        reduced[static_cast<size_t>(d)] = true;
    }
    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(nd));
    for (int64_t i = 0; i < nd; ++i) {
        if (!reduced[static_cast<size_t>(i)]) {
            out.push_back(self.size(i));
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

    // Copy.
    m.impl("copy_", copy_kernel);
}

} // namespace meta
} // namespace tensorplay
