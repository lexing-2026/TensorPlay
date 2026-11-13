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
#include "DTypeNames.h"
#include "DistributionDispatch.h"
#include "TypePromotion.h"
#include "TypeProperties.h"
#include "../BlasComposite.h"

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

// A float-only pointwise op answers an integral or boolean operand in the
// default floating type, because the operation has no integral form to answer
// in; a floating or complex operand keeps its own type.
DType unary_float_dtype(DType dtype) {
    if (isIntegralType(dtype, /*includeBool=*/true)) {
        return globalContext().defaultDType();
    }
    return dtype;
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

Tensor arange_start_stub(const Scalar& start, const Scalar& end,
                         std::optional<DType> dtype,
                         std::optional<int64_t> /*layout*/,
                         std::optional<Device> device,
                         std::optional<bool> /*pin_memory*/) {
    return arange_start_step_stub(
        start, end, Scalar(1),
        dtype.has_value() ? *dtype : DType::Undefined, device);
}

Tensor arange_start_step_opt_stub(const Scalar& start, const Scalar& end,
                                  const Scalar& step,
                                  std::optional<DType> dtype,
                                  std::optional<int64_t> /*layout*/,
                                  std::optional<Device> device,
                                  std::optional<bool> /*pin_memory*/) {
    return arange_start_step_stub(
        start, end, step, dtype.has_value() ? *dtype : DType::Undefined,
        device);
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
// Random fills.
//
// A fill writes values the meta device does not hold, so the tensor itself is
// the result, with unchanged metadata.  Arguments whose validity does not
// depend on stored data are still checked, so a call the computing backends
// would reject is rejected here too.  Probability and bound tensors carry
// their data elsewhere and are not read.
// ---------------------------------------------------------------------------

void check_random_writable(const Tensor& t) {
    const auto sizes = static_cast<std::vector<int64_t>>(t.shape());
    const auto strides = t.strides();
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (strides[i] == 0 && sizes[i] > 1) {
            TP_THROW(RuntimeError,
                     "unsupported operation: more than one element of the "
                     "written-to tensor refers to a single memory location. "
                     "Please clone() the tensor before performing the "
                     "operation.");
        }
    }
}

void check_floating_fill_dtype(const Tensor& self) {
    if (!isFloatingType(self.dtype())) {
        TP_THROW(NotImplementedError,
                 "distribution only supports floating dtypes");
    }
}

Tensor& uniform_inplace_kernel(Tensor& self, double from, double to,
                               std::optional<Generator> /*generator*/) {
    const DType value_dtype = isComplexType(self.dtype())
                                  ? toRealValueType(self.dtype())
                                  : self.dtype();
    if (!isFloatingType(value_dtype)) {
        TP_THROW(NotImplementedError,
                 "\"check_uniform_bounds\" not implemented for '",
                 toString(self.dtype()), "'");
    }
    const char* dtype_name = distribution::bounds_dtype_name(value_dtype);
    distribution::dispatch_dtype(value_dtype, [&](auto tag) {
        using scalar_t = decltype(tag);
        const double min = distribution::fp_dtype_lowest<scalar_t>();
        const double max = distribution::fp_dtype_max<scalar_t>();
        if (!(from >= min && from <= max)) {
            TP_THROW(RuntimeError, "from is out of bounds for ", dtype_name);
        }
        if (!(to >= min && to <= max)) {
            TP_THROW(RuntimeError, "to is out of bounds for ", dtype_name);
        }
        if (from > to) {
            TP_THROW(RuntimeError,
                     "uniform_ expects to return a [from, to) range, but "
                     "found from=", from, " > to=", to);
        }
        if ((to - from) > max) {
            TP_THROW(RuntimeError,
                     "uniform_ expects to-from <= std::numeric_limits<",
                     dtype_name, ">::max(), but found to=", to, " and from=",
                     from, " which result in to-from to exceed the limit");
        }
    });
    if (self.numel() == 0) return self;
    check_random_writable(self);
    return self;
}

Tensor& normal_inplace_kernel(Tensor& self, double /*mean*/, double std,
                              std::optional<Generator> /*generator*/) {
    if (!(std >= 0.0)) {
        TP_THROW(RuntimeError, "normal expects std >= 0.0, but found std ", std);
    }
    if (self.numel() == 0) return self;
    check_random_writable(self);
    check_floating_fill_dtype(self);
    return self;
}

Tensor& cauchy_inplace_kernel(Tensor& self, double /*median*/, double sigma,
                              std::optional<Generator> /*generator*/) {
    if (!(sigma > 0.0)) {
        TP_THROW(RuntimeError,
                 "cauchy_ expects sigma > 0.0, but found sigma=", sigma);
    }
    if (self.numel() == 0) return self;
    check_random_writable(self);
    check_floating_fill_dtype(self);
    return self;
}

Tensor& exponential_inplace_kernel(Tensor& self, double lambd,
                                   std::optional<Generator> /*generator*/) {
    if (!(lambd > 0.0)) {
        TP_THROW(RuntimeError,
                 "exponential_ expects lambda > 0.0, but found lambda=", lambd);
    }
    if (self.numel() == 0) return self;
    check_random_writable(self);
    check_floating_fill_dtype(self);
    return self;
}

Tensor& log_normal_inplace_kernel(Tensor& self, double /*mean*/, double std,
                                  std::optional<Generator> /*generator*/) {
    if (!(std > 0.0)) {
        TP_THROW(RuntimeError,
                 "log_normal_ expects std > 0.0, but found std=", std);
    }
    if (self.numel() == 0) return self;
    check_random_writable(self);
    check_floating_fill_dtype(self);
    return self;
}

Tensor& geometric_inplace_kernel(Tensor& self, double p,
                                 std::optional<Generator> /*generator*/) {
    if (!(0.0 < p && p < 1.0)) {
        TP_THROW(RuntimeError,
                 "geometric_ expects p to be in (0, 1), but got p=", p);
    }
    if (self.numel() == 0) return self;
    check_random_writable(self);
    return self;
}

Tensor& bernoulli_float_inplace_kernel(Tensor& self, double p,
                                       std::optional<Generator> /*generator*/) {
    if (!(p >= 0.0 && p <= 1.0)) {
        TP_THROW(ValueError, "bernoulli_ expects p to be in [0, 1]");
    }
    if (self.numel() == 0) return self;
    check_random_writable(self);
    return self;
}

Tensor& bernoulli_tensor_inplace_kernel(
        Tensor& self, const Tensor& probabilities,
        std::optional<Generator> /*generator*/) {
    if (self.device() != probabilities.device()) {
        TP_THROW(DeviceMismatchError,
                 "bernoulli_: probability tensor must be on the same device");
    }
    if (self.numel() == 0) return self;
    check_random_writable(self);
    if (!isFloatingType(probabilities.dtype())) {
        TP_THROW(TypeError,
                 "bernoulli_ probability tensor must have a floating dtype");
    }
    return self;
}

Tensor& random_inplace_kernel(Tensor& self,
                              std::optional<Generator> /*generator*/) {
    if (self.numel() == 0) return self;
    check_random_writable(self);
    return self;
}

Tensor& random_from_inplace_kernel(Tensor& self, int64_t from,
                                   std::optional<int64_t> to,
                                   std::optional<Generator> /*generator*/) {
    if (to.has_value()) {
        if (*to <= from) {
            TP_THROW(RuntimeError,
                     "random_ expects 'from' to be less than 'to', but got "
                     "from=", from, " >= to=", *to);
        }
        distribution::check_random_from_to_bounds(from, *to, self.dtype());
        if (self.numel() == 0) return self;
        check_random_writable(self);
        return self;
    }
    if (self.numel() == 0) return self;
    check_random_writable(self);
    distribution::dispatch_dtype(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        const uint64_t top = distribution::unbounded_top<scalar_t>();
        if (from >= 0 && static_cast<uint64_t>(from) > top) {
            TP_THROW(RuntimeError,
                     "random_ expects 'from' to be at most ", top, " for ",
                     self.dtype(), ", but got from=", from);
        }
    });
    return self;
}

Tensor& random_to_inplace_kernel(Tensor& self, int64_t to,
                                 std::optional<Generator> generator) {
    return random_from_inplace_kernel(self, 0, to, std::move(generator));
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
// Pointwise maps beyond the four arithmetic ones.  The pair maps that compute
// at floating precision answer whole-number inputs at the default real type;
// the modulo and division families keep an integral pair integral, and true
// division lifts every integral pair to a real type.
// ---------------------------------------------------------------------------

DType float_pair_dtype(DType a, DType b) {
    const DType out = promoteTypes(a, b);
    return isFloatingType(out) ? out : DType::Float32;
}

Tensor float_pair_kernel(const Tensor& self, const Tensor& other) {
    check_all_meta(self, other);
    return empty_meta(broadcast2(self, other),
                      float_pair_dtype(self.dtype(), other.dtype()),
                      self.device());
}

Tensor float_pair_scalar_kernel(const Tensor& self, const Scalar& other) {
    return empty_meta(shape_of(self),
                      float_pair_dtype(self.dtype(),
                                       result_type(other, self.dtype())),
                      self.device());
}

Tensor same_pair_kernel(const Tensor& self, const Tensor& other) {
    check_all_meta(self, other);
    return empty_meta(broadcast2(self, other),
                      promoteTypes(self.dtype(), other.dtype()), self.device());
}

// A complex scalar carries no lane in the whole-number maps, so it answers
// at the operand's own type instead of widening the result.
Tensor same_pair_scalar_kernel(const Tensor& self, const Scalar& other) {
    const DType out = other.isComplex()
                          ? self.dtype()
                          : promoteTypes(self.dtype(),
                                         result_type(other, self.dtype()));
    return empty_meta(shape_of(self), out, self.device());
}

void reject_complex_pair(const Tensor& a, const Tensor& b, const char* op) {
    if (isComplexType(a.dtype()) || isComplexType(b.dtype())) {
        TP_THROW(TypeError, op, ": unsupported dtype for complex operands");
    }
}

void reject_complex_scalar(const Tensor& self, const Scalar& other,
                           const char* op) {
    if (isComplexType(self.dtype()) || other.isComplex()) {
        TP_THROW(TypeError, op, ": unsupported dtype for complex operands");
    }
}

Tensor fmod_pair_kernel(const Tensor& self, const Tensor& other) {
    reject_complex_pair(self, other, "fmod");
    return same_pair_kernel(self, other);
}

Tensor fmod_scalar_kernel(const Tensor& self, const Scalar& other) {
    reject_complex_scalar(self, other, "fmod");
    return same_pair_scalar_kernel(self, other);
}

Tensor remainder_pair_kernel(const Tensor& self, const Tensor& other) {
    reject_complex_pair(self, other, "remainder");
    return same_pair_kernel(self, other);
}

Tensor remainder_scalar_kernel(const Tensor& self, const Scalar& other) {
    reject_complex_scalar(self, other, "remainder");
    return same_pair_scalar_kernel(self, other);
}

Tensor remainder_scalar_tensor_kernel(const Scalar& self, const Tensor& other) {
    if (isComplexType(other.dtype()) || self.isComplex()) {
        TP_THROW(TypeError,
                 "remainder: unsupported dtype for complex operands");
    }
    const DType out = self.isComplex()
                          ? other.dtype()
                          : promoteTypes(result_type(self, other.dtype()),
                                         other.dtype());
    return empty_meta(shape_of(other), out, other.device());
}

Tensor floor_divide_pair_kernel(const Tensor& self, const Tensor& other) {
    reject_complex_pair(self, other, "floor_divide");
    return same_pair_kernel(self, other);
}

Tensor floor_divide_scalar_kernel(const Tensor& self, const Scalar& other) {
    reject_complex_scalar(self, other, "floor_divide");
    return same_pair_scalar_kernel(self, other);
}

Tensor heaviside_kernel(const Tensor& self, const Tensor& other) {
    reject_complex_pair(self, other, "heaviside");
    return same_pair_kernel(self, other);
}

Tensor copysign_scalar_kernel(const Tensor& self, const Scalar& other) {
    (void)other;
    // The sign is carried at single precision and never widens the answer.
    return empty_meta(shape_of(self),
                      float_pair_dtype(self.dtype(), DType::Float32),
                      self.device());
}

Tensor true_divide_kernel(const Tensor& self, const Tensor& other) {
    check_all_meta(self, other);
    return empty_meta(broadcast2(self, other),
                      div_like_dtype(self.dtype(), other.dtype()), self.device());
}

Tensor true_divide_scalar_kernel(const Tensor& self, const Scalar& other) {
    return empty_meta(shape_of(self), div_like_scalar_dtype(self.dtype(), other),
                      self.device());
}

// An in-place pair map writes element data only, so the operand's geometry
// and value type are already the ones the write would leave behind.
Tensor& pair_inplace_kernel(Tensor& self, const Tensor& other) {
    check_inplace_fits(self, other);
    return self;
}

Tensor& pair_scalar_inplace_kernel(Tensor& self, const Scalar& other) {
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
    return empty_meta(shape_of(self), unary_float_dtype(self.dtype()),
                      self.device());
}

Tensor log_kernel(const Tensor& self) {
    return empty_meta(shape_of(self), unary_float_dtype(self.dtype()),
                      self.device());
}

Tensor sqrt_kernel(const Tensor& self) {
    return empty_meta(shape_of(self), unary_float_dtype(self.dtype()),
                      self.device());
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

std::tuple<Tensor, Tensor> aminmax_all_kernel(const Tensor& self) {
    return aminmax_cpu(self, {}, false);
}

std::tuple<Tensor, Tensor> aminmax_dim_kernel(const Tensor& self, int64_t dim,
                                              bool keepdim) {
    return aminmax_cpu(self, {dim}, keepdim);
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
    if (self.dtype() != mat2.dtype()) {
        TP_THROW(RuntimeError,
                 "expected m1 and m2 to have the same dtype, but got: ",
                 elementTypeName(self.dtype()), " != ",
                 elementTypeName(mat2.dtype()));
    }
    if (self.dtype() == DType::Bool) {
        TP_THROW(NotImplementedError,
                 "\"addmm_impl_cpu_\" not implemented for 'Bool'");
    }
    return empty_meta({self.size(0), mat2.size(1)}, self.dtype(),
                      self.device());
}

// General N-D product shapes: vector operands fold to rows/columns, batch
// dimensions broadcast, and the matrix part follows mm.
Tensor matmul_kernel(const Tensor& self, const Tensor& other) {
    if (self.dim() < 1 || other.dim() < 1) {
        TP_THROW(RuntimeError, "matmul(): input operands must be at least 1D");
    }
    check_all_meta(self, other);
    const std::vector<int64_t> a = shape_of(self);
    const std::vector<int64_t> b = shape_of(other);
    const int64_t dim1 = self.dim();
    const int64_t dim2 = other.dim();
    std::vector<int64_t> out;

    if (dim1 == 1 && dim2 == 1) {
        if (a[0] != b[0]) {
            TP_THROW(RuntimeError, "inconsistent tensor size, expected tensor [",
                     a[0], "] and src [", b[0],
                     "] to have the same number of elements");
        }
    } else if (dim1 == 1) {
        const int64_t k = a[0];
        const int64_t other_k = b[b.size() - 2];
        if (k != other_k) {
            TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (1x", k,
                     " and ", other_k, "x", b.back(), ")");
        }
        out.assign(b.begin(), b.end() - 2);
        out.push_back(b.back());
    } else if (dim2 == 1) {
        const int64_t k = a.back();
        if (k != b[0]) {
            int64_t folded_m = a[a.size() - 2];
            for (size_t i = 0; i + 2 < a.size(); ++i) folded_m *= a[i];
            TP_THROW(RuntimeError, "size mismatch, got input (", folded_m,
                     "), mat (", folded_m, "x", k, "), vec (", b[0], ")");
        }
        out.assign(a.begin(), a.end() - 2);
        out.push_back(a[a.size() - 2]);
    } else {
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
        out = broadcast_shapes(
            std::vector<int64_t>(a.begin(), a.end() - 2),
            std::vector<int64_t>(b.begin(), b.end() - 2));
        out.push_back(a[a.size() - 2]);
        out.push_back(b.back());
    }
    if (self.dtype() != other.dtype()) {
        TP_THROW(RuntimeError,
                 "expected m1 and m2 to have the same dtype, but got: ",
                 elementTypeName(self.dtype()), " != ",
                 elementTypeName(other.dtype()));
    }
    if (self.dtype() == DType::Bool) {
        TP_THROW(NotImplementedError,
                 "\"addmm_impl_cpu_\" not implemented for 'Bool'");
    }
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
    if (self.dim() != 3) TP_THROW(RuntimeError, "batch1 must be a 3D tensor");
    if (batch2.dim() != 3) TP_THROW(RuntimeError, "batch2 must be a 3D tensor");
    if (self.size(0) != batch2.size(0) || self.size(2) != batch2.size(1)) {
        TP_THROW(RuntimeError,
                 "Expected size for first two dimensions of batch2 tensor to "
                 "be: [", self.size(0), ", ", self.size(2), "] but got: [",
                 batch2.size(0), ", ", batch2.size(1), "].");
    }
    if (self.dtype() != batch2.dtype()) {
        TP_THROW(RuntimeError, "expected scalar type ",
                 scalarTypeName(self.dtype()), " but found ",
                 scalarTypeName(batch2.dtype()));
    }
    if (self.dtype() == DType::Bool) {
        TP_THROW(NotImplementedError,
                 "\"addmm_impl_cpu_\" not implemented for 'Bool'");
    }
    return empty_meta({self.size(0), self.size(1), batch2.size(2)},
                      self.dtype(), self.device());
}

// addmm broadcasts the bias input against the product of the two matrices.
// Every operand shares one dtype; a whole-number product is blended at single
// precision.  A zero weight leaves the bias unread.
std::string matmul_operand_desc(const Tensor& t) {
    if (t.dim() != 2) return std::to_string(t.dim()) + "D";
    return std::to_string(t.size(0)) + "x" + std::to_string(t.size(1));
}

void check_addmm_operands(const Tensor& input, const Tensor& mat1,
                          const Tensor& mat2) {
    if (mat1.dim() != 2 || mat2.dim() != 2) {
        TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (",
                 matmul_operand_desc(mat1), " and ",
                 matmul_operand_desc(mat2), ")");
    }
    if (mat1.size(1) != mat2.size(0)) {
        TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (",
                 mat1.size(0), "x", mat1.size(1), " and ", mat2.size(0), "x",
                 mat2.size(1), ")");
    }
    if (mat1.dtype() != mat2.dtype()) {
        TP_THROW(RuntimeError,
                 "mat1 and mat2 must have the same dtype, but got ",
                 scalarTypeName(mat1.dtype()), " and ",
                 scalarTypeName(mat2.dtype()));
    }
    if (input.dtype() != mat2.dtype()) {
        TP_THROW(RuntimeError,
                 "self and mat2 must have the same dtype, but got ",
                 scalarTypeName(input.dtype()), " and ",
                 scalarTypeName(mat2.dtype()));
    }
    if (mat2.dtype() == DType::Bool) {
        TP_THROW(NotImplementedError,
                 "\"addmm_impl_cpu_\" not implemented for 'Bool'");
    }
}

bool scalar_is_zero(const Scalar& s) {
    if (s.isComplex()) {
        return s.to<complex<double>>() == complex<double>(0.0, 0.0);
    }
    return s.toDouble() == 0.0;
}

void require_expandable_to(const Tensor& self,
                           const std::vector<int64_t>& target,
                           const char* op) {
    if (self.dim() > static_cast<int64_t>(target.size()) ||
        broadcast_shapes(shape_of(self), target) != target) {
        TP_THROW(RuntimeError,
                 op, ": the addend's shape cannot be expanded to the result's "
                 "shape");
    }
}

Tensor addmm_kernel(const Tensor& input, const Tensor& mat1, const Tensor& mat2,
                    const Scalar& beta, const Scalar& /*alpha*/) {
    check_addmm_operands(input, mat1, mat2);
    const std::vector<int64_t> target{mat1.size(0), mat2.size(1)};
    if (!scalar_is_zero(beta)) {
        require_expandable_to(input, target, "addmm");
    }
    const DType out = isIntegralType(mat2.dtype(), /*include_bool=*/true)
                          ? DType::Float32
                          : mat2.dtype();
    return empty_meta(target, out, mat1.device());
}

// The in-place form lands the product back on the bias, which must be able to
// hold the whole result.
Tensor& addmm_inplace_kernel(Tensor& self, const Tensor& mat1,
                             const Tensor& mat2, const Scalar& beta,
                             const Scalar& alpha) {
    const Tensor result = addmm_kernel(self, mat1, mat2, beta, alpha);
    require_expandable_to(result, shape_of(self), "addmm_");
    return self;
}

// ---------------------------------------------------------------------------
// Vector products and the batch accumulators.  dot and mv require equal value
// types, inner contracts the last axes and stacks the leading ones, and the
// accumulators seed a broadcast addend with the product.
// ---------------------------------------------------------------------------

void check_matmul_value_dtype(DType dtype, const char* op) {
    switch (dtype) {
        case DType::Bool:
        case DType::UInt16:
        case DType::UInt32:
        case DType::UInt64:
        case DType::ComplexHalf:
        case DType::BComplex32:
            TP_THROW(NotImplementedError, op, ": unsupported dtype ",
                     scalarTypeName(dtype));
        default:
            return;
    }
}

Tensor dot_kernel(const Tensor& self, const Tensor& other) {
    if (self.dim() != 1 || other.dim() != 1) {
        TP_THROW(RuntimeError, "1D tensors expected, but got ", self.dim(),
                 "D and ", other.dim(), "D tensors");
    }
    if (self.size(0) != other.size(0)) {
        TP_THROW(RuntimeError, "inconsistent tensor size, expected tensor [",
                 self.size(0), "] and src [", other.size(0),
                 "] to have the same number of elements, but got ",
                 self.size(0), " and ", other.size(0),
                 " elements respectively");
    }
    if (self.dtype() != other.dtype()) {
        TP_THROW(RuntimeError,
                 "dot : expected both vectors to have same dtype, but found ",
                 scalarTypeName(self.dtype()), " and ",
                 scalarTypeName(other.dtype()));
    }
    check_matmul_value_dtype(self.dtype(), "dot");
    return empty_meta({}, self.dtype(), self.device());
}

Tensor mv_kernel(const Tensor& self, const Tensor& vec) {
    if (self.dim() != 2 || vec.dim() != 1) {
        TP_THROW(RuntimeError, "vector + matrix @ vector expected, got ", 1,
                 ", ", self.dim(), ", ", vec.dim());
    }
    if (self.size(1) != vec.size(0)) {
        TP_THROW(RuntimeError, "size mismatch, got input (", self.size(0),
                 "), mat (", self.size(0), "x", self.size(1), "), vec (",
                 vec.size(0), ")");
    }
    if (self.dtype() != vec.dtype()) {
        // The leading slot is the accumulator's; a matrix-vector product
        // carries the vector's type there.
        TP_THROW(RuntimeError,
                 "addmv input tensors must have the same dtype, but got ",
                 scalarTypeName(vec.dtype()), ", ",
                 scalarTypeName(self.dtype()), ", and ",
                 scalarTypeName(vec.dtype()));
    }
    check_matmul_value_dtype(self.dtype(), "mv");
    return empty_meta({self.size(0)}, self.dtype(), self.device());
}

Tensor inner_kernel(const Tensor& self, const Tensor& other) {
    check_all_meta(self, other);
    if (self.dim() == 0 || other.dim() == 0) {
        return empty_meta(broadcast2(self, other),
                          promoteTypes(self.dtype(), other.dtype()),
                          self.device());
    }
    if (self.size(-1) != other.size(-1)) {
        TP_THROW(RuntimeError,
                 "inner() the last dimension must match on both input tensors "
                 "but got shapes ",
                 Size(shape_of(self)).toString(), " and ",
                 Size(shape_of(other)).toString());
    }
    if (self.dtype() != other.dtype()) {
        TP_THROW(RuntimeError,
                 "expected m1 and m2 to have the same dtype, but got: ",
                 elementTypeName(self.dtype()), " != ",
                 elementTypeName(other.dtype()));
    }
    if (self.dim() == 1 && other.dim() == 1) {
        return dot_kernel(self, other);
    }
    check_matmul_value_dtype(self.dtype(), "inner");
    const std::vector<int64_t> self_shape = shape_of(self);
    const std::vector<int64_t> other_shape = shape_of(other);
    std::vector<int64_t> out(self_shape.begin(), self_shape.end() - 1);
    out.insert(out.end(), other_shape.begin(), other_shape.end() - 1);
    return empty_meta(out, self.dtype(), self.device());
}

Tensor ger_kernel(const Tensor& self, const Tensor& vec2) {
    if (self.dim() != 1 || vec2.dim() != 1) {
        TP_THROW(RuntimeError, "1D tensors expected, but got ", self.dim(),
                 "D and ", vec2.dim(), "D tensors");
    }
    return empty_meta({self.numel(), vec2.numel()},
                      promoteTypes(self.dtype(), vec2.dtype()),
                      self.device());
}

Tensor baddbmm_kernel(const Tensor& self, const Tensor& batch1,
                      const Tensor& batch2, const Scalar& beta,
                      const Scalar& /*alpha*/) {
    if (batch1.dim() != 3) TP_THROW(RuntimeError, "batch1 must be a 3D tensor");
    if (batch2.dim() != 3) TP_THROW(RuntimeError, "batch2 must be a 3D tensor");
    if (batch1.size(0) != batch2.size(0) || batch1.size(2) != batch2.size(1)) {
        TP_THROW(RuntimeError,
                 "Expected size for first two dimensions of batch2 tensor to "
                 "be: [", batch1.size(0), ", ", batch1.size(2), "] but got: [",
                 batch2.size(0), ", ", batch2.size(1), "].");
    }
    if (self.dtype() != batch1.dtype() || batch1.dtype() != batch2.dtype()) {
        TP_THROW(RuntimeError, "Input dtypes must be the same, got: input ",
                 elementTypeName(self.dtype()), ", batch1: ",
                 elementTypeName(batch1.dtype()), ", batch2: ",
                 elementTypeName(batch2.dtype()));
    }
    check_matmul_value_dtype(self.dtype(), "baddbmm");
    const std::vector<int64_t> target{batch1.size(0), batch1.size(1),
                                      batch2.size(2)};
    // A zero weight leaves the addend unread, so only a weighted seed has to
    // reach the product's shape.
    if (!scalar_is_zero(beta)) {
        require_expandable_to(self, target, "baddbmm");
    }
    // The product is blended back through an add whose weight carries single
    // precision, which lifts a whole-number result to a real type.
    const DType out =
        add_like_dtype(self.dtype(), self.dtype(), Scalar(1.0));
    return empty_meta(target, out, self.device());
}

Tensor addbmm_kernel(const Tensor& self, const Tensor& batch1,
                     const Tensor& batch2, const Scalar& beta,
                     const Scalar& alpha) {
    if (batch1.dim() != 3) TP_THROW(RuntimeError, "batch1 must be a 3D tensor");
    if (batch2.dim() != 3) TP_THROW(RuntimeError, "batch2 must be a 3D tensor");
    if (batch1.size(0) != batch2.size(0) || batch1.size(2) != batch2.size(1)) {
        TP_THROW(RuntimeError, "Incompatible matrix sizes for bmm (",
                 batch1.size(1), "x", batch1.size(2), " and ",
                 batch2.size(1), "x", batch2.size(2), ")");
    }
    const DType out = promoteTypes(promoteTypes(batch1.dtype(), batch2.dtype()),
                                   self.dtype());
    if (out == DType::Bool) {
        TP_THROW(NotImplementedError, "\"addbmm\" not implemented for 'Bool'");
    }
    if (!isFloatingType(out) && !isComplexType(out)) {
        if (!beta.isIntegral(/*includeBool=*/true)) {
            TP_THROW(RuntimeError,
                     "For integral input tensors, argument beta must not be a "
                     "floating point number.");
        }
        if (!alpha.isIntegral(/*includeBool=*/true)) {
            TP_THROW(RuntimeError,
                     "For integral input tensors, argument alpha must not be a "
                     "floating point number.");
        }
    }
    const std::vector<int64_t> target{batch1.size(1), batch2.size(2)};
    // A zero weight on a whole-number addend leaves it unread, and that route
    // skips the broadcast check.
    if (!(scalar_is_zero(beta) && !isFloatingType(out))) {
        require_expandable_to(self, target, "addbmm");
    }
    return empty_meta(target, out, self.device());
}

Tensor& baddbmm_inplace_kernel(Tensor& self, const Tensor& batch1,
                               const Tensor& batch2, const Scalar& /*beta*/,
                               const Scalar& /*alpha*/) {
    baddbmm_kernel(self, batch1, batch2, Scalar(1), Scalar(1));
    return self;
}

Tensor& addbmm_inplace_kernel(Tensor& self, const Tensor& batch1,
                              const Tensor& batch2, const Scalar& /*beta*/,
                              const Scalar& /*alpha*/) {
    addbmm_kernel(self, batch1, batch2, Scalar(1), Scalar(1));
    return self;
}

// addr and addmv blend a broadcast addend with a product under one shared
// scalar reading: a truth-value weight needs a truth-value result, and a
// whole-number result takes only whole-number weights.  The floating route
// answers at the promotion of all three operands; other value types compose
// through broadcast arithmetic, and a zero weight leaves the addend unread.
bool all_floating(const Tensor& a, const Tensor& b, const Tensor& c) {
    return isFloatingType(a.dtype()) && isFloatingType(b.dtype()) &&
           isFloatingType(c.dtype());
}

Tensor addr_kernel(const Tensor& self, const Tensor& vec1, const Tensor& vec2,
                   const Scalar& beta, const Scalar& alpha) {
    const DType dtype =
        promoteTypes(promoteTypes(vec1.dtype(), vec2.dtype()), self.dtype());
    const bool native = all_floating(self, vec1, vec2);
    if (!native) {
        if (vec1.dim() != 1) {
            TP_THROW(RuntimeError, "addr: Expected 1-D argument vec1, but got ",
                     vec1.dim(), "-D");
        }
        if (vec2.dim() != 1) {
            TP_THROW(RuntimeError, "addr: Expected 1-D argument vec2, but got ",
                     vec2.dim(), "-D");
        }
    }
    blas_composite::check_scalar(dtype, beta, "beta");
    blas_composite::check_scalar(dtype, alpha, "alpha");
    const std::vector<int64_t> target{vec1.numel(), vec2.numel()};
    // The floating product reaches for the addend whatever its weight; the
    // composed product only reads it when the weight is nonzero.
    if (native || !scalar_is_zero(beta)) {
        require_expandable_to(self, target, "addr");
    }
    return empty_meta(target, dtype, self.device());
}

Tensor& addr_inplace_kernel(Tensor& self, const Tensor& vec1,
                            const Tensor& vec2, const Scalar& beta,
                            const Scalar& alpha) {
    const Tensor result = addr_kernel(self, vec1, vec2, beta, alpha);
    require_expandable_to(result, shape_of(self), "addr_");
    return self;
}

Tensor addmv_kernel(const Tensor& self, const Tensor& mat, const Tensor& vec,
                    const Scalar& beta, const Scalar& alpha) {
    const DType dtype =
        promoteTypes(promoteTypes(mat.dtype(), vec.dtype()), self.dtype());
    const bool native = all_floating(self, mat, vec);
    if (!native && dtype == DType::Bool) {
        TP_THROW(NotImplementedError, "\"addmv\" not implemented for 'Bool'");
    }
    if (mat.dim() != 2) TP_THROW(RuntimeError, "addmv: mat must be a matrix");
    if (vec.dim() != 1) TP_THROW(RuntimeError, "addmv: vec must be a vector");
    if (vec.numel() != mat.size(1)) {
        TP_THROW(RuntimeError, "addmv: both args should have matching shapes");
    }
    if (!native) {
        blas_composite::check_scalar(dtype, beta, "beta");
        blas_composite::check_scalar(dtype, alpha, "alpha");
    }
    if (!scalar_is_zero(beta)) {
        require_expandable_to(self, {mat.size(0)}, "addmv");
    }
    return empty_meta({mat.size(0)}, dtype, self.device());
}

Tensor& addmv_inplace_kernel(Tensor& self, const Tensor& mat, const Tensor& vec,
                             const Scalar& beta, const Scalar& alpha) {
    const Tensor result = addmv_kernel(self, mat, vec, beta, alpha);
    require_expandable_to(result, shape_of(self), "addmv_");
    return self;
}

// The ternary blend of a product or quotient into a scaled addend.  Division
// lifts a whole-number result to single precision and refuses an all-integer
// quotient outright; the in-place forms land back on the addend, which has to
// already hold the three-way broadcast.
Tensor addcmul_kernel(const Tensor& self, const Tensor& tensor1,
                      const Tensor& tensor2, const Scalar& /*value*/) {
    return empty_meta(broadcast_shapes(shape_of(self), shape_of(tensor1),
                                       shape_of(tensor2)),
                      promoteTypes(promoteTypes(self.dtype(), tensor1.dtype()),
                                   tensor2.dtype()),
                      self.device());
}

Tensor& addcmul_inplace_kernel(Tensor& self, const Tensor& tensor1,
                               const Tensor& tensor2, const Scalar& /*value*/) {
    if (broadcast_shapes(shape_of(self), shape_of(tensor1),
                         shape_of(tensor2)) != shape_of(self)) {
        TP_THROW(RuntimeError, "addcmul_: output shape does not match self");
    }
    return self;
}

void check_addcdiv_dtypes(const Tensor& tensor1, const Tensor& tensor2) {
    if (isIntegralType(tensor1.dtype(), /*include_bool=*/true) &&
        isIntegralType(tensor2.dtype(), /*include_bool=*/true)) {
        TP_THROW(RuntimeError, "Integer division with addcdiv is not supported");
    }
}

Tensor addcdiv_kernel(const Tensor& self, const Tensor& tensor1,
                      const Tensor& tensor2, const Scalar& /*value*/) {
    check_addcdiv_dtypes(tensor1, tensor2);
    DType out = promoteTypes(promoteTypes(self.dtype(), tensor1.dtype()),
                             tensor2.dtype());
    if (isIntegralType(out)) out = DType::Float32;
    return empty_meta(broadcast_shapes(shape_of(self), shape_of(tensor1),
                                       shape_of(tensor2)),
                      out, self.device());
}

Tensor& addcdiv_inplace_kernel(Tensor& self, const Tensor& tensor1,
                               const Tensor& tensor2, const Scalar& /*value*/) {
    check_addcdiv_dtypes(tensor1, tensor2);
    if (broadcast_shapes(shape_of(self), shape_of(tensor1),
                         shape_of(tensor2)) != shape_of(self)) {
        TP_THROW(RuntimeError, "addcdiv_: output shape does not match self");
    }
    return self;
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

Tensor conv1d_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     const std::vector<int64_t>& stride,
                     const std::vector<int64_t>& padding,
                     const std::vector<int64_t>& dilation, int64_t groups) {
    if (input.dim() != 3 || weight.dim() != 3) {
        TP_THROW(RuntimeError, "conv1d: Expected 3D input and weight");
    }
    convolution::check_conv_shapes(input, weight, bias, groups, /*transposed=*/false);
    convolution::check_conv_geometry(input, weight, stride, padding, dilation, "conv1d");
    const std::vector<int64_t> spatial =
        convolution::conv_output_spatial_sizes(input, weight, stride, padding, dilation);
    std::vector<int64_t> out{input.size(0), weight.size(0)};
    out.insert(out.end(), spatial.begin(), spatial.end());
    return empty_meta(out, input.dtype(), input.device());
}

Tensor conv3d_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     const std::vector<int64_t>& stride,
                     const std::vector<int64_t>& padding,
                     const std::vector<int64_t>& dilation, int64_t groups) {
    if (input.dim() != 5 || weight.dim() != 5) {
        TP_THROW(RuntimeError, "conv3d: Expected 5D input and weight");
    }
    convolution::check_conv_shapes(input, weight, bias, groups, /*transposed=*/false);
    convolution::check_conv_geometry(input, weight, stride, padding, dilation, "conv3d");
    const std::vector<int64_t> spatial =
        convolution::conv_output_spatial_sizes(input, weight, stride, padding, dilation);
    std::vector<int64_t> out{input.size(0), weight.size(0)};
    out.insert(out.end(), spatial.begin(), spatial.end());
    return empty_meta(out, input.dtype(), input.device());
}

// A transposed convolution scatters each input element through the kernel, so
// its output grows with the stride and the dilation and shrinks by the
// cropping padding: (in - 1) * stride - 2 * padding + dilation * (kernel - 1)
// + output_padding + 1.  The weight's first channel dimension is the input's.
Tensor conv_transpose2d_kernel(const Tensor& input, const Tensor& weight,
                               const Tensor& bias,
                               const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& padding,
                               const std::vector<int64_t>& output_padding,
                               int64_t groups,
                               const std::vector<int64_t>& dilation) {
    if (input.dim() != 4 || weight.dim() != 4) {
        TP_THROW(RuntimeError, "conv_transpose2d: Expected 4D input and weight");
    }
    convolution::check_conv_shapes(input, weight, bias, groups, /*transposed=*/true);
    const auto expand = [](const std::vector<int64_t>& list, const char* name) {
        if (list.empty()) return std::vector<int64_t>{1, 1};
        if (list.size() == 1) return std::vector<int64_t>{list[0], list[0]};
        if (list.size() != 2) {
            TP_THROW(ValueError, "conv_transpose2d: ", name, " needs two values");
        }
        return list;
    };
    const std::vector<int64_t> st = expand(stride, "stride");
    const std::vector<int64_t> pd = expand(padding, "padding");
    const std::vector<int64_t> op = expand(output_padding, "output_padding");
    const std::vector<int64_t> dl = expand(dilation, "dilation");
    if (st[0] <= 0 || st[1] <= 0) {
        TP_THROW(RuntimeError, "conv_transpose2d: stride must be positive");
    }
    if (dl[0] <= 0 || dl[1] <= 0) {
        TP_THROW(RuntimeError, "conv_transpose2d: dilation must be positive");
    }
    if (pd[0] < 0 || pd[1] < 0 || op[0] < 0 || op[1] < 0) {
        TP_THROW(RuntimeError,
                 "conv_transpose2d: padding and output_padding must be "
                 "non-negative");
    }
    if ((op[0] >= st[0] && op[0] >= dl[0]) ||
        (op[1] >= st[1] && op[1] >= dl[1])) {
        TP_THROW(RuntimeError,
                 "conv_transpose2d: output_padding must be smaller than either "
                 "stride or dilation");
    }
    const int64_t out_channels = weight.size(1) * groups;
    const int64_t h_out =
        (input.size(2) - 1) * st[0] - 2 * pd[0] + dl[0] * (weight.size(2) - 1) +
        op[0] + 1;
    const int64_t w_out =
        (input.size(3) - 1) * st[1] - 2 * pd[1] + dl[1] * (weight.size(3) - 1) +
        op[1] + 1;
    if (h_out <= 0 || w_out <= 0) {
        TP_THROW(RuntimeError,
                 "conv_transpose2d: Calculated output size is too small");
    }
    return empty_meta({input.size(0), out_channels, h_out, w_out},
                      input.dtype(), input.device());
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

// A float-only unary op answers an integral operand in the default floating
// type; see ``unary_float_dtype``.
#define META_UNARY_FLOAT_KERNEL(NAME)                                         \
    Tensor NAME##_kernel(const Tensor& self) {                                \
        return empty_meta(shape_of(self), unary_float_dtype(self.dtype()),     \
                          self.device());                                     \
    }

META_UNARY_FLOAT_KERNEL(silu)
META_UNARY_FLOAT_KERNEL(rsqrt)
META_UNARY_FLOAT_KERNEL(cos)
META_UNARY_FLOAT_KERNEL(sin)
META_UNARY_FLOAT_KERNEL(exp2)
META_UNARY_FLOAT_KERNEL(expm1)
META_UNARY_FLOAT_KERNEL(log2)
META_UNARY_FLOAT_KERNEL(log10)
META_UNARY_FLOAT_KERNEL(log1p)
META_UNARY_FLOAT_KERNEL(tan)
META_UNARY_FLOAT_KERNEL(asin)
META_UNARY_FLOAT_KERNEL(acos)
META_UNARY_FLOAT_KERNEL(atan)
META_UNARY_FLOAT_KERNEL(sinh)
META_UNARY_FLOAT_KERNEL(cosh)
META_UNARY_FLOAT_KERNEL(asinh)
META_UNARY_FLOAT_KERNEL(acosh)
META_UNARY_FLOAT_KERNEL(atanh)
META_UNARY_FLOAT_KERNEL(reciprocal)
META_UNARY_KERNEL(square)
META_UNARY_FLOAT_KERNEL(sinc)
META_UNARY_FLOAT_KERNEL(erf)
META_UNARY_FLOAT_KERNEL(erfc)
META_UNARY_FLOAT_KERNEL(erfinv)
META_UNARY_FLOAT_KERNEL(lgamma)
META_UNARY_FLOAT_KERNEL(digamma)
META_UNARY_FLOAT_KERNEL(i0)
META_UNARY_FLOAT_KERNEL(deg2rad)
META_UNARY_FLOAT_KERNEL(rad2deg)
#undef META_UNARY_KERNEL
#undef META_UNARY_FLOAT_KERNEL

Tensor absolute_kernel(const Tensor& self) {
    return abs_kernel(self);
}

// An in-place pointwise operation overwrites the operand's own bytes, so the
// geometry and the value type both stay as they were.
#define META_UNARY_INPLACE_KERNEL(NAME)                                       \
    Tensor& NAME##_inplace_kernel(Tensor& self) { return self; }

META_UNARY_INPLACE_KERNEL(abs)
META_UNARY_INPLACE_KERNEL(absolute)
META_UNARY_INPLACE_KERNEL(sqrt)
META_UNARY_INPLACE_KERNEL(rsqrt)
META_UNARY_INPLACE_KERNEL(exp)
META_UNARY_INPLACE_KERNEL(exp2)
META_UNARY_INPLACE_KERNEL(expm1)
META_UNARY_INPLACE_KERNEL(log)
META_UNARY_INPLACE_KERNEL(log2)
META_UNARY_INPLACE_KERNEL(log10)
META_UNARY_INPLACE_KERNEL(log1p)
META_UNARY_INPLACE_KERNEL(sin)
META_UNARY_INPLACE_KERNEL(cos)
META_UNARY_INPLACE_KERNEL(tan)
META_UNARY_INPLACE_KERNEL(asin)
META_UNARY_INPLACE_KERNEL(acos)
META_UNARY_INPLACE_KERNEL(atan)
META_UNARY_INPLACE_KERNEL(sinh)
META_UNARY_INPLACE_KERNEL(cosh)
META_UNARY_INPLACE_KERNEL(asinh)
META_UNARY_INPLACE_KERNEL(acosh)
META_UNARY_INPLACE_KERNEL(atanh)
META_UNARY_INPLACE_KERNEL(ceil)
META_UNARY_INPLACE_KERNEL(floor)
META_UNARY_INPLACE_KERNEL(round)
META_UNARY_INPLACE_KERNEL(reciprocal)
META_UNARY_INPLACE_KERNEL(square)
META_UNARY_INPLACE_KERNEL(sinc)
META_UNARY_INPLACE_KERNEL(erf)
META_UNARY_INPLACE_KERNEL(erfc)
META_UNARY_INPLACE_KERNEL(erfinv)
META_UNARY_INPLACE_KERNEL(lgamma)
META_UNARY_INPLACE_KERNEL(digamma)
META_UNARY_INPLACE_KERNEL(i0)
META_UNARY_INPLACE_KERNEL(deg2rad)
META_UNARY_INPLACE_KERNEL(rad2deg)
#undef META_UNARY_INPLACE_KERNEL

// ---------------------------------------------------------------------------
// Predicates and the maps that answer beside the operand's own type.  A
// predicate reports element data, so its answer is Bool in the operand's
// geometry; the floating predicates have no complex lane, while isreal and
// is_complex inspect only the value type.  The remaining maps compute in real
// arithmetic: a complex operand answers at its real width and a whole-number
// operand at the default real type.
// ---------------------------------------------------------------------------

Tensor bool_answer(const Tensor& self, const char* op) {
    const DType dtype = self.dtype();
    const bool has_real_lane = dtype == DType::Bool || isIntegralType(dtype) ||
                               (isFloatingType(dtype) && !isFloat8Type(dtype));
    if (!has_real_lane) {
        TP_THROW(TypeError, op, ": unsupported dtype");
    }
    return empty_meta(shape_of(self), DType::Bool, self.device());
}

#define META_BOOL_PREDICATE(NAME)                                             \
    Tensor NAME##_kernel(const Tensor& self) { return bool_answer(self, #NAME); }

META_BOOL_PREDICATE(isnan)
META_BOOL_PREDICATE(isinf)
META_BOOL_PREDICATE(isfinite)
META_BOOL_PREDICATE(isneginf)
META_BOOL_PREDICATE(isposinf)
META_BOOL_PREDICATE(signbit)
#undef META_BOOL_PREDICATE

Tensor isreal_kernel(const Tensor& self) {
    return empty_meta(shape_of(self), DType::Bool, self.device());
}

bool is_complex_kernel(const Tensor& self) {
    return isComplexType(self.dtype());
}

DType float_unary_dtype(DType in) {
    if (isComplexType(in)) return toRealValueType(in);
    return isFloatingType(in) ? in : DType::Float32;
}

Tensor angle_kernel(const Tensor& self) {
    if (self.dtype() == DType::Bool) {
        TP_THROW(TypeError, "Unsupported dtype (expected float)");
    }
    return empty_meta(shape_of(self), float_unary_dtype(self.dtype()),
                      self.device());
}

Tensor positive_kernel(const Tensor& self) {
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor entr_kernel(const Tensor& self) {
    return empty_meta(shape_of(self), float_unary_dtype(self.dtype()),
                      self.device());
}

Tensor logit_kernel(const Tensor& self, const std::optional<Scalar>& /*eps*/) {
    return empty_meta(shape_of(self), float_unary_dtype(self.dtype()),
                      self.device());
}

Tensor& logit_inplace_kernel(Tensor& self, const std::optional<Scalar>& /*eps*/) {
    return self;
}

Tensor nan_to_num_kernel(const Tensor& self, const Scalar& /*nan*/,
                         const std::optional<Scalar>& /*posinf*/,
                         const std::optional<Scalar>& /*neginf*/) {
    if (isComplexType(self.dtype())) {
        TP_THROW(TypeError, "nan_to_num: unsupported dtype");
    }
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor& nan_to_num_inplace_kernel(Tensor& self, const Scalar& /*nan*/,
                                  const std::optional<Scalar>& /*posinf*/,
                                  const std::optional<Scalar>& /*neginf*/) {
    if (isComplexType(self.dtype())) {
        TP_THROW(TypeError, "nan_to_num: unsupported dtype");
    }
    return self;
}

// ---------------------------------------------------------------------------
// Linear interpolation: both endpoints take part in the broadcast, and the
// weight adds its own type when it is a tensor.  A whole-numbered promotion
// answers at single precision, and a floating scalar weight lifts it too.
// ---------------------------------------------------------------------------

Tensor lerp_scalar_kernel(const Tensor& self, const Tensor& end,
                          const Scalar& weight) {
    check_all_meta(self, end);
    DType out = promoteTypes(self.dtype(), end.dtype());
    if (weight.isFloatingPoint()) out = promoteTypes(out, DType::Float32);
    if (isIntegralType(out)) out = DType::Float32;
    return empty_meta(broadcast2(self, end), out, self.device());
}

Tensor lerp_tensor_kernel(const Tensor& self, const Tensor& end,
                          const Tensor& weight) {
    check_all_meta(self, end);
    check_all_meta(self, weight);
    DType out = promoteTypes(promoteTypes(self.dtype(), end.dtype()),
                             weight.dtype());
    if (isIntegralType(out)) out = DType::Float32;
    return empty_meta(
        broadcast_shapes(broadcast2(self, end), shape_of(weight)), out,
        self.device());
}

void check_lerp_destination(const Tensor& self,
                            const std::vector<int64_t>& out, const char* op) {
    if (out != shape_of(self)) {
        TP_THROW(RuntimeError, op,
                 ": the broadcast result does not take the destination's shape");
    }
}

Tensor& lerp_scalar_inplace_kernel(Tensor& self, const Tensor& end,
                                   const Scalar& weight) {
    check_lerp_destination(self, broadcast2(self, end), "lerp_");
    (void)weight;
    return self;
}

Tensor& lerp_tensor_inplace_kernel(Tensor& self, const Tensor& end,
                                   const Tensor& weight) {
    check_lerp_destination(self,
                           broadcast_shapes(broadcast2(self, end),
                                            shape_of(weight)),
                           "lerp_");
    return self;
}

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
    m.impl("arange.start", arange_start_stub);
    m.impl("arange.start_step", arange_start_step_opt_stub);
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
    m.impl("uniform_", uniform_inplace_kernel);
    m.impl("normal_", normal_inplace_kernel);
    m.impl("cauchy_", cauchy_inplace_kernel);
    m.impl("exponential_", exponential_inplace_kernel);
    m.impl("geometric_", geometric_inplace_kernel);
    m.impl("log_normal_", log_normal_inplace_kernel);
    m.impl("bernoulli_.float", bernoulli_float_inplace_kernel);
    m.impl("bernoulli_.Tensor", bernoulli_tensor_inplace_kernel);
    m.impl("random_", random_inplace_kernel);
    m.impl("random_.from", random_from_inplace_kernel);
    m.impl("random_.to", random_to_inplace_kernel);

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

    // Pointwise pair maps: modulo, division, sign transfer and the maps that
    // compute at floating precision.
    m.impl("fmod.Tensor", fmod_pair_kernel);
    m.impl("fmod.Scalar", fmod_scalar_kernel);
    m.impl("remainder.Tensor", remainder_pair_kernel);
    m.impl("remainder.Scalar", remainder_scalar_kernel);
    m.impl("remainder.Scalar_Tensor", remainder_scalar_tensor_kernel);
    m.impl("floor_divide", floor_divide_pair_kernel);
    m.impl("floor_divide.Scalar", floor_divide_scalar_kernel);
    m.impl("true_divide.Tensor", true_divide_kernel);
    m.impl("true_divide.Scalar", true_divide_scalar_kernel);
    m.impl("copysign.Tensor", float_pair_kernel);
    m.impl("copysign.Scalar", copysign_scalar_kernel);
    m.impl("nextafter", float_pair_kernel);
    m.impl("hypot", float_pair_kernel);
    m.impl("atan2", float_pair_kernel);
    m.impl("heaviside", heaviside_kernel);
    m.impl("xlogy", float_pair_kernel);
    m.impl("atan2_", pair_inplace_kernel);
    m.impl("heaviside_", pair_inplace_kernel);
    m.impl("hypot_", pair_inplace_kernel);
    m.impl("nextafter_", pair_inplace_kernel);

    // Pointwise unary.
    m.impl("abs", abs_kernel);
    m.impl("absolute", absolute_kernel);
    m.impl("abs_", abs_inplace_kernel);
    m.impl("absolute_", absolute_inplace_kernel);
    m.impl("neg", neg_kernel);
    m.impl("neg_", neg_inplace_kernel);
    m.impl("exp", exp_kernel);
    m.impl("exp2", exp2_kernel);
    m.impl("expm1", expm1_kernel);
    m.impl("exp_", exp_inplace_kernel);
    m.impl("exp2_", exp2_inplace_kernel);
    m.impl("expm1_", expm1_inplace_kernel);
    m.impl("log", log_kernel);
    m.impl("log2", log2_kernel);
    m.impl("log10", log10_kernel);
    m.impl("log1p", log1p_kernel);
    m.impl("log_", log_inplace_kernel);
    m.impl("log2_", log2_inplace_kernel);
    m.impl("log10_", log10_inplace_kernel);
    m.impl("log1p_", log1p_inplace_kernel);
    m.impl("sqrt", sqrt_kernel);
    m.impl("rsqrt", rsqrt_kernel);
    m.impl("sqrt_", sqrt_inplace_kernel);
    m.impl("rsqrt_", rsqrt_inplace_kernel);
    m.impl("tan", tan_kernel);
    m.impl("asin", asin_kernel);
    m.impl("acos", acos_kernel);
    m.impl("atan", atan_kernel);
    m.impl("sinh", sinh_kernel);
    m.impl("cosh", cosh_kernel);
    m.impl("asinh", asinh_kernel);
    m.impl("acosh", acosh_kernel);
    m.impl("atanh", atanh_kernel);
    m.impl("tan_", tan_inplace_kernel);
    m.impl("asin_", asin_inplace_kernel);
    m.impl("acos_", acos_inplace_kernel);
    m.impl("atan_", atan_inplace_kernel);
    m.impl("sinh_", sinh_inplace_kernel);
    m.impl("cosh_", cosh_inplace_kernel);
    m.impl("asinh_", asinh_inplace_kernel);
    m.impl("acosh_", acosh_inplace_kernel);
    m.impl("atanh_", atanh_inplace_kernel);
    m.impl("reciprocal", reciprocal_kernel);
    m.impl("square", square_kernel);
    m.impl("sinc", sinc_kernel);
    m.impl("angle", angle_kernel);
    m.impl("reciprocal_", reciprocal_inplace_kernel);
    m.impl("square_", square_inplace_kernel);
    m.impl("sinc_", sinc_inplace_kernel);
    m.impl("erf", erf_kernel);
    m.impl("erfc", erfc_kernel);
    m.impl("erfinv", erfinv_kernel);
    m.impl("lgamma", lgamma_kernel);
    m.impl("digamma", digamma_kernel);
    m.impl("i0", i0_kernel);
    m.impl("erf_", erf_inplace_kernel);
    m.impl("erfc_", erfc_inplace_kernel);
    m.impl("erfinv_", erfinv_inplace_kernel);
    m.impl("lgamma_", lgamma_inplace_kernel);
    m.impl("digamma_", digamma_inplace_kernel);
    m.impl("i0_", i0_inplace_kernel);
    m.impl("deg2rad", deg2rad_kernel);
    m.impl("rad2deg", rad2deg_kernel);
    m.impl("deg2rad_", deg2rad_inplace_kernel);
    m.impl("rad2deg_", rad2deg_inplace_kernel);
    m.impl("ceil_", ceil_inplace_kernel);
    m.impl("floor_", floor_inplace_kernel);
    m.impl("round_", round_inplace_kernel);
    m.impl("clamp", clamp_kernel);

    // Predicates and the maps that keep the operand's own geometry.
    m.impl("isnan", isnan_kernel);
    m.impl("isinf", isinf_kernel);
    m.impl("isfinite", isfinite_kernel);
    m.impl("isneginf", isneginf_kernel);
    m.impl("isposinf", isposinf_kernel);
    m.impl("signbit", signbit_kernel);
    m.impl("isreal", isreal_kernel);
    m.impl("is_complex", is_complex_kernel);
    m.impl("positive", positive_kernel);
    m.impl("entr", entr_kernel);
    m.impl("logit", logit_kernel);
    m.impl("logit_", logit_inplace_kernel);
    m.impl("nan_to_num", nan_to_num_kernel);
    m.impl("nan_to_num_", nan_to_num_inplace_kernel);

    // Linear interpolation.
    m.impl("lerp", lerp_scalar_kernel);
    m.impl("lerp.Tensor", lerp_tensor_kernel);
    m.impl("lerp_.Scalar", lerp_scalar_inplace_kernel);
    m.impl("lerp_.Tensor", lerp_tensor_inplace_kernel);

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
    m.impl("_aminmax", aminmax_all_kernel);
    m.impl("_aminmax.dim", aminmax_dim_kernel);

    // Matrix products and concatenation.
    m.impl("mm", mm_kernel);
    m.impl("matmul", matmul_kernel);
    m.impl("reshape", reshape_kernel);
    m.impl("bmm", bmm_kernel);
    m.impl("addmm", addmm_kernel);
    m.impl("addmm_", addmm_inplace_kernel);
    m.impl("dot", dot_kernel);
    m.impl("mv", mv_kernel);
    m.impl("inner", inner_kernel);
    m.impl("ger", ger_kernel);
    m.impl("baddbmm", baddbmm_kernel);
    m.impl("baddbmm_", baddbmm_inplace_kernel);
    m.impl("addbmm", addbmm_kernel);
    m.impl("addbmm_", addbmm_inplace_kernel);
    m.impl("addr", addr_kernel);
    m.impl("addr_", addr_inplace_kernel);
    m.impl("addmv", addmv_kernel);
    m.impl("addmv_", addmv_inplace_kernel);
    m.impl("addcmul", addcmul_kernel);
    m.impl("addcmul_", addcmul_inplace_kernel);
    m.impl("addcdiv", addcdiv_kernel);
    m.impl("addcdiv_", addcdiv_inplace_kernel);
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
    m.impl("conv1d", conv1d_kernel);
    m.impl("conv3d", conv3d_kernel);
    m.impl("conv_transpose2d", conv_transpose2d_kernel);
    m.impl("group_norm", group_norm_kernel);
    m.impl("native_group_norm", native_group_norm_kernel);

    // Pointwise activations and roots.
    m.impl("silu", silu_kernel);
    m.impl("cos", cos_kernel);
    m.impl("sin", sin_kernel);
    m.impl("cos_", cos_inplace_kernel);
    m.impl("sin_", sin_inplace_kernel);

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
