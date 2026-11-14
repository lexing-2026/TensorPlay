// Shape-only kernels for the factories and copies on the meta device.
//
// Fill-style factories are empty allocations: the fill value lives in
// element data, which a meta tensor does not carry.  The random fills
// validate their scalar arguments and leave the operand's metadata
// unchanged, since a probability or bound tensor is never read.  A copy
// answers its own geometry; one that would move a meta tensor to another
// device is refused outright, because moving reads the elements it
// transports.
#include "MetaCommon.h"

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

#include <algorithm>
#include <cctype>
#include <cmath>
#include <initializer_list>
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
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Windows.  A window factory answers a one-dimensional buffer whose length
// is the requested window length.  The cosine or polynomial samples live in
// element data, which meta tensors do not carry, but the length and the
// requested type are shape questions and still get policed here.
// ---------------------------------------------------------------------------
Tensor window_meta(int64_t window_length, std::optional<DType> dtype_opt,
                   const char* name) {
    if (window_length < 0) {
        TP_THROW(ValueError, name, ": window_length must be non-negative");
    }
    DType dt = dtype_opt.value_or(DType::Float32);
    if (dt == DType::Undefined) dt = DType::Float32;
    if (dt != DType::Float32 && dt != DType::Float64) {
        TP_THROW(NotImplementedError, name, ": only float32/float64 windows are supported");
    }
    return empty_meta({window_length}, dt, Device(DeviceType::Meta));
}

Tensor bartlett_window_kernel(int64_t window_length, bool /*periodic*/,
                              std::optional<DType> dtype) {
    return window_meta(window_length, dtype, "bartlett_window");
}

Tensor blackman_window_kernel(int64_t window_length, bool /*periodic*/,
                              std::optional<DType> dtype) {
    return window_meta(window_length, dtype, "blackman_window");
}

Tensor hann_window_kernel(int64_t window_length, bool /*periodic*/,
                          std::optional<DType> dtype) {
    return window_meta(window_length, dtype, "hann_window");
}

Tensor hamming_window_kernel(int64_t window_length, bool /*periodic*/,
                             double /*alpha*/, double /*beta*/,
                             std::optional<DType> dtype) {
    return window_meta(window_length, dtype, "hamming_window");
}

// Bernoulli draws land in the operand's own geometry; the per-element
// probabilities live in element data and are never read here.
Tensor bernoulli_kernel(const Tensor& self, std::optional<Generator> /*generator*/) {
    check_all_meta(self, self);
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor bernoulli_p_kernel(const Tensor& self, double /*p*/,
                          std::optional<Generator> /*generator*/) {
    check_all_meta(self, self);
    return empty_meta(shape_of(self), self.dtype(), self.device());
}
} // anonymous namespace
TENSORPLAY_LIBRARY_IMPL(Meta, MetaFactory) {
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
    m.impl("bartlett_window", bartlett_window_kernel);
    m.impl("blackman_window", blackman_window_kernel);
    m.impl("hann_window", hann_window_kernel);
    m.impl("hamming_window", hamming_window_kernel);
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
    m.impl("bernoulli", bernoulli_kernel);
    m.impl("bernoulli.p", bernoulli_p_kernel);
    m.impl("bernoulli_.float", bernoulli_float_inplace_kernel);
    m.impl("bernoulli_.Tensor", bernoulli_tensor_inplace_kernel);
    m.impl("random_", random_inplace_kernel);
    m.impl("random_.from", random_from_inplace_kernel);
    m.impl("random_.to", random_to_inplace_kernel);
    m.impl("copy_", copy_kernel);
}

} // namespace meta
} // namespace tensorplay
