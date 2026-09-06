// Hand-written overload wiring: entries whose argument shapes differ from
// any registered sibling in ways the mechanical matcher rejects, but whose
// semantics are a plain forward to an already-registered kernel.
#include "Tensor.h"
#include "SparseKernels.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Context.h"
#include "Quantizer.h"
#include "Scalar.h"
#include "TypePromotion.h"
#include "CompositeCommon.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "cpu/Lapack.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

// A scalar taking part in tensor arithmetic materializes with its natural
// dtype promoted against the reference tensor, matching wrapped-number
// promotion for the scalar overloads implemented here.
Tensor scalar_like(const Scalar& s, const Tensor& ref) {
    const DType prom = promoteTypes(scalar_natural_dtype(s), ref.dtype());
    return ops::scalar_tensor(s, prom, ref.device());
}

Tensor quantile_scalar_tensor(const Tensor& self, double q) {
    if (!(q >= 0.0 && q <= 1.0)) {
        TP_THROW(ValueError,
                 "quantile() q must be in the range [0, 1] but got ", q);
    }
    return ops::scalar_tensor(Scalar(q), self.dtype(), self.device());
}

Tensor& write_reduction_out(const char* op, Tensor value, Tensor& out) {
    if (!out.defined()) {
        out = std::move(value);
        return out;
    }
    if (out.device() != value.device()) {
        TP_THROW(DeviceMismatchError,
                 op, ": output device must match input device");
    }
    if (out.dtype() != value.dtype()) {
        if (!canCast(value.dtype(), out.dtype())) {
            TP_THROW(TypeError,
                     op, ": result type cannot be cast to output type");
        }
        value = value.to(out.dtype());
    }
    const auto target = static_cast<std::vector<int64_t>>(value.shape());
    if (static_cast<std::vector<int64_t>>(out.shape()) != target) {
        out.resize_(target);
    }
    out.copy_(value);
    return out;
}

Tensor& write_exact_out(const char* op, Tensor value, Tensor& out) {
    if (!out.defined()) {
        out = std::move(value);
        return out;
    }
    if (out.dtype() != value.dtype()) {
        TP_THROW(TypeError, op, ": output dtype must match result dtype");
    }
    if (out.device() != value.device()) {
        TP_THROW(DeviceMismatchError,
                 op, ": output device must match input device");
    }
    const auto target = static_cast<std::vector<int64_t>>(value.shape());
    if (static_cast<std::vector<int64_t>>(out.shape()) != target) {
        out.resize_(target);
    }
    out.copy_(value);
    return out;
}

} // namespace

// ---- norm family -----------------------------------------------------------
Tensor norm_scalar_native(const Tensor& self, const Scalar& p) {
    if (p.isComplex()) {
        TP_THROW(NotImplementedError, "norm with a complex exponent is not supported");
    }
    return ops::norm(self, p.toDouble());
}

Tensor norm_scalar_opt_dim_native(const Tensor& self, const std::optional<Scalar>& p,
                                  const std::vector<int64_t>& dim, bool keepdim) {
    return ops::norm(self, dim, p.has_value() ? p->toDouble() : 2.0, keepdim);
}

Tensor norm_scalar_opt_dtype_native(const Tensor& self, const std::optional<Scalar>& p,
                                    DType dtype) {
    if (p.has_value()) {
        Tensor r = ops::norm(self, p->toDouble());
        return r.to(dtype);
    }
    Tensor r = ops::norm(self, 2.0);
    return r.to(dtype);
}

Tensor norm_scalar_opt_dim_dtype_native(const Tensor& self, const std::optional<Scalar>& p,
                                        const std::vector<int64_t>& dim, bool keepdim,
                                        DType dtype) {
    Tensor r = ops::norm(self, dim, p.has_value() ? p->toDouble() : 2.0, keepdim);
    return r.to(dtype);
}

Tensor& norm_out_native(const Tensor& self, const std::optional<Scalar>& p,
                        const std::vector<int64_t>& dim, bool keepdim, Tensor& out) {
    return write_reduction_out(
        "norm", ops::norm(self, dim, p.has_value() ? p->toDouble() : 2.0,
                           keepdim),
        out);
}

Tensor& norm_dtype_out_native(const Tensor& self, const std::optional<Scalar>& p,
                              const std::vector<int64_t>& dim, bool keepdim, DType dtype,
                              Tensor& out) {
    return write_reduction_out(
        "norm", ops::norm(self, dim, p.has_value() ? p->toDouble() : 2.0,
                           keepdim)
                    .to(dtype),
        out);
}

// ---- reductions ------------------------------------------------------------
Tensor prod_dim_int_native(const Tensor& self, int64_t dim, bool keepdim,
                           std::optional<DType> dtype) {
    // Forward to the registered int-list reduction; calling the int-dim
    // overload here would dispatch back to this very kernel.
    Tensor r = ops::prod(self, std::vector<int64_t>{dim}, keepdim);
    if (dtype.has_value() && r.dtype() != *dtype) r = r.to(*dtype);
    return r;
}

Tensor& prod_int_out_native(const Tensor& self, int64_t dim, bool keepdim,
                            std::optional<DType> dtype, Tensor& out) {
    return write_reduction_out("prod", prod_dim_int_native(self, dim, keepdim,
                                                            dtype),
                               out);
}

Tensor std_correction_native(const Tensor& self, const std::optional<std::vector<int64_t>>& dim,
                             const std::optional<Scalar>& correction, bool keepdim) {
    if (dim.has_value()) {
        const int64_t c = correction.has_value() ? correction->to<int64_t>() : 1;
        return ops::std(self, *dim, c, keepdim);
    }
    return ops::std(self, correction.has_value() ? correction->to<int64_t>() : 1);
}

Tensor& std_correction_out_native(const Tensor& self,
                                  const std::optional<std::vector<int64_t>>& dim,
                                  const std::optional<Scalar>& correction, bool keepdim,
                                  Tensor& out) {
    return write_reduction_out(
        "std", std_correction_native(self, dim, correction, keepdim), out);
}

Tensor& std_out_native(const Tensor& self, const std::optional<std::vector<int64_t>>& dim,
                       bool unbiased, bool keepdim, Tensor& out) {
    const std::optional<Scalar> correction = Scalar(int64_t(unbiased ? 1 : 0));
    return write_reduction_out(
        "std", std_correction_native(self, dim, correction, keepdim), out);
}

Tensor var_correction_native(const Tensor& self, const std::optional<std::vector<int64_t>>& dim,
                             const std::optional<Scalar>& correction, bool keepdim) {
    if (dim.has_value()) {
        const int64_t c = correction.has_value() ? correction->to<int64_t>() : 1;
        return ops::var(self, *dim, c, keepdim);
    }
    return ops::var(self, correction.has_value() ? correction->to<int64_t>() : 1);
}

Tensor& var_correction_out_native(const Tensor& self,
                                  const std::optional<std::vector<int64_t>>& dim,
                                  const std::optional<Scalar>& correction, bool keepdim,
                                  Tensor& out) {
    return write_reduction_out(
        "var", var_correction_native(self, dim, correction, keepdim), out);
}

Tensor& var_out_native(const Tensor& self, const std::optional<std::vector<int64_t>>& dim,
                       bool unbiased, bool keepdim, Tensor& out) {
    const std::optional<Scalar> correction = Scalar(int64_t(unbiased ? 1 : 0));
    return write_reduction_out(
        "var", var_correction_native(self, dim, correction, keepdim), out);
}

std::tuple<Tensor, Tensor> std_mean_correction_native(
        const Tensor& self, const std::optional<std::vector<int64_t>>& dim,
        const std::optional<Scalar>& correction, bool keepdim) {
    Tensor s = std_correction_native(self, dim, correction, keepdim);
    Tensor v = var_correction_native(self, dim, correction, keepdim);
    return {s, v};
}

std::tuple<Tensor, Tensor> var_mean_correction_native(
        const Tensor& self, const std::optional<std::vector<int64_t>>& dim,
        const std::optional<Scalar>& correction, bool keepdim) {
    return std_mean_correction_native(self, dim, correction, keepdim);
}

std::tuple<Tensor, Tensor> median_dim_native(const Tensor& self, int64_t dim, bool keepdim) {
    // the lower of the two central order statistics, i.e. the
    // ((n + 1) / 2)-th smallest element along the dimension
    const int64_t n = self.size(wrap_dim(dim, self.dim()));
    return ops::kthvalue(self, (n + 1) / 2, dim, keepdim);
}

std::tuple<Tensor, Tensor> median_dim_values_native(const Tensor& self, int64_t dim,
                                                     bool keepdim, Tensor& values,
                                                     Tensor& indices) {
    auto r = median_dim_native(self, dim, keepdim);
    write_exact_out("median", std::get<0>(r), values);
    write_exact_out("median", std::get<1>(r), indices);
    return {values, indices};
}

// ---- sort / topk ------------------------------------------------------------
// `stable` requests that equal elements keep their input order; it does not
// forbid an implementation from always doing so.  Every backend sort here
// breaks ties on the source index (the CPU comparator's final `a.second <
// b.second`, the radix passes' index key), so the flag selects between two
// identical orderings and only has to be accepted, not acted on.
std::tuple<Tensor, Tensor> sort_stable_native(const Tensor& self,
                                              std::optional<bool> stable, int64_t dim,
                                              bool descending) {
    (void)stable;
    return ops::sort(self, dim, descending);
}

std::tuple<Tensor, Tensor> sort_values_stable_native(const Tensor& self,
                                                      std::optional<bool> stable,
                                                      int64_t dim, bool descending,
                                                      Tensor& values, Tensor& indices) {
    auto r = sort_stable_native(self, stable, dim, descending);
    write_exact_out("sort", std::get<0>(r), values);
    write_exact_out("sort", std::get<1>(r), indices);
    return {values, indices};
}

Tensor argsort_stable_native(const Tensor& self, bool stable, int64_t dim, bool descending) {
    (void)stable;
    return ops::argsort(self, dim, descending);
}

Tensor& argsort_stable_out_native(const Tensor& self, bool stable, int64_t dim,
                                  bool descending, Tensor& out) {
    return write_exact_out("argsort", argsort_stable_native(self, stable, dim,
                                                              descending),
                           out);
}

std::tuple<Tensor, Tensor> topk_values_native(const Tensor& self, int64_t k, int64_t dim,
                                               bool largest, bool sorted, Tensor& values,
                                               Tensor& indices) {
    auto r = ops::topk(self, k, dim, largest, sorted);
    values = std::get<0>(r);
    indices = std::get<1>(r);
    return {values, indices};
}

Tensor& logsumexp_out_native(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim,
                             Tensor& out) {
    // log-sum-exp over disjoint dimension sets composes, but each reduction
    // removes an axis, so the dims are consumed from the highest index down
    // to keep the remaining indices valid.
    Tensor r = self;
    std::vector<int64_t> dims = dim;
    std::sort(dims.begin(), dims.end(), std::greater<int64_t>());
    for (int64_t d : dims) {
        r = ops::logsumexp(r, d, keepdim);
    }
    return write_reduction_out("logsumexp", std::move(r), out);
}

// ---- shape ops --------------------------------------------------------------
Tensor movedim_int_native(const Tensor& self, int64_t source, int64_t destination) {
    return ops::movedim(self, std::vector<int64_t>{source},
                        std::vector<int64_t>{destination});
}

Tensor narrow_tensor_native(const Tensor& self, int64_t dim, const Tensor& start,
                            int64_t length) {
    return ops::narrow(self, dim, start.item().to<int64_t>(), length);
}

Tensor max_other_native(const Tensor& self, const Tensor& other) {
    return ops::maximum(self, other);
}

std::tuple<Tensor, Tensor> aminmax_out_native(const Tensor& self,
                                               std::optional<int64_t> dim,
                                               bool keepdim, Tensor& min, Tensor& max) {
    std::vector<int64_t> dims;
    if (dim.has_value()) dims.push_back(*dim);
    auto r = ops::aminmax(self, dims, keepdim);
    write_exact_out("aminmax", std::get<0>(r), min);
    write_exact_out("aminmax", std::get<1>(r), max);
    return {min, max};
}

Tensor round_decimals_native(const Tensor& self, int64_t decimals) {
    Tensor scaled = self * Scalar(std::pow(10.0, static_cast<double>(decimals)));
    Tensor r = ops::round(scaled);
    return r * Scalar(std::pow(10.0, static_cast<double>(-decimals)));
}

Tensor& round__decimals_native(Tensor& self, int64_t decimals) {
    ops::copy_(self, round_decimals_native(self, decimals));
    return self;
}

Tensor& nan_to_num_out_native(const Tensor& self, std::optional<double> nan,
                              std::optional<double> posinf,
                              std::optional<double> neginf, Tensor& out) {
    const std::optional<Scalar> posinf_value =
        posinf.has_value() ? std::optional<Scalar>(Scalar(*posinf)) : std::nullopt;
    const std::optional<Scalar> neginf_value =
        neginf.has_value() ? std::optional<Scalar>(Scalar(*neginf)) : std::nullopt;
    return write_exact_out(
        "nan_to_num",
        ops::nan_to_num(self, Scalar(nan.value_or(0.0)), posinf_value,
                        neginf_value),
        out);
}

Tensor& nanmean_out_native(const Tensor& self, const std::optional<std::vector<int64_t>>& dim,
                           bool keepdim, std::optional<DType> dtype, Tensor& out) {
    std::optional<DType> result_dtype = dtype;
    if (!result_dtype.has_value() && out.defined()) result_dtype = out.dtype();
    Tensor result;
    if (dim.has_value() && dim->size() == 1) {
        result = ops::nanmean(self, (*dim)[0], keepdim, result_dtype);
    } else if (!dim.has_value()) {
        result = ops::nanmean(self, std::nullopt, keepdim, result_dtype);
    } else {
        TP_THROW(NotImplementedError, "nanmean.out with multiple dims is not supported");
    }
    if (!out.defined()) {
        out = result;
        return out;
    }
    if (out.dtype() != result.dtype()) {
        TP_THROW(TypeError, "nanmean(): provided dtype must match dtype of out");
    }
    if (out.device() != result.device()) {
        TP_THROW(DeviceMismatchError,
                 "nanmean(): out tensor must be on the same device as the input");
    }
    out.resize_(static_cast<std::vector<int64_t>>(result.shape()));
    out.copy_(result);
    return out;
}

// ---- numeric helpers --------------------------------------------------------
Tensor trapezoid_dx_native(const Tensor& y, const Scalar& dx, int64_t dim) {
    return ops::trapezoid(y, std::nullopt, dx, dim);
}

Tensor trapezoid_x_native(const Tensor& y, const Tensor& x, int64_t dim) {
    return ops::trapezoid(y, x, Scalar(1), dim);
}

Tensor cumulative_trapezoid_dx_native(const Tensor& y, const Scalar& dx, int64_t dim) {
    return ops::cumulative_trapezoid(y, std::nullopt, dx, dim);
}

Tensor cumulative_trapezoid_x_native(const Tensor& y, const Tensor& x, int64_t dim) {
    return ops::cumulative_trapezoid(y, x, Scalar(1), dim);
}

// Every gradient overload funnels into the tensor-spacing base kernel, so the
// scalar spacings of the variants below are materialized as 0-d tensors before
// delegating. Calling a scalar-spacing overload from here would dispatch back
// into the very kernels registered just below.
static std::vector<Tensor> gradient_scalar_spacings(const std::vector<Scalar>& spacing,
                                                    const Tensor& ref) {
    std::vector<Tensor> sp;
    sp.reserve(spacing.size());
    for (const Scalar& s : spacing) {
        sp.push_back(ops::scalar_tensor(s, DType::Undefined, ref.device()));
    }
    return sp;
}

std::vector<Tensor> gradient_scalarint_native(const Tensor& self,
                                              const std::optional<Scalar>& spacing,
                                              std::optional<int64_t> dim,
                                              int64_t edge_order) {
    const std::vector<int64_t> dims{dim.has_value() ? *dim : self.dim() - 1};
    return ops::gradient(self,
                         gradient_scalar_spacings({spacing.value_or(Scalar(1))}, self),
                         dims, edge_order);
}

std::vector<Tensor> gradient_scalararray_native(const Tensor& self, const Scalar& spacing,
                                                const std::vector<int64_t>& dim,
                                                int64_t edge_order) {
    return ops::gradient(self,
                         gradient_scalar_spacings(std::vector<Scalar>(dim.size(), spacing), self),
                         dim, edge_order);
}

std::vector<Tensor> gradient_array_native(const Tensor& self, const std::vector<int64_t>& dim,
                                          int64_t edge_order) {
    return ops::gradient(self,
                         gradient_scalar_spacings(std::vector<Scalar>(dim.size(), Scalar(1)), self),
                         dim, edge_order);
}

std::vector<Tensor> gradient_scalarrayint_native(const Tensor& self,
                                                 const std::vector<Scalar>& spacing,
                                                 std::optional<int64_t> dim,
                                                 int64_t edge_order) {
    const std::vector<int64_t> dims{dim.has_value() ? *dim : self.dim() - 1};
    return ops::gradient(self, gradient_scalar_spacings(spacing, self), dims, edge_order);
}

std::vector<Tensor> gradient_scalarrayarray_native(const Tensor& self,
                                                   const std::vector<Scalar>& spacing,
                                                   const std::vector<int64_t>& dim,
                                                   int64_t edge_order) {
    return ops::gradient(self, gradient_scalar_spacings(spacing, self), dim, edge_order);
}

std::vector<Tensor> gradient_tensorarrayint_native(const Tensor& self,
                                                   const std::vector<Tensor>& spacing,
                                                   std::optional<int64_t> dim,
                                                   int64_t edge_order) {
    const std::vector<int64_t> dims{dim.has_value() ? *dim : self.dim() - 1};
    return ops::gradient(self, spacing, dims, edge_order);
}

Tensor quantile_scalar_native(const Tensor& self, double q, std::optional<int64_t> dim,
                              bool keepdim, const std::string& interpolation) {
    Tensor qv = quantile_scalar_tensor(self, q);
    return ops::quantile(self, qv, dim, keepdim, interpolation);
}

Tensor nanquantile_scalar_native(const Tensor& self, double q,
                                 std::optional<int64_t> dim, bool keepdim,
                                 const std::string& interpolation) {
    Tensor qv = quantile_scalar_tensor(self, q);
    return ops::nanquantile(self, qv, dim, keepdim, interpolation);
}

// ---- ormqr -----------------------------------------------------------------
// Applies the orthogonal matrix Q carried by Householder reflectors (self,
// input2=tau) to the rows or columns of input3: C := op(Q)·C when left, else
// C·op(Q), with op = transpose ? Q^T : Q.  Q has order q = the matched
// dimension of input3, while only k reflectors are given; the product is
// materialized explicitly by expanding the reflector block into a (q x q)
// column-major buffer whose remaining columns are zero and reducing with
// orgqr, then applied through the registered batched matmul.
template <typename T>
static void row_to_col_major(const T* src, int64_t rows, int64_t cols, T* dst) {
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c)
            dst[c * rows + r] = src[r * cols + c];
}

template <typename T>
static void col_to_row_major(const T* src, int64_t rows, int64_t cols, T* dst) {
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c)
            dst[r * cols + c] = src[c * rows + r];
}

// ---- linalg_vdot -----------------------------------------------------------
// Flattened 1-D dot product with the conjugate applied to self: sum(conj(x)
// * y) over the whole tensor regardless of the input rank.
Tensor linalg_vdot_composite(const Tensor& self, const Tensor& other) {
    if (self.shape() != other.shape()) {
        TP_THROW(RuntimeError,
                 "linalg.vdot: both tensors must have the same shape, got ",
                 self.shape(), " and ", other.shape());
    }
    Tensor x = ops::conj(self.reshape({-1}));
    Tensor y = other.reshape({-1});
    return ops::sum(x * y);
}

Tensor ormqr_composite(const Tensor& self, const Tensor& input2, const Tensor& input3,
                       bool left, bool transpose) {
    using namespace cpu;
    require_lapack("ormqr");
    if (self.dtype() != input2.dtype() || self.dtype() != input3.dtype()) {
        TP_THROW(RuntimeError, "ormqr(): self, tau and other must share the same dtype");
    }
    const int64_t m = self.size(-2);
    const int64_t k = self.size(-1);
    if (input2.size(-1) != k) {
        TP_THROW(RuntimeError, "ormqr(): tau must have shape (..., k) matching the ",
                 k, " reflectors of self, but got ", input2.size(-1));
    }
    const int64_t q = left ? input3.size(-2) : input3.size(-1);
    if (q != m) {
        TP_THROW(RuntimeError, "ormqr(): reflector matrix and other disagree on the "
                 "matched dimension: got ", m, " vs ", q);
    }
    if (k == 0 || q == 0) {
        return input3.clone();
    }
    if (self.dtype() != DType::Float32 && self.dtype() != DType::Float64) {
        TP_THROW(RuntimeError, "ormqr(): only float32 and float64 reflectors are "
                 "supported on the CPU backend");
    }
    const Tensor self_c = self.contiguous();
    const Tensor tau_c = input2.contiguous();
    const Tensor other_c = input3.contiguous();
    const int64_t self_plane = m * k;
    const int64_t tau_plane = k;
    const int64_t crows = input3.size(-2);
    const int64_t ccols = input3.size(-1);
    const int64_t other_plane = crows * ccols;
    const int64_t batch = self.numel() / self_plane;
    const DType dt = self.dtype();

    Tensor q_full = Tensor::empty(std::vector<int64_t>{batch, q, q}, dt, self.device());
    auto run_one_dtype = [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const T* sbase = self_c.data_ptr<T>();
        const T* tbase = tau_c.data_ptr<T>();
        T* qbase = q_full.data_ptr<T>();
        for (int64_t i = 0; i < batch; ++i) {
            const T* refl = sbase + i * self_plane;
            const T* tau = tbase + i * tau_plane;
            T* qrow = qbase + i * q * q;
            {
                // Expand the k reflector columns into a (q x q) column-major
                // buffer (extra columns zero) and reduce with orgqr, then
                // convert the resulting Q back to row-major.
                std::vector<T> a(static_cast<size_t>(q * q), T(0));
                row_to_col_major<T>(refl, q, k, a.data());
                int64_t lwork = -1;
                std::vector<T> work(1);
                if constexpr (std::is_same_v<T, float>) {
                    lapack_sorgqr(q, q, k, a.data(), q, tau, work.data(), lwork);
                    lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0]));
                    work.resize(static_cast<size_t>(lwork));
                    lapack_sorgqr(q, q, k, a.data(), q, tau, work.data(), lwork);
                } else {
                    lapack_dorgqr(q, q, k, a.data(), q, tau, work.data(), lwork);
                    lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0]));
                    work.resize(static_cast<size_t>(lwork));
                    lapack_dorgqr(q, q, k, a.data(), q, tau, work.data(), lwork);
                }
                col_to_row_major<T>(a.data(), q, q, qrow);
            }
        }
    };
    if (dt == DType::Float32) run_one_dtype(static_cast<float*>(nullptr));
    else run_one_dtype(static_cast<double*>(nullptr));

    const std::vector<int64_t> qshape{batch, q, q};
    Tensor q_t = q_full.view(qshape);
    Tensor qt = ops::transpose(q_t, -2, -1);
    if (left) {
        return transpose ? ops::matmul(qt, other_c) : ops::matmul(q_t, other_c);
    }
    return transpose ? ops::matmul(other_c, qt) : ops::matmul(other_c, q_t);
}

// ---- geqrf -----------------------------------------------------------------
// Raw Householder factorization A = Q·R.  The reflector factors stay in the
// lower triangle of the output and the elementary reflector scales in tau,
// exactly the form consumed by orgqr/ormqr.  Batched, real dtypes only.
std::tuple<Tensor, Tensor> geqrf_composite(const Tensor& self) {
    using namespace cpu;
    require_lapack("geqrf");
    if (self.dtype() != DType::Float32 && self.dtype() != DType::Float64) {
        TP_THROW(RuntimeError, "geqrf(): only float32 and float64 matrices are "
                 "supported on the CPU backend");
    }
    const int64_t m = self.size(-2);
    const int64_t n = self.size(-1);
    const int64_t k = std::min(m, n);
    const Tensor self_c = self.contiguous();
    const int64_t plane = m * n;
    const int64_t batch = plane == 0 ? 0 : self.numel() / plane;
    // LAPACK works on column-major buffers: allocate with swapped dims and
    // transpose so the physical layout matches, then copy logically.
    std::vector<int64_t> phys(self_c.dim());
    for (int64_t i = 0; i < self_c.dim(); ++i) phys[i] = self_c.size(i);
    std::swap(phys[self_c.dim() - 2], phys[self_c.dim() - 1]);
    Tensor a = Tensor::empty(phys, self.dtype(), self.device()).transpose(-2, -1);
    a.copy_(self_c);
    std::vector<int64_t> tau_shape(self_c.dim() - 1);
    for (int64_t i = 0; i < self_c.dim() - 2; ++i) tau_shape[i] = self_c.size(i);
    if (self_c.dim() >= 2) tau_shape.back() = k;
    Tensor tau = Tensor::empty(tau_shape, self.dtype(), self.device());
    auto run_one_dtype = [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        T* abase = a.data_ptr<T>();
        T* tbase = tau.data_ptr<T>();
        int64_t lwork = -1;
        std::vector<T> work(1);
        if constexpr (std::is_same_v<T, float>) {
            lapack_sgeqrf(m, n, abase, std::max<int64_t>(1, m), tbase, work.data(), lwork);
            lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0]));
            work.resize(static_cast<size_t>(lwork));
            for (int64_t i = 0; i < batch; ++i) {
                lapack_sgeqrf(m, n, abase + i * plane, std::max<int64_t>(1, m),
                              tbase + i * k, work.data(), lwork);
            }
        } else {
            lapack_dgeqrf(m, n, abase, std::max<int64_t>(1, m), tbase, work.data(), lwork);
            lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0]));
            work.resize(static_cast<size_t>(lwork));
            for (int64_t i = 0; i < batch; ++i) {
                lapack_dgeqrf(m, n, abase + i * plane, std::max<int64_t>(1, m),
                              tbase + i * k, work.data(), lwork);
            }
        }
    };
    if (self.dtype() == DType::Float32) run_one_dtype(static_cast<float*>(nullptr));
    else run_one_dtype(static_cast<double*>(nullptr));
    return std::tuple<Tensor, Tensor>(std::move(a), std::move(tau));
}

// ---- matmul dtype overloads --------------------------------------------------
Tensor mm_dtype_native(const Tensor& self, const Tensor& mat2, DType out_dtype) {
    return ops::mm(self, mat2).to(out_dtype);
}

Tensor& mm_dtype_out_native(const Tensor& self, const Tensor& mat2, DType out_dtype,
                            Tensor& out) {
    return write_exact_out("mm", mm_dtype_native(self, mat2, out_dtype), out);
}

Tensor addmm_dtype_native(const Tensor& self, const Tensor& mat1, const Tensor& mat2,
                          DType out_dtype, const Scalar& beta, const Scalar& alpha) {
    return ops::addmm(self, mat1, mat2, beta, alpha).to(out_dtype);
}

Tensor& addmm_dtype_out_native(const Tensor& self, const Tensor& mat1, const Tensor& mat2,
                               DType out_dtype, const Scalar& beta, const Scalar& alpha,
                               Tensor& out) {
    return write_exact_out(
        "addmm", addmm_dtype_native(self, mat1, mat2, out_dtype, beta, alpha),
        out);
}

Tensor bmm_dtype_native(const Tensor& self, const Tensor& mat2, DType out_dtype) {
    return ops::bmm(self, mat2).to(out_dtype);
}

Tensor& bmm_dtype_out_native(const Tensor& self, const Tensor& mat2, DType out_dtype,
                             Tensor& out) {
    return write_exact_out("bmm", bmm_dtype_native(self, mat2, out_dtype), out);
}

Tensor baddbmm_dtype_native(const Tensor& self, const Tensor& batch1, const Tensor& batch2,
                            DType out_dtype, const Scalar& beta, const Scalar& alpha) {
    return ops::baddbmm(self, batch1, batch2, beta, alpha).to(out_dtype);
}

Tensor& baddbmm_dtype_out_native(const Tensor& self, const Tensor& batch1, const Tensor& batch2,
                                 DType out_dtype, const Scalar& beta, const Scalar& alpha,
                                 Tensor& out) {
    return write_exact_out(
        "baddbmm",
        baddbmm_dtype_native(self, batch1, batch2, out_dtype, beta, alpha), out);
}

// ---- conv padding overloads ---------------------------------------------------
namespace {

std::vector<int64_t> padding_from_mode(const std::string& padding, int64_t k) {
    if (padding == "same" || padding == "valid") {
        return std::vector<int64_t>(2 * k, 0);
    }
    TP_THROW(RuntimeError, "conv: unsupported padding mode ", padding);
}

} // namespace

Tensor conv1d_padding_native(const Tensor& input, const Tensor& weight,
                             const std::optional<Tensor>& bias,
                             const std::vector<int64_t>& stride, const std::string& padding,
                             const std::vector<int64_t>& dilation, int64_t groups) {
    const auto pad = padding_from_mode(padding, 1);
    return ops::conv1d(input, weight, bias, stride, pad, dilation, groups);
}

Tensor conv2d_padding_native(const Tensor& input, const Tensor& weight,
                             const std::optional<Tensor>& bias,
                             const std::vector<int64_t>& stride, const std::string& padding,
                             const std::vector<int64_t>& dilation, int64_t groups) {
    const auto pad = padding_from_mode(padding, 2);
    return ops::conv2d(input, weight, bias, stride, pad, dilation, groups);
}

Tensor conv3d_padding_native(const Tensor& input, const Tensor& weight,
                             const std::optional<Tensor>& bias,
                             const std::vector<int64_t>& stride, const std::string& padding,
                             const std::vector<int64_t>& dilation, int64_t groups) {
    const auto pad = padding_from_mode(padding, 3);
    return ops::conv3d(input, weight, bias, stride, pad, dilation, groups);
}

Tensor _convolution_deprecated_native(const Tensor& input, const Tensor& weight,
                                      const std::optional<Tensor>& bias,
                                      const std::vector<int64_t>& stride,
                                      const std::vector<int64_t>& padding,
                                      const std::vector<int64_t>& dilation, bool transposed,
                                      const std::vector<int64_t>& output_padding,
                                      int64_t groups, bool benchmark, bool deterministic,
                                      bool cudnn_enabled) {
    (void)benchmark; (void)deterministic; (void)cudnn_enabled;
    if (transposed) {
        return ops::conv_transpose2d(input, weight, bias, stride, padding, output_padding,
                                     groups, dilation);
    }
    const int64_t k = static_cast<int64_t>(weight.dim()) - 2;
    if (k == 1) return ops::conv1d(input, weight, bias, stride, padding, dilation, groups);
    if (k == 3) return ops::conv3d(input, weight, bias, stride, padding, dilation, groups);
    return ops::conv2d(input, weight, bias, stride, padding, dilation, groups);
}

// ---- pooling out/backward overloads --------------------------------------------
Tensor& adaptive_max_pool2d_backward_gi_native(const Tensor& grad_output, const Tensor& self,
                                               const Tensor& indices, Tensor& grad_input) {
    (void)indices;
    return write_exact_out("adaptive_max_pool2d_backward",
                           ops::adaptive_max_pool2d_backward(grad_output, self),
                           grad_input);
}

Tensor& adaptive_max_pool3d_backward_gi_native(const Tensor& grad_output, const Tensor& self,
                                               const Tensor& indices, Tensor& grad_input) {
    (void)indices;
    return write_exact_out("adaptive_max_pool3d_backward",
                           ops::adaptive_max_pool3d_backward(grad_output, self),
                           grad_input);
}

Tensor& max_pool2d_with_indices_backward_gi_native(const Tensor& grad_output,
                                                   const Tensor& self,
                                                   const std::vector<int64_t>& kernel_size,
                                                   const std::vector<int64_t>& stride,
                                                   const std::vector<int64_t>& padding,
                                                   const std::vector<int64_t>& dilation,
                                                   bool ceil_mode, const Tensor& indices,
                                                   Tensor& grad_input) {
    grad_input = ops::max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride,
                                                       padding, dilation, ceil_mode, indices);
    return grad_input;
}

Tensor& max_pool3d_with_indices_backward_gi_native(const Tensor& grad_output,
                                                   const Tensor& self,
                                                   const std::vector<int64_t>& kernel_size,
                                                   const std::vector<int64_t>& stride,
                                                   const std::vector<int64_t>& padding,
                                                   const std::vector<int64_t>& dilation,
                                                   bool ceil_mode, const Tensor& indices,
                                                   Tensor& grad_input) {
    grad_input = ops::max_pool3d_with_indices_backward(grad_output, self, kernel_size, stride,
                                                       padding, dilation, ceil_mode, indices);
    return grad_input;
}

// ---- loss out overloads ---------------------------------------------------------
Tensor& nll_loss_out_native(const Tensor& self, const Tensor& target,
                            const std::optional<Tensor>& weight, int64_t reduction,
                            int64_t ignore_index, Tensor& out) {
    auto r = ops::nll_loss(self, target, weight, reduction, ignore_index);
    return write_exact_out("nll_loss", std::get<0>(r), out);
}

Tensor& nll_loss2d_out_native(const Tensor& self, const Tensor& target,
                              const std::optional<Tensor>& weight, int64_t reduction,
                              int64_t ignore_index, Tensor& out) {
    auto r = ops::nll_loss2d(self, target, weight, reduction, ignore_index);
    return write_exact_out("nll_loss2d", std::get<0>(r), out);
}

// ---- rnn dispatcher overloads ----------------------------------------------------
std::tuple<Tensor, Tensor> rnn_input_overload(
        int kind, const Tensor& input, const Tensor& hx, const std::vector<Tensor>& params,
        bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional,
        bool batch_first) {
    if (kind == 1) {
        return ops::gru(input, std::vector<Tensor>{hx}, params, has_biases, num_layers,
                        static_cast<float>(dropout), train, bidirectional, batch_first);
    }
    if (kind == 2) {
        return ops::rnn_tanh(input, std::vector<Tensor>{hx}, params, has_biases, num_layers,
                             static_cast<float>(dropout), train, bidirectional, batch_first);
    }
    return ops::rnn_relu(input, std::vector<Tensor>{hx}, params, has_biases, num_layers,
                         static_cast<float>(dropout), train, bidirectional, batch_first);
}

std::tuple<Tensor, Tensor> gru_input_native(const Tensor& input, const Tensor& hx,
                                            const std::vector<Tensor>& params,
                                            bool has_biases, int64_t num_layers, double dropout,
                                            bool train, bool bidirectional, bool batch_first) {
    return rnn_input_overload(1, input, hx, params, has_biases, num_layers, dropout, train,
                              bidirectional, batch_first);
}

std::tuple<Tensor, Tensor> rnn_relu_input_native(const Tensor& input, const Tensor& hx,
                                                 const std::vector<Tensor>& params,
                                                 bool has_biases, int64_t num_layers,
                                                 double dropout, bool train,
                                                 bool bidirectional, bool batch_first) {
    return rnn_input_overload(3, input, hx, params, has_biases, num_layers, dropout, train,
                              bidirectional, batch_first);
}

std::tuple<Tensor, Tensor> rnn_tanh_input_native(const Tensor& input, const Tensor& hx,
                                                 const std::vector<Tensor>& params,
                                                 bool has_biases, int64_t num_layers,
                                                 double dropout, bool train,
                                                 bool bidirectional, bool batch_first) {
    return rnn_input_overload(2, input, hx, params, has_biases, num_layers, dropout, train,
                              bidirectional, batch_first);
}

std::tuple<Tensor, Tensor, Tensor> lstm_input_native(const Tensor& input,
                                                     const std::vector<Tensor>& hx,
                                                     const std::vector<Tensor>& params,
                                                     bool has_biases, int64_t num_layers,
                                                     double dropout, bool train,
                                                     bool bidirectional, bool batch_first) {
    return ops::lstm(input, hx, params, has_biases, num_layers, static_cast<float>(dropout),
                     train, bidirectional, batch_first);
}

std::tuple<Tensor, Tensor> gru_data_native(const Tensor& data, const Tensor& batch_sizes,
                                           const Tensor& hx, const std::vector<Tensor>& params,
                                           bool has_biases, int64_t num_layers, double dropout,
                                           bool train, bool bidirectional) {
    if (batch_sizes.dim() != 1) {
        TP_THROW(RuntimeError, "gru.data: batch_sizes must be 1-dimensional");
    }
    const int64_t num_steps = batch_sizes.size(0);
    const int64_t total = data.size(0);
    if (num_steps == 0 || total == 0) {
        auto r = ops::gru(data.unsqueeze(0), std::vector<Tensor>{hx}, params, has_biases,
                          num_layers, static_cast<float>(dropout), train, bidirectional, false);
        return {std::get<0>(r).reshape({total, std::get<0>(r).size(-1)}), std::get<1>(r)};
    }
    Tensor bs_cpu = batch_sizes.to(Device(DeviceType::CPU)).contiguous();
    const int64_t* bs_ptr = bs_cpu.data_ptr<int64_t>();
    const int64_t max_batch = bs_ptr[0];
    const int64_t feat = data.size(1);
    Tensor padded = Tensor::zeros({num_steps, max_batch, feat}, data.dtype(), data.device());
    int64_t offset = 0;
    for (int64_t i = 0; i < num_steps; ++i) {
        const int64_t b = bs_ptr[i];
        if (b > 0) {
            padded.select(0, i).narrow(0, 0, b).copy_(data.narrow(0, offset, b));
        }
        offset += b;
    }
    auto r = ops::gru(padded, std::vector<Tensor>{hx}, params, has_biases, num_layers,
                      static_cast<float>(dropout), train, bidirectional, false);
    Tensor out_padded = std::get<0>(r);
    Tensor hn = std::get<1>(r);
    std::vector<Tensor> steps;
    steps.reserve(num_steps);
    for (int64_t i = 0; i < num_steps; ++i) {
        const int64_t b = bs_ptr[i];
        if (b > 0) steps.push_back(out_padded.select(0, i).narrow(0, 0, b));
    }
    Tensor out = steps.empty() ? Tensor::empty({0, out_padded.size(-1)}, out_padded.dtype(), out_padded.device())
                               : ops::cat(steps, 0);
    return {out, hn};
}

std::tuple<Tensor, Tensor> rnn_relu_data_native(const Tensor& data, const Tensor& batch_sizes,
                                                const Tensor& hx,
                                                const std::vector<Tensor>& params,
                                                bool has_biases, int64_t num_layers,
                                                double dropout, bool train,
                                                bool bidirectional) {
    if (batch_sizes.dim() != 1) {
        TP_THROW(RuntimeError, "rnn_relu.data: batch_sizes must be 1-dimensional");
    }
    const int64_t num_steps = batch_sizes.size(0);
    const int64_t total = data.size(0);
    if (num_steps == 0 || total == 0) {
        auto r = ops::rnn_relu(data.unsqueeze(0), std::vector<Tensor>{hx}, params, has_biases,
                               num_layers, static_cast<float>(dropout), train, bidirectional, false);
        return {std::get<0>(r).reshape({total, std::get<0>(r).size(-1)}), std::get<1>(r)};
    }
    Tensor bs_cpu = batch_sizes.to(Device(DeviceType::CPU)).contiguous();
    const int64_t* bs_ptr = bs_cpu.data_ptr<int64_t>();
    const int64_t max_batch = bs_ptr[0];
    const int64_t feat = data.size(1);
    Tensor padded = Tensor::zeros({num_steps, max_batch, feat}, data.dtype(), data.device());
    int64_t offset = 0;
    for (int64_t i = 0; i < num_steps; ++i) {
        const int64_t b = bs_ptr[i];
        if (b > 0) padded.select(0, i).narrow(0, 0, b).copy_(data.narrow(0, offset, b));
        offset += b;
    }
    auto r = ops::rnn_relu(padded, std::vector<Tensor>{hx}, params, has_biases, num_layers,
                           static_cast<float>(dropout), train, bidirectional, false);
    Tensor out_padded = std::get<0>(r);
    Tensor hn = std::get<1>(r);
    std::vector<Tensor> steps;
    steps.reserve(num_steps);
    for (int64_t i = 0; i < num_steps; ++i) {
        const int64_t b = bs_ptr[i];
        if (b > 0) steps.push_back(out_padded.select(0, i).narrow(0, 0, b));
    }
    Tensor out = steps.empty() ? Tensor::empty({0, out_padded.size(-1)}, out_padded.dtype(), out_padded.device())
                               : ops::cat(steps, 0);
    return {out, hn};
}

std::tuple<Tensor, Tensor> rnn_tanh_data_native(const Tensor& data, const Tensor& batch_sizes,
                                                const Tensor& hx,
                                                const std::vector<Tensor>& params,
                                                bool has_biases, int64_t num_layers,
                                                double dropout, bool train,
                                                bool bidirectional) {
    if (batch_sizes.dim() != 1) {
        TP_THROW(RuntimeError, "rnn_tanh.data: batch_sizes must be 1-dimensional");
    }
    const int64_t num_steps = batch_sizes.size(0);
    const int64_t total = data.size(0);
    if (num_steps == 0 || total == 0) {
        auto r = ops::rnn_tanh(data.unsqueeze(0), std::vector<Tensor>{hx}, params, has_biases,
                               num_layers, static_cast<float>(dropout), train, bidirectional, false);
        return {std::get<0>(r).reshape({total, std::get<0>(r).size(-1)}), std::get<1>(r)};
    }
    Tensor bs_cpu = batch_sizes.to(Device(DeviceType::CPU)).contiguous();
    const int64_t* bs_ptr = bs_cpu.data_ptr<int64_t>();
    const int64_t max_batch = bs_ptr[0];
    const int64_t feat = data.size(1);
    Tensor padded = Tensor::zeros({num_steps, max_batch, feat}, data.dtype(), data.device());
    int64_t offset = 0;
    for (int64_t i = 0; i < num_steps; ++i) {
        const int64_t b = bs_ptr[i];
        if (b > 0) padded.select(0, i).narrow(0, 0, b).copy_(data.narrow(0, offset, b));
        offset += b;
    }
    auto r = ops::rnn_tanh(padded, std::vector<Tensor>{hx}, params, has_biases, num_layers,
                           static_cast<float>(dropout), train, bidirectional, false);
    Tensor out_padded = std::get<0>(r);
    Tensor hn = std::get<1>(r);
    std::vector<Tensor> steps;
    steps.reserve(num_steps);
    for (int64_t i = 0; i < num_steps; ++i) {
        const int64_t b = bs_ptr[i];
        if (b > 0) steps.push_back(out_padded.select(0, i).narrow(0, 0, b));
    }
    Tensor out = steps.empty() ? Tensor::empty({0, out_padded.size(-1)}, out_padded.dtype(), out_padded.device())
                               : ops::cat(steps, 0);
    return {out, hn};
}

namespace {

// One packed-sequence LSTM layer.  batch_sizes is a non-increasing profile
// over the packed time axis; at each step the rows whose sequences just
// ended are frozen into the final-state list, and the remaining active
// prefix keeps running.  Reversing the frozen list at the end restores the
// original row order, so row j gets the state after its true last input.
// For the reverse direction the walk starts from the smallest batch and
// grows it, attaching fresh initial states for rows whose reversed sequence
// begins at the current step; every row is active at step 0, so the loop's
// final state is already the complete h_n/c_n in row order.
std::pair<Tensor, std::pair<Tensor, Tensor>> packed_lstm_layer(
    const Tensor& data, const int64_t* batch_sizes, int64_t num_steps,
    const Tensor& hx0, const Tensor& cx0, bool reverse,
    const Tensor& w_ih, const Tensor& w_hh,
    const std::optional<Tensor>& b_ih, const std::optional<Tensor>& b_hh) {
    std::vector<Tensor> step_outputs;
    std::vector<Tensor> frozen_h, frozen_c;
    Tensor h, c;
    if (!reverse) {
        h = hx0.narrow(0, 0, batch_sizes[0]);
        c = cx0.narrow(0, 0, batch_sizes[0]);
        int64_t input_offset = 0;
        int64_t last = batch_sizes[0];
        for (int64_t i = 0; i < num_steps; ++i) {
            const int64_t b = batch_sizes[i];
            Tensor step_input = data.narrow(0, input_offset, b);
            input_offset += b;
            const int64_t dec = last - b;
            if (dec > 0) {
                frozen_h.push_back(h.narrow(0, last - dec, dec));
                frozen_c.push_back(c.narrow(0, last - dec, dec));
                h = h.narrow(0, 0, last - dec);
                c = c.narrow(0, 0, last - dec);
            }
            last = b;
            auto cell = ops::lstm_cell(step_input, h, c, w_ih, w_hh, b_ih, b_hh);
            h = std::get<0>(cell);
            c = std::get<1>(cell);
            step_outputs.push_back(h);
        }
        frozen_h.push_back(h);
        frozen_c.push_back(c);
        std::reverse(frozen_h.begin(), frozen_h.end());
        std::reverse(frozen_c.begin(), frozen_c.end());
        return {ops::cat(step_outputs, 0),
                std::make_pair(ops::cat(frozen_h, 0), ops::cat(frozen_c, 0))};
    }
    h = hx0.narrow(0, 0, batch_sizes[num_steps - 1]);
    c = cx0.narrow(0, 0, batch_sizes[num_steps - 1]);
    int64_t input_offset = data.size(0);
    int64_t last = batch_sizes[num_steps - 1];
    for (int64_t i = num_steps - 1; i >= 0; --i) {
        const int64_t b = batch_sizes[i];
        const int64_t inc = b - last;
        if (inc > 0) {
            h = ops::cat(std::vector<Tensor>{
                h, hx0.narrow(0, last, inc)}, 0);
            c = ops::cat(std::vector<Tensor>{
                c, cx0.narrow(0, last, inc)}, 0);
        }
        Tensor step_input = data.narrow(0, input_offset - b, b);
        input_offset -= b;
        last = b;
        auto cell = ops::lstm_cell(step_input, h, c, w_ih, w_hh, b_ih, b_hh);
        h = std::get<0>(cell);
        c = std::get<1>(cell);
        step_outputs.push_back(h);
    }
    std::reverse(step_outputs.begin(), step_outputs.end());
    return {ops::cat(step_outputs, 0), std::make_pair(h, c)};
}

}  // namespace

std::tuple<Tensor, Tensor, Tensor> lstm_data_native(const Tensor& data,
                                                    const Tensor& batch_sizes,
                                                    const std::vector<Tensor>& hx,
                                                    const std::vector<Tensor>& params,
                                                    bool has_biases, int64_t num_layers,
                                                    double dropout, bool train,
                                                    bool bidirectional) {
    if (hx.size() != 2) {
        TP_THROW(RuntimeError, "lstm.data: expects two hidden states");
    }
    if (batch_sizes.dim() != 1) {
        TP_THROW(RuntimeError, "lstm.data: batch_sizes must be 1-dimensional");
    }
    const int64_t num_steps = batch_sizes.size(0);
    const int64_t total = data.size(0);
    if (num_steps == 0 || total == 0) {
        auto r = ops::lstm(data.unsqueeze(0), hx, params, has_biases, num_layers,
                           static_cast<float>(dropout), train, bidirectional, false);
        Tensor out = std::get<0>(r);
        return {out.reshape({total, out.size(-1)}), std::get<1>(r), std::get<2>(r)};
    }
    Tensor bs_cpu = batch_sizes.to(Device(DeviceType::CPU)).contiguous();
    const int64_t* bs_ptr = bs_cpu.data_ptr<int64_t>();
    const int64_t dirs = bidirectional ? 2 : 1;
    const int64_t max_batch = bs_ptr[0];
    if (hx[0].dim() != 3 || hx[1].dim() != 3) {
        TP_THROW(RuntimeError,
                 "lstm.data: hidden states must be [num_layers * dirs, batch, "
                 "hidden_size]");
    }
    if (hx[0].size(0) != num_layers * dirs || hx[1].size(0) != num_layers * dirs) {
        TP_THROW(RuntimeError,
                 "lstm.data: hidden states must have num_layers * dirs leading "
                 "rows, got ", hx[0].size(0));
    }
    if (hx[0].size(1) < max_batch || hx[1].size(1) < max_batch) {
        TP_THROW(RuntimeError,
                 "lstm.data: initial hidden batch must cover the largest "
                 "batch size (", max_batch, ")");
    }
    const int64_t param_stride = has_biases ? 4 : 2;
    if (static_cast<int64_t>(params.size()) < num_layers * dirs * param_stride) {
        TP_THROW(RuntimeError, "lstm.data: missing parameters");
    }
    std::vector<Tensor> hy_list, cy_list;
    hy_list.reserve(num_layers * dirs);
    cy_list.reserve(num_layers * dirs);
    Tensor layer_input_data = data;
    for (int64_t layer = 0; layer < num_layers; ++layer) {
        Tensor fw_out, fw_h, fw_c, rv_out, rv_h, rv_c;
        {
            const int64_t si = layer * dirs;
            const int64_t pbase = si * param_stride;
            std::optional<Tensor> b_ih, b_hh;
            if (has_biases) {
                b_ih = params[pbase + 2];
                b_hh = params[pbase + 3];
            }
            auto fw = packed_lstm_layer(
                layer_input_data, bs_ptr, num_steps,
                hx[0].select(0, si).contiguous(),
                hx[1].select(0, si).contiguous(),
                false, params[pbase], params[pbase + 1], b_ih, b_hh);
            fw_out = fw.first;
            fw_h = fw.second.first;
            fw_c = fw.second.second;
            hy_list.push_back(fw_h);
            cy_list.push_back(fw_c);
        }
        if (bidirectional) {
            const int64_t si = layer * dirs + 1;
            const int64_t pbase = si * param_stride;
            std::optional<Tensor> b_ih, b_hh;
            if (has_biases) {
                b_ih = params[pbase + 2];
                b_hh = params[pbase + 3];
            }
            auto rv = packed_lstm_layer(
                layer_input_data, bs_ptr, num_steps,
                hx[0].select(0, si).contiguous(),
                hx[1].select(0, si).contiguous(),
                true, params[pbase], params[pbase + 1], b_ih, b_hh);
            rv_out = rv.first;
            rv_h = rv.second.first;
            rv_c = rv.second.second;
            hy_list.push_back(rv_h);
            cy_list.push_back(rv_c);
            layer_input_data =
                ops::cat(std::vector<Tensor>{fw_out, rv_out}, fw_out.dim() - 1);
        } else {
            layer_input_data = fw_out;
        }
        if (train && dropout != 0.0 && layer < num_layers - 1) {
            layer_input_data = ops::dropout(layer_input_data, dropout, true);
        }
    }
    return {layer_input_data, ops::stack(hy_list, 0), ops::stack(cy_list, 0)};
}

// ---- sparse / misc ------------------------------------------------------------
Tensor to_sparse_sparse_dim_native(const Tensor& self, int64_t sparse_dim) {
    if (self.device().is_cpu()) {
        return cpu::to_sparse_coo_cpu_sparse_dim(self, sparse_dim);
    }
#ifdef USE_CUDA
    if (self.device().is_cuda()) {
        return cuda::to_sparse_coo_cuda_sparse_dim(self, sparse_dim);
    }
#endif
    TP_THROW(NotImplementedError,
             "to_sparse(): sparse_dim conversion is not implemented for this device");
}

Tensor _to_sparse_sparse_dim_native(const Tensor& self, int64_t sparse_dim) {
    return to_sparse_sparse_dim_native(self, sparse_dim);
}

Tensor sparse_coo_tensor_indices_native(const Tensor& indices, const Tensor& values,
                                        std::optional<DType> dtype,
                                        std::optional<int64_t> layout,
                                        std::optional<Device> device,
                                        std::optional<bool> pin_memory,
                                        std::optional<bool> is_coalesced) {
    if (layout.has_value() && *layout != 0) {
        TP_THROW(ValueError, "sparse_coo_tensor: layout must be sparse_coo");
    }
    Tensor i = indices;
    Tensor v = values;
    if (device.has_value()) {
        if (i.device() != *device) i = i.to(*device);
        if (v.device() != *device) v = v.to(*device);
    }
    if (dtype.has_value() && *dtype != DType::Undefined && v.dtype() != *dtype) {
        v = v.to(*dtype);
    }
    if (pin_memory.value_or(false)) {
        i = i.pin_memory();
        v = v.pin_memory();
    }
    return ops::sparse_coo_tensor(i, v, std::optional<std::vector<int64_t>>{},
                                  is_coalesced.value_or(false));
}

Tensor sparse_coo_tensor_indices_size_native(const Tensor& indices, const Tensor& values,
                                             const std::vector<int64_t>& size,
                                             std::optional<DType> dtype,
                                             std::optional<int64_t> layout,
                                             std::optional<Device> device,
                                             std::optional<bool> pin_memory,
                                             std::optional<bool> is_coalesced) {
    if (layout.has_value() && *layout != 0) {
        TP_THROW(ValueError, "sparse_coo_tensor: layout must be sparse_coo");
    }
    Tensor i = indices;
    Tensor v = values;
    if (device.has_value()) {
        if (i.device() != *device) i = i.to(*device);
        if (v.device() != *device) v = v.to(*device);
    }
    if (dtype.has_value() && *dtype != DType::Undefined && v.dtype() != *dtype) {
        v = v.to(*dtype);
    }
    if (pin_memory.value_or(false)) {
        i = i.pin_memory();
        v = v.pin_memory();
    }
    return ops::sparse_coo_tensor(i, v, size, is_coalesced.value_or(false));
}

Tensor sparse_coo_tensor_size_native(const std::vector<int64_t>& size,
                                     std::optional<DType> dtype,
                                     std::optional<int64_t> layout,
                                     std::optional<Device> device,
                                     std::optional<bool> pin_memory) {
    if (layout.has_value() && *layout != 0) {
        TP_THROW(ValueError, "sparse_coo_tensor: layout must be sparse_coo");
    }
    Tensor indices = ops::empty(
        {static_cast<int64_t>(size.size()), 0}, DType::Int64, device,
        pin_memory.value_or(false));
    Tensor values = ops::empty({0}, dtype, device,
                               pin_memory.value_or(false));
    return ops::sparse_coo_tensor(indices, values, size, false);
}

Tensor build_compressed_layout_tensor(
    const Tensor& compressed_indices, const Tensor& plain_indices,
    const Tensor& values, std::optional<std::vector<int64_t>> size,
    std::optional<DType> dtype, std::optional<int64_t> layout,
    std::optional<Device> device, std::optional<bool> pin_memory,
    bool validate_inputs, int fallback_layout, const char* name);

void validate_compressed_components(
    const Tensor& compressed_indices, const Tensor& plain_indices,
    const Tensor& values, const std::vector<int64_t>& size, int layout,
    const char* name);

void validate_csr_components(const Tensor& crow_indices,
                             const Tensor& col_indices,
                             const Tensor& values,
                             const std::vector<int64_t>& size) {
    if (crow_indices.dim() != 1 || col_indices.dim() != 1) {
        TP_THROW(ValueError, "sparse_csr_tensor: crow/col must be 1-D tensors");
    }
    if (values.dim() != 1) {
        TP_THROW(ValueError, "sparse_csr_tensor: values must be 1-D");
    }
    if (crow_indices.dtype() != DType::Int32 &&
        crow_indices.dtype() != DType::Int64) {
        TP_THROW(TypeError,
                 "sparse_csr_tensor: crow_indices must be Int32 or Int64");
    }
    if (col_indices.dtype() != DType::Int32 &&
        col_indices.dtype() != DType::Int64) {
        TP_THROW(TypeError,
                 "sparse_csr_tensor: col_indices must be Int32 or Int64");
    }
    if (crow_indices.device() != col_indices.device() ||
        crow_indices.device() != values.device()) {
        TP_THROW(DeviceMismatchError,
                 "sparse_csr_tensor: crow/col/values must share one device");
    }
    if (size.size() != 2 || size[0] < 0 || size[1] < 0) {
        TP_THROW(ValueError,
                 "sparse_csr_tensor: size must contain two non-negative entries");
    }
    const int64_t rows = size[0];
    const int64_t nnz = col_indices.size(0);
    if (crow_indices.size(0) != rows + 1) {
        TP_THROW(ValueError,
                 "sparse_csr_tensor: crow must have rows+1 entries");
    }
    if (values.size(0) != nnz) {
        TP_THROW(ValueError,
                 "sparse_csr_tensor: col and values must have the same nnz");
    }

    Tensor crow_host = crow_indices.device().is_cpu()
        ? crow_indices.contiguous()
        : crow_indices.to(Device(DeviceType::CPU)).contiguous();
    Tensor col_host = col_indices.device().is_cpu()
        ? col_indices.contiguous()
        : col_indices.to(Device(DeviceType::CPU)).contiguous();
    Tensor crow64 = crow_host.dtype() == DType::Int64
        ? crow_host : crow_host.to(DType::Int64);
    Tensor col64 = col_host.dtype() == DType::Int64
        ? col_host : col_host.to(DType::Int64);
    const int64_t* crow = crow64.data_ptr<int64_t>();
    const int64_t* col = col64.data_ptr<int64_t>();
    if (crow[0] != 0) {
        TP_THROW(ValueError, "sparse_csr_tensor: crow must start at zero");
    }
    for (int64_t row = 0; row < rows; ++row) {
        if (crow[row] > crow[row + 1]) {
            TP_THROW(ValueError,
                     "sparse_csr_tensor: crow must be non-decreasing");
        }
    }
    if (crow[rows] != nnz) {
        TP_THROW(ValueError,
                 "sparse_csr_tensor: crow last entry must equal nnz");
    }
    for (int64_t index = 0; index < nnz; ++index) {
        if (col[index] < 0 || col[index] >= size[1]) {
            TP_THROW(ValueError,
                     "sparse_csr_tensor: column index is out of range");
        }
    }
}

std::vector<int64_t> infer_csr_size(const Tensor& crow_indices,
                                    const Tensor& col_indices) {
    if (crow_indices.dim() != 1 || col_indices.dim() != 1) {
        TP_THROW(ValueError, "sparse_csr_tensor: crow/col must be 1-D tensors");
    }
    if (crow_indices.size(0) == 0) {
        TP_THROW(ValueError, "sparse_csr_tensor: crow must not be empty");
    }
    if (col_indices.dtype() != DType::Int32 &&
        col_indices.dtype() != DType::Int64) {
        TP_THROW(TypeError,
                 "sparse_csr_tensor: col_indices must be Int32 or Int64");
    }
    Tensor col_host = col_indices.device().is_cpu()
        ? col_indices.contiguous()
        : col_indices.to(Device(DeviceType::CPU)).contiguous();
    Tensor col64 = col_host.dtype() == DType::Int64
        ? col_host : col_host.to(DType::Int64);
    const int64_t nnz = col64.size(0);
    int64_t columns = 0;
    const int64_t* data = col64.data_ptr<int64_t>();
    for (int64_t index = 0; index < nnz; ++index) {
        if (data[index] < 0) {
            TP_THROW(ValueError,
                     "sparse_csr_tensor: column index must be non-negative");
        }
        columns = std::max(columns, data[index] + 1);
    }
    return {crow_indices.size(0) - 1, columns};
}

Tensor build_compressed_tensor(const Tensor& crow_indices,
                               const Tensor& col_indices,
                               const Tensor& values,
                               std::optional<std::vector<int64_t>> size,
                               std::optional<DType> dtype,
                               std::optional<int64_t> layout,
                               std::optional<Device> device,
                               std::optional<bool> pin_memory,
                               bool validate_inputs,
                               std::array<int64_t, 2> blocksize) {
    const int resolved_layout = layout.value_or(TensorImpl::kSparseCSRLayout);
    if (resolved_layout != TensorImpl::kSparseCSRLayout &&
        resolved_layout != TensorImpl::kSparseCSCLayout &&
        resolved_layout != TensorImpl::kSparseBSRLayout &&
        resolved_layout != TensorImpl::kSparseBSCLayout) {
        TP_THROW(ValueError,
                 "sparse_compressed_tensor: layout must be one of "
                 "sparse_csr/sparse_csc/sparse_bsr/sparse_bsc");
    }
    Tensor target_crow = crow_indices;
    Tensor target_col = col_indices;
    Tensor target_values = values;
    if (device.has_value()) {
        target_crow = target_crow.to(*device);
        target_col = target_col.to(*device);
        target_values = target_values.to(*device);
    }
    if (dtype.has_value() && *dtype != DType::Undefined &&
        target_values.dtype() != *dtype) {
        target_values = target_values.to(*dtype);
    }
    if (pin_memory.value_or(false)) {
        target_crow = target_crow.pin_memory();
        target_col = target_col.pin_memory();
        target_values = target_values.pin_memory();
    }
    if (!size.has_value()) {
        size = infer_csr_size(target_crow, target_col);
    }
    if (validate_inputs) {
        validate_csr_components(target_crow, target_col, target_values, *size);
    }
    return Tensor::make_sparse_compressed_tensor(
        target_crow, target_col, target_values, *size, resolved_layout,
        blocksize);
}

Tensor build_csr_tensor(const Tensor& crow_indices,
                        const Tensor& col_indices,
                        const Tensor& values,
                        std::optional<std::vector<int64_t>> size,
                        std::optional<DType> dtype,
                        std::optional<int64_t> layout,
                        std::optional<Device> device,
                        std::optional<bool> pin_memory,
                        bool validate_inputs) {
    if (layout.has_value() &&
        *layout != TensorImpl::kSparseCSRLayout) {
        TP_THROW(ValueError, "sparse_csr_tensor: layout must be sparse_csr");
    }
    return build_compressed_layout_tensor(
        crow_indices, col_indices, values, size, dtype, layout, device,
        pin_memory, validate_inputs, TensorImpl::kSparseCSRLayout,
        "sparse_csr_tensor");
}

// Dense -> compressed sparse wrappers.  Each public spelling pins the
// target layout and forwards to the shared conversion kernel.
Tensor dense_to_sparse_compressed_native(
    const Tensor& self, int layout,
    std::array<int64_t, 2> blocksize,
    std::optional<int64_t> dense_dim, const char* name) {
    if (self.is_sparse()) {
        const int self_layout = self.unsafeGetTensorImpl()->sparse_layout();
        if (self_layout == layout) {
            return self;
        }
        // Cross-layout: go through the dense form and re-compress.
        return tensorplay::cpu::to_sparse_compressed_cpu(
            self.to_dense(), layout, blocksize, dense_dim, name);
    }
    return tensorplay::cpu::to_sparse_compressed_cpu(self, layout, blocksize, dense_dim,
                                         name);
}

Tensor to_sparse_csc_native(const Tensor& self, std::optional<int64_t> dense_dim) {
    reject_active_transform(self, "to_sparse_csc");
    return dense_to_sparse_compressed_native(
        self, TensorImpl::kSparseCSCLayout, {0, 0}, dense_dim, "to_sparse_csc");
}

Tensor _to_sparse_csc_native(const Tensor& self, std::optional<int64_t> dense_dim) {
    return to_sparse_csc_native(self, dense_dim);
}

Tensor to_sparse_bsr_native(const Tensor& self, const std::vector<int64_t>& blocksize,
                            std::optional<int64_t> dense_dim) {
    reject_active_transform(self, "to_sparse_bsr");
    TP_CHECK(blocksize.size() == 2, "to_sparse_bsr: blocksize must have 2 elements");
    return dense_to_sparse_compressed_native(
        self, TensorImpl::kSparseBSRLayout,
        {blocksize[0], blocksize[1]}, dense_dim, "to_sparse_bsr");
}

Tensor _to_sparse_bsr_native(const Tensor& self, const std::vector<int64_t>& blocksize,
                             std::optional<int64_t> dense_dim) {
    return to_sparse_bsr_native(self, blocksize, dense_dim);
}

Tensor to_sparse_bsc_native(const Tensor& self, const std::vector<int64_t>& blocksize,
                            std::optional<int64_t> dense_dim) {
    reject_active_transform(self, "to_sparse_bsc");
    TP_CHECK(blocksize.size() == 2, "to_sparse_bsc: blocksize must have 2 elements");
    return dense_to_sparse_compressed_native(
        self, TensorImpl::kSparseBSCLayout,
        {blocksize[0], blocksize[1]}, dense_dim, "to_sparse_bsc");
}

Tensor _to_sparse_bsc_native(const Tensor& self, const std::vector<int64_t>& blocksize,
                             std::optional<int64_t> dense_dim) {
    return to_sparse_bsc_native(self, blocksize, dense_dim);
}

Tensor sparse_csr_tensor_crow_col_value_native(
    const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values,
    std::optional<DType> dtype, std::optional<int64_t> layout,
    std::optional<Device> device, std::optional<bool> pin_memory) {
    return build_csr_tensor(crow_indices, col_indices, values, std::nullopt,
                            dtype, layout, device, pin_memory, true);
}

Tensor sparse_csr_tensor_crow_col_value_size_native(
    const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values,
    const std::vector<int64_t>& size, std::optional<DType> dtype,
    std::optional<int64_t> layout, std::optional<Device> device,
    std::optional<bool> pin_memory) {
    return build_csr_tensor(crow_indices, col_indices, values, size, dtype,
                            layout, device, pin_memory, true);
}

Tensor sparse_csr_tensor_unsafe_native(
    const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values,
    const std::vector<int64_t>& size, std::optional<DType> dtype,
    std::optional<int64_t> layout, std::optional<Device> device,
    std::optional<bool> pin_memory) {
    return build_csr_tensor(crow_indices, col_indices, values, size, dtype,
                            layout, device, pin_memory, false);
}

void validate_sparse_csr_tensor_args_native(
    const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values,
    const std::vector<int64_t>& size, std::optional<bool> check_pinning) {
    (void)check_pinning;
    validate_compressed_components(crow_indices, col_indices, values, size,
                                   TensorImpl::kSparseCSRLayout,
                                   "_validate_sparse_csr_tensor_args");
}

// ---------------------------------------------------------------------------
// Compressed-sparse constructors.
//
// Batch axes stay on the compressed/plain index tensors and on the values
// tensor: compressed/plain are (*batch, units+1)/(*batch, nnz), while values
// are (*batch, nnz, *block, *dense).  The shared helper supports CSR, CSC, BSR,
// and BSC with that representation.
// ---------------------------------------------------------------------------

bool layout_is_row_compressed(int layout) {
    return layout == TensorImpl::kSparseCSRLayout ||
           layout == TensorImpl::kSparseBSRLayout;
}

bool layout_is_blocked(int layout) {
    return layout == TensorImpl::kSparseBSRLayout ||
           layout == TensorImpl::kSparseBSCLayout;
}

int resolve_compressed_layout(std::optional<int64_t> layout, int fallback,
                              const char* name) {
    const int resolved =
        layout.has_value() ? static_cast<int>(*layout) : fallback;
    if (resolved != TensorImpl::kSparseCSRLayout &&
        resolved != TensorImpl::kSparseCSCLayout &&
        resolved != TensorImpl::kSparseBSRLayout &&
        resolved != TensorImpl::kSparseBSCLayout) {
        TP_THROW(ValueError, name,
                 ": layout must be one of "
                 "sparse_csr/sparse_csc/sparse_bsr/sparse_bsc");
    }
    return resolved;
}

// Block extents a values payload declares: (*batch, nnz, b0, b1, *dense) for
// the blocked layouts, (*batch, nnz, *dense) otherwise.
std::array<int64_t, 2> blocksize_of_values(const Tensor& values,
                                           int64_t batch_ndim, int layout,
                                           const char* name) {
    if (!layout_is_blocked(layout)) return {1, 1};
    if (values.dim() < batch_ndim + 3) {
        TP_THROW(ValueError, name,
                 ": blocked layouts need values shaped "
                 "(*batch, nnz, block_rows, block_cols, ...)");
    }
    return {values.size(batch_ndim + 1), values.size(batch_ndim + 2)};
}

struct CompressedGeometry {
    int64_t comp_units;   // entries the compressed pointer array indexes
    int64_t plain_units;  // exclusive upper bound of a plain coordinate
};

CompressedGeometry compressed_geometry(const std::vector<int64_t>& size,
                                       int64_t dense_dim,
                                       int layout,
                                       std::array<int64_t, 2> blocksize,
                                       const char* name) {
    const int64_t ndim = static_cast<int64_t>(size.size());
    const int64_t batch_dim = ndim - 2 - dense_dim;
    if (batch_dim < 0) {
        TP_THROW(ValueError, name, ": size must hold at least dense_dim(=",
                 dense_dim, ") + 2 entries, got ", ndim);
    }
    for (int64_t extent : size) {
        if (extent < 0) {
            TP_THROW(ValueError, name, ": sizes must be non-negative");
        }
    }
    const int64_t rows = size[batch_dim];
    const int64_t cols = size[batch_dim + 1];

    const bool row_compressed = layout_is_row_compressed(layout);
    const int64_t b0 = blocksize[0];
    const int64_t b1 = blocksize[1];
    if (b0 <= 0 || b1 <= 0) {
        TP_THROW(ValueError, name, ": block sizes must be positive");
    }
    const int64_t comp_axis = row_compressed ? rows : cols;
    const int64_t plain_axis = row_compressed ? cols : rows;
    const int64_t comp_extent = row_compressed ? b0 : b1;
    const int64_t plain_extent = row_compressed ? b1 : b0;
    if (comp_axis % comp_extent != 0 || plain_axis % plain_extent != 0) {
        TP_THROW(ValueError, name,
                 ": sizes must be divisible by the block sizes");
    }
    return {comp_axis / comp_extent, plain_axis / plain_extent};
}

// Dense trailing rank a values payload carries beyond (*batch, nnz[, b0, b1]).
int64_t dense_dim_of_values(const Tensor& values, int64_t batch_ndim,
                            int layout, const char* name) {
    const int64_t payload_rank =
        batch_ndim + (layout_is_blocked(layout) ? 3 : 1);
    if (values.dim() < payload_rank) {
        TP_THROW(ValueError, name, ": values rank is incompatible with the "
                 "index rank");
    }
    return values.dim() - payload_rank;
}

// Read an index tensor as host Int64 so the structural checks below can walk
// it regardless of the device or index width it was built with.
Tensor index_as_host_int64(const Tensor& indices) {
    Tensor host = indices.device().is_cpu()
        ? indices.contiguous()
        : indices.to(Device(DeviceType::CPU)).contiguous();
    return host.dtype() == DType::Int64 ? host : host.to(DType::Int64);
}

void validate_compressed_components(const Tensor& compressed_indices,
                                    const Tensor& plain_indices,
                                    const Tensor& values,
                                    const std::vector<int64_t>& size,
                                    int layout,
                                    const char* name) {
    if (compressed_indices.dim() < 1 ||
        plain_indices.dim() != compressed_indices.dim()) {
        TP_THROW(ValueError, name,
                 ": compressed/plain indices must have the same rank and "
                 "at least one dimension");
    }
    if ((compressed_indices.dtype() != DType::Int32 &&
         compressed_indices.dtype() != DType::Int64) ||
        plain_indices.dtype() != compressed_indices.dtype()) {
        TP_THROW(TypeError, name,
                 ": compressed/plain indices must have the same Int32 or "
                 "Int64 dtype");
    }
    if (compressed_indices.device() != plain_indices.device() ||
        compressed_indices.device() != values.device()) {
        TP_THROW(DeviceMismatchError, name,
                 ": indices and values must share one device");
    }

    const int64_t batch_ndim = compressed_indices.dim() - 1;
    const std::array<int64_t, 2> blocksize =
        blocksize_of_values(values, batch_ndim, layout, name);
    const int64_t dense_dim =
        dense_dim_of_values(values, batch_ndim, layout, name);
    const CompressedGeometry geom =
        compressed_geometry(size, dense_dim, layout, blocksize, name);
    const int64_t block_ndim = layout_is_blocked(layout) ? 2 : 0;
    if (static_cast<int64_t>(size.size()) !=
        batch_ndim + 2 + dense_dim) {
        TP_THROW(ValueError, name,
                 ": size rank does not match the index/value ranks");
    }
    for (int64_t d = 0; d < batch_ndim; ++d) {
        if (compressed_indices.size(d) != size[static_cast<size_t>(d)] ||
            plain_indices.size(d) != size[static_cast<size_t>(d)] ||
            values.size(d) != size[static_cast<size_t>(d)]) {
            TP_THROW(ValueError, name,
                     ": batch dimensions of indices, values, and size must "
                     "match");
        }
    }
    const int64_t nnz = values.size(batch_ndim);

    if (compressed_indices.size(batch_ndim) != geom.comp_units + 1) {
        TP_THROW(ValueError, name, ": compressed indices must have ",
                 geom.comp_units + 1, " entries, got ",
                 compressed_indices.size(batch_ndim));
    }
    if (plain_indices.size(batch_ndim) != nnz) {
        TP_THROW(ValueError, name,
                 ": plain indices and values must agree on nnz");
    }
    if (compressed_indices.stride(-1) != 1 ||
        plain_indices.stride(-1) != 1) {
        TP_THROW(ValueError, name,
                 ": the last index dimension must have stride 1");
    }
    if (layout_is_blocked(layout) &&
        (values.size(batch_ndim + 1) != blocksize[0] ||
         values.size(batch_ndim + 2) != blocksize[1])) {
        TP_THROW(ValueError, name,
                 ": values block dimensions do not match the layout");
    }
    const int64_t dense_start = batch_ndim + 1 + block_ndim;
    for (int64_t d = 0; d < dense_dim; ++d) {
        if (values.size(dense_start + d) !=
            size[static_cast<size_t>(batch_ndim + 2 + d)]) {
            TP_THROW(ValueError, name,
                     ": values dense dimensions must match size");
        }
    }

    const Tensor comp64 = index_as_host_int64(compressed_indices);
    const Tensor plain64 = index_as_host_int64(plain_indices);
    const int64_t* comp = comp64.data_ptr<int64_t>();
    const int64_t* plain = plain64.data_ptr<int64_t>();
    int64_t batch_count = 1;
    for (int64_t d = 0; d < batch_ndim; ++d) {
        batch_count *= compressed_indices.size(d);
    }
    for (int64_t batch = 0; batch < batch_count; ++batch) {
        const int64_t comp_base = batch * (geom.comp_units + 1);
        const int64_t plain_base = batch * nnz;
        if (comp[comp_base] != 0) {
            TP_THROW(ValueError, name,
                     ": compressed indices must start at zero in every "
                     "batch");
        }
        for (int64_t unit = 0; unit < geom.comp_units; ++unit) {
            if (comp[comp_base + unit] > comp[comp_base + unit + 1]) {
                TP_THROW(ValueError, name,
                         ": compressed indices must be non-decreasing");
            }
        }
        if (comp[comp_base + geom.comp_units] != nnz) {
            TP_THROW(ValueError, name,
                     ": last compressed index must equal nnz in every batch");
        }
        for (int64_t i = 0; i < nnz; ++i) {
            if (plain[plain_base + i] < 0 ||
                plain[plain_base + i] >= geom.plain_units) {
                TP_THROW(ValueError, name, ": plain index is out of range");
            }
        }
    }
}

// Recover (*batch, rows, cols, *dense) for the spellings that omit `size`: the
// compressed axis is described by the pointer array, the plain axis by the
// largest coordinate that appears, and the remaining shape comes from values.
std::vector<int64_t> infer_compressed_size(const Tensor& compressed_indices,
                                           const Tensor& plain_indices,
                                           const Tensor& values,
                                           int layout,
                                           std::array<int64_t, 2> blocksize,
                                           const char* name) {
    if (compressed_indices.dim() < 1 ||
        plain_indices.dim() != compressed_indices.dim()) {
        TP_THROW(ValueError, name,
                 ": compressed/plain indices must have the same rank and "
                 "at least one dimension");
    }
    if (compressed_indices.dtype() != plain_indices.dtype() ||
        (compressed_indices.dtype() != DType::Int32 &&
         compressed_indices.dtype() != DType::Int64)) {
        TP_THROW(TypeError, name,
                 ": compressed/plain indices must have the same Int32 or "
                 "Int64 dtype");
    }
    const int64_t batch_ndim = compressed_indices.dim() - 1;
    if (compressed_indices.size(batch_ndim) == 0) {
        TP_THROW(ValueError, name, ": compressed indices must not be empty");
    }
    const int64_t dense_dim =
        dense_dim_of_values(values, batch_ndim, layout, name);
    if (dense_dim < 0) {
        TP_THROW(ValueError, name,
                 ": values rank is incompatible with the index rank");
    }
    for (int64_t d = 0; d < batch_ndim; ++d) {
        if (compressed_indices.size(d) != plain_indices.size(d) ||
            compressed_indices.size(d) != values.size(d)) {
            TP_THROW(ValueError, name,
                     ": batch dimensions of indices and values must match");
        }
    }
    const Tensor plain64 = index_as_host_int64(plain_indices);
    const int64_t nnz = plain_indices.size(batch_ndim);
    const int64_t* plain = plain64.data_ptr<int64_t>();
    int64_t plain_units = 0;
    for (int64_t i = 0; i < plain64.numel(); ++i) {
        if (plain[i] < 0) {
            TP_THROW(ValueError, name, ": plain index must be non-negative");
        }
        plain_units = std::max(plain_units, plain[i] + 1);
    }
    const int64_t comp_units = compressed_indices.size(batch_ndim) - 1;
    const bool row_compressed = layout_is_row_compressed(layout);
    const int64_t comp_extent = row_compressed ? blocksize[0] : blocksize[1];
    const int64_t plain_extent = row_compressed ? blocksize[1] : blocksize[0];
    const int64_t comp_axis = comp_units * comp_extent;
    const int64_t plain_axis = plain_units * plain_extent;
    std::vector<int64_t> result;
    for (int64_t d = 0; d < batch_ndim; ++d) {
        result.push_back(compressed_indices.size(d));
    }
    if (row_compressed) {
        result.push_back(comp_axis);
        result.push_back(plain_axis);
    } else {
        result.push_back(plain_axis);
        result.push_back(comp_axis);
    }
    const int64_t dense_start = batch_ndim + 1 +
        (layout_is_blocked(layout) ? 2 : 0);
    for (int64_t d = 0; d < dense_dim; ++d) {
        result.push_back(values.size(dense_start + d));
    }
    return result;
}

// Shared body for every compressed constructor: move the components to the
// requested dtype/device, fill in an omitted size, validate unless the caller
// asked for the unchecked spelling, and install them under `layout`.
Tensor build_compressed_layout_tensor(const Tensor& compressed_indices,
                                      const Tensor& plain_indices,
                                      const Tensor& values,
                                      std::optional<std::vector<int64_t>> size,
                                      std::optional<DType> dtype,
                                      std::optional<int64_t> layout,
                                      std::optional<Device> device,
                                      std::optional<bool> pin_memory,
                                      bool validate_inputs,
                                      int fallback_layout,
                                      const char* name) {
    if (fallback_layout < TensorImpl::kSparseCSRLayout &&
        !layout.has_value()) {
        TP_THROW(ValueError, name,
                 " expected sparse compressed tensor layout but got none");
    }
    if (fallback_layout >= TensorImpl::kSparseCSRLayout && layout.has_value() &&
        *layout != fallback_layout) {
        TP_THROW(ValueError, name,
                 ": layout does not match the constructor spelling");
    }
    const int resolved_layout =
        resolve_compressed_layout(layout, fallback_layout, name);
    Tensor target_compressed = compressed_indices;
    Tensor target_plain = plain_indices;
    Tensor target_values = values;
    if (device.has_value()) {
        target_compressed = target_compressed.to(*device);
        target_plain = target_plain.to(*device);
        target_values = target_values.to(*device);
    }
    if (dtype.has_value() && *dtype != DType::Undefined &&
        target_values.dtype() != *dtype) {
        target_values = target_values.to(*dtype);
    }
    if (pin_memory.value_or(false)) {
        target_compressed = target_compressed.pin_memory();
        target_plain = target_plain.pin_memory();
        target_values = target_values.pin_memory();
    }
    const int64_t batch_ndim = target_compressed.dim() - 1;
    const std::array<int64_t, 2> blocksize =
        blocksize_of_values(target_values, batch_ndim, resolved_layout, name);
    if (!size.has_value()) {
        size = infer_compressed_size(target_compressed, target_plain,
                                     target_values, resolved_layout, blocksize,
                                     name);
    }
    if (validate_inputs) {
        validate_compressed_components(target_compressed, target_plain,
                                       target_values, *size, resolved_layout,
                                       name);
    }
    return Tensor::make_sparse_compressed_tensor(
        target_compressed, target_plain, target_values, *size, resolved_layout,
        blocksize);
}

Tensor sparse_compressed_tensor_comp_plain_value_size_native(
    const Tensor& compressed_indices, const Tensor& plain_indices,
    const Tensor& values, const std::vector<int64_t>& size,
    std::optional<DType> dtype, std::optional<int64_t> layout,
    std::optional<Device> device, std::optional<bool> pin_memory) {
    return build_compressed_layout_tensor(
        compressed_indices, plain_indices, values, size, dtype, layout, device,
        pin_memory, true, -1,
        "sparse_compressed_tensor");
}

Tensor sparse_compressed_tensor_comp_plain_value_native(
    const Tensor& compressed_indices, const Tensor& plain_indices,
    const Tensor& values, std::optional<DType> dtype,
    std::optional<int64_t> layout, std::optional<Device> device,
    std::optional<bool> pin_memory) {
    return build_compressed_layout_tensor(
        compressed_indices, plain_indices, values, std::nullopt, dtype, layout,
        device, pin_memory, true, -1,
        "sparse_compressed_tensor");
}

Tensor sparse_compressed_tensor_unsafe_native(
    const Tensor& compressed_indices, const Tensor& plain_indices,
    const Tensor& values, const std::vector<int64_t>& size,
    std::optional<DType> dtype, std::optional<int64_t> layout,
    std::optional<Device> device, std::optional<bool> pin_memory) {
    return build_compressed_layout_tensor(
        compressed_indices, plain_indices, values, size, dtype, layout, device,
        pin_memory, false, -1,
        "_sparse_compressed_tensor_unsafe");
}

// The layout-pinned spellings differ only in which layout they fall back to
// when the caller leaves the kwarg unset.
Tensor sparse_csc_tensor_ccol_row_value_size_native(
    const Tensor& ccol_indices, const Tensor& row_indices, const Tensor& values,
    const std::vector<int64_t>& size, std::optional<DType> dtype,
    std::optional<int64_t> layout, std::optional<Device> device,
    std::optional<bool> pin_memory) {
    return build_compressed_layout_tensor(
        ccol_indices, row_indices, values, size, dtype, layout, device,
        pin_memory, true, TensorImpl::kSparseCSCLayout, "sparse_csc_tensor");
}

Tensor sparse_csc_tensor_ccol_row_value_native(
    const Tensor& ccol_indices, const Tensor& row_indices, const Tensor& values,
    std::optional<DType> dtype, std::optional<int64_t> layout,
    std::optional<Device> device, std::optional<bool> pin_memory) {
    return build_compressed_layout_tensor(
        ccol_indices, row_indices, values, std::nullopt, dtype, layout, device,
        pin_memory, true, TensorImpl::kSparseCSCLayout, "sparse_csc_tensor");
}

Tensor sparse_csc_tensor_unsafe_native(
    const Tensor& ccol_indices, const Tensor& row_indices, const Tensor& values,
    const std::vector<int64_t>& size, std::optional<DType> dtype,
    std::optional<int64_t> layout, std::optional<Device> device,
    std::optional<bool> pin_memory) {
    return build_compressed_layout_tensor(
        ccol_indices, row_indices, values, size, dtype, layout, device,
        pin_memory, false, TensorImpl::kSparseCSCLayout,
        "_sparse_csc_tensor_unsafe");
}

Tensor sparse_bsr_tensor_crow_col_value_size_native(
    const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values,
    const std::vector<int64_t>& size, std::optional<DType> dtype,
    std::optional<int64_t> layout, std::optional<Device> device,
    std::optional<bool> pin_memory) {
    return build_compressed_layout_tensor(
        crow_indices, col_indices, values, size, dtype, layout, device,
        pin_memory, true, TensorImpl::kSparseBSRLayout, "sparse_bsr_tensor");
}

Tensor sparse_bsr_tensor_crow_col_value_native(
    const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values,
    std::optional<DType> dtype, std::optional<int64_t> layout,
    std::optional<Device> device, std::optional<bool> pin_memory) {
    return build_compressed_layout_tensor(
        crow_indices, col_indices, values, std::nullopt, dtype, layout, device,
        pin_memory, true, TensorImpl::kSparseBSRLayout, "sparse_bsr_tensor");
}

Tensor sparse_bsr_tensor_unsafe_native(
    const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values,
    const std::vector<int64_t>& size, std::optional<DType> dtype,
    std::optional<int64_t> layout, std::optional<Device> device,
    std::optional<bool> pin_memory) {
    return build_compressed_layout_tensor(
        crow_indices, col_indices, values, size, dtype, layout, device,
        pin_memory, false, TensorImpl::kSparseBSRLayout,
        "_sparse_bsr_tensor_unsafe");
}

Tensor sparse_bsc_tensor_ccol_row_value_size_native(
    const Tensor& ccol_indices, const Tensor& row_indices, const Tensor& values,
    const std::vector<int64_t>& size, std::optional<DType> dtype,
    std::optional<int64_t> layout, std::optional<Device> device,
    std::optional<bool> pin_memory) {
    return build_compressed_layout_tensor(
        ccol_indices, row_indices, values, size, dtype, layout, device,
        pin_memory, true, TensorImpl::kSparseBSCLayout, "sparse_bsc_tensor");
}

Tensor sparse_bsc_tensor_ccol_row_value_native(
    const Tensor& ccol_indices, const Tensor& row_indices, const Tensor& values,
    std::optional<DType> dtype, std::optional<int64_t> layout,
    std::optional<Device> device, std::optional<bool> pin_memory) {
    return build_compressed_layout_tensor(
        ccol_indices, row_indices, values, std::nullopt, dtype, layout, device,
        pin_memory, true, TensorImpl::kSparseBSCLayout, "sparse_bsc_tensor");
}

Tensor sparse_bsc_tensor_unsafe_native(
    const Tensor& ccol_indices, const Tensor& row_indices, const Tensor& values,
    const std::vector<int64_t>& size, std::optional<DType> dtype,
    std::optional<int64_t> layout, std::optional<Device> device,
    std::optional<bool> pin_memory) {
    return build_compressed_layout_tensor(
        ccol_indices, row_indices, values, size, dtype, layout, device,
        pin_memory, false, TensorImpl::kSparseBSCLayout,
        "_sparse_bsc_tensor_unsafe");
}

void validate_sparse_compressed_tensor_args_native(
    const Tensor& compressed_indices, const Tensor& plain_indices,
    const Tensor& values, const std::vector<int64_t>& size, int64_t layout,
    std::optional<bool> check_pinning) {
    (void)check_pinning;
    const int resolved = resolve_compressed_layout(
        layout, TensorImpl::kSparseCSRLayout,
        "_validate_sparse_compressed_tensor_args");
    validate_compressed_components(compressed_indices, plain_indices, values,
                                   size, resolved,
                                   "_validate_sparse_compressed_tensor_args");
}

void validate_sparse_csc_tensor_args_native(
    const Tensor& ccol_indices, const Tensor& row_indices, const Tensor& values,
    const std::vector<int64_t>& size, std::optional<bool> check_pinning) {
    (void)check_pinning;
    validate_compressed_components(ccol_indices, row_indices, values, size,
                                   TensorImpl::kSparseCSCLayout,
                                   "_validate_sparse_csc_tensor_args");
}

void validate_sparse_bsr_tensor_args_native(
    const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values,
    const std::vector<int64_t>& size, std::optional<bool> check_pinning) {
    (void)check_pinning;
    validate_compressed_components(crow_indices, col_indices, values, size,
                                   TensorImpl::kSparseBSRLayout,
                                   "_validate_sparse_bsr_tensor_args");
}

void validate_sparse_bsc_tensor_args_native(
    const Tensor& ccol_indices, const Tensor& row_indices, const Tensor& values,
    const std::vector<int64_t>& size, std::optional<bool> check_pinning) {
    (void)check_pinning;
    validate_compressed_components(ccol_indices, row_indices, values, size,
                                   TensorImpl::kSparseBSCLayout,
                                   "_validate_sparse_bsc_tensor_args");
}

// Uninitialized compressed tensor with room for `nnz` stored units.  The
// indices are left empty on purpose: this is the allocation half of a
// build, and the caller fills the structure in afterwards, so the result does
// not satisfy the layout invariants until it does.
Tensor sparse_compressed_tensor_with_dims_native(
    int64_t nnz, int64_t dense_dim, const std::vector<int64_t>& size,
    const std::vector<int64_t>& blocksize, DType index_dtype,
    std::optional<DType> dtype, std::optional<int64_t> layout,
    std::optional<Device> device, std::optional<bool> pin_memory) {
    const char* name = "_sparse_compressed_tensor_with_dims";
    if (!layout.has_value()) {
        TP_THROW(ValueError, name,
                 ": expected a sparse compressed layout but got none");
    }
    const int resolved =
        resolve_compressed_layout(layout, TensorImpl::kSparseCSRLayout, name);
    if (nnz < 0) {
        TP_THROW(ValueError, name, ": nnz must be non-negative, got ", nnz);
    }
    if (dense_dim < 0) {
        TP_THROW(ValueError, name, ": dense_dim must be non-negative, got ",
                 dense_dim);
    }
    if (index_dtype != DType::Int32 && index_dtype != DType::Int64) {
        TP_THROW(TypeError, name, ": index dtype must be Int32 or Int64");
    }
    const bool blocked = layout_is_blocked(resolved);
    if (blocked) {
        if (blocksize.size() != 2) {
            TP_THROW(ValueError, name,
                     ": blocksize must have 2 entries for a blocked layout, got ",
                     blocksize.size());
        }
    } else if (!blocksize.empty()) {
        TP_THROW(ValueError, name,
                 ": blocksize cannot be given for a non-blocked layout");
    }
    const std::array<int64_t, 2> extents =
        blocked ? std::array<int64_t, 2>{blocksize[0], blocksize[1]}
                : std::array<int64_t, 2>{1, 1};
    const CompressedGeometry geom =
        compressed_geometry(size, dense_dim, resolved, extents, name);

    const int64_t batch_dim =
        static_cast<int64_t>(size.size()) - 2 - dense_dim;
    if (batch_dim < 0) {
        TP_THROW(ValueError, name, ": size rank is smaller than dense_dim + 2");
    }
    std::vector<int64_t> batch_shape(size.begin(),
                                     size.begin() + batch_dim);
    std::vector<int64_t> compressed_shape = batch_shape;
    compressed_shape.push_back(geom.comp_units + 1);
    std::vector<int64_t> plain_shape = batch_shape;
    plain_shape.push_back(nnz);
    std::vector<int64_t> values_shape = batch_shape;
    values_shape.push_back(nnz);
    if (blocked) {
        values_shape.push_back(extents[0]);
        values_shape.push_back(extents[1]);
    }
    for (int64_t d = batch_dim + 2; d < static_cast<int64_t>(size.size()); ++d) {
        values_shape.push_back(size[d]);
    }

    Tensor compressed_indices =
        Tensor::empty(compressed_shape, index_dtype,
                      device.value_or(globalContext().defaultDevice()));
    Tensor plain_indices = Tensor::empty(
        plain_shape, index_dtype,
        device.value_or(globalContext().defaultDevice()));
    Tensor values = Tensor::empty(
        values_shape, dtype.value_or(globalContext().defaultDType()),
        device.value_or(globalContext().defaultDevice()));
    if (pin_memory.value_or(false)) {
        compressed_indices = compressed_indices.pin_memory();
        plain_indices = plain_indices.pin_memory();
        values = values.pin_memory();
    }
    return Tensor::make_sparse_compressed_tensor(
        compressed_indices, plain_indices, values, size, resolved, extents);
}

Tensor& multinomial_out_native(const Tensor& self, int64_t num_samples, bool replacement,
                               std::optional<Generator> generator, Tensor& out) {
    (void)generator;
    return write_exact_out("multinomial",
                           ops::multinomial(self, num_samples, replacement),
                           out);
}

Tensor& linalg_lu_solve_out_native(const Tensor& LU, const Tensor& pivots, const Tensor& B,
                                   bool left, bool adjoint, Tensor& out) {
    return write_exact_out("linalg_lu_solve",
                           ops::linalg_lu_solve(LU, pivots, B, left, adjoint),
                           out);
}

void split_with_sizes_copy_out_native(const Tensor& self,
                                      const std::vector<int64_t>& split_sizes,
                                      int64_t dim, std::vector<Tensor> outs) {
    auto parts = ops::split_with_sizes_copy(self, split_sizes, dim);
    for (size_t i = 0; i < outs.size() && i < parts.size(); ++i) {
        ops::copy_(outs[i], parts[i]);
    }
}

// ---- generator-qualified factory overloads ---------------------------------
Tensor rand_generator_native(const std::vector<int64_t>& size,
                             std::optional<Generator> generator,
                             std::optional<DType> dtype,
                             std::optional<int64_t> layout,
                             std::optional<Device> device,
                             std::optional<bool> pin_memory) {
    (void)generator; (void)layout; (void)pin_memory;
    return ops::rand(size, dtype, device);
}

Tensor rand_like_generator_native(const Tensor& self,
                                  std::optional<Generator> generator,
                                  std::optional<DType> dtype,
                                  std::optional<int64_t> layout,
                                  std::optional<Device> device,
                                  std::optional<bool> pin_memory,
                                  std::optional<int64_t> memory_format) {
    (void)generator; (void)layout; (void)pin_memory; (void)memory_format;
    return ops::rand_like(self, dtype.value_or(DType::Undefined), device);
}

Tensor randint_low_native(int64_t low, int64_t high, const std::vector<int64_t>& size,
                          std::optional<DType> dtype,
                          std::optional<int64_t> layout,
                          std::optional<Device> device,
                          std::optional<bool> pin_memory) {
    (void)layout; (void)pin_memory;
    return ops::randint(low, high, size, dtype.value_or(DType::Int64), device);
}

Tensor randint_generator_native(int64_t high, const std::vector<int64_t>& size,
                                std::optional<Generator> generator,
                                std::optional<DType> dtype,
                                std::optional<int64_t> layout,
                                std::optional<Device> device,
                                std::optional<bool> pin_memory) {
    (void)generator; (void)layout; (void)pin_memory;
    return ops::randint(0, high, size, dtype.value_or(DType::Int64), device);
}

Tensor randint_low_generator_native(int64_t low, int64_t high,
                                    const std::vector<int64_t>& size,
                                    std::optional<Generator> generator,
                                    std::optional<DType> dtype,
                                    std::optional<int64_t> layout,
                                    std::optional<Device> device,
                                    std::optional<bool> pin_memory) {
    (void)generator; (void)layout; (void)pin_memory;
    return ops::randint(low, high, size, dtype.value_or(DType::Int64), device);
}

Tensor randint_like_low_dtype_native(const Tensor& self, int64_t low, int64_t high,
                                     std::optional<DType> dtype,
                                     std::optional<int64_t> layout,
                                     std::optional<Device> device,
                                     std::optional<bool> pin_memory,
                                     std::optional<int64_t> memory_format) {
    (void)layout; (void)pin_memory; (void)memory_format;
    Tensor t = ops::randint(low, high, self.shape(),
                            dtype.value_or(DType::Undefined), self.device());
    return t;
}

Tensor randint_like_tensor_native(const Tensor& self, const Tensor& high,
                                  std::optional<DType> dtype,
                                  std::optional<int64_t> layout,
                                  std::optional<Device> device,
                                  std::optional<bool> pin_memory,
                                  std::optional<int64_t> memory_format) {
    (void)layout; (void)pin_memory; (void)memory_format;
    return ops::randint_like(self, 0, high.item().to<int64_t>(),
                             dtype.value_or(DType::Undefined), device);
}

Tensor randint_like_generator_native(const Tensor& self, int64_t high,
                                     std::optional<Generator> generator,
                                     std::optional<DType> dtype,
                                     std::optional<int64_t> layout,
                                     std::optional<Device> device,
                                     std::optional<bool> pin_memory,
                                     std::optional<int64_t> memory_format) {
    (void)generator; (void)layout; (void)pin_memory; (void)memory_format;
    return ops::randint_like(self, 0, high, dtype.value_or(DType::Undefined), device);
}

Tensor randint_like_tensor_generator_native(const Tensor& self, const Tensor& high,
                                            std::optional<Generator> generator,
                                            std::optional<DType> dtype,
                                            std::optional<int64_t> layout,
                                            std::optional<Device> device,
                                            std::optional<bool> pin_memory,
                                            std::optional<int64_t> memory_format) {
    (void)generator; (void)layout; (void)pin_memory; (void)memory_format;
    return ops::randint_like(self, 0, high.item().to<int64_t>(),
                             dtype.value_or(DType::Undefined), device);
}

Tensor randint_like_low_generator_dtype_native(const Tensor& self, int64_t low, int64_t high,
                                               std::optional<Generator> generator,
                                               std::optional<DType> dtype,
                                               std::optional<int64_t> layout,
                                               std::optional<Device> device,
                                               std::optional<bool> pin_memory,
                                               std::optional<int64_t> memory_format) {
    (void)generator; (void)layout; (void)pin_memory; (void)memory_format;
    return ops::randint_like(self, low, high, dtype.value_or(DType::Undefined), device);
}

Tensor randn_like_generator_native(const Tensor& self,
                                   std::optional<Generator> generator,
                                   std::optional<DType> dtype,
                                   std::optional<int64_t> layout,
                                   std::optional<Device> device,
                                   std::optional<bool> pin_memory,
                                   std::optional<int64_t> memory_format) {
    (void)generator; (void)layout; (void)pin_memory; (void)memory_format;
    return ops::randn_like(self, dtype.value_or(DType::Undefined), device);
}

Tensor randperm_generator_native(int64_t n, std::optional<Generator> generator,
                                 std::optional<DType> dtype,
                                 std::optional<int64_t> layout,
                                 std::optional<Device> device,
                                 std::optional<bool> pin_memory) {
    (void)layout; (void)pin_memory;
    DType dt = dtype.value_or(DType::Int64);
    if (dt != DType::Int64 && dt != DType::Int32) {
        TP_THROW(NotImplementedError, "randperm() only supports Int64/Int32");
    }
    Device dev = device.has_value() ? *device : Device(DeviceType::CPU);
    if (!generator.has_value()) {
        return ops::randperm(n, dt, dev);
    }
    // The permutation values depend only on the generator stream, so the
    // Fisher-Yates pass runs on CPU against the explicit generator and the
    // result is then placed on the requested device; the draw order matches
    // the default-generator randperm kernel element for element.
    Tensor t({n}, dt, Device(DeviceType::CPU));
    if (n > 0) {
        if (dt == DType::Int64) {
            int64_t* data = t.data_ptr<int64_t>();
            for (int64_t i = 0; i < n; ++i) data[i] = i;
            for (int64_t i = 0; i < n - 1; ++i) {
                int64_t z = static_cast<int64_t>(
                    generator->random() % static_cast<uint32_t>(n - i));
                int64_t sav = data[i];
                data[i] = data[z + i];
                data[z + i] = sav;
            }
        } else {
            int32_t* data = t.data_ptr<int32_t>();
            for (int64_t i = 0; i < n; ++i) data[i] = static_cast<int32_t>(i);
            for (int64_t i = 0; i < n - 1; ++i) {
                int64_t z = static_cast<int64_t>(
                    generator->random() % static_cast<uint32_t>(n - i));
                int32_t sav = data[i];
                data[i] = data[z + i];
                data[z + i] = sav;
            }
        }
    }
    if (dev.type() == DeviceType::CPU) return t;
    return t.to(dev);
}

Tensor& random_to_native(Tensor& self, int64_t to, std::optional<Generator> generator) {
    (void)generator;
    return ops::random_(self, 0, to);
}

Tensor range_step_native(const Scalar& start, const Scalar& end, const Scalar& step,
                         std::optional<DType> dtype,
                         std::optional<int64_t> layout,
                         std::optional<Device> device,
                         std::optional<bool> pin_memory) {
    (void)layout; (void)pin_memory;
    return ops::range(start, end, step, dtype, device);
}

// ---- xlogy scalar overloads ---------------------------------------------------
Tensor xlogy_scalar_other_native(const Tensor& self, const Scalar& other) {
    return ops::xlogy(self, scalar_like(other, self));
}

Tensor xlogy_scalar_self_native(const Scalar& self, const Tensor& other) {
    return ops::xlogy(scalar_like(self, other), other);
}

Tensor& xlogy_out_scalar_other_native(const Tensor& self, const Scalar& other, Tensor& out) {
    return write_reduction_out("xlogy", xlogy_scalar_other_native(self, other), out);
}

Tensor& xlogy_out_scalar_self_native(const Scalar& self, const Tensor& other, Tensor& out) {
    return write_reduction_out("xlogy", xlogy_scalar_self_native(self, other), out);
}

Tensor& xlogy__scalar_other_native(Tensor& self, const Scalar& other) {
    ops::copy_(self, xlogy_scalar_other_native(self, other));
    return self;
}

Tensor float_power_scalar_native(const Scalar& self, const Tensor& exponent) {
    return ops::float_power(scalar_like(self, exponent), exponent);
}

Tensor float_power_tensor_scalar_native(const Tensor& self, const Scalar& exponent) {
    return ops::float_power(self, scalar_like(exponent, self));
}

// ---- fft out overloads ----------------------------------------------------
Tensor& fft_fft2_out_native(const Tensor& self, const std::optional<std::vector<int64_t>>& s,
                            const std::vector<int64_t>& dim,
                            const std::optional<std::string>& norm, Tensor& out) {
    return write_exact_out("fft_fft2", ops::fft_fft2(self, s, dim,
                                                      norm.value_or("backward")),
                           out);
}

Tensor& fft_ifft2_out_native(const Tensor& self, const std::optional<std::vector<int64_t>>& s,
                             const std::vector<int64_t>& dim,
                             const std::optional<std::string>& norm, Tensor& out) {
    return write_exact_out("fft_ifft2", ops::fft_ifft2(self, s, dim,
                                                       norm.value_or("backward")),
                           out);
}

Tensor& fft_rfft2_out_native(const Tensor& self, const std::optional<std::vector<int64_t>>& s,
                             const std::vector<int64_t>& dim,
                             const std::optional<std::string>& norm, Tensor& out) {
    return write_exact_out("fft_rfft2", ops::fft_rfft2(self, s, dim,
                                                       norm.value_or("backward")),
                           out);
}

Tensor& fft_irfft2_out_native(const Tensor& self, const std::optional<std::vector<int64_t>>& s,
                              const std::vector<int64_t>& dim,
                              const std::optional<std::string>& norm, Tensor& out) {
    return write_exact_out("fft_irfft2", ops::fft_irfft2(self, s, dim,
                                                        norm.value_or("backward")),
                           out);
}

// ---- upsample vec overloads -------------------------------------------------
namespace {

// Upstream .vec overloads accept either an explicit output size or per-dim
// scale factors; when only scales are given the target size is the floor of
// each spatial input extent times its factor.
std::vector<int64_t> upsample_out_size(const Tensor& input,
                                       const std::optional<std::vector<int64_t>>& output_size,
                                       const std::optional<std::vector<double>>& scale_factors) {
    if (output_size.has_value()) return *output_size;
    if (!scale_factors.has_value()) {
        TP_THROW(RuntimeError, "upsample: either output_size or scale_factors must be set");
    }
    const std::vector<int64_t> in = static_cast<std::vector<int64_t>>(input.shape());
    if (scale_factors->size() != in.size() - 2) {
        TP_THROW(RuntimeError, "upsample: scale_factors must match the spatial dims");
    }
    std::vector<int64_t> sizes = {in[0], in[1]};
    for (size_t i = 2; i < in.size(); ++i) {
        sizes.push_back(static_cast<int64_t>(std::floor(
            static_cast<double>(in[i]) * (*scale_factors)[i - 2])));
    }
    return sizes;
}

} // namespace

Tensor upsample_linear1d_vec_native(const Tensor& input,
                                    const std::optional<std::vector<int64_t>>& output_size,
                                    bool align_corners,
                                    const std::optional<std::vector<double>>& scale_factors) {
    const auto sz = upsample_out_size(input, output_size, scale_factors);
    return ops::upsample_linear1d(input, sz, align_corners);
}

Tensor upsample_nearest1d_vec_native(const Tensor& input,
                                     const std::optional<std::vector<int64_t>>& output_size,
                                     const std::optional<std::vector<double>>& scale_factors) {
    const auto sz = upsample_out_size(input, output_size, scale_factors);
    return ops::upsample_nearest1d(input, sz);
}

Tensor upsample_bilinear2d_vec_native(const Tensor& input,
                                      const std::optional<std::vector<int64_t>>& output_size,
                                      bool align_corners,
                                      const std::optional<std::vector<double>>& scale_factors) {
    const auto sz = upsample_out_size(input, output_size, scale_factors);
    return ops::upsample_bilinear2d(input, sz, align_corners);
}

Tensor upsample_bicubic2d_vec_native(const Tensor& input,
                                     const std::optional<std::vector<int64_t>>& output_size,
                                     bool align_corners,
                                     const std::optional<std::vector<double>>& scale_factors) {
    const auto sz = upsample_out_size(input, output_size, scale_factors);
    return ops::upsample_bicubic2d(input, sz, align_corners);
}

Tensor upsample_trilinear3d_vec_native(const Tensor& input,
                                       const std::optional<std::vector<int64_t>>& output_size,
                                       bool align_corners,
                                       const std::optional<std::vector<double>>& scale_factors) {
    const auto sz = upsample_out_size(input, output_size, scale_factors);
    return ops::upsample_trilinear3d(input, sz, align_corners);
}

Tensor upsample_nearest2d_vec_native(const Tensor& input,
                                     const std::optional<std::vector<int64_t>>& output_size,
                                     const std::optional<std::vector<double>>& scale_factors) {
    const auto sz = upsample_out_size(input, output_size, scale_factors);
    return ops::upsample_nearest2d(input, sz);
}

Tensor upsample_nearest3d_vec_native(const Tensor& input,
                                     const std::optional<std::vector<int64_t>>& output_size,
                                     const std::optional<std::vector<double>>& scale_factors) {
    const auto sz = upsample_out_size(input, output_size, scale_factors);
    return ops::upsample_nearest3d(input, sz);
}

// nearest-exact .vec entry points: same size/scale handling as the legacy
// nearest family, but they resolve to the pixel-center exact kernels.
Tensor _upsample_nearest_exact1d_vec_native(const Tensor& input,
                                            const std::optional<std::vector<int64_t>>& output_size,
                                            const std::optional<std::vector<double>>& scale_factors) {
    const auto sz = upsample_out_size(input, output_size, scale_factors);
    return ops::_upsample_nearest_exact1d(input, sz);
}

Tensor _upsample_nearest_exact2d_vec_native(const Tensor& input,
                                            const std::optional<std::vector<int64_t>>& output_size,
                                            const std::optional<std::vector<double>>& scale_factors) {
    const auto sz = upsample_out_size(input, output_size, scale_factors);
    return ops::_upsample_nearest_exact2d(input, sz);
}

Tensor _upsample_nearest_exact3d_vec_native(const Tensor& input,
                                            const std::optional<std::vector<int64_t>>& output_size,
                                            const std::optional<std::vector<double>>& scale_factors) {
    const auto sz = upsample_out_size(input, output_size, scale_factors);
    return ops::_upsample_nearest_exact3d(input, sz);
}

// ---- misc forwards ------------------------------------------------------------
Tensor& logit_backward_gi_native(const Tensor& grad_output, const Tensor& self,
                                 std::optional<double> eps, Tensor& grad_input) {
    return write_reduction_out(
        "logit_backward",
        ops::logit_backward(grad_output, self,
                            eps.has_value() ? std::optional<Scalar>(Scalar(*eps))
                                            : std::nullopt),
        grad_input);
}

Tensor quantize_per_tensor_tq_native(const Tensor& self, const Tensor& scale,
                                     const Tensor& zero_point, DType dtype) {
    const double sc = scale.item().toDouble();
    const int64_t zp = zero_point.item().to<int64_t>();
    return make_per_tensor_affine_quantizer(sc, zp, dtype)->quantize(self);
}

std::vector<Tensor> quantize_per_tensor_tensors_native(const std::vector<Tensor>& tensors,
                                                       const Tensor& scales,
                                                       const Tensor& zero_points, DType dtype) {
    std::vector<Tensor> out;
    out.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        out.push_back(quantize_per_tensor_tq_native(
            tensors[i], scales.select(0, static_cast<int64_t>(i)),
            zero_points.select(0, static_cast<int64_t>(i)), dtype));
    }
    return out;
}

std::vector<Tensor> dequantize_tensors_native(const std::vector<Tensor>& tensors) {
    std::vector<Tensor> out;
    out.reserve(tensors.size());
    for (const Tensor& t : tensors) {
        out.push_back(t.dequantize());
    }
    return out;
}

Tensor stft_center_native(const Tensor& self, int64_t n_fft,
                           std::optional<int64_t> hop_length,
                           std::optional<int64_t> win_length,
                           const std::optional<Tensor>& window, bool center,
                           const std::string& pad_mode, bool normalized,
                           std::optional<bool> onesided,
                           std::optional<bool> return_complex,
                           std::optional<bool> align_to_window) {
    (void)align_to_window;
    return ops::stft(self, n_fft, hop_length, win_length, window, center, pad_mode, normalized,
                     onesided.value_or(true), return_complex.value_or(true));
}

TENSORPLAY_LIBRARY_IMPL(Composite, VariantHandOps) {
    m.impl("norm.Scalar", norm_scalar_native);
    m.impl("norm.ScalarOpt_dim", norm_scalar_opt_dim_native);
    m.impl("norm.ScalarOpt_dtype", norm_scalar_opt_dtype_native);
    m.impl("norm.ScalarOpt_dim_dtype", norm_scalar_opt_dim_dtype_native);
    m.impl("norm.out", norm_out_native);
    m.impl("norm.dtype_out", norm_dtype_out_native);

    m.impl("prod.dim_int", prod_dim_int_native);
    m.impl("prod.int_out", prod_int_out_native);
    m.impl("std.correction", std_correction_native);
    m.impl("std.correction_out", std_correction_out_native);
    m.impl("std.out", std_out_native);
    m.impl("var.correction", var_correction_native);
    m.impl("var.correction_out", var_correction_out_native);
    m.impl("var.out", var_out_native);
    m.impl("std_mean.correction", std_mean_correction_native);
    m.impl("var_mean.correction", var_mean_correction_native);
    m.impl("median.dim", median_dim_native);
    m.impl("median.dim_values", median_dim_values_native);

    m.impl("sort.stable", sort_stable_native);
    m.impl("sort.values_stable", sort_values_stable_native);
    m.impl("argsort.stable", argsort_stable_native);
    m.impl("argsort.stable_out", argsort_stable_out_native);
    // duplicate of generated out wrapper // m.impl("topk.values", topk_values_native);
    m.impl("logsumexp.out", logsumexp_out_native);

    m.impl("movedim.int", movedim_int_native);
    m.impl("narrow.Tensor", narrow_tensor_native);
    m.impl("max.other", max_other_native);
    m.impl("aminmax.out", aminmax_out_native);
    m.impl("round.decimals", round_decimals_native);
    m.impl("round_.decimals", round__decimals_native);
    m.impl("nan_to_num.out", nan_to_num_out_native);
    m.impl("nanmean.out", nanmean_out_native);

    m.impl("trapezoid.dx", trapezoid_dx_native);
    m.impl("trapezoid.x", trapezoid_x_native);
    m.impl("cumulative_trapezoid.dx", cumulative_trapezoid_dx_native);
    m.impl("cumulative_trapezoid.x", cumulative_trapezoid_x_native);
    m.impl("gradient.scalarint", gradient_scalarint_native);
    m.impl("gradient.scalararray", gradient_scalararray_native);
    m.impl("gradient.array", gradient_array_native);
    m.impl("gradient.scalarrayint", gradient_scalarrayint_native);
    m.impl("gradient.scalarrayarray", gradient_scalarrayarray_native);
    m.impl("gradient.tensorarrayint", gradient_tensorarrayint_native);
    m.impl("quantile.scalar", quantile_scalar_native);
    m.impl("nanquantile.scalar", nanquantile_scalar_native);
    m.impl("ormqr", ormqr_composite);
    m.impl("geqrf", geqrf_composite);
    m.impl("linalg_vdot", linalg_vdot_composite);

    m.impl("mm.dtype", mm_dtype_native);
    m.impl("mm.dtype_out", mm_dtype_out_native);
    m.impl("addmm.dtype", addmm_dtype_native);
    m.impl("addmm.dtype_out", addmm_dtype_out_native);
    m.impl("bmm.dtype", bmm_dtype_native);
    m.impl("bmm.dtype_out", bmm_dtype_out_native);
    m.impl("baddbmm.dtype", baddbmm_dtype_native);
    m.impl("baddbmm.dtype_out", baddbmm_dtype_out_native);

    m.impl("conv1d.padding", conv1d_padding_native);
    m.impl("conv2d.padding", conv2d_padding_native);
    m.impl("conv3d.padding", conv3d_padding_native);
    m.impl("_convolution.deprecated", _convolution_deprecated_native);

    m.impl("adaptive_max_pool2d_backward.grad_input", adaptive_max_pool2d_backward_gi_native);
    m.impl("adaptive_max_pool3d_backward.grad_input", adaptive_max_pool3d_backward_gi_native);
    // duplicates of generated out wrappers
    // m.impl("max_pool2d_with_indices_backward.grad_input", max_pool2d_with_indices_backward_gi_native);
    // m.impl("max_pool3d_with_indices_backward.grad_input", max_pool3d_with_indices_backward_gi_native);

    m.impl("nll_loss.out", nll_loss_out_native);
    m.impl("nll_loss2d.out", nll_loss2d_out_native);

    m.impl("gru.input", gru_input_native);
    m.impl("rnn_relu.input", rnn_relu_input_native);
    m.impl("rnn_tanh.input", rnn_tanh_input_native);
    // dropped duplicate: lstm.input // m.impl("lstm.input", lstm_input_native);
    m.impl("gru.data", gru_data_native);
    m.impl("rnn_relu.data", rnn_relu_data_native);
    m.impl("rnn_tanh.data", rnn_tanh_data_native);
    m.impl("lstm.data", lstm_data_native);

    m.impl("to_sparse_csc", to_sparse_csc_native);
    m.impl("_to_sparse_csc", _to_sparse_csc_native);
    m.impl("to_sparse_bsr", to_sparse_bsr_native);
    m.impl("_to_sparse_bsr", _to_sparse_bsr_native);
    m.impl("to_sparse_bsc", to_sparse_bsc_native);
    m.impl("_to_sparse_bsc", _to_sparse_bsc_native);
    m.impl("to_sparse.sparse_dim", to_sparse_sparse_dim_native);
    m.impl("_to_sparse.sparse_dim", _to_sparse_sparse_dim_native);
    m.impl("sparse_coo_tensor.indices", sparse_coo_tensor_indices_native);
    m.impl("sparse_coo_tensor.indices_size", sparse_coo_tensor_indices_size_native);
    m.impl("sparse_coo_tensor.size", sparse_coo_tensor_size_native);
    m.impl("sparse_csr_tensor.crow_col_value", sparse_csr_tensor_crow_col_value_native);
    m.impl("sparse_csr_tensor.crow_col_value_size",
           sparse_csr_tensor_crow_col_value_size_native);
    m.impl("_sparse_csr_tensor_unsafe", sparse_csr_tensor_unsafe_native);
    m.impl("_validate_sparse_csr_tensor_args",
           validate_sparse_csr_tensor_args_native);
    m.impl("sparse_compressed_tensor.comp_plain_value_size",
           sparse_compressed_tensor_comp_plain_value_size_native);
    m.impl("sparse_compressed_tensor.comp_plain_value",
           sparse_compressed_tensor_comp_plain_value_native);
    m.impl("_sparse_compressed_tensor_unsafe",
           sparse_compressed_tensor_unsafe_native);
    m.impl("sparse_csc_tensor.ccol_row_value_size",
           sparse_csc_tensor_ccol_row_value_size_native);
    m.impl("sparse_csc_tensor.ccol_row_value",
           sparse_csc_tensor_ccol_row_value_native);
    m.impl("_sparse_csc_tensor_unsafe", sparse_csc_tensor_unsafe_native);
    m.impl("sparse_bsr_tensor.crow_col_value_size",
           sparse_bsr_tensor_crow_col_value_size_native);
    m.impl("sparse_bsr_tensor.crow_col_value",
           sparse_bsr_tensor_crow_col_value_native);
    m.impl("_sparse_bsr_tensor_unsafe", sparse_bsr_tensor_unsafe_native);
    m.impl("sparse_bsc_tensor.ccol_row_value_size",
           sparse_bsc_tensor_ccol_row_value_size_native);
    m.impl("sparse_bsc_tensor.ccol_row_value",
           sparse_bsc_tensor_ccol_row_value_native);
    m.impl("_sparse_bsc_tensor_unsafe", sparse_bsc_tensor_unsafe_native);
    m.impl("_validate_sparse_compressed_tensor_args",
           validate_sparse_compressed_tensor_args_native);
    m.impl("_validate_sparse_csc_tensor_args",
           validate_sparse_csc_tensor_args_native);
    m.impl("_validate_sparse_bsr_tensor_args",
           validate_sparse_bsr_tensor_args_native);
    m.impl("_validate_sparse_bsc_tensor_args",
           validate_sparse_bsc_tensor_args_native);
    m.impl("_sparse_compressed_tensor_with_dims",
           sparse_compressed_tensor_with_dims_native);
    m.impl("multinomial.out", multinomial_out_native);
    m.impl("linalg_lu_solve.out", linalg_lu_solve_out_native);
    m.impl("split_with_sizes_copy.out", split_with_sizes_copy_out_native);

    m.impl("rand.generator", rand_generator_native);
    m.impl("rand_like.generator", rand_like_generator_native);
    m.impl("randint.low", randint_low_native);
    m.impl("randint.generator", randint_generator_native);
    m.impl("randint.low_generator", randint_low_generator_native);
    m.impl("randint_like.low_dtype", randint_like_low_dtype_native);
    m.impl("randint_like.Tensor", randint_like_tensor_native);
    m.impl("randint_like.generator", randint_like_generator_native);
    m.impl("randint_like.Tensor_generator", randint_like_tensor_generator_native);
    m.impl("randint_like.low_generator_dtype", randint_like_low_generator_dtype_native);
    m.impl("randn_like.generator", randn_like_generator_native);
    m.impl("randperm.generator", randperm_generator_native);
    m.impl("random_.to", random_to_native);
    m.impl("range.step", range_step_native);

    m.impl("xlogy.Scalar_Other", xlogy_scalar_other_native);
    m.impl("xlogy.Scalar_Self", xlogy_scalar_self_native);
    m.impl("xlogy.OutScalar_Other", xlogy_out_scalar_other_native);
    m.impl("xlogy.OutScalar_Self", xlogy_out_scalar_self_native);
    m.impl("xlogy_.Scalar_Other", xlogy__scalar_other_native);
    m.impl("float_power.Scalar", float_power_scalar_native);
    m.impl("float_power.Tensor_Scalar", float_power_tensor_scalar_native);

    // dropped duplicate: fft_fft2.out // m.impl("fft_fft2.out", fft_fft2_out_native);
    // dropped duplicate: fft_ifft2.out // m.impl("fft_ifft2.out", fft_ifft2_out_native);
    // dropped duplicate: fft_rfft2.out // m.impl("fft_rfft2.out", fft_rfft2_out_native);
    // dropped duplicate: fft_irfft2.out // m.impl("fft_irfft2.out", fft_irfft2_out_native);

    m.impl("upsample_linear1d.vec", upsample_linear1d_vec_native);
    m.impl("upsample_nearest1d.vec", upsample_nearest1d_vec_native);
    m.impl("upsample_bilinear2d.vec", upsample_bilinear2d_vec_native);
    m.impl("upsample_bicubic2d.vec", upsample_bicubic2d_vec_native);
    m.impl("upsample_trilinear3d.vec", upsample_trilinear3d_vec_native);
    m.impl("upsample_nearest2d.vec", upsample_nearest2d_vec_native);
    m.impl("upsample_nearest3d.vec", upsample_nearest3d_vec_native);
    m.impl("_upsample_nearest_exact1d.vec", _upsample_nearest_exact1d_vec_native);
    m.impl("_upsample_nearest_exact2d.vec", _upsample_nearest_exact2d_vec_native);
    m.impl("_upsample_nearest_exact3d.vec", _upsample_nearest_exact3d_vec_native);

    m.impl("logit_backward.grad_input", logit_backward_gi_native);
    m.impl("quantize_per_tensor.tensor_qparams", quantize_per_tensor_tq_native);
    m.impl("quantize_per_tensor.tensors", quantize_per_tensor_tensors_native);
    m.impl("dequantize.tensors", dequantize_tensors_native);
    // dropped duplicate: stft.center // m.impl("stft.center", stft_center_native);
}

} // namespace composite
} // namespace tensorplay
