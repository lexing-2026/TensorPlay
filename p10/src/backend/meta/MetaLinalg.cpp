// Shape-only kernels for the linear algebra family on the meta device.
// A factorization's observable result is its extents: the factor planes
// repeat the operand's batch geometry, the status planes are integers,
// and a real operand's eigenvectors answer complex tensors.  The
// squareness, floating-dtype and driver/mode contracts are enforced
// here; the numeric failures that depend on stored values cannot fire.
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
// Linear algebra.  A factorization's observable result is its extents: the
// packed factor copies the input, the pivot sequence follows the shorter side
// and every batch member owns one status entry.  Everything else on this
// device is decided by values (singularity, convergence, pivot validity), so
// a status plane carries no entry and a value-dependent failure cannot fire.
// ---------------------------------------------------------------------------
bool is_linalg_dtype(DType dtype) {
    return dtype == DType::Float32 || dtype == DType::Float64 ||
           dtype == DType::ComplexFloat || dtype == DType::ComplexDouble;
}

void check_linalg_dtype(DType dtype) {
    if (is_linalg_dtype(dtype)) return;
    TP_THROW(NotImplementedError, "unsupported dtype ", scalarTypeName(dtype),
             " for linalg on meta");
}

void check_complex_linalg_dtype(DType dtype) {
    if (dtype == DType::ComplexFloat || dtype == DType::ComplexDouble) return;
    TP_THROW(NotImplementedError, "unsupported dtype ", scalarTypeName(dtype),
             " for complex linalg on meta");
}

void check_real_linalg_dtype(DType dtype) {
    if (dtype == DType::Float32 || dtype == DType::Float64) return;
    TP_THROW(NotImplementedError, "unsupported dtype ", scalarTypeName(dtype),
             " for linalg on meta (only float32/float64 are implemented)");
}

std::vector<int64_t> drop_last_two(std::vector<int64_t> shape) {
    shape.resize(shape.size() >= 2 ? shape.size() - 2 : 0);
    return shape;
}

std::vector<int64_t> linalg_batch_shape(const Tensor& t) {
    return drop_last_two(shape_of(t));
}

std::vector<int64_t> append_dims(std::vector<int64_t> shape,
                                 std::initializer_list<int64_t> tail) {
    shape.insert(shape.end(), tail);
    return shape;
}

void check_linalg_matrix(const Tensor& t, const char* fn,
                         const char* arg = "A") {
    if (t.dim() < 2) {
        TP_THROW(RuntimeError, fn, ": The input tensor ", arg,
                 " must have at least 2 dimensions.");
    }
}

void check_linalg_square(const Tensor& t, const char* fn,
                         const char* arg = "A") {
    check_linalg_matrix(t, fn, arg);
    if (t.size(-1) != t.size(-2)) {
        TP_THROW(RuntimeError, fn, ": ", arg,
                 " must be batches of square matrices, but they are ",
                 t.size(-2), " by ", t.size(-1), " matrices");
    }
}

// Right-aligned batch broadcast; the reported dimension index counts from the
// start of the batch rank, not of the full shape.
std::vector<int64_t> linalg_broadcast(const std::vector<int64_t>& a,
                                      const std::vector<int64_t>& b) {
    const size_t rank = std::max(a.size(), b.size());
    std::vector<int64_t> out(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
        const int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
        if (da != db && da != 1 && db != 1) {
            TP_THROW(RuntimeError, "The size of tensor a (", da,
                     ") must match the size of tensor b (", db,
                     ") at non-singleton dimension ", i);
        }
        out[i] = std::max(da, db);
    }
    return out;
}

// A right-hand side whose leading extents mirror the coefficient matrix is a
// stack of vectors, one per batch member, rather than a stack of matrices.
bool linalg_vector_rhs(const Tensor& A, const Tensor& B) {
    if (B.dim() == 1) return true;
    if (A.dim() - 1 != B.dim()) return false;
    for (int64_t i = 0; i < A.dim() - 1; ++i) {
        if (A.size(i) != B.size(i)) return false;
    }
    return true;
}

// The right-hand side with a trailing column added when it is a vector.
std::vector<int64_t> linalg_rhs_shape(const Tensor& B, bool vector_case) {
    std::vector<int64_t> b2 = shape_of(B);
    if (vector_case) b2.push_back(1);
    return b2;
}

void check_linalg_same_device(const Tensor& a, const Tensor& b,
                              const char* fn) {
    if (a.device() != b.device()) {
        TP_THROW(DeviceMismatchError, fn,
                 ": A and B must be on the same device");
    }
}

void check_linalg_same_dtype(const Tensor& a, const Tensor& b,
                             const char* fn) {
    if (a.dtype() != b.dtype()) {
        TP_THROW(RuntimeError, fn,
                 ": A and B must have the same dtype, but got ",
                 scalarTypeName(a.dtype()), " and ", scalarTypeName(b.dtype()));
    }
}

// Reading the status planes would need element data; a meta factorization
// leaves them at zero and never reports a failure.
void linalg_check_errors_kernel(const Tensor& /*infos*/,
                                const std::string& /*api_name*/,
                                bool /*is_matrix*/) {}

std::tuple<Tensor, Tensor> cholesky_ex_kernel(const Tensor& A,
                                              bool /*upper*/,
                                              bool check_errors) {
    const char* api = check_errors ? "linalg.cholesky_ex" : "linalg.cholesky";
    check_linalg_square(A, api);
    check_linalg_dtype(A.dtype());
    return {empty_meta(shape_of(A), A.dtype(), A.device()),
            empty_meta(linalg_batch_shape(A), DType::Int32, A.device())};
}

std::tuple<Tensor, Tensor> inv_ex_kernel(const Tensor& A,
                                         bool /*check_errors*/) {
    check_linalg_dtype(A.dtype());
    // The identity right-hand side has the input's shape, so the square
    // requirement reports the solve entry point.
    check_linalg_square(A, "linalg.solve_ex");
    return {empty_meta(shape_of(A), A.dtype(), A.device()),
            empty_meta(linalg_batch_shape(A), DType::Int32, A.device())};
}

std::tuple<Tensor, Tensor, Tensor> det_kernel(const Tensor& A) {
    check_linalg_square(A, "linalg.det");
    check_linalg_dtype(A.dtype());
    const auto batch = linalg_batch_shape(A);
    const int64_t k = std::min(A.size(-2), A.size(-1));
    return {empty_meta(batch, A.dtype(), A.device()),
            empty_meta(shape_of(A), A.dtype(), A.device()),
            empty_meta(append_dims(batch, {k}), DType::Int32, A.device())};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> slogdet_kernel(const Tensor& A) {
    check_linalg_square(A, "linalg.slogdet");
    check_linalg_dtype(A.dtype());
    const auto batch = linalg_batch_shape(A);
    const int64_t k = std::min(A.size(-2), A.size(-1));
    return {empty_meta(batch, A.dtype(), A.device()),
            empty_meta(batch, toRealValueType(A.dtype()), A.device()),
            empty_meta(shape_of(A), A.dtype(), A.device()),
            empty_meta(append_dims(batch, {k}), DType::Int32, A.device())};
}

std::tuple<Tensor, Tensor, Tensor> lu_factor_ex_kernel(const Tensor& A,
                                                       bool /*pivot*/,
                                                       bool /*check_errors*/) {
    check_linalg_matrix(A, "linalg.lu_factor_ex");
    check_linalg_dtype(A.dtype());
    const auto batch = linalg_batch_shape(A);
    const int64_t k = std::min(A.size(-2), A.size(-1));
    return {empty_meta(shape_of(A), A.dtype(), A.device()),
            empty_meta(append_dims(batch, {k}), DType::Int32, A.device()),
            empty_meta(batch, DType::Int32, A.device())};
}

std::tuple<Tensor, Tensor, Tensor> lu_kernel(const Tensor& A, bool pivot) {
    if (!pivot) {
        TP_THROW(RuntimeError,
                 "linalg.lu: LU without pivoting is not implemented");
    }
    check_linalg_matrix(A, "linalg.lu");
    check_linalg_dtype(A.dtype());
    const auto batch = linalg_batch_shape(A);
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    const int64_t k = std::min(m, n);
    return {empty_meta(append_dims(batch, {m, m}), A.dtype(), A.device()),
            empty_meta(append_dims(batch, {m, k}), A.dtype(), A.device()),
            empty_meta(append_dims(batch, {k, n}), A.dtype(), A.device())};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> solve_ex_kernel(
        const Tensor& A, const Tensor& B, bool left, bool /*check_errors*/) {
    const char* api = "linalg.solve_ex";
    check_linalg_square(A, api, "A");
    const bool vector_case = linalg_vector_rhs(A, B);
    const std::vector<int64_t> b2 = linalg_rhs_shape(B, vector_case);
    if (b2.size() < 2) {
        TP_THROW(RuntimeError, api,
                 ": The input tensor B must have at least 2 dimensions.");
    }
    check_linalg_same_device(A, B, api);
    check_linalg_same_dtype(A, B, api);
    const int64_t brows = b2[b2.size() - 2];
    const int64_t bcols = b2[b2.size() - 1];
    if (!(left ? A.size(-2) == brows : A.size(-1) == bcols)) {
        TP_THROW(RuntimeError, api,
                 ": Incompatible shapes of A and B for the equation ",
                 left ? "AX = B" : "XA = B", " (", A.size(-2), "x",
                 A.size(-1), " and ", brows, "x", bcols, ")");
    }
    if (!left && vector_case) {
        TP_THROW(RuntimeError,
                 "linalg.solve: Vector right-hand sides are not supported "
                 "when left is false");
    }
    const auto batch = linalg_broadcast(linalg_batch_shape(A),
                                        drop_last_two(b2));
    std::vector<int64_t> result = append_dims(batch, {brows, bcols});
    if (vector_case) result.pop_back();
    const int64_t n = A.size(-2);
    return {empty_meta(result, A.dtype(), A.device()),
            empty_meta(append_dims(batch, {n, n}), A.dtype(), A.device()),
            empty_meta(append_dims(batch, {n}), DType::Int32, A.device()),
            empty_meta(batch, DType::Int32, A.device())};
}

Tensor solve_triangular_kernel(const Tensor& A, const Tensor& B,
                               bool /*upper*/, bool left,
                               bool /*unitriangular*/) {
    const char* api = "linalg.solve_triangular";
    check_linalg_matrix(A, api, "A");
    check_linalg_matrix(B, api, "B");
    check_linalg_same_device(A, B, api);
    check_linalg_same_dtype(A, B, api);
    if (!(left ? A.size(-1) == B.size(-2) : A.size(-1) == B.size(-1))) {
        TP_THROW(RuntimeError, api,
                 ": Incompatible shapes of A and B for the equation ",
                 left ? "AX = B" : "XA = B");
    }
    check_linalg_dtype(A.dtype());
    const auto batch = linalg_broadcast(linalg_batch_shape(A),
                                        linalg_batch_shape(B));
    return empty_meta(append_dims(batch, {B.size(-2), B.size(-1)}),
                      B.dtype(), B.device());
}

Tensor lu_solve_kernel(const Tensor& LU, const Tensor& pivots,
                       const Tensor& B, bool left, bool /*adjoint*/) {
    const char* api = "linalg.lu_solve";
    check_linalg_square(LU, api, "LU");
    if (B.dim() < 1) {
        TP_THROW(RuntimeError, api, ": B must have at least 1 dimension");
    }
    if (LU.device() != B.device()) {
        TP_THROW(DeviceMismatchError, api,
                 ": LU and B must be on the same device");
    }
    if (LU.dtype() != B.dtype()) {
        TP_THROW(RuntimeError, api,
                 ": LU and B must have the same dtype, but got ",
                 scalarTypeName(LU.dtype()), " and ",
                 scalarTypeName(B.dtype()));
    }
    const bool vector_case = linalg_vector_rhs(LU, B);
    const std::vector<int64_t> b2 = linalg_rhs_shape(B, vector_case);
    if (pivots.dtype() != DType::Int32 || pivots.device() != LU.device()) {
        TP_THROW(RuntimeError, api,
                 ": pivots must be Int32 on the same device as LU");
    }
    if (pivots.dim() < 1 || pivots.size(-1) != LU.size(-1)) {
        TP_THROW(RuntimeError, api,
                 ": pivots must contain one entry per matrix column");
    }
    const int64_t brows = b2[b2.size() - 2];
    const int64_t bcols = b2[b2.size() - 1];
    if (!(left ? LU.size(-2) == brows : LU.size(-1) == bcols)) {
        TP_THROW(RuntimeError, api,
                 ": Incompatible shapes of LU and B for the equation ",
                 left ? "AX = B" : "XA = B", " (", LU.size(-2), "x",
                 LU.size(-1), " and ", brows, "x", bcols, ")");
    }
    check_linalg_dtype(LU.dtype());
    const auto batch = linalg_broadcast(linalg_batch_shape(LU),
                                        drop_last_two(b2));
    std::vector<int64_t> result = append_dims(batch, {brows, bcols});
    if (vector_case) result.pop_back();
    return empty_meta(result, LU.dtype(), LU.device());
}

std::tuple<Tensor, Tensor> eigh_kernel(const Tensor& A,
                                       const std::string& UPLO,
                                       bool /*compute_v*/) {
    if (UPLO != "U" && UPLO != "L") {
        TP_THROW(RuntimeError,
                 "linalg.eigh: UPLO argument must be 'U' or 'L', got ", UPLO);
    }
    check_linalg_square(A, "linalg.eigh");
    check_linalg_dtype(A.dtype());
    // The vector plane is filled with the input's own extent even when only
    // the eigenvalues were asked for.
    return {empty_meta(append_dims(linalg_batch_shape(A), {A.size(-1)}),
                       toRealValueType(A.dtype()), A.device()),
            empty_meta(shape_of(A), A.dtype(), A.device())};
}

DType eig_values_dtype(DType dtype) {
    if (isComplexType(dtype)) return dtype;
    return dtype == DType::Float32 ? DType::ComplexFloat
                                   : DType::ComplexDouble;
}

void check_eig_input(const Tensor& A) {
    check_linalg_square(A, "linalg.eig");
    if (isComplexType(A.dtype())) {
        check_complex_linalg_dtype(A.dtype());
    } else {
        check_real_linalg_dtype(A.dtype());
    }
}

std::tuple<Tensor, Tensor> eig_kernel(const Tensor& A) {
    check_eig_input(A);
    const DType values_dtype = eig_values_dtype(A.dtype());
    return {empty_meta(append_dims(linalg_batch_shape(A), {A.size(-1)}),
                       values_dtype, A.device()),
            empty_meta(shape_of(A), values_dtype, A.device())};
}

Tensor eigvals_kernel(const Tensor& A) {
    return std::get<0>(eig_kernel(A));
}

std::tuple<Tensor, Tensor, Tensor> svd_kernel(
        const Tensor& A, bool full_matrices, bool compute_uv,
        const std::optional<std::string>& driver) {
    if (driver.has_value()) {
        TP_THROW(RuntimeError,
                 "linalg.svd: keyword argument `driver=` is only supported on "
                 "CUDA inputs");
    }
    check_linalg_matrix(A, "linalg.svd");
    // An empty operand is never factored, so its element type does not have
    // to be one the factorization supports.
    if (A.numel() != 0) check_linalg_dtype(A.dtype());
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    const int64_t k = std::min(m, n);
    const auto batch = linalg_batch_shape(A);
    Tensor U;
    Tensor Vh;
    if (compute_uv) {
        U = empty_meta(append_dims(batch, {m, full_matrices ? m : k}),
                       A.dtype(), A.device());
        Vh = empty_meta(append_dims(batch, {full_matrices ? n : k, n}),
                        A.dtype(), A.device());
    } else {
        U = empty_meta(append_dims(batch, {m, 0}), A.dtype(), A.device());
        Vh = empty_meta(append_dims(batch, {0, n}), A.dtype(), A.device());
    }
    return {U,
            empty_meta(append_dims(batch, {k}), toRealValueType(A.dtype()),
                       A.device()),
            Vh};
}

std::tuple<Tensor, Tensor> qr_kernel(const Tensor& A,
                                     const std::string& mode) {
    check_linalg_matrix(A, "linalg.qr");
    if (mode != "reduced" && mode != "complete" && mode != "r") {
        TP_THROW(RuntimeError, "linalg.qr: mode '", mode,
                 "' not recognized. Mode must be one of 'reduced', "
                 "'complete' or 'r'");
    }
    check_linalg_dtype(A.dtype());
    const bool compute_q = mode != "r";
    const bool reduced = mode == "reduced" || !compute_q;
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    const int64_t k = std::min(m, n);
    const auto batch = linalg_batch_shape(A);
    Tensor Q;
    if (compute_q) {
        Q = empty_meta(append_dims(batch, {m, reduced ? k : m}), A.dtype(),
                       A.device());
    } else {
        Q = empty_meta({0}, A.dtype(), A.device());
    }
    const int64_t rrows = reduced ? k : m;
    return {Q, empty_meta(append_dims(batch, {rrows, n}), A.dtype(),
                          A.device())};
}

Tensor householder_product_kernel(const Tensor& input, const Tensor& tau) {
    const char* api = "linalg.householder_product";
    check_linalg_matrix(input, api);
    if (input.size(-2) < input.size(-1)) {
        TP_THROW(RuntimeError, api,
                 ": If input has size (..., m, n), n must be less than or "
                 "equal to m, but got n = ", input.size(-1), " and m = ",
                 input.size(-2));
    }
    if (tau.dim() < 1 || input.size(-1) < tau.size(-1)) {
        TP_THROW(RuntimeError, api,
                 ": input.shape[-1] must be greater than or equal to "
                 "tau.shape[-1]");
    }
    if (input.dim() - tau.dim() != 1) {
        TP_THROW(RuntimeError, api,
                 ": Expected tau to have one dimension less than input, but "
                 "got tau.ndim equal to ", tau.dim(),
                 " and input.ndim is equal to ", input.dim());
    }
    for (int64_t i = 0; i < input.dim() - 2; ++i) {
        if (input.size(i) != tau.size(i)) {
            TP_THROW(RuntimeError, api,
                     ": Expected batch dimensions of tau to be equal to "
                     "input.shape[:-2]");
        }
    }
    if (tau.size(-1) != std::min(input.size(-2), input.size(-1))) {
        TP_THROW(RuntimeError, api,
                 ": If tau has size (..., k), then when input has size "
                 "(..., m, n) we require k == min(m, n)");
    }
    if (tau.dtype() != input.dtype()) {
        TP_THROW(RuntimeError, api,
                 ": input and tau must have the same dtype");
    }
    if (tau.device() != input.device()) {
        TP_THROW(DeviceMismatchError, api,
                 ": input and tau must be on the same device");
    }
    check_linalg_dtype(input.dtype());
    return empty_meta(shape_of(input), input.dtype(), input.device());
}

std::tuple<Tensor, Tensor, Tensor, Tensor> lstsq_kernel(
        const Tensor& A, const Tensor& B, std::optional<double> /*rcond*/,
        const std::optional<std::string>& driver_opt) {
    const char* api = "linalg.lstsq";
    check_linalg_matrix(A, api);
    if (B.dim() < 1) {
        TP_THROW(RuntimeError, api, ": B must have at least 1 dimension");
    }
    const int64_t dim_diff = A.dim() - B.dim();
    if (dim_diff < 0 || dim_diff > 1) {
        TP_THROW(RuntimeError, api,
                 ": A and B must have compatible numbers of dimensions");
    }
    check_linalg_same_device(A, B, api);
    check_linalg_same_dtype(A, B, api);
    const bool vector_case = linalg_vector_rhs(A, B);
    const std::vector<int64_t> b2 = linalg_rhs_shape(B, vector_case);
    if (A.size(-2) != b2[b2.size() - 2]) {
        TP_THROW(RuntimeError, api,
                 ": A and B have incompatible row dimensions: ",
                 A.size(-2), " and ", b2[b2.size() - 2]);
    }
    std::string driver = driver_opt.value_or("gelsy");
    std::transform(driver.begin(), driver.end(), driver.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (driver != "gels" && driver != "gelsy" && driver != "gelsd" &&
        driver != "gelss") {
        TP_THROW(RuntimeError, api,
                 ": driver must be one of gels, gelsy, gelsd, or gelss");
    }
    check_linalg_dtype(A.dtype());
    const auto batch = linalg_broadcast(linalg_batch_shape(A),
                                        drop_last_two(b2));
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    const int64_t nrhs = b2[b2.size() - 1];
    const DType real = toRealValueType(A.dtype());
    std::vector<int64_t> solution = append_dims(batch, {n, nrhs});
    if (vector_case) solution.pop_back();
    // The rank is a value, so the residual plane assumes a full-rank least
    // squares problem and reports the extent that rank would produce.
    Tensor residuals =
        (m > n && driver != "gelsy")
            ? empty_meta(append_dims(batch, {nrhs}), real, A.device())
            : empty_meta({0}, real, A.device());
    Tensor rank = driver == "gels"
        ? empty_meta({0}, DType::Int64, A.device())
        : empty_meta(batch, DType::Int64, A.device());
    Tensor singular_values =
        (driver == "gelsd" || driver == "gelss")
            ? empty_meta(append_dims(batch, {std::min(m, n)}), real,
                         A.device())
            : empty_meta({0}, real, A.device());
    return {empty_meta(solution, A.dtype(), A.device()), residuals, rank,
            singular_values};
}

std::tuple<Tensor, Tensor, Tensor> ldl_factor_ex_kernel(
        const Tensor& A, bool /*hermitian*/, bool check_errors) {
    const char* api =
        check_errors ? "linalg.ldl_factor_ex" : "linalg.ldl_factor";
    check_linalg_square(A, api);
    check_linalg_dtype(A.dtype());
    const auto batch = linalg_batch_shape(A);
    return {empty_meta(shape_of(A), A.dtype(), A.device()),
            empty_meta(append_dims(batch, {A.size(-1)}), DType::Int32,
                       A.device()),
            empty_meta(batch, DType::Int32, A.device())};
}

std::tuple<Tensor, Tensor> ldl_factor_only_kernel(const Tensor& A,
                                                  bool hermitian) {
    auto values = ldl_factor_ex_kernel(A, hermitian, false);
    return {std::get<0>(values), std::get<1>(values)};
}

Tensor ldl_solve_kernel(const Tensor& LD, const Tensor& pivots,
                        const Tensor& B, bool /*hermitian*/) {
    const char* api = "linalg.ldl_solve";
    check_linalg_square(LD, api, "LD");
    check_linalg_matrix(B, api, "B");
    const auto batch = linalg_broadcast(linalg_batch_shape(LD),
                                        linalg_batch_shape(B));
    check_linalg_dtype(LD.dtype());
    return empty_meta(append_dims(batch, {B.size(-2), B.size(-1)}),
                      B.dtype(), B.device());
}

Tensor diagonal_kernel(const Tensor& A, int64_t offset, int64_t dim1,
                       int64_t dim2) {
    const char* api = "linalg.diagonal";
    check_linalg_matrix(A, api);
    const int64_t ndim = A.dim();
    const auto norm_dim = [&](int64_t d) {
        const int64_t r = d < 0 ? d + ndim : d;
        if (r < 0 || r >= ndim) {
            TP_THROW(RuntimeError,
                     "Dimension out of range (expected to be in range of [",
                     -ndim, ", ", ndim - 1, "], but got ", d, ")");
        }
        return r;
    };
    const int64_t d1_axis = norm_dim(dim1);
    const int64_t d2_axis = norm_dim(dim2);
    if (d1_axis == d2_axis) {
        TP_THROW(RuntimeError,
                 "linalg.diagonal: dimension 1 and dimension 2 cannot be "
                 "equal");
    }
    check_linalg_dtype(A.dtype());
    const int64_t d1 = A.size(d1_axis);
    const int64_t d2 = A.size(d2_axis);
    const int64_t diag_len =
        offset >= 0 ? std::max<int64_t>(std::min(d1, d2 - offset), 0)
                    : std::max<int64_t>(std::min(d1 + offset, d2), 0);
    std::vector<int64_t> out;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != d1_axis && i != d2_axis) out.push_back(A.size(i));
    }
    out.push_back(diag_len);
    return empty_meta(out, A.dtype(), A.device());
}

std::tuple<Tensor, Tensor, Tensor> lu_unpack_kernel(const Tensor& LU,
                                                    const Tensor& pivots,
                                                    bool unpack_data,
                                                    bool unpack_pivots) {
    if (LU.dim() < 2) {
        TP_THROW(RuntimeError,
                 "lu_unpack: expected LU_data with 2 or more dimensions but "
                 "got ", LU.dim());
    }
    const int64_t m = LU.size(-2);
    const int64_t n = LU.size(-1);
    const int64_t k = std::min(m, n);
    if (unpack_pivots) {
        if (pivots.dtype() != DType::Int32) {
            TP_THROW(RuntimeError,
                     "lu_unpack: LU_pivots must be an int32 tensor as "
                     "produced by an LU factorization");
        }
        if (pivots.dim() != LU.dim() - 1 || pivots.size(-1) != k) {
            TP_THROW(RuntimeError,
                     "lu_unpack: LU_pivots does not match the shape LU_data "
                     "implies; expected a trailing extent of ", k);
        }
    }
    const auto batch = linalg_batch_shape(LU);
    Tensor P;
    if (unpack_pivots) {
        P = empty_meta(append_dims(batch, {m, m}), LU.dtype(), LU.device());
    } else {
        P = empty_meta({0}, LU.dtype(), LU.device());
    }
    if (!unpack_data) {
        return {P, empty_meta({0}, LU.dtype(), LU.device()),
                empty_meta({0}, LU.dtype(), LU.device())};
    }
    return {P, empty_meta(append_dims(batch, {m, k}), LU.dtype(),
                          LU.device()),
            empty_meta(append_dims(batch, {k, n}), LU.dtype(), LU.device())};
}

void check_linalg_tolerance(const Tensor& input, const Tensor& tolerance) {
    if (tolerance.device() != input.device()) {
        TP_THROW(DeviceMismatchError,
                 "linalg tolerance tolerance must be on the input device");
    }
    if (isComplexType(tolerance.dtype())) {
        TP_THROW(TypeError, "linalg tolerance tolerance must not be complex");
    }
}

Tensor pinv_tensor_kernel(const Tensor& self,
                          const std::optional<Tensor>& atol,
                          const std::optional<Tensor>& rtol, bool hermitian) {
    if (atol.has_value()) check_linalg_tolerance(self, *atol);
    if (rtol.has_value()) check_linalg_tolerance(self, *rtol);
    if (!is_linalg_dtype(self.dtype()) || self.dim() < 2) {
        TP_THROW(TypeError,
                 "linalg.pinv expects a tensor with at least 2 dimensions of "
                 "a supported dtype");
    }
    if (hermitian && self.size(-2) != self.size(-1)) {
        TP_THROW(RuntimeError, "linalg.pinv hermitian input must be square");
    }
    if (self.numel() == 0) {
        std::vector<int64_t> shape = shape_of(self);
        std::swap(shape[shape.size() - 2], shape[shape.size() - 1]);
        return empty_meta(shape, self.dtype(), self.device());
    }
    // Both tolerances broadcast against the batch with a trailing singleton
    // for the singular-value axis, and the reciprocal scaling of the right
    // factor carries that batch into the result.
    std::vector<int64_t> tolerance = append_dims(linalg_batch_shape(self), {1});
    if (atol.has_value()) {
        tolerance = broadcast_shapes(tolerance,
                                     append_dims(shape_of(*atol), {1}));
    }
    if (rtol.has_value()) {
        tolerance = broadcast_shapes(tolerance,
                                     append_dims(shape_of(*rtol), {1}));
    }
    tolerance.pop_back();
    const std::vector<int64_t> out = append_dims(
        broadcast_shapes(linalg_batch_shape(self), tolerance),
        {self.size(-1), self.size(-2)});
    return empty_meta(out, self.dtype(), self.device());
}

Tensor matrix_power_kernel(const Tensor& self, int64_t n) {
    if (self.dim() < 2 || self.size(-2) != self.size(-1)) {
        TP_THROW(RuntimeError, "matrix_power(): expected a square matrix");
    }
    if (n == std::numeric_limits<int64_t>::min()) {
        TP_THROW(RuntimeError, "matrix_power(): exponent is too small");
    }
    return empty_meta(shape_of(self), self.dtype(), self.device());
}
} // anonymous namespace
TENSORPLAY_LIBRARY_IMPL(Meta, MetaLinalg) {
    m.impl("_linalg_check_errors", linalg_check_errors_kernel);
    m.impl("linalg_cholesky_ex", cholesky_ex_kernel);
    m.impl("linalg_inv_ex", inv_ex_kernel);
    m.impl("_linalg_det", det_kernel);
    m.impl("_linalg_slogdet", slogdet_kernel);
    m.impl("_linalg_solve_ex", solve_ex_kernel);
    m.impl("linalg_lu_factor_ex", lu_factor_ex_kernel);
    m.impl("linalg_lu", lu_kernel);
    m.impl("linalg_lu_solve", lu_solve_kernel);
    m.impl("linalg_solve_triangular", solve_triangular_kernel);
    m.impl("_linalg_eigh", eigh_kernel);
    m.impl("_linalg_eigvals", eigvals_kernel);
    m.impl("linalg_eig", eig_kernel);
    m.impl("_linalg_svd", svd_kernel);
    m.impl("linalg_lstsq", lstsq_kernel);
    m.impl("linalg_qr", qr_kernel);
    m.impl("linalg_householder_product", householder_product_kernel);
    m.impl("linalg_ldl_factor", ldl_factor_only_kernel);
    m.impl("linalg_ldl_factor_ex", ldl_factor_ex_kernel);
    m.impl("linalg_ldl_solve", ldl_solve_kernel);
    m.impl("linalg_diagonal", diagonal_kernel);
    m.impl("linalg_pinv.atol_rtol_tensor", pinv_tensor_kernel);
    m.impl("lu_unpack", lu_unpack_kernel);
    // Every power multiplies square matrices with themselves, so the batch
    // geometry and the matrix plane are the operand's own; the exponent
    // never changes them, and the zeroth power answers an identity of the
    // same width.
    m.impl("matrix_power", matrix_power_kernel);
    m.impl("linalg_matrix_power", matrix_power_kernel);
}

} // namespace meta
} // namespace tensorplay
