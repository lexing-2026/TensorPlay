#pragma once

// Pieces of lu_unpack that do not depend on where the data lives.
//
// An LU factorization stores both triangles in one matrix and records the row
// interchanges as a sequence of 1-based pivots.  Unpacking splits the two
// triangles apart (the unit diagonal of L is implicit in the packed form) and
// replays the pivot sequence into a permutation matrix.  Only the replay is
// order-dependent and therefore needs a per-device kernel; everything here is
// shape work and triangular masking.

#include "Tensor.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace lu_unpack_detail {

namespace ops = tensorplay::tpx::ops;

struct LuShape {
    int64_t m;
    int64_t n;
    int64_t k;
};

inline LuShape validate(const Tensor& LU, const Tensor& pivots,
                        bool unpack_pivots) {
    TP_CHECK(LU.dim() >= 2,
             "lu_unpack: expected LU_data with 2 or more dimensions but got ",
             LU.dim());
    const int64_t m = LU.size(-2);
    const int64_t n = LU.size(-1);
    const int64_t k = std::min(m, n);
    if (unpack_pivots) {
        TP_CHECK(pivots.dtype() == DType::Int32,
                 "lu_unpack: LU_pivots must be an int32 tensor as produced by "
                 "an LU factorization");
        TP_CHECK(pivots.dim() == LU.dim() - 1 && pivots.size(-1) == k,
                 "lu_unpack: LU_pivots does not match the shape LU_data "
                 "implies; expected a trailing extent of ", k);
    }
    return {m, n, k};
}

// L keeps the strictly lower triangle and gains the unit diagonal the packed
// form leaves implicit; U keeps the upper triangle.  The wider of the two
// dimensions decides which factor is truncated to the square core.
inline std::pair<Tensor, Tensor> split_triangles(const Tensor& LU,
                                                 const LuShape& shape) {
    Tensor lower;
    Tensor upper;
    if (shape.m > shape.n) {
        upper = ops::triu(ops::narrow(LU, -2, 0, shape.n), 0);
        lower = ops::tril(LU, -1);
    } else if (shape.m < shape.n) {
        lower = ops::tril(ops::narrow(LU, -1, 0, shape.m), -1);
        upper = ops::triu(LU, 0);
    } else {
        lower = ops::tril(LU, -1);
        upper = ops::triu(LU, 0);
    }
    lower = ops::add(lower,
                     ops::eye(shape.m, shape.k, LU.dtype(), LU.device(), false),
                     Scalar(1));
    return {lower, upper};
}

// The permutation carries one 1 per column: column j selects row perm[j].
inline Tensor permutation_matrix(const Tensor& perm, const Tensor& LU,
                                 const LuShape& shape) {
    std::vector<int64_t> sizes = static_cast<std::vector<int64_t>>(LU.shape());
    sizes[sizes.size() - 2] = shape.m;
    sizes[sizes.size() - 1] = shape.m;
    const Tensor zero =
        ops::zeros(sizes, LU.dtype(), LU.device(), false, false);
    return ops::scatter(zero, -2, ops::unsqueeze(perm, -2), Scalar(1.0));
}

// Shape of the identity permutation the replay starts from.
inline std::vector<int64_t> perm_sizes(const Tensor& LU, const LuShape& shape) {
    std::vector<int64_t> sizes = static_cast<std::vector<int64_t>>(LU.shape());
    sizes.pop_back();
    sizes.back() = shape.m;
    return sizes;
}

}  // namespace lu_unpack_detail
}  // namespace tensorplay
