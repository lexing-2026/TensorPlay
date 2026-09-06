// Host kernel for lu_unpack.
//
// The triangular split is shape work shared with every backend; what belongs
// here is the pivot replay, which walks each matrix's interchange sequence in
// order and is therefore serialized per matrix but independent across the
// batch.

#include "../LuUnpackShared.h"

#include "Dispatcher.h"
#include "Parallel.h"

#include <cstdint>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cpu {

namespace detail_lu = tensorplay::lu_unpack_detail;

namespace {

// Replays the 1-based interchange sequence of every matrix in the batch onto
// its own identity permutation.
Tensor replay_pivots(const Tensor& pivots, int64_t m, int64_t k) {
    const Tensor source = pivots.contiguous();
    std::vector<int64_t> sizes =
        static_cast<std::vector<int64_t>>(source.shape());
    sizes.back() = m;
    Tensor perm = Tensor::empty(sizes, DType::Int64, source.device());
    const int64_t batch = source.numel() / std::max<int64_t>(k, 1);
    if (perm.numel() == 0) return perm;

    const int32_t* pivot_data = source.data_ptr<int32_t>();
    int64_t* perm_data = perm.data_ptr<int64_t>();
    parallel::parallel_for(0, batch, 1, [&](int64_t begin, int64_t end) {
        for (int64_t b = begin; b < end; ++b) {
            int64_t* row = perm_data + b * m;
            std::iota(row, row + m, int64_t(0));
            const int32_t* pivot_row = pivot_data + b * k;
            for (int64_t i = 0; i < k; ++i) {
                const int64_t target = static_cast<int64_t>(pivot_row[i]) - 1;
                TP_CHECK(target >= 0 && target < m,
                         "lu_unpack: pivots must lie between 1 and ", m,
                         " inclusive");
                std::swap(row[i], row[target]);
            }
        }
    });
    return perm;
}

}  // namespace

std::tuple<Tensor, Tensor, Tensor> lu_unpack_cpu(const Tensor& LU,
                                                 const Tensor& pivots,
                                                 bool unpack_data,
                                                 bool unpack_pivots) {
    const detail_lu::LuShape shape = detail_lu::validate(LU, pivots, unpack_pivots);

    Tensor P;
    if (unpack_pivots) {
        const Tensor perm = replay_pivots(pivots, shape.m, shape.k);
        P = detail_lu::permutation_matrix(perm, LU, shape);
    } else {
        P = Tensor::empty({0}, LU.dtype(), LU.device());
    }

    Tensor L;
    Tensor U;
    if (unpack_data) {
        std::tie(L, U) = detail_lu::split_triangles(LU, shape);
    } else {
        L = Tensor::empty({0}, LU.dtype(), LU.device());
        U = Tensor::empty({0}, LU.dtype(), LU.device());
    }
    return {P, L, U};
}

TENSORPLAY_LIBRARY_IMPL(CPU, LuUnpackKernels) {
    m.impl("lu_unpack", lu_unpack_cpu);
}

}  // namespace cpu
}  // namespace tensorplay
