// Composite kernel: unique_consecutive.
// via adjacent inequality, run ids via cumsum over the run-start mask.

#include "CompositeCommon.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

std::tuple<Tensor, Tensor, Tensor> unique_consecutive_flat(
        const Tensor& self, bool return_inverse, bool return_counts) {
    const Tensor flat = ops::reshape(self, {-1});
    const int64_t n = flat.numel();
    const Device dev = self.device();
    Tensor inverse = ops::empty({0}, DType::Int64, dev);
    Tensor counts = ops::empty({0}, DType::Int64, dev);
    if (n == 0) {
        Tensor output = ops::empty({0}, self.dtype(), dev);
        if (return_inverse) {
            inverse = ops::empty(static_cast<std::vector<int64_t>>(self.shape()),
                                 DType::Int64, dev);
        }
        return {output, inverse, counts};
    }
    const Tensor neq = ops::ne(ops::slice(flat, 0, 1, std::nullopt, 1),
                               ops::slice(flat, 0, std::nullopt, n - 1, 1));
    const Tensor mark = ops::cat(
        {ops::ones({1}, DType::Bool, dev), neq}, 0);
    Tensor output = ops::masked_select(flat, mark);
    const Tensor ids = ops::sub(ops::cumsum(mark, 0, DType::Int64),
                                Scalar(int64_t(1)));
    if (return_inverse) {
        inverse = ops::reshape(ids, static_cast<std::vector<int64_t>>(self.shape()));
    }
    if (return_counts) {
        counts = ops::bincount(ids, std::nullopt, output.numel());
    }
    return {output, inverse, counts};
}

std::tuple<Tensor, Tensor, Tensor> unique_consecutive_dim(
        const Tensor& self, int64_t dim, bool return_inverse,
        bool return_counts) {
    const int64_t ndim = self.dim();
    dim = wrap_dim(dim, ndim);
    const int64_t n = self.size(dim);
    const Device dev = self.device();

    const int64_t zero_dims = [&] {
        int64_t c = 0;
        for (int64_t d = 0; d < ndim; ++d) c += self.size(d) == 0;
        return c;
    }();
    if (n == 0) {
        if (zero_dims != 1) {
            TP_THROW(RuntimeError,
                     "unique_consecutive(): Expected exactly one zero-sized dimension");
        }
        return {ops::empty(static_cast<std::vector<int64_t>>(self.shape()),
                           self.dtype(), dev),
                ops::empty({0}, DType::Int64, dev),
                ops::empty({0}, DType::Int64, dev)};
    }
    if (zero_dims != 0) {
        TP_THROW(RuntimeError,
                 "unique_consecutive(): Expected no zero-sized dimensions when the selected dim is non-empty");
    }

    const Tensor front = ops::moveaxis(self, dim, int64_t(0));
    const Tensor flat = ops::reshape(front, {n, -1});
    const Tensor neq_rows = ops::any(
        ops::ne(ops::slice(flat, 0, 1, std::nullopt, 1),
                ops::slice(flat, 0, std::nullopt, n - 1, 1)),
        int64_t(1), false);
    const Tensor mark = ops::cat(
        {ops::ones({1}, DType::Bool, dev), neq_rows}, 0);
    const Tensor kept_idx = ops::reshape(ops::nonzero(mark), {-1});
    const int64_t num_runs = kept_idx.size(0);

    Tensor kept = ops::index_select(flat, 0, kept_idx);
    std::vector<int64_t> front_sizes =
        static_cast<std::vector<int64_t>>(front.shape());
    front_sizes[0] = num_runs;
    kept = ops::reshape(kept, front_sizes);
    Tensor output = ops::moveaxis(kept, int64_t(0), dim);

    Tensor inverse = ops::empty({0}, DType::Int64, dev);
    Tensor counts = ops::empty({0}, DType::Int64, dev);
    const Tensor ids = ops::sub(ops::cumsum(mark, 0, DType::Int64),
                                Scalar(int64_t(1)));
    if (return_inverse) inverse = ids;
    if (return_counts) counts = ops::bincount(ids, std::nullopt, num_runs);
    return {output, inverse, counts};
}

} // anonymous namespace

std::tuple<Tensor, Tensor, Tensor> unique_consecutive_native(
        const Tensor& self, bool return_inverse, bool return_counts,
        std::optional<int64_t> dim) {
    if (!dim.has_value()) {
        return unique_consecutive_flat(self, return_inverse, return_counts);
    }
    return unique_consecutive_dim(self, *dim, return_inverse, return_counts);
}

TENSORPLAY_LIBRARY_IMPL(Composite, UniqueComposite) {
    m.impl("unique_consecutive", unique_consecutive_native);
}

} // namespace composite
} // namespace tensorplay
