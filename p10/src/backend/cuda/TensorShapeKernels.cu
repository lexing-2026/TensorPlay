#include "CudaDispatchHelpers.cuh"

namespace tensorplay {
namespace cuda {


namespace {


// ---------------------------------------------------------------------------
// _chunk_cat: split the list into num_chunks even parts along dim, then cat.
// ---------------------------------------------------------------------------

Tensor interop_chunk_cat_cuda(const std::vector<Tensor>& tensors, int64_t dim,
                              int64_t num_chunks) {
    if (tensors.empty()) {
        TP_THROW(RuntimeError, "expected a non-empty list of Tensors");
    }
    if (num_chunks <= 0) {
        TP_THROW(RuntimeError, "num_chunks must be positive, got ", num_chunks);
    }
    std::vector<Tensor> parts;
    parts.reserve(static_cast<size_t>(num_chunks));
    const int64_t per = (static_cast<int64_t>(tensors.size()) + num_chunks - 1) /
                        num_chunks;
    for (int64_t c = 0; c < num_chunks; ++c) {
        const int64_t begin = c * per;
        const int64_t end = std::min<int64_t>(begin + per, tensors.size());
        if (begin >= end) break;
        parts.push_back(ops::cat(std::vector<Tensor>(tensors.begin() + begin,
                                                     tensors.begin() + end),
                                 dim));
    }
    return ops::cat(parts, dim);
}


Tensor& interop__chunk_cat_out_cuda(const std::vector<Tensor>& tensors,
                                    int64_t dim, int64_t num_chunks,
                                    Tensor& out) {
    write_out(out, interop_chunk_cat_cuda(tensors, dim, num_chunks));
    return out;
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, TensorShapeKernels) {
    m.impl("_chunk_cat", interop_chunk_cat_cuda);
    m.impl("_chunk_cat.out", interop__chunk_cat_out_cuda);
}

} // namespace cuda
} // namespace tensorplay
