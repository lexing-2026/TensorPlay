// Sparse BLAS entry points.  The sampled SpMM arithmetic itself runs in the
// per-ISA tier file (SparseBlasKernelsImpl.cpp) behind sampled_addmm_stub;
// this base tier owns validation, tensor preparation and registration.

#include "SparseKernels.h"
#include "cpu/SparseBlasKernels.h"
#include "Exception.h"

#include <vector>

namespace tensorplay {
namespace cpu {

namespace {

void sparse_sampled_addmm_check_inputs(const Tensor& self,
                                       const Tensor& mat1,
                                       const Tensor& mat2,
                                       const Tensor& result) {
    if (!self.is_sparse_csr()) {
        TP_THROW(RuntimeError,
                 "sparse_sampled_addmm: expected self to be a sparse CSR "
                 "tensor, got layout ",
                 (self.is_sparse() ? "sparse" : "strided"));
    }
    if (mat1.is_sparse()) {
        TP_THROW(RuntimeError,
                 "sparse_sampled_addmm: expected mat1 to have strided "
                 "layout, got ",
                 (mat1.is_sparse() ? "sparse" : "strided"));
    }
    if (mat2.is_sparse()) {
        TP_THROW(RuntimeError,
                 "sparse_sampled_addmm: expected mat2 to have strided "
                 "layout, got ",
                 (mat2.is_sparse() ? "sparse" : "strided"));
    }
    if (!result.is_sparse_csr()) {
        TP_THROW(RuntimeError,
                 "sparse_sampled_addmm: expected result to have sparse csr "
                 "layout, got ",
                 (result.is_sparse() ? "sparse" : "strided"));
    }
    if (self.dense_dim() != 0 || result.dense_dim() != 0) {
        TP_THROW(RuntimeError,
                 "sparse_sampled_addmm: expected non-hybrid self and result "
                 "tensors");
    }
    if (mat1.dtype() != mat2.dtype() || mat1.dtype() != self.dtype() ||
        result.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "sparse_sampled_addmm: mat1, mat2, self and result must "
                 "share one dtype");
    }
    if (mat1.dim() < 2 || mat2.dim() < 2 || result.dim() < 2) {
        TP_THROW(RuntimeError,
                 "sparse_sampled_addmm: mat1, mat2 and result must be "
                 "matrices");
    }
    const std::vector<int64_t> mat1_batch(
        mat1.sizes().begin(), mat1.sizes().begin() + mat1.dim() - 2);
    const std::vector<int64_t> mat2_batch(
        mat2.sizes().begin(), mat2.sizes().begin() + mat2.dim() - 2);
    if (mat1_batch != mat2_batch) {
        TP_THROW(RuntimeError,
                 "sparse_sampled_addmm: mat1 and mat2 must have the same "
                 "batch size");
    }
    if (self.dim() > 2) {
        const std::vector<int64_t> self_batch(
            self.sizes().begin(), self.sizes().begin() + self.dim() - 2);
        if (self_batch != mat1_batch) {
            TP_THROW(RuntimeError,
                     "sparse_sampled_addmm: self and mat1 must have the "
                     "same batch size");
        }
    }
    if (mat1.size(mat1.dim() - 1) != mat2.size(mat2.dim() - 2)) {
        TP_THROW(RuntimeError,
                 "sparse_sampled_addmm: mat1 and mat2 shapes cannot be "
                 "multiplied (",
                 mat1.size(mat1.dim() - 2), "x", mat1.size(mat1.dim() - 1),
                 " and ", mat2.size(mat2.dim() - 2), "x",
                 mat2.size(mat2.dim() - 1), ")");
    }
    if (self.size(self.dim() - 2) != mat1.size(mat1.dim() - 2)) {
        TP_THROW(RuntimeError,
                 "sparse_sampled_addmm: self.shape[-2] must match "
                 "mat1.shape[-2]");
    }
    if (self.size(self.dim() - 1) != mat2.size(mat2.dim() - 1)) {
        TP_THROW(RuntimeError,
                 "sparse_sampled_addmm: self.shape[-1] must match "
                 "mat2.shape[-1]");
    }
}

} // namespace

Tensor& sparse_sampled_addmm_out_sparse_csr_cpu(
    const Tensor& self, const Tensor& mat1, const Tensor& mat2,
    const Scalar& beta, const Scalar& alpha, Tensor& result) {
    if (!self.is_sparse_csr()) {
        TP_THROW(RuntimeError,
                 "sparse_sampled_addmm: expected self to be a sparse CSR "
                 "tensor, got layout ",
                 (self.is_sparse() ? "sparse" : "strided"));
    }
    if (&result != &self || !result.is_sparse_csr() ||
        !result._values().is_contiguous()) {
        // Rebuild `result` to carry self's sparsity pattern and shape.  The
        // components are cloned so later in-place updates never alias self
        // and the kernel always writes into contiguous storage.  This also
        // turns the strided seed created by the function variant into a CSR
        // tensor before any layout validation runs.
        Tensor crow = self._crow_indices().contiguous().clone();
        Tensor col = self._col_indices().contiguous().clone();
        Tensor values = self._values().contiguous().clone();
        result.unsafeGetTensorImpl()->set_sparse_compressed_state(
            crow.unsafeGetTensorImpl(), col.unsafeGetTensorImpl(),
            values.unsafeGetTensorImpl(),
            static_cast<std::vector<int64_t>>(self.shape()),
            TensorImpl::kSparseCSRLayout, {0, 0});
        result.unsafeGetTensorImpl()->set_sizes_contiguous(
            static_cast<std::vector<int64_t>>(self.shape()));
    }
    sparse_sampled_addmm_check_inputs(self, mat1, mat2, result);
    const DType dtype = self.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float64 &&
        dtype != DType::ComplexFloat && dtype != DType::ComplexDouble) {
        TP_THROW(RuntimeError,
                 "sparse_sampled_addmm: expected self to be a "
                 "floating-point or complex tensor, but got ",
                 toString(dtype));
    }
    if (self.device() != mat1.device() || self.device() != mat2.device() ||
        self.device() != result.device()) {
        TP_THROW(DeviceMismatchError,
                 "sparse_sampled_addmm: all operands must be on the same "
                 "device");
    }

    const int64_t nnz = self._values().size(-1);
    if (mat1.numel() == 0 || mat2.numel() == 0 || nnz == 0) {
        result._values().mul_(beta);
        return result;
    }

    // The right operand is stored transposed (batch, n, k) so every dot
    // product reads two contiguous rows.
    Tensor mat2_t = mat2.transpose(-1, -2).contiguous();
    Tensor mat1_c = mat1.is_contiguous() ? mat1 : mat1.contiguous();
    sampled_addmm_stub(DeviceType::CPU, mat1_c, mat2_t, beta, alpha, result);
    return result;
}

Tensor sparse_sampled_addmm_sparse_csr_cpu(
    const Tensor& self, const Tensor& mat1, const Tensor& mat2,
    const Scalar& beta, const Scalar& alpha) {
    // Start from a dense empty seed; the out path below replaces it with a
    // CSR tensor carrying self's pattern.
    Tensor result = Tensor::empty({0}, self.dtype(), self.device());
    sparse_sampled_addmm_out_sparse_csr_cpu(self, mat1, mat2, beta, alpha,
                                            result);
    return result;
}

DEFINE_DISPATCH(sampled_addmm_stub);

} // namespace cpu
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(SparseCPU, SparseBlasKernels) {
    using namespace tensorplay::cpu;
    m.impl("sparse_sampled_addmm", sparse_sampled_addmm_sparse_csr_cpu);
    m.impl("sparse_sampled_addmm.out",
           sparse_sampled_addmm_out_sparse_csr_cpu);
}
