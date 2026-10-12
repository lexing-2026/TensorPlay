// Entry points for _sparse_mm_reduce_impl and its backward on SparseCPU.
// The arithmetic lives in the per-ISA stub kernels (SpmmReduceKernelsImpl
// .cpp); this file owns validation, the CSR<->CSC bookkeeping and the
// operator registrations.

#include "SparseKernels.h"
#include "cpu/SpmmReduceKernels.h"
#include "Exception.h"
#include "GradMode.h"

#include <array>
#include <string>
#include <tuple>

namespace tensorplay {
namespace cpu {

namespace {

int get_reduction_enum(const std::string& reduce) {
    if (reduce == "sum") return 0;
    if (reduce == "mean") return 1;
    if (reduce == "amax") return 2;
    if (reduce == "amin") return 3;
    TP_THROW(RuntimeError, "sparse_mm_reduce: unsupported reduce type \"",
             reduce, "\"");
}

void check_spmm_reduce_inputs(const Tensor& self, const Tensor& other,
                                const std::string& reduce) {
    if (!self.is_sparse_csr()) {
        TP_THROW(RuntimeError,
                 "sparse_mm_reduce: expected self to be a sparse CSR "
                 "tensor, got layout ",
                 (self.is_sparse() ? "sparse" : "strided"));
    }
    if (self.dense_dim() != 0) {
        TP_THROW(RuntimeError,
                 "sparse_mm_reduce: expected non-hybrid self tensor");
    }
    if (self.dim() != 2) {
        TP_THROW(RuntimeError,
                 "sparse_mm_reduce: expected self to be a 2-D tensor, got ",
                 self.dim(), "-D tensor");
    }
    if (other.is_sparse()) {
        TP_THROW(RuntimeError,
                 "sparse_mm_reduce: expected other to be a strided tensor, "
                 "got layout ",
                 (other.is_sparse() ? "sparse" : "strided"));
    }
    if (other.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "sparse_mm_reduce: self and other must share one dtype");
    }
    if (other.dim() != 2 || other.size(0) != self.size(1)) {
        TP_THROW(RuntimeError,
                 "sparse_mm_reduce: other must be a 2-D tensor whose "
                 "leading dimension matches self.size(1)");
    }
    if (self.device() != other.device()) {
        TP_THROW(DeviceMismatchError,
                 "sparse_mm_reduce: self and other must be on the same "
                 "device");
    }
    // Match the reference contract: prod is declared but not enabled.
    if (reduce == "prod") {
        TP_THROW(RuntimeError,
                 "sparse_mm_reduce: reduce type of prod has not been "
                 "enabled");
    }
}

void check_spmm_reduce_backward_inputs(const Tensor& self,
                                       const Tensor& grad_out,
                                       const Tensor& other) {
    if (!self.is_sparse_csr()) {
        TP_THROW(RuntimeError,
                 "sparse_mm_reduce_backward: expected self to be a sparse "
                 "CSR tensor, got layout ",
                 (self.is_sparse() ? "sparse" : "strided"));
    }
    if (grad_out.is_sparse() ||
        grad_out.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "sparse_mm_reduce_backward: grad_out must be a strided "
                 "tensor sharing self's dtype");
    }
    if (grad_out.dim() != 2 || grad_out.size(0) != self.size(0) ||
        grad_out.size(1) != other.size(1)) {
        TP_THROW(RuntimeError,
                 "sparse_mm_reduce_backward: grad_out shape mismatch");
    }
    if (other.is_sparse() ||
        other.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "sparse_mm_reduce_backward: other must be a strided tensor "
                 "sharing self's dtype");
    }
    if (other.dim() != 2 || other.size(0) != self.size(1)) {
        TP_THROW(RuntimeError,
                 "sparse_mm_reduce_backward: other shape mismatch");
    }
}

} // namespace

std::tuple<Tensor, Tensor> _sparse_mm_reduce_impl_sparse_csr_cpu(
    const Tensor& self, const Tensor& other, const std::string& reduce) {
    check_spmm_reduce_inputs(self, other, reduce);
    const int op = get_reduction_enum(reduce);

    Tensor crow = self._crow_indices().contiguous();
    Tensor col = self._col_indices().contiguous();
    Tensor val = self._values().contiguous();
    Tensor other_c = other.is_contiguous() ? other : other.contiguous();

    Tensor out = Tensor::zeros({self.size(0), other.size(1)}, other.dtype(),
                               other.device());
    Tensor arg_out = Tensor::empty({0}, col.dtype(), col.device());

    const int64_t nnz = self._nnz();
    if (nnz == 0) {
        return std::make_tuple(std::move(out), std::move(arg_out));
    }

    // amax/amin only need the winning indices during autograd.
    const bool need_arg = GradMode::is_enabled() &&
                          (self.requires_grad() || other.requires_grad()) &&
                          (op == 2 || op == 3);
    if (need_arg) {
        arg_out.resize_(static_cast<std::vector<int64_t>>(out.shape()));
        arg_out.fill_(nnz);
    }

    spmm_reduce_stub(DeviceType::CPU, out, arg_out, crow, col, val, other_c,
                     op, need_arg);
    return std::make_tuple(std::move(out), std::move(arg_out));
}

std::tuple<Tensor, Tensor> _sparse_mm_reduce_impl_backward_sparse_csr_cpu(
    const Tensor& self, const Tensor& grad_out, const Tensor& other,
    const std::string& reduce, const Tensor& arg_out,
    const std::vector<bool>& output_mask) {
    check_spmm_reduce_backward_inputs(self, grad_out, other);
    const int op = get_reduction_enum(reduce);
    const std::array<bool, 2> mask = {output_mask[0], output_mask[1]};

Tensor crow = self._crow_indices().contiguous();
    Tensor col = self._col_indices().contiguous();
    Tensor val = self._values().contiguous();
    Tensor grad_out_c = grad_out.is_contiguous() ? grad_out
                                                 : grad_out.contiguous();
    Tensor other_c = other.is_contiguous() ? other : other.contiguous();

    // CSC bookkeeping (row indices plus column pointers/permutation) is only
    // needed by the sum/mean grad_other path.
    Tensor row;
    Tensor ccol;
    Tensor permute;
    if (op == 0 || op == 1) {
        const bool out_int32 = crow.dtype() == DType::Int32;
        Tensor coo_indices = Tensor::_convert_indices_from_csr_to_coo(
            crow, col, out_int32, /*transpose=*/false);
        row = coo_indices.select(0, 0);
        Tensor index = col.mul(self.size(0)).add_(row);
        permute = index.argsort();
        ccol = Tensor::_convert_indices_from_coo_to_csr(
            col.index_select(0, permute), self.size(1), out_int32);
    }

    Tensor grad_self;
    if (mask[0]) {
        grad_self = Tensor::make_sparse_csr_tensor(
            self._crow_indices().clone(), self._col_indices().clone(),
            Tensor::zeros_like(self._values()),
            static_cast<std::vector<int64_t>>(self.shape()));
        spmm_reduce_backward_input_stub(
            DeviceType::CPU, grad_self, grad_out_c, crow, col, other_c,
            arg_out, row, op);
    }
    Tensor grad_other;
    if (mask[1]) {
        grad_other = Tensor::zeros({other.size(0), other.size(1)},
                                    other.dtype(), other.device());
        spmm_reduce_backward_other_stub(
            DeviceType::CPU, grad_other, grad_out_c, crow, col, val, arg_out,
            row, ccol, permute, op);
    }
    return std::make_tuple(std::move(grad_self), std::move(grad_other));
}

DEFINE_DISPATCH(spmm_reduce_stub);
DEFINE_DISPATCH(spmm_reduce_backward_input_stub);
DEFINE_DISPATCH(spmm_reduce_backward_other_stub);

} // namespace cpu
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(SparseCPU, SpmmReduceKernels) {
    using namespace tensorplay::cpu;
    using tensorplay::Tensor;
    m.impl("_sparse_mm_reduce_impl",
           [](const Tensor& self, const Tensor& other,
              const std::string& reduce) {
               return _sparse_mm_reduce_impl_sparse_csr_cpu(self, other,
                                                            reduce);
           });
    m.impl("_sparse_mm_reduce_impl_backward",
           [](const Tensor& self, const Tensor& grad_out, const Tensor& other,
              const std::string& reduce, const Tensor& arg_out,
              const std::vector<bool>& output_mask) {
               return _sparse_mm_reduce_impl_backward_sparse_csr_cpu(
                   self, grad_out, other, reduce, arg_out, output_mask);
           });
}
