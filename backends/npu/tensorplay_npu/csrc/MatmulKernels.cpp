// Matrix products through the Ascend operator library: mm, bmm, addmm and
// the general batched matmul.  Every case is one operator call; the library
// owns the tiling and the cube configuration.

#include <optional>
#include <vector>

#include "AscendOpApi.h"
#include "AscendTensor.h"
#include "CpuFallback.h"
#include "Dispatcher.h"
#include "Utils.h"
#include "tensorplay/ops/KernelSignatures.h"

namespace tensorplay {
namespace npu {

namespace {

namespace sig = kernel_signatures;

// The cube multiplies float16, bfloat16 and float32; a pair the device
// cannot multiply runs on the host.
bool native_pair(const Tensor& a, const Tensor& b) {
    return a.device().is_privateuse1() && b.device().is_privateuse1() &&
           a.dtype() == b.dtype() && is_native_float(a.dtype()) && a.dim() >= 1 &&
           b.dim() >= 1 && a.dim() <= kMaxAclDims && b.dim() <= kMaxAclDims;
}

const OpApi& mm_api() {
    static const OpApi api("aclnnMm");
    return api;
}

const OpApi& bmm_api() {
    static const OpApi api("aclnnBatchMatMul");
    return api;
}

const OpApi& matmul_api() {
    static const OpApi api("aclnnMatmul");
    return api;
}

const OpApi& addmm_api() {
    static const OpApi api("aclnnAddmm");
    return api;
}

// The cube keeps the operand dtype: no substitution by a narrower
// compute type.
constexpr int8_t kKeepDtype = 0;

// One product of dense operands into a fresh result of ``shape``.
Tensor product(const OpApi& api, const Tensor& a, const Tensor& b,
               const std::vector<int64_t>& shape) {
    const int index = index_of(a.device());
    Tensor out = empty_on(shape, a.dtype(), index);
    if (out.numel() == 0) return out;
    const Tensor x = compute_operand(a, a.dtype());
    const Tensor y = compute_operand(b, a.dtype());
    AclArgs args;
    const AclTensorRef xr = args.tensor(x);
    const AclTensorRef yr = args.tensor(y);
    const AclTensorRef outr = args.tensor(out);
    api.run(index, xr, yr, outr, kKeepDtype);
    return out;
}

// [.., M, K] @ [.., K, N] -> broadcast batch + [M, N], with the operand
// shapes as the product and the scalar case of two 1-d operands dropped.
std::optional<std::vector<int64_t>> matmul_shape(const Tensor& a, const Tensor& b) {
    const int64_t da = a.dim();
    const int64_t db = b.dim();
    const std::vector<int64_t> sa = a.sizes().vec();
    const std::vector<int64_t> sb = b.sizes().vec();
    const int64_t inner_a = sa[static_cast<size_t>(da - 1)];
    const int64_t inner_b = sb[static_cast<size_t>(db - 1 == 0 ? 0 : db - 2)];
    if (inner_a != inner_b) return std::nullopt;
    std::vector<int64_t> shape;
    if (da > 2 || db > 2) {
        const std::vector<int64_t> batch_a(sa.begin(), sa.end() - 2);
        const std::vector<int64_t> batch_b(sb.begin(), sb.end() - 2);
        shape = broadcast_shapes(IntArrayRef(batch_a), IntArrayRef(batch_b));
    }
    if (da > 1) shape.push_back(sa[static_cast<size_t>(da - 2)]);
    if (db > 1) shape.push_back(sb[static_cast<size_t>(db - 1)]);
    return shape;
}

Tensor mm_kernel(const Tensor& self, const Tensor& mat2) {
    if (!native_pair(self, mat2) || self.dim() != 2 || mat2.dim() != 2 || !mm_api().valid()) {
        return cpu_fallback::Kernel<"mm", sig::mm>::call(self, mat2);
    }
    TP_CHECK(self.sizes()[1] == mat2.sizes()[0], "mm: shapes ", self.sizes()[0], "x",
             self.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1],
             " cannot be multiplied");
    return product(mm_api(), self, mat2, {self.sizes()[0], mat2.sizes()[1]});
}

Tensor bmm_kernel(const Tensor& self, const Tensor& mat2) {
    if (!native_pair(self, mat2) || self.dim() != 3 || mat2.dim() != 3 || !bmm_api().valid()) {
        return cpu_fallback::Kernel<"bmm", sig::bmm>::call(self, mat2);
    }
    TP_CHECK(self.sizes()[0] == mat2.sizes()[0] && self.sizes()[2] == mat2.sizes()[1],
             "bmm: batch or inner dimensions do not match");
    return product(bmm_api(), self, mat2,
                   {self.sizes()[0], self.sizes()[1], mat2.sizes()[2]});
}

Tensor addmm_kernel(const Tensor& self, const Tensor& mat1, const Tensor& mat2,
                    const Scalar& beta, const Scalar& alpha) {
    if (!native_pair(mat1, mat2) || mat1.dim() != 2 || mat2.dim() != 2 ||
        !self.device().is_privateuse1() || self.dtype() != mat1.dtype() ||
        self.dim() > 2 || !addmm_api().valid()) {
        return cpu_fallback::Kernel<"addmm", sig::addmm>::call(self, mat1, mat2, beta, alpha);
    }
    TP_CHECK(mat1.sizes()[1] == mat2.sizes()[0], "addmm: shapes ", mat1.sizes()[0], "x",
             mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1],
             " cannot be multiplied");
    const std::vector<int64_t> shape = {mat1.sizes()[0], mat2.sizes()[1]};
    const int index = index_of(mat1.device());
    Tensor out = empty_on(shape, mat1.dtype(), index);
    if (out.numel() == 0) return out;
    const Tensor bias = compute_operand(self, self.dtype());
    const Tensor x = compute_operand(mat1, mat1.dtype());
    const Tensor y = compute_operand(mat2, mat2.dtype());
    AclArgs args;
    const AclTensorRef biasr = args.tensor(bias);
    const AclTensorRef xr = args.tensor(x);
    const AclTensorRef yr = args.tensor(y);
    const AclTensorRef outr = args.tensor(out);
    addmm_api().run(index, biasr, xr, yr, args.scalar(beta, self.dtype()),
                    args.scalar(alpha, self.dtype()), outr, kKeepDtype);
    return out;
}

Tensor matmul_kernel(const Tensor& self, const Tensor& other) {
    if (!native_pair(self, other) || !matmul_api().valid()) {
        return cpu_fallback::Kernel<"matmul", sig::matmul>::call(self, other);
    }
    const std::optional<std::vector<int64_t>> shape = matmul_shape(self, other);
    // Two 1-d operands produce a 0-d result, which the operator sees as a
    // 1-element tensor; the host keeps that shape exact.
    if (!shape || shape->empty()) {
        return cpu_fallback::Kernel<"matmul", sig::matmul>::call(self, other);
    }
    const int64_t inner = self.sizes()[self.dim() - 1];
    if (inner == 0) {
        return cpu_fallback::Kernel<"matmul", sig::matmul>::call(self, other);
    }
    return product(matmul_api(), self, other, *shape);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, NPUMatmulKernels) {
    m.impl("mm", static_cast<sig::mm*>(&mm_kernel));
    m.impl("bmm", static_cast<sig::bmm*>(&bmm_kernel));
    m.impl("addmm", static_cast<sig::addmm*>(&addmm_kernel));
    m.impl("matmul", static_cast<sig::matmul*>(&matmul_kernel));
}

} // namespace npu
} // namespace tensorplay
