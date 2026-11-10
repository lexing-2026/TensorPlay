// Matrix products on CNNL: mm, addmm, bmm and matmul.
//
// A transposed operand (a contiguous matrix viewed through .t()) is passed
// with CNNL's transpose flag instead of being copied, which is what
// backward passes produce.  The algorithm CNNL's heuristic picks for a
// problem (shape, dtype, transposes, beta) is looked up once per thread
// and reused.

#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <tuple>

#include "Cnnl.h"
#include "CpuFallback.h"
#include "Dispatcher.h"
#include "tensorplay/ops/KernelSignatures.h"

namespace tensorplay {
namespace mlu {

namespace {

namespace sig = kernel_signatures;

// A matrix operand as CNNL sees it: a dense row-major matrix and whether
// the product reads it transposed.  ``batch`` is 0 for 2-d problems.
struct Operand {
    Tensor storage;  // the dense tensor whose memory is read
    bool transposed = false;
    int64_t batch = 0, rows = 0, cols = 0;  // of the dense layout
};

// ``t`` is [.., r, c].  Accept it as-is (row-major) or as the transpose of
// a row-major [.., c, r]; otherwise make it contiguous.
Operand as_operand(const Tensor& t) {
    const int64_t nd = t.dim();
    const auto sizes = t.sizes().vec();
    const auto strides = t.strides();
    const int64_t r = sizes[nd - 2], c = sizes[nd - 1];
    const int64_t batch = nd == 3 ? sizes[0] : 0;
    const bool batch_ok = nd == 2 || strides[0] == r * c || sizes[0] == 1;
    if (batch_ok && (strides[nd - 1] == 1 || c == 1) && (strides[nd - 2] == c || r == 1)) {
        return {t, false, batch, r, c};
    }
    if (batch_ok && (strides[nd - 2] == 1 || r == 1) && (strides[nd - 1] == r || c == 1)) {
        return {t, true, batch, c, r};
    }
    return {contiguous_on_device(t), false, batch, r, c};
}

TensorDesc matrix_desc(const Operand& op, DType dtype) {
    if (op.batch == 0) return TensorDesc({op.rows, op.cols}, {op.cols, 1}, dtype);
    return TensorDesc({op.batch, op.rows, op.cols}, {op.rows * op.cols, op.cols, 1}, dtype);
}

struct Plan {
    cnnlMatMulDescriptor_t desc = nullptr;
    cnnlMatMulAlgo_t algo = nullptr;
    size_t workspace = 0;
    ~Plan() {
        if (algo) cnnlMatMulAlgoDestroy(algo);
        if (desc) cnnlMatMulDescDestroy(desc);
    }
};

using PlanKey = std::tuple<int, int, int64_t, int64_t, int64_t, int64_t, bool, bool, bool>;

const Plan& plan_for(const PlanKey& key, cnnlHandle_t h, const TensorDesc& a,
                     const TensorDesc& b, const TensorDesc& c, bool batched) {
    thread_local std::map<PlanKey, std::unique_ptr<Plan>> plans;
    auto it = plans.find(key);
    if (it != plans.end()) return *it->second;
    auto plan = std::make_unique<Plan>();
    TP_CNNL_CHECK(cnnlMatMulDescCreate(&plan->desc));
    const int32_t trans_a = std::get<6>(key), trans_b = std::get<7>(key);
    const int32_t use_beta = std::get<8>(key);
    const int32_t compute = CNNL_DTYPE_FLOAT;
    TP_CNNL_CHECK(cnnlSetMatMulDescAttr(plan->desc, CNNL_MATMUL_DESC_TRANSA, &trans_a,
                                        sizeof(trans_a)));
    TP_CNNL_CHECK(cnnlSetMatMulDescAttr(plan->desc, CNNL_MATMUL_DESC_TRANSB, &trans_b,
                                        sizeof(trans_b)));
    TP_CNNL_CHECK(cnnlSetMatMulDescAttr(plan->desc, CNNL_MATMUL_DESC_COMPUTE_TYPE, &compute,
                                        sizeof(compute)));
    TP_CNNL_CHECK(cnnlSetMatMulDescAttr(plan->desc, CNNL_MATMUL_USE_BETA, &use_beta,
                                        sizeof(use_beta)));
    TP_CNNL_CHECK(cnnlMatMulAlgoCreate(&plan->algo));
    cnnlMatMulHeuristicResult_t result = nullptr;
    TP_CNNL_CHECK(cnnlCreateMatMulHeuristicResult(&result));
    int count = 0;
    if (batched) {
        TP_CNNL_CHECK(cnnlGetBatchMatMulExAlgoHeuristic(h, plan->desc, a, b, c, nullptr, 1,
                                                        &result, &count));
    } else {
        TP_CNNL_CHECK(cnnlGetMatMulAlgoHeuristic(h, plan->desc, a, b, c, c, nullptr, 1, &result,
                                                 &count));
    }
    TP_CHECK(count > 0, "CNNL found no matmul algorithm for the problem");
    if (batched) {
        TP_CNNL_CHECK(cnnlGetBatchMatMulExHeuristicResult(result, plan->algo, &plan->workspace));
    } else {
        TP_CNNL_CHECK(cnnlGetMatMulHeuristicResult(result, plan->algo, &plan->workspace));
    }
    cnnlDestroyMatMulHeuristicResult(result);
    return *plans.emplace(key, std::move(plan)).first->second;
}

// out (dense [.., M, N]) = alpha * a @ b + beta * out.
void gemm(const Tensor& a, const Tensor& b, const Tensor& out, float alpha, float beta) {
    const int index = index_of(out.device());
    const DType dtype = out.dtype();
    const Operand oa = as_operand(a), ob = as_operand(b);
    const bool batched = out.dim() == 3;
    const auto osz = out.sizes().vec();
    const int64_t B = batched ? osz[0] : 0, M = osz[out.dim() - 2], N = osz[out.dim() - 1];
    const int64_t K = a.sizes()[a.dim() - 1];
    Operand oc{out, false, B, M, N};
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    TensorDesc a_desc = matrix_desc(oa, dtype), b_desc = matrix_desc(ob, dtype);
    TensorDesc c_desc = matrix_desc(oc, dtype);
    const PlanKey key{index, static_cast<int>(dtype), B, M, N, K, oa.transposed, ob.transposed,
                      beta != 0.0f};
    const Plan& plan = plan_for(key, h, a_desc, b_desc, c_desc, batched);
    Workspace ws(plan.workspace, index);
    if (batched) {
        TP_CNNL_CHECK(cnnlBatchMatMulEx(h, plan.desc, plan.algo, &alpha, a_desc,
                                        oa.storage.data_ptr(), b_desc, ob.storage.data_ptr(),
                                        &beta, c_desc, out.data_ptr(), ws.get(), ws.size()));
    } else {
        TP_CNNL_CHECK(cnnlMatMul_v2(h, plan.desc, plan.algo, &alpha, a_desc,
                                    oa.storage.data_ptr(), b_desc, ob.storage.data_ptr(), &beta,
                                    c_desc, out.data_ptr(), ws.get(), ws.size(), c_desc,
                                    out.data_ptr()));
    }
}

bool native_pair(const Tensor& a, const Tensor& b) {
    return a.device().is_privateuse1() && b.device().is_privateuse1() &&
           is_native_float(a.dtype()) && a.dtype() == b.dtype() &&
           index_of(a.device()) == index_of(b.device());
}

// A dense tensor's memory under a new contiguous shape.
Tensor reshape_dense(const Tensor& t, const std::vector<int64_t>& sizes) {
    const Tensor c = contiguous_on_device(t);
    const auto& impl = c.unsafeGetTensorImpl();
    std::vector<int64_t> strides(sizes.size(), 1);
    for (int64_t i = static_cast<int64_t>(sizes.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * std::max<int64_t>(sizes[i + 1], 1);
    }
    return Tensor(impl->storage(), sizes, strides, c.dtype(), impl->storage_offset());
}

Tensor mm_native(const Tensor& a, const Tensor& b) {
    const int64_t M = a.sizes()[0], N = b.sizes()[1];
    Tensor out = empty_on({M, N}, a.dtype(), index_of(a.device()));
    if (out.numel() == 0) return out;
    if (a.sizes()[1] == 0) {
        out.zero_();
        return out;
    }
    gemm(a, b, out, 1.0f, 0.0f);
    return out;
}

Tensor bmm_native(const Tensor& a, const Tensor& b) {
    const int64_t B = a.sizes()[0], M = a.sizes()[1], N = b.sizes()[2];
    Tensor out = empty_on({B, M, N}, a.dtype(), index_of(a.device()));
    if (out.numel() == 0) return out;
    if (a.sizes()[2] == 0) {
        out.zero_();
        return out;
    }
    gemm(a, b, out, 1.0f, 0.0f);
    return out;
}

Tensor mm_kernel(const Tensor& self, const Tensor& mat2) {
    if (!native_pair(self, mat2) || self.dim() != 2 || mat2.dim() != 2) {
        return cpu_fallback::Kernel<"mm", sig::mm>::call(self, mat2);
    }
    TP_CHECK(self.sizes()[1] == mat2.sizes()[0], "mm: shapes ", self.sizes()[0], "x",
             self.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1],
             " cannot be multiplied");
    return mm_native(self, mat2);
}

Tensor bmm_kernel(const Tensor& self, const Tensor& mat2) {
    if (!native_pair(self, mat2) || self.dim() != 3 || mat2.dim() != 3) {
        return cpu_fallback::Kernel<"bmm", sig::bmm>::call(self, mat2);
    }
    TP_CHECK(self.sizes()[0] == mat2.sizes()[0] && self.sizes()[2] == mat2.sizes()[1],
             "bmm: batch or inner dimensions do not match");
    return bmm_native(self, mat2);
}

Tensor addmm_kernel(const Tensor& self, const Tensor& mat1, const Tensor& mat2,
                    const Scalar& beta, const Scalar& alpha) {
    if (!native_pair(mat1, mat2) || mat1.dim() != 2 || mat2.dim() != 2 ||
        !self.device().is_privateuse1() || self.dtype() != mat1.dtype()) {
        return cpu_fallback::Kernel<"addmm", sig::addmm>::call(self, mat1, mat2, beta, alpha);
    }
    TP_CHECK(mat1.sizes()[1] == mat2.sizes()[0], "addmm: shapes cannot be multiplied");
    const int64_t M = mat1.sizes()[0], N = mat2.sizes()[1];
    Tensor out = empty_on({M, N}, mat1.dtype(), index_of(mat1.device()));
    if (out.numel() == 0) return out;
    const float b = static_cast<float>(beta.toDouble());
    const float a = static_cast<float>(alpha.toDouble());
    if (b != 0.0f) {
        // The bias goes into the output first and the product accumulates
        // onto it.
        strided_copy(out, self.dim() == 2 && self.sizes().vec() == out.sizes().vec()
                              ? self
                              : self.expand({M, N}));
    }
    if (mat1.sizes()[1] == 0) {
        if (b == 0.0f) out.zero_();
        else if (b != 1.0f) out.mul_(Scalar(static_cast<double>(b)));
        return out;
    }
    gemm(mat1, mat2, out, a, b);
    return out;
}

// matmul over the broadcast batch dimensions: every case becomes one mm or
// one bmm on dense operands.
Tensor matmul_kernel(const Tensor& self, const Tensor& other) {
    if (!native_pair(self, other) || self.dim() == 0 || other.dim() == 0) {
        return cpu_fallback::Kernel<"matmul", sig::matmul>::call(self, other);
    }
    const int64_t da = self.dim(), db = other.dim();
    if (da <= 2 && db <= 2) {
        const Tensor a = da == 1 ? reshape_dense(self, {1, self.sizes()[0]}) : self;
        const Tensor b = db == 1 ? reshape_dense(other, {other.sizes()[0], 1}) : other;
        TP_CHECK(a.sizes()[1] == b.sizes()[0], "matmul: inner dimensions do not match");
        Tensor out = mm_native(a, b);
        std::vector<int64_t> shape;
        if (da == 2) shape.push_back(a.sizes()[0]);
        if (db == 2) shape.push_back(b.sizes()[1]);
        return reshape_dense(out, shape);
    }
    if (db <= 2 && da >= 3) {
        // [..., M, K] @ [K, N]: fold the batch into the rows.
        const auto sa = self.sizes().vec();
        const int64_t K = sa.back();
        const int64_t rows = self.numel() / std::max<int64_t>(K, 1);
        const Tensor b = db == 1 ? reshape_dense(other, {other.sizes()[0], 1}) : other;
        TP_CHECK(b.sizes()[0] == K, "matmul: inner dimensions do not match");
        Tensor out = mm_native(reshape_dense(self, {rows, K}), b);
        std::vector<int64_t> shape(sa.begin(), sa.end() - 1);
        if (db == 2) shape.push_back(b.sizes()[1]);
        return reshape_dense(out, shape);
    }
    // General case: broadcast the batch dimensions, then one bmm.
    const Tensor a = da == 1 ? reshape_dense(self, {1, self.sizes()[0]}) : self;
    const Tensor b = db == 1 ? reshape_dense(other, {other.sizes()[0], 1}) : other;
    const auto sa = a.sizes().vec(), sb = b.sizes().vec();
    const int64_t M = sa[sa.size() - 2], K = sa.back(), N = sb.back();
    TP_CHECK(sb[sb.size() - 2] == K, "matmul: inner dimensions do not match");
    std::vector<int64_t> batch = broadcast_shape(std::vector<int64_t>(sa.begin(), sa.end() - 2),
                                                 std::vector<int64_t>(sb.begin(), sb.end() - 2));
    const int64_t B = std::accumulate(batch.begin(), batch.end(), int64_t(1),
                                      std::multiplies<int64_t>());
    auto expanded = [&](const Tensor& t, int64_t r, int64_t c) {
        std::vector<int64_t> shape = batch;
        shape.push_back(r);
        shape.push_back(c);
        const Tensor e = t.sizes().vec() == shape ? t : t.expand(shape);
        return reshape_dense(e, {B, r, c});
    };
    Tensor out = bmm_native(expanded(a, M, K), expanded(b, K, N));
    std::vector<int64_t> shape = batch;
    if (da > 1) shape.push_back(M);
    if (db > 1) shape.push_back(N);
    return reshape_dense(out, shape);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, MLUBlasKernels) {
    m.impl("mm", static_cast<sig::mm*>(&mm_kernel));
    m.impl("bmm", static_cast<sig::bmm*>(&bmm_kernel));
    m.impl("addmm", static_cast<sig::addmm*>(&addmm_kernel));
    m.impl("matmul", static_cast<sig::matmul*>(&matmul_kernel));
}

} // namespace mlu
} // namespace tensorplay
