// Whether the generic matrix product accepts four-bit weights in groups of
// thirty-two, and how fast it runs when it does.
//
// The library's resident-weight product only takes groups of sixty-four and
// up, but the general product takes the group size from the caller and its
// documentation admits four-bit weights when the weights are group-wise
// quantized.  This builds that problem directly against the library: a
// single-row activation, a four-bit weight stored two codes to a byte with
// the first code in the low field, and one scale per group of thirty-two
// along the contracted axis.  The result is compared with a product of the
// values the codes stand for, then timed.
//
//   g++ -O2 -I$NEUWARE_HOME/include probe_vendor_cnnl.cpp \
//       -L$NEUWARE_HOME/lib64 -lcnnl -lcnrt -o probe_vendor_cnnl
//   ./probe_vendor_cnnl [M] [K] [N] [group] [repeat]

#include <cnnl.h>
#include <cnrt.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

const char* g_stage = "init";

#define STAGE(name) (g_stage = (name))
#define CK(expr)                                                                     \
    do {                                                                             \
        cnnlStatus_t s_ = (expr);                                                    \
        if (s_ != CNNL_STATUS_SUCCESS) {                                             \
            std::printf("  [%s] %s -> cnnl status %d\n", g_stage, #expr, (int)s_);   \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)
#define RK(expr)                                                                     \
    do {                                                                             \
        cnrtRet_t s_ = (expr);                                                       \
        if (s_ != cnrtSuccess) {                                                     \
            std::printf("  [%s] %s -> cnrt status %d\n", g_stage, #expr, (int)s_);   \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)
#define NOTE(...) (std::printf("  [%s] ", g_stage), std::printf(__VA_ARGS__))

// The half values are carried as raw 16-bit words on the host; only the
// conversion is needed here.
uint16_t to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000);
    const int exp = static_cast<int>((x >> 23) & 0xff) - 127 + 15;
    const uint16_t man = static_cast<uint16_t>((x >> 13) & 0x3ff);
    if (exp <= 0) return sign;
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00);
    return static_cast<uint16_t>(sign | (exp << 10) | man);
}

float from_half(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    const int exp = (h >> 10) & 0x1f;
    const uint32_t man = h & 0x3ff;
    uint32_t x = sign;
    if (exp != 0) x |= static_cast<uint32_t>(exp - 15 + 127) << 23 | man << 13;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

void host_tensor(cnnlTensorDescriptor_t* desc, cnnlDataType_t dtype, int dim,
                 const int64_t* sizes, const int64_t* strides) {
    CK(cnnlCreateTensorDescriptor(desc));
    CK(cnnlSetTensorDescriptorEx_v2(*desc, CNNL_LAYOUT_ARRAY, dtype, dim, sizes, strides));
}

} // namespace

int main(int argc, char** argv) {
    const int M = argc > 1 ? std::atoi(argv[1]) : 1;
    const int K = argc > 2 ? std::atoi(argv[2]) : 4096;
    const int N = argc > 3 ? std::atoi(argv[3]) : 4096;
    const int G = argc > 4 ? std::atoi(argv[4]) : 32;
    const int repeat = argc > 5 ? std::atoi(argv[5]) : 30;
    const int groups = K / G;

    std::printf("M=%d K=%d N=%d group=%d repeat=%d\n", M, K, N, G, repeat);

    std::mt19937 rng(0);
    std::uniform_int_distribution<int> code_dist(-8, 7);
    std::uniform_real_distribution<float> scale_dist(0.005f, 0.05f);
    std::normal_distribution<float> act_dist(0.0f, 1.0f);

    // Host operands.  Codes are signed four-bit values; a byte holds the
    // first of its pair in the low field, which is the stored order.
    std::vector<unsigned char> b_host(static_cast<size_t>(N) * K / 2);
    std::vector<int8_t> codes(static_cast<size_t>(N) * K);
    for (size_t i = 0; i < codes.size(); ++i) codes[i] = static_cast<int8_t>(code_dist(rng));
    for (size_t i = 0; i < codes.size(); i += 2) {
        b_host[i / 2] = static_cast<unsigned char>((codes[i] & 0xF) | ((codes[i + 1] & 0xF) << 4));
    }
    std::vector<uint16_t> scale_host(static_cast<size_t>(N) * groups);
    std::vector<float> scale_value(scale_host.size());
    for (size_t i = 0; i < scale_host.size(); ++i) {
        scale_value[i] = scale_dist(rng);
        scale_host[i] = to_half(scale_value[i]);
        scale_value[i] = from_half(scale_host[i]);
    }
    std::vector<uint16_t> a_host(static_cast<size_t>(M) * K);
    std::vector<float> a_value(a_host.size());
    for (size_t i = 0; i < a_host.size(); ++i) {
        a_value[i] = act_dist(rng);
        a_host[i] = to_half(a_value[i]);
        a_value[i] = from_half(a_host[i]);
    }

    // The value each code stands for, and the product of those values.
    std::vector<double> ref(static_cast<size_t>(M) * N, 0.0);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<double>(a_value[static_cast<size_t>(m) * K + k]) *
                       codes[static_cast<size_t>(n) * K + k] *
                       scale_value[static_cast<size_t>(n) * groups + k / G];
            }
            ref[static_cast<size_t>(m) * N + n] = acc;
        }
    }

    RK(cnrtSetDevice(0));
    cnrtQueue_t queue = nullptr;
    RK(cnrtQueueCreate(&queue));

    void *a_dev = nullptr, *b_dev = nullptr, *s_dev = nullptr, *c_dev = nullptr;
    RK(cnrtMalloc(&a_dev, a_host.size() * sizeof(uint16_t)));
    RK(cnrtMalloc(&b_dev, b_host.size()));
    RK(cnrtMalloc(&s_dev, scale_host.size() * sizeof(uint16_t)));
    RK(cnrtMalloc(&c_dev, static_cast<size_t>(M) * N * sizeof(uint16_t)));
    RK(cnrtMemcpy(a_dev, a_host.data(), a_host.size() * sizeof(uint16_t), cnrtMemcpyHostToDev));
    RK(cnrtMemcpy(b_dev, b_host.data(), b_host.size(), cnrtMemcpyHostToDev));
    RK(cnrtMemcpy(s_dev, scale_host.data(), scale_host.size() * sizeof(uint16_t),
                  cnrtMemcpyHostToDev));

    cnnlHandle_t handle = nullptr;
    STAGE("handle");
    CK(cnnlCreate(&handle));
    CK(cnnlSetQueue(handle, queue));

    // Descriptors.  The weight is described in its logical shape and the
    // four-bit element type, so the contraction length is the logical one.
    const int64_t a_sizes[2] = {M, K}, a_strides[2] = {K, 1};
    const int64_t b_sizes[2] = {N, K}, b_strides[2] = {K, 1};
    const int64_t c_sizes[2] = {M, N}, c_strides[2] = {N, 1};
    const int64_t s_sizes[2] = {N, groups}, s_strides[2] = {groups, 1};

    cnnlTensorDescriptor_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr;
    cnnlTensorDescriptor_t s_desc = nullptr;
    STAGE("tensors");
    host_tensor(&a_desc, CNNL_DTYPE_HALF, 2, a_sizes, a_strides);
    host_tensor(&b_desc, CNNL_DTYPE_INT4X2, 2, b_sizes, b_strides);
    host_tensor(&c_desc, CNNL_DTYPE_HALF, 2, c_sizes, c_strides);
    host_tensor(&s_desc, CNNL_DTYPE_HALF, 2, s_sizes, s_strides);

    cnnlMatMulDescriptor_t mm = nullptr;
    cnnlMatMulAlgo_t algo = nullptr;
    STAGE("matmul-desc");
    CK(cnnlMatMulDescCreate(&mm));
    const int32_t trans_a = 0, trans_b = 1, compute = CNNL_DTYPE_FLOAT;
    CK(cnnlSetMatMulDescAttr(mm, CNNL_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    CK(cnnlSetMatMulDescAttr(mm, CNNL_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
    CK(cnnlSetMatMulDescAttr(mm, CNNL_MATMUL_DESC_COMPUTE_TYPE, &compute, sizeof(compute)));

    // The activation carries no quantization of its own, but the library
    // wants both sides described together; its policy is "none" for the
    // left operand.
    cnnlQuantizeExDescriptor_t aq = nullptr, bq = nullptr;
    STAGE("quant-desc");
    CK(cnnlCreateQuantizeExDescriptor(&aq));
    CK(cnnlCreateQuantizeExDescriptor(&bq));
    CK(cnnlSetQuantizeExDescriptorQuantSchemeAndDtype(aq, CNNL_QUANTIZE_NONE, CNNL_DTYPE_HALF));
    // This attribute takes the scale by host pointer: the descriptor copies
    // it when it is set, which suits weights that do not change.
    CK(cnnlSetQuantizeExDescriptor(bq, nullptr, nullptr, s_desc, scale_host.data(), nullptr,
                                   nullptr, CNNL_QUANTIZE_GROUP_WISE, CNNL_QUANTIZE_SCALE,
                                   CNNL_POINTER_MODE_HOST, CNNL_DTYPE_HALF, G));
    CK(cnnlSetMatMulDescAttr(mm, CNNL_MATMUL_A_QUANT, &aq, sizeof(aq)));
    CK(cnnlSetMatMulDescAttr(mm, CNNL_MATMUL_B_QUANT, &bq, sizeof(bq)));

    STAGE("heuristic");
    CK(cnnlMatMulAlgoCreate(&algo));
    cnnlMatMulHeuristicResult_t result = nullptr;
    CK(cnnlCreateMatMulHeuristicResult(&result));
    int count = 0;
    cnnlStatus_t status = cnnlGetMatMulAlgoHeuristic(handle, mm, a_desc, b_desc, c_desc, c_desc,
                                                     nullptr, 1, &result, &count);
    NOTE("heuristic status %d count %d\n", (int)status, count);
    size_t workspace = 0;
    if (status == CNNL_STATUS_SUCCESS && count > 0) {
        CK(cnnlGetMatMulHeuristicResult(result, algo, &workspace));
    } else if (status != CNNL_STATUS_SUCCESS) {
        std::printf("  no algorithm offered; aborting\n");
        return 1;
    }
    NOTE("workspace %zu bytes\n", workspace);

    void* ws_dev = nullptr;
    if (workspace) RK(cnrtMalloc(&ws_dev, workspace));

    const float alpha = 1.0f, beta = 0.0f;
    STAGE("matmul");
    CK(cnnlMatMul_v2(handle, mm, algo, &alpha, a_desc, a_dev, b_desc, b_dev, &beta, c_desc, c_dev,
                     ws_dev, workspace, c_desc, c_dev));
    RK(cnrtQueueSync(queue));

    std::vector<uint16_t> out_host(static_cast<size_t>(M) * N);
    RK(cnrtMemcpy(out_host.data(), c_dev, out_host.size() * sizeof(uint16_t),
                  cnrtMemcpyDevToHost));

    double worst = 0.0, peak = 0.0;
    for (size_t i = 0; i < out_host.size(); ++i) {
        peak = std::max(peak, std::fabs(ref[i]));
        worst = std::max(worst, std::fabs(static_cast<double>(from_half(out_host[i])) - ref[i]));
    }
    NOTE("relative error %.2e\n", peak > 0 ? worst / peak : worst);

    auto run = [&] {
        CK(cnnlMatMul_v2(handle, mm, algo, &alpha, a_desc, a_dev, b_desc, b_dev, &beta, c_desc,
                         c_dev, ws_dev, workspace, c_desc, c_dev));
    };
    for (int i = 0; i < 3; ++i) run();
    RK(cnrtQueueSync(queue));
    cnrtNotifier_t t0 = nullptr, t1 = nullptr;
    RK(cnrtNotifierCreate(&t0));
    RK(cnrtNotifierCreate(&t1));
    RK(cnrtPlaceNotifier(t0, queue));
    for (int i = 0; i < repeat; ++i) run();
    RK(cnrtPlaceNotifier(t1, queue));
    RK(cnrtQueueSync(queue));
    float ms = 0.0f;
    RK(cnrtNotifierElapsedTime(t0, t1, &ms));
    ms /= repeat;
    const double bytes = static_cast<double>(N) * K / 2.0 + static_cast<double>(N) * groups * 2.0;
    NOTE("%.4f ms  %.1f GB/s\n", ms, bytes / (ms / 1e3) / 1e9);
    return 0;
}
