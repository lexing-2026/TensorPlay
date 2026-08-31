#include "OneDNNContext.h"

#ifdef USE_ONEDNN
#include "Parallel.h"
#include <functional>
#include <algorithm>
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#endif
#endif

namespace tensorplay {

#ifdef USE_ONEDNN

std::atomic<bool> OneDNNContext::enabled_(true);

bool OneDNNContext::is_available() {
    return true;
}

dnnl::engine& OneDNNContext::get_engine() {
    static dnnl::engine* eng = new dnnl::engine(dnnl::engine::kind::cpu, 0);
    return *eng;
}

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
namespace {

// Bridges oneDNN's primitive partitioning onto TensorPlay's intra-op pool.
// Synchronous contract (no ASYNCHRONOUS flag): parallel_for() blocks until
// every submitted chunk finished, mirroring how the pool already serves
// pointwise kernels -- one shared worker team, no OMP oversubscription.
//
// Note: oneDNN must not be driven from inside a callback that is itself
// running on the intra-op pool (the same single-pool constraint that
// applies to nested pointwise work).
class DnnlThreadpoolBridge final
    : public dnnl::threadpool_interop::threadpool_iface {
public:
    int get_num_threads() const override {
        return std::max(1, parallel::get_num_threads());
    }

    bool get_in_parallel() const override {
        return t_inside_job_ || parallel::in_parallel_region();
    }

    uint64_t get_flags() const override { return 0; /* synchronous */ }

    void parallel_for(int n,
                      const std::function<void(int, int)>& fn) override {
        if (n <= 0) return;
        // oneDNN contract (dnnl_threadpool_iface.hpp):
        //     for (int i = 0; i < n; i++) fn(i, n);
        // Each of the n instances does the work of "thread i of n", so fn
        // must be invoked once per instance with the TOTAL count. Passing
        // sub-ranges (b, e) instead makes the library read them as
        // (ithr, nthr): every instance then computes only a partial share
        // (corrupting reduction-based kernels such as the gemm conv
        // backward-weights path, and dropping most of the work when the
        // pool runs a single chunk).
        const int nthr = get_num_threads();
        if (n == 1) {
            parallel::internal::ThreadIdGuard tid_guard(0);
            fn(0, 1);
            return;
        }
        if (nthr <= 1) {
            parallel::internal::ThreadIdGuard tid_guard(0);
            for (int i = 0; i < n; ++i) fn(i, n);
            return;
        }
        // Chunking (≈num_threads slices, chunk >= grain) is delegated so
        // primitive execution follows the exact same partition/threading
        // policy -- including the warm-team OpenMP fast path -- as every
        // other TensorPlay kernel.  Synchronous: returns after all chunks.
        JobCtx ctx{fn, n};
        parallel::internal::invoke_parallel_impl(
            0, static_cast<int64_t>(n), /*grain=*/1,
            [&ctx](int64_t b, int64_t e) {
                ThreadLocalFlag guard;
                for (int64_t i = b; i < e; ++i)
                    ctx.fn(static_cast<int>(i), ctx.n);
            });
    }

private:
    struct JobCtx {
        const std::function<void(int, int)>& fn;
        int n;
    };
    // RAII marker so get_in_parallel() reports truthfully for any nested
    // threadpool queries issued while a bridge job runs.
    struct ThreadLocalFlag {
        ThreadLocalFlag() { t_inside_job_ = true; }
        ~ThreadLocalFlag() { t_inside_job_ = false; }
    };
    static inline thread_local bool t_inside_job_ = false;
};

DnnlThreadpoolBridge& get_bridge() {
    static DnnlThreadpoolBridge* bridge = new DnnlThreadpoolBridge();
    return *bridge;
}

} // namespace
#endif // THREADPOOL runtime

dnnl::stream& OneDNNContext::get_stream() {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    static dnnl::stream* s = new dnnl::stream(
        dnnl::threadpool_interop::make_stream(get_engine(), &get_bridge()));
    return *s;
#else
    static dnnl::stream* s = new dnnl::stream(get_engine());
    return *s;
#endif
}

#else
std::atomic<bool> OneDNNContext::enabled_(false);

bool OneDNNContext::is_available() {
    return false;
}
#endif

bool OneDNNContext::is_enabled() {
    return enabled_.load();
}

void OneDNNContext::set_enabled(bool enabled) {
    enabled_.store(enabled);
}

} // namespace tensorplay
