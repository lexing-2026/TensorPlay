#pragma once

#include "Macros.h"
#include <cstdint>
#include <functional>
#include <string>

namespace tensorplay {
namespace parallel {

// Default grain size for elementwise operations. A parallel_for is only
// dispatched to the thread pool when the iteration count exceeds the grain
// size; below that threshold it runs serially on the calling thread. This
// keeps tiny operators (e.g. a 1-element add) free of per-op thread
// overhead even when many threads are configured.
constexpr int64_t GRAIN_SIZE = 32768;

// Called during new thread initialization.
P10_API void init_num_threads();

// Sets the number of threads to be used in a parallel region.
P10_API void set_num_threads(int nthreads);

// Returns the maximum number of threads that may be used in a parallel region.
P10_API int get_num_threads();

// Returns the current thread number (starting from 0) inside a parallel
// region, or 0 in the sequential region.
P10_API int get_thread_num();

// Checks whether the code runs in a parallel region.
P10_API bool in_parallel_region();

// Returns a detailed string describing parallelization settings.
P10_API std::string get_parallel_info();

namespace internal {

// Initializes num_threads lazily at the first parallel call.
inline void lazy_init_num_threads() {
  static bool init = false;
  if (!init) {
    init_num_threads();
    init = true;
  }
}

P10_API void set_thread_num(int id);

class ThreadIdGuard {
 public:
  ThreadIdGuard(int new_id) : old_id_(get_thread_num()) { set_thread_num(new_id); }
  ThreadIdGuard(const ThreadIdGuard&) = delete;
  ThreadIdGuard& operator=(const ThreadIdGuard&) = delete;
  ~ThreadIdGuard() { set_thread_num(old_id_); }

 private:
  int old_id_;
};

template <class F>
void invoke_parallel(int64_t begin, int64_t end, int64_t grain_size, const F& f);

// Non-template core dispatched by invoke_parallel; implemented in Parallel.cpp.
P10_API void invoke_parallel_impl(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const std::function<void(int64_t, int64_t)>& f);

} // namespace internal

/*
parallel_for

begin: index at which to start applying user function
end:   index at which to stop applying user function
grain_size: number of elements per chunk. impacts the degree of parallelization.
f: user function applied in parallel to the chunks, signature:
   void f(int64_t begin, int64_t end)

Runs serially on the calling thread when the iteration count does not exceed
grain_size, when a single thread is configured, or when already inside a
parallel region. Otherwise dispatches the work to the intra-op thread pool.
*/
template <class F>
inline void parallel_for(int64_t begin, int64_t end, int64_t grain_size, const F& f) {
  if (begin >= end) {
    return;
  }
  internal::lazy_init_num_threads();
  const int64_t numiter = end - begin;
  const bool use_parallel =
      (numiter > grain_size && numiter > 1 && !in_parallel_region() && get_num_threads() > 1);
  if (!use_parallel) {
    internal::ThreadIdGuard tid_guard(0);
    f(begin, end);
    return;
  }
  internal::invoke_parallel(begin, end, grain_size, f);
}

namespace internal {

template <class F>
void invoke_parallel(int64_t begin, int64_t end, int64_t grain_size, const F& f) {
  invoke_parallel_impl(begin, end, grain_size, [&f](int64_t b, int64_t e) { f(b, e); });
}

} // namespace internal

} // namespace parallel
} // namespace tensorplay

// C-ABI worksharing bridge: lets runtime-generated kernels share the same
// intra-op pool as every in-tree kernel instead of carrying a second thread
// runtime.  The body callback receives ``[begin, end)`` per chunk.
extern "C" {

typedef void (*tp_parallel_body_c)(void* ctx, long long begin, long long end);

P10_API void tp_parallel_for_c(
    long long begin,
    long long end,
    long long grain,
    tp_parallel_body_c body,
    void* ctx);

} // extern "C"