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

// Configures the number of threads assigned to inter-op parallelism
// (independent tasks submitted by higher-level execution layers).
P10_API void set_num_interop_threads(int nthreads);

// Returns the number of threads assigned to inter-op parallelism.
P10_API int get_num_interop_threads();

// Returns the current thread number (starting from 0) inside a parallel
// region, or 0 in the sequential region.
P10_API int get_thread_num();

// Checks whether the code runs in a parallel region.
P10_API bool in_parallel_region();

// Returns a detailed string describing parallelization settings.
P10_API std::string get_parallel_info();

namespace internal {

// Initializes num_threads lazily on the first parallel call of each thread;
// the OpenMP team-size ICV is per thread, so every joining thread must run
// this before dispatching parallel work.
inline void lazy_init_num_threads() {
  static thread_local bool init = false;
  if (!init) {
    init_num_threads();
    init = true;
  }
}

// Team-size cap for OpenMP-backed GEMM kernels (oneDNN) on hybrid CPUs with
// two frequency tiers; 0 when no cap applies (uniform-frequency parts).
// Explicit thread counts (environment, set_num_threads) bypass it.
P10_API int hybrid_thread_cap();

// True when the thread count was fixed explicitly -- a set_num_threads call
// or a positive OMP_NUM_THREADS/MKL_NUM_THREADS. Cap heuristics must not
// second-guess an explicit configuration.
P10_API bool user_pins_thread_count();

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

template <class F>
void invoke_parallel_balanced(int64_t begin, int64_t end, int64_t grain_size,
                              const F& f);

template <class F>
void invoke_parallel_weighted(int64_t begin, int64_t end, int64_t grain_size,
                              const F& f);

// Non-template core dispatched by invoke_parallel; implemented in Parallel.cpp.
P10_API void invoke_parallel_impl(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const std::function<void(int64_t, int64_t)>& f);

// Non-template core dispatched by invoke_parallel_balanced; implemented in
// Parallel.cpp.  The work is handed out range by range from a shared counter
// rather than split once among the team, so a thread that answers quickly
// takes more of the walk than one that answers slowly.  The size of a range
// is decided here from the team and the iteration count; grain_size only
// decides, as in parallel_for, whether the walk is worth spreading.  A body
// is safe here only when it writes the range it is given and keeps nothing of
// its own per thread: a body that folds into a per-thread slot must stay with
// the split, because a slot would then hold several ranges.
P10_API void invoke_parallel_balanced_impl(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const std::function<void(int64_t, int64_t)>& f);

// Non-template core dispatched by invoke_parallel_weighted; implemented in
// Parallel.cpp.  The walk is split once among the team, as in the plain
// split, but the shares follow the capacity each core publishes rather than
// the thread count alone.  A team that spans two kinds of core answers a
// range at two rates, and an even split leaves the slower kind holding the
// closing stretch while the faster kind has nothing left: the walk then ends
// when the slower cores are done, not when the team is.  A body here keeps
// the same contract as the plain split -- it may fold into a per-thread slot
// indexed by get_thread_num(), because it sees one range.
P10_API void invoke_parallel_weighted_impl(
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

/*
parallel_for_balanced

begin: index at which to start applying user function
end:   index at which to stop applying user function
grain_size: iteration count below which the walk is not spread.
f: user function applied to each handed-out range, signature:
   void f(int64_t begin, int64_t end)

The same guards as parallel_for.  The ranges the team answers come from a
counter the threads draw on as they free up, so a thread on a core that
answers the walk quickly draws more ranges than a thread on one that answers
it slowly.  Use it for a walk that only writes the range it is handed, and
whose range takes longer to answer than the counter takes to reach the next
thread: a walk that answers a range sooner spends its time queueing on that
one line, and the fixed split has no such line.
*/
template <class F>
inline void parallel_for_balanced(int64_t begin, int64_t end,
                                  int64_t grain_size, const F& f) {
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
  internal::invoke_parallel_balanced(begin, end, grain_size, f);
}

/*
parallel_for_weighted

begin: index at which to start applying user function
end:   index at which to stop applying user function
grain_size: iteration count below which the walk is not spread.
f: user function applied to each share, signature:
   void f(int64_t begin, int64_t end)

The same guards as parallel_for.  The walk is cut once, but the cut follows
the capacity the firmware publishes for the core each thread is running on,
so a team that spans two kinds of core finishes together instead of waiting
for the slower kind.  Use it where the per-thread contract of the plain split
is enough and the walk costs enough per element for the capacity gap to show.
*/
template <class F>
inline void parallel_for_weighted(int64_t begin, int64_t end,
                                  int64_t grain_size, const F& f) {
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
  internal::invoke_parallel_weighted(begin, end, grain_size, f);
}

namespace internal {

template <class F>
void invoke_parallel(int64_t begin, int64_t end, int64_t grain_size, const F& f) {
  invoke_parallel_impl(begin, end, grain_size, [&f](int64_t b, int64_t e) { f(b, e); });
}

template <class F>
void invoke_parallel_balanced(int64_t begin, int64_t end, int64_t grain_size,
                              const F& f) {
  invoke_parallel_balanced_impl(
      begin, end, grain_size, [&f](int64_t b, int64_t e) { f(b, e); });
}

template <class F>
void invoke_parallel_weighted(int64_t begin, int64_t end, int64_t grain_size,
                              const F& f) {
  invoke_parallel_weighted_impl(
      begin, end, grain_size, [&f](int64_t b, int64_t e) { f(b, e); });
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