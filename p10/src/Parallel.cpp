#include "Parallel.h"
#include "Exception.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif

namespace tensorplay {
namespace parallel {

namespace {

constexpr int NOT_SET = -1;
constexpr int CONSUMED = -2;

// Number of intra-op threads set by the user.
// NOT_SET -> positive value -> CONSUMED (pool created)
std::atomic<int> num_intraop_threads{NOT_SET};

thread_local bool in_parallel_region_ = false;
thread_local int thread_num_ = 0;
// Set on intraop-pool threads only; lets in_parallel_region() answer without
// taking the pool mutex (the old implementation locked + scanned thread ids
// on every parallel_for call from the main thread).
thread_local bool t_is_pool_thread_ = false;

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
inline void cpu_relax() { _mm_pause(); }
#else
inline void cpu_relax() { std::this_thread::yield(); }
#endif

// TaskThreadPoolBase::defaultNumThreads() (cpuinfo "cores" vs "processors"):
// SMT siblings share execution ports, so defaulting to logical CPUs
// oversubscribes compute-bound elementwise kernels.
int physical_core_count() {
  int logical = static_cast<int>(std::thread::hardware_concurrency());
#ifdef __linux__
  FILE* f = std::fopen("/proc/cpuinfo", "re");
  if (!f) {
    return logical;
  }
  int siblings = -1;
  int cores = -1;
  char line[256];
  while (std::fgets(line, sizeof(line), f)) {
    int v = -1;
    if (siblings < 0 && std::sscanf(line, "siblings : %d", &v) == 1 && v > 0) {
      siblings = v;
    } else if ((v = -1, std::sscanf(line, "cpu cores : %d", &v) == 1) && v > 0) {
      cores = v;
    }
    if (siblings > 0 && cores > 0) {
      break;
    }
  }
  std::fclose(f);
  if (siblings > 0 && cores > 0 && siblings > cores) {
    return std::max(1, logical * cores / siblings);
  }
#endif
  return logical;
}

int intraop_default_num_threads() {
  // Computed once: the default path parses /proc/cpuinfo, which costs
  // ~100us+ per call on many-core machines -- fatal for per-op hot paths.
  static const int cached = [] {
    const char* env = std::getenv("OMP_NUM_THREADS");
    if (env) {
      int n = std::atoi(env);
      if (n > 0) {
        return n;
      }
    }
    env = std::getenv("MKL_NUM_THREADS");
    if (env) {
      int n = std::atoi(env);
      if (n > 0) {
        return n;
      }
    }
    // On a part with two CPU frequency tiers the default is the capped
    // count, so the OpenMP team, the BLAS thread count, the per-thread
    // scratch sizing and the thread count compiled into generated kernels
    // all agree.  Counts that differ between consecutive regions make the
    // OpenMP runtime resize its team at every region.
    int n = physical_core_count();
    const int cap = internal::hybrid_thread_cap();
    if (cap > 0 && cap < n) {
      n = cap;
    }
    return n;
  }();
  return cached;
}

bool env_pins_thread_count() {
  const char* env = std::getenv("OMP_NUM_THREADS");
  if (env && std::atoi(env) > 0) {
    return true;
  }
  env = std::getenv("MKL_NUM_THREADS");
  return env && std::atoi(env) > 0;
}

// Team-size cap for OpenMP-backed GEMM kernels (oneDNN) on parts with two
// CPU frequency tiers lives in parallel::internal (see hybrid_thread_cap).

// Persistent pool of worker threads. The calling (master) thread participates
// in the work itself, so the pool holds nthreads - 1 workers.
class IntraopThreadPool {
 public:
  explicit IntraopThreadPool(int num_workers) {
    for (int i = 0; i < num_workers; ++i) {
      threads_.emplace_back([this] { worker_loop(); });
    }
  }

  IntraopThreadPool(const IntraopThreadPool&) = delete;
  IntraopThreadPool& operator=(const IntraopThreadPool&) = delete;

  ~IntraopThreadPool() {
    {
      std::unique_lock<std::mutex> lk(mutex_);
      running_ = false;
      condition_.notify_all();
    }
    for (auto& t : threads_) {
      t.join();
    }
  }

  size_t size() const { return threads_.size(); }

  void run(std::function<void()> func) {
    {
      std::unique_lock<std::mutex> lk(mutex_);
      tasks_.push(std::move(func));
    }
    // Publish after the push (release), then wake.  Spinning workers poll
    // pending_ lock-free and only fall through to the condvar when idle for
    // longer than the spin budget.
    pending_.fetch_add(1, std::memory_order_release);
    condition_.notify_one();
  }

 private:
  // Steals one queued task if any is visibly pending.  Returns false both
  // when the queue looks empty and when a peer stole it between our load and
  // the locked pop; callers simply retry or fall asleep.
  bool try_pop(std::function<void()>& out) {
    if (pending_.load(std::memory_order_acquire) == 0) {
      return false;
    }
    std::unique_lock<std::mutex> lk(mutex_);
    if (tasks_.empty()) {
      return false;
    }
    out = std::move(tasks_.front());
    tasks_.pop();
    pending_.fetch_sub(1, std::memory_order_relaxed);
    return true;
  }

  void worker_loop() {
    t_is_pool_thread_ = true;
    // Tiered idle wait.  A pure condvar park costs a futex wakeup (tens of us)
    // per op burst; an unbounded busy-wait makes idle workers steal physical
    // cores from tasks that are still computing (fatal once #tasks exceeds
    // #physical cores).  So: a few microseconds of tight polling to catch
    // immediate follow-on work, a short yield phase to stay responsive
    // without burning a core, then park.
    constexpr size_t kPauseSpins = 128; // ~0.5-2us of _mm_pause polling
    constexpr size_t kYieldSpins = 96;  // sched_yield decay before parking
    size_t spins = 0;
    while (true) {
      std::function<void()> task;
      bool got = false;
      while (spins < kPauseSpins) {
        if (!running_.load(std::memory_order_acquire)) break;
        if (try_pop(task)) { got = true; break; }
        cpu_relax();
        ++spins;
      }
      while (!got && spins < kPauseSpins + kYieldSpins) {
        if (!running_.load(std::memory_order_acquire)) break;
        if (try_pop(task)) { got = true; break; }
        std::this_thread::yield();
        ++spins;
      }
      if (!got) {
        std::unique_lock<std::mutex> lk(mutex_);
        condition_.wait(lk, [this] { return !running_ || !tasks_.empty(); });
        if (!running_ && tasks_.empty()) {
          return;
        }
        task = std::move(tasks_.front());
        tasks_.pop();
        pending_.fetch_sub(1, std::memory_order_relaxed);
      }
      spins = 0;
      task();
    }
  }

  std::queue<std::function<void()>> tasks_;
  std::vector<std::thread> threads_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::atomic<bool> running_{true};
  // Queue length snapshot for lock-free polling by spinning workers; own cache
  // line so producer mutex/queue traffic doesn't invalidate the polled line.
  alignas(64) std::atomic<size_t> pending_{0};
};

int _num_pool_threads(int nthreads) {
  if (nthreads == NOT_SET) {
    nthreads = intraop_default_num_threads();
  } else {
    nthreads = std::max(nthreads, 1);
  }
  // minus one because the calling (master) thread participates in the work
  return nthreads - 1;
}

IntraopThreadPool& get_intraop_pool() {
  static IntraopThreadPool pool(_num_pool_threads(num_intraop_threads.exchange(CONSUMED)));
  return pool;
}

// RAII guard for in_parallel_region() / get_thread_num() inside tasks.
struct ParallelRegionGuard {
  ParallelRegionGuard(int task_id) {
    internal::set_thread_num(task_id);
    in_parallel_region_ = true;
  }
  ParallelRegionGuard(const ParallelRegionGuard&) = delete;
  ParallelRegionGuard& operator=(const ParallelRegionGuard&) = delete;
  ~ParallelRegionGuard() {
    in_parallel_region_ = false;
    internal::set_thread_num(0);
  }
};

} // namespace

void init_num_threads() {
  int nthreads = num_intraop_threads.load();
  const bool user_pinned = nthreads > 0;
  if (nthreads <= 0) {
    nthreads = intraop_default_num_threads();
  }

#ifdef _OPENMP
  // oneDNN is invoked outside TensorPlay's native elementwise pool.  Keep its
  // ParallelOpenMP.cpp; forcing one thread here serializes every convolution.
  // The OpenMP team size also decides how oneDNN blocks its GEMM kernels at
  // primitive creation, so it must stay stable for the life of the process:
  // a cap re-applied after a primitive was built for a different team size
  // leaves part of the team idle, and the reverse floods the slow tier.
  int omp_threads = nthreads;
  if (!user_pinned && !env_pins_thread_count()) {
    const int cap = internal::hybrid_thread_cap();
    if (cap > 0 && cap < omp_threads) {
      omp_threads = cap;
    }
  }
  omp_set_num_threads(omp_threads);
#endif

#ifdef USE_MKL
  mkl_set_num_threads_local(nthreads);
  mkl_set_dynamic(false);
#endif
}

void set_num_threads(int nthreads) {
  TP_CHECK_VALUE(nthreads > 0, "Expected positive number of threads");
  int expected = NOT_SET;
  if (!num_intraop_threads.compare_exchange_strong(expected, nthreads)) {
    // num_intraop_threads either stores a positive integer or CONSUMED,
    // check that the requested size matches the current one
    int stored_nthreads = num_intraop_threads.load();
    if (stored_nthreads <= 0) {
      // plus one because of the master thread
      stored_nthreads = static_cast<int>(get_intraop_pool().size() + 1);
    }
    if (stored_nthreads != nthreads) {
      TP_WARN(
          "Cannot set number of intraop threads "
          "after parallel work has started or after set_num_threads call "
          "when using native parallel backend");
      return;
    }
  }

  // TensorPlay's own native parallel regions.
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#endif
#ifdef USE_MKL
  mkl_set_num_threads_local(nthreads);
  mkl_set_dynamic(false);
#endif
}

int get_num_threads() {
  internal::lazy_init_num_threads();
  int nthreads = num_intraop_threads.load();
  if (nthreads > 0) {
    return nthreads;
  }
  if (nthreads == NOT_SET) {
    return intraop_default_num_threads();
  }
  return static_cast<int>(get_intraop_pool().size() + 1);
}

// This parallel layer runs one shared worker pool that serves both intra-op
// (chunked kernel work) and inter-op (independent higher-level tasks)
// parallelism. There is no separately sized inter-op pool, so the inter-op
// entry points report the shared pool's size, and a resize request is only
// honored when it asks for the size the pool already has. A pool cannot be
// resized once parallel work has started, so any other request is an error.
void set_num_interop_threads(int nthreads) {
  TP_CHECK_VALUE(nthreads > 0, "Expected positive number of threads");
  int current = get_num_threads();
  if (nthreads == current) {
    return;
  }
  TP_THROW(
      RuntimeError,
      "cannot set number of interop threads to ",
      nthreads,
      ": the parallel layer uses a single shared thread pool whose size (",
      current,
      ") is configured via set_num_threads before parallel work has started");
}

int get_num_interop_threads() {
  return get_num_threads();
}

int get_thread_num() {
  return thread_num_;
}

bool in_parallel_region() {
  if (in_parallel_region_) {
    return true;
  }
  // t_is_pool_thread_ is a thread_local set at worker startup: no locks, no
  // thread-id scan on the hot parallel_for path.
  return num_intraop_threads.load() == CONSUMED && t_is_pool_thread_;
}

std::string get_parallel_info() {
  std::ostringstream ss;
  ss << "TensorPlay/Parallel:\n";
  ss << "\tget_num_threads() : " << get_num_threads() << '\n';
  ss << "\tget_thread_num() : " << get_thread_num() << '\n';
  ss << "\tin_parallel_region() : " << (in_parallel_region() ? "true" : "false") << '\n';
  ss << "\tGRAIN_SIZE : " << GRAIN_SIZE << '\n';
  ss << "\tstd::thread::hardware_concurrency() : "
     << std::thread::hardware_concurrency() << '\n';
  ss << "\tparallel backend : intraop thread pool\n";
  return ss.str();
}

namespace internal {

// Team-size cap for OpenMP-backed GEMM kernels (oneDNN) on parts with two
// CPU frequency tiers.  A team spanning every logical CPU spreads GEMM tiles
// onto the slow tier, and the slowest core paces the whole team through the
// kernel's internal barriers; the straggler effect then dominates the
// compute itself.  Cap the team near the fast-tier size, never below half
// the logical count.  Uniform-frequency parts keep the full team; explicit
// thread counts (environment or set_num_threads) are never second-guessed.
int hybrid_thread_cap() {
  static const int cached = []() -> int {
#ifdef __linux__
    std::vector<int> max_freqs;
    for (int cpu = 0; cpu < 512; ++cpu) {
      std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                         "/cpufreq/cpuinfo_max_freq";
      std::FILE* f = std::fopen(path.c_str(), "re");
      if (!f) break;
      int khz = 0;
      if (std::fscanf(f, "%d", &khz) == 1 && khz > 0) {
        max_freqs.push_back(khz);
      }
      std::fclose(f);
    }
    if (max_freqs.size() > 1) {
      const int top = *std::max_element(max_freqs.begin(), max_freqs.end());
      const int bottom = *std::min_element(max_freqs.begin(), max_freqs.end());
      if (top > bottom) {  // two tiers -> hybrid
        int fast = 0;
        for (int f : max_freqs) fast += (f == top);
        const int logical = static_cast<int>(max_freqs.size());
        return std::max(fast, logical / 2);
      }
    }
#endif
    return 0;
  }();
  return cached;
}

void set_thread_num(int id) {
  thread_num_ = id;
}

void invoke_parallel_impl(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const std::function<void(int64_t, int64_t)>& f) {
  const int64_t numiter = end - begin;
  const int64_t num_threads = get_num_threads();

#ifdef _OPENMP
  // Kernels run inside an OpenMP region so libgomp keeps the team warm
  // between dispatches.  The native pool below stays as the fallback for
  // non-OpenMP builds.
  //
  // TP_PARALLEL_BACKEND=native forces the in-house pool even in OpenMP builds.
  // Back-to-back small regions (RNN cell loops) measured ~600us cheaper per
  // region transition on Zen4 with the native pool: libgomp's post-barrier
  // spin taxes the serial op that follows each region.
  static const bool force_native_pool = [] {
      const char* e = std::getenv("TP_PARALLEL_BACKEND");
      return e && std::strcmp(e, "native") == 0;
  }();
  if (!force_native_pool && omp_get_max_threads() > 1 && !omp_in_parallel()) {
    std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr eptr;
    // The region always opens with the full team and the chunking is decided
    // inside it.  A num_threads clause that differs from one region to the
    // next makes libgomp retire the surplus workers and spawn them again at
    // every change, so a short loop followed by a long one pays thread
    // creation, and the fresh threads start on whichever CPU happens to be
    // idle.  A loop with fewer chunks than threads leaves the surplus
    // threads without a chunk instead.
    #pragma omp parallel
    {
      // Never more chunks than the configured intra-op count: kernels size
      // their per-thread scratch from get_num_threads() and index it with
      // get_thread_num().
      int64_t team = std::min<int64_t>(omp_get_num_threads(), num_threads);
      if (grain_size > 0) {
        team = std::min(team, (numiter + grain_size - 1) / grain_size);
      }
      team = std::max<int64_t>(team, 1);
      const int64_t tid = omp_get_thread_num();
      const int64_t chunk = (numiter + team - 1) / team;
      const int64_t local_begin = begin + tid * chunk;
      if (tid < team && local_begin < end) {
        try {
          ParallelRegionGuard guard(static_cast<int>(tid));
          f(local_begin, std::min(end, local_begin + chunk));
        } catch (...) {
          if (!err_flag.test_and_set()) {
            eptr = std::current_exception();
          }
        }
      }
    }
    if (eptr) {
      std::rethrow_exception(eptr);
    }
    return;
  }
#endif

  // Chunk size is at least grain_size, matching the semantics of
  // parallel_for: work below grain_size never reaches this point.
  int64_t chunk_size = std::max(grain_size, (numiter + num_threads - 1) / num_threads);
  const size_t num_tasks = static_cast<size_t>((numiter + chunk_size - 1) / chunk_size);

  struct State {
    std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr eptr;
    std::mutex mutex;
    std::atomic<size_t> remaining{0};
    std::condition_variable cv;
  } state;

  auto task = [&state, f, begin, chunk_size, end](size_t task_id) {
    int64_t local_begin = begin + static_cast<int64_t>(task_id) * chunk_size;
    if (local_begin < end) {
      int64_t local_end = std::min(end, chunk_size + local_begin);
      try {
        ParallelRegionGuard guard(static_cast<int>(task_id));
        f(local_begin, local_end);
      } catch (...) {
        if (!state.err_flag.test_and_set()) {
          state.eptr = std::current_exception();
        }
      }
    }
    {
      std::unique_lock<std::mutex> lk(state.mutex);
      if (--state.remaining == 0) {
        state.cv.notify_one();
      }
    }
  };

  state.remaining = num_tasks;
  for (size_t i = 1; i < num_tasks; ++i) {
    get_intraop_pool().run(std::bind(task, i));
  }
  // Run the first task on the calling thread directly.
  task(0);

  // Wait for all tasks to finish.
  {
    std::unique_lock<std::mutex> lk(state.mutex);
    if (state.remaining != 0) {
      state.cv.wait(lk);
    }
  }
  if (state.eptr) {
    std::rethrow_exception(state.eptr);
  }
}

} // namespace internal
} // namespace parallel
} // namespace tensorplay

void tp_parallel_for_c(
    long long begin,
    long long end,
    long long grain,
    tp_parallel_body_c body,
    void* ctx) {
  tensorplay::parallel::parallel_for(
      static_cast<int64_t>(begin),
      static_cast<int64_t>(end),
      static_cast<int64_t>(grain),
      [body, ctx](int64_t b, int64_t e) { body(ctx, b, e); });
}
