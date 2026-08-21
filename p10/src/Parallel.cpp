#include "Parallel.h"
#include "Exception.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <queue>
#include <sstream>
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

int intraop_default_num_threads() {
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
  unsigned int hc = std::thread::hardware_concurrency();
  return hc > 0 ? static_cast<int>(hc) : 1;
}

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

  bool in_thread_pool() const {
    std::thread::id id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(mutex_);
    for (const auto& t : threads_) {
      if (t.get_id() == id) {
        return true;
      }
    }
    return false;
  }

  void run(std::function<void()> func) {
    {
      std::unique_lock<std::mutex> lk(mutex_);
      tasks_.push(std::move(func));
      condition_.notify_one();
    }
  }

 private:
  void worker_loop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lk(mutex_);
        condition_.wait(lk, [this] { return !running_ || !tasks_.empty(); });
        if (!running_ && tasks_.empty()) {
          return;
        }
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
    }
  }

  std::queue<std::function<void()>> tasks_;
  std::vector<std::thread> threads_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::atomic<bool> running_{true};
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
  if (nthreads <= 0) {
    nthreads = intraop_default_num_threads();
  }

#ifdef _OPENMP
  // oneDNN is invoked outside TensorPlay's native elementwise pool.  Keep its
  // OpenMP team aligned with the public intra-op setting, as PyTorch does in
  // ParallelOpenMP.cpp; forcing one thread here serializes every convolution.
  omp_set_num_threads(nthreads);
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

  // Match PyTorch's thread contract for the oneDNN/OpenMP kernels as well as
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

int get_thread_num() {
  return thread_num_;
}

bool in_parallel_region() {
  if (in_parallel_region_) {
    return true;
  }
  return num_intraop_threads.load() == CONSUMED &&
      get_intraop_pool().in_thread_pool();
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
