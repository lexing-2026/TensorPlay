#pragma once
#include <vector>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <thread>
#include <optional>
#include <atomic>
#include <unordered_map>
#include <cstdint>
#include "Macros.h"
#include "Tensor.h"
#include "Edge.h"
#include "Node.h"
#include "InputBuffer.h"
#include "GraphTask.h"

namespace tensorplay {
namespace tpx {

// Thread-safe ready queue ordered by sequence_nr (max first), matching
// CPU work; the thread that initiates backward() drains the CPU queue itself.
class TENSORPLAY_API ReadyQueue {
public:
    struct NodeTask {
        std::shared_ptr<Node> fn_;
        InputBuffer input_buffer_;
        // The graph this task belongs to. Raw pointer: the initiating thread
        // blocks until every enqueued task has been evaluated, so the
        // GraphTask always outlives its tasks.
        GraphTask* graph_;

        NodeTask(std::shared_ptr<Node> fn, InputBuffer input_buffer, GraphTask* graph)
            : fn_(std::move(fn)), input_buffer_(std::move(input_buffer)), graph_(graph) {}

        // Max heap by sequence_nr
        bool operator<(const NodeTask& other) const {
            return fn_->sequence_nr() < other.fn_->sequence_nr();
        }
    };

    // Pre-reserving the heap keeps the first graph's task churn from growing
    // the bucket repeatedly; queues are process-lifetime so the capacity
    // persists across all executions.
    void reserve(size_t n) { heap_.reserve(n); }

    void push(NodeTask task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            heap_.push_back(std::move(task));
            sift_up(heap_.size() - 1);
        }
        cv_.notify_one();
    }

    // Blocks until a task is available. `stop` is polled so a caller that is
    // only draining the queue on behalf of one GraphTask can leave when that
    // graph completes even if this queue stays idle.
    template <typename StopFn>
    std::optional<NodeTask> pop_until(const StopFn& stop) {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            if (!heap_.empty()) {
                // Take the last element; when it is the root (size == 1) no
                // reheapify is needed.  Swapping rather than assigning keeps
                // self-move out of the picture (front == back at size 1).
                NodeTask task = std::move(heap_.back());
                heap_.pop_back();
                if (!heap_.empty()) {
                    std::swap(heap_.front(), task);
                    sift_down(0);
                }
                return task;
            }
            if (stop()) return std::nullopt;
            // Short poll: the stop predicate turns true as soon as this
            // thread's GraphTask completes on another queue.
            cv_.wait_for(lock, std::chrono::milliseconds(5));
        }
    }

    // Wake any thread blocked in pop_until without enqueuing work. Used to
    // unblock the initiating thread the instant its GraphTask completes on a
    // device worker; otherwise it would wait out a full poll interval above.
    // The mutex is taken so the notify cannot be lost between pop_until's
    // a dummy wakeup task under the queue lock).
    void notify() {
        std::lock_guard<std::mutex> lock(mutex_);
        cv_.notify_all();
    }

private:
    // Flat binary max-heap over a vector (ordering by sequence_nr, max at
    // root).  The vector is reserved at construction, so steady-state
    // push/pop never allocates; a single mutex still guards it.
    void sift_up(size_t idx) {
        while (idx > 0) {
            size_t parent = (idx - 1) / 2;
            if (!(heap_[parent] < heap_[idx])) break;
            std::swap(heap_[parent], heap_[idx]);
            idx = parent;
        }
    }

    void sift_down(size_t idx) {
        const size_t n = heap_.size();
        for (;;) {
            size_t largest = idx;
            size_t l = 2 * idx + 1;
            size_t r = l + 1;
            if (l < n && heap_[largest] < heap_[l]) largest = l;
            if (r < n && heap_[largest] < heap_[r]) largest = r;
            if (largest == idx) break;
            std::swap(heap_[idx], heap_[largest]);
            idx = largest;
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<NodeTask> heap_;
};

class TENSORPLAY_API Engine {
private:
public:
    static Engine& get_default_engine();
    static bool current_graph_task_keep_graph();

    ~Engine();

    // Executes the graph rooted at `roots` with the given `inputs`.
    // accumulate_grad == true corresponds to backward(), false to grad().
    // Returns the captured gradients for `outputs` (empty for backward()).
    variable_list execute(const edge_list& roots,
                          const variable_list& inputs,
                          bool keep_graph,
                          bool create_graph,
                          bool accumulate_grad,
                          const edge_list& outputs);

    void queue_callback(std::function<void()> callback);

private:
    Engine() = default;

    void compute_dependencies(Node* root, GraphTask& task, uint64_t min_topo_nr);

    // Evaluates one dequeued function and distributes its outputs. When
    // `local_queue` is non-null the engine runs in nested (reentrant) mode:
    // every follow-up task is routed there instead of device queues, which
    // keeps reentrant backward deadlock-free.
    void evaluate_function(GraphTask& task, Node* func, InputBuffer& inputs,
                           ReadyQueue& cpu_queue, ReadyQueue* local_queue);

    // Routes a follow-up task: CUDA devices get dedicated worker threads,
    // everything else lands on the CPU queue processed by the initiator.
    void enqueue_task(GraphTask& task, ReadyQueue::NodeTask&& node_task,
                      ReadyQueue& cpu_queue, ReadyQueue* local_queue);

    // Entry point shared by device workers and the initiating thread.
    void execute_task(ReadyQueue::NodeTask&& task, ReadyQueue& cpu_queue, ReadyQueue* local_queue);

    ReadyQueue* queue_for_device(int device_index);

    void worker_main(ReadyQueue& queue);

    std::mutex queues_mutex_;
    // -1 -> CPU queue; >= 0 -> CUDA device index. Queues are leaked by design
    // (the engine is a process-lifetime singleton; workers may outlive users).
    std::unordered_map<int, ReadyQueue*> ready_queues_;
    std::unordered_map<int, std::thread> device_threads_;
    // Cached pointer to the CPU queue (device index -1).  The queue is
    // process-lifetime and never replaced, so the hot path reads this instead
    // of taking queues_mutex_ on every node/edge completion.
    std::atomic<ReadyQueue*> cpu_queue_cache_{nullptr};

    // Depth of nested execute() calls on this thread. Backed by a
    // file-local thread_local in Engine.cpp: MSVC forbids thread storage
    // duration on members of a dll-interface class.
    static int& nested_depth();
};

} // namespace tpx
} // namespace tensorplay
