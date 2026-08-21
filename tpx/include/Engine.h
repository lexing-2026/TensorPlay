#pragma once
#include <vector>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <optional>
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

// Thread-safe ready queue ordered by sequence_nr (max first), mirroring
// torch::autograd::ReadyQueue. One queue per execution device plus one for
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

    void push(NodeTask task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            heap_.push(std::move(task));
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
                auto task = std::move(const_cast<NodeTask&>(heap_.top()));
                heap_.pop();
                return task;
            }
            if (stop()) return std::nullopt;
            // Short poll: the stop predicate turns true as soon as this
            // thread's GraphTask completes on another queue.
            cv_.wait_for(lock, std::chrono::milliseconds(5));
        }
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::priority_queue<NodeTask> heap_;
};

class TENSORPLAY_API Engine {
private:
public:
    static Engine& get_default_engine();

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

    // Depth of nested execute() calls on this thread.
    static thread_local int nested_depth_;
};

} // namespace tpx
} // namespace tensorplay
