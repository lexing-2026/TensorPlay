#pragma once

#include "agent_utils.h"
#include "message.h"
#include "rpc_agent.h"
#include "rref_impl.h"
#include "tensorpipe_utils.h"

#include "metrics/RpcMetricsHandler.h"
#include "profiler/remote_profiler_manager.h"
#include "profiler/server_process_global_profiler.h"

#include <pybind11/pybind11.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tensorplay::distributed {
class Store;
}

namespace tensorplay::distributed::rpc {

namespace py = pybind11;

class RpcRuntime : public RpcAgent {
public:
    RpcRuntime();
    ~RpcRuntime() override;

    RpcRuntime(const RpcRuntime&) = delete;
    RpcRuntime& operator=(const RpcRuntime&) = delete;

    void init(
        const std::string& name,
        worker_id_t rank,
        worker_id_t world_size,
        int num_worker_threads,
        double timeout_seconds,
        std::string init_method = "env://",
        std::shared_ptr<tensorplay::distributed::Store> store = nullptr);
    void configure_backend(
        std::optional<std::vector<std::string>> transports,
        std::optional<std::vector<std::string>> channels);
    bool initialized() const;
    bool started() const;
    WorkerInfo current_worker() const;
    std::vector<WorkerInfo> workers() const;

    RpcFuturePtr submit(
        const std::string& target,
        py::object callable,
        py::tuple args,
        py::dict kwargs,
        double timeout_seconds = -1.0);
    std::shared_ptr<RpcRRef> remote(
        const std::string& target,
        py::object callable,
        py::tuple args,
        py::dict kwargs,
        double timeout_seconds = -1.0);
    std::shared_ptr<RpcRRef> restore_rref(
        const std::string& owner,
        worker_id_t owner_id,
        RRefId rref_id,
        ForkId fork_id);

    std::shared_ptr<tensorplay::distributed::Store> store() const;
    DeviceMap get_device_map(const WorkerInfo& destination) const;

    py::object fetch_rref(const RpcRRef& rref, double timeout_seconds) const;
    void fork_rref(const RpcRRef& rref) const;
    void delete_rref(const RpcRRef& rref) const;

    py::dict all_gather(
        py::object value,
        const std::vector<std::string>& worker_names,
        double timeout_seconds);
    void barrier(
        const std::vector<std::string>& worker_names,
        double timeout_seconds);

    std::unordered_map<std::string, std::string> get_metrics() const override;
    std::unordered_map<std::string, std::string> get_debug_info() const override;
    RpcFuturePtr send(
        const WorkerInfo& to,
        MessagePtr message,
        double timeout_seconds = -1.0,
        const DeviceMap& device_map = {}) override;
    RpcFuturePtr send_with_retries(
        const WorkerInfo& to,
        MessagePtr message,
        RpcRetryOptions options = {}) override;
    const WorkerInfo& get_worker_info(const std::string& name) const override;
    const WorkerInfo& get_worker_info(worker_id_t id) const override;
    std::vector<WorkerInfo> get_worker_infos() const override;
    void start() override;
    void shutdown() override;
    void join(bool shutdown = false, double timeout_seconds = 0.0) override;
    void sync(double timeout_seconds = -1.0) override;
    void set_device_map(
        const std::string& worker,
        std::unordered_map<std::string, std::string> device_map);

    void profiler_start(bool record_call_stack);
    void profiler_stop();
    std::vector<profiler::Event> profiler_events() const;

    struct RpcFrame final {
        uint64_t request_id = 0;
        MessagePtr message;
    };

    struct CollectiveState final {
        std::mutex mutex;
        std::condition_variable condition;
        std::vector<worker_id_t> expected;
        std::unordered_map<worker_id_t, py::object> values;
        py::dict gathered;
        bool ready = false;
    };

private:
    enum class TaskKind : uint8_t {
        CALL,
        REMOTE_CALL,
        MESSAGE,
        INCOMING,
    };

    struct Task final {
        bool valid = false;
        TaskKind kind = TaskKind::CALL;
        py::object callable;
        py::object args;
        py::object kwargs;
        MessagePtr message;
        RpcFuturePtr future;
        std::string target;
        double timeout_seconds = -1.0;
        RpcRetryOptions retry_options;
        RRefId rref_id;
        int64_t autograd_context_id = -1;
        int64_t autograd_message_id = -1;
        std::shared_ptr<tensorpipe::Pipe> pipe;
        uint64_t request_id = 0;
        DeviceMap device_map;
        bool has_device_map = false;
    };

    struct ClientPipe final {
        explicit ClientPipe(std::shared_ptr<tensorpipe::Pipe> value)
            : pipe(std::move(value)) {}

        std::shared_ptr<tensorpipe::Pipe> pipe;
        mutable std::mutex mutex;
        bool in_error = false;
    };

    mutable std::mutex mutex_;
    mutable std::mutex shutdown_mutex_;
    std::condition_variable condition_;
    std::condition_variable idle_condition_;
    std::deque<Task> queue_;
    std::vector<std::thread> worker_threads_;
    size_t active_tasks_ = 0;
    std::vector<WorkerInfo> workers_;
    std::string master_addr_;
    uint16_t master_port_ = 29500;
    worker_id_t current_rank_ = 0;
    worker_id_t world_size_ = 1;
    int num_worker_threads_ = 1;
    bool initialized_ = false;
    bool started_ = false;
    bool stopping_ = false;
    mutable std::mutex client_mutex_;
    mutable std::unordered_map<worker_id_t, std::shared_ptr<ClientPipe>>
        client_pipes_;
    std::vector<std::thread> pipe_threads_;
    mutable std::atomic<uint64_t> next_request_id_{1};
    std::atomic<local_id_t> next_local_id_{1};
    mutable std::mutex collective_mutex_;
    std::unordered_map<std::string, uint64_t> collective_sequences_;
    std::unordered_map<
        std::string,
        std::shared_ptr<CollectiveState>>
        collective_states_;
    std::shared_ptr<tensorplay::distributed::Store> rendezvous_store_;
    std::string rendezvous_prefix_;
    std::string bootstrap_transport_;
    std::unordered_map<worker_id_t, std::string> worker_urls_;
    std::optional<std::vector<std::string>> transports_;
    std::optional<std::vector<std::string>> channels_;
    std::shared_ptr<tensorpipe::Context> tensorpipe_context_;
    std::shared_ptr<tensorpipe::Listener> tensorpipe_listener_;
    mutable std::mutex device_map_mutex_;
    std::unordered_map<std::string, DeviceMap> device_maps_;
    std::unordered_map<std::string, DeviceMap> reverse_device_maps_;
    mutable RRefContext rrefs_;
    mutable metrics::RpcMetricsHandler metrics_;
    mutable std::mutex profiler_mutex_;
    std::shared_ptr<std::atomic<bool>> lifetime_token_ =
        std::make_shared<std::atomic<bool>>(true);

    WorkerInfo resolve_worker(const std::string& name) const;
    void task_started();
    void task_finished();
    Task pop_task();
    void worker_loop();
    void execute_task(Task task);
    void execute_callable(Task& task);
    void execute_message(Task& task);
    void execute_incoming(Task& task);
    void enqueue(Task task);

    void start_listener();
    void initialize_rendezvous(
        const std::string& init_method,
        std::shared_ptr<tensorplay::distributed::Store> store);
    void exchange_worker_urls();
    void accept_pipe(
        const tensorpipe::Error& error,
        std::shared_ptr<tensorpipe::Pipe> pipe);
    void handle_pipe(std::shared_ptr<tensorpipe::Pipe> pipe);

    RpcFrame handle_frame(const RpcFrame& frame);
    RpcFrame handle_call(
        const RpcFrame& frame,
        bool remote_call,
        int64_t autograd_context_id = -1);
    RpcFrame handle_forward_autograd(const RpcFrame& frame);
    RpcFrame handle_backward_autograd(const RpcFrame& frame);
    RpcFrame handle_cleanup_autograd(const RpcFrame& frame);
    RpcFrame handle_rref_backward(const RpcFrame& frame);
    RpcFrame handle_fetch(const RpcFrame& frame);
    RpcFrame handle_fork(const RpcFrame& frame);
    RpcFrame handle_delete(const RpcFrame& frame);
    RpcFrame handle_gather(const RpcFrame& frame);

    MessagePtr send_message(
        const WorkerInfo& to,
        MessagePtr message,
        double timeout_seconds,
        const RpcRetryOptions& retry_options,
        const DeviceMap* device_map = nullptr) const;
    MessagePtr send_task(Task& task, const WorkerInfo& to);
    MessagePtr send_collective(
        const WorkerInfo& to,
        const std::string& collective_id,
        worker_id_t leader_id,
        const std::vector<worker_id_t>& group_ids,
        int phase,
        py::object value,
        double timeout_seconds) const;

    static MessagePtr make_python_message(
        SerializedPyObj object,
        MessageType type,
        int64_t id = -1);
    static SerializedPyObj serialize_result(bool success, py::object value);
    static std::tuple<bool, py::object> deserialize_result(
        const Message& message);
    static py::object make_exception(const std::string& message);
    static uint64_t now_ns();

    DeviceMap device_map_for(const std::string& worker) const;
    DeviceMap reverse_device_map_for(const std::string& worker) const;

    static std::vector<std::string> normalize_worker_names(
        const std::vector<std::string>& names,
        const std::vector<WorkerInfo>& workers);

    friend class RpcRRef;
};

RpcRuntime& global_rpc_runtime();

}  // namespace tensorplay::distributed::rpc
