#include "rpc_runtime.h"

#include "../autograd/autograd.h"
#include "../autograd/context/container.h"
#include "../autograd/engine/dist_engine.h"
#include "../autograd/rpc_messages/cleanup_autograd_context_req.h"
#include "../autograd/rpc_messages/cleanup_autograd_context_resp.h"
#include "../autograd/rpc_messages/propagate_gradients_req.h"
#include "../autograd/rpc_messages/propagate_gradients_resp.h"
#include "../autograd/rpc_messages/rref_backward_req.h"
#include "../autograd/rpc_messages/rref_backward_resp.h"
#include "../autograd/rpc_messages/rpc_with_autograd.h"
#include "../autograd/utils.h"
#include "python_functions.h"
#include "store/FileStore.h"
#include "store/PrefixStore.h"
#include "store/Store.h"
#include "store/TCPStore.h"
#include "tensorpipe_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace tensorplay::distributed::rpc {
namespace {

std::atomic<uint64_t> rpc_prefix_counter{0};

constexpr int kGatherPhase = 0;
constexpr int kBroadcastPhase = 1;

int64_t current_autograd_context_id() {
    try {
        auto& container =
            tensorplay::distributed::autograd::DistAutogradContainer::instance();
        return container.has_current()
            ? tensorplay::distributed::autograd::DistAutogradContainer::current_context_id()
            : -1;
    } catch (...) {
        return -1;
    }
}

std::chrono::milliseconds effective_duration(
    double requested_seconds,
    std::chrono::milliseconds default_timeout) {
    if (requested_seconds < 0.0) {
        return default_timeout.count() == 0
            ? std::chrono::milliseconds(-1)
            : default_timeout;
    }
    if (requested_seconds == 0.0) {
        return std::chrono::milliseconds(-1);
    }
    return timeout_to_duration(requested_seconds);
}

struct PipeCompletion {
    mutable std::mutex mutex;
    std::condition_variable condition;
    bool completed = false;
    std::string error;

    void complete(const tensorpipe::Error& value) {
        std::lock_guard<std::mutex> lock(mutex);
        if (completed) {
            return;
        }
        if (value) {
            error = value.what();
        }
        completed = true;
        condition.notify_all();
    }

    bool wait(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        if (timeout.count() < 0) {
            condition.wait(lock, [this]() { return completed; });
            return true;
        }
        return condition.wait_for(
            lock, timeout, [this]() { return completed; });
    }
};

struct DescriptorCompletion final : PipeCompletion {
    tensorpipe::Descriptor descriptor;

    void complete(
        const tensorpipe::Error& value,
        tensorpipe::Descriptor descriptor_value) {
        std::lock_guard<std::mutex> lock(mutex);
        if (completed) {
            return;
        }
        if (value) {
            error = value.what();
        }
        descriptor = std::move(descriptor_value);
        completed = true;
        condition.notify_all();
    }
};

std::chrono::milliseconds store_timeout(
    std::chrono::milliseconds rpc_timeout) {
    if (rpc_timeout.count() <= 0) {
        return tensorplay::distributed::Store::kDefaultTimeout;
    }
    return rpc_timeout;
}

std::shared_ptr<tensorplay::distributed::Store> create_rendezvous_store(
    const std::string& init_method,
    const std::string& master_address,
    uint16_t master_port,
    worker_id_t rank,
    std::chrono::milliseconds timeout) {
    if (init_method.rfind("file://", 0) == 0) {
        const std::string path = init_method.substr(7);
        if (path.empty()) {
            throw std::invalid_argument("file rendezvous path is empty");
        }
        return std::make_shared<tensorplay::distributed::FileStore>(
            path, timeout);
    }
    if (init_method.rfind("env://", 0) != 0 &&
        init_method.rfind("tcp://", 0) != 0) {
        throw std::invalid_argument(
            "RPC initialization method must use env://, tcp://, or file://");
    }
    if (master_address.empty() || master_port == 0) {
        throw std::invalid_argument(
            "RPC rendezvous requires a master address and port");
    }
    return std::make_shared<tensorplay::distributed::TCPStore>(
        master_address,
        master_port,
        rank == 0,
        timeout);
}

void wait_for_pipe_operation(
    const std::shared_ptr<PipeCompletion>& completion,
    std::chrono::milliseconds timeout) {
    if (!completion->wait(timeout)) {
        throw std::runtime_error("TensorPipe operation timed out");
    }
    std::lock_guard<std::mutex> lock(completion->mutex);
    if (!completion->error.empty()) {
        throw std::runtime_error(completion->error);
    }
}

MessagePtr read_tensorpipe_message(
    const std::shared_ptr<tensorpipe::Pipe>& pipe,
    std::chrono::milliseconds timeout) {
    auto descriptor_completion = std::make_shared<DescriptorCompletion>();
    pipe->readDescriptor(
        [descriptor_completion](
            const tensorpipe::Error& error,
            tensorpipe::Descriptor descriptor) {
            descriptor_completion->complete(error, std::move(descriptor));
        });
    wait_for_pipe_operation(descriptor_completion, timeout);
    TensorPipeReadAllocation allocation = allocate_tensorpipe_message(
        descriptor_completion->descriptor);
    auto read_completion = std::make_shared<PipeCompletion>();
    auto state = allocation.state;
    pipe->read(
        std::move(allocation.allocation),
        [read_completion, state](const tensorpipe::Error& error) {
            read_completion->complete(error);
        });
    wait_for_pipe_operation(read_completion, timeout);
    py::gil_scoped_acquire gil;
    return decode_tensorpipe_message(
        descriptor_completion->descriptor, *state);
}

void write_tensorpipe_message(
    const std::shared_ptr<tensorpipe::Pipe>& pipe,
    std::shared_ptr<TensorPipeWriteState> state,
    std::chrono::milliseconds timeout) {
    auto completion = std::make_shared<PipeCompletion>();
    pipe->write(
        std::move(state->message),
        [completion, state](const tensorpipe::Error& error) {
            completion->complete(error);
        });
    wait_for_pipe_operation(completion, timeout);
}

RpcRuntime::RpcFrame response_frame(
    uint64_t request_id,
    MessageType type,
    SerializedPyObj object) {
    RpcRuntime::RpcFrame response;
    response.request_id = request_id;
    response.message = std::make_shared<Message>(
        std::vector<uint8_t>(object.payload_.begin(), object.payload_.end()),
        std::move(object.tensors_),
        type,
        static_cast<int64_t>(request_id));
    return response;
}

SerializedPyObj serialized_message(const RpcRuntime::RpcFrame& frame) {
    return SerializedPyObj(
        std::string(
            frame.message->payload().begin(), frame.message->payload().end()),
        std::vector<py::object>(
            frame.message->tensors().begin(), frame.message->tensors().end()));
}

}  // namespace

RpcRuntime::RpcRuntime() : RpcAgent(WorkerInfo{"", 0}, 60.0) {}

RpcRuntime::~RpcRuntime() {
    if (initialized()) {
        shutdown();
    }
}

void RpcRuntime::configure_backend(
    std::optional<std::vector<std::string>> transports,
    std::optional<std::vector<std::string>> channels) {
    if (transports.has_value()) {
        for (const auto& name : *transports) {
            if (!TensorPipeTransportRegistry::instance().has(name)) {
                throw std::invalid_argument(
                    "unknown RPC transport: " + name);
            }
        }
    }
    if (channels.has_value()) {
        for (const auto& name : *channels) {
            if (!TensorPipeChannelRegistry::instance().has(name)) {
                throw std::invalid_argument(
                    "unknown RPC channel: " + name);
            }
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
        throw std::runtime_error(
            "RPC backend options cannot change after initialization");
    }
    transports_ = std::move(transports);
    channels_ = std::move(channels);
}

MessagePtr RpcRuntime::make_python_message(
    SerializedPyObj object,
    MessageType type,
    int64_t id) {
    return std::make_shared<Message>(
        std::vector<uint8_t>(object.payload_.begin(), object.payload_.end()),
        std::move(object.tensors_),
        type,
        id);
}

SerializedPyObj RpcRuntime::serialize_result(bool success, py::object value) {
    return serialize_python_object(py::make_tuple(success, std::move(value)));
}

std::tuple<bool, py::object> RpcRuntime::deserialize_result(
    const Message& message) {
    py::object value = deserialize_python_object(SerializedPyObj(
        std::string(message.payload().begin(), message.payload().end()),
        std::vector<py::object>(
            message.tensors().begin(), message.tensors().end())));
    py::tuple result = value.cast<py::tuple>();
    if (result.size() != 2) {
        throw std::runtime_error("RPC response must contain status and value");
    }
    return {result[0].cast<bool>(), result[1]};
}

py::object RpcRuntime::make_exception(const std::string& message) {
    return py::reinterpret_steal<py::object>(
        PyObject_CallOneArg(PyExc_RuntimeError, py::str(message).ptr()));
}

uint64_t RpcRuntime::now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

void RpcRuntime::init(
    const std::string& name,
    worker_id_t rank,
    worker_id_t world_size,
    int num_worker_threads,
    double timeout_seconds,
    std::string init_method,
    std::shared_ptr<tensorplay::distributed::Store> store) {
    if (name.empty() || rank < 0 || world_size <= 0 || rank >= world_size) {
        throw std::invalid_argument("invalid RPC worker configuration");
    }
    validate_worker_name(name);
    if (num_worker_threads <= 0) {
        throw std::invalid_argument("number of RPC worker threads must be positive");
    }
    if (timeout_seconds < 0.0) {
        throw std::invalid_argument("RPC timeout must be non-negative");
    }
    std::vector<WorkerInfo> configured_workers{{name, rank}};
    std::string address = environment_value("MASTER_ADDR");
    std::string port_string = environment_value("MASTER_PORT");
    if (init_method.rfind("tcp://", 0) == 0) {
        const Endpoint endpoint = parse_endpoint(init_method.substr(6), 0);
        if (endpoint.port == 0) {
            throw std::invalid_argument("TCP initialization endpoint needs a port");
        }
        address = endpoint.host;
        port_string = std::to_string(endpoint.port);
    }
    if (address.empty() && world_size == 1) {
        address = "127.0.0.1";
    }
    if (init_method.rfind("env://", 0) == 0 && world_size > 1 &&
        (address.empty() || port_string.empty())) {
        throw std::invalid_argument(
            "env rendezvous requires MASTER_ADDR and MASTER_PORT");
    }
    if (address.empty()) {
        address = "127.0.0.1";
    }
    uint16_t base_port = 29500;
    if (!port_string.empty()) {
        const auto parsed = std::stoll(port_string);
        if (parsed <= 0 || parsed > 65535) {
            throw std::invalid_argument("RPC master port is out of range");
        }
        base_port = static_cast<uint16_t>(parsed);
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) {
            throw std::runtime_error("RPC runtime is already initialized");
        }
        workers_ = std::move(configured_workers);
        worker_info_ = workers_[static_cast<size_t>(rank)];
        current_rank_ = rank;
        world_size_ = world_size;
        num_worker_threads_ = num_worker_threads;
        master_addr_ = std::move(address);
        master_port_ = base_port;
        set_rpc_timeout(timeout_to_duration(timeout_seconds));
        stopping_ = false;
        started_ = false;
        initialized_ = true;
    }
    tensorplay::distributed::autograd::DistAutogradContainer::init(
        rank, this);
    RpcAgent::set_current_rpc_agent(this);
    try {
        initialize_rendezvous(init_method, std::move(store));
    } catch (...) {
        shutdown();
        throw;
    }
}

bool RpcRuntime::initialized() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return initialized_;
}

bool RpcRuntime::started() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return started_;
}

void RpcRuntime::initialize_rendezvous(
    const std::string& init_method,
    std::shared_ptr<tensorplay::distributed::Store> store) {
    const uint64_t prefix_id = rpc_prefix_counter.fetch_add(1);
    rendezvous_prefix_ = "rpc_prefix_" + std::to_string(prefix_id);
    const auto timeout = store_timeout(rpc_timeout());
    if (!store) {
        if (world_size_ <= 1) {
            return;
        }
        store = create_rendezvous_store(
            init_method,
            master_addr_,
            master_port_,
            current_rank_,
            timeout);
    }
    store->setTimeout(timeout);
    rendezvous_store_ = std::make_shared<tensorplay::distributed::PrefixStore>(
        rendezvous_prefix_, std::move(store), timeout);

    if (world_size_ <= 1) {
        return;
    }

    auto workers = collect_worker_infos(
        rendezvous_store_,
        worker_info_.name,
        current_rank_,
        world_size_,
        timeout);
    std::lock_guard<std::mutex> lock(mutex_);
    workers_ = std::move(workers);
    worker_info_ = workers_[static_cast<size_t>(current_rank_)];
}

void RpcRuntime::exchange_worker_urls() {
    std::shared_ptr<tensorpipe::Listener> listener;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        listener = tensorpipe_listener_;
    }
    if (!listener || !rendezvous_store_ || bootstrap_transport_.empty()) {
        throw std::runtime_error("RPC listener rendezvous is not initialized");
    }
    const std::string own_url = listener->url(bootstrap_transport_);
    if (own_url.empty()) {
        throw std::runtime_error("RPC listener did not materialize an address");
    }
    const std::string own_key = "worker/" + std::to_string(current_rank_);
    rendezvous_store_->set(
        own_key,
        std::vector<uint8_t>(own_url.begin(), own_url.end()));

    std::vector<std::string> keys;
    keys.reserve(workers_.size());
    for (const auto& worker : workers_) {
        keys.push_back(
            "worker/" + std::to_string(worker.id));
    }
    if (!rendezvous_store_->wait(keys, store_timeout(rpc_timeout()))) {
        throw std::runtime_error("RPC worker address exchange timed out");
    }

    std::unordered_map<worker_id_t, std::string> urls;
    for (const auto& worker : workers_) {
        const auto value = rendezvous_store_->get(
            "worker/" + std::to_string(worker.id));
        std::string url(value.begin(), value.end());
        if (url.empty()) {
            throw std::runtime_error("RPC worker address is empty");
        }
        urls.emplace(worker.id, std::move(url));
    }
    std::lock_guard<std::mutex> lock(mutex_);
    worker_urls_ = std::move(urls);
}

WorkerInfo RpcRuntime::current_worker() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        throw std::runtime_error("RPC runtime is not initialized");
    }
    return worker_info_;
}

std::vector<WorkerInfo> RpcRuntime::workers() const {
    return get_worker_infos();
}

std::shared_ptr<tensorplay::distributed::Store> RpcRuntime::store() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return rendezvous_store_;
}

void RpcRuntime::task_started() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++active_tasks_;
}

void RpcRuntime::task_finished() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_tasks_ > 0) {
            --active_tasks_;
        }
    }
    idle_condition_.notify_all();
}

RpcRuntime::Task RpcRuntime::pop_task() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this]() { return stopping_ || !queue_.empty(); });
    if (queue_.empty()) {
        Task task;
        return task;
    }
    Task task = std::move(queue_.front());
    queue_.pop_front();
    return task;
}

void RpcRuntime::worker_loop() {
    for (;;) {
        Task task = pop_task();
        if (!task.valid) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_ && queue_.empty()) {
                return;
            }
            continue;
        }
        execute_task(std::move(task));
    }
}

void RpcRuntime::enqueue(Task task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_ || stopping_) {
            throw std::runtime_error("RPC runtime is not accepting work");
        }
        const auto worker = std::find_if(
            workers_.begin(),
            workers_.end(),
            [&task](const WorkerInfo& value) {
                return value.name == task.target;
            });
        if (worker == workers_.end()) {
            throw std::invalid_argument(
                "worker is not registered: " + task.target);
        }
        queue_.push_back(std::move(task));
    }
    condition_.notify_one();
}

RpcFuturePtr RpcRuntime::submit(
    const std::string& target,
    py::object callable,
    py::tuple args,
    py::dict kwargs,
    double timeout_seconds) {
    const WorkerInfo worker = resolve_worker(target);
    auto future = std::make_shared<RpcFuture>();
    Task task;
    task.valid = true;
    task.kind = TaskKind::CALL;
    task.callable = std::move(callable);
    task.args = std::move(args);
    task.kwargs = std::move(kwargs);
    task.future = future;
    task.target = worker.name;
    task.timeout_seconds = timeout_seconds;
    task.autograd_context_id = current_autograd_context_id();
    enqueue(std::move(task));
    return future;
}

std::shared_ptr<RpcRRef> RpcRuntime::remote(
    const std::string& target,
    py::object callable,
    py::tuple args,
    py::dict kwargs,
    double timeout_seconds) {
    const WorkerInfo worker = resolve_worker(target);
    const RRefId id(current_rank_, next_local_id_.fetch_add(1));
    const ForkId fork(current_rank_, next_local_id_.fetch_add(1));
    auto creation = std::make_shared<RpcFuture>();
    std::shared_ptr<RRefState> local_state;
    if (worker.id == current_rank_) {
        local_state = rrefs_.create(id);
    }
    Task task;
    task.valid = true;
    task.kind = TaskKind::REMOTE_CALL;
    task.callable = std::move(callable);
    task.args = std::move(args);
    task.kwargs = std::move(kwargs);
    task.future = worker.id == current_rank_ ? nullptr : creation;
    task.target = worker.name;
    task.timeout_seconds = timeout_seconds;
    task.rref_id = id;
    task.autograd_context_id = current_autograd_context_id();
    try {
        enqueue(std::move(task));
        if (worker.id == current_rank_) {
            creation->set_result(py::none());
        }
    } catch (...) {
        if (local_state) {
            rrefs_.release(id);
        }
        throw;
    }
    return std::make_shared<RpcRRef>(
        this, worker, id, fork, std::move(creation), std::move(local_state));
}

std::shared_ptr<RpcRRef> RpcRuntime::restore_rref(
    const std::string& owner,
    worker_id_t owner_id,
    RRefId rref_id,
    ForkId fork_id) {
    const WorkerInfo worker = resolve_worker(owner);
    if (worker.id != owner_id) {
        throw std::invalid_argument("RRef owner name and id do not match");
    }
    auto creation = std::make_shared<RpcFuture>();
    creation->set_result(py::none());
    std::shared_ptr<RRefState> local_state;
    if (worker.id == current_rank_) {
        local_state = rrefs_.find(rref_id);
        if (!local_state) {
            throw std::runtime_error("local RRef owner entry does not exist");
        }
        rrefs_.retain(rref_id);
    }
    auto result = std::make_shared<RpcRRef>(
        this,
        worker,
        rref_id,
        ForkId(current_rank_, next_local_id_.fetch_add(1)),
        std::move(creation),
        std::move(local_state));
    if (worker.id != current_rank_) {
        fork_rref(*result);
    }
    return result;
}

void RpcRuntime::execute_callable(Task& task) {
    const uint64_t started = now_ns();
    bool failed = false;
    std::optional<tensorplay::distributed::autograd::ContextGuard>
        autograd_context_guard;
    if (task.autograd_context_id >= 0) {
        autograd_context_guard.emplace(task.autograd_context_id);
    }
    py::gil_scoped_acquire gil;
    try {
        PyObject* raw = PyObject_Call(
            task.callable.ptr(), task.args.ptr(), task.kwargs.ptr());
        if (raw == nullptr) {
            throw py::error_already_set();
        }
        py::object result = py::reinterpret_steal<py::object>(raw);
        if (PyObject_HasAttrString(task.callable.ptr(), "_wrapped_async_rpc_function") &&
            PyObject_HasAttrString(result.ptr(), "wait")) {
            result = result.attr("wait")();
        }
        if (task.kind == TaskKind::REMOTE_CALL) {
            rrefs_.set_value(task.rref_id, result);
            if (task.future) {
                task.future->set_result(py::none());
            }
        } else if (task.future) {
            task.future->set_result(std::move(result));
        }
    } catch (py::error_already_set& error) {
        failed = true;
        py::object exception = py::reinterpret_borrow<py::object>(error.value());
        if (task.kind == TaskKind::REMOTE_CALL) {
            rrefs_.set_exception(task.rref_id, exception);
            if (task.future) {
                task.future->set_exception(std::move(exception));
            }
        } else if (task.future) {
            task.future->set_exception(std::move(exception));
        }
        error.restore();
        PyErr_Clear();
    } catch (const std::exception& error) {
        failed = true;
        py::object exception = make_exception(error.what());
        if (task.kind == TaskKind::REMOTE_CALL) {
            rrefs_.set_exception(task.rref_id, exception);
            if (task.future) {
                task.future->set_exception(std::move(exception));
            }
        } else if (task.future) {
            task.future->set_exception(std::move(exception));
        }
    }
    const uint64_t finished = now_ns();
    metrics_.record_call(0, 0);
    profiler::record_server_event(
        task.kind == TaskKind::REMOTE_CALL ? "remote" : "call",
        worker_info_.name,
        task.target,
        started,
        finished,
        failed);
    task.callable = py::none();
    task.args = py::tuple();
    task.kwargs = py::dict();
}

void RpcRuntime::execute_message(Task& task) {
    try {
        const WorkerInfo worker = resolve_worker(task.target);
        auto complete = [this, &task](const MessagePtr& response) {
            if (!task.future) {
                return;
            }
            py::gil_scoped_acquire gil;
            if (response->type() == MessageType::EXCEPTION) {
                try {
                    auto [success, value] = deserialize_result(*response);
                    if (!success) {
                        task.future->set_exception(
                            make_exception(value.cast<std::string>()));
                    } else {
                        task.future->set_exception(
                            make_exception("RPC request failed"));
                    }
                } catch (const std::exception& error) {
                    task.future->set_exception(make_exception(error.what()));
                }
                return;
            }
            task.future->set_result(py::bytes(
                reinterpret_cast<const char*>(response->payload().data()),
                response->payload().size()));
        };
        if (worker.id == current_rank_) {
            if (task.message->type() == MessageType::BACKWARD_AUTOGRAD_REQ ||
                task.message->type() == MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ ||
                task.message->type() == MessageType::RREF_BACKWARD_REQ) {
                py::gil_scoped_acquire gil;
                const auto frame = RpcFrame{
                    static_cast<uint64_t>(task.message->id()), task.message};
                const auto response = handle_frame(frame);
                complete(response.message);
            } else {
                complete(task.message);
            }
            return;
        }
        MessagePtr response = send_message(
            worker,
            task.message,
            task.timeout_seconds,
            task.retry_options,
            task.has_device_map ? &task.device_map : nullptr);
        complete(response);
    } catch (py::error_already_set& error) {
        if (task.future) {
            py::gil_scoped_acquire gil;
            py::object exception = py::reinterpret_borrow<py::object>(error.value());
            task.future->set_exception(std::move(exception));
        }
        error.restore();
        PyErr_Clear();
    } catch (const std::exception& error) {
        if (task.future) {
            py::gil_scoped_acquire gil;
            task.future->set_exception(make_exception(error.what()));
        }
    }
}

void RpcRuntime::execute_incoming(Task& task) {
    if (!task.pipe || !task.message) {
        return;
    }
    const RpcFrame request_frame{task.request_id, task.message};
    RpcFrame response;
    try {
        py::gil_scoped_acquire gil;
        response = handle_frame(request_frame);
    } catch (py::error_already_set& error) {
        py::gil_scoped_acquire gil;
        const std::string message = error.what();
        error.restore();
        PyErr_Clear();
        response = response_frame(
            request_frame.request_id,
            MessageType::EXCEPTION,
            serialize_result(false, py::str(message)));
    } catch (const std::exception& error) {
        py::gil_scoped_acquire gil;
        response = response_frame(
            request_frame.request_id,
            MessageType::EXCEPTION,
            serialize_result(false, py::str(error.what())));
    }

    try {
        std::shared_ptr<TensorPipeWriteState> write_state;
        {
            py::gil_scoped_acquire gil;
            write_state = make_tensorpipe_message(
                *response.message,
                reverse_device_map_for(task.pipe->getRemoteName()));
        }
        write_tensorpipe_message(
            task.pipe,
            std::move(write_state),
            effective_duration(-1.0, rpc_timeout()));
    } catch (...) {
    }
}

void RpcRuntime::execute_task(Task task) {
    task_started();
    struct TaskGuard final {
        RpcRuntime* runtime;
        ~TaskGuard() {
            runtime->task_finished();
        }
    } guard{this};
    if (task.kind == TaskKind::INCOMING) {
        execute_incoming(task);
        return;
    }
    try {
        const WorkerInfo worker = resolve_worker(task.target);
        if (task.kind == TaskKind::MESSAGE) {
            execute_message(task);
        } else if (worker.id == current_rank_) {
            execute_callable(task);
        } else {
            MessagePtr response = send_task(task, worker);
            py::gil_scoped_acquire gil;
            auto [success, value] = deserialize_result(*response);
            if (!success) {
                py::object exception = make_exception(value.cast<std::string>());
                if (task.kind == TaskKind::REMOTE_CALL) {
                    if (task.future) {
                        task.future->set_exception(std::move(exception));
                    }
                } else if (task.future) {
                    task.future->set_exception(std::move(exception));
                }
            } else if (task.future) {
                task.future->set_result(
                    task.kind == TaskKind::REMOTE_CALL ? py::none() : std::move(value));
            }
        }
    } catch (py::error_already_set& error) {
        if (task.future) {
            py::gil_scoped_acquire gil;
            py::object exception = py::reinterpret_borrow<py::object>(error.value());
            task.future->set_exception(std::move(exception));
        }
        error.restore();
        PyErr_Clear();
    } catch (const std::exception& error) {
        if (task.future) {
            py::gil_scoped_acquire gil;
            task.future->set_exception(make_exception(error.what()));
        }
    }
    {
        py::gil_scoped_acquire gil;
        task.callable = py::none();
        task.args = py::tuple();
        task.kwargs = py::dict();
    }
    if (task.message) {
        task.message.reset();
    }
}

WorkerInfo RpcRuntime::resolve_worker(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& worker : workers_) {
        if (worker.name == name) {
            return worker;
        }
    }
    throw std::invalid_argument("worker is not registered: " + name);
}

MessagePtr RpcRuntime::send_task(Task& task, const WorkerInfo& to) {
    py::gil_scoped_acquire gil;
    py::object call;
    if (task.kind == TaskKind::REMOTE_CALL) {
        call = py::make_tuple(task.rref_id.to_python(), task.callable, task.args, task.kwargs);
    } else {
        call = py::make_tuple(task.callable, task.args, task.kwargs);
    }
    SerializedPyObj object = serialize_python_object(call);
    auto message = make_python_message(
        std::move(object),
        task.kind == TaskKind::REMOTE_CALL
            ? MessageType::PYTHON_REMOTE_CALL
            : MessageType::PYTHON_CALL);
    bool has_grad_tensor = false;
    for (const auto& value : message->tensors()) {
        if (value.cast<tensorplay::Tensor>().requires_grad()) {
            has_grad_tensor = true;
            break;
        }
    }

    tensorplay::distributed::autograd::AutogradMetadata metadata;
    if (task.autograd_context_id >= 0 && has_grad_tensor) {
        auto& container =
            tensorplay::distributed::autograd::DistAutogradContainer::instance();
        const auto context = container.retrieve(task.autograd_context_id);
        metadata = {
            task.autograd_context_id, container.new_message_id()};
        task.autograd_message_id = metadata.message_id;
        tensorplay::distributed::autograd::add_send_rpc_backward(
            context, metadata, message->tensors());
        context->add_known_worker(to.id);
        message = std::move(
            tensorplay::distributed::autograd::RpcWithAutograd(
                current_rank_,
                MessageType::FORWARD_AUTOGRAD_REQ,
                metadata,
                std::move(message),
                device_map_for(to.name)))
                      .to_message();
    }

    MessagePtr response;
    {
        py::gil_scoped_release release;
        response = send_message(
            to,
            std::move(message),
            task.timeout_seconds,
            task.retry_options);
    }
    if (metadata.valid() &&
        response->type() == MessageType::FORWARD_AUTOGRAD_RESP) {
        auto wrapped_response =
            tensorplay::distributed::autograd::RpcWithAutograd::from_message(
                *response);
        if (wrapped_response.metadata().context_id != metadata.context_id) {
            throw std::runtime_error(
                "distributed autograd response metadata does not match request");
        }
        const auto& inner = wrapped_response.wrapped_message();
        if (!inner) {
            throw std::runtime_error(
                "distributed autograd response has no wrapped message");
        }
        tensorplay::distributed::autograd::add_recv_rpc_backward(
            wrapped_response.metadata(),
            inner->tensors(),
            to.id,
            reverse_device_map_for(to.name));
        return inner;
    }
    return response;
}

MessagePtr RpcRuntime::send_message(
    const WorkerInfo& to,
    MessagePtr message,
    double timeout_seconds,
    const RpcRetryOptions& retry_options,
    const DeviceMap* device_map) const {
    if (!message) {
        throw std::invalid_argument("RPC message cannot be null");
    }
    const auto timeout = effective_duration(timeout_seconds, rpc_timeout());
    const int max_retries = std::max(0, retry_options.max_retries);
    std::string last_error = "RPC worker did not accept the request";
    for (int attempt = 0; attempt <= max_retries; ++attempt) {
        const uint64_t request_id = next_request_id_.fetch_add(1);
        message->set_id(static_cast<int64_t>(request_id));
        std::shared_ptr<ClientPipe> client_pipe;
        try {
            std::shared_ptr<tensorpipe::Context> context;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                context = tensorpipe_context_;
            }
            if (!context) {
                throw std::runtime_error("RPC TensorPipe context is not initialized");
            }
            std::shared_ptr<TensorPipeWriteState> write_state;
            {
                py::gil_scoped_acquire gil;
                write_state = make_tensorpipe_message(
                    *message,
                    device_map != nullptr ? *device_map : device_map_for(to.name));
            }
            std::string url;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                const auto iterator = worker_urls_.find(to.id);
                if (iterator == worker_urls_.end()) {
                    throw std::runtime_error(
                        "RPC destination has no rendezvous address");
                }
                url = iterator->second;
            }
            {
                std::lock_guard<std::mutex> lock(client_mutex_);
                auto iterator = client_pipes_.find(to.id);
                if (iterator == client_pipes_.end()) {
                    auto pipe = context->connect(
                        url,
                        tensorpipe::PipeOptions().remoteName(to.name));
                    if (!pipe) {
                        throw std::runtime_error(
                            "RPC TensorPipe connection failed");
                    }
                    client_pipe = std::make_shared<ClientPipe>(std::move(pipe));
                    client_pipes_.emplace(to.id, client_pipe);
                } else {
                    client_pipe = iterator->second;
                }
            }
            std::unique_lock<std::mutex> pipe_lock(client_pipe->mutex);
            if (client_pipe->in_error || !client_pipe->pipe) {
                throw std::runtime_error("RPC TensorPipe connection is closed");
            }
            write_tensorpipe_message(client_pipe->pipe, write_state, timeout);
            MessagePtr response = read_tensorpipe_message(
                client_pipe->pipe, timeout);
            if (response && response->id() == static_cast<int64_t>(request_id)) {
                metrics_.record_call(
                    static_cast<uint64_t>(message->payload().size()),
                    static_cast<uint64_t>(response->payload().size()));
                return response;
            }
            last_error = "RPC response was invalid or timed out";
            client_pipe->in_error = true;
            client_pipe->pipe->close();
            pipe_lock.unlock();
            std::lock_guard<std::mutex> lock(client_mutex_);
            const auto iterator = client_pipes_.find(to.id);
            if (iterator != client_pipes_.end() &&
                iterator->second == client_pipe) {
                client_pipes_.erase(iterator);
            }
        } catch (const std::exception& error) {
            last_error = error.what();
            if (client_pipe) {
                std::unique_lock<std::mutex> pipe_lock(client_pipe->mutex);
                client_pipe->in_error = true;
                if (client_pipe->pipe) {
                    client_pipe->pipe->close();
                }
                pipe_lock.unlock();
                std::lock_guard<std::mutex> lock(client_mutex_);
                const auto iterator = client_pipes_.find(to.id);
                if (iterator != client_pipes_.end() &&
                    iterator->second == client_pipe) {
                    client_pipes_.erase(iterator);
                }
            }
        }
        metrics_.record_error();
        if (attempt < max_retries) {
            const double factor = std::pow(retry_options.retry_backoff, attempt);
            const auto delay = std::chrono::milliseconds(static_cast<int64_t>(
                std::max(0.0, static_cast<double>(retry_options.retry_duration_ms) * factor)));
            std::this_thread::sleep_for(delay);
        }
    }
    throw std::runtime_error(last_error);
}

void RpcRuntime::start_listener() {
    if (world_size_ <= 1) {
        return;
    }
    auto context = std::make_shared<tensorpipe::Context>(
        tensorpipe::ContextOptions().name(worker_info_.name));
    std::vector<std::string> listen_urls;
    int64_t lowest_priority = std::numeric_limits<int64_t>::max();
    std::string lowest_priority_transport;
    for (const auto& key : TensorPipeTransportRegistry::instance().keys()) {
        int64_t priority = -1;
        if (transports_.has_value()) {
            const auto iterator =
                std::find(transports_->begin(), transports_->end(), key);
            if (iterator == transports_->end()) {
                continue;
            }
            priority = static_cast<int64_t>(transports_->size() - 1 -
                                            (iterator - transports_->begin()));
        }
        auto registration =
            TensorPipeTransportRegistry::instance().create(key);
        if (!registration || !registration->transport ||
            !registration->transport->isViable()) {
            continue;
        }
        if (priority == -1) {
            priority = registration->priority;
        }
        if (priority < lowest_priority) {
            lowest_priority = priority;
            lowest_priority_transport = key;
        }
        listen_urls.emplace_back(key + "://" + registration->address);
        context->registerTransport(
            priority, key, std::move(registration->transport));
    }
    if (lowest_priority_transport.empty()) {
        throw std::runtime_error("no viable RPC transport is available");
    }
    bootstrap_transport_ = lowest_priority_transport;

    for (const auto& key : TensorPipeChannelRegistry::instance().keys()) {
        int64_t priority = -1;
        if (channels_.has_value()) {
            const auto iterator =
                std::find(channels_->begin(), channels_->end(), key);
            if (iterator == channels_->end()) {
                continue;
            }
            priority = static_cast<int64_t>(channels_->size() - 1 -
                                            (iterator - channels_->begin()));
        }
        auto registration = TensorPipeChannelRegistry::instance().create(key);
        if (!registration || !registration->channel ||
            !registration->channel->isViable()) {
            continue;
        }
        if (priority == -1) {
            priority = registration->priority;
        }
        context->registerChannel(
            priority, key, std::move(registration->channel));
    }
    auto listener = context->listen(listen_urls);
    if (!listener) {
        throw std::runtime_error("RPC TensorPipe listener creation failed");
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tensorpipe_context_ = context;
        tensorpipe_listener_ = listener;
    }
    listener->accept([this](
                         const tensorpipe::Error& error,
                         std::shared_ptr<tensorpipe::Pipe> pipe) {
        accept_pipe(error, std::move(pipe));
    });
    exchange_worker_urls();
}

void RpcRuntime::accept_pipe(
    const tensorpipe::Error& error,
    std::shared_ptr<tensorpipe::Pipe> pipe) {
    std::shared_ptr<tensorpipe::Listener> listener;
    bool stopping = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping = stopping_;
        listener = tensorpipe_listener_;
    }
    if (!stopping && listener && !error) {
        listener->accept([this](
                             const tensorpipe::Error& next_error,
                             std::shared_ptr<tensorpipe::Pipe> next_pipe) {
            accept_pipe(next_error, std::move(next_pipe));
        });
    }
    if (error || !pipe || stopping) {
        if (pipe) {
            pipe->close();
        }
        return;
    }
    std::lock_guard<std::mutex> lock(client_mutex_);
    pipe_threads_.emplace_back(
        [this, pipe = std::move(pipe)]() mutable { handle_pipe(std::move(pipe)); });
}

void RpcRuntime::handle_pipe(std::shared_ptr<tensorpipe::Pipe> pipe) {
    for (;;) {
        MessagePtr request;
        try {
            request = read_tensorpipe_message(pipe, std::chrono::milliseconds(-1));
        } catch (...) {
            break;
        }
        if (!request || request->id() < 0) {
            break;
        }

        Task task;
        task.valid = true;
        task.kind = TaskKind::INCOMING;
        task.message = std::move(request);
        task.pipe = pipe;
        task.request_id = static_cast<uint64_t>(task.message->id());
        task.target = worker_info_.name;
        try {
            enqueue(std::move(task));
        } catch (...) {
            break;
        }
    }
    pipe->close();
}

RpcRuntime::RpcFrame RpcRuntime::handle_frame(const RpcFrame& frame) {
    switch (frame.message->type()) {
        case MessageType::PYTHON_CALL:
            return handle_call(frame, false);
        case MessageType::PYTHON_REMOTE_CALL:
            return handle_call(frame, true);
        case MessageType::PYTHON_RREF_FETCH_CALL:
            return handle_fetch(frame);
        case MessageType::RREF_FORK_REQUEST:
            return handle_fork(frame);
        case MessageType::RREF_USER_DELETE:
            return handle_delete(frame);
        case MessageType::PYTHON_GATHER_CALL:
            return handle_gather(frame);
        case MessageType::FORWARD_AUTOGRAD_REQ:
            return handle_forward_autograd(frame);
        case MessageType::BACKWARD_AUTOGRAD_REQ:
            return handle_backward_autograd(frame);
        case MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ:
            return handle_cleanup_autograd(frame);
        case MessageType::RREF_BACKWARD_REQ:
            return handle_rref_backward(frame);
        default:
            throw std::runtime_error("unexpected RPC message type");
    }
}

RpcRuntime::RpcFrame RpcRuntime::handle_call(
    const RpcFrame& frame,
    bool remote_call,
    int64_t autograd_context_id) {
    py::gil_scoped_acquire gil;
    try {
        SerializedPyObj serialized = serialized_message(frame);
        py::object value = deserialize_python_object(serialized);
        py::tuple call = value.cast<py::tuple>();
        py::object callable;
        py::tuple args;
        py::dict kwargs;
        RRefId rref_id;
        if (remote_call) {
            if (call.size() != 4) {
                throw std::runtime_error("remote call payload must contain four values");
            }
            rref_id = GloballyUniqueId::from_python(call[0]);
            callable = call[1];
            args = call[2].cast<py::tuple>();
            kwargs = call[3].cast<py::dict>();
        } else {
            if (call.size() != 3) {
                throw std::runtime_error("call payload must contain three values");
            }
            callable = call[0];
            args = call[1].cast<py::tuple>();
            kwargs = call[2].cast<py::dict>();
        }
        auto future = std::make_shared<RpcFuture>();
        if (remote_call) {
            rrefs_.create(rref_id);
        }
        Task task;
        task.valid = true;
        task.kind = remote_call ? TaskKind::REMOTE_CALL : TaskKind::CALL;
        task.callable = std::move(callable);
        task.args = std::move(args);
        task.kwargs = std::move(kwargs);
        task.future = remote_call ? nullptr : future;
        task.target = worker_info_.name;
        task.timeout_seconds = -1.0;
        task.rref_id = rref_id;
        task.autograd_context_id = autograd_context_id;
        // Incoming requests already run on an RPC worker.  Running the
        // decoded call directly keeps a one-thread runtime reentrant and
        // avoids waiting for a queue slot held by this same request.
        execute_callable(task);
        if (remote_call) {
            return response_frame(
                frame.request_id,
                MessageType::REMOTE_RET,
                serialize_result(true, py::none()));
        }
        const auto timeout = effective_duration(-1.0, rpc_timeout());
        py::object result = future->wait(
            timeout.count() < 0 ? -1.0 : static_cast<double>(timeout.count()) / 1000.0);
        return response_frame(
            frame.request_id,
            MessageType::PYTHON_RET,
            serialize_result(true, std::move(result)));
    } catch (py::error_already_set& error) {
        const std::string message = error.what();
        error.restore();
        PyErr_Clear();
        return response_frame(
            frame.request_id,
            remote_call ? MessageType::REMOTE_RET : MessageType::PYTHON_RET,
            serialize_result(false, py::str(message)));
    } catch (const std::exception& error) {
        return response_frame(
            frame.request_id,
            remote_call ? MessageType::REMOTE_RET : MessageType::PYTHON_RET,
            serialize_result(false, py::str(error.what())));
    }
}

RpcRuntime::RpcFrame RpcRuntime::handle_forward_autograd(
    const RpcFrame& frame) {
    py::gil_scoped_acquire gil;
    auto request =
        tensorplay::distributed::autograd::RpcWithAutograd::from_message(
            *frame.message);
    const auto& wrapped = request.wrapped_message();
    if (!wrapped ||
        (wrapped->type() != MessageType::PYTHON_CALL &&
         wrapped->type() != MessageType::PYTHON_REMOTE_CALL)) {
        throw std::runtime_error(
            "distributed autograd request contains an invalid RPC");
    }

    DeviceMap reverse_device_map;
    for (const auto& entry : request.device_map()) {
        reverse_device_map[entry.second] = entry.first;
    }
    auto context = tensorplay::distributed::autograd::add_recv_rpc_backward(
        request.metadata(),
        wrapped->tensors(),
        request.from_worker(),
        std::move(reverse_device_map));
    tensorplay::distributed::autograd::ContextGuard context_guard(
        request.metadata().context_id);

    RpcFrame inner_request{frame.request_id, wrapped};
    const bool remote_call = wrapped->type() == MessageType::PYTHON_REMOTE_CALL;
    RpcFrame inner_response = handle_call(
        inner_request,
        remote_call,
        request.metadata().context_id);
    bool success = false;
    {
        auto result = deserialize_result(*inner_response.message);
        success = std::get<0>(result);
    }
    if (!success) {
        return inner_response;
    }

    bool has_grad_tensor = false;
    for (const auto& value : inner_response.message->tensors()) {
        if (value.cast<tensorplay::Tensor>().requires_grad()) {
            has_grad_tensor = true;
            break;
        }
    }
    if (!has_grad_tensor) {
        return inner_response;
    }

    auto& container =
        tensorplay::distributed::autograd::DistAutogradContainer::instance();
    const tensorplay::distributed::autograd::AutogradMetadata response_metadata{
        request.metadata().context_id, container.new_message_id()};
    tensorplay::distributed::autograd::add_send_rpc_backward(
        context, response_metadata, inner_response.message->tensors());
    auto response = std::move(
        tensorplay::distributed::autograd::RpcWithAutograd(
            current_rank_,
            MessageType::FORWARD_AUTOGRAD_RESP,
            response_metadata,
            std::move(inner_response.message),
            {}))
                        .to_message();
    return {frame.request_id, std::move(response)};
}

RpcRuntime::RpcFrame RpcRuntime::handle_backward_autograd(
    const RpcFrame& frame) {
    try {
        auto request =
            tensorplay::distributed::autograd::PropagateGradientsReq::from_message(
                *frame.message);
        auto& container =
            tensorplay::distributed::autograd::DistAutogradContainer::instance();
        const auto context = container.retrieve(request.metadata().context_id);
        const auto function =
            context->send_function(request.metadata().message_id);
        function->set_grads(request.gradients());
        tensorplay::distributed::autograd::DistEngine::getInstance()
            .execute_send_function(context, function, request.retain_graph());
    } catch (py::error_already_set& error) {
        const std::string message = error.what();
        error.restore();
        PyErr_Clear();
        return response_frame(
            frame.request_id,
            MessageType::EXCEPTION,
            serialize_result(false, py::str(message)));
    } catch (const std::exception& error) {
        return response_frame(
            frame.request_id,
            MessageType::EXCEPTION,
            serialize_result(false, py::str(error.what())));
    }
    auto response =
        tensorplay::distributed::autograd::PropagateGradientsResp().to_message();
    response->set_id(static_cast<int64_t>(frame.request_id));
    return {frame.request_id, std::move(response)};
}

RpcRuntime::RpcFrame RpcRuntime::handle_cleanup_autograd(
    const RpcFrame& frame) {
    try {
        const auto request =
            tensorplay::distributed::autograd::CleanupAutogradContextReq::from_message(
                *frame.message);
        tensorplay::distributed::autograd::DistAutogradContainer::instance()
            .release_if_present(request.context_id());
        auto response =
            tensorplay::distributed::autograd::CleanupAutogradContextResp()
                .to_message();
        response->set_id(static_cast<int64_t>(frame.request_id));
        return {frame.request_id, std::move(response)};
    } catch (py::error_already_set& error) {
        const std::string message = error.what();
        error.restore();
        PyErr_Clear();
        return response_frame(
            frame.request_id,
            MessageType::EXCEPTION,
            serialize_result(false, py::str(message)));
    } catch (const std::exception& error) {
        return response_frame(
            frame.request_id,
            MessageType::EXCEPTION,
            serialize_result(false, py::str(error.what())));
    }
}

RpcRuntime::RpcFrame RpcRuntime::handle_rref_backward(
    const RpcFrame& frame) {
    py::gil_scoped_acquire gil;
    try {
        const auto request =
            tensorplay::distributed::autograd::RRefBackwardReq::from_message(
                *frame.message);
        const auto root = rrefs_.wait(request.rref_id(), -1.0).cast<tensorplay::Tensor>();
        if (request.context_id() < 0) {
            tensorplay::tpx::backward(root);
        } else {
            tensorplay::distributed::autograd::backward(
                request.context_id(), {root}, request.retain_graph());
        }
        auto response =
            tensorplay::distributed::autograd::RRefBackwardResp().to_message();
        response->set_id(static_cast<int64_t>(frame.request_id));
        return {frame.request_id, std::move(response)};
    } catch (py::error_already_set& error) {
        const std::string message = error.what();
        error.restore();
        PyErr_Clear();
        return response_frame(
            frame.request_id,
            MessageType::EXCEPTION,
            serialize_result(false, py::str(message)));
    } catch (const std::exception& error) {
        return response_frame(
            frame.request_id,
            MessageType::EXCEPTION,
            serialize_result(false, py::str(error.what())));
    }
}

RpcRuntime::RpcFrame RpcRuntime::handle_fetch(const RpcFrame& frame) {
    py::gil_scoped_acquire gil;
    try {
        const auto id = GloballyUniqueId::from_python(
            deserialize_python_object(serialized_message(frame)));
        const auto timeout = effective_duration(-1.0, rpc_timeout());
        py::object value = rrefs_.wait(
            id,
            timeout.count() < 0 ? -1.0 : static_cast<double>(timeout.count()) / 1000.0);
        return response_frame(
            frame.request_id,
            MessageType::PYTHON_RREF_FETCH_RET,
            serialize_result(true, std::move(value)));
    } catch (py::error_already_set& error) {
        const std::string message = error.what();
        error.restore();
        PyErr_Clear();
        return response_frame(
            frame.request_id,
            MessageType::PYTHON_RREF_FETCH_RET,
            serialize_result(false, py::str(message)));
    } catch (const std::exception& error) {
        return response_frame(
            frame.request_id,
            MessageType::PYTHON_RREF_FETCH_RET,
            serialize_result(false, py::str(error.what())));
    }
}

RpcRuntime::RpcFrame RpcRuntime::handle_fork(const RpcFrame& frame) {
    py::gil_scoped_acquire gil;
    try {
        const auto id = GloballyUniqueId::from_python(
            deserialize_python_object(serialized_message(frame)));
        rrefs_.retain(id);
        return response_frame(
            frame.request_id,
            MessageType::RREF_ACK,
            serialize_result(true, py::none()));
    } catch (const std::exception& error) {
        return response_frame(
            frame.request_id,
            MessageType::RREF_ACK,
            serialize_result(false, py::str(error.what())));
    }
}

RpcRuntime::RpcFrame RpcRuntime::handle_delete(const RpcFrame& frame) {
    py::gil_scoped_acquire gil;
    try {
        const auto id = GloballyUniqueId::from_python(
            deserialize_python_object(serialized_message(frame)));
        rrefs_.release(id);
        return response_frame(
            frame.request_id,
            MessageType::RREF_ACK,
            serialize_result(true, py::none()));
    } catch (const std::exception& error) {
        return response_frame(
            frame.request_id,
            MessageType::RREF_ACK,
            serialize_result(false, py::str(error.what())));
    }
}

RpcRuntime::RpcFrame RpcRuntime::handle_gather(const RpcFrame& frame) {
    py::gil_scoped_acquire gil;
    try {
        const py::tuple request = deserialize_python_object(
            serialized_message(frame)).cast<py::tuple>();
        if (request.size() != 6) {
            throw std::runtime_error(
                "collective request must contain six values");
        }
        const std::string collective_id = request[0].cast<std::string>();
        const worker_id_t source_id = request[1].cast<worker_id_t>();
        const worker_id_t leader_id = request[2].cast<worker_id_t>();
        const int phase = request[4].cast<int>();
        const py::object value = request[5];
        const auto expected = request[3].cast<std::vector<worker_id_t>>();
        if (collective_id.empty() || expected.empty()) {
            throw std::runtime_error("collective request is invalid");
        }
        if (phase != kGatherPhase && phase != kBroadcastPhase) {
            throw std::runtime_error("collective request phase is invalid");
        }
        if (std::find(expected.begin(), expected.end(), leader_id) ==
            expected.end()) {
            throw std::runtime_error("collective leader is not in the group");
        }
        std::vector<worker_id_t> sorted_expected = expected;
        std::sort(sorted_expected.begin(), sorted_expected.end());
        if (std::adjacent_find(
                sorted_expected.begin(), sorted_expected.end()) !=
            sorted_expected.end()) {
            throw std::runtime_error(
                "collective group contains duplicate workers");
        }
        std::shared_ptr<CollectiveState> state;
        {
            std::lock_guard<std::mutex> lock(collective_mutex_);
            auto& entry = collective_states_[collective_id];
            if (!entry) {
                entry = std::make_shared<CollectiveState>();
                entry->expected = sorted_expected;
            } else if (entry->expected != sorted_expected) {
                throw std::runtime_error(
                    "collective group changed during an active operation");
            }
            state = entry;
        }
        if (phase == kGatherPhase) {
            if (current_rank_ != leader_id ||
                std::find(
                    sorted_expected.begin(), sorted_expected.end(), source_id) ==
                    sorted_expected.end()) {
                throw std::runtime_error(
                    "collective gather destination is invalid");
            }
            std::lock_guard<std::mutex> lock(state->mutex);
            if (!state->values.emplace(source_id, value).second) {
                throw std::runtime_error(
                    "collective worker reported more than once");
            }
            if (state->values.size() == state->expected.size()) {
                state->ready = true;
                state->condition.notify_all();
            }
        } else {
            if (current_rank_ == leader_id || source_id != leader_id) {
                throw std::runtime_error(
                    "collective broadcast destination is invalid");
            }
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->ready) {
                throw std::runtime_error(
                    "collective broadcast was received more than once");
            }
            state->gathered = value.cast<py::dict>();
            state->ready = true;
            state->condition.notify_all();
        }
        return response_frame(
            frame.request_id,
            MessageType::PYTHON_GATHER_RET,
            serialize_result(true, py::none()));
    } catch (py::error_already_set& error) {
        const std::string message = error.what();
        error.restore();
        PyErr_Clear();
        return response_frame(
            frame.request_id,
            MessageType::PYTHON_GATHER_RET,
            serialize_result(false, py::str(message)));
    } catch (const std::exception& error) {
        return response_frame(
            frame.request_id,
            MessageType::PYTHON_GATHER_RET,
            serialize_result(false, py::str(error.what())));
    }
}

py::object RpcRuntime::fetch_rref(
    const RpcRRef& rref,
    double timeout_seconds) const {
    if (rref.owner().id == current_rank_) {
        return rrefs_.wait(rref.rref_id(), timeout_seconds);
    }
    py::gil_scoped_acquire gil;
    SerializedPyObj object = serialize_python_object(rref.rref_id().to_python());
    auto message = make_python_message(std::move(object),
                                       MessageType::PYTHON_RREF_FETCH_CALL);
    tensorplay::distributed::autograd::AutogradMetadata metadata;
    const auto context_id = current_autograd_context_id();
    if (context_id >= 0) {
        auto& container =
            tensorplay::distributed::autograd::DistAutogradContainer::instance();
        metadata = {context_id, container.new_message_id()};
        auto context = container.retrieve(context_id);
        context->add_known_worker(rref.owner().id);
        message = std::move(
            tensorplay::distributed::autograd::RpcWithAutograd(
                current_rank_,
                MessageType::FORWARD_AUTOGRAD_REQ,
                metadata,
                std::move(message),
                device_map_for(rref.owner().name)))
                      .to_message();
    }
    MessagePtr response;
    {
        py::gil_scoped_release release;
        response = send_message(
            rref.owner(),
            std::move(message),
            timeout_seconds,
            RpcRetryOptions{0, 1000, 1.5});
    }
    if (metadata.valid() && response->type() == MessageType::FORWARD_AUTOGRAD_RESP) {
        auto wrapped_response =
            tensorplay::distributed::autograd::RpcWithAutograd::from_message(
                *response);
        if (wrapped_response.metadata().context_id != metadata.context_id) {
            throw std::runtime_error(
                "distributed autograd RRef response metadata does not match request");
        }
        const auto& inner = wrapped_response.wrapped_message();
        if (!inner || inner->type() != MessageType::PYTHON_RREF_FETCH_RET) {
            throw std::runtime_error(
                "distributed autograd RRef response is malformed");
        }
        tensorplay::distributed::autograd::add_recv_rpc_backward(
            wrapped_response.metadata(),
            inner->tensors(),
            rref.owner().id,
            reverse_device_map_for(rref.owner().name));
        response = inner;
    }
    auto [success, value] = deserialize_result(*response);
    if (!success) {
        throw std::runtime_error(value.cast<std::string>());
    }
    return value;
}

void RpcRuntime::fork_rref(const RpcRRef& rref) const {
    if (rref.owner().id == current_rank_) {
        rrefs_.retain(rref.rref_id());
        return;
    }
    py::gil_scoped_acquire gil;
    SerializedPyObj object = serialize_python_object(rref.rref_id().to_python());
    auto message = make_python_message(std::move(object),
                                       MessageType::RREF_FORK_REQUEST);
    {
        py::gil_scoped_release release;
        send_message(
            rref.owner(),
            std::move(message),
            -1.0,
            RpcRetryOptions{});
    }
}

void RpcRuntime::delete_rref(const RpcRRef& rref) const {
    if (!initialized()) {
        return;
    }
    if (rref.owner().id == current_rank_) {
        rrefs_.release(rref.rref_id());
        return;
    }
    py::gil_scoped_acquire gil;
    SerializedPyObj object = serialize_python_object(rref.rref_id().to_python());
    auto message = make_python_message(std::move(object),
                                       MessageType::RREF_USER_DELETE);
    try {
        py::gil_scoped_release release;
        send_message(
            rref.owner(),
            std::move(message),
            -1.0,
            RpcRetryOptions{0, 1000, 1.5});
    } catch (...) {
    }
}

std::vector<std::string> RpcRuntime::normalize_worker_names(
    const std::vector<std::string>& names,
    const std::vector<WorkerInfo>& workers) {
    if (names.empty()) {
        std::vector<std::string> result;
        result.reserve(workers.size());
        for (const auto& worker : workers) {
            result.push_back(worker.name);
        }
        return result;
    }
    std::vector<std::string> result;
    result.reserve(names.size());
    for (const auto& name : names) {
        for (const auto& worker : workers) {
            if (worker.name == name) {
                result.push_back(name);
                goto found;
            }
        }
        throw std::invalid_argument("worker is not registered: " + name);
    found:
        continue;
    }
    return result;
}

MessagePtr RpcRuntime::send_collective(
    const WorkerInfo& to,
    const std::string& collective_id,
    worker_id_t leader_id,
    const std::vector<worker_id_t>& group_ids,
    int phase,
    py::object value,
    double timeout_seconds) const {
    SerializedPyObj object = serialize_python_object(
        py::make_tuple(
            collective_id,
            current_rank_,
            leader_id,
            group_ids,
            phase,
            std::move(value)));
    auto message = make_python_message(
        std::move(object), MessageType::PYTHON_GATHER_CALL);
    py::gil_scoped_release release;
    return send_message(
        to,
        std::move(message),
        timeout_seconds,
        RpcRetryOptions{0, 1000, 1.5});
}

py::dict RpcRuntime::all_gather(
    py::object value,
    const std::vector<std::string>& worker_names,
    double timeout_seconds) {
    std::vector<WorkerInfo> group;
    for (const auto& name : normalize_worker_names(
             worker_names, get_worker_infos())) {
        const WorkerInfo worker = resolve_worker(name);
        const auto duplicate = std::find_if(
            group.begin(),
            group.end(),
            [&worker](const WorkerInfo& entry) { return entry.id == worker.id; });
        if (duplicate == group.end()) {
            group.push_back(worker);
        }
    }
    const auto current = std::find_if(
        group.begin(),
        group.end(),
        [this](const WorkerInfo& worker) { return worker.id == current_rank_; });
    if (current == group.end()) {
        throw std::invalid_argument(
            "collective group must contain the current worker");
    }
    std::sort(
        group.begin(),
        group.end(),
        [](const WorkerInfo& left, const WorkerInfo& right) {
            return left.name < right.name;
        });
    const WorkerInfo leader = group.front();
    std::vector<worker_id_t> group_ids;
    group_ids.reserve(group.size());
    std::string group_key;
    for (const auto& worker : group) {
        group_ids.push_back(worker.id);
        group_key.append(std::to_string(worker.name.size()));
        group_key.push_back(':');
        group_key.append(worker.name);
        group_key.push_back(';');
    }
    std::sort(group_ids.begin(), group_ids.end());

    uint64_t sequence = 0;
    {
        std::lock_guard<std::mutex> lock(collective_mutex_);
        sequence = collective_sequences_[group_key]++;
    }
    const std::string collective_id =
        group_key + "#" + std::to_string(sequence);
    auto state = std::make_shared<CollectiveState>();
    state->expected = group_ids;
    {
        std::lock_guard<std::mutex> lock(collective_mutex_);
        const auto [iterator, inserted] =
            collective_states_.emplace(collective_id, state);
        if (!inserted) {
            throw std::runtime_error("collective sequence is already active");
        }
    }
    const auto remove_state = [this, &collective_id, &state]() {
        std::lock_guard<std::mutex> lock(collective_mutex_);
        const auto iterator = collective_states_.find(collective_id);
        if (iterator != collective_states_.end() && iterator->second == state) {
            collective_states_.erase(iterator);
        }
    };
    const auto signal_timeout = effective_duration(timeout_seconds, rpc_timeout());

    if (leader.id == current_rank_) {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->values.emplace(current_rank_, value);
        if (state->values.size() == state->expected.size()) {
            state->ready = true;
            state->condition.notify_all();
        }
    } else {
        try {
            MessagePtr response = send_collective(
                leader,
                collective_id,
                leader.id,
                group_ids,
                kGatherPhase,
                value,
                timeout_seconds);
            auto [success, remote_value] = deserialize_result(*response);
            if (!success) {
                remove_state();
                throw std::runtime_error(remote_value.cast<std::string>());
            }
        } catch (...) {
            remove_state();
            throw;
        }
    }

    bool ready = false;
    {
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(state->mutex);
        if (signal_timeout.count() < 0) {
            state->condition.wait(lock, [&state]() { return state->ready; });
            ready = true;
        } else {
            ready = state->condition.wait_for(
                lock,
                signal_timeout,
                [&state]() { return state->ready; });
        }
    }
    if (!ready) {
        remove_state();
        throw std::runtime_error("collective gather timed out");
    }

    if (leader.id == current_rank_) {
        py::dict gathered;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            for (const auto& worker : group) {
                const auto iterator = state->values.find(worker.id);
                if (iterator == state->values.end()) {
                    remove_state();
                    throw std::runtime_error(
                        "collective result is missing a worker");
                }
                gathered[worker.name.c_str()] = iterator->second;
            }
            state->gathered = gathered;
        }
        try {
            for (const auto& worker : group) {
                if (worker.id == current_rank_) {
                    continue;
                }
                MessagePtr response = send_collective(
                    worker,
                    collective_id,
                    leader.id,
                    group_ids,
                    kBroadcastPhase,
                    gathered,
                    timeout_seconds);
                auto [success, remote_value] = deserialize_result(*response);
                if (!success) {
                    throw std::runtime_error(remote_value.cast<std::string>());
                }
            }
        } catch (...) {
            remove_state();
            throw;
        }
        remove_state();
        return gathered;
    }

    py::dict gathered;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        gathered = state->gathered;
    }
    remove_state();
    return gathered;
}

void RpcRuntime::barrier(
    const std::vector<std::string>& worker_names,
    double timeout_seconds) {
    all_gather(py::none(), worker_names, timeout_seconds);
}

RpcFuturePtr RpcRuntime::send(
    const WorkerInfo& to,
    MessagePtr message,
    double timeout_seconds,
    const DeviceMap& device_map) {
    if (!message || !message->is_request()) {
        throw std::invalid_argument(
            "RPC agent send requires a request message");
    }
    const WorkerInfo registered = resolve_worker(to.name);
    if (registered.id != to.id) {
        throw std::invalid_argument(
            "RPC destination name and id do not match");
    }
    auto future = std::make_shared<RpcFuture>();
    Task task;
    task.valid = true;
    task.kind = TaskKind::MESSAGE;
    task.message = std::move(message);
    task.future = future;
    task.target = to.name;
    task.timeout_seconds = timeout_seconds;
    task.device_map = device_map;
    task.has_device_map = true;
    enqueue(std::move(task));
    return future;
}

RpcFuturePtr RpcRuntime::send_with_retries(
    const WorkerInfo& to,
    MessagePtr message,
    RpcRetryOptions options) {
    if (!message || !message->is_request()) {
        throw std::invalid_argument(
            "RPC agent send requires a request message");
    }
    const WorkerInfo registered = resolve_worker(to.name);
    if (registered.id != to.id) {
        throw std::invalid_argument(
            "RPC destination name and id do not match");
    }
    if (options.max_retries < 0 || options.retry_duration_ms < 0 ||
        options.retry_backoff < 1.0) {
        throw std::invalid_argument("invalid RPC retry options");
    }
    auto future = std::make_shared<RpcFuture>();
    Task task;
    task.valid = true;
    task.kind = TaskKind::MESSAGE;
    task.message = std::move(message);
    task.future = future;
    task.target = to.name;
    task.timeout_seconds = -1.0;
    task.retry_options = options;
    enqueue(std::move(task));
    return future;
}

const WorkerInfo& RpcRuntime::get_worker_info(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& worker : workers_) {
        if (worker.name == name) {
            return worker;
        }
    }
    throw std::invalid_argument("worker is not registered: " + name);
}

const WorkerInfo& RpcRuntime::get_worker_info(worker_id_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& worker : workers_) {
        if (worker.id == id) {
            return worker;
        }
    }
    throw std::invalid_argument("worker id is not registered");
}

DeviceMap RpcRuntime::get_device_map(const WorkerInfo& destination) const {
    return device_map_for(destination.name);
}

std::vector<WorkerInfo> RpcRuntime::get_worker_infos() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return workers_;
}

void RpcRuntime::start() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            throw std::runtime_error("RPC runtime is not initialized");
        }
        if (started_) {
            return;
        }
        started_ = true;
        stopping_ = false;
    }
    try {
        start_listener();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            worker_threads_.reserve(static_cast<size_t>(num_worker_threads_));
            for (int index = 0; index < num_worker_threads_; ++index) {
                worker_threads_.emplace_back([this]() { worker_loop(); });
            }
        }
    } catch (...) {
        shutdown();
        throw;
    }
}

void RpcRuntime::set_device_map(
    const std::string& worker,
    std::unordered_map<std::string, std::string> device_map) {
    const WorkerInfo target = resolve_worker(worker);
    if (target.name == worker_info_.name) {
        throw std::invalid_argument("device map target must be a remote worker");
    }
    std::unordered_map<std::string, std::string> reverse;
    for (const auto& entry : device_map) {
        if (entry.first.empty() || entry.second.empty()) {
            throw std::invalid_argument("device map contains an empty device");
        }
        auto [iterator, inserted] = reverse.emplace(entry.second, entry.first);
        if (!inserted && iterator->second != entry.first) {
            throw std::invalid_argument("device map must be one-to-one");
        }
    }
    std::lock_guard<std::mutex> lock(device_map_mutex_);
    auto& current = device_maps_[target.name];
    auto& reverse_current = reverse_device_maps_[target.name];
    for (const auto& entry : device_map) {
        auto current_entry = current.find(entry.first);
        if (current_entry != current.end() && current_entry->second != entry.second) {
            throw std::invalid_argument("device map source is already mapped");
        }
        for (const auto& existing : current) {
            if (existing.first != entry.first && existing.second == entry.second) {
                throw std::invalid_argument("device map target is already mapped");
            }
        }
        auto reverse_entry = reverse_current.find(entry.second);
        if (reverse_entry != reverse_current.end() &&
            reverse_entry->second != entry.first) {
            throw std::invalid_argument("device map target is already mapped");
        }
        current[entry.first] = entry.second;
        reverse_current[entry.second] = entry.first;
    }
}

DeviceMap RpcRuntime::device_map_for(const std::string& worker) const {
    std::lock_guard<std::mutex> lock(device_map_mutex_);
    const auto iterator = device_maps_.find(worker);
    return iterator == device_maps_.end() ? DeviceMap{} : iterator->second;
}

DeviceMap RpcRuntime::reverse_device_map_for(const std::string& worker) const {
    std::lock_guard<std::mutex> lock(device_map_mutex_);
    const auto iterator = reverse_device_maps_.find(worker);
    return iterator == reverse_device_maps_.end() ? DeviceMap{} : iterator->second;
}

void RpcRuntime::shutdown() {
    std::shared_ptr<tensorpipe::Listener> listener;
    std::shared_ptr<tensorpipe::Context> context;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            return;
        }
        stopping_ = true;
        started_ = false;
        listener = tensorpipe_listener_;
        context = tensorpipe_context_;
    }
    condition_.notify_all();
    if (listener) {
        listener->close();
    }
    if (context) {
        context->close();
    }
    std::vector<std::shared_ptr<ClientPipe>> client_pipes;
    {
        std::lock_guard<std::mutex> lock(client_mutex_);
        client_pipes.reserve(client_pipes_.size());
        for (const auto& entry : client_pipes_) {
            client_pipes.push_back(entry.second);
        }
    }
    for (const auto& client_pipe : client_pipes) {
        if (client_pipe && client_pipe->pipe) {
            client_pipe->pipe->close();
        }
    }
    for (auto& worker : worker_threads_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    std::vector<std::thread> pipes;
    {
        std::lock_guard<std::mutex> lock(client_mutex_);
        pipes.swap(pipe_threads_);
    }
    for (auto& pipe : pipes) {
        if (pipe.joinable()) {
            pipe.join();
        }
    }
    if (context) {
        context->join();
    }
    {
        std::lock_guard<std::mutex> lock(client_mutex_);
        client_pipes_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        worker_threads_.clear();
        active_tasks_ = 0;
        queue_.clear();
        workers_.clear();
        worker_urls_.clear();
        rendezvous_store_.reset();
        tensorpipe_listener_.reset();
        tensorpipe_context_.reset();
        initialized_ = false;
        stopping_ = false;
        rendezvous_prefix_.clear();
        bootstrap_transport_.clear();
        transports_.reset();
        channels_.reset();
    }
    {
        std::lock_guard<std::mutex> lock(collective_mutex_);
        collective_states_.clear();
        collective_sequences_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(device_map_mutex_);
        device_maps_.clear();
        reverse_device_maps_.clear();
    }
    rrefs_.clear();
    profiler::shutdown_server_profiler();
}

void RpcRuntime::join(bool should_shutdown, double timeout_seconds) {
    if (should_shutdown) {
        shutdown();
        return;
    }
    const auto timeout = effective_duration(timeout_seconds, rpc_timeout());
    {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto ready = [this]() {
            return queue_.empty() && active_tasks_ == 0;
        };
        if (timeout.count() < 0) {
            idle_condition_.wait(lock, ready);
        } else if (!idle_condition_.wait_for(lock, timeout, ready)) {
            throw std::runtime_error("RPC join timed out");
        }
    }
    if (world_size_ > 1) {
        barrier({}, timeout_seconds);
    }
}

void RpcRuntime::sync(double timeout_seconds) {
    barrier({}, timeout_seconds);
}

std::unordered_map<std::string, std::string> RpcRuntime::get_metrics() const {
    auto result = metrics_.snapshot();
    result["rref_count"] = std::to_string(rrefs_.size());
    result["initialized"] = initialized() ? "1" : "0";
    return result;
}

std::unordered_map<std::string, std::string> RpcRuntime::get_debug_info() const {
    auto result = get_metrics();
    result["worker_name"] = worker_info_.name;
    result["worker_id"] = std::to_string(worker_info_.id);
    result["world_size"] = std::to_string(world_size_);
    result["master_addr"] = master_addr_;
    result["master_port"] = std::to_string(master_port_);
    return result;
}

void RpcRuntime::profiler_start(bool record_call_stack) {
    profiler::global_profiler().start(record_call_stack);
}

void RpcRuntime::profiler_stop() {
    profiler::global_profiler().stop();
}

std::vector<profiler::Event> RpcRuntime::profiler_events() const {
    return profiler::global_profiler().events();
}

RpcRuntime& global_rpc_runtime() {
    static RpcRuntime runtime;
    return runtime;
}

}  // namespace tensorplay::distributed::rpc
