#include "python_bindings.h"

#include "../../distributed/rpc/python_call.h"
#include "../../distributed/rpc/python_functions.h"
#include "../../distributed/rpc/rpc_agent.h"
#include "../../distributed/rpc/rpc_runtime.h"
#include "../../distributed/rpc/rref_impl.h"
#include "../../distributed/store/Store.h"
#include "../../distributed/rpc/tensorpipe_agent.h"

#include <pybind11/gil.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using tensorplay::distributed::rpc::GloballyUniqueId;
using tensorplay::distributed::rpc::MessageType;
using tensorplay::distributed::rpc::RpcFuture;
using tensorplay::distributed::rpc::RpcFuturePtr;
using tensorplay::distributed::rpc::RpcBackendOptions;
using tensorplay::distributed::rpc::RpcRetryOptions;
using tensorplay::distributed::rpc::RpcRRef;
using tensorplay::distributed::rpc::RpcRuntime;
using tensorplay::distributed::rpc::TensorPipeAgent;
using tensorplay::distributed::rpc::TensorPipeRpcBackendOptions;
using tensorplay::distributed::rpc::WorkerInfo;
using tensorplay::distributed::rpc::global_rpc_runtime;

RpcRuntime& runtime() {
    return global_rpc_runtime();
}

tensorplay::distributed::rpc::worker_id_t checked_worker_id(int64_t id) {
    if (id < std::numeric_limits<tensorplay::distributed::rpc::worker_id_t>::min() ||
        id > std::numeric_limits<tensorplay::distributed::rpc::worker_id_t>::max()) {
        throw py::value_error("RPC worker id is out of range");
    }
    return static_cast<tensorplay::distributed::rpc::worker_id_t>(id);
}

py::dict event_to_dict(
    const tensorplay::distributed::rpc::profiler::Event& event) {
    py::dict result;
    result["name"] = event.name;
    result["source"] = event.source;
    result["destination"] = event.destination;
    result["start_ns"] = event.start_ns;
    result["end_ns"] = event.end_ns;
    result["error"] = event.error;
    return result;
}

}  // namespace

void init_rpc(py::module_& m) {
    py::module_ rpc = m.def_submodule("_distributed_rpc", "Native distributed runtime");

    py::class_<RpcRetryOptions>(rpc, "RpcRetryOptions")
        .def(py::init<>())
        .def_readwrite("max_retries", &RpcRetryOptions::max_retries)
        .def_readwrite("retry_duration_ms", &RpcRetryOptions::retry_duration_ms)
        .def_readwrite("retry_backoff", &RpcRetryOptions::retry_backoff);

    py::class_<RpcBackendOptions, std::shared_ptr<RpcBackendOptions>>(
        rpc, "RpcBackendOptions")
        .def(py::init<>())
        .def(
            py::init<double, std::string>(),
            py::arg("rpc_timeout") = 60.0,
            py::arg("init_method") = "env://")
        .def_readwrite("rpc_timeout", &RpcBackendOptions::rpc_timeout_seconds)
        .def_readwrite("init_method", &RpcBackendOptions::init_method)
        .def_readwrite("num_worker_threads", &RpcBackendOptions::num_worker_threads)
        .def_readwrite("transports", &RpcBackendOptions::transports)
        .def_readwrite("channels", &RpcBackendOptions::channels)
        .def_readwrite("device_maps", &RpcBackendOptions::device_maps)
        .def_readwrite("devices", &RpcBackendOptions::devices)
        .def(
            "set_device_map",
            &RpcBackendOptions::set_device_map,
            py::arg("worker"),
            py::arg("device_map"));

    py::class_<TensorPipeRpcBackendOptions,
               RpcBackendOptions,
               std::shared_ptr<TensorPipeRpcBackendOptions>>(
        rpc, "TensorPipeRpcBackendOptions")
        .def(py::init<>())
        .def(
            py::init<
                int,
                std::optional<std::vector<std::string>>,
                std::optional<std::vector<std::string>>,
                double,
                std::string>(),
            py::arg("num_worker_threads") = 16,
            py::arg("transports") = std::nullopt,
            py::arg("channels") = std::nullopt,
            py::arg("rpc_timeout") = 60.0,
            py::arg("init_method") = "env://")
        .def(
            "set_device_map",
            &TensorPipeRpcBackendOptions::set_device_map,
            py::arg("worker"),
            py::arg("device_map"));

    py::class_<tensorplay::distributed::rpc::RpcAgent,
               std::shared_ptr<tensorplay::distributed::rpc::RpcAgent>>(
        rpc, "RpcAgent")
        .def("get_worker_info", [](const tensorplay::distributed::rpc::RpcAgent& agent) {
            return agent.worker_info();
        })
        .def("get_worker_info", [](const tensorplay::distributed::rpc::RpcAgent& agent, const std::string& name) {
            return agent.get_worker_info(name);
        })
        .def("get_worker_info", [](const tensorplay::distributed::rpc::RpcAgent& agent, int64_t id) {
            return agent.get_worker_info(checked_worker_id(id));
        })
        .def("get_worker_infos", &tensorplay::distributed::rpc::RpcAgent::get_worker_infos)
        .def("get_metrics", &tensorplay::distributed::rpc::RpcAgent::get_metrics)
        .def("get_debug_info", &tensorplay::distributed::rpc::RpcAgent::get_debug_info)
        .def(
            "shutdown",
            &tensorplay::distributed::rpc::RpcAgent::shutdown,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "start",
            &tensorplay::distributed::rpc::RpcAgent::start,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "join",
            &tensorplay::distributed::rpc::RpcAgent::join,
            py::arg("shutdown") = false,
            py::arg("timeout") = 0.0,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "sync",
            &tensorplay::distributed::rpc::RpcAgent::sync,
            py::call_guard<py::gil_scoped_release>());

    py::class_<RpcRuntime,
               tensorplay::distributed::rpc::RpcAgent,
               std::shared_ptr<RpcRuntime>>(rpc, "RpcRuntime")
        .def("initialized", &RpcRuntime::initialized)
        .def("started", &RpcRuntime::started)
        .def("current_worker", &RpcRuntime::current_worker)
        .def("workers", &RpcRuntime::workers)
        .def(
            "configure_backend",
            &RpcRuntime::configure_backend,
            py::arg("transports") = std::nullopt,
            py::arg("channels") = std::nullopt)
        .def("submit", &RpcRuntime::submit)
        .def("remote", &RpcRuntime::remote)
        .def(
            "restore_rref",
            [](RpcRuntime& runtime,
               const std::string& owner,
               int64_t owner_id,
               py::handle rref_id,
               py::handle fork_id) {
                return runtime.restore_rref(
                    owner,
                    checked_worker_id(owner_id),
                    GloballyUniqueId::from_python(rref_id),
                    GloballyUniqueId::from_python(fork_id));
            },
            py::arg("owner"),
            py::arg("owner_id"),
            py::arg("rref_id"),
            py::arg("fork_id"))
        .def("store", &RpcRuntime::store)
        .def("get_device_map", &RpcRuntime::get_device_map)
        .def("all_gather", &RpcRuntime::all_gather)
        .def("barrier", &RpcRuntime::barrier)
        .def("set_device_map", &RpcRuntime::set_device_map);

    py::class_<TensorPipeAgent,
               RpcRuntime,
               std::shared_ptr<TensorPipeAgent>>(rpc, "TensorPipeAgent")
        .def(
            py::init<std::shared_ptr<tensorplay::distributed::Store>,
                     std::string,
                     tensorplay::distributed::rpc::worker_id_t,
                     tensorplay::distributed::rpc::worker_id_t,
                     TensorPipeRpcBackendOptions>(),
            py::arg("store"),
            py::arg("name"),
            py::arg("rank"),
            py::arg("world_size"),
            py::arg("options"),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "get_backend_options",
            &TensorPipeAgent::backend_options,
            py::return_value_policy::reference_internal)
        .def("set_device_map", &TensorPipeAgent::set_device_map)
        .def("device_maps", &TensorPipeAgent::device_maps);

    py::enum_<MessageType>(rpc, "MessageType")
        .value("PYTHON_CALL", MessageType::PYTHON_CALL)
        .value("PYTHON_RET", MessageType::PYTHON_RET)
        .value("PYTHON_REMOTE_CALL", MessageType::PYTHON_REMOTE_CALL)
        .value("REMOTE_RET", MessageType::REMOTE_RET)
        .value("PYTHON_RREF_FETCH_CALL", MessageType::PYTHON_RREF_FETCH_CALL)
        .value("PYTHON_RREF_FETCH_RET", MessageType::PYTHON_RREF_FETCH_RET)
        .value("RREF_USER_DELETE", MessageType::RREF_USER_DELETE)
        .value("RREF_FORK_REQUEST", MessageType::RREF_FORK_REQUEST)
        .value("RREF_CHILD_ACCEPT", MessageType::RREF_CHILD_ACCEPT)
        .value("RREF_ACK", MessageType::RREF_ACK)
        .value("FORWARD_AUTOGRAD_REQ", MessageType::FORWARD_AUTOGRAD_REQ)
        .value("FORWARD_AUTOGRAD_RESP", MessageType::FORWARD_AUTOGRAD_RESP)
        .value("BACKWARD_AUTOGRAD_REQ", MessageType::BACKWARD_AUTOGRAD_REQ)
        .value("BACKWARD_AUTOGRAD_RESP", MessageType::BACKWARD_AUTOGRAD_RESP)
        .value("CLEANUP_AUTOGRAD_CONTEXT_REQ", MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ)
        .value("CLEANUP_AUTOGRAD_CONTEXT_RESP", MessageType::CLEANUP_AUTOGRAD_CONTEXT_RESP)
        .value("RUN_WITH_PROFILING_REQ", MessageType::RUN_WITH_PROFILING_REQ)
        .value("RUN_WITH_PROFILING_RESP", MessageType::RUN_WITH_PROFILING_RESP)
        .value("RREF_BACKWARD_REQ", MessageType::RREF_BACKWARD_REQ)
        .value("RREF_BACKWARD_RESP", MessageType::RREF_BACKWARD_RESP)
        .value("PYTHON_GATHER_CALL", MessageType::PYTHON_GATHER_CALL)
        .value("PYTHON_GATHER_RET", MessageType::PYTHON_GATHER_RET)
        .value("EXCEPTION", MessageType::EXCEPTION)
        .value("UNKNOWN", MessageType::UNKNOWN)
        .export_values();

    py::class_<GloballyUniqueId>(rpc, "GloballyUniqueId")
        .def(py::init<tensorplay::distributed::rpc::worker_id_t,
                      tensorplay::distributed::rpc::local_id_t>())
        .def_readonly("created_on", &GloballyUniqueId::created_on)
        .def_readonly("local_id", &GloballyUniqueId::local_id)
        .def("to_tuple", &GloballyUniqueId::to_python)
        .def_static("from_tuple", &GloballyUniqueId::from_python)
        .def("__repr__", &GloballyUniqueId::to_string)
        .def("__hash__", [](const GloballyUniqueId& value) {
            return GloballyUniqueId::Hash{}(value);
        })
        .def("__eq__", &GloballyUniqueId::operator==);

    py::class_<WorkerInfo>(rpc, "WorkerInfo")
        .def(py::init<std::string, tensorplay::distributed::rpc::worker_id_t>())
        .def_readonly("name", &WorkerInfo::name)
        .def_readonly("id", &WorkerInfo::id)
        .def("__repr__", [](const WorkerInfo& value) {
            return "WorkerInfo(name='" + value.name + "', id=" +
                std::to_string(value.id) + ")";
        })
        .def("__eq__", &WorkerInfo::operator==);

    py::class_<RpcFuture, RpcFuturePtr>(rpc, "Future")
        .def("done", &RpcFuture::done)
        .def("wait", &RpcFuture::wait, py::arg("timeout") = -1.0)
        .def("value", &RpcFuture::value)
        .def("exception", &RpcFuture::exception, py::arg("timeout") = -1.0)
        .def("set_result", &RpcFuture::set_result, py::arg("result"))
        .def("set_exception", &RpcFuture::set_exception, py::arg("error"))
        .def("then", &RpcFuture::then, py::arg("callback"))
        .def("add_done_callback", &RpcFuture::add_done_callback, py::arg("callback"));

    py::class_<RpcRRef, std::shared_ptr<RpcRRef>>(rpc, "RRef")
        .def("owner", [](const RpcRRef& value) { return value.owner().name; })
        .def("owner_id", [](const RpcRRef& value) { return value.owner().id; })
        .def("rref_id", [](const RpcRRef& value) { return value.rref_id().to_python(); })
        .def("fork_id", [](const RpcRRef& value) { return value.fork_id().to_python(); })
        .def("is_owner", &RpcRRef::is_owner)
        .def("confirmed_by_owner", &RpcRRef::confirmed_by_owner)
        .def("to_here", &RpcRRef::to_here, py::arg("timeout") = -1.0)
        .def("local_value", &RpcRRef::local_value)
        .def(
            "backward",
            &RpcRRef::backward,
            py::arg("context_id") = -1,
            py::arg("retain_graph") = false)
        .def("fork", &RpcRRef::fork)
        .def("__repr__", &RpcRRef::repr);

    rpc.def(
        "init",
        [](const std::string& name,
           int64_t rank,
           int64_t world_size,
           int threads,
           double timeout,
           const std::string& init_method) {
            if (rank < 0 || rank > std::numeric_limits<tensorplay::distributed::rpc::worker_id_t>::max() ||
                world_size <= 0 ||
                world_size > std::numeric_limits<tensorplay::distributed::rpc::worker_id_t>::max()) {
                throw py::value_error("RPC worker ids are out of range");
            }
            runtime().init(
                name,
                static_cast<tensorplay::distributed::rpc::worker_id_t>(rank),
                static_cast<tensorplay::distributed::rpc::worker_id_t>(world_size),
                threads,
                timeout,
                init_method);
            runtime().start();
        },
        py::arg("name"),
        py::arg("rank"),
        py::arg("world_size"),
        py::arg("threads"),
        py::arg("timeout") = 60.0,
        py::arg("init_method") = "env://");
    rpc.def(
        "configure_backend",
        [](std::optional<std::vector<std::string>> transports,
           std::optional<std::vector<std::string>> channels) {
            runtime().configure_backend(
                std::move(transports), std::move(channels));
        },
        py::arg("transports") = std::nullopt,
        py::arg("channels") = std::nullopt);
    rpc.def("is_initialized", []() { return runtime().initialized(); });
    rpc.def(
        "set_device_map",
        [](const std::string& worker,
           std::unordered_map<std::string, std::string> device_map) {
            runtime().set_device_map(worker, std::move(device_map));
        },
        py::arg("worker"),
        py::arg("device_map"));
    rpc.def("current_worker", []() { return runtime().current_worker(); });
    rpc.def("workers", []() { return runtime().workers(); });
    rpc.def(
        "submit",
        [](const std::string& target,
           py::object callable,
           py::tuple args,
           py::dict kwargs,
           double timeout) {
            return runtime().submit(
                target, std::move(callable), std::move(args), std::move(kwargs), timeout);
        },
        py::arg("target"),
        py::arg("callable"),
        py::arg("args"),
        py::arg("kwargs"),
        py::arg("timeout") = -1.0);
    rpc.def(
        "remote",
        [](const std::string& target,
           py::object callable,
           py::tuple args,
           py::dict kwargs,
           double timeout) {
            return runtime().remote(
                target, std::move(callable), std::move(args), std::move(kwargs), timeout);
        },
        py::arg("target"),
        py::arg("callable"),
        py::arg("args"),
        py::arg("kwargs"),
        py::arg("timeout") = -1.0);
    rpc.def(
        "restore_rref",
        [](const std::string& owner,
           int64_t owner_id,
           py::handle rref_id,
           py::handle fork_id) {
            return runtime().restore_rref(
                owner,
                checked_worker_id(owner_id),
                GloballyUniqueId::from_python(rref_id),
                GloballyUniqueId::from_python(fork_id));
        },
        py::arg("owner"),
        py::arg("owner_id"),
        py::arg("rref_id"),
        py::arg("fork_id"));
    rpc.def(
        "all_gather",
        [](py::object value,
           const std::vector<std::string>& workers,
           double timeout) { return runtime().all_gather(value, workers, timeout); },
        py::arg("value"),
        py::arg("workers") = std::vector<std::string>(),
        py::arg("timeout") = -1.0);
    rpc.def(
        "barrier",
        [](const std::vector<std::string>& workers, double timeout) {
            runtime().barrier(workers, timeout);
        },
        py::arg("workers") = std::vector<std::string>(),
        py::arg("timeout") = -1.0);
    rpc.def("metrics", []() { return runtime().get_metrics(); });
    rpc.def("debug_info", []() { return runtime().get_debug_info(); });
    rpc.def("profiler_start", [](bool record_call_stack) {
        runtime().profiler_start(record_call_stack);
    }, py::arg("record_call_stack") = false);
    rpc.def("profiler_stop", []() { runtime().profiler_stop(); });
    rpc.def("profiler_events", []() {
        py::list result;
        for (const auto& event : runtime().profiler_events()) {
            result.append(event_to_dict(event));
        }
        return result;
    });
    rpc.def("shutdown", []() { runtime().shutdown(); });
}
