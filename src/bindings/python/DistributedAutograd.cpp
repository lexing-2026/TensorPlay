#include "python_bindings.h"

#include "../../distributed/autograd/autograd.h"
#include "../../distributed/autograd/context/container.h"
#include "../../distributed/autograd/engine/dist_engine.h"
#include "../../distributed/autograd/functions/recvrpc_backward.h"
#include "../../distributed/autograd/functions/sendrpc_backward.h"
#include "../../distributed/rpc/rpc_agent.h"

#include <pybind11/stl.h>

#include <memory>

namespace {

using tensorplay::distributed::autograd::ContextPtr;
using tensorplay::distributed::autograd::DistAutogradContainer;
using tensorplay::distributed::autograd::DistAutogradContext;
using tensorplay::distributed::autograd::DistEngine;
using tensorplay::distributed::autograd::RecvRpcBackward;
using tensorplay::distributed::autograd::SendRpcBackward;
using tensorplay::distributed::rpc::RpcAgent;

py::dict node_map(
    const std::unordered_map<
        int64_t,
        std::shared_ptr<tensorplay::tpx::Node>>& functions) {
    py::dict result;
    for (const auto& entry : functions) {
        result[py::int_(entry.first)] = py::cast(entry.second);
    }
    return result;
}

}  // namespace

void init_distributed_autograd(py::module_& m) {
    py::module_ distributed_autograd = m.def_submodule(
        "_distributed_autograd", "Native distributed autograd runtime");

    py::class_<SendRpcBackward,
               tensorplay::tpx::Node,
               std::shared_ptr<SendRpcBackward>>(
        distributed_autograd, "SendRpcBackward");
    py::class_<RecvRpcBackward,
               tensorplay::tpx::Node,
               std::shared_ptr<RecvRpcBackward>>(
        distributed_autograd, "RecvRpcBackward");

    py::class_<DistAutogradContext,
               std::shared_ptr<DistAutogradContext>>(
        distributed_autograd, "DistAutogradContext")
        .def(
            "_context_id",
            &DistAutogradContext::context_id,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "_send_functions",
            [](const DistAutogradContext& context) {
                std::unordered_map<
                    int64_t,
                    std::shared_ptr<tensorplay::tpx::Node>> functions;
                for (const auto& entry : context.send_functions()) {
                    functions.emplace(entry.first, entry.second);
                }
                return node_map(functions);
            })
        .def(
            "_recv_functions",
            [](const DistAutogradContext& context) {
                std::unordered_map<
                    int64_t,
                    std::shared_ptr<tensorplay::tpx::Node>> functions;
                for (const auto& entry : context.recv_functions()) {
                    functions.emplace(entry.first, entry.second);
                }
                return node_map(functions);
            })
        .def(
            "_known_worker_ids",
            &DistAutogradContext::known_workers,
            py::call_guard<py::gil_scoped_release>());

    distributed_autograd.def(
        "_new_context",
        []() { return DistAutogradContainer::instance().new_context(); },
        py::call_guard<py::gil_scoped_release>());
    distributed_autograd.def(
        "_release_context",
        [](int64_t context_id) {
            DistAutogradContainer::instance().release(context_id);
        },
        py::call_guard<py::gil_scoped_release>());
    distributed_autograd.def(
        "_get_max_id",
        []() { return DistAutogradContainer::instance().max_id(); },
        py::call_guard<py::gil_scoped_release>());
    distributed_autograd.def(
        "_is_valid_context",
        [](int64_t context_id) {
            DistAutogradContainer::instance().validate(context_id);
        },
        py::call_guard<py::gil_scoped_release>());
    distributed_autograd.def(
        "_retrieve_context",
        [](int64_t context_id) {
            return DistAutogradContainer::instance().retrieve(context_id);
        },
        py::call_guard<py::gil_scoped_release>());
    distributed_autograd.def(
        "_current_context",
        []() { return DistAutogradContainer::instance().current(); },
        py::call_guard<py::gil_scoped_release>());
    distributed_autograd.def(
        "_init",
        [](int64_t worker_id) {
            DistAutogradContainer::init(worker_id, RpcAgent::current_rpc_agent());
        },
        py::call_guard<py::gil_scoped_release>());
    distributed_autograd.def(
        "_is_initialized",
        []() { return DistAutogradContainer::is_initialized_global(); });
    distributed_autograd.def(
        "_get_debug_info",
        []() { return DistEngine::getInstance().get_debug_info(); },
        py::call_guard<py::gil_scoped_release>());
    distributed_autograd.def(
        "backward",
        [](int64_t context_id,
           const std::vector<tensorplay::Tensor>& roots,
           bool retain_graph) {
            tensorplay::distributed::autograd::backward(
                context_id, roots, retain_graph);
        },
        py::arg("context_id"),
        py::arg("roots"),
        py::arg("retain_graph") = false,
        py::call_guard<py::gil_scoped_release>());
    distributed_autograd.def(
        "get_gradients",
        [](int64_t context_id) {
            const auto context =
                DistAutogradContainer::instance().retrieve(context_id);
            py::dict result;
            for (const auto& entry : context->gradients()) {
                result[py::cast(entry.first)] = py::cast(entry.second);
            }
            return result;
        },
        py::arg("context_id"));
}
