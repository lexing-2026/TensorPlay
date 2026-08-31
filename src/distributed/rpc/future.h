#pragma once

#include <pybind11/pybind11.h>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace tensorplay::distributed::rpc {

namespace py = pybind11;

class RpcFuture final : public std::enable_shared_from_this<RpcFuture> {
public:
    RpcFuture();
    ~RpcFuture();

    RpcFuture(const RpcFuture&) = delete;
    RpcFuture& operator=(const RpcFuture&) = delete;

    bool done() const;
    py::object wait(double timeout_seconds = -1.0);
    py::object value() const;
    py::object exception(double timeout_seconds = -1.0);
    void set_result(py::object value);
    void set_exception(py::object error);
    std::shared_ptr<RpcFuture> then(py::function callback);
    void add_done_callback(py::function callback);

private:
    struct State;
    std::shared_ptr<State> state_;

    py::object outcome() const;
    void invoke_callbacks(
        std::vector<std::function<void(py::object)>> callbacks);
};

using RpcFuturePtr = std::shared_ptr<RpcFuture>;

}  // namespace tensorplay::distributed::rpc
