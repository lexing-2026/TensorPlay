#include "future.h"

#include <stdexcept>
#include <utility>

namespace tensorplay::distributed::rpc {

namespace {

py::object make_callback_exception(const char* message) {
    PyObject* value = PyObject_CallFunction(PyExc_RuntimeError, "s", message);
    if (value == nullptr) {
        PyErr_Clear();
        return py::none();
    }
    return py::reinterpret_steal<py::object>(value);
}

void fail_child(
    const std::shared_ptr<RpcFuture>& child,
    const char* message) {
    try {
        py::object exception = make_callback_exception(message);
        if (!exception.is_none()) {
            child->set_exception(std::move(exception));
        }
    } catch (...) {
    }
}

} // namespace

struct RpcFuture::State {
    mutable std::mutex mutex;
    std::condition_variable condition;
    bool completed = false;
    bool has_exception = false;
    py::object result = py::none();
    py::object error = py::none();
    std::vector<std::function<void(py::object)>> callbacks;

    ~State() {
        if (!Py_IsInitialized()) {
            return;
        }
        py::gil_scoped_acquire gil;
        callbacks.clear();
        result = py::none();
        error = py::none();
    }
};

RpcFuture::RpcFuture() : state_(std::make_shared<State>()) {}

RpcFuture::~RpcFuture() = default;

bool RpcFuture::done() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->completed;
}

py::object RpcFuture::outcome() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->completed) {
        throw std::runtime_error("RPC future has not completed");
    }
    if (state_->has_exception) {
        PyErr_SetObject(
            PyExceptionInstance_Class(state_->error.ptr()), state_->error.ptr());
        throw py::error_already_set();
    }
    return state_->result;
}

py::object RpcFuture::wait(double timeout_seconds) {
    bool ready = false;
    {
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(state_->mutex);
        if (timeout_seconds < 0.0) {
            state_->condition.wait(lock, [this]() { return state_->completed; });
            ready = true;
        } else {
            ready = state_->condition.wait_for(
                lock,
                std::chrono::duration<double>(timeout_seconds),
                [this]() { return state_->completed; });
        }
    }
    if (!ready) {
        PyErr_SetString(PyExc_TimeoutError, "RPC future wait timed out");
        throw py::error_already_set();
    }
    return outcome();
}

py::object RpcFuture::value() const {
    return outcome();
}

py::object RpcFuture::exception(double timeout_seconds) {
    if (timeout_seconds >= 0.0 && !done()) {
        wait(timeout_seconds);
    } else if (timeout_seconds < 0.0 && !done()) {
        wait();
    }
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->has_exception ? state_->error : py::none();
}

void RpcFuture::set_result(py::object value) {
    std::vector<std::function<void(py::object)>> callbacks;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->completed) {
            throw std::runtime_error("RPC future can only be completed once");
        }
        state_->result = std::move(value);
        state_->error = py::none();
        state_->has_exception = false;
        state_->completed = true;
        callbacks.swap(state_->callbacks);
    }
    state_->condition.notify_all();
    invoke_callbacks(std::move(callbacks));
}

void RpcFuture::set_exception(py::object error) {
    if (!PyExceptionInstance_Check(error.ptr())) {
        throw py::type_error("RPC future error must be an exception instance");
    }
    std::vector<std::function<void(py::object)>> callbacks;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->completed) {
            throw std::runtime_error("RPC future can only be completed once");
        }
        state_->error = std::move(error);
        state_->result = py::none();
        state_->has_exception = true;
        state_->completed = true;
        callbacks.swap(state_->callbacks);
    }
    state_->condition.notify_all();
    invoke_callbacks(std::move(callbacks));
}

void RpcFuture::invoke_callbacks(
    std::vector<std::function<void(py::object)>> callbacks) {
    py::gil_scoped_acquire gil;
    py::object self = py::cast(shared_from_this());
    for (auto& callback : callbacks) {
        try {
            callback(self);
        } catch (py::error_already_set& error) {
            error.discard_as_unraisable("RPC future callback");
        } catch (...) {
        }
    }
}

std::shared_ptr<RpcFuture> RpcFuture::then(py::function callback) {
    auto child = std::make_shared<RpcFuture>();
    py::object self = py::cast(shared_from_this());
    bool call_now = false;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->completed) {
            call_now = true;
        } else {
            state_->callbacks.emplace_back(
                [child, callback = std::move(callback)](py::object value) mutable {
                    try {
                        child->set_result(callback(value));
                    } catch (py::error_already_set& error) {
                        py::object exception = py::reinterpret_borrow<py::object>(
                            error.value());
                        child->set_exception(std::move(exception));
                        error.restore();
                        PyErr_Clear();
                    } catch (const std::exception& error) {
                        fail_child(child, error.what());
                    } catch (...) {
                        fail_child(child, "RPC future callback failed");
                    }
                });
        }
    }
    if (call_now) {
        try {
            child->set_result(callback(self));
        } catch (py::error_already_set& error) {
            py::object exception = py::reinterpret_borrow<py::object>(error.value());
            child->set_exception(std::move(exception));
            error.restore();
            PyErr_Clear();
        } catch (const std::exception& error) {
            fail_child(child, error.what());
        } catch (...) {
            fail_child(child, "RPC future callback failed");
        }
    }
    return child;
}

void RpcFuture::add_done_callback(py::function callback) {
    bool call_now = false;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->completed) {
            call_now = true;
        } else {
            state_->callbacks.emplace_back(
                [callback = std::move(callback)](py::object value) mutable {
                    callback(value);
                });
        }
    }
    if (call_now) {
        try {
            callback(py::cast(shared_from_this()));
        } catch (py::error_already_set& error) {
            error.discard_as_unraisable("RPC future callback");
        }
    }
}

}  // namespace tensorplay::distributed::rpc
