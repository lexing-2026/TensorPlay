#include "python_bindings.h"

#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class NativeFuture;
using FuturePtr = std::shared_ptr<NativeFuture>;
using Callback = std::function<void()>;

py::object make_runtime_exception(const std::string& message) {
    py::object exception_type = py::reinterpret_borrow<py::object>(PyExc_RuntimeError);
    return exception_type(message);
}

py::object make_callback_exception(const std::string& message) {
    return make_runtime_exception(
        "Got the following error when running the callback: " + message);
}

std::string python_type_name(py::handle object) {
    return "<class '" + std::string(Py_TYPE(object.ptr())->tp_name) + "'>";
}

[[noreturn]] void raise_python_exception(const py::object& exception) {
    PyErr_SetObject(PyExceptionInstance_Class(exception.ptr()), exception.ptr());
    throw py::error_already_set();
}

void log_callback_error(const std::string& message) {
    std::cerr << "Future callback raised an exception: " << message << '\n';
}

class NativeFuture : public std::enable_shared_from_this<NativeFuture> {
    struct State {
        mutable std::mutex mutex;
        std::condition_variable condition;
        bool completed = false;
        bool has_exception = false;
        bool resolving = false;
        py::object result;
        py::object exception;
        std::vector<Device> devices;
        py::function completer;
        py::function unwrap_func;
        std::function<void()> native_completer;
        std::vector<Callback> callbacks;

        explicit State(std::vector<Device> devices_) : devices(std::move(devices_)) {}

        ~State() {
            py::gil_scoped_acquire gil;
            callbacks.clear();
            native_completer = nullptr;
            completer = py::function();
            unwrap_func = py::function();
            result = py::object();
            exception = py::object();
            devices.clear();
        }
    };

    std::shared_ptr<State> state_;

public:
    static std::vector<Device> normalize_devices(const py::object& devices) {
        if (devices.is_none()) {
            return {};
        }
        std::vector<Device> normalized;
        bool has_type = false;
        DeviceType device_type = DeviceType::CPU;
        for (py::handle item : py::iterable(devices)) {
            Device device;
            if (PyLong_Check(item.ptr()) && !PyBool_Check(item.ptr())) {
                device = Device(DeviceType::CUDA, py::cast<int64_t>(item));
            } else if (PyUnicode_Check(item.ptr())) {
                device = Device(py::cast<std::string>(item));
            } else {
                try {
                    device = py::cast<Device>(item);
                } catch (const py::cast_error&) {
                    throw py::type_error(
                        "devices must contain device values or accelerator indices");
                }
            }
            if (has_type && device.type() != device_type) {
                throw py::value_error(
                    "Expected all devices to be of the same type");
            }
            device_type = device.type();
            has_type = true;
            if (device.index() < 0) {
                throw py::value_error(
                    "Expected devices to have indices, got " + device.toString());
            }
#ifndef USE_CUDA
            if (device.is_cuda()) {
                throw std::runtime_error("CUDA support is not available");
            }
#endif
            normalized.push_back(device);
        }
        return normalized;
    }

private:
    static void invoke_callback(Callback& callback) {
        try {
            callback();
        } catch (py::error_already_set& error) {
            log_callback_error(error.what());
        } catch (const std::exception& error) {
            log_callback_error(error.what());
        } catch (...) {
            log_callback_error("unknown callback failure");
        }
    }

public:
    void complete(
        py::object result,
        py::object exception,
        bool has_exception,
        bool ignore_if_done) {
        py::gil_scoped_acquire gil;
        std::vector<Callback> callbacks;
        {
            std::unique_lock<std::mutex> lock(state_->mutex);
            if (state_->completed) {
                if (ignore_if_done) {
                    return;
                }
                throw std::runtime_error(
                    "Attempting to mark a completed Future as complete again. "
                    "Note that a Future can only be marked completed once.");
            }

            state_->result = std::move(result);
            state_->exception = std::move(exception);
            state_->has_exception = has_exception;
            state_->completed = true;
            state_->completer = py::function();
            state_->native_completer = nullptr;
            callbacks.swap(state_->callbacks);
        }
        state_->condition.notify_all();
        for (auto& callback : callbacks) {
            invoke_callback(callback);
        }
    }

private:
    void start_completion() {
        py::gil_scoped_acquire gil;
        std::function<void()> native_completer;
        py::function completer;
        {
            std::unique_lock<std::mutex> lock(state_->mutex);
            if (state_->completed || state_->resolving) {
                return;
            }
            if (state_->native_completer) {
                native_completer = std::move(state_->native_completer);
            } else if (state_->completer) {
                completer = std::move(state_->completer);
            } else {
                return;
            }
            state_->resolving = true;
        }

        try {
            if (native_completer) {
                native_completer();
            } else {
                completer();
            }
        } catch (py::error_already_set& error) {
            complete(py::object(), error.value(), true, true);
        } catch (const std::exception& error) {
            complete(
                py::object(),
                make_runtime_exception(error.what()),
                true,
                true);
        } catch (...) {
            complete(
                py::object(),
                make_runtime_exception("unknown completion failure"),
                true,
                true);
        }

        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->resolving = false;
        }
        state_->condition.notify_all();
    }

    py::object outcome() const {
        py::gil_scoped_acquire gil;
        py::object result;
        py::object exception;
        py::function unwrap_func;
        bool has_exception;
        {
            std::unique_lock<std::mutex> lock(state_->mutex);
            if (!state_->completed) {
                PyErr_SetString(PyExc_RuntimeError, "Future has not completed yet.");
                throw py::error_already_set();
            }
            has_exception = state_->has_exception;
            if (has_exception) {
                exception = state_->exception;
            } else {
                result = state_->result;
                unwrap_func = state_->unwrap_func;
            }
        }
        if (has_exception) {
            raise_python_exception(exception);
        }
        if (unwrap_func) {
            unwrap_func(result);
        }
        return result;
    }

public:
    explicit NativeFuture(std::vector<Device> devices)
        : state_(std::make_shared<State>(std::move(devices))) {}

    bool done() const {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return state_->completed;
    }

    bool is_done() const {
        return done();
    }

    py::object wait() {
        start_completion();
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(state_->mutex);
            state_->condition.wait(lock, [this]() { return state_->completed; });
        }
        return outcome();
    }

    py::object value() const {
        return outcome();
    }

    void set_result(py::object result) {
        complete(std::move(result), py::object(), false, false);
    }

    void set_exception(py::object exception) {
        py::gil_scoped_acquire gil;
        int is_exception = PyObject_IsInstance(exception.ptr(), PyExc_Exception);
        if (is_exception < 0) {
            throw py::error_already_set();
        }
        if (is_exception == 0) {
            std::string message = py::str(exception).cast<std::string>();
            message += " is of type ";
            message += python_type_name(exception);
            message += ", not an Exception.";
            PyErr_SetString(PyExc_AssertionError, message.c_str());
            throw py::error_already_set();
        }
        complete(py::object(), std::move(exception), true, false);
    }

public:
    py::object stored_exception() const {
        py::gil_scoped_acquire gil;
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (!state_->completed || !state_->has_exception) {
            return py::none();
        }
        return state_->exception;
    }

    void set_python_completer(py::object completer) {
        py::gil_scoped_acquire gil;
        py::function function;
        if (!completer.is_none()) {
            function = completer.cast<py::function>();
        }
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->completer = std::move(function);
        state_->native_completer = nullptr;
    }

    py::object python_completer() const {
        py::gil_scoped_acquire gil;
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (!state_->completer) {
            return py::none();
        }
        return state_->completer;
    }

    void set_unwrap_func(py::function unwrap_func) {
        py::gil_scoped_acquire gil;
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->unwrap_func = std::move(unwrap_func);
    }

    void set_native_completer(std::function<void()> completer) {
        py::gil_scoped_acquire gil;
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->native_completer = std::move(completer);
        state_->completer = py::function();
    }

    void add_native_callback(Callback callback) {
        py::gil_scoped_acquire gil;
        bool run_now;
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            if (state_->completed) {
                run_now = true;
            } else {
                state_->callbacks.push_back(std::move(callback));
                run_now = false;
            }
        }
        if (run_now) {
            invoke_callback(callback);
        }
    }

    void add_python_callback(py::object self, py::function callback) {
        add_native_callback(
            [self = std::move(self), callback = std::move(callback)]() mutable {
                try {
                    callback(self);
                } catch (py::error_already_set& error) {
                    log_callback_error(error.what());
                } catch (const std::exception& error) {
                    log_callback_error(error.what());
                } catch (...) {
                    log_callback_error("unknown callback failure");
                }
            });
    }

    FuturePtr then(py::object self, py::function callback) {
        py::gil_scoped_acquire gil;
        std::vector<Device> devices;
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            devices = state_->devices;
        }
        auto child = std::make_shared<NativeFuture>(devices);
        auto parent = shared_from_this();
        child->set_native_completer([parent = std::move(parent)]() {
            parent->wait();
        });

        add_native_callback(
            [self = std::move(self), callback = std::move(callback), child]() mutable {
                try {
                    child->complete(callback(self), py::object(), false, true);
                } catch (py::error_already_set& error) {
                    child->complete(
                        py::object(),
                        make_callback_exception(error.what()),
                        true,
                        true);
                } catch (const std::exception& error) {
                    child->complete(
                        py::object(),
                        make_callback_exception(error.what()),
                        true,
                        true);
                } catch (...) {
                    child->complete(
                        py::object(),
                        make_callback_exception("unknown callback failure"),
                        true,
                        true);
                }
            });
        return child;
    }
};

struct ParsedFutures {
    std::vector<FuturePtr> sources;
    py::list originals;
};

ParsedFutures parse_futures(const py::iterable& input) {
    ParsedFutures parsed;
    for (py::handle item : input) {
        if (item.is_none()) {
            throw std::runtime_error("Future can't be None");
        }
        try {
            parsed.sources.push_back(item.cast<FuturePtr>());
        } catch (const py::cast_error&) {
            throw py::type_error("futures must contain Future objects");
        }
        parsed.originals.append(item);
    }
    return parsed;
}

struct CollectContext {
    std::vector<FuturePtr> sources;
    py::object originals;
    FuturePtr combined;
    size_t remaining;
    std::mutex mutex;
    bool finalized = false;

    CollectContext(
        std::vector<FuturePtr> sources_,
        py::object originals_,
        FuturePtr combined_)
        : sources(std::move(sources_)),
          originals(std::move(originals_)),
          combined(std::move(combined_)),
          remaining(sources.size()) {}

    ~CollectContext() {
        py::gil_scoped_acquire gil;
        originals = py::object();
        sources.clear();
        combined.reset();
    }

    void finalize() {
        py::gil_scoped_acquire gil;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (finalized) {
                return;
            }
            finalized = true;
        }

        for (const auto& source : sources) {
            py::object exception = source->stored_exception();
            if (!exception.is_none()) {
                combined->complete(py::object(), std::move(exception), true, true);
                return;
            }
        }
        combined->complete(originals, py::object(), false, true);
    }

    void source_done(const FuturePtr& source) {
        if (!source->stored_exception().is_none()) {
            finalize();
            return;
        }
        bool all_done = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (remaining != 0) {
                --remaining;
                all_done = remaining == 0;
            }
        }
        if (all_done) {
            finalize();
        }
    }

    void drive() {
        py::gil_scoped_acquire gil;
        for (const auto& source : sources) {
            try {
                source->wait();
            } catch (py::error_already_set&) {
            } catch (const std::exception&) {
            } catch (...) {
            }
        }
        if (!combined->done()) {
            finalize();
        }
    }
};

FuturePtr make_collect_all(ParsedFutures parsed) {
    auto combined = std::make_shared<NativeFuture>(std::vector<Device>{});
    if (parsed.sources.empty()) {
        combined->complete(parsed.originals, py::object(), false, false);
        return combined;
    }

    auto context = std::make_shared<CollectContext>(
        std::move(parsed.sources), parsed.originals, combined);
    combined->set_native_completer([context]() { context->drive(); });
    const auto sources = context->sources;
    combined->set_unwrap_func(py::cpp_function(
        [sources](py::object) {
            for (const auto& source : sources) {
                source->wait();
            }
        }));
    for (const auto& source : context->sources) {
        source->add_native_callback(
            [context, source]() { context->source_done(source); });
    }
    return combined;
}

FuturePtr collect_all(const py::iterable& input) {
    return make_collect_all(parse_futures(input));
}

py::list wait_all(const py::iterable& input) {
    ParsedFutures parsed = parse_futures(input);
    std::vector<FuturePtr> sources = parsed.sources;
    auto combined = make_collect_all(std::move(parsed));
    combined->wait();

    py::list results;
    for (const auto& source : sources) {
        results.append(source->wait());
    }
    return results;
}

}  // namespace

void init_futures(py::module_& m) {
    py::class_<NativeFuture, std::shared_ptr<NativeFuture>>(m, "Future")
        .def(
            py::init([](py::object devices, py::object completer) {
                auto future = std::make_shared<NativeFuture>(
                    NativeFuture::normalize_devices(devices));
                if (!completer.is_none()) {
                    future->set_python_completer(std::move(completer));
                }
                return future;
            }),
            "devices"_a = py::none(),
            "_completer"_a = py::none())
        .def("done", &NativeFuture::done)
        .def("is_done", &NativeFuture::is_done)
        .def("wait", &NativeFuture::wait)
        .def("value", &NativeFuture::value)
        .def(
            "then",
            [](py::object self, py::function callback) {
                auto future = self.cast<FuturePtr>();
                return future->then(std::move(self), std::move(callback));
            },
            "callback"_a)
        .def(
            "add_done_callback",
            [](py::object self, py::function callback) {
                self.cast<FuturePtr>()->add_python_callback(
                    std::move(self), std::move(callback));
            },
            "callback"_a)
        .def("set_result", &NativeFuture::set_result, "result"_a)
        .def("set_exception", &NativeFuture::set_exception, "result"_a)
        .def(
            "_set_completer",
            &NativeFuture::set_python_completer,
            "completer"_a)
        .def("_set_unwrap_func", &NativeFuture::set_unwrap_func, "func"_a)
        .def_property(
            "_completer",
            &NativeFuture::python_completer,
            &NativeFuture::set_python_completer)
        .def(
            "__reduce__",
            [](const NativeFuture&) -> py::object {
                throw std::runtime_error("Future objects cannot be serialized.");
            })
        .def(
            "__reduce_ex__",
            [](const NativeFuture&, int) -> py::object {
                throw std::runtime_error("Future objects cannot be serialized.");
            });

    m.def("_collect_all", &collect_all, "futures"_a);
    m.def("_wait_all", &wait_all, "futures"_a);
}
