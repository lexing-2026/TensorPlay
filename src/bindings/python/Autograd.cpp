#include "python_bindings.h"
#include "PythonRuntime.h"
#include "Node.h"
#include "AccumulateGrad.h"
#include "Autograd.h"
#include "Engine.h"
#include "AnomalyMode.h"
#include "SavedVariable.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#ifdef USE_CUDA
#include "CUDAGenerator.h"
#include "CUDARuntime.h"
#endif
#include <algorithm>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <set>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <typeinfo>
#include <string>
#include <pybind11/functional.h>

namespace {
// Cached tensor PyTypeObject for ns-scale type checks on the custom-function
// hot path (using a cached CPython tensor type).
PyTypeObject* g_fast_tensor_type = nullptr;
inline bool fast_is_tensor(PyObject* obj) {
    // Cache the registered base class, never the type of the first object
    // seen: a subclass cached here would reject every plain tensor.
    if (!g_fast_tensor_type)
        g_fast_tensor_type = reinterpret_cast<PyTypeObject*>(py::type::of<Tensor>().ptr());
    return PyObject_TypeCheck(obj, g_fast_tensor_type) != 0;
}

// Requires-grad scan over one Python argument.  A direct tensor is
// inspected; a list is descended only when every element is a tensor; any
// other object (tuple, dict, mixed list) contributes nothing.  Stops at the
// first tensor whose requires_grad bit is set.
bool visit_requires_grad(PyObject* object) {
    if (fast_is_tensor(object)) {
        return py::cast<const Tensor&>(py::handle(object)).requires_grad();
    }
    if (PyList_Check(object)) {
        const Py_ssize_t size = PyList_GET_SIZE(object);
        for (Py_ssize_t i = 0; i < size; ++i) {
            if (!fast_is_tensor(PyList_GET_ITEM(object, i))) {
                return false;
            }
        }
        for (Py_ssize_t i = 0; i < size; ++i) {
            if (visit_requires_grad(PyList_GET_ITEM(object, i))) {
                return true;
            }
        }
    }
    return false;
}

using PyObjectRef = std::shared_ptr<PyObject>;

PyObjectRef retain_pyobject(py::handle object) {
    PyObject* ptr = object.ptr();
    Py_XINCREF(ptr);
    return PyObjectRef(ptr, [](PyObject* value) noexcept {
        if (!value || !tensorplay::python_c::interpreter_active()) return;
        if (PyGILState_Check()) {
            Py_DECREF(value);
            return;
        }
        try {
            py::gil_scoped_acquire gil;
            Py_DECREF(value);
        } catch (...) {
        }
    });
}

py::object borrow_pyobject(const PyObjectRef& object) {
    if (!object) return py::none();
    return py::reinterpret_borrow<py::object>(object.get());
}

inline bool checkpoint_differentiable(const Tensor& tensor) {
    return tensor.defined() &&
           tensorplay::isFloatingOrComplexType(tensor.dtype());
}

void collect_checkpoint_inputs(
    py::handle object,
    std::vector<Tensor>& saved,
    std::vector<Tensor>& original_inputs,
    std::vector<tensorplay::tpx::Edge>& edges,
    bool& any_requires_grad) {
    if (fast_is_tensor(object.ptr())) {
        const Tensor& input = py::cast<const Tensor&>(object);
        Tensor detached = input.detach();
        if (input.requires_grad()) {
            any_requires_grad = true;
            tensorplay::tpx::impl::set_requires_grad(detached, true);
            auto input_edges = tensorplay::tpx::collect_next_edges(input);
            edges.insert(edges.end(),
                         std::make_move_iterator(input_edges.begin()),
                         std::make_move_iterator(input_edges.end()));
        } else {
            edges.emplace_back();
        }
        saved.push_back(std::move(detached));
        original_inputs.push_back(input);
        return;
    }

    if (PyTuple_Check(object.ptr())) {
        const Py_ssize_t size = PyTuple_GET_SIZE(object.ptr());
        for (Py_ssize_t i = 0; i < size; ++i) {
            collect_checkpoint_inputs(
                py::handle(PyTuple_GET_ITEM(object.ptr(), i)),
                saved, original_inputs, edges, any_requires_grad);
        }
        return;
    }

    if (PyList_Check(object.ptr())) {
        const Py_ssize_t size = PyList_GET_SIZE(object.ptr());
        for (Py_ssize_t i = 0; i < size; ++i) {
            collect_checkpoint_inputs(
                py::handle(PyList_GET_ITEM(object.ptr(), i)),
                saved, original_inputs, edges, any_requires_grad);
        }
        return;
    }

    if (PyDict_Check(object.ptr())) {
        PyObject* key = nullptr;
        PyObject* value = nullptr;
        Py_ssize_t position = 0;
        while (PyDict_Next(object.ptr(), &position, &key, &value)) {
            collect_checkpoint_inputs(
                py::handle(value), saved, original_inputs, edges,
                any_requires_grad);
        }
    }
}

py::object rebuild_checkpoint_tree(
    py::handle object,
    const std::vector<Tensor>& saved,
    size_t& position) {
    if (fast_is_tensor(object.ptr())) {
        if (position >= saved.size()) {
            throw std::runtime_error("checkpoint input tree changed during replay");
        }
        return py::cast(saved[position++]);
    }

    if (PyTuple_Check(object.ptr())) {
        const Py_ssize_t size = PyTuple_GET_SIZE(object.ptr());
        py::tuple result(size);
        for (Py_ssize_t i = 0; i < size; ++i) {
            result[i] = rebuild_checkpoint_tree(
                py::handle(PyTuple_GET_ITEM(object.ptr(), i)), saved, position);
        }
        return result;
    }

    if (PyList_Check(object.ptr())) {
        const Py_ssize_t size = PyList_GET_SIZE(object.ptr());
        py::list result(size);
        for (Py_ssize_t i = 0; i < size; ++i) {
            result[i] = rebuild_checkpoint_tree(
                py::handle(PyList_GET_ITEM(object.ptr(), i)), saved, position);
        }
        return result;
    }

    if (PyDict_Check(object.ptr())) {
        py::dict result;
        PyObject* key = nullptr;
        PyObject* value = nullptr;
        Py_ssize_t dict_position = 0;
        while (PyDict_Next(object.ptr(), &dict_position, &key, &value)) {
            result[py::handle(key)] = rebuild_checkpoint_tree(
                py::handle(value), saved, position);
        }
        return result;
    }

    return py::reinterpret_borrow<py::object>(object);
}

void collect_checkpoint_outputs(
    py::handle object,
    std::vector<Tensor>& outputs,
    bool include_non_differentiable) {
    if (fast_is_tensor(object.ptr())) {
        const Tensor& output = py::cast<const Tensor&>(object);
        if (checkpoint_differentiable(output) &&
            (include_non_differentiable || output.requires_grad())) {
            outputs.push_back(output);
        }
        return;
    }

    if (PyTuple_Check(object.ptr())) {
        const Py_ssize_t size = PyTuple_GET_SIZE(object.ptr());
        for (Py_ssize_t i = 0; i < size; ++i) {
            collect_checkpoint_outputs(
                py::handle(PyTuple_GET_ITEM(object.ptr(), i)), outputs,
                include_non_differentiable);
        }
        return;
    }

    if (PyList_Check(object.ptr())) {
        const Py_ssize_t size = PyList_GET_SIZE(object.ptr());
        for (Py_ssize_t i = 0; i < size; ++i) {
            collect_checkpoint_outputs(
                py::handle(PyList_GET_ITEM(object.ptr(), i)), outputs,
                include_non_differentiable);
        }
        return;
    }

    if (PyDict_Check(object.ptr())) {
        PyObject* key = nullptr;
        PyObject* value = nullptr;
        Py_ssize_t position = 0;
        while (PyDict_Next(object.ptr(), &position, &key, &value)) {
            collect_checkpoint_outputs(
                py::handle(value), outputs, include_non_differentiable);
        }
    }
}

py::object attach_checkpoint_outputs(
    py::handle object,
    const std::shared_ptr<tensorplay::tpx::Node>& node,
    std::vector<tensorplay::tpx::OutputSlotMeta>& metas,
    size_t& output_index,
    bool force_requires_grad) {
    if (fast_is_tensor(object.ptr())) {
        const Tensor& output = py::cast<const Tensor&>(object);
        if (!checkpoint_differentiable(output) ||
            (!force_requires_grad && !output.requires_grad())) {
            return py::reinterpret_borrow<py::object>(object);
        }

        tensorplay::tpx::OutputSlotMeta meta;
        meta.shape = static_cast<std::vector<int64_t>>(output.shape());
        meta.dtype = output.dtype();
        meta.device_type = output.device().type();
        meta.device_index = output.device().index();
        meta.valid = true;
        metas.push_back(std::move(meta));
        Tensor wrapped = output.detach();
        tensorplay::tpx::impl::set_requires_grad(wrapped, true);
        tensorplay::tpx::impl::set_grad_fn(
            wrapped, node, static_cast<uint32_t>(output_index));
        ++output_index;
        return py::cast(std::move(wrapped));
    }

    if (PyTuple_Check(object.ptr())) {
        const Py_ssize_t size = PyTuple_GET_SIZE(object.ptr());
        py::tuple result(size);
        for (Py_ssize_t i = 0; i < size; ++i) {
            result[i] = attach_checkpoint_outputs(
                py::handle(PyTuple_GET_ITEM(object.ptr(), i)),
                node, metas, output_index, force_requires_grad);
        }
        return result;
    }

    if (PyList_Check(object.ptr())) {
        const Py_ssize_t size = PyList_GET_SIZE(object.ptr());
        py::list result(size);
        for (Py_ssize_t i = 0; i < size; ++i) {
            result[i] = attach_checkpoint_outputs(
                py::handle(PyList_GET_ITEM(object.ptr(), i)),
                node, metas, output_index, force_requires_grad);
        }
        return result;
    }

    if (PyDict_Check(object.ptr())) {
        py::dict result;
        PyObject* key = nullptr;
        PyObject* value = nullptr;
        Py_ssize_t position = 0;
        while (PyDict_Next(object.ptr(), &position, &key, &value)) {
            result[py::handle(key)] = attach_checkpoint_outputs(
                py::handle(value), node, metas, output_index,
                force_requires_grad);
        }
        return result;
    }

    return py::reinterpret_borrow<py::object>(object);
}

void collect_grad_parameters(
    const py::object& function,
    std::vector<Tensor>& grad_parameters) {
    if (!py::hasattr(function, "parameters")) return;
    py::object parameter_iterable = function.attr("parameters")();
    for (py::handle item :
         py::reinterpret_borrow<py::iterable>(parameter_iterable)) {
        if (fast_is_tensor(item.ptr()) &&
            py::cast<const Tensor&>(item).requires_grad()) {
            grad_parameters.push_back(py::cast<const Tensor&>(item));
        }
    }
}

std::pair<py::object, py::object> make_checkpoint_contexts(
    const py::object& context_fn) {
    if (context_fn.is_none()) return {py::none(), py::none()};

    py::object result = context_fn();
    if (!PyTuple_Check(result.ptr()) && !PyList_Check(result.ptr())) {
        throw std::invalid_argument(
            "context_fn must return a pair of context managers");
    }
    if (PySequence_Size(result.ptr()) != 2) {
        throw std::invalid_argument(
            "context_fn must return exactly two context managers");
    }
    py::object forward_context = py::reinterpret_steal<py::object>(
        PySequence_GetItem(result.ptr(), 0));
    py::object replay_context = py::reinterpret_steal<py::object>(
        PySequence_GetItem(result.ptr(), 1));
    return {std::move(forward_context), std::move(replay_context)};
}

class PyContextScope {
public:
    explicit PyContextScope(py::object context)
        : context_(std::move(context)) {
        if (!context_.is_none()) {
            context_.attr("__enter__")();
            active_ = true;
        }
    }

    void close() {
        if (!active_) return;
        active_ = false;
        context_.attr("__exit__")(py::none(), py::none(), py::none());
    }

    ~PyContextScope() {
        if (!tensorplay::python_c::interpreter_active()) {
            context_.release();
            return;
        }
        if (!active_) return;
        try {
            close();
        } catch (...) {
            PyErr_WriteUnraisable(context_.ptr());
        }
    }

private:
    py::object context_;
    bool active_ = false;
};

class CpuRngScope {
public:
    CpuRngScope(const Tensor& target, bool enabled) : active_(enabled) {
        if (!active_) return;
        previous_ = default_generator().get_state();
        default_generator().set_state(target);
    }

    void restore() {
        if (!active_) return;
        default_generator().set_state(previous_);
        active_ = false;
    }

    ~CpuRngScope() {
        if (!active_) return;
        try {
            default_generator().set_state(previous_);
        } catch (...) {
        }
    }

private:
    Tensor previous_;
    bool active_ = false;
};

struct CheckpointCudaRngState {
    int device = -1;
    Tensor state;
};

std::vector<CheckpointCudaRngState> capture_checkpoint_cuda_rng_states(
    const std::vector<Tensor>& inputs,
    const std::vector<Tensor>& parameters,
    bool enabled) {
    std::vector<CheckpointCudaRngState> states;
#ifdef USE_CUDA
    if (!enabled) return states;
    if (tensorplay::cuda::deviceCount() == 0) return states;
    std::set<int> devices;
    devices.insert(tensorplay::cuda::currentDevice());
    auto collect = [&devices](const Tensor& tensor) {
        if (tensor.defined() && tensor.device().is_cuda() &&
            tensor.device().index() >= 0) {
            devices.insert(static_cast<int>(tensor.device().index()));
        }
    };
    for (const Tensor& tensor : inputs) collect(tensor);
    for (const Tensor& tensor : parameters) collect(tensor);
    states.reserve(devices.size());
    for (int device : devices) {
        tensorplay::cuda::CUDAGuard guard(device);
        states.push_back({device, tensorplay::cuda::get_rng_state()});
    }
#else
    (void)inputs;
    (void)parameters;
    (void)enabled;
#endif
    return states;
}

class CheckpointCudaRngScope {
public:
    CheckpointCudaRngScope(
        const std::vector<CheckpointCudaRngState>& target,
        bool enabled)
        : active_(enabled && !target.empty()), target_(target) {
#ifdef USE_CUDA
        if (!active_) return;
        previous_.reserve(target_.size());
        for (const auto& state : target_) {
            tensorplay::cuda::CUDAGuard guard(state.device);
            previous_.push_back({state.device,
                                 tensorplay::cuda::get_rng_state()});
            tensorplay::cuda::set_rng_state(state.state);
        }
#else
        (void)target;
#endif
    }

    CheckpointCudaRngScope(const CheckpointCudaRngScope&) = delete;
    CheckpointCudaRngScope& operator=(const CheckpointCudaRngScope&) = delete;

    void restore() {
        if (!active_) return;
#ifdef USE_CUDA
        for (const auto& state : previous_) {
            tensorplay::cuda::CUDAGuard guard(state.device);
            tensorplay::cuda::set_rng_state(state.state);
        }
#endif
        active_ = false;
    }

    ~CheckpointCudaRngScope() {
        if (!active_) return;
        try {
            restore();
        } catch (...) {
        }
    }

private:
    bool active_ = false;
    const std::vector<CheckpointCudaRngState>& target_;
    std::vector<CheckpointCudaRngState> previous_;
};

class CheckpointReplayStop final : public std::exception {
public:
    const char* what() const noexcept override {
        return "checkpoint replay reached its required saved values";
    }
};

struct CheckpointSavedToken {
    size_t index = 0;
};

class SavedVariableHooksScope {
public:
    explicit SavedVariableHooksScope(
        std::shared_ptr<tensorplay::tpx::SavedVariableHooks> hooks)
        : active_(true) {
        tensorplay::tpx::push_saved_variable_hooks(std::move(hooks));
    }

    SavedVariableHooksScope(const SavedVariableHooksScope&) = delete;
    SavedVariableHooksScope& operator=(const SavedVariableHooksScope&) = delete;

    ~SavedVariableHooksScope() {
        if (!active_) return;
        try {
            tensorplay::tpx::pop_saved_variable_hooks();
        } catch (...) {
        }
    }

    void close() {
        if (!active_) return;
        tensorplay::tpx::pop_saved_variable_hooks();
        active_ = false;
    }

private:
    bool active_ = false;
};

class PythonSavedVariableHooks final
    : public tensorplay::tpx::SavedVariableHooks {
public:
    PythonSavedVariableHooks(py::object pack, py::object unpack)
        : pack_(retain_pyobject(pack)), unpack_(retain_pyobject(unpack)) {}

    std::shared_ptr<void> pack(const Tensor& tensor) override {
        py::gil_scoped_acquire gil;
        py::object result = borrow_pyobject(pack_)(py::cast(tensor));
        return std::static_pointer_cast<void>(retain_pyobject(result));
    }

    Tensor unpack(const std::shared_ptr<void>& packed) override {
        py::gil_scoped_acquire gil;
        auto object = std::static_pointer_cast<PyObject>(packed);
        py::object result = borrow_pyobject(unpack_)(
            py::reinterpret_borrow<py::object>(object.get()));
        if (!fast_is_tensor(result.ptr())) {
            throw py::type_error("saved tensor unpack hook must return a Tensor");
        }
        return py::cast<Tensor>(result);
    }

private:
    PyObjectRef pack_;
    PyObjectRef unpack_;
};

struct SavedTensorToken {
    std::shared_ptr<tensorplay::tpx::SavedVariableHooks> hooks;
    std::shared_ptr<void> packed;
};

class NativeCheckpointFrame final
    : public tensorplay::tpx::SavedVariableHooks,
      public std::enable_shared_from_this<NativeCheckpointFrame> {
private:
    enum class Phase { Forward, Idle, Replay };

    struct Slot {
        std::vector<int64_t> shape;
        DType dtype{DType::Undefined};
        DeviceType device_type{DeviceType::CPU};
        int64_t device_index = -1;
        uint32_t version = 0;
        Tensor forward_value;
        Tensor value;
        bool recomputed = false;
    };

    struct Operation {
        size_t slot_start = 0;
        size_t slot_end = 0;
        bool cache_forward_values = false;
    };

public:
    NativeCheckpointFrame(
        py::object function,
        py::tuple args,
        py::dict kwargs,
        std::vector<Tensor> saved_inputs,
        std::vector<Tensor> original_inputs,
        Tensor cpu_rng_state,
        std::vector<CheckpointCudaRngState> cuda_rng_states,
        bool preserve_rng_state,
        py::object replay_context,
        std::string determinism_check,
        bool debug,
        bool early_stop)
        : function_(retain_pyobject(function)),
          args_(retain_pyobject(args)),
          kwargs_(retain_pyobject(kwargs)),
          saved_inputs_(std::move(saved_inputs)),
          original_inputs_(std::move(original_inputs)),
          cpu_rng_state_(std::move(cpu_rng_state)),
          cuda_rng_states_(std::move(cuda_rng_states)),
          replay_context_(retain_pyobject(replay_context)),
          determinism_check_(std::move(determinism_check)),
          debug_(debug),
          early_stop_(early_stop),
          preserve_rng_state_(preserve_rng_state) {}

    std::shared_ptr<void> pack(const Tensor& tensor) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (phase_ == Phase::Forward) {
            Slot slot;
            slot.shape = static_cast<std::vector<int64_t>>(tensor.shape());
            slot.dtype = tensor.dtype();
            slot.device_type = tensor.device().type();
            slot.device_index = tensor.device().index();
            slot.version = tensor.unsafeGetTensorImpl()->version();
            if (!forward_operation_stack_.empty()) {
                slot.forward_value = tensor;
            }
            slots_.push_back(std::move(slot));
            return make_token(slots_.size() - 1);
        }

        if (phase_ != Phase::Replay) {
            throw std::runtime_error(
                "checkpoint saved-value hook used outside an active replay");
        }

        const size_t index = replay_count_++;
        if (index >= slots_.size()) {
            slots_.resize(index + 1);
            replay_extra_ = true;
        }
        Slot& slot = slots_[index];
        slot.value = replay_create_graph_ ? tensor : tensor.detach();
        slot.recomputed = true;
        if (early_stop_ && replay_count_ == forward_slot_count_) {
            stop_requested_ = true;
            throw CheckpointReplayStop();
        }
        return make_token(index);
    }

    Tensor unpack(const std::shared_ptr<void>& packed) override {
        if (!packed) {
            throw std::runtime_error("checkpoint saved-value token is empty");
        }
        const auto token = std::static_pointer_cast<CheckpointSavedToken>(packed);
        const size_t index = token->index;

        bool owner = false;
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            if (index >= slots_.size()) {
                throw std::runtime_error("checkpoint saved-value index is invalid");
            }
            if (slots_[index].recomputed && slots_[index].value.defined()) {
                return slots_[index].value;
            }
            if (replay_error_) {
                auto error = replay_error_;
                lock.unlock();
                std::rethrow_exception(error);
            }
            if (replaying_) {
                if (active_replay_frame_ == this) {
                    throw std::runtime_error(
                        "checkpoint replay requested a value before it was recomputed");
                }
                replay_cv_.wait(lock, [this] { return !replaying_; });
                continue;
            }
            if (replay_started_) {
                replay_cv_.wait(lock, [this] { return !replaying_; });
                continue;
            }
            replay_started_ = true;
            replaying_ = true;
            phase_ = Phase::Replay;
            replay_operation_count_ = 0;
            owner = true;
            break;
        }

        if (owner) {
            std::exception_ptr error;
            try {
                recompute();
            } catch (...) {
                error = std::current_exception();
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                replay_error_ = error;
                replaying_ = false;
                phase_ = Phase::Idle;
                replay_cv_.notify_all();
            }
            if (error) std::rethrow_exception(error);
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= slots_.size() || !slots_[index].recomputed ||
            !slots_[index].value.defined()) {
            throw std::runtime_error(
                "checkpoint replay did not produce the requested saved value");
        }
        return slots_[index].value;
    }

    void finish_forward() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (phase_ != Phase::Forward) {
            throw std::runtime_error("checkpoint forward phase is not active");
        }
        if (!forward_operation_stack_.empty()) {
            throw std::runtime_error(
                "checkpoint operation scope was not closed before forward completion");
        }
        for (const auto& operation : operations_) {
            if (!operation.cache_forward_values) {
                for (size_t i = operation.slot_start;
                     i < operation.slot_end && i < slots_.size(); ++i) {
                    slots_[i].forward_value = Tensor();
                }
            }
        }
        forward_slot_count_ = slots_.size();
        phase_ = Phase::Idle;
    }

    int64_t begin_operation() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (phase_ == Phase::Forward) {
            const size_t index = operations_.size();
            operations_.push_back(Operation{slots_.size(), slots_.size(), false});
            forward_operation_stack_.push_back(index);
            return static_cast<int64_t>(index);
        }
        if (phase_ != Phase::Replay) {
            return -1;
        }
        return static_cast<int64_t>(replay_operation_count_++);
    }

    void end_operation(int64_t operation_index) {
        if (operation_index < 0) return;
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t index = static_cast<size_t>(operation_index);
        if (phase_ == Phase::Forward) {
            if (forward_operation_stack_.empty() ||
                forward_operation_stack_.back() != index ||
                index >= operations_.size()) {
                throw std::runtime_error(
                    "checkpoint operation scope is not properly nested");
            }
            operations_[index].slot_end = slots_.size();
            forward_operation_stack_.pop_back();
        }
    }

    void cache_operation(int64_t operation_index, bool cache) {
        if (operation_index < 0) return;
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t index = static_cast<size_t>(operation_index);
        if (phase_ != Phase::Forward || index >= operations_.size()) {
            throw std::runtime_error(
                "checkpoint operation index is invalid");
        }
        auto& operation = operations_[index];
        operation.cache_forward_values = cache;
        if (!cache) {
            for (size_t i = operation.slot_start;
                 i < operation.slot_end && i < slots_.size(); ++i) {
                slots_[i].forward_value = Tensor();
            }
        }
    }

    void reuse_operation(int64_t operation_index) {
        if (operation_index < 0) return;
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t index = static_cast<size_t>(operation_index);
        if (phase_ != Phase::Replay || index >= operations_.size()) {
            throw std::runtime_error(
                "checkpoint replay operation index is invalid");
        }
        const auto& operation = operations_[index];
        if (!operation.cache_forward_values) {
            throw std::runtime_error(
                "checkpoint replay requested an uncached operation");
        }
        for (size_t i = operation.slot_start; i < operation.slot_end; ++i) {
            if (i >= slots_.size() || !slots_[i].forward_value.defined()) {
                throw std::runtime_error(
                    "checkpoint cached operation has no saved value");
            }
            const Tensor& value = slots_[i].forward_value;
            if (value.unsafeGetTensorImpl()->version() != slots_[i].version) {
                throw std::runtime_error(
                    "checkpoint cached saved value was modified in-place");
            }
            slots_[i].value = value;
            slots_[i].recomputed = true;
            ++replay_count_;
        }
        if (early_stop_ && replay_count_ == forward_slot_count_) {
            stop_requested_ = true;
            throw CheckpointReplayStop();
        }
    }

    py::object forward_call() const {
        py::object function = borrow_pyobject(function_);
        py::tuple args = borrow_pyobject(args_).cast<py::tuple>();
        py::dict kwargs = borrow_pyobject(kwargs_).cast<py::dict>();
        return function(*args, **kwargs);
    }

private:
    static std::shared_ptr<void> make_token(size_t index) {
        return std::make_shared<CheckpointSavedToken>(
            CheckpointSavedToken{index});
    }

    static std::string describe_slot(const Slot& slot) {
        std::ostringstream out;
        out << "shape=[";
        for (size_t i = 0; i < slot.shape.size(); ++i) {
            if (i) out << ',';
            out << slot.shape[i];
        }
        out << "] dtype=" << static_cast<int>(slot.dtype)
            << " device=" << static_cast<int>(slot.device_type)
            << ':' << slot.device_index;
        return out.str();
    }

    std::string mismatch_message(const char* reason) const {
        std::ostringstream out;
        out << "activation checkpoint replay mismatch: " << reason
            << "; forward saved values=" << forward_slot_count_
            << ", replay saved values=" << replay_count_;
        if (debug_) {
            out << "\nforward slots:";
            for (size_t i = 0; i < forward_slot_count_ && i < slots_.size(); ++i) {
                out << "\n  " << i << ": " << describe_slot(slots_[i]);
            }
            out << "\nreplay slots:";
            for (size_t i = 0; i < replay_count_ && i < slots_.size(); ++i) {
                out << "\n  " << i << ": " << describe_slot(slots_[i]);
            }
        }
        return out.str();
    }

    void validate_replay(bool stopped) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (replay_count_ != forward_slot_count_) {
            throw std::runtime_error(mismatch_message(
                stopped ? "early-stop terminated before all values were produced"
                        : "the number of saved values changed"));
        }
        if (determinism_check_ != "default") return;
        for (size_t i = 0; i < forward_slot_count_; ++i) {
            const Slot& slot = slots_[i];
            const Tensor& value = slot.value;
            if (slot.shape != static_cast<std::vector<int64_t>>(value.shape()) ||
                slot.dtype != value.dtype() ||
                slot.device_type != value.device().type() ||
                slot.device_index != value.device().index()) {
                throw std::runtime_error(mismatch_message(
                    "saved value metadata changed"));
            }
        }
    }

    void recompute() {
        py::gil_scoped_acquire gil;
        const bool previous_grad = tensorplay::tpx::GradMode::is_enabled();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            replay_create_graph_ = previous_grad;
        }
        tensorplay::tpx::GradMode::set_enabled(true);
        NativeCheckpointFrame* previous_active = active_replay_frame_;
        active_replay_frame_ = this;
        bool stopped = false;
        try {
            std::vector<Tensor> replay_inputs;
            replay_inputs.reserve(saved_inputs_.size());
            for (size_t i = 0; i < saved_inputs_.size(); ++i) {
                Tensor replay = saved_inputs_[i].detach();
                if (saved_inputs_[i].requires_grad()) {
                    Tensor correction = tensorplay::tpx::ops::sub(
                        original_inputs_[i], original_inputs_[i].detach());
                    replay = tensorplay::tpx::ops::add(replay, correction);
                }
                replay_inputs.push_back(std::move(replay));
            }

            size_t position = 0;
            py::object replay_args = rebuild_checkpoint_tree(
                borrow_pyobject(args_), replay_inputs, position);
            py::object replay_kwargs = rebuild_checkpoint_tree(
                borrow_pyobject(kwargs_), replay_inputs, position);
            if (position != replay_inputs.size()) {
                throw std::runtime_error(
                    "checkpoint input tree changed during replay");
            }

            CpuRngScope rng_scope(cpu_rng_state_, preserve_rng_state_);
            CheckpointCudaRngScope cuda_rng_scope(
                cuda_rng_states_, preserve_rng_state_);
            try {
                PyContextScope context_scope(borrow_pyobject(replay_context_));
                SavedVariableHooksScope hooks_scope(shared_from_this());
                py::object function = borrow_pyobject(function_);
                function(*replay_args.cast<py::tuple>(),
                         **replay_kwargs.cast<py::dict>());
                hooks_scope.close();
                context_scope.close();
            } catch (const CheckpointReplayStop&) {
                stopped = true;
            } catch (const py::error_already_set&) {
                bool requested = false;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    requested = stop_requested_;
                }
                if (!requested) throw;
                PyErr_Clear();
                stopped = true;
            }
            cuda_rng_scope.restore();
            rng_scope.restore();
            validate_replay(stopped);
        } catch (...) {
            active_replay_frame_ = previous_active;
            tensorplay::tpx::GradMode::set_enabled(previous_grad);
            throw;
        }
        active_replay_frame_ = previous_active;
        tensorplay::tpx::GradMode::set_enabled(previous_grad);
    }

    PyObjectRef function_;
    PyObjectRef args_;
    PyObjectRef kwargs_;
    std::vector<Tensor> saved_inputs_;
    std::vector<Tensor> original_inputs_;
    Tensor cpu_rng_state_;
    std::vector<CheckpointCudaRngState> cuda_rng_states_;
    PyObjectRef replay_context_;
    std::string determinism_check_;
    bool debug_ = false;
    bool early_stop_ = true;
    bool preserve_rng_state_ = true;

    mutable std::mutex mutex_;
    std::condition_variable replay_cv_;
    Phase phase_ = Phase::Forward;
    std::vector<Slot> slots_;
    size_t forward_slot_count_ = 0;
    size_t replay_count_ = 0;
    bool replay_started_ = false;
    bool replaying_ = false;
    bool replay_extra_ = false;
    bool stop_requested_ = false;
    bool replay_create_graph_ = false;
    std::exception_ptr replay_error_;
    std::vector<Operation> operations_;
    std::vector<size_t> forward_operation_stack_;
    size_t replay_operation_count_ = 0;

    static thread_local NativeCheckpointFrame* active_replay_frame_;
};

thread_local NativeCheckpointFrame* NativeCheckpointFrame::active_replay_frame_ = nullptr;

std::shared_ptr<NativeCheckpointFrame> current_native_checkpoint_frame() {
    auto hooks = tensorplay::tpx::current_saved_variable_hooks();
    if (!hooks) return nullptr;
    return std::dynamic_pointer_cast<NativeCheckpointFrame>(hooks);
}

class ActivationCheckpointNode : public tensorplay::tpx::Node {
public:
    ActivationCheckpointNode(
        py::object function,
        py::tuple args,
        py::dict kwargs,
        std::vector<Tensor> saved_inputs,
        std::vector<Tensor> original_inputs,
        std::vector<Tensor> grad_parameters,
        Tensor cpu_rng_state,
        std::vector<CheckpointCudaRngState> cuda_rng_states,
        bool use_reentrant,
        bool preserve_rng_state,
        py::object replay_context,
        std::string determinism_check,
        bool debug,
        bool early_stop)
        : function_(retain_pyobject(function)),
          args_(retain_pyobject(args)),
          kwargs_(retain_pyobject(kwargs)),
          saved_inputs_(std::move(saved_inputs)),
          original_inputs_(std::move(original_inputs)),
          grad_parameters_(std::move(grad_parameters)),
          cpu_rng_state_(std::move(cpu_rng_state)),
          cuda_rng_states_(std::move(cuda_rng_states)),
          use_reentrant_(use_reentrant),
          preserve_rng_state_(preserve_rng_state),
          replay_context_(retain_pyobject(replay_context)),
          determinism_check_(std::move(determinism_check)),
          debug_(debug),
          early_stop_(early_stop) {}

    size_t num_inputs() const override {
        return output_metas().size();
    }

    std::string name() const override {
        return "ActivationCheckpointBackward";
    }

    py::object forward_call() {
        py::object function = borrow_pyobject(function_);
        py::tuple args = borrow_pyobject(args_).cast<py::tuple>();
        py::dict kwargs = borrow_pyobject(kwargs_).cast<py::dict>();
        return function(*args, **kwargs);
    }

    tensorplay::tpx::variable_list apply(
        tensorplay::tpx::variable_list&& inputs) override {
        py::gil_scoped_acquire gil;
        if (released_) {
            throw std::runtime_error(tensorplay::tpx::backward_twice_message());
        }
        const bool create_graph = tensorplay::tpx::GradMode::is_enabled();

        std::vector<Tensor> replay_inputs;
        replay_inputs.reserve(saved_inputs_.size());
        for (size_t i = 0; i < saved_inputs_.size(); ++i) {
            const Tensor& saved = saved_inputs_[i];
            Tensor replay = saved.detach();
            if (saved.requires_grad()) {
                if (create_graph) {
                    Tensor correction = tensorplay::tpx::ops::sub(
                        original_inputs_[i], original_inputs_[i].detach());
                    replay = tensorplay::tpx::ops::add(replay, correction);
                } else {
                    tensorplay::tpx::impl::set_requires_grad(replay, true);
                }
            }
            replay_inputs.push_back(std::move(replay));
        }

        size_t position = 0;
        py::object saved_args = borrow_pyobject(args_);
        py::object saved_kwargs = borrow_pyobject(kwargs_);
        py::object replay_args = rebuild_checkpoint_tree(
            saved_args, replay_inputs, position);
        py::object replay_kwargs = rebuild_checkpoint_tree(
            saved_kwargs, replay_inputs, position);
        if (position != replay_inputs.size()) {
            throw std::runtime_error(
                "checkpoint input tree changed during replay");
        }

        CpuRngScope rng_scope(cpu_rng_state_, preserve_rng_state_);
        CheckpointCudaRngScope cuda_rng_scope(
            cuda_rng_states_, preserve_rng_state_);
        py::object replay_output;
        const bool previous_grad = tensorplay::tpx::GradMode::is_enabled();
        tensorplay::tpx::GradMode::set_enabled(true);
        try {
            py::object function = borrow_pyobject(function_);
            PyContextScope context_scope(borrow_pyobject(replay_context_));
            replay_output = function(
                *replay_args.cast<py::tuple>(),
                **replay_kwargs.cast<py::dict>());
            context_scope.close();
        } catch (...) {
            tensorplay::tpx::GradMode::set_enabled(previous_grad);
            throw;
        }
        tensorplay::tpx::GradMode::set_enabled(previous_grad);
        cuda_rng_scope.restore();
        rng_scope.restore();

        std::vector<Tensor> replay_outputs;
        collect_checkpoint_outputs(replay_output, replay_outputs, false);
        const auto& metas = output_metas();
        if (replay_outputs.size() != metas.size()) {
            throw std::runtime_error(
                "checkpoint replay returned a different number of tensor outputs");
        }
        if (!use_reentrant_ && determinism_check_ == "default") {
            for (size_t i = 0; i < replay_outputs.size(); ++i) {
                const Tensor& output = replay_outputs[i];
                const auto& meta = metas[i];
                if (static_cast<std::vector<int64_t>>(output.shape()) != meta.shape ||
                    output.dtype() != meta.dtype ||
                    output.device().type() != meta.device_type ||
                    output.device().index() != meta.device_index) {
                    throw std::runtime_error(
                        "checkpoint replay returned different tensor metadata");
                }
            }
        }

        tensorplay::tpx::variable_list backward_outputs;
        tensorplay::tpx::variable_list backward_grads;
        backward_outputs.reserve(replay_outputs.size());
        backward_grads.reserve(replay_outputs.size());
        for (size_t i = 0; i < replay_outputs.size(); ++i) {
            if (!replay_outputs[i].requires_grad()) continue;
            Tensor gradient;
            if (i < inputs.size() && inputs[i].defined()) {
                gradient = inputs[i];
            } else {
                gradient = tensorplay::tpx::ops::zeros(
                    static_cast<std::vector<int64_t>>(replay_outputs[i].shape()),
                    replay_outputs[i].dtype(), replay_outputs[i].device());
            }
            backward_outputs.push_back(replay_outputs[i]);
            backward_grads.push_back(std::move(gradient));
        }

        if (backward_outputs.empty()) {
            if (use_reentrant_) {
                throw std::runtime_error(
                    "checkpoint replay produced no differentiable outputs");
            }
            return tensorplay::tpx::variable_list(saved_inputs_.size());
        }

        std::vector<Tensor> grad_targets;
        std::vector<size_t> replay_target_indices;
        grad_targets.reserve(replay_inputs.size() + grad_parameters_.size());
        replay_target_indices.reserve(replay_inputs.size());
        for (size_t i = 0; i < replay_inputs.size(); ++i) {
            if (!replay_inputs[i].requires_grad()) continue;
            replay_target_indices.push_back(i);
            grad_targets.push_back(replay_inputs[i]);
        }
        for (const Tensor& parameter : grad_parameters_) {
            grad_targets.push_back(parameter);
        }

        tensorplay::tpx::variable_list results(
            saved_inputs_.size() + grad_parameters_.size());
        if (!grad_targets.empty()) {
            std::vector<Tensor> captured;
            {
                py::gil_scoped_release release;
                captured = tensorplay::tpx::grad(
                    backward_outputs, grad_targets, backward_grads,
                    true, create_graph, true);
            }
            for (size_t i = 0; i < replay_target_indices.size(); ++i) {
                if (i < captured.size()) {
                    results[replay_target_indices[i]] = captured[i];
                }
            }
            const size_t parameter_offset = replay_target_indices.size();
            for (size_t i = 0; i < grad_parameters_.size(); ++i) {
                const size_t captured_index = parameter_offset + i;
                if (captured_index >= captured.size() ||
                    !captured[captured_index].defined()) {
                    continue;
                }
                results[saved_inputs_.size() + i] =
                    captured[captured_index];
            }
        }
        return results;
    }

    void release_variables() override {
        released_ = true;
        saved_inputs_.clear();
        original_inputs_.clear();
        grad_parameters_.clear();
        cpu_rng_state_ = Tensor();
        function_.reset();
        args_.reset();
        kwargs_.reset();
        replay_context_.reset();
        tensorplay::tpx::Node::release_variables();
    }

private:
    PyObjectRef function_;
    PyObjectRef args_;
    PyObjectRef kwargs_;
    std::vector<Tensor> saved_inputs_;
    std::vector<Tensor> original_inputs_;
    std::vector<Tensor> grad_parameters_;
    Tensor cpu_rng_state_;
    std::vector<CheckpointCudaRngState> cuda_rng_states_;
    bool released_ = false;
    bool use_reentrant_ = false;
    bool preserve_rng_state_ = true;
    PyObjectRef replay_context_;
    std::string determinism_check_;
    bool debug_ = false;
    bool early_stop_ = true;
};

// Interned attribute names for the custom-apply hot paths.  Setting a
// context attribute with a freshly built string would allocate (and hash)
// a new unicode object on every forward; interning once turns each setattr
// into a dict hit on a shared key.
struct ApplyAttrNames {
    PyObject* needs_input_grad = nullptr;
    PyObject* backward_fn = nullptr;
    PyObject* node_name = nullptr;
    PyObject* backward = nullptr;
    PyObject* backward_entry = nullptr;
    PyObject* engine_materializes = nullptr;
    PyObject* n_outputs = nullptr;
    PyObject* materialize_grads = nullptr;
    PyObject* node_id = nullptr;
    PyObject* saved_anchors = nullptr;   // "_saved_anchors"
};
ApplyAttrNames g_apply_names;

void init_apply_attr_names() {
    g_apply_names.needs_input_grad = PyUnicode_InternFromString("needs_input_grad");
    g_apply_names.backward_fn = PyUnicode_InternFromString("backward_fn");
    g_apply_names.node_name = PyUnicode_InternFromString("_node_name");
    g_apply_names.backward = PyUnicode_InternFromString("backward");
    g_apply_names.backward_entry = PyUnicode_InternFromString("_backward_entry");
    g_apply_names.engine_materializes = PyUnicode_InternFromString("_engine_materializes");
    g_apply_names.n_outputs = PyUnicode_InternFromString("_n_outputs");
    g_apply_names.materialize_grads = PyUnicode_InternFromString("materialize_grads");
    g_apply_names.node_id = PyUnicode_InternFromString("_node_id");
    g_apply_names.saved_anchors = PyUnicode_InternFromString("_saved_anchors");
}

// Borrowed-value setattr for pre-interned names; errors propagate as
// pybind exceptions so the caller's cleanup paths stay uniform.
inline void set_attr(PyObject* obj, PyObject* name, PyObject* value) {
    if (PyObject_SetAttr(obj, name, value) < 0) {
        throw py::error_already_set();
    }
}

// True only when the attribute exists and is truthy; a missing attribute
// reads as false, which matches the class-level default on the context.
inline bool get_attr_flag(PyObject* obj, PyObject* name) {
    PyObject* value = PyObject_GetAttr(obj, name);
    if (value == nullptr) {
        PyErr_Clear();
        return false;
    }
    int truth = PyObject_IsTrue(value);
    Py_DECREF(value);
    return truth == 1;
}
} // namespace

class PyNode;

// Live nodes from the fused custom-apply paths, keyed by the handle
// recorded on the Python context.  The context itself only holds this
// integer: a strong context-to-node edge would let any context that saved
// an output tensor pin the node (and through it the output) in a cycle
// the Python collector cannot see through the C++ shared_ptr.  When a
// context is finalized while its node is still owned elsewhere -- a live
// output, an in-flight backward -- the finalizer adopts the node here,
// moving the keep-alive onto the context so the backward entry can still
// reach it.
std::mutex g_live_py_nodes_mutex;
std::unordered_map<int64_t, PyNode*> g_live_py_nodes;
int64_t g_next_py_node_id = 1;

// Drop an owning Python reference during node teardown: never touch a
// dead interpreter, and take the GIL only when the calling thread does
// not already hold it.
void py_node_decref(py::object& obj) {
    PyObject* raw = obj.release().ptr();
    if (raw == nullptr) return;
    if (!tensorplay::python_c::interpreter_active()) return;
    if (PyGILState_Check()) {
        Py_DECREF(raw);
        return;
    }
    py::gil_scoped_acquire gil;
    Py_DECREF(raw);
}

// Custom Node for Python-defined Autograd Functions
class PyNode : public tensorplay::tpx::Node {
public:
    explicit PyNode(py::object py_ctx) {
        // The node holds its context only weakly.  A strong member here
        // would close, for any context that saved an output tensor, a
        // context -> output -> this node -> context cycle the Python
        // collector cannot see through; the strong reference comes back
        // through adopt() when the context is finalized while the node is
        // still owned elsewhere.
        PyObject* ref = PyWeakref_NewRef(py_ctx.ptr(), nullptr);
        if (ref == nullptr) {
            throw py::error_already_set();
        }
        ctx_ref_ = py::reinterpret_steal<py::object>(ref);
        adopted_ = py::none();
        {
            std::lock_guard<std::mutex> lock(g_live_py_nodes_mutex);
            registry_id_ = g_next_py_node_id++;
            g_live_py_nodes.emplace(registry_id_, this);
        }
        // The context carries the handle instead of the node: its
        // finalizer uses the handle to find this node and decide whether
        // the node still needs a keeper.
        PyObject* handle = PyLong_FromLongLong(registry_id_);
        if (handle == nullptr) {
            std::lock_guard<std::mutex> lock(g_live_py_nodes_mutex);
            g_live_py_nodes.erase(registry_id_);
            throw py::error_already_set();
        }
        try {
            set_attr(py_ctx.ptr(), g_apply_names.node_id, handle);
        } catch (...) {
            std::lock_guard<std::mutex> lock(g_live_py_nodes_mutex);
            g_live_py_nodes.erase(registry_id_);
            Py_DECREF(handle);
            throw;
        }
        Py_DECREF(handle);
    }

    ~PyNode() {
        // Leave the registry before dropping the Python references: a
        // finalizer they may trigger looks this node up by handle and must
        // not find a half-destroyed entry.
        {
            std::lock_guard<std::mutex> lock(g_live_py_nodes_mutex);
            g_live_py_nodes.erase(registry_id_);
        }
        py_node_decref(adopted_);
        py_node_decref(ctx_ref_);
    }

    // The wrapped context, honoring an adoption.  Raises when neither the
    // adoption slot nor the weak reference resolves, so a caller surfaces
    // a clear error instead of silently running against a dead context.
    py::object resolve_ctx() const {
        if (!adopted_.is_none()) {
            return adopted_;
        }
        PyObject* result = PyObject_CallNoArgs(ctx_ref_.ptr());
        if (result == nullptr) {
            throw py::error_already_set();
        }
        py::object ctx = py::reinterpret_steal<py::object>(result);
        if (ctx.is_none()) {
            throw std::runtime_error(
                "the autograd context of this node is no longer alive");
        }
        return ctx;
    }

    // Called from the context's finalizer while the node is still owned
    // elsewhere.  Taking the context strongly here resurrects it, which
    // finalization permits, and keeps the context alive exactly as long
    // as the node itself lives.
    void adopt(py::object ctx) { adopted_ = std::move(ctx); }

    // Number of external owners (live outputs, engine queues).  Zero means
    // the node is dying on its own and must not be pinned by an adoption.
    long owner_count() const { return weak_from_this().use_count(); }

    // Backward input slots correspond to forward OUTPUTS for custom
    // this node's incoming gradient buffer by the attached output count.
    size_t num_inputs() const override {
        return output_metas().empty() ? Node::num_inputs()
                                      : output_metas().size();
    }

    tensorplay::tpx::variable_list apply(tensorplay::tpx::variable_list&& inputs) override {
        // One static probe instead of a getenv call per backward evaluation.
        static const bool kTrace = [] {
            const char* e = std::getenv("TP_ENGINE_TRACE");
            return e && e[0] != '\0';
        }();
        if (kTrace) fprintf(stderr, "[tp-engine] PyNode: acquiring GIL\n");
        py::gil_scoped_acquire gil;
        if (kTrace) fprintf(stderr, "[tp-engine] PyNode: GIL acquired, calling backward\n");

        // Convert C++ grads to a positional args TUPLE directly (no
        // intermediate py::list): one allocation, PyTuple_SET_ITEM fills.
        size_t n_in = inputs.size();
        py::tuple py_inputs(static_cast<Py_ssize_t>(n_in));
        for (size_t i = 0; i < n_in; ++i) {
            if (inputs[i].defined()) {
                py_inputs[i] = py::cast(inputs[i]);
            } else {
                py_inputs[i] = py::none();
            }
        }
        inputs.clear();

        // Call backward on the context object.  One lookup covers both the
        // presence check and the invocation (a missing attribute is turned
        // into the runtime error below, matching the previous hasattr gate).
        py::object ctx;
        try {
            ctx = resolve_ctx();
        } catch (const std::runtime_error&) {
            throw std::runtime_error(
                "the context of this autograd node is no longer alive; "
                "its backward pass cannot run");
        }
        py::object backward_fn;
        try {
            backward_fn = ctx.attr("backward");
        } catch (const py::error_already_set&) {
            throw std::runtime_error("PyNode context object has no 'backward' method");
        }
        py::object result_obj = backward_fn(*py_inputs);
        if (std::getenv("TP_ENGINE_TRACE")) fprintf(stderr, "[tp-engine] PyNode: backward returned\n");

        tensorplay::tpx::variable_list results;

        if (result_obj.is_none()) {
            return results;
        } else if (py::isinstance<Tensor>(result_obj)) {
            results.push_back(py::cast<Tensor>(result_obj));
        } else if (py::isinstance<py::sequence>(result_obj)) {
            for (auto item : py::cast<py::sequence>(result_obj)) {
                if (item.is_none()) {
                    results.push_back(Tensor());
                } else {
                    results.push_back(py::cast<Tensor>(item));
                }
            }
        } else {
            throw std::runtime_error("backward must return a Tensor, a sequence of Tensors, or None");
        }

        return results;
    }

    // A custom function's node is named after the function class
    // (``<Function>Backward``), which the apply path records on the context.
    std::string name() const override {
        if (!tensorplay::python_c::interpreter_active()) {
            return "PyNode";
        }
        py::gil_scoped_acquire gil;
        try {
            py::object ctx = resolve_ctx();
            py::object label = py::getattr(ctx, "_node_name", py::none());
            if (label.is_none()) {
                return "PyNode";
            }
            return label.cast<std::string>();
        } catch (const std::exception&) {
            PyErr_Clear();
            return "PyNode";
        }
    }

    // A context holds what its backward pass will read, and that pass is the
    // last reader.  The engine releases a node's variables once the graph is
    // no longer kept; a context's saved tensors are the node's variables for
    // a function-backed node, so they are released here rather than left for
    // the context to be collected -- which a caller holding an output keeps
    // alive for as long as the graph is reachable.
    void release_variables() override {
        if (tensorplay::python_c::interpreter_active()) {
            py::gil_scoped_acquire gil;
            try {
                py::object ctx = resolve_ctx();
                py::object release = py::getattr(ctx, "release_saved", py::none());
                if (!release.is_none()) {
                    try {
                        release();
                    } catch (const py::error_already_set&) {
                        PyErr_Clear();
                    }
                }
            } catch (const std::runtime_error&) {
                // The context is already gone, so it released its own saved
                // state when it died.
            }
        }
        tensorplay::tpx::Node::release_variables();
    }

    // Weak reference to the context (never a strong edge; see the ctor).
    py::object ctx_ref_;
    // Strong reference taken back from the context's finalizer while the
    // node is still owned elsewhere.
    py::object adopted_;
    int64_t registry_id_ = 0;
public:
    py::object ctx() const { return resolve_ctx(); }
};

void init_autograd(py::module_& m) {
    init_apply_attr_names();
    py::class_<tensorplay::tpx::Node, std::shared_ptr<tensorplay::tpx::Node>>(m, "Node")
        .def("name", [](const tensorplay::tpx::Node& self) {
            return self.name();
        })
        .def("_raw_ptr", [](const tensorplay::tpx::Node& self) -> int64_t {
            return reinterpret_cast<int64_t>(&self);
        })
        .def("add_pre_hook", [](tensorplay::tpx::Node& self,
                                std::function<std::vector<tensorplay::tpx::Tensor>(
                                    std::vector<tensorplay::tpx::Tensor>)> hook) {
            // Hooks may fire on engine worker threads; manage the GIL here so
            // the C++ hook invocation is always Python-safe.
            self.add_pre_hook([hook](std::vector<tensorplay::tpx::Tensor>&& grads) {
                py::gil_scoped_acquire gil;
                return hook(std::move(grads));
            });
        }, py::arg("hook"))
        .def("_add_tensor_pre_hook", [](tensorplay::tpx::Node& self,
                                        std::function<std::vector<tensorplay::tpx::Tensor>(
                                            std::vector<tensorplay::tpx::Tensor>)> hook) {
            // A hook a tensor registers on the gradient its node receives;
            // it runs before the retained-gradient hooks and the node's own.
            self.add_tensor_pre_hook([hook](std::vector<tensorplay::tpx::Tensor>&& grads) {
                py::gil_scoped_acquire gil;
                return hook(std::move(grads));
            });
        }, py::arg("hook"))
        .def("add_post_hook", [](tensorplay::tpx::Node& self,
                                 std::function<std::vector<tensorplay::tpx::Tensor>(
                                     const std::vector<tensorplay::tpx::Tensor>&,
                                     std::vector<tensorplay::tpx::Tensor>)> hook) {
            self.add_post_hook([hook](const std::vector<tensorplay::tpx::Tensor>& inputs,
                                      std::vector<tensorplay::tpx::Tensor>&& outputs) {
                py::gil_scoped_acquire gil;
                return hook(inputs, std::move(outputs));
            });
        }, py::arg("hook"))
        .def_property_readonly("next_functions", [](const tensorplay::tpx::Node& self) {
            std::vector<std::pair<std::shared_ptr<tensorplay::tpx::Node>, int>> result;
            for (const auto& edge : self.next_edges()) {
                result.push_back({edge.function, (int)edge.input_nr});
            }
            return result;
        })
        .def_property_readonly("variable", [](const tensorplay::tpx::Node& self) -> std::optional<tensorplay::tpx::Tensor> {
            auto* acc = dynamic_cast<const tensorplay::tpx::AccumulateGrad*>(&self);
            if (acc) {
                return acc->value_;
            }
            return std::nullopt;
        });

    py::module_ autograd = m.def_submodule("_autograd", "Autograd mechanism");

    py::class_<SavedTensorToken>(autograd, "_SavedTensorToken");

    autograd.def(
        "_push_saved_tensors_hooks",
        [](py::object pack, py::object unpack) {
            if (!PyCallable_Check(pack.ptr()) ||
                !PyCallable_Check(unpack.ptr())) {
                throw py::type_error(
                    "saved tensor hooks must both be callable");
            }
            tensorplay::tpx::push_saved_variable_hooks(
                std::make_shared<PythonSavedVariableHooks>(
                    std::move(pack), std::move(unpack)));
        },
        "pack"_a,
        "unpack"_a);
    autograd.def("_pop_saved_tensors_hooks", []() {
        tensorplay::tpx::pop_saved_variable_hooks();
    });
    autograd.def("_saved_variable_hooks_active", []() {
        return static_cast<bool>(
            tensorplay::tpx::current_saved_variable_hooks());
    });
    autograd.def("_pack_saved_tensor", [](const Tensor& tensor) {
        auto hooks = tensorplay::tpx::current_saved_variable_hooks();
        if (!hooks) return py::object(py::none());
        SavedTensorToken token;
        token.hooks = std::move(hooks);
        token.packed = token.hooks->pack(tensor);
        return py::cast(std::move(token));
    });
    autograd.def("_unpack_saved_tensor", [](const SavedTensorToken& token) {
        if (!token.hooks) {
            throw std::runtime_error("saved tensor token has no hook");
        }
        return token.hooks->unpack(token.packed);
    });
    autograd.def("_checkpoint_operation_begin", []() -> int64_t {
        auto frame = current_native_checkpoint_frame();
        return frame ? frame->begin_operation() : -1;
    });
    autograd.def("_checkpoint_operation_end", [](int64_t operation_index) {
        auto frame = current_native_checkpoint_frame();
        if (frame) frame->end_operation(operation_index);
    }, "operation_index"_a);
    autograd.def(
        "_checkpoint_operation_cache",
        [](int64_t operation_index, bool cache) {
            auto frame = current_native_checkpoint_frame();
            if (frame) frame->cache_operation(operation_index, cache);
        },
        "operation_index"_a,
        "cache"_a);
    autograd.def("_checkpoint_operation_reuse", [](int64_t operation_index) {
        auto frame = current_native_checkpoint_frame();
        if (frame) frame->reuse_operation(operation_index);
    }, "operation_index"_a);

    py::class_<PyNode, tensorplay::tpx::Node, std::shared_ptr<PyNode>>(autograd, "PyNode")
        .def(py::init<py::object>())
        .def("add_next_edge", [](PyNode& self, std::shared_ptr<tensorplay::tpx::Node> next_node, int input_nr) {
            if (next_node) {
                self.add_next_edge(tensorplay::tpx::Edge(next_node, input_nr));
            } else {
                self.add_next_edge(tensorplay::tpx::Edge());
            }
        }, "next_node"_a.none(), "input_nr"_a = 0)
        .def("set_materialize_grads", &PyNode::set_materialize_grads,
             py::arg("value"))
        .def_property_readonly(
            "_py_ctx", [](PyNode& self) -> py::object { return self.resolve_ctx(); },
            "The Python context object this node wraps.")
        .def_property_readonly(
            "_node_id", [](PyNode& self) { return self.registry_id_; },
            "Registry handle the node recorded on its context.")
        .def(
            "register_hook",
            [](PyNode& self, py::function hook) {
                self.ctx().attr("_hooks").cast<py::list>().append(hook);
            },
            py::keep_alive<1, 2>())
        .def(
            "register_prehook",
            [](PyNode& self, py::function hook) {
                self.ctx().attr("_prehooks").cast<py::list>().append(hook);
            },
            py::keep_alive<1, 2>())
        .def(
            "attach_outputs",
            [](PyNode& self, py::handle outputs) {
                // Single C++ crossing for graph attachment: marks every
                // tensor output as requiring grad and wires this node as
                // its grad_fn.  Non-tensor slots are skipped so multi-output
                // functions can return Nones mixed with Tensors.  Also
                // engine to zero-fill missing gradients.
                auto node = std::shared_ptr<tensorplay::tpx::Node>(
                    std::static_pointer_cast<tensorplay::tpx::Node>(
                        self.shared_from_this()));
                auto& metas = self.output_metas();
                metas.clear();
                int idx = 0;
                auto record_slot = [&](py::handle item) {
                    tensorplay::tpx::OutputSlotMeta m;
                    if (py::isinstance<Tensor>(item)) {
                        const Tensor& t = py::cast<const Tensor&>(item);
                        m.shape = static_cast<std::vector<int64_t>>(t.shape());
                        m.dtype = t.dtype();
                        m.device_type = t.device().type();
                        m.device_index = t.device().index();
                        m.valid = true;
                    }
                    metas.push_back(std::move(m));
                };
                if (py::isinstance<Tensor>(outputs)) {
                    record_slot(outputs);
                    Tensor& t = py::cast<Tensor&>(outputs);
                    tensorplay::tpx::impl::set_requires_grad(t, true);
                    tensorplay::tpx::impl::set_grad_fn(t, node, 0);
                    return;
                }
                for (auto item : outputs.cast<py::sequence>()) {
                    record_slot(item);
                    if (py::isinstance<Tensor>(item)) {
                        Tensor& t = py::cast<Tensor&>(item);
                        tensorplay::tpx::impl::set_requires_grad(t, true);
                        tensorplay::tpx::impl::set_grad_fn(t, node, idx);
                    }
                    ++idx;
                }
            },
            "outputs"_a);

    autograd.def(
        "_activation_checkpoint",
        [](py::object function,
           py::tuple args,
           py::dict kwargs,
           bool use_reentrant,
           bool preserve_rng_state,
           py::object context_fn,
           std::string determinism_check,
           bool debug,
           bool early_stop) -> py::object {
            if (!PyCallable_Check(function.ptr())) {
                throw py::type_error("checkpoint function must be callable");
            }
            if (use_reentrant && !context_fn.is_none()) {
                throw std::invalid_argument(
                    "context_fn is not supported for reentrant checkpointing");
            }
            if (use_reentrant && debug) {
                throw std::invalid_argument(
                    "debug is not supported for reentrant checkpointing");
            }
            if (!use_reentrant && debug && !context_fn.is_none()) {
                throw std::invalid_argument(
                    "debug is incompatible with a custom context_fn");
            }
            if (!use_reentrant && determinism_check != "default" &&
                determinism_check != "none") {
                throw std::invalid_argument(
                    "determinism_check must be 'default' or 'none'");
            }

            std::vector<Tensor> saved_inputs;
            std::vector<Tensor> original_inputs;
            std::vector<tensorplay::tpx::Edge> edges;
            bool any_requires_grad = false;
            const bool previous_grad =
                tensorplay::tpx::GradMode::is_enabled();
            std::vector<Tensor> grad_parameters;
            collect_checkpoint_inputs(args, saved_inputs, original_inputs,
                                      edges,
                                      any_requires_grad);
            collect_checkpoint_inputs(kwargs, saved_inputs, original_inputs,
                                      edges,
                                      any_requires_grad);
            if (previous_grad) {
                collect_grad_parameters(function, grad_parameters);
            }
            const bool has_grad_parameters =
                previous_grad && !grad_parameters.empty();
            if (!previous_grad ||
                (!any_requires_grad && !has_grad_parameters)) {
                return function(*args, **kwargs);
            }

            for (const Tensor& parameter : grad_parameters) {
                auto parameter_edges =
                    tensorplay::tpx::collect_next_edges(parameter);
                edges.insert(edges.end(),
                             std::make_move_iterator(parameter_edges.begin()),
                             std::make_move_iterator(parameter_edges.end()));
            }

            auto contexts = make_checkpoint_contexts(context_fn);
            Tensor cpu_rng_state;
            if (preserve_rng_state) {
                cpu_rng_state = default_generator().get_state();
            }
            auto cuda_rng_states = capture_checkpoint_cuda_rng_states(
                saved_inputs, grad_parameters, preserve_rng_state);

            if (!use_reentrant) {
                auto frame = std::make_shared<NativeCheckpointFrame>(
                    function, args, kwargs, std::move(saved_inputs),
                    std::move(original_inputs), std::move(cpu_rng_state),
                    std::move(cuda_rng_states), preserve_rng_state,
                    std::move(contexts.second),
                    std::move(determinism_check), debug, early_stop);
                py::object output;
                tensorplay::tpx::GradMode::set_enabled(true);
                try {
                    PyContextScope context_scope(std::move(contexts.first));
                    SavedVariableHooksScope hooks_scope(frame);
                    output = frame->forward_call();
                    frame->finish_forward();
                    hooks_scope.close();
                    context_scope.close();
                } catch (...) {
                    tensorplay::tpx::GradMode::set_enabled(previous_grad);
                    throw;
                }
                tensorplay::tpx::GradMode::set_enabled(previous_grad);
                return output;
            }

            auto node = std::make_shared<ActivationCheckpointNode>(
                std::move(function), std::move(args), std::move(kwargs),
                std::move(saved_inputs), std::move(original_inputs),
                std::move(grad_parameters), std::move(cpu_rng_state),
                std::move(cuda_rng_states), use_reentrant, preserve_rng_state,
                std::move(contexts.second), std::move(determinism_check),
                debug, early_stop);
            node->add_next_edge_list(std::move(edges));
            node->set_materialize_grads(true);

            py::object output;
            tensorplay::tpx::GradMode::set_enabled(!use_reentrant);
            try {
                PyContextScope context_scope(std::move(contexts.first));
                output = node->forward_call();
                context_scope.close();
            } catch (...) {
                tensorplay::tpx::GradMode::set_enabled(previous_grad);
                throw;
            }
            tensorplay::tpx::GradMode::set_enabled(previous_grad);

            auto base_node = std::static_pointer_cast<tensorplay::tpx::Node>(
                node);
            auto& metas = node->output_metas();
            metas.clear();
            std::vector<Tensor> forward_outputs;
            collect_checkpoint_outputs(
                output, forward_outputs, use_reentrant);
            if (forward_outputs.empty()) return output;
            size_t output_index = 0;
            return attach_checkpoint_outputs(
                output, base_node, metas, output_index, use_reentrant);
        },
        "function"_a,
        "args"_a,
        "kwargs"_a,
        "use_reentrant"_a,
        "preserve_rng_state"_a,
        "context_fn"_a,
        "determinism_check"_a,
        "debug"_a,
        "early_stop"_a);
    
    autograd.def("collect_next_edges", [](const Tensor& t) {
        auto edges = tensorplay::tpx::collect_next_edges(t);
        std::vector<std::pair<std::shared_ptr<tensorplay::tpx::Node>, int>> result;
        for (const auto& edge : edges) {
            result.push_back({edge.function, (int)edge.input_nr});
        }
        return result;
    });

    // Single-crossing requires-grad gate for generated autograd classes:
    // without it, a call whose inputs need no gradient builds and discards
    // an empty graph node.  Tensors nested in plain lists count; tensors
    // inside dicts or mixed lists do not.
    autograd.def("_any_requires_grad", [](py::args args, py::kwargs kwargs) {
        for (py::handle item : args) {
            if (visit_requires_grad(item.ptr())) {
                return true;
            }
        }
        if (kwargs) {
            PyObject* key = nullptr;
            PyObject* value = nullptr;
            Py_ssize_t position = 0;
            while (PyDict_Next(kwargs.ptr(), &position, &key, &value)) {
                if (visit_requires_grad(value)) {
                    return true;
                }
            }
        }
        return false;
    });

    // tuple producing needs_input_grad bits AND wiring this node's
    // next_edges.  Returns (needs_list, any_requires_grad) so the Python
    // layer avoids N per-input pybind round-trips.
    autograd.def("setup_custom_function_graph",
        [](py::object node_obj, py::sequence args) {            auto node = node_obj.cast<std::shared_ptr<tensorplay::tpx::Node>>();
            Py_ssize_t n = PyTuple_GET_SIZE(args.ptr());
            py::list needs(n);
            bool any_rg = false;
            std::vector<tensorplay::tpx::Edge> edges;
            edges.reserve((size_t)n);
            for (Py_ssize_t i = 0; i < n; ++i) {
                PyObject* item = PyTuple_GET_ITEM(args.ptr(), i);
                if (py::isinstance<Tensor>(item)) {
                    const Tensor& t = py::cast<const Tensor&>(item);
                    bool rg = t.requires_grad();
                    any_rg |= rg;
                    needs[i] = py::bool_(rg);
                    if (rg) {
                        for (auto& e : tensorplay::tpx::collect_next_edges(t)) {
                            edges.push_back(std::move(e));
                        }
                    } else {
                        edges.emplace_back();
                    }
                } else {
                    needs[i] = py::bool_(false);
                    edges.emplace_back();
                }
            }
            if (any_rg) {
                node->add_next_edge_list(std::move(edges));
            }
            return py::make_tuple(needs, any_rg);
        },
        "node"_a, "args"_a);

    // Custom-function forward block: toggles grad off, calls
    // the user forward, then setup_context -- all inside ONE crossing so
    // the Python layer pays no per-step pybind/GIL-mode round-trips.
    autograd.def("run_custom_function_forward",
        [](py::object ctx, py::object forward_fn,
           std::optional<py::object> setup_ctx_fn, py::sequence args) {
            const bool prev = tensorplay::tpx::GradMode::is_enabled();
            tensorplay::tpx::GradMode::set_enabled(false);
            py::object output;
            try {
                if (setup_ctx_fn.has_value()) {
                    // new style: forward(*args); setup_context(ctx, args, out)
                    output = (*forward_fn)(*(args.cast<py::tuple>()));
                    if (output) {
                        (*setup_ctx_fn)(
                            ctx, args, py::object(output));
                    }
                } else {
                    // legacy style: forward(ctx, *args)
                    py::tuple full(args.size() + 1);
                    full[0] = ctx;
                    for (Py_ssize_t i = 0; i < args.size(); ++i) {
                        full[i + 1] = args[i];
                    }
                    output = (*forward_fn)(*full);
                }
            } catch (...) {
                tensorplay::tpx::GradMode::set_enabled(prev);
                throw;
            }
            tensorplay::tpx::GradMode::set_enabled(prev);
            return output;
        },
        "ctx"_a, "forward_fn"_a, "setup_ctx_fn"_a.none(), "args"_a);

    // THE single-entry hot path: node creation, unpack_input, the
    // AutoGradMode(false) forward block, setup_context, output marking and
    // the backward-entry wiring all happen inside ONE pybind crossing, on
    // the same Node/Edge machinery native kernels build their graph with.
    // Returns (output, ctx, needs, executable, node); Python only handles
    // the rare direct-backward override afterwards.  The node is
    // constructed natively (same translation unit), so the graph node never
    // round-trips through a Python factory callback.
    autograd.def("custom_function_apply",
        [](py::object ctx_factory,
           py::object forward_fn, std::optional<py::object> setup_ctx_fn,
           py::sequence args,
           py::object backward_fn, py::object node_name) {
            auto ctx = ctx_factory();

            // ---- unpack_input ----
            Py_ssize_t n_args = PyTuple_GET_SIZE(args.ptr());
            py::tuple needs(n_args);
            bool any_rg = false;
            for (Py_ssize_t i = 0; i < n_args; ++i) {
                PyObject* item = PyTuple_GET_ITEM(args.ptr(), i);
                if (fast_is_tensor(item)) {
                    bool rg = py::cast<const Tensor&>(item).requires_grad();
                    any_rg |= rg;
                    PyObject* bit = rg ? Py_True : Py_False;
                    Py_INCREF(bit);
                    PyTuple_SET_ITEM(needs.ptr(), i, bit);
                } else if (py::isinstance<py::sequence>(item)) {
                    // Nested containers are rare; mark conservatively and
                    // let the Python fallback re-wire if needed.
                    bool nested_rg = false;
                    for (auto inner : py::reinterpret_borrow<py::sequence>(item)) {
                        if (py::isinstance<Tensor>(inner)
                            && py::cast<const Tensor&>(inner).requires_grad()) {
                            nested_rg = true;
                            break;
                        }
                    }
                    any_rg |= nested_rg;
                    PyObject* bit = nested_rg ? Py_True : Py_False;
                    Py_INCREF(bit);
                    PyTuple_SET_ITEM(needs.ptr(), i, bit);
                } else {
                    PyObject* bit = Py_False;
                    Py_INCREF(bit);
                    PyTuple_SET_ITEM(needs.ptr(), i, bit);
                }
            }

            const bool prev_grad = tensorplay::tpx::GradMode::is_enabled();
            const bool executable = prev_grad && any_rg;
            // The node only exists when this call is actually tracked:
            // with the gate closed there is no backward to serve, and
            // skipping it keeps the untracked path free of the weakref and
            // registry bookkeeping the ownership design needs.
            py::object node = py::none();
            PyNode* py_node = nullptr;
            if (executable) {
                node = py::cast(std::make_shared<PyNode>(ctx));
                py_node = node.cast<PyNode*>();
                // next_edges from every tensor arg (single pass)
                std::vector<tensorplay::tpx::Edge> edges;
                edges.reserve((size_t)n_args);
                for (Py_ssize_t i = 0; i < n_args; ++i) {
                    PyObject* item = PyTuple_GET_ITEM(args.ptr(), i);
                    if (fast_is_tensor(item)
                        && py::cast<const Tensor&>(item).requires_grad()) {
                        for (auto& e : tensorplay::tpx::collect_next_edges(
                                 py::cast<const Tensor&>(item))) {
                            edges.push_back(std::move(e));
                        }
                    } else {
                        edges.emplace_back();
                    }
                }
                py_node->add_next_edge_list(std::move(edges));
            }

            // ---- forward block under AutoGradMode(false) ----
            py::object output;
            tensorplay::tpx::GradMode::set_enabled(false);
            try {
                if (setup_ctx_fn.has_value()) {
                    output = (*forward_fn)(*(args.cast<py::tuple>()));
                    if (!output) throw py::error_already_set();
                    (*setup_ctx_fn)(ctx, args, output);
                } else {
                    py::tuple full(n_args + 1);
                    full[0] = ctx;
                    for (Py_ssize_t i = 0; i < n_args; ++i) {
                        full[i + 1] = PyTuple_GET_ITEM(args.ptr(), i);
                    }
                    output = (*forward_fn)(*full);
                }
            } catch (...) {
                tensorplay::tpx::GradMode::set_enabled(prev_grad);
                throw;
            }
            tensorplay::tpx::GradMode::set_enabled(prev_grad);

            // ---- _wrap_outputs (executable only) ----
            if (executable) {
                auto shared = std::shared_ptr<tensorplay::tpx::Node>(
                    std::static_pointer_cast<tensorplay::tpx::Node>(
                        py_node->shared_from_this()));
                auto& metas = py_node->output_metas();
                metas.clear();
                int idx = 0;
                auto mark = [&](py::handle item) {
                    tensorplay::tpx::OutputSlotMeta m;
                    if (fast_is_tensor(item.ptr())) {
                        Tensor& t = py::cast<Tensor&>(item);
                        tensorplay::tpx::impl::set_requires_grad(t, true);
                        tensorplay::tpx::impl::set_grad_fn(t, shared, idx);
                        m.shape =
                            static_cast<std::vector<int64_t>>(t.shape());
                        m.dtype = t.dtype();
                        m.device_type = t.device().type();
                        m.device_index = t.device().index();
                        m.valid = true;
                    }
                    metas.push_back(std::move(m));
                    ++idx;
                };
                if (py::isinstance<Tensor>(output)) {
                    mark(output);
                } else if (py::isinstance<py::sequence>(output)) {
                    for (auto item : output.cast<py::sequence>()) mark(item);
                }
                // The engine zero-fills missing gradient slots from the
                // per-output metadata recorded above.  The raw outputs are
                // never retained on the context: doing so would close a
                // reference cycle (ctx -> output -> grad_fn -> ctx) that
                // Python's collector cannot see through the C++ node.
                PyObject* num = PyLong_FromLong(idx);
                set_attr(ctx.ptr(), g_apply_names.n_outputs, num);
                Py_DECREF(num);
                bool materialize = get_attr_flag(
                    ctx.ptr(), g_apply_names.materialize_grads);
                py_node->set_materialize_grads(materialize);
                set_attr(ctx.ptr(), g_apply_names.engine_materializes,
                         materialize ? Py_True : Py_False);
            }
            // Backward-entry wiring.  ``needs_input_grad`` is read by user
            // backward code, the rest only by the entry itself; setting it
            // here keeps the Python layer off the hot path entirely.
            set_attr(ctx.ptr(), g_apply_names.needs_input_grad, needs.ptr());
            if (executable) {
                if (backward_fn.ptr() != nullptr && !backward_fn.is_none()) {
                    set_attr(ctx.ptr(), g_apply_names.backward_fn,
                             backward_fn.ptr());
                }
                if (node_name.ptr() != nullptr && !node_name.is_none()) {
                    set_attr(ctx.ptr(), g_apply_names.node_name,
                             node_name.ptr());
                }
                PyObject* entry = PyObject_GetAttr(
                    ctx.ptr(), g_apply_names.backward_entry);
                if (entry == nullptr) {
                    throw py::error_already_set();
                }
                set_attr(ctx.ptr(), g_apply_names.backward, entry);
                Py_DECREF(entry);
            }
            return py::make_tuple(output, ctx, needs, executable,
                                   node);
        },
        "ctx_factory"_a, "forward_fn"_a,
        "setup_ctx_fn"_a.none(), "args"_a,
        "backward_fn"_a = py::none(), "node_name"_a = py::none());

    // Full eager custom-operator call in ONE crossing: grad gate,
    // requires-grad scan, graph node, kernel invocation and context wiring.
    // ``kernel_fn`` is the generated legacy forward (ctx, *args, **kwargs)
    // so user setup_context runs inside the same disabled-grad block;
    // ``backward_fn`` is the registered backward formula, stored on the
    // context for the backward entry.  Returns the kernel output, or None
    // when no input requires grad (or gradients are disabled) — the caller
    // then invokes the kernel directly.
    autograd.def("custom_op_autograd_apply",
        [](py::object ctx_factory, py::object kernel_fn, py::object backward_fn,
           py::object node_name, py::sequence args, py::dict kwargs) -> py::object {
            if (!tensorplay::tpx::GradMode::is_enabled()) {
                return py::none();
            }

            // ---- unpack_input: needs bits + any-requires-grad ----
            Py_ssize_t n_args = PyTuple_GET_SIZE(args.ptr());
            py::tuple needs(n_args);
            bool any_rg = false;
            for (Py_ssize_t i = 0; i < n_args; ++i) {
                PyObject* item = PyTuple_GET_ITEM(args.ptr(), i);
                if (fast_is_tensor(item)) {
                    bool rg = py::cast<const Tensor&>(item).requires_grad();
                    any_rg |= rg;
                    PyObject* bit = rg ? Py_True : Py_False;
                    Py_INCREF(bit);
                    PyTuple_SET_ITEM(needs.ptr(), i, bit);
                } else if (py::isinstance<py::sequence>(item)) {
                    // Nested containers are rare; mark conservatively.
                    bool nested_rg = false;
                    for (auto inner : py::reinterpret_borrow<py::sequence>(item)) {
                        if (py::isinstance<Tensor>(inner)
                            && py::cast<const Tensor&>(inner).requires_grad()) {
                            nested_rg = true;
                            break;
                        }
                    }
                    any_rg |= nested_rg;
                    PyObject* bit = nested_rg ? Py_True : Py_False;
                    Py_INCREF(bit);
                    PyTuple_SET_ITEM(needs.ptr(), i, bit);
                } else {
                    PyObject* bit = Py_False;
                    Py_INCREF(bit);
                    PyTuple_SET_ITEM(needs.ptr(), i, bit);
                }
            }
            if (!any_rg) {
                for (auto value : kwargs) {
                    if (visit_requires_grad(value.second.ptr())) {
                        any_rg = true;
                        break;
                    }
                }
                if (!any_rg) {
                    return py::none();
                }
            }

            auto ctx = ctx_factory();
            // The node stays a pure C++ object: nothing on this path hands
            // it back to Python, so no Python wrapper is allocated for it.
            auto node = std::make_shared<PyNode>(ctx);
            std::shared_ptr<tensorplay::tpx::Node> shared = node;

            {
                std::vector<tensorplay::tpx::Edge> edges;
                edges.reserve((size_t)n_args);
                for (Py_ssize_t i = 0; i < n_args; ++i) {
                    PyObject* item = PyTuple_GET_ITEM(args.ptr(), i);
                    if (fast_is_tensor(item)
                        && py::cast<const Tensor&>(item).requires_grad()) {
                        for (auto& e : tensorplay::tpx::collect_next_edges(
                                 py::cast<const Tensor&>(item))) {
                            edges.push_back(std::move(e));
                        }
                    } else {
                        edges.emplace_back();
                    }
                }
                node->add_next_edge_list(std::move(edges));
            }

            // ---- forward block under AutoGradMode(false) ----
            py::object output;
            tensorplay::tpx::GradMode::set_enabled(false);
            try {
                py::tuple full(n_args + 1);
                full[0] = ctx;
                for (Py_ssize_t i = 0; i < n_args; ++i) {
                    full[i + 1] = PyTuple_GET_ITEM(args.ptr(), i);
                }
                output = py::reinterpret_steal<py::object>(
                    PyObject_Call(kernel_fn.ptr(), full.ptr(), kwargs.ptr()));
                if (!output) throw py::error_already_set();
            } catch (...) {
                tensorplay::tpx::GradMode::set_enabled(true);
                throw;
            }
            tensorplay::tpx::GradMode::set_enabled(true);

            // ---- context wiring (everything backward needs), with
            // pre-interned names so no per-call string is built ----
            PyObject* ctx_ptr = ctx.ptr();
            set_attr(ctx_ptr, g_apply_names.needs_input_grad, needs.ptr());
            set_attr(ctx_ptr, g_apply_names.backward_fn, backward_fn.ptr());
            set_attr(ctx_ptr, g_apply_names.node_name, node_name.ptr());
            PyObject* entry = PyObject_GetAttr(ctx_ptr, g_apply_names.backward_entry);
            if (entry == nullptr) {
                throw py::error_already_set();
            }
            set_attr(ctx_ptr, g_apply_names.backward, entry);
            Py_DECREF(entry);
            bool materialize = get_attr_flag(ctx_ptr, g_apply_names.materialize_grads);
            node->set_materialize_grads(materialize);
            set_attr(ctx_ptr, g_apply_names.engine_materializes,
                     materialize ? Py_True : Py_False);
            // The raw outputs are never retained on the context: that would
            // close a reference cycle (ctx -> output -> grad_fn -> ctx) the
            // Python collector cannot see through the C++ node.  With
            // materialization opted out the engine hands undefined slots to
            // backward as-is, which is the point of opting out.

            // ---- attach outputs ----
            auto& metas = node->output_metas();
            metas.clear();
            int idx = 0;
            auto mark = [&](py::handle item) {
                tensorplay::tpx::OutputSlotMeta m;
                if (fast_is_tensor(item.ptr())) {
                    Tensor& t = py::cast<Tensor&>(item);
                    tensorplay::tpx::impl::set_requires_grad(t, true);
                    tensorplay::tpx::impl::set_grad_fn(t, shared, idx);
                    m.shape = static_cast<std::vector<int64_t>>(t.shape());
                    m.dtype = t.dtype();
                    m.device_type = t.device().type();
                    m.device_index = t.device().index();
                    m.valid = true;
                }
                metas.push_back(std::move(m));
                ++idx;
            };
            if (py::isinstance<Tensor>(output)) {
                mark(output);
            } else if (py::isinstance<py::sequence>(output)) {
                for (auto item : output.cast<py::sequence>()) mark(item);
            }
            PyObject* num = PyLong_FromLong(idx);
            set_attr(ctx_ptr, g_apply_names.n_outputs, num);
            Py_DECREF(num);

            return output;
        },
        "ctx_factory"_a, "kernel_fn"_a, "backward_fn"_a,
        "node_name"_a, "args"_a, "kwargs"_a);

    // Adoption entry point, called from a context's finalizer with the
    // handle the node recorded on it.  True means the node is still alive
    // and owned from elsewhere, so the context -- resurrected by this call
    // -- takes the keep-alive over.  The whole decision runs under the
    // registry lock so a node being torn down concurrently finishes its
    // destructor (which needs the same lock) before this could touch it.
    //
    // Owners whose only life support is the context's own saved tensors do
    // not count: a context that saved one of this node's outputs pins the
    // node through that tensor's grad_fn, and adopting on top of it would
    // close a context -> saved output -> node -> context cycle the
    // collector cannot see through.  Such a context collects together with
    // its saved tensors instead (each carries the context back), so the
    // node dies with the island.  A positive surplus means real external
    // owners -- a kept output, downstream graph edges, an in-flight
    // backward -- and the context must outlive them for backward to run.
    autograd.def("_adopt_node_if_needed",
        [](int64_t node_id, py::object ctx) -> bool {
            std::lock_guard<std::mutex> lock(g_live_py_nodes_mutex);
            auto it = g_live_py_nodes.find(node_id);
            if (it == g_live_py_nodes.end()) {
                return false;
            }
            PyNode* node = it->second;
            long owners = node->owner_count();
            if (owners == 0) {
                return false;
            }
            long self_saved = 0;
            PyObject* anchors = PyObject_GetAttr(
                ctx.ptr(), g_apply_names.saved_anchors);
            if (anchors == nullptr) {
                PyErr_Clear();
            } else {
                if (PyTuple_Check(anchors)) {
                    Py_ssize_t n = PyTuple_GET_SIZE(anchors);
                    for (Py_ssize_t i = 0; i < n; ++i) {
                        PyObject* item = PyTuple_GET_ITEM(anchors, i);
                        if (fast_is_tensor(item)) {
                            auto fn = tensorplay::tpx::impl::grad_fn(
                                py::cast<const Tensor&>(item));
                            if (fn.get() ==
                                static_cast<tensorplay::tpx::Node*>(node)) {
                                ++self_saved;
                            }
                        }
                    }
                }
                Py_DECREF(anchors);
            }
            if (owners <= self_saved) {
                return false;
            }
            node->adopt(std::move(ctx));
            return true;
        });

    autograd.def("_get_current_graph_task_keep_graph",
                 &tensorplay::tpx::Engine::current_graph_task_keep_graph);
    using WeakTensorRef = tensorplay::weak_intrusive_ptr<tensorplay::TensorImpl>;
    py::class_<WeakTensorRef>(m, "_WeakTensorRef")
        .def(py::init([](const Tensor& tensor) {
            return WeakTensorRef(tensor.unsafeGetTensorImpl());
        }))
        .def("expired", &WeakTensorRef::expired);

    autograd.def("backward", [](const std::vector<Tensor>& tensors, std::optional<std::vector<Tensor>> grad_tensors, std::optional<bool> retain_graph, bool create_graph, std::optional<std::vector<Tensor>> inputs) {
        bool keep_graph = retain_graph.value_or(create_graph);
        std::vector<Tensor> grads;
        if (grad_tensors) grads = *grad_tensors;
        std::vector<Tensor> only;
        if (inputs) only = *inputs;
        // The engine may evaluate nodes on worker threads that need the GIL
        // for Python-backed autograd functions; the initiating thread must
        // not hold it while it waits for the graph to drain.
        py::gil_scoped_release release;
        tensorplay::tpx::backward(tensors, grads, keep_graph, create_graph, only);
    }, "tensors"_a, "grad_tensors"_a = py::none(), "retain_graph"_a = py::none(), "create_graph"_a = false, "inputs"_a = py::none());

    autograd.def("queue_callback", [](py::function callback) {
        PyObject* raw_callback = callback.ptr();
        Py_INCREF(raw_callback);
        std::shared_ptr<PyObject> callback_ref(
            raw_callback,
            [](PyObject* object) {
                if (!tensorplay::python_c::interpreter_active()) return;
                py::gil_scoped_acquire gil;
                Py_DECREF(object);
            });
        tensorplay::tpx::Engine::get_default_engine().queue_callback(
            [callback_ref = std::move(callback_ref)]() {
                py::gil_scoped_acquire gil;
                py::reinterpret_borrow<py::object>(callback_ref.get())();
            });
    }, "callback"_a);

    autograd.def("grad", [](const std::vector<Tensor>& outputs, const std::vector<Tensor>& inputs, std::optional<std::vector<Tensor>> grad_outputs, std::optional<bool> retain_graph, bool create_graph, bool allow_unused) {
        bool keep_graph = retain_graph.value_or(create_graph);
        std::vector<Tensor> grads;
        if (grad_outputs) grads = *grad_outputs;
        // Undefined gradients (unused inputs, or grads that arrive as
        // on it.
        std::vector<tensorplay::Tensor> captured;
        {
            py::gil_scoped_release release;
            captured = tensorplay::tpx::grad(outputs, inputs, grads,
                                             keep_graph, create_graph,
                                             allow_unused);
        }
        py::tuple result(captured.size());
        for (size_t i = 0; i < captured.size(); ++i) {
            if (captured[i].defined()) result[i] = py::cast(std::move(captured[i]));
            else result[i] = py::none();
        }
        return result;
    }, "outputs"_a, "inputs"_a, "grad_outputs"_a = py::none(), "retain_graph"_a = py::none(), "create_graph"_a = false, "allow_unused"_a = false);

    autograd.def("is_grad_enabled", &tensorplay::tpx::GradMode::is_enabled);
    autograd.def("set_grad_enabled", &tensorplay::tpx::GradMode::set_enabled);

    // Restores each tensor's version counter to a previously observed value.
    // Used by the Python context manager of the same purpose so a tensor whose
    // storage is freed and re-populated (sharded-parameter recycling) stays
    // valid for saved-tensor mutation checks.
    autograd.def(
        "_unsafe_set_version_counter",
        [](const std::vector<Tensor>& tensors, const std::vector<int64_t>& versions) {
            if (tensors.size() != versions.size()) {
                throw std::runtime_error(
                    "tensors and versions must have the same length");
            }
            for (size_t i = 0; i < tensors.size(); ++i) {
                tensors[i].unsafeGetTensorImpl()->set_version(versions[i]);
            }
        });

    // Python wrapper drives through __enter__/__exit__. Entering disables
    // autograd recording and freezes version counters; exit restores the
    struct PyInferenceMode {
        bool prev_ = false;
        bool prev_grad_ = true;
        explicit PyInferenceMode(bool mode) {
            prev_ = tensorplay::tpx::InferenceMode::is_enabled();
            prev_grad_ = tensorplay::tpx::GradMode::is_enabled();
            tensorplay::tpx::InferenceMode::set_enabled(mode);
            tensorplay::tpx::GradMode::set_enabled(!mode);
        }
        void enter() {}
        void exit(const std::optional<py::object>&,
                  const std::optional<py::object>&,
                  const std::optional<py::object>&) {
            tensorplay::tpx::InferenceMode::set_enabled(prev_);
            tensorplay::tpx::GradMode::set_enabled(prev_grad_);
        }
    };

    py::class_<PyInferenceMode>(autograd, "_InferenceMode")
        .def(py::init<bool>(), py::arg("mode") = true)
        .def("__enter__", &PyInferenceMode::enter)
        .def("__exit__", &PyInferenceMode::exit);

    autograd.def("is_inference_mode_enabled", &tensorplay::tpx::InferenceMode::is_enabled);

    // creation happens deep inside C++ op wrappers while the calling thread
    // holds the GIL, so capturing the Python traceback at that point records
    // the user-level call site of each forward op.
    autograd.def("is_anomaly_enabled", &tensorplay::tpx::AnomalyMode::is_enabled);
    autograd.def("is_anomaly_check_nan_enabled", &tensorplay::tpx::AnomalyMode::should_check_nan);
    autograd.def("set_anomaly_enabled",
                 [](bool enabled, bool check_nan) { tensorplay::tpx::AnomalyMode::set_enabled(enabled, check_nan); },
                 "enabled"_a, "check_nan"_a = true);

    // Profiler submodule
    py::module_ profiler = m.def_submodule("profiler", "Profiler");

    // Parallel submodule
    py::module_ parallel = m.def_submodule("parallel", "Parallel computing");

    // Install the anomaly-mode stack capturer: records the Python traceback
    // overrides the C++ backtrace default for the Python engine).
    tensorplay::tpx::set_anomaly_stack_capture([]() -> std::string {
        if (!tensorplay::python_c::interpreter_active()) return {};
        try {
            py::gil_scoped_acquire gil;
            if (!tensorplay::python_c::interpreter_active()) return {};
            auto traceback = py::module_::import("traceback");
            auto stack = traceback.attr("format_stack")();
            std::string out = py::str(stack).cast<std::string>();
            return out;
        } catch (const std::exception&) {
            return {};
        }
    });
}
