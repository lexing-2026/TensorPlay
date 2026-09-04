#include "python_bindings.h"
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
    if (g_fast_tensor_type)
        return PyObject_TypeCheck(obj, g_fast_tensor_type) != 0;
    bool ok = py::isinstance<Tensor>(py::handle(obj));
    if (ok) g_fast_tensor_type = Py_TYPE(obj);
    return ok;
}

using PyObjectRef = std::shared_ptr<PyObject>;

PyObjectRef retain_pyobject(py::handle object) {
    PyObject* ptr = object.ptr();
    Py_XINCREF(ptr);
    return PyObjectRef(ptr, [](PyObject* value) noexcept {
        if (!value || !Py_IsInitialized()) return;
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
        if (!active_ || !Py_IsInitialized()) return;
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
    bool use_reentrant_ = false;
    bool preserve_rng_state_ = true;
    PyObjectRef replay_context_;
    std::string determinism_check_;
    bool debug_ = false;
    bool early_stop_ = true;
};
} // namespace

// Custom Node for Python-defined Autograd Functions
class PyNode : public tensorplay::tpx::Node {
public:
    PyNode(py::object py_ctx) : py_ctx_(std::move(py_ctx)) {}

    // Backward input slots correspond to forward OUTPUTS for custom
    // this node's incoming gradient buffer by the attached output count.
    size_t num_inputs() const override {
        return output_metas().empty() ? Node::num_inputs()
                                      : output_metas().size();
    }

    tensorplay::tpx::variable_list apply(tensorplay::tpx::variable_list&& inputs) override {
        if (std::getenv("TP_ENGINE_TRACE")) fprintf(stderr, "[tp-engine] PyNode: acquiring GIL\n");
        py::gil_scoped_acquire gil;
        if (std::getenv("TP_ENGINE_TRACE")) fprintf(stderr, "[tp-engine] PyNode: GIL acquired, calling backward\n");

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

        // Call backward on the context object
        if (!py::hasattr(py_ctx_, "backward")) {
             throw std::runtime_error("PyNode context object has no 'backward' method");
        }

        py::object result_obj = py_ctx_.attr("backward")(*py_inputs);
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

    py::object py_ctx_;
public:
    py::object ctx() const { return py_ctx_; }
};

void init_autograd(py::module_& m) {
    py::class_<tensorplay::tpx::Node, std::shared_ptr<tensorplay::tpx::Node>>(m, "Node")
        .def_property_readonly("name", [](const tensorplay::tpx::Node& self) {
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
            "_py_ctx", [](PyNode& self) -> py::object { return self.py_ctx_; },
            "The Python context object this node wraps.")
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
    // AutoGradMode(false) forward block, setup_context and _wrap_outputs
    // all happen inside ONE pybind crossing.  Returns (output, ctx, needs,
    // executable); Python only builds the backward closure afterwards.
    autograd.def("custom_function_apply",
        [](py::object ctx_factory, py::object node_factory,
           py::object forward_fn, std::optional<py::object> setup_ctx_fn,
           py::sequence args) {
            auto ctx = ctx_factory();
            auto node = node_factory(ctx);
            auto* py_node = node.cast<PyNode*>();

            // ---- unpack_input ----
            Py_ssize_t n_args = PyTuple_GET_SIZE(args.ptr());
            py::tuple needs(n_args);
            bool any_rg = false;
            for (Py_ssize_t i = 0; i < n_args; ++i) {
                PyObject* item = PyTuple_GET_ITEM(args.ptr(), i);
                if (fast_is_tensor(item)) {
                    bool rg = py::cast<const Tensor&>(item).requires_grad();
                    any_rg |= rg;
                    needs[i] = py::bool_(rg);
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
                    needs[i] = py::bool_(nested_rg);
                } else {
                    needs[i] = py::bool_(false);
                }
            }

            const bool prev_grad = tensorplay::tpx::GradMode::is_enabled();
            const bool executable = prev_grad && any_rg;
            if (executable) {
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
                py_node->set_materialize_grads(true);
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
                ctx.attr("_outputs") = py::isinstance<py::tuple>(output)
                    ? output
                    : (py::isinstance<py::list>(output)
                           ? py::tuple(output.cast<py::sequence>())
                           : py::make_tuple(output));
                ctx.attr("requires_grad") = true;
                ctx.attr("backward_fn") = py::none();  // set by Python later
            }
            return py::make_tuple(output, ctx, needs, executable,
                                   node);
        },
        "ctx_factory"_a, "node_factory"_a, "forward_fn"_a,
        "setup_ctx_fn"_a.none(), "args"_a);

    autograd.def("backward", [](const std::vector<Tensor>& tensors, std::optional<std::vector<Tensor>> grad_tensors, std::optional<bool> retain_graph, bool create_graph) {
        bool keep_graph = retain_graph.value_or(create_graph);
        std::vector<Tensor> grads;
        if (grad_tensors) grads = *grad_tensors;
        // The engine may evaluate nodes on worker threads that need the GIL
        // for Python-backed autograd functions; the initiating thread must
        // not hold it while it waits for the graph to drain.
        py::gil_scoped_release release;
        tensorplay::tpx::backward(tensors, grads, keep_graph, create_graph);
    }, "tensors"_a, "grad_tensors"_a = py::none(), "retain_graph"_a = py::none(), "create_graph"_a = false);

    autograd.def("queue_callback", [](py::function callback) {
        PyObject* raw_callback = callback.ptr();
        Py_INCREF(raw_callback);
        std::shared_ptr<PyObject> callback_ref(
            raw_callback,
            [](PyObject* object) {
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
        if (!Py_IsInitialized()) return {};
        try {
            py::gil_scoped_acquire gil;
            if (!Py_IsInitialized()) return {};
            auto traceback = py::module_::import("traceback");
            auto stack = traceback.attr("format_stack")();
            std::string out = py::str(stack).cast<std::string>();
            return out;
        } catch (const std::exception&) {
            return {};
        }
    });
}
