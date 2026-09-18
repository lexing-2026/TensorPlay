#include "PythonDispatchModeTLS.h"

#include <atomic>
#include <utility>

#include "Exception.h"
#include "LocalDispatchKeySet.h"

namespace tensorplay::impl {
namespace {

std::atomic<PyObjectReleaseHook> release_hook{nullptr};

thread_local DispatchModeState mode_state;

void sync_python_key() {
    tls_set_dispatch_key_included(DispatchKey::Python, !mode_state.stack.empty());
}

} // namespace

SafePyObject::~SafePyObject() {
    // Without an installed hook the interpreter is gone (or never loaded);
    // leaking the reference is the only safe choice then.
    if (auto hook = release_hook.load(std::memory_order_acquire)) {
        hook(object_);
    }
}

void set_pyobject_release_hook(PyObjectReleaseHook hook) {
    release_hook.store(hook, std::memory_order_release);
}

void DispatchModeTLS::push(DispatchModeHandle mode) {
    TP_CHECK(mode != nullptr, "cannot push a null dispatch mode");
    mode_state.stack.push_back(std::move(mode));
    sync_python_key();
}

DispatchModeHandle DispatchModeTLS::pop() {
    TP_CHECK(!mode_state.stack.empty(),
             "trying to pop from an empty dispatch mode stack");
    DispatchModeHandle mode = std::move(mode_state.stack.back());
    mode_state.stack.pop_back();
    sync_python_key();
    return mode;
}

DispatchModeHandle DispatchModeTLS::get_at(int64_t index) {
    const auto len = static_cast<int64_t>(mode_state.stack.size());
    TP_CHECK(index >= 0 && index < len, "dispatch mode stack index ", index,
             " out of range for a stack of ", len);
    return mode_state.stack[static_cast<size_t>(index)];
}

int64_t DispatchModeTLS::stack_len() {
    return static_cast<int64_t>(mode_state.stack.size());
}

DispatchModeState DispatchModeTLS::get_state() {
    return mode_state;
}

void DispatchModeTLS::set_state(DispatchModeState state) {
    mode_state = std::move(state);
    sync_python_key();
}

DispatchModeStateGuard::DispatchModeStateGuard(DispatchModeState state)
    : saved_(DispatchModeTLS::get_state()) {
    DispatchModeTLS::set_state(std::move(state));
}

DispatchModeStateGuard::~DispatchModeStateGuard() {
    DispatchModeTLS::set_state(std::move(saved_));
}

} // namespace tensorplay::impl
