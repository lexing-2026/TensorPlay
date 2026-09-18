#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "Macros.h"

namespace tensorplay::impl {

// Owning reference to a Python object that may be dropped on any thread.
// p10 never calls into Python: the Python layer installs the release hook,
// which acquires the interpreter lock before releasing the reference.
class P10_API SafePyObject {
public:
    explicit SafePyObject(void* object) noexcept : object_(object) {}
    ~SafePyObject();

    SafePyObject(const SafePyObject&) = delete;
    SafePyObject& operator=(const SafePyObject&) = delete;

    // Borrowed pointer; valid while this holder is alive.
    void* get() const noexcept { return object_; }

private:
    void* object_;
};

using PyObjectReleaseHook = void (*)(void*);

// Installed once by the Python extension at import.
P10_API void set_pyobject_release_hook(PyObjectReleaseHook hook);

using DispatchModeHandle = std::shared_ptr<SafePyObject>;

// Snapshot of the dispatch-mode stack, carried across threads by the
// autograd engine so backward operators reach the modes active when
// backward was requested.
struct P10_API DispatchModeState {
    std::vector<DispatchModeHandle> stack;
};

// Thread-local stack of Python dispatch modes (innermost last).  The Python
// dispatch key is included in the thread's local key set exactly while the
// stack is non-empty.
class P10_API DispatchModeTLS {
public:
    static void push(DispatchModeHandle mode);
    // Removes and returns the innermost mode; throws when the stack is empty.
    static DispatchModeHandle pop();
    static DispatchModeHandle get_at(int64_t index);
    static int64_t stack_len();

    static DispatchModeState get_state();
    static void set_state(DispatchModeState state);

private:
    DispatchModeTLS() = delete;
};

// Installs a mode-stack snapshot for a scope and restores the previous one.
class P10_API DispatchModeStateGuard {
public:
    explicit DispatchModeStateGuard(DispatchModeState state);
    ~DispatchModeStateGuard();

    DispatchModeStateGuard(const DispatchModeStateGuard&) = delete;
    DispatchModeStateGuard& operator=(const DispatchModeStateGuard&) = delete;

private:
    DispatchModeState saved_;
};

} // namespace tensorplay::impl
