#pragma once

#include <Python.h>

namespace tensorplay::python_c {

inline bool interpreter_active() noexcept {
    if (!Py_IsInitialized()) return false;
#if PY_VERSION_HEX >= 0x030D00A1
    return !Py_IsFinalizing();
#else
    return !_Py_IsFinalizing();
#endif
}

}  // namespace tensorplay::python_c
