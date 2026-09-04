#pragma once

#include <Python.h>

// Py_NewRef arrived in 3.10; the extension still builds against 3.9.
#if PY_VERSION_HEX < 0x030A0000
inline PyObject* Py_NewRef(PyObject* o) {
    Py_INCREF(o);
    return o;
}
#endif
