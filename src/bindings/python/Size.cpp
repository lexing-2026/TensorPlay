// tensorplay.Size -- a C-level tuple subclass.
//
// Shape objects cross every int[] binding path (allocations, conv/reshape
// parameters, runtime-generated kernels), so the type must behave like a
// tuple at the C protocol level: iteration and sequence access resolve
// through the C tuple slots with no Python-level round trip, and
// PySequence_Fast sees a tuple directly (zero-copy).  The rich comparison
// and concatenation keep the historical list/tuple compatibility of the
// previous pybind wrapper.

#include "python_bindings.h"

#include <string>

namespace {

struct SizeObject {
    PyTupleObject tuple;
};

PyObject* SizeTypeObj = nullptr;  // &SizeType as PyObject*

// Slots borrowed from the tuple implementation.  File-scope statics so the
// wrapper templates below can take their addresses as constant expressions.
ssizeargfunc tuple_repeat = PyTuple_Type.tp_as_sequence->sq_repeat;
binaryfunc tuple_subscript = PyTuple_Type.tp_as_mapping->mp_subscript;

inline bool item_is_int(PyObject* item) {
    return PyIndex_Check(item) != 0;
}

PyObject* size_repr(PyObject* self) {
    Py_ssize_t n = PyTuple_GET_SIZE(self);
    std::string repr("tensorplay.Size(");
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (i != 0) repr += ", ";
        repr += std::to_string(PyLong_AsLongLong(PyTuple_GET_ITEM(self, i)));
    }
    repr += ")";
    return PyUnicode_FromString(repr.c_str());
}

// Build a Size from the first `n` items of an already-materialized tuple.
PyObject* size_from_tuple(PyObject* tup) {
    if (Py_TYPE(tup) == reinterpret_cast<PyTypeObject*>(SizeTypeObj)) {
        Py_INCREF(tup);
        return tup;
    }
    return PyObject_CallFunctionObjArgs(SizeTypeObj, tup, nullptr);
}

// Wrap any slot borrowed from the tuple implementation: tuple results come
// back as Size so slicing/concatenation keeps the type.  The borrowed slots
// are resolved through file-scope pointers set at static-init time.
static PyObject* size_repeat(PyObject* self, Py_ssize_t count) {
    PyObject* result = tuple_repeat(self, count);
    if (result == nullptr) return nullptr;
    PyObject* wrapped = size_from_tuple(result);
    Py_DECREF(result);
    return wrapped;
}

static PyObject* size_subscript(PyObject* self, PyObject* key) {
    // tuple's subscript covers int (negative indexing, bounds errors) and
    // slices; only the slice result needs re-wrapping into a Size.
    PyObject* result = tuple_subscript(self, key);
    if (result == nullptr) return nullptr;
    PyObject* wrapped = size_from_tuple(result);
    Py_DECREF(result);
    return wrapped;
}

PyObject* size_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    PyObject* self = PyTuple_Type.tp_new(type, args, kwargs);
    if (self == nullptr) return nullptr;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(self); ++i) {
        PyObject* item = PyTuple_GET_ITEM(self, i);
        if (item_is_int(item)) continue;
        PyObject* number = PyNumber_Index(item);
        if (number != nullptr && item_is_int(number)) {
            PyTuple_SET_ITEM(self, i, number);  // steals the reference
            continue;
        }
        Py_XDECREF(number);
        PyErr_Format(
            PyExc_TypeError,
            "tensorplay.Size() takes an iterable of 'int' (item %zd is '%s')",
            i, Py_TYPE(item)->tp_name);
        Py_DECREF(self);
        return nullptr;
    }
    return self;
}

PyObject* size_concat(PyObject* left, PyObject* right) {
    if (!PySequence_Check(right)) {
        PyErr_Format(
            PyExc_TypeError, "can only concatenate sequence (not '%s') to tensorplay.Size",
            Py_TYPE(right)->tp_name);
        return nullptr;
    }
    PyObject* seq = PySequence_Fast(right, nullptr);
    if (seq == nullptr) {
        PyErr_Clear();
        PyErr_Format(
            PyExc_TypeError, "can only concatenate sequence (not '%s') to tensorplay.Size",
            Py_TYPE(right)->tp_name);
        return nullptr;
    }
    Py_ssize_t ln = PyTuple_GET_SIZE(left);
    Py_ssize_t rn = PySequence_Fast_GET_SIZE(seq);
    PyObject* combined = PyTuple_New(ln + rn);
    if (combined == nullptr) {
        Py_DECREF(seq);
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < ln; ++i) {
        PyObject* item = PyTuple_GET_ITEM(left, i);
        Py_INCREF(item);
        PyTuple_SET_ITEM(combined, i, item);
    }
    for (Py_ssize_t i = 0; i < rn; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        if (!item_is_int(item)) {
            Py_DECREF(combined);
            Py_DECREF(seq);
            PyErr_Format(
                PyExc_TypeError,
                "can only concatenate an iterable of 'int' (item %zd is '%s') "
                "to tensorplay.Size",
                i, Py_TYPE(item)->tp_name);
            return nullptr;
        }
        PyObject* number = PyNumber_Index(item);
        if (number == nullptr) {
            Py_DECREF(combined);
            Py_DECREF(seq);
            return nullptr;
        }
        PyTuple_SET_ITEM(combined, ln + i, number);
    }
    Py_DECREF(seq);
    PyObject* wrapped = size_from_tuple(combined);
    Py_DECREF(combined);
    return wrapped;
}

// The interpreter probes right.nb_add first only for subclass instances;
// `list + Size` and `tuple + Size` both land here through the number slot,
// matching the historical reflected-addition surface of the pybind wrapper.
PyObject* size_add(PyObject* left, PyObject* right) {
    if (!PySequence_Check(left) || !PySequence_Check(right)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return size_concat(left, right);
}

PyNumberMethods size_as_number = {
    &size_add,   /* nb_add */
    nullptr,     /* nb_subtract */
    nullptr,     /* nb_multiply */
    nullptr,     /* nb_remainder */
    nullptr,     /* nb_divmod */
    nullptr,     /* nb_power */
    nullptr,     /* nb_negative */
    nullptr,     /* nb_positive */
    nullptr,     /* nb_absolute */
    nullptr,     /* nb_bool */
    nullptr,     /* nb_invert */
    nullptr,     /* nb_lshift */
    nullptr,     /* nb_rshift */
    nullptr,     /* nb_and */
    nullptr,     /* nb_xor */
    nullptr,     /* nb_or */
    nullptr,     /* nb_int */
    nullptr,     /* nb_reserved */
    nullptr,     /* nb_float */
    nullptr,     /* nb_inplace_add */
    nullptr,     /* nb_inplace_subtract */
    nullptr,     /* nb_inplace_multiply */
    nullptr,     /* nb_inplace_remainder */
    nullptr,     /* nb_inplace_power */
    nullptr,     /* nb_inplace_lshift */
    nullptr,     /* nb_inplace_rshift */
    nullptr,     /* nb_inplace_and */
    nullptr,     /* nb_inplace_xor */
    nullptr,     /* nb_inplace_or */
    nullptr,     /* nb_floor_divide */
    nullptr,     /* nb_true_divide */
    nullptr,     /* nb_inplace_floor_divide */
    nullptr,     /* nb_inplace_true_divide */
    nullptr,     /* nb_index */
    nullptr,     /* nb_matrix_multiply */
    nullptr,     /* nb_inplace_matrix_multiply */
};

PySequenceMethods size_as_sequence = {
    nullptr,       /* sq_length (inherited) */
    &size_concat,  /* sq_concat */
    &size_repeat,  /* sq_repeat */
    nullptr,       /* sq_item (inherited) */
    nullptr,       /* was_sq_slice */
    nullptr,       /* sq_ass_item */
    nullptr,       /* was_sq_ass_slice */
    nullptr,       /* sq_contains (inherited) */
};

PyMappingMethods size_as_mapping = {
    nullptr,        /* mp_length (inherited) */
    &size_subscript, /* mp_subscript */
    nullptr,        /* mp_ass_subscript (immutable) */
};

PyObject* size_richcompare(PyObject* self, PyObject* other, int op) {
    // Tuple semantics for tuple/Size operands; a list operand keeps the
    // historical elementwise equality of the pybind wrapper.
    if (PyList_Check(other) && (op == Py_EQ || op == Py_NE)) {
        PyObject* as_tuple = PyList_AsTuple(other);
        if (as_tuple == nullptr) return nullptr;
        PyObject* result = PyTuple_Type.tp_richcompare(self, as_tuple, op);
        Py_DECREF(as_tuple);
        return result;
    }
    return PyTuple_Type.tp_richcompare(self, other, op);
}

PyObject* size_numel(PyObject* self, PyObject*) {
    Py_ssize_t n = PyTuple_GET_SIZE(self);
    long long numel = 1;
    for (Py_ssize_t i = 0; i < n; ++i) {
        numel *= PyLong_AsLongLong(PyTuple_GET_ITEM(self, i));
        if (numel == -1 && PyErr_Occurred()) return nullptr;
    }
    return PyLong_FromLongLong(numel);
}

PyObject* size_reduce(PyObject* self, PyObject*) {
    Py_ssize_t n = PyTuple_GET_SIZE(self);
    PyObject* items = PyTuple_New(n);
    if (items == nullptr) return nullptr;
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PyTuple_GET_ITEM(self, i);
        Py_INCREF(item);
        PyTuple_SET_ITEM(items, i, item);
    }
    PyObject* result = Py_BuildValue("(O(O))", SizeTypeObj, items);
    Py_DECREF(items);
    return result;
}

PyMethodDef size_methods[] = {
    {"numel", size_numel, METH_NOARGS, "Product of the dimensions."},
    {"__reduce__", size_reduce, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject SizeType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "tensorplay.Size",       /* tp_name */
    sizeof(SizeObject),      /* tp_basicsize */
    0,                       /* tp_itemsize */
    nullptr,                 /* tp_dealloc (tuple's, inherited) */
    0,                       /* tp_vectorcall_offset */
    nullptr,                 /* tp_getattr */
    nullptr,                 /* tp_setattr */
    nullptr,                 /* tp_as_async */
    size_repr,               /* tp_repr */
    &size_as_number,         /* tp_as_number */
    &size_as_sequence,       /* tp_as_sequence */
    &size_as_mapping,        /* tp_as_mapping */
    nullptr,                 /* tp_hash (tuple's, inherited) */
    nullptr,                 /* tp_call */
    nullptr,                 /* tp_str */
    nullptr,                 /* tp_getattro */
    nullptr,                 /* tp_setattro */
    nullptr,                 /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,      /* tp_flags */
    "Sequence of tensor dimension sizes (tuple subclass).", /* tp_doc */
    nullptr,                 /* tp_traverse */
    nullptr,                 /* tp_clear */
    size_richcompare,        /* tp_richcompare */
    0,                       /* tp_weaklistoffset */
    nullptr,                 /* tp_iter (tuple's, inherited) */
    nullptr,                 /* tp_iternext */
    size_methods,            /* tp_methods */
    nullptr,                 /* tp_members */
    nullptr,                 /* tp_getset */
    &PyTuple_Type,           /* tp_base */
    nullptr,                 /* tp_dict */
    nullptr,                 /* tp_descr_get */
    nullptr,                 /* tp_descr_set */
    0,                       /* tp_dictoffset */
    nullptr,                 /* tp_init */
    nullptr,                 /* tp_alloc */
    size_new,                /* tp_new */
};

}  // namespace

PyObject* Size_NewFromSizes(Py_ssize_t dim, const int64_t* sizes) {
    PyObject* self = SizeType.tp_alloc(&SizeType, dim);
    if (self == nullptr) return nullptr;
    for (Py_ssize_t i = 0; i < dim; ++i) {
        PyObject* item = PyLong_FromLongLong(sizes[i]);
        if (item == nullptr) {
            Py_DECREF(self);
            return nullptr;
        }
        PyTuple_SET_ITEM(self, i, item);
    }
    return self;
}

PyObject* Size_New(const tensorplay::Size& size) {
    return Size_NewFromSizes(
        static_cast<Py_ssize_t>(size.size()), size.data());
}

bool Size_Check(PyObject* obj) {
    return Py_TYPE(obj) == reinterpret_cast<PyTypeObject*>(SizeTypeObj);
}

void init_size(py::module_& m) {
    if (PyType_Ready(&SizeType) < 0) throw py::error_already_set();
    SizeTypeObj = reinterpret_cast<PyObject*>(&SizeType);
    Py_INCREF(SizeTypeObj);
    if (PyModule_AddObject(m.ptr(), "Size", SizeTypeObj) < 0) {
        Py_DECREF(SizeTypeObj);
        throw py::error_already_set();
    }
}
