#include "python_bindings.h"

namespace {

const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::UInt8: return "uint8";
        case DType::Int8: return "int8";
        case DType::Int16: return "int16";
        case DType::Int32: return "int32";
        case DType::Int64: return "int64";
        case DType::UInt16: return "uint16";
        case DType::UInt32: return "uint32";
        case DType::UInt64: return "uint64";
        case DType::Float16: return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::ComplexHalf: return "complex32";
        case DType::ComplexFloat: return "complex64";
        case DType::ComplexDouble: return "complex128";
        case DType::BComplex32: return "bcomplex32";
        case DType::QInt8: return "qint8";
        case DType::QUInt8: return "quint8";
        case DType::QInt32: return "qint32";
        case DType::Bool: return "bool";
        default: return "undefined";
    }
}

std::string dtype_repr(DType dtype) {
    return std::string("tensorplay.") + dtype_name(dtype);
}

} // namespace

void init_dtype(py::module_& m) {
    auto dtype = py::enum_<DType>(m, "DType");
    dtype
        .value("uint8", DType::UInt8)
        .value("int8", DType::Int8)
        .value("int16", DType::Int16)
        .value("int32", DType::Int32)
        .value("int64", DType::Int64)
        .value("uint16", DType::UInt16)
        .value("uint32", DType::UInt32)
        .value("uint64", DType::UInt64)
        .value("float16", DType::Float16)
        .value("bfloat16", DType::BFloat16)
        .value("float32", DType::Float32)
        .value("float64", DType::Float64)
        .value("complex32", DType::ComplexHalf)
        .value("complex64", DType::ComplexFloat)
        .value("complex128", DType::ComplexDouble)
        .value("bcomplex32", DType::BComplex32)
        .value("qint8", DType::QInt8)
        .value("quint8", DType::QUInt8)
        .value("qint32", DType::QInt32)
        .value("bool", DType::Bool)
        .value("undefined", DType::Undefined)
        .def("__str__", [](DType d) { return dtype_repr(d); })
        .def("__repr__", [](DType d) { return dtype_repr(d); })
        .def_property_readonly("is_floating_point", [](DType d) {
            return tensorplay::isFloatingType(d);
        })
        .def_property_readonly("is_complex", [](DType d) {
            return tensorplay::isComplexType(d);
        })
        .def_property_readonly("is_quantized", [](DType d) {
            return tensorplay::isQuantizedType(d);
        })
        .def_property_readonly("is_signed", [](DType d) {
            return tensorplay::isSignedType(d);
        })
        .def_property_readonly("itemsize", [](DType d) {
            return tensorplay::elementSize(d);
        });

    m.attr("uint8") = DType::UInt8;
    m.attr("int8") = DType::Int8;
    m.attr("int16") = DType::Int16;
    m.attr("int32") = DType::Int32;
    m.attr("int64") = DType::Int64;
    m.attr("uint16") = DType::UInt16;
    m.attr("uint32") = DType::UInt32;
    m.attr("uint64") = DType::UInt64;
    m.attr("float16") = DType::Float16;
    m.attr("bfloat16") = DType::BFloat16;
    m.attr("float32") = DType::Float32;
    m.attr("float64") = DType::Float64;
    m.attr("complex32") = DType::ComplexHalf;
    m.attr("complex64") = DType::ComplexFloat;
    m.attr("complex128") = DType::ComplexDouble;
    m.attr("bcomplex32") = DType::BComplex32;
    m.attr("qint8") = DType::QInt8;
    m.attr("quint8") = DType::QUInt8;
    m.attr("qint32") = DType::QInt32;
    m.attr("bool") = DType::Bool;
    m.attr("undefined") = DType::Undefined;

    m.attr("half") = DType::Float16;
    m.attr("float") = DType::Float32;
    m.attr("double") = DType::Float64;
    m.attr("short") = DType::Int16;
    m.attr("int") = DType::Int32;
    m.attr("long") = DType::Int64;
    m.attr("cfloat") = DType::ComplexFloat;
    m.attr("cdouble") = DType::ComplexDouble;
    m.attr("chalf") = DType::ComplexHalf;
}
