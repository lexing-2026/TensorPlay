#include "python_bindings.h"

void init_dtype(nb::module_& m) {
    nb::enum_<DType>(m, "DType")
        .value("float32", DType::Float32)
        .value("float64", DType::Float64)
        .value("int32", DType::Int32)
        .value("int64", DType::Int64)
        .value("uint8", DType::UInt8)
        .value("int8", DType::Int8)
        .value("int16", DType::Int16)
        .value("uint16", DType::UInt16)
        .value("uint32", DType::UInt32)
        .value("uint64", DType::UInt64)
        .value("bool", DType::Bool)
        .value("undefined", DType::Undefined)
        .def("__str__", [](DType d) {
            switch(d) {
                case DType::Float32: return "tensorplay.float32";
                case DType::Float64: return "tensorplay.float64";
                case DType::Int32: return "tensorplay.int32";
                case DType::Int64: return "tensorplay.int64";
                case DType::UInt8: return "tensorplay.uint8";
                case DType::Int8: return "tensorplay.int8";
                case DType::Int16: return "tensorplay.int16";
                case DType::UInt16: return "tensorplay.uint16";
                case DType::UInt32: return "tensorplay.uint32";
                case DType::UInt64: return "tensorplay.uint64";
                case DType::Bool: return "tensorplay.bool";
                default: return "tensorplay.undefined";
            }
        })
        .def("__repr__", [](DType d) {
            switch(d) {
                case DType::Float32: return "tensorplay.float32";
                case DType::Float64: return "tensorplay.float64";
                case DType::Int32: return "tensorplay.int32";
                case DType::Int64: return "tensorplay.int64";
                case DType::UInt8: return "tensorplay.uint8";
                case DType::Int8: return "tensorplay.int8";
                case DType::Int16: return "tensorplay.int16";
                case DType::UInt16: return "tensorplay.uint16";
                case DType::UInt32: return "tensorplay.uint32";
                case DType::UInt64: return "tensorplay.uint64";
                case DType::Bool: return "tensorplay.bool";
                default: return "tensorplay.undefined";
            }
        })
        .export_values();
}
