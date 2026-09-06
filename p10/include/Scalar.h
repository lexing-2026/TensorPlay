#pragma once

#include <variant>
#include <iostream>
#include <string>
#include <type_traits>
#include <cmath>
#include <complex>
#include <sstream>

#include "DType.h"
#include "Complex.h"
#include "Macros.h"
#include "Exception.h"

namespace tensorplay {

// Scalar class integrated with the Tensor DType system
class P10_API Scalar {
public:
    Scalar() : type_(DType::Undefined) {}

    // Strict constructors for core types
    // We allow implicit conversions for usability
    
    Scalar(int32_t v) : val_(v), type_(DType::Int32) {}
    Scalar(int64_t v) : val_(v), type_(DType::Int64) {}
    Scalar(uint64_t v) : val_(v), type_(DType::UInt64) {}
    Scalar(float v) : val_(v), type_(DType::Float32) {}
    Scalar(double v) : val_(v), type_(DType::Float64) {}
    Scalar(std::complex<float> v) : val_(v), type_(DType::ComplexFloat) {}
    Scalar(std::complex<double> v) : val_(v), type_(DType::ComplexDouble) {}
    Scalar(complex<float> v)
        : val_(static_cast<std::complex<float>>(v)), type_(DType::ComplexFloat) {}
    Scalar(complex<double> v)
        : val_(static_cast<std::complex<double>>(v)), type_(DType::ComplexDouble) {}
    Scalar(bool v) : val_(v), type_(DType::Bool) {}

    // Copy/Move
    Scalar(const Scalar&) = default;
    Scalar(Scalar&&) = default;
    Scalar& operator=(const Scalar&) = default;
    Scalar& operator=(Scalar&&) = default;
    ~Scalar() {
        // std::cout << "Scalar destructor called for " << toString() << std::endl;
    }

    // Accessors
    double toDouble() const {
        if (type_ == DType::Float64) return std::get<double>(val_);
        if (type_ == DType::Float32) return static_cast<double>(std::get<float>(val_));
        if (type_ == DType::Int64) return static_cast<double>(std::get<int64_t>(val_));
        if (type_ == DType::UInt64) return static_cast<double>(std::get<uint64_t>(val_));
        if (type_ == DType::Int32) return static_cast<double>(std::get<int32_t>(val_));
        if (type_ == DType::Bool) return static_cast<double>(std::get<bool>(val_));
        if (isComplexType(type_)) TP_THROW(TypeError, "Cannot convert a complex Scalar to double");
        TP_THROW(RuntimeError, "Scalar is undefined");
    }

    template<typename T>
    T to() const {
        if (type_ == DType::Undefined) {
            TP_THROW(RuntimeError, "Scalar is undefined");
        }

        return std::visit([](const auto& value) -> T {
            using source_t = std::decay_t<decltype(value)>;
            if constexpr (is_complex_type_v<T>) {
                using target_value_t = typename is_complex_type<T>::value_type;
                if constexpr (is_complex_type_v<source_t>) {
                    return T(static_cast<target_value_t>(value.real()),
                             static_cast<target_value_t>(value.imag()));
                } else {
                    return T(static_cast<target_value_t>(value), target_value_t(0));
                }
            } else if constexpr (is_complex_type_v<source_t>) {
                // the real component and discards the imaginary component.
                return static_cast<T>(value.real());
            } else {
                return static_cast<T>(value);
            }
        }, val_);
    }

    // Type checking
    DType dtype() const { return type_; }
    
    bool isFloatingPoint() const {
        return isFloatingType(type_);
    }

    bool isComplex() const {
        return isComplexType(type_);
    }

    bool isIntegral(bool includeBool = false) const {
        return isIntegralType(type_, includeBool);
    }
    
    bool isBoolean() const {
        return type_ == DType::Bool;
    }
    
    bool is_dtype(DType dt) const {
        return type_ == dt;
    }

    // String representation
    std::string toString() const {
        if (type_ == DType::Undefined) return "Scalar(Undefined)";
        std::string s = "Scalar(";
        std::ostringstream value_stream;
        if (type_ == DType::Bool) {
            value_stream << (std::get<bool>(val_) ? "true" : "false");
        } else {
            std::visit([&value_stream](const auto& value) { value_stream << value; }, val_);
        }
        s += value_stream.str();
        s += ", dtype=";
        s += ::tensorplay::toString(type_);
        s += ")";
        return s;
    }

    // Operators
    Scalar operator+(const Scalar& other) const;
    Scalar operator-(const Scalar& other) const;
    Scalar operator*(const Scalar& other) const;
    Scalar operator/(const Scalar& other) const;
    
    bool operator==(const Scalar& other) const;
    bool operator!=(const Scalar& other) const;
    bool operator>(const Scalar& other) const;
    bool operator<(const Scalar& other) const;

private:
    std::variant<int32_t, int64_t, uint64_t, float, double, bool,
                 std::complex<float>, std::complex<double>> val_;
    DType type_;
    
    // Helper for promotion
    static DType promote_types(DType a, DType b);
};

// Stream operator
inline std::ostream& operator<<(std::ostream& os, const Scalar& s) {
    os << s.toString();
    return os;
}

template <>
struct TypeTraits<Scalar> {
    static constexpr ScalarType scalar_type = ScalarType::Undefined;
    static constexpr ScalarType dtype = ScalarType::Undefined;
};

} // namespace tensorplay
