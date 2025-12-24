#pragma once

#include <variant>
#include <iostream>
#include <string>
#include <type_traits>
#include <cmath>

#include "DType.h"
#include "Macros.h"
#include "Exception.h"

namespace tensorplay {

// Scalar class aligned with Tensor DType system
class P10_API Scalar {
public:
    Scalar() : type_(DType::Undefined) {}

    // Strict constructors for core types
    // We allow implicit conversions for usability
    
    Scalar(int32_t v) : val_(v), type_(DType::Int32) {}
    Scalar(int64_t v) : val_(v), type_(DType::Int64) {}
    Scalar(float v) : val_(v), type_(DType::Float32) {}
    Scalar(double v) : val_(v), type_(DType::Float64) {}
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
        if (type_ == DType::Int32) return static_cast<double>(std::get<int32_t>(val_));
        if (type_ == DType::Bool) return static_cast<double>(std::get<bool>(val_));
        TP_THROW(RuntimeError, "Scalar is undefined");
    }

    template<typename T>
    T to() const {
        // Strict safe conversion or same-precision conversion
        if (type_ == DType::Float64) {
            double v = std::get<double>(val_);
            if constexpr (std::is_integral_v<T>) {
                if (v != std::floor(v)) TP_THROW(TypeError, "Cannot safely convert non-integer Float64 Scalar to Integral type");
            }
            return static_cast<T>(v);
        } else if (type_ == DType::Float32) {
            float v = std::get<float>(val_);
            if constexpr (std::is_integral_v<T>) {
                if (v != std::floor(v)) TP_THROW(TypeError, "Cannot safely convert non-integer Float32 Scalar to Integral type");
            }
            return static_cast<T>(v);
        } else if (type_ == DType::Int64) {
            int64_t v = std::get<int64_t>(val_);
            // Safe narrowing check for int32
            if constexpr (std::is_same_v<T, int32_t>) {
                if (v > INT32_MAX || v < INT32_MIN) TP_THROW(RuntimeError, "Scalar value overflow for Int32");
            }
            return static_cast<T>(v);
        } else if (type_ == DType::Int32) {
            return static_cast<T>(std::get<int32_t>(val_));
        } else if (type_ == DType::Bool) {
            if constexpr (!std::is_same_v<T, bool>) TP_THROW(TypeError, "Cannot convert Bool Scalar to non-Bool type implicitly");
            return static_cast<T>(std::get<bool>(val_));
        }
        TP_THROW(RuntimeError, "Scalar is undefined");
    }

    // Type checking
    DType dtype() const { return type_; }
    
    bool isFloatingPoint() const {
        return type_ == DType::Float64 || type_ == DType::Float32;
    }

    bool isIntegral(bool includeBool = false) const {
        return type_ == DType::Int64 || type_ == DType::Int32 || (includeBool && type_ == DType::Bool);
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
        if (type_ == DType::Float64) s += std::to_string(std::get<double>(val_));
        else if (type_ == DType::Float32) s += std::to_string(std::get<float>(val_));
        else if (type_ == DType::Int64) s += std::to_string(std::get<int64_t>(val_));
        else if (type_ == DType::Int32) s += std::to_string(std::get<int32_t>(val_));
        else if (type_ == DType::Bool) s += (std::get<bool>(val_) ? "true" : "false");
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
    std::variant<int32_t, int64_t, float, double, bool> val_;
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
