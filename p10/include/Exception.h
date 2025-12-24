#pragma once

#include <exception>
#include <string>
#include <sstream>
#include <utility>
#include <iostream>
#include "Macros.h"
#include "Stacktrace.h"

namespace tensorplay {

struct SourceLocation {
    const char* file;
    const char* func;
    int line;
};

// Base class for TensorPlay exceptions
class P10_API Exception : public std::exception {
public:
    Exception(SourceLocation source, std::string msg);
    
    const char* what() const noexcept override;
    
    const std::string& msg() const { return msg_; }
    const SourceLocation& source() const { return source_; }
    const std::string& stacktrace() const { return stacktrace_; }

private:
    SourceLocation source_;
    std::string msg_;
    std::string what_;
    std::string stacktrace_;
};

// Derived classes mapping to Python exceptions
class P10_API IndexError : public Exception { using Exception::Exception; };
class P10_API ValueError : public Exception { using Exception::Exception; };
class P10_API TypeError : public Exception { using Exception::Exception; };
class P10_API NotImplementedError : public Exception { using Exception::Exception; };
class P10_API RuntimeError : public Exception { using Exception::Exception; };
class P10_API DeviceMismatchError : public RuntimeError { using RuntimeError::RuntimeError; };

// Warning system
using WarningHandler = void(*)(const SourceLocation& source, const std::string& msg);
P10_API void setWarningHandler(WarningHandler handler);
P10_API void warn(SourceLocation source, const std::string& msg);

namespace detail {
    template <typename T>
    void msg_builder(std::ostream& os, const T& t) {
        os << t;
    }

    template <typename T, typename... Args>
    void msg_builder(std::ostream& os, const T& t, const Args&... args) {
        os << t;
        msg_builder(os, args...);
    }

    template <typename... Args>
    std::string format_msg(const Args&... args) {
        std::ostringstream ss;
        msg_builder(ss, args...);
        return ss.str();
    }
}

} // namespace tensorplay

// Macros
#define TP_THROW(Type, ...) \
    throw tensorplay::Type({__FILE__, __func__, __LINE__}, tensorplay::detail::format_msg(__VA_ARGS__))

#define TP_CHECK(cond, ...) \
    if (!(cond)) { \
        TP_THROW(Exception, "Expected " #cond " to be true, but got false. ", __VA_ARGS__); \
    }

#define TP_CHECK_INDEX(cond, ...) \
    if (!(cond)) { \
        TP_THROW(IndexError, __VA_ARGS__); \
    }

#define TP_CHECK_VALUE(cond, ...) \
    if (!(cond)) { \
        TP_THROW(ValueError, __VA_ARGS__); \
    }

#define TP_CHECK_TYPE(cond, ...) \
    if (!(cond)) { \
        TP_THROW(TypeError, __VA_ARGS__); \
    }

#define TP_CHECK_NOT_IMPLEMENTED(cond, ...) \
    if (!(cond)) { \
        TP_THROW(NotImplementedError, __VA_ARGS__); \
    }

#define TP_WARN(...) \
    tensorplay::warn({__FILE__, __func__, __LINE__}, tensorplay::detail::format_msg(__VA_ARGS__))
