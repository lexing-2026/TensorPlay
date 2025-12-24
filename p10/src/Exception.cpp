#include "Exception.h"
#include "Stacktrace.h"
#include <iostream>

namespace tensorplay {

Exception::Exception(SourceLocation source, std::string msg) 
    : source_(source), msg_(std::move(msg)) {
    stacktrace_ = get_stacktrace();
    what_ = msg_;
}

const char* Exception::what() const noexcept {
    return what_.c_str();
}

namespace {
    void defaultWarningHandler(const SourceLocation& source, const std::string& msg) {
        std::cerr << "Warning: " << msg << " (" << source.file << ":" << source.line << ")" << std::endl;
    }
    
    WarningHandler currentWarningHandler = defaultWarningHandler;
}

void setWarningHandler(WarningHandler handler) {
    if (handler) {
        currentWarningHandler = handler;
    } else {
        currentWarningHandler = defaultWarningHandler;
    }
}

void warn(SourceLocation source, const std::string& msg) {
    currentWarningHandler(source, msg);
}

} // namespace tensorplay
