#include "Stacktrace.h"
#include <sstream>
#include <vector>
#include <mutex>
#include <iomanip>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#elif defined(__linux__) || defined(__APPLE__)
#include <execinfo.h>
#include <cxxabi.h>
#else
// Other platforms: no stack capture yet
#endif

namespace tensorplay {

namespace {
// Stack capture is opt-in via TENSORPLAY_SHOW_CPP_STACKTRACES (the same switch
// honored by the Python bindings), so throwing never pays symbolization cost
// unless the user asked for traces. The check is cached after first read.
bool stacktrace_enabled() {
    static const bool enabled = []() {
        const char* v = std::getenv("TENSORPLAY_SHOW_CPP_STACKTRACES");
        return v != nullptr && v[0] == '1';
    }();
    return enabled;
}
} // namespace

#ifdef _WIN32

// Helper to initialize symbols only once
struct SymbolHelper {
    HANDLE process;
    SymbolHelper() {
        process = GetCurrentProcess();
        SymInitialize(process, NULL, TRUE);
    }
    ~SymbolHelper() {
        SymCleanup(process);
    }
};

// DbgHelp is single-threaded, so every capture runs under this lock.
static std::mutex& symbol_mutex() {
    static std::mutex mtx;
    return mtx;
}

// The capture itself, with no switch on it; the caller holds symbol_mutex().
static std::string capture_stacktrace() {
    static SymbolHelper* symHelper = new SymbolHelper(); // Initialized once, never destroyed

    void* stack[64];
    unsigned short frames;
    HANDLE process = GetCurrentProcess();

    frames = CaptureStackBackTrace(0, 64, stack, NULL);

    std::ostringstream ss;
    ss << "C++ Stack Trace:\n";

    for (unsigned short i = 0; i < frames; i++) {
        DWORD64 address = (DWORD64)(stack[i]);

        char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
        PSYMBOL_INFO pSymbol = (PSYMBOL_INFO)buffer;
        pSymbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        pSymbol->MaxNameLen = MAX_SYM_NAME;

        DWORD64 displacement = 0;
        if (SymFromAddr(process, address, &displacement, pSymbol)) {
            ss << "  Frame " << i << ": " << pSymbol->Name << " + 0x" << std::hex << displacement << std::dec << "\n";
            
            // Try to get line number
            IMAGEHLP_LINE64 line;
            line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
            DWORD displacementLine = 0;
            if (SymGetLineFromAddr64(process, address, &displacementLine, &line)) {
                ss << "    at " << line.FileName << ":" << line.LineNumber << "\n";
            }
        } else {
            ss << "  Frame " << i << ": [Unknown Address: 0x" << std::hex << address << std::dec << "]\n";
        }
    }
    return ss.str();
}

std::string get_stacktrace() {
    if (!stacktrace_enabled()) {
        return "";
    }
    std::lock_guard<std::mutex> lock(symbol_mutex());
    return capture_stacktrace();
}

std::string capture_stacktrace_unconditional() {
    // A thread that faulted inside DbgHelp still holds the lock; waiting for it
    // would hang the dying process instead of letting it exit.
    std::unique_lock<std::mutex> lock(symbol_mutex(), std::try_to_lock);
    if (!lock.owns_lock()) {
        return "C++ Stack Trace: unavailable, the symbolizer was busy.\n";
    }
    return capture_stacktrace();
}

#else // non-Windows

#if defined(__linux__) || defined(__APPLE__)

// The capture itself, with no switch on it.  The switch exists so that
// ordinary code does not pay for symbolization nobody asked for; a process
// that is already dying has nothing left to protect, and a trace it declines
// to print is the one thing that cannot be recovered afterwards.
static std::string capture_stacktrace() {
    void* frames[64];
    int n = ::backtrace(frames, 64);
    if (n <= 0) {
        return "";
    }
    char** symbols = backtrace_symbols(frames, n);
    if (!symbols) {
        return "";
    }

    std::ostringstream ss;
    ss << "C++ Stack Trace:\n";
    for (int i = 0; i < n; ++i) {
        std::string frame = symbols[i];
        ss << "  Frame " << i << ": ";
        // backtrace_symbols format: "module(mangled_name+0xoffset) [addr]"
        const size_t begin = frame.find('(');
        const size_t end =
            (begin == std::string::npos) ? std::string::npos : frame.find(')', begin);
        if (begin != std::string::npos && end != std::string::npos && end > begin) {
            const size_t plus = frame.find('+', begin + 1);
            const size_t name_len =
                (plus != std::string::npos && plus < end) ? plus - begin - 1 : end - begin - 1;
            const std::string mangled = frame.substr(begin + 1, name_len);
            int status = -1;
            char* demangled =
                abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
            if (status == 0 && demangled != nullptr) {
                ss << demangled;
                std::free(demangled);
                ss << " [" << frame.substr(0, begin) << "]";
            } else if (!mangled.empty()) {
                ss << mangled << " [" << frame.substr(0, begin) << "]";
            } else {
                ss << frame;
            }
        } else {
            ss << frame;
        }
        ss << "\n";
    }
    std::free(symbols);
    return ss.str();
}

std::string get_stacktrace() {
    if (!stacktrace_enabled()) {
        return "";
    }
    return capture_stacktrace();
}

std::string capture_stacktrace_unconditional() {
    return capture_stacktrace();
}

#else

std::string get_stacktrace() {
    if (!stacktrace_enabled()) {
        return "";
    }
    return "Stack trace not implemented for this platform yet.";
}

std::string capture_stacktrace_unconditional() {
    return "Stack trace not implemented for this platform yet.";
}

#endif // __linux__ || __APPLE__

#endif

std::vector<StackFrame> get_stack_frames() {
    std::vector<StackFrame> out;
#if defined(_WIN32)
    // The Windows capture above produces one string for the whole stack; there
    // is no per-frame form to hand back, so an empty result says "nothing
    // captured" rather than inventing frames.
    (void)out;
#elif defined(__linux__) || defined(__APPLE__)
    void* frames[64];
    int n = ::backtrace(frames, 64);
    if (n <= 0) {
        return out;
    }
    char** symbols = backtrace_symbols(frames, n);
    if (!symbols) {
        return out;
    }
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        StackFrame f;
        // backtrace_symbols format: "module(mangled_name+0xoffset) [addr]"
        const std::string text = symbols[i];
        const size_t begin = text.find('(');
        const size_t end =
            (begin == std::string::npos) ? std::string::npos : text.find(')', begin);
        if (begin != std::string::npos && end != std::string::npos && end > begin) {
            f.filename = text.substr(0, begin);
            const size_t plus = text.find('+', begin + 1);
            const size_t name_len = (plus != std::string::npos && plus < end)
                                        ? plus - begin - 1
                                        : end - begin - 1;
            const std::string mangled = text.substr(begin + 1, name_len);
            int status = -1;
            char* demangled =
                abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
            if (status == 0 && demangled != nullptr) {
                f.function = demangled;
                std::free(demangled);
            } else {
                f.function = mangled;
            }
        } else {
            f.function = text;
        }
        out.push_back(std::move(f));
    }
    std::free(symbols);
#endif
    return out;
}

} // namespace tensorplay
