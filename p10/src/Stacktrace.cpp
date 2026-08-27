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

std::string get_stacktrace() {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    if (!stacktrace_enabled()) {
        return "";
    }
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

#else // non-Windows

#if defined(__linux__) || defined(__APPLE__)

std::string get_stacktrace() {
    if (!stacktrace_enabled()) {
        return "";
    }
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

#else

std::string get_stacktrace() {
    if (!stacktrace_enabled()) {
        return "";
    }
    return "Stack trace not implemented for this platform yet.";
}

#endif // __linux__ || __APPLE__

#endif

} // namespace tensorplay
