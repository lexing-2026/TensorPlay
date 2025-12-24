#include "Stacktrace.h"
#include <sstream>
#include <vector>
#include <mutex>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#else
// Add Linux/Mac implementation later
#endif

namespace tensorplay {

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

#else

std::string get_stacktrace() {
    // TODO: Add Linux/Mac implementation
    return "Stack trace not implemented for non-Windows platforms yet.";
}

#endif

} // namespace tensorplay
