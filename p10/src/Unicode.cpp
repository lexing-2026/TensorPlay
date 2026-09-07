// Definitions for the UTF-8 / UTF-16 helpers declared in Unicode.h.
// Windows only; other platforms keep paths in their native form.

#include "Unicode.h"

#include "Exception.h"

#include <stdexcept>
#include <string>

#if defined(_WIN32)
#include <windows.h>

namespace tensorplay {

std::wstring utf8_to_utf16(const std::string& str) {
    if (str.empty()) {
        return std::wstring();
    }
    // First call measures the destination; the second performs the
    // conversion.  A non-positive measurement means the input is not
    // valid UTF-8 (ERROR_NO_UNICODE_TRANSLATION) or exceeds INT_MAX.
    int size_needed = ::MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, str.data(),
        static_cast<int>(str.size()), nullptr, 0);
    if (size_needed <= 0) {
        TP_THROW(RuntimeError, "Error converting the content to Unicode");
    }
    std::wstring wstr(static_cast<size_t>(size_needed), L'\0');
    int written = ::MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, str.data(),
        static_cast<int>(str.size()), &wstr[0], size_needed);
    if (written != size_needed) {
        TP_THROW(RuntimeError, "Error converting the content to Unicode");
    }
    return wstr;
}

std::string utf16_to_utf8(const std::wstring& wstr) {
    if (wstr.empty()) {
        return std::string();
    }
    int size_needed = ::WideCharToMultiByte(
        CP_UTF8, WC_ERR_INVALID_CHARS, wstr.data(),
        static_cast<int>(wstr.size()), nullptr, 0, nullptr, nullptr);
    if (size_needed <= 0) {
        TP_THROW(RuntimeError, "Error converting the content to UTF8");
    }
    std::string str(static_cast<size_t>(size_needed), '\0');
    int written = ::WideCharToMultiByte(
        CP_UTF8, WC_ERR_INVALID_CHARS, wstr.data(),
        static_cast<int>(wstr.size()), &str[0], size_needed, nullptr, nullptr);
    if (written != size_needed) {
        TP_THROW(RuntimeError, "Error converting the content to UTF8");
    }
    return str;
}

}  // namespace tensorplay

#endif  // _WIN32
