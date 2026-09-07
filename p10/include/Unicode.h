// Cross-platform UTF-8 / UTF-16 conversion helpers.
//
// Windows file APIs come in narrow (ANSI codepage) and wide (UTF-16) forms;
// the wide forms are the only ones that address files whose names fall
// outside the active ANSI codepage.  Paths held by the framework are UTF-8,
// so every Win32 call that receives a user-visible name goes through
// utf8_to_utf16 first (and utf16_to_utf8 on the way out).
//
// The conversion is lossless for any valid UTF-8 input: UTF-16 surrogate
// pairs cover the full scalar range.  Malformed sequences fail with a
// runtime error rather than being silently replaced.

#pragma once

#include "Macros.h"

#include <string>

namespace tensorplay {

#if defined(_WIN32)

// Converts a UTF-8 encoded string to a UTF-16 wide string.
// Throws std::runtime_error when the input is not valid UTF-8.
P10_API std::wstring utf8_to_utf16(const std::string& str);

// Converts a UTF-16 wide string back to a UTF-8 encoded string.
// Throws std::runtime_error when the conversion fails.
P10_API std::string utf16_to_utf8(const std::wstring& wstr);

#endif  // _WIN32

}  // namespace tensorplay
