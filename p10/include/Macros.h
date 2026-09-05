#pragma once

#ifdef TP_STATIC_BUILD
    #define P10_API
    #define TENSORPLAY_API
#elif defined(_WIN32)
    #if defined(p10_EXPORTS) || defined(P10_EXPORTS)
        #define P10_API __declspec(dllexport)
    #else
        #define P10_API __declspec(dllimport)
    #endif

    #if defined(TENSORPLAY_EXPORTS)
        #define TENSORPLAY_API __declspec(dllexport)
    #else
        #define TENSORPLAY_API __declspec(dllimport)
    #endif
#else
    #define P10_API
    #define TENSORPLAY_API
#endif

#define TP_CONCAT_IMPL(x, y) x##y
#define TP_CONCAT(x, y) TP_CONCAT_IMPL(x, y)

// Restricted-pointer annotation: MSVC spells it __restrict, GCC/Clang
// spell it __restrict__.
#if defined(_MSC_VER) && !defined(__clang__)
#define TP_RESTRICT __restrict
#else
#define TP_RESTRICT __restrict__
#endif

// Force-inline hint: only GNU-style compilers accept the always_inline
// attribute; MSVC relies on its own inliner without extra decoration.
#if defined(_MSC_VER) && !defined(__clang__)
#define TP_ALWAYS_INLINE inline
#else
#define TP_ALWAYS_INLINE inline __attribute__((always_inline))
#endif
