#pragma once

#ifdef TP_STATIC_BUILD
    #define P10_API
    #define TENSORPLAY_API
#elif defined(_WIN32)
    // The single shared library carries both namespaces, so either export
    // marker turns both macros into export form while building it.
    #if defined(p10_EXPORTS) || defined(P10_EXPORTS) || \
        defined(TENSORPLAY_EXPORTS)
        #define P10_API __declspec(dllexport)
        #define TENSORPLAY_API __declspec(dllexport)
    #else
        #define P10_API __declspec(dllimport)
        #define TENSORPLAY_API __declspec(dllimport)
    #endif
#else
    #define P10_API
    #define TENSORPLAY_API
#endif

// Branch-prediction hints: a no-op where the compiler has no equivalent.
// The bool cast is required: the builtin takes a long, and an implicit
// narrowing conversion here would silently change the predicted value.
#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
    #define TP_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
    #define TP_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
    #define TP_LIKELY(expr) (expr)
    #define TP_UNLIKELY(expr) (expr)
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
