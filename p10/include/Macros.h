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
