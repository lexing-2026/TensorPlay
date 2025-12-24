#pragma once

#ifdef TP_STATIC_BUILD
    #define STAX_API
#elif defined(_WIN32)
    #if defined(stax_EXPORTS)
        #define STAX_API __declspec(dllexport)
    #else
        #define STAX_API __declspec(dllimport)
    #endif
#else
    #define STAX_API
#endif
