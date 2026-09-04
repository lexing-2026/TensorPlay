#pragma once

// Slim declaration layer for the vendored SLEEF vector math library.
//
// The generated sleef.h gates every per-ISA declaration behind the ISA
// predefined macros of the translating TU (__AVX__, __AVX512F__, ...), so it
// cannot be included from translation units that mix capability levels or
// rely on per-function target attributes.  p10 needs exactly that mixing,
// so the entry points used here are declared directly.
//
// The non-suffixed symbols below are runtime CPU dispatchers compiled into
// libsleef; they pick the AVX-512/AVX2 implementation for the running CPU.
// No ISA macros are required to declare or call them.
//
// Precision tiers follow the reference vec layer: u10 for most functions,
// u35 for single-precision sin/cos, u15 for erfc and u05 for hypot.
// Gate for consumers: the SIMD call sites themselves must be compiled only
// where the surrounding code already guarantees AVX2/AVX-512 support
// (capability TUs or target-attributed functions).

#include "cpu/vec/Intrinsics.h"

#if defined(_MSC_VER) && defined(_M_X64)
#define TP_SLEEF_CC __vectorcall
#else
#define TP_SLEEF_CC
#endif

// The vector-math entry points exist only on the x86-64 paths; other
// architectures keep the file includable and compile the scalar fallbacks.
#if defined(__x86_64__) || defined(_M_X64)

#ifdef __cplusplus
extern "C" {
#endif

// __m256 (f8)
__m256 TP_SLEEF_CC Sleef_expf8_u10(__m256);
__m256 TP_SLEEF_CC Sleef_exp2f8_u10(__m256);
__m256 TP_SLEEF_CC Sleef_expm1f8_u10(__m256);
__m256 TP_SLEEF_CC Sleef_logf8_u10(__m256);
__m256 TP_SLEEF_CC Sleef_log2f8_u10(__m256);
__m256 TP_SLEEF_CC Sleef_log10f8_u10(__m256);
__m256 TP_SLEEF_CC Sleef_log1pf8_u10(__m256);
__m256 TP_SLEEF_CC Sleef_sinf8_u35(__m256);
__m256 TP_SLEEF_CC Sleef_cosf8_u35(__m256);
__m256 TP_SLEEF_CC Sleef_tanf8_u10(__m256);
__m256 TP_SLEEF_CC Sleef_asinf8_u10(__m256);
__m256 TP_SLEEF_CC Sleef_acosf8_u10(__m256);
__m256 TP_SLEEF_CC Sleef_atanf8_u10(__m256);
__m256 TP_SLEEF_CC Sleef_atan2f8_u10(__m256, __m256);
__m256 TP_SLEEF_CC Sleef_sinhf8_u10(__m256);
__m256 TP_SLEEF_CC Sleef_coshf8_u10(__m256);
__m256 TP_SLEEF_CC Sleef_tanhf8_u10(__m256);
__m256 TP_SLEEF_CC Sleef_asinhf8_u10(__m256);
__m256 TP_SLEEF_CC Sleef_acoshf8_u10(__m256);
__m256 TP_SLEEF_CC Sleef_atanhf8_u10(__m256);
__m256 TP_SLEEF_CC Sleef_erff8_u10(__m256);
__m256 TP_SLEEF_CC Sleef_erfcf8_u15(__m256);
__m256 TP_SLEEF_CC Sleef_powf8_u10(__m256, __m256);
__m256 TP_SLEEF_CC Sleef_hypotf8_u05(__m256, __m256);

// __m256d (d4)
__m256d TP_SLEEF_CC Sleef_expd4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_exp2d4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_expm1d4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_logd4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_log2d4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_log10d4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_log1pd4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_sind4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_cosd4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_tand4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_asind4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_acosd4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_atand4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_atan2d4_u10(__m256d, __m256d);
__m256d TP_SLEEF_CC Sleef_sinhd4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_coshd4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_tanhd4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_asinhd4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_acoshd4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_atanhd4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_erfd4_u10(__m256d);
__m256d TP_SLEEF_CC Sleef_erfcd4_u15(__m256d);
__m256d TP_SLEEF_CC Sleef_powd4_u10(__m256d, __m256d);
__m256d TP_SLEEF_CC Sleef_hypotd4_u05(__m256d, __m256d);

// __m512 (f16)
__m512 TP_SLEEF_CC Sleef_expf16_u10(__m512);
__m512 TP_SLEEF_CC Sleef_exp2f16_u10(__m512);
__m512 TP_SLEEF_CC Sleef_expm1f16_u10(__m512);
__m512 TP_SLEEF_CC Sleef_logf16_u10(__m512);
__m512 TP_SLEEF_CC Sleef_log2f16_u10(__m512);
__m512 TP_SLEEF_CC Sleef_log10f16_u10(__m512);
__m512 TP_SLEEF_CC Sleef_log1pf16_u10(__m512);
__m512 TP_SLEEF_CC Sleef_sinf16_u35(__m512);
__m512 TP_SLEEF_CC Sleef_cosf16_u35(__m512);
__m512 TP_SLEEF_CC Sleef_tanf16_u10(__m512);
__m512 TP_SLEEF_CC Sleef_asinf16_u10(__m512);
__m512 TP_SLEEF_CC Sleef_acosf16_u10(__m512);
__m512 TP_SLEEF_CC Sleef_atanf16_u10(__m512);
__m512 TP_SLEEF_CC Sleef_atan2f16_u10(__m512, __m512);
__m512 TP_SLEEF_CC Sleef_sinhf16_u10(__m512);
__m512 TP_SLEEF_CC Sleef_coshf16_u10(__m512);
__m512 TP_SLEEF_CC Sleef_tanhf16_u10(__m512);
__m512 TP_SLEEF_CC Sleef_asinhf16_u10(__m512);
__m512 TP_SLEEF_CC Sleef_acoshf16_u10(__m512);
__m512 TP_SLEEF_CC Sleef_atanhf16_u10(__m512);
__m512 TP_SLEEF_CC Sleef_erff16_u10(__m512);
__m512 TP_SLEEF_CC Sleef_erfcf16_u15(__m512);
__m512 TP_SLEEF_CC Sleef_powf16_u10(__m512, __m512);
__m512 TP_SLEEF_CC Sleef_hypotf16_u05(__m512, __m512);

// __m512d (d8)
__m512d TP_SLEEF_CC Sleef_expd8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_exp2d8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_expm1d8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_logd8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_log2d8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_log10d8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_log1pd8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_sind8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_cosd8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_tand8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_asind8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_acosd8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_atand8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_atan2d8_u10(__m512d, __m512d);
__m512d TP_SLEEF_CC Sleef_sinhd8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_coshd8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_tanhd8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_asinhd8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_acoshd8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_atanhd8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_erfd8_u10(__m512d);
__m512d TP_SLEEF_CC Sleef_erfcd8_u15(__m512d);
__m512d TP_SLEEF_CC Sleef_powd8_u10(__m512d, __m512d);
__m512d TP_SLEEF_CC Sleef_hypotd8_u05(__m512d, __m512d);

#ifdef __cplusplus
} // extern "C"
#endif

namespace tensorplay {
namespace tpsleef {
inline __m256 exp(__m256 a) { return Sleef_expf8_u10(a); }
inline __m256d exp(__m256d a) { return Sleef_expd4_u10(a); }
inline __m512 exp(__m512 a) { return Sleef_expf16_u10(a); }
inline __m512d exp(__m512d a) { return Sleef_expd8_u10(a); }

inline __m256 exp2(__m256 a) { return Sleef_exp2f8_u10(a); }
inline __m256d exp2(__m256d a) { return Sleef_exp2d4_u10(a); }
inline __m512 exp2(__m512 a) { return Sleef_exp2f16_u10(a); }
inline __m512d exp2(__m512d a) { return Sleef_exp2d8_u10(a); }

inline __m256 expm1(__m256 a) { return Sleef_expm1f8_u10(a); }
inline __m256d expm1(__m256d a) { return Sleef_expm1d4_u10(a); }
inline __m512 expm1(__m512 a) { return Sleef_expm1f16_u10(a); }
inline __m512d expm1(__m512d a) { return Sleef_expm1d8_u10(a); }

inline __m256 log(__m256 a) { return Sleef_logf8_u10(a); }
inline __m256d log(__m256d a) { return Sleef_logd4_u10(a); }
inline __m512 log(__m512 a) { return Sleef_logf16_u10(a); }
inline __m512d log(__m512d a) { return Sleef_logd8_u10(a); }

inline __m256 log2(__m256 a) { return Sleef_log2f8_u10(a); }
inline __m256d log2(__m256d a) { return Sleef_log2d4_u10(a); }
inline __m512 log2(__m512 a) { return Sleef_log2f16_u10(a); }
inline __m512d log2(__m512d a) { return Sleef_log2d8_u10(a); }

inline __m256 log10(__m256 a) { return Sleef_log10f8_u10(a); }
inline __m256d log10(__m256d a) { return Sleef_log10d4_u10(a); }
inline __m512 log10(__m512 a) { return Sleef_log10f16_u10(a); }
inline __m512d log10(__m512d a) { return Sleef_log10d8_u10(a); }

inline __m256 log1p(__m256 a) { return Sleef_log1pf8_u10(a); }
inline __m256d log1p(__m256d a) { return Sleef_log1pd4_u10(a); }
inline __m512 log1p(__m512 a) { return Sleef_log1pf16_u10(a); }
inline __m512d log1p(__m512d a) { return Sleef_log1pd8_u10(a); }

inline __m256 sin(__m256 a) { return Sleef_sinf8_u35(a); }
inline __m256d sin(__m256d a) { return Sleef_sind4_u10(a); }
inline __m512 sin(__m512 a) { return Sleef_sinf16_u35(a); }
inline __m512d sin(__m512d a) { return Sleef_sind8_u10(a); }

inline __m256 cos(__m256 a) { return Sleef_cosf8_u35(a); }
inline __m256d cos(__m256d a) { return Sleef_cosd4_u10(a); }
inline __m512 cos(__m512 a) { return Sleef_cosf16_u35(a); }
inline __m512d cos(__m512d a) { return Sleef_cosd8_u10(a); }

inline __m256 tan(__m256 a) { return Sleef_tanf8_u10(a); }
inline __m256d tan(__m256d a) { return Sleef_tand4_u10(a); }
inline __m512 tan(__m512 a) { return Sleef_tanf16_u10(a); }
inline __m512d tan(__m512d a) { return Sleef_tand8_u10(a); }

inline __m256 asin(__m256 a) { return Sleef_asinf8_u10(a); }
inline __m256d asin(__m256d a) { return Sleef_asind4_u10(a); }
inline __m512 asin(__m512 a) { return Sleef_asinf16_u10(a); }
inline __m512d asin(__m512d a) { return Sleef_asind8_u10(a); }

inline __m256 acos(__m256 a) { return Sleef_acosf8_u10(a); }
inline __m256d acos(__m256d a) { return Sleef_acosd4_u10(a); }
inline __m512 acos(__m512 a) { return Sleef_acosf16_u10(a); }
inline __m512d acos(__m512d a) { return Sleef_acosd8_u10(a); }

inline __m256 atan(__m256 a) { return Sleef_atanf8_u10(a); }
inline __m256d atan(__m256d a) { return Sleef_atand4_u10(a); }
inline __m512 atan(__m512 a) { return Sleef_atanf16_u10(a); }
inline __m512d atan(__m512d a) { return Sleef_atand8_u10(a); }

inline __m256 atan2(__m256 a, __m256 b) { return Sleef_atan2f8_u10(a, b); }
inline __m256d atan2(__m256d a, __m256d b) { return Sleef_atan2d4_u10(a, b); }
inline __m512 atan2(__m512 a, __m512 b) { return Sleef_atan2f16_u10(a, b); }
inline __m512d atan2(__m512d a, __m512d b) { return Sleef_atan2d8_u10(a, b); }

inline __m256 sinh(__m256 a) { return Sleef_sinhf8_u10(a); }
inline __m256d sinh(__m256d a) { return Sleef_sinhd4_u10(a); }
inline __m512 sinh(__m512 a) { return Sleef_sinhf16_u10(a); }
inline __m512d sinh(__m512d a) { return Sleef_sinhd8_u10(a); }

inline __m256 cosh(__m256 a) { return Sleef_coshf8_u10(a); }
inline __m256d cosh(__m256d a) { return Sleef_coshd4_u10(a); }
inline __m512 cosh(__m512 a) { return Sleef_coshf16_u10(a); }
inline __m512d cosh(__m512d a) { return Sleef_coshd8_u10(a); }

inline __m256 tanh(__m256 a) { return Sleef_tanhf8_u10(a); }
inline __m256d tanh(__m256d a) { return Sleef_tanhd4_u10(a); }
inline __m512 tanh(__m512 a) { return Sleef_tanhf16_u10(a); }
inline __m512d tanh(__m512d a) { return Sleef_tanhd8_u10(a); }

inline __m256 asinh(__m256 a) { return Sleef_asinhf8_u10(a); }
inline __m256d asinh(__m256d a) { return Sleef_asinhd4_u10(a); }
inline __m512 asinh(__m512 a) { return Sleef_asinhf16_u10(a); }
inline __m512d asinh(__m512d a) { return Sleef_asinhd8_u10(a); }

inline __m256 acosh(__m256 a) { return Sleef_acoshf8_u10(a); }
inline __m256d acosh(__m256d a) { return Sleef_acoshd4_u10(a); }
inline __m512 acosh(__m512 a) { return Sleef_acoshf16_u10(a); }
inline __m512d acosh(__m512d a) { return Sleef_acoshd8_u10(a); }

inline __m256 atanh(__m256 a) { return Sleef_atanhf8_u10(a); }
inline __m256d atanh(__m256d a) { return Sleef_atanhd4_u10(a); }
inline __m512 atanh(__m512 a) { return Sleef_atanhf16_u10(a); }
inline __m512d atanh(__m512d a) { return Sleef_atanhd8_u10(a); }

inline __m256 erf(__m256 a) { return Sleef_erff8_u10(a); }
inline __m256d erf(__m256d a) { return Sleef_erfd4_u10(a); }
inline __m512 erf(__m512 a) { return Sleef_erff16_u10(a); }
inline __m512d erf(__m512d a) { return Sleef_erfd8_u10(a); }

inline __m256 erfc(__m256 a) { return Sleef_erfcf8_u15(a); }
inline __m256d erfc(__m256d a) { return Sleef_erfcd4_u15(a); }
inline __m512 erfc(__m512 a) { return Sleef_erfcf16_u15(a); }
inline __m512d erfc(__m512d a) { return Sleef_erfcd8_u15(a); }

inline __m256 pow(__m256 a, __m256 b) { return Sleef_powf8_u10(a, b); }
inline __m256d pow(__m256d a, __m256d b) { return Sleef_powd4_u10(a, b); }
inline __m512 pow(__m512 a, __m512 b) { return Sleef_powf16_u10(a, b); }
inline __m512d pow(__m512d a, __m512d b) { return Sleef_powd8_u10(a, b); }

inline __m256 hypot(__m256 a, __m256 b) { return Sleef_hypotf8_u05(a, b); }
inline __m256d hypot(__m256d a, __m256d b) { return Sleef_hypotd4_u05(a, b); }
inline __m512 hypot(__m512 a, __m512 b) { return Sleef_hypotf16_u05(a, b); }
inline __m512d hypot(__m512d a, __m512d b) { return Sleef_hypotd8_u05(a, b); }

} // namespace tpsleef
} // namespace tensorplay
#endif // x86-64 vector helpers
