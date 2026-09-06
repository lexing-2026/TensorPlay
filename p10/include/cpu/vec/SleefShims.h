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
// Precision tiers: u10 for most functions, u35 for single-precision sin/cos,
// u15 for erfc and u05 for hypot.
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

#if defined(__VSX__) || defined(CPU_CAPABILITY_VSX)
// VSX tier: SLEEF's 2x double / 4x float VSX entry points. The libm
// dispatcher symbols are used so the running POWER generation picks its
// best implementation at load time.
#ifdef __cplusplus
extern "C" {
#endif

// (f4 = 4 x float)
__vector float Sleef_expf4_u10(__vector float);
__vector float Sleef_exp2f4_u10(__vector float);
__vector float Sleef_expm1f4_u10(__vector float);
__vector float Sleef_logf4_u10(__vector float);
__vector float Sleef_log2f4_u10(__vector float);
__vector float Sleef_log10f4_u10(__vector float);
__vector float Sleef_log1pf4_u10(__vector float);
__vector float Sleef_sinf4_u10(__vector float);
__vector float Sleef_cosf4_u10(__vector float);
__vector float Sleef_tanf4_u10(__vector float);
__vector float Sleef_asinf4_u10(__vector float);
__vector float Sleef_acosf4_u10(__vector float);
__vector float Sleef_atanf4_u10(__vector float);
__vector float Sleef_atan2f4_u10(__vector float, __vector float);
__vector float Sleef_sinhf4_u10(__vector float);
__vector float Sleef_coshf4_u10(__vector float);
__vector float Sleef_tanhf4_u10(__vector float);
__vector float Sleef_asinhf4_u10(__vector float);
__vector float Sleef_acoshf4_u10(__vector float);
__vector float Sleef_atanhf4_u10(__vector float);
__vector float Sleef_erff4_u10(__vector float);
__vector float Sleef_erfcf4_u15(__vector float);
__vector float Sleef_powf4_u10(__vector float, __vector float);
__vector float Sleef_hypotf4_u05(__vector float, __vector float);

// (d2 = 2 x double)
__vector double Sleef_expd2_u10(__vector double);
__vector double Sleef_exp2d2_u10(__vector double);
__vector double Sleef_expm1d2_u10(__vector double);
__vector double Sleef_logd2_u10(__vector double);
__vector double Sleef_log2d2_u10(__vector double);
__vector double Sleef_log10d2_u10(__vector double);
__vector double Sleef_log1pd2_u10(__vector double);
__vector double Sleef_sind2_u10(__vector double);
__vector double Sleef_cosd2_u10(__vector double);
__vector double Sleef_tand2_u10(__vector double);
__vector double Sleef_asind2_u10(__vector double);
__vector double Sleef_acosd2_u10(__vector double);
__vector double Sleef_atand2_u10(__vector double);
__vector double Sleef_atan2d2_u10(__vector double, __vector double);
__vector double Sleef_sinhd2_u10(__vector double);
__vector double Sleef_coshd2_u10(__vector double);
__vector double Sleef_tanhd2_u10(__vector double);
__vector double Sleef_asinhd2_u10(__vector double);
__vector double Sleef_acoshd2_u10(__vector double);
__vector double Sleef_atanhd2_u10(__vector double);
__vector double Sleef_erfd2_u10(__vector double);
__vector double Sleef_erfcd2_u15(__vector double);
__vector double Sleef_powd2_u10(__vector double, __vector double);
__vector double Sleef_hypotd2_u05(__vector double, __vector double);

#ifdef __cplusplus
} // extern "C"
#endif

namespace tensorplay {
namespace tpsleef {
inline __vector float exp(__vector float a) { return Sleef_expf4_u10(a); }
inline __vector double exp(__vector double a) { return Sleef_expd2_u10(a); }
inline __vector float exp2(__vector float a) { return Sleef_exp2f4_u10(a); }
inline __vector double exp2(__vector double a) { return Sleef_exp2d2_u10(a); }
inline __vector float expm1(__vector float a) { return Sleef_expm1f4_u10(a); }
inline __vector double expm1(__vector double a) { return Sleef_expm1d2_u10(a); }
inline __vector float log(__vector float a) { return Sleef_logf4_u10(a); }
inline __vector double log(__vector double a) { return Sleef_logd2_u10(a); }
inline __vector float log2(__vector float a) { return Sleef_log2f4_u10(a); }
inline __vector double log2(__vector double a) { return Sleef_log2d2_u10(a); }
inline __vector float log10(__vector float a) { return Sleef_log10f4_u10(a); }
inline __vector double log10(__vector double a) { return Sleef_log10d2_u10(a); }
inline __vector float log1p(__vector float a) { return Sleef_log1pf4_u10(a); }
inline __vector double log1p(__vector double a) { return Sleef_log1pd2_u10(a); }
inline __vector float sin(__vector float a) { return Sleef_sinf4_u10(a); }
inline __vector double sin(__vector double a) { return Sleef_sind2_u10(a); }
inline __vector float cos(__vector float a) { return Sleef_cosf4_u10(a); }
inline __vector double cos(__vector double a) { return Sleef_cosd2_u10(a); }
inline __vector float tan(__vector float a) { return Sleef_tanf4_u10(a); }
inline __vector double tan(__vector double a) { return Sleef_tand2_u10(a); }
inline __vector float asin(__vector float a) { return Sleef_asinf4_u10(a); }
inline __vector double asin(__vector double a) { return Sleef_asind2_u10(a); }
inline __vector float acos(__vector float a) { return Sleef_acosf4_u10(a); }
inline __vector double acos(__vector double a) { return Sleef_acosd2_u10(a); }
inline __vector float atan(__vector float a) { return Sleef_atanf4_u10(a); }
inline __vector double atan(__vector double a) { return Sleef_atand2_u10(a); }
inline __vector float atan2(__vector float a, __vector float b) { return Sleef_atan2f4_u10(a, b); }
inline __vector double atan2(__vector double a, __vector double b) { return Sleef_atan2d2_u10(a, b); }
inline __vector float sinh(__vector float a) { return Sleef_sinhf4_u10(a); }
inline __vector double sinh(__vector double a) { return Sleef_sinhd2_u10(a); }
inline __vector float cosh(__vector float a) { return Sleef_coshf4_u10(a); }
inline __vector double cosh(__vector double a) { return Sleef_coshd2_u10(a); }
inline __vector float tanh(__vector float a) { return Sleef_tanhf4_u10(a); }
inline __vector double tanh(__vector double a) { return Sleef_tanhd2_u10(a); }
inline __vector float asinh(__vector float a) { return Sleef_asinhf4_u10(a); }
inline __vector double asinh(__vector double a) { return Sleef_asinhd2_u10(a); }
inline __vector float acosh(__vector float a) { return Sleef_acoshf4_u10(a); }
inline __vector double acosh(__vector double a) { return Sleef_acoshd2_u10(a); }
inline __vector float atanh(__vector float a) { return Sleef_atanhf4_u10(a); }
inline __vector double atanh(__vector double a) { return Sleef_atanhd2_u10(a); }
inline __vector float erf(__vector float a) { return Sleef_erff4_u10(a); }
inline __vector double erf(__vector double a) { return Sleef_erfd2_u10(a); }
inline __vector float erfc(__vector float a) { return Sleef_erfcf4_u15(a); }
inline __vector double erfc(__vector double a) { return Sleef_erfcd2_u15(a); }
inline __vector float pow(__vector float a, __vector float b) { return Sleef_powf4_u10(a, b); }
inline __vector double pow(__vector double a, __vector double b) { return Sleef_powd2_u10(a, b); }
inline __vector float hypot(__vector float a, __vector float b) { return Sleef_hypotf4_u05(a, b); }
inline __vector double hypot(__vector double a, __vector double b) { return Sleef_hypotd2_u05(a, b); }
} // namespace tpsleef
} // namespace tensorplay
#endif // VSX helpers

#if defined(__s390x__) && (defined(CPU_CAPABILITY_ZVECTOR) || defined(__VX__))
// ZVECTOR tier: SLEEF's 2x double / 4x float VXE entry points.
#ifdef __cplusplus
extern "C" {
#endif

__vector float Sleef_expf4_u10(__vector float);
__vector float Sleef_exp2f4_u10(__vector float);
__vector float Sleef_expm1f4_u10(__vector float);
__vector float Sleef_logf4_u10(__vector float);
__vector float Sleef_log2f4_u10(__vector float);
__vector float Sleef_log10f4_u10(__vector float);
__vector float Sleef_log1pf4_u10(__vector float);
__vector float Sleef_sinf4_u10(__vector float);
__vector float Sleef_cosf4_u10(__vector float);
__vector float Sleef_tanf4_u10(__vector float);
__vector float Sleef_asinf4_u10(__vector float);
__vector float Sleef_acosf4_u10(__vector float);
__vector float Sleef_atanf4_u10(__vector float);
__vector float Sleef_atan2f4_u10(__vector float, __vector float);
__vector float Sleef_sinhf4_u10(__vector float);
__vector float Sleef_coshf4_u10(__vector float);
__vector float Sleef_tanhf4_u10(__vector float);
__vector float Sleef_asinhf4_u10(__vector float);
__vector float Sleef_acoshf4_u10(__vector float);
__vector float Sleef_atanhf4_u10(__vector float);
__vector float Sleef_erff4_u10(__vector float);
__vector float Sleef_erfcf4_u15(__vector float);
__vector float Sleef_powf4_u10(__vector float, __vector float);
__vector float Sleef_hypotf4_u05(__vector float, __vector float);

__vector double Sleef_expd2_u10(__vector double);
__vector double Sleef_exp2d2_u10(__vector double);
__vector double Sleef_expm1d2_u10(__vector double);
__vector double Sleef_logd2_u10(__vector double);
__vector double Sleef_log2d2_u10(__vector double);
__vector double Sleef_log10d2_u10(__vector double);
__vector double Sleef_log1pd2_u10(__vector double);
__vector double Sleef_sind2_u10(__vector double);
__vector double Sleef_cosd2_u10(__vector double);
__vector double Sleef_tand2_u10(__vector double);
__vector double Sleef_asind2_u10(__vector double);
__vector double Sleef_acosd2_u10(__vector double);
__vector double Sleef_atand2_u10(__vector double);
__vector double Sleef_atan2d2_u10(__vector double, __vector double);
__vector double Sleef_sinhd2_u10(__vector double);
__vector double Sleef_coshd2_u10(__vector double);
__vector double Sleef_tanhd2_u10(__vector double);
__vector double Sleef_asinhd2_u10(__vector double);
__vector double Sleef_acoshd2_u10(__vector double);
__vector double Sleef_atanhd2_u10(__vector double);
__vector double Sleef_erfd2_u10(__vector double);
__vector double Sleef_erfcd2_u15(__vector double);
__vector double Sleef_powd2_u10(__vector double, __vector double);
__vector double Sleef_hypotd2_u05(__vector double, __vector double);

#ifdef __cplusplus
} // extern "C"
#endif

namespace tensorplay {
namespace tpsleef {
inline __vector float exp(__vector float a) { return Sleef_expf4_u10(a); }
inline __vector double exp(__vector double a) { return Sleef_expd2_u10(a); }
inline __vector float exp2(__vector float a) { return Sleef_exp2f4_u10(a); }
inline __vector double exp2(__vector double a) { return Sleef_exp2d2_u10(a); }
inline __vector float expm1(__vector float a) { return Sleef_expm1f4_u10(a); }
inline __vector double expm1(__vector double a) { return Sleef_expm1d2_u10(a); }
inline __vector float log(__vector float a) { return Sleef_logf4_u10(a); }
inline __vector double log(__vector double a) { return Sleef_logd2_u10(a); }
inline __vector float log2(__vector float a) { return Sleef_log2f4_u10(a); }
inline __vector double log2(__vector double a) { return Sleef_log2d2_u10(a); }
inline __vector float log10(__vector float a) { return Sleef_log10f4_u10(a); }
inline __vector double log10(__vector double a) { return Sleef_log10d2_u10(a); }
inline __vector float log1p(__vector float a) { return Sleef_log1pf4_u10(a); }
inline __vector double log1p(__vector double a) { return Sleef_log1pd2_u10(a); }
inline __vector float sin(__vector float a) { return Sleef_sinf4_u10(a); }
inline __vector double sin(__vector double a) { return Sleef_sind2_u10(a); }
inline __vector float cos(__vector float a) { return Sleef_cosf4_u10(a); }
inline __vector double cos(__vector double a) { return Sleef_cosd2_u10(a); }
inline __vector float tan(__vector float a) { return Sleef_tanf4_u10(a); }
inline __vector double tan(__vector double a) { return Sleef_tand2_u10(a); }
inline __vector float asin(__vector float a) { return Sleef_asinf4_u10(a); }
inline __vector double asin(__vector double a) { return Sleef_asind2_u10(a); }
inline __vector float acos(__vector float a) { return Sleef_acosf4_u10(a); }
inline __vector double acos(__vector double a) { return Sleef_acosd2_u10(a); }
inline __vector float atan(__vector float a) { return Sleef_atanf4_u10(a); }
inline __vector double atan(__vector double a) { return Sleef_atand2_u10(a); }
inline __vector float atan2(__vector float a, __vector float b) { return Sleef_atan2f4_u10(a, b); }
inline __vector double atan2(__vector double a, __vector double b) { return Sleef_atan2d2_u10(a, b); }
inline __vector float sinh(__vector float a) { return Sleef_sinhf4_u10(a); }
inline __vector double sinh(__vector double a) { return Sleef_sinhd2_u10(a); }
inline __vector float cosh(__vector float a) { return Sleef_coshf4_u10(a); }
inline __vector double cosh(__vector double a) { return Sleef_coshd2_u10(a); }
inline __vector float tanh(__vector float a) { return Sleef_tanhf4_u10(a); }
inline __vector double tanh(__vector double a) { return Sleef_tanhd2_u10(a); }
inline __vector float asinh(__vector float a) { return Sleef_asinhf4_u10(a); }
inline __vector double asinh(__vector double a) { return Sleef_asinhd2_u10(a); }
inline __vector float acosh(__vector float a) { return Sleef_acoshf4_u10(a); }
inline __vector double acosh(__vector double a) { return Sleef_acoshd2_u10(a); }
inline __vector float atanh(__vector float a) { return Sleef_atanhf4_u10(a); }
inline __vector double atanh(__vector double a) { return Sleef_atanhd2_u10(a); }
inline __vector float erf(__vector float a) { return Sleef_erff4_u10(a); }
inline __vector double erf(__vector double a) { return Sleef_erfd2_u10(a); }
inline __vector float erfc(__vector float a) { return Sleef_erfcf4_u15(a); }
inline __vector double erfc(__vector double a) { return Sleef_erfcd2_u15(a); }
inline __vector float pow(__vector float a, __vector float b) { return Sleef_powf4_u10(a, b); }
inline __vector double pow(__vector double a, __vector double b) { return Sleef_powd2_u10(a, b); }
inline __vector float hypot(__vector float a, __vector float b) { return Sleef_hypotf4_u05(a, b); }
inline __vector double hypot(__vector double a, __vector double b) { return Sleef_hypotd2_u05(a, b); }
} // namespace tpsleef
} // namespace tensorplay
#endif // ZVECTOR helpers

#if defined(__aarch64__) && !defined(__ANDROID__) && \
    !(defined(CPU_CAPABILITY_SVE256) || defined(CPU_CAPABILITY_SVE128))
// aarch64 NEON tier: SLEEF's 2x double / 4x float ADVSIMD runtime
// dispatchers. Follows the desktop default (Linux and macOS; Android keeps
// the scalar <cmath> fallback).
#ifdef __cplusplus
extern "C" {
#endif

float32x4_t Sleef_expf4_u10(float32x4_t);
float32x4_t Sleef_exp2f4_u10(float32x4_t);
float32x4_t Sleef_expm1f4_u10(float32x4_t);
float32x4_t Sleef_logf4_u10(float32x4_t);
float32x4_t Sleef_log2f4_u10(float32x4_t);
float32x4_t Sleef_log10f4_u10(float32x4_t);
float32x4_t Sleef_log1pf4_u10(float32x4_t);
float32x4_t Sleef_sinf4_u10(float32x4_t);
float32x4_t Sleef_cosf4_u10(float32x4_t);
float32x4_t Sleef_tanf4_u10(float32x4_t);
float32x4_t Sleef_asinf4_u10(float32x4_t);
float32x4_t Sleef_acosf4_u10(float32x4_t);
float32x4_t Sleef_atanf4_u10(float32x4_t);
float32x4_t Sleef_atan2f4_u10(float32x4_t, float32x4_t);
float32x4_t Sleef_sinhf4_u10(float32x4_t);
float32x4_t Sleef_coshf4_u10(float32x4_t);
float32x4_t Sleef_tanhf4_u10(float32x4_t);
float32x4_t Sleef_asinhf4_u10(float32x4_t);
float32x4_t Sleef_acoshf4_u10(float32x4_t);
float32x4_t Sleef_atanhf4_u10(float32x4_t);
float32x4_t Sleef_erff4_u10(float32x4_t);
float32x4_t Sleef_erfcf4_u15(float32x4_t);
float32x4_t Sleef_powf4_u10(float32x4_t, float32x4_t);
float32x4_t Sleef_hypotf4_u05(float32x4_t, float32x4_t);
float32x4_t Sleef_copysignf4(float32x4_t, float32x4_t);
float32x4_t Sleef_fmodf4(float32x4_t, float32x4_t);
float32x4_t Sleef_nextafterf4(float32x4_t, float32x4_t);

float64x2_t Sleef_expd2_u10(float64x2_t);
float64x2_t Sleef_exp2d2_u10(float64x2_t);
float64x2_t Sleef_expm1d2_u10(float64x2_t);
float64x2_t Sleef_logd2_u10(float64x2_t);
float64x2_t Sleef_log2d2_u10(float64x2_t);
float64x2_t Sleef_log10d2_u10(float64x2_t);
float64x2_t Sleef_log1pd2_u10(float64x2_t);
float64x2_t Sleef_sind2_u10(float64x2_t);
float64x2_t Sleef_cosd2_u10(float64x2_t);
float64x2_t Sleef_tand2_u10(float64x2_t);
float64x2_t Sleef_asind2_u10(float64x2_t);
float64x2_t Sleef_acosd2_u10(float64x2_t);
float64x2_t Sleef_atand2_u10(float64x2_t);
float64x2_t Sleef_atan2d2_u10(float64x2_t, float64x2_t);
float64x2_t Sleef_sinhd2_u10(float64x2_t);
float64x2_t Sleef_coshd2_u10(float64x2_t);
float64x2_t Sleef_tanhd2_u10(float64x2_t);
float64x2_t Sleef_asinhd2_u10(float64x2_t);
float64x2_t Sleef_acoshd2_u10(float64x2_t);
float64x2_t Sleef_atanhd2_u10(float64x2_t);
float64x2_t Sleef_erfd2_u10(float64x2_t);
float64x2_t Sleef_erfcd2_u15(float64x2_t);
float64x2_t Sleef_powd2_u10(float64x2_t, float64x2_t);
float64x2_t Sleef_hypotd2_u05(float64x2_t, float64x2_t);
float64x2_t Sleef_copysignd2(float64x2_t, float64x2_t);
float64x2_t Sleef_fmodd2(float64x2_t, float64x2_t);
float64x2_t Sleef_nextafterd2(float64x2_t, float64x2_t);

#ifdef __cplusplus
} // extern "C"
#endif

namespace tensorplay {
namespace tpsleef {
inline float32x4_t exp(float32x4_t a) { return Sleef_expf4_u10(a); }
inline float64x2_t exp(float64x2_t a) { return Sleef_expd2_u10(a); }
inline float32x4_t exp2(float32x4_t a) { return Sleef_exp2f4_u10(a); }
inline float64x2_t exp2(float64x2_t a) { return Sleef_exp2d2_u10(a); }
inline float32x4_t expm1(float32x4_t a) { return Sleef_expm1f4_u10(a); }
inline float64x2_t expm1(float64x2_t a) { return Sleef_expm1d2_u10(a); }
inline float32x4_t log(float32x4_t a) { return Sleef_logf4_u10(a); }
inline float64x2_t log(float64x2_t a) { return Sleef_logd2_u10(a); }
inline float32x4_t log2(float32x4_t a) { return Sleef_log2f4_u10(a); }
inline float64x2_t log2(float64x2_t a) { return Sleef_log2d2_u10(a); }
inline float32x4_t log10(float32x4_t a) { return Sleef_log10f4_u10(a); }
inline float64x2_t log10(float64x2_t a) { return Sleef_log10d2_u10(a); }
inline float32x4_t log1p(float32x4_t a) { return Sleef_log1pf4_u10(a); }
inline float64x2_t log1p(float64x2_t a) { return Sleef_log1pd2_u10(a); }
inline float32x4_t sin(float32x4_t a) { return Sleef_sinf4_u10(a); }
inline float64x2_t sin(float64x2_t a) { return Sleef_sind2_u10(a); }
inline float32x4_t cos(float32x4_t a) { return Sleef_cosf4_u10(a); }
inline float64x2_t cos(float64x2_t a) { return Sleef_cosd2_u10(a); }
inline float32x4_t tan(float32x4_t a) { return Sleef_tanf4_u10(a); }
inline float64x2_t tan(float64x2_t a) { return Sleef_tand2_u10(a); }
inline float32x4_t asin(float32x4_t a) { return Sleef_asinf4_u10(a); }
inline float64x2_t asin(float64x2_t a) { return Sleef_asind2_u10(a); }
inline float32x4_t acos(float32x4_t a) { return Sleef_acosf4_u10(a); }
inline float64x2_t acos(float64x2_t a) { return Sleef_acosd2_u10(a); }
inline float32x4_t atan(float32x4_t a) { return Sleef_atanf4_u10(a); }
inline float64x2_t atan(float64x2_t a) { return Sleef_atand2_u10(a); }
inline float32x4_t atan2(float32x4_t a, float32x4_t b) { return Sleef_atan2f4_u10(a, b); }
inline float64x2_t atan2(float64x2_t a, float64x2_t b) { return Sleef_atan2d2_u10(a, b); }
inline float32x4_t sinh(float32x4_t a) { return Sleef_sinhf4_u10(a); }
inline float64x2_t sinh(float64x2_t a) { return Sleef_sinhd2_u10(a); }
inline float32x4_t cosh(float32x4_t a) { return Sleef_coshf4_u10(a); }
inline float64x2_t cosh(float64x2_t a) { return Sleef_coshd2_u10(a); }
inline float32x4_t tanh(float32x4_t a) { return Sleef_tanhf4_u10(a); }
inline float64x2_t tanh(float64x2_t a) { return Sleef_tanhd2_u10(a); }
inline float32x4_t asinh(float32x4_t a) { return Sleef_asinhf4_u10(a); }
inline float64x2_t asinh(float64x2_t a) { return Sleef_asinhd2_u10(a); }
inline float32x4_t acosh(float32x4_t a) { return Sleef_acoshf4_u10(a); }
inline float64x2_t acosh(float64x2_t a) { return Sleef_acoshd2_u10(a); }
inline float32x4_t atanh(float32x4_t a) { return Sleef_atanhf4_u10(a); }
inline float64x2_t atanh(float64x2_t a) { return Sleef_atanhd2_u10(a); }
inline float32x4_t erf(float32x4_t a) { return Sleef_erff4_u10(a); }
inline float64x2_t erf(float64x2_t a) { return Sleef_erfd2_u10(a); }
inline float32x4_t erfc(float32x4_t a) { return Sleef_erfcf4_u15(a); }
inline float64x2_t erfc(float64x2_t a) { return Sleef_erfcd2_u15(a); }
inline float32x4_t pow(float32x4_t a, float32x4_t b) { return Sleef_powf4_u10(a, b); }
inline float64x2_t pow(float64x2_t a, float64x2_t b) { return Sleef_powd2_u10(a, b); }
inline float32x4_t hypot(float32x4_t a, float32x4_t b) { return Sleef_hypotf4_u05(a, b); }
inline float64x2_t hypot(float64x2_t a, float64x2_t b) { return Sleef_hypotd2_u05(a, b); }
inline float32x4_t fmod(float32x4_t a, float32x4_t b) { return Sleef_fmodf4(a, b); }
inline float64x2_t fmod(float64x2_t a, float64x2_t b) { return Sleef_fmodd2(a, b); }
inline float32x4_t nextafter(float32x4_t a, float32x4_t b) { return Sleef_nextafterf4(a, b); }
inline float64x2_t nextafter(float64x2_t a, float64x2_t b) { return Sleef_nextafterd2(a, b); }
} // namespace tpsleef
} // namespace tensorplay
#endif // aarch64 NEON helpers

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && \
    (defined(CPU_CAPABILITY_SVE256) || defined(CPU_CAPABILITY_SVE128))
// SVE tiers: SLEEF's vector-length-agnostic entry points operate on
// svfloat32_t/svfloat64_t with explicit predicate control. Precision
// suffixes match the other tiers (u10 / u15 for erfc / u05 for hypot).
#ifdef __cplusplus
extern "C" {
#endif

svfloat32_t Sleef_expfx_u10sve(svfloat32_t);
svfloat32_t Sleef_exp2fx_u10sve(svfloat32_t);
svfloat32_t Sleef_expm1fx_u10sve(svfloat32_t);
svfloat32_t Sleef_logfx_u10sve(svfloat32_t);
svfloat32_t Sleef_log2fx_u10sve(svfloat32_t);
svfloat32_t Sleef_log10fx_u10sve(svfloat32_t);
svfloat32_t Sleef_log1pfx_u10sve(svfloat32_t);
svfloat32_t Sleef_sinfx_u10sve(svfloat32_t);
svfloat32_t Sleef_cosfx_u10sve(svfloat32_t);
svfloat32_t Sleef_tanfx_u10sve(svfloat32_t);
svfloat32_t Sleef_asinfx_u10sve(svfloat32_t);
svfloat32_t Sleef_acosfx_u10sve(svfloat32_t);
svfloat32_t Sleef_atanfx_u10sve(svfloat32_t);
svfloat32_t Sleef_atan2fx_u10sve(svfloat32_t, svfloat32_t);
svfloat32_t Sleef_sinhfx_u10sve(svfloat32_t);
svfloat32_t Sleef_coshfx_u10sve(svfloat32_t);
svfloat32_t Sleef_tanhfx_u10sve(svfloat32_t);
svfloat32_t Sleef_asinhfx_u10sve(svfloat32_t);
svfloat32_t Sleef_acoshfx_u10sve(svfloat32_t);
svfloat32_t Sleef_atanhfx_u10sve(svfloat32_t);
svfloat32_t Sleef_erffx_u10sve(svfloat32_t);
svfloat32_t Sleef_erfcfx_u15sve(svfloat32_t);
svfloat32_t Sleef_powfx_u10sve(svfloat32_t, svfloat32_t);
svfloat32_t Sleef_hypotfx_u05sve(svfloat32_t, svfloat32_t);

svfloat64_t Sleef_expdx_u10sve(svfloat64_t);
svfloat64_t Sleef_exp2dx_u10sve(svfloat64_t);
svfloat64_t Sleef_expm1dx_u10sve(svfloat64_t);
svfloat64_t Sleef_logdx_u10sve(svfloat64_t);
svfloat64_t Sleef_log2dx_u10sve(svfloat64_t);
svfloat64_t Sleef_log10dx_u10sve(svfloat64_t);
svfloat64_t Sleef_log1pdx_u10sve(svfloat64_t);
svfloat64_t Sleef_sindx_u10sve(svfloat64_t);
svfloat64_t Sleef_cosdx_u10sve(svfloat64_t);
svfloat64_t Sleef_tandx_u10sve(svfloat64_t);
svfloat64_t Sleef_asindx_u10sve(svfloat64_t);
svfloat64_t Sleef_acosdx_u10sve(svfloat64_t);
svfloat64_t Sleef_atandx_u10sve(svfloat64_t);
svfloat64_t Sleef_atan2dx_u10sve(svfloat64_t, svfloat64_t);
svfloat64_t Sleef_sinhdx_u10sve(svfloat64_t);
svfloat64_t Sleef_coshdx_u10sve(svfloat64_t);
svfloat64_t Sleef_tanhdx_u10sve(svfloat64_t);
svfloat64_t Sleef_asinhdx_u10sve(svfloat64_t);
svfloat64_t Sleef_acoshdx_u10sve(svfloat64_t);
svfloat64_t Sleef_atanhdx_u10sve(svfloat64_t);
svfloat64_t Sleef_erfdx_u10sve(svfloat64_t);
svfloat64_t Sleef_erfcdx_u15sve(svfloat64_t);
svfloat64_t Sleef_powdx_u10sve(svfloat64_t, svfloat64_t);
svfloat64_t Sleef_hypotdx_u05sve(svfloat64_t, svfloat64_t);

#ifdef __cplusplus
} // extern "C"
#endif

namespace tensorplay {
namespace tpsleef {
inline svfloat32_t exp(svfloat32_t a) { return Sleef_expfx_u10sve(a); }
inline svfloat64_t exp(svfloat64_t a) { return Sleef_expdx_u10sve(a); }
inline svfloat32_t exp2(svfloat32_t a) { return Sleef_exp2fx_u10sve(a); }
inline svfloat64_t exp2(svfloat64_t a) { return Sleef_exp2dx_u10sve(a); }
inline svfloat32_t expm1(svfloat32_t a) { return Sleef_expm1fx_u10sve(a); }
inline svfloat64_t expm1(svfloat64_t a) { return Sleef_expm1dx_u10sve(a); }
inline svfloat32_t log(svfloat32_t a) { return Sleef_logfx_u10sve(a); }
inline svfloat64_t log(svfloat64_t a) { return Sleef_logdx_u10sve(a); }
inline svfloat32_t log2(svfloat32_t a) { return Sleef_log2fx_u10sve(a); }
inline svfloat64_t log2(svfloat64_t a) { return Sleef_log2dx_u10sve(a); }
inline svfloat32_t log10(svfloat32_t a) { return Sleef_log10fx_u10sve(a); }
inline svfloat64_t log10(svfloat64_t a) { return Sleef_log10dx_u10sve(a); }
inline svfloat32_t log1p(svfloat32_t a) { return Sleef_log1pfx_u10sve(a); }
inline svfloat64_t log1p(svfloat64_t a) { return Sleef_log1pdx_u10sve(a); }
inline svfloat32_t sin(svfloat32_t a) { return Sleef_sinfx_u10sve(a); }
inline svfloat64_t sin(svfloat64_t a) { return Sleef_sindx_u10sve(a); }
inline svfloat32_t cos(svfloat32_t a) { return Sleef_cosfx_u10sve(a); }
inline svfloat64_t cos(svfloat64_t a) { return Sleef_cosdx_u10sve(a); }
inline svfloat32_t tan(svfloat32_t a) { return Sleef_tanfx_u10sve(a); }
inline svfloat64_t tan(svfloat64_t a) { return Sleef_tandx_u10sve(a); }
inline svfloat32_t asin(svfloat32_t a) { return Sleef_asinfx_u10sve(a); }
inline svfloat64_t asin(svfloat64_t a) { return Sleef_asindx_u10sve(a); }
inline svfloat32_t acos(svfloat32_t a) { return Sleef_acosfx_u10sve(a); }
inline svfloat64_t acos(svfloat64_t a) { return Sleef_acosdx_u10sve(a); }
inline svfloat32_t atan(svfloat32_t a) { return Sleef_atanfx_u10sve(a); }
inline svfloat64_t atan(svfloat64_t a) { return Sleef_atandx_u10sve(a); }
inline svfloat32_t atan2(svfloat32_t a, svfloat32_t b) { return Sleef_atan2fx_u10sve(a, b); }
inline svfloat64_t atan2(svfloat64_t a, svfloat64_t b) { return Sleef_atan2dx_u10sve(a, b); }
inline svfloat32_t sinh(svfloat32_t a) { return Sleef_sinhfx_u10sve(a); }
inline svfloat64_t sinh(svfloat64_t a) { return Sleef_sinhdx_u10sve(a); }
inline svfloat32_t cosh(svfloat32_t a) { return Sleef_coshfx_u10sve(a); }
inline svfloat64_t cosh(svfloat64_t a) { return Sleef_coshdx_u10sve(a); }
inline svfloat32_t tanh(svfloat32_t a) { return Sleef_tanhfx_u10sve(a); }
inline svfloat64_t tanh(svfloat64_t a) { return Sleef_tanhdx_u10sve(a); }
inline svfloat32_t asinh(svfloat32_t a) { return Sleef_asinhfx_u10sve(a); }
inline svfloat64_t asinh(svfloat64_t a) { return Sleef_asinhdx_u10sve(a); }
inline svfloat32_t acosh(svfloat32_t a) { return Sleef_acoshfx_u10sve(a); }
inline svfloat64_t acosh(svfloat64_t a) { return Sleef_acoshdx_u10sve(a); }
inline svfloat32_t atanh(svfloat32_t a) { return Sleef_atanhfx_u10sve(a); }
inline svfloat64_t atanh(svfloat64_t a) { return Sleef_atanhdx_u10sve(a); }
inline svfloat32_t erf(svfloat32_t a) { return Sleef_erffx_u10sve(a); }
inline svfloat64_t erf(svfloat64_t a) { return Sleef_erfdx_u10sve(a); }
inline svfloat32_t erfc(svfloat32_t a) { return Sleef_erfcfx_u15sve(a); }
inline svfloat64_t erfc(svfloat64_t a) { return Sleef_erfcdx_u15sve(a); }
inline svfloat32_t pow(svfloat32_t a, svfloat32_t b) { return Sleef_powfx_u10sve(a, b); }
inline svfloat64_t pow(svfloat64_t a, svfloat64_t b) { return Sleef_powdx_u10sve(a, b); }
inline svfloat32_t hypot(svfloat32_t a, svfloat32_t b) { return Sleef_hypotfx_u05sve(a, b); }
inline svfloat64_t hypot(svfloat64_t a, svfloat64_t b) { return Sleef_hypotdx_u05sve(a, b); }
} // namespace tpsleef
} // namespace tensorplay
#endif // SVE helpers
