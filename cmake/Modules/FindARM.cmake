# Detect whether the toolchain can compile the SVE (aarch64) vector tiers.
# The tiers are keyed to a fixed vector length (-msve-vector-bits), and the
# SVE kernels rely on bf16 arithmetic, so the probe includes a bf16 NEON
# conversion to reject compilers whose SVE support lacks the bf16 feature.
# The runtime vector length is probed separately by the dispatch layer.
IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
  INCLUDE(CheckCXXSourceCompiles)

  SET(SVE_BF16_PROBE_CODE "
    #include <arm_sve.h>
    #include <arm_neon.h>
    int main() {
      svfloat64_t a = svdup_n_f64(0);
      (void)a;
      float32x4_t b = vdupq_n_f32(0);
      bfloat16x8_t c = vreinterpretq_bf16_f32(b);
      bfloat16x4_t d = vget_low_bf16(c);
      (void)d;
      return 0;
    }
  ")

  SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
  SET(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS_INIT} -march=armv8-a+sve+bf16 -msve-vector-bits=256")
  CHECK_CXX_SOURCE_COMPILES("${SVE_BF16_PROBE_CODE}" TP_CXX_SVE256_PROBE)
  SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

  if(TP_CXX_SVE256_PROBE)
    set(TP_CXX_SVE256_FOUND TRUE)
    set(TP_CXX_SVE256_FLAGS "-march=armv8-a+sve+bf16 -msve-vector-bits=256")
    set(TP_CXX_SVE128_FLAGS "-march=armv8-a+sve+bf16 -msve-vector-bits=128")
    message(STATUS "SVE vector tiers available (256/128).")
  else()
    set(TP_CXX_SVE256_FOUND FALSE)
    message(STATUS "SVE vector tiers not available.")
  endif()
  mark_as_advanced(TP_CXX_SVE256_PROBE)
ENDIF()
