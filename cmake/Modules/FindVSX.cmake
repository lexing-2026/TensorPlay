# Detect whether the toolchain can compile the VSX (PowerPC) vector tier.
# The probe compiles real vector code so that both the header and the
# codegen path are exercised, not just the flag itself.
IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
  INCLUDE(CheckCXXSourceCompiles)

  SET(VSX_PROBE_CODE "
    #include <altivec.h>
    #undef bool
    #undef vector
    #undef pixel
    int main() {
      float __attribute__((aligned(16))) xs[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
      float __attribute__((aligned(16))) ys[4] = { 4.0f, 3.0f, 2.0f, 1.0f };
      __vector float vx = vec_vsx_ld(0, xs);
      __vector float vy = vec_vsx_ld(0, ys);
      __vector float vz = vec_add(vx, vy);
      float out[4] __attribute__((aligned(16)));
      vec_vsx_st(vz, 0, out);
      return (out[0] == 5.0f && out[3] == 5.0f) ? 0 : 1;
    }
  ")

  SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
  SET(CMAKE_REQUIRED_FLAGS "-mvsx")
  CHECK_CXX_SOURCE_COMPILES("${VSX_PROBE_CODE}" TP_CXX_VSX_PROBE)
  SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

  if(TP_CXX_VSX_PROBE)
    set(TP_CXX_VSX_FOUND TRUE)
    set(TP_CXX_VSX_FLAGS "-mvsx")
    message(STATUS "VSX vector tier available.")
  else()
    set(TP_CXX_VSX_FOUND FALSE)
    message(STATUS "VSX vector tier not available.")
  endif()
  mark_as_advanced(TP_CXX_VSX_PROBE)
ENDIF()
