# Detect whether the toolchain can compile the ZVECTOR (s390x) vector tier.
# The vector extension headers provide the intrinsics; -mvx/-mzvector enable
# the codegen. Runtime usability additionally requires the VXE hardware
# facility, probed by the dispatch layer itself.
IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
  INCLUDE(CheckCXXSourceCompiles)

  SET(ZVECTOR_PROBE_CODE "
    #include <vecintrin.h>
    int main() {
      float __attribute__((aligned(16))) xs[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
      float __attribute__((aligned(16))) ys[4] = { 4.0f, 3.0f, 2.0f, 1.0f };
      float32x4 vx = vec_xl(0, xs);
      float32x4 vy = vec_xl(0, ys);
      float32x4 vz = vec_add(vx, vy);
      float out[4] __attribute__((aligned(16)));
      vec_st(vz, 0, out);
      return (out[0] == 5.0f && out[3] == 5.0f) ? 0 : 1;
    }
  ")

  SET(TP_ZVECTOR_TEST_FLAGS "-mvx -mzvector")
  SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
  SET(CMAKE_REQUIRED_FLAGS "${TP_ZVECTOR_TEST_FLAGS}")
  CHECK_CXX_SOURCE_COMPILES("${ZVECTOR_PROBE_CODE}" TP_CXX_ZVECTOR_PROBE)
  SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

  if(TP_CXX_ZVECTOR_PROBE)
    set(TP_CXX_ZVECTOR_FOUND TRUE)
    set(TP_CXX_ZVECTOR_FLAGS "${TP_ZVECTOR_TEST_FLAGS}")
    message(STATUS "ZVECTOR vector tier available.")
  else()
    set(TP_CXX_ZVECTOR_FOUND FALSE)
    message(STATUS "ZVECTOR vector tier not available.")
  endif()
  mark_as_advanced(TP_CXX_ZVECTOR_PROBE)
ENDIF()
