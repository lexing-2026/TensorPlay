# Detect whether the toolchain can compile the RISC-V vector tiers.
# The tiers are keyed to a fixed vector length (-mrvv-vector-bits on
# clang / -march with _zvl256b on gcc), and the vector intrinsics gate on
# the v1.0 spec (__riscv_v_intrinsic >= 12000).  The runtime vector
# register width is probed separately by the dispatch layer.
IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
  INCLUDE(CheckCXXSourceCompiles)

  SET(RVV_PROBE_CODE "
    #ifndef __riscv
    #error \"not riscv\"
    #endif
    #ifndef __riscv_v_intrinsic
    #error \"vector intrinsics unavailable\"
    #endif
    #if __riscv_v_intrinsic < 12000
    #error \"vector intrinsics pre-1.0\"
    #endif
    #include <riscv_vector.h>
    int main() {
      vfloat32m1_t a = __riscv_vfmv_v_f_f32m1(0.f, 4);
      vfloat32m1_t b = __riscv_vfadd_vv_f32m1(a, a, 4);
      (void)b;
      return 0;
    }
  ")

  SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
  # Fixed-length vector view: 256-bit registers for the RVVM2 tier.
  SET(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS_INIT} -march=rv64gcv_zvl256b -mrvv-vector-bits=256")
  CHECK_CXX_SOURCE_COMPILES("${RVV_PROBE_CODE}" TP_CXX_RVVM2_PROBE)
  if(NOT TP_CXX_RVVM2_PROBE)
    SET(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS_INIT} -march=rv64gcv")
    CHECK_CXX_SOURCE_COMPILES("${RVV_PROBE_CODE}" TP_CXX_RVVM1_PROBE)
  endif()
  SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

  if(TP_CXX_RVVM2_PROBE)
    set(TP_CXX_RVVM2_FOUND TRUE)
    set(TP_CXX_RVVM2_FLAGS "-march=rv64gcv_zvl256b -mrvv-vector-bits=256")
    set(TP_CXX_RVVM1_FLAGS "-march=rv64gcv_zvl128b -mrvv-vector-bits=128")
    message(STATUS "RISC-V vector tiers available (RVVM2/RVVM1).")
  elseif(TP_CXX_RVVM1_PROBE)
    set(TP_CXX_RVVM1_FOUND TRUE)
    set(TP_CXX_RVVM1_FLAGS "-march=rv64gcv")
    message(STATUS "RISC-V vector tier available (RVVM1).")
  else()
    set(TP_CXX_RVVM1_FOUND FALSE)
    set(TP_CXX_RVVM2_FOUND FALSE)
    message(STATUS "RISC-V vector tiers not available.")
  endif()
  mark_as_advanced(TP_CXX_RVVM1_PROBE TP_CXX_RVVM2_PROBE)
ENDIF()
