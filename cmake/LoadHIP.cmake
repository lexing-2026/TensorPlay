# Locate the AMD GPU toolchain and enable the HIP language.
#
# Sets TP_FOUND_HIP=TRUE and the variables listed at the bottom when a
# usable install is present; returns silently otherwise so the caller can
# fall back to a CPU-only build.

set(TP_FOUND_HIP FALSE)

if(DEFINED ENV{ROCM_PATH})
  file(TO_CMAKE_PATH "$ENV{ROCM_PATH}" ROCM_PATH)
  if(NOT EXISTS ${ROCM_PATH})
    message(FATAL_ERROR
      "ROCM_PATH is set to ${ROCM_PATH} but does not exist.")
  endif()
else()
  if(UNIX)
    set(ROCM_PATH /opt/rocm)
  else()
    set(ROCM_PATH C:/opt/rocm)
  endif()
  if(NOT EXISTS ${ROCM_PATH})
    message(STATUS "No ROCm install found at ${ROCM_PATH}; building without AMD GPU support.")
    return()
  endif()
endif()

message(STATUS "Found ROCm: ${ROCM_PATH}")
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

# GPU architectures to codegen for.  The value accepts an
# env var list ("gfx1103;gfx1100"); a bare arch string is accepted too.  A
# cache variable set with -DTP_ROCM_ARCH=... takes precedence.
if(NOT TP_ROCM_ARCH AND DEFINED ENV{TP_ROCM_ARCH})
  set(TP_ROCM_ARCH $ENV{TP_ROCM_ARCH})
elseif(NOT TP_ROCM_ARCH)
  set(TP_ROCM_ARCH "gfx1103")
endif()
string(REPLACE "," ";" TP_ROCM_ARCH "${TP_ROCM_ARCH}")
string(REPLACE " " ";" TP_ROCM_ARCH "${TP_ROCM_ARCH}")
message(STATUS "AMD GPU arch list: ${TP_ROCM_ARCH}")

if(DEFINED ENV{HIP_CLANG_PATH})
  file(TO_CMAKE_PATH "$ENV{HIP_CLANG_PATH}" _tp_hip_clang_dir)
else()
  set(_tp_hip_clang_dir "${ROCM_PATH}/lib/llvm/bin")
endif()
if(NOT EXISTS "${_tp_hip_clang_dir}/clang++")
  message(STATUS "HIP compiler not found at ${_tp_hip_clang_dir}/clang++.")
  return()
endif()
set(CMAKE_HIP_COMPILER "${_tp_hip_clang_dir}/clang++")

set(CMAKE_HIP_PLATFORM "amd" CACHE STRING "HIP platform" FORCE)
set(CMAKE_HIP_ARCHITECTURES ${TP_ROCM_ARCH})

enable_language(HIP)
message(STATUS "HIP language enabled with compiler: ${CMAKE_HIP_COMPILER}")
message(STATUS "HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")

if(NOT WIN32)
  string(APPEND CMAKE_HIP_FLAGS " -fPIC")
endif()
string(APPEND CMAKE_HIP_FLAGS " -std=c++20")

# The hipify-staged tree keeps the CUDA namespace/class spelling; only API
# calls are rewritten.  These defines match what the staged sources expect
# from the device-code compiler.
set(TP_HIP_CXX_FLAGS
    -D__HIP_PLATFORM_AMD__=1
    -DCUDA_HAS_FP16=1
    -DUSE_ROCM
    -D__HIP_NO_HALF_OPERATORS__=1
    -D__HIP_NO_HALF_CONVERSIONS__=1
    CACHE INTERNAL "HIP build flags applied to staged AMD GPU sources")

find_path(ROCPRIM_INCLUDE_DIR rocprim/rocprim.hpp
  HINTS ${ROCM_PATH}/include)
find_path(HIPCUB_INCLUDE_DIR hipcub/hipcub.hpp
  HINTS ${ROCM_PATH}/include)

set(TP_HIP_LIBRARIES
    ${ROCM_PATH}/lib/libamdhip64.so
    ${ROCM_PATH}/lib/libhipblas.so
    ${ROCM_PATH}/lib/libhipblaslt.so
    ${ROCM_PATH}/lib/libhipsolver.so
    ${ROCM_PATH}/lib/libhipsparse.so
    ${ROCM_PATH}/lib/libhipfft.so
    ${ROCM_PATH}/lib/libhiprand.so
    ${ROCM_PATH}/lib/librocblas.so
    CACHE INTERNAL "HIP runtime and math libraries")
if(EXISTS ${ROCM_PATH}/lib/libMIOpen.so)
  list(APPEND TP_HIP_LIBRARIES ${ROCM_PATH}/lib/libMIOpen.so)
endif()
if(NOT WIN32)
  list(APPEND TP_HIP_LIBRARIES dl)
endif()

set(TP_HIP_INCLUDE_DIRS ${ROCM_PATH}/include CACHE INTERNAL "HIP include dirs")
set(TP_ROCM_PATH ${ROCM_PATH} CACHE INTERNAL "ROCm root")
set(TP_FOUND_HIP TRUE)
