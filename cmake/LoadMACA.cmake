# Locate the MACA SDK and expose what a device build needs.
#
# The SDK ships its own compiler driver (mxcc) and a CUDA-compatible surface
# under tools/cu-bridge: the runtime, BLAS, DNN, FFT, RNG, solver, sparse and
# collective headers keep their CUDA spelling, and the wrapper headers route
# every call to the mc* libraries.  A translation unit therefore keeps
# #include <cuda_runtime.h> and friends and is compiled by mxcc instead of a
# host C++ compiler.
#
# Sets:
#   TP_MACA_FOUND          TRUE when the SDK and cu-bridge are usable
#   TP_MACA_PATH           SDK root (/opt/maca by default)
#   TP_MACA_MXCC           compiler driver
#   TP_MACA_INCLUDE_DIRS   include search path for device and host TUs
#   TP_MACA_LIBRARIES      libraries a device build links
#   TP_MACA_DEVICE_ARCH    offload architecture (xcore1000 on C500)

set(TP_MACA_FOUND FALSE)

if(DEFINED ENV{MACA_PATH} AND EXISTS "$ENV{MACA_PATH}")
  set(TP_MACA_PATH "$ENV{MACA_PATH}" CACHE PATH "MACA SDK root")
else()
  set(TP_MACA_PATH "/opt/maca" CACHE PATH "MACA SDK root")
endif()

if(NOT EXISTS "${TP_MACA_PATH}/include/mcr/mc_runtime_api.h")
  message(STATUS "MACA SDK not found at ${TP_MACA_PATH}; the metax backend is unavailable.")
  return()
endif()

set(TP_MACA_CU_BRIDGE "${TP_MACA_PATH}/tools/cu-bridge" CACHE PATH
    "cu-bridge root inside the MACA SDK")
if(NOT EXISTS "${TP_MACA_CU_BRIDGE}/include/cuda_runtime.h")
  message(STATUS "cu-bridge not found at ${TP_MACA_CU_BRIDGE}; the metax backend is unavailable.")
  return()
endif()

set(TP_MACA_MXCC "${TP_MACA_PATH}/mxgpu_llvm/bin/mxcc" CACHE FILEPATH
    "MACA compiler driver")
if(NOT EXISTS "${TP_MACA_MXCC}")
  message(STATUS "mxcc not found at ${TP_MACA_MXCC}; the metax backend is unavailable.")
  return()
endif()

if(NOT TP_MACA_DEVICE_ARCH)
  set(TP_MACA_DEVICE_ARCH "xcore1000" CACHE STRING
      "MACA offload architecture (xcore1000 covers the C series)")
endif()

# Header layout of the SDK: the compatibility surface lives in cu-bridge,
# the native mc* headers in per-library subdirectories of include/.
#
# The cutlass compatibility layer ships only a handful of forwards; the
# full header set lives under mctlass/ with its own namespace and a macro
# rename in cutlass/cutlass_wrapper.h.  Generate a complete forward tree so
# a translation unit that includes <cutlass/...> resolves to the native
# headers.
set(TP_MACA_CUTLASS_SHIM "${CMAKE_BINARY_DIR}/metax_cutlass_shim")
file(GLOB_RECURSE _tp_mctlass_headers
     "${TP_MACA_PATH}/include/mctlass/*.h"
     "${TP_MACA_PATH}/include/mctlass/*.hpp")
foreach(_tp_hdr IN LISTS _tp_mctlass_headers)
  file(RELATIVE_PATH _tp_rel "${TP_MACA_PATH}/include/mctlass" "${_tp_hdr}")
  set(_tp_out "${TP_MACA_CUTLASS_SHIM}/cutlass/${_tp_rel}")
  set(_tp_body "#pragma once\n#include <cutlass/cutlass_wrapper.h>\n#include <mctlass/${_tp_rel}>\n")
  get_filename_component(_tp_dir "${_tp_out}" DIRECTORY)
  file(MAKE_DIRECTORY "${_tp_dir}")
  # Rewriting an unchanged file updates its timestamp and invalidates every
  # translation unit that includes it, turning each configure into a full
  # device rebuild.
  set(_tp_old "")
  if(EXISTS "${_tp_out}")
    file(READ "${_tp_out}" _tp_old)
  endif()
  if(NOT _tp_old STREQUAL _tp_body)
    file(WRITE "${_tp_out}" "${_tp_body}")
  endif()
endforeach()

set(TP_MACA_INCLUDE_DIRS
  "${TP_MACA_CUTLASS_SHIM}"
  "${TP_MACA_CU_BRIDGE}/include"
  "${TP_MACA_PATH}/include"
)
foreach(_tp_maca_sub
    mcr mcblas mcdnn mccl mcrand mcfft mcsolver mcsparse mcpti mcimage
    mcjpeg mctlass mctlassEx mcflashinfer flash_attn mccub cute)
  if(EXISTS "${TP_MACA_PATH}/include/${_tp_maca_sub}")
    list(APPEND TP_MACA_INCLUDE_DIRS "${TP_MACA_PATH}/include/${_tp_maca_sub}")
  endif()
endforeach()

# Host translation units see only the library headers.  The template-library
# directories (mctlass, cute, mccub, ...) ship files named like system
# headers -- mctlass/semaphore.h shadows <semaphore.h>, which <thread> and
# <mutex> pull in -- so they stay on the device compile line alone.  The
# common directory is part of the device compiler's own search path and holds
# the fp16 and complex type headers that the compatibility headers reach for,
# so it joins the host line explicitly.
set(TP_MACA_HOST_INCLUDE_DIRS
  "${TP_MACA_CU_BRIDGE}/include"
  "${TP_MACA_PATH}/include"
  "${TP_MACA_PATH}/include/common"
)
foreach(_tp_maca_sub
    mcr mcblas mcdnn mccl mcrand mcfft mcsolver mcsparse mcpti)
  if(EXISTS "${TP_MACA_PATH}/include/${_tp_maca_sub}")
    list(APPEND TP_MACA_HOST_INCLUDE_DIRS "${TP_MACA_PATH}/include/${_tp_maca_sub}")
  endif()
endforeach()

# Libraries a device build links against.  The compatibility surface and the
# native runtime are distinct libraries: the CUDA-spelling headers rename every
# runtime and driver call to a w-prefixed symbol, which lives in
# libruntime_cu.so, while the objects produced by the device compiler call the
# native mc* runtime in libmcruntime.so and the fat-binary registration entry
# points in libmccompiler.so.  The RTC calls go through the same split: the
# native compiler interface is in libmcruntime.so, its nvrtc-spelling face in
# libruntime_cu.so.
set(TP_MACA_LIBRARY_DIR "${TP_MACA_PATH}/lib64")
set(TP_MACA_LIBRARIES
  "${TP_MACA_LIBRARY_DIR}/libmcruntime.so"
  "${TP_MACA_LIBRARY_DIR}/libruntime_cu.so"
  "${TP_MACA_LIBRARY_DIR}/libmccompiler.so"
  "${TP_MACA_LIBRARY_DIR}/libmcblas.so"
  "${TP_MACA_LIBRARY_DIR}/libmcblasLt.so"
  "${TP_MACA_LIBRARY_DIR}/libmcdnn.so"
  "${TP_MACA_LIBRARY_DIR}/libmccl.so"
  "${TP_MACA_LIBRARY_DIR}/libmcrand.so"
  "${TP_MACA_LIBRARY_DIR}/libmcfft.so"
  "${TP_MACA_LIBRARY_DIR}/libmcsolver.so"
  "${TP_MACA_LIBRARY_DIR}/libmcsparse.so"
  "${TP_MACA_LIBRARY_DIR}/libmcpti.so"
)
foreach(_tp_maca_lib IN LISTS TP_MACA_LIBRARIES)
  if(NOT EXISTS "${_tp_maca_lib}")
    message(STATUS "MACA library missing: ${_tp_maca_lib}")
  endif()
endforeach()

# Flags shared by every device translation unit.
#
#   -D__CUDACC__          the sources mark host/device helpers with
#                         __host__ __device__, which the compatibility
#                         headers only enable in a CUDA compilation.
#   -Xdevice -D__CUDA_ARCH__=800
#                         device passes select the architecture-specific
#                         branches; the host pass keeps the portable ones.
#   -include climits      mc_runtime_api.h uses UINT_MAX without including
#                         the header that declares it.
set(TP_MACA_COMPILE_FLAGS
  -std=c++20
  -offload-arch=${TP_MACA_DEVICE_ARCH}
  -D__CUDACC__
  -DUSE_CUDA
  -DUSE_CUDNN
  -Xdevice
  -D__CUDA_ARCH__=800
  -include
  climits
  -fPIC
  -O3
)

# Host translation units that still pull the SDK's CUDA-spelling headers are
# compiled by the SDK driver as plain C++: no device architecture and no
# device-language macro, so the compatibility headers keep the host
# definitions of the standard library intact.  The language level, the
# backend defines and the workaround include stay the same.
set(TP_MACA_HOST_COMPILE_FLAGS
  -std=c++20
  -DUSE_CUDA
  -DUSE_CUDNN
  -DUSE_METAX
  -include
  climits
  -fPIC
  -O3
)

set(TP_MACA_FOUND TRUE)
message(STATUS "MACA SDK: ${TP_MACA_PATH} (mxcc: ${TP_MACA_MXCC}, arch: ${TP_MACA_DEVICE_ARCH})")

# A device translation unit holds several GB while mxcc optimizes it (the
# elementwise kernel files with their full dtype fan-out run past 10 GB),
# and the build host's core count says nothing about that.  Device compiles
# share one Ninja pool sized from physical memory; the host objects of the
# same target wait for them anyway, so the pool bounds peak memory without
# idling anything else.  Override with -DTP_MACA_JOBS=<n>.
if(NOT TP_MACA_MB_PER_JOB)
  set(TP_MACA_MB_PER_JOB 6000)
endif()
if(NOT TP_MACA_JOBS)
  cmake_host_system_information(RESULT _tp_maca_mem_mb
                                QUERY TOTAL_PHYSICAL_MEMORY)
  math(EXPR _tp_maca_jobs "${_tp_maca_mem_mb} / ${TP_MACA_MB_PER_JOB}")
  if(_tp_maca_jobs LESS 1)
    set(_tp_maca_jobs 1)
  elseif(_tp_maca_jobs GREATER 8)
    set(_tp_maca_jobs 8)
  endif()
  set(TP_MACA_JOBS "${_tp_maca_jobs}")
endif()
set_property(GLOBAL APPEND PROPERTY JOB_POOLS "tp_maca_device=${TP_MACA_JOBS}")
message(STATUS "MACA device compiles: ${TP_MACA_JOBS} concurrent")

# Compile one source into an object file with mxcc.
#
#   tp_maca_compile_object(<output-object> <source> <mode> <include-dirs...>)
#
# <mode> is DEVICE for a translation unit that carries device code and HOST
# for one that only happens to include the SDK's headers (see the flag sets
# above).  mxcc writes a make-style depfile next to the object, so a header
# edit recompiles exactly the translation units that include it.
function(tp_maca_compile_object out_obj src mode)
  if(mode STREQUAL "HOST")
    set(_flags ${TP_MACA_HOST_COMPILE_FLAGS})
  else()
    set(_flags ${TP_MACA_COMPILE_FLAGS})
  endif()
  set(_includes "")
  foreach(_dir IN LISTS ARGN)
    list(APPEND _includes "-I${_dir}")
  endforeach()
  get_filename_component(_obj_name "${src}" NAME)
  set(_obj_dir "${CMAKE_CURRENT_BINARY_DIR}/metax_objects")
  set(_obj "${_obj_dir}/${_obj_name}.o")
  # mxcc names the compiled object in the depfile after the path it was given
  # with -o.  That path is absolute, while the build edge declares the object
  # relative to the binary directory; the mismatch makes Ninja discard the
  # depfile and rebuild the object on every run.  -MT spells the target the
  # way the edge declares it.
  file(RELATIVE_PATH _obj_rel "${CMAKE_BINARY_DIR}" "${_obj}")
  add_custom_command(
    OUTPUT "${_obj}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_obj_dir}"
    COMMAND "${TP_MACA_MXCC}" ${_flags} ${_includes}
            -MD -MF "${_obj}.d" -MT "${_obj_rel}"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/${src}" -o "${_obj}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${src}"
    DEPFILE "${_obj}.d"
    JOB_POOL tp_maca_device
    VERBATIM
    COMMENT "mxcc ${_obj_name}")
  set(${out_obj} "${_obj}" PARENT_SCOPE)
endfunction()
