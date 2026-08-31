# FindMKLDNN
# ----------
#
# Provide the oneDNN (MKL-DNN) primitives for the CPU backend.
#
# The default path builds the vendored source tree at third_party/oneDNN as a
# static library.  A system installation is used only when MKLDNN_USE_SYSTEM
# is ON, or as a fallback when the vendored checkout is absent.
#
# Input:
#   MKLDNN_USE_SYSTEM    prefer a system oneDNN (DNNL_ROOT/ONEDNN_ROOT hints)
#   MKLDNN_CPU_RUNTIME   CPU threading runtime for the source build:
#                        "OMP" (default), "TBB" (requires TBB::tbb), "SEQ".
#   MKLDNN_USE_NATIVE_ARCH  when TRUE, let oneDNN tune for the build host
#                        (HostOpts); otherwise portable codegen ("" on x86).
#
# Output:
#   MKLDNN_FOUND         - TRUE when primitives are available
#   MKLDNN_INCLUDE_DIR   - include directories for consumers
#   MKLDNN_INCLUDE_DIRS  - alias of MKLDNN_INCLUDE_DIR
#   MKLDNN_LIBRARIES     - libraries (or the dnnl target) to link
#   MKLDNN_CPU_RUNTIME   - the runtime actually used by the source build

option(MKLDNN_USE_SYSTEM "Prefer a system oneDNN over the vendored source tree" OFF)

macro(_TP_MKLDNN_TRY_SYSTEM)
    find_path(MKLDNN_SYSTEM_INCLUDE_DIR
        NAMES dnnl.h dnnl.hpp
        HINTS
            $ENV{DNNL_ROOT}
            $ENV{DNNL_ROOT}/include
            $ENV{ONEDNN_ROOT}
            $ENV{ONEDNN_ROOT}/include
    )
    find_library(MKLDNN_SYSTEM_LIBRARY
        NAMES dnnl
        HINTS
            $ENV{DNNL_ROOT}/lib
            $ENV{DNNL_ROOT}/lib64
            $ENV{ONEDNN_ROOT}/lib
            $ENV{ONEDNN_ROOT}/lib64
    )
    if(MKLDNN_SYSTEM_INCLUDE_DIR AND MKLDNN_SYSTEM_LIBRARY)
        set(MKLDNN_INCLUDE_DIR "${MKLDNN_SYSTEM_INCLUDE_DIR}")
        set(MKLDNN_LIBRARIES "${MKLDNN_SYSTEM_LIBRARY}")
        set(MKLDNN_FOUND TRUE)
    endif()
endmacro()

if(MKLDNN_USE_SYSTEM)
    _TP_MKLDNN_TRY_SYSTEM()
    if(MKLDNN_FOUND)
        message(STATUS "Found oneDNN (system): ${MKLDNN_LIBRARIES}")
        return()
    endif()
endif()

# ---------------------------------------------------------------------------
# Vendored source tree (subdirectory build)
# ---------------------------------------------------------------------------
set(MKLDNN_ROOT "${PROJECT_SOURCE_DIR}/third_party/oneDNN")

find_path(MKLDNN_VENDORED_INCLUDE_DIR
    NAMES dnnl.h dnnl.hpp
    PATHS "${MKLDNN_ROOT}"
    PATH_SUFFIXES include include/oneapi/dnnl
)

if(NOT MKLDNN_VENDORED_INCLUDE_DIR)
    if(NOT MKLDNN_USE_SYSTEM)
        message(STATUS "Vendored oneDNN checkout not found; trying a system installation.")
        _TP_MKLDNN_TRY_SYSTEM()
        if(MKLDNN_FOUND)
            message(STATUS "Found oneDNN (system fallback): ${MKLDNN_LIBRARIES}")
            return()
        endif()
    endif()
    message(STATUS "oneDNN source files not found! Initialize the checkout:"
        "\n  mkdir -p third_party && cp -a <oneDNN source> third_party/oneDNN")
    set(MKLDNN_FOUND FALSE)
    return()
endif()

if(NOT MKLDNN_CPU_RUNTIME)
    set(MKLDNN_CPU_RUNTIME "OMP" CACHE STRING "oneDNN CPU runtime" FORCE)
elseif(MKLDNN_CPU_RUNTIME STREQUAL "TBB" AND NOT TARGET TBB::tbb)
    message(FATAL_ERROR "MKLDNN_CPU_RUNTIME=TBB requires TBB::tbb")
endif()
set(MKLDNN_CPU_RUNTIME "${MKLDNN_CPU_RUNTIME}" CACHE STRING "" FORCE)
message(STATUS "MKLDNN_CPU_RUNTIME = ${MKLDNN_CPU_RUNTIME}")

set(DNNL_CPU_RUNTIME "${MKLDNN_CPU_RUNTIME}" CACHE STRING "" FORCE)
set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(DNNL_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
set(DNNL_ENABLE_PRIMITIVE_CACHE ON CACHE BOOL "" FORCE)
set(DNNL_GRAPH_CPU_RUNTIME "${MKLDNN_CPU_RUNTIME}" CACHE STRING "" FORCE)
set(DNNL_GRAPH_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)

# The uKernels built-ins ship on x86; other CPU targets fall back to the
# classic primitive implementations.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i686)")
    set(DNNL_EXPERIMENTAL_UKERNEL ON CACHE BOOL "" FORCE)
endif()

# The graph API ships on Linux desktop/server configurations.
if(NOT APPLE AND NOT WIN32)
    set(ONEDNN_BUILD_GRAPH ON CACHE BOOL "" FORCE)
endif()

# Portable codegen by default; HostOpts only under an explicit native-arch
# request (release packaging stays compatible with older CPUs).
if(MKLDNN_USE_NATIVE_ARCH)
    set(DNNL_ARCH_OPT_FLAGS "HostOpts" CACHE STRING "" FORCE)
else()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"
       AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
        set(DNNL_ARCH_OPT_FLAGS "-mcpu=generic" CACHE STRING "" FORCE)
    else()
        set(DNNL_ARCH_OPT_FLAGS "" CACHE STRING "" FORCE)
    endif()
endif()

add_subdirectory("${MKLDNN_ROOT}" "${CMAKE_BINARY_DIR}/third_party/oneDNN")

if(NOT TARGET dnnl)
    message(STATUS "oneDNN source build did not produce the dnnl target")
    set(MKLDNN_FOUND FALSE)
    return()
endif()

# Build-tree only: the static archive is linked into p10, nothing is installed.
set_target_properties(dnnl PROPERTIES EXCLUDE_FROM_ALL ON)

# GCC emits a handful of known warnings inside oneDNN; keep the build log clean.
if(NOT APPLE AND CMAKE_COMPILER_IS_GNUCC)
    target_compile_options(dnnl PRIVATE -Wno-maybe-uninitialized)
    target_compile_options(dnnl PRIVATE -Wno-strict-overflow)
    target_compile_options(dnnl PRIVATE -Wno-error=strict-overflow)
endif()

# The OMP-runtime build compiles with -fopenmp but attaches no OpenMP link
# dependency to the dnnl target, expecting the consumer to provide the
# runtime. Carry ours on the interface so every consumer resolves omp_*/GOMP_*
# symbols without per-target bookkeeping.
if(MKLDNN_CPU_RUNTIME STREQUAL "OMP" AND TARGET OpenMP::OpenMP_CXX)
    set_property(TARGET dnnl APPEND PROPERTY INTERFACE_LINK_LIBRARIES
        OpenMP::OpenMP_CXX)
endif()

set(MKLDNN_INCLUDE_DIR
    "${MKLDNN_ROOT}/include"
    "${CMAKE_BINARY_DIR}/third_party/oneDNN/include")
set(MKLDNN_LIBRARIES dnnl)
set(MKLDNN_FOUND TRUE)
message(STATUS "Found oneDNN (vendored source build): ${MKLDNN_ROOT}")

mark_as_advanced(MKLDNN_INCLUDE_DIR MKLDNN_LIBRARIES MKLDNN_CPU_RUNTIME)
set(MKLDNN_INCLUDE_DIRS "${MKLDNN_INCLUDE_DIR}")
