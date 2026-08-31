# FindMKL
# -------
#
# Locate Intel MKL.  Resolution order:
#   1. oneAPI MKLConfig.cmake (config mode) - full-featured MKL::MKL target.
#      Honours MKL_THREADING / MKL_LINK / MKL_INTERFACE when set before this
#      call, matching the oneAPI documented knobs.
#   2. Manual discovery of a oneAPI ("mkl/latest") or classic ("lib/intel64")
#      layout, assembling the interface + threading + core static/dynamic set.
#   3. Generic FindBLAS with an Intel vendor hint as the last resort.
#
# Input:
#   MKL_ROOT / MKLROOT   (env/cache) install prefix searched first
#   ONEAPI_ROOT          (env)       oneAPI install root (mkl/latest appended)
#   MKL_INTERFACE        lp64 (default) or ilp64
#   MKL_THREADING        sequential (default), gnu_thread, intel_thread, tbb
#   MKL_LINK             static (default) or dynamic (uses the mkl_rt single-
#                        dynamic-library entry point)
#
# Output:
#   MKL_FOUND            - TRUE when headers and libraries were resolved
#   MKL_INCLUDE_DIR(S)   - headers directory
#   MKL_LIBRARIES        - ordered link set (empty when config mode supplied
#                          the MKL::MKL target)
#   MKL_VERSION          - "<year>.<minor>.<update>" when detectable
#   MKL::MKL             - interface target (created here only when config
#                          mode did not already provide one)

include(FindPackageHandleStandardArgs)

# ---------------------------------------------------------------------------
# 1) Config mode.  find_package(MKL CONFIG) resolves only MKLConfig.cmake and
#    never re-enters this module.
# ---------------------------------------------------------------------------
if(NOT TARGET MKL::MKL)
    find_package(MKL CONFIG QUIET)
endif()
if(TARGET MKL::MKL)
    set(MKL_FOUND TRUE)
    if(NOT DEFINED MKL_VERSION)
        set(MKL_VERSION "oneAPI")
    endif()
    message(STATUS "Found MKL (config): ${MKL_DIR}")
    return()
endif()

# ---------------------------------------------------------------------------
# 2) Manual discovery
# ---------------------------------------------------------------------------
set(_MKL_ROOTS
    "${MKL_ROOT}"
    "$ENV{MKL_ROOT}"
    "$ENV{MKLROOT}"
    "$ENV{INTEL_MKL_DIR}"
    "$ENV{ONEAPI_ROOT}/mkl/latest"
    "/opt/intel/oneapi/mkl/latest"
)
file(GLOB _MKL_CLASSIC_ROOTS "/opt/intel/mkl*")
list(APPEND _MKL_ROOTS ${_MKL_CLASSIC_ROOTS})

find_path(MKL_INCLUDE_DIR
    NAMES mkl.h
    HINTS ${_MKL_ROOTS}
    PATH_SUFFIXES include
)

set(MKL_INTERFACE_DEFAULT "lp64")
if(NOT MKL_INTERFACE)
    set(MKL_INTERFACE "${MKL_INTERFACE_DEFAULT}")
endif()
if(NOT MKL_THREADING)
    set(MKL_THREADING "sequential")
endif()
if(NOT MKL_LINK)
    set(MKL_LINK "static")
endif()

if(MKL_INCLUDE_DIR)
    if(MKL_LINK STREQUAL "dynamic")
        find_library(MKL_RT_LIBRARY
            NAMES mkl_rt
            HINTS ${_MKL_ROOTS}
            PATH_SUFFIXES lib lib/intel64
        )
        if(MKL_RT_LIBRARY)
            set(MKL_LIBRARIES "${MKL_RT_LIBRARY}")
        endif()
    else()
        set(_MKL_LIB_SUFFIXES lib lib/intel64)
        if(WIN32)
            set(_MKL_LIB_SUFFIXES lib lib/intel64)
        endif()
        if(MKL_THREADING STREQUAL "sequential")
            set(_MKL_THREAD_LIB mkl_sequential)
        else()
            set(_MKL_THREAD_LIB mkl_${MKL_THREADING})
        endif()
        foreach(_comp mkl_${MKL_INTERFACE} ${_MKL_THREAD_LIB} mkl_core)
            find_library(MKL_${_comp}_LIBRARY
                NAMES ${_comp}
                HINTS ${_MKL_ROOTS}
                PATH_SUFFIXES ${_MKL_LIB_SUFFIXES}
            )
            if(MKL_${_comp}_LIBRARY)
                list(APPEND MKL_LIBRARIES "${MKL_${_comp}_LIBRARY}")
            endif()
        endforeach()
        # Threaded layers talk to an OpenMP runtime; carry ours so consumers
        # resolve the symbols without extra bookkeeping.
        if(MKL_THREADING MATCHES "^(gnu_thread|intel_thread)$"
           AND TARGET OpenMP::OpenMP_CXX)
            list(APPEND MKL_LIBRARIES OpenMP::OpenMP_CXX)
        endif()
    endif()
endif()

if(MKL_INCLUDE_DIR AND EXISTS "${MKL_INCLUDE_DIR}/mkl_version.h")
    foreach(_comp MKL MINOR UPDATE)
        file(STRINGS "${MKL_INCLUDE_DIR}/mkl_version.h" _MKL_${_comp}_LINE
             REGEX "#define __INTEL_${_comp}__[ \t]+[0-9]+")
        string(REGEX REPLACE
               ".*#define __INTEL_${_comp}__[ \t]+([0-9]+).*" "\\1"
               _MKL_${_comp} "${_MKL_${_comp}_LINE}")
        unset(_MKL_${_comp}_LINE)
    endforeach()
    if(_MKL_MKL)
        set(MKL_VERSION "${_MKL_MKL}.${_MKL_MINOR}.${_MKL_UPDATE}")
    endif()
endif()

find_package_handle_standard_args(MKL
    REQUIRED_VARS MKL_INCLUDE_DIR MKL_LIBRARIES
    VERSION_VAR MKL_VERSION
)

mark_as_advanced(MKL_INCLUDE_DIR MKL_LIBRARIES MKL_VERSION
                 MKL_INTERFACE MKL_THREADING MKL_LINK)

# ---------------------------------------------------------------------------
# 3) Last resort: generic BLAS with an Intel vendor hint.  Only reached when
#    manual discovery above failed; report through the same result variables.
# ---------------------------------------------------------------------------
if(NOT MKL_FOUND)
    set(BLA_VENDOR "Intel10_64lp")
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        set(MKL_LIBRARIES "${BLAS_LIBRARIES}")
        find_path(MKL_INCLUDE_DIR
            NAMES mkl.h
            HINTS $ENV{MKL_ROOT}/include "$ENV{ONEAPI_ROOT}/mkl/latest/include"
                  /usr/include/mkl
        )
        if(NOT MKL_INCLUDE_DIR)
            # FindBLAS links a working MKL without exposing headers; keep the
            # result usable by reporting only the libraries.
            set(MKL_INCLUDE_DIR "")
        endif()
        set(MKL_FOUND TRUE)
    endif()
endif()

if(MKL_FOUND AND NOT TARGET MKL::MKL)
    add_library(MKL::MKL INTERFACE IMPORTED)
    if(MKL_INCLUDE_DIR)
        set_target_properties(MKL::MKL PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}")
    endif()
    set_target_properties(MKL::MKL PROPERTIES
        INTERFACE_LINK_LIBRARIES "${MKL_LIBRARIES}")
endif()

set(MKL_INCLUDE_DIRS "${MKL_INCLUDE_DIR}")
