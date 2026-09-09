# FindMKL
# -------
#
# Locate Intel MKL.  Resolution order:
#   1. Manual discovery of a oneAPI ("mkl/latest") or classic ("lib/intel64")
#      layout, assembling the interface + threading + core static/dynamic set.
#   2. Generic FindBLAS with an Intel vendor hint as the last resort.
#
# Input:
#   MKL_ROOT / MKLROOT   (env/cache) install prefix searched first
#   ONEAPI_ROOT          (env)       oneAPI install root (mkl/latest appended)
#   MKL_INTERFACE        lp64 (default) or ilp64
#   MKL_THREADING        optional threading preference: sequential,
#                        gnu_thread, intel_thread, or tbb
#   MKL_LINK             static (default) or dynamic (uses the mkl_rt single-
#                        dynamic-library entry point)
#
# Output:
#   MKL_FOUND            - TRUE when headers and libraries were resolved
#   MKL_INCLUDE_DIR(S)   - headers directory
#   MKL_LIBRARIES        - ordered link set (empty when config mode supplied
#                          the MKL::MKL target)
#   MKL_VERSION          - "<year>.<minor>.<update>" when detectable
#   MKL_OPENMP_TYPE      - selected OpenMP implementation (GNU or Intel)
#   MKL_OPENMP_LIBRARY   - selected OpenMP runtime link item
#   MKL::MKL             - interface target (created here only when config
#                          mode did not already provide one)

include(FindPackageHandleStandardArgs)

# ---------------------------------------------------------------------------
# 1) Reuse an existing target when the caller supplied one.
# ---------------------------------------------------------------------------
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
if(NOT MKL_LINK)
    set(MKL_LINK "static")
endif()

include(CheckCXXSourceCompiles)

set(MKL_OPENMP_TYPE "")
set(MKL_OPENMP_LIBRARY "")

# Static archives need the platform services used internally by the dispatch
# and threading layers.  Apple provides these through its system runtime.
set(_MKL_SYSTEM_LIBRARIES)
if(UNIX AND NOT APPLE)
    list(APPEND _MKL_SYSTEM_LIBRARIES pthread m ${CMAKE_DL_LIBS})
endif()

function(_tp_mkl_link_works _result)
    set(CMAKE_REQUIRED_LIBRARIES ${ARGN} ${ARGN})
    set(CMAKE_REQUIRED_QUIET TRUE)
    unset(TP_MKL_CANDIDATE_LINKS CACHE)
    check_cxx_source_compiles(
        "extern \"C\" void cblas_sgemm(); int main() { cblas_sgemm(); return 0; }"
        TP_MKL_CANDIDATE_LINKS)
    set(${_result} "${TP_MKL_CANDIDATE_LINKS}" PARENT_SCOPE)
endfunction()

if(MKL_INCLUDE_DIR)
    if(MKL_LINK STREQUAL "dynamic")
        find_library(MKL_RT_LIBRARY
            NAMES mkl_rt
            HINTS ${_MKL_ROOTS}
            PATH_SUFFIXES lib lib/intel64
        )
        if(MKL_RT_LIBRARY)
            _tp_mkl_link_works(_MKL_RT_WORKS "${MKL_RT_LIBRARY}")
            if(_MKL_RT_WORKS)
                set(MKL_LIBRARIES "${MKL_RT_LIBRARY}")
            endif()
        endif()
    else()
        set(_MKL_LIB_SUFFIXES lib lib/intel64)
        set(_MKL_INTERFACE_NAMES
            "mkl_${MKL_INTERFACE}"
            "mkl_intel_${MKL_INTERFACE}")
        if(NOT WIN32)
            list(APPEND _MKL_INTERFACE_NAMES "mkl_gf_${MKL_INTERFACE}")
        endif()

        if(MKL_THREADING)
            set(_MKL_THREADING_CANDIDATES "${MKL_THREADING}")
            if(NOT MKL_THREADING STREQUAL "sequential")
                list(APPEND _MKL_THREADING_CANDIDATES sequential)
            endif()
        elseif(WIN32)
            set(_MKL_THREADING_CANDIDATES intel_thread sequential)
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set(_MKL_THREADING_CANDIDATES gnu_thread intel_thread sequential)
        else()
            set(_MKL_THREADING_CANDIDATES intel_thread sequential)
        endif()
        list(REMOVE_DUPLICATES _MKL_THREADING_CANDIDATES)

        find_library(MKL_CORE_LIBRARY
            NAMES mkl_core
            HINTS ${_MKL_ROOTS}
            PATH_SUFFIXES ${_MKL_LIB_SUFFIXES})

        foreach(_MKL_INTERFACE_NAME IN LISTS _MKL_INTERFACE_NAMES)
            string(MAKE_C_IDENTIFIER "${_MKL_INTERFACE_NAME}" _MKL_INTERFACE_ID)
            find_library(MKL_INTERFACE_${_MKL_INTERFACE_ID}_LIBRARY
                NAMES "${_MKL_INTERFACE_NAME}"
                HINTS ${_MKL_ROOTS}
                PATH_SUFFIXES ${_MKL_LIB_SUFFIXES}
            )
            if(NOT MKL_INTERFACE_${_MKL_INTERFACE_ID}_LIBRARY OR
               NOT MKL_CORE_LIBRARY)
                continue()
            endif()

            foreach(_MKL_THREADING_NAME IN LISTS _MKL_THREADING_CANDIDATES)
                if(_MKL_THREADING_NAME STREQUAL "sequential")
                    set(_MKL_THREAD_LIBRARY_NAMES mkl_sequential)
                    set(_MKL_RUNTIME_LIBRARIES)
                    set(_MKL_CANDIDATE_OPENMP_TYPE "")
                    set(_MKL_CANDIDATE_OPENMP_LIBRARY "")
                elseif(_MKL_THREADING_NAME STREQUAL "gnu_thread")
                    set(_MKL_THREAD_LIBRARY_NAMES mkl_gnu_thread)
                    if(NOT TARGET OpenMP::OpenMP_CXX)
                        continue()
                    endif()
                    set(_MKL_RUNTIME_LIBRARIES OpenMP::OpenMP_CXX)
                    set(_MKL_CANDIDATE_OPENMP_TYPE GNU)
                    set(_MKL_CANDIDATE_OPENMP_LIBRARY OpenMP::OpenMP_CXX)
                elseif(_MKL_THREADING_NAME STREQUAL "intel_thread")
                    set(_MKL_THREAD_LIBRARY_NAMES mkl_intel_thread)
                    find_library(MKL_INTEL_RUNTIME_LIBRARY
                        NAMES libiomp5md iomp5
                        HINTS ${_MKL_ROOTS}
                        PATH_SUFFIXES ${_MKL_LIB_SUFFIXES})
                    if(NOT MKL_INTEL_RUNTIME_LIBRARY)
                        continue()
                    endif()
                    set(_MKL_RUNTIME_LIBRARIES "${MKL_INTEL_RUNTIME_LIBRARY}")
                    set(_MKL_CANDIDATE_OPENMP_TYPE Intel)
                    set(_MKL_CANDIDATE_OPENMP_LIBRARY "${MKL_INTEL_RUNTIME_LIBRARY}")
                elseif(_MKL_THREADING_NAME STREQUAL "tbb")
                    set(_MKL_THREAD_LIBRARY_NAMES mkl_tbb_thread)
                    find_library(MKL_TBB_RUNTIME_LIBRARY
                        NAMES tbb12 tbb
                        HINTS ${_MKL_ROOTS}
                        PATH_SUFFIXES ${_MKL_LIB_SUFFIXES})
                    if(NOT MKL_TBB_RUNTIME_LIBRARY)
                        continue()
                    endif()
                    set(_MKL_RUNTIME_LIBRARIES "${MKL_TBB_RUNTIME_LIBRARY}")
                    set(_MKL_CANDIDATE_OPENMP_TYPE "")
                    set(_MKL_CANDIDATE_OPENMP_LIBRARY "")
                else()
                    message(FATAL_ERROR "Unsupported MKL_THREADING value: ${_MKL_THREADING_NAME}")
                endif()

                string(MAKE_C_IDENTIFIER "${_MKL_THREADING_NAME}" _MKL_THREADING_ID)
                find_library(MKL_THREAD_${_MKL_THREADING_ID}_LIBRARY
                    NAMES ${_MKL_THREAD_LIBRARY_NAMES}
                    HINTS ${_MKL_ROOTS}
                    PATH_SUFFIXES ${_MKL_LIB_SUFFIXES})
                if(NOT MKL_THREAD_${_MKL_THREADING_ID}_LIBRARY)
                    continue()
                endif()

                set(_MKL_CANDIDATE_LIBRARIES
                    "${MKL_INTERFACE_${_MKL_INTERFACE_ID}_LIBRARY}"
                    "${MKL_THREAD_${_MKL_THREADING_ID}_LIBRARY}"
                    "${MKL_CORE_LIBRARY}"
                    ${_MKL_RUNTIME_LIBRARIES}
                    ${_MKL_SYSTEM_LIBRARIES})
                _tp_mkl_link_works(_MKL_CANDIDATE_WORKS
                    ${_MKL_CANDIDATE_LIBRARIES})
                if(_MKL_CANDIDATE_WORKS)
                    set(MKL_LIBRARIES ${_MKL_CANDIDATE_LIBRARIES})
                    set(MKL_THREADING "${_MKL_THREADING_NAME}")
                    set(MKL_OPENMP_TYPE "${_MKL_CANDIDATE_OPENMP_TYPE}")
                    set(MKL_OPENMP_LIBRARY "${_MKL_CANDIDATE_OPENMP_LIBRARY}")
                    break()
                endif()
            endforeach()
            if(MKL_LIBRARIES)
                break()
            endif()
        endforeach()
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
