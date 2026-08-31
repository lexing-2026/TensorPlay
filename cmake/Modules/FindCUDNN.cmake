# FindCUDNN
# ---------
#
# Locate the NVIDIA cuDNN deep-learning library and (optionally) the
# header-only cuDNN frontend.
#
# Input:
#   CUDNN_ROOT       (env/cache) install prefix searched first; CUDNN_ROOT_DIR
#                    is accepted as a legacy spelling
#   CUDNN_INCLUDE_DIR (env/cache) explicit header directory
#   CUDNN_LIBRARY    (env/cache) explicit library file; CUDNN_LIB_DIR is an
#                    alias handled by EnvVarForwarding
#   CUDNN_STATIC     (option)  prefer the static archive (default OFF)
#   CUDNN_FRONTEND_DIR (env)    cuDNN frontend root
#
# Output:
#   CUDNN_FOUND               - TRUE when both header and library were found
#   CUDNN_INCLUDE_PATH        - directory holding cudnn.h
#   CUDNN_LIBRARY_PATH        - library to link
#   CUDNN_VERSION             - "<major>.<minor>.<patch>" parsed from headers
#   CUDNN_INCLUDE_DIR/CUDNN_LIBRARY - aliases of the *_PATH variables kept for
#                    the existing p10 wiring
#   CUDNN_FRONTEND_INCLUDE_DIR - directory holding cudnn_frontend.h (optional;
#                    no failure when absent)

include(FindPackageHandleStandardArgs)

set(CUDNN_ROOT "$ENV{CUDNN_ROOT_DIR}" CACHE PATH "Folder containing NVIDIA cuDNN")
if(DEFINED ENV{CUDNN_ROOT_DIR})
    message(STATUS "CUDNN_ROOT_DIR is a legacy spelling; prefer CUDNN_ROOT.")
endif()
list(APPEND CUDNN_ROOT $ENV{CUDNN_ROOT_DIR} $ENV{CUDNN_ROOT} ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CUDNN_ROOT})

set(CUDNN_INCLUDE_DIR
    "$ENV{CUDNN_INCLUDE_DIR}"
    CACHE PATH "Folder containing NVIDIA cuDNN header files")

find_path(CUDNN_INCLUDE_PATH
    NAMES cudnn.h
    HINTS ${CUDNN_INCLUDE_DIR}
    PATH_SUFFIXES cuda/include cuda include
)

option(CUDNN_STATIC "Look for the static cuDNN archive" OFF)
if(CUDNN_STATIC)
    set(CUDNN_LIBNAME "cudnn_static")
else()
    set(CUDNN_LIBNAME "cudnn")
endif()

set(CUDNN_LIBRARY
    "$ENV{CUDNN_LIBRARY}"
    CACHE PATH "Path to the cuDNN library file (e.g., libcudnn.so)")
if(CUDNN_LIBRARY MATCHES "cudnn_static" AND NOT CUDNN_STATIC)
    message(STATUS "CUDNN_LIBRARY points at a static archive but CUDNN_STATIC is OFF.")
endif()

find_library(CUDNN_LIBRARY_PATH
    NAMES ${CUDNN_LIBNAME}
    PATHS ${CUDNN_LIBRARY}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64
)

# The pip nvidia-cudnn wheel ships headers and versioned sonames outside any
# standard prefix; ask the running interpreter where the package lives.  This
# is a TensorPlay addition covering pip-installed CUDA runtimes.
if((NOT CUDNN_LIBRARY_PATH OR NOT CUDNN_INCLUDE_PATH) AND DEFINED Python_EXECUTABLE)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0])"
        OUTPUT_VARIABLE _cudnn_wheel_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_cudnn_wheel_dir)
        find_library(CUDNN_LIBRARY_PATH
            NAMES ${CUDNN_LIBNAME} libcudnn.so libcudnn.so.9
            HINTS "${_cudnn_wheel_dir}/lib"
            NO_DEFAULT_PATH
        )
        find_path(CUDNN_INCLUDE_PATH
            NAMES cudnn.h
            HINTS "${_cudnn_wheel_dir}/include"
            NO_DEFAULT_PATH
        )
    endif()
    unset(_cudnn_wheel_dir)
endif()

# Version detection: v8+ exposes the triple in cudnn_version.h; older
# releases carry it directly in cudnn.h.
set(CUDNN_VERSION "")
if(CUDNN_INCLUDE_PATH)
    set(_CUDNN_VERSION_HEADER "${CUDNN_INCLUDE_PATH}/cudnn_version.h")
    if(NOT EXISTS "${_CUDNN_VERSION_HEADER}")
        set(_CUDNN_VERSION_HEADER "${CUDNN_INCLUDE_PATH}/cudnn.h")
    endif()
    if(EXISTS "${_CUDNN_VERSION_HEADER}")
        foreach(_comp MAJOR MINOR PATCHLEVEL)
            file(STRINGS "${_CUDNN_VERSION_HEADER}" _CUDNN_${_comp}_LINE
                 REGEX "#define CUDNN_${_comp}[ \t]+[0-9]+")
            string(REGEX REPLACE
                   ".*#define CUDNN_${_comp}[ \t]+([0-9]+).*" "\\1"
                   CUDNN_VERSION_${_comp} "${_CUDNN_${_comp}_LINE}")
            unset(_CUDNN_${_comp}_LINE)
        endforeach()
        if(CUDNN_VERSION_MAJOR)
            set(CUDNN_VERSION
                "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCHLEVEL}")
        else()
            set(CUDNN_VERSION "?")
        endif()
    endif()
    unset(_CUDNN_VERSION_HEADER)
endif()

# Header-only frontend (graph API). Optional: the runtime-only path builds
# fine without it.
find_path(CUDNN_FRONTEND_INCLUDE_DIR
    NAMES cudnn_frontend.h
    HINTS
        "$ENV{CUDNN_FRONTEND_DIR}/include"
        "${CUDNN_INCLUDE_PATH}/cudnn_frontend"
        "${CUDNN_INCLUDE_PATH}"
)

find_package_handle_standard_args(CUDNN
    REQUIRED_VARS CUDNN_LIBRARY_PATH CUDNN_INCLUDE_PATH
    VERSION_VAR CUDNN_VERSION
)

mark_as_advanced(CUDNN_ROOT CUDNN_INCLUDE_PATH CUDNN_LIBRARY_PATH
                 CUDNN_VERSION CUDNN_FRONTEND_INCLUDE_DIR)

if(CUDNN_FOUND)
    set(CUDNN_INCLUDE_DIR "${CUDNN_INCLUDE_PATH}")
    set(CUDNN_LIBRARY "${CUDNN_LIBRARY_PATH}")
    set(CUDNN_INCLUDE_DIRS "${CUDNN_INCLUDE_PATH}")
endif()
