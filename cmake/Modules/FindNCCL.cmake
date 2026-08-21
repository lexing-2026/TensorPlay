# FindNCCL
# -------
# Mirrors pytorch's cmake/Modules/FindNCCL.cmake.
#
# Input:
#   NCCL_ROOT / ENV{NCCL_ROOT} - install prefix to search first
#
# Output:
#   NCCL_FOUND        - TRUE when both header and library were found
#   NCCL_INCLUDE_DIRS - where nccl.h lives
#   NCCL_LIBRARIES    - library to link
#   NCCL_VERSION      - numeric version parsed from nccl.h (e.g. 22907)

find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS
        ${NCCL_ROOT}/include
        $ENV{NCCL_ROOT}/include
        ${CUDA_HOME}/include
)

# The pip wheel ships only the versioned soname (libnccl.so.2), so search for
# that explicitly in addition to the dev names torch looks for.
find_library(NCCL_LIBRARY
    NAMES nccl libnccl nccl_static libnccl.so.2
    HINTS
        ${NCCL_ROOT}/lib
        ${NCCL_ROOT}/lib64
        $ENV{NCCL_ROOT}/lib
        $ENV{NCCL_ROOT}/lib64
        ${CUDA_HOME}/lib64
)

# Extra hint beyond torch: the nvidia-nccl-cuXX pip wheel layout.
if(DEFINED Python_EXECUTABLE AND NOT NCCL_LIBRARY)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import nvidia.nccl; print(list(nvidia.nccl.__path__)[0])"
        OUTPUT_VARIABLE _nccl_wheel_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_nccl_wheel_dir)
        find_library(NCCL_LIBRARY
            NAMES nccl libnccl libnccl.so.2
            HINTS "${_nccl_wheel_dir}/lib"
            NO_DEFAULT_PATH
        )
        if(NOT NCCL_INCLUDE_DIR)
            find_path(NCCL_INCLUDE_DIR NAMES nccl.h
                HINTS "${_nccl_wheel_dir}/include" NO_DEFAULT_PATH)
        endif()
    endif()
    unset(_nccl_wheel_dir)
endif()

if(NCCL_INCLUDE_DIR AND EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
    file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" NCCL_VERSION_LINE
         REGEX "#define NCCL_VERSION_CODE")
    string(REGEX REPLACE ".*#define NCCL_VERSION_CODE ([0-9]+).*" "\\1"
           NCCL_VERSION "${NCCL_VERSION_LINE}")
    unset(NCCL_VERSION_LINE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    NCCL REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR VERSION_VAR NCCL_VERSION)
mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY NCCL_VERSION)

if(NCCL_FOUND)
    set(NCCL_LIBRARIES ${NCCL_LIBRARY})
    set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
endif()
