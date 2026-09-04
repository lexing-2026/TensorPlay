find_package(rccl CONFIG QUIET)

if(TARGET roc::rccl)
    set(RCCL_FOUND TRUE)
    set(RCCL_INCLUDE_DIRS "${rccl_INCLUDE_DIRS}")
    set(RCCL_LIBRARIES roc::rccl)
endif()

if(NOT RCCL_FOUND)
    find_path(RCCL_INCLUDE_DIR
    NAMES rccl/rccl.h
    HINTS
        ${RCCL_ROOT}
        $ENV{RCCL_ROOT}
        ${ROCM_PATH}
        $ENV{ROCM_PATH}
        /opt/rocm
        PATH_SUFFIXES include)

    find_library(RCCL_LIBRARY
    NAMES rccl librccl librccl.so.1
    HINTS
        ${RCCL_ROOT}
        $ENV{RCCL_ROOT}
        ${ROCM_PATH}
        $ENV{ROCM_PATH}
        /opt/rocm
        PATH_SUFFIXES lib lib64)

    if(RCCL_INCLUDE_DIR AND EXISTS "${RCCL_INCLUDE_DIR}/rccl/rccl.h")
        file(STRINGS "${RCCL_INCLUDE_DIR}/rccl/rccl.h" RCCL_VERSION_LINE
             REGEX "^[ \t]*#define[ \t]+NCCL_VERSION_CODE[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
        string(REGEX REPLACE ".*NCCL_VERSION_CODE[ \t]+([0-9]+).*" "\\1"
               RCCL_VERSION_CODE "${RCCL_VERSION_LINE}")
        unset(RCCL_VERSION_LINE)
    endif()

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(
        RCCL REQUIRED_VARS RCCL_LIBRARY RCCL_INCLUDE_DIR
        VERSION_VAR RCCL_VERSION_CODE)

    if(RCCL_FOUND)
        set(RCCL_INCLUDE_DIRS "${RCCL_INCLUDE_DIR}")
        set(RCCL_LIBRARIES "${RCCL_LIBRARY}")
    endif()
endif()

if(RCCL_FOUND AND NOT TARGET tp::rccl)
    add_library(tp::rccl INTERFACE IMPORTED)
    target_include_directories(tp::rccl INTERFACE "${RCCL_INCLUDE_DIRS}")
    target_link_libraries(tp::rccl INTERFACE "${RCCL_LIBRARIES}")
endif()

if(RCCL_FOUND AND RCCL_INCLUDE_DIRS AND
   EXISTS "${RCCL_INCLUDE_DIRS}/rccl/rccl.h")
    file(STRINGS "${RCCL_INCLUDE_DIRS}/rccl/rccl.h" RCCL_GATHER_SCATTER
         REGEX "^[ \t]*#define[ \t]+RCCL_GATHER_SCATTER[ \t]+1")
    file(STRINGS "${RCCL_INCLUDE_DIRS}/rccl/rccl.h" RCCL_ALLTOALLV
         REGEX "^[ \t]*#define[ \t]+RCCL_ALLTOALLV[ \t]+1")
    if(NOT RCCL_GATHER_SCATTER OR NOT RCCL_ALLTOALLV)
        message(FATAL_ERROR
            "RCCL must provide native gather/scatter and all-to-all-v APIs.")
    endif()
    unset(RCCL_GATHER_SCATTER)
    unset(RCCL_ALLTOALLV)
endif()

mark_as_advanced(RCCL_INCLUDE_DIR RCCL_LIBRARY RCCL_VERSION_CODE)
