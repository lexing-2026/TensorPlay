# FindOpenBLAS
# ------------
#
# Locate an OpenBLAS installation.  Callers normally attempt config mode
# (OpenBLASConfig.cmake) first; this module covers pkg-config and manual
# prefix layouts.
#
# Input:
#   OpenBLAS_ROOT / OPENBLAS_ROOT_DIR  (env/cache) install prefix searched first
#
# Output:
#   OpenBLAS_FOUND         - TRUE when headers and library were found
#   OpenBLAS_INCLUDE_DIR(S)- header directory
#   OpenBLAS_LIBRARIES     - library to link
#   OpenBLAS_VERSION       - parsed from openblas_config.h when available
#   OpenBLAS::OpenBLAS     - interface target (created here only when config
#                            mode did not already provide one)

include(FindPackageHandleStandardArgs)

find_path(OpenBLAS_INCLUDE_DIR
    NAMES openblas_config.h cblas.h
    HINTS
        ${OpenBLAS_ROOT}
        $ENV{OPENBLAS_ROOT_DIR}
        $ENV{OpenBLAS_ROOT}
    PATH_SUFFIXES
        include
        include/openblas
        include/x86_64
)

find_library(OpenBLAS_LIBRARY
    NAMES openblas libopenblas openblas_static
    HINTS
        ${OpenBLAS_ROOT}
        $ENV{OPENBLAS_ROOT_DIR}
        $ENV{OpenBLAS_ROOT}
    PATH_SUFFIXES
        lib
        lib64
)

if(NOT OpenBLAS_LIBRARY)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(OpenBLAS_PC QUIET openblas)
        if(OpenBLAS_PC_FOUND)
            find_path(OpenBLAS_INCLUDE_DIR
                NAMES openblas_config.h cblas.h
                HINTS ${OpenBLAS_PC_INCLUDE_DIRS}
                NO_DEFAULT_PATH
            )
            find_library(OpenBLAS_LIBRARY
                NAMES openblas libopenblas
                HINTS ${OpenBLAS_PC_LIBRARY_DIRS}
                NO_DEFAULT_PATH
            )
        endif()
    endif()
endif()

if(OpenBLAS_INCLUDE_DIR AND EXISTS
   "${OpenBLAS_INCLUDE_DIR}/openblas_config.h")
    file(STRINGS "${OpenBLAS_INCLUDE_DIR}/openblas_config.h"
         OpenBLAS_VERSION_LINE
         REGEX "#define OPENBLAS_VERSION[ \t]+\"[0-9.]+\"")
    string(REGEX REPLACE
           ".*#define OPENBLAS_VERSION[ \t]+\"([0-9.]+)\".*" "\\1"
           OpenBLAS_VERSION "${OpenBLAS_VERSION_LINE}")
    unset(OpenBLAS_VERSION_LINE)
endif()

find_package_handle_standard_args(OpenBLAS
    REQUIRED_VARS OpenBLAS_LIBRARY OpenBLAS_INCLUDE_DIR
    VERSION_VAR OpenBLAS_VERSION
)

mark_as_advanced(OpenBLAS_INCLUDE_DIR OpenBLAS_LIBRARY OpenBLAS_VERSION)

if(OpenBLAS_FOUND)
    set(OpenBLAS_INCLUDE_DIRS "${OpenBLAS_INCLUDE_DIR}")
    set(OpenBLAS_LIBRARIES "${OpenBLAS_LIBRARY}")
    if(NOT TARGET OpenBLAS::OpenBLAS)
        add_library(OpenBLAS::OpenBLAS INTERFACE IMPORTED)
        set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${OpenBLAS_LIBRARY}")
    endif()
endif()
