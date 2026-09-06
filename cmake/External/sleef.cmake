# SLEEF vector math library (vendored at third_party/sleef).
#
# Built as a static libsleef with libm only: no DFT, no quad, no tests.
# SLEEF 4.x compiles its scalar fallback objects against TLFloat, which it
# provisions through its own ExternalProject from submodules/tlfloat; the
# install prefix is therefore scoped to a private directory under the build
# tree so nothing leaks into the project's own install rules.
#
# After this file is included, the `sleef` target exists (unless disabled)
# and p10 links it.  Call sites go through p10/include/cpu/vec/SleefShims.h,
# which declares the runtime-dispatched entry points directly and therefore
# does not need sleef.h on the include path.

option(USE_SLEEF "Build the vendored SLEEF vector math library" ON)
option(USE_SYSTEM_SLEEF "Link a system-provided libsleef instead of the vendored one" OFF)

if(NOT USE_SLEEF)
    return()
endif()

if(USE_SYSTEM_SLEEF)
    find_library(SLEEF_LIBRARY NAMES sleef)
    if(NOT SLEEF_LIBRARY)
        message(FATAL_ERROR "USE_SYSTEM_SLEEF=ON but libsleef was not found")
    endif()
    message(STATUS "Found system SLEEF: ${SLEEF_LIBRARY}")
    add_library(sleef UNKNOWN IMPORTED)
    set_target_properties(sleef PROPERTIES IMPORTED_LOCATION "${SLEEF_LIBRARY}")
    return()
endif()

if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../../third_party/sleef/CMakeLists.txt")
    # No vendored checkout (for example a CI build without third_party):
    # degrade to libm scalar paths instead of failing the configure.
    message(STATUS
        "Vendored SLEEF checkout not found at third_party/sleef; "
        "building without vector math acceleration.")
    set(USE_SLEEF OFF CACHE BOOL "" FORCE)
    return()
endif()
if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../../third_party/sleef/submodules/tlfloat/CMakeLists.txt")
    message(FATAL_ERROR
        "third_party/sleef/submodules/tlfloat is empty; SLEEF 4.x needs it for "
        "the scalar fallback objects (git submodule update --init submodules/tlfloat).")
endif()

# Everything set inside this function stays function-local except the
# install-prefix cache entry SLEEF's own configure checks; the private
# prefix keeps the vendored tree's install rules out of the project's.
function(_tp_add_vendored_sleef)
    set(TP_SAVED_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/third_party/sleef-prefix"
        CACHE PATH "" FORCE)
    unset(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    unset(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT CACHE)
    # The vendored tree provisions tlfloat itself; never let SLEEF's
    # pkg-config probe answer instead, because it resolves the dependency
    # to a bare -l name whose search directory stays scoped to SLEEF's
    # subdirectory and never reaches the compute library's link line.
    set(CMAKE_DISABLE_FIND_PACKAGE_PkgConfig ON)
    set(BUILD_SHARED_LIBS OFF)
    set(SLEEF_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(SLEEF_BUILD_LIBM ON CACHE BOOL "" FORCE)
    set(SLEEF_BUILD_DFT OFF CACHE BOOL "" FORCE)
    set(SLEEF_BUILD_QUAD OFF CACHE BOOL "" FORCE)
    set(SLEEF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(SLEEF_ENABLE_TESTER4 OFF CACHE BOOL "" FORCE)
    set(SLEEF_ENABLE_TLFLOAT ON CACHE BOOL "" FORCE)
    set(SLEEF_ENABLE_MPFR OFF CACHE BOOL "" FORCE)
    set(SLEEF_ENABLE_SSL OFF CACHE BOOL "" FORCE)
    set(SLEEF_ENABLE_FFTW OFF CACHE BOOL "" FORCE)
    set(SLEEF_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
    set(SLEEF_ENABLE_LTO OFF CACHE BOOL "" FORCE)
    set(SLEEF_SHOW_CONFIG OFF CACHE BOOL "" FORCE)
    # Per-target SIMD variants: the vec backends of the compute library call
    # the SLEEF entry points compiled for the matching ISA (VSX on PowerPC,
    # VXE on s390x, SVE on aarch64), so each architecture must request its
    # variant here in addition to the host-detected x86 set.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc|ppc)64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc|ppc)")
        set(SLEEF_ENABLE_VSX ON CACHE BOOL "" FORCE)
        set(SLEEF_ENABLE_VSX3 ON CACHE BOOL "" FORCE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "s390x")
        set(SLEEF_ENABLE_VXE ON CACHE BOOL "" FORCE)
        set(SLEEF_ENABLE_VXE2 ON CACHE BOOL "" FORCE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(SLEEF_ENABLE_SVE ON CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(
        "${CMAKE_CURRENT_LIST_DIR}/../../third_party/sleef"
        "${CMAKE_BINARY_DIR}/third_party/sleef")
    # SLEEF's tlfloat ExternalProject bakes CMAKE_INSTALL_PREFIX into its
    # configure step at generate time; restore the project prefix only
    # after that value has been captured.
    set(CMAKE_INSTALL_PREFIX "${TP_SAVED_INSTALL_PREFIX}"
        CACHE PATH "" FORCE)
    set(CMAKE_DISABLE_FIND_PACKAGE_PkgConfig OFF)
endfunction()

_tp_add_vendored_sleef()
message(STATUS "Building vendored SLEEF (static)")
# Some of SLEEF's tlfloat resolution paths answer with a bare -l style
# entry whose search directory stays scoped to SLEEF's subdirectory and
# never reaches the compute library's link line. Scan the resulting link
# interface and swap any such entry for the archive the vendored install
# step produces.
if(TARGET sleef)
    get_target_property(TP_SLEEF_IFACE sleef INTERFACE_LINK_LIBRARIES)
    if(TP_SLEEF_IFACE)
        set(TP_TLFLOAT_ARCHIVE
            "${CMAKE_BINARY_DIR}/third_party/sleef-prefix/lib/${CMAKE_STATIC_LIBRARY_PREFIX}tlfloat${CMAKE_STATIC_LIBRARY_SUFFIX}")
        set(TP_SLEEF_IFACE_FIXED "")
        set(TP_SLEEF_IFACE_CHANGED FALSE)
        foreach(TP_IFACE_ITEM ${TP_SLEEF_IFACE})
            if(TP_IFACE_ITEM STREQUAL "tlfloat" OR TP_IFACE_ITEM STREQUAL "-ltlfloat")
                list(APPEND TP_SLEEF_IFACE_FIXED "${TP_TLFLOAT_ARCHIVE}")
                set(TP_SLEEF_IFACE_CHANGED TRUE)
            else()
                list(APPEND TP_SLEEF_IFACE_FIXED "${TP_IFACE_ITEM}")
            endif()
        endforeach()
        if(TP_SLEEF_IFACE_CHANGED)
            # The archive is produced by the tlfloat ExternalProject's
            # install step during the build, not by a node in this build
            # graph; ninja rejects a missing file it has no rule for, so
            # seed the location now. The dependency chain (p10 -> sleef ->
            # ext_tlfloat) guarantees the real archive overwrites the seed
            # before anything links against it.
            file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/third_party/sleef-prefix/lib")
            file(TOUCH "${TP_TLFLOAT_ARCHIVE}")
            set_target_properties(sleef PROPERTIES
                INTERFACE_LINK_LIBRARIES "${TP_SLEEF_IFACE_FIXED}")
            message(STATUS "Pinned the sleef tlfloat link entry to ${TP_TLFLOAT_ARCHIVE}")
        else()
            message(STATUS "sleef link interface: ${TP_SLEEF_IFACE_FIXED}")
        endif()
    endif()
endif()
