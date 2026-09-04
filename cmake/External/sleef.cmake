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
    add_subdirectory(
        "${CMAKE_CURRENT_LIST_DIR}/../../third_party/sleef"
        "${CMAKE_BINARY_DIR}/third_party/sleef")
    # SLEEF's tlfloat ExternalProject bakes CMAKE_INSTALL_PREFIX into its
    # configure step at generate time; restore the project prefix only
    # after that value has been captured.
    set(CMAKE_INSTALL_PREFIX "${TP_SAVED_INSTALL_PREFIX}"
        CACHE PATH "" FORCE)
endfunction()

_tp_add_vendored_sleef()
message(STATUS "Building vendored SLEEF (static)")
