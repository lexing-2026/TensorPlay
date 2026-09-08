cmake_minimum_required(VERSION 3.18)

project(tpx LANGUAGES CXX)

set(TPX_SOURCES
    src/Autograd.cpp
    src/AnomalyMode.cpp
    src/autocast_mode.cpp
    src/Engine.cpp
    src/SavedVariable.cpp
    ${GEN_TPX_OPS_CPP}
    ${GEN_TPX_AUTOGRAD_REG_CPP}
    ${GEN_AUTOCAST_CPP}
)

# The autograd surface is compiled into libp10, not into a separate dylib.
# Backend and composite kernels call tensorplay::tpx::ops entry points, so a
# standalone autograd library would create a p10<->tpx dependency cycle that
# linkers with strict shared-library symbol resolution (macOS two-level
# namespace) reject.  A single core library also binds these symbols without
# cross-image indirection and ships one fewer dylib per wheel.
# The GENERATED marker for the generated list lives in p10's scope, next to
# the p10 target; a directory-local property here would not cover the
# existence check CMake runs for target_sources at the owning scope.
target_sources(p10 PRIVATE ${TPX_SOURCES})
add_dependencies(p10 generate_code)

# Engine.cpp instantiates cpu::veccomplex AVX2 inline helpers (VecComplex.h);
# compile this TU with matching ISA flags so always_inline succeeds.  The
# complex fast paths themselves are runtime-guarded at their dispatch sites,
# and the helpers compile away entirely on non-x86 hosts, where the flags
# below are not valid options.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|i[3-9]86)")
    set_source_files_properties(src/Engine.cpp PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mfma")
endif()

# Header-only usage point: consumers keep including tpx headers and linking
# the tpx target for its include paths; the symbols now come from libp10.
add_library(tpx INTERFACE)
target_include_directories(tpx INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${GEN_BASE_DIR}
)
target_link_libraries(tpx INTERFACE p10)
