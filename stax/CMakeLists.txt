cmake_minimum_required(VERSION 3.18)
project(stax LANGUAGES CXX)

file(GLOB_RECURSE STAX_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(stax ${STAX_SOURCES})
set_target_properties(stax PROPERTIES UNITY_BUILD ON)
target_link_libraries(stax PRIVATE tpx p10)

target_include_directories(stax PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Static library: the Export.h header must expand STAX_API to nothing, the
# same contract the shared p10 library satisfies through its real export
# define. stax_EXPORTS is only defined by CMake for shared libraries, so the
# static branch is pinned explicitly here.
target_compile_definitions(stax PRIVATE TP_STATIC_BUILD)

target_link_libraries(stax PUBLIC p10)

# Output directories
if(NOT DEFINED TP_PACKAGE_DIR)
    set(TP_PACKAGE_DIR "${CMAKE_SOURCE_DIR}/tensorplay")
endif()
set(TP_LIB_DIR "${TP_PACKAGE_DIR}/lib")

set_target_properties(stax PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${TP_LIB_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TP_LIB_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${TP_LIB_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${TP_LIB_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${TP_LIB_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${TP_LIB_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${TP_LIB_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${TP_LIB_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${TP_LIB_DIR}"
)

install(TARGETS stax
    RUNTIME DESTINATION tensorplay/lib
    LIBRARY DESTINATION tensorplay/lib
    ARCHIVE DESTINATION tensorplay/lib
)
