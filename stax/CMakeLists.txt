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

target_compile_definitions(stax PRIVATE STAX_EXPORTS)

target_link_libraries(stax PUBLIC p10)

# Output directories
set(TP_PACKAGE_DIR "${CMAKE_SOURCE_DIR}/tensorplay")
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
