cmake_minimum_required(VERSION 3.18)

project(tpx LANGUAGES CXX)

set(TPX_SOURCES
    src/Autograd.cpp
    src/Engine.cpp
    src/Tensor.cpp
    src/AutogradMetaInterface.cpp
    ${GEN_TPX_OPS_CPP}
)

set_source_files_properties(${GEN_TPX_OPS_CPP} PROPERTIES GENERATED TRUE)

add_library(tpx ${TPX_SOURCES})
set_target_properties(tpx PROPERTIES UNITY_BUILD ON)
add_dependencies(tpx generate_code)
target_link_libraries(tpx PRIVATE p10)

if(BUILD_SHARED_LIBS)
    target_compile_definitions(tpx PRIVATE TENSORPLAY_EXPORTS)
endif()

target_include_directories(tpx PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${GEN_BASE_DIR}
)

target_compile_definitions(tpx PRIVATE TENSORPLAY_EXPORTS)

target_link_libraries(tpx PUBLIC p10)

# Installation logic for tpx
set(TP_PACKAGE_DIR "${CMAKE_SOURCE_DIR}/tensorplay")
set(TP_LIB_DIR "${TP_PACKAGE_DIR}/lib")

set_target_properties(tpx PROPERTIES
    OUTPUT_NAME "tpx"
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

install(TARGETS tpx
    RUNTIME DESTINATION tensorplay/lib
    LIBRARY DESTINATION tensorplay/lib
    ARCHIVE DESTINATION tensorplay/lib
)
