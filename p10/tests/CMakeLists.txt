cmake_minimum_required(VERSION 3.18)
project(p10_native_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Location of the TensorPlay checkout that owns this test suite.
set(TP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(TP_GENERATED "${TP_ROOT}/build/generated")

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

add_executable(p10_tests
    main.cpp
    filecheck.cpp
    dtype_scalar.cpp
    tensor.cpp
    dispatch.cpp
    distributions.cpp
    generator.cpp
    exception.cpp
)

target_include_directories(p10_tests PRIVATE
    ${TP_ROOT}/p10/include
    ${TP_ROOT}/tpx/include
    ${TP_GENERATED}
)

target_link_libraries(p10_tests PRIVATE
    GTest::gtest
    Threads::Threads
    ${TP_ROOT}/tensorplay/lib/libp10.so
    ${TP_ROOT}/tensorplay/lib/libtpx.so
)

set_target_properties(p10_tests PROPERTIES
    BUILD_RPATH "${TP_ROOT}/tensorplay/lib"
)

include(GoogleTest)
gtest_discover_tests(p10_tests)
