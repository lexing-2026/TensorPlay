# C++ test suite, built with -DBUILD_TESTS=ON.
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

add_executable(test_process_group_gloo
    cpp/distributed/test_process_group_gloo.cpp)
target_compile_definitions(test_process_group_gloo
    PRIVATE TENSORPLAY_ROOT="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_process_group_gloo
    PRIVATE tp_distributed pybind11::embed GTest::gtest Threads::Threads
            tpx stax)
add_test(NAME test_process_group_gloo COMMAND test_process_group_gloo)

add_subdirectory(cpp/vulkan)
