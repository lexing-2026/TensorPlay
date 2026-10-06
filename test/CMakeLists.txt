# C++ test suite, built with -DBUILD_TESTS=ON.
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

add_executable(test_intrusive_ptr
    cpp/test_intrusive_ptr.cpp)
target_link_libraries(test_intrusive_ptr
    PRIVATE p10 GTest::gtest Threads::Threads)
add_test(NAME test_intrusive_ptr COMMAND test_intrusive_ptr)

add_executable(test_process_group_gloo
    cpp/distributed/test_process_group_gloo.cpp)
target_compile_definitions(test_process_group_gloo
    PRIVATE TENSORPLAY_ROOT="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_process_group_gloo
    PRIVATE tp_distributed pybind11::embed GTest::gtest Threads::Threads
            tpx stax)
add_test(NAME test_process_group_gloo COMMAND test_process_group_gloo)

add_subdirectory(cpp/vulkan)

if(TP_USE_CUDA)
    add_executable(test_cuda_reduction_dimensions
        cpp/cuda/test_reduction_dimensions.cu)
    target_link_libraries(test_cuda_reduction_dimensions PRIVATE p10 CUDA::cudart)
    target_compile_definitions(test_cuda_reduction_dimensions PRIVATE
        CCCL_DISABLE_CTK_COMPATIBILITY_CHECK)
    target_compile_options(test_cuda_reduction_dimensions PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    add_test(NAME test_cuda_reduction_dimensions COMMAND test_cuda_reduction_dimensions)
    set_tests_properties(test_cuda_reduction_dimensions PROPERTIES SKIP_RETURN_CODE 77)
endif()
