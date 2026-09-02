# Vulkan backend C++ tests, mirroring the per-backend subdirectory layout of
# the C++ suite.  Only added when the Vulkan backend is in the build.
if(USE_VULKAN)
    add_executable(test_vulkan
        api_test.cpp
        op_test.cpp
        op_extended_test.cpp
        test_vulkan_factory.cpp)
    target_include_directories(test_vulkan
        PRIVATE "${CMAKE_SOURCE_DIR}/tpx/include"
                "${CMAKE_SOURCE_DIR}/p10/src"
                "${CMAKE_SOURCE_DIR}/third_party/vulkan-headers/include"
                "${CMAKE_SOURCE_DIR}/third_party/vulkan-memory-allocator")
    target_link_libraries(test_vulkan PRIVATE p10 tpx GTest::gtest GTest::gtest_main Threads::Threads ${VULKAN_LIBRARY})
    add_test(NAME test_vulkan COMMAND test_vulkan)
endif()
