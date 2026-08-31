# Vulkan backend C++ tests, mirroring the per-backend subdirectory layout of
# the C++ suite.  Only added when the Vulkan backend is in the build.
if(USE_VULKAN)
    add_executable(test_vulkan
        api_test.cpp
        op_test.cpp
        test_vulkan_factory.cpp)
    target_link_libraries(test_vulkan PRIVATE p10 GTest::gtest Threads::Threads)
    add_test(NAME test_vulkan COMMAND test_vulkan)
endif()
