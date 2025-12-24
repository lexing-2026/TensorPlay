cmake_minimum_required(VERSION 3.18)

project(p10 LANGUAGES CXX)

set(P10_SOURCES
    src/Allocator.cpp
    src/Device.cpp
    src/Dispatcher.cpp
    src/Exception.cpp
    src/Generator.cpp
    src/Scalar.cpp
    src/SizesAndStrides.cpp
    src/Stacktrace.cpp
    src/TensorImpl.cpp
    src/OneDNNContext.cpp
    # src/AutogradMetaInterface.cpp (Moved to TPX)
    src/Tensor.cpp
    
    # Kernels
    src/backend/cpu/ArithmeticKernels.cpp
    src/backend/cpu/ComparisonKernels.cpp
    src/backend/cpu/ConvKernels.cpp
    src/backend/cpu/CopyKernels.cpp
    src/backend/cpu/FactoryKernels.cpp
    src/backend/cpu/LinearAlgebraKernels.cpp
    src/backend/cpu/PointwiseKernels.cpp
    src/backend/cpu/PoolingKernels.cpp
    src/backend/cpu/PadKernels.cpp
    src/backend/cpu/NormalizationKernels.cpp
    src/backend/cpu/RandomKernels.cpp
    src/backend/cpu/ReductionKernels.cpp
    src/backend/cpu/LossKernels.cpp
    src/backend/cpu/ViewKernels.cpp
    
    # Generated
    ${GEN_CPP}
)

if(USE_CUDA)
    list(APPEND P10_SOURCES
        src/backend/cuda/CUDAAllocator.cpp
        src/backend/cuda/CUDAContext.cpp
        src/backend/cuda/ArithmeticKernels.cu
        src/backend/cuda/CopyKernels.cu
        src/backend/cuda/FactoryKernels.cu
        src/backend/cuda/EmbeddingKernels.cu
        src/backend/cuda/PointwiseKernels.cu
        src/backend/cuda/LinearAlgebraKernels.cu
        src/backend/cuda/ActivationKernels.cu
        src/backend/cuda/LossKernels.cu
        src/backend/cuda/ConvKernels.cu
        src/backend/cuda/PoolingKernels.cu
        src/backend/cuda/RandomKernels.cu
        src/backend/cuda/ReductionKernels.cu
        src/backend/cuda/ViewKernels.cu
    )
endif()

set_source_files_properties(${GEN_CPP} PROPERTIES GENERATED TRUE)

add_library(p10 ${P10_SOURCES})
# Enable UNITY_BUILD for faster compilation
set_target_properties(p10 PROPERTIES UNITY_BUILD OFF)
# Exclude generated files from UNITY_BUILD as they might cause conflicts or be too large
set_source_files_properties(${GEN_CPP} PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

add_dependencies(p10 generate_code)

target_include_directories(p10 PUBLIC  
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${GEN_BASE_DIR}
)

target_compile_definitions(p10 PRIVATE P10_EXPORTS)
target_compile_definitions(p10 PRIVATE ${BLAS_COMPILE_DEFS})

if(WIN32)
    target_link_libraries(p10 PRIVATE Dbghelp)
    if(MSVC)
        # Prevent linker from discarding unused object files (like kernels with static registration)
        target_link_options(p10 PRIVATE "/OPT:NOREF")
    endif()
endif()

if(USE_CUDA)
    target_link_libraries(p10 PRIVATE CUDA::cudart CUDA::cublas CUDA::curand)
    target_compile_definitions(p10 PRIVATE USE_CUDA)
    
    if(USE_CUDNN)
         target_link_libraries(p10 PRIVATE ${CUDNN_LIBRARY})
         target_include_directories(p10 PRIVATE ${CUDNN_INCLUDE_DIR})
         target_compile_definitions(p10 PRIVATE USE_CUDNN)
    endif()
endif()

if(COMPUTE_LIBS)
    target_link_libraries(p10 PUBLIC ${COMPUTE_LIBS})
endif()

# Installation logic for p10
set(TP_PACKAGE_DIR "${CMAKE_SOURCE_DIR}/tensorplay")
set(TP_LIB_DIR "${TP_PACKAGE_DIR}/lib")

set_target_properties(p10 PROPERTIES
    OUTPUT_NAME "p10"
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

install(TARGETS p10
    RUNTIME DESTINATION tensorplay/lib
    LIBRARY DESTINATION tensorplay/lib
    ARCHIVE DESTINATION tensorplay/lib
)
