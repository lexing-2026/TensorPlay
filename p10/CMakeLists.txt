cmake_minimum_required(VERSION 3.18)

project(p10 LANGUAGES CXX)

set(P10_SOURCES
    src/Allocator.cpp
    src/Context.cpp
    src/Device.cpp
    src/Dispatcher.cpp
    src/DispatchStub.cpp
    src/Exception.cpp
    src/Generator.cpp
    src/GradMode.cpp
    src/Parallel.cpp
    src/Scalar.cpp
    src/SizesAndStrides.cpp
    src/Stacktrace.cpp
    src/TensorImpl.cpp
    src/OneDNNContext.cpp
    src/StaxPointwise.cpp
    src/autocast_mode.cpp
    # src/AutogradMetaInterface.cpp (Moved to TPX)
    src/Tensor.cpp
    
    # Kernels
    src/backend/cpu/AmpKernels.cpp
    src/backend/cpu/ArithmeticKernels.cpp
    src/backend/cpu/ComparisonKernels.cpp
    src/backend/cpu/ConvKernels.cpp
    src/backend/cpu/CopyKernels.cpp
    src/backend/cpu/SparseKernels.cpp
    src/backend/cpu/FactoryKernels.cpp
    src/backend/cpu/LinearAlgebraKernels.cpp
    src/backend/cpu/LinalgKernels.cpp
    src/backend/cpu/Lapack.cpp
    src/backend/cpu/PointwiseKernels.cpp
    src/backend/cpu/PoolingKernels.cpp
    src/backend/cpu/UpsampleKernels.cpp
    src/backend/cpu/MiscKernels.cpp
    src/backend/cpu/PadKernels.cpp
    src/backend/cpu/NormalizationKernels.cpp
    src/backend/cpu/RandomKernels.cpp
    src/backend/cpu/ReductionKernels.cpp
    src/backend/cpu/LossKernels.cpp
    src/backend/cpu/ViewKernels.cpp
    src/backend/cpu/ForeachKernels.cpp
    src/backend/cpu/OptimizerKernels.cpp
    src/backend/cpu/LlmReferenceKernels.cpp
    src/backend/cpu/IndexingKernels.cpp
    src/backend/cpu/TierOpsKernels.cpp
    src/backend/cpu/Tier5OpsKernels.cpp
    src/backend/cpu/SpecialKernels.cpp
    src/backend/cpu/SpectralKernels.cpp
    src/Einsum.cpp
    src/TensorIterator.cpp
    src/TensorIteratorReduce.cpp
    
    # Generated
    ${GEN_CPP}
)

if(USE_CUDA)
    list(APPEND P10_SOURCES
        src/backend/cuda/CUDAAllocator.cpp
        src/backend/cuda/CUDAContext.cpp
        src/backend/cuda/CUDAGenerator.cpp
        src/backend/cuda/NCCLContext.cpp
        src/backend/cuda/CUDARuntime.cpp
        src/backend/cuda/CUDAGraph.cpp
        src/backend/cuda/CUDARuntimeKernels.cu
        src/backend/cuda/ArithmeticKernels.cu
        src/backend/cuda/CopyKernels.cu
        src/backend/cuda/SparseKernels.cu
        src/backend/cuda/FactoryKernels.cu
        src/backend/cuda/EmbeddingKernels.cu
        src/backend/cuda/PointwiseKernels.cu
        src/backend/cuda/MiscKernels.cu
        src/backend/cuda/UpsampleKernels.cu
        src/backend/cuda/LinearAlgebraKernels.cu
        src/backend/cuda/LinalgKernels.cu
        src/backend/cuda/CudaBlasGemm.cpp
        src/backend/cuda/ActivationKernels.cu
        src/backend/cuda/NormalizationKernels.cu
        src/backend/cuda/LossKernels.cu
        src/backend/cuda/ConvKernels.cu
        src/backend/cuda/PadKernels.cu
        src/backend/cuda/PoolingKernels.cu
        src/backend/cuda/RandomKernels.cu
        src/backend/cuda/ReductionKernels.cu
        src/backend/cuda/ViewKernels.cu
        src/backend/cuda/OptimizerKernels.cu
        src/backend/cuda/AmpKernels.cu
        src/backend/cuda/SamplingKernels.cu
        src/backend/cuda/AttentionKernels.cu
        src/backend/cuda/IndexingKernels.cu
        src/backend/cuda/TierOpsKernels.cu
        src/backend/cuda/TierOpsReduceKernels.cu
        src/backend/cuda/RNNKernels.cu
        src/backend/cuda/Tier5OpsKernels.cu
        src/backend/cuda/Tier5LossesKernels.cu
        src/backend/cuda/SpecialKernels.cu
        src/backend/cuda/SpectralKernels.cu
    )
endif()

set_source_files_properties(${GEN_CPP} PROPERTIES GENERATED TRUE)

# ------------------------------------------------------------------
# CPU kernel multi-version compilation (DEFAULT + AVX2)
# ------------------------------------------------------------------
# Kernels in TP_CPU_KERNEL_SRCS are compiled twice, mirroring PyTorch's
# native/cpu mechanism: once with no ISA-specific flags
# (CPU_CAPABILITY=DEFAULT) and once with AVX2 enabled
# (CPU_CAPABILITY_AVX2 -DCPU_CAPABILITY=AVX2 -mavx2 -mfma). Both copies are
# linked into p10 under different inline namespaces (see
# Note [CPU_CAPABILITY namespace] in cpu/vec/vec_base.h); DispatchStub
# selects the AVX2 copy at runtime when the CPU supports it.
# NB: every kernel in this list MUST register an AVX2 kernel via
# REGISTER_DISPATCH (its DEFAULT-only registration is not enough when
# HAVE_AVX2_CPU_DEFINITION is defined).
set(TP_CPU_KERNEL_SRCS
    src/backend/cpu/ReductionKernelsImpl.cpp
    src/backend/cpu/StaxPointwiseKernels.cpp
)

if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i386|i686")
    foreach(KERNEL_SRC ${TP_CPU_KERNEL_SRCS})
        get_filename_component(KERNEL_NAME ${KERNEL_SRC} NAME_WE)
        add_library(p10_kernel_default_${KERNEL_NAME} OBJECT ${KERNEL_SRC})
        set_target_properties(p10_kernel_default_${KERNEL_NAME} PROPERTIES
            POSITION_INDEPENDENT_CODE ON)
        target_compile_definitions(p10_kernel_default_${KERNEL_NAME} PRIVATE
            CPU_CAPABILITY_DEFAULT CPU_CAPABILITY=DEFAULT)
        target_include_directories(p10_kernel_default_${KERNEL_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include ${GEN_BASE_DIR})
        target_compile_definitions(p10_kernel_default_${KERNEL_NAME} PRIVATE
            ${BLAS_COMPILE_DEFS})
        add_dependencies(p10_kernel_default_${KERNEL_NAME} generate_code)

        add_library(p10_kernel_avx2_${KERNEL_NAME} OBJECT ${KERNEL_SRC})
        set_target_properties(p10_kernel_avx2_${KERNEL_NAME} PROPERTIES
            POSITION_INDEPENDENT_CODE ON)
        target_compile_definitions(p10_kernel_avx2_${KERNEL_NAME} PRIVATE
            CPU_CAPABILITY_AVX2 CPU_CAPABILITY=AVX2 HAVE_AVX2_CPU_DEFINITION)
        target_compile_options(p10_kernel_avx2_${KERNEL_NAME} PRIVATE -mavx2 -mfma)
        target_include_directories(p10_kernel_avx2_${KERNEL_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include ${GEN_BASE_DIR})
        target_compile_definitions(p10_kernel_avx2_${KERNEL_NAME} PRIVATE
            ${BLAS_COMPILE_DEFS})
        add_dependencies(p10_kernel_avx2_${KERNEL_NAME} generate_code)

        list(APPEND TP_CPU_KERNEL_DEFAULT_OBJS $<TARGET_OBJECTS:p10_kernel_default_${KERNEL_NAME}>)
        list(APPEND TP_CPU_KERNEL_AVX2_OBJS $<TARGET_OBJECTS:p10_kernel_avx2_${KERNEL_NAME}>)
    endforeach()
    list(APPEND P10_SOURCES ${TP_CPU_KERNEL_DEFAULT_OBJS} ${TP_CPU_KERNEL_AVX2_OBJS})
endif()

add_library(p10 ${P10_SOURCES})
# Enable UNITY_BUILD for faster compilation
set_target_properties(p10 PROPERTIES UNITY_BUILD OFF)

if(USE_CUDA)
    # ATen alignment: allow __device__ lambdas and constexpr libm calls in .cu kernels
    target_compile_options(p10 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
    # The local toolkit ships 13.0 headers while nvcc is 13.2; cccl's strict
    # CUDART-vs-compiler check otherwise rejects any TU that includes
    # <cub/cub.cuh> (e.g. SparseKernels.cu).  The macro is cccl's documented
    # escape hatch for exactly this newer-compiler/older-headers pairing.
    target_compile_definitions(p10 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:CCCL_DISABLE_CTK_COMPATIBILITY_CHECK>)
endif()

if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i386|i686")
    # Tell DispatchStub (and every TU) that AVX2 kernel copies exist, so the
    # AVX2 dispatch slots are registered and selected at runtime.
    target_compile_definitions(p10 PUBLIC HAVE_AVX2_CPU_DEFINITION)
endif()
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
    target_link_libraries(p10 PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cufft CUDA::cusolver)
    target_compile_definitions(p10 PRIVATE USE_CUDA)

    # CUDA 13 moved CUB/thrust under <toolkit>/include/cccl; SparseKernels
    # includes <cub/cub.cuh>, which resolves only with that dir on the path.
    if(EXISTS "${CUDAToolkit_INCLUDE_DIRS}/cccl/cub")
        target_include_directories(p10 PRIVATE "${CUDAToolkit_INCLUDE_DIRS}/cccl")
    endif()

    # NCCL, mirroring pytorch's Dependencies.cmake: option + FindNCCL.cmake,
    # positive USE_NCCL define, graceful downgrade when absent.
    option(USE_NCCL "Use NCCL for tensorplay.distributed" ON)
    if(USE_NCCL)
        find_package(NCCL 2.4.8 QUIET)
        if(NCCL_FOUND)
            target_link_libraries(p10 PRIVATE ${NCCL_LIBRARIES})
            target_include_directories(p10 SYSTEM PRIVATE ${NCCL_INCLUDE_DIRS})
            target_compile_definitions(p10 PRIVATE USE_NCCL)
            message(STATUS "Found NCCL: ${NCCL_LIBRARIES} (version ${NCCL_VERSION})")
        else()
            message(WARNING
                "Not using NCCL (tensorplay.distributed will be unavailable). "
                "Set NCCL_ROOT to an NCCL install prefix to enable it.")
        endif()
    endif()

    if(USE_CUDNN)
         target_link_libraries(p10 PRIVATE ${CUDNN_LIBRARY})
         target_include_directories(p10 PRIVATE ${CUDNN_INCLUDE_DIR})
         if(CUDNN_FRONTEND_INCLUDE_DIR)
             target_include_directories(p10 PRIVATE ${CUDNN_FRONTEND_INCLUDE_DIR})
         endif()
         target_compile_definitions(p10 PRIVATE USE_CUDNN)
    endif()
endif()

if(COMPUTE_LIBS)
    target_link_libraries(p10 PUBLIC ${COMPUTE_LIBS})
endif()

find_package(Threads REQUIRED)
target_link_libraries(p10 PUBLIC Threads::Threads)

if(UNIX AND NOT APPLE)
    target_link_libraries(p10 PUBLIC m)
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
