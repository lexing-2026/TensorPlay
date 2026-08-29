cmake_minimum_required(VERSION 3.18)

project(p10 LANGUAGES CXX)

set(P10_SOURCES
    src/Allocator.cpp
    src/Context.cpp
    src/Device.cpp
    src/Dispatcher.cpp
    src/DispatchStub.cpp
    src/Exception.cpp
    src/FileCheck.cpp
    src/Generator.cpp
    src/GradMode.cpp
    src/InferenceMode.cpp
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
    src/RegisterComposites.cpp
    
    # Kernels
    src/backend/cpu/AmpKernels.cpp
    src/backend/cpu/ArithmeticKernels.cpp
    src/backend/cpu/ComparisonKernels.cpp
    src/backend/cpu/ConvKernels.cpp
    src/backend/cpu/CopyKernels.cpp
    src/backend/cpu/Cross.cpp
    src/backend/cpu/SparseKernels.cpp
    src/backend/cpu/QuantKernels.cpp
    src/backend/cpu/ForwardKernels.cpp
    src/backend/cpu/FactoryKernels.cpp
    src/backend/cpu/LinearAlgebraKernels.cpp
    src/backend/cpu/LinalgKernels.cpp
    src/backend/cpu/Lapack.cpp
    src/backend/cpu/PointwiseKernels.cpp
    src/backend/cpu/PoolingKernels.cpp
    src/backend/cpu/GridSamplerKernels.cpp
    src/backend/cpu/FractionalMaxPoolKernels.cpp
    src/backend/cpu/MaxUnpoolKernels.cpp
    src/backend/cpu/MarginLossKernels.cpp
    src/backend/cpu/UpsampleKernels.cpp
    src/backend/cpu/MiscKernels.cpp
    # TensorShape.cpp, LinearAlgebra.cpp, TensorFactories.cpp, Itertools.cpp.
    src/backend/cpu/UnaryOps.cpp
    src/backend/cpu/Pooling.cpp
    src/backend/cpu/MaxPooling.cpp
    src/backend/cpu/TensorShape.cpp
    src/backend/cpu/LinearAlgebra.cpp
    src/backend/cpu/TensorFactories.cpp
    src/backend/cpu/Itertools.cpp
    src/backend/cpu/PadKernels.cpp
    src/backend/cpu/NormalizationKernels.cpp
    src/backend/cpu/RandomKernels.cpp
    src/backend/cpu/ReductionKernels.cpp
    src/backend/cpu/LossKernels.cpp
    src/backend/cpu/ViewKernels.cpp
    src/backend/cpu/ShapeAlignKernels.cpp
    src/backend/cpu/GradientKernels.cpp
    src/backend/cpu/ForeachKernels.cpp
    src/backend/cpu/OptimizerKernels.cpp
    src/backend/cpu/SamplingKernels.cpp
    src/backend/cpu/TopKKernels.cpp
    src/backend/cpu/TransformersKernels.cpp
    src/backend/cpu/IndexingKernels.cpp
    src/backend/cpu/TierOpsKernels.cpp
    src/backend/cpu/Tier5OpsKernels.cpp
    src/backend/cpu/SpecialKernels.cpp
    src/backend/cpu/SpectralKernels.cpp
    src/backend/cpu/Spectral2DKernels.cpp
    src/backend/cpu/InplaceCompletionKernels.cpp

    # Backend-neutral composite kernels, one file per operator category
    # (CompositeCommon.h holds the shared helpers).
    # Registered under the Composite dispatch key; see the files themselves.
    src/backend/composite/Activation.cpp
    src/backend/composite/ChanelShuffle.cpp
    src/backend/composite/CompositeViewCopyKernels.cpp
    src/backend/composite/ConvolutionTBC.cpp
    src/backend/composite/Distance.cpp
    src/backend/composite/Dropout.cpp
    src/backend/composite/Histogram.cpp
    src/backend/composite/Integration.cpp
    src/backend/composite/Itertools.cpp
    src/backend/composite/Linear.cpp
    src/backend/composite/LinearAlgebra.cpp
    src/backend/composite/Pointwise.cpp
    src/backend/composite/RNN.cpp
    src/backend/composite/TensorAdvancedIndexing.cpp
    src/backend/composite/TensorCompare.cpp
    src/backend/composite/TensorFactories.cpp
    src/backend/composite/TensorProperties.cpp
    src/backend/composite/TensorShape.cpp
    src/backend/composite/TypeProperties.cpp
    src/backend/composite/UnaryOps.cpp
    src/backend/composite/Unique.cpp

    src/Einsum.cpp
    src/Profiler.cpp
    src/ProfilerGpu.cpp
    src/ProfilerCupti.cpp
    src/ProfilerNvtx.cpp
    src/ProfilerItt.cpp
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
        src/backend/cuda/CUDAGraphKernels.cu
        src/backend/cuda/CUDARuntimeKernels.cu
        src/backend/cuda/ArithmeticKernels.cu
        src/backend/cuda/CopyKernels.cu
        src/backend/cuda/Cross.cu
        src/backend/cuda/SparseKernels.cu
        src/backend/cuda/QuantKernels.cu
        src/backend/cuda/ForwardKernels.cu
        src/backend/cuda/FactoryKernels.cu
        src/backend/cuda/EmbeddingKernels.cu
        src/backend/cuda/PointwiseKernels.cu
        src/backend/cuda/UnaryOps.cu
        src/backend/cuda/Pooling.cu
        src/backend/cuda/MaxPooling.cu
        src/backend/cuda/AveragePool3d.cu
        src/backend/cuda/AdaptiveAveragePooling3d.cu
        src/backend/cuda/TensorShape.cu
        src/backend/cuda/LinearAlgebra.cu
        src/backend/cuda/TensorFactories.cu
        src/backend/cuda/Itertools.cu
        src/backend/cuda/MiscKernels.cu
        src/backend/cuda/UpsampleKernels.cu
        src/backend/cuda/LinearAlgebraKernels.cu
        src/backend/cuda/LinalgKernels.cu
        src/backend/cuda/LinalgSVD.cu
        src/backend/cuda/LinalgLDL.cu
        src/backend/cuda/CudaBlasGemm.cpp
        src/backend/cuda/ActivationKernels.cu
        src/backend/cuda/NormalizationKernels.cu
        src/backend/cuda/LossKernels.cu
        src/backend/cuda/LossCTC.cu
        src/backend/cuda/ConvKernels.cu
        src/backend/cuda/PadKernels.cu
        src/backend/cuda/PoolingKernels.cu
        src/backend/cuda/GridSamplerKernels.cu
        src/backend/cuda/FractionalMaxPoolKernels.cu
        src/backend/cuda/MaxUnpoolKernels.cu
        src/backend/cuda/MarginLossKernels.cu
        src/backend/cuda/RandomKernels.cu
        src/backend/cuda/ReductionKernels.cu
        src/backend/cuda/ViewKernels.cu
        src/backend/cuda/ShapeAlignKernels.cu
        src/backend/cuda/ForeachFallbackKernels.cu
        src/backend/cuda/ForeachOutKernels.cu
        src/backend/cuda/ForeachMTAKernels.cu
        src/backend/cuda/OptimizerKernels.cu
        src/backend/cuda/AmpKernels.cu
        src/backend/cuda/SamplingKernels.cu
        src/backend/cuda/TopKKernels.cu
        src/backend/cuda/AttentionKernels.cu
        src/backend/cuda/IndexingKernels.cu
        src/backend/cuda/TierOpsKernels.cu
        src/backend/cuda/TierOpsReduceKernels.cu
        src/backend/cuda/RNNKernels.cu
        src/backend/cuda/Tier5OpsKernels.cu
        src/backend/cuda/Tier5LossesKernels.cu
        src/backend/cuda/SpecialUnaryKernels.cu
        src/backend/cuda/SpecialBinaryKernels.cu
        src/backend/cuda/IncompleteGammaKernels.cu
        src/backend/cuda/SpectralKernels.cu
        src/backend/cuda/Spectral2DKernels.cu
        src/backend/cuda/InplaceCompletionKernels.cu
    )
endif()

set_source_files_properties(${GEN_CPP} PROPERTIES GENERATED TRUE)

# ------------------------------------------------------------------
# CPU kernel multi-version compilation (DEFAULT + AVX2)
# ------------------------------------------------------------------
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
    src/backend/cpu/ActivationUnaryKernels.cpp
    src/backend/cpu/StaxPointwiseKernels.cpp
    src/backend/cpu/ComplexKernels.cpp
)

if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i386|i686")
    # (CXX_AVX512_FOUND): only build the tier when the compiler accepts the
    # ISA flags defined by the capability tables.
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma"
        TP_CXX_AVX512_FOUND)
    foreach(KERNEL_SRC ${TP_CPU_KERNEL_SRCS})
        get_filename_component(KERNEL_NAME ${KERNEL_SRC} NAME_WE)
        add_library(p10_kernel_default_${KERNEL_NAME} OBJECT ${KERNEL_SRC})
        set_target_properties(p10_kernel_default_${KERNEL_NAME} PROPERTIES
            POSITION_INDEPENDENT_CODE ON)
        # (add_compile_definitions), so every tier sees the same DispatchStub
        # layout; only CPU_CAPABILITY differs per copy.
        target_compile_definitions(p10_kernel_default_${KERNEL_NAME} PRIVATE
            CPU_CAPABILITY_DEFAULT CPU_CAPABILITY=DEFAULT
            HAVE_AVX2_CPU_DEFINITION)
        if(TP_CXX_AVX512_FOUND)
            target_compile_definitions(p10_kernel_default_${KERNEL_NAME} PRIVATE
                HAVE_AVX512_CPU_DEFINITION)
        endif()
        target_include_directories(p10_kernel_default_${KERNEL_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include ${GEN_BASE_DIR})
        target_compile_definitions(p10_kernel_default_${KERNEL_NAME} PRIVATE
            ${BLAS_COMPILE_DEFS})
        # ComplexKernels owns the DispatchStub definitions.  Only its DEFAULT
        # copy emits those definitions; AVX2/AVX512 copies contribute the
        # capability-specific implementations and registrations.
        if(KERNEL_NAME STREQUAL "ComplexKernels")
            target_compile_definitions(p10_kernel_default_${KERNEL_NAME} PRIVATE
                TP_COMPLEX_KERNELS_DEFINE_DISPATCH)
        endif()
        add_dependencies(p10_kernel_default_${KERNEL_NAME} generate_code)

        add_library(p10_kernel_avx2_${KERNEL_NAME} OBJECT ${KERNEL_SRC})
        set_target_properties(p10_kernel_avx2_${KERNEL_NAME} PROPERTIES
            POSITION_INDEPENDENT_CODE ON)
        target_compile_definitions(p10_kernel_avx2_${KERNEL_NAME} PRIVATE
            CPU_CAPABILITY_AVX2 CPU_CAPABILITY=AVX2 HAVE_AVX2_CPU_DEFINITION)
        if(TP_CXX_AVX512_FOUND)
            target_compile_definitions(p10_kernel_avx2_${KERNEL_NAME} PRIVATE
                HAVE_AVX512_CPU_DEFINITION)
        endif()
        target_compile_options(p10_kernel_avx2_${KERNEL_NAME} PRIVATE -mavx2 -mfma -mf16c)
        if(KERNEL_NAME STREQUAL "ComplexKernels")
            target_compile_definitions(p10_kernel_avx2_${KERNEL_NAME} PRIVATE
                TP_COMPLEX_KERNELS_NO_DISPATCH_DEFINITION)
        endif()
        target_include_directories(p10_kernel_avx2_${KERNEL_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include ${GEN_BASE_DIR})
        target_compile_definitions(p10_kernel_avx2_${KERNEL_NAME} PRIVATE
            ${BLAS_COMPILE_DEFS})
        add_dependencies(p10_kernel_avx2_${KERNEL_NAME} generate_code)

        list(APPEND TP_CPU_KERNEL_DEFAULT_OBJS $<TARGET_OBJECTS:p10_kernel_default_${KERNEL_NAME}>)
        list(APPEND TP_CPU_KERNEL_AVX2_OBJS $<TARGET_OBJECTS:p10_kernel_avx2_${KERNEL_NAME}>)

        if(TP_CXX_AVX512_FOUND)
            add_library(p10_kernel_avx512_${KERNEL_NAME} OBJECT ${KERNEL_SRC})
            set_target_properties(p10_kernel_avx512_${KERNEL_NAME} PROPERTIES
                POSITION_INDEPENDENT_CODE ON)
            target_compile_definitions(p10_kernel_avx512_${KERNEL_NAME} PRIVATE
                CPU_CAPABILITY_AVX512 CPU_CAPABILITY=AVX512 HAVE_AVX512_CPU_DEFINITION HAVE_AVX2_CPU_DEFINITION)
            # Same flag set as Codegen.cmake's AVX512 capability entry.
            target_compile_options(p10_kernel_avx512_${KERNEL_NAME} PRIVATE
                -mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma)
            if(KERNEL_NAME STREQUAL "ComplexKernels")
                target_compile_definitions(p10_kernel_avx512_${KERNEL_NAME} PRIVATE
                    TP_COMPLEX_KERNELS_NO_DISPATCH_DEFINITION)
            endif()
            target_include_directories(p10_kernel_avx512_${KERNEL_NAME} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include ${GEN_BASE_DIR})
            target_compile_definitions(p10_kernel_avx512_${KERNEL_NAME} PRIVATE
                ${BLAS_COMPILE_DEFS})
            add_dependencies(p10_kernel_avx512_${KERNEL_NAME} generate_code)
            list(APPEND TP_CPU_KERNEL_AVX512_OBJS $<TARGET_OBJECTS:p10_kernel_avx512_${KERNEL_NAME}>)
        endif()
    endforeach()
    list(APPEND P10_SOURCES ${TP_CPU_KERNEL_DEFAULT_OBJS} ${TP_CPU_KERNEL_AVX2_OBJS})
    if(TP_CXX_AVX512_FOUND)
        list(APPEND P10_SOURCES ${TP_CPU_KERNEL_AVX512_OBJS})
    endif()
endif()

add_library(p10 ${P10_SOURCES})
# Enable UNITY_BUILD for faster compilation
set_target_properties(p10 PROPERTIES UNITY_BUILD OFF)

if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i386|i686")
    # Tell DispatchStub (and every TU) that AVX2 kernel copies exist, so the
    # AVX2 dispatch slots are registered and selected at runtime.
    target_compile_definitions(p10 PUBLIC HAVE_AVX2_CPU_DEFINITION)
    if(TP_CXX_AVX512_FOUND)
        target_compile_definitions(p10 PUBLIC HAVE_AVX512_CPU_DEFINITION)
    endif()
endif()

if(USE_CUDA)
    # The native aligned flash path consumes only the standalone CUTE/CUTLASS
    # headers and flash kernels.  It intentionally does not include or link
    # kernels remain buildable.
    set(TP_NATIVE_FLASH_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/pytorch/third_party/flash-attention/csrc/flash_attn/src")
    set(TP_NATIVE_CUTLASS_INCLUDE
        "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/pytorch/third_party/cutlass/include")
    if(EXISTS "${TP_NATIVE_FLASH_SRC}/flash.h" AND
       EXISTS "${TP_NATIVE_CUTLASS_INCLUDE}/cute/tensor.hpp")
        target_include_directories(p10 PRIVATE
            "${TP_NATIVE_FLASH_SRC}"
            "${TP_NATIVE_CUTLASS_INCLUDE}")
    endif()
    # CUDA kernel alignment: allow __device__ lambdas and constexpr libm calls in .cu kernels
    target_compile_options(p10 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
    # The local toolkit ships 13.0 headers while nvcc is 13.2; cccl's strict
    # CUDART-vs-compiler check otherwise rejects any TU that includes
    # <cub/cub.cuh> (e.g. SparseKernels.cu).  The macro is cccl's documented
    # escape hatch for exactly this newer-compiler/older-headers pairing.
    target_compile_definitions(p10 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:CCCL_DISABLE_CTK_COMPATIBILITY_CHECK>)
endif()

# Exclude generated files from UNITY_BUILD as they might cause conflicts or be too large
set_source_files_properties(${GEN_CPP} PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

add_dependencies(p10 generate_code)

target_include_directories(p10 PUBLIC  
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${GEN_BASE_DIR}
)

# hand-written tpx headers (Autograd.h, Node.h).  Expose tpx/include so those
# generated/native translation units resolve.
target_include_directories(p10 PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../tpx/include>
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
# ProfilerCupti dlopen's libcupti at runtime; libdl is part of libc on
# glibc >= 2.34 but must be linked explicitly elsewhere.
target_link_libraries(p10 PRIVATE ${CMAKE_DL_LIBS})

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
