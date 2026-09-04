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
    src/SymNodeImpl.cpp
    src/SymInt.cpp
    src/SymBool.cpp
    src/SymFloat.cpp
    src/SizesAndStrides.cpp
    src/Stacktrace.cpp
    src/TensorImpl.cpp
    src/TransformDispatch.cpp
    src/BatchingKernels.cpp
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
    src/backend/cpu/Convolution.cpp
    src/backend/cpu/CopyKernels.cpp
    src/backend/cpu/EmbeddingBagKernels.cpp
    src/backend/cpu/Cross.cpp
    src/backend/cpu/SparseKernels.cpp
    src/backend/cpu/SemiStructuredKernels.cpp
    src/backend/cpu/QuantKernels.cpp
    src/backend/cpu/QuantConvPacking.cpp
    src/quantized/QTensorImpl.cpp
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
    src/backend/cpu/BatchNormKernels.cpp
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
    src/backend/cpu/BitwiseKernels.cpp
    src/backend/cpu/Blas.cpp
    src/backend/cpu/BatchLinearAlgebra.cpp
    src/backend/cpu/RNNKernels.cpp
    src/backend/cpu/DistanceKernels.cpp
    src/backend/cpu/ComplexFactories.cpp
    src/backend/cpu/ExtraLossKernels.cpp
    src/backend/cpu/SpecialKernels.cpp
    src/backend/cpu/SpecialAliasKernels.cpp
    src/backend/cpu/MetaViewKernels.cpp
    src/backend/cpu/FactoryToolsKernels.cpp
    src/backend/cpu/ActivationMoreKernels.cpp
    src/backend/cpu/SpectralKernels.cpp
    src/backend/cpu/Spectral2DKernels.cpp
    src/backend/cpu/InplaceCompletionKernels.cpp
    src/backend/cpu/SymScalarKernels.cpp
    # Backend-neutral composite kernels, one file per operator category
    # (CompositeCommon.h holds the shared helpers).
    # Registered under the Composite dispatch key; see the files themselves.
    src/backend/composite/Activation.cpp
    src/backend/composite/ActivationMore.cpp
    src/backend/composite/BackwardHelperComposites.cpp
    src/backend/composite/OutVariantComposites.cpp
    src/backend/composite/InplaceAliasComposites.cpp
    src/backend/composite/VariantOutComposites.cpp
    src/backend/composite/VariantAliasComposites.cpp
    src/backend/composite/VariantHandComposites.cpp
    src/backend/composite/MiscBridgeComposites.cpp
    src/backend/composite/DeadEndBridgeComposites.cpp
    src/backend/composite/DunderBinaryComposite.cpp
    src/backend/composite/FactoryToolsComposite.cpp
    src/backend/composite/ChanelShuffle.cpp
    src/backend/composite/BaseMetaComposites.cpp
    src/backend/composite/CompositeViewCopyKernels.cpp
    src/backend/composite/ConvolutionTBC.cpp
    src/backend/composite/Distance.cpp
    src/backend/composite/Dropout.cpp
    src/backend/composite/FactoryToolsComposite.cpp
    src/backend/composite/Histogram.cpp
    src/backend/composite/Integration.cpp
    src/backend/composite/Itertools.cpp
    src/backend/composite/Linear.cpp
    src/backend/composite/LinearAlgebra.cpp
    src/backend/composite/Pointwise.cpp
    src/backend/composite/Reduction.cpp
    src/backend/composite/RNN.cpp
    src/backend/composite/TensorAdvancedIndexing.cpp
    src/backend/composite/TensorCompare.cpp
    src/backend/composite/TensorFactories.cpp
    src/backend/composite/TensorProperties.cpp
    src/backend/composite/TensorShape.cpp
    src/backend/composite/TypeProperties.cpp
    src/backend/composite/UnaryOps.cpp
    src/backend/composite/Unique.cpp
    # Unconditionally compiled Vulkan dispatch surface: exposes the
    # availability probe and registry so callers link regardless of whether
    # the backend itself is in the build.
    src/vulkan/Context.cpp
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
if(USE_CUDA OR USE_ROCM)
    set(TP_CUDA_BACKEND_SOURCES
        src/backend/cuda/CUDAAllocator.cpp
        src/backend/cuda/CUDAContext.cpp
        src/backend/cuda/CUDAGenerator.cpp
        src/backend/cuda/CUDARuntime.cpp
        src/backend/cuda/CUDAGraph.cpp
        src/backend/cuda/CUDAGraphKernels.cu
        src/backend/cuda/CUDARuntimeKernels.cu
        src/backend/cuda/ArithmeticKernels.cu
        src/backend/cuda/CopyKernels.cu
        src/backend/cuda/Cross.cu
        src/backend/cuda/SparseKernels.cu
        src/backend/cuda/SemiStructuredKernels.cu
        src/backend/cuda/QuantKernels.cu
        src/backend/cuda/ForwardKernels.cu
        src/backend/cuda/FactoryKernels.cu
        src/backend/cuda/EmbeddingKernels.cu
        src/backend/cuda/EmbeddingBagKernels.cu
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
        src/backend/cuda/LossFillKernels.cu
        src/backend/cuda/LossCTC.cu
        src/backend/cuda/ConvKernels.cu
        src/backend/cuda/Convolution.cu
        src/backend/cuda/PadKernels.cu
        src/backend/cuda/PoolingKernels.cu
        src/backend/cuda/GridSamplerKernels.cu
        src/backend/cuda/FractionalMaxPoolKernels.cu
        src/backend/cuda/MaxUnpoolKernels.cu
        src/backend/cuda/MarginLossKernels.cu
        src/backend/cuda/RandomKernels.cu
        src/backend/cuda/ReductionKernels.cu
        src/backend/cuda/DistanceKernels.cu
        src/backend/cuda/ViewKernels.cu
        src/backend/cuda/ShapeAlignKernels.cu
        src/backend/cuda/ForeachFallbackKernels.cu
        src/backend/cuda/ForeachOutKernels.cu
        src/backend/cuda/ForeachMTAKernels.cu
        src/backend/cuda/OptimizerKernels.cu
        src/backend/cuda/AmpKernels.cu
        src/backend/cuda/SamplingKernels.cu
        src/backend/cuda/TensorTopK.cu
        src/backend/cuda/SpecialAliasKernels.cu
        src/backend/cuda/AttentionKernels.cu
        src/backend/cuda/IndexingKernels.cu
        src/backend/cuda/InteropAliasKernels.cu
        src/backend/cuda/TierOpsKernels.cu
        src/backend/cuda/BitwiseKernels.cu
        src/backend/cuda/TierOpsReduceKernels.cu
        src/backend/cuda/RNNKernels.cu
        src/backend/cuda/Blas.cu
        src/backend/cuda/BatchLinearAlgebra.cu
        src/backend/cuda/ComplexFactories.cu
        src/backend/cuda/LossKernels.cu
        src/backend/cuda/SpecialUnaryKernels.cu
        src/backend/cuda/SpecialBinaryKernels.cu
        src/backend/cuda/IncompleteGammaKernels.cu
        src/backend/cuda/SpectralKernels.cu
        src/backend/cuda/Spectral2DKernels.cu
        src/backend/cuda/InplaceCompletionKernels.cu
    )
    if(USE_ROCM)
        # Compile the hipify-staged translations instead of the originals.
        tp_map_sources(TP_CUDA_BACKEND_SOURCES ${TP_CUDA_BACKEND_SOURCES})
        list(APPEND P10_SOURCES ${TP_CUDA_BACKEND_SOURCES})
        list(APPEND P10_SOURCES
            "${TP_HIP_STAGING}/p10/src/backend/hip/NCCLContext.cpp")
        # The staged tree also carries translated copies of every shared
        # source; swap the ones whose original TU touches GPU APIs.
        # ProfilerCupti is dropped; the stub keeps its symbols as no-ops
        # (the GPU collector is not available in HIP builds).
        set(TP_P10_SHARED_GPU_SOURCES
            src/Allocator.cpp
            src/ProfilerGpu.cpp
            src/backend/cpu/CopyKernels.cpp
        )
        tp_map_sources(TP_P10_SHARED_GPU_SOURCES ${TP_P10_SHARED_GPU_SOURCES})
        list(REMOVE_ITEM P10_SOURCES
            src/Allocator.cpp
            src/ProfilerGpu.cpp
            src/backend/cpu/CopyKernels.cpp
            src/ProfilerCupti.cpp)
        list(APPEND P10_SOURCES ${TP_P10_SHARED_GPU_SOURCES}
            src/ProfilerCuptiStub.cpp)
    else()
        list(APPEND TP_CUDA_BACKEND_SOURCES src/backend/cuda/NCCLContext.cpp)
        list(APPEND P10_SOURCES ${TP_CUDA_BACKEND_SOURCES})
    endif()
endif()
if(USE_VULKAN)
    # Vulkan backend: api layer, ops and the generated SPIR-V registry.
    # The shader registry is produced at build time from the GLSL sources by
    # tools/gen_vulkan_spv.py (GLSL -> SPIR-V -> embedded C++ pipeline).
    set(TP_VULKAN_GEN_DIR "${GEN_BASE_DIR}/vulkan")
    set(TP_VULKAN_GEN_CPP "${TP_VULKAN_GEN_DIR}/spv.cpp")
    set(TP_VULKAN_GEN_H "${TP_VULKAN_GEN_DIR}/spv.h")
    find_program(TP_GLSLANG_VALIDATOR glslangValidator
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../tools/bin"
        NO_DEFAULT_PATH)
    if(NOT TP_GLSLANG_VALIDATOR)
        find_program(TP_GLSLANG_VALIDATOR glslangValidator)
    endif()
    if(NOT TP_GLSLANG_VALIDATOR)
        message(FATAL_ERROR "USE_VULKAN requires glslangValidator (expected under tools/bin).")
    endif()
    file(GLOB_RECURSE TP_VULKAN_GLSL CONFIG_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/vulkan/glsl/*.glsl")
    list(APPEND TP_VULKAN_GLSL
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/vulkan/glsl/shader_params.yaml")
    add_custom_command(
        OUTPUT ${TP_VULKAN_GEN_CPP} ${TP_VULKAN_GEN_H}
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/..
                ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../tools/gen_vulkan_spv.py
                --glsl-path ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/vulkan/glsl
                --output-path ${TP_VULKAN_GEN_DIR}
                --glslang-path ${TP_GLSLANG_VALIDATOR}
        DEPENDS ${TP_VULKAN_GLSL} ${CMAKE_CURRENT_SOURCE_DIR}/../tools/gen_vulkan_spv.py
        COMMENT "Generating Vulkan SPIR-V shader registry"
    )
    set_source_files_properties(${TP_VULKAN_GEN_CPP} PROPERTIES GENERATED TRUE)
    add_custom_target(vulkan_spv_gen DEPENDS ${TP_VULKAN_GEN_CPP} ${TP_VULKAN_GEN_H})
    list(APPEND P10_SOURCES
        src/backend/vulkan/api/Adapter.cpp
        src/backend/vulkan/api/Allocator.cpp
        src/backend/vulkan/api/Command.cpp
        src/backend/vulkan/api/Context.cpp
        src/backend/vulkan/api/Descriptor.cpp
        src/backend/vulkan/api/Exception.cpp
        src/backend/vulkan/api/Pipeline.cpp
        src/backend/vulkan/api/QueryPool.cpp
        src/backend/vulkan/api/Resource.cpp
        src/backend/vulkan/api/Runtime.cpp
        src/backend/vulkan/api/Shader.cpp
        src/backend/vulkan/api/ShaderRegistry.cpp
        src/backend/vulkan/api/Tensor.cpp
        src/backend/vulkan/VulkanRuntime.cpp
        src/backend/vulkan/impl/Common.cpp
        src/backend/vulkan/impl/Packing.cpp
        src/backend/vulkan/ops/Activation.cpp
        src/backend/vulkan/ops/Allocator.cpp
        src/backend/vulkan/ops/BinaryOp.cpp
        src/backend/vulkan/ops/Clamp.cpp
        src/backend/vulkan/ops/Clone.cpp
        src/backend/vulkan/ops/Conv.cpp
        src/backend/vulkan/ops/Copy.cpp
        src/backend/vulkan/ops/Cumsum.cpp
        src/backend/vulkan/ops/Factory.cpp
        src/backend/vulkan/ops/Glu.cpp
        src/backend/vulkan/ops/MaskedFill.cpp
        src/backend/vulkan/ops/Matmul.cpp
        src/backend/vulkan/ops/Misc.cpp
        src/backend/vulkan/ops/Normalization.cpp
        src/backend/vulkan/ops/ParamCache.cpp
        src/backend/vulkan/ops/Comparison.cpp
        src/backend/vulkan/ops/Indexing.cpp
        src/backend/vulkan/ops/Upsample.cpp
        src/backend/vulkan/ops/Where.cpp
        src/backend/vulkan/ops/Pooling.cpp
        src/backend/vulkan/ops/Quant.cpp
        src/backend/vulkan/ops/Random.cpp
        src/backend/vulkan/ops/Reduction.cpp
        src/backend/vulkan/ops/Shape.cpp
        src/backend/vulkan/ops/Softmax.cpp
        src/backend/vulkan/ops/UnaryOp.cpp
        src/backend/vulkan/ops/Utils.cpp
        src/backend/vulkan/ops/View.cpp
        src/backend/vulkan/ops/Zero.cpp
        ${TP_VULKAN_GEN_CPP}
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
    src/backend/cpu/EmbeddingBagKernelsImpl.cpp
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
else()
    # Targets without the x86 ISA tiers still need one copy of each kernel, or
    # the DispatchStub DEFAULT slots stay empty and the ops report a missing
    # kernel at runtime.  Same defines as the DEFAULT tier above, minus the
    # AVX capability flags.
    foreach(KERNEL_SRC ${TP_CPU_KERNEL_SRCS})
        get_filename_component(KERNEL_NAME ${KERNEL_SRC} NAME_WE)
        add_library(p10_kernel_default_${KERNEL_NAME} OBJECT ${KERNEL_SRC})
        set_target_properties(p10_kernel_default_${KERNEL_NAME} PROPERTIES
            POSITION_INDEPENDENT_CODE ON)
        target_compile_definitions(p10_kernel_default_${KERNEL_NAME} PRIVATE
            CPU_CAPABILITY_DEFAULT CPU_CAPABILITY=DEFAULT)
        if(KERNEL_NAME STREQUAL "ComplexKernels")
            target_compile_definitions(p10_kernel_default_${KERNEL_NAME} PRIVATE
                TP_COMPLEX_KERNELS_DEFINE_DISPATCH)
        endif()
        target_include_directories(p10_kernel_default_${KERNEL_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include ${GEN_BASE_DIR})
        target_compile_definitions(p10_kernel_default_${KERNEL_NAME} PRIVATE
            ${BLAS_COMPILE_DEFS})
        add_dependencies(p10_kernel_default_${KERNEL_NAME} generate_code)
        list(APPEND P10_SOURCES $<TARGET_OBJECTS:p10_kernel_default_${KERNEL_NAME}>)
    endforeach()
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
    # them into the library; ordinary kernels remain buildable.
    file(GLOB TP_VENDOR_ROOTS LIST_DIRECTORIES true
        "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/*")
    foreach(TP_VENDOR_ROOT ${TP_VENDOR_ROOTS})
        if(EXISTS "${TP_VENDOR_ROOT}/third_party/flash-attention/csrc/flash_attn/src/flash.h" AND
           EXISTS "${TP_VENDOR_ROOT}/third_party/cutlass/include/cute/tensor.hpp")
            set(TP_NATIVE_FLASH_SRC
                "${TP_VENDOR_ROOT}/third_party/flash-attention/csrc/flash_attn/src")
            set(TP_NATIVE_CUTLASS_INCLUDE
                "${TP_VENDOR_ROOT}/third_party/cutlass/include")
            break()
        endif()
    endforeach()
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
if(USE_ROCM)
    # Staged headers (translated CUDA-spelling names) must win over the
    # originals for every TU in this target, hence BEFORE on the prefix.
    target_include_directories(p10 BEFORE PUBLIC
        $<BUILD_INTERFACE:${TP_HIP_STAGING}/p10/include>
    )
    target_include_directories(p10 BEFORE PRIVATE ${TP_HIP_STAGING}/src)
    target_include_directories(p10 SYSTEM PRIVATE ${TP_HIP_INCLUDE_DIRS})
endif()
# Backend-internal cross-directory includes (e.g. backend/vulkan/... from
# cpu kernels) resolve against src/.
target_include_directories(p10 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
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
if(USE_VULKAN)
    target_include_directories(p10 SYSTEM PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/vulkan-headers"
        "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/vulkan-memory-allocator"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/vulkan/api"
        "${TP_VULKAN_GEN_DIR}")
    target_link_libraries(p10 PRIVATE ${VULKAN_LIBRARY})
    target_compile_definitions(p10 PRIVATE USE_VULKAN)
    add_dependencies(p10 vulkan_spv_gen)
endif()
if(USE_CUDA)
    target_link_libraries(p10 PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cufft CUDA::cusolver)
    target_compile_definitions(p10 PRIVATE USE_CUDA)
    # CUDA 13 moved CUB/thrust under <toolkit>/include/cccl; SparseKernels
    # includes <cub/cub.cuh>, which resolves only with that dir on the path.
    if(EXISTS "${CUDAToolkit_INCLUDE_DIRS}/cccl/cub")
        target_include_directories(p10 PRIVATE "${CUDAToolkit_INCLUDE_DIRS}/cccl")
    endif()
    # Enable NCCL for CUDA builds when it is selected.
    option(USE_NCCL "Use NCCL for tensorplay.distributed" ON)
    if(USE_NCCL)
        option(USE_SYSTEM_NCCL "Use a system-wide NCCL" OFF)
        include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/External/nccl.cmake)
        if(TARGET tp::nccl)
            target_link_libraries(p10 PRIVATE tp::nccl)
            target_compile_definitions(p10 PRIVATE USE_NCCL)
            message(STATUS "Found NCCL: ${NCCL_LIBRARIES}")
        else()
            message(WARNING
                "Not using NCCL (tensorplay.distributed will be unavailable). "
                "Set NCCL_ROOT to an NCCL install prefix or check out "
                "third_party/nccl to enable it.")
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
if(USE_ROCM)
    target_include_directories(p10 SYSTEM PRIVATE ${RCCL_INCLUDE_DIRS})
    target_link_libraries(p10 PRIVATE tp::rccl)
    target_link_libraries(p10 PRIVATE ${TP_HIP_LIBRARIES})
    # The staged tree keeps the CUDA-spelling guard names, so the target
    # carries both defines: existing #ifdef USE_CUDA blocks activate and
    # route to the HIP-backed implementations.
    target_compile_definitions(p10 PRIVATE USE_CUDA USE_ROCM USE_RCCL)
    if(USE_CUDNN)
        # The staged compatibility header re-creates the descriptor-style
        # DNN surface on top of MIOpen; this define activates the guarded
        # code paths (the library itself is already on the link line via
        # TP_HIP_LIBRARIES).
        target_compile_definitions(p10 PRIVATE USE_CUDNN)
    endif()
    target_compile_options(p10 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX,HIP>:${TP_HIP_CXX_FLAGS}>)
endif()
if(COMPUTE_LIBS)
    target_link_libraries(p10 PUBLIC ${COMPUTE_LIBS})
endif()
# Vector math (runtime-dispatched entry points; see cpu/vec/SleefShims.h).
if(TARGET sleef)
    target_link_libraries(p10 PRIVATE sleef)
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
if(NOT DEFINED TP_PACKAGE_DIR)
    set(TP_PACKAGE_DIR "${CMAKE_SOURCE_DIR}/tensorplay")
endif()
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
