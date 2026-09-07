cmake_minimum_required(VERSION 3.18)
project(TensorPlay LANGUAGES CXX)

include(CMakeDependentOption)

# EnvVarForwarding reads os.environ via Python (see module header), so the
# CMakeLists.txt. Forwarded BUILD_*/USE_*/CMAKE_* env vars become cache
# variables before the USE_CUDA auto-detection below consumes them.
find_package(Python COMPONENTS Interpreter REQUIRED)
include(cmake/EnvVarForwarding.cmake)

# Find modules (cuDNN, MKL, OpenBLAS, MKLDNN, NCCL, ...) live in cmake/Modules
# and must be resolvable by every find_package() below.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Auto-detect CUDA: default to ON only when an nvcc is available so that a
# plain `pip install .` on a CPU-only machine still builds successfully.
if(NOT DEFINED USE_CUDA)
    find_program(NVCC_CANDIDATE nvcc)
    if(NVCC_CANDIDATE)
        set(USE_CUDA ON)
    else()
        set(USE_CUDA OFF)
    endif()
endif()
option(USE_CUDA "Enable CUDA support" ${USE_CUDA})
set(TP_CUDA_ENABLED ${USE_CUDA})

# AMD GPU support via the HIP toolchain.  Mutually exclusive with CUDA: the
# HIP backend registers the same device and dispatch slots, so one build can
# serve only one of the two runtimes.
option(USE_ROCM "Enable AMD GPU (HIP) support" OFF)
if(USE_ROCM AND USE_CUDA)
    message(FATAL_ERROR "USE_CUDA and USE_ROCM are mutually exclusive; pick one GPU backend per build.")
endif()

if(USE_ROCM)
    include(cmake/LoadHIP.cmake)
    if(NOT TP_FOUND_HIP)
        message(FATAL_ERROR "USE_ROCM requested but no usable HIP toolchain was found.")
    endif()
endif()

if(USE_ROCM)
    # Stage the CUDA-backed sources into the build tree and translate them
    # to HIP (API-name rewrite + file renames; see tools/amd_build).
    set(TP_HIP_STAGING "${CMAKE_BINARY_DIR}/hipified")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
                ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/amd_build/build_amd.py
                --output-dir "${TP_HIP_STAGING}"
                --subdir p10
                --subdir src/bindings/python
                --ignore tests
                --ignore benchmark
        RESULT_VARIABLE TP_HIPIFY_RESULT)
    if(NOT TP_HIPIFY_RESULT EQUAL 0)
        message(FATAL_ERROR "GPU source staging failed with ${TP_HIPIFY_RESULT}")
    endif()
    include(cmake/HipifyStage.cmake)
endif()

if(USE_CUDA)
    message(STATUS "Checking for CUDA compiler...")
    message(STATUS "CMAKE_CUDA_COMPILER is set to: '${CMAKE_CUDA_COMPILER}'")
    message(STATUS "CACHE{CMAKE_CUDA_COMPILER}: '$CACHE{CMAKE_CUDA_COMPILER}'")
    message(STATUS "ENV{CUDACXX}: '$ENV{CUDACXX}'")

    # Check for CUDA availability before enabling language
    if(NOT CMAKE_CUDA_COMPILER)
        include(CheckLanguage)
        check_language(CUDA)
    endif()

    if(CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER_ID)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        # Sources genuinely require C++20 even for nvcc (the split float8
        # headers use
        # requires-clauses).  cmake 3.25's NVIDIA module predates nvcc's
        # -std=c++20 and defines no CUDA20 options, so provide them here;
        # newer cmakes already do this natively and the guards below simply
        # overwrite with identical values.
        list(APPEND CMAKE_CUDA_COMPILE_FEATURES cuda_std_20)
        set(CMAKE_CUDA20_COMPILE_FEATURES cuda_std_20)
        set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "-std=c++20")
        set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "-std=c++20")
        # Add basic CUDA flags
        if(MSVC)
            # Windows/MSVC specific flags
            # Flags are handled via add_compile_options below
        else()
            # Linux/GCC specific flags
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
        endif()
    else()
        message(FATAL_ERROR "CUDA compiler not found. USE_CUDA is ON, so this is a fatal error. Please ensure nvcc is in your PATH or set CMAKE_CUDA_COMPILER.")
    endif()
else()
    message(STATUS "CUDA support explicitly disabled by USE_CUDA=OFF")
endif()

# Find CUDA Toolkit if enabled
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")
        string(APPEND CMAKE_CXX_FLAGS " -DUSE_CUDA")
else()
    message(STATUS "CUDA support disabled by user or not found.")
endif()

# Check for cuDNN.  Discovery (root hints, CUDA toolkit layout, pip wheel,
# Windows v9 nesting, version parsing, frontend header) lives in
# cmake/Modules/FindCUDNN.cmake; only the enable/bundling policy stays here.
set(USE_CUDNN OFF)
if(USE_ROCM)
    # The AMD backend drives the descriptor-style DNN calls through the
    # compatibility header staged by tools/amd_build, which re-creates the
    # cuDNN-style surface on top of MIOpen.  libMIOpen is already on the link
    # line via LoadHIP; enabling the guard here activates the DNN-backed code
    # paths in the staged sources.
    if(EXISTS "${TP_ROCM_PATH}/lib/libMIOpen.so")
        set(USE_CUDNN ON)
        message(STATUS "AMD backend: DNN support enabled via MIOpen (${TP_ROCM_PATH}/lib/libMIOpen.so)")
    else()
        message(STATUS "MIOpen not found; AMD backend will be built without DNN support.")
    endif()
endif()
if(USE_CUDA)
    option(USE_STATIC_CUDNN "Use cuDNN static libraries" OFF)
    if(USE_STATIC_CUDNN)
        set(CUDNN_STATIC ON CACHE BOOL "")
    else()
        set(CUDNN_STATIC OFF CACHE BOOL "")
    endif()

    find_package(CUDNN QUIET)
    if(CUDNN_FOUND)
        if(CUDNN_VERSION VERSION_LESS "8.1.0")
            message(FATAL_ERROR "TensorPlay requires cuDNN 8.1 and above.")
        endif()
        set(USE_CUDNN ON)
        message(STATUS "Found cuDNN ${CUDNN_VERSION}: ${CUDNN_INCLUDE_PATH}")
        string(APPEND CMAKE_CXX_FLAGS " -DUSE_CUDNN")
        if(CUDNN_FRONTEND_INCLUDE_DIR)
            message(STATUS "Found cuDNN frontend: ${CUDNN_FRONTEND_INCLUDE_DIR}")
        endif()

        add_library(tp::cudnn INTERFACE IMPORTED)
        target_include_directories(tp::cudnn INTERFACE ${CUDNN_INCLUDE_PATH})
        if(CUDNN_STATIC AND NOT WIN32)
            target_link_options(tp::cudnn INTERFACE
                "-Wl,--exclude-libs,libcudnn_static.a")
        else()
            target_link_libraries(tp::cudnn INTERFACE ${CUDNN_LIBRARY_PATH})
        endif()
    else()
        message(STATUS "cuDNN not found; CUDA backend will be built without cuDNN support.")
    endif()

    # Install CUDA Runtime Libraries (Windows)
    if(WIN32)
        set(CUDA_BIN_DIR "")
        if(DEFINED CUDAToolkit_BIN_DIR)
            set(CUDA_BIN_DIR "${CUDAToolkit_BIN_DIR}")
        elseif(CMAKE_CUDA_COMPILER)
             get_filename_component(CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
        elseif(DEFINED ENV{CUDA_PATH})
             set(CUDA_BIN_DIR "$ENV{CUDA_PATH}/bin")
        else()
            # Try to find CUDA bin directory
            # CUDAToolkit_LIBRARY_DIR usually points to lib/x64
            if(DEFINED CUDAToolkit_LIBRARY_DIR)
                get_filename_component(CUDA_LIB_DIR "${CUDAToolkit_LIBRARY_DIR}" ABSOLUTE)
                get_filename_component(CUDA_ROOT_DIR "${CUDA_LIB_DIR}/../.." ABSOLUTE)
                set(CUDA_BIN_DIR "${CUDA_ROOT_DIR}/bin")
            endif()
        endif()
        
        # Convert to forward slashes for CMake consistency
        file(TO_CMAKE_PATH "${CUDA_BIN_DIR}" CUDA_BIN_DIR)
        
        if(EXISTS "${CUDA_BIN_DIR}")
            message(STATUS "Found CUDA bin dir: ${CUDA_BIN_DIR}")
            
            # List of DLLs to bundle
            # We use glob to find versioned DLLs (e.g. cublas64_11.dll)
            set(CUDA_DLL_PATTERNS
                "cudart64_*.dll"
                "cublas64_*.dll"
                "cublasLt64_*.dll"
                "nvrtc64_*.dll"
            )
            
            foreach(PATTERN ${CUDA_DLL_PATTERNS})
                file(GLOB CUDA_DLLS "${CUDA_BIN_DIR}/${PATTERN}")
                if(NOT CUDA_DLLS)
                     # Try x64 subdirectory (seen in CUDA 13.0+)
                     file(GLOB CUDA_DLLS "${CUDA_BIN_DIR}/x64/${PATTERN}")
                endif()

                if(CUDA_DLLS)
                    foreach(DLL ${CUDA_DLLS})
                        # Install to package lib directory
                        install(FILES ${DLL} DESTINATION tensorplay/lib)
                        # Also copy to source for local dev
                        file(COPY ${DLL} DESTINATION ${CMAKE_SOURCE_DIR}/tensorplay/lib)
                        message(STATUS "  - Bundling CUDA DLL: ${DLL}")
                    endforeach()
                else()
                    message(STATUS "  - No match for pattern ${PATTERN} in ${CUDA_BIN_DIR}")
                endif()
            endforeach()
        else()
            message(WARNING "Could not determine CUDA bin directory. CUDAToolkit_LIBRARY_DIR=${CUDAToolkit_LIBRARY_DIR}")
        endif()
    endif()

    # On Windows, copy cuDNN DLLs if they are not in PATH
    if(WIN32 AND USE_CUDNN)
        get_filename_component(CUDNN_LIB_DIR "${CUDNN_LIBRARY}" DIRECTORY)
        # Handle v9.x structure: lib/13.1/x64 -> ../../../bin/13.1
        # Or standard: lib/x64 -> ../../bin

        # Try to find root by going up until we find 'bin'
        get_filename_component(CUDNN_ROOT_V9 "${CUDNN_LIB_DIR}/../../.." ABSOLUTE)
        get_filename_component(CUDNN_ROOT_STD "${CUDNN_LIB_DIR}/../.." ABSOLUTE)

        set(CUDNN_DLLS "")

        if(EXISTS "${CUDNN_ROOT_V9}/bin/13.1")
            file(GLOB CUDNN_DLLS "${CUDNN_ROOT_V9}/bin/13.1/cudnn*.dll")
        elseif(EXISTS "${CUDNN_ROOT_STD}/bin")
            file(GLOB CUDNN_DLLS "${CUDNN_ROOT_STD}/bin/cudnn*.dll")
        endif()

        if(NOT CUDNN_DLLS)
            # Fallback: try to find adjacent to lib if flattened, or in typical paths
            file(GLOB CUDNN_DLLS "${CUDNN_LIB_DIR}/*.dll")
        endif()

        foreach(DLL ${CUDNN_DLLS})
            file(COPY ${DLL} DESTINATION ${CMAKE_BINARY_DIR}/bin)
            file(COPY ${DLL} DESTINATION ${CMAKE_SOURCE_DIR}/tensorplay/lib)
            install(FILES ${DLL} DESTINATION tensorplay/lib)
        endforeach()
    endif()
endif()

if(WIN32)
    # Apply C++ standard only to C++ files (CUDA has its own standard setting)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/std:c++20>)
    
    if(MSVC)
        # Common MSVC flags for both CXX and CUDA (Host)
        # /EHsc: Enable exceptions
        # /wd...: Suppress warnings
        # /MP: Multi-processor compilation
        # /Gm-: Disable minimal rebuild
        set(MSVC_COMMON_FLAGS /EHsc /wd4251 /wd4275 /wd4819 /MP /Gm-)
        # M_PI and friends are gated behind this define on MSVC
        add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:_USE_MATH_DEFINES>)
        set(MSVC_REL_FLAGS /O2 /arch:AVX2)
        
        foreach(FLAG ${MSVC_COMMON_FLAGS})
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${FLAG}>)
            add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${FLAG}>)
        endforeach()
        
        # Release flags
        foreach(FLAG ${MSVC_REL_FLAGS})
            add_compile_options($<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:${FLAG}>)
            add_compile_options($<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${FLAG}>)
        endforeach()
        
        # Suppress NVCC (EDG) frontend warnings on Windows
        # 1394: field of class type without a DLL interface used in a class with a DLL interface
        # 1388: base class dllexport/dllimport specification differs from that of the derived class
        # Note: Use CMAKE_CUDA_FLAGS to avoid list splitting issues with commas in add_compile_options
        # Quote the argument to handle the comma correctly in MSBuild/Command line
        # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=1394,1388\"")

    endif()
else()
    string(APPEND CMAKE_CXX_FLAGS " -std=c++20 -O3 -Wall -Wextra -Wpedantic")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        string(APPEND CMAKE_CXX_FLAGS " -fPIC")
    endif()
endif()

include_directories(include)

# Nanobind requires "Python" package to be found first
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Derive Python Root if not set
if(NOT DEFINED Python_ROOT_DIR)
    get_filename_component(PYTHON_BIN_DIR "${Python_EXECUTABLE}" DIRECTORY)
    # Check if executable is in Scripts or bin or root
    if(EXISTS "${PYTHON_BIN_DIR}/python.exe" OR EXISTS "${PYTHON_BIN_DIR}/python")
        set(Python_ROOT_DIR "${PYTHON_BIN_DIR}")
    endif()
endif()
message(STATUS "Python Root: ${Python_ROOT_DIR}")

# Code generation variables
if(NOT DEFINED GEN_BASE_DIR)
    set(GEN_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/generated")
endif()
set(GEN_INCLUDE_DIR "${GEN_BASE_DIR}/tensorplay/ops")
set(GEN_HEADER "${GEN_INCLUDE_DIR}/TensorGenerated.h")
set(GEN_CPP "${GEN_INCLUDE_DIR}/TensorGenerated.cpp")
set(GEN_BINDINGS_HEADER "${GEN_INCLUDE_DIR}/TensorBindingsGenerated.h")
set(GEN_CPYTHON_HEADER "${GEN_INCLUDE_DIR}/TensorCPythonGenerated.h")
set(GEN_AUTOGRAD_NODES_H "${GEN_INCLUDE_DIR}/AutogradNodesGenerated.h")
set(GEN_TPX_OPS_H "${GEN_INCLUDE_DIR}/TPXOpsGenerated.h")
set(GEN_TPX_OPS_CPP "${GEN_INCLUDE_DIR}/TPXOpsGenerated.cpp")
set(GEN_TPX_AUTOGRAD_REG_CPP "${GEN_INCLUDE_DIR}/TPXAutogradRegistration.cpp")
set(GEN_AUTOCAST_CPP "${GEN_INCLUDE_DIR}/AutocastGenerated.cpp")
set(GEN_TPX_REDISPATCH_H "${GEN_INCLUDE_DIR}/TensorRedispatchGenerated.h")
set(GEN_STRUCTURED_H "${GEN_INCLUDE_DIR}/StructuredGenerated.h")

# Python Interface Stub Generation
if(NOT DEFINED TP_PACKAGE_DIR)
    set(TP_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tensorplay")
endif()

# Generate tensorplay/version.py from version.txt + git metadata
set(TP_VERSION_PY "${TP_PACKAGE_DIR}/version.py")
add_custom_command(
    OUTPUT ${TP_VERSION_PY}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
            ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tensorplay_version.py
            --is-debug $<IF:$<CONFIG:Debug>,1,0>
            --cuda-version $<IF:$<BOOL:${CUDAToolkit_VERSION}>,${CUDAToolkit_VERSION},>
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tensorplay_version.py
            ${CMAKE_CURRENT_SOURCE_DIR}/version.txt
    COMMENT "Generating tensorplay/version.py"
)

# Add a custom target to ensure code generation runs before library building
add_custom_target(generate_code DEPENDS ${GEN_HEADER} ${GEN_CPP} ${GEN_TPX_OPS_CPP} ${GEN_TPX_AUTOGRAD_REG_CPP} ${GEN_AUTOCAST_CPP} ${GEN_TPX_REDISPATCH_H} ${GEN_STRUCTURED_H} ${TP_VERSION_PY})

include_directories(${GEN_BASE_DIR})

# pybind11 via pip installation
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

# --- Compute Backend Selection ---
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    # Apply OpenMP flags only to CXX files directly
    # For CUDA, pass them to the host compiler via -Xcompiler
    string(APPEND CMAKE_CXX_FLAGS " ${OpenMP_CXX_FLAGS}")
    
    if(MSVC)
        # MSVC flag is usually -openmp or /openmp
        # Use -Xcompiler to pass flags to host compiler from nvcc
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp>)
    else()
        # GCC/Clang flag is -fopenmp
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
    endif()
endif()

# --- OneAPI Configuration ---
if(WIN32)
    set(ONEAPI_ROOT "C:/Program Files (x86)/Intel/oneAPI")
    if(EXISTS "${ONEAPI_ROOT}")
        message(STATUS "Found OneAPI Root: ${ONEAPI_ROOT}")
        # Add OneAPI root to prefix path so find_package can locate config files
        # OneDNN config: <root>/dnnl/latest/lib/cmake/dnnl
        # MKL config: <root>/mkl/latest/lib/cmake/mkl
        list(APPEND CMAKE_PREFIX_PATH "${ONEAPI_ROOT}")
    endif()
else()
    # Linux/macOS: honor ONEAPI_ROOT, else the default install location, so
    # find_package(MKL CONFIG) can discover a system oneAPI install.
    if(NOT DEFINED ENV{ONEAPI_ROOT} AND EXISTS "/opt/intel/oneapi")
        set(ENV{ONEAPI_ROOT} "/opt/intel/oneapi")
    endif()
    if(DEFINED ENV{ONEAPI_ROOT} AND EXISTS "$ENV{ONEAPI_ROOT}")
        message(STATUS "Found OneAPI Root: $ENV{ONEAPI_ROOT}")
        list(APPEND CMAKE_PREFIX_PATH "$ENV{ONEAPI_ROOT}/mkl/latest"
                                      "$ENV{ONEAPI_ROOT}/dnnl/latest")
    endif()
endif()

option(USE_BLAS "Enable BLAS linear algebra acceleration" ON)

# Vulkan backend: enabled when a Vulkan loader can be located.  The loader
# is linked from the system; headers are vendored under
# third_party/vulkan-headers and the GLSL compiler is vendored under
# tools/bin/glslangValidator, so no Vulkan SDK install is required.
if(NOT DEFINED USE_VULKAN)
    find_library(VULKAN_LOADER_CANDIDATE vulkan)
    if(VULKAN_LOADER_CANDIDATE)
        set(USE_VULKAN ON)
    else()
        set(USE_VULKAN OFF)
    endif()
endif()
option(USE_VULKAN "Enable Vulkan support" ${USE_VULKAN})
if(USE_VULKAN)
    find_library(VULKAN_LIBRARY NAMES vulkan)
    if(NOT VULKAN_LIBRARY)
        message(FATAL_ERROR "USE_VULKAN is ON but the Vulkan loader library was not found.")
    endif()
    message(STATUS "Found Vulkan loader: ${VULKAN_LIBRARY}")
    string(APPEND CMAKE_CXX_FLAGS " -DUSE_VULKAN")
endif()

set(COMPUTE_LIBS "")
set(BLAS_COMPILE_DEFS "")
set(BLAS_COMPILE_OPTIONS "")

if(USE_BLAS)
    # Default MKL; when it is absent the detector falls through to OpenBLAS
    # (system, then the vendored copy below) so wheels never ship the
    set(BLAS_PROVIDER "MKL" CACHE STRING "BLAS provider (MKL/OpenBLAS)")
    set_property(CACHE BLAS_PROVIDER PROPERTY STRINGS "MKL" "OpenBLAS")

    if(BLAS_PROVIDER STREQUAL "MKL")
        # 1. Try MKL Config mode (Best for OneAPI)
        # statically into our .so -- shares the libgomp runtime with our own
        # parallel_for pool, so no second threading runtime is loaded.
        set(MKL_THREADING "gnu_thread")
        set(MKL_LINK "static")
        # Our kernels use 32-bit cblas indices; MKLConfig defaults to ilp64.
        set(MKL_INTERFACE lp64)
        # Config mode first (oneAPI MKLConfig), then the module fallback:
        # CI unpacks the mkl-static / mkl-include wheels under /opt/intel
        # (MKLROOT), which only the module layout knows how to assemble.
        find_package(MKL CONFIG QUIET)
        if(NOT MKL_FOUND)
            find_package(MKL MODULE QUIET)
        endif()
        
        if(MKL_FOUND)
            message(STATUS "Found MKL: ${MKL_DIR}")
            list(APPEND COMPUTE_LIBS MKL::MKL)
            set(BLAS_FOUND TRUE)
            list(APPEND BLAS_COMPILE_DEFS "USE_MKL")
            
            # Install MKL DLLs (Windows)
            if(WIN32)
                set(MKL_BIN_DIR "")
                # Try to derive from MKL_DIR (Config mode)
                if(MKL_DIR)
                    get_filename_component(MKL_CMAKE_DIR "${MKL_DIR}" ABSOLUTE)
                    # MKL_DIR is usually <root>/lib/cmake/mkl
                    get_filename_component(MKL_ROOT_DERIVED "${MKL_CMAKE_DIR}/../../.." ABSOLUTE)
                    if(EXISTS "${MKL_ROOT_DERIVED}/bin")
                        set(MKL_BIN_DIR "${MKL_ROOT_DERIVED}/bin")
                    endif()
                endif()

                # Fallback to standard locations
                if(NOT MKL_BIN_DIR)
                    if(EXISTS "${ONEAPI_ROOT}/mkl/latest/bin")
                         set(MKL_BIN_DIR "${ONEAPI_ROOT}/mkl/latest/bin")
                    elseif(EXISTS "${ONEAPI_ROOT}/bin")
                         set(MKL_BIN_DIR "${ONEAPI_ROOT}/bin")
                    endif()
                endif()
                
                if(MKL_BIN_DIR AND EXISTS "${MKL_BIN_DIR}")
                    message(STATUS "Found MKL bin dir: ${MKL_BIN_DIR}")
                    # Bundle MKL DLLs (using glob to match version suffixes like .2.dll)
                    # Core, Sequential, Interface, Dispatchers
                    set(MKL_PATTERNS 
                        "mkl_core*.dll"
                        "mkl_sequential*.dll" 
                        "mkl_intel_lp64*.dll"
                        "mkl_def*.dll"
                        "mkl_avx2*.dll"
                        "mkl_vml_avx2*.dll"
                        "mkl_vml_def*.dll"
                    )
                    foreach(PATTERN ${MKL_PATTERNS})
                        file(GLOB MKL_DLLS "${MKL_BIN_DIR}/${PATTERN}")
                        foreach(DLL ${MKL_DLLS})
                             install(FILES "${DLL}" DESTINATION tensorplay/lib)
                             file(COPY "${DLL}" DESTINATION ${CMAKE_SOURCE_DIR}/tensorplay/lib)
                             message(STATUS "  - Bundling MKL DLL: ${DLL}")
                        endforeach()
                    endforeach()
                else()
                    message(WARNING "Could not find MKL bin directory to bundle DLLs.")
                endif()
            endif()
        else()
            # Module-mode fallback (cmake/Modules/FindMKL.cmake): manual
            # oneAPI/classic layouts, then a generic FindBLAS pass.  The
            # module reports the result through MKL::MKL.
            find_package(MKL MODULE QUIET)
            if(MKL_FOUND)
                message(STATUS "Found MKL (module): ${MKL_LIBRARIES}")
                list(APPEND COMPUTE_LIBS MKL::MKL)
                set(BLAS_FOUND TRUE)
                list(APPEND BLAS_COMPILE_DEFS "USE_MKL")
            endif()
        endif()

        # MKL absent (CI runners, clean dev boxes): still try OpenBLAS below
        # instead of silently dropping to the naive-GEMM fallback.
        if(NOT BLAS_FOUND)
            set(TP_TRY_OPENBLAS TRUE)
        endif()

    endif()

    if(NOT BLAS_FOUND AND (BLAS_PROVIDER STREQUAL "OpenBLAS" OR TP_TRY_OPENBLAS))
        find_package(OpenBLAS CONFIG QUIET)
        if(OpenBLAS_FOUND AND NOT TARGET OpenBLAS::OpenBLAS)
            # Distribution config scripts only export the plain target name;
            # provide the namespaced alias so downstream references resolve.
            if(TARGET openblas)
                add_library(OpenBLAS::OpenBLAS ALIAS openblas)
            elseif(TARGET libopenblas)
                add_library(OpenBLAS::OpenBLAS ALIAS libopenblas)
            elseif(TARGET openblas_static)
                add_library(OpenBLAS::OpenBLAS ALIAS openblas_static)
            else()
                message(STATUS
                    "OpenBLAS config package found but exports no known "
                    "target; falling back to module-mode detection")
                set(OpenBLAS_FOUND FALSE)
            endif()
        endif()
        if(OpenBLAS_FOUND AND TARGET OpenBLAS::OpenBLAS)
             list(APPEND COMPUTE_LIBS OpenBLAS::OpenBLAS)
             set(BLAS_FOUND TRUE)
             list(APPEND BLAS_COMPILE_DEFS "USE_OPENBLAS" "USE_BLAS")
        else()
             # Module-mode fallback (cmake/Modules/FindOpenBLAS.cmake):
             # pkg-config or manual prefix layouts.
             find_package(OpenBLAS MODULE QUIET)
             if(OpenBLAS_FOUND AND TARGET OpenBLAS::OpenBLAS)
                 list(APPEND COMPUTE_LIBS OpenBLAS::OpenBLAS)
                 set(BLAS_FOUND TRUE)
                 list(APPEND BLAS_COMPILE_DEFS "USE_OPENBLAS" "USE_BLAS")
             endif()
        endif()
    endif()
    
    # Last resort for CI runners / clean machines: build the vendored OpenBLAS
    # (C-only BLAS, no LAPACK/Fortran) as static PIC and link it in, so shipped
    # wheels never fall back to gemm_naive.  Same pattern as the oneDNN vendoring.
    if(NOT BLAS_FOUND AND NOT BLAS_PROVIDER STREQUAL "None"
       AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/OpenBLAS-0.3.29.tar.gz")
        message(STATUS "Configuring vendored OpenBLAS 0.3.29 (static, C-only)...")
        include(FetchContent)  # normally pulled in by the oneDNN section below
        # BUILD_SHARED_LIBS must be scoped to the vendored subbuild ONLY.
        # A bare set() here leaks into every later add_subdirectory: p10
        # silently builds static, the linker then discards the registration-
        # only kernel objects (nothing references them) and ALL CUDA/CPU
        # kernels vanish from the dispatcher at runtime.
        set(TP_SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
        set(BUILD_SHARED_LIBS OFF)
        set(BUILD_STATIC_LIBS ON)
        set(NO_LAPACK ON)
        set(C_LAPACK OFF)
        set(NOFORTRAN ON)
        FetchContent_Declare(openblas_vendor
            URL "${CMAKE_CURRENT_SOURCE_DIR}/third_party/OpenBLAS-0.3.29.tar.gz"
            URL_HASH SHA256=38240eee1b29e2bde47ebb5d61160207dc68668a54cac62c076bb5032013b1eb
        )
        FetchContent_MakeAvailable(openblas_vendor)
        set(BUILD_SHARED_LIBS ${TP_SAVED_BUILD_SHARED_LIBS})
        set(BUILD_STATIC_LIBS ${TP_SAVED_BUILD_SHARED_LIBS})
        # `openblas` is an ALIAS; the static archive is the real target.
        if(TARGET openblas_static)
            set_property(TARGET openblas_static PROPERTY POSITION_INDEPENDENT_CODE ON)
            list(APPEND COMPUTE_LIBS openblas_static)
            # Scope the vendored headers to consumers of the target (p10).
            # A global include_directories() here would leak this directory
            # into EVERY later target — oneDNN's bundled xbyak then resolves
            # <cpuid.h> to OpenBLAS's helper of the same name (which does not
            # define GCC's __cpuid macro) and fails to compile on MKL-less
            # machines.  Consumers inherit these dirs transitively.
            target_include_directories(openblas_static SYSTEM INTERFACE
                $<BUILD_INTERFACE:${openblas_vendor_SOURCE_DIR}>
                $<BUILD_INTERFACE:${openblas_vendor_BINARY_DIR}>)
            set(BLAS_FOUND TRUE)
            list(APPEND BLAS_COMPILE_DEFS "USE_OPENBLAS" "USE_BLAS")
            message(STATUS "✅ Vendored OpenBLAS configured")
        endif()
    endif()

    if(BLAS_FOUND)
        # Report the provider that actually configured; the cache variable
        # keeps its requested default even when detection fell through to
        # another backend.
        if("USE_MKL" IN_LIST BLAS_COMPILE_DEFS)
            set(TP_BLAS_ACTIVE "MKL")
        elseif("USE_OPENBLAS" IN_LIST BLAS_COMPILE_DEFS)
            set(TP_BLAS_ACTIVE "OpenBLAS")
        else()
            set(TP_BLAS_ACTIVE "${BLAS_PROVIDER}")
        endif()
        message(STATUS "✅ Successfully configured BLAS: ${TP_BLAS_ACTIVE} (${BLAS_COMPILE_DEFS})")
    else()
        message(WARNING "Could not find ${BLAS_PROVIDER}.")
    endif()
endif()

# --- oneDNN Integration (cmake/Modules/FindMKLDNN.cmake) ---
# Resolution order inside the module: a system installation first, then the
# vendored source tree at third_party/oneDNN built as a static library with
# the OMP CPU runtime (default), the graph API on Linux, and portable codegen
# unless MKLDNN_USE_NATIVE_ARCH is set.
option(USE_MKLDNN "Enable oneDNN for deep learning primitives" ON)
option(USE_NATIVE_ARCH "Let dependencies tune for the build host (-march=native / HostOpts)" OFF)
option(USE_NCCL "Use NCCL for tensorplay.distributed" ON)
option(USE_RCCL "Use RCCL for tensorplay.distributed" OFF)
if(USE_ROCM)
    set(USE_RCCL ON CACHE BOOL "Use RCCL for tensorplay.distributed" FORCE)
    set(USE_NCCL OFF CACHE BOOL "Use NCCL for tensorplay.distributed" FORCE)
    find_package(RCCL REQUIRED)
elseif(USE_RCCL)
    message(FATAL_ERROR "USE_RCCL requires USE_ROCM=ON.")
endif()
# Compatibility alias for the historical knob name.
if(DEFINED USE_ONEDNN AND NOT DEFINED USE_MKLDNN)
    set(USE_MKLDNN "${USE_ONEDNN}")
endif()

set(MKLDNN_USE_NATIVE_ARCH ${USE_NATIVE_ARCH})
set(ONEDNN_FOUND FALSE)
if(USE_MKLDNN)
    find_package(MKLDNN QUIET)
    if(MKLDNN_FOUND)
        list(APPEND COMPUTE_LIBS ${MKLDNN_LIBRARIES})
        foreach(_inc ${MKLDNN_INCLUDE_DIRS})
            include_directories(SYSTEM ${_inc})
        endforeach()
        set(ONEDNN_FOUND TRUE)
        message(STATUS "✅ Successfully configured oneDNN (MKLDNN_CPU_RUNTIME=${MKLDNN_CPU_RUNTIME})")
    else()
        message(STATUS "oneDNN not found; building without MKL-DNN primitives.")
    endif()
endif()

# --- NNPACK Integration ---
option(USE_NNPACK "Use NNPACK" ON)

# ---[ NNPACK
if(USE_NNPACK)
    include(${CMAKE_CURRENT_LIST_DIR}/cmake/External/nnpack.cmake)
    if(NNPACK_FOUND)
        if(TARGET nnpack)
            # ---[ NNPACK is built together with TensorPlay: explicitly specify dependency
            list(APPEND COMPUTE_LIBS nnpack)
        else()
            include_directories(SYSTEM ${NNPACK_INCLUDE_DIRS})
            list(APPEND COMPUTE_LIBS ${NNPACK_LIBRARIES})
        endif()
    else()
        message(WARNING "Not compiling with NNPACK. Suppress this warning with -DUSE_NNPACK=OFF")
        set(USE_NNPACK OFF)
    endif()
endif()

if(NNPACK_FOUND AND NOT CMAKE_CXX_FLAGS MATCHES "USE_NNPACK")
    string(APPEND CMAKE_CXX_FLAGS " -DUSE_NNPACK")
endif()

# --- SLEEF Integration ---
include(${CMAKE_CURRENT_LIST_DIR}/cmake/External/sleef.cmake)

if(ONEDNN_FOUND)
    string(APPEND CMAKE_CXX_FLAGS " -DUSE_MKLDNN -DUSE_ONEDNN")
endif()
foreach(_def ${BLAS_COMPILE_DEFS})
    string(APPEND CMAKE_CXX_FLAGS " -D${_def}")
endforeach()

# Link the OpenMP runtime into every compute library and the Python extension.
# Compile-only flags (-fopenmp) emit GOMP_* symbols without resolving them,
# producing shared libraries with undefined OpenMP symbols (e.g. GOMP_critical_start).
# COMPUTE_LIBS is propagated PUBLICly through p10 -> tpx -> stax -> _C.
if(OpenMP_CXX_FOUND)
    list(APPEND COMPUTE_LIBS OpenMP::OpenMP_CXX)
endif()

if(MSVC)
  list(APPEND BLAS_COMPILE_OPTIONS "/O2" "/arch:AVX2")
else()
  list(APPEND BLAS_COMPILE_OPTIONS "-O3")
  if(USE_NATIVE_ARCH)
      list(APPEND BLAS_COMPILE_OPTIONS "-march=native")
  endif()
endif()

if(NOT BLAS_FOUND)
    if(WIN32)
        message(STATUS "⚠️ Compute library not found. Attempting to auto-download OpenBLAS (Windows Pre-built)...")
        include(FetchContent)
        
        # Use OpenBLAS 0.3.26 which is stable and has pre-built binaries
        FetchContent_Declare(
            openblas_prebuilt
            URL https://github.com/xianyi/OpenBLAS/releases/download/v0.3.26/OpenBLAS-0.3.26-x64.zip
        )
        
        FetchContent_GetProperties(openblas_prebuilt)
        if(NOT openblas_prebuilt_POPULATED)
            message(STATUS "Downloading OpenBLAS 0.3.26...")
            FetchContent_Populate(openblas_prebuilt)
        endif()
        
        set(OpenBLAS_ROOT ${openblas_prebuilt_SOURCE_DIR})
        set(OpenBLAS_INCLUDE_DIR "${OpenBLAS_ROOT}/include")
        set(OpenBLAS_LIBRARY "${OpenBLAS_ROOT}/lib/libopenblas.lib")
        
        if(EXISTS "${OpenBLAS_INCLUDE_DIR}" AND EXISTS "${OpenBLAS_LIBRARY}")
             message(STATUS "✅ Successfully downloaded OpenBLAS to ${OpenBLAS_ROOT}")
             
             include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
             list(APPEND COMPUTE_LIBS ${OpenBLAS_LIBRARY})
             list(APPEND BLAS_COMPILE_DEFS "USE_OPENBLAS")
             list(APPEND BLAS_COMPILE_DEFS "USE_BLAS")
             set(BLAS_FOUND TRUE)
             
             # Copy DLLs to binary directories for runtime
             file(GLOB OPENBLAS_DLLS "${OpenBLAS_ROOT}/bin/*.dll")
             foreach(DLL ${OPENBLAS_DLLS})
                 # Copy to build output dirs
                 file(COPY ${DLL} DESTINATION ${CMAKE_BINARY_DIR}/Release)
                 file(COPY ${DLL} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
                 # Copy to Python package lib dir (for import tensorplay)
                 file(COPY ${DLL} DESTINATION ${CMAKE_SOURCE_DIR}/tensorplay/lib)
                 # Install to package
                 install(FILES ${DLL} DESTINATION tensorplay/lib)
                 message(STATUS "  - Copied DLL: ${DLL}")
             endforeach()
        else()
             message(WARNING "Downloaded OpenBLAS but verification failed (include/lib not found).")
        endif()
    else()
        message(WARNING "No accelerated compute library (MKL/BLAS) found. Auto-download is only supported on Windows. Using naive implementation.")
    endif()
endif()

if(NOT BLAS_FOUND)
    message(WARNING "No accelerated compute library (MKL/BLAS) found. Using naive implementation.")
endif()

# --- Config Generation ---
set(TP_BUILD_TYPE ${CMAKE_BUILD_TYPE})
if(NOT TP_BUILD_TYPE)
    set(TP_BUILD_TYPE "Release")
endif()

file(STRINGS "version.txt" TENSORPLAY_VERSION)
string(STRIP "${TENSORPLAY_VERSION}" TENSORPLAY_VERSION)

# Collect Build Info
set(TP_CXX_VERSION "202002") # C++20

if(MSVC)
    set(TP_COMPILER_INFO "MSVC ${MSVC_VERSION}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    string(REGEX REPLACE "^([0-9]+)\\.([0-9]+).*" "\\1.\\2" _compiler_ver_short "${CMAKE_CXX_COMPILER_VERSION}")
    set(TP_COMPILER_INFO "GCC ${_compiler_ver_short}")
else()
    set(TP_COMPILER_INFO "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
endif()

if(BLAS_PROVIDER STREQUAL "MKL" AND BLAS_FOUND)
    # Try to find MKL version if possible, otherwise use generic text or placeholder
    # Config mode often sets MKL_VERSION or similar variables.
    if(DEFINED MKL_VERSION)
        set(MKL_VER_STR "${MKL_VERSION}")
    else()
        # Fallback if not explicitly set, though Config mode usually sets it
        set(MKL_VER_STR "Unknown")
    endif()
    set(TP_MKL_INFO "Intel(R) oneAPI Math Kernel Library Version ${MKL_VER_STR} for Intel(R) 64 architecture applications")
    set(TP_LAPACK_INFO "LAPACK is enabled (usually provided by MKL)")
elseif(BLAS_PROVIDER STREQUAL "OpenBLAS" AND BLAS_FOUND)
    set(TP_MKL_INFO "OpenBLAS ${OpenBLAS_VERSION}")
    set(TP_LAPACK_INFO "LAPACK is enabled (via OpenBLAS)")
else()
    set(TP_MKL_INFO "N/A")
    set(TP_LAPACK_INFO "LAPACK is disabled")
endif()

if(ONEDNN_FOUND)
    set(TP_ONEDNN_INFO "Intel(R) MKL-DNN runtime-detected")
else()
    set(TP_ONEDNN_INFO "N/A")
endif()

if(OpenMP_CXX_FOUND)
    set(TP_OPENMP_INFO "OpenMP runtime-detected")
else()
    set(TP_OPENMP_INFO "OpenMP disabled")
endif()

if(CUDAToolkit_FOUND)
    set(TP_CUDA_INFO "CUDA ${CUDAToolkit_VERSION}")
    if(USE_CUDNN)
        set(TP_CUDNN_INFO "cuDNN enabled")
    else()
        set(TP_CUDNN_INFO "cuDNN disabled")
    endif()
else()
    set(TP_CUDA_INFO "CUDA disabled")
    set(TP_CUDNN_INFO "cuDNN disabled")
endif()
if(MSVC)
    set(TP_CPU_CAPABILITY "AVX2")
else()
    set(TP_CPU_CAPABILITY "Native")
endif()

# Normalize CMake TRUE/FALSE to ON/OFF for build settings
function(tp_bool_to_onoff VAR)
    if(${VAR})
        set(_out "ON")
    else()
        set(_out "OFF")
    endif()
    set("${VAR}_OUT" "${_out}" PARENT_SCOPE)
endfunction()

# Call normalize functions before build settings are assembled
tp_bool_to_onoff(OpenMP_CXX_FOUND)
tp_bool_to_onoff(BLAS_FOUND)
tp_bool_to_onoff(ONEDNN_FOUND)
tp_bool_to_onoff(USE_NNPACK)

# Build Settings String
set(TP_BUILD_SETTINGS_LIST)

if(BLAS_PROVIDER STREQUAL "MKL")
    set(BLAS_INFO_VAL "mkl")
else()
    set(BLAS_INFO_VAL "${BLAS_PROVIDER}")
endif()

# Normalize the CUDA toggle to 0/1 for build settings
if(USE_CUDA)
    set(TP_USE_CUDA_VAL "1")
else()
    set(TP_USE_CUDA_VAL "0")
endif()

# Entries are emitted in alphabetical key order
list(APPEND TP_BUILD_SETTINGS_LIST "BLAS_INFO=${BLAS_INFO_VAL}")
list(APPEND TP_BUILD_SETTINGS_LIST "BUILD_TYPE=${TP_BUILD_TYPE}")

# Get git commit SHA (full hash)
execute_process(
    COMMAND git rev-parse HEAD
    OUTPUT_VARIABLE TP_COMMIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_RESULT
)
if(NOT GIT_RESULT EQUAL 0)
    set(TP_COMMIT_SHA "unknown")
endif()
list(APPEND TP_BUILD_SETTINGS_LIST "COMMIT_SHA=${TP_COMMIT_SHA}")

# CUDA compile flags are recorded even for CPU-only builds; when CUDA is
# disabled this carries the baseline flags a CUDA build would use.
set(TP_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
if(NOT TP_CUDA_FLAGS)
    set(TP_CUDA_FLAGS "-Xcompiler -fPIC")
endif()
list(APPEND TP_BUILD_SETTINGS_LIST "CUDA_FLAGS=${TP_CUDA_FLAGS}")

# Sanitize CXX_COMPILER (remove quotes, use forward slashes)
string(REPLACE "\"" "" CXX_COMPILER_NO_QUOTES "${CMAKE_CXX_COMPILER}")
file(TO_CMAKE_PATH "${CXX_COMPILER_NO_QUOTES}" CXX_COMPILER_FWD)
list(APPEND TP_BUILD_SETTINGS_LIST "CXX_COMPILER=${CXX_COMPILER_FWD}")

list(APPEND TP_BUILD_SETTINGS_LIST "CXX_FLAGS=${CMAKE_CXX_FLAGS}")
list(APPEND TP_BUILD_SETTINGS_LIST "LAPACK_INFO=${BLAS_INFO_VAL}")
list(APPEND TP_BUILD_SETTINGS_LIST "PERF_WITH_AVX=1")
list(APPEND TP_BUILD_SETTINGS_LIST "PERF_WITH_AVX2=1")
list(APPEND TP_BUILD_SETTINGS_LIST "TENSORPLAY_VERSION=${TENSORPLAY_VERSION}")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_CUDA=${TP_USE_CUDA_VAL}")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_CUDNN=${USE_CUDNN}")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_CUSPARSELT=OFF")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_GFLAGS=OFF")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_GLOG=OFF")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_GLOO=ON")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_MKL=${BLAS_FOUND_OUT}")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_MKLDNN=${ONEDNN_FOUND_OUT}")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_MPI=OFF")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_NCCL=${USE_NCCL}")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_RCCL=${USE_RCCL}")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_NNPACK=${USE_NNPACK_OUT}")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_OPENMP=${OpenMP_CXX_FOUND_OUT}")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_ROCM=${USE_ROCM}")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_ROCM_KERNEL_ASSERT=OFF")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_XCCL=OFF")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_XPU=OFF")

string(REPLACE ";" ", " TP_BUILD_SETTINGS_STR "${TP_BUILD_SETTINGS_LIST}")
string(APPEND TP_BUILD_SETTINGS_STR ", ")
# Escape backslashes for C++ string literal
string(REPLACE "\\" "\\\\" TP_BUILD_SETTINGS_STR "${TP_BUILD_SETTINGS_STR}")
# Escape quotes for C++ string literal
string(REPLACE "\"" "\\\"" TP_BUILD_SETTINGS_STR "${TP_BUILD_SETTINGS_STR}")
# Escape CXX_FLAGS for C++ string literal
set(ESCAPED_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "\\" "\\\\" ESCAPED_CXX_FLAGS "${ESCAPED_CXX_FLAGS}")
string(REPLACE "\"" "\\\"" ESCAPED_CXX_FLAGS "${ESCAPED_CXX_FLAGS}")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/ConfigInfo.h.in"
               "${GEN_BASE_DIR}/tensorplay/ops/Config.h")

# Code generation: the schema engine is self-contained under tools/codegen.
file(GLOB_RECURSE tp_codegen_reference CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/*.py")
set(TP_CODEGEN_TAGS "${CMAKE_CURRENT_SOURCE_DIR}/config/tags.yaml")
add_custom_command(
            OUTPUT ${GEN_HEADER} ${GEN_CPP} ${GEN_BINDINGS_HEADER}
                   ${GEN_CPYTHON_HEADER}
                   ${GEN_AUTOGRAD_NODES_H} ${GEN_TPX_OPS_H} ${GEN_TPX_OPS_CPP}
                   ${GEN_TPX_AUTOGRAD_REG_CPP} ${GEN_AUTOCAST_CPP} ${GEN_TPX_REDISPATCH_H}
                   ${GEN_STRUCTURED_H}
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR} ${Python_EXECUTABLE} -m tools.codegen.main
                    --yaml ${CMAKE_CURRENT_SOURCE_DIR}/config/native_functions.yaml
                    --derivatives ${CMAKE_CURRENT_SOURCE_DIR}/config/derivatives.yaml
                    --out_dir ${GEN_INCLUDE_DIR}
                    --pkg_out ${CMAKE_CURRENT_SOURCE_DIR}/tensorplay
            DEPENDS
                ${CMAKE_CURRENT_SOURCE_DIR}/config/native_functions.yaml
                ${CMAKE_CURRENT_SOURCE_DIR}/config/derivatives.yaml
                ${TP_CODEGEN_TAGS}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/main.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/model.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/api_types.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/gen_api.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/gen_autograd.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/gen_autocast.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/gen_bindings.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/gen_python.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/gen_pyi.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/gen_tpx.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/gen_python_c.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/gen_structured.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/gen_inplace_or_view.py
                ${tp_codegen_reference}
                ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/python/CPythonBridge.h
                ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/python/CPythonBridge.cpp
            COMMENT "Generating Tensor methods, bindings and pyi stubs from native_functions.yaml"
        )

set_source_files_properties(
    ${GEN_HEADER} ${GEN_CPP} ${GEN_BINDINGS_HEADER}
    ${GEN_AUTOGRAD_NODES_H} ${GEN_TPX_OPS_H} ${GEN_TPX_OPS_CPP}
    ${GEN_TPX_AUTOGRAD_REG_CPP} ${GEN_AUTOCAST_CPP} ${GEN_TPX_REDISPATCH_H}
    ${GEN_STRUCTURED_H}
    PROPERTIES GENERATED TRUE
)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
# statically linked into the single Python extension (_C.pyd).  This avoids
# needing __declspec(dllexport) annotations on every p10 symbol that crosses
# the DLL boundary, which the codebase does not carry.
if(WIN32 AND BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS OFF)
endif()

# Add p10
add_subdirectory(p10)
# Add tpx
add_subdirectory(tpx)

# Add stax (Optional)
add_subdirectory(stax)

# ONE shared copy keeps the pybind11 caster substrate and the wrap-identity
# cache registered from a single module boundary: _C registers the Tensor
# type through it at import time, and user op modules (add_tensorplay_op)
# link the same library so their tpx_py_wrap calls resolve against that
# registration.  Compiling CPythonBridge.cpp per-extension segfaults on the
# first returned Tensor (each copy would carry its own registry/cache).
# Windows keeps the historical layout (statically embedded everywhere)
# because the codebase carries no __declspec(dllexport) annotations.
if(NOT WIN32)
    add_library(tp_python SHARED src/bindings/python/CPythonBridge.cpp)
    target_include_directories(tp_python
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/p10/include"
               "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/python"
               "${CMAKE_CURRENT_SOURCE_DIR}/tpx/include")
    target_link_libraries(tp_python PUBLIC p10 pybind11::pybind11)
endif()

# ---------------------------------------------------------------------------
# Distributed transports.  Platform defaults and dependent options follow
# the reference layout: distributed is off by default outside Linux and
# Windows; tensorpipe never builds on Windows; gloo requires a 64-bit host.
# ---------------------------------------------------------------------------
if(NOT LINUX AND NOT WIN32)
    set(USE_TP_DISTRIBUTED OFF CACHE BOOL "Use distributed" FORCE)
else()
    set(USE_TP_DISTRIBUTED ON CACHE BOOL "Use distributed")
endif()
cmake_dependent_option(TP_USE_GLOO "Use the gloo process-group transport" ON
    "USE_TP_DISTRIBUTED AND NOT WIN32" OFF)
# The tensorpipe sources use POSIX-only interfaces (ssize_t and the fd
# sharing socket API) and do not compile under MSVC; Windows keeps the
# gloo transport only.
cmake_dependent_option(TP_USE_TENSORPIPE "Use the tensorpipe RPC transport" ON
    "USE_TP_DISTRIBUTED AND NOT WIN32" OFF)
cmake_dependent_option(TP_USE_MPI "Enable the MPI process group backend" ON
    "USE_TP_DISTRIBUTED" OFF)

if(TP_USE_GLOO AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(WARNING "Gloo can only be used on 64-bit systems.")
    set(TP_USE_GLOO OFF)
endif()

# ---------------------------------------------------------------------------
# CPU process-group backends (gloo transport + MPI), kept as a separate
# static library so the pybind11 module stays free of unity-build collisions.
# ---------------------------------------------------------------------------
if(TP_USE_GLOO)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_TEST OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(USE_REDIS OFF CACHE BOOL "" FORCE)
set(USE_IBVERBS OFF CACHE BOOL "" FORCE)
set(TP_GLOO_USE_RCCL_SAVE "${USE_RCCL}")
set(USE_NCCL OFF CACHE BOOL "" FORCE)
set(USE_RCCL OFF)
set(USE_RCCL OFF CACHE BOOL "" FORCE)
# libuv ships as a non-PIC static archive on Linux and cannot be linked
# into the shared transport library there (the TCP transport covers our
# use); Windows gloo only supports the libuv transport, so it stays on and
# consumes the libuv_ROOT provisioned by CI.
if(NOT WIN32)
    set(USE_LIBUV OFF CACHE BOOL "" FORCE)
endif()
set(USE_TCP_OPENSSL_LINK OFF CACHE BOOL "" FORCE)
set(USE_TCP_OPENSSL_LOAD OFF CACHE BOOL "" FORCE)
# Gloo links statically into the distributed objects and is never installed.
# A shared gloo would leave a libgloo.so NEEDED entry on _C plus another
# install tree to ship, and a wheel import then fails wherever the loader
# does not search the extra directory.
set(GLOO_INSTALL OFF CACHE BOOL "" FORCE)
set(GLOO_STATIC_OR_SHARED STATIC CACHE STRING "" FORCE)
# Gloo builds CPU-only, but its USE_CUDA knob lives in the global namespace:
# save the project's own switch and restore it after the subdirectory, or a
# fresh configure silently turns the whole CUDA build off.
set(TP_GLOO_USE_CUDA_SAVE "${USE_CUDA}")
set(USE_CUDA OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/gloo)
set(USE_CUDA "${TP_GLOO_USE_CUDA_SAVE}" CACHE BOOL "" FORCE)
set(USE_RCCL "${TP_GLOO_USE_RCCL_SAVE}")
set(USE_RCCL "${TP_GLOO_USE_RCCL_SAVE}" CACHE BOOL "" FORCE)
endif()

if(TP_USE_TENSORPIPE)
set(TP_USE_CUDA ${TP_CUDA_ENABLED} CACHE BOOL "" FORCE)
set(TP_BUILD_LIBUV OFF CACHE BOOL "" FORCE)
set(TP_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(TP_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(TP_BUILD_MISC OFF CACHE BOOL "" FORCE)
set(TP_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(TP_STATIC_OR_SHARED STATIC CACHE STRING "" FORCE)
add_subdirectory(
    third_party/tensorpipe
    third_party/tensorpipe-build
    EXCLUDE_FROM_ALL)
set_target_properties(tensorpipe PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(TARGET tensorpipe_cuda)
    set_target_properties(tensorpipe_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
endif()

# Process-group library.  Gloo sources join only when the transport was
# built; the store layer and MPI backend stay available otherwise.
set(TP_DISTRIBUTED_SOURCES
    src/distributed/mpi/ProcessGroupMPI.cpp
    src/distributed/store/FileStore.cpp
    src/distributed/store/HashStore.cpp
    src/distributed/store/PrefixStore.cpp
    src/distributed/store/Store.cpp
    src/distributed/store/TCPStore.cpp
)
set(TP_DISTRIBUTED_LINK p10 pybind11::pybind11)
set(TP_DISTRIBUTED_DEFINES)
if(TP_USE_GLOO)
    list(APPEND TP_DISTRIBUTED_SOURCES src/distributed/gloo/ProcessGroupGloo.cpp)
    list(APPEND TP_DISTRIBUTED_LINK gloo)
    list(APPEND TP_DISTRIBUTED_DEFINES USE_P10D_GLOO)
endif()
add_library(tp_distributed STATIC ${TP_DISTRIBUTED_SOURCES})
target_include_directories(tp_distributed
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/distributed"
           "${CMAKE_CURRENT_SOURCE_DIR}/src/distributed/autograd"
           "${CMAKE_CURRENT_SOURCE_DIR}/p10/include"
           "${CMAKE_CURRENT_SOURCE_DIR}/tpx/include"
           "${CMAKE_CURRENT_SOURCE_DIR}/third_party/gloo")
target_link_libraries(tp_distributed PUBLIC ${TP_DISTRIBUTED_LINK})
target_compile_definitions(tp_distributed PUBLIC ${TP_DISTRIBUTED_DEFINES})

if(TP_USE_MPI)
    find_package(MPI COMPONENTS CXX QUIET)
    if(MPI_CXX_FOUND)
        target_link_libraries(tp_distributed PUBLIC MPI::MPI_CXX)
        target_compile_definitions(tp_distributed PUBLIC USE_P10D_MPI)
        message(STATUS "MPI backend enabled (${MPI_CXX_LIBRARIES})")
    else()
        message(STATUS "MPI runtime not found; the MPI backend stays disabled")
    endif()
endif()

# Python Bindings
set(TP_C_SOURCES
    src/bindings/python/init.cpp
    src/bindings/python/DType.cpp
    src/bindings/python/Scalar.cpp
    src/bindings/python/SymInt.cpp
    src/bindings/python/Device.cpp
    src/bindings/python/Size.cpp
    src/bindings/python/Generator.cpp
    src/bindings/python/Storage.cpp
    src/bindings/python/TensorNumpy.cpp
    src/bindings/python/Autograd.cpp
    src/bindings/python/autocast_mode.cpp
    src/bindings/python/Transforms.cpp
    src/bindings/python/Ops.cpp
    src/bindings/python/Dispatch.cpp
    src/bindings/python/Tensor.cpp
    src/bindings/python/Stax.cpp
    src/bindings/python/Parallel.cpp
    src/bindings/python/Distributed.cpp
    src/bindings/python/DistributedAutograd.cpp
    src/distributed/bindings.cpp
    src/bindings/python/Futures.cpp
    src/bindings/python/Rpc.cpp
    src/distributed/rpc/agent_utils.cpp
    src/distributed/rpc/future.cpp
    src/distributed/rpc/message.cpp
    src/distributed/rpc/python_call.cpp
    src/distributed/rpc/python_functions.cpp
    src/distributed/rpc/python_remote_call.cpp
    src/distributed/rpc/python_resp.cpp
    src/distributed/rpc/python_rpc_handler.cpp
    src/distributed/rpc/py_rref.cpp
    src/distributed/rpc/rpc_agent.cpp
    src/distributed/rpc/request_callback.cpp
    src/distributed/rpc/request_callback_impl.cpp
    src/distributed/rpc/request_callback_no_python.cpp
    src/distributed/rpc/rpc_runtime.cpp
    src/distributed/rpc/rref_context.cpp
    src/distributed/rpc/rref_impl.cpp
    src/distributed/rpc/rref_proto.cpp
    src/distributed/rpc/types.cpp
    src/distributed/rpc/unpickled_python_call.cpp
    src/distributed/rpc/unpickled_python_remote_call.cpp
    src/distributed/rpc/utils.cpp
    src/distributed/autograd/autograd.cpp
    src/distributed/autograd/context/container.cpp
    src/distributed/autograd/context/context.cpp
    src/distributed/autograd/engine/dist_engine.cpp
    src/distributed/autograd/functions/recvrpc_backward.cpp
    src/distributed/autograd/functions/sendrpc_backward.cpp
    src/distributed/autograd/rpc_messages/autograd_metadata.cpp
    src/distributed/autograd/rpc_messages/cleanup_autograd_context_req.cpp
    src/distributed/autograd/rpc_messages/cleanup_autograd_context_resp.cpp
    src/distributed/autograd/rpc_messages/propagate_gradients_req.cpp
    src/distributed/autograd/rpc_messages/propagate_gradients_resp.cpp
    src/distributed/autograd/rpc_messages/rpc_with_autograd.cpp
    src/distributed/autograd/rpc_messages/rref_backward_req.cpp
    src/distributed/autograd/rpc_messages/rref_backward_resp.cpp
    src/distributed/autograd/utils.cpp
    src/distributed/rpc/metrics/registry.cpp
    src/distributed/rpc/profiler/remote_profiler_manager.cpp
    src/distributed/rpc/profiler/server_process_global_profiler.cpp
    src/bindings/python/CudaGraph.cpp
     src/bindings/python/FileCheck.cpp
     src/Version.cpp
)
if(TP_USE_TENSORPIPE)
    # The tensorpipe transport TUs require its headers; they only join the
    # extension where the transport was built.
    list(APPEND TP_C_SOURCES
        src/distributed/rpc/tensorpipe_agent.cpp
        src/distributed/rpc/tensorpipe_backend.cpp
        src/distributed/rpc/tensorpipe_utils.cpp)
endif()
if(USE_ROCM)
    # Swap every binding TU over to its hipify-staged copy where one exists
    # (raw CUDA runtime calls in the bindings need the HIP translation too).
    tp_map_sources(TP_C_SOURCES ${TP_C_SOURCES})
endif()
pybind11_add_module(_C ${TP_C_SOURCES})
set_target_properties(_C PROPERTIES UNITY_BUILD ON)
# The NumPy C-API bootstrap headers (numpy_stub.h) need the NumPy include
# directory; locate it through the interpreter that owns the extension.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import numpy; print(numpy.get_include())"
    RESULT_VARIABLE TP_NUMPY_PROBE_RESULT
    OUTPUT_VARIABLE TP_NUMPY_INCLUDE
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
if(TP_NUMPY_PROBE_RESULT EQUAL 0 AND TP_NUMPY_INCLUDE)
    target_include_directories(_C PRIVATE "${TP_NUMPY_INCLUDE}")
endif()
# Transport selection follows the dependent options above; the fallback
# branch keeps the wheel buildable where a transport is unavailable.
if(TARGET tp_python)
    target_link_libraries(_C PRIVATE p10 tpx stax tp_python tp_distributed)
else()
    target_link_libraries(_C PRIVATE p10 tpx stax tp_distributed)
endif()
if(TP_USE_TENSORPIPE)
    target_link_libraries(_C PRIVATE tensorpipe)
endif()

if(TP_CUDA_ENABLED AND TARGET tensorpipe_cuda)
    target_link_libraries(_C PRIVATE tensorpipe_cuda)
endif()

if(TP_CUDA_ENABLED)
    target_compile_definitions(_C PRIVATE TENSORPLAY_TENSORPIPE_CUDA)
endif()

if(TP_CUDA_ENABLED)
    target_link_libraries(_C PRIVATE CUDA::cudart)
endif()

# if(MSVC)
#     # Force link all symbols from static libraries to ensure kernels are registered
#     target_link_options(_C PRIVATE "/WHOLEARCHIVE:p10" "/WHOLEARCHIVE:tpx" "/WHOLEARCHIVE:stax")
# endif()

# target_compile_definitions(_C PRIVATE TP_STATIC_BUILD)
# Distributed.cpp shares the NCCL stub/context declaration with p10.  Keep
# this include path explicit: p10's include directory is public, but the
# context header intentionally lives beside its CUDA implementation and is
# not part of the public tensor API.
target_include_directories(_C PRIVATE
    ${GEN_BASE_DIR}
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/python
    ${CMAKE_CURRENT_SOURCE_DIR}/p10/src/backend/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/p10/include
)

if(USE_ROCM)
    # Staged headers must win over the repo originals, hence BEFORE.
    target_include_directories(_C BEFORE PRIVATE
        ${TP_HIP_STAGING}/p10/include
        ${TP_HIP_STAGING}/src
    )
    target_include_directories(_C PRIVATE ${TP_HIP_INCLUDE_DIRS})
    target_link_libraries(_C PRIVATE ${TP_HIP_LIBRARIES})
    target_compile_options(_C PRIVATE ${TP_HIP_CXX_FLAGS})
    # Only the translated binding TUs activate the GPU paths; the rest of
    # the module (and every vendored third-party TU it absorbs in the unity
    # build) keeps CPU semantics, matching a plain CPU configuration.
    foreach(_src IN LISTS TP_C_SOURCES)
        if(_src MATCHES "^${TP_HIP_STAGING}")
            if(USE_CUDNN)
                set_source_files_properties(${_src} PROPERTIES
                    COMPILE_OPTIONS "-DUSE_CUDA;-DUSE_CUDNN;-DUSE_RCCL")
            else()
                set_source_files_properties(${_src} PROPERTIES
                    COMPILE_OPTIONS "-DUSE_CUDA;-DUSE_RCCL")
            endif()
        endif()
    endforeach()
endif()

if(NOT DEFINED TP_PACKAGE_DIR)
    set(TP_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tensorplay")
endif()
set(TP_LIB_DIR "${TP_PACKAGE_DIR}/lib")
file(MAKE_DIRECTORY "${TP_LIB_DIR}")

set_target_properties(_C PROPERTIES
    OUTPUT_NAME "__init__"
    RUNTIME_OUTPUT_DIRECTORY "${TP_PACKAGE_DIR}/_C"
    LIBRARY_OUTPUT_DIRECTORY "${TP_PACKAGE_DIR}/_C"
    ARCHIVE_OUTPUT_DIRECTORY "${TP_PACKAGE_DIR}/_C"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${TP_PACKAGE_DIR}/_C"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${TP_PACKAGE_DIR}/_C"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${TP_PACKAGE_DIR}/_C"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${TP_PACKAGE_DIR}/_C"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${TP_PACKAGE_DIR}/_C"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${TP_PACKAGE_DIR}/_C"
)

if(NOT WIN32)
    set_target_properties(tp_python PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${TP_LIB_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${TP_LIB_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${TP_LIB_DIR}")
endif()

# RPATH: the extension lives in tensorplay/_C while its dependencies
# (libp10.so, libtpx.so, libstax.so) live in tensorplay/lib. Without an
# explicit relative RPATH, importing tensorplay from a wheel/venv fails
# with "cannot open shared object file". Windows ignores RPATH (DLL
# search is handled by tensorplay/__init__.py via os.add_dll_directory).
if(NOT WIN32)
    set_target_properties(p10 tpx stax PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_RPATH "$ORIGIN"
    )
    set_target_properties(tp_python PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_RPATH "$ORIGIN"
    )
    set_target_properties(_C PROPERTIES
        INSTALL_RPATH "$ORIGIN/../lib"
        BUILD_RPATH "$ORIGIN/../lib"
    )
endif()

# Installation
set(TP_INSTALL_TARGETS p10 tpx stax)
if(TARGET tp_python)
    list(APPEND TP_INSTALL_TARGETS tp_python)
endif()
install(TARGETS ${TP_INSTALL_TARGETS}
    LIBRARY DESTINATION tensorplay/lib
    RUNTIME DESTINATION tensorplay/lib
)

install(TARGETS _C
    LIBRARY DESTINATION tensorplay/_C
    RUNTIME DESTINATION tensorplay/_C
)

# version.py is generated at build time (gitignored, so scikit-build-core's
# wheel.packages pass cannot pick it up); it must be installed explicitly or
# the shipped wheel cannot `import tensorplay`.
install(FILES "${TP_VERSION_PY}"
    DESTINATION tensorplay
)

# Install generated stub files (.pyi) to tensorplay/_C
# Generated by the codegen custom target into ${CMAKE_SOURCE_DIR}/tensorplay/_C
# (or ${CMAKE_CURRENT_BINARY_DIR}/tensorplay/_C, whichever exists)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/tensorplay/_C/"
    DESTINATION tensorplay/_C
    FILES_MATCHING PATTERN "*.pyi"
)
install(FILES "${CMAKE_SOURCE_DIR}/tensorplay/functional.pyi"
    DESTINATION tensorplay
)

# Type stubs for the _C extension are generated into the package source tree
# so IDEs and type checkers pick them up both in-tree and from the install
# rule above.  Like the other codegen outputs, the stub is a dedicated custom
# command: its outputs, template, schema, binding source, and documentation
# module are all tracked as dependencies, and the stub target is a dependency
# of the extension target so a stale stub cannot survive a build.
file(GLOB_RECURSE tp_pyi_python CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/*.py")
add_custom_command(
    OUTPUT
    "${TP_PACKAGE_DIR}/_C/__init__.pyi"
    "${TP_PACKAGE_DIR}/_C/_VariableFunctions.pyi"
    "${TP_PACKAGE_DIR}/functional.pyi"
    COMMAND
    ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
        ${Python_EXECUTABLE} -m tools.codegen.main
        --yaml config/native_functions.yaml
        --out_dir ${GEN_INCLUDE_DIR}
        --targets PyiStub
        --pyi_template "${TP_PACKAGE_DIR}"
        --pyi_out "${TP_PACKAGE_DIR}"
    DEPENDS
        "${TP_PACKAGE_DIR}/_C/__init__.pyi.in"
        "${TP_PACKAGE_DIR}/_C/_VariableFunctions.pyi.in"
        "${TP_PACKAGE_DIR}/functional.pyi.in"
        "${TP_PACKAGE_DIR}/functional.py"
        "${TP_PACKAGE_DIR}/_docs.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/config/native_functions.yaml"
        ${TP_CODEGEN_TAGS}
        "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/python/DType.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/gen_pyi.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/api_types.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/model.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/main.py"
        ${tp_pyi_python}
        ${tp_codegen_reference}
    WORKING_DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Generating type stubs from native_functions.yaml"
)
add_custom_target(tensorplay_stubs DEPENDS
    "${TP_PACKAGE_DIR}/_C/__init__.pyi"
    "${TP_PACKAGE_DIR}/_C/_VariableFunctions.pyi"
    "${TP_PACKAGE_DIR}/functional.pyi"
)
add_dependencies(_C tensorplay_stubs)

# Add custom target for building python extension
add_custom_target(build_python
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target _C
    COMMENT "Building Python extension"
)


# Testing support
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
        add_subdirectory(test)
    endif()
endif()
