cmake_minimum_required(VERSION 3.18)
project(TensorPlay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Enable CUDA support" ON)

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
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
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
    add_compile_definitions(USE_CUDA)
else()
    message(STATUS "CUDA support disabled by user or not found.")
endif()

# Check for cuDNN
if(USE_CUDA)
    find_path(CUDNN_INCLUDE_DIR cudnn.h
        HINTS 
            ${CUDAToolkit_INCLUDE_DIRS} 
            $ENV{CUDNN_ROOT_DIR}/include 
            $ENV{CUDNN_INCLUDE_DIR}
            "C:/Program Files/NVIDIA/CUDNN/v9.17/include/13.1"
            "C:/Program Files/NVIDIA/CUDNN/v9.17/include"
            "C:/Program Files/NVIDIA/CUDNN/v9.x/include"
        DOC "Path to cuDNN include directory"
    )

    # Look for cudnn library (windows: cudnn.lib, linux: libcudnn.so)
    find_library(CUDNN_LIBRARY NAMES cudnn
        HINTS 
            ${CUDAToolkit_LIBRARY_DIR} 
            $ENV{CUDNN_ROOT_DIR}/lib/x64 
            $ENV{CUDNN_LIBRARY_DIR}
            "C:/Program Files/NVIDIA/CUDNN/v9.17/lib/13.1/x64"
            "C:/Program Files/NVIDIA/CUDNN/v9.17/lib/x64"
            "C:/Program Files/NVIDIA/CUDNN/v9.x/lib/x64"
        DOC "Path to cuDNN library"
    )

    if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
        message(WARNING "cuDNN not found in standard locations.")
    endif()

    # Install CUDA Runtime Libraries (Windows)
    if(WIN32 AND USE_CUDA)
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
                "curand64_*.dll"
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

    if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        set(USE_CUDNN ON)
        message(STATUS "Found cuDNN: ${CUDNN_INCLUDE_DIR}")
        add_compile_definitions(USE_CUDNN)
        
        # On Windows, we might need to copy DLLs if they are not in PATH
        if(WIN32)
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
                # Don't copy if already exists to avoid permission issues or redundancy
                # But for development it's safer to copy to build dir
                file(COPY ${DLL} DESTINATION ${CMAKE_BINARY_DIR}/bin) 
                file(COPY ${DLL} DESTINATION ${CMAKE_SOURCE_DIR}/tensorplay/lib)
                install(FILES ${DLL} DESTINATION tensorplay/lib)
            endforeach()
        endif()
    else()
        message(WARNING "cuDNN not found. CUDA backend will be built without cuDNN support.")
    endif()
else()
    set(USE_CUDNN OFF)
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
    add_compile_options(-std=c++20 -O3 -Wall -Wextra -Wpedantic)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(-fPIC)
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
set(GEN_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/generated")
set(GEN_INCLUDE_DIR "${GEN_BASE_DIR}/tensorplay/ops")
set(GEN_HEADER "${GEN_INCLUDE_DIR}/TensorGenerated.h")
set(GEN_CPP "${GEN_INCLUDE_DIR}/TensorGenerated.cpp")
set(GEN_BINDINGS_HEADER "${GEN_INCLUDE_DIR}/TensorBindingsGenerated.h")
set(GEN_AUTOGRAD_NODES_H "${GEN_INCLUDE_DIR}/AutogradNodesGenerated.h")
set(GEN_TPX_OPS_H "${GEN_INCLUDE_DIR}/TPXOpsGenerated.h")
set(GEN_TPX_OPS_CPP "${GEN_INCLUDE_DIR}/TPXOpsGenerated.cpp")

# Python Interface Stub Generation
set(TP_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tensorplay")

# Add a custom target to ensure code generation runs before library building
add_custom_target(generate_code DEPENDS ${GEN_HEADER} ${GEN_CPP} ${GEN_TPX_OPS_CPP})

include_directories(${GEN_BASE_DIR})

# Nanobind via pip installation
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

# --- Compute Backend Selection ---
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    # Apply OpenMP flags only to CXX files directly
    # For CUDA, pass them to the host compiler via -Xcompiler
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
    
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
endif()

option(USE_BLAS "Enable BLAS linear algebra acceleration" ON)
set(COMPUTE_LIBS "")
set(BLAS_COMPILE_DEFS "")
set(BLAS_COMPILE_OPTIONS "")

if(USE_BLAS)
    set(BLAS_PROVIDER "MKL" CACHE STRING "BLAS provider (MKL/OpenBLAS)")
    set_property(CACHE BLAS_PROVIDER PROPERTY STRINGS "MKL" "OpenBLAS")

    if(BLAS_PROVIDER STREQUAL "MKL")
        # 1. Try MKL Config mode (Best for OneAPI)
        set(MKL_THREADING "sequential")
        set(MKL_LINK "static")
        find_package(MKL CONFIG QUIET)
        
        if(MKL_FOUND)
            message(STATUS "Found MKL (Config): ${MKL_DIR}")
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
            # Fallback: Try finding BLAS/MKL via standard FindBLAS
            set(BLA_VENDOR "Intel10_64lp")
            set(BLA_STATIC ON) # Force static linking
            find_package(BLAS QUIET)
            if(BLAS_FOUND)
                message(STATUS "Found MKL via BLAS: ${BLAS_LIBRARIES}")
                list(APPEND COMPUTE_LIBS ${BLAS_LIBRARIES})
                list(APPEND BLAS_COMPILE_DEFS "USE_MKL")
                
                # Try to find headers if using generic FindBLAS
                find_path(MKL_INCLUDE_DIR mkl.h HINTS $ENV{MKL_ROOT}/include "${ONEAPI_ROOT}/mkl/latest/include")
                if(MKL_INCLUDE_DIR)
                    include_directories(SYSTEM ${MKL_INCLUDE_DIR})
                else()
                    # Assume standard path
                    include_directories(SYSTEM "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/include")
                endif()
            else()
                 # Try manual search
                 if(EXISTS "${ONEAPI_ROOT}/mkl/latest/lib/intel64/mkl_core.lib")
                     message(STATUS "Found MKL (Manual): ${ONEAPI_ROOT}/mkl/latest")
                     set(MKL_LIB_DIR "${ONEAPI_ROOT}/mkl/latest/lib/intel64")
                     set(MKL_INC_DIR "${ONEAPI_ROOT}/mkl/latest/include")
                     include_directories(SYSTEM ${MKL_INC_DIR})
                     list(APPEND COMPUTE_LIBS "${MKL_LIB_DIR}/mkl_intel_lp64.lib" "${MKL_LIB_DIR}/mkl_sequential.lib" "${MKL_LIB_DIR}/mkl_core.lib")
                     set(BLAS_FOUND TRUE)
                     list(APPEND BLAS_COMPILE_DEFS "USE_MKL")
                 endif()
            endif()
        endif()

    elseif(BLAS_PROVIDER STREQUAL "OpenBLAS")
        find_package(OpenBLAS CONFIG QUIET)
        if(OpenBLAS_FOUND)
             list(APPEND COMPUTE_LIBS OpenBLAS::OpenBLAS)
             set(BLAS_FOUND TRUE)
             list(APPEND BLAS_COMPILE_DEFS "USE_OPENBLAS" "USE_BLAS")
        else()
             find_package(PkgConfig QUIET)
             if(PKG_CONFIG_FOUND)
                pkg_check_modules(OpenBLAS_PC openblas QUIET)
                if(OpenBLAS_PC_FOUND)
                    list(APPEND COMPUTE_LIBS ${OpenBLAS_PC_LIBRARIES})
                    include_directories(SYSTEM ${OpenBLAS_PC_INCLUDE_DIRS})
                    set(BLAS_FOUND TRUE)
                    list(APPEND BLAS_COMPILE_DEFS "USE_OPENBLAS" "USE_BLAS")
                endif()
             endif()
        endif()
    endif()
    
    if(BLAS_FOUND)
        message(STATUS "✅ Successfully configured BLAS: ${BLAS_PROVIDER}")
    else()
        message(WARNING "Could not find ${BLAS_PROVIDER}.")
    endif()
endif()

# --- oneDNN Integration ---
option(USE_ONEDNN "Enable oneDNN for deep learning primitives" ON)

if(USE_ONEDNN)
    message(STATUS "Configuring oneDNN (Static)...")
    
    include(FetchContent)
    
    set(DNNL_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
    set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(DNNL_ENABLE_CONCURRENT_EXEC ON CACHE BOOL "" FORCE)
    set(DNNL_CPU_RUNTIME "OMP" CACHE STRING "" FORCE)
    set(DNNL_ENABLE_WORKLOAD "TRAINING" CACHE STRING "" FORCE)
    set(DNNL_ENABLE_PRIMITIVE "CONVOLUTION;MATMUL;REORDER" CACHE STRING "" FORCE)
    set(DNNL_ENABLE_PRIMITIVE_CACHE ON CACHE BOOL "" FORCE)
    set(ONEDNN_BUILD_GRAPH OFF CACHE BOOL "" FORCE) # Graph API incompatible with primitive selection (Use ONEDNN_ prefix)
    set(DNNL_BUILD_GRAPH OFF CACHE BOOL "" FORCE) # Try legacy name too
    set(DNNL_ARCH_OPT_FLAGS "" CACHE STRING "" FORCE) # Let user flags decide
    # OneDNN v3+ might use these
    set(DNNL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    
    FetchContent_Declare(
        onednn
        URL "${CMAKE_CURRENT_SOURCE_DIR}/third_party/v3.4.1.zip"
    )
    
    FetchContent_MakeAvailable(onednn)
    
    if(TARGET dnnl)
        message(STATUS "✅ Successfully configured oneDNN (Static)")
        list(APPEND COMPUTE_LIBS dnnl)
        set(ONEDNN_FOUND TRUE)
        add_compile_definitions(USE_ONEDNN)
        
        # OneDNN headers
        get_target_property(DNNL_INCLUDE_DIRS dnnl INTERFACE_INCLUDE_DIRECTORIES)
        include_directories(${DNNL_INCLUDE_DIRS})

        # Disable OneDNN installation to prevent polluting site-packages
        # We only need the static library linked into p10
        set_target_properties(dnnl PROPERTIES EXCLUDE_FROM_ALL ON)
    else()
        message(FATAL_ERROR "Failed to build oneDNN")
    endif()
endif()

add_compile_definitions(${BLAS_COMPILE_DEFS})

if(MSVC)
  list(APPEND BLAS_COMPILE_OPTIONS "/O2" "/arch:AVX2")
else()
  list(APPEND BLAS_COMPILE_OPTIONS "-march=native" "-O3")
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
    set(TP_ONEDNN_INFO "Intel(R) MKL-DNN v${dnnl_VERSION}")
else()
    set(TP_ONEDNN_INFO "N/A")
endif()

if(OpenMP_CXX_FOUND)
    set(TP_OPENMP_INFO "OpenMP ${OpenMP_CXX_VERSION}")
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

# Build Settings String
set(TP_BUILD_SETTINGS_LIST)

if(BLAS_PROVIDER STREQUAL "MKL")
    set(BLAS_INFO_VAL "mkl")
else()
    set(BLAS_INFO_VAL "${BLAS_PROVIDER}")
endif()

list(APPEND TP_BUILD_SETTINGS_LIST "BLAS_INFO=${BLAS_INFO_VAL}")
list(APPEND TP_BUILD_SETTINGS_LIST "BUILD_TYPE=${TP_BUILD_TYPE}")

# Sanitize CXX_COMPILER (remove quotes, use forward slashes)
string(REPLACE "\"" "" CXX_COMPILER_NO_QUOTES "${CMAKE_CXX_COMPILER}")
file(TO_CMAKE_PATH "${CXX_COMPILER_NO_QUOTES}" CXX_COMPILER_FWD)
list(APPEND TP_BUILD_SETTINGS_LIST "CXX_COMPILER=${CXX_COMPILER_FWD}")

list(APPEND TP_BUILD_SETTINGS_LIST "CXX_FLAGS=${CMAKE_CXX_FLAGS}")
list(APPEND TP_BUILD_SETTINGS_LIST "LAPACK_INFO=${BLAS_INFO_VAL}")
list(APPEND TP_BUILD_SETTINGS_LIST "PERF_WITH_AVX=1")
list(APPEND TP_BUILD_SETTINGS_LIST "PERF_WITH_AVX2=1")
list(APPEND TP_BUILD_SETTINGS_LIST "TENSORPLAY_VERSION=${TENSORPLAY_VERSION}")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_OPENMP=${OpenMP_CXX_FOUND}")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_MKL=${BLAS_FOUND}")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_MKLDNN=${ONEDNN_FOUND}")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_CUDA=${USE_CUDA}")
list(APPEND TP_BUILD_SETTINGS_LIST "USE_CUDNN=${USE_CUDNN}")

string(REPLACE ";" ", " TP_BUILD_SETTINGS_STR "${TP_BUILD_SETTINGS_LIST}")
# Escape backslashes for C++ string literal
string(REPLACE "\\" "\\\\" TP_BUILD_SETTINGS_STR "${TP_BUILD_SETTINGS_STR}")
# Escape quotes for C++ string literal
string(REPLACE "\"" "\\\"" TP_BUILD_SETTINGS_STR "${TP_BUILD_SETTINGS_STR}")

set(CONFIG_CONTENT "
#pragma once
#include <string>
#include <map>

namespace tensorplay {
    inline std::string show_config() {
        std::string s = \"TensorPlay built with:\\n\";
        s += \"   - C++ Version: ${TP_CXX_VERSION}\\n\";
        s += \"   - ${TP_COMPILER_INFO}\\n\";
        s += \"   - ${TP_MKL_INFO}\\n\";
        s += \"   - ${TP_ONEDNN_INFO}\\n\";
        s += \"   - ${TP_OPENMP_INFO}\\n\";
        s += \"   - ${TP_CUDA_INFO}\\n\";
        s += \"   - ${TP_CUDNN_INFO}\\n\";
        s += \"   - ${TP_LAPACK_INFO}\\n\";
        s += \"   - CPU capability usage: ${TP_CPU_CAPABILITY}\\n\";
        s += \"   - Build settings: ${TP_BUILD_SETTINGS_STR}\\n\";
        return s;
    }

    inline std::string _cxx_flags() {
        return \"${ESCAPED_CXX_FLAGS}\";
    }

    inline std::string _parallel_info() {
        return \"${TP_OPENMP_INFO}\";
    }

    inline std::map<std::string, std::string> get_build_info() {
        std::map<std::string, std::string> info;
        info[\"CXX_VERSION\"] = \"${TP_CXX_VERSION}\";
        info[\"COMPILER_INFO\"] = \"${TP_COMPILER_INFO}\";
        info[\"MKL_INFO\"] = \"${TP_MKL_INFO}\";
        info[\"ONEDNN_INFO\"] = \"${TP_ONEDNN_INFO}\";
        info[\"OPENMP_INFO\"] = \"${TP_OPENMP_INFO}\";
        info[\"CUDA_INFO\"] = \"${TP_CUDA_INFO}\";
        info[\"CUDNN_INFO\"] = \"${TP_CUDNN_INFO}\";
        info[\"LAPACK_INFO\"] = \"${TP_LAPACK_INFO}\";
        info[\"CPU_CAPABILITY\"] = \"${TP_CPU_CAPABILITY}\";
        info[\"BUILD_SETTINGS\"] = \"${TP_BUILD_SETTINGS_STR}\";
        return info;
    }
}
")
file(WRITE "${GEN_BASE_DIR}/tensorplay/ops/Config.h" "${CONFIG_CONTENT}")

# Code generation
add_custom_command(
            OUTPUT ${GEN_HEADER} ${GEN_CPP} ${GEN_BINDINGS_HEADER} 
                   ${GEN_AUTOGRAD_NODES_H} ${GEN_TPX_OPS_H} ${GEN_TPX_OPS_CPP}
            COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/gen.py 
                    --yaml ${CMAKE_CURRENT_SOURCE_DIR}/config/native_functions.yaml 
                    --derivatives ${CMAKE_CURRENT_SOURCE_DIR}/config/derivatives.yaml
                    --out_dir ${GEN_INCLUDE_DIR}
                    --pkg_out ${CMAKE_CURRENT_SOURCE_DIR}/tensorplay
            DEPENDS 
                ${CMAKE_CURRENT_SOURCE_DIR}/config/native_functions.yaml
                ${CMAKE_CURRENT_SOURCE_DIR}/config/derivatives.yaml
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/gen.py
            COMMENT "Generating Tensor methods, bindings and pyi stubs from native_functions.yaml"
        )

set_source_files_properties(
    ${GEN_HEADER} ${GEN_CPP} ${GEN_BINDINGS_HEADER} 
    ${GEN_AUTOGRAD_NODES_H} ${GEN_TPX_OPS_H} ${GEN_TPX_OPS_CPP}
    PROPERTIES GENERATED TRUE
)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Add p10
add_subdirectory(p10)

# Add tpx
add_subdirectory(tpx)

# Add stax (Optional)
add_subdirectory(stax)

# Python Bindings
nanobind_add_module(_C 
    src/bindings/python/init.cpp
    src/bindings/python/DType.cpp
    src/bindings/python/Scalar.cpp
    src/bindings/python/Device.cpp
    src/bindings/python/Size.cpp
    src/bindings/python/Generator.cpp
    src/bindings/python/Autograd.cpp
    src/bindings/python/Ops.cpp
    src/bindings/python/Tensor.cpp
    src/bindings/python/Stax.cpp
)
set_target_properties(_C PROPERTIES UNITY_BUILD ON)
target_link_libraries(_C PRIVATE p10 tpx stax)

if(USE_CUDA)
    target_link_libraries(_C PRIVATE CUDA::cudart)
endif()

# if(MSVC)
#     # Force link all symbols from static libraries to ensure kernels are registered
#     target_link_options(_C PRIVATE "/WHOLEARCHIVE:p10" "/WHOLEARCHIVE:tpx" "/WHOLEARCHIVE:stax")
# endif()

# target_compile_definitions(_C PRIVATE TP_STATIC_BUILD)
target_include_directories(_C PRIVATE ${GEN_BASE_DIR} include)

set(TP_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tensorplay")
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

# Installation
install(TARGETS p10 tpx stax
    LIBRARY DESTINATION tensorplay/lib
    RUNTIME DESTINATION tensorplay/lib
)

install(TARGETS _C
    LIBRARY DESTINATION tensorplay/_C
    RUNTIME DESTINATION tensorplay/_C
)

# Install generated stub files (.pyi) to tensorplay/_C
# Note: These are generated during build (by rebuild.py pre-step or custom command)
# We assume they are available in ${CMAKE_SOURCE_DIR}/tensorplay/_C or ${CMAKE_CURRENT_BINARY_DIR}/tensorplay/_C
# But rebuild.py puts them in ${CMAKE_SOURCE_DIR}/tensorplay/_C
install(DIRECTORY "${CMAKE_SOURCE_DIR}/tensorplay/_C/"
    DESTINATION tensorplay/_C
    FILES_MATCHING PATTERN "*.pyi"
)

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
