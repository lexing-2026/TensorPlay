# NCCL backend for tensorplay.distributed.
#
# USE_SYSTEM_NCCL=ON  -> find_package(NCCL) against a system install
#                        (NCCL_ROOT / NCCL_LIB_DIR / NCCL_INCLUDE_DIR hints).
# USE_SYSTEM_NCCL=OFF -> build the vendored tree at third_party/nccl with its
#                        own makefiles, producing a static archive.  When the
#                        checkout is absent the caller downgrades gracefully.
#
# On success this defines:
#   tp::nccl           interface target carrying include dirs + libraries
#   NCCL_LIBRARIES     library (or target) to link
#   NCCL_INCLUDE_DIRS  header directory

if(NOT __TP_NCCL_INCLUDED)
    set(__TP_NCCL_INCLUDED TRUE)

    if(USE_SYSTEM_NCCL)
        find_package(NCCL QUIET)
        if(NCCL_FOUND)
            add_library(tp::nccl INTERFACE IMPORTED)
            target_link_libraries(tp::nccl INTERFACE ${NCCL_LIBRARIES})
            target_include_directories(tp::nccl INTERFACE ${NCCL_INCLUDE_DIRS})
        endif()
    else()
        if(NOT EXISTS "${PROJECT_SOURCE_DIR}/third_party/nccl")
            message(STATUS
                "Vendored NCCL checkout not found at third_party/nccl; "
                "falling back to a system search.")
            find_package(NCCL QUIET)
            if(NCCL_FOUND)
                add_library(tp::nccl INTERFACE IMPORTED)
                target_link_libraries(tp::nccl INTERFACE ${NCCL_LIBRARIES})
                target_include_directories(tp::nccl INTERFACE ${NCCL_INCLUDE_DIRS})
            endif()
            return()
        endif()

        # Gencode flags for the bundled nvcc invocation, derived from
        # CMAKE_CUDA_ARCHITECTURES (e.g. "70;75" -> "-gencode=arch=compute_70,code=sm_70 ...").
        set(NVCC_GENCODE "")
        foreach(_arch ${CMAKE_CUDA_ARCHITECTURES})
            string(APPEND NVCC_GENCODE
                "-gencode=arch=compute_${_arch},code=sm_${_arch} ")
        endforeach()
        string(STRIP "${NVCC_GENCODE}" NVCC_GENCODE)

        if(DEFINED ENV{MAX_JOBS})
            set(MAX_JOBS "$ENV{MAX_JOBS}")
        else()
            include(ProcessorCount)
            ProcessorCount(NUM_HARDWARE_THREADS)
            math(EXPR MAX_JOBS "${NUM_HARDWARE_THREADS} / 2")
            if(MAX_JOBS LESS 2)
                set(MAX_JOBS 2)
            endif()
        endif()

        if("${CMAKE_GENERATOR}" MATCHES "Make")
            # Recursive make with the jobserver plus a load limit to keep the
            # parallel nccl build from oversubscribing the machine.
            set(MAKE_COMMAND "$(MAKE)" "-l${MAX_JOBS}")
        else()
            set(MAKE_COMMAND "make" "-j${MAX_JOBS}" "-l${MAX_JOBS}")
        endif()

        set(__NCCL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/nccl")
        ExternalProject_Add(nccl_external
            SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/nccl
            BUILD_IN_SOURCE 1
            CONFIGURE_COMMAND ""
            BUILD_COMMAND
                ${MAKE_COMMAND}
                "CXX=${CMAKE_CXX_COMPILER}"
                "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}"
                "NVCC=${CUDA_NVCC_EXECUTABLE}"
                "NVCC_GENCODE=${NVCC_GENCODE}"
                "BUILDDIR=${__NCCL_BUILD_DIR}"
                "VERBOSE=0"
                "DEBUG=0"
            BUILD_BYPRODUCTS "${__NCCL_BUILD_DIR}/lib/libnccl_static.a"
            INSTALL_COMMAND ""
        )

        set(NCCL_LIBRARIES "${__NCCL_BUILD_DIR}/lib/libnccl_static.a")
        set(NCCL_INCLUDE_DIRS "${__NCCL_BUILD_DIR}/include")
        set(NCCL_FOUND TRUE)

        add_library(tp::nccl INTERFACE IMPORTED)
        add_dependencies(tp::nccl nccl_external)
        target_link_libraries(tp::nccl INTERFACE ${NCCL_LIBRARIES})
        target_include_directories(tp::nccl INTERFACE ${NCCL_INCLUDE_DIRS})
        # nccl uses shm_open/shm_close, which requires librt on Linux.
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            target_link_libraries(tp::nccl INTERFACE rt)
        endif()
    endif()
endif()
