# Helper for building custom operators

function(add_tensorplay_op)
    set(options)
    set(oneValueArgs NAME YAML OUT_DIR)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    if(NOT ARG_NAME)
        message(FATAL_ERROR "NAME is required")
    endif()
    if(NOT ARG_YAML)
        message(FATAL_ERROR "YAML is required")
    endif()
    
    # Ensure YAML path is absolute
    get_filename_component(ARG_YAML_ABS "${ARG_YAML}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    
    if(NOT ARG_OUT_DIR)
        set(ARG_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    endif()
    
    file(MAKE_DIRECTORY ${ARG_OUT_DIR})
    
    # Locate tensorplaygen.py
    # Assuming this file is in cmake/ and tools/ is at ../tools/
    get_filename_component(TP_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
    set(TP_GEN_TOOL "${TP_CMAKE_DIR}/../tools/codegen/tensorplaygen.py")
    
    set(GEN_HEADER "${ARG_OUT_DIR}/OpsGenerated.h")
    set(GEN_BINDING "${ARG_OUT_DIR}/OpsBinding.cpp")
    
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    
    add_custom_command(
        OUTPUT ${GEN_HEADER} ${GEN_BINDING}
        COMMAND "${Python_EXECUTABLE}" ${TP_GEN_TOOL} --yaml ${ARG_YAML_ABS} --out_dir ${ARG_OUT_DIR} --module_name ${ARG_NAME}
        DEPENDS ${TP_GEN_TOOL} ${ARG_YAML_ABS}
        COMMENT "Generating code for ${ARG_NAME}"
    )
    
    # Nanobind module
    nanobind_add_module(${ARG_NAME} ${ARG_SOURCES} ${GEN_BINDING})
    target_include_directories(${ARG_NAME} PRIVATE ${ARG_OUT_DIR})
    
    # Standard compilation options
    if(MSVC)
        target_compile_options(${ARG_NAME} PRIVATE /std:c++20 /EHsc /wd4251 /wd4275)
    else()
        target_compile_options(${ARG_NAME} PRIVATE -std=c++20 -O3 -fPIC)
    endif()
    
endfunction()
