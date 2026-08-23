# Helper for building custom operators.
#
# Layout mirrors upstream torch: the tensorplay::python_c conversion surface
# (CPythonBridge.cpp) plays the role of torch_python and lives in its own
# shared library; every op module is a plain Python_add_library(MODULE
# WITH_SOABI) linking it plus p10 (the libtorch analog).  Generated bindings
# are raw CPython (PyMethodDef/METH_FASTCALL) with no pybind11 in the
# extension itself.

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
    set(TP_REPO_DIR "${TP_CMAKE_DIR}/..")
    set(TP_GEN_TOOL "${TP_REPO_DIR}/tools/codegen/tensorplaygen.py")

    set(GEN_HEADER "${ARG_OUT_DIR}/OpsGenerated.h")
    set(GEN_BINDING "${ARG_OUT_DIR}/OpsBinding.cpp")

    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

    # Conversion surface shared across op modules -- one copy keeps the
    # pybind11 caster substrate (used internally by the bridge) registered
    # from a single module boundary, like torch_python.
    if(NOT TARGET tp_python)
        find_package(pybind11 CONFIG REQUIRED)
        add_library(tp_python SHARED "${TP_REPO_DIR}/src/bindings/python/CPythonBridge.cpp")
        target_include_directories(tp_python PUBLIC
            "${TP_REPO_DIR}/p10/include"
            "${TP_REPO_DIR}/src/bindings/python"
        )
        target_link_libraries(tp_python PUBLIC p10 pybind11::pybind11)
    endif()

    add_custom_command(
        OUTPUT ${GEN_HEADER} ${GEN_BINDING}
        COMMAND "${Python_EXECUTABLE}" ${TP_GEN_TOOL} --yaml ${ARG_YAML_ABS} --out_dir ${ARG_OUT_DIR} --module_name ${ARG_NAME}
        DEPENDS ${TP_GEN_TOOL} ${ARG_YAML_ABS}
        COMMENT "Generating code for ${ARG_NAME}"
    )

    Python_add_library(${ARG_NAME} MODULE WITH_SOABI ${ARG_SOURCES} ${GEN_BINDING})
    target_include_directories(${ARG_NAME} PRIVATE ${ARG_OUT_DIR})
    target_link_libraries(${ARG_NAME} PRIVATE tp_python)

    # Standard compilation options
    if(MSVC)
        target_compile_options(${ARG_NAME} PRIVATE /std:c++20 /EHsc /wd4251 /wd4275)
    else()
        target_compile_options(${ARG_NAME} PRIVATE -std=c++20 -O3 -fPIC)
    endif()

endfunction()
