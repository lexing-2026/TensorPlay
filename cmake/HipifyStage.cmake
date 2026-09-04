# Helpers for compiling the hipify-staged AMD GPU sources.
#
# The build-time translation tool renames transformed files following fixed
# rules (directory component cuda/CUDA -> hip/HIP, file stem cuda/CUDA ->
# hip/HIP, extension .cu -> .hip, and an explicit "_hip" suffix when neither
# the directory nor the stem changed).  tp_hipify_path reconstructs those rules so
# the build can reference the staged copy of any source file, and
# tp_map_sources swaps a normal source list over to the staged tree wherever
# a translated copy exists.

function(tp_hipify_path out_var path)
  cmake_path(GET path PARENT_PATH _dir)
  cmake_path(GET path STEM _stem)
  cmake_path(GET path EXTENSION _ext)
  cmake_path(GET path FILENAME _fname)

  set(_orig_dir "${_dir}")
  string(REPLACE "cuda" "hip" _dir "${_dir}")
  string(REPLACE "CUDA" "HIP" _dir "${_dir}")
  set(_orig_stem "${_stem}")
  string(REPLACE "cuda" "hip" _stem "${_stem}")
  string(REPLACE "CUDA" "HIP" _stem "${_stem}")

  # cmake_path(EXTENSION) keeps the leading dot (".cu"); the rebuilt
  # filename below joins stem and extension with a dot, so the replacement
  # value must be dot-free.
  if(_ext STREQUAL ".cu")
    set(_ext "hip")
  endif()

  if(_dir STREQUAL _orig_dir AND _stem STREQUAL _orig_stem)
    string(APPEND _stem "_hip")
  endif()

  set(${out_var} "${_dir}/${_stem}.${_ext}" PARENT_SCOPE)
endfunction()

function(tp_map_sources out_var)
  set(_mapped)
  foreach(_src IN LISTS ARGN)
    tp_hipify_path(_hip "${_src}")
    # Sources may be relative to the repo root or to a staged subdir
    # (p10/); accept whichever layout the staging tree actually has.
    if(EXISTS "${TP_HIP_STAGING}/${_hip}")
      list(APPEND _mapped "${TP_HIP_STAGING}/${_hip}")
    elseif(EXISTS "${TP_HIP_STAGING}/p10/${_hip}")
      list(APPEND _mapped "${TP_HIP_STAGING}/p10/${_hip}")
    elseif(EXISTS "${TP_HIP_STAGING}/${_src}")
      list(APPEND _mapped "${TP_HIP_STAGING}/${_src}")
    elseif(EXISTS "${TP_HIP_STAGING}/p10/${_src}")
      list(APPEND _mapped "${TP_HIP_STAGING}/p10/${_src}")
    else()
      list(APPEND _mapped "${CMAKE_CURRENT_SOURCE_DIR}/${_src}")
    endif()
  endforeach()
  set(${out_var} "${_mapped}" PARENT_SCOPE)
endfunction()
