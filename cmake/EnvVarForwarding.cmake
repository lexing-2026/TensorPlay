# Forward environment variables to CMake variables.
#
# Forwarding rule: an environment variable reaches CMake (as a cache variable of
# the same name) if it starts with BUILD_, USE_, or CMAKE_, or appears in the
# _ENV_PASSTHROUGH list below. Anything else is not forwarded -- set it as a
# CMake option with -D / cmake.define instead.
#
# The variable tables below are scaled to TensorPlay's knobs.
#
# Everyday knobs:
#   USE_CUDA=0               disable the CUDA build
#   USE_CUDNN=0              disable cuDNN (consumed below via USE_CUDA block)
#   USE_BLAS=0               disable BLAS linear algebra acceleration
#   USE_ONEDNN=0             disable oneDNN primitives
#   BUILD_TESTS=1            enable the test build
#   BUILD_SHARED_LIBS=0      build static libraries where supported
#   CMAKE_CUDA_ARCHITECTURES e.g. "61" or "70;75;86"
#   DEBUG=1 / REL_WITH_DEB_INFO=1  mapped to the CMake build type by
#                            [[tool.scikit-build.overrides]]-style handling in
#                            scikit-build-core; CMAKE_BUILD_TYPE also honored
#   MAX_JOBS                 compile parallelism; aliased to
#                            CMAKE_BUILD_PARALLEL_LEVEL by
#                            [tool.scikit-build.env] in pyproject.toml
#
# Handled outside this module (NOT forwarded here):
#   TENSORPLAY_BUILD_VERSION / TENSORPLAY_BUILD_NUMBER  wheel version;
#                            consumed by the version metadata provider
#                            (tools/metadata)

# Additional env vars forwarded with the same name.
set(_ENV_PASSTHROUGH
  BLAS_PROVIDER
  CUDA_HOST_COMPILER
  CUDA_NVCC_EXECUTABLE
  CUDA_SEPARABLE_COMPILATION
  CUDAToolkit_ROOT
  CUDNN_INCLUDE_DIR
  CUDNN_LIBRARY
  CUDNN_ROOT
  MKL_INTERFACE
  MKL_LINK
  MKL_ROOT
  MKL_THREADING
  MKLDNN_CPU_RUNTIME
  WERROR
)

# Forward passthrough env vars (same name)
foreach(_var IN LISTS _ENV_PASSTHROUGH)
  if(DEFINED ENV{${_var}} AND NOT DEFINED ${_var})
    set(${_var} "$ENV{${_var}}" CACHE STRING "From env ${_var}" FORCE)
  endif()
endforeach()

# Forward all BUILD_*, USE_*, CMAKE_* environment variables into the CMake
# cache, matching the -D flags setup.py used to pass.
#
# CMake cannot enumerate environment variables, and serializing the whole
# environment to text and re-parsing it in CMake is unsafe: values such as PS1
# contain ';' and '\' (and some exported shell functions even contain newlines),
# all of which collide with CMake's list, escape, and line semantics and
# silently corrupt unrelated variables. The top-level CMakeLists.txt already
# requires Python (find_package(Python COMPONENTS Interpreter REQUIRED)) before
# including this module, so read os.environ directly there -- the full
# environment is never serialized -- and have it emit only the selected,
# properly escaped cache assignments for CMake to evaluate.

# Applies one forwarded variable. An explicitly-set environment variable takes
# priority, matching the -D semantics this module emulates: override the cache
# (do not merely fill when undefined) so a value left by an earlier env-less
# configure -- an option() default or a ninja-triggered reconfigure -- cannot
# permanently shadow the environment.
function(_envfwd_apply _name _value)
  if(NOT DEFINED ${_name} OR NOT "${${_name}}" STREQUAL "${_value}")
    set(${_name} "${_value}" CACHE STRING "From environment" FORCE)
  endif()
endfunction()

# Reads os.environ and prints `_envfwd_apply("<name>" "<value>")` for each
# selected variable, escaping the value for a CMake double-quoted argument.
set(_envfwd_script [==[
import os, re, sys

select = re.compile(r"^(BUILD_|USE_|CMAKE_)")

def q(s):
    # Escape for a CMake double-quoted argument. Backslash and quote are
    # structural; '$' is escaped to suppress ${}/$ENV{} expansion. ';' and
    # newlines are literal inside quotes and need no escaping.
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")

sys.stdout.write("\n".join(
    '_envfwd_apply("%s" "%s")' % (q(name), q(value))
    for name, value in os.environ.items()
    if select.search(name)
))
]==])

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "${_envfwd_script}"
  OUTPUT_VARIABLE _envfwd_code
  RESULT_VARIABLE _envfwd_rc
)
if(NOT _envfwd_rc EQUAL 0)
  message(FATAL_ERROR
    "EnvVarForwarding: failed to read the environment via Python (exit ${_envfwd_rc}).")
endif()
cmake_language(EVAL CODE "${_envfwd_code}")
