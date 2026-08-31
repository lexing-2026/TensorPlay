
#include "tensorplay/ops/Config.h"
#include "DispatchStub.h"
#include "Parallel.h"

#include <map>
#include <sstream>
#include <cstring>

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_ONEDNN
#include "oneapi/dnnl/dnnl_common.h"
#endif

namespace tensorplay {

std::string get_mkl_version() {
  std::string version;
#ifdef USE_MKL
  {
    version.resize(198, '\0');
    mkl_get_version_string(version.data(), 198);
    version.resize(strlen(version.c_str()));
  }
#else
  version = "N/A";
#endif
  return version;
}

std::string get_mkldnn_version() {
  std::ostringstream ss;
#ifdef USE_ONEDNN
  const dnnl_version_t* ver = dnnl_version();
  ss << "Intel(R) MKL-DNN v" << ver->major << '.' << ver->minor << '.'
     << ver->patch << " (Git Hash " << ver->hash << ')';
#else
  ss << "MKLDNN not found";
#endif
  return std::move(ss).str();
}

std::string get_openmp_version() {
  std::ostringstream ss;
#ifdef _OPENMP
  ss << "OpenMP " << _OPENMP;
  const char* ver_str = nullptr;
  switch (_OPENMP) {
    case 200505:
      ver_str = "2.5";
      break;
    case 200805:
      ver_str = "3.0";
      break;
    case 201107:
      ver_str = "3.1";
      break;
    case 201307:
      ver_str = "4.0";
      break;
    case 201511:
      ver_str = "4.5";
      break;
    default:
      ver_str = nullptr;
      break;
  }
  if (ver_str) {
    ss << " (a.k.a. OpenMP " << ver_str << ')';
  }
#else
  ss << "OpenMP not found";
#endif
  return std::move(ss).str();
}

std::string get_cpu_capability_str() {
  auto cap = cpu::get_cpu_capability();
  switch (cap) {
    case cpu::CPUCapability::DEFAULT:
      return "DEFAULT";
#ifdef HAVE_AVX2_CPU_DEFINITION
    case cpu::CPUCapability::AVX2:
      return "AVX2";
#endif
#ifdef HAVE_AVX512_CPU_DEFINITION
    case cpu::CPUCapability::AVX512:
      return "AVX512";
#endif
    default:
      break;
  }
  return "";
}

std::string show_config() {
  std::ostringstream ss;
  ss << "TensorPlay built with:\n";

#if defined(__GNUC__)
  ss << "  - GCC " << __GNUC__ << '.' << __GNUC_MINOR__ << '\n';
#endif

#if defined(__cplusplus)
  ss << "  - C++ Version: " << __cplusplus << '\n';
#endif

#if defined(__clang_major__)
  ss << "  - clang " << __clang_major__ << '.' << __clang_minor__ << '.'
     << __clang_patchlevel__ << '\n';
#endif

#ifdef USE_MKL
  ss << "  - " << get_mkl_version() << '\n';
#endif

#ifdef USE_ONEDNN
  ss << "  - " << get_mkldnn_version() << '\n';
#endif

#ifdef _OPENMP
  ss << "  - " << get_openmp_version() << '\n';
#endif

#ifdef USE_MKL
  ss << "  - LAPACK is enabled (usually provided by MKL)\n";
#endif

#ifdef USE_NNPACK
  ss << "  - NNPACK is enabled\n";
#endif

  ss << "  - CPU capability usage: " << get_cpu_capability_str() << '\n';

  ss << "  - Build settings: ";
  for (const auto& pair : get_build_info()) {
    if (pair.first == "BUILD_SETTINGS") {
      ss << pair.second;
    }
  }
  ss << '\n';

  return ss.str();
}

std::string _cxx_flags() {
  for (const auto& pair : get_build_info()) {
    if (pair.first == "CXX_FLAGS") {
      return pair.second;
    }
  }
  return "";
}

std::string _parallel_info() {
  return tensorplay::parallel::get_parallel_info();
}

std::map<std::string, std::string> get_build_info() {
  std::map<std::string, std::string> info = TP_BUILD_INFO;
  info["MKL_INFO"]          = get_mkl_version();
  info["ONEDNN_INFO"]       = get_mkldnn_version();
  info["OPENMP_INFO"]       = get_openmp_version();
  info["CPU_CAPABILITY"]    = get_cpu_capability_str();
  return info;
}

}  // namespace tensorplay
