// Runtime resolution of an ILP64 LAPACK (see cpu/Lapack.h).
//
// Search order:
//   1. $TP_LAPACK_LIB (explicit override)
//   2. a libscipy_openblas64_* already mapped into this process (numpy
//      imported before the first linalg call)
//   3. numpy's wheel directory: <env>/numpy.libs/libscipy_openblas64_*
//
// Symbols are tried as `scipy_<name>_64_` (scipy-openblas wheels) and then as
// `<name>_64_` / `<name>_` for other ILP64 OpenBLAS builds.

#include "cpu/Lapack.h"

#include "Exception.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#include <dirent.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace tensorplay {
namespace cpu {

namespace {

using F = void*;  // raw Fortran-ABI entry point

F g_sgetrf, g_dgetrf, g_sgetrs, g_dgetrs;
F g_cgetrf, g_zgetrf, g_cgetrs, g_zgetrs;
F g_spotrf, g_dpotrf;
F g_cpotrf, g_zpotrf;
F g_spotrs, g_dpotrs;
F g_cpotrs, g_zpotrs;
F g_spotri, g_dpotri;
F g_cpotri, g_zpotri;
F g_sgeqrf, g_dgeqrf, g_sorgqr, g_dorgqr;
F g_cgeqrf, g_zgeqrf, g_cungqr, g_zungqr;
F g_sgesdd, g_dgesdd;
F g_cgesdd, g_zgesdd;
F g_ssyevd, g_dsyevd;
F g_cheevd, g_zheevd;
F g_sgeev, g_dgeev;
F g_cgeev, g_zgeev;
F g_strtrs, g_dtrtrs;
F g_ctrtrs, g_ztrtrs;
F g_strsm, g_dtrsm, g_ctrsm, g_ztrsm;
F g_sgels, g_dgels;
F g_cgels, g_zgels;
F g_sgelsy, g_dgelsy, g_cgelsy, g_zgelsy;
F g_sgelsd, g_dgelsd, g_cgelsd, g_zgelsd;
F g_sgelss, g_dgelss, g_cgelss, g_zgelss;
F g_ssytrf, g_dsytrf, g_ssytrs, g_dsytrs;
F g_csytrf, g_zsytrf, g_chetrf, g_zhetrf;
F g_csytrs, g_zsytrs, g_chetrs, g_zhetrs;

#ifdef _WIN32
using LibHandle = HMODULE;
static LibHandle open_lib(const char* path) { return LoadLibraryA(path); }
static void* sym(LibHandle h, const char* n) {
    return reinterpret_cast<void*>(GetProcAddress(h, n));
}
#else
using LibHandle = void*;
static LibHandle open_lib(const char* path) {
    return dlopen(path, RTLD_LAZY | RTLD_LOCAL);
}
static void* sym(LibHandle h, const char* n) { return dlsym(h, n); }
#endif

void* resolve_one(LibHandle handle, const char* base) {
    char buf[128];
    const char* patterns[] = {"scipy_%s_64_", "%s_64_", "%s_", "scipy_%s64_"};
    for (const char* pattern : patterns) {
        std::snprintf(buf, sizeof(buf), pattern, base);
        if (void* p = sym(handle, buf)) return p;
    }
    return nullptr;
}

bool resolve_all(LibHandle handle) {
    struct Pair { F& slot; const char* name; };
    const Pair pairs[] = {
        {g_sgetrf, "sgetrf"}, {g_dgetrf, "dgetrf"},
        {g_sgetrs, "sgetrs"}, {g_dgetrs, "dgetrs"},
        {g_cgetrf, "cgetrf"}, {g_zgetrf, "zgetrf"},
        {g_cgetrs, "cgetrs"}, {g_zgetrs, "zgetrs"},
        {g_spotrf, "spotrf"}, {g_dpotrf, "dpotrf"},
        {g_cpotrf, "cpotrf"}, {g_zpotrf, "zpotrf"},
        {g_spotrs, "spotrs"}, {g_dpotrs, "dpotrs"},
        {g_cpotrs, "cpotrs"}, {g_zpotrs, "zpotrs"},
        {g_spotri, "spotri"}, {g_dpotri, "dpotri"},
        {g_cpotri, "cpotri"}, {g_zpotri, "zpotri"},
        // LAPACK's generic-name routines (geqrf/gesdd/syevd/...) carry the
        // s/d precision prefix in their symbols; the scipy-openblas wheel
        // exports them as scipy_<s|d><name>_64_.
        {g_sgeqrf, "sgeqrf"}, {g_dgeqrf, "dgeqrf"},
        {g_sorgqr, "sorgqr"}, {g_dorgqr, "dorgqr"},
        {g_cgeqrf, "cgeqrf"}, {g_zgeqrf, "zgeqrf"},
        {g_cungqr, "cungqr"}, {g_zungqr, "zungqr"},
        {g_sgesdd, "sgesdd"}, {g_dgesdd, "dgesdd"},
        {g_cgesdd, "cgesdd"}, {g_zgesdd, "zgesdd"},
        {g_ssyevd, "ssyevd"}, {g_dsyevd, "dsyevd"},
        {g_cheevd, "cheevd"}, {g_zheevd, "zheevd"},
        {g_sgeev,  "sgeev"},  {g_dgeev,  "dgeev"},
        {g_cgeev,  "cgeev"},  {g_zgeev,  "zgeev"},
        {g_strtrs, "strtrs"}, {g_dtrtrs, "dtrtrs"},
        {g_ctrtrs, "ctrtrs"}, {g_ztrtrs, "ztrtrs"},
        {g_strsm, "cblas_strsm"}, {g_dtrsm, "cblas_dtrsm"},
        {g_ctrsm, "cblas_ctrsm"}, {g_ztrsm, "cblas_ztrsm"},
        {g_sgels,  "sgels"},  {g_dgels,  "dgels"},
        {g_cgels,  "cgels"},  {g_zgels,  "zgels"},
        {g_sgelsy, "sgelsy"}, {g_dgelsy, "dgelsy"},
        {g_cgelsy, "cgelsy"}, {g_zgelsy, "zgelsy"},
        {g_sgelsd, "sgelsd"}, {g_dgelsd, "dgelsd"},
        {g_cgelsd, "cgelsd"}, {g_zgelsd, "zgelsd"},
        {g_sgelss, "sgelss"}, {g_dgelss, "dgelss"},
        {g_cgelss, "cgelss"}, {g_zgelss, "zgelss"},
        {g_ssytrf, "ssytrf"}, {g_dsytrf, "dsytrf"},
        {g_ssytrs, "ssytrs"}, {g_dsytrs, "dsytrs"},
        {g_csytrf, "csytrf"}, {g_zsytrf, "zsytrf"},
        {g_chetrf, "chetrf"}, {g_zhetrf, "zhetrf"},
        {g_csytrs, "csytrs"}, {g_zsytrs, "zsytrs"},
        {g_chetrs, "chetrs"}, {g_zhetrs, "zhetrs"},
    };
    for (const auto& p : pairs) {
        p.slot = resolve_one(handle, p.name);
        if (!p.slot) return false;
    }
    return true;
}

LibHandle find_library() {
#ifdef _WIN32
    if (const char* env = std::getenv("TP_LAPACK_LIB")) {
        if (LibHandle h = open_lib(env)) return h;
    }
    namespace fs = std::filesystem;
    std::vector<std::string> bases;
    if (const char* cp = std::getenv("CONDA_PREFIX")) bases.push_back(cp);
    if (const char* vp = std::getenv("VIRTUAL_ENV")) bases.push_back(vp);
    if (const char* py = std::getenv("Python_ROOT_DIR")) {
        bases.push_back((fs::path(py) / "Lib" / "site-packages").string());
        bases.push_back((fs::path(py) / "lib" / "site-packages").string());
    }
    std::vector<char> executable_path(MAX_PATH);
    for (;;) {
        const DWORD length = GetModuleFileNameA(
            nullptr, executable_path.data(),
            static_cast<DWORD>(executable_path.size()));
        if (length == 0) break;
        if (length + 1 < executable_path.size()) {
            const fs::path executable(
                std::string(executable_path.data(), length));
            const fs::path python_root = executable.parent_path();
            bases.push_back(
                (python_root / "Lib" / "site-packages").string());
            bases.push_back(
                (python_root / "lib" / "site-packages").string());
            break;
        }
        executable_path.resize(executable_path.size() * 2);
    }
    bases.push_back("C:\\Python313\\Lib\\site-packages");
    for (const auto& base : bases) {
        std::error_code ec;
        const fs::path dir_path = fs::path(base) / "numpy.libs";
        if (!fs::exists(dir_path, ec)) continue;
        for (const auto& ent : fs::directory_iterator(dir_path, ec)) {
            const std::string name = ent.path().filename().string();
            if (name.find("scipy_openblas") == std::string::npos &&
                name.find("openblas") == std::string::npos) continue;
            if (LibHandle h = open_lib(ent.path().string().c_str())) return h;
        }
    }
    return open_lib("libscipy_openblas.dll");
#else
    if (const char* env = std::getenv("TP_LAPACK_LIB")) {
        if (void* h = dlopen(env, RTLD_LAZY)) return h;
    }
    // Already mapped (e.g. numpy imported earlier in this process)?
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        const auto pos = line.find("libscipy_openblas");
        if (pos == std::string::npos) continue;
        const auto start = line.find_first_of('/');
        if (start == std::string::npos) continue;
        auto end = line.size();
        while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t')) --end;
        if (line.find(".so") == std::string::npos) continue;
        if (void* h = dlopen(line.substr(start, end - start).c_str(), RTLD_LAZY | RTLD_NOLOAD)) return h;
    }
    // numpy's wheel directory inside a conda/venv prefix.  Distros install
    // under versioned python3.X paths, so glob every interpreter we can see.
    std::vector<std::string> bases;
    if (const char* cp = std::getenv("CONDA_PREFIX")) bases.push_back(cp);
    if (const char* vp = std::getenv("VIRTUAL_ENV")) bases.push_back(vp);
    {
        const char* home = std::getenv("PYTHONHOME");
        if (home) bases.push_back(std::string(home) + "/lib");
    }
    bases.push_back("/usr/local/lib");  // Debian/Ubuntu pip: python3.X/dist-packages
    bases.push_back("/usr/lib");        // apt: python3/dist-packages
    for (const auto& base : bases) {
        const std::string dir_path = base + "/numpy.libs";
        DIR* dir = opendir(dir_path.c_str());
        if (!dir) continue;
        while (const dirent* ent = readdir(dir)) {
            const std::string name = ent->d_name;
            if (name.find("scipy_openblas") == std::string::npos &&
                name.find("openblas64") == std::string::npos) continue;
            const std::string path = dir_path + "/" + name;
            if (void* h = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL)) return h;
        }
        closedir(dir);
    }
    // Versioned interpreter prefixes (python3.Y layout varies across distros).
    for (const char* prefix : {"/usr/local/lib", "/usr/lib"}) {
        for (int minor = 8; minor <= 14; ++minor) {
            for (const char* kind : {"dist-packages", "site-packages"}) {
                const std::string dir_path = std::string(prefix) + "/python3." +
                    std::to_string(minor) + "/" + kind + "/numpy.libs";
                DIR* dir = opendir(dir_path.c_str());
                if (!dir) continue;
                while (const dirent* ent = readdir(dir)) {
                    const std::string name = ent->d_name;
                    if (name.find("scipy_openblas") == std::string::npos &&
                        name.find("openblas64") == std::string::npos) continue;
                    const std::string path = dir_path + "/" + name;
                    if (void* h = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL)) return h;
                }
                closedir(dir);
            }
        }
    }
    // Last resort: loader search path.
    return dlopen("libscipy_openblas.so", RTLD_LAZY | RTLD_LOCAL);
#endif
}

std::once_flag g_once;
bool g_ok = false;

void init_once() {
    auto handle = find_library();
    if (!handle) return;
    g_ok = resolve_all(handle);
}

}  // namespace

bool lapack_available() {
    std::call_once(g_once, init_once);
    return g_ok;
}

void require_lapack(const char* api_name) {
    if (lapack_available()) return;
    TP_THROW(RuntimeError,
             "Calling ", api_name, " on a CPU tensor requires a LAPACK library. ",
             "TensorPlay resolves one at runtime from numpy's bundled OpenBLAS ",
             "(install numpy and import it first, or set TP_LAPACK_LIB to an ",
             "ILP64 libscipy_openblas64 / libopenblas64 path).");
}

#define TP_LAPACK_CALL(sym, ...) \
    reinterpret_cast<void (*)(__VA_ARGS__)>(g_##sym)

int64_t lapack_sgetrf(int64_t m, int64_t n, float* a, int64_t lda, int64_t* ipiv) {
    int64_t info = 0;
    TP_LAPACK_CALL(sgetrf, const int64_t*, const int64_t*, float*, const int64_t*, int64_t*, int64_t*)(&m, &n, a, &lda, ipiv, &info);
    return info;
}
int64_t lapack_dgetrf(int64_t m, int64_t n, double* a, int64_t lda, int64_t* ipiv) {
    int64_t info = 0;
    TP_LAPACK_CALL(dgetrf, const int64_t*, const int64_t*, double*, const int64_t*, int64_t*, int64_t*)(&m, &n, a, &lda, ipiv, &info);
    return info;
}
int64_t lapack_sgetrs(char trans, int64_t n, int64_t nrhs, const float* a,
                      int64_t lda, const int64_t* ipiv, float* b, int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(sgetrs, const char*, const int64_t*, const int64_t*, const float*, const int64_t*, const int64_t*, float*, const int64_t*, int64_t*)(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}
int64_t lapack_dgetrs(char trans, int64_t n, int64_t nrhs, const double* a,
                      int64_t lda, const int64_t* ipiv, double* b, int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(dgetrs, const char*, const int64_t*, const int64_t*, const double*, const int64_t*, const int64_t*, double*, const int64_t*, int64_t*)(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}
int64_t lapack_cgetrf(int64_t m, int64_t n, complex<float>* a, int64_t lda,
                      int64_t* ipiv) {
    int64_t info = 0;
    TP_LAPACK_CALL(cgetrf, const int64_t*, const int64_t*, complex<float>*,
                   const int64_t*, int64_t*, int64_t*)(&m, &n, a, &lda, ipiv, &info);
    return info;
}
int64_t lapack_zgetrf(int64_t m, int64_t n, complex<double>* a, int64_t lda,
                      int64_t* ipiv) {
    int64_t info = 0;
    TP_LAPACK_CALL(zgetrf, const int64_t*, const int64_t*, complex<double>*,
                   const int64_t*, int64_t*, int64_t*)(&m, &n, a, &lda, ipiv, &info);
    return info;
}
int64_t lapack_cgetrs(char trans, int64_t n, int64_t nrhs,
                      const complex<float>* a, int64_t lda,
                      const int64_t* ipiv, complex<float>* b, int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(cgetrs, const char*, const int64_t*, const int64_t*,
                   const complex<float>*, const int64_t*, const int64_t*,
                   complex<float>*, const int64_t*, int64_t*)(&trans, &n,
                   &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}
int64_t lapack_zgetrs(char trans, int64_t n, int64_t nrhs,
                      const complex<double>* a, int64_t lda,
                      const int64_t* ipiv, complex<double>* b, int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(zgetrs, const char*, const int64_t*, const int64_t*,
                   const complex<double>*, const int64_t*, const int64_t*,
                   complex<double>*, const int64_t*, int64_t*)(&trans, &n,
                   &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}
int64_t lapack_spotrf(char uplo, int64_t n, float* a, int64_t lda) {
    int64_t info = 0;
    TP_LAPACK_CALL(spotrf, const char*, const int64_t*, float*, const int64_t*, int64_t*)(&uplo, &n, a, &lda, &info);
    return info;
}
int64_t lapack_dpotrf(char uplo, int64_t n, double* a, int64_t lda) {
    int64_t info = 0;
    TP_LAPACK_CALL(dpotrf, const char*, const int64_t*, double*, const int64_t*, int64_t*)(&uplo, &n, a, &lda, &info);
    return info;
}
int64_t lapack_cpotrf(char uplo, int64_t n, complex<float>* a, int64_t lda) {
    int64_t info = 0;
    TP_LAPACK_CALL(cpotrf, const char*, const int64_t*, complex<float>*,
                   const int64_t*, int64_t*)(&uplo, &n, a, &lda, &info);
    return info;
}
int64_t lapack_zpotrf(char uplo, int64_t n, complex<double>* a, int64_t lda) {
    int64_t info = 0;
    TP_LAPACK_CALL(zpotrf, const char*, const int64_t*, complex<double>*,
                   const int64_t*, int64_t*)(&uplo, &n, a, &lda, &info);
    return info;
}
int64_t lapack_spotrs(char uplo, int64_t n, int64_t nrhs, const float* a,
                      int64_t lda, float* b, int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(spotrs, const char*, const int64_t*, const int64_t*, const float*, const int64_t*, float*, const int64_t*, int64_t*)(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}
int64_t lapack_dpotrs(char uplo, int64_t n, int64_t nrhs, const double* a,
                      int64_t lda, double* b, int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(dpotrs, const char*, const int64_t*, const int64_t*, const double*, const int64_t*, double*, const int64_t*, int64_t*)(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}
int64_t lapack_cpotrs(char uplo, int64_t n, int64_t nrhs,
                      const complex<float>* a, int64_t lda,
                      complex<float>* b, int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(cpotrs, const char*, const int64_t*, const int64_t*,
                   const complex<float>*, const int64_t*, complex<float>*,
                   const int64_t*, int64_t*)(&uplo, &n, &nrhs, a, &lda, b, &ldb,
                                              &info);
    return info;
}
int64_t lapack_zpotrs(char uplo, int64_t n, int64_t nrhs,
                      const complex<double>* a, int64_t lda,
                      complex<double>* b, int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(zpotrs, const char*, const int64_t*, const int64_t*,
                   const complex<double>*, const int64_t*, complex<double>*,
                   const int64_t*, int64_t*)(&uplo, &n, &nrhs, a, &lda, b, &ldb,
                                              &info);
    return info;
}
int64_t lapack_spotri(char uplo, int64_t n, float* a, int64_t lda) {
    int64_t info = 0;
    TP_LAPACK_CALL(spotri, const char*, const int64_t*, float*, const int64_t*, int64_t*)(&uplo, &n, a, &lda, &info);
    return info;
}
int64_t lapack_dpotri(char uplo, int64_t n, double* a, int64_t lda) {
    int64_t info = 0;
    TP_LAPACK_CALL(dpotri, const char*, const int64_t*, double*, const int64_t*, int64_t*)(&uplo, &n, a, &lda, &info);
    return info;
}
int64_t lapack_cpotri(char uplo, int64_t n, complex<float>* a, int64_t lda) {
    int64_t info = 0;
    TP_LAPACK_CALL(cpotri, const char*, const int64_t*, complex<float>*,
                   const int64_t*, int64_t*)(&uplo, &n, a, &lda, &info);
    return info;
}
int64_t lapack_zpotri(char uplo, int64_t n, complex<double>* a, int64_t lda) {
    int64_t info = 0;
    TP_LAPACK_CALL(zpotri, const char*, const int64_t*, complex<double>*,
                   const int64_t*, int64_t*)(&uplo, &n, a, &lda, &info);
    return info;
}
int64_t lapack_sgeqrf(int64_t m, int64_t n, float* a, int64_t lda, float* tau,
                      float* work, int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(sgeqrf, const int64_t*, const int64_t*, float*, const int64_t*, float*, float*, const int64_t*, int64_t*)(&m, &n, a, &lda, tau, work, &lwork, &info);
    return info;
}
int64_t lapack_dgeqrf(int64_t m, int64_t n, double* a, int64_t lda, double* tau,
                      double* work, int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(dgeqrf, const int64_t*, const int64_t*, double*, const int64_t*, double*, double*, const int64_t*, int64_t*)(&m, &n, a, &lda, tau, work, &lwork, &info);
    return info;
}
int64_t lapack_sorgqr(int64_t m, int64_t n, int64_t k, float* a, int64_t lda,
                      const float* tau, float* work, int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(sorgqr, const int64_t*, const int64_t*, const int64_t*, float*, const int64_t*, const float*, float*, const int64_t*, int64_t*)(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
    return info;
}
int64_t lapack_dorgqr(int64_t m, int64_t n, int64_t k, double* a, int64_t lda,
                      const double* tau, double* work, int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(dorgqr, const int64_t*, const int64_t*, const int64_t*, double*, const int64_t*, const double*, double*, const int64_t*, int64_t*)(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
    return info;
}
int64_t lapack_cgeqrf(int64_t m, int64_t n, complex<float>* a, int64_t lda,
                      complex<float>* tau, complex<float>* work,
                      int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(cgeqrf, const int64_t*, const int64_t*, complex<float>*,
                   const int64_t*, complex<float>*, complex<float>*,
                   const int64_t*, int64_t*)(&m, &n, a, &lda, tau, work, &lwork,
                                              &info);
    return info;
}
int64_t lapack_zgeqrf(int64_t m, int64_t n, complex<double>* a, int64_t lda,
                      complex<double>* tau, complex<double>* work,
                      int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(zgeqrf, const int64_t*, const int64_t*, complex<double>*,
                   const int64_t*, complex<double>*, complex<double>*,
                   const int64_t*, int64_t*)(&m, &n, a, &lda, tau, work, &lwork,
                                              &info);
    return info;
}
int64_t lapack_cungqr(int64_t m, int64_t n, int64_t k, complex<float>* a,
                      int64_t lda, const complex<float>* tau,
                      complex<float>* work, int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(cungqr, const int64_t*, const int64_t*, const int64_t*,
                   complex<float>*, const int64_t*, const complex<float>*,
                   complex<float>*, const int64_t*, int64_t*)(&m, &n, &k, a,
                   &lda, tau, work, &lwork, &info);
    return info;
}
int64_t lapack_zungqr(int64_t m, int64_t n, int64_t k, complex<double>* a,
                      int64_t lda, const complex<double>* tau,
                      complex<double>* work, int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(zungqr, const int64_t*, const int64_t*, const int64_t*,
                   complex<double>*, const int64_t*, const complex<double>*,
                   complex<double>*, const int64_t*, int64_t*)(&m, &n, &k, a,
                   &lda, tau, work, &lwork, &info);
    return info;
}
int64_t lapack_sgesdd(char jobz, int64_t m, int64_t n, float* a, int64_t lda,
                      float* s, float* u, int64_t ldu, float* vt, int64_t ldvt,
                      float* work, int64_t lwork, int64_t* iwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(sgesdd, const char*, const int64_t*, const int64_t*, float*, const int64_t*, float*, float*, const int64_t*, float*, const int64_t*, float*, const int64_t*, int64_t*, int64_t*)(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info);
    return info;
}
int64_t lapack_dgesdd(char jobz, int64_t m, int64_t n, double* a, int64_t lda,
                      double* s, double* u, int64_t ldu, double* vt,
                      int64_t ldvt, double* work, int64_t lwork, int64_t* iwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(dgesdd, const char*, const int64_t*, const int64_t*, double*, const int64_t*, double*, double*, const int64_t*, double*, const int64_t*, double*, const int64_t*, int64_t*, int64_t*)(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info);
    return info;
}
int64_t lapack_cgesdd(char jobz, int64_t m, int64_t n, complex<float>* a,
                      int64_t lda, float* s, complex<float>* u,
                      int64_t ldu, complex<float>* vt, int64_t ldvt,
                      complex<float>* work, int64_t lwork, float* rwork,
                      int64_t* iwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(cgesdd, const char*, const int64_t*, const int64_t*,
                   complex<float>*, const int64_t*, float*, complex<float>*,
                   const int64_t*, complex<float>*, const int64_t*,
                   complex<float>*, const int64_t*, float*, int64_t*, int64_t*)(&jobz,
                   &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork,
                   iwork, &info);
    return info;
}
int64_t lapack_zgesdd(char jobz, int64_t m, int64_t n, complex<double>* a,
                      int64_t lda, double* s, complex<double>* u,
                      int64_t ldu, complex<double>* vt, int64_t ldvt,
                      complex<double>* work, int64_t lwork, double* rwork,
                      int64_t* iwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(zgesdd, const char*, const int64_t*, const int64_t*,
                   complex<double>*, const int64_t*, double*, complex<double>*,
                   const int64_t*, complex<double>*, const int64_t*,
                   complex<double>*, const int64_t*, double*, int64_t*, int64_t*)(&jobz,
                   &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork,
                   iwork, &info);
    return info;
}
int64_t lapack_ssyevd(char jobz, char uplo, int64_t n, float* a, int64_t lda,
                      float* w, float* work, int64_t lwork, int64_t* iwork,
                      int64_t liwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(ssyevd, const char*, const char*, const int64_t*, float*, const int64_t*, float*, float*, const int64_t*, int64_t*, const int64_t*, int64_t*)(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, &info);
    return info;
}
int64_t lapack_dsyevd(char jobz, char uplo, int64_t n, double* a, int64_t lda,
                      double* w, double* work, int64_t lwork, int64_t* iwork,
                      int64_t liwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(dsyevd, const char*, const char*, const int64_t*, double*, const int64_t*, double*, double*, const int64_t*, int64_t*, const int64_t*, int64_t*)(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, &info);
    return info;
}
int64_t lapack_cheevd(char jobz, char uplo, int64_t n, complex<float>* a,
                      int64_t lda, float* w, complex<float>* work,
                      int64_t lwork, float* rwork, int64_t lrwork,
                      int64_t* iwork, int64_t liwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(cheevd, const char*, const char*, const int64_t*,
                   complex<float>*, const int64_t*, float*, complex<float>*,
                   const int64_t*, float*, const int64_t*, int64_t*, const int64_t*,
                   int64_t*)(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork,
                              &lrwork, iwork, &liwork, &info);
    return info;
}
int64_t lapack_zheevd(char jobz, char uplo, int64_t n, complex<double>* a,
                      int64_t lda, double* w, complex<double>* work,
                      int64_t lwork, double* rwork, int64_t lrwork,
                      int64_t* iwork, int64_t liwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(zheevd, const char*, const char*, const int64_t*,
                   complex<double>*, const int64_t*, double*, complex<double>*,
                   const int64_t*, double*, const int64_t*, int64_t*, const int64_t*,
                   int64_t*)(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork,
                              &lrwork, iwork, &liwork, &info);
    return info;
}
int64_t lapack_sgeev(char jobvl, char jobvr, int64_t n, float* a, int64_t lda,
                     float* wr, float* wi, float* vl, int64_t ldvl, float* vr,
                     int64_t ldvr, float* work, int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(sgeev, const char*, const char*, const int64_t*, float*, const int64_t*, float*, float*, float*, const int64_t*, float*, const int64_t*, float*, const int64_t*, int64_t*)(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info);
    return info;
}
int64_t lapack_dgeev(char jobvl, char jobvr, int64_t n, double* a, int64_t lda,
                     double* wr, double* wi, double* vl, int64_t ldvl,
                     double* vr, int64_t ldvr, double* work, int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(dgeev, const char*, const char*, const int64_t*, double*, const int64_t*, double*, double*, double*, const int64_t*, double*, const int64_t*, double*, const int64_t*, int64_t*)(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info);
    return info;
}
int64_t lapack_cgeev(char jobvl, char jobvr, int64_t n, complex<float>* a,
                     int64_t lda, complex<float>* w,
                     complex<float>* vl, int64_t ldvl,
                     complex<float>* vr, int64_t ldvr,
                     complex<float>* work, int64_t lwork, float* rwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(cgeev, const char*, const char*, const int64_t*,
                   complex<float>*, const int64_t*, complex<float>*,
                   complex<float>*, const int64_t*, complex<float>*,
                   const int64_t*, complex<float>*, const int64_t*, float*,
                   int64_t*)(&jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr,
                              work, &lwork, rwork, &info);
    return info;
}
int64_t lapack_zgeev(char jobvl, char jobvr, int64_t n, complex<double>* a,
                     int64_t lda, complex<double>* w,
                     complex<double>* vl, int64_t ldvl,
                     complex<double>* vr, int64_t ldvr,
                     complex<double>* work, int64_t lwork, double* rwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(zgeev, const char*, const char*, const int64_t*,
                   complex<double>*, const int64_t*, complex<double>*,
                   complex<double>*, const int64_t*, complex<double>*,
                   const int64_t*, complex<double>*, const int64_t*, double*,
                   int64_t*)(&jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr,
                              work, &lwork, rwork, &info);
    return info;
}
int64_t lapack_strtrs(char uplo, char transa, char diag, int64_t n,
                      int64_t nrhs, const float* a, int64_t lda, float* b,
                      int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(strtrs, const char*, const char*, const char*, const int64_t*, const int64_t*, const float*, const int64_t*, float*, const int64_t*, int64_t*)(&uplo, &transa, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}
int64_t lapack_dtrtrs(char uplo, char transa, char diag, int64_t n,
                      int64_t nrhs, const double* a, int64_t lda, double* b,
                      int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(dtrtrs, const char*, const char*, const char*, const int64_t*, const int64_t*, const double*, const int64_t*, double*, const int64_t*, int64_t*)(&uplo, &transa, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}
int64_t lapack_ctrtrs(char uplo, char transa, char diag, int64_t n,
                      int64_t nrhs, const complex<float>* a, int64_t lda,
                      complex<float>* b, int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(ctrtrs, const char*, const char*, const char*,
                   const int64_t*, const int64_t*, const complex<float>*,
                   const int64_t*, complex<float>*, const int64_t*, int64_t*)(&uplo,
                   &transa, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}
int64_t lapack_ztrtrs(char uplo, char transa, char diag, int64_t n,
                      int64_t nrhs, const complex<double>* a, int64_t lda,
                      complex<double>* b, int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(ztrtrs, const char*, const char*, const char*,
                   const int64_t*, const int64_t*, const complex<double>*,
                   const int64_t*, complex<double>*, const int64_t*, int64_t*)(&uplo,
                   &transa, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}

// CBLAS trsm bindings (order/side/uplo/trans/diag pass through the CBLAS
// enum values; see cpu/Lapack.h for why these back the triangular solves).
void lapack_strsm(int64_t order, int64_t side, int64_t uplo, int64_t trans,
                  int64_t diag, int64_t m, int64_t n, float alpha,
                  const float* a, int64_t lda, float* b, int64_t ldb) {
    // CBLAS takes every scalar by value (netlib ABI), unlike the Fortran
    // LAPACK entries above which take pointers.
    TP_LAPACK_CALL(strsm, const int64_t, const int64_t, const int64_t,
                   const int64_t, const int64_t, const int64_t, const int64_t,
                   const float, const float*, const int64_t, float*,
                   const int64_t)(order, side, uplo, trans, diag, m, n,
                                  alpha, a, lda, b, ldb);
}
void lapack_dtrsm(int64_t order, int64_t side, int64_t uplo, int64_t trans,
                  int64_t diag, int64_t m, int64_t n, double alpha,
                  const double* a, int64_t lda, double* b, int64_t ldb) {
    TP_LAPACK_CALL(dtrsm, const int64_t, const int64_t, const int64_t,
                   const int64_t, const int64_t, const int64_t, const int64_t,
                   const double, const double*, const int64_t, double*,
                   const int64_t)(order, side, uplo, trans, diag, m, n,
                                  alpha, a, lda, b, ldb);
}
void lapack_ctrsm(int64_t order, int64_t side, int64_t uplo, int64_t trans,
                  int64_t diag, int64_t m, int64_t n,
                  const complex<float>* alpha,
                  const complex<float>* a, int64_t lda,
                  complex<float>* b, int64_t ldb) {
    TP_LAPACK_CALL(ctrsm, const int64_t, const int64_t, const int64_t,
                   const int64_t, const int64_t, const int64_t, const int64_t,
                   const complex<float>*, const complex<float>*,
                   const int64_t, complex<float>*, const int64_t)(
                           order, side, uplo, trans, diag, m, n, alpha, a, lda,
                           b, ldb);
}
void lapack_ztrsm(int64_t order, int64_t side, int64_t uplo, int64_t trans,
                  int64_t diag, int64_t m, int64_t n,
                  const complex<double>* alpha,
                  const complex<double>* a, int64_t lda,
                  complex<double>* b, int64_t ldb) {
    TP_LAPACK_CALL(ztrsm, const int64_t, const int64_t, const int64_t,
                   const int64_t, const int64_t, const int64_t, const int64_t,
                   const complex<double>*, const complex<double>*,
                   const int64_t, complex<double>*, const int64_t)(
                           order, side, uplo, trans, diag, m, n, alpha, a, lda,
                           b, ldb);
}
int64_t lapack_sgels(char trans, int64_t m, int64_t n, int64_t nrhs, float* a,
                     int64_t lda, float* b, int64_t ldb, float* work,
                     int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(sgels, const char*, const int64_t*, const int64_t*, const int64_t*, float*, const int64_t*, float*, const int64_t*, float*, const int64_t*, int64_t*)(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);
    return info;
}
int64_t lapack_dgels(char trans, int64_t m, int64_t n, int64_t nrhs, double* a,
                     int64_t lda, double* b, int64_t ldb, double* work,
                     int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(dgels, const char*, const int64_t*, const int64_t*, const int64_t*, double*, const int64_t*, double*, const int64_t*, double*, const int64_t*, int64_t*)(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);
    return info;
}
int64_t lapack_cgels(char trans, int64_t m, int64_t n, int64_t nrhs,
                     complex<float>* a, int64_t lda,
                     complex<float>* b, int64_t ldb,
                     complex<float>* work, int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(cgels, const char*, const int64_t*, const int64_t*, const int64_t*,
                   complex<float>*, const int64_t*, complex<float>*,
                   const int64_t*, complex<float>*, const int64_t*, int64_t*)(&trans,
                   &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);
    return info;
}
int64_t lapack_zgels(char trans, int64_t m, int64_t n, int64_t nrhs,
                     complex<double>* a, int64_t lda,
                     complex<double>* b, int64_t ldb,
                     complex<double>* work, int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(zgels, const char*, const int64_t*, const int64_t*, const int64_t*,
                   complex<double>*, const int64_t*, complex<double>*,
                   const int64_t*, complex<double>*, const int64_t*, int64_t*)(&trans,
                   &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);
    return info;
}
int64_t lapack_sgelsy(int64_t m, int64_t n, int64_t nrhs, float* a,
                      int64_t lda, float* b, int64_t ldb, int64_t* jpvt,
                      float rcond, int64_t* rank, float* work, int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(sgelsy, const int64_t*, const int64_t*, const int64_t*, float*,
                   const int64_t*, float*, const int64_t*, int64_t*, const float*,
                   int64_t*, float*, const int64_t*, int64_t*)(&m, &n, &nrhs, a,
                   &lda, b, &ldb, jpvt, &rcond, rank, work, &lwork, &info);
    return info;
}
int64_t lapack_dgelsy(int64_t m, int64_t n, int64_t nrhs, double* a,
                      int64_t lda, double* b, int64_t ldb, int64_t* jpvt,
                      double rcond, int64_t* rank, double* work, int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(dgelsy, const int64_t*, const int64_t*, const int64_t*, double*,
                   const int64_t*, double*, const int64_t*, int64_t*, const double*,
                   int64_t*, double*, const int64_t*, int64_t*)(&m, &n, &nrhs, a,
                   &lda, b, &ldb, jpvt, &rcond, rank, work, &lwork, &info);
    return info;
}
int64_t lapack_cgelsy(int64_t m, int64_t n, int64_t nrhs,
                      complex<float>* a, int64_t lda,
                      complex<float>* b, int64_t ldb, int64_t* jpvt,
                      float rcond, int64_t* rank, complex<float>* work,
                      int64_t lwork, float* rwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(cgelsy, const int64_t*, const int64_t*, const int64_t*,
                   complex<float>*, const int64_t*, complex<float>*,
                   const int64_t*, int64_t*, const float*, int64_t*,
                   complex<float>*, const int64_t*, float*, int64_t*)(&m,
                   &n, &nrhs, a, &lda, b, &ldb, jpvt, &rcond, rank, work,
                   &lwork, rwork, &info);
    return info;
}
int64_t lapack_zgelsy(int64_t m, int64_t n, int64_t nrhs,
                      complex<double>* a, int64_t lda,
                      complex<double>* b, int64_t ldb, int64_t* jpvt,
                      double rcond, int64_t* rank, complex<double>* work,
                      int64_t lwork, double* rwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(zgelsy, const int64_t*, const int64_t*, const int64_t*,
                   complex<double>*, const int64_t*, complex<double>*,
                   const int64_t*, int64_t*, const double*, int64_t*,
                   complex<double>*, const int64_t*, double*, int64_t*)(&m,
                   &n, &nrhs, a, &lda, b, &ldb, jpvt, &rcond, rank, work,
                   &lwork, rwork, &info);
    return info;
}
int64_t lapack_sgelsd(int64_t m, int64_t n, int64_t nrhs, float* a,
                      int64_t lda, float* b, int64_t ldb, float* s,
                      float rcond, int64_t* rank, float* work, int64_t lwork,
                      int64_t* iwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(sgelsd, const int64_t*, const int64_t*, const int64_t*, float*,
                   const int64_t*, float*, const int64_t*, float*, const float*,
                   int64_t*, float*, const int64_t*, int64_t*, int64_t*)(&m, &n,
                   &nrhs, a, &lda, b, &ldb, s, &rcond, rank, work, &lwork,
                   iwork, &info);
    return info;
}
int64_t lapack_dgelsd(int64_t m, int64_t n, int64_t nrhs, double* a,
                      int64_t lda, double* b, int64_t ldb, double* s,
                      double rcond, int64_t* rank, double* work, int64_t lwork,
                      int64_t* iwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(dgelsd, const int64_t*, const int64_t*, const int64_t*, double*,
                   const int64_t*, double*, const int64_t*, double*, const double*,
                   int64_t*, double*, const int64_t*, int64_t*, int64_t*)(&m, &n,
                   &nrhs, a, &lda, b, &ldb, s, &rcond, rank, work, &lwork,
                   iwork, &info);
    return info;
}
int64_t lapack_cgelsd(int64_t m, int64_t n, int64_t nrhs,
                      complex<float>* a, int64_t lda,
                      complex<float>* b, int64_t ldb, float* s,
                      float rcond, int64_t* rank, complex<float>* work,
                      int64_t lwork, float* rwork, int64_t* iwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(cgelsd, const int64_t*, const int64_t*, const int64_t*,
                   complex<float>*, const int64_t*, complex<float>*,
                   const int64_t*, float*, const float*, int64_t*,
                   complex<float>*, const int64_t*, float*, int64_t*,
                   int64_t*)(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, rank,
                   work, &lwork, rwork, iwork, &info);
    return info;
}
int64_t lapack_zgelsd(int64_t m, int64_t n, int64_t nrhs,
                      complex<double>* a, int64_t lda,
                      complex<double>* b, int64_t ldb, double* s,
                      double rcond, int64_t* rank, complex<double>* work,
                      int64_t lwork, double* rwork, int64_t* iwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(zgelsd, const int64_t*, const int64_t*, const int64_t*,
                   complex<double>*, const int64_t*, complex<double>*,
                   const int64_t*, double*, const double*, int64_t*,
                   complex<double>*, const int64_t*, double*, int64_t*,
                   int64_t*)(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, rank,
                   work, &lwork, rwork, iwork, &info);
    return info;
}
int64_t lapack_sgelss(int64_t m, int64_t n, int64_t nrhs, float* a,
                      int64_t lda, float* b, int64_t ldb, float* s,
                      float rcond, int64_t* rank, float* work, int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(sgelss, const int64_t*, const int64_t*, const int64_t*, float*,
                   const int64_t*, float*, const int64_t*, float*, const float*,
                   int64_t*, float*, const int64_t*, int64_t*)(&m, &n, &nrhs, a,
                   &lda, b, &ldb, s, &rcond, rank, work, &lwork, &info);
    return info;
}
int64_t lapack_dgelss(int64_t m, int64_t n, int64_t nrhs, double* a,
                      int64_t lda, double* b, int64_t ldb, double* s,
                      double rcond, int64_t* rank, double* work, int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(dgelss, const int64_t*, const int64_t*, const int64_t*, double*,
                   const int64_t*, double*, const int64_t*, double*, const double*,
                   int64_t*, double*, const int64_t*, int64_t*)(&m, &n, &nrhs, a,
                   &lda, b, &ldb, s, &rcond, rank, work, &lwork, &info);
    return info;
}
int64_t lapack_cgelss(int64_t m, int64_t n, int64_t nrhs,
                      complex<float>* a, int64_t lda,
                      complex<float>* b, int64_t ldb, float* s,
                      float rcond, int64_t* rank, complex<float>* work,
                      int64_t lwork, float* rwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(cgelss, const int64_t*, const int64_t*, const int64_t*,
                   complex<float>*, const int64_t*, complex<float>*,
                   const int64_t*, float*, const float*, int64_t*,
                   complex<float>*, const int64_t*, float*, int64_t*)(&m,
                   &n, &nrhs, a, &lda, b, &ldb, s, &rcond, rank, work, &lwork,
                   rwork, &info);
    return info;
}
int64_t lapack_zgelss(int64_t m, int64_t n, int64_t nrhs,
                      complex<double>* a, int64_t lda,
                      complex<double>* b, int64_t ldb, double* s,
                      double rcond, int64_t* rank, complex<double>* work,
                      int64_t lwork, double* rwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(zgelss, const int64_t*, const int64_t*, const int64_t*,
                   complex<double>*, const int64_t*, complex<double>*,
                   const int64_t*, double*, const double*, int64_t*,
                   complex<double>*, const int64_t*, double*, int64_t*)(&m,
                   &n, &nrhs, a, &lda, b, &ldb, s, &rcond, rank, work, &lwork,
                   rwork, &info);
    return info;
}
int64_t lapack_ssytrf(char uplo, int64_t n, float* a, int64_t lda, int64_t* ipiv,
                      float* work, int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(ssytrf, const char*, const int64_t*, float*, const int64_t*, int64_t*, float*, const int64_t*, int64_t*)(&uplo, &n, a, &lda, ipiv, work, &lwork, &info);
    return info;
}
int64_t lapack_dsytrf(char uplo, int64_t n, double* a, int64_t lda,
                      int64_t* ipiv, double* work, int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(dsytrf, const char*, const int64_t*, double*, const int64_t*, int64_t*, double*, const int64_t*, int64_t*)(&uplo, &n, a, &lda, ipiv, work, &lwork, &info);
    return info;
}
int64_t lapack_ssytrs(char uplo, int64_t n, int64_t nrhs, const float* a,
                      int64_t lda, const int64_t* ipiv, float* b, int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(ssytrs, const char*, const int64_t*, const int64_t*, const float*, const int64_t*, const int64_t*, float*, const int64_t*, int64_t*)(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}
int64_t lapack_dsytrs(char uplo, int64_t n, int64_t nrhs, const double* a,
                      int64_t lda, const int64_t* ipiv, double* b, int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(dsytrs, const char*, const int64_t*, const int64_t*, const double*, const int64_t*, const int64_t*, double*, const int64_t*, int64_t*)(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}
int64_t lapack_csytrf(char uplo, int64_t n, complex<float>* a, int64_t lda,
                      int64_t* ipiv, complex<float>* work, int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(csytrf, const char*, const int64_t*, complex<float>*,
                   const int64_t*, int64_t*, complex<float>*, const int64_t*,
                   int64_t*)(&uplo, &n, a, &lda, ipiv, work, &lwork, &info);
    return info;
}
int64_t lapack_zsytrf(char uplo, int64_t n, complex<double>* a,
                      int64_t lda, int64_t* ipiv, complex<double>* work,
                      int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(zsytrf, const char*, const int64_t*, complex<double>*,
                   const int64_t*, int64_t*, complex<double>*, const int64_t*,
                   int64_t*)(&uplo, &n, a, &lda, ipiv, work, &lwork, &info);
    return info;
}
int64_t lapack_chetrf(char uplo, int64_t n, complex<float>* a, int64_t lda,
                      int64_t* ipiv, complex<float>* work, int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(chetrf, const char*, const int64_t*, complex<float>*,
                   const int64_t*, int64_t*, complex<float>*, const int64_t*,
                   int64_t*)(&uplo, &n, a, &lda, ipiv, work, &lwork, &info);
    return info;
}
int64_t lapack_zhetrf(char uplo, int64_t n, complex<double>* a,
                      int64_t lda, int64_t* ipiv, complex<double>* work,
                      int64_t lwork) {
    int64_t info = 0;
    TP_LAPACK_CALL(zhetrf, const char*, const int64_t*, complex<double>*,
                   const int64_t*, int64_t*, complex<double>*, const int64_t*,
                   int64_t*)(&uplo, &n, a, &lda, ipiv, work, &lwork, &info);
    return info;
}
int64_t lapack_csytrs(char uplo, int64_t n, int64_t nrhs,
                      const complex<float>* a, int64_t lda,
                      const int64_t* ipiv, complex<float>* b, int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(csytrs, const char*, const int64_t*, const int64_t*,
                   const complex<float>*, const int64_t*, const int64_t*,
                   complex<float>*, const int64_t*, int64_t*)(&uplo, &n, &nrhs,
                   a, &lda, ipiv, b, &ldb, &info);
    return info;
}
int64_t lapack_zsytrs(char uplo, int64_t n, int64_t nrhs,
                      const complex<double>* a, int64_t lda,
                      const int64_t* ipiv, complex<double>* b, int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(zsytrs, const char*, const int64_t*, const int64_t*,
                   const complex<double>*, const int64_t*, const int64_t*,
                   complex<double>*, const int64_t*, int64_t*)(&uplo, &n, &nrhs,
                   a, &lda, ipiv, b, &ldb, &info);
    return info;
}
int64_t lapack_chetrs(char uplo, int64_t n, int64_t nrhs,
                      const complex<float>* a, int64_t lda,
                      const int64_t* ipiv, complex<float>* b, int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(chetrs, const char*, const int64_t*, const int64_t*,
                   const complex<float>*, const int64_t*, const int64_t*,
                   complex<float>*, const int64_t*, int64_t*)(&uplo, &n, &nrhs,
                   a, &lda, ipiv, b, &ldb, &info);
    return info;
}
int64_t lapack_zhetrs(char uplo, int64_t n, int64_t nrhs,
                      const complex<double>* a, int64_t lda,
                      const int64_t* ipiv, complex<double>* b, int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(zhetrs, const char*, const int64_t*, const int64_t*,
                   const complex<double>*, const int64_t*, const int64_t*,
                   complex<double>*, const int64_t*, int64_t*)(&uplo, &n, &nrhs,
                   a, &lda, ipiv, b, &ldb, &info);
    return info;
}

}  // namespace cpu
}  // namespace tensorplay
