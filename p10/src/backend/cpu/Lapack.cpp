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
F g_spotrf, g_dpotrf;
F g_spotrs, g_dpotrs;
F g_spotri, g_dpotri;
F g_sgeqrf, g_dgeqrf, g_sorgqr, g_dorgqr;
F g_sgesdd, g_dgesdd;
F g_ssyevd, g_dsyevd;
F g_sgeev, g_dgeev;
F g_strtrs, g_dtrtrs;
F g_strsm, g_dtrsm;
F g_sgels, g_dgels;
F g_ssytrf, g_dsytrf, g_ssytrs, g_dsytrs;

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

void* resolve_one(void* handle, const char* base) {
    char buf[128];
    const char* patterns[] = {"scipy_%s_64_", "%s_64_", "%s_", "scipy_%s64_"};
    for (const char* pattern : patterns) {
        std::snprintf(buf, sizeof(buf), pattern, base);
        if (void* p = sym(handle, buf)) return p;
    }
    return nullptr;
}

bool resolve_all(void* handle) {
    struct Pair { F& slot; const char* name; };
    const Pair pairs[] = {
        {g_sgetrf, "sgetrf"}, {g_dgetrf, "dgetrf"},
        {g_sgetrs, "sgetrs"}, {g_dgetrs, "dgetrs"},
        {g_spotrf, "spotrf"}, {g_dpotrf, "dpotrf"},
        {g_spotrs, "spotrs"}, {g_dpotrs, "dpotrs"},
        {g_spotri, "spotri"}, {g_dpotri, "dpotri"},
        // LAPACK's generic-name routines (geqrf/gesdd/syevd/...) carry the
        // s/d precision prefix in their symbols; the scipy-openblas wheel
        // exports them as scipy_<s|d><name>_64_.
        {g_sgeqrf, "sgeqrf"}, {g_dgeqrf, "dgeqrf"},
        {g_sorgqr, "sorgqr"}, {g_dorgqr, "dorgqr"},
        {g_sgesdd, "sgesdd"}, {g_dgesdd, "dgesdd"},
        {g_ssyevd, "ssyevd"}, {g_dsyevd, "dsyevd"},
        {g_sgeev,  "sgeev"},  {g_dgeev,  "dgeev"},
        {g_strtrs, "strtrs"}, {g_dtrtrs, "dtrtrs"},
        {g_strsm, "cblas_strsm"}, {g_dtrsm, "cblas_dtrsm"},
        {g_sgels,  "sgels"},  {g_dgels,  "dgels"},
        {g_ssytrf, "ssytrf"}, {g_dsytrf, "dsytrf"},
        {g_ssytrs, "ssytrs"}, {g_dsytrs, "dsytrs"},
    };
    for (const auto& p : pairs) {
        p.slot = resolve_one(handle, p.name);
        if (!p.slot) return false;
    }
    return true;
}

void* find_library() {
#ifdef _WIN32
    if (const char* env = std::getenv("TP_LAPACK_LIB")) {
        if (LibHandle h = open_lib(env)) return h;
    }
    namespace fs = std::filesystem;
    std::vector<std::string> bases;
    if (const char* cp = std::getenv("CONDA_PREFIX")) bases.push_back(cp);
    if (const char* vp = std::getenv("VIRTUAL_ENV")) bases.push_back(vp);
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
    void* handle = find_library();
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
int64_t lapack_strtrs(char side, char uplo, char transa, char diag, int64_t n,
                      int64_t nrhs, const float* a, int64_t lda, float* b,
                      int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(strtrs, const char*, const char*, const char*, const char*, const int64_t*, const int64_t*, const float*, const int64_t*, float*, const int64_t*, int64_t*)(&side, &uplo, &transa, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}
int64_t lapack_dtrtrs(char side, char uplo, char transa, char diag, int64_t n,
                      int64_t nrhs, const double* a, int64_t lda, double* b,
                      int64_t ldb) {
    int64_t info = 0;
    TP_LAPACK_CALL(dtrtrs, const char*, const char*, const char*, const char*, const int64_t*, const int64_t*, const double*, const int64_t*, double*, const int64_t*, int64_t*)(&side, &uplo, &transa, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);
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

}  // namespace cpu
}  // namespace tensorplay
