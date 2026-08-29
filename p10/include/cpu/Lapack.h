// Runtime LAPACK binding (ILP64).
//
// runtime.  The default provider is numpy's bundled scipy-openblas wheel
// library (libscipy_openblas64_*), which exports the full LAPACK with 64-bit
// integers (`scipy_<name>_64_`).  `TP_LAPACK_LIB` may point at any other
// ILP64 OpenBLAS build exposing the same symbols.
//
// The raw entries follow the Fortran ABI (everything by pointer, no return
// value); the typed `lapack_*` helpers below wrap them with C++ value
// BatchLinearAlgebra.h so the kernel ports read the same way.

#pragma once

#include <cstdint>

namespace tensorplay {
namespace cpu {

// True once the shared library has been located and every routine used by the
// linalg kernels has been resolved.  Thread-safe; lazily initializes.
bool lapack_available();

// Raises RuntimeError with setup guidance when !lapack_available().
void require_lapack(const char* api_name);

int64_t lapack_sgetrf(int64_t m, int64_t n, float* a, int64_t lda, int64_t* ipiv);
int64_t lapack_dgetrf(int64_t m, int64_t n, double* a, int64_t lda, int64_t* ipiv);
int64_t lapack_sgetrs(char trans, int64_t n, int64_t nrhs, const float* a,
                      int64_t lda, const int64_t* ipiv, float* b, int64_t ldb);
int64_t lapack_dgetrs(char trans, int64_t n, int64_t nrhs, const double* a,
                      int64_t lda, const int64_t* ipiv, double* b, int64_t ldb);

int64_t lapack_spotrf(char uplo, int64_t n, float* a, int64_t lda);
int64_t lapack_dpotrf(char uplo, int64_t n, double* a, int64_t lda);

int64_t lapack_sgeqrf(int64_t m, int64_t n, float* a, int64_t lda, float* tau,
                      float* work, int64_t lwork);
int64_t lapack_dgeqrf(int64_t m, int64_t n, double* a, int64_t lda, double* tau,
                      double* work, int64_t lwork);
int64_t lapack_sorgqr(int64_t m, int64_t n, int64_t k, float* a, int64_t lda,
                      const float* tau, float* work, int64_t lwork);
int64_t lapack_dorgqr(int64_t m, int64_t n, int64_t k, double* a, int64_t lda,
                      const double* tau, double* work, int64_t lwork);

int64_t lapack_sgesdd(char jobz, int64_t m, int64_t n, float* a, int64_t lda,
                      float* s, float* u, int64_t ldu, float* vt, int64_t ldvt,
                      float* work, int64_t lwork, int64_t* iwork);
int64_t lapack_dgesdd(char jobz, int64_t m, int64_t n, double* a, int64_t lda,
                      double* s, double* u, int64_t ldu, double* vt,
                      int64_t ldvt, double* work, int64_t lwork, int64_t* iwork);

int64_t lapack_ssyevd(char jobz, char uplo, int64_t n, float* a, int64_t lda,
                      float* w, float* work, int64_t lwork, int64_t* iwork,
                      int64_t liwork);
int64_t lapack_dsyevd(char jobz, char uplo, int64_t n, double* a, int64_t lda,
                      double* w, double* work, int64_t lwork, int64_t* iwork,
                      int64_t liwork);

int64_t lapack_sgeev(char jobvl, char jobvr, int64_t n, float* a, int64_t lda,
                     float* wr, float* wi, float* vl, int64_t ldvl, float* vr,
                     int64_t ldvr, float* work, int64_t lwork);
int64_t lapack_dgeev(char jobvl, char jobvr, int64_t n, double* a, int64_t lda,
                     double* wr, double* wi, double* vl, int64_t ldvl,
                     double* vr, int64_t ldvr, double* work, int64_t lwork);

int64_t lapack_strtrs(char side, char uplo, char transa, char diag, int64_t n,
                      int64_t nrhs, const float* a, int64_t lda, float* b,
                      int64_t ldb);
int64_t lapack_dtrtrs(char side, char uplo, char transa, char diag, int64_t n,
                      int64_t nrhs, const double* a, int64_t lda, double* b,
                      int64_t ldb);

int64_t lapack_sgels(char trans, int64_t m, int64_t n, int64_t nrhs, float* a,
                     int64_t lda, float* b, int64_t ldb, float* work,
                     int64_t lwork);
int64_t lapack_dgels(char trans, int64_t m, int64_t n, int64_t nrhs, double* a,
                     int64_t lda, double* b, int64_t ldb, double* work,
                     int64_t lwork);

int64_t lapack_ssytrf(char uplo, int64_t n, float* a, int64_t lda, int64_t* ipiv,
                      float* work, int64_t lwork);
int64_t lapack_dsytrf(char uplo, int64_t n, double* a, int64_t lda,
                      int64_t* ipiv, double* work, int64_t lwork);
int64_t lapack_ssytrs(char uplo, int64_t n, int64_t nrhs, const float* a,
                      int64_t lda, const int64_t* ipiv, float* b, int64_t ldb);
int64_t lapack_dsytrs(char uplo, int64_t n, int64_t nrhs, const double* a,
                      int64_t lda, const int64_t* ipiv, double* b, int64_t ldb);

}  // namespace cpu
}  // namespace tensorplay
