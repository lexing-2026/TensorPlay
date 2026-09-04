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
#include <complex>

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
int64_t lapack_cgetrf(int64_t m, int64_t n, std::complex<float>* a, int64_t lda,
                      int64_t* ipiv);
int64_t lapack_zgetrf(int64_t m, int64_t n, std::complex<double>* a, int64_t lda,
                      int64_t* ipiv);
int64_t lapack_cgetrs(char trans, int64_t n, int64_t nrhs,
                      const std::complex<float>* a, int64_t lda,
                      const int64_t* ipiv, std::complex<float>* b, int64_t ldb);
int64_t lapack_zgetrs(char trans, int64_t n, int64_t nrhs,
                      const std::complex<double>* a, int64_t lda,
                      const int64_t* ipiv, std::complex<double>* b, int64_t ldb);

int64_t lapack_spotrf(char uplo, int64_t n, float* a, int64_t lda);
int64_t lapack_dpotrf(char uplo, int64_t n, double* a, int64_t lda);

int64_t lapack_spotrs(char uplo, int64_t n, int64_t nrhs, const float* a,
                      int64_t lda, float* b, int64_t ldb);
int64_t lapack_dpotrs(char uplo, int64_t n, int64_t nrhs, const double* a,
                      int64_t lda, double* b, int64_t ldb);
int64_t lapack_cpotrf(char uplo, int64_t n, std::complex<float>* a, int64_t lda);
int64_t lapack_zpotrf(char uplo, int64_t n, std::complex<double>* a, int64_t lda);
int64_t lapack_cpotrs(char uplo, int64_t n, int64_t nrhs,
                      const std::complex<float>* a, int64_t lda,
                      std::complex<float>* b, int64_t ldb);
int64_t lapack_zpotrs(char uplo, int64_t n, int64_t nrhs,
                      const std::complex<double>* a, int64_t lda,
                      std::complex<double>* b, int64_t ldb);

int64_t lapack_spotri(char uplo, int64_t n, float* a, int64_t lda);
int64_t lapack_dpotri(char uplo, int64_t n, double* a, int64_t lda);
int64_t lapack_cpotri(char uplo, int64_t n, std::complex<float>* a, int64_t lda);
int64_t lapack_zpotri(char uplo, int64_t n, std::complex<double>* a, int64_t lda);

int64_t lapack_sgeqrf(int64_t m, int64_t n, float* a, int64_t lda, float* tau,
                      float* work, int64_t lwork);
int64_t lapack_dgeqrf(int64_t m, int64_t n, double* a, int64_t lda, double* tau,
                      double* work, int64_t lwork);
int64_t lapack_sorgqr(int64_t m, int64_t n, int64_t k, float* a, int64_t lda,
                      const float* tau, float* work, int64_t lwork);
int64_t lapack_dorgqr(int64_t m, int64_t n, int64_t k, double* a, int64_t lda,
                      const double* tau, double* work, int64_t lwork);
int64_t lapack_cgeqrf(int64_t m, int64_t n, std::complex<float>* a, int64_t lda,
                      std::complex<float>* tau, std::complex<float>* work,
                      int64_t lwork);
int64_t lapack_zgeqrf(int64_t m, int64_t n, std::complex<double>* a, int64_t lda,
                      std::complex<double>* tau, std::complex<double>* work,
                      int64_t lwork);
int64_t lapack_cungqr(int64_t m, int64_t n, int64_t k, std::complex<float>* a,
                      int64_t lda, const std::complex<float>* tau,
                      std::complex<float>* work, int64_t lwork);
int64_t lapack_zungqr(int64_t m, int64_t n, int64_t k, std::complex<double>* a,
                      int64_t lda, const std::complex<double>* tau,
                      std::complex<double>* work, int64_t lwork);

int64_t lapack_sgesdd(char jobz, int64_t m, int64_t n, float* a, int64_t lda,
                      float* s, float* u, int64_t ldu, float* vt, int64_t ldvt,
                      float* work, int64_t lwork, int64_t* iwork);
int64_t lapack_dgesdd(char jobz, int64_t m, int64_t n, double* a, int64_t lda,
                      double* s, double* u, int64_t ldu, double* vt,
                      int64_t ldvt, double* work, int64_t lwork, int64_t* iwork);
int64_t lapack_cgesdd(char jobz, int64_t m, int64_t n, std::complex<float>* a,
                      int64_t lda, float* s, std::complex<float>* u,
                      int64_t ldu, std::complex<float>* vt, int64_t ldvt,
                      std::complex<float>* work, int64_t lwork, float* rwork,
                      int64_t* iwork);
int64_t lapack_zgesdd(char jobz, int64_t m, int64_t n, std::complex<double>* a,
                      int64_t lda, double* s, std::complex<double>* u,
                      int64_t ldu, std::complex<double>* vt, int64_t ldvt,
                      std::complex<double>* work, int64_t lwork, double* rwork,
                      int64_t* iwork);

int64_t lapack_ssyevd(char jobz, char uplo, int64_t n, float* a, int64_t lda,
                      float* w, float* work, int64_t lwork, int64_t* iwork,
                      int64_t liwork);
int64_t lapack_dsyevd(char jobz, char uplo, int64_t n, double* a, int64_t lda,
                      double* w, double* work, int64_t lwork, int64_t* iwork,
                      int64_t liwork);
int64_t lapack_cheevd(char jobz, char uplo, int64_t n, std::complex<float>* a,
                      int64_t lda, float* w, std::complex<float>* work,
                      int64_t lwork, float* rwork, int64_t lrwork,
                      int64_t* iwork, int64_t liwork);
int64_t lapack_zheevd(char jobz, char uplo, int64_t n, std::complex<double>* a,
                      int64_t lda, double* w, std::complex<double>* work,
                      int64_t lwork, double* rwork, int64_t lrwork,
                      int64_t* iwork, int64_t liwork);

int64_t lapack_sgeev(char jobvl, char jobvr, int64_t n, float* a, int64_t lda,
                     float* wr, float* wi, float* vl, int64_t ldvl, float* vr,
                     int64_t ldvr, float* work, int64_t lwork);
int64_t lapack_dgeev(char jobvl, char jobvr, int64_t n, double* a, int64_t lda,
                     double* wr, double* wi, double* vl, int64_t ldvl,
                     double* vr, int64_t ldvr, double* work, int64_t lwork);
int64_t lapack_cgeev(char jobvl, char jobvr, int64_t n, std::complex<float>* a,
                     int64_t lda, std::complex<float>* w,
                     std::complex<float>* vl, int64_t ldvl,
                     std::complex<float>* vr, int64_t ldvr,
                     std::complex<float>* work, int64_t lwork, float* rwork);
int64_t lapack_zgeev(char jobvl, char jobvr, int64_t n, std::complex<double>* a,
                     int64_t lda, std::complex<double>* w,
                     std::complex<double>* vl, int64_t ldvl,
                     std::complex<double>* vr, int64_t ldvr,
                     std::complex<double>* work, int64_t lwork, double* rwork);

int64_t lapack_strtrs(char uplo, char transa, char diag, int64_t n,
                      int64_t nrhs, const float* a, int64_t lda, float* b,
                      int64_t ldb);
int64_t lapack_dtrtrs(char uplo, char transa, char diag, int64_t n,
                      int64_t nrhs, const double* a, int64_t lda, double* b,
                      int64_t ldb);
int64_t lapack_ctrtrs(char uplo, char transa, char diag, int64_t n,
                      int64_t nrhs, const std::complex<float>* a, int64_t lda,
                      std::complex<float>* b, int64_t ldb);
int64_t lapack_ztrtrs(char uplo, char transa, char diag, int64_t n,
                      int64_t nrhs, const std::complex<double>* a, int64_t lda,
                      std::complex<double>* b, int64_t ldb);

// CBLAS triangular multiply-solve: X = op(A)^-1 * alpha * B (side) in the
// given order (0 = row-major, 1 = column-major).  The strtrs Fortran entry
// rejects every UPLO value in the bundled scipy-openblas64 wheel, so the
// triangular solves route through the working CBLAS trsm instead.
void lapack_strsm(int64_t order, int64_t side, int64_t uplo, int64_t trans,
                  int64_t diag, int64_t m, int64_t n, float alpha,
                  const float* a, int64_t lda, float* b, int64_t ldb);
void lapack_dtrsm(int64_t order, int64_t side, int64_t uplo, int64_t trans,
                  int64_t diag, int64_t m, int64_t n, double alpha,
                  const double* a, int64_t lda, double* b, int64_t ldb);
void lapack_ctrsm(int64_t order, int64_t side, int64_t uplo, int64_t trans,
                  int64_t diag, int64_t m, int64_t n,
                  const std::complex<float>* alpha,
                  const std::complex<float>* a, int64_t lda,
                  std::complex<float>* b, int64_t ldb);
void lapack_ztrsm(int64_t order, int64_t side, int64_t uplo, int64_t trans,
                  int64_t diag, int64_t m, int64_t n,
                  const std::complex<double>* alpha,
                  const std::complex<double>* a, int64_t lda,
                  std::complex<double>* b, int64_t ldb);

int64_t lapack_sgels(char trans, int64_t m, int64_t n, int64_t nrhs, float* a,
                     int64_t lda, float* b, int64_t ldb, float* work,
                     int64_t lwork);
int64_t lapack_dgels(char trans, int64_t m, int64_t n, int64_t nrhs, double* a,
                     int64_t lda, double* b, int64_t ldb, double* work,
                     int64_t lwork);
int64_t lapack_cgels(char trans, int64_t m, int64_t n, int64_t nrhs,
                     std::complex<float>* a, int64_t lda,
                     std::complex<float>* b, int64_t ldb,
                     std::complex<float>* work, int64_t lwork);
int64_t lapack_zgels(char trans, int64_t m, int64_t n, int64_t nrhs,
                     std::complex<double>* a, int64_t lda,
                     std::complex<double>* b, int64_t ldb,
                     std::complex<double>* work, int64_t lwork);

int64_t lapack_sgelsy(int64_t m, int64_t n, int64_t nrhs, float* a,
                      int64_t lda, float* b, int64_t ldb, int64_t* jpvt,
                      float rcond, int64_t* rank, float* work, int64_t lwork);
int64_t lapack_dgelsy(int64_t m, int64_t n, int64_t nrhs, double* a,
                      int64_t lda, double* b, int64_t ldb, int64_t* jpvt,
                      double rcond, int64_t* rank, double* work, int64_t lwork);
int64_t lapack_cgelsy(int64_t m, int64_t n, int64_t nrhs,
                      std::complex<float>* a, int64_t lda,
                      std::complex<float>* b, int64_t ldb, int64_t* jpvt,
                      float rcond, int64_t* rank, std::complex<float>* work,
                      int64_t lwork, float* rwork);
int64_t lapack_zgelsy(int64_t m, int64_t n, int64_t nrhs,
                      std::complex<double>* a, int64_t lda,
                      std::complex<double>* b, int64_t ldb, int64_t* jpvt,
                      double rcond, int64_t* rank, std::complex<double>* work,
                      int64_t lwork, double* rwork);

int64_t lapack_sgelsd(int64_t m, int64_t n, int64_t nrhs, float* a,
                      int64_t lda, float* b, int64_t ldb, float* s,
                      float rcond, int64_t* rank, float* work, int64_t lwork,
                      int64_t* iwork);
int64_t lapack_dgelsd(int64_t m, int64_t n, int64_t nrhs, double* a,
                      int64_t lda, double* b, int64_t ldb, double* s,
                      double rcond, int64_t* rank, double* work, int64_t lwork,
                      int64_t* iwork);
int64_t lapack_cgelsd(int64_t m, int64_t n, int64_t nrhs,
                      std::complex<float>* a, int64_t lda,
                      std::complex<float>* b, int64_t ldb, float* s,
                      float rcond, int64_t* rank, std::complex<float>* work,
                      int64_t lwork, float* rwork, int64_t* iwork);
int64_t lapack_zgelsd(int64_t m, int64_t n, int64_t nrhs,
                      std::complex<double>* a, int64_t lda,
                      std::complex<double>* b, int64_t ldb, double* s,
                      double rcond, int64_t* rank, std::complex<double>* work,
                      int64_t lwork, double* rwork, int64_t* iwork);

int64_t lapack_sgelss(int64_t m, int64_t n, int64_t nrhs, float* a,
                      int64_t lda, float* b, int64_t ldb, float* s,
                      float rcond, int64_t* rank, float* work, int64_t lwork);
int64_t lapack_dgelss(int64_t m, int64_t n, int64_t nrhs, double* a,
                      int64_t lda, double* b, int64_t ldb, double* s,
                      double rcond, int64_t* rank, double* work, int64_t lwork);
int64_t lapack_cgelss(int64_t m, int64_t n, int64_t nrhs,
                      std::complex<float>* a, int64_t lda,
                      std::complex<float>* b, int64_t ldb, float* s,
                      float rcond, int64_t* rank, std::complex<float>* work,
                      int64_t lwork, float* rwork);
int64_t lapack_zgelss(int64_t m, int64_t n, int64_t nrhs,
                      std::complex<double>* a, int64_t lda,
                      std::complex<double>* b, int64_t ldb, double* s,
                      double rcond, int64_t* rank, std::complex<double>* work,
                      int64_t lwork, double* rwork);

int64_t lapack_ssytrf(char uplo, int64_t n, float* a, int64_t lda, int64_t* ipiv,
                      float* work, int64_t lwork);
int64_t lapack_dsytrf(char uplo, int64_t n, double* a, int64_t lda,
                      int64_t* ipiv, double* work, int64_t lwork);
int64_t lapack_ssytrs(char uplo, int64_t n, int64_t nrhs, const float* a,
                      int64_t lda, const int64_t* ipiv, float* b, int64_t ldb);
int64_t lapack_dsytrs(char uplo, int64_t n, int64_t nrhs, const double* a,
                      int64_t lda, const int64_t* ipiv, double* b, int64_t ldb);
int64_t lapack_csytrf(char uplo, int64_t n, std::complex<float>* a, int64_t lda,
                      int64_t* ipiv, std::complex<float>* work, int64_t lwork);
int64_t lapack_zsytrf(char uplo, int64_t n, std::complex<double>* a,
                      int64_t lda, int64_t* ipiv, std::complex<double>* work,
                      int64_t lwork);
int64_t lapack_chetrf(char uplo, int64_t n, std::complex<float>* a, int64_t lda,
                      int64_t* ipiv, std::complex<float>* work, int64_t lwork);
int64_t lapack_zhetrf(char uplo, int64_t n, std::complex<double>* a,
                      int64_t lda, int64_t* ipiv, std::complex<double>* work,
                      int64_t lwork);
int64_t lapack_csytrs(char uplo, int64_t n, int64_t nrhs,
                      const std::complex<float>* a, int64_t lda,
                      const int64_t* ipiv, std::complex<float>* b, int64_t ldb);
int64_t lapack_zsytrs(char uplo, int64_t n, int64_t nrhs,
                      const std::complex<double>* a, int64_t lda,
                      const int64_t* ipiv, std::complex<double>* b, int64_t ldb);
int64_t lapack_chetrs(char uplo, int64_t n, int64_t nrhs,
                      const std::complex<float>* a, int64_t lda,
                      const int64_t* ipiv, std::complex<float>* b, int64_t ldb);
int64_t lapack_zhetrs(char uplo, int64_t n, int64_t nrhs,
                      const std::complex<double>* a, int64_t lda,
                      const int64_t* ipiv, std::complex<double>* b, int64_t ldb);

}  // namespace cpu
}  // namespace tensorplay
