#include "Tensor.h"
#include "Complex.h"
#include "SparseKernels.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "OpMathType.h"
#include "TypePromotion.h"
#include "TypeProperties.h"
#include "TensorIterator.h"
#include "TensorIteratorOps.h"
#include "Utils.h"
#include "Exception.h"
#include "Parallel.h"
#include "OneDNNContext.h"
#include "Allocator.h"
#include "GradMode.h"
#include "OutWrite.h"
#include "cpu/VecComplex.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#ifdef USE_ONEDNN
#include "dnnl.hpp"
#endif

#ifdef USE_MKL
#include <mkl.h>
#elif defined(USE_BLAS)
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tensorplay {
namespace cpu {
using namespace tensorplay::parallel;

// Forward declarations for the scalar fallback used by the fused kernel.
Tensor add_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha);
Tensor mul_scalar_kernel(const Tensor& self, const Scalar& other);
Tensor& relu_inplace_kernel(Tensor& self);

// --- AVX-512 runtime-dispatched contiguous binary kernels -------------------
// Zen4-class CPUs run zmm natively; the build carries no global -mavx512f so
// these carry their own target attribute and are gated by cpuid at runtime.
// native fused multiply-add path, while mul/div are single IEEE ops.

// Operation codes shared by the runtime-dispatched x86 kernels below and the
// generic binary callers later in this file; the enum itself is portable.
enum : int { BIN_ADD = 0, BIN_MUL = 1, BIN_DIV = 2 };

#if defined(__x86_64__)
namespace {

inline bool cpu_has_avx512() {
    static const bool ok = __builtin_cpu_supports("avx512f") != 0 &&
                           __builtin_cpu_supports("avx512vl") != 0 &&
                           __builtin_cpu_supports("avx512dq") != 0;
    return ok;
}

// Client cores without the 512-bit datapath still carry AVX2+FMA; the
// broadcast kernels below run 256-bit loops there.
inline bool cpu_has_avx2f() {
    static const bool ok = __builtin_cpu_supports("avx2") != 0 &&
                           __builtin_cpu_supports("fma") != 0;
    return ok;
}

__attribute__((target("avx512f,fma")))
void binary_f32_avx512(int code, const float* a, const float* b, float* y,
                       int64_t n, float alpha) {
    const __m512 va = _mm512_set1_ps(alpha);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(a + i);
        __m512 w = _mm512_loadu_ps(b + i);
        __m512 r;
        switch (code) {
            case BIN_ADD:
                r = alpha == 1.0f   ? _mm512_add_ps(x, w)
                  : alpha == -1.0f  ? _mm512_sub_ps(x, w)
                                    : _mm512_fmadd_ps(va, w, x);
                break;
            case BIN_MUL: r = _mm512_mul_ps(x, w); break;
            default:      r = _mm512_div_ps(x, w); break;
        }
        _mm512_storeu_ps(y + i, r);
    }
    for (; i < n; ++i) {
        switch (code) {
            case BIN_ADD: y[i] = a[i] + alpha * b[i]; break;
            case BIN_MUL: y[i] = a[i] * b[i]; break;
            default:      y[i] = a[i] / b[i]; break;
        }
    }
}

__attribute__((target("avx512f")))
void binary_f64_avx512(int code, const double* a, const double* b, double* y,
                       int64_t n, double alpha) {
    const __m512d va = _mm512_set1_pd(alpha);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d x = _mm512_loadu_pd(a + i);
        __m512d w = _mm512_loadu_pd(b + i);
        __m512d r;
        switch (code) {
            case BIN_ADD: r = _mm512_add_pd(x, _mm512_mul_pd(va, w)); break;
            case BIN_MUL: r = _mm512_mul_pd(x, w); break;
            default:      r = _mm512_div_pd(x, w); break;
        }
        _mm512_storeu_pd(y + i, r);
    }
    for (; i < n; ++i) {
        switch (code) {
            case BIN_ADD: y[i] = a[i] + alpha * b[i]; break;
            case BIN_MUL: y[i] = a[i] * b[i]; break;
            default:      y[i] = a[i] / b[i]; break;
        }
    }
}

__attribute__((target("avx512f")))
void add_f32_avx512(const float* a, const float* b, float* y, int64_t n) {
    int64_t i = 0;
    for (; i + 64 <= n; i += 64) {
        _mm512_storeu_ps(y + i, _mm512_add_ps(_mm512_loadu_ps(a + i),
                                              _mm512_loadu_ps(b + i)));
        _mm512_storeu_ps(y + i + 16, _mm512_add_ps(_mm512_loadu_ps(a + i + 16),
                                                    _mm512_loadu_ps(b + i + 16)));
        _mm512_storeu_ps(y + i + 32, _mm512_add_ps(_mm512_loadu_ps(a + i + 32),
                                                    _mm512_loadu_ps(b + i + 32)));
        _mm512_storeu_ps(y + i + 48, _mm512_add_ps(_mm512_loadu_ps(a + i + 48),
                                                    _mm512_loadu_ps(b + i + 48)));
    }
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(y + i, _mm512_add_ps(_mm512_loadu_ps(a + i),
                                              _mm512_loadu_ps(b + i)));
    }
    for (; i < n; ++i) y[i] = a[i] + b[i];
}

__attribute__((target("avx512f")))
void sub_f32_avx512(const float* a, const float* b, float* y, int64_t n) {
    // Dedicated row for alpha == -1 (x - y): keeps the subtraction on the
    // same unrolled schedule as addition instead of a fused mul-add chain,
    // so subtracting large contiguous tensors stays pure bandwidth.
    int64_t i = 0;
    for (; i + 64 <= n; i += 64) {
        _mm512_storeu_ps(y + i, _mm512_sub_ps(_mm512_loadu_ps(a + i),
                                              _mm512_loadu_ps(b + i)));
        _mm512_storeu_ps(y + i + 16, _mm512_sub_ps(_mm512_loadu_ps(a + i + 16),
                                                    _mm512_loadu_ps(b + i + 16)));
        _mm512_storeu_ps(y + i + 32, _mm512_sub_ps(_mm512_loadu_ps(a + i + 32),
                                                    _mm512_loadu_ps(b + i + 32)));
        _mm512_storeu_ps(y + i + 48, _mm512_sub_ps(_mm512_loadu_ps(a + i + 48),
                                                    _mm512_loadu_ps(b + i + 48)));
    }
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(y + i, _mm512_sub_ps(_mm512_loadu_ps(a + i),
                                              _mm512_loadu_ps(b + i)));
    }
    for (; i < n; ++i) y[i] = a[i] - b[i];
}

__attribute__((target("avx512f")))
void add_f64_avx512(const double* a, const double* b, double* y, int64_t n) {
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        _mm512_storeu_pd(y + i, _mm512_add_pd(_mm512_loadu_pd(a + i),
                                              _mm512_loadu_pd(b + i)));
        _mm512_storeu_pd(y + i + 8, _mm512_add_pd(_mm512_loadu_pd(a + i + 8),
                                                  _mm512_loadu_pd(b + i + 8)));
        _mm512_storeu_pd(y + i + 16, _mm512_add_pd(_mm512_loadu_pd(a + i + 16),
                                                   _mm512_loadu_pd(b + i + 16)));
        _mm512_storeu_pd(y + i + 24, _mm512_add_pd(_mm512_loadu_pd(a + i + 24),
                                                   _mm512_loadu_pd(b + i + 24)));
    }
    for (; i + 8 <= n; i += 8) {
        _mm512_storeu_pd(y + i, _mm512_add_pd(_mm512_loadu_pd(a + i),
                                              _mm512_loadu_pd(b + i)));
    }
    for (; i < n; ++i) y[i] = a[i] + b[i];
}

__attribute__((target("avx512f")))
void scalar_f32_avx512(int code, const float* a, float* y, int64_t n,
                       float scalar) {
    const __m512 vscalar = _mm512_set1_ps(scalar);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 x = _mm512_loadu_ps(a + i);
        __m512 r;
        switch (code) {
            case BIN_ADD: r = _mm512_add_ps(x, vscalar); break;
            case BIN_MUL: r = _mm512_mul_ps(x, vscalar); break;
            default:      r = _mm512_div_ps(x, vscalar); break;
        }
        _mm512_storeu_ps(y + i, r);
    }
    for (; i < n; ++i) {
        y[i] = code == BIN_ADD ? a[i] + scalar
             : code == BIN_MUL ? a[i] * scalar : a[i] / scalar;
    }
}

__attribute__((target("avx512f")))
void scalar_f64_avx512(int code, const double* a, double* y, int64_t n,
                       double scalar) {
    const __m512d vscalar = _mm512_set1_pd(scalar);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512d x = _mm512_loadu_pd(a + i);
        __m512d r;
        switch (code) {
            case BIN_ADD: r = _mm512_add_pd(x, vscalar); break;
            case BIN_MUL: r = _mm512_mul_pd(x, vscalar); break;
            default:      r = _mm512_div_pd(x, vscalar); break;
        }
        _mm512_storeu_pd(y + i, r);
    }
    for (; i < n; ++i) {
        y[i] = code == BIN_ADD ? a[i] + scalar
             : code == BIN_MUL ? a[i] * scalar : a[i] / scalar;
    }
}

// 256-bit twins of the scalar kernels above for client cores whose 512-bit
// datapath is emulated: same op folding, half the lanes.
__attribute__((target("avx2")))
void scalar_f32_avx256(int code, const float* a, float* y, int64_t n,
                       float scalar) {
    const __m256 vscalar = _mm256_set1_ps(scalar);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 x = _mm256_loadu_ps(a + i);
        switch (code) {
            case BIN_ADD: _mm256_storeu_ps(y + i, _mm256_add_ps(x, vscalar)); break;
            case BIN_MUL: _mm256_storeu_ps(y + i, _mm256_mul_ps(x, vscalar)); break;
            default:      _mm256_storeu_ps(y + i, _mm256_div_ps(x, vscalar)); break;
        }
    }
    for (; i < n; ++i) {
        y[i] = code == BIN_ADD ? a[i] + scalar
             : code == BIN_MUL ? a[i] * scalar : a[i] / scalar;
    }
}

__attribute__((target("avx2")))
void scalar_f64_avx256(int code, const double* a, double* y, int64_t n,
                       double scalar) {
    const __m256d vscalar = _mm256_set1_pd(scalar);
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d x = _mm256_loadu_pd(a + i);
        switch (code) {
            case BIN_ADD: _mm256_storeu_pd(y + i, _mm256_add_pd(x, vscalar)); break;
            case BIN_MUL: _mm256_storeu_pd(y + i, _mm256_mul_pd(x, vscalar)); break;
            default:      _mm256_storeu_pd(y + i, _mm256_div_pd(x, vscalar)); break;
        }
    }
    for (; i < n; ++i) {
        y[i] = code == BIN_ADD ? a[i] + scalar
             : code == BIN_MUL ? a[i] * scalar : a[i] / scalar;
    }
}

// Broadcast variants of the dense kernels above: one operand is constant
// along the inner run (its TensorIterator stride is zero), so it loads once
// and the dense side streams at full width.

__attribute__((target("avx512f,fma")))
void binary_f32_avx512_scalar_b(int code, const float* a, float b, float* y,
                                int64_t n, float alpha) {
    const __m512 vb = _mm512_set1_ps(b);
    const __m512 valpha = _mm512_set1_ps(alpha);
    int64_t i = 0;
    // The op selection stays outside the streaming loop: a branch per vector
    // costs issue slots at the rates these loops run.
    if (code == BIN_MUL) {
        for (; i + 32 <= n; i += 32) {
            _mm512_storeu_ps(y + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), vb));
            _mm512_storeu_ps(y + i + 16,
                             _mm512_mul_ps(_mm512_loadu_ps(a + i + 16), vb));
        }
        for (; i + 16 <= n; i += 16)
            _mm512_storeu_ps(y + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), vb));
    } else if (code == BIN_DIV) {
        for (; i + 32 <= n; i += 32) {
            _mm512_storeu_ps(y + i, _mm512_div_ps(_mm512_loadu_ps(a + i), vb));
            _mm512_storeu_ps(y + i + 16,
                             _mm512_div_ps(_mm512_loadu_ps(a + i + 16), vb));
        }
        for (; i + 16 <= n; i += 16)
            _mm512_storeu_ps(y + i, _mm512_div_ps(_mm512_loadu_ps(a + i), vb));
    } else if (alpha == 1.0f) {
        for (; i + 32 <= n; i += 32) {
            _mm512_storeu_ps(y + i, _mm512_add_ps(_mm512_loadu_ps(a + i), vb));
            _mm512_storeu_ps(y + i + 16,
                             _mm512_add_ps(_mm512_loadu_ps(a + i + 16), vb));
        }
        for (; i + 16 <= n; i += 16)
            _mm512_storeu_ps(y + i, _mm512_add_ps(_mm512_loadu_ps(a + i), vb));
    } else if (alpha == -1.0f) {
        for (; i + 32 <= n; i += 32) {
            _mm512_storeu_ps(y + i, _mm512_sub_ps(_mm512_loadu_ps(a + i), vb));
            _mm512_storeu_ps(y + i + 16,
                             _mm512_sub_ps(_mm512_loadu_ps(a + i + 16), vb));
        }
        for (; i + 16 <= n; i += 16)
            _mm512_storeu_ps(y + i, _mm512_sub_ps(_mm512_loadu_ps(a + i), vb));
    } else {
        for (; i + 32 <= n; i += 32) {
            _mm512_storeu_ps(
                y + i, _mm512_fmadd_ps(valpha, vb, _mm512_loadu_ps(a + i)));
            _mm512_storeu_ps(y + i + 16,
                             _mm512_fmadd_ps(valpha, vb,
                                             _mm512_loadu_ps(a + i + 16)));
        }
        for (; i + 16 <= n; i += 16)
            _mm512_storeu_ps(
                y + i, _mm512_fmadd_ps(valpha, vb, _mm512_loadu_ps(a + i)));
    }
    for (; i < n; ++i) {
        switch (code) {
            case BIN_ADD: y[i] = a[i] + alpha * b; break;
            case BIN_MUL: y[i] = a[i] * b; break;
            default:      y[i] = a[i] / b; break;
        }
    }
}

__attribute__((target("avx512f,fma")))
void binary_f32_avx512_scalar_a(int code, float a, const float* b, float* y,
                                int64_t n, float alpha) {
    const __m512 va = _mm512_set1_ps(a);
    const __m512 valpha = _mm512_set1_ps(alpha);
    int64_t i = 0;
    if (code == BIN_MUL) {
        for (; i + 32 <= n; i += 32) {
            _mm512_storeu_ps(y + i, _mm512_mul_ps(va, _mm512_loadu_ps(b + i)));
            _mm512_storeu_ps(y + i + 16,
                             _mm512_mul_ps(va, _mm512_loadu_ps(b + i + 16)));
        }
        for (; i + 16 <= n; i += 16)
            _mm512_storeu_ps(y + i, _mm512_mul_ps(va, _mm512_loadu_ps(b + i)));
    } else if (code == BIN_DIV) {
        for (; i + 32 <= n; i += 32) {
            _mm512_storeu_ps(y + i, _mm512_div_ps(va, _mm512_loadu_ps(b + i)));
            _mm512_storeu_ps(y + i + 16,
                             _mm512_div_ps(va, _mm512_loadu_ps(b + i + 16)));
        }
        for (; i + 16 <= n; i += 16)
            _mm512_storeu_ps(y + i, _mm512_div_ps(va, _mm512_loadu_ps(b + i)));
    } else if (alpha == 1.0f) {
        for (; i + 32 <= n; i += 32) {
            _mm512_storeu_ps(y + i, _mm512_add_ps(va, _mm512_loadu_ps(b + i)));
            _mm512_storeu_ps(y + i + 16,
                             _mm512_add_ps(va, _mm512_loadu_ps(b + i + 16)));
        }
        for (; i + 16 <= n; i += 16)
            _mm512_storeu_ps(y + i, _mm512_add_ps(va, _mm512_loadu_ps(b + i)));
    } else if (alpha == -1.0f) {
        for (; i + 32 <= n; i += 32) {
            _mm512_storeu_ps(y + i, _mm512_sub_ps(va, _mm512_loadu_ps(b + i)));
            _mm512_storeu_ps(y + i + 16,
                             _mm512_sub_ps(va, _mm512_loadu_ps(b + i + 16)));
        }
        for (; i + 16 <= n; i += 16)
            _mm512_storeu_ps(y + i, _mm512_sub_ps(va, _mm512_loadu_ps(b + i)));
    } else {
        for (; i + 32 <= n; i += 32) {
            _mm512_storeu_ps(
                y + i, _mm512_fmadd_ps(valpha, _mm512_loadu_ps(b + i), va));
            _mm512_storeu_ps(y + i + 16,
                             _mm512_fmadd_ps(valpha,
                                             _mm512_loadu_ps(b + i + 16), va));
        }
        for (; i + 16 <= n; i += 16)
            _mm512_storeu_ps(
                y + i, _mm512_fmadd_ps(valpha, _mm512_loadu_ps(b + i), va));
    }
    for (; i < n; ++i) {
        switch (code) {
            case BIN_ADD: y[i] = a + alpha * b[i]; break;
            case BIN_MUL: y[i] = a * b[i]; break;
            default:      y[i] = a / b[i]; break;
        }
    }
}

__attribute__((target("avx512f,fma")))
void binary_f64_avx512_scalar_b(int code, const double* a, double b, double* y,
                                int64_t n, double alpha) {
    const __m512d vb = _mm512_set1_pd(b);
    const __m512d valpha = _mm512_set1_pd(alpha);
    int64_t i = 0;
    if (code == BIN_MUL) {
        for (; i + 16 <= n; i += 16) {
            _mm512_storeu_pd(y + i, _mm512_mul_pd(_mm512_loadu_pd(a + i), vb));
            _mm512_storeu_pd(y + i + 8,
                             _mm512_mul_pd(_mm512_loadu_pd(a + i + 8), vb));
        }
        for (; i + 8 <= n; i += 8)
            _mm512_storeu_pd(y + i, _mm512_mul_pd(_mm512_loadu_pd(a + i), vb));
    } else if (code == BIN_DIV) {
        for (; i + 16 <= n; i += 16) {
            _mm512_storeu_pd(y + i, _mm512_div_pd(_mm512_loadu_pd(a + i), vb));
            _mm512_storeu_pd(y + i + 8,
                             _mm512_div_pd(_mm512_loadu_pd(a + i + 8), vb));
        }
        for (; i + 8 <= n; i += 8)
            _mm512_storeu_pd(y + i, _mm512_div_pd(_mm512_loadu_pd(a + i), vb));
    } else if (alpha == 1.0) {
        for (; i + 16 <= n; i += 16) {
            _mm512_storeu_pd(y + i, _mm512_add_pd(_mm512_loadu_pd(a + i), vb));
            _mm512_storeu_pd(y + i + 8,
                             _mm512_add_pd(_mm512_loadu_pd(a + i + 8), vb));
        }
        for (; i + 8 <= n; i += 8)
            _mm512_storeu_pd(y + i, _mm512_add_pd(_mm512_loadu_pd(a + i), vb));
    } else if (alpha == -1.0) {
        for (; i + 16 <= n; i += 16) {
            _mm512_storeu_pd(y + i, _mm512_sub_pd(_mm512_loadu_pd(a + i), vb));
            _mm512_storeu_pd(y + i + 8,
                             _mm512_sub_pd(_mm512_loadu_pd(a + i + 8), vb));
        }
        for (; i + 8 <= n; i += 8)
            _mm512_storeu_pd(y + i, _mm512_sub_pd(_mm512_loadu_pd(a + i), vb));
    } else {
        for (; i + 16 <= n; i += 16) {
            _mm512_storeu_pd(
                y + i, _mm512_fmadd_pd(valpha, vb, _mm512_loadu_pd(a + i)));
            _mm512_storeu_pd(y + i + 8,
                             _mm512_fmadd_pd(valpha, vb,
                                             _mm512_loadu_pd(a + i + 8)));
        }
        for (; i + 8 <= n; i += 8)
            _mm512_storeu_pd(
                y + i, _mm512_fmadd_pd(valpha, vb, _mm512_loadu_pd(a + i)));
    }
    for (; i < n; ++i) {
        switch (code) {
            case BIN_ADD: y[i] = a[i] + alpha * b; break;
            case BIN_MUL: y[i] = a[i] * b; break;
            default:      y[i] = a[i] / b; break;
        }
    }
}

__attribute__((target("avx512f,fma")))
void binary_f64_avx512_scalar_a(int code, double a, const double* b, double* y,
                                int64_t n, double alpha) {
    const __m512d va = _mm512_set1_pd(a);
    const __m512d valpha = _mm512_set1_pd(alpha);
    int64_t i = 0;
    if (code == BIN_MUL) {
        for (; i + 16 <= n; i += 16) {
            _mm512_storeu_pd(y + i, _mm512_mul_pd(va, _mm512_loadu_pd(b + i)));
            _mm512_storeu_pd(y + i + 8,
                             _mm512_mul_pd(va, _mm512_loadu_pd(b + i + 8)));
        }
        for (; i + 8 <= n; i += 8)
            _mm512_storeu_pd(y + i, _mm512_mul_pd(va, _mm512_loadu_pd(b + i)));
    } else if (code == BIN_DIV) {
        for (; i + 16 <= n; i += 16) {
            _mm512_storeu_pd(y + i, _mm512_div_pd(va, _mm512_loadu_pd(b + i)));
            _mm512_storeu_pd(y + i + 8,
                             _mm512_div_pd(va, _mm512_loadu_pd(b + i + 8)));
        }
        for (; i + 8 <= n; i += 8)
            _mm512_storeu_pd(y + i, _mm512_div_pd(va, _mm512_loadu_pd(b + i)));
    } else if (alpha == 1.0) {
        for (; i + 16 <= n; i += 16) {
            _mm512_storeu_pd(y + i, _mm512_add_pd(va, _mm512_loadu_pd(b + i)));
            _mm512_storeu_pd(y + i + 8,
                             _mm512_add_pd(va, _mm512_loadu_pd(b + i + 8)));
        }
        for (; i + 8 <= n; i += 8)
            _mm512_storeu_pd(y + i, _mm512_add_pd(va, _mm512_loadu_pd(b + i)));
    } else if (alpha == -1.0) {
        for (; i + 16 <= n; i += 16) {
            _mm512_storeu_pd(y + i, _mm512_sub_pd(va, _mm512_loadu_pd(b + i)));
            _mm512_storeu_pd(y + i + 8,
                             _mm512_sub_pd(va, _mm512_loadu_pd(b + i + 8)));
        }
        for (; i + 8 <= n; i += 8)
            _mm512_storeu_pd(y + i, _mm512_sub_pd(va, _mm512_loadu_pd(b + i)));
    } else {
        for (; i + 16 <= n; i += 16) {
            _mm512_storeu_pd(
                y + i, _mm512_fmadd_pd(valpha, _mm512_loadu_pd(b + i), va));
            _mm512_storeu_pd(y + i + 8,
                             _mm512_fmadd_pd(valpha,
                                             _mm512_loadu_pd(b + i + 8), va));
        }
        for (; i + 8 <= n; i += 8)
            _mm512_storeu_pd(
                y + i, _mm512_fmadd_pd(valpha, _mm512_loadu_pd(b + i), va));
    }
    for (; i < n; ++i) {
        switch (code) {
            case BIN_ADD: y[i] = a + alpha * b[i]; break;
            case BIN_MUL: y[i] = a * b[i]; break;
            default:      y[i] = a / b[i]; break;
        }
    }
}

// --- 256-bit twins for hosts without the 512-bit datapath ------------------
// Same shapes as the zmm kernels above; every branch is hoisted out of the
// streaming loop and runs two-way unrolled.

__attribute__((target("avx2,fma")))
void binary_f32_avx256(int code, const float* a, const float* b, float* y,
                       int64_t n, float alpha) {
    const __m256 va = _mm256_set1_ps(alpha);
    int64_t i = 0;
    if (code == BIN_MUL) {
        for (; i + 32 <= n; i += 32) {
            _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(a + i),
                                                  _mm256_loadu_ps(b + i)));
            _mm256_storeu_ps(y + i + 8, _mm256_mul_ps(_mm256_loadu_ps(a + i + 8),
                                                      _mm256_loadu_ps(b + i + 8)));
            _mm256_storeu_ps(y + i + 16, _mm256_mul_ps(_mm256_loadu_ps(a + i + 16),
                                                       _mm256_loadu_ps(b + i + 16)));
            _mm256_storeu_ps(y + i + 24, _mm256_mul_ps(_mm256_loadu_ps(a + i + 24),
                                                       _mm256_loadu_ps(b + i + 24)));
        }
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(a + i),
                                                  _mm256_loadu_ps(b + i)));
    } else if (code == BIN_DIV) {
        for (; i + 32 <= n; i += 32) {
            _mm256_storeu_ps(y + i, _mm256_div_ps(_mm256_loadu_ps(a + i),
                                                  _mm256_loadu_ps(b + i)));
            _mm256_storeu_ps(y + i + 8, _mm256_div_ps(_mm256_loadu_ps(a + i + 8),
                                                      _mm256_loadu_ps(b + i + 8)));
            _mm256_storeu_ps(y + i + 16, _mm256_div_ps(_mm256_loadu_ps(a + i + 16),
                                                       _mm256_loadu_ps(b + i + 16)));
            _mm256_storeu_ps(y + i + 24, _mm256_div_ps(_mm256_loadu_ps(a + i + 24),
                                                       _mm256_loadu_ps(b + i + 24)));
        }
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_div_ps(_mm256_loadu_ps(a + i),
                                                  _mm256_loadu_ps(b + i)));
    } else if (alpha == 1.0f) {
        for (; i + 32 <= n; i += 32) {
            _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(a + i),
                                                  _mm256_loadu_ps(b + i)));
            _mm256_storeu_ps(y + i + 8, _mm256_add_ps(_mm256_loadu_ps(a + i + 8),
                                                      _mm256_loadu_ps(b + i + 8)));
            _mm256_storeu_ps(y + i + 16, _mm256_add_ps(_mm256_loadu_ps(a + i + 16),
                                                       _mm256_loadu_ps(b + i + 16)));
            _mm256_storeu_ps(y + i + 24, _mm256_add_ps(_mm256_loadu_ps(a + i + 24),
                                                       _mm256_loadu_ps(b + i + 24)));
        }
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(a + i),
                                                  _mm256_loadu_ps(b + i)));
    } else if (alpha == -1.0f) {
        for (; i + 32 <= n; i += 32) {
            _mm256_storeu_ps(y + i, _mm256_sub_ps(_mm256_loadu_ps(a + i),
                                                  _mm256_loadu_ps(b + i)));
            _mm256_storeu_ps(y + i + 8, _mm256_sub_ps(_mm256_loadu_ps(a + i + 8),
                                                      _mm256_loadu_ps(b + i + 8)));
            _mm256_storeu_ps(y + i + 16, _mm256_sub_ps(_mm256_loadu_ps(a + i + 16),
                                                       _mm256_loadu_ps(b + i + 16)));
            _mm256_storeu_ps(y + i + 24, _mm256_sub_ps(_mm256_loadu_ps(a + i + 24),
                                                       _mm256_loadu_ps(b + i + 24)));
        }
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_sub_ps(_mm256_loadu_ps(a + i),
                                                  _mm256_loadu_ps(b + i)));
    } else {
        for (; i + 32 <= n; i += 32) {
            _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(a + i),
                                                  _mm256_mul_ps(va, _mm256_loadu_ps(b + i))));
            _mm256_storeu_ps(y + i + 8, _mm256_add_ps(_mm256_loadu_ps(a + i + 8),
                                                      _mm256_mul_ps(va, _mm256_loadu_ps(b + i + 8))));
            _mm256_storeu_ps(y + i + 16, _mm256_add_ps(_mm256_loadu_ps(a + i + 16),
                                                       _mm256_mul_ps(va, _mm256_loadu_ps(b + i + 16))));
            _mm256_storeu_ps(y + i + 24, _mm256_add_ps(_mm256_loadu_ps(a + i + 24),
                                                       _mm256_mul_ps(va, _mm256_loadu_ps(b + i + 24))));
        }
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(a + i),
                                                  _mm256_mul_ps(va, _mm256_loadu_ps(b + i))));
    }
    for (; i < n; ++i) {
        switch (code) {
            case BIN_ADD: y[i] = a[i] + alpha * b[i]; break;
            case BIN_MUL: y[i] = a[i] * b[i]; break;
            default:      y[i] = a[i] / b[i]; break;
        }
    }
}

__attribute__((target("avx2,fma")))
void binary_f64_avx256(int code, const double* a, const double* b, double* y,
                       int64_t n, double alpha) {
    const __m256d va = _mm256_set1_pd(alpha);
    int64_t i = 0;
    if (code == BIN_MUL) {
        for (; i + 16 <= n; i += 16) {
            _mm256_storeu_pd(y + i, _mm256_mul_pd(_mm256_loadu_pd(a + i),
                                                  _mm256_loadu_pd(b + i)));
            _mm256_storeu_pd(y + i + 4, _mm256_mul_pd(_mm256_loadu_pd(a + i + 4),
                                                      _mm256_loadu_pd(b + i + 4)));
            _mm256_storeu_pd(y + i + 8, _mm256_mul_pd(_mm256_loadu_pd(a + i + 8),
                                                      _mm256_loadu_pd(b + i + 8)));
            _mm256_storeu_pd(y + i + 12, _mm256_mul_pd(_mm256_loadu_pd(a + i + 12),
                                                       _mm256_loadu_pd(b + i + 12)));
        }
        for (; i + 4 <= n; i += 4)
            _mm256_storeu_pd(y + i, _mm256_mul_pd(_mm256_loadu_pd(a + i),
                                                  _mm256_loadu_pd(b + i)));
    } else if (code == BIN_DIV) {
        for (; i + 16 <= n; i += 16) {
            _mm256_storeu_pd(y + i, _mm256_div_pd(_mm256_loadu_pd(a + i),
                                                  _mm256_loadu_pd(b + i)));
            _mm256_storeu_pd(y + i + 4, _mm256_div_pd(_mm256_loadu_pd(a + i + 4),
                                                      _mm256_loadu_pd(b + i + 4)));
            _mm256_storeu_pd(y + i + 8, _mm256_div_pd(_mm256_loadu_pd(a + i + 8),
                                                      _mm256_loadu_pd(b + i + 8)));
            _mm256_storeu_pd(y + i + 12, _mm256_div_pd(_mm256_loadu_pd(a + i + 12),
                                                       _mm256_loadu_pd(b + i + 12)));
        }
        for (; i + 4 <= n; i += 4)
            _mm256_storeu_pd(y + i, _mm256_div_pd(_mm256_loadu_pd(a + i),
                                                  _mm256_loadu_pd(b + i)));
    } else if (alpha == 1.0) {
        for (; i + 16 <= n; i += 16) {
            _mm256_storeu_pd(y + i, _mm256_add_pd(_mm256_loadu_pd(a + i),
                                                  _mm256_loadu_pd(b + i)));
            _mm256_storeu_pd(y + i + 4, _mm256_add_pd(_mm256_loadu_pd(a + i + 4),
                                                      _mm256_loadu_pd(b + i + 4)));
            _mm256_storeu_pd(y + i + 8, _mm256_add_pd(_mm256_loadu_pd(a + i + 8),
                                                      _mm256_loadu_pd(b + i + 8)));
            _mm256_storeu_pd(y + i + 12, _mm256_add_pd(_mm256_loadu_pd(a + i + 12),
                                                       _mm256_loadu_pd(b + i + 12)));
        }
        for (; i + 4 <= n; i += 4)
            _mm256_storeu_pd(y + i, _mm256_add_pd(_mm256_loadu_pd(a + i),
                                                  _mm256_loadu_pd(b + i)));
    } else if (alpha == -1.0) {
        for (; i + 16 <= n; i += 16) {
            _mm256_storeu_pd(y + i, _mm256_sub_pd(_mm256_loadu_pd(a + i),
                                                  _mm256_loadu_pd(b + i)));
            _mm256_storeu_pd(y + i + 4, _mm256_sub_pd(_mm256_loadu_pd(a + i + 4),
                                                      _mm256_loadu_pd(b + i + 4)));
            _mm256_storeu_pd(y + i + 8, _mm256_sub_pd(_mm256_loadu_pd(a + i + 8),
                                                      _mm256_loadu_pd(b + i + 8)));
            _mm256_storeu_pd(y + i + 12, _mm256_sub_pd(_mm256_loadu_pd(a + i + 12),
                                                       _mm256_loadu_pd(b + i + 12)));
        }
        for (; i + 4 <= n; i += 4)
            _mm256_storeu_pd(y + i, _mm256_sub_pd(_mm256_loadu_pd(a + i),
                                                  _mm256_loadu_pd(b + i)));
    } else {
        for (; i + 16 <= n; i += 16) {
            _mm256_storeu_pd(y + i, _mm256_add_pd(_mm256_loadu_pd(a + i),
                                                  _mm256_mul_pd(va, _mm256_loadu_pd(b + i))));
            _mm256_storeu_pd(y + i + 4, _mm256_add_pd(_mm256_loadu_pd(a + i + 4),
                                                      _mm256_mul_pd(va, _mm256_loadu_pd(b + i + 4))));
            _mm256_storeu_pd(y + i + 8, _mm256_add_pd(_mm256_loadu_pd(a + i + 8),
                                                      _mm256_mul_pd(va, _mm256_loadu_pd(b + i + 8))));
            _mm256_storeu_pd(y + i + 12, _mm256_add_pd(_mm256_loadu_pd(a + i + 12),
                                                       _mm256_mul_pd(va, _mm256_loadu_pd(b + i + 12))));
        }
        for (; i + 4 <= n; i += 4)
            _mm256_storeu_pd(y + i, _mm256_add_pd(_mm256_loadu_pd(a + i),
                                                  _mm256_mul_pd(va, _mm256_loadu_pd(b + i))));
    }
    for (; i < n; ++i) {
        switch (code) {
            case BIN_ADD: y[i] = a[i] + alpha * b[i]; break;
            case BIN_MUL: y[i] = a[i] * b[i]; break;
            default:      y[i] = a[i] / b[i]; break;
        }
    }
}

__attribute__((target("avx2,fma")))
void binary_f32_avx256_scalar_b(int code, const float* a, float b, float* y,
                                int64_t n, float alpha) {
    const __m256 vb = _mm256_set1_ps(b);
    const __m256 valpha = _mm256_set1_ps(alpha);
    int64_t i = 0;
    if (code == BIN_MUL) {
        for (; i + 32 <= n; i += 32) {
            _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), vb));
            _mm256_storeu_ps(y + i + 8, _mm256_mul_ps(_mm256_loadu_ps(a + i + 8), vb));
            _mm256_storeu_ps(y + i + 16, _mm256_mul_ps(_mm256_loadu_ps(a + i + 16), vb));
            _mm256_storeu_ps(y + i + 24, _mm256_mul_ps(_mm256_loadu_ps(a + i + 24), vb));
        }
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), vb));
    } else if (code == BIN_DIV) {
        for (; i + 32 <= n; i += 32) {
            _mm256_storeu_ps(y + i, _mm256_div_ps(_mm256_loadu_ps(a + i), vb));
            _mm256_storeu_ps(y + i + 8, _mm256_div_ps(_mm256_loadu_ps(a + i + 8), vb));
            _mm256_storeu_ps(y + i + 16, _mm256_div_ps(_mm256_loadu_ps(a + i + 16), vb));
            _mm256_storeu_ps(y + i + 24, _mm256_div_ps(_mm256_loadu_ps(a + i + 24), vb));
        }
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_div_ps(_mm256_loadu_ps(a + i), vb));
    } else if (alpha == 1.0f) {
        for (; i + 32 <= n; i += 32) {
            _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(a + i), vb));
            _mm256_storeu_ps(y + i + 8, _mm256_add_ps(_mm256_loadu_ps(a + i + 8), vb));
            _mm256_storeu_ps(y + i + 16, _mm256_add_ps(_mm256_loadu_ps(a + i + 16), vb));
            _mm256_storeu_ps(y + i + 24, _mm256_add_ps(_mm256_loadu_ps(a + i + 24), vb));
        }
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(a + i), vb));
    } else if (alpha == -1.0f) {
        for (; i + 32 <= n; i += 32) {
            _mm256_storeu_ps(y + i, _mm256_sub_ps(_mm256_loadu_ps(a + i), vb));
            _mm256_storeu_ps(y + i + 8, _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), vb));
            _mm256_storeu_ps(y + i + 16, _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), vb));
            _mm256_storeu_ps(y + i + 24, _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), vb));
        }
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_sub_ps(_mm256_loadu_ps(a + i), vb));
    } else {
        for (; i + 32 <= n; i += 32) {
            _mm256_storeu_ps(y + i, _mm256_fmadd_ps(valpha, vb, _mm256_loadu_ps(a + i)));
            _mm256_storeu_ps(y + i + 8, _mm256_fmadd_ps(valpha, vb, _mm256_loadu_ps(a + i + 8)));
            _mm256_storeu_ps(y + i + 16, _mm256_fmadd_ps(valpha, vb, _mm256_loadu_ps(a + i + 16)));
            _mm256_storeu_ps(y + i + 24, _mm256_fmadd_ps(valpha, vb, _mm256_loadu_ps(a + i + 24)));
        }
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_fmadd_ps(valpha, vb, _mm256_loadu_ps(a + i)));
    }
    for (; i < n; ++i) {
        switch (code) {
            case BIN_ADD: y[i] = a[i] + alpha * b; break;
            case BIN_MUL: y[i] = a[i] * b; break;
            default:      y[i] = a[i] / b; break;
        }
    }
}

__attribute__((target("avx2,fma")))
void binary_f32_avx256_scalar_a(int code, float a, const float* b, float* y,
                                int64_t n, float alpha) {
    const __m256 va = _mm256_set1_ps(a);
    const __m256 valpha = _mm256_set1_ps(alpha);
    int64_t i = 0;
    if (code == BIN_MUL) {
        for (; i + 32 <= n; i += 32) {
            _mm256_storeu_ps(y + i, _mm256_mul_ps(va, _mm256_loadu_ps(b + i)));
            _mm256_storeu_ps(y + i + 8, _mm256_mul_ps(va, _mm256_loadu_ps(b + i + 8)));
            _mm256_storeu_ps(y + i + 16, _mm256_mul_ps(va, _mm256_loadu_ps(b + i + 16)));
            _mm256_storeu_ps(y + i + 24, _mm256_mul_ps(va, _mm256_loadu_ps(b + i + 24)));
        }
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_mul_ps(va, _mm256_loadu_ps(b + i)));
    } else if (code == BIN_DIV) {
        for (; i + 32 <= n; i += 32) {
            _mm256_storeu_ps(y + i, _mm256_div_ps(va, _mm256_loadu_ps(b + i)));
            _mm256_storeu_ps(y + i + 8, _mm256_div_ps(va, _mm256_loadu_ps(b + i + 8)));
            _mm256_storeu_ps(y + i + 16, _mm256_div_ps(va, _mm256_loadu_ps(b + i + 16)));
            _mm256_storeu_ps(y + i + 24, _mm256_div_ps(va, _mm256_loadu_ps(b + i + 24)));
        }
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_div_ps(va, _mm256_loadu_ps(b + i)));
    } else if (alpha == 1.0f) {
        for (; i + 32 <= n; i += 32) {
            _mm256_storeu_ps(y + i, _mm256_add_ps(va, _mm256_loadu_ps(b + i)));
            _mm256_storeu_ps(y + i + 8, _mm256_add_ps(va, _mm256_loadu_ps(b + i + 8)));
            _mm256_storeu_ps(y + i + 16, _mm256_add_ps(va, _mm256_loadu_ps(b + i + 16)));
            _mm256_storeu_ps(y + i + 24, _mm256_add_ps(va, _mm256_loadu_ps(b + i + 24)));
        }
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_add_ps(va, _mm256_loadu_ps(b + i)));
    } else if (alpha == -1.0f) {
        for (; i + 32 <= n; i += 32) {
            _mm256_storeu_ps(y + i, _mm256_sub_ps(va, _mm256_loadu_ps(b + i)));
            _mm256_storeu_ps(y + i + 8, _mm256_sub_ps(va, _mm256_loadu_ps(b + i + 8)));
            _mm256_storeu_ps(y + i + 16, _mm256_sub_ps(va, _mm256_loadu_ps(b + i + 16)));
            _mm256_storeu_ps(y + i + 24, _mm256_sub_ps(va, _mm256_loadu_ps(b + i + 24)));
        }
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_sub_ps(va, _mm256_loadu_ps(b + i)));
    } else {
        for (; i + 32 <= n; i += 32) {
            _mm256_storeu_ps(y + i, _mm256_fmadd_ps(valpha, _mm256_loadu_ps(b + i), va));
            _mm256_storeu_ps(y + i + 8, _mm256_fmadd_ps(valpha, _mm256_loadu_ps(b + i + 8), va));
            _mm256_storeu_ps(y + i + 16, _mm256_fmadd_ps(valpha, _mm256_loadu_ps(b + i + 16), va));
            _mm256_storeu_ps(y + i + 24, _mm256_fmadd_ps(valpha, _mm256_loadu_ps(b + i + 24), va));
        }
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_fmadd_ps(valpha, _mm256_loadu_ps(b + i), va));
    }
    for (; i < n; ++i) {
        switch (code) {
            case BIN_ADD: y[i] = a + alpha * b[i]; break;
            case BIN_MUL: y[i] = a * b[i]; break;
            default:      y[i] = a / b[i]; break;
        }
    }
}

__attribute__((target("avx2,fma")))
void binary_f64_avx256_scalar_b(int code, const double* a, double b, double* y,
                                int64_t n, double alpha) {
    const __m256d vb = _mm256_set1_pd(b);
    const __m256d valpha = _mm256_set1_pd(alpha);
    int64_t i = 0;
    if (code == BIN_MUL) {
        for (; i + 16 <= n; i += 16) {
            _mm256_storeu_pd(y + i, _mm256_mul_pd(_mm256_loadu_pd(a + i), vb));
            _mm256_storeu_pd(y + i + 4, _mm256_mul_pd(_mm256_loadu_pd(a + i + 4), vb));
            _mm256_storeu_pd(y + i + 8, _mm256_mul_pd(_mm256_loadu_pd(a + i + 8), vb));
            _mm256_storeu_pd(y + i + 12, _mm256_mul_pd(_mm256_loadu_pd(a + i + 12), vb));
        }
        for (; i + 4 <= n; i += 4)
            _mm256_storeu_pd(y + i, _mm256_mul_pd(_mm256_loadu_pd(a + i), vb));
    } else if (code == BIN_DIV) {
        for (; i + 16 <= n; i += 16) {
            _mm256_storeu_pd(y + i, _mm256_div_pd(_mm256_loadu_pd(a + i), vb));
            _mm256_storeu_pd(y + i + 4, _mm256_div_pd(_mm256_loadu_pd(a + i + 4), vb));
            _mm256_storeu_pd(y + i + 8, _mm256_div_pd(_mm256_loadu_pd(a + i + 8), vb));
            _mm256_storeu_pd(y + i + 12, _mm256_div_pd(_mm256_loadu_pd(a + i + 12), vb));
        }
        for (; i + 4 <= n; i += 4)
            _mm256_storeu_pd(y + i, _mm256_div_pd(_mm256_loadu_pd(a + i), vb));
    } else if (alpha == 1.0) {
        for (; i + 16 <= n; i += 16) {
            _mm256_storeu_pd(y + i, _mm256_add_pd(_mm256_loadu_pd(a + i), vb));
            _mm256_storeu_pd(y + i + 4, _mm256_add_pd(_mm256_loadu_pd(a + i + 4), vb));
            _mm256_storeu_pd(y + i + 8, _mm256_add_pd(_mm256_loadu_pd(a + i + 8), vb));
            _mm256_storeu_pd(y + i + 12, _mm256_add_pd(_mm256_loadu_pd(a + i + 12), vb));
        }
        for (; i + 4 <= n; i += 4)
            _mm256_storeu_pd(y + i, _mm256_add_pd(_mm256_loadu_pd(a + i), vb));
    } else if (alpha == -1.0) {
        for (; i + 16 <= n; i += 16) {
            _mm256_storeu_pd(y + i, _mm256_sub_pd(_mm256_loadu_pd(a + i), vb));
            _mm256_storeu_pd(y + i + 4, _mm256_sub_pd(_mm256_loadu_pd(a + i + 4), vb));
            _mm256_storeu_pd(y + i + 8, _mm256_sub_pd(_mm256_loadu_pd(a + i + 8), vb));
            _mm256_storeu_pd(y + i + 12, _mm256_sub_pd(_mm256_loadu_pd(a + i + 12), vb));
        }
        for (; i + 4 <= n; i += 4)
            _mm256_storeu_pd(y + i, _mm256_sub_pd(_mm256_loadu_pd(a + i), vb));
    } else {
        for (; i + 16 <= n; i += 16) {
            _mm256_storeu_pd(y + i, _mm256_fmadd_pd(valpha, vb, _mm256_loadu_pd(a + i)));
            _mm256_storeu_pd(y + i + 4, _mm256_fmadd_pd(valpha, vb, _mm256_loadu_pd(a + i + 4)));
            _mm256_storeu_pd(y + i + 8, _mm256_fmadd_pd(valpha, vb, _mm256_loadu_pd(a + i + 8)));
            _mm256_storeu_pd(y + i + 12, _mm256_fmadd_pd(valpha, vb, _mm256_loadu_pd(a + i + 12)));
        }
        for (; i + 4 <= n; i += 4)
            _mm256_storeu_pd(y + i, _mm256_fmadd_pd(valpha, vb, _mm256_loadu_pd(a + i)));
    }
    for (; i < n; ++i) {
        switch (code) {
            case BIN_ADD: y[i] = a[i] + alpha * b; break;
            case BIN_MUL: y[i] = a[i] * b; break;
            default:      y[i] = a[i] / b; break;
        }
    }
}

__attribute__((target("avx2,fma")))
void binary_f64_avx256_scalar_a(int code, double a, const double* b, double* y,
                                int64_t n, double alpha) {
    const __m256d va = _mm256_set1_pd(a);
    const __m256d valpha = _mm256_set1_pd(alpha);
    int64_t i = 0;
    if (code == BIN_MUL) {
        for (; i + 16 <= n; i += 16) {
            _mm256_storeu_pd(y + i, _mm256_mul_pd(va, _mm256_loadu_pd(b + i)));
            _mm256_storeu_pd(y + i + 4, _mm256_mul_pd(va, _mm256_loadu_pd(b + i + 4)));
            _mm256_storeu_pd(y + i + 8, _mm256_mul_pd(va, _mm256_loadu_pd(b + i + 8)));
            _mm256_storeu_pd(y + i + 12, _mm256_mul_pd(va, _mm256_loadu_pd(b + i + 12)));
        }
        for (; i + 4 <= n; i += 4)
            _mm256_storeu_pd(y + i, _mm256_mul_pd(va, _mm256_loadu_pd(b + i)));
    } else if (code == BIN_DIV) {
        for (; i + 16 <= n; i += 16) {
            _mm256_storeu_pd(y + i, _mm256_div_pd(va, _mm256_loadu_pd(b + i)));
            _mm256_storeu_pd(y + i + 4, _mm256_div_pd(va, _mm256_loadu_pd(b + i + 4)));
            _mm256_storeu_pd(y + i + 8, _mm256_div_pd(va, _mm256_loadu_pd(b + i + 8)));
            _mm256_storeu_pd(y + i + 12, _mm256_div_pd(va, _mm256_loadu_pd(b + i + 12)));
        }
        for (; i + 4 <= n; i += 4)
            _mm256_storeu_pd(y + i, _mm256_div_pd(va, _mm256_loadu_pd(b + i)));
    } else if (alpha == 1.0) {
        for (; i + 16 <= n; i += 16) {
            _mm256_storeu_pd(y + i, _mm256_add_pd(va, _mm256_loadu_pd(b + i)));
            _mm256_storeu_pd(y + i + 4, _mm256_add_pd(va, _mm256_loadu_pd(b + i + 4)));
            _mm256_storeu_pd(y + i + 8, _mm256_add_pd(va, _mm256_loadu_pd(b + i + 8)));
            _mm256_storeu_pd(y + i + 12, _mm256_add_pd(va, _mm256_loadu_pd(b + i + 12)));
        }
        for (; i + 4 <= n; i += 4)
            _mm256_storeu_pd(y + i, _mm256_add_pd(va, _mm256_loadu_pd(b + i)));
    } else if (alpha == -1.0) {
        for (; i + 16 <= n; i += 16) {
            _mm256_storeu_pd(y + i, _mm256_sub_pd(va, _mm256_loadu_pd(b + i)));
            _mm256_storeu_pd(y + i + 4, _mm256_sub_pd(va, _mm256_loadu_pd(b + i + 4)));
            _mm256_storeu_pd(y + i + 8, _mm256_sub_pd(va, _mm256_loadu_pd(b + i + 8)));
            _mm256_storeu_pd(y + i + 12, _mm256_sub_pd(va, _mm256_loadu_pd(b + i + 12)));
        }
        for (; i + 4 <= n; i += 4)
            _mm256_storeu_pd(y + i, _mm256_sub_pd(va, _mm256_loadu_pd(b + i)));
    } else {
        for (; i + 16 <= n; i += 16) {
            _mm256_storeu_pd(y + i, _mm256_fmadd_pd(valpha, _mm256_loadu_pd(b + i), va));
            _mm256_storeu_pd(y + i + 4, _mm256_fmadd_pd(valpha, _mm256_loadu_pd(b + i + 4), va));
            _mm256_storeu_pd(y + i + 8, _mm256_fmadd_pd(valpha, _mm256_loadu_pd(b + i + 8), va));
            _mm256_storeu_pd(y + i + 12, _mm256_fmadd_pd(valpha, _mm256_loadu_pd(b + i + 12), va));
        }
        for (; i + 4 <= n; i += 4)
            _mm256_storeu_pd(y + i, _mm256_fmadd_pd(valpha, _mm256_loadu_pd(b + i), va));
    }
    for (; i < n; ++i) {
        switch (code) {
            case BIN_ADD: y[i] = a + alpha * b[i]; break;
            case BIN_MUL: y[i] = a * b[i]; break;
            default:      y[i] = a / b[i]; break;
        }
    }
}

// ISA selectors: keep every TensorIterator branch ISA-agnostic by picking
// the widest kernel the host actually runs.
static void binary_f32_vec(int code, const float* a, const float* b, float* y,
                           int64_t n, float alpha) {
    if (cpu_has_avx512()) binary_f32_avx512(code, a, b, y, n, alpha);
    else binary_f32_avx256(code, a, b, y, n, alpha);
}
static void binary_f32_vec_scalar_b(int code, const float* a, float b, float* y,
                                    int64_t n, float alpha) {
    if (cpu_has_avx512()) binary_f32_avx512_scalar_b(code, a, b, y, n, alpha);
    else binary_f32_avx256_scalar_b(code, a, b, y, n, alpha);
}
static void binary_f32_vec_scalar_a(int code, float a, const float* b, float* y,
                                    int64_t n, float alpha) {
    if (cpu_has_avx512()) binary_f32_avx512_scalar_a(code, a, b, y, n, alpha);
    else binary_f32_avx256_scalar_a(code, a, b, y, n, alpha);
}
static void binary_f64_vec(int code, const double* a, const double* b, double* y,
                           int64_t n, double alpha) {
    if (cpu_has_avx512()) binary_f64_avx512(code, a, b, y, n, alpha);
    else binary_f64_avx256(code, a, b, y, n, alpha);
}
static void binary_f64_vec_scalar_b(int code, const double* a, double b, double* y,
                                    int64_t n, double alpha) {
    if (cpu_has_avx512()) binary_f64_avx512_scalar_b(code, a, b, y, n, alpha);
    else binary_f64_avx256_scalar_b(code, a, b, y, n, alpha);
}
static void binary_f64_vec_scalar_a(int code, double a, const double* b, double* y,
                                    int64_t n, double alpha) {
    if (cpu_has_avx512()) binary_f64_avx512_scalar_a(code, a, b, y, n, alpha);
    else binary_f64_avx256_scalar_a(code, a, b, y, n, alpha);
}

// TensorIterator's native CPU kernels parallelize contiguous scalar and
// wrapped-scalar operations at the same grain size as ordinary pointwise
// kernels.  Keep the target-specific loop above single-purpose, but add the
// same scheduling around it so a large optimizer tensor does not silently
// become a one-thread AVX-512 loop.
inline void scalar_f32_contiguous(int code, const float* a, float* y,
                                   int64_t n, float scalar) {
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        if (cpu_has_avx512())
            scalar_f32_avx512(code, a + begin, y + begin, end - begin, scalar);
        else
            scalar_f32_avx256(code, a + begin, y + begin, end - begin, scalar);
    });
}

inline void scalar_f64_contiguous(int code, const double* a, double* y,
                                   int64_t n, double scalar) {
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        if (cpu_has_avx512())
            scalar_f64_avx512(code, a + begin, y + begin, end - begin, scalar);
        else
            scalar_f64_avx256(code, a + begin, y + begin, end - begin, scalar);
    });
}

inline void binary_f32_contiguous(int code, const float* a, const float* b,
                                   float* y, int64_t n, float alpha) {
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        if (cpu_has_avx512())
            binary_f32_avx512(code, a + begin, b + begin, y + begin,
                              end - begin, alpha);
        else
            binary_f32_avx256(code, a + begin, b + begin, y + begin,
                              end - begin, alpha);
    });
}

inline void binary_f64_contiguous(int code, const double* a, const double* b,
                                   double* y, int64_t n, double alpha) {
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        if (cpu_has_avx512())
            binary_f64_avx512(code, a + begin, b + begin, y + begin,
                              end - begin, alpha);
        else
            binary_f64_avx256(code, a + begin, b + begin, y + begin,
                              end - begin, alpha);
    });
}

// Muon keeps parameters in FP32 but its Newton-Schulz update in BF16.  The
// native reduced-precision TensorIterator path widens BF16 to FP32, performs
// the scaled add in FP32, and stores back to FP32.  This specialization avoids
// sending that very common optimizer update through the generic mixed-dtype
// iterator/oneDNN path.
__attribute__((target("avx512f,fma")))
void f32_bf16_add_inplace_avx512(float* dst, const uint16_t* src,
                                 int64_t n, float alpha) {
    const __m512 valpha = _mm512_set1_ps(alpha);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256i raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(src + i));
        const __m512 value = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
        const __m512 current = _mm512_loadu_ps(dst + i);
        _mm512_storeu_ps(dst + i, _mm512_fmadd_ps(valpha, value, current));
    }
    for (; i < n; ++i) {
        dst[i] += alpha * detail::bfloat16_to_float_bits(src[i]);
    }
}

inline void f32_bf16_add_inplace_contiguous(float* dst,
                                            const BFloat16* src,
                                            int64_t n, float alpha) {
    const auto* bits = reinterpret_cast<const uint16_t*>(src);
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        f32_bf16_add_inplace_avx512(dst + begin, bits + begin,
                                    end - begin, alpha);
    });
}

inline bool cpu_has_avx512_bf16() {
    static const bool ok = __builtin_cpu_supports("avx512f") != 0 &&
                           __builtin_cpu_supports("avx512bw") != 0 &&
                           __builtin_cpu_supports("avx512bf16") != 0;
    return ok;
}

// BF16 arithmetic is performed in float32 and rounded once on store, just as
// target-attributed routine: the rest of p10 remains safe to load on hosts
// without AVX-512 BF16, while Zen4-class hosts get the native 16-lane path.
__attribute__((target("avx512f,avx512bw,avx512bf16,fma")))
void bf16_binary_avx512(int code, const uint16_t* a, const uint16_t* b,
                        uint16_t* y, int64_t n, float alpha) {
    const __m512 valpha = _mm512_set1_ps(alpha);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256i ar = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a + i));
        const __m256i br = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(b + i));
        const __m512 af = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(ar), 16));
        const __m512 bf = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(br), 16));
        __m512 out;
        switch (code) {
            // Match that accumulation order before the single BF16 store.
            case BIN_ADD: out = _mm512_fmadd_ps(valpha, bf, af); break;
            case BIN_MUL: out = _mm512_mul_ps(af, bf); break;
            default:      out = _mm512_div_ps(af, bf); break;
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(y + i),
                            (__m256i)_mm512_cvtneps_pbh(out));
    }
    for (; i < n; ++i) {
        const float af = detail::bfloat16_to_float_bits(a[i]);
        const float bf = detail::bfloat16_to_float_bits(b[i]);
        float out;
        switch (code) {
            case BIN_ADD: out = af + alpha * bf; break;
            case BIN_MUL: out = af * bf; break;
            default:      out = af / bf; break;
        }
        y[i] = detail::float_to_bfloat16_bits(out);
    }
}

__attribute__((target("avx512f,avx512bw,avx512bf16")))
void bf16_scalar_avx512(int code, const uint16_t* a, uint16_t* y,
                        int64_t n, float scalar) {
    const __m512 vscalar = _mm512_set1_ps(scalar);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256i ar = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a + i));
        const __m512 af = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(ar), 16));
        __m512 out;
        switch (code) {
            case BIN_ADD: out = _mm512_add_ps(af, vscalar); break;
            case BIN_MUL: out = _mm512_mul_ps(af, vscalar); break;
            default:      out = _mm512_div_ps(af, vscalar); break;
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(y + i),
                            (__m256i)_mm512_cvtneps_pbh(out));
    }
    for (; i < n; ++i) {
        const float af = detail::bfloat16_to_float_bits(a[i]);
        float out;
        switch (code) {
            case BIN_ADD: out = af + scalar; break;
            case BIN_MUL: out = af * scalar; break;
            default:      out = af / scalar; break;
        }
        y[i] = detail::float_to_bfloat16_bits(out);
    }
}

inline void bf16_binary_contiguous(int code, const BFloat16* a,
                                   const BFloat16* b, BFloat16* y,
                                   int64_t n, float alpha) {
    const auto* ap = reinterpret_cast<const uint16_t*>(a);
    const auto* bp = reinterpret_cast<const uint16_t*>(b);
    auto* yp = reinterpret_cast<uint16_t*>(y);
    if (cpu_has_avx512_bf16()) {
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            bf16_binary_avx512(code, ap + begin, bp + begin, yp + begin,
                               end - begin, alpha);
        });
        return;
    }
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const float av = static_cast<float>(a[i]);
            const float bv = static_cast<float>(b[i]);
            const float out = code == BIN_ADD ? av + alpha * bv
                               : code == BIN_MUL ? av * bv : av / bv;
            y[i] = static_cast<BFloat16>(out);
        }
    });
}

inline void bf16_scalar_contiguous(int code, const BFloat16* a, BFloat16* y,
                                   int64_t n, float scalar) {
    const auto* ap = reinterpret_cast<const uint16_t*>(a);
    auto* yp = reinterpret_cast<uint16_t*>(y);
    if (cpu_has_avx512_bf16()) {
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            bf16_scalar_avx512(code, ap + begin, yp + begin, end - begin,
                               scalar);
        });
        return;
    }
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const float av = static_cast<float>(a[i]);
            const float out = code == BIN_ADD ? av + scalar
                               : code == BIN_MUL ? av * scalar : av / scalar;
            y[i] = static_cast<BFloat16>(out);
        }
    });
}

}  // namespace
#endif  // __x86_64__

// --- Helper for Binary Ops ---

// Scalar-division element step. Arithmetic happens in the compute domain
// (opmath) so Half/BFloat16 inputs divide in float, and reduced complex
// values divide in a float complex domain.
template <typename ctype>
ctype div_scalar_element(ctype a, const Scalar& other) {
    if constexpr (is_complex_type_v<ctype>) {
        using math_t = opmath_type<ctype>;
        const math_t am(a.real(), a.imag());
        const math_t bm = other.to<math_t>();
        const math_t rm = am / bm;
        return ctype(static_cast<typename ctype::value_type>(rm.real()),
                     static_cast<typename ctype::value_type>(rm.imag()));
    } else {
        using math_t = opmath_type<ctype>;
        return static_cast<ctype>(static_cast<math_t>(a) /
                                  static_cast<math_t>(other.to<ctype>()));
    }
}

// Same-shape elementwise results keep a channels-last operand layout, so
// residual chains around channels-last convolutions stay repack-free.
static bool binary_keep_channels_last(const Tensor& a, const Tensor& b) {
    return a.dim() == 4 && a.dim() == b.dim() &&
           static_cast<std::vector<int64_t>>(a.shape()) ==
               static_cast<std::vector<int64_t>>(b.shape()) &&
           a.is_contiguous(MemoryFormat::ChannelsLast) &&
           b.is_contiguous(MemoryFormat::ChannelsLast);
}

static Tensor empty_binary_result(const Tensor& a, const Tensor& b,
                                  const std::vector<int64_t>& out_shape, DType dt) {
    Tensor result = Tensor::empty(out_shape, dt, a.device());
    if (binary_keep_channels_last(a, b)) {
        result = result.as_strided(out_shape, get_channels_last_strides(out_shape), 0);
    }
    return result;
}

// A scalar operand carries no layout, so a channels-last tensor keeps its
// layout through scalar arithmetic.
static Tensor empty_scalar_result(const Tensor& self, DType dt) {
    const auto sizes = static_cast<std::vector<int64_t>>(self.shape());
    Tensor result = Tensor::empty(sizes, dt, self.device());
    if (self.dim() == 4 && self.is_contiguous(MemoryFormat::ChannelsLast)) {
        result = result.as_strided(sizes, get_channels_last_strides(sizes), 0);
    }
    return result;
}

template<typename Op, typename MklOp>
Tensor binary_op_kernel_impl(const Tensor& self, const Tensor& other, Op op, MklOp mkl_op, bool use_mkl_op = false, bool force_float = false, int vec_code = -1) {
    std::vector<int64_t> out_shape = broadcast_shapes(self.sizes(), other.sizes());
    DType result_dtype = native::result_type(self, other);
    if (force_float && isIntegralType(result_dtype, true)) {
        result_dtype = DType::Float32;
    }

    Tensor result = empty_binary_result(self, other, out_shape, result_dtype);

    bool optimized = false;
    if (result_dtype == DType::Float32 &&
        self.dtype() == DType::Float32 &&
        other.dtype() == DType::Float32 &&
        self.is_contiguous() && other.is_contiguous() && result.is_contiguous() &&
        self.shape() == other.shape()) {
        
        #ifdef USE_MKL
        if (use_mkl_op) {
            int64_t n = self.numel();
            mkl_op((int)n, self.data_ptr<float>(), other.data_ptr<float>(), result.data_ptr<float>());
            optimized = true;
        }
        #endif
    }
    
    if (!optimized) {
        Tensor self_casted = (self.dtype() == result_dtype) ? self : self.to(result_dtype);
        Tensor other_casted = (other.dtype() == result_dtype) ? other : other.to(result_dtype);

#if defined(__x86_64__)
        // The four arithmetic callers pass a BIN_* code so the vector
        // kernels can take over: TensorIterator hands the inner run with
        // per-operand strides, and a zero stride is a value constant along
        // the run (a broadcast operand), which the vector kernels accept as
        // a scalar.  Everything else keeps the generic iterator loop below.
        if (vec_code >= 0 && cpu_has_avx2f() &&
            (result_dtype == DType::Float32 || result_dtype == DType::Float64)) {
            const bool is_f32 = (result_dtype == DType::Float32);
            TensorIterator iter = TensorIterator::binary_op(result, self_casted, other_casted);
            iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
                if (is_f32) {
                    float* r = reinterpret_cast<float*>(data[0]);
                    const float* a = reinterpret_cast<const float*>(data[1]);
                    const float* b = reinterpret_cast<const float*>(data[2]);
                    if (strides[0] == 4 && strides[1] == 4 && strides[2] == 4) {
                        binary_f32_vec(vec_code, a, b, r, n, 1.0f);
                    } else if (strides[0] == 4 && strides[1] == 4 && strides[2] == 0) {
                        binary_f32_vec_scalar_b(vec_code, a, *b, r, n, 1.0f);
                    } else if (strides[0] == 4 && strides[1] == 0 && strides[2] == 4) {
                        binary_f32_vec_scalar_a(vec_code, *a, b, r, n, 1.0f);
                    } else if (strides[0] == 4 && strides[1] == 0 && strides[2] == 0) {
                        const float va = *a, vb = *b;
                        for (int64_t i = 0; i < n; ++i)
                            r[i] = vec_code == BIN_MUL ? va * vb
                                 : vec_code == BIN_DIV ? va / vb : va + vb;
                    } else {
                        for (int64_t i = 0; i < n; ++i) {
                            const float av = *reinterpret_cast<const float*>(data[1] + i * strides[1]);
                            const float bv = *reinterpret_cast<const float*>(data[2] + i * strides[2]);
                            *reinterpret_cast<float*>(data[0] + i * strides[0]) =
                                vec_code == BIN_MUL ? av * bv
                              : vec_code == BIN_DIV ? av / bv : av + bv;
                        }
                    }
                } else {
                    double* r = reinterpret_cast<double*>(data[0]);
                    const double* a = reinterpret_cast<const double*>(data[1]);
                    const double* b = reinterpret_cast<const double*>(data[2]);
                    if (strides[0] == 8 && strides[1] == 8 && strides[2] == 8) {
                        binary_f64_vec(vec_code, a, b, r, n, 1.0);
                    } else if (strides[0] == 8 && strides[1] == 8 && strides[2] == 0) {
                        binary_f64_vec_scalar_b(vec_code, a, *b, r, n, 1.0);
                    } else if (strides[0] == 8 && strides[1] == 0 && strides[2] == 8) {
                        binary_f64_vec_scalar_a(vec_code, *a, b, r, n, 1.0);
                    } else if (strides[0] == 8 && strides[1] == 0 && strides[2] == 0) {
                        const double va = *a, vb = *b;
                        for (int64_t i = 0; i < n; ++i)
                            r[i] = vec_code == BIN_MUL ? va * vb
                                 : vec_code == BIN_DIV ? va / vb : va + vb;
                    } else {
                        for (int64_t i = 0; i < n; ++i) {
                            const double av = *reinterpret_cast<const double*>(data[1] + i * strides[1]);
                            const double bv = *reinterpret_cast<const double*>(data[2] + i * strides[2]);
                            *reinterpret_cast<double*>(data[0] + i * strides[0]) =
                                vec_code == BIN_MUL ? av * bv
                              : vec_code == BIN_DIV ? av / bv : av + bv;
                        }
                    }
                }
            });
            return result;
        }
#endif

        // Route through the shared TensorIterator: it broadcasts, reorders
        // dimensions for memory locality, coalesces adjacent dims and
        // parallelizes the inner loop (the old path was a serial recursion).
        TensorIterator iter = TensorIterator::binary_op(result, self_casted, other_casted);

        #define TI_OP_CASE(ctype, name) \
        case DType::name: { \
            iter.for_each([&op](char** data, const int64_t* strides, int64_t n) { \
                char* r = data[0]; \
                const char* a = data[1]; \
                const char* b = data[2]; \
                /* strides are in bytes */ \
                if (strides[0] == static_cast<int64_t>(sizeof(ctype)) && \
                    strides[1] == static_cast<int64_t>(sizeof(ctype)) && \
                    strides[2] == static_cast<int64_t>(sizeof(ctype))) { \
                    ctype* rp = reinterpret_cast<ctype*>(r); \
                    const ctype* ap = reinterpret_cast<const ctype*>(a); \
                    const ctype* bp = reinterpret_cast<const ctype*>(b); \
                    for (int64_t i = 0; i < n; ++i) rp[i] = op(ap[i], bp[i]); \
                } else { \
                    for (int64_t i = 0; i < n; ++i) { \
                        *reinterpret_cast<ctype*>(r + i * strides[0]) = op( \
                            *reinterpret_cast<const ctype*>(a + i * strides[1]), \
                            *reinterpret_cast<const ctype*>(b + i * strides[2])); \
                    } \
                } \
            }); \
            break; \
        }

        switch (result_dtype) {
            // defined component-wise over the complex dtypes.
            TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TI_OP_CASE)
            default: TP_THROW(TypeError, "binary_op: unsupported dtype");
        }
        #undef TI_OP_CASE
    }
    
    return result;
}

// --- Binary Kernels ---

// alpha * y with per-dtype semantics.  Template + if constexpr so each
// instantiation only compiles its own branch (a non-template context would
// semantically check discarded branches too, breaking e.g. y.real() on
// integral dtypes).  Complex values scale the low-precision components via
// double, matching the real-path alpha.toDouble() behavior.
template <typename T>
inline T tp_alpha_scaled(const tensorplay::Scalar& alpha, const T& y) {
    if constexpr (is_complex_type_v<T>) {
        using v_t = typename is_complex_type<T>::value_type;
        const v_t a = static_cast<v_t>(alpha.toDouble());
        return T(a * y.real(), a * y.imag());
    } else if constexpr (std::is_floating_point_v<T>) {
        return alpha.to<T>() * y;
    } else if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(alpha.to<int64_t>()) * y;
    } else {
        // Half/BFloat16 and other low-precision scalars: compute in float.
        return static_cast<T>(static_cast<float>(alpha.toDouble()) *
                              static_cast<float>(y));
    }
}

DType add_result_dtype(DType self_dtype, DType other_dtype,
                       const Scalar& alpha) {
    DType result_dtype = promoteTypes(self_dtype, other_dtype);
    if (alpha.isFloatingPoint() && !isFloatingType(result_dtype)) {
        result_dtype = promoteTypes(result_dtype, DType::Float32);
    }
    return result_dtype;
}

bool has_output_alias(const Tensor& out, const Tensor& input) {
    auto out_impl = out.unsafeGetTensorImpl();
    auto input_impl = input.unsafeGetTensorImpl();
    return out_impl != nullptr && input_impl != nullptr &&
           out_impl->storage().defined() && input_impl->storage().defined() &&
           out_impl->storage().is_same(input_impl->storage());
}

bool try_add_float32_out(const Tensor& self, const Tensor& other,
                         Scalar alpha, Tensor& out) {
    if (self.dtype() != DType::Float32 || other.dtype() != DType::Float32 ||
        out.dtype() != DType::Float32 || self.shape() != other.shape() ||
        self.shape() != out.shape() || !self.is_contiguous() ||
        !other.is_contiguous() || !out.is_contiguous() ||
        has_output_alias(out, self) || has_output_alias(out, other)) {
        return false;
    }

    const int64_t n = self.numel();
    const float alpha_val = alpha.to<float>();
    const float* self_ptr = self.data_ptr<float>();
    const float* other_ptr = other.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

#if defined(__x86_64__)
    if (cpu_has_avx512()) {
        if (alpha_val == 1.0f) {
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                add_f32_avx512(self_ptr + begin, other_ptr + begin,
                               out_ptr + begin, end - begin);
            });
        } else if (alpha_val == -1.0f) {
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                sub_f32_avx512(self_ptr + begin, other_ptr + begin,
                               out_ptr + begin, end - begin);
            });
        } else {
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                binary_f32_avx512(BIN_ADD, self_ptr + begin,
                                  other_ptr + begin, out_ptr + begin,
                                  end - begin, alpha_val);
            });
        }
        return true;
    }
#endif

    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            out_ptr[i] = self_ptr[i] + alpha_val * other_ptr[i];
        }
    });
    return true;
}

bool try_add_tensor_iterator_out(const Tensor& self, const Tensor& other,
                                 Scalar alpha, DType result_dtype,
                                 Tensor& out) {
    if (has_output_alias(out, self) || has_output_alias(out, other)) {
        return false;
    }

    Tensor self_casted =
        (self.dtype() == result_dtype) ? self : self.to(result_dtype);
    Tensor other_casted =
        (other.dtype() == result_dtype) ? other : other.to(result_dtype);
    TensorIterator iter =
        TensorIterator::binary_op(out, self_casted, other_casted);

#if defined(__x86_64__)
    if (result_dtype == DType::Float32 && cpu_has_avx2f()) {
        const float alpha_val = alpha.to<float>();
        iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
            if (strides[0] == 4 && strides[1] == 4 && strides[2] == 4) {
                binary_f32_vec(
                    BIN_ADD, reinterpret_cast<const float*>(data[1]),
                    reinterpret_cast<const float*>(data[2]),
                    reinterpret_cast<float*>(data[0]), n, alpha_val);
                return;
            }
            if (strides[0] == 4 && strides[1] == 4 && strides[2] == 0) {
                binary_f32_vec_scalar_b(
                    BIN_ADD, reinterpret_cast<const float*>(data[1]),
                    *reinterpret_cast<const float*>(data[2]),
                    reinterpret_cast<float*>(data[0]), n, alpha_val);
                return;
            }
            if (strides[0] == 4 && strides[1] == 0 && strides[2] == 4) {
                binary_f32_vec_scalar_a(
                    BIN_ADD, *reinterpret_cast<const float*>(data[1]),
                    reinterpret_cast<const float*>(data[2]),
                    reinterpret_cast<float*>(data[0]), n, alpha_val);
                return;
            }
            for (int64_t i = 0; i < n; ++i) {
                *reinterpret_cast<float*>(data[0] + i * strides[0]) =
                    *reinterpret_cast<const float*>(data[1] + i * strides[1]) +
                    alpha_val *
                        *reinterpret_cast<const float*>(data[2] + i * strides[2]);
            }
        });
        return true;
    }
#endif

#define TI_ADD_OUT_CASE(ctype, name) \
    case DType::name: { \
        using ctype_ = ctype; \
        iter.for_each([&alpha](char** data, const int64_t* strides, int64_t n) { \
            for (int64_t i = 0; i < n; ++i) { \
                auto* dst = reinterpret_cast<ctype_*>(data[0] + i * strides[0]); \
                const auto* lhs = reinterpret_cast<const ctype_*>( \
                    data[1] + i * strides[1]); \
                const auto* rhs = reinterpret_cast<const ctype_*>( \
                    data[2] + i * strides[2]); \
                *dst = *lhs + tp_alpha_scaled(alpha, *rhs); \
            } \
        }); \
        break; \
    }
    switch (result_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TI_ADD_OUT_CASE)
        default: TP_THROW(TypeError, "add.out: unsupported dtype");
    }
#undef TI_ADD_OUT_CASE
    return true;
}

Tensor add_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha) {
#if defined(__x86_64__)
    bool plain_layout = true;
#ifdef USE_ONEDNN
    plain_layout =
        !self.unsafeGetTensorImpl()->has_onednn_md() &&
        !other.unsafeGetTensorImpl()->has_onednn_md();
#endif
    const float fast_alpha = alpha.to<float>();
    if (plain_layout && cpu_has_avx512() &&
        self.dtype() == DType::Float32 && other.dtype() == DType::Float32 &&
        self.shape() == other.shape() && self.is_contiguous() &&
        other.is_contiguous() &&
        (fast_alpha == 1.0f || fast_alpha == -1.0f)) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::Float32,
            self.device());
        const int64_t n = self.numel();
        const float* a = self.data_ptr<float>();
        const float* b = other.data_ptr<float>();
        float* y = result.data_ptr<float>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            if (fast_alpha == 1.0f) {
                add_f32_avx512(a + begin, b + begin, y + begin, end - begin);
            } else {
                sub_f32_avx512(a + begin, b + begin, y + begin, end - begin);
            }
        });
        return result;
    }

    // Operands that share one dense layout stream flat in storage order:
    // allocate the result with the same strides so a permuted pair (e.g.
    // two transposes) runs one unit-stride pass instead of gathering a
    // cache line per element.
    if (plain_layout && (cpu_has_avx512() || cpu_has_avx2f()) &&
        (self.dtype() == DType::Float32 || self.dtype() == DType::Float64) &&
        other.dtype() == self.dtype() && self.shape() == other.shape() &&
        !alpha.isComplex()) {
        const std::vector<int64_t> sizes =
            static_cast<std::vector<int64_t>>(self.shape());
        const std::vector<int64_t> strides =
            static_cast<std::vector<int64_t>>(self.strides());
        if (strides == static_cast<std::vector<int64_t>>(other.strides()) &&
            SizesAndStrides::is_non_overlapping_and_dense(sizes, strides)) {
            Tensor result = Tensor::empty(sizes, self.dtype(), self.device());
            result = result.as_strided(sizes, strides, 0);
            const int64_t n = self.numel();
            if (self.dtype() == DType::Float32) {
                binary_f32_vec(BIN_ADD, self.data_ptr<float>(),
                               other.data_ptr<float>(),
                               result.data_ptr<float>(), n, alpha.to<float>());
            } else {
                binary_f64_vec(BIN_ADD, self.data_ptr<double>(),
                               other.data_ptr<double>(),
                               result.data_ptr<double>(), n, alpha.to<double>());
            }
            return result;
        }
    }
#endif
    #ifdef USE_ONEDNN
    if (OneDNNContext::is_enabled()) {
        auto self_impl = self.unsafeGetTensorImpl();
        auto other_impl = other.unsafeGetTensorImpl();
        bool self_blocked = self_impl->has_onednn_md();
        bool other_blocked = other_impl->has_onednn_md();

        if (self_blocked || other_blocked) {
            bool match = false;
            if (self_blocked && other_blocked) {
                auto md1 = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                auto md2 = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                if (*md1 == *md2) match = true;
            }

            if (match) {
                 // Optimization: Treat as contiguous 1D array
                 auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());

                 std::vector<int64_t> out_shape = broadcast_shapes(self.sizes(), other.sizes());

                 DType result_dtype = promoteTypes(self.dtype(), other.dtype());
                 if (alpha.isFloatingPoint() && !isFloatingType(result_dtype)) {
                     result_dtype = promoteTypes(result_dtype, DType::Float32);
                 }
                 Tensor result = Tensor::empty(out_shape, result_dtype, self.device());

                 size_t req_size = md->get_size();
                 if (result.numel() * sizeof(float) < req_size) {
                      Allocator* allocator = getAllocator(result.device().type());
                      Storage new_storage(req_size, allocator);
                      result.unsafeGetTensorImpl()->set_storage(new_storage);
                 }
                 result.unsafeGetTensorImpl()->set_onednn_md(self_impl->get_onednn_md());

                 int64_t n = req_size / sizeof(float);
                 float alpha_val = alpha.to<float>();
                 float* r_ptr = result.data_ptr<float>();
                 const float* s_ptr = self.data_ptr<float>();
                 const float* o_ptr = other.data_ptr<float>();

                 // Reuse AVX logic
                 #if defined(__AVX512F__)
                 if (std::abs(alpha_val - 1.0f) < 1e-6) {
                      parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                      for (int64_t i = begin; i < end; i += 16) {
                          if (i + 16 <= end) {
                              __m512 a = _mm512_loadu_ps(s_ptr + i);
                              __m512 b = _mm512_loadu_ps(o_ptr + i);
                              _mm512_storeu_ps(r_ptr + i, _mm512_add_ps(a, b));
                          } else {
                              for (int64_t j = i; j < end; ++j) r_ptr[j] = s_ptr[j] + o_ptr[j];
                          }
                      }
                      });
                 } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                      parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                      for (int64_t i = begin; i < end; i += 16) {
                          if (i + 16 <= end) {
                              __m512 a = _mm512_loadu_ps(s_ptr + i);
                              __m512 b = _mm512_loadu_ps(o_ptr + i);
                              _mm512_storeu_ps(r_ptr + i, _mm512_sub_ps(a, b));
                          } else {
                              for (int64_t j = i; j < end; ++j) r_ptr[j] = s_ptr[j] - o_ptr[j];
                          }
                      }
                      });
                 } else {
                      __m512 valpha = _mm512_set1_ps(alpha_val);
                      parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                      for (int64_t i = begin; i < end; i += 16) {
                          if (i + 16 <= end) {
                              __m512 a = _mm512_loadu_ps(s_ptr + i);
                              __m512 b = _mm512_loadu_ps(o_ptr + i);
                              _mm512_storeu_ps(r_ptr + i, _mm512_fmadd_ps(valpha, b, a));
                          } else {
                              for (int64_t j = i; j < end; ++j) r_ptr[j] = s_ptr[j] + alpha_val * o_ptr[j];
                          }
                      }
                      });
                 }
                 #elif defined(__AVX2__)
                 if (std::abs(alpha_val - 1.0f) < 1e-6) {
                      parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                      for (int64_t i = begin; i < end; i += 8) {
                          if (i + 8 <= end) {
                              __m256 a = _mm256_loadu_ps(s_ptr + i);
                              __m256 b = _mm256_loadu_ps(o_ptr + i);
                              _mm256_storeu_ps(r_ptr + i, _mm256_add_ps(a, b));
                          } else {
                              for (int64_t j = i; j < end; ++j) r_ptr[j] = s_ptr[j] + o_ptr[j];
                          }
                      }
                      });
                 } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                      parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                      for (int64_t i = begin; i < end; i += 8) {
                          if (i + 8 <= end) {
                              __m256 a = _mm256_loadu_ps(s_ptr + i);
                              __m256 b = _mm256_loadu_ps(o_ptr + i);
                              _mm256_storeu_ps(r_ptr + i, _mm256_sub_ps(a, b));
                          } else {
                              for (int64_t j = i; j < end; ++j) r_ptr[j] = s_ptr[j] - o_ptr[j];
                          }
                      }
                      });
                 } else {
                      __m256 valpha = _mm256_set1_ps(alpha_val);
                      parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                      for (int64_t i = begin; i < end; i += 8) {
                          if (i + 8 <= end) {
                              __m256 a = _mm256_loadu_ps(s_ptr + i);
                              __m256 b = _mm256_loadu_ps(o_ptr + i);
                              _mm256_storeu_ps(r_ptr + i, _mm256_add_ps(a, _mm256_mul_ps(valpha, b)));
                          } else {
                              for (int64_t j = i; j < end; ++j) r_ptr[j] = s_ptr[j] + alpha_val * o_ptr[j];
                          }
                      }
                      });
                 }
                 #else
                 for(int64_t i=0; i<n; ++i) r_ptr[i] = s_ptr[i] + alpha_val * o_ptr[i];
                 #endif
                 
                 return result;
            } else {
                 // Reorder to NCHW
                 auto& eng = OneDNNContext::get_engine();
                 auto& s = OneDNNContext::get_stream();
                 
                 Tensor self_nchw = self;
                 if (self_blocked) {
                      auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                      dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(self.shape());
                      auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
                      self_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
                      auto src_mem = dnnl::memory(*md, eng, self.data_ptr<float>());
                      auto dst_mem = dnnl::memory(nchw_md, eng, self_nchw.data_ptr<float>());
                      dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                 }
                 
                 Tensor other_nchw = other;
                 if (other_blocked) {
                      auto md = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                      dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(other.shape());
                      auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
                      other_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(other.shape()), other.dtype(), other.device());
                      auto src_mem = dnnl::memory(*md, eng, other.data_ptr<float>());
                      auto dst_mem = dnnl::memory(nchw_md, eng, other_nchw.data_ptr<float>());
                      dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                 }
                 if (self_blocked || other_blocked) s.wait();
                 
                 return add_kernel(self_nchw, other_nchw, alpha);
            }
        }
    }
    #endif

    std::vector<int64_t> out_shape = broadcast_shapes(self.sizes(), other.sizes());
    DType result_dtype = native::result_type(self, other);
    if (alpha.isFloatingPoint() && !isFloatingType(result_dtype)) {
        result_dtype = promoteTypes(result_dtype, DType::Float32);
    }
    Tensor result = empty_binary_result(self, other, out_shape, result_dtype);

    bool optimized = false;
    // Optimization for same-shape tensors (handles contiguous and non-contiguous via temporary copies)
    if (result_dtype == DType::Float32 &&
        self.dtype() == DType::Float32 &&
        other.dtype() == DType::Float32 &&
        self.shape() == other.shape()) {

        // Flat iteration needs one consistent storage order: channels-last
        // operands are consumed in their own physical order, everything else
        // normalizes to row-major.
        const bool flat_cl = binary_keep_channels_last(self, other) &&
                             result.is_contiguous(MemoryFormat::ChannelsLast);
        // Create contiguous accessors (might trigger copy)
        Tensor self_contig = (flat_cl || self.is_contiguous()) ? self : self.contiguous();
        Tensor other_contig = (flat_cl || other.is_contiguous()) ? other : other.contiguous();
        // Result matches the chosen order (allocated channels-last above when
        // both operands are channels-last), but check to be safe or if passed in
        Tensor result_contig = (flat_cl || result.is_contiguous()) ? result : result.contiguous();

        int64_t n = self_contig.numel();
        float alpha_val = alpha.to<float>();
        float* r_ptr = result_contig.data_ptr<float>();
        const float* s_ptr = self_contig.data_ptr<float>();
        const float* o_ptr = other_contig.data_ptr<float>();

#if defined(__x86_64__)
        // AVX-512 runtime dispatch: full-width add for any alpha.
        if (cpu_has_avx512()) {
            if (alpha_val == 1.0f) {
                parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                    add_f32_avx512(s_ptr + begin, o_ptr + begin, r_ptr + begin,
                                   end - begin);
                });
            } else {
                parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                    binary_f32_avx512(BIN_ADD, s_ptr + begin, o_ptr + begin,
                                      r_ptr + begin, end - begin, alpha_val);
                });
            }
            optimized = true;
        }
#endif
        #ifdef USE_MKL
        if (std::abs(alpha_val - 1.0f) < 1e-6) {
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                if (begin < end) {
                    vsAdd(end - begin, s_ptr + begin, o_ptr + begin, r_ptr + begin);
                }
            });
            optimized = true;
        } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                if (begin < end) {
                    vsSub(end - begin, s_ptr + begin, o_ptr + begin, r_ptr + begin);
                }
            });
            optimized = true;
        }
        #endif
        
        if (!optimized) {
            #if defined(__AVX512F__)
            if (std::abs(alpha_val - 1.0f) < 1e-6) {
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 16) {
                     if (i + 16 <= end) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(r_ptr + i, _mm512_add_ps(a, b));
                     } else {
                         for (int64_t j = i; j < end; ++j) r_ptr[j] = s_ptr[j] + o_ptr[j];
                     }
                 }
                 });
            } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 16) {
                     if (i + 16 <= end) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(r_ptr + i, _mm512_sub_ps(a, b));
                     } else {
                         for (int64_t j = i; j < end; ++j) r_ptr[j] = s_ptr[j] - o_ptr[j];
                     }
                 }
                 });
            } else {
                 __m512 valpha = _mm512_set1_ps(alpha_val);
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 16) {
                     if (i + 16 <= end) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(r_ptr + i, _mm512_fmadd_ps(valpha, b, a));
                     } else {
                         for (int64_t j = i; j < end; ++j) r_ptr[j] = s_ptr[j] + alpha_val * o_ptr[j];
                     }
                 }
                 });
            }
            #elif defined(__AVX2__)
            if (std::abs(alpha_val - 1.0f) < 1e-6) {
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 8) {
                     if (i + 8 <= end) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(r_ptr + i, _mm256_add_ps(a, b));
                     } else {
                         for (int64_t j = i; j < end; ++j) r_ptr[j] = s_ptr[j] + o_ptr[j];
                     }
                 }
                 });
            } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 8) {
                     if (i + 8 <= end) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(r_ptr + i, _mm256_sub_ps(a, b));
                     } else {
                         for (int64_t j = i; j < end; ++j) r_ptr[j] = s_ptr[j] - o_ptr[j];
                     }
                 }
                 });
            } else {
                 __m256 valpha = _mm256_set1_ps(alpha_val);
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 8) {
                     if (i + 8 <= end) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         // Use mul+add instead of fmadd to be safe with AVX2 but no FMA3
                         _mm256_storeu_ps(r_ptr + i, _mm256_add_ps(a, _mm256_mul_ps(valpha, b)));
                     } else {
                         for (int64_t j = i; j < end; ++j) r_ptr[j] = s_ptr[j] + alpha_val * o_ptr[j];
                     }
                 }
                 });
            }
            #else
            if (std::abs(alpha_val - 1.0f) < 1e-6) {
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; ++i) {
                     r_ptr[i] = s_ptr[i] + o_ptr[i];
                 }
                 });
             } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; ++i) {
                     r_ptr[i] = s_ptr[i] - o_ptr[i];
                 }
                 });
             } else {
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; ++i) {
                     r_ptr[i] = s_ptr[i] + alpha_val * o_ptr[i];
                 }
                 });
             }
            #endif
            optimized = true;
        }
    }
    
#if defined(__x86_64__)
    // Native BF16 add: widen once, apply alpha in float32, and round once on
    // store.  The generic TensorIterator path is scalar for this dtype.
    if (self.dtype() == DType::BFloat16 && other.dtype() == DType::BFloat16 &&
        self.is_contiguous() && other.is_contiguous() &&
        self.shape() == other.shape() && !alpha.isComplex()) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::BFloat16,
            self.device());
        bf16_binary_contiguous(BIN_ADD, self.data_ptr<BFloat16>(),
                               other.data_ptr<BFloat16>(),
                               result.data_ptr<BFloat16>(), self.numel(),
                               static_cast<float>(static_cast<BFloat16>(
                                   alpha.to<float>())));
        return result;
    }
#endif

    // AVX2 complex fast path (cpu/VecComplex.h): contiguous same-shape
    // add/sub with alpha == +/-1 (bit-identical to the generic path since
    // v_t(±1.0)*x == ±x exactly); other alphas keep the iterator loop.
    if (!optimized &&
        (self.dtype() == DType::ComplexFloat ||
         self.dtype() == DType::ComplexDouble) &&
        other.dtype() == self.dtype() && self.shape() == other.shape() &&
        self.numel() >= 4096) {
        const double a = alpha.toDouble();
        if (a == 1.0 || a == -1.0) {
            const bool flat_cl = binary_keep_channels_last(self, other);
            Tensor self_contig = (flat_cl || self.is_contiguous()) ? self : self.contiguous();
            Tensor other_contig = (flat_cl || other.is_contiguous()) ? other : other.contiguous();
            Tensor out = empty_binary_result(self, other,
                static_cast<std::vector<int64_t>>(self.shape()), self.dtype());
            const veccomplex::Op vop =
                a == 1.0 ? veccomplex::Op::Add : veccomplex::Op::Sub;
            if (veccomplex::try_binary(self_contig.data_ptr(),
                                       other_contig.data_ptr(), out.data_ptr(),
                                       self.numel(), self.dtype(), vop)) {
                return out;
            }
        }
    }

#if defined(__x86_64__)
    // AVX-512 runtime dispatch for float64 same-shape add (no fast path
    // existed before -- f64 went through the TensorIterator scalar loop).
    if (!optimized && cpu_has_avx512() &&
        result_dtype == DType::Float64 &&
        self.dtype() == DType::Float64 &&
        other.dtype() == DType::Float64 &&
        self.shape() == other.shape()) {
        const bool flat_cl = binary_keep_channels_last(self, other) &&
                             result.is_contiguous(MemoryFormat::ChannelsLast);
        Tensor self_contig = (flat_cl || self.is_contiguous()) ? self : self.contiguous();
        Tensor other_contig = (flat_cl || other.is_contiguous()) ? other : other.contiguous();
        Tensor result_contig = (flat_cl || result.is_contiguous()) ? result : result.contiguous();
        int64_t n = self_contig.numel();
        const double alpha_val = alpha.toDouble();
        double* r_ptr = result_contig.data_ptr<double>();
        const double* s_ptr = self_contig.data_ptr<double>();
        const double* o_ptr = other_contig.data_ptr<double>();
        if (alpha_val == 1.0) {
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                add_f64_avx512(s_ptr + begin, o_ptr + begin, r_ptr + begin,
                               end - begin);
            });
        } else {
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                binary_f64_avx512(BIN_ADD, s_ptr + begin, o_ptr + begin,
                                  r_ptr + begin, end - begin, alpha_val);
            });
        }
        optimized = true;
    }
#endif

    if (!optimized) {
        Tensor self_casted = (self.dtype() == result_dtype) ? self : self.to(result_dtype);
        Tensor other_casted = (other.dtype() == result_dtype) ? other : other.to(result_dtype);

        // alpha * y with per-dtype semantics; a template so the if constexpr
        // branches are truly discarded per instantiation.
        TensorIterator iter = TensorIterator::binary_op(result, self_casted, other_casted);

#if defined(__x86_64__)
        // contiguous dim into each run (broadcast operands included), so runs
        // whose strides all equal the element size vectorize exactly like the
        // same-shape path.  This is the bias-add pattern ((N,G)+(G,)) that
        // the RNN cells hit at every timestep; the scalar run loop below was
        if (result_dtype == DType::Float32 && cpu_has_avx2f()) {
            const float alpha_val = alpha.to<float>();
            iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
                if (strides[0] == 4 && strides[1] == 4 && strides[2] == 4) {
                    binary_f32_vec(BIN_ADD,
                                      reinterpret_cast<const float*>(data[1]),
                                      reinterpret_cast<const float*>(data[2]),
                                      reinterpret_cast<float*>(data[0]),
                                      n, alpha_val);
                } else if (strides[0] == 4 && strides[1] == 4 && strides[2] == 0) {
                    binary_f32_vec_scalar_b(BIN_ADD,
                                               reinterpret_cast<const float*>(data[1]),
                                               *reinterpret_cast<const float*>(data[2]),
                                               reinterpret_cast<float*>(data[0]),
                                               n, alpha_val);
                } else if (strides[0] == 4 && strides[1] == 0 && strides[2] == 4) {
                    binary_f32_vec_scalar_a(BIN_ADD,
                                               *reinterpret_cast<const float*>(data[1]),
                                               reinterpret_cast<const float*>(data[2]),
                                               reinterpret_cast<float*>(data[0]),
                                               n, alpha_val);
                } else {
                    for (int64_t i = 0; i < n; ++i)
                        *reinterpret_cast<float*>(data[0] + i * strides[0]) =
                            *reinterpret_cast<const float*>(data[1] + i * strides[1]) +
                            alpha_val *
                            *reinterpret_cast<const float*>(data[2] + i * strides[2]);
                }
            });
            optimized = true;
        }

        // Broadcast float64 add: same-shape dense runs took the AVX-512 path
        // above, so what reaches the iterator here is exactly the
        // broadcast/strided residue.
        if (!optimized && result_dtype == DType::Float64 && cpu_has_avx2f()) {
            const double alpha_val = alpha.toDouble();
            iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
                if (strides[0] == 8 && strides[1] == 8 && strides[2] == 8) {
                    binary_f64_vec(BIN_ADD,
                                      reinterpret_cast<const double*>(data[1]),
                                      reinterpret_cast<const double*>(data[2]),
                                      reinterpret_cast<double*>(data[0]),
                                      n, alpha_val);
                } else if (strides[0] == 8 && strides[1] == 8 && strides[2] == 0) {
                    binary_f64_vec_scalar_b(BIN_ADD,
                                               reinterpret_cast<const double*>(data[1]),
                                               *reinterpret_cast<const double*>(data[2]),
                                               reinterpret_cast<double*>(data[0]),
                                               n, alpha_val);
                } else if (strides[0] == 8 && strides[1] == 0 && strides[2] == 8) {
                    binary_f64_vec_scalar_a(BIN_ADD,
                                               *reinterpret_cast<const double*>(data[1]),
                                               reinterpret_cast<const double*>(data[2]),
                                               reinterpret_cast<double*>(data[0]),
                                               n, alpha_val);
                } else {
                    for (int64_t i = 0; i < n; ++i)
                        *reinterpret_cast<double*>(data[0] + i * strides[0]) =
                            *reinterpret_cast<const double*>(data[1] + i * strides[1]) +
                            alpha_val *
                            *reinterpret_cast<const double*>(data[2] + i * strides[2]);
                }
            });
            optimized = true;
        }
#endif

        if (!optimized) {
        #define TI_ALPHA_CASE(ctype, name) \
        case DType::name: { \
            using ctype_ = ctype; \
            iter.for_each([&alpha](char** data, const int64_t* strides, int64_t n) { \
                auto op = [alpha](ctype_ x, ctype_ y) -> ctype_ { \
                    return x + tp_alpha_scaled(alpha, y); \
                }; \
                for (int64_t i = 0; i < n; ++i) \
                    *reinterpret_cast<ctype_*>(data[0] + i * strides[0]) = op( \
                        *reinterpret_cast<const ctype_*>(data[1] + i * strides[1]), \
                        *reinterpret_cast<const ctype_*>(data[2] + i * strides[2])); \
            }); \
            break; \
        }
        switch (result_dtype) {
            TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TI_ALPHA_CASE)
            default: TP_THROW(TypeError, "add_out: unsupported dtype");
        }
        #undef TI_ALPHA_CASE
        }
    }
    return result;
}

Tensor& add_out_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha,
                       Tensor& out) {
    if (self.device() != other.device() || self.device() != out.device()) {
        TP_THROW(DeviceMismatchError,
                 "add.out: all tensors must be on the same device");
    }
    if (GradMode::is_enabled() &&
        (self.requires_grad() || other.requires_grad() || out.requires_grad())) {
        TP_THROW(RuntimeError,
                 "add.out: functions with out arguments do not support automatic differentiation");
    }

    const DType result_dtype =
        add_result_dtype(self.dtype(), other.dtype(), alpha);
    if (out.dtype() != result_dtype) {
        TP_THROW(RuntimeError,
                 "add.out: expected output dtype ",
                 static_cast<int>(result_dtype), ", but got ",
                 static_cast<int>(out.dtype()));
    }

    const std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    if (static_cast<std::vector<int64_t>>(out.shape()) == out_shape) {
        if (try_add_float32_out(self, other, alpha, out) ||
            try_add_tensor_iterator_out(self, other, alpha, result_dtype, out)) {
            return out;
        }
    }

    Tensor result = add_kernel(self, other, alpha);
    if (out.shape() == result.shape()) {
        out.copy_(result);
    } else {
        out.unsafeGetTensorImpl()->copy_metadata_from(
            *result.unsafeGetTensorImpl());
    }
    return out;
}

Tensor add_relu_same_shape(const Tensor& self, const Tensor& other) {
    Tensor result = Tensor::empty(
        static_cast<std::vector<int64_t>>(self.shape()), DType::Float32, self.device());
    const int64_t n = self.numel();
    const float* self_ptr = self.data_ptr<float>();
    const float* other_ptr = other.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();

    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        int64_t i = begin;
#if defined(__AVX512F__)
        const __m512 zero = _mm512_setzero_ps();
        for (; i + 16 <= end; i += 16) {
            const __m512 sum = _mm512_add_ps(
                _mm512_loadu_ps(self_ptr + i), _mm512_loadu_ps(other_ptr + i));
            _mm512_storeu_ps(result_ptr + i, _mm512_max_ps(zero, sum));
        }
#elif defined(__AVX2__)
        const __m256 zero = _mm256_setzero_ps();
        for (; i + 8 <= end; i += 8) {
            const __m256 sum = _mm256_add_ps(
                _mm256_loadu_ps(self_ptr + i), _mm256_loadu_ps(other_ptr + i));
            _mm256_storeu_ps(result_ptr + i, _mm256_max_ps(zero, sum));
        }
#endif
        for (; i < end; ++i) {
            result_ptr[i] = std::max(0.0f, self_ptr[i] + other_ptr[i]);
        }
    });
    return result;
}

Tensor add_relu_cpu(const Tensor& self, const Tensor& other) {
    bool plain_layout = true;
#ifdef USE_ONEDNN
    plain_layout =
        !self.unsafeGetTensorImpl()->has_onednn_md() &&
        !other.unsafeGetTensorImpl()->has_onednn_md();
#endif
    if (self.dtype() == DType::Float32 && other.dtype() == DType::Float32 &&
        self.shape() == other.shape() && self.is_contiguous() &&
        other.is_contiguous() && plain_layout) {
        return add_relu_same_shape(self, other);
    }

    Tensor result = add_kernel(self, other, Scalar(1));
    return relu_inplace_kernel(result);
}

Tensor sub_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha) {
    if (self.dtype() == DType::Bool && other.dtype() == DType::Bool) {
        TP_THROW(RuntimeError,
                 "Subtraction, the `-` operator, with two bool tensors is not "
                 "supported. Use the `^` or `logical_xor()` operator instead.");
    }
    if (alpha.isFloatingPoint()) {
        return add_kernel(self, other, Scalar(-alpha.toDouble()));
    } else {
        if (alpha.isIntegral()) {
             return add_kernel(self, other, Scalar(-alpha.to<int64_t>()));
        }
        return add_kernel(self, other, Scalar(-alpha.to<double>()));
    }
}

Tensor mul_kernel(const Tensor& self, const Tensor& other) {
#if defined(__x86_64__)
    // Operands that share one dense layout stream flat in storage order:
    // allocate the result with the same strides so a permuted pair (e.g.
    // two transposes) runs one unit-stride pass instead of gathering a
    // cache line per element.
    bool plain_layout = true;
#ifdef USE_ONEDNN
    plain_layout =
        !self.unsafeGetTensorImpl()->has_onednn_md() &&
        !other.unsafeGetTensorImpl()->has_onednn_md();
#endif
    if (plain_layout && (cpu_has_avx512() || cpu_has_avx2f()) &&
        (self.dtype() == DType::Float32 || self.dtype() == DType::Float64) &&
        other.dtype() == self.dtype() && self.shape() == other.shape()) {
        const std::vector<int64_t> sizes =
            static_cast<std::vector<int64_t>>(self.shape());
        const std::vector<int64_t> strides =
            static_cast<std::vector<int64_t>>(self.strides());
        if (strides == static_cast<std::vector<int64_t>>(other.strides()) &&
            SizesAndStrides::is_non_overlapping_and_dense(sizes, strides)) {
            Tensor result = Tensor::empty(sizes, self.dtype(), self.device());
            result = result.as_strided(sizes, strides, 0);
            const int64_t n = self.numel();
            if (self.dtype() == DType::Float32) {
                binary_f32_vec(BIN_MUL, self.data_ptr<float>(),
                               other.data_ptr<float>(),
                               result.data_ptr<float>(), n, 1.0f);
            } else {
                binary_f64_vec(BIN_MUL, self.data_ptr<double>(),
                               other.data_ptr<double>(),
                               result.data_ptr<double>(), n, 1.0);
            }
            return result;
        }
    }
#endif
    #ifdef USE_ONEDNN
    if (OneDNNContext::is_enabled()) {
        auto self_impl = self.unsafeGetTensorImpl();
        auto other_impl = other.unsafeGetTensorImpl();
        bool self_blocked = self_impl->has_onednn_md();
        bool other_blocked = other_impl->has_onednn_md();

        if (self_blocked || other_blocked) {
            bool match = false;
            if (self_blocked && other_blocked) {
                auto md1 = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                auto md2 = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                if (*md1 == *md2) match = true;
            }

            if (match) {
                 auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                 std::vector<int64_t> out_shape = broadcast_shapes(self.sizes(), other.sizes());
                 DType result_dtype = promoteTypes(self.dtype(), other.dtype());
                 Tensor result = Tensor::empty(out_shape, result_dtype, self.device());

                 size_t req_size = md->get_size();
                 if (result.numel() * sizeof(float) < req_size) {
                      Allocator* allocator = getAllocator(result.device().type());
                      Storage new_storage(req_size, allocator);
                      result.unsafeGetTensorImpl()->set_storage(new_storage);
                 }
                 result.unsafeGetTensorImpl()->set_onednn_md(self_impl->get_onednn_md());

                 int64_t n = req_size / sizeof(float);
                 float* r_ptr = result.data_ptr<float>();
                 const float* s_ptr = self.data_ptr<float>();
                 const float* o_ptr = other.data_ptr<float>();

                 #if defined(__AVX512F__)
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 16) {
                     if (i + 16 <= end) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(r_ptr + i, _mm512_mul_ps(a, b));
                     } else {
                         for (int64_t j = i; j < end; ++j) r_ptr[j] = s_ptr[j] * o_ptr[j];
                     }
                 }
                 });
                 #elif defined(__AVX2__)
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 8) {
                     if (i + 8 <= end) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(r_ptr + i, _mm256_mul_ps(a, b));
                     } else {
                         for (int64_t j = i; j < end; ++j) r_ptr[j] = s_ptr[j] * o_ptr[j];
                     }
                 }
                 });
                 #else
                 for(int64_t i=0; i<n; ++i) r_ptr[i] = s_ptr[i] * o_ptr[i];
                 #endif
                 return result;
            } else {
                 // Reorder to NCHW
                 auto& eng = OneDNNContext::get_engine();
                 auto& s = OneDNNContext::get_stream();
                 
                 Tensor self_nchw = self;
                 if (self_blocked) {
                      auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                      dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(self.shape());
                      auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
                      self_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
                      auto src_mem = dnnl::memory(*md, eng, self.data_ptr<float>());
                      auto dst_mem = dnnl::memory(nchw_md, eng, self_nchw.data_ptr<float>());
                      dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                 }
                 
                 Tensor other_nchw = other;
                 if (other_blocked) {
                      auto md = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                      dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(other.shape());
                      auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
                      other_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(other.shape()), other.dtype(), other.device());
                      auto src_mem = dnnl::memory(*md, eng, other.data_ptr<float>());
                      auto dst_mem = dnnl::memory(nchw_md, eng, other_nchw.data_ptr<float>());
                      dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                 }
                 if (self_blocked || other_blocked) s.wait();
                 return mul_kernel(self_nchw, other_nchw);
            }
        }
    }
    #endif

#if defined(__x86_64__)
    if (self.dtype() == DType::BFloat16 && other.dtype() == DType::BFloat16 &&
        self.is_contiguous() && other.is_contiguous() &&
        self.shape() == other.shape()) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::BFloat16,
            self.device());
        bf16_binary_contiguous(BIN_MUL, self.data_ptr<BFloat16>(),
                               other.data_ptr<BFloat16>(),
                               result.data_ptr<BFloat16>(), self.numel(), 1.0f);
        return result;
    }
#endif

    // AVX2 complex fast path (cpu/VecComplex.h): contiguous same-shape mul.
    if ((self.dtype() == DType::ComplexFloat ||
         self.dtype() == DType::ComplexDouble) &&
        other.dtype() == self.dtype() && self.shape() == other.shape() &&
        self.numel() >= 4096) {
        Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
        Tensor other_contig = other.is_contiguous() ? other : other.contiguous();
        Tensor out = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), self.dtype(),
            self.device());
        if (veccomplex::try_binary(self_contig.data_ptr(),
                                   other_contig.data_ptr(), out.data_ptr(),
                                   self.numel(), self.dtype(),
                                   veccomplex::Op::Mul)) {
            return out;
        }
    }

#if defined(__x86_64__)
    // AVX-512 runtime dispatch: contiguous same-shape real mul.
    if ((self.dtype() == DType::Float32 || self.dtype() == DType::Float64) &&
        other.dtype() == self.dtype() && self.shape() == other.shape() &&
        self.numel() >= 4096 && cpu_has_avx512()) {
        Tensor a = self.is_contiguous() ? self : self.contiguous();
        Tensor b = other.is_contiguous() ? other : other.contiguous();
        Tensor out = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), self.dtype(),
            self.device());
        const bool f32 = self.dtype() == DType::Float32;
        parallel_for(0, self.numel(), GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            if (f32)
                binary_f32_avx512(BIN_MUL, a.data_ptr<float>() + begin,
                                  b.data_ptr<float>() + begin,
                                  out.data_ptr<float>() + begin, end - begin, 0.f);
            else
                binary_f64_avx512(BIN_MUL, a.data_ptr<double>() + begin,
                                  b.data_ptr<double>() + begin,
                                  out.data_ptr<double>() + begin, end - begin, 0.0);
        });
        return out;
    }
#endif

    auto op = [](auto a, auto b) { return a * b; };
    auto mkl_op = [](int n, float* a, float* b, float* y) {
        #ifdef USE_MKL
        vsMul(n, a, b, y);
        #endif
    };
    return binary_op_kernel_impl(self, other, op, mkl_op, true, false, BIN_MUL);
}

Tensor div_kernel(const Tensor& self, const Tensor& other) {
#if defined(__x86_64__)
    // Operands that share one dense layout stream flat in storage order:
    // allocate the result with the same strides so a permuted pair (e.g.
    // two transposes) runs one unit-stride pass instead of gathering a
    // cache line per element.
    bool plain_layout = true;
#ifdef USE_ONEDNN
    plain_layout =
        !self.unsafeGetTensorImpl()->has_onednn_md() &&
        !other.unsafeGetTensorImpl()->has_onednn_md();
#endif
    if (plain_layout && (cpu_has_avx512() || cpu_has_avx2f()) &&
        (self.dtype() == DType::Float32 || self.dtype() == DType::Float64) &&
        other.dtype() == self.dtype() && self.shape() == other.shape()) {
        const std::vector<int64_t> sizes =
            static_cast<std::vector<int64_t>>(self.shape());
        const std::vector<int64_t> strides =
            static_cast<std::vector<int64_t>>(self.strides());
        if (strides == static_cast<std::vector<int64_t>>(other.strides()) &&
            SizesAndStrides::is_non_overlapping_and_dense(sizes, strides)) {
            Tensor result = Tensor::empty(sizes, self.dtype(), self.device());
            result = result.as_strided(sizes, strides, 0);
            const int64_t n = self.numel();
            if (self.dtype() == DType::Float32) {
                binary_f32_vec(BIN_DIV, self.data_ptr<float>(),
                               other.data_ptr<float>(),
                               result.data_ptr<float>(), n, 1.0f);
            } else {
                binary_f64_vec(BIN_DIV, self.data_ptr<double>(),
                               other.data_ptr<double>(),
                               result.data_ptr<double>(), n, 1.0);
            }
            return result;
        }
    }
#endif
    #ifdef USE_ONEDNN
    if (OneDNNContext::is_enabled()) {
        auto self_impl = self.unsafeGetTensorImpl();
        auto other_impl = other.unsafeGetTensorImpl();
        bool self_blocked = self_impl->has_onednn_md();
        bool other_blocked = other_impl->has_onednn_md();

        if (self_blocked || other_blocked) {
            bool match = false;
            if (self_blocked && other_blocked) {
                auto md1 = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                auto md2 = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                if (*md1 == *md2) match = true;
            }

            if (match) {
                 auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                 std::vector<int64_t> out_shape = broadcast_shapes(self.sizes(), other.sizes());
                 DType result_dtype = promoteTypes(self.dtype(), other.dtype());
                 if (result_dtype != DType::Float32) result_dtype = DType::Float32; // Div always produces float
                 Tensor result = Tensor::empty(out_shape, result_dtype, self.device());

                 size_t req_size = md->get_size();
                 if (result.numel() * sizeof(float) < req_size) {
                      Allocator* allocator = getAllocator(result.device().type());
                      Storage new_storage(req_size, allocator);
                      result.unsafeGetTensorImpl()->set_storage(new_storage);
                 }
                 result.unsafeGetTensorImpl()->set_onednn_md(self_impl->get_onednn_md());

                 int64_t n = req_size / sizeof(float);
                 float* r_ptr = result.data_ptr<float>();
                 const float* s_ptr = self.data_ptr<float>();
                 const float* o_ptr = other.data_ptr<float>();

                 #if defined(__AVX512F__)
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 16) {
                     if (i + 16 <= end) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(r_ptr + i, _mm512_div_ps(a, b));
                     } else {
                         for (int64_t j = i; j < end; ++j) r_ptr[j] = s_ptr[j] / o_ptr[j];
                     }
                 }
                 });
                 #elif defined(__AVX2__)
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 8) {
                     if (i + 8 <= end) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(r_ptr + i, _mm256_div_ps(a, b));
                     } else {
                         for (int64_t j = i; j < end; ++j) r_ptr[j] = s_ptr[j] / o_ptr[j];
                     }
                 }
                 });
                 #else
                 for(int64_t i=0; i<n; ++i) r_ptr[i] = s_ptr[i] / o_ptr[i];
                 #endif
                 return result;
            } else {
                 // Reorder to NCHW
                 auto& eng = OneDNNContext::get_engine();
                 auto& s = OneDNNContext::get_stream();
                 
                 Tensor self_nchw = self;
                 if (self_blocked) {
                      auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                      dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(self.shape());
                      auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
                      self_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
                      auto src_mem = dnnl::memory(*md, eng, self.data_ptr<float>());
                      auto dst_mem = dnnl::memory(nchw_md, eng, self_nchw.data_ptr<float>());
                      dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                 }
                 
                 Tensor other_nchw = other;
                 if (other_blocked) {
                      auto md = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                      dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(other.shape());
                      auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
                      other_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(other.shape()), other.dtype(), other.device());
                      auto src_mem = dnnl::memory(*md, eng, other.data_ptr<float>());
                      auto dst_mem = dnnl::memory(nchw_md, eng, other_nchw.data_ptr<float>());
                      dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                 }
                 if (self_blocked || other_blocked) s.wait();
                 return div_kernel(self_nchw, other_nchw);
            }
        }
    }
    #endif

#if defined(__x86_64__)
    if (self.dtype() == DType::BFloat16 && other.dtype() == DType::BFloat16 &&
        self.is_contiguous() && other.is_contiguous() &&
        self.shape() == other.shape()) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::BFloat16,
            self.device());
        bf16_binary_contiguous(BIN_DIV, self.data_ptr<BFloat16>(),
                               other.data_ptr<BFloat16>(),
                               result.data_ptr<BFloat16>(), self.numel(), 1.0f);
        return result;
    }
#endif

    // AVX2 complex fast path (cpu/VecComplex.h): contiguous same-shape div
    if ((self.dtype() == DType::ComplexFloat ||
         self.dtype() == DType::ComplexDouble) &&
        other.dtype() == self.dtype() && self.shape() == other.shape() &&
        self.numel() >= 4096) {
        Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
        Tensor other_contig = other.is_contiguous() ? other : other.contiguous();
        Tensor out = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), self.dtype(),
            self.device());
        if (veccomplex::try_binary(self_contig.data_ptr(),
                                   other_contig.data_ptr(), out.data_ptr(),
                                   self.numel(), self.dtype(),
                                   veccomplex::Op::Div)) {
            return out;
        }
    }

#if defined(__x86_64__)
    // AVX-512 runtime dispatch: contiguous same-shape real div (IEEE vdivps,
    // up here).
    if ((self.dtype() == DType::Float32 || self.dtype() == DType::Float64) &&
        other.dtype() == self.dtype() && self.shape() == other.shape() &&
        self.numel() >= 4096 && cpu_has_avx512()) {
        Tensor a = self.is_contiguous() ? self : self.contiguous();
        Tensor b = other.is_contiguous() ? other : other.contiguous();
        Tensor out = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), self.dtype(),
            self.device());
        const bool f32 = self.dtype() == DType::Float32;
        parallel_for(0, self.numel(), GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            if (f32)
                binary_f32_avx512(BIN_DIV, a.data_ptr<float>() + begin,
                                  b.data_ptr<float>() + begin,
                                  out.data_ptr<float>() + begin, end - begin, 0.f);
            else
                binary_f64_avx512(BIN_DIV, a.data_ptr<double>() + begin,
                                  b.data_ptr<double>() + begin,
                                  out.data_ptr<double>() + begin, end - begin, 0.0);
        });
        return out;
    }
#endif

    auto op = [](auto a, auto b) {
        using T = std::decay_t<decltype(a)>;
        if constexpr (std::is_same_v<T, bool>) return static_cast<float>(a) / static_cast<float>(b);
        else if constexpr (is_complex_type_v<T>) {
            // Reduced-precision complex elements compute in their opmath
            // domain; float/double complex divide in their own precision.
            using F = opmath_type<T>;
            const F af(a.real(), a.imag());
            const F bf(b.real(), b.imag());
            const F rf = af / bf;
            return T(static_cast<typename T::value_type>(rf.real()),
                     static_cast<typename T::value_type>(rf.imag()));
        } else return a / b;
    };
    auto mkl_op = [](int n, float* a, float* b, float* y) {
        #ifdef USE_MKL
        vsDiv(n, a, b, y);
        #endif
    };
    return binary_op_kernel_impl(self, other, op, mkl_op, true, true, BIN_DIV);
}

// Stax pointwise fusion primitive.  Keep the generic path on the existing
// p10 kernels so broadcasting, promotion, layouts, and future devices retain
// their established semantics.  The contiguous float32 path is the hot path
// emitted by the Stax scheduler and performs one allocation and one parallel
// pass for mul+add.
Tensor fused_mul_add_kernel(const Tensor& self, const Tensor& other, const Tensor& addend) {
    if (self.device() == other.device() && self.device() == addend.device() &&
        self.dtype() == DType::Float32 && other.dtype() == DType::Float32 &&
        addend.dtype() == DType::Float32 && self.is_contiguous() &&
        other.is_contiguous() && addend.is_contiguous() &&
        self.shape() == other.shape() && self.shape() == addend.shape()) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::Float32, self.device());
        const float* self_ptr = self.data_ptr<float>();
        const float* other_ptr = other.data_ptr<float>();
        const float* addend_ptr = addend.data_ptr<float>();
        float* result_ptr = result.data_ptr<float>();
        const int64_t n = self.numel();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                result_ptr[i] = self_ptr[i] * other_ptr[i] + addend_ptr[i];
            }
        });
        return result;
    }

    return add_kernel(mul_kernel(self, other), addend, Scalar(1));
}

Tensor fused_mul_add_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& addend) {
    if (self.dtype() == DType::Float32 && self.is_contiguous()) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::Float32, self.device());
        const float* self_ptr = self.data_ptr<float>();
        float* result_ptr = result.data_ptr<float>();
        const float mul_value = static_cast<float>(other.toDouble());
        const float add_value = static_cast<float>(addend.toDouble());
        const int64_t n = self.numel();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                result_ptr[i] = self_ptr[i] * mul_value + add_value;
            }
        });
        return result;
    }

    return add_scalar_kernel(mul_scalar_kernel(self, other), addend, Scalar(1));
}

// --- Inplace Binary Kernels ---

Tensor& add_inplace_kernel(Tensor& self, const Tensor& other, const Scalar& alpha) {
    if (other.is_sparse()) {
        return add_sparse_to_dense_cpu(self, other, alpha);
    }
    std::vector<int64_t> out_shape = broadcast_shapes(self.sizes(), other.sizes());

    if (static_cast<std::vector<int64_t>>(self.shape()) != out_shape) {
        TP_THROW(RuntimeError, "output with shape " + self.shape().toString() + " doesn't match the broadcast shape " + Size(out_shape).toString());
    }

    if (self.shape() != other.shape()) {
    }

#if defined(__x86_64__)
    if (self.dtype() == DType::BFloat16 && other.dtype() == DType::BFloat16 &&
        self.is_contiguous() && other.is_contiguous() &&
        self.shape() == other.shape() && !alpha.isComplex()) {
        bf16_binary_contiguous(BIN_ADD, self.data_ptr<BFloat16>(),
                               other.data_ptr<BFloat16>(),
                               self.data_ptr<BFloat16>(), self.numel(),
                               static_cast<float>(static_cast<BFloat16>(
                                   alpha.to<float>())));
        return self;
    }

    // Muon's common mixed-precision update: FP32 parameter plus BF16 update.
    // Compute in FP32 and keep the destination dtype unchanged, matching the
    // native TensorIterator opmath path.
    if (self.dtype() == DType::Float32 && other.dtype() == DType::BFloat16 &&
        self.is_contiguous() && other.is_contiguous() &&
        self.shape() == other.shape() && !alpha.isComplex() &&
        cpu_has_avx512()) {
        f32_bf16_add_inplace_contiguous(
            self.data_ptr<float>(), other.data_ptr<BFloat16>(), self.numel(),
            alpha.to<float>());
        return self;
    }

    // Match the contiguous FP32 tensor-add path used by the native optimizer
    // kernels without paying the generic iterator or BLAS wrapper overhead.
    if (self.dtype() == DType::Float32 && other.dtype() == DType::Float32 &&
        self.is_contiguous() && other.is_contiguous() &&
        self.shape() == other.shape() && !alpha.isComplex() &&
        (cpu_has_avx512() || cpu_has_avx2f())) {
        binary_f32_contiguous(BIN_ADD, self.data_ptr<float>(),
                              other.data_ptr<float>(), self.data_ptr<float>(),
                              self.numel(), alpha.to<float>());
        return self;
    }
#endif

    #ifdef USE_ONEDNN
    if (OneDNNContext::is_enabled()) {
        auto self_impl = self.unsafeGetTensorImpl();
        auto other_impl = other.unsafeGetTensorImpl();
        bool self_blocked = self_impl->has_onednn_md();
        bool other_blocked = other_impl->has_onednn_md();
        
        if (self_blocked && self.shape() != other.shape()) {
            // Broadcasting required. Unblock self (convert to dense).
            auto& eng = OneDNNContext::get_engine();
            auto& s = OneDNNContext::get_stream();
            auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
            
            dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(self.shape());
            auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);

            Tensor self_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
            auto src_mem = dnnl::memory(*md, eng, self.data_ptr<float>());
            auto dst_mem = dnnl::memory(nchw_md, eng, self_nchw.data_ptr<float>());
            
            try {
                dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                s.wait();
            } catch (const dnnl::error& e) {
                TP_THROW(RuntimeError, "OneDNN reorder failed in add_inplace_kernel (unblock): " + std::string(e.message));
            }
            
            // Replace storage and clear MD
            self.unsafeGetTensorImpl()->set_storage(self_nchw.unsafeGetTensorImpl()->storage());
            self.unsafeGetTensorImpl()->set_onednn_md(nullptr);
            self_blocked = false;
        }

        if (self_blocked) {
             // Self is blocked, we MUST preserve it.
             Tensor other_matching = other;
             bool match = false;
             if (other_blocked) {
                 auto md1 = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                 auto md2 = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                 if (*md1 == *md2) match = true;
             }
             
             if (!match) {
                  auto& eng = OneDNNContext::get_engine();
                  auto& s = OneDNNContext::get_stream();
                  auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                  
                  size_t req_size = md->get_size();
                  
                  Tensor temp = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
                  if (temp.numel() * sizeof(float) < req_size) {
                      Allocator* allocator = getAllocator(self.device().type());
                      Storage new_storage(req_size, allocator);
                      temp.unsafeGetTensorImpl()->set_storage(new_storage);
                  }
                  temp.unsafeGetTensorImpl()->set_onednn_md(self_impl->get_onednn_md());
                  
                  auto dst_mem = dnnl::memory(*md, eng, temp.data_ptr<float>());
                  
                  try {
                       if (other_blocked) {
                            auto other_md = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                            auto src_mem = dnnl::memory(*other_md, eng, other.data_ptr<float>());
                            dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                       } else {
                            dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(other.shape());
                            auto src_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
                            auto src_mem = dnnl::memory(src_md, eng, other.data_ptr<float>());
                            dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                       }
                       s.wait();
                   } catch (const std::exception& e) {
                      TP_THROW(RuntimeError, "OneDNN reorder failed in add_inplace_kernel (match block): " + std::string(e.what()));
                  }
                  other_matching = temp;
             }
             
             float alpha_val = alpha.to<float>();
             auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
             int64_t n = md->get_size() / sizeof(float);
             
             float* s_ptr = self.data_ptr<float>();
             const float* o_ptr = other_matching.data_ptr<float>();
             
             #if defined(__AVX512F__)
             if (std::abs(alpha_val - 1.0f) < 1e-6) {
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 16) {
                     if (i + 16 <= end) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(s_ptr + i, _mm512_add_ps(a, b));
                     } else {
                         for (int64_t j = i; j < end; ++j) s_ptr[j] += o_ptr[j];
                     }
                 }
                 });
             } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 16) {
                     if (i + 16 <= end) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(s_ptr + i, _mm512_sub_ps(a, b));
                     } else {
                         for (int64_t j = i; j < end; ++j) s_ptr[j] -= o_ptr[j];
                     }
                 }
                 });
             } else {
                 __m512 valpha = _mm512_set1_ps(alpha_val);
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 16) {
                     if (i + 16 <= end) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(s_ptr + i, _mm512_fmadd_ps(valpha, b, a));
                     } else {
                         for (int64_t j = i; j < end; ++j) s_ptr[j] += alpha_val * o_ptr[j];
                     }
                 }
                 });
             }
             #elif defined(__AVX2__)
             if (std::abs(alpha_val - 1.0f) < 1e-6) {
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 8) {
                     if (i + 8 <= end) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(s_ptr + i, _mm256_add_ps(a, b));
                     } else {
                         for (int64_t j = i; j < end; ++j) s_ptr[j] += o_ptr[j];
                     }
                 }
                 });
             } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 8) {
                     if (i + 8 <= end) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(s_ptr + i, _mm256_sub_ps(a, b));
                     } else {
                         for (int64_t j = i; j < end; ++j) s_ptr[j] -= o_ptr[j];
                     }
                 }
                 });
             } else {
                 __m256 valpha = _mm256_set1_ps(alpha_val);
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 8) {
                     if (i + 8 <= end) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(s_ptr + i, _mm256_add_ps(a, _mm256_mul_ps(valpha, b)));
                     } else {
                         for (int64_t j = i; j < end; ++j) s_ptr[j] += alpha_val * o_ptr[j];
                     }
                 }
                 });
             }
             #else
             for (int64_t i = 0; i < n; ++i) s_ptr[i] += alpha_val * o_ptr[i];
             #endif
             return self;
        } else if (other_blocked) {
             auto& eng = OneDNNContext::get_engine();
             auto& s = OneDNNContext::get_stream();
             auto md = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
             dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(other.shape());
             auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
             
             Tensor other_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(other.shape()), other.dtype(), other.device());
             try {
                 auto src_mem = dnnl::memory(*md, eng, other.data_ptr<float>());
                 auto dst_mem = dnnl::memory(nchw_md, eng, other_nchw.data_ptr<float>());
                 dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                 s.wait();
             } catch (const std::exception& e) {
                 TP_THROW(RuntimeError, "OneDNN reorder failed in add_inplace_kernel (other_blocked): " + std::string(e.what()));
             }
             
             return add_inplace_kernel(self, other_nchw, alpha);
        }
    }
    #endif

    bool optimized = false;
    if (self.dtype() == DType::Float32 && 
        other.dtype() == DType::Float32 &&
        self.is_contiguous() && other.is_contiguous() &&
        self.shape() == other.shape()) {
        
        float alpha_val = alpha.to<float>();
        int64_t n = self.numel();

        #if defined(USE_MKL) || defined(USE_BLAS)
        cblas_saxpy((int)n, alpha_val, other.data_ptr<float>(), 1, self.data_ptr<float>(), 1);
        optimized = true;
        #endif

        if (!optimized) {
             float* s_ptr = self.data_ptr<float>();
             const float* o_ptr = other.data_ptr<float>();
             
             #if defined(__AVX512F__)
             if (std::abs(alpha_val - 1.0f) < 1e-6) {
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 16) {
                     if (i + 16 <= end) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(s_ptr + i, _mm512_add_ps(a, b));
                     } else {
                         for (int64_t j = i; j < end; ++j) s_ptr[j] += o_ptr[j];
                     }
                 }
                 });
             } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 16) {
                     if (i + 16 <= end) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(s_ptr + i, _mm512_sub_ps(a, b));
                     } else {
                         for (int64_t j = i; j < end; ++j) s_ptr[j] -= o_ptr[j];
                     }
                 }
                 });
             } else {
                 __m512 valpha = _mm512_set1_ps(alpha_val);
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 16) {
                     if (i + 16 <= end) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(s_ptr + i, _mm512_fmadd_ps(valpha, b, a));
                     } else {
                         for (int64_t j = i; j < end; ++j) s_ptr[j] += alpha_val * o_ptr[j];
                     }
                 }
                 });
             }
             #elif defined(__AVX2__)
             if (std::abs(alpha_val - 1.0f) < 1e-6) {
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 8) {
                     if (i + 8 <= end) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(s_ptr + i, _mm256_add_ps(a, b));
                     } else {
                         for (int64_t j = i; j < end; ++j) s_ptr[j] += o_ptr[j];
                     }
                 }
                 });
             } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 8) {
                     if (i + 8 <= end) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(s_ptr + i, _mm256_sub_ps(a, b));
                     } else {
                         for (int64_t j = i; j < end; ++j) s_ptr[j] -= o_ptr[j];
                     }
                 }
                 });
             } else {
                 __m256 valpha = _mm256_set1_ps(alpha_val);
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; i += 8) {
                     if (i + 8 <= end) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(s_ptr + i, _mm256_add_ps(a, _mm256_mul_ps(valpha, b)));
                     } else {
                         for (int64_t j = i; j < end; ++j) s_ptr[j] += alpha_val * o_ptr[j];
                     }
                 }
                 });
             }
             #else
             if (std::abs(alpha_val - 1.0f) < 1e-6) {
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; ++i) {
                     s_ptr[i] += o_ptr[i];
                 }
                 });
             } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; ++i) {
                     s_ptr[i] -= o_ptr[i];
                 }
                 });
             } else {
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                 for (int64_t i = begin; i < end; ++i) {
                     s_ptr[i] += alpha_val * o_ptr[i];
                 }
                 });
             }
             #endif
             optimized = true;
        }
    }
    
    if (!optimized) {
        Tensor other_c = (other.dtype() == self.dtype()) ? other : other.to(self.dtype());
        { auto op = [alpha](auto x, auto y) {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_floating_point_v<T> || is_complex_type_v<T>) return x + alpha.to<T>() * y;
            else if (alpha.isFloatingPoint()) return static_cast<T>(x + alpha.toDouble() * y);
            else return static_cast<T>(x + alpha.to<int64_t>() * y);
        };
            ti_apply_arith(self, self, other_c, op);
        }
    }
    return self;
}

Tensor& sub_inplace_kernel(Tensor& self, const Tensor& other, const Scalar& alpha) {
    if (alpha.isFloatingPoint()) {
        return add_inplace_kernel(self, other, Scalar(-alpha.toDouble()));
    } else {
        if (alpha.isIntegral()) {
             return add_inplace_kernel(self, other, Scalar(-alpha.to<int64_t>()));
        }
        return add_inplace_kernel(self, other, Scalar(-alpha.to<double>()));
    }
}

Tensor& mul_inplace_kernel(Tensor& self, const Tensor& other) {
    std::vector<int64_t> out_shape = broadcast_shapes(self.sizes(), other.sizes());
    if (static_cast<std::vector<int64_t>>(self.shape()) != out_shape) TP_THROW(RuntimeError, "mul_: shape mismatch");

    #ifdef USE_ONEDNN
    if (OneDNNContext::is_enabled()) {
        auto self_impl = self.unsafeGetTensorImpl();
        auto other_impl = other.unsafeGetTensorImpl();
        bool self_blocked = self_impl->has_onednn_md();
        bool other_blocked = other_impl->has_onednn_md();
        
        if (self_blocked) {
             Tensor other_matching = other;
             bool match = false;
             if (other_blocked) {
                 auto md1 = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                 auto md2 = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                 if (*md1 == *md2) match = true;
             }
             
             if (!match) {
                  auto& eng = OneDNNContext::get_engine();
                  auto& s = OneDNNContext::get_stream();
                  auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                  
                  size_t req_size = md->get_size();
                  Tensor temp = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
                  if (temp.numel() * sizeof(float) < req_size) {
                      Allocator* allocator = getAllocator(self.device().type());
                      Storage new_storage(req_size, allocator);
                      temp.unsafeGetTensorImpl()->set_storage(new_storage);
                  }
                  temp.unsafeGetTensorImpl()->set_onednn_md(self_impl->get_onednn_md());
                  
                  auto dst_mem = dnnl::memory(*md, eng, temp.data_ptr<float>());
                  
                  if (other_blocked) {
                       auto other_md = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                       auto src_mem = dnnl::memory(*other_md, eng, other.data_ptr<float>());
                       dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                  } else {
                       dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(other.shape());
                       auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
                       auto src_mem = dnnl::memory(nchw_md, eng, other.data_ptr<float>());
                       dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                  }
                  s.wait();
                  other_matching = temp;
             }
             
             auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
             int64_t n = md->get_size() / sizeof(float);
             
             float* s_ptr = self.data_ptr<float>();
             const float* o_ptr = other_matching.data_ptr<float>();
             
             #if defined(__AVX512F__)
             parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
             for (int64_t i = begin; i < end; i += 16) {
                 if (i + 16 <= end) {
                     __m512 a = _mm512_loadu_ps(s_ptr + i);
                     __m512 b = _mm512_loadu_ps(o_ptr + i);
                     _mm512_storeu_ps(s_ptr + i, _mm512_mul_ps(a, b));
                 } else {
                     for (int64_t j = i; j < end; ++j) s_ptr[j] *= o_ptr[j];
                 }
             }
             });
             #elif defined(__AVX2__)
             parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
             for (int64_t i = begin; i < end; i += 8) {
                 if (i + 8 <= end) {
                     __m256 a = _mm256_loadu_ps(s_ptr + i);
                     __m256 b = _mm256_loadu_ps(o_ptr + i);
                     _mm256_storeu_ps(s_ptr + i, _mm256_mul_ps(a, b));
                 } else {
                     for (int64_t j = i; j < end; ++j) s_ptr[j] *= o_ptr[j];
                 }
             }
             });
             #else
             for (int64_t i = 0; i < n; ++i) s_ptr[i] *= o_ptr[i];
             #endif
             return self;
        } else if (other_blocked) {
             auto& eng = OneDNNContext::get_engine();
             auto& s = OneDNNContext::get_stream();
             auto md = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
             dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(other.shape());
             auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
             
             Tensor other_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(other.shape()), other.dtype(), other.device());
             auto src_mem = dnnl::memory(*md, eng, other.data_ptr<float>());
             auto dst_mem = dnnl::memory(nchw_md, eng, other_nchw.data_ptr<float>());
             dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
             s.wait();
             
             return mul_inplace_kernel(self, other_nchw);
        }
    }
    #endif

    bool optimized = false;
    if (self.dtype() == DType::Float32 && other.dtype() == DType::Float32 &&
        self.is_contiguous() && other.is_contiguous() && self.shape() == other.shape()) {
        #ifdef USE_MKL
        int64_t n = self.numel();
        vsMul((int)n, self.data_ptr<float>(), other.data_ptr<float>(), self.data_ptr<float>());
        optimized = true;
        #endif
    }
    
    if (!optimized) {
        Tensor other_c = (other.dtype() == self.dtype()) ? other : other.to(self.dtype());
        { auto op = [](auto x, auto y) { return x * y; };
            ti_apply_arith(self, self, other_c, op);
        }
    }
    return self;
}

Tensor& div_inplace_kernel(Tensor& self, const Tensor& other) {
    std::vector<int64_t> out_shape = broadcast_shapes(self.sizes(), other.sizes());
    if (static_cast<std::vector<int64_t>>(self.shape()) != out_shape) TP_THROW(RuntimeError, "div_: shape mismatch");

#if defined(__x86_64__)
    // Muon's norm is a 0-d FP32 tensor while the working matrix is BF16.
    // TensorIterator handles that broadcast correctly, but its per-element
    // dtype dispatch is needlessly expensive for this hot scalar-broadcast
    // case.  Load the scalar once and use the same float32-widen/BF16-store
    // kernel as the tensor-scalar operator.  A one-element tensor with a
    // non-scalar shape (e.g. [1]) is broadcast identically here.
    if (self.dtype() == DType::BFloat16 && self.is_contiguous() &&
        self.numel() >= 4096 && other.numel() == 1 &&
        other.is_contiguous() && !other.is_sparse()) {
        float divisor = 0.0f;
        bool scalar_loaded = true;
        switch (other.dtype()) {
            case DType::Float32:
                divisor = other.data_ptr<float>()[0];
                break;
            case DType::Float64:
                divisor = static_cast<float>(other.data_ptr<double>()[0]);
                break;
            case DType::Float16:
                divisor = static_cast<float>(other.data_ptr<Half>()[0]);
                break;
            case DType::BFloat16:
                divisor = static_cast<float>(other.data_ptr<BFloat16>()[0]);
                break;
            default:
                scalar_loaded = false;
                break;
        }
        if (scalar_loaded) {
            bf16_scalar_contiguous(BIN_DIV, self.data_ptr<BFloat16>(),
                                   self.data_ptr<BFloat16>(), self.numel(),
                                   divisor);
            return self;
        }
    }
#endif

    #ifdef USE_ONEDNN
    if (OneDNNContext::is_enabled()) {
        auto self_impl = self.unsafeGetTensorImpl();
        auto other_impl = other.unsafeGetTensorImpl();
        bool self_blocked = self_impl->has_onednn_md();
        bool other_blocked = other_impl->has_onednn_md();
        
        if (self_blocked) {
             Tensor other_matching = other;
             bool match = false;
             if (other_blocked) {
                 auto md1 = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                 auto md2 = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                 if (*md1 == *md2) match = true;
             }
             
             if (!match) {
                  auto& eng = OneDNNContext::get_engine();
                  auto& s = OneDNNContext::get_stream();
                  auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                  
                  size_t req_size = md->get_size();
                  Tensor temp = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
                  if (temp.numel() * sizeof(float) < req_size) {
                      Allocator* allocator = getAllocator(self.device().type());
                      Storage new_storage(req_size, allocator);
                      temp.unsafeGetTensorImpl()->set_storage(new_storage);
                  }
                  temp.unsafeGetTensorImpl()->set_onednn_md(self_impl->get_onednn_md());
                  
                  auto dst_mem = dnnl::memory(*md, eng, temp.data_ptr<float>());
                  
                  if (other_blocked) {
                       auto other_md = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                       auto src_mem = dnnl::memory(*other_md, eng, other.data_ptr<float>());
                       dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                  } else {
                       dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(other.shape());
                       auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
                       auto src_mem = dnnl::memory(nchw_md, eng, other.data_ptr<float>());
                       dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                  }
                  s.wait();
                  other_matching = temp;
             }
             
             auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
             int64_t n = md->get_size() / sizeof(float);
             
             float* s_ptr = self.data_ptr<float>();
             const float* o_ptr = other_matching.data_ptr<float>();
             
             #if defined(__AVX512F__)
             parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
             for (int64_t i = begin; i < end; i += 16) {
                 if (i + 16 <= end) {
                     __m512 a = _mm512_loadu_ps(s_ptr + i);
                     __m512 b = _mm512_loadu_ps(o_ptr + i);
                     _mm512_storeu_ps(s_ptr + i, _mm512_div_ps(a, b));
                 } else {
                     for (int64_t j = i; j < end; ++j) s_ptr[j] /= o_ptr[j];
                 }
             }
             });
             #elif defined(__AVX2__)
             parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
             for (int64_t i = begin; i < end; i += 8) {
                 if (i + 8 <= end) {
                     __m256 a = _mm256_loadu_ps(s_ptr + i);
                     __m256 b = _mm256_loadu_ps(o_ptr + i);
                     _mm256_storeu_ps(s_ptr + i, _mm256_div_ps(a, b));
                 } else {
                     for (int64_t j = i; j < end; ++j) s_ptr[j] /= o_ptr[j];
                 }
             }
             });
             #else
             for (int64_t i = 0; i < n; ++i) s_ptr[i] /= o_ptr[i];
             #endif
             return self;
        } else if (other_blocked) {
             auto& eng = OneDNNContext::get_engine();
             auto& s = OneDNNContext::get_stream();
             auto md = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
             dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(other.shape());
             auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
             
             Tensor other_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(other.shape()), other.dtype(), other.device());
             auto src_mem = dnnl::memory(*md, eng, other.data_ptr<float>());
             auto dst_mem = dnnl::memory(nchw_md, eng, other_nchw.data_ptr<float>());
             dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
             s.wait();
             
             return div_inplace_kernel(self, other_nchw);
        }
    }
    #endif

    bool optimized = false;
    if (self.dtype() == DType::Float32 && other.dtype() == DType::Float32 &&
        self.is_contiguous() && other.is_contiguous() && self.shape() == other.shape()) {
        #ifdef USE_MKL
        int64_t n = self.numel();
        vsDiv((int)n, self.data_ptr<float>(), other.data_ptr<float>(), self.data_ptr<float>());
        optimized = true;
        #endif
    }
    
    if (!optimized) {
        Tensor other_c = (other.dtype() == self.dtype()) ? other : other.to(self.dtype());
        { auto op = [](auto x, auto y) {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, bool>) return static_cast<bool>(static_cast<int>(x) / static_cast<int>(y));
            else return x / y;
        };
            ti_apply_arith(self, self, other_c, op);
        }
    }
    return self;
}



// --- Scalar Kernels ---

namespace {
// complex scalar widens any REAL tensor to its complex width (float64 ->
// complex128, everything else -> complex64); a wrapped float promotes
// integral tensors to Float32.
inline DType scalar_result_dtype(DType self_dt, const Scalar& other,
                                 const Scalar* alpha = nullptr) {
    DType dt = result_type(other, self_dt);
    // A real or complex alpha scales the scalar in its own category, so it
    // moves the result the way such a scalar would.
    if (alpha != nullptr && (alpha->isFloatingPoint() || alpha->isComplex())) {
        dt = result_type(*alpha, dt);
    }
    return dt;
}
} // namespace

Tensor add_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha) {
    DType result_dtype = scalar_result_dtype(self.dtype(), other, &alpha);

#if defined(__x86_64__)
    // Native add.Scalar lowers a wrapped scalar to the same vectorized
    // TensorIterator loop as add.Tensor.  Keep the scalar product in the
    // tensor dtype before entering the AVX-512 loop so promotion and rounding
    // agree with the generic path.
    if (self.dtype() == DType::Float32 && result_dtype == DType::Float32 &&
        !other.isComplex() && !alpha.isComplex() && self.is_contiguous() &&
        (cpu_has_avx512() || cpu_has_avx2f())) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::Float32,
            self.device());
        const float scalar = alpha.to<float>() * other.to<float>();
        scalar_f32_contiguous(BIN_ADD, self.data_ptr<float>(),
                              result.data_ptr<float>(), self.numel(), scalar);
        return result;
    }
    if (self.dtype() == DType::Float64 && result_dtype == DType::Float64 &&
        !other.isComplex() && !alpha.isComplex() && self.is_contiguous() &&
        (cpu_has_avx512() || cpu_has_avx2f())) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::Float64,
            self.device());
        const double scalar = alpha.to<double>() * other.to<double>();
        scalar_f64_contiguous(BIN_ADD, self.data_ptr<double>(),
                              result.data_ptr<double>(), self.numel(), scalar);
        return result;
    }
#endif

    Tensor result = empty_scalar_result(self, result_dtype);
    Tensor self_casted = (self.dtype() == result_dtype) ? self : self.to(result_dtype);

    #define OP_CASE(ctype, name) \
    case DType::name: { \
        auto op = [other, alpha](ctype a) -> ctype { \
            return static_cast<ctype>(a + alpha.to<ctype>() * other.to<ctype>()); \
        }; \
        apply_unary_op_recursive<ctype>(result.data_ptr<ctype>(), result.strides(), \
                                       self_casted, self_casted.strides(), \
                                       0, 0, 0, static_cast<std::vector<int64_t>>(self.shape()), op); \
        break; \
    }

    switch (result_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(OP_CASE)
        default: TP_THROW(TypeError, "add_scalar: unsupported dtype");
    }
    #undef OP_CASE

    return result;
}

Tensor sub_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha) {
    DType result_dtype = scalar_result_dtype(self.dtype(), other, &alpha);

#if defined(__x86_64__)
    // x - alpha*other folds into one scalar term, so the contiguous float
    // surface takes the same vectorized loop as add.Scalar with the negated
    // product instead of the recursive elementwise functor.
    if (self.dtype() == DType::Float32 && result_dtype == DType::Float32 &&
        !other.isComplex() && !alpha.isComplex() && self.is_contiguous() &&
        (cpu_has_avx512() || cpu_has_avx2f())) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::Float32,
            self.device());
        const float scalar = alpha.to<float>() * other.to<float>();
        scalar_f32_contiguous(BIN_ADD, self.data_ptr<float>(),
                              result.data_ptr<float>(), self.numel(), -scalar);
        return result;
    }
    if (self.dtype() == DType::Float64 && result_dtype == DType::Float64 &&
        !other.isComplex() && !alpha.isComplex() && self.is_contiguous() &&
        (cpu_has_avx512() || cpu_has_avx2f())) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::Float64,
            self.device());
        const double scalar = alpha.to<double>() * other.to<double>();
        scalar_f64_contiguous(BIN_ADD, self.data_ptr<double>(),
                              result.data_ptr<double>(), self.numel(), -scalar);
        return result;
    }
#endif

    Tensor result = empty_scalar_result(self, result_dtype);
    Tensor self_casted = (self.dtype() == result_dtype) ? self : self.to(result_dtype);

    #define OP_CASE(ctype, name) \
    case DType::name: { \
        auto op = [other, alpha](ctype a) -> ctype { \
            return static_cast<ctype>(a - alpha.to<ctype>() * other.to<ctype>()); \
        }; \
        apply_unary_op_recursive<ctype>(result.data_ptr<ctype>(), result.strides(), \
                                       self_casted, self_casted.strides(), \
                                       0, 0, 0, static_cast<std::vector<int64_t>>(self.shape()), op); \
        break; \
    }

    switch (result_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(OP_CASE)
        default: TP_THROW(TypeError, "sub_scalar: unsupported dtype");
    }
    #undef OP_CASE

    return result;
}

Tensor mul_scalar_kernel(const Tensor& self, const Scalar& other) {
    DType result_dtype = scalar_result_dtype(self.dtype(), other);

#if defined(__x86_64__)
    if (self.dtype() == DType::Float32 && result_dtype == DType::Float32 &&
        !other.isComplex() && self.is_contiguous() && (cpu_has_avx512() || cpu_has_avx2f())) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::Float32,
            self.device());
        scalar_f32_contiguous(BIN_MUL, self.data_ptr<float>(),
                              result.data_ptr<float>(), self.numel(),
                              other.to<float>());
        return result;
    }
    if (self.dtype() == DType::Float64 && result_dtype == DType::Float64 &&
        !other.isComplex() && self.is_contiguous() && (cpu_has_avx512() || cpu_has_avx2f())) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::Float64,
            self.device());
        scalar_f64_contiguous(BIN_MUL, self.data_ptr<double>(),
                              result.data_ptr<double>(), self.numel(),
                              other.to<double>());
        return result;
    }
#endif

#if defined(__x86_64__)
    if (self.dtype() == DType::BFloat16 && result_dtype == DType::BFloat16 &&
        !other.isComplex() && self.is_contiguous()) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::BFloat16,
            self.device());
        bf16_scalar_contiguous(BIN_MUL, self.data_ptr<BFloat16>(),
                               result.data_ptr<BFloat16>(), self.numel(),
                               other.to<float>());
        return result;
    }
#endif

    Tensor result = empty_scalar_result(self, result_dtype);
    Tensor self_casted = (self.dtype() == result_dtype) ? self : self.to(result_dtype);

    #define OP_CASE(ctype, name) \
    case DType::name: { \
        auto op = [other](ctype a) -> ctype { \
            return static_cast<ctype>(a * other.to<ctype>()); \
        }; \
        apply_unary_op_recursive<ctype>(result.data_ptr<ctype>(), result.strides(), \
                                       self_casted, self_casted.strides(), \
                                       0, 0, 0, static_cast<std::vector<int64_t>>(self.shape()), op); \
        break; \
    }

    switch (result_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(OP_CASE)
        default: TP_THROW(TypeError, "mul_scalar: unsupported dtype");
    }
    #undef OP_CASE

    return result;
}

Tensor div_scalar_kernel(const Tensor& self, const Scalar& other) {
    // True division promotes integral tensors to Float32 (or ComplexFloat
    // for a wrapped complex divisor), while preserving floating tensor
    DType result_dtype = self.dtype();
    if (!isFloatingOrComplexType(result_dtype)) {
        result_dtype = other.isComplex() ? DType::ComplexFloat : DType::Float32;
    } else if (!isComplexType(result_dtype) && other.isComplex()) {
        result_dtype = promoteTypes(toComplexType(result_dtype), other.dtype());
    }

#if defined(__x86_64__)
    if (self.dtype() == DType::Float32 && result_dtype == DType::Float32 &&
        !other.isComplex() && self.is_contiguous() && (cpu_has_avx512() || cpu_has_avx2f())) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::Float32,
            self.device());
        scalar_f32_contiguous(BIN_DIV, self.data_ptr<float>(),
                              result.data_ptr<float>(), self.numel(),
                              other.to<float>());
        return result;
    }
    if (self.dtype() == DType::Float64 && result_dtype == DType::Float64 &&
        !other.isComplex() && self.is_contiguous() && (cpu_has_avx512() || cpu_has_avx2f())) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::Float64,
            self.device());
        scalar_f64_contiguous(BIN_DIV, self.data_ptr<double>(),
                              result.data_ptr<double>(), self.numel(),
                              other.to<double>());
        return result;
    }
#endif

#if defined(__x86_64__)
    if (self.dtype() == DType::BFloat16 && result_dtype == DType::BFloat16 &&
        !other.isComplex() && self.is_contiguous()) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::BFloat16,
            self.device());
        bf16_scalar_contiguous(BIN_DIV, self.data_ptr<BFloat16>(),
                               result.data_ptr<BFloat16>(), self.numel(),
                               other.to<float>());
        return result;
    }
#endif

    Tensor result = empty_scalar_result(self, result_dtype);
    Tensor self_casted = (self.dtype() == result_dtype) ? self : self.to(result_dtype);

    // Half/BFloat16 element types do not model a floating-point type, so the
    // division runs in a float complex domain. The template makes the branch
    // dependent; a lambda inside the switch would still semantically check
    // the dead branch for every element type the macro expands.
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        auto op = [other](ctype a) -> ctype { \
            return div_scalar_element(a, other); \
        }; \
        apply_unary_op_recursive<ctype>(result.data_ptr<ctype>(), result.strides(), \
                                       self_casted, self_casted.strides(), \
                                       0, 0, 0, static_cast<std::vector<int64_t>>(self.shape()), op); \
        break; \
    }

    switch (result_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(OP_CASE)
        default: TP_THROW(TypeError, "div_scalar: unsupported dtype");
    }
    #undef OP_CASE

    return result;
}

// Inplace Scalar
Tensor& add_scalar_inplace_kernel(Tensor& self, const Scalar& other, const Scalar& alpha) {
#if defined(__x86_64__)
    if ((self.dtype() == DType::Float32 || self.dtype() == DType::Float64) &&
        !other.isComplex() && !alpha.isComplex() && self.is_contiguous() &&
        (cpu_has_avx512() || cpu_has_avx2f())) {
        if (self.dtype() == DType::Float32) {
            const float scalar = alpha.to<float>() * other.to<float>();
            scalar_f32_contiguous(BIN_ADD, self.data_ptr<float>(),
                                  self.data_ptr<float>(), self.numel(), scalar);
        } else {
            const double scalar = alpha.to<double>() * other.to<double>();
            scalar_f64_contiguous(BIN_ADD, self.data_ptr<double>(),
                                  self.data_ptr<double>(), self.numel(), scalar);
        }
        return self;
    }
#endif

    #define OP_CASE(ctype, name) \
    case DType::name: { \
        auto op = [other, alpha](ctype a) -> ctype { \
            return static_cast<ctype>(a + alpha.to<ctype>() * other.to<ctype>()); \
        }; \
        apply_unary_op_recursive<ctype>(self.data_ptr<ctype>(), self.strides(), \
                                       self, self.strides(), \
                                       0, 0, 0, static_cast<std::vector<int64_t>>(self.shape()), op); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(OP_CASE)
        default: TP_THROW(TypeError, "add_scalar_: unsupported dtype");
    }
    #undef OP_CASE
    return self;
}

Tensor& sub_scalar_inplace_kernel(Tensor& self, const Scalar& other, const Scalar& alpha) {
#if defined(__x86_64__)
    if ((self.dtype() == DType::Float32 || self.dtype() == DType::Float64) &&
        !other.isComplex() && !alpha.isComplex() && self.is_contiguous() &&
        (cpu_has_avx512() || cpu_has_avx2f())) {
        if (self.dtype() == DType::Float32) {
            const float scalar = alpha.to<float>() * other.to<float>();
            scalar_f32_contiguous(BIN_ADD, self.data_ptr<float>(),
                                  self.data_ptr<float>(), self.numel(), -scalar);
        } else {
            const double scalar = alpha.to<double>() * other.to<double>();
            scalar_f64_contiguous(BIN_ADD, self.data_ptr<double>(),
                                  self.data_ptr<double>(), self.numel(), -scalar);
        }
        return self;
    }
#endif

    #define OP_CASE(ctype, name) \
    case DType::name: { \
        auto op = [other, alpha](ctype a) -> ctype { \
            return static_cast<ctype>(a - alpha.to<ctype>() * other.to<ctype>()); \
        }; \
        apply_unary_op_recursive<ctype>(self.data_ptr<ctype>(), self.strides(), \
                                        self, self.strides(), \
                                        0, 0, 0, static_cast<std::vector<int64_t>>(self.shape()), op); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(OP_CASE)
        default: TP_THROW(TypeError, "sub_scalar_: unsupported dtype");
    }
    #undef OP_CASE
    return self;
}

Tensor& mul_scalar_inplace_kernel(Tensor& self, const Scalar& other) {
#if defined(__x86_64__)
    if (self.dtype() == DType::Float32 && !other.isComplex() &&
        self.is_contiguous() && (cpu_has_avx512() || cpu_has_avx2f())) {
        scalar_f32_contiguous(BIN_MUL, self.data_ptr<float>(),
                              self.data_ptr<float>(), self.numel(),
                              other.to<float>());
        return self;
    }
    if (self.dtype() == DType::Float64 && !other.isComplex() &&
        self.is_contiguous() && (cpu_has_avx512() || cpu_has_avx2f())) {
        scalar_f64_contiguous(BIN_MUL, self.data_ptr<double>(),
                              self.data_ptr<double>(), self.numel(),
                              other.to<double>());
        return self;
    }
#endif

#if defined(__x86_64__)
    if (self.dtype() == DType::BFloat16 && !other.isComplex() &&
        self.is_contiguous()) {
        bf16_scalar_contiguous(BIN_MUL, self.data_ptr<BFloat16>(),
                               self.data_ptr<BFloat16>(), self.numel(),
                               other.to<float>());
        return self;
    }
#endif

    #define OP_CASE(ctype, name) \
    case DType::name: { \
        auto op = [other](ctype a) -> ctype { \
            return static_cast<ctype>(a * other.to<ctype>()); \
        }; \
        apply_unary_op_recursive<ctype>(self.data_ptr<ctype>(), self.strides(), \
                                       self, self.strides(), \
                                       0, 0, 0, static_cast<std::vector<int64_t>>(self.shape()), op); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(OP_CASE)
        default: TP_THROW(TypeError, "mul_scalar_: unsupported dtype");
    }
    #undef OP_CASE
    return self;
}

Tensor& div_scalar_inplace_kernel(Tensor& self, const Scalar& other) {
#if defined(__x86_64__)
    if (self.dtype() == DType::Float32 && !other.isComplex() &&
        self.is_contiguous() && (cpu_has_avx512() || cpu_has_avx2f())) {
        scalar_f32_contiguous(BIN_DIV, self.data_ptr<float>(),
                              self.data_ptr<float>(), self.numel(),
                              other.to<float>());
        return self;
    }
    if (self.dtype() == DType::Float64 && !other.isComplex() &&
        self.is_contiguous() && (cpu_has_avx512() || cpu_has_avx2f())) {
        scalar_f64_contiguous(BIN_DIV, self.data_ptr<double>(),
                              self.data_ptr<double>(), self.numel(),
                              other.to<double>());
        return self;
    }
#endif

#if defined(__x86_64__)
    if (self.dtype() == DType::BFloat16 && !other.isComplex() &&
        self.is_contiguous()) {
        bf16_scalar_contiguous(BIN_DIV, self.data_ptr<BFloat16>(),
                               self.data_ptr<BFloat16>(), self.numel(),
                               other.to<float>());
        return self;
    }
#endif

    #define OP_CASE(ctype, name) \
    case DType::name: { \
        auto op = [other](ctype a) -> ctype { \
            return div_scalar_element(a, other); \
        }; \
        apply_unary_op_recursive<ctype>(self.data_ptr<ctype>(), self.strides(), \
                                       self, self.strides(), \
                                       0, 0, 0, static_cast<std::vector<int64_t>>(self.shape()), op); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(OP_CASE)
        default: TP_THROW(TypeError, "div_scalar_: unsupported dtype");
    }
    #undef OP_CASE
    return self;
}

// Keep the common contiguous float32 case fused, and use one native ternary
// traversal for every broadcasted/promotion path.  The latter is important:
// composing mul() and add() here would reintroduce the Python/composite
// implementation that this operator is meant to replace.
Tensor addcmul_cpu(const Tensor& self, const Tensor& tensor1,
                   const Tensor& tensor2, const Scalar& value) {
    if (self.dtype() == DType::Float32 && tensor1.dtype() == DType::Float32 &&
        tensor2.dtype() == DType::Float32 && self.is_contiguous() &&
        tensor1.is_contiguous() && tensor2.is_contiguous() &&
        self.shape() == tensor1.shape() && self.shape() == tensor2.shape()) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::Float32, self.device());
        const float* self_ptr = self.data_ptr<float>();
        const float* tensor1_ptr = tensor1.data_ptr<float>();
        const float* tensor2_ptr = tensor2.data_ptr<float>();
        float* result_ptr = result.data_ptr<float>();
        const float alpha = value.to<float>();
        const int64_t n = self.numel();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                result_ptr[i] = self_ptr[i] + alpha * tensor1_ptr[i] * tensor2_ptr[i];
            }
        });
        return result;
    }

    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(tensor1.shape()),
        static_cast<std::vector<int64_t>>(tensor2.shape()));
    DType result_dtype = promoteTypes(
        promoteTypes(self.dtype(), tensor1.dtype()), tensor2.dtype());
    Tensor result = Tensor::empty(out_shape, result_dtype, self.device());
    Tensor a = self.dtype() == result_dtype ? self : self.to(result_dtype);
    Tensor b = tensor1.dtype() == result_dtype ? tensor1 : tensor1.to(result_dtype);
    Tensor c = tensor2.dtype() == result_dtype ? tensor2 : tensor2.to(result_dtype);
    auto a_strides = broadcast_strides(a, out_shape);
    auto b_strides = broadcast_strides(b, out_shape);
    auto c_strides = broadcast_strides(c, out_shape);

    #define ADDCMUL_CASE(ctype, name) \
        case DType::name: { \
            const ctype alpha = value.to<ctype>(); \
            auto op = [alpha](ctype x, ctype y, ctype z) -> ctype { \
                return x + alpha * y * z; \
            }; \
            apply_ternary_op_recursive<ctype, ctype>( \
                result.data_ptr<ctype>(), result.strides(), a, a_strides, \
                b, b_strides, c, c_strides, 0, 0, 0, 0, 0, out_shape, op); \
            break; \
        }
    switch (result_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(ADDCMUL_CASE)
        case DType::ComplexFloat: {
            const tensorplay::complex<float> alpha = value.to<tensorplay::complex<float>>();
            auto op = [alpha](tensorplay::complex<float> x, tensorplay::complex<float> y,
                              tensorplay::complex<float> z) { return x + alpha * y * z; };
            apply_ternary_op_recursive<tensorplay::complex<float>, tensorplay::complex<float>>( \
                result.data_ptr<tensorplay::complex<float>>(), result.strides(), a, a_strides, \
                b, b_strides, c, c_strides, 0, 0, 0, 0, 0, out_shape, op);
            break;
        }
        case DType::ComplexDouble: {
            const tensorplay::complex<double> alpha = value.to<tensorplay::complex<double>>();
            auto op = [alpha](tensorplay::complex<double> x, tensorplay::complex<double> y,
                              tensorplay::complex<double> z) { return x + alpha * y * z; };
            apply_ternary_op_recursive<tensorplay::complex<double>, tensorplay::complex<double>>( \
                result.data_ptr<tensorplay::complex<double>>(), result.strides(), a, a_strides, \
                b, b_strides, c, c_strides, 0, 0, 0, 0, 0, out_shape, op);
            break;
        }
        default: TP_THROW(TypeError, "addcmul: unsupported dtype");
    }
    #undef ADDCMUL_CASE
    return result;
}

Tensor& addcmul_inplace_cpu(Tensor& self, const Tensor& tensor1,
                            const Tensor& tensor2, const Scalar& value) {
    if (self.dtype() == DType::Float32 && tensor1.dtype() == DType::Float32 &&
        tensor2.dtype() == DType::Float32 && self.is_contiguous() &&
        tensor1.is_contiguous() && tensor2.is_contiguous() &&
        self.shape() == tensor1.shape() && self.shape() == tensor2.shape()) {
        float* self_ptr = self.data_ptr<float>();
        const float* tensor1_ptr = tensor1.data_ptr<float>();
        const float* tensor2_ptr = tensor2.data_ptr<float>();
        const float alpha = value.to<float>();
        const int64_t n = self.numel();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                self_ptr[i] += alpha * tensor1_ptr[i] * tensor2_ptr[i];
            }
        });
        return self;
    }

    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(tensor1.shape()),
        static_cast<std::vector<int64_t>>(tensor2.shape()));
    if (out_shape != static_cast<std::vector<int64_t>>(self.shape())) {
        TP_THROW(RuntimeError, "addcmul_: output shape does not match self");
    }
    Tensor b = tensor1.dtype() == self.dtype() ? tensor1 : tensor1.to(self.dtype());
    Tensor c = tensor2.dtype() == self.dtype() ? tensor2 : tensor2.to(self.dtype());
    auto self_strides = self.strides();
    auto b_strides = broadcast_strides(b, out_shape);
    auto c_strides = broadcast_strides(c, out_shape);
    #define ADDCMUL_INPLACE_CASE(ctype, name) \
        case DType::name: { \
            const ctype alpha = value.to<ctype>(); \
            auto op = [alpha](ctype x, ctype y, ctype z) -> ctype { \
                return x + alpha * y * z; \
            }; \
            apply_ternary_op_recursive<ctype, ctype>( \
                self.data_ptr<ctype>(), self_strides, self, self_strides, \
                b, b_strides, c, c_strides, 0, 0, 0, 0, 0, out_shape, op); \
            break; \
        }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(ADDCMUL_INPLACE_CASE)
        case DType::ComplexFloat: {
            const tensorplay::complex<float> alpha = value.to<tensorplay::complex<float>>();
            auto op = [alpha](tensorplay::complex<float> x, tensorplay::complex<float> y,
                              tensorplay::complex<float> z) { return x + alpha * y * z; };
            apply_ternary_op_recursive<tensorplay::complex<float>, tensorplay::complex<float>>( \
                self.data_ptr<tensorplay::complex<float>>(), self_strides, self, self_strides, \
                b, b_strides, c, c_strides, 0, 0, 0, 0, 0, out_shape, op);
            break;
        }
        case DType::ComplexDouble: {
            const tensorplay::complex<double> alpha = value.to<tensorplay::complex<double>>();
            auto op = [alpha](tensorplay::complex<double> x, tensorplay::complex<double> y,
                              tensorplay::complex<double> z) { return x + alpha * y * z; };
            apply_ternary_op_recursive<tensorplay::complex<double>, tensorplay::complex<double>>( \
                self.data_ptr<tensorplay::complex<double>>(), self_strides, self, self_strides, \
                b, b_strides, c, c_strides, 0, 0, 0, 0, 0, out_shape, op);
            break;
        }
        default: TP_THROW(TypeError, "addcmul_: unsupported dtype");
    }
    #undef ADDCMUL_INPLACE_CASE
    return self;
}

Tensor& addcmul_out_cpu(const Tensor& self, const Tensor& tensor1,
                        const Tensor& tensor2, const Scalar& value, Tensor& out) {
    write_out(out, addcmul_cpu(self, tensor1, tensor2, value));
    return out;
}

Tensor addcdiv_cpu(const Tensor& self, const Tensor& tensor1,
                   const Tensor& tensor2, const Scalar& value) {
    if (isIntegralType(tensor1.dtype(), true) && isIntegralType(tensor2.dtype(), true)) {
        TP_THROW(RuntimeError, "Integer division with addcdiv is not supported");
    }
    if (self.dtype() == DType::Float32 && tensor1.dtype() == DType::Float32 &&
        tensor2.dtype() == DType::Float32 && self.is_contiguous() &&
        tensor1.is_contiguous() && tensor2.is_contiguous() &&
        self.shape() == tensor1.shape() && self.shape() == tensor2.shape()) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::Float32, self.device());
        const float* self_ptr = self.data_ptr<float>();
        const float* tensor1_ptr = tensor1.data_ptr<float>();
        const float* tensor2_ptr = tensor2.data_ptr<float>();
        float* result_ptr = result.data_ptr<float>();
        const float alpha = value.to<float>();
        const int64_t n = self.numel();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                result_ptr[i] = self_ptr[i] + alpha * tensor1_ptr[i] / tensor2_ptr[i];
            }
        });
        return result;
    }

    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(tensor1.shape()),
        static_cast<std::vector<int64_t>>(tensor2.shape()));
    DType result_dtype = promoteTypes(
        promoteTypes(self.dtype(), tensor1.dtype()), tensor2.dtype());
    if (isIntegralType(result_dtype)) result_dtype = DType::Float32;
    Tensor result = Tensor::empty(out_shape, result_dtype, self.device());
    Tensor a = self.dtype() == result_dtype ? self : self.to(result_dtype);
    Tensor b = tensor1.dtype() == result_dtype ? tensor1 : tensor1.to(result_dtype);
    Tensor c = tensor2.dtype() == result_dtype ? tensor2 : tensor2.to(result_dtype);
    auto a_strides = broadcast_strides(a, out_shape);
    auto b_strides = broadcast_strides(b, out_shape);
    auto c_strides = broadcast_strides(c, out_shape);
    #define ADDCDIV_CASE(ctype, name) \
        case DType::name: { \
            const ctype alpha = value.to<ctype>(); \
            auto op = [alpha](ctype x, ctype y, ctype z) -> ctype { \
                return x + alpha * (y / z); \
            }; \
            apply_ternary_op_recursive<ctype, ctype>( \
                result.data_ptr<ctype>(), result.strides(), a, a_strides, \
                b, b_strides, c, c_strides, 0, 0, 0, 0, 0, out_shape, op); \
            break; \
        }
    switch (result_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(ADDCDIV_CASE)
        case DType::ComplexFloat: {
            const tensorplay::complex<float> alpha = value.to<tensorplay::complex<float>>();
            auto op = [alpha](tensorplay::complex<float> x, tensorplay::complex<float> y,
                              tensorplay::complex<float> z) { return x + alpha * (y / z); };
            apply_ternary_op_recursive<tensorplay::complex<float>, tensorplay::complex<float>>( \
                result.data_ptr<tensorplay::complex<float>>(), result.strides(), a, a_strides, \
                b, b_strides, c, c_strides, 0, 0, 0, 0, 0, out_shape, op);
            break;
        }
        case DType::ComplexDouble: {
            const tensorplay::complex<double> alpha = value.to<tensorplay::complex<double>>();
            auto op = [alpha](tensorplay::complex<double> x, tensorplay::complex<double> y,
                              tensorplay::complex<double> z) { return x + alpha * (y / z); };
            apply_ternary_op_recursive<tensorplay::complex<double>, tensorplay::complex<double>>( \
                result.data_ptr<tensorplay::complex<double>>(), result.strides(), a, a_strides, \
                b, b_strides, c, c_strides, 0, 0, 0, 0, 0, out_shape, op);
            break;
        }
        default: TP_THROW(TypeError, "addcdiv: unsupported dtype");
    }
    #undef ADDCDIV_CASE
    return result;
}

Tensor& addcdiv_inplace_cpu(Tensor& self, const Tensor& tensor1,
                            const Tensor& tensor2, const Scalar& value) {
    if (isIntegralType(tensor1.dtype(), true) && isIntegralType(tensor2.dtype(), true)) {
        TP_THROW(RuntimeError, "Integer division with addcdiv is not supported");
    }
    if (self.dtype() == DType::Float32 && tensor1.dtype() == DType::Float32 &&
        tensor2.dtype() == DType::Float32 && self.is_contiguous() &&
        tensor1.is_contiguous() && tensor2.is_contiguous() &&
        self.shape() == tensor1.shape() && self.shape() == tensor2.shape()) {
        float* self_ptr = self.data_ptr<float>();
        const float* tensor1_ptr = tensor1.data_ptr<float>();
        const float* tensor2_ptr = tensor2.data_ptr<float>();
        const float alpha = value.to<float>();
        const int64_t n = self.numel();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                self_ptr[i] += alpha * tensor1_ptr[i] / tensor2_ptr[i];
            }
        });
        return self;
    }

    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(tensor1.shape()),
        static_cast<std::vector<int64_t>>(tensor2.shape()));
    if (out_shape != static_cast<std::vector<int64_t>>(self.shape())) {
        TP_THROW(RuntimeError, "addcdiv_: output shape does not match self");
    }
    Tensor b = tensor1.dtype() == self.dtype() ? tensor1 : tensor1.to(self.dtype());
    Tensor c = tensor2.dtype() == self.dtype() ? tensor2 : tensor2.to(self.dtype());
    auto self_strides = self.strides();
    auto b_strides = broadcast_strides(b, out_shape);
    auto c_strides = broadcast_strides(c, out_shape);
    #define ADDCDIV_INPLACE_CASE(ctype, name) \
        case DType::name: { \
            const ctype alpha = value.to<ctype>(); \
            auto op = [alpha](ctype x, ctype y, ctype z) -> ctype { \
                return x + alpha * (y / z); \
            }; \
            apply_ternary_op_recursive<ctype, ctype>( \
                self.data_ptr<ctype>(), self_strides, self, self_strides, \
                b, b_strides, c, c_strides, 0, 0, 0, 0, 0, out_shape, op); \
            break; \
        }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(ADDCDIV_INPLACE_CASE)
        case DType::ComplexFloat: {
            const tensorplay::complex<float> alpha = value.to<tensorplay::complex<float>>();
            auto op = [alpha](tensorplay::complex<float> x, tensorplay::complex<float> y,
                              tensorplay::complex<float> z) { return x + alpha * (y / z); };
            apply_ternary_op_recursive<tensorplay::complex<float>, tensorplay::complex<float>>( \
                self.data_ptr<tensorplay::complex<float>>(), self_strides, self, self_strides, \
                b, b_strides, c, c_strides, 0, 0, 0, 0, 0, out_shape, op);
            break;
        }
        case DType::ComplexDouble: {
            const tensorplay::complex<double> alpha = value.to<tensorplay::complex<double>>();
            auto op = [alpha](tensorplay::complex<double> x, tensorplay::complex<double> y,
                              tensorplay::complex<double> z) { return x + alpha * (y / z); };
            apply_ternary_op_recursive<tensorplay::complex<double>, tensorplay::complex<double>>( \
                self.data_ptr<tensorplay::complex<double>>(), self_strides, self, self_strides, \
                b, b_strides, c, c_strides, 0, 0, 0, 0, 0, out_shape, op);
            break;
        }
        default: TP_THROW(TypeError, "addcdiv_: unsupported dtype");
    }
    #undef ADDCDIV_INPLACE_CASE
    return self;
}

// Registration
TENSORPLAY_LIBRARY_IMPL(CPU, ArithmeticKernels) {
    m.impl("add.Tensor", add_kernel);
    m.impl("add.out", add_out_kernel);
    m.impl("add_relu", add_relu_cpu);
    m.impl("sub.Tensor", sub_kernel);
    m.impl("mul.Tensor", mul_kernel);
    m.impl("div.Tensor", div_kernel);
    m.impl("addcmul", addcmul_cpu);
    m.impl("addcmul_", addcmul_inplace_cpu);
    m.impl("addcmul.out", addcmul_out_cpu);
    m.impl("addcdiv", addcdiv_cpu);
    m.impl("addcdiv_", addcdiv_inplace_cpu);
    m.impl("fused_mul_add", fused_mul_add_kernel);
    m.impl("fused_mul_add.Scalar", fused_mul_add_scalar_kernel);

    m.impl("add_.Tensor", add_inplace_kernel);
    m.impl("sub_.Tensor", sub_inplace_kernel);
    m.impl("mul_.Tensor", mul_inplace_kernel);
    m.impl("div_.Tensor", div_inplace_kernel);

    m.impl("add.Scalar", add_scalar_kernel);
    m.impl("sub.Scalar", sub_scalar_kernel);
    m.impl("mul.Scalar", mul_scalar_kernel);
    m.impl("div.Scalar", div_scalar_kernel);

    m.impl("add_.Scalar", add_scalar_inplace_kernel);
    m.impl("sub_.Scalar", sub_scalar_inplace_kernel);
    m.impl("mul_.Scalar", mul_scalar_inplace_kernel);
    m.impl("div_.Scalar", div_scalar_inplace_kernel);
}

// A sparse operand outranks the dense one when the dispatch key is derived
// from the arguments, so in-place accumulation of a sparse tensor into a
// dense destination registers under the sparse backend; the kernel itself
// keeps branching on whether the trailing operand is sparse.
TENSORPLAY_LIBRARY_IMPL(SparseCPU, ArithSparseInplace) {
    m.impl("add_.Tensor", add_inplace_kernel);
}

} // namespace cpu
} // namespace tensorplay
