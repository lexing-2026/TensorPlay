// CPU scaled-dot-product attention kernels.
// lives under transformers/, not an "llm" grab-bag.
//
// Forward f32/f16/bf16 path: BLAS sgemm for QK^T and PV, fused causal-prefix
// row softmax with runtime-dispatched libmvec vector exp (AVX-512 16-wide ->
// AVX2 8-wide -> scalar).  f64 keeps a serial double reference oracle.

#include <immintrin.h>

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "GradMode.h"
#include "LinearAlgebraNames.h"
#include "Parallel.h"

// Dispatcher-level primitives for the differentiable composite path
// (defined in TPXOpsGenerated.cpp; declared locally because tpx headers are
// not visible below the p10 layer -- same pattern as Einsum.cpp).
namespace tensorplay {
namespace tpx {
namespace ops {
TENSORPLAY_API Tensor mm(const Tensor& self, const Tensor& mat2);
TENSORPLAY_API Tensor narrow(const Tensor& self, int64_t dim, int64_t start,
                             int64_t length);
TENSORPLAY_API Tensor reshape(const Tensor& self,
                              const std::vector<int64_t>& shape);
TENSORPLAY_API Tensor cat(const std::vector<Tensor>& tensors, int64_t dim);
} // namespace ops
} // namespace tpx
} // namespace tensorplay

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <vector>

#if defined(USE_MKL)
#include <mkl.h>
#elif defined(USE_BLAS)
#include <cblas.h>
#endif

#if defined(__x86_64__) && defined(__GLIBC__)
extern "C" {
__m256 _ZGVdN8v_expf(__m256);
__m512 _ZGVeN16v_expf(__m512);
}
#define TP_SDPA_LIBMVEC 1
#define TP_SDPA_LIBMVEC512 1
#endif

namespace tensorplay {
namespace cpu {

namespace {

// Vectorized expf: AVX-512 16 lanes -> AVX2 8 lanes -> scalar libm tail.
// libmvec symbols carry C linkage; dispatch is per-call CPUID so one binary
// serves Zen4 (AVX-512) and older guests alike.  The target attribute is
// required because the TU compiles with base x86-64 flags (repo convention:
// per-function targets, cf. VecUnary.h).
#if defined(TP_SDPA_LIBMVEC512)
__attribute__((target("avx2,avx512f")))
#elif defined(TP_SDPA_LIBMVEC)
__attribute__((target("avx2")))
#endif
void vexp_f32(const float* x, float* y, int64_t n) {
  int64_t i = 0;
#if defined(TP_SDPA_LIBMVEC)
  const bool avx512 =
#if defined(TP_SDPA_LIBMVEC512)
      __builtin_cpu_supports("avx512f");
#else
      false;
#endif
  if (avx512) {
    for (; i + 16 <= n; i += 16)
      _mm512_storeu_ps(y + i, _ZGVeN16v_expf(_mm512_loadu_ps(x + i)));
  } else if (__builtin_cpu_supports("avx2")) {
    for (; i + 8 <= n; i += 8)
      _mm256_storeu_ps(y + i, _ZGVdN8v_expf(_mm256_loadu_ps(x + i)));
  }
#endif
  for (; i < n; ++i) y[i] = std::exp(x[i]);
}

} // namespace

Tensor sdpa_kernel_cpu(const Tensor& query, const Tensor& key,
                       const Tensor& value, bool is_causal, int64_t impl) {
  Tensor q = query.contiguous();
  Tensor k = key.contiguous();
  Tensor v = value.contiguous();
  if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4) {
    TP_THROW(RuntimeError, "sdpa: query/key/value must be 4D [B, H, T, D]");
  }
  int64_t B = q.size(0), H = q.size(1), T = q.size(2), D = q.size(3);
  // Cross-attention: query length may differ from key/value length
  int64_t Tq = T;
  int64_t Skv = k.size(2);
  if (k.size(0) != B || k.size(1) != H || v.size(0) != B || v.size(1) != H ||
      v.size(2) != Skv || k.size(3) != D || v.size(3) != D) {
    TP_THROW(RuntimeError, "sdpa: key/value shapes must match [B, H, S, D]");
  }
  if (q.dtype() != DType::Float32 && q.dtype() != DType::Float64 &&
      q.dtype() != DType::Float16 && q.dtype() != DType::BFloat16) {
    TP_THROW(NotImplementedError,
             "sdpa cpu: expected float32/float64/float16/bfloat16");
  }
  const DType original_dtype = q.dtype();

  if (original_dtype == DType::Float64) {
    // Serial double reference for validation.
    Tensor out = Tensor::empty({B, H, Tq, D}, DType::Float64, q.device());
    const double* qd = q.data_ptr<double>();
    const double* kd = k.data_ptr<double>();
    const double* vd = v.data_ptr<double>();
    double* od = out.data_ptr<double>();
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t h = 0; h < H; ++h) {
        const double* qh = qd + ((b * H + h) * Tq) * D;
        const double* kh = kd + ((b * H + h) * Skv) * D;
        const double* vh = vd + ((b * H + h) * Skv) * D;
        double* oh = od + ((b * H + h) * Tq) * D;
        std::vector<double> scores(Skv);
        for (int64_t t = 0; t < Tq; ++t) {
          const int64_t visible = is_causal ? std::min(t + 1, Skv) : Skv;
          const double* qrow = qh + t * D;
          double mx = -INFINITY;
          for (int64_t kk = 0; kk < Skv; ++kk) {
            double s = -INFINITY;
            if (kk < visible) {
              const double* krow = kh + kk * D;
              s = 0.0;
              for (int64_t d = 0; d < D; ++d) s += qrow[d] * krow[d];
              s *= scale;
            }
            scores[kk] = s;
            mx = std::max(mx, s);
          }
          double total = 0.0;
          for (int64_t kk = 0; kk < Skv; ++kk) {
            double e = std::exp(scores[kk] - mx);
            scores[kk] = e;
            total += e;
          }
          double* orow = oh + t * D;
          for (int64_t d = 0; d < D; ++d) {
            double acc = 0.0;
            for (int64_t kk = 0; kk < Skv; ++kk)
              acc += scores[kk] * vh[kk * D + d];
            orow[d] = acc / total;
          }
        }
      }
    }
    return out;
  }

  // ---- Fast f32 path ----
#if !defined(USE_MKL) && !defined(USE_BLAS)
  TP_THROW(NotImplementedError, "sdpa cpu fast path requires BLAS");
#else
  if (Tq <= 0 || Skv <= 0 || D <= 0 || B * H * Tq * D == 0) {
    return Tensor::empty({B, H, Tq, D}, original_dtype, query.device());
  }
  if (Tq * Skv > static_cast<int64_t>(INT32_MAX) ||
      Skv * D > static_cast<int64_t>(INT32_MAX)) {
    TP_THROW(RuntimeError, "sdpa cpu: shape too large for BLAS ints");
  }
  Tensor qf = q.to(DType::Float32);
  Tensor kf = k.to(DType::Float32);
  Tensor vf = v.to(DType::Float32);
  Tensor out = Tensor::empty({B, H, Tq, D}, DType::Float32, qf.device());
  const float* qd = qf.data_ptr<float>();
  const float* kd = kf.data_ptr<float>();
  const float* vd = vf.data_ptr<float>();
  float* od = out.data_ptr<float>();
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  // Scores scratch reused across calls: a fresh 134MB allocation per prefill
  // pays ~32k page faults before the first GEMM lane runs (measured as the
  // concurrent GIL-released callers don't share.
  static thread_local std::vector<float> scores_scratch;
  if (scores_scratch.size() < static_cast<size_t>(B * H * Tq * Skv))
    scores_scratch.resize(static_cast<size_t>(B * H * Tq * Skv));
  float* sc = scores_scratch.data();
  std::vector<float> shifted(Skv);

  const int64_t bh_total = B * H;
  const int64_t threads = parallel::get_num_threads();

  if (Tq == 1) {
    // Decode: one query row per head.  Per-head work is dot-products +
    // softmax + a D-wide axpy accumulation -- pure SIMD loops beat 64 tiny
    // before this path).  Heads parallelize across cores.
    parallel::parallel_for(0, bh_total, 1, [&](int64_t b0, int64_t b1) {
      std::vector<float> probs(Skv);
      std::vector<float> acc(D);
      for (int64_t m = b0; m < b1; ++m) {
        const float* qh = qd + m * D;
        const float* kh = kd + m * Skv * D;
        const float* vh = vd + m * Skv * D;
        float* oh = od + m * D;
        for (int64_t j = 0; j < Skv; ++j) {
          const float* krow = kh + j * D;
          float s = 0.0f;
          for (int64_t d = 0; d < D; ++d) s += qh[d] * krow[d];
          probs[j] = s * scale;
        }
        float mx = probs[0];
        for (int64_t j = 1; j < Skv; ++j) mx = std::max(mx, probs[j]);
        vexp_f32(probs.data(), probs.data(), Skv);
        float total = 0.0f;
        for (int64_t j = 0; j < Skv; ++j) total += probs[j];
        const float inv = 1.0f / total;
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (int64_t j = 0; j < Skv; ++j) {
          const float p = probs[j] * inv;
          const float* vrow = vh + j * D;
          for (int64_t d = 0; d < D; ++d) acc[d] += p * vrow[d];
        }
        for (int64_t d = 0; d < D; ++d) oh[d] = acc[d];
      }
    });
    return out.to(original_dtype);
  }

  // Prefill: heads in parallel; each worker pins MKL to single-thread so the
  // per-head sgemms don't nest OpenMP regions (same runtime -> serialized
  // inner region, oversubscription-free).  Sequential when there's less
  // parallelism than the gemms can use themselves.
  auto run_heads = [&](int64_t mb, int64_t me) {
#if defined(USE_MKL)
    mkl_set_num_threads_local(mb >= 0 ? 1 : 0);
#endif
    std::vector<float> shifted_loc(Skv);
    for (int64_t m = (mb >= 0 ? mb : 0); m < (me >= 0 ? me : bh_total); ++m) {
      const float* qh = qd + m * Tq * D;
      const float* kh = kd + m * Skv * D;
      const float* vh = vd + m * Skv * D;
      float* sh = sc + m * Tq * Skv;
      float* oh = od + m * Tq * D;

      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                  static_cast<int>(Tq), static_cast<int>(Skv),
                  static_cast<int>(D), scale, qh, static_cast<int>(D), kh,
                  static_cast<int>(D), 0.0f, sh, static_cast<int>(Skv));

      for (int64_t t = 0; t < Tq; ++t) {
        float* row = sh + t * Skv;
        const int64_t nvis = is_causal ? std::min(t + 1, Skv) : Skv;
        float mx = row[0];
        for (int64_t j = 1; j < nvis; ++j) mx = std::max(mx, row[j]);
        for (int64_t j = 0; j < nvis; ++j) shifted_loc[j] = row[j] - mx;
        vexp_f32(shifted_loc.data(), row, nvis);
        float total = 0.0f;
        for (int64_t j = 0; j < nvis; ++j) total += row[j];
        const float inv = 1.0f / total;
        for (int64_t j = 0; j < nvis; ++j) row[j] *= inv;
        if (nvis < Skv)
          std::memset(row + nvis, 0,
                      static_cast<size_t>(Skv - nvis) * sizeof(float));
      }

      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  static_cast<int>(Tq), static_cast<int>(D),
                  static_cast<int>(Skv), 1.0f, sh, static_cast<int>(Skv), vh,
                  static_cast<int>(D), 0.0f, oh, static_cast<int>(D));
    }
#if defined(USE_MKL)
    if (mb >= 0) mkl_set_num_threads_local(0);
#endif
  };

  if (bh_total > threads && threads > 1) {
    const int64_t grain = std::max<int64_t>(1, bh_total / (threads * 4));
    parallel::parallel_for(0, bh_total, grain,
                           [&](int64_t b0, int64_t b1) { run_heads(b0, b1); });
  } else {
    run_heads(-1, -1);
  }
  return out.to(original_dtype);
#endif
}

std::tuple<Tensor, Tensor, Tensor> sdpa_backward_kernel_cpu(
    const Tensor& grad_output, const Tensor& query, const Tensor& key,
    const Tensor& value, bool is_causal, int64_t impl) {
  (void)impl;
  Tensor q = query.contiguous();
  Tensor k = key.contiguous();
  Tensor v = value.contiguous();
  Tensor go = grad_output.contiguous();
  if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4 || go.dim() != 4) {
    TP_THROW(RuntimeError, "sdpa backward: q/k/v/grad_output must be 4D");
  }
  const int64_t B = q.size(0), H = q.size(1), T = q.size(2), D = q.size(3);
  if (k.size(0) != B || k.size(1) != H || k.size(2) != T || k.size(3) != D ||
      v.size(0) != B || v.size(1) != H || v.size(2) != T || v.size(3) != D ||
      go.size(0) != B || go.size(1) != H || go.size(2) != T ||
      go.size(3) != D) {
    TP_THROW(RuntimeError, "sdpa backward: q/k/v/grad_output shapes must match");
  }
  if (q.dtype() != DType::Float32 && q.dtype() != DType::Float64 &&
      q.dtype() != DType::Float16 && q.dtype() != DType::BFloat16) {
    TP_THROW(NotImplementedError,
             "sdpa backward CPU: expected float32/float64/float16/bfloat16");
  }
  if (k.dtype() != q.dtype() || v.dtype() != q.dtype() ||
      go.dtype() != q.dtype()) {
    TP_THROW(RuntimeError,
             "sdpa backward: q/k/v/grad_output dtypes must match");
  }

  const DType original_dtype = q.dtype();
  // Serial double reference; backward shapes in practice are small. BLAS
  // rewrite lands with the autograd-perf pass.
  q = q.to(DType::Float64);
  k = k.to(DType::Float64);
  v = v.to(DType::Float64);
  go = go.to(DType::Float64);
  const double* qd = q.data_ptr<double>();
  const double* kd = k.data_ptr<double>();
  const double* vd = v.data_ptr<double>();
  const double* god = go.data_ptr<double>();
  const double scale = 1.0 / std::sqrt(static_cast<double>(D));
  const int64_t rows = B * H;

  Tensor probs = Tensor::empty({B, H, T, T}, DType::Float64, q.device());
  Tensor dprob = Tensor::empty({B, H, T, T}, DType::Float64, q.device());
  Tensor dscore = Tensor::empty({B, H, T, T}, DType::Float64, q.device());
  double* pd = probs.data_ptr<double>();
  double* dpd = dprob.data_ptr<double>();
  double* dsd = dscore.data_ptr<double>();

  for (int64_t bh = 0; bh < rows; ++bh) {
    const double* qh = qd + bh * T * D;
    const double* kh = kd + bh * T * D;
    for (int64_t t = 0; t < T; ++t) {
      double max_score = -INFINITY;
      for (int64_t kk = 0; kk < T; ++kk) {
        double score = -INFINITY;
        if (!is_causal || kk <= t) {
          score = 0.0;
          for (int64_t d = 0; d < D; ++d)
            score += qh[t * D + d] * kh[kk * D + d];
          score *= scale;
        }
        pd[(bh * T + t) * T + kk] = score;
        max_score = std::max(max_score, score);
      }
      double total = 0.0;
      for (int64_t kk = 0; kk < T; ++kk) {
        double p = (pd[(bh * T + t) * T + kk] == -INFINITY)
                       ? 0.0
                       : std::exp(pd[(bh * T + t) * T + kk] - max_score);
        pd[(bh * T + t) * T + kk] = p;
        total += p;
      }
      for (int64_t kk = 0; kk < T; ++kk) pd[(bh * T + t) * T + kk] /= total;
    }
  }

  for (int64_t bh = 0; bh < rows; ++bh) {
    const double* gh = god + bh * T * D;
    const double* vh = vd + bh * T * D;
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t kk = 0; kk < T; ++kk) {
        double dot = 0.0;
        for (int64_t d = 0; d < D; ++d) dot += gh[t * D + d] * vh[kk * D + d];
        dpd[(bh * T + t) * T + kk] = dot;
      }
      double row_dot = 0.0;
      for (int64_t kk = 0; kk < T; ++kk)
        row_dot += dpd[(bh * T + t) * T + kk] * pd[(bh * T + t) * T + kk];
      for (int64_t kk = 0; kk < T; ++kk)
        dsd[(bh * T + t) * T + kk] =
            pd[(bh * T + t) * T + kk] * (dpd[(bh * T + t) * T + kk] - row_dot) *
            scale;
    }
  }

  Tensor d_q = Tensor::zeros({B, H, T, D}, DType::Float64, q.device());
  Tensor d_k = Tensor::zeros({B, H, T, D}, DType::Float64, q.device());
  Tensor d_v = Tensor::zeros({B, H, T, D}, DType::Float64, q.device());
  double* dqd = d_q.data_ptr<double>();
  double* dkd = d_k.data_ptr<double>();
  double* dvd = d_v.data_ptr<double>();
  for (int64_t bh = 0; bh < rows; ++bh) {
    const double* qh = qd + bh * T * D;
    const double* kh = kd + bh * T * D;
    const double* gh = god + bh * T * D;
    const double* vh = vd + bh * T * D;
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t d = 0; d < D; ++d) {
        double q_acc = 0.0;
        for (int64_t kk = 0; kk < T; ++kk)
          q_acc += dsd[(bh * T + t) * T + kk] * kh[kk * D + d];
        dqd[(bh * T + t) * D + d] = q_acc;
      }
    }
    for (int64_t kk = 0; kk < T; ++kk) {
      for (int64_t d = 0; d < D; ++d) {
        double k_acc = 0.0, v_acc = 0.0;
        for (int64_t t = 0; t < T; ++t) {
          k_acc += dsd[(bh * T + t) * T + kk] * qh[t * D + d];
          v_acc += pd[(bh * T + t) * T + kk] * gh[t * D + d];
        }
        dkd[(bh * T + kk) * D + d] = k_acc;
        dvd[(bh * T + kk) * D + d] = v_acc;
      }
    }
  }
  return {d_q.to(original_dtype), d_k.to(original_dtype),
          d_v.to(original_dtype)};
}

// ---------------------------------------------------------------------------
// semantics): self [M_total, K] @ mat2 [G, K, N] -> [M_total, N]; offs [G]
// holds cumulative END offsets, group g spans [prev_end, offs[g]).
// No-grad path is one dispatcher op wrapping per-group cblas_sgemm calls --
// the composite narrow/mm/cat loop it replaces pays ~4us dispatch overhead
// per expert per token-batch.  GradMode falls back to the differentiable
// composite so CIA records inner nodes automatically.
// ---------------------------------------------------------------------------
Tensor grouped_mm_cpu(const Tensor& self, const Tensor& mat2,
                      const Tensor& offs) {
  if (self.dim() != 2 || mat2.dim() != 3) {
    TP_THROW(RuntimeError,
             "grouped_mm(): expected 2D self and 3D mat2, got ", self.dim(),
             "D and ", mat2.dim(), "D");
  }
  const int64_t M = self.size(0), K = self.size(1);
  const int64_t G = mat2.size(0);
  if (mat2.size(1) != K) {
    TP_THROW(RuntimeError, "grouped_mm(): self.size(1) must match mat2.size(1): ",
             K, " vs ", mat2.size(1));
  }
  if (offs.dim() != 1 || offs.numel() != G) {
    TP_THROW(RuntimeError, "grouped_mm(): offs must be 1D of length mat2.size(0)=",
             G, ", got ", offs.dim(), "D/", offs.numel(), " elements");
  }
  if (self.dtype() != mat2.dtype()) {
    TP_THROW(RuntimeError, "grouped_mm(): expected self and mat2 to have the same dtype, but got: ",
             c10_style_dtype_name(self.dtype()), " != ",
             c10_style_dtype_name(mat2.dtype()));
  }
  if (self.dtype() != DType::Float32 && self.dtype() != DType::Float64) {
    TP_THROW(NotImplementedError,
             "grouped_mm cpu: expected float32/float64");
  }

  // Validate offsets: non-decreasing int32/int64 within [0, M].
  auto read_off = [&](int64_t i) -> int64_t {
    if (offs.dtype() == DType::Int32) return offs.data_ptr<int32_t>()[i];
    if (offs.dtype() == DType::Int64) return offs.data_ptr<int64_t>()[i];
    TP_THROW(TypeError, "grouped_mm(): offs must be int32 or int64");
  };
  int64_t prev = 0;
  for (int64_t g = 0; g < G; ++g) {
    const int64_t end = read_off(g);
    if (end < prev || end > M) {
      TP_THROW(RuntimeError, "grouped_mm(): offs must be non-decreasing in [0, M_total=",
               M, "], got offs[", g, "]=", end);
    }
    prev = end;
  }

  const bool needs_grad =
      GradMode::is_enabled() && (self.requires_grad() || mat2.requires_grad());
  if (needs_grad) {
    // Differentiable composite: mm over row slices, cat back together.
    std::vector<Tensor> parts;
    parts.reserve(G);
    int64_t start = 0;
    for (int64_t g = 0; g < G; ++g) {
      const int64_t end = read_off(g);
      const int64_t len = end - start;
      if (len > 0) {
        Tensor wg = tpx::ops::narrow(mat2, 0, g, 1);
        wg = tpx::ops::reshape(wg, {K, mat2.size(2)});
        parts.push_back(tpx::ops::mm(tpx::ops::narrow(self, 0, start, len), wg));
      }
      start = end;
    }
    if (parts.empty()) {
      return Tensor::zeros({M, mat2.size(2)}, self.dtype(), self.device());
    }
    return tpx::ops::cat(parts, 0);
  }

#if !defined(USE_MKL) && !defined(USE_BLAS)
  TP_THROW(NotImplementedError, "grouped_mm cpu requires BLAS");
#else
  Tensor out = Tensor::empty({M, mat2.size(2)}, self.dtype(), self.device());
  if (M == 0 || mat2.size(2) == 0) return out;
  const int N = static_cast<int>(mat2.size(2));
  if (static_cast<int64_t>(N) * K > INT32_MAX ||
      static_cast<int64_t>(N) * M > INT32_MAX) {
    TP_THROW(RuntimeError, "grouped_mm cpu: shape too large for BLAS ints");
  }

  if (self.dtype() == DType::Float32) {
    float* od = out.data_ptr<float>();
    std::memset(od, 0, static_cast<size_t>(M) * N * sizeof(float));
    const float* ad = self.data_ptr<float>();
    const float* bd = mat2.data_ptr<float>();
    int64_t start = 0;
    for (int64_t g = 0; g < G; ++g) {
      const int64_t end = read_off(g);
      const int len = static_cast<int>(end - start);
      if (len > 0) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, len, N,
                    static_cast<int>(K), 1.0f, ad + start * K,
                    static_cast<int>(K), bd + g * K * N, N, 1.0f,
                    od + start * N, N);
      }
      start = end;
    }
  } else {
    double* od = out.data_ptr<double>();
    std::memset(od, 0, static_cast<size_t>(M) * N * sizeof(double));
    const double* ad = self.data_ptr<double>();
    const double* bd = mat2.data_ptr<double>();
    int64_t start = 0;
    for (int64_t g = 0; g < G; ++g) {
      const int64_t end = read_off(g);
      const int len = static_cast<int>(end - start);
      if (len > 0) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, len, N,
                    static_cast<int>(K), 1.0, ad + start * K,
                    static_cast<int>(K), bd + g * K * N, N, 1.0,
                    od + start * N, N);
      }
      start = end;
    }
  }
  return out;
#endif
}

// RoPE primitives follow the transformer attention path.  The layout matches
// [x0, x1, x2, x3, ...], while cos/sin are [positions, head_dim / 2].
namespace {

inline bool rope_float_dtype(DType dtype) {
  return dtype == DType::Float32 || dtype == DType::Float64 ||
         dtype == DType::Float16 || dtype == DType::BFloat16;
}

inline int64_t rope_tokens(const Tensor& input, const char* op) {
  if (input.dim() < 2) {
    TP_THROW(RuntimeError, op, ": input must have at least 2 dimensions");
  }
  if ((input.size(-1) & 1) != 0) {
    TP_THROW(RuntimeError, op, ": the last dimension must be even");
  }
  if (!rope_float_dtype(input.dtype())) {
    TP_THROW(NotImplementedError, op,
             ": only float32/float64/float16/bfloat16 are supported");
  }
  return input.size(-2);
}

struct RopeTable {
  int64_t rows;
  int64_t half_dim;
};

RopeTable check_rope_table(const Tensor& cos, const Tensor& sin,
                           int64_t half_dim, int64_t tokens,
                           int64_t position_offset, const Device& device,
                           const char* op) {
  if (position_offset < 0) {
    TP_THROW(RuntimeError, op, ": position_offset must be non-negative");
  }
  if (cos.device() != device || sin.device() != device ||
      cos.device() != sin.device()) {
    TP_THROW(DeviceMismatchError, op,
             ": input and cos/sin must be on the same device");
  }
  if (cos.dtype() != sin.dtype() || !rope_float_dtype(cos.dtype())) {
    TP_THROW(RuntimeError, op,
             ": cos and sin must have the same floating dtype");
  }

  int64_t rows = 0;
  if (cos.dim() == 1 && sin.dim() == 1) {
    if (cos.size(0) != half_dim || sin.size(0) != half_dim) {
      TP_THROW(RuntimeError, op,
               ": 1D cos/sin tables must have head_dim/2 entries");
    }
    rows = 1;
  } else if (cos.dim() == 2 && sin.dim() == 2) {
    if (cos.size(1) != half_dim || sin.size(1) != half_dim ||
        cos.size(0) != sin.size(0)) {
      TP_THROW(RuntimeError, op,
               ": cos/sin tables must be [positions, head_dim/2]");
    }
    rows = cos.size(0);
  } else {
    TP_THROW(RuntimeError, op,
             ": cos and sin must both be 1D or both be 2D");
  }

  if (rows != 1 &&
      (position_offset > rows || tokens > rows - position_offset)) {
    TP_THROW(RuntimeError, op,
             ": cos/sin table is shorter than the requested positions");
  }
  if (rows == 1 && position_offset != 0) {
    TP_THROW(RuntimeError, op,
             ": position_offset must be zero for a one-row table");
  }
  return {rows, half_dim};
}

template <typename T, typename C>
void rope_loop(const T* input, T* output, const C* cos, const C* sin,
               int64_t pairs, int64_t tokens, int64_t half_dim,
               int64_t table_rows, int64_t position_offset) {
  using Acc = std::conditional_t<std::is_same_v<T, double>, double, float>;
  parallel::parallel_for(0, pairs, parallel::GRAIN_SIZE,
                         [&](int64_t begin, int64_t end) {
    for (int64_t pair = begin; pair < end; ++pair) {
      const int64_t row = pair / half_dim;
      const int64_t pair_in_row = pair - row * half_dim;
      const int64_t token = row % tokens;
      const int64_t table_row = table_rows == 1
                                    ? 0
                                    : position_offset + token;
      const int64_t input_offset = row * (2 * half_dim) + 2 * pair_in_row;
      const int64_t table_offset = table_row * half_dim + pair_in_row;
      const Acc x0 = static_cast<Acc>(input[input_offset]);
      const Acc x1 = static_cast<Acc>(input[input_offset + 1]);
      const Acc c = static_cast<Acc>(cos[table_offset]);
      const Acc s = static_cast<Acc>(sin[table_offset]);
      output[input_offset] = static_cast<T>(x0 * c - x1 * s);
      output[input_offset + 1] = static_cast<T>(x0 * s + x1 * c);
    }
  });
}

template <typename T, typename C>
Tensor rope_single_typed(const Tensor& input, const Tensor& cos,
                         const Tensor& sin, const RopeTable& table,
                         int64_t tokens, int64_t position_offset) {
  Tensor input_c = input.is_contiguous() ? input : input.contiguous();
  Tensor cos_c = cos.is_contiguous() ? cos : cos.contiguous();
  Tensor sin_c = sin.is_contiguous() ? sin : sin.contiguous();
  Tensor output = Tensor::empty(
      static_cast<std::vector<int64_t>>(input_c.shape()), input_c.dtype(),
      input_c.device());
  rope_loop<T, C>(input_c.data_ptr<T>(), output.data_ptr<T>(),
                  cos_c.data_ptr<C>(), sin_c.data_ptr<C>(),
                  input_c.numel() / 2, tokens, table.half_dim, table.rows,
                  position_offset);
  return output;
}

template <typename T, typename C>
std::tuple<Tensor, Tensor> rope_pair_typed(
    const Tensor& query, const Tensor& key, const Tensor& cos,
    const Tensor& sin, const RopeTable& table, int64_t query_tokens,
    int64_t key_tokens, int64_t position_offset) {
  Tensor query_c = query.is_contiguous() ? query : query.contiguous();
  Tensor key_c = key.is_contiguous() ? key : key.contiguous();
  Tensor cos_c = cos.is_contiguous() ? cos : cos.contiguous();
  Tensor sin_c = sin.is_contiguous() ? sin : sin.contiguous();
  Tensor query_out = Tensor::empty(
      static_cast<std::vector<int64_t>>(query_c.shape()), query_c.dtype(),
      query_c.device());
  Tensor key_out = Tensor::empty(
      static_cast<std::vector<int64_t>>(key_c.shape()), key_c.dtype(),
      key_c.device());

  const int64_t query_pairs = query_c.numel() / 2;
  const int64_t key_pairs = key_c.numel() / 2;
  using Acc = std::conditional_t<std::is_same_v<T, double>, double, float>;
  parallel::parallel_for(
      0, query_pairs + key_pairs, parallel::GRAIN_SIZE,
      [&](int64_t begin, int64_t end) {
        for (int64_t pair = begin; pair < end; ++pair) {
          const T* input = query_c.data_ptr<T>();
          T* output = query_out.data_ptr<T>();
          int64_t local_pair = pair;
          int64_t tokens = query_tokens;
          if (pair >= query_pairs) {
            local_pair -= query_pairs;
            input = key_c.data_ptr<T>();
            output = key_out.data_ptr<T>();
            tokens = key_tokens;
          }
          const int64_t row = local_pair / table.half_dim;
          const int64_t pair_in_row = local_pair - row * table.half_dim;
          const int64_t token = row % tokens;
          const int64_t table_row = table.rows == 1
                                        ? 0
                                        : position_offset + token;
          const int64_t input_offset =
              row * (2 * table.half_dim) + 2 * pair_in_row;
          const int64_t table_offset = table_row * table.half_dim + pair_in_row;
          const Acc x0 = static_cast<Acc>(input[input_offset]);
          const Acc x1 = static_cast<Acc>(input[input_offset + 1]);
          const Acc c = static_cast<Acc>(cos_c.data_ptr<C>()[table_offset]);
          const Acc s = static_cast<Acc>(sin_c.data_ptr<C>()[table_offset]);
          output[input_offset] = static_cast<T>(x0 * c - x1 * s);
          output[input_offset + 1] = static_cast<T>(x0 * s + x1 * c);
        }
      });
  return std::make_tuple(query_out, key_out);
}

template <typename T>
Tensor rope_single_dispatch(const Tensor& input, const Tensor& cos,
                            const Tensor& sin, const RopeTable& table,
                            int64_t tokens, int64_t position_offset) {
  switch (cos.dtype()) {
    case DType::Float32:
      return rope_single_typed<T, float>(input, cos, sin, table, tokens,
                                         position_offset);
    case DType::Float64:
      return rope_single_typed<T, double>(input, cos, sin, table, tokens,
                                          position_offset);
    case DType::Float16:
      return rope_single_typed<T, Half>(input, cos, sin, table, tokens,
                                        position_offset);
    case DType::BFloat16:
      return rope_single_typed<T, BFloat16>(input, cos, sin, table, tokens,
                                            position_offset);
    default:
      TP_THROW(NotImplementedError, "rotary_embedding: unsupported table dtype");
  }
}

template <typename T>
std::tuple<Tensor, Tensor> rope_pair_dispatch(
    const Tensor& query, const Tensor& key, const Tensor& cos,
    const Tensor& sin, const RopeTable& table, int64_t query_tokens,
    int64_t key_tokens, int64_t position_offset) {
  switch (cos.dtype()) {
    case DType::Float32:
      return rope_pair_typed<T, float>(query, key, cos, sin, table,
                                       query_tokens, key_tokens,
                                       position_offset);
    case DType::Float64:
      return rope_pair_typed<T, double>(query, key, cos, sin, table,
                                        query_tokens, key_tokens,
                                        position_offset);
    case DType::Float16:
      return rope_pair_typed<T, Half>(query, key, cos, sin, table,
                                      query_tokens, key_tokens,
                                      position_offset);
    case DType::BFloat16:
      return rope_pair_typed<T, BFloat16>(query, key, cos, sin, table,
                                          query_tokens, key_tokens,
                                          position_offset);
    default:
      TP_THROW(NotImplementedError, "fused_rope: unsupported table dtype");
  }
}

} // namespace

Tensor rotary_embedding_cpu(const Tensor& input, const Tensor& cos,
                            const Tensor& sin, int64_t position_offset) {
  const int64_t tokens = rope_tokens(input, "rotary_embedding");
  const RopeTable table = check_rope_table(
      cos, sin, input.size(-1) / 2, tokens, position_offset, input.device(),
      "rotary_embedding");
  switch (input.dtype()) {
    case DType::Float32:
      return rope_single_dispatch<float>(input, cos, sin, table, tokens,
                                         position_offset);
    case DType::Float64:
      return rope_single_dispatch<double>(input, cos, sin, table, tokens,
                                          position_offset);
    case DType::Float16:
      return rope_single_dispatch<Half>(input, cos, sin, table, tokens,
                                        position_offset);
    case DType::BFloat16:
      return rope_single_dispatch<BFloat16>(input, cos, sin, table, tokens,
                                            position_offset);
    default:
      TP_THROW(NotImplementedError, "rotary_embedding: unsupported input dtype");
  }
}

std::tuple<Tensor, Tensor> fused_rope_cpu(
    const Tensor& query, const Tensor& key, const Tensor& cos,
    const Tensor& sin, int64_t position_offset) {
  const int64_t query_tokens = rope_tokens(query, "fused_rope");
  const int64_t key_tokens = rope_tokens(key, "fused_rope");
  if (query.device() != key.device()) {
    TP_THROW(DeviceMismatchError,
             "fused_rope: query and key must be on the same device");
  }
  if (query.dtype() != key.dtype()) {
    TP_THROW(RuntimeError, "fused_rope: query and key must have the same dtype");
  }
  if (query.dim() != key.dim() || query.size(-1) != key.size(-1) ||
      query_tokens != key_tokens) {
    TP_THROW(RuntimeError,
             "fused_rope: query/key must have the same rank, token length, and head dimension");
  }
  const RopeTable table = check_rope_table(
      cos, sin, query.size(-1) / 2, query_tokens, position_offset,
      query.device(), "fused_rope");
  switch (query.dtype()) {
    case DType::Float32:
      return rope_pair_dispatch<float>(query, key, cos, sin, table,
                                       query_tokens, key_tokens,
                                       position_offset);
    case DType::Float64:
      return rope_pair_dispatch<double>(query, key, cos, sin, table,
                                        query_tokens, key_tokens,
                                        position_offset);
    case DType::Float16:
      return rope_pair_dispatch<Half>(query, key, cos, sin, table,
                                      query_tokens, key_tokens,
                                      position_offset);
    case DType::BFloat16:
      return rope_pair_dispatch<BFloat16>(query, key, cos, sin, table,
                                          query_tokens, key_tokens,
                                          position_offset);
    default:
      TP_THROW(NotImplementedError, "fused_rope: unsupported input dtype");
  }
}

TENSORPLAY_LIBRARY_IMPL(CPU, TransformersKernels) {
  m.impl("scaled_dot_product_attention", sdpa_kernel_cpu);
  m.impl("scaled_dot_product_attention_backward", sdpa_backward_kernel_cpu);
  m.impl("grouped_mm", grouped_mm_cpu);
  m.impl("rotary_embedding", rotary_embedding_cpu);
  m.impl("fused_rope", fused_rope_cpu);
}

} // namespace cpu
} // namespace tensorplay
