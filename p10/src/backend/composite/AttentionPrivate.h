// Bodies of the attention contracts that name their own layout, mask and
// dropout state: the flash, memory-efficient and cuDNN shaped forwards, the
// scaled-dot-product entry points over them, and the backwards that read the
// saved logsumexp.
//
// Each body forms the score matrix, so it is the answer for every call a
// fused schedule cannot express.  They are registered under the Composite key
// (AttentionPrivate.cpp), which serves every backend; the CUDA kernels answer
// what their schedule can and hand everything else to these same functions.
//
// Two conventions hold throughout:
// - The logsumexp is kept at float32, or at float64 for float64 inputs, so a
//   backward pass that reads it back is not limited by its precision.  A row
//   with no visible key reports positive infinity, which makes exp(score -
//   lse) zero there -- the right weight for a row that contributes nothing.
// - Dropout is drawn from a seed the forward records in its seed outputs.
//   The positions dropped are a function of that seed alone, so the backward
//   draws the same positions from it on any backend.
#pragma once

#include "AttentionComposite.h"
#include "Generator.h"
#include "GradMode.h"

#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace composite {
namespace attention {

namespace ops = tpx::ops;

// Mask alignment codes of the memory-efficient contract.
constexpr int64_t kNoCustomMask = 0;
constexpr int64_t kCausalFromTopLeft = 1;
constexpr int64_t kCausalFromBottomRight = 2;

// The contracts with a derivative of their own are differentiated as a whole,
// so the primitives they are built from are not recorded one by one.
struct NoGradScope {
  bool previous;
  NoGradScope() : previous(GradMode::is_enabled()) { GradMode::set_enabled(false); }
  ~NoGradScope() { GradMode::set_enabled(previous); }
  NoGradScope(const NoGradScope&) = delete;
  NoGradScope& operator=(const NoGradScope&) = delete;
};

inline bool is_reduced(DType dtype) {
  return dtype == DType::Float16 || dtype == DType::BFloat16;
}

inline DType working_dtype(DType dtype) {
  return is_reduced(dtype) ? DType::Float32 : dtype;
}

inline DType lse_dtype(DType dtype) {
  return dtype == DType::Float64 ? DType::Float64 : DType::Float32;
}

inline void check_floating(const Tensor& query, const char* contract) {
  const DType dt = query.dtype();
  if (dt != DType::Float32 && dt != DType::Float64 && !is_reduced(dt)) {
    TP_THROW(NotImplementedError, contract,
             ": expected float32, float64, float16 or bfloat16 inputs, got ",
             dt);
  }
}

inline double resolve_scale(std::optional<double> scale, int64_t head_dim) {
  return scale.has_value() ? *scale
                           : 1.0 / std::sqrt(static_cast<double>(head_dim));
}

// ---------------------------------------------------------------------------
// Layout checks.  The batched contracts differ only in which of the second
// and third axes is the sequence, and a product along the wrong one still has
// the right shape, so the rank is checked by name before anything is formed.
// ---------------------------------------------------------------------------

inline std::string shape_text(const Tensor& t) {
  std::string text = "(";
  for (int64_t i = 0; i < t.dim(); ++i) {
    if (i) text += ", ";
    text += std::to_string(t.size(i));
  }
  if (t.dim() == 1) text += ",";
  return text + ")";
}

inline void check_layout(const Tensor& query, const Tensor& key,
                         const Tensor& value, bool packed, bool heads_second) {
  const std::pair<const char*, const Tensor*> named[] = {
      {"query", &query}, {"key", &key}, {"value", &value}};
  for (const auto& [name, tensor] : named) {
    if (packed && tensor->dim() != 3) {
      TP_THROW(ValueError, "packed attention takes ", name,
               " as (total, heads, dim), got ", tensor->dim(), " axes ",
               shape_text(*tensor));
    }
    if (!packed && tensor->dim() != 4) {
      TP_THROW(ValueError, "batched attention takes ", name, " as (batch, ",
               heads_second ? "heads, sequence" : "sequence, heads",
               ", dim), got ", tensor->dim(), " axes ", shape_text(*tensor));
    }
  }
}

// Grouped heads: one key head serves a contiguous run of query heads, so the
// query head count has to be a whole number of runs.
inline void check_heads(int64_t heads_q, int64_t heads_k) {
  if (heads_k == 0 || heads_q % heads_k != 0) {
    TP_THROW(ValueError, "query heads ", heads_q,
             " must be a multiple of key/value heads ", heads_k);
  }
}

// ---------------------------------------------------------------------------
// Masks.  Every mask is a keep-mask over one (queries, keys) block, restated
// additively as 0 or -inf in the working precision.
// ---------------------------------------------------------------------------

inline Tensor positions(int64_t n, const Device& device) {
  return ops::arange(Scalar(0), Scalar(n), Scalar(1), DType::Int64, device);
}

// A window measured from the diagonal running from the top-left corner to the
// bottom-right one: query r sees the keys from r + lk - lq - left up to but
// not including r + lk - lq + right + 1.  A negative bound is unbounded.
inline std::optional<Tensor> window_keep(int64_t lq, int64_t lk, int64_t left,
                                         int64_t right, const Device& device) {
  if (left < 0 && right < 0) return std::nullopt;
  Tensor rows = ops::add(ops::view(positions(lq, device), {lq, 1}),
                         Scalar(lk - lq));
  Tensor cols = ops::view(positions(lk, device), {1, lk});
  std::optional<Tensor> keep;
  if (left >= 0) keep = ops::ge(cols, ops::sub(rows, Scalar(left)));
  if (right >= 0) {
    Tensor nearer = ops::lt(cols, ops::add(rows, Scalar(right + 1)));
    keep = keep.has_value() ? ops::logical_and(*keep, nearer) : nearer;
  }
  return keep;
}

// Query r sees key c when c <= r + offset.
inline Tensor diagonal_keep(int64_t lq, int64_t lk, int64_t offset,
                            const Device& device) {
  Tensor rows = ops::view(positions(lq, device), {lq, 1});
  Tensor cols = ops::view(positions(lk, device), {1, lk});
  return ops::le(cols, ops::add(rows, Scalar(offset)));
}

inline Tensor and_keep(const std::optional<Tensor>& a, const Tensor& b) {
  return a.has_value() ? ops::logical_and(*a, b) : b;
}

inline Tensor additive(const Tensor& keep, DType dtype) {
  return ops::where(keep, ops::full({}, Scalar(0), dtype, keep.device()),
                    ops::full({}, Scalar(kNegInf), dtype, keep.device()));
}

inline Tensor add_mask(const std::optional<Tensor>& mask, const Tensor& term) {
  return mask.has_value() ? ops::add(*mask, term) : term;
}

// A bias given as booleans keeps where it is true.
inline Tensor bias_term(const Tensor& bias, DType dtype) {
  return bias.dtype() == DType::Bool ? additive(bias, dtype) : ops::to(bias, dtype);
}

// The linear bias of the flash contract: -slope * |r + lk - lq - c| for query
// r and key c, one slope per head or per (batch, head).
inline Tensor alibi_term(const Tensor& slopes, int64_t batch, int64_t heads,
                         int64_t lq, int64_t lk, DType dtype) {
  const Device device = slopes.device();
  Tensor rows = ops::view(positions(lq, device), {lq, 1});
  Tensor cols = ops::view(positions(lk, device), {1, lk});
  Tensor distance = ops::abs(ops::sub(ops::add(rows, Scalar(lk - lq)), cols));
  Tensor s = ops::to(slopes, dtype);
  if (s.dim() == 1) {
    s = ops::view(s, {1, heads, 1, 1});
  } else {
    TP_CHECK(s.dim() == 2 && s.size(0) == batch && s.size(1) == heads,
             "alibi_slopes must be (heads,) or (batch, heads), got ",
             shape_text(slopes));
    s = ops::view(s, {batch, heads, 1, 1});
  }
  return ops::neg(ops::mul(s, ops::to(distance, dtype)));
}

// The keys a batch entry reads are a prefix of the ones it was handed.
inline Tensor prefix_keep(const Tensor& lengths, int64_t lk) {
  Tensor cols = ops::view(positions(lk, lengths.device()), {1, 1, 1, lk});
  Tensor bound = ops::view(ops::to(lengths, DType::Int64), {lengths.numel(), 1, 1, 1});
  return ops::lt(cols, bound);
}

// ---------------------------------------------------------------------------
// Dropout state.
// ---------------------------------------------------------------------------

// A fresh seed for a call that drops, drawn from the default generator so a
// seeded program repeats its draws.
inline uint64_t draw_seed() { return default_generator().random64(); }

// The seed a forward recorded in a seed tensor, whatever integer type holds it.
inline uint64_t read_seed(const Tensor& seed) {
  TP_CHECK(seed.defined() && seed.numel() >= 1,
           "attention backward: dropout needs the seed its forward recorded");
  Tensor host = seed.contiguous().to(Device(DeviceType::CPU));
  uint64_t value = 0;
  std::memcpy(&value, host.data_ptr(), std::min<size_t>(sizeof(value), host.itemsize()));
  return value;
}

inline Tensor seed_tensor(uint64_t seed, DType dtype, const Device& device) {
  Tensor host = Tensor::zeros({}, dtype, Device(DeviceType::CPU));
  std::memcpy(host.data_ptr(), &seed, std::min<size_t>(sizeof(seed), host.itemsize()));
  return host.to(device);
}

// The positions one (batch, heads, queries, keys) block drops.  Draws are taken
// from ``generator`` in call order, so a forward and its backward that walk the
// blocks in the same order with generators built from the same seed agree.
inline Tensor dropped_positions(Generator& generator,
                                const std::vector<int64_t>& shape, double p,
                                const Device& device) {
  Tensor draws = ops::rand(shape, generator, DType::Float32, std::nullopt,
                           device, std::nullopt);
  return ops::lt(draws, Scalar(p));
}

// ---------------------------------------------------------------------------
// The computation on heads-second blocks: query (B, Hq, Lq, D), key and value
// (B, Hk, Lk, D / Dv), with Hq a multiple of Hk.
// ---------------------------------------------------------------------------

inline Tensor expand_heads(const Tensor& t, int64_t heads_q) {
  const int64_t heads_k = t.size(1);
  if (heads_k == heads_q) return t;
  const int64_t group = heads_q / heads_k;
  std::vector<int64_t> shape = t.shape();
  std::vector<int64_t> grouped = {shape[0], heads_k, group, shape[2], shape[3]};
  return ops::reshape(
      ops::expand(ops::unsqueeze(t, 2), grouped),
      {shape[0], heads_q, shape[2], shape[3]});
}

inline Tensor fold_heads(const Tensor& t, int64_t heads_k) {
  const int64_t heads_q = t.size(1);
  if (heads_k == heads_q) return t;
  std::vector<int64_t> shape = t.shape();
  return ops::sum(ops::view(t, {shape[0], heads_k, heads_q / heads_k, shape[2],
                                shape[3]}),
                  {2}, false);
}

struct BlockResult {
  Tensor out;
  Tensor lse;
};

// ``mask`` is additive and broadcasts against (B, Hq, Lq, Lk); ``dropped``
// marks the positions dropped with probability ``p``.
inline BlockResult block_forward(const Tensor& query, const Tensor& key,
                                 const Tensor& value,
                                 const std::optional<Tensor>& mask, double scale,
                                 const std::optional<Tensor>& dropped, double p) {
  const DType origin = query.dtype();
  const DType working = working_dtype(origin);
  check_heads(query.size(1), key.size(1));
  Tensor q = ops::to(query, working);
  Tensor k = expand_heads(ops::to(key, working), q.size(1));
  Tensor v = expand_heads(ops::to(value, working), q.size(1));
  Tensor scores = ops::matmul(ops::mul(q, Scalar(scale)), ops::transpose(k, -2, -1));
  if (mask.has_value()) scores = ops::add(scores, ops::to(*mask, working));
  // A row whose largest score is minus infinity has nothing to exponentiate
  // against; its reference point is zero and its entries still exponentiate
  // to zero, which leaves the row's total at zero.
  Tensor row_max = ops::amax(scores, {-1}, true);
  row_max = ops::where(ops::eq(row_max, Scalar(kNegInf)), ops::zeros_like(row_max),
                       row_max);
  Tensor weights = ops::exp(ops::sub(scores, row_max));
  Tensor total = ops::sum(weights, {-1}, true);
  Tensor empty = ops::eq(total, Scalar(0));
  Tensor probs = ops::div(weights, ops::where(empty, ops::ones_like(total), total));
  Tensor lse = ops::where(
      ops::squeeze(empty, -1),
      ops::full({}, Scalar(std::numeric_limits<double>::infinity()), working,
                query.device()),
      ops::squeeze(ops::add(row_max, ops::log(total)), -1));
  if (p > 0.0) {
    TP_CHECK(dropped.has_value(), "attention: dropout needs its drawn positions");
    probs = ops::mul(ops::where(*dropped, Scalar(0), probs), Scalar(1.0 / (1.0 - p)));
  }
  Tensor out = ops::matmul(probs, v);
  return {ops::to(out, origin), ops::to(lse, lse_dtype(origin))};
}

struct BlockGrads {
  Tensor query;
  Tensor key;
  Tensor value;
  Tensor scores;  // the gradient of the masked scores, when asked for
};

// The backward of ``block_forward`` from the logsumexp it reported, which is
// what lets a caller that merged constants across shards differentiate each
// shard against the merged one.
inline BlockGrads block_backward(const Tensor& grad_out, const Tensor& query,
                                 const Tensor& key, const Tensor& value,
                                 const Tensor& out, const Tensor& lse,
                                 const std::optional<Tensor>& mask, double scale,
                                 const std::optional<Tensor>& dropped, double p,
                                 bool want_scores) {
  const DType working = working_dtype(query.dtype());
  const int64_t heads_q = query.size(1);
  const int64_t heads_k = key.size(1);
  check_heads(heads_q, heads_k);
  Tensor q = ops::to(query, working);
  Tensor k = expand_heads(ops::to(key, working), heads_q);
  Tensor v = expand_heads(ops::to(value, working), heads_q);
  Tensor go = ops::to(grad_out, working);
  Tensor o = ops::to(out, working);
  Tensor scores = ops::matmul(ops::mul(q, Scalar(scale)), ops::transpose(k, -2, -1));
  if (mask.has_value()) scores = ops::add(scores, ops::to(*mask, working));
  Tensor probs = ops::exp(ops::sub(scores, ops::unsqueeze(ops::to(lse, working), -1)));
  const double keep_scale = p > 0.0 ? 1.0 / (1.0 - p) : 1.0;
  auto drop = [&](const Tensor& t) {
    if (p <= 0.0) return t;
    TP_CHECK(dropped.has_value(), "attention backward: dropout needs its drawn positions");
    return ops::mul(ops::where(*dropped, Scalar(0), t), Scalar(keep_scale));
  };
  Tensor grad_v = ops::matmul(ops::transpose(drop(probs), -2, -1), go);
  Tensor grad_probs = drop(ops::matmul(go, ops::transpose(v, -2, -1)));
  // The row sums of probs * grad_probs are the row sums of grad_out * out.
  Tensor delta = ops::sum(ops::mul(go, o), {-1}, true);
  Tensor grad_scores = ops::mul(probs, ops::sub(grad_probs, delta));
  Tensor grad_q = ops::mul(ops::matmul(grad_scores, k), Scalar(scale));
  Tensor grad_k = ops::mul(
      ops::matmul(ops::transpose(grad_scores, -2, -1), q), Scalar(scale));
  BlockGrads grads;
  grads.query = ops::to(grad_q, query.dtype());
  grads.key = ops::to(fold_heads(grad_k, heads_k), key.dtype());
  grads.value = ops::to(fold_heads(grad_v, heads_k), value.dtype());
  if (want_scores) grads.scores = grad_scores;
  return grads;
}

// ---------------------------------------------------------------------------
// Packed sequences: the cumulative-length tables are read on the host, and
// each sequence is a block of its own.
// ---------------------------------------------------------------------------

inline std::vector<int64_t> read_bounds(const Tensor& table) {
  TP_CHECK(table.dim() == 1, "a cumulative-length table must be 1-D, got ",
           shape_text(table));
  Tensor host = ops::to(table.contiguous().to(Device(DeviceType::CPU)), DType::Int64);
  const int64_t* data = static_cast<const int64_t*>(host.data_ptr());
  return std::vector<int64_t>(data, data + host.numel());
}

struct Bounds {
  std::vector<int64_t> q;
  std::vector<int64_t> k;
  size_t count() const { return q.empty() ? 0 : q.size() - 1; }
};

inline Bounds read_tables(const Tensor& cum_q, const std::optional<Tensor>& cum_k) {
  Bounds b;
  b.q = read_bounds(cum_q);
  b.k = cum_k.has_value() && cum_k->defined() ? read_bounds(*cum_k) : b.q;
  if (b.q.size() != b.k.size()) {
    TP_THROW(ValueError, "query and key sequence tables disagree: ", b.q.size(),
             " vs ", b.k.size(), " entries");
  }
  return b;
}

// A (tokens, heads, dim) slice as a one-entry heads-second batch.
inline Tensor packed_block(const Tensor& t, int64_t start, int64_t stop) {
  return ops::transpose(ops::unsqueeze(ops::slice(t, 0, start, stop, 1), 0), 1, 2);
}

// And back: (1, heads, tokens, dim) to (tokens, heads, dim).
inline Tensor packed_rows(const Tensor& t) {
  return ops::transpose(ops::squeeze(t, 0), 0, 1);
}

// What masks one block: the per-contract rules below fill these in.
struct BlockMaskRule {
  int64_t window_left = -1;     // bottom-right window bounds
  int64_t window_right = -1;
  std::optional<int64_t> diagonal;  // keep c <= r + diagonal
  std::optional<int64_t> band;      // keep c > r + diagonal_band - band
  int64_t band_offset = 0;
};

inline std::optional<Tensor> block_keep(const BlockMaskRule& rule, int64_t lq,
                                        int64_t lk, const Device& device) {
  std::optional<Tensor> keep = window_keep(lq, lk, rule.window_left,
                                           rule.window_right, device);
  if (rule.diagonal.has_value()) {
    keep = and_keep(keep, diagonal_keep(lq, lk, *rule.diagonal, device));
  }
  if (rule.band.has_value()) {
    Tensor rows = ops::view(positions(lq, device), {lq, 1});
    Tensor cols = ops::view(positions(lk, device), {1, lk});
    keep = and_keep(keep, ops::gt(cols, ops::add(rows, Scalar(rule.band_offset - *rule.band))));
  }
  return keep;
}

// ---------------------------------------------------------------------------
// Shared pieces of the contract bodies.
// ---------------------------------------------------------------------------

inline void check_dropout(double p, const char* contract) {
  if (!(p >= 0.0 && p < 1.0)) {
    TP_THROW(ValueError, contract, ": dropout probability must be in [0, 1), got ", p);
  }
}

// One (heads-second) block: masks, dropout positions and the computation.
struct BlockSpec {
  BlockMaskRule rule;
  std::optional<Tensor> key_lengths;  // (B,) prefix of keys each entry reads
  std::optional<Tensor> bias;         // broadcasts against (B, Hq, Lq, Lk)
  std::optional<Tensor> alibi;        // (Hq,) or (B, Hq)
};

inline std::optional<Tensor> block_mask(const BlockSpec& spec, int64_t batch,
                                        int64_t heads, int64_t lq, int64_t lk,
                                        DType working, const Device& device) {
  std::optional<Tensor> keep = block_keep(spec.rule, lq, lk, device);
  if (spec.key_lengths.has_value()) {
    keep = and_keep(keep, prefix_keep(*spec.key_lengths, lk));
  }
  std::optional<Tensor> mask;
  if (keep.has_value()) mask = additive(*keep, working);
  if (spec.bias.has_value()) mask = add_mask(mask, bias_term(*spec.bias, working));
  if (spec.alibi.has_value()) {
    mask = add_mask(mask, alibi_term(*spec.alibi, batch, heads, lq, lk, working));
  }
  return mask;
}

// A block with no keys at all: every row is empty, so it outputs zeros and an
// infinite constant.
inline BlockResult empty_block(const Tensor& query, const Tensor& value) {
  std::vector<int64_t> out_shape = query.shape();
  out_shape.back() = value.size(-1);
  std::vector<int64_t> lse_shape(out_shape.begin(), out_shape.end() - 1);
  return {Tensor::zeros(out_shape, query.dtype(), query.device()),
          ops::full(lse_shape, Scalar(std::numeric_limits<double>::infinity()),
                    lse_dtype(query.dtype()), query.device())};
}

inline BlockResult run_block(const Tensor& q, const Tensor& k, const Tensor& v,
                             const BlockSpec& spec, double scale, double p,
                             std::optional<Generator>& generator) {
  const int64_t batch = q.size(0), heads = q.size(1);
  const int64_t lq = q.size(2), lk = k.size(2);
  if (lk == 0) return empty_block(q, v);
  std::optional<Tensor> mask = block_mask(spec, batch, heads, lq, lk,
                                          working_dtype(q.dtype()), q.device());
  std::optional<Tensor> dropped;
  if (p > 0.0) dropped = dropped_positions(*generator, {batch, heads, lq, lk}, p, q.device());
  return block_forward(q, k, v, mask, scale, dropped, p);
}

inline BlockGrads run_block_backward(const Tensor& go, const Tensor& q,
                                     const Tensor& k, const Tensor& v,
                                     const Tensor& out, const Tensor& lse,
                                     const BlockSpec& spec, double scale,
                                     double p, std::optional<Generator>& generator,
                                     bool want_scores) {
  const int64_t batch = q.size(0), heads = q.size(1);
  const int64_t lq = q.size(2), lk = k.size(2);
  if (lk == 0) {
    return {Tensor::zeros(q.shape(), q.dtype(), q.device()),
            Tensor::zeros(k.shape(), k.dtype(), k.device()),
            Tensor::zeros(v.shape(), v.dtype(), v.device()), Tensor()};
  }
  std::optional<Tensor> mask = block_mask(spec, batch, heads, lq, lk,
                                          working_dtype(q.dtype()), q.device());
  std::optional<Tensor> dropped;
  if (p > 0.0) dropped = dropped_positions(*generator, {batch, heads, lq, lk}, p, q.device());
  return block_backward(go, q, k, v, out, lse, mask, scale, dropped, p, want_scores);
}

inline std::optional<Generator> generator_for(double p, uint64_t seed) {
  if (p <= 0.0) return std::nullopt;
  return Generator(seed);
}

inline int64_t round_up_32(int64_t n) { return (n + 31) / 32 * 32; }

inline Tensor infinite(const std::vector<int64_t>& shape, DType dtype,
                       const Device& device) {
  return ops::full(shape, Scalar(std::numeric_limits<double>::infinity()), dtype,
                   device);
}

// Writes ``piece`` into rows [start, start + piece rows) of ``dst`` along ``dim``.
inline void write_rows(Tensor dst, int64_t dim, int64_t start, const Tensor& piece) {
  Tensor window = ops::slice(dst, dim, start, start + piece.size(dim), 1);
  window.copy_(piece);
}

inline void add_rows(Tensor dst, int64_t dim, int64_t start, const Tensor& piece) {
  Tensor window = ops::slice(dst, dim, start, start + piece.size(dim), 1);
  window.add_(piece);
}

// A (tokens, heads, dim) packed layout, computed one sequence at a time.
struct PackedResult {
  Tensor out;                 // (total_q, Hq, Dv)
  std::vector<Tensor> lses;   // per sequence, (Hq, lq), empty for skipped ones
};

inline PackedResult packed_forward(const Tensor& query, const Tensor& key,
                                   const Tensor& value, const Bounds& bounds,
                                   const std::function<BlockSpec(size_t, int64_t, int64_t)>& spec_for,
                                   const std::optional<std::vector<int64_t>>& key_extent,
                                   double scale, double p,
                                   std::optional<Generator>& generator) {
  const int64_t total_q = query.size(0), heads = query.size(1);
  check_heads(heads, key.size(1));
  PackedResult result;
  result.out = Tensor::zeros({total_q, heads, value.size(-1)}, query.dtype(),
                             query.device());
  result.lses.resize(bounds.count());
  for (size_t seq = 0; seq < bounds.count(); ++seq) {
    const int64_t start_q = bounds.q[seq], stop_q = bounds.q[seq + 1];
    const int64_t start_k = bounds.k[seq];
    int64_t stop_k = bounds.k[seq + 1];
    if (key_extent.has_value()) stop_k = std::min(stop_k, start_k + (*key_extent)[seq]);
    const int64_t lq = stop_q - start_q, lk = stop_k - start_k;
    if (lq <= 0) continue;
    BlockSpec spec = spec_for(seq, lq, lk);
    BlockResult r = run_block(packed_block(query, start_q, stop_q),
                              packed_block(key, start_k, stop_k),
                              packed_block(value, start_k, stop_k), spec, scale,
                              p, generator);
    write_rows(result.out, 0, start_q, packed_rows(r.out));
    result.lses[seq] = ops::squeeze(r.lse, 0);
  }
  return result;
}

struct PackedGrads {
  Tensor query, key, value;
};

inline PackedGrads packed_backward(const Tensor& grad_out, const Tensor& query,
                                   const Tensor& key, const Tensor& value,
                                   const Tensor& out, const Bounds& bounds,
                                   const std::function<Tensor(size_t, int64_t, int64_t)>& lse_for,
                                   const std::function<BlockSpec(size_t, int64_t, int64_t)>& spec_for,
                                   double scale, double p,
                                   std::optional<Generator>& generator) {
  PackedGrads grads;
  grads.query = Tensor::zeros(query.shape(), query.dtype(), query.device());
  grads.key = Tensor::zeros(key.shape(), key.dtype(), key.device());
  grads.value = Tensor::zeros(value.shape(), value.dtype(), value.device());
  for (size_t seq = 0; seq < bounds.count(); ++seq) {
    const int64_t start_q = bounds.q[seq], stop_q = bounds.q[seq + 1];
    const int64_t start_k = bounds.k[seq], stop_k = bounds.k[seq + 1];
    const int64_t lq = stop_q - start_q, lk = stop_k - start_k;
    if (lq <= 0) continue;
    BlockSpec spec = spec_for(seq, lq, lk);
    BlockGrads g = run_block_backward(
        packed_block(grad_out, start_q, stop_q), packed_block(query, start_q, stop_q),
        packed_block(key, start_k, stop_k), packed_block(value, start_k, stop_k),
        packed_block(out, start_q, stop_q), ops::unsqueeze(lse_for(seq, start_q, lq), 0),
        spec, scale, p, generator, false);
    add_rows(grads.query, 0, start_q, packed_rows(g.query));
    add_rows(grads.key, 0, start_k, packed_rows(g.key));
    add_rows(grads.value, 0, start_k, packed_rows(g.value));
  }
  return grads;
}

inline std::vector<int64_t> host_lengths(const Tensor& lengths) {
  return read_bounds(lengths);
}

inline Tensor flash_rng_state(uint64_t seed, const Device& device) {
  Tensor host = Tensor::zeros({2}, DType::UInt64, Device(DeviceType::CPU));
  std::memcpy(host.data_ptr(), &seed, sizeof(seed));
  return host.to(device);
}

// The host's own fused kernel answers a batched call with no window, or with
// the causal one on equal lengths (where its upper-left corner is the same as
// the lower-right one), equal head widths and nothing dropped.
inline bool host_kernel_serves(const Tensor& query, const Tensor& key,
                               const Tensor& value, int64_t left, int64_t right,
                               double dropout_p) {
  if (!query.device().is_cpu() || dropout_p != 0.0) return false;
  if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4) return false;
  const DType dt = query.dtype();
  if (key.dtype() != dt || value.dtype() != dt) return false;
  if (value.size(-1) != query.size(-1) || key.size(-1) != query.size(-1)) return false;
  if (left >= 0) return false;
  if (right < 0) return true;
  return right == 0 && query.size(1) == key.size(1);
}

// ---------------------------------------------------------------------------
// `_flash_attention_forward`: batched inputs are (batch, sequence, heads, dim),
// a cumulative-length table means they are packed as (total, heads, dim).  The
// causal flag is the window whose right bound is zero, so it aligns to the
// lower-right corner when the lengths differ.  The constant is (batch, heads,
// queries), or (heads, total) when packed.
// ---------------------------------------------------------------------------

using FlashForward = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>;

inline FlashForward flash_forward(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const std::optional<Tensor>& cum_seq_q, const std::optional<Tensor>& cum_seq_k,
    int64_t max_q, int64_t max_k, double dropout_p, bool is_causal,
    bool return_debug_mask, std::optional<double> scale,
    std::optional<int64_t> window_size_left, std::optional<int64_t> window_size_right,
    const std::optional<Tensor>& seqused_k, const std::optional<Tensor>& alibi_slopes,
    const std::optional<Tensor>& block_table, std::optional<int64_t> num_splits) {
  (void)max_q, (void)max_k, (void)return_debug_mask, (void)num_splits;
  NoGradScope no_grad;
  check_floating(query, "flash attention");
  check_dropout(dropout_p, "flash attention");
  if (block_table.has_value()) {
    TP_THROW(NotImplementedError,
             "flash attention: block_table names a paged key/value store, "
             "which only a kernel that walks the pages can read");
  }
  const bool packed = cum_seq_q.has_value();
  if (packed != cum_seq_k.has_value()) {
    TP_THROW(ValueError,
             "cumulative query and key lengths must both be given or both be absent");
  }
  check_layout(query, key, value, packed, /*heads_second=*/false);
  const double s = resolve_scale(scale, query.size(-1));
  BlockMaskRule rule;
  rule.window_left = window_size_left.value_or(-1);
  rule.window_right = is_causal ? 0 : window_size_right.value_or(-1);
  const uint64_t seed = dropout_p > 0.0 ? draw_seed() : 0;
  std::optional<Generator> generator = generator_for(dropout_p, seed);
  const Device device = query.device();
  Tensor out, lse;
  if (!packed && !seqused_k.has_value() && !alibi_slopes.has_value() &&
      host_kernel_serves(query, key, value, rule.window_left, rule.window_right,
                         dropout_p)) {
    check_heads(query.size(2), key.size(2));
    auto [heads_out, heads_lse] =
        ::tensorplay::detail::redispatch__scaled_dot_product_flash_attention_for_cpu_function(
            ops::transpose(query, 1, 2), ops::transpose(key, 1, 2),
            ops::transpose(value, 1, 2), 0.0, rule.window_right == 0,
            std::nullopt, s);
    out = ops::transpose(heads_out, 1, 2);
    lse = heads_lse;
  } else if (!packed) {
    const int64_t heads = query.size(2);
    check_heads(heads, key.size(2));
    BlockSpec spec;
    spec.rule = rule;
    if (seqused_k.has_value()) spec.key_lengths = *seqused_k;
    if (alibi_slopes.has_value()) spec.alibi = *alibi_slopes;
    BlockResult r = run_block(ops::transpose(query, 1, 2), ops::transpose(key, 1, 2),
                              ops::transpose(value, 1, 2), spec, s, dropout_p,
                              generator);
    out = ops::transpose(r.out, 1, 2);
    lse = r.lse;
  } else {
    const Bounds bounds = read_tables(*cum_seq_q, cum_seq_k);
    std::optional<std::vector<int64_t>> used;
    if (seqused_k.has_value()) used = host_lengths(*seqused_k);
    auto spec_for = [&](size_t seq, int64_t, int64_t) {
      BlockSpec spec;
      spec.rule = rule;
      if (alibi_slopes.has_value()) {
        spec.alibi = alibi_slopes->dim() == 2
                         ? ops::select(*alibi_slopes, 0, static_cast<int64_t>(seq))
                         : *alibi_slopes;
      }
      return spec;
    };
    PackedResult r = packed_forward(query, key, value, bounds, spec_for, used, s,
                                    dropout_p, generator);
    out = r.out;
    const int64_t heads = query.size(1);
    lse = infinite({heads, query.size(0)}, lse_dtype(query.dtype()), device);
    for (size_t seq = 0; seq < bounds.count(); ++seq) {
      if (r.lses[seq].defined()) write_rows(lse, 1, bounds.q[seq], r.lses[seq]);
    }
  }
  return {out, lse, flash_rng_state(seed, device),
          Tensor::zeros({}, DType::UInt64, device),
          Tensor::empty({0}, query.dtype(), device)};
}

inline Tensor flash_forward_into(
    Tensor& out, const Tensor& query, const Tensor& key, const Tensor& value,
    const std::optional<Tensor>& cum_seq_q, const std::optional<Tensor>& cum_seq_k,
    int64_t max_q, int64_t max_k, double dropout_p, bool is_causal,
    bool return_debug_mask, std::optional<double> scale,
    std::optional<int64_t> window_size_left, std::optional<int64_t> window_size_right,
    const std::optional<Tensor>& seqused_k, const std::optional<Tensor>& alibi_slopes,
    const std::optional<Tensor>& block_table, std::optional<int64_t> num_splits) {
  if (dropout_p != 0.0) {
    TP_THROW(ValueError,
             "_flash_attention_forward_no_dropout_inplace: dropout_p must be 0, got ",
             dropout_p);
  }
  auto result = ::tensorplay::detail::redispatch__flash_attention_forward_function(
      query, key, value, cum_seq_q, cum_seq_k, max_q, max_k, 0.0, is_causal,
      return_debug_mask, scale, window_size_left, window_size_right, seqused_k,
      alibi_slopes, block_table, num_splits);
  NoGradScope no_grad;
  out.copy_(std::get<0>(result));
  return std::get<1>(result);
}

inline std::tuple<Tensor, Tensor, Tensor> flash_backward(
    const Tensor& grad_out, const Tensor& query, const Tensor& key,
    const Tensor& value, const Tensor& out, const Tensor& logsumexp,
    const Tensor& cum_seq_q, const Tensor& cum_seq_k, int64_t max_q,
    int64_t max_k, double dropout_p, bool is_causal, const Tensor& rng_state,
    const Tensor& unused, std::optional<double> scale,
    std::optional<int64_t> window_size_left, std::optional<int64_t> window_size_right) {
  (void)max_q, (void)max_k, (void)unused;
  if (!grad_out.defined()) return {Tensor(), Tensor(), Tensor()};
  NoGradScope no_grad;
  check_dropout(dropout_p, "flash attention backward");
  const bool packed = cum_seq_q.defined();
  check_layout(query, key, value, packed, /*heads_second=*/false);
  const double s = resolve_scale(scale, query.size(-1));
  BlockSpec spec;
  spec.rule.window_left = window_size_left.value_or(-1);
  spec.rule.window_right = is_causal ? 0 : window_size_right.value_or(-1);
  std::optional<Generator> generator =
      generator_for(dropout_p, dropout_p > 0.0 ? read_seed(rng_state) : 0);
  if (!packed && host_kernel_serves(query, key, value, spec.rule.window_left,
                                    spec.rule.window_right, dropout_p)) {
    check_heads(query.size(2), key.size(2));
    auto [grad_q, grad_k, grad_v] =
        ::tensorplay::detail::redispatch__scaled_dot_product_flash_attention_for_cpu_backward_function(
            ops::transpose(grad_out, 1, 2), ops::transpose(query, 1, 2),
            ops::transpose(key, 1, 2), ops::transpose(value, 1, 2),
            ops::transpose(out, 1, 2), logsumexp, 0.0,
            spec.rule.window_right == 0, std::nullopt, s);
    return {ops::transpose(grad_q, 1, 2), ops::transpose(grad_k, 1, 2),
            ops::transpose(grad_v, 1, 2)};
  }
  if (!packed) {
    BlockGrads g = run_block_backward(
        ops::transpose(grad_out, 1, 2), ops::transpose(query, 1, 2),
        ops::transpose(key, 1, 2), ops::transpose(value, 1, 2),
        ops::transpose(out, 1, 2), logsumexp, spec, s, dropout_p, generator, false);
    return {ops::transpose(g.query, 1, 2), ops::transpose(g.key, 1, 2),
            ops::transpose(g.value, 1, 2)};
  }
  const Bounds bounds = read_tables(cum_seq_q, cum_seq_k);
  auto lse_for = [&](size_t, int64_t start, int64_t length) {
    return ops::slice(logsumexp, 1, start, start + length, 1);
  };
  auto spec_for = [&](size_t, int64_t, int64_t) { return spec; };
  PackedGrads g = packed_backward(grad_out, query, key, value, out, bounds,
                                  lse_for, spec_for, s, dropout_p, generator);
  return {g.query, g.key, g.value};
}

// ---------------------------------------------------------------------------
// `_efficient_attention_forward`: batched inputs are (batch, sequence, heads,
// dim); packed ones are one batch entry of every token, (1, total, heads,
// dim), or the same without the leading axis.  The mask is a code: one keeps
// the keys at or before the query index, two the keys at or before the
// lower-right diagonal; a window keeps only the last ``window_size`` of those.
// The constant is (batch, heads, queries rounded up to 32), or (sequences,
// heads, longest sequence rounded up to 32) when packed, with the rows past a
// sequence's end at positive infinity.
// ---------------------------------------------------------------------------

inline BlockMaskRule efficient_rule(int64_t custom_mask_type, int64_t lq, int64_t lk,
                                    std::optional<int64_t> window_size) {
  if (custom_mask_type < kNoCustomMask || custom_mask_type > kCausalFromBottomRight) {
    TP_THROW(ValueError, "memory-efficient attention: unknown custom_mask_type ",
             custom_mask_type);
  }
  BlockMaskRule rule;
  const int64_t offset = custom_mask_type == kCausalFromBottomRight ? lk - lq : 0;
  if (custom_mask_type != kNoCustomMask) rule.diagonal = offset;
  if (window_size.has_value() && *window_size > 0) {
    rule.band = *window_size;
    rule.band_offset = offset;
  }
  return rule;
}

using EfficientForward = std::tuple<Tensor, Tensor, Tensor, Tensor, SymInt, SymInt>;

inline EfficientForward efficient_forward(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const std::optional<Tensor>& bias, const std::optional<Tensor>& cu_seqlens_q,
    const std::optional<Tensor>& cu_seqlens_k, std::optional<int64_t> max_seqlen_q,
    std::optional<int64_t> max_seqlen_k, double dropout_p, int64_t custom_mask_type,
    bool compute_log_sumexp, std::optional<double> scale,
    const std::optional<Tensor>& seqlen_k, std::optional<int64_t> window_size) {
  NoGradScope no_grad;
  check_floating(query, "memory-efficient attention");
  check_dropout(dropout_p, "memory-efficient attention");
  const bool packed = cu_seqlens_q.has_value();
  if (packed != cu_seqlens_k.has_value()) {
    TP_THROW(ValueError,
             "cumulative query and key lengths must both be given or both be absent");
  }
  // The packed contract carries a batch axis of one; it is read without it.
  const bool batch_axis = packed && query.dim() == 4;
  if (batch_axis) {
    TP_CHECK(query.size(0) == 1 && key.size(0) == 1 && value.size(0) == 1,
             "memory-efficient attention: packed inputs carry a batch axis of 1, got ",
             shape_text(query));
  }
  const Tensor q_in = batch_axis ? ops::squeeze(query, 0) : query;
  const Tensor k_in = batch_axis ? ops::squeeze(key, 0) : key;
  const Tensor v_in = batch_axis ? ops::squeeze(value, 0) : value;
  check_layout(q_in, k_in, v_in, packed, /*heads_second=*/false);
  if (packed && bias.has_value()) {
    TP_THROW(NotImplementedError,
             "memory-efficient attention: a bias over packed sequences has no "
             "batch axis to be read along");
  }
  const double s = resolve_scale(scale, query.size(-1));
  const uint64_t seed = dropout_p > 0.0 ? draw_seed() : 0;
  std::optional<Generator> generator = generator_for(dropout_p, seed);
  const Device device = query.device();
  const DType stat_dtype = lse_dtype(query.dtype());
  Tensor out, lse;
  int64_t result_max_q = 0, result_max_k = 0;
  if (!packed) {
    const int64_t batch = q_in.size(0), lq = q_in.size(1), heads = q_in.size(2);
    const int64_t lk = k_in.size(1);
    check_heads(heads, k_in.size(2));
    BlockSpec spec;
    spec.rule = efficient_rule(custom_mask_type, lq, lk, window_size);
    if (seqlen_k.has_value()) spec.key_lengths = *seqlen_k;
    if (bias.has_value()) spec.bias = *bias;
    BlockResult r = run_block(ops::transpose(q_in, 1, 2), ops::transpose(k_in, 1, 2),
                              ops::transpose(v_in, 1, 2), spec, s, dropout_p,
                              generator);
    out = ops::transpose(r.out, 1, 2);
    if (compute_log_sumexp) {
      lse = infinite({batch, heads, round_up_32(lq)}, stat_dtype, device);
      write_rows(lse, 2, 0, r.lse);
    } else {
      lse = Tensor::empty({batch, heads, 0}, stat_dtype, device);
    }
    result_max_q = lq;
    result_max_k = lk;
  } else {
    const Bounds bounds = read_tables(*cu_seqlens_q, cu_seqlens_k);
    std::optional<std::vector<int64_t>> extent;
    if (seqlen_k.has_value()) extent = host_lengths(*seqlen_k);
    int64_t longest_q = 0, longest_k = 0;
    for (size_t seq = 0; seq < bounds.count(); ++seq) {
      longest_q = std::max(longest_q, bounds.q[seq + 1] - bounds.q[seq]);
      longest_k = std::max(longest_k, bounds.k[seq + 1] - bounds.k[seq]);
    }
    result_max_q = max_seqlen_q.value_or(longest_q);
    result_max_k = max_seqlen_k.value_or(longest_k);
    auto spec_for = [&](size_t, int64_t lq, int64_t lk) {
      BlockSpec spec;
      spec.rule = efficient_rule(custom_mask_type, lq, lk, window_size);
      return spec;
    };
    PackedResult r = packed_forward(q_in, k_in, v_in, bounds, spec_for, extent, s,
                                    dropout_p, generator);
    out = r.out;
    const int64_t heads = q_in.size(1);
    const int64_t count = static_cast<int64_t>(bounds.count());
    if (compute_log_sumexp) {
      lse = infinite({count, heads, round_up_32(result_max_q)}, stat_dtype, device);
      for (size_t seq = 0; seq < bounds.count(); ++seq) {
        if (!r.lses[seq].defined()) continue;
        write_rows(ops::select(lse, 0, static_cast<int64_t>(seq)), 1, 0, r.lses[seq]);
      }
    } else {
      lse = Tensor::empty({count, heads, 0}, stat_dtype, device);
    }
    if (batch_axis) out = ops::unsqueeze(out, 0);
  }
  return {out, lse, seed_tensor(seed, DType::Int64, device),
          Tensor::zeros({}, DType::Int64, device), SymInt(result_max_q),
          SymInt(result_max_k)};
}

inline std::tuple<Tensor, Tensor, Tensor, Tensor> efficient_backward(
    const Tensor& grad_out_, const Tensor& query, const Tensor& key,
    const Tensor& value, const std::optional<Tensor>& bias, const Tensor& out,
    const std::optional<Tensor>& cu_seqlens_q, const std::optional<Tensor>& cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k, const Tensor& logsumexp,
    double dropout_p, const Tensor& philox_seed, const Tensor& philox_offset,
    int64_t custom_mask_type, bool bias_requires_grad, std::optional<double> scale,
    std::optional<int64_t> num_splits_key, std::optional<int64_t> window_size,
    bool shared_storage_dqdkdv) {
  (void)max_seqlen_q, (void)max_seqlen_k, (void)philox_offset, (void)num_splits_key,
      (void)shared_storage_dqdkdv;
  if (!grad_out_.defined()) return {Tensor(), Tensor(), Tensor(), Tensor()};
  NoGradScope no_grad;
  check_dropout(dropout_p, "memory-efficient attention backward");
  // An optional saved for the backward arrives as an undefined tensor.
  std::optional<Tensor> bias_in;
  if (bias.has_value() && bias->defined()) bias_in = *bias;
  const bool packed = cu_seqlens_q.has_value() && cu_seqlens_q->defined();
  if (bias_requires_grad && !bias_in.has_value()) {
    TP_THROW(ValueError, "bias_requires_grad is true but no bias was provided");
  }
  const double s = resolve_scale(scale, query.size(-1));
  std::optional<Generator> generator =
      generator_for(dropout_p, dropout_p > 0.0 ? read_seed(philox_seed) : 0);
  if (!packed) {
    const int64_t lq = query.size(1), lk = key.size(1);
    check_layout(query, key, value, false, /*heads_second=*/false);
    BlockSpec spec;
    spec.rule = efficient_rule(custom_mask_type, lq, lk, window_size);
    if (bias_in.has_value()) spec.bias = *bias_in;
    Tensor lse = ops::slice(logsumexp, 2, 0, lq, 1);
    BlockGrads g = run_block_backward(
        ops::transpose(grad_out_, 1, 2), ops::transpose(query, 1, 2),
        ops::transpose(key, 1, 2), ops::transpose(value, 1, 2),
        ops::transpose(out, 1, 2), lse, spec, s, dropout_p, generator,
        bias_requires_grad);
    Tensor grad_bias;
    if (bias_requires_grad) {
      grad_bias = ops::to(ops::sum_to_size(g.scores, bias_in->shape()), bias_in->dtype());
    }
    return {ops::transpose(g.query, 1, 2), ops::transpose(g.key, 1, 2),
            ops::transpose(g.value, 1, 2), grad_bias};
  }
  const bool batch_axis = query.dim() == 4;
  auto drop_axis = [&](const Tensor& t) { return batch_axis ? ops::squeeze(t, 0) : t; };
  const Bounds bounds = read_tables(*cu_seqlens_q, cu_seqlens_k);
  auto lse_for = [&](size_t seq, int64_t, int64_t length) {
    return ops::slice(ops::select(logsumexp, 0, static_cast<int64_t>(seq)), 1, 0,
                      length, 1);
  };
  auto spec_for = [&](size_t, int64_t lq, int64_t lk) {
    BlockSpec spec;
    spec.rule = efficient_rule(custom_mask_type, lq, lk, window_size);
    return spec;
  };
  PackedGrads g = packed_backward(drop_axis(grad_out_), drop_axis(query),
                                  drop_axis(key), drop_axis(value), drop_axis(out),
                                  bounds, lse_for, spec_for, s, dropout_p, generator);
  auto add_axis = [&](const Tensor& t) { return batch_axis ? ops::unsqueeze(t, 0) : t; };
  return {add_axis(g.query), add_axis(g.key), add_axis(g.value), Tensor()};
}

// ---------------------------------------------------------------------------
// `_cudnn_attention_forward`: batched inputs are head-major, (batch, heads,
// sequence, dim); packed ones are (total, heads, dim) like the flash contract.
// The causal flag keeps the keys at or before the query index.  The constant
// is (batch, heads, queries, 1), or (heads, total) when packed, and the tables
// a packed call was given are handed back as they are.
// ---------------------------------------------------------------------------

using CudnnForward = std::tuple<Tensor, Tensor, Tensor, Tensor, SymInt, SymInt,
                                Tensor, Tensor, Tensor>;

inline CudnnForward cudnn_forward(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const std::optional<Tensor>& attn_bias, const std::optional<Tensor>& cum_seq_q,
    const std::optional<Tensor>& cum_seq_k, int64_t max_q, int64_t max_k,
    bool compute_log_sumexp, double dropout_p, bool is_causal,
    bool return_debug_mask, std::optional<double> scale,
    const std::optional<Tensor>& seqused_k, const std::optional<Tensor>& block_table) {
  (void)return_debug_mask;
  NoGradScope no_grad;
  check_floating(query, "cuDNN attention");
  check_dropout(dropout_p, "cuDNN attention");
  if (block_table.has_value()) {
    TP_THROW(NotImplementedError,
             "cuDNN attention: block_table names a paged key/value store, "
             "which only a kernel that walks the pages can read");
  }
  const bool packed = cum_seq_q.has_value();
  if (packed && !cum_seq_k.has_value()) {
    TP_THROW(ValueError, "cuDNN varlen attention requires cum_seq_k");
  }
  check_layout(query, key, value, packed, /*heads_second=*/true);
  if (packed && attn_bias.has_value()) {
    TP_THROW(NotImplementedError,
             "cuDNN attention: a bias over packed sequences has no batch axis "
             "to be read along");
  }
  const double s = resolve_scale(scale, query.size(-1));
  const uint64_t seed = dropout_p > 0.0 ? draw_seed() : 0;
  std::optional<Generator> generator = generator_for(dropout_p, seed);
  const Device device = query.device();
  BlockMaskRule rule;
  if (is_causal) rule.diagonal = 0;
  Tensor out, lse;
  SymInt result_max_q, result_max_k;
  Tensor echo_q, echo_k;
  if (!packed) {
    check_heads(query.size(1), key.size(1));
    BlockSpec spec;
    spec.rule = rule;
    if (seqused_k.has_value()) spec.key_lengths = *seqused_k;
    if (attn_bias.has_value()) spec.bias = *attn_bias;
    BlockResult r = run_block(query, key, value, spec, s, dropout_p, generator);
    out = r.out;
    if (compute_log_sumexp) lse = ops::unsqueeze(r.lse, -1);
    result_max_q = SymInt(query.size(2));
    result_max_k = SymInt(key.size(2));
  } else {
    const Bounds bounds = read_tables(*cum_seq_q, cum_seq_k);
    std::optional<std::vector<int64_t>> used;
    if (seqused_k.has_value()) used = host_lengths(*seqused_k);
    auto spec_for = [&](size_t, int64_t, int64_t) {
      BlockSpec spec;
      spec.rule = rule;
      return spec;
    };
    PackedResult r = packed_forward(query, key, value, bounds, spec_for, used, s,
                                    dropout_p, generator);
    out = r.out;
    if (compute_log_sumexp) {
      lse = infinite({query.size(1), query.size(0)}, lse_dtype(query.dtype()), device);
      for (size_t seq = 0; seq < bounds.count(); ++seq) {
        if (r.lses[seq].defined()) write_rows(lse, 1, bounds.q[seq], r.lses[seq]);
      }
    }
    echo_q = *cum_seq_q;
    echo_k = *cum_seq_k;
    result_max_q = SymInt(max_q);
    result_max_k = SymInt(max_k);
  }
  return {out, lse, echo_q, echo_k, result_max_q, result_max_k,
          seed_tensor(seed, DType::Int64, device),
          Tensor::zeros({}, DType::Int64, device), Tensor()};
}

inline std::tuple<Tensor, Tensor, Tensor> cudnn_backward(
    const Tensor& grad_out, const Tensor& query, const Tensor& key,
    const Tensor& value, const Tensor& out, const Tensor& logsumexp,
    const Tensor& philox_seed, const Tensor& philox_offset, const Tensor& attn_bias,
    const Tensor& cum_seq_q, const Tensor& cum_seq_k, int64_t max_q, int64_t max_k,
    double dropout_p, bool is_causal, std::optional<double> scale) {
  (void)philox_offset, (void)max_q, (void)max_k;
  if (!grad_out.defined()) return {Tensor(), Tensor(), Tensor()};
  NoGradScope no_grad;
  check_dropout(dropout_p, "cuDNN attention backward");
  TP_CHECK(logsumexp.defined(),
           "cuDNN attention backward needs the logsumexp its forward computed; "
           "run the forward with compute_log_sumexp=True");
  const bool packed = cum_seq_q.defined() && cum_seq_q.numel() > 0;
  check_layout(query, key, value, packed, /*heads_second=*/true);
  const double s = resolve_scale(scale, query.size(-1));
  std::optional<Generator> generator =
      generator_for(dropout_p, dropout_p > 0.0 ? read_seed(philox_seed) : 0);
  BlockSpec spec;
  if (is_causal) spec.rule.diagonal = 0;
  if (!packed) {
    if (attn_bias.defined()) spec.bias = attn_bias;
    BlockGrads g = run_block_backward(grad_out, query, key, value, out,
                                      ops::squeeze(logsumexp, -1), spec, s,
                                      dropout_p, generator, false);
    return {g.query, g.key, g.value};
  }
  const Bounds bounds = read_tables(cum_seq_q, cum_seq_k);
  auto lse_for = [&](size_t, int64_t start, int64_t length) {
    return ops::slice(logsumexp, 1, start, start + length, 1);
  };
  auto spec_for = [&](size_t, int64_t, int64_t) { return spec; };
  PackedGrads g = packed_backward(grad_out, query, key, value, out, bounds,
                                  lse_for, spec_for, s, dropout_p, generator);
  return {g.query, g.key, g.value};
}

// ---------------------------------------------------------------------------
// The scaled-dot-product entry points: head-major (batch, heads, sequence,
// dim) inputs over the contracts above.  They reach those through the
// dispatcher so a backend's own kernel answers when it has one.
// ---------------------------------------------------------------------------

using SdpaFlash = std::tuple<Tensor, Tensor, Tensor, Tensor, SymInt, SymInt,
                             Tensor, Tensor, Tensor>;

inline SdpaFlash sdpa_flash(const Tensor& query, const Tensor& key,
                            const Tensor& value, double dropout_p, bool is_causal,
                            bool return_debug_mask, std::optional<double> scale) {
  NoGradScope no_grad;
  check_layout(query, key, value, false, /*heads_second=*/true);
  const int64_t max_q = query.size(2), max_k = key.size(2);
  TP_CHECK(key.size(2) == value.size(2),
           "Key and Value must have the same sequence length");
  auto [out, lse, rng_state, unused, debug] =
      ::tensorplay::detail::redispatch__flash_attention_forward_function(
          ops::transpose(query, 1, 2), ops::transpose(key, 1, 2),
          ops::transpose(value, 1, 2), std::nullopt, std::nullopt, max_q, max_k,
          dropout_p, is_causal, return_debug_mask, scale, std::nullopt,
          std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  return {ops::transpose(out, 1, 2), lse, Tensor(), Tensor(), SymInt(max_q),
          SymInt(max_k), rng_state, unused, debug};
}

inline std::tuple<Tensor, Tensor, Tensor> sdpa_flash_backward(
    const Tensor& grad_out, const Tensor& query, const Tensor& key,
    const Tensor& value, const Tensor& out, const Tensor& logsumexp,
    const Tensor& cum_seq_q, const Tensor& cum_seq_k, int64_t max_q, int64_t max_k,
    double dropout_p, bool is_causal, const Tensor& philox_seed,
    const Tensor& philox_offset, std::optional<double> scale) {
  if (!grad_out.defined()) return {Tensor(), Tensor(), Tensor()};
  NoGradScope no_grad;
  auto [grad_q, grad_k, grad_v] =
      ::tensorplay::detail::redispatch__flash_attention_backward_function(
          ops::transpose(grad_out, 1, 2), ops::transpose(query, 1, 2),
          ops::transpose(key, 1, 2), ops::transpose(value, 1, 2),
          ops::transpose(out, 1, 2), logsumexp, cum_seq_q, cum_seq_k, max_q,
          max_k, dropout_p, is_causal, philox_seed, philox_offset, scale,
          std::nullopt, std::nullopt);
  return {ops::transpose(grad_q, 1, 2), ops::transpose(grad_k, 1, 2),
          ops::transpose(grad_v, 1, 2)};
}

inline std::tuple<Tensor, Tensor, Tensor, Tensor> sdpa_efficient(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const std::optional<Tensor>& attn_bias, bool compute_log_sumexp,
    double dropout_p, bool is_causal, std::optional<double> scale) {
  NoGradScope no_grad;
  check_layout(query, key, value, false, /*heads_second=*/true);
  auto [out, lse, seed, offset, max_q, max_k] =
      ::tensorplay::detail::redispatch__efficient_attention_forward_function(
          ops::transpose(query, 1, 2), ops::transpose(key, 1, 2),
          ops::transpose(value, 1, 2), attn_bias, std::nullopt, std::nullopt,
          std::nullopt, std::nullopt, dropout_p,
          is_causal ? kCausalFromTopLeft : kNoCustomMask, compute_log_sumexp,
          scale, std::nullopt, std::nullopt);
  (void)max_q, (void)max_k;
  return {ops::transpose(out, 1, 2), lse, seed, offset};
}

inline std::tuple<Tensor, Tensor, Tensor, Tensor> sdpa_efficient_backward(
    const Tensor& grad_out_, const Tensor& query, const Tensor& key,
    const Tensor& value, const Tensor& attn_bias, const Tensor& out,
    const Tensor& logsumexp, const Tensor& philox_seed, const Tensor& philox_offset,
    double dropout_p, const std::vector<bool>& grad_input_mask, bool is_causal,
    std::optional<double> scale) {
  if (!grad_out_.defined()) return {Tensor(), Tensor(), Tensor(), Tensor()};
  NoGradScope no_grad;
  std::optional<Tensor> bias;
  if (attn_bias.defined()) bias = attn_bias;
  const bool bias_grad = grad_input_mask.size() > 3 && grad_input_mask[3] && bias.has_value();
  auto [grad_q, grad_k, grad_v, grad_bias] =
      ::tensorplay::detail::redispatch__efficient_attention_backward_function(
          ops::transpose(grad_out_, 1, 2), ops::transpose(query, 1, 2),
          ops::transpose(key, 1, 2), ops::transpose(value, 1, 2), bias,
          ops::transpose(out, 1, 2), std::nullopt, std::nullopt, query.size(2),
          key.size(2), logsumexp, dropout_p, philox_seed, philox_offset,
          is_causal ? kCausalFromTopLeft : kNoCustomMask, bias_grad, scale,
          std::nullopt, std::nullopt, false);
  return {ops::transpose(grad_q, 1, 2), ops::transpose(grad_k, 1, 2),
          ops::transpose(grad_v, 1, 2), grad_bias};
}

inline CudnnForward sdpa_cudnn(const Tensor& query, const Tensor& key,
                               const Tensor& value,
                               const std::optional<Tensor>& attn_bias,
                               bool compute_log_sumexp, double dropout_p,
                               bool is_causal, bool return_debug_mask,
                               std::optional<double> scale) {
  NoGradScope no_grad;
  check_layout(query, key, value, false, /*heads_second=*/true);
  return ::tensorplay::detail::redispatch__cudnn_attention_forward_function(
      query, key, value, attn_bias, std::nullopt, std::nullopt, query.size(2),
      key.size(2), compute_log_sumexp, dropout_p, is_causal, return_debug_mask,
      scale, std::nullopt, std::nullopt);
}

inline std::tuple<Tensor, Tensor, Tensor> sdpa_cudnn_backward(
    const Tensor& grad_out, const Tensor& query, const Tensor& key,
    const Tensor& value, const Tensor& out, const Tensor& logsumexp,
    const Tensor& philox_seed, const Tensor& philox_offset, const Tensor& attn_bias,
    const Tensor& cum_seq_q, const Tensor& cum_seq_k, int64_t max_q, int64_t max_k,
    double dropout_p, bool is_causal, std::optional<double> scale) {
  NoGradScope no_grad;
  return ::tensorplay::detail::redispatch__cudnn_attention_backward_function(
      grad_out, query, key, value, out, logsumexp, philox_seed, philox_offset,
      attn_bias, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, is_causal, scale);
}

}  // namespace attention
}  // namespace composite
}  // namespace tensorplay
