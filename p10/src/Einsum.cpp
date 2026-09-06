//
// through differentiable primitives (mul/sum/bmm/diagonal/movedim/permute),
// so autograd is derived automatically from those inner calls and no
// derivatives.yaml entry is needed.  The device-agnostic implementation is
// registered for both CPU and CUDA; every primitive dispatches on the
// operand's own device.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "GradMode.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace tensorplay {
namespace tpx {

namespace {

constexpr uint8_t kNumLetters = 'z' - 'a' + 1;   // per case: [A-Z] or [a-z]
constexpr uint8_t kTotalLabels = kNumLetters * 2;  // labels live in [A-Za-z]
constexpr uint8_t kEllipsis = kTotalLabels;        // code identifying "..."

uint8_t subscript_of(unsigned char label) {
    return std::isupper(label) ? label - 'A' : label - 'a' + kNumLetters;
}

// ---------------------------------------------------------------------------
// Native contraction-path planning.
//
// package is installed on the Python side; without it every pairwise step
// contracts strictly left to right, which can inflate an intermediate tensor
// exponentially (e.g. "abc,cde,bd,de->ab").  We plan natively instead: an
// exhaustive dynamic program while the operand count is small, then the
// opt-einsum-style greedy heuristic (minimise intermediate size first, total
// flops as tie-break).  Both emit the SSA-style path consumed by the
// contraction loop below: each pair indexes positions in the evolving operand
// list and every result takes a fresh slot at the end.
// ---------------------------------------------------------------------------

constexpr double kPlanCostCap = 1e300;

double plan_mask_cost(const std::vector<int64_t>& slot_sizes,
                      const std::vector<char>& mask) {
    double p = 1.0;
    for (size_t d = 0; d < mask.size(); ++d) {
        if (mask[d]) p *= static_cast<double>(std::max<int64_t>(slot_sizes[d], 1));
    }
    return std::min(p, kPlanCostCap);
}

// Walks the optimal binary tree recorded in `best_split` (subset -> its two
// halves) post-order and translates it into SSA path pairs.
std::vector<int64_t> plan_emit_path(
        size_t n, const std::vector<std::pair<uint32_t, uint32_t>>& best_split) {
    std::vector<int64_t> path;
    if (n < 2) return path;
    const size_t full = size_t{1} << n;
    std::vector<size_t> pos(full);   // subset -> current list position
    std::vector<uint32_t> alive;     // subsets in list order
    alive.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        pos[size_t{1} << i] = i;
        alive.push_back(static_cast<uint32_t>(size_t{1} << i));
    }
    std::function<void(uint32_t)> emit = [&](uint32_t s) {
        if (!(s & (s - 1))) return;  // single operand: a leaf
        const auto [a, b] = best_split[s];
        emit(a);
        emit(b);
        const size_t pa = pos[a];
        const size_t pb = pos[b];
        path.push_back(static_cast<int64_t>(pa));
        path.push_back(static_cast<int64_t>(pb));
        // Remove both entries and append the result -- exactly how the
        // contraction loop consumes a path.
        alive.erase(alive.begin() + static_cast<std::ptrdiff_t>(std::max(pa, pb)));
        alive.erase(alive.begin() + static_cast<std::ptrdiff_t>(std::min(pa, pb)));
        alive.push_back(s);
        for (size_t k = 0; k + 1 < alive.size(); ++k) pos[alive[k]] = k;
        pos[s] = alive.size() - 1;
    };
    emit(static_cast<uint32_t>(full - 1));
    return path;
}

// Exhaustive DP over operand subsets: dp[S] is the minimal accumulated
// elementwise cost of fully contracting S.  A dim survives S iff it appears
// in exactly one member (contraction dims) or is an output dim.
std::vector<int64_t> plan_exact(const std::vector<std::vector<char>>& occ,
                                const std::vector<int64_t>& slot_sizes,
                                int64_t out_num_dim) {
    const size_t n = occ.size();
    const size_t ndims = occ[0].size();
    const size_t full = size_t{1} << n;
    std::vector<std::vector<char>> survive(full, std::vector<char>(ndims, 0));
    for (size_t s = 1; s < full; ++s) {
        auto& sv = survive[s];  // reused as occurrence counts first
        for (size_t i = 0; i < n; ++i) {
            if (!(s >> i & 1)) continue;
            for (size_t d = 0; d < ndims; ++d) sv[d] += occ[i][d];
        }
        for (size_t d = 0; d < ndims; ++d) {
            const char cnt = sv[d];
            sv[d] = d < static_cast<size_t>(out_num_dim) ? cnt > 0 : cnt == 1;
        }
    }
    std::vector<double> dp(full, std::numeric_limits<double>::infinity());
    for (size_t i = 0; i < n; ++i) dp[size_t{1} << i] = 0.0;  // leaves cost nothing
    std::vector<std::pair<uint32_t, uint32_t>> best_split(full, {0, 0});
    for (size_t s = 3; s < full; ++s) {
        if (!(s & (s - 1))) continue;  // single-operand subset
        const size_t low = s & (~s + 1);
        for (size_t a = (s - 1) & s; a; a = (a - 1) & s) {
            if (!(a & low)) continue;  // deduplicate symmetric splits
            const size_t b = s ^ a;
            double inter = 1.0;
            for (size_t d = 0; d < ndims; ++d) {
                if (survive[a][d] || survive[b][d]) {
                    inter *= static_cast<double>(std::max<int64_t>(slot_sizes[d], 1));
                }
            }
            const double cost = dp[a] + dp[b] + std::min(inter, kPlanCostCap);
            if (cost < dp[s]) {
                dp[s] = cost;
                best_split[s] = {static_cast<uint32_t>(a), static_cast<uint32_t>(b)};
            }
        }
    }
    return plan_emit_path(n, best_split);
}

// Greedy over live operands: contract the pair producing the smallest
// intermediate tensor (flops as tie-break), following opt_einsum's default
// heuristic so large operand counts stay near-optimal.
std::vector<int64_t> plan_greedy(const std::vector<std::vector<char>>& occ,
                                 const std::vector<int64_t>& slot_sizes,
                                 int64_t out_num_dim) {
    struct Entry {
        std::vector<char> dims;
    };
    std::vector<Entry> live(occ.size());
    for (size_t i = 0; i < occ.size(); ++i) live[i].dims = occ[i];

    std::vector<int64_t> counts(slot_sizes.size(), 0);
    for (const auto& e : live) {
        for (size_t d = 0; d < e.dims.size(); ++d) counts[d] += e.dims[d] ? 1 : 0;
    }

    std::vector<int64_t> path;
    const size_t ndims = slot_sizes.size();
    while (live.size() > 1) {
        double best_mem = std::numeric_limits<double>::infinity();
        double best_flops = std::numeric_limits<double>::infinity();
        size_t bi = 0, bj = 0;
        std::vector<char> best_res;
        for (size_t i = 0; i < live.size(); ++i) {
            for (size_t j = i + 1; j < live.size(); ++j) {
                const auto& a = live[i].dims;
                const auto& b = live[j].dims;
                std::vector<char> res(ndims, 0);
                double flops = 1.0, mem = 1.0;
                for (size_t d = 0; d < ndims; ++d) {
                    const bool sa = a[d] != 0, sb = b[d] != 0;
                    bool keep = sa || sb;
                    if (d >= static_cast<size_t>(out_num_dim)) {
                        if ((sa && sb && counts[d] == 2) ||
                            ((sa != sb) && counts[d] == 1)) {
                            keep = false;  // consumed by this contraction
                        }
                    }
                    res[d] = keep ? 1 : 0;
                    if (sa || sb) flops *= static_cast<double>(std::max<int64_t>(slot_sizes[d], 1));
                    if (keep) mem *= static_cast<double>(std::max<int64_t>(slot_sizes[d], 1));
                }
                flops = std::min(flops, kPlanCostCap);
                mem = std::min(mem, kPlanCostCap);
                if (mem < best_mem || (mem == best_mem && flops < best_flops)) {
                    best_mem = mem;
                    best_flops = flops;
                    bi = i;
                    bj = j;
                    best_res = std::move(res);
                }
            }
        }
        path.push_back(static_cast<int64_t>(bi));
        path.push_back(static_cast<int64_t>(bj));
        const auto& a = live[bi].dims;
        const auto& b = live[bj].dims;
        for (size_t d = 0; d < ndims; ++d) {
            const int64_t removed =
                static_cast<int64_t>(a[d] != 0) + static_cast<int64_t>(b[d] != 0);
            counts[d] += -removed + static_cast<int64_t>(best_res[d] != 0);
        }
        live.erase(live.begin() + static_cast<std::ptrdiff_t>(bj));
        live.erase(live.begin() + static_cast<std::ptrdiff_t>(bi));
        live.push_back({std::move(best_res)});
    }
    return path;
}

// ---------------------------------------------------------------------------
// Two-operand fast paths.
//
// Equations without ellipsis, without repeated labels inside one operand and
// with strictly matching shared sizes reduce to a single BLAS call --
// mm / bmm / mv / dot / outer over permuted views plus an optional output
// permutation view -- skipping the align-and-bmm pipeline (~9 dispatched ops
// per pair).  Every precondition miss returns nullopt so the general
// ---------------------------------------------------------------------------

std::optional<Tensor> try_two_operand_fast(
        const std::string& equation, size_t arrow_pos,
        const std::vector<Tensor>& operands,
        const std::vector<std::vector<uint8_t>>& op_labels,
        const std::vector<int64_t>& label_count) {
    constexpr uint8_t NONE = 0, IN_A = 1, IN_B = 2, IN_BOTH = 3;
    if (operands.size() != 2) return std::nullopt;
    const Tensor& A = operands[0];
    const Tensor& B = operands[1];
    if (op_labels[0].empty() || op_labels[1].empty()) return std::nullopt;
    if (A.dtype() != B.dtype()) return std::nullopt;

    // No ellipsis and no intra-operand repeats (diagonals).
    for (int side = 0; side < 2; ++side) {
        std::array<char, kTotalLabels> seen{};
        for (const uint8_t s : op_labels[side]) {
            if (s == kEllipsis) return std::nullopt;
            if (seen[s]) return std::nullopt;
            seen[s] = 1;
        }
    }

    // Output labels, in order: explicit RHS, or implicit singles in ascending
    // subscript order.
    std::vector<uint8_t> out_seq;
    if (arrow_pos != std::string::npos) {
        for (const char rc : equation.substr(arrow_pos + 2)) {
            const unsigned char label = static_cast<unsigned char>(rc);
            if (label == ' ') continue;
            // '.' or any other stray character: let the general path report it.
            if (!std::isalpha(label)) return std::nullopt;
            out_seq.push_back(subscript_of(label));
        }
    } else {
        for (uint8_t l = 0; l < kTotalLabels; ++l) {
            if (label_count[l] == 1) out_seq.push_back(l);
        }
    }
    std::array<char, kTotalLabels> in_out{};
    for (const uint8_t s : out_seq) {
        if (in_out[s]) return std::nullopt;            // duplicated output label
        if (label_count[s] < 1 || label_count[s] > 2) return std::nullopt;
        in_out[s] = 1;
    }

    std::array<uint8_t, kTotalLabels> where{};
    for (int side = 0; side < 2; ++side) {
        for (const uint8_t s : op_labels[side]) where[s] |= 1u << side;
    }
    for (uint8_t l = 0; l < kTotalLabels; ++l) {
        if (where[l] == NONE || in_out[l]) continue;   // absent, free or batch
        if (where[l] != IN_BOTH) return std::nullopt;  // lone dim needing a pre-sum
    }

    // Strict sizes: any broadcast across the two operands -> general path.
    auto size_on = [&](int side, uint8_t l) -> int64_t {
        const auto& labels = op_labels[side];
        const auto& t = operands[side];
        for (size_t k = 0; k < labels.size(); ++k) {
            if (labels[k] == l) return t.size(static_cast<int64_t>(k));
        }
        return -1;
    };
    for (uint8_t l = 0; l < kTotalLabels; ++l) {
        if (where[l] == IN_BOTH && size_on(0, l) != size_on(1, l)) return std::nullopt;
    }

    // Partition: BO batch (shared+output), FA/FB free singles, C contracted.
    std::vector<uint8_t> bo_a, bo_b, fa, fb, cseq;
    for (const uint8_t s : op_labels[0]) {
        if (where[s] != IN_BOTH) continue;
        if (in_out[s]) bo_a.push_back(s); else cseq.push_back(s);
    }
    for (const uint8_t s : op_labels[1]) {
        if (where[s] == IN_BOTH && in_out[s]) bo_b.push_back(s);
    }
    if (bo_a != bo_b) return std::nullopt;             // batch order must align
    for (const uint8_t s : out_seq) {
        if (where[s] == IN_A) fa.push_back(s);
        else if (where[s] == IN_B) fb.push_back(s);
    }
    const bool has_batch = !bo_a.empty();
    if (cseq.empty() && has_batch) return std::nullopt;
    if (!cseq.empty() && fa.empty() && fb.empty() && has_batch) return std::nullopt;

    auto group_size = [&](const std::vector<uint8_t>& labels) {
        int64_t p = 1;
        for (const uint8_t s : labels) {
            p *= size_on(where[s] == IN_B ? 1 : 0, s);
        }
        return p;
    };
    const int64_t Bn = group_size(bo_a), M = group_size(fa);
    const int64_t K = group_size(cseq), N = group_size(fb);

    auto perm_of = [&](const std::vector<uint8_t>& side_labels,
                       const std::vector<uint8_t>& wanted) {
        std::array<int64_t, kTotalLabels> pos{};
        pos.fill(-1);
        for (size_t k = 0; k < side_labels.size(); ++k) pos[side_labels[k]] = static_cast<int64_t>(k);
        std::vector<int64_t> p;
        p.reserve(wanted.size());
        for (const uint8_t s : wanted) p.push_back(pos[s]);
        return p;
    };
    std::vector<uint8_t> want_a = bo_a;
    want_a.insert(want_a.end(), fa.begin(), fa.end());
    want_a.insert(want_a.end(), cseq.begin(), cseq.end());
    std::vector<uint8_t> want_b = bo_b;
    want_b.insert(want_b.end(), cseq.begin(), cseq.end());
    want_b.insert(want_b.end(), fb.begin(), fb.end());
    const Tensor ta = ops::permute(A, perm_of(op_labels[0], want_a));
    const Tensor tb = ops::permute(B, perm_of(op_labels[1], want_b));

    // Natural result order before applying the requested output permutation.
    std::vector<uint8_t> natural = bo_b;
    natural.insert(natural.end(), fa.begin(), fa.end());
    natural.insert(natural.end(), fb.begin(), fb.end());

    Tensor result;
    if (cseq.empty()) {  // pure outer product (no batch reaches here)
        result = ops::mm(ops::reshape(ta, {M, 1}), ops::reshape(tb, {1, N}));
    } else if (fa.empty() && fb.empty()) {
        return ops::dot(ops::reshape(ta, {K}), ops::reshape(tb, {K}));
    } else if (!fa.empty() && !fb.empty()) {
        if (has_batch) {
            result = ops::bmm(ops::reshape(ta, {Bn, M, K}), ops::reshape(tb, {Bn, K, N}));
        } else {
            result = ops::mm(ops::reshape(ta, {M, K}), ops::reshape(tb, {K, N}));
        }
    } else if (!fa.empty()) {  // matrix @ vector
        if (has_batch) {
            result = ops::bmm(ops::reshape(ta, {Bn, M, K}), ops::reshape(tb, {Bn, K, 1}));
        } else {
            result = ops::mv(ops::reshape(ta, {M, K}), ops::reshape(tb, {K}));
        }
    } else {                   // vector @ matrix: result = Bᵀ·v, mv takes (N,K)
        if (has_batch) {
            result = ops::bmm(ops::reshape(ta, {Bn, 1, K}), ops::reshape(tb, {Bn, K, N}));
        } else {
            // Flat (N,) in FB order; the assembly below reshapes/permutess it
            // to the requested output layout.
            result = ops::mv(ops::transpose(ops::reshape(tb, {K, N}), 0, 1),
                             ops::reshape(ta, {K}));
        }
    }

    // Assemble the requested output layout.
    std::vector<int64_t> nat_sizes;
    nat_sizes.reserve(natural.size());
    for (const uint8_t s : natural) nat_sizes.push_back(size_on(where[s] == IN_B ? 1 : 0, s));
    Tensor r = result.numel() == 1 && nat_sizes.empty()
                   ? result
                   : ops::reshape(result, nat_sizes);
    if (natural == out_seq) return r;
    std::array<int64_t, kTotalLabels> nat_pos{};
    nat_pos.fill(-1);
    for (size_t k = 0; k < natural.size(); ++k) nat_pos[natural[k]] = static_cast<int64_t>(k);
    std::vector<int64_t> perm;
    perm.reserve(out_seq.size());
    for (const uint8_t s : out_seq) perm.push_back(nat_pos[s]);
    return ops::permute(r, perm);
}


// sumproduct_pair computes `(left*right).sum(sum_dims)` by means of
// permutation and batch matrix multiplication; its main purpose is to provide
// a pairwise reduction for einsum.
Tensor sumproduct_pair(const Tensor& left_, const Tensor& right_,
                       const std::vector<int64_t>& sum_dims_, bool keepdim) {
    if (left_.dim() != right_.dim()) {
        TP_THROW(RuntimeError, "number of dimensions must match");
    }
    if (sum_dims_.empty()) return left_.mul(right_);

    const int64_t dim = left_.dim();
    std::vector<bool> sum_dims(dim, false);
    for (const auto i : sum_dims_) sum_dims[i] = true;

    // Dimensions that will be part of the output (i.e. not summed over) in
    // three groups: lro appear in left+right+output, lo: left+output,
    // ro: right+output.  The sizes are kept track of for reshaping.
    std::vector<int64_t> lro, lo, ro;
    int64_t lro_size = 1, lo_size = 1, ro_size = 1, sum_size = 1;
    Tensor left = left_;
    Tensor right = right_;
    for (int64_t i = 0; i < dim; ++i) {
        const bool sl = left.size(i) != 1;
        const bool sr = right.size(i) != 1;
        if (sum_dims[i]) {  // first dimensions that will be summed over after multiplication
            if (sl && sr) {  // dimensions nontrivially in both must be of the same size
                if (left.size(i) != right.size(i)) {
                    TP_THROW(RuntimeError, "non-broadcast dimensions must match");
                }
                sum_size *= left.size(i);
            } else if (sl) {  // only in one of left/right: sum right away
                left = left.sum({i}, true);
            } else if (sr) {
                right = right.sum({i}, true);
            }
        } else if (sl && sr) {  // dimensions in the output
            if (left.size(i) != right.size(i)) {
                TP_THROW(RuntimeError, "non-broadcast dimensions must match");
            }
            lro.push_back(i);
            lro_size *= left.size(i);
        } else if (sl) {  // dimensions appearing only once
            lo.push_back(i);
            lo_size *= left.size(i);
        } else {
            ro.push_back(i);
            ro_size *= right.size(i);
        }
    }

    // The pipeline is permute inputs -> reshape inputs -> bmm ->
    // reshape(view) output -> permute output.  By default the permuted output
    // is "lro, lo, 1-for-summed-dims, ro"; however if all dimensions from the
    // right operand appear before those from the left operand in memory we
    // can swap the operands so that bmm directly produces the natural order.
    const bool swap_lo_ro = !lo.empty() && !ro.empty() && ro.back() < lo.front();
    if (swap_lo_ro) {
        std::swap(left, right);
        std::swap(lo, ro);
        std::swap(lo_size, ro_size);
    }

    const int64_t out_num_dim =
        static_cast<int64_t>(lro.size() + lo.size() + sum_dims_.size() + ro.size());
    std::vector<int64_t> out_size;
    out_size.reserve(out_num_dim);
    for (const auto d : lro) out_size.push_back(left.size(d));
    for (const auto d : lo) out_size.push_back(left.size(d));
    for (const auto d : sum_dims_) { out_size.push_back(1); (void)d; }
    for (const auto d : ro) out_size.push_back(right.size(d));

    std::vector<int64_t> lpermutation(lro);
    lpermutation.insert(lpermutation.end(), lo.begin(), lo.end());
    lpermutation.insert(lpermutation.end(), sum_dims_.begin(), sum_dims_.end());
    lpermutation.insert(lpermutation.end(), ro.begin(), ro.end());

    std::vector<int64_t> rpermutation(lro);
    rpermutation.insert(rpermutation.end(), sum_dims_.begin(), sum_dims_.end());
    rpermutation.insert(rpermutation.end(), ro.begin(), ro.end());
    rpermutation.insert(rpermutation.end(), lo.begin(), lo.end());

    std::vector<int64_t> opermutation(out_num_dim, -1);
    int64_t i = 0;
    for (auto it = lro.cbegin(); it != lro.cend(); ++i, ++it) opermutation[*it] = i;
    for (auto it = lo.cbegin(); it != lo.cend(); ++i, ++it) opermutation[*it] = i;
    for (auto it = sum_dims_.cbegin(); it != sum_dims_.cend(); ++i, ++it) opermutation[*it] = i;
    for (auto it = ro.cbegin(); it != ro.cend(); ++i, ++it) opermutation[*it] = i;

    left = left.permute(lpermutation).reshape({lro_size, lo_size, sum_size});
    right = right.permute(rpermutation).reshape({lro_size, sum_size, ro_size});
    // Route through the dispatcher-level ops so the contraction stays inside
    // the autograd graph (CompositeImplicitAutograd semantics).  reshape()
    // tolerates non-contiguous intermediates where view() would reject them.
    Tensor result = ops::bmm(left, right);
    // Reshape the result so non-contiguous gradients from reordered
    // contraction paths are handled.
    result = ops::view(result, out_size).permute(opermutation);

    // Finally squeeze summed dimensions if desired.
    if (!keepdim) {
        std::vector<int64_t> sizes = static_cast<std::vector<int64_t>>(result.shape());
        for (int64_t j = dim - 1; j >= 0; --j) {
            if (sum_dims[j]) sizes.erase(sizes.begin() + j);
        }
        result = result.reshape(sizes);
    }
    return result;
}

std::vector<int64_t> normalize_tensordot_dims(
        const std::vector<int64_t>& dims, int64_t ndim, const char* name) {
    const int64_t dim_post_expr = ndim == 0 ? 1 : ndim;
    std::vector<int64_t> normalized;
    normalized.reserve(dims.size());
    std::vector<bool> seen(static_cast<size_t>(ndim), false);
    for (const int64_t dim : dims) {
        if (dim < -dim_post_expr || dim >= dim_post_expr) {
            TP_THROW(IndexError,
                     "tensordot: dimension ", dim,
                     " in ", name, " is out of range for a ", ndim,
                     "-D tensor");
        }
        const int64_t wrapped = dim < 0 ? dim + dim_post_expr : dim;
        if (wrapped >= ndim) {
            TP_THROW(IndexError,
                     "tensordot: dimension ", dim,
                     " in ", name, " is invalid for a ", ndim,
                     "-D tensor");
        }
        if (seen[static_cast<size_t>(wrapped)]) {
            TP_THROW(RuntimeError,
                     "tensordot: dimension ", wrapped,
                     " appears multiple times in ", name);
        }
        seen[static_cast<size_t>(wrapped)] = true;
        normalized.push_back(wrapped);
    }
    return normalized;
}

Tensor tensordot_kernel(const Tensor& input1, const Tensor& input2,
                        const std::vector<int64_t>& dims1_arg,
                        const std::vector<int64_t>& dims2_arg) {
    if (dims1_arg.size() != dims2_arg.size()) {
        TP_THROW(RuntimeError,
                 "tensordot: both dimension lists should have the same length");
    }
    if (input1.dtype() != input2.dtype()) {
        TP_THROW(RuntimeError, "tensordot: both inputs should have the same dtype");
    }
    if (input1.device() != input2.device()) {
        TP_THROW(DeviceMismatchError,
                 "tensordot: both inputs must be on the same device");
    }

    const std::vector<int64_t> dims1 =
        normalize_tensordot_dims(dims1_arg, input1.dim(), "dims_self");
    const std::vector<int64_t> dims2 =
        normalize_tensordot_dims(dims2_arg, input2.dim(), "dims_other");

    int64_t contraction_size = 1;
    Tensor t1 = input1;
    Tensor t2 = input2;
    for (size_t i = 0; i < dims1.size(); ++i) {
        const int64_t size1 = input1.size(dims1[i]);
        const int64_t size2 = input2.size(dims2[i]);
        if (size2 == 1) {
            t1 = t1.sum({dims1[i]}, true, t1.dtype());
        } else if (size1 == 1) {
            t2 = t2.sum({dims2[i]}, true, t2.dtype());
        } else {
            if (size1 != size2) {
                TP_THROW(RuntimeError,
                         "tensordot: contracted dimensions need to match, but "
                         "first has size ", size1, " in dim ", dims1[i],
                         " and second has size ", size2, " in dim ", dims2[i]);
            }
            contraction_size *= size1;
        }
    }

    std::vector<bool> contracted1(static_cast<size_t>(input1.dim()), false);
    std::vector<bool> contracted2(static_cast<size_t>(input2.dim()), false);
    for (const int64_t dim : dims1) contracted1[static_cast<size_t>(dim)] = true;
    for (const int64_t dim : dims2) contracted2[static_cast<size_t>(dim)] = true;

    std::vector<int64_t> permutation1;
    std::vector<int64_t> permutation2;
    std::vector<int64_t> result_sizes;
    permutation1.reserve(static_cast<size_t>(input1.dim()));
    permutation2.reserve(static_cast<size_t>(input2.dim()));
    result_sizes.reserve(static_cast<size_t>(input1.dim() + input2.dim()));

    int64_t free_size1 = 1;
    int64_t free_size2 = 1;
    for (int64_t dim = 0; dim < input1.dim(); ++dim) {
        if (!contracted1[static_cast<size_t>(dim)]) {
            permutation1.push_back(dim);
            free_size1 *= t1.size(dim);
            result_sizes.push_back(t1.size(dim));
        }
    }
    permutation1.insert(permutation1.end(), dims1.begin(), dims1.end());
    permutation2.insert(permutation2.end(), dims2.begin(), dims2.end());
    for (int64_t dim = 0; dim < input2.dim(); ++dim) {
        if (!contracted2[static_cast<size_t>(dim)]) {
            permutation2.push_back(dim);
            free_size2 *= t2.size(dim);
            result_sizes.push_back(t2.size(dim));
        }
    }

    if (free_size1 != 1 || free_size2 != 1) {
        t1 = t1.permute(permutation1).reshape({free_size1, contraction_size});
        t2 = t2.permute(permutation2).reshape({contraction_size, free_size2});
        return ops::reshape(ops::mm(t1, t2), result_sizes);
    }

    t1 = t1.permute(permutation1);
    t2 = t2.permute(permutation2);
    if (t1.is_contiguous() && t2.is_contiguous()) {
        return ops::reshape(
            ops::dot(t1.reshape({-1}), t2.reshape({-1})), result_sizes);
    }
    return ops::reshape(
        ops::sum(ops::mul(t1.squeeze(), t2.squeeze()), t1.dtype()),
        result_sizes);
}

Tensor& tensordot_out_kernel(const Tensor& input1, const Tensor& input2,
                             const std::vector<int64_t>& dims1,
                             const std::vector<int64_t>& dims2, Tensor& out) {
    if (out.device() != input1.device() || input1.device() != input2.device()) {
        TP_THROW(DeviceMismatchError,
                 "tensordot: all tensors must be on the same device");
    }
    if (out.dtype() != input1.dtype()) {
        TP_THROW(RuntimeError,
                 "tensordot: output dtype must match the input dtype");
    }
    if (GradMode::is_enabled() &&
        (input1.requires_grad() || input2.requires_grad() || out.requires_grad())) {
        TP_THROW(RuntimeError,
                 "tensordot: out variants do not support automatic differentiation "
                 "when an argument requires grad");
    }

    Tensor result = tensordot_kernel(input1, input2, dims1, dims2);
    if (out.shape() == result.shape()) {
        out.copy_(result);
    } else {
        out.unsafeGetTensorImpl()->copy_metadata_from(*result.unsafeGetTensorImpl());
    }
    return out;
}

} // anonymous namespace

Tensor einsum_kernel(const std::string& equation,
                     const std::vector<Tensor>& operands,
                     const std::vector<int64_t>& path_arg) {
    if (operands.empty()) {
        TP_THROW(RuntimeError, "einsum(): must provide at least one operand");
    }
    const size_t num_ops = operands.size();

    const bool has_path = !path_arg.empty();
    if (has_path) {
        const int64_t path_size = num_ops == 1 ? 1 : static_cast<int64_t>(num_ops - 1) * 2;
        if (static_cast<int64_t>(path_arg.size()) != path_size) {
            TP_THROW(RuntimeError, "einsum(): expected contraction path given in path parameter to have size ",
                     path_size, " but got ", path_arg.size());
        }
    }

    // Labels must be in range [A-Za-z]
    constexpr uint8_t NUM_OF_LETTERS = 'z' - 'a' + 1;
    constexpr uint8_t TOTAL_LABELS = NUM_OF_LETTERS * 2;
    // Code used to identify ELLIPSIS ("...")
    constexpr uint8_t ELLIPSIS = TOTAL_LABELS;

    auto label_to_subscript = [=](unsigned char label) -> uint8_t {
        return std::isupper(label) ? label - 'A' : label - 'a' + NUM_OF_LETTERS;
    };
    auto subscript_to_label = [=](uint8_t s) -> char {
        return s < NUM_OF_LETTERS ? static_cast<char>(s + 'A')
                                  : static_cast<char>(s + 'a' - NUM_OF_LETTERS);
    };

    // Find arrow (->) to split equation into lhs and rhs
    const size_t arrow_pos = equation.find("->");
    const std::string lhs = equation.substr(0, arrow_pos);

    // Convert labels for input operands into an index in [0, 52) and store
    // them in op_labels for each operand along with ELLIPSIS if present.
    std::vector<std::vector<uint8_t>> op_labels(num_ops);
    bool ell_in_input = false;
    size_t curr_op = 0;
    for (size_t i = 0; i < lhs.length(); ++i) {
        const unsigned char label = static_cast<unsigned char>(lhs[i]);
        switch (label) {
            case ' ':
                break;
            case '.': {
                if (ell_in_input) {
                    TP_THROW(RuntimeError, "einsum(): found '.' for operand ", curr_op,
                             " for which an ellipsis was already found");
                }
                bool ok = i + 2 < lhs.length() && lhs[++i] == '.' && lhs[++i] == '.';
                if (!ok) {
                    TP_THROW(RuntimeError, "einsum(): found '.' for operand ", curr_op,
                             " that is not part of any ellipsis");
                }
                op_labels[curr_op].push_back(ELLIPSIS);
                ell_in_input = true;
                break;
            }
            case ',':
                ++curr_op;
                if (curr_op >= num_ops) {
                    TP_THROW(RuntimeError, "einsum(): fewer operands were provided than specified in the equation");
                }
                ell_in_input = false;
                break;
            default:
                if (!std::isalpha(label)) {
                    TP_THROW(RuntimeError, "einsum(): invalid subscript given at index ", i,
                             " in the equation string, subscripts must be in [a-zA-Z]");
                }
                op_labels[curr_op].push_back(label_to_subscript(label));
        }
    }
    if (curr_op != num_ops - 1) {
        TP_THROW(RuntimeError, "einsum(): more operands were provided than specified in the equation");
    }

    std::vector<int64_t> label_count(TOTAL_LABELS, 0);

    // The maximum number of dimensions covered by any ellipsis, needed when
    // unsqueezing missing dimensions from operands to permute and broadcast.
    int64_t ell_num_dim = 0;
    for (size_t i = 0; i < num_ops; ++i) {
        const auto& operand = operands[i];
        const auto& labels = op_labels[i];
        const int64_t ndims = operand.dim();
        int64_t nlabels = static_cast<int64_t>(labels.size());
        bool has_ellipsis = false;
        for (const auto label : labels) {
            if (label == ELLIPSIS) {
                --nlabels;
                has_ellipsis = true;
                ell_num_dim = std::max(ell_num_dim, ndims - nlabels);
            } else {
                ++label_count[label];
            }
        }
        const bool ok = has_ellipsis ? nlabels <= ndims : nlabels == ndims;
        if (!ok) {
            if (has_ellipsis) {
                TP_THROW(RuntimeError, "einsum(): the number of subscripts in the equation (",
                         nlabels, ") is more than the number of dimensions (", ndims,
                         ") for operand ", i);
            } else {
                TP_THROW(RuntimeError, "einsum(): the number of subscripts in the equation (",
                         nlabels, ") does not match the number of dimensions (", ndims,
                         ") for operand ", i, " and no ellipsis was given");
            }
        }
    }

    // Two-operand equations without ellipsis or diagonals reduce to a single
    // BLAS call; anything unusual falls through to the general machinery
    if (auto fast = try_two_operand_fast(equation, arrow_pos, operands,
                                         op_labels, label_count)) {
        return *fast;
    }

    // Map every label to an index in the aligned permuted shape.
    std::vector<int64_t> label_perm_index(TOTAL_LABELS, -1);
    int64_t perm_index = 0;
    int64_t ell_index = 0;
    bool ell_in_output = false;

    if (arrow_pos == std::string::npos) {
        // Implicit output is ellipsis (...) + labels seen only once.
        perm_index = ell_num_dim;
        ell_in_output = true;
        for (uint8_t label = 0; label < TOTAL_LABELS; ++label) {
            if (label_count[label] == 1) label_perm_index[label] = perm_index++;
        }
    } else {
        const std::string rhs = equation.substr(arrow_pos + 2);
        for (size_t i = 0; i < rhs.length(); ++i) {
            const unsigned char label = static_cast<unsigned char>(rhs[i]);
            switch (label) {
                case ' ':
                    break;
                case '.': {
                    if (ell_in_output) {
                        TP_THROW(RuntimeError, "einsum(): found '.' for output but an ellipsis (...) was already found");
                    }
                    bool ok = i + 2 < rhs.length() && rhs[++i] == '.' && rhs[++i] == '.';
                    if (!ok) {
                        TP_THROW(RuntimeError, "einsum(): found '.' for output that is not part of any ellipsis (...)");
                    }
                    ell_index = perm_index;
                    perm_index += ell_num_dim;
                    ell_in_output = true;
                    break;
                }
                default: {
                    if (!std::isalpha(label)) {
                        TP_THROW(RuntimeError, "einsum(): invalid subscript given at index ",
                                 lhs.size() + 2 + i, " in the equation string, subscripts must be in [a-zA-Z]");
                    }
                    const auto index = label_to_subscript(label);
                    if (!(label_count[index] > 0 && label_perm_index[index] == -1)) {
                        if (label_perm_index[index] > -1) {
                            TP_THROW(RuntimeError, "einsum(): output subscript ", subscript_to_label(index),
                                     " appears more than once in the output");
                        } else {
                            TP_THROW(RuntimeError, "einsum(): output subscript ", subscript_to_label(index),
                                     " does not appear in the equation for any input operand");
                        }
                    }
                    label_perm_index[index] = perm_index++;
                }
            }
        }
    }

    // Save number of dimensions in output before adding contraction dims.
    const int64_t out_num_dim = perm_index;

    // If ellipsis is not part of the output, add it to contraction dims.
    if (!ell_in_output) {
        ell_index = perm_index;
        perm_index += ell_num_dim;
    }

    // Add contraction labels (labels not present in output).
    for (uint8_t label = 0; label < TOTAL_LABELS; ++label) {
        if (label_count[label] > 0 && label_perm_index[label] == -1) {
            label_perm_index[label] = perm_index++;
        }
    }

    // Check sizes, take diagonals for repeated labels, unsqueeze missing
    // dimensions so all operands share the aligned layout, then permute.
    std::vector<int64_t> label_size(TOTAL_LABELS, 1);
    std::vector<int64_t> ell_sizes(ell_num_dim, 1);
    std::vector<int64_t> dim_counts(perm_index, 0);
    std::deque<Tensor> ops_stack;
    for (size_t i = 0; i < num_ops; ++i) {
        Tensor op = operands[i];
        std::vector<int64_t> permutation(perm_index, -1);
        int64_t dim = 0;
        for (const auto s : op_labels[i]) {
            if (s == ELLIPSIS) {
                const int64_t ndim = op.dim() - (static_cast<int64_t>(op_labels[i].size()) - 1);
                for (int64_t j = ell_num_dim - ndim; j < ell_num_dim; ++j) {
                    if (op.size(dim) != 1) {
                        if (ell_sizes[j] != 1 && ell_sizes[j] != op.size(dim)) {
                            TP_THROW(RuntimeError, "einsum(): dimension ", dim,
                                     " covered by ellipsis in operand ", i, "has size ", op.size(dim),
                                     " which does not broadcast with previously seen ellipsis with size ",
                                     ell_sizes[j], " for the respective dimension");
                        }
                        ell_sizes[j] = op.size(dim);
                        ++dim_counts[ell_index + j];
                    }
                    permutation[ell_index + j] = dim++;
                }
            } else if (permutation[label_perm_index[s]] == -1) {
                if (op.size(dim) != 1) {
                    if (label_size[s] != 1 && label_size[s] != op.size(dim)) {
                        TP_THROW(RuntimeError, "einsum(): subscript ", subscript_to_label(s),
                                 " has size ", op.size(dim), " for operand ", i,
                                 " which does not broadcast with previously seen size ", label_size[s]);
                    }
                    label_size[s] = op.size(dim);
                    ++dim_counts[label_perm_index[s]];
                }
                permutation[label_perm_index[s]] = dim++;
            } else {
                // Repeated label, take diagonal.
                const auto prev_dim = permutation[label_perm_index[s]];
                if (op.size(dim) != op.size(prev_dim)) {
                    TP_THROW(RuntimeError, "einsum(): subscript ", subscript_to_label(s),
                             " is repeated for operand ", i, " but the sizes don't match, ",
                             op.size(dim), " != ", op.size(prev_dim));
                }
                op = op.diagonal(0, prev_dim, dim).movedim({-1}, {prev_dim});
            }
        }

        // Add dimensions for missing labels.
        for (int64_t k = 0; k < perm_index; ++k) {
            if (permutation[k] == -1) {
                op = op.unsqueeze(dim);
                permutation[k] = dim++;
            }
        }
        ops_stack.emplace_back(op.permute(permutation));
    }

    // Contract.  Without a caller-supplied path, plan the contraction order
    // left-to-right fallback, which can blow up intermediate sizes.
    std::vector<int64_t> contract_path(path_arg);
    bool use_path = has_path;
    if (!use_path && num_ops > 2) {
        std::vector<int64_t> slot_sizes(perm_index, 0);
        std::vector<std::vector<char>> occ(num_ops, std::vector<char>(perm_index, 0));
        for (size_t i = 0; i < num_ops; ++i) {
            for (int64_t d = 0; d < perm_index; ++d) {
                const int64_t sz = ops_stack[i].size(d);
                occ[i][d] = sz != 1 ? 1 : 0;
                slot_sizes[d] = std::max(slot_sizes[d], sz);
            }
        }
        contract_path = num_ops <= 8 ? plan_exact(occ, slot_sizes, out_num_dim)
                                     : plan_greedy(occ, slot_sizes, out_num_dim);
        use_path = true;
    }
    size_t it = 0;
    while (ops_stack.size() > 1) {
        int64_t pi = 0;
        int64_t pj = 1;
        if (use_path) {
            pi = contract_path[it++];
            pj = contract_path[it++];
            if (pj < pi) std::swap(pi, pj);
            if (pi == pj || pi < 0 || pj >= static_cast<int64_t>(ops_stack.size())) {
                TP_THROW(RuntimeError, "einsum(): invalid contraction (", pi, ", ", pj,
                         pi == pj ? ") cannot contract an operand with itself"
                                  : ") operand index is out of bounds");
            }
        }

        Tensor a = ops_stack[pi];
        Tensor b = ops_stack[pj];
        ops_stack.erase(ops_stack.begin() + pj);
        ops_stack.erase(ops_stack.begin() + pi);

        // Collect dimensions that can be summed now.
        std::vector<int64_t> sum_dims;
        std::vector<int64_t> a_dims_to_sum;
        std::vector<int64_t> b_dims_to_sum;
        for (int64_t d = out_num_dim; d < perm_index; ++d) {
            const bool sa = a.size(d) != 1;
            const bool sb = b.size(d) != 1;
            if (sa && sb) {
                if (a.size(d) != b.size(d)) {
                    TP_THROW(RuntimeError, "non-broadcast dimensions must match");
                }
                if (--dim_counts[d] == 1) {
                    sum_dims.push_back(d);
                    dim_counts[d] = 0;
                }
            } else if (dim_counts[d] == 1) {
                if (sa) {
                    a_dims_to_sum.push_back(d);
                    dim_counts[d] = 0;
                } else if (sb) {
                    b_dims_to_sum.push_back(d);
                    dim_counts[d] = 0;
                }
            }
        }

        // Sum multiple dims at a time to minimize kernel calls to sum.
        if (!a_dims_to_sum.empty()) a = a.sum(a_dims_to_sum, true);
        if (!b_dims_to_sum.empty()) b = b.sum(b_dims_to_sum, true);

        Tensor pair = sumproduct_pair(a, b, sum_dims, true);
        if (use_path) {
            ops_stack.emplace_back(pair);
        } else {
            ops_stack.emplace_front(pair);
        }
    }

    // Sum out remaining contraction dims.
    if (perm_index - out_num_dim > 0) {
        if (num_ops > 1) {
            // All contraction dims are already size 1 after contraction.
            std::vector<int64_t> sizes = static_cast<std::vector<int64_t>>(ops_stack[0].shape());
            for (int64_t d = perm_index - 1; d >= out_num_dim; --d) {
                sizes.erase(sizes.begin() + d);
            }
            // reshape = zero-copy view whenever the layout permits (always
            // true for left-to-right); a reordered contraction path can leave
            // the final intermediate non-viewable, where it materializes.
            return ops::reshape(ops_stack[0], sizes);
        } else {
            std::vector<int64_t> sum_dims(perm_index - out_num_dim);
            for (int64_t k = 0; k < perm_index - out_num_dim; ++k) sum_dims[k] = out_num_dim + k;
            return ops_stack[0].sum(sum_dims);
        }
    }

    return ops_stack[0];
}

TENSORPLAY_LIBRARY_IMPL(CPU, EinsumKernels) {
    m.impl("einsum", einsum_kernel);
    m.impl("tensordot", tensordot_kernel);
    m.impl("tensordot.out", tensordot_out_kernel);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, EinsumKernelsCUDA) {
    m.impl("einsum", einsum_kernel);
    m.impl("tensordot", tensordot_kernel);
    m.impl("tensordot.out", tensordot_out_kernel);
}

} // namespace tpx
} // namespace tensorplay
