// Native einsum: faithful port of ATen's implementation
// (aten/src/ATen/native/Linear.cpp, einsum + sumproduct_pair).
//
// Like torch's CompositeImplicitAutograd operator, the kernel is expressed
// through differentiable primitives (mul/sum/bmm/diagonal/movedim/permute),
// so autograd is derived automatically from those inner calls and no
// derivatives.yaml entry is needed.  The device-agnostic implementation is
// registered for both CPU and CUDA; every primitive dispatches on the
// operand's own device.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace tensorplay {
namespace tpx {

namespace {

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
    Tensor result = left.bmm(right);
    result = result.view(out_size).permute(opermutation);

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

    // Contract.  Without a path, reduce from left to right.
    std::vector<int64_t> contract_path(path_arg);
    size_t it = 0;
    while (ops_stack.size() > 1) {
        int64_t pi = 0;
        int64_t pj = 1;
        if (has_path) {
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
        if (has_path) {
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
            return ops_stack[0].view(sizes);
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
}

TENSORPLAY_LIBRARY_IMPL(CUDA, EinsumKernelsCUDA) {
    m.impl("einsum", einsum_kernel);
}

} // namespace tpx
} // namespace tensorplay
