// Composite kernels: isin (3 overloads) and is_nonzero.
// single-element truthiness check.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "TypePromotion.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cmath>
#include <cstdint>
#include <tuple>
#include <utility>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

void isin_dtype_check(DType dt) {
    if (dt == DType::Bool || isComplexType(dt)) {
        TP_THROW(NotImplementedError,
                 "isin is not supported for bool/complex dtypes");
    }
}

} // anonymous namespace

// Membership by one of two routes chosen on the test-set size: a linear scan
// whose working set stays at O(elements) for a handful of probes, and a
// sort-based join that never materializes the elements-by-test comparison
// matrix.  Both are built from dispatcher-visible primitives, so every
// backend that provides unique/sort/cat/index inherits them.
Tensor isin_tensor_tensor_native(const Tensor& elements,
                                 const Tensor& test_elements,
                                 bool assume_unique, bool invert) {
    isin_dtype_check(elements.dtype());
    isin_dtype_check(test_elements.dtype());
    if (elements.numel() == 0) {
        return ops::full_like(elements, Scalar(invert), DType::Bool);
    }

    const DType common = promoteTypes(elements.dtype(), test_elements.dtype());
    const Tensor elements_p =
        elements.dtype() == common ? elements : tpx::to(elements, common);
    const Tensor test_p =
        test_elements.dtype() == common ? test_elements : tpx::to(test_elements, common);
    const Tensor elements_flat = ops::reshape(elements_p, {-1});
    const Tensor test_flat = ops::reshape(test_p, {-1});

    // Below this bound the scan wins: the sort-based join would pay a full
    // sort to resolve a few probes.  The scan keeps one boolean work tensor
    // alive instead of the broadcast matrix of size elements x test.
    const int64_t scan_bound = static_cast<int64_t>(
        10.0 * std::pow(static_cast<double>(elements.numel()), 0.145));
    if (test_flat.numel() < scan_bound) {
        Tensor found = ops::full_like(elements_flat, Scalar(false), DType::Bool);
        for (int64_t j = 0; j < test_flat.numel(); ++j) {
            found = ops::logical_or(std::move(found),
                                    ops::eq(elements_flat, ops::select(test_flat, 0, j)));
        }
        if (invert) found = ops::logical_not(found);
        return ops::reshape(found, static_cast<std::vector<int64_t>>(elements.shape()));
    }

    // 1. Collapse both sides to their distinct values; the element inverse
    //    maps each original position back to its distinct value at the end.
    Tensor elements_distinct, unique_order;
    if (assume_unique) {
        elements_distinct = elements_flat;
    } else {
        Tensor inverse;
        std::tie(elements_distinct, inverse, std::ignore) =
            ops::unique(elements_flat, /*sorted=*/true, /*return_inverse=*/true,
                        /*return_counts=*/false);
        unique_order = std::move(inverse);
    }
    const Tensor test_distinct =
        assume_unique
            ? test_flat
            : std::get<0>(ops::unique(test_flat, /*sorted=*/true,
                                      /*return_inverse=*/false,
                                      /*return_counts=*/false));

    // 2. Stable sort of the concatenation.  Elements occupy the first slots
    //    of the input, so among equal values every element entry precedes
    //    its test entry.
    const int64_t n_elements = elements_distinct.numel();
    const Tensor all_values = ops::cat({elements_distinct, test_distinct}, 0);
    Tensor sorted_values, sorted_order;
    std::tie(sorted_values, sorted_order) = ops::sort(all_values, 0, false);
    const int64_t total = all_values.numel();

    // 3. One flag per sorted slot: true where the sequence holds the same
    //    value twice in a row, which under distinct inputs pairs an element
    //    entry with its test entry.  The trailing slot carries `invert` so
    //    values with no test match resolve to it.
    Tensor flags;
    if (total > 1) {
        const Tensor left = ops::slice(sorted_values, 0, 0, total - 1);
        const Tensor right = ops::slice(sorted_values, 0, 1, total);
        flags = ops::cat({invert ? ops::ne(right, left) : ops::eq(right, left),
                          ops::full({1}, Scalar(invert), DType::Bool,
                                     all_values.device())},
                         0);
    } else {
        flags = ops::full({1}, Scalar(invert), DType::Bool, all_values.device());
    }

    // 4. Undo the sort so each flag sits at its concatenated position.
    const Tensor ordered = ops::index_copy(
        ops::empty({total}, DType::Bool, all_values.device()), 0, sorted_order,
        flags);

    // 5. Restore the original element order and shape.
    Tensor flat_result = assume_unique
                             ? ops::slice(ordered, 0, 0, n_elements)
                             : ops::index(ordered, {unique_order});
    return ops::reshape(flat_result,
                        static_cast<std::vector<int64_t>>(elements.shape()));
}

Tensor isin_tensor_scalar_native(const Tensor& elements,
                                 const Scalar& test_element,
                                 bool /*assume_unique*/, bool invert) {
    isin_dtype_check(elements.dtype());
    Tensor result = invert ? ops::ne(elements, test_element)
                           : ops::eq(elements, test_element);
    return result;
}

Tensor isin_scalar_tensor_native(const Scalar& element,
                                 const Tensor& test_elements,
                                 bool /*assume_unique*/, bool invert) {
    isin_dtype_check(test_elements.dtype());
    // bool result.
    Tensor result = ops::any(ops::eq(test_elements, element), std::nullopt, false);
    if (invert) result = ops::logical_not(result);
    return result;
}

bool is_nonzero_native(const Tensor& self) {
    const int64_t n = self.numel();
    if (n == 0) {
        TP_THROW(RuntimeError,
                 "Boolean value of Tensor with no values is ambiguous");
    }
    if (n > 1) {
        TP_THROW(RuntimeError,
                 "Boolean value of Tensor with more than one value is ambiguous");
    }
    const Scalar s = self.item();
    if (s.isComplex()) {
        const auto c = s.to<complex<double>>();
        return c.real() != 0.0 || c.imag() != 0.0;
    }
    return s.toDouble() != 0.0;
}

TENSORPLAY_LIBRARY_IMPL(Composite, TensorCompareComposite) {
    m.impl("isin.Tensor_Tensor", isin_tensor_tensor_native);
    m.impl("isin.Tensor_Scalar", isin_tensor_scalar_native);
    m.impl("isin.Scalar_Tensor", isin_scalar_tensor_native);
    m.impl("is_nonzero", is_nonzero_native);
}

} // namespace composite
} // namespace tensorplay
