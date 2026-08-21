#include "TensorIterator.h"
#include "TensorIteratorInternal.h"
#include "Parallel.h"
#include "irange.h"
#include <algorithm>
#include <tuple>

/// Contains the implementation of parallel reductions in TensorIterator.

namespace tensorplay {

using loop2d_t = TensorIteratorBase::loop2d_t;

static bool use_two_pass_reduction(TensorIteratorBase& iter);
static void two_pass_reduction(TensorIteratorBase& iter, loop2d_t loop);
static void parallel_dim_reduction(TensorIteratorBase& iter, loop2d_t loop);

void TensorIteratorBase::parallel_reduce(loop2d_t loop) {
  TP_CHECK(
      ntensors() == 2,
      "parallel_reduce only supports one input and one output");
  int64_t numel = this->numel();
  if (numel < parallel::GRAIN_SIZE || parallel::get_num_threads() == 1 ||
      parallel::in_parallel_region()) {
    serial_for_each(loop, {0, numel});
  } else if (use_two_pass_reduction(*this)) {
    two_pass_reduction(*this, loop);
  } else {
    parallel_dim_reduction(*this, loop);
  }
}

static bool use_two_pass_reduction(TensorIteratorBase& iter) {
  return iter.output(0).numel() == 1;
}

static void two_pass_reduction(TensorIteratorBase& iter, loop2d_t loop) {
  const int max_threads = parallel::get_num_threads();

  const auto& dst = iter.output(0);
  auto unsqueezed = dst.unsqueeze(0);
  DimVector buffer_shape = unsqueezed.shape();
  buffer_shape[0] = max_threads;
  auto buffer = Tensor::empty(buffer_shape, dst.dtype(), dst.device());
  // Fill with the identity. use_two_pass_reduction guarantees a single output
  // element, and iter.output_base() was pre-filled with the identity by the
  // caller, so the output's value IS the identity.
  buffer.fill_(dst.item());

  auto buffer_stride = buffer.strides()[0] * buffer.itemsize();
  auto buffer_0 = buffer.select(0, 0);
  auto first_reduce = TensorIterator::reduce_op(buffer_0, iter.input(0));

  parallel::parallel_for(
      0, iter.numel(), parallel::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        const auto thread_num = parallel::get_thread_num();
        auto shape = first_reduce.shape();
        auto strides = first_reduce.get_strides();

        // Bump output ptr so each thread has its own output slice
        auto base_ptrs = first_reduce.get_base_ptrs();
        base_ptrs[0] += buffer_stride * thread_num;

        internal::serial_for_each(
            shape,
            strides,
            base_ptrs.data(),
            base_ptrs.size(),
            loop,
            {begin, end});
      });

  auto final_reduce = TensorIterator::reduce_op(unsqueezed, buffer);
  final_reduce.for_each(loop);
}

/// Chooses a dimension over which to parallelize. Prefers the outer-most
/// dimension that's larger than the number of available threads.
static int find_split_dim(TensorIteratorBase& iter) {
  int num_threads = parallel::get_num_threads();
  auto shape = iter.shape();

  // start with the outer-most dimension
  int best_dim = iter.ndim() - 1;
  for (int dim = best_dim; dim >= 0 && !iter.is_dim_reduced(dim); dim--) {
    if (shape[dim] >= num_threads) {
      return dim;
    } else if (shape[dim] > shape[best_dim]) {
      best_dim = dim;
    }
  }

  TP_CHECK(!iter.is_dim_reduced(best_dim), "no split dim found");
  return best_dim;
}

static std::tuple<int64_t, int64_t> round_columns(
    TensorIteratorBase& iter,
    int dim,
    int multiple,
    int64_t begin,
    int64_t end) {
  begin = begin - (begin % multiple);
  if (end != iter.shape()[dim]) {
    // only round the 'end' column down if it's not the final column
    end = end - (end % multiple);
  }
  return std::make_tuple(begin, end);
}

static void parallel_dim_reduction(TensorIteratorBase& iter, loop2d_t loop) {
  TP_CHECK(iter.ndim() >= 1, "at least one dim required");
  int dim = find_split_dim(iter);
  int64_t cols = iter.shape()[dim];
  int element_size = iter.element_size(/*arg=*/1);

  bool should_round_columns = iter.strides(1)[dim] == element_size;
  parallel::parallel_for(0, cols, 1, [&](int64_t begin, int64_t end) {
    if (should_round_columns) {
      // round columns to multiples of 128 bytes if adjacent columns are
      // contiguous in memory.
      int64_t cols_per_128_bytes = 128 / element_size;
      std::tie(begin, end) =
          round_columns(iter, dim, cols_per_128_bytes, begin, end);
    }
    if (begin == end) {
      return;
    }
    auto sub_iter = TensorIterator(iter);
    sub_iter.narrow(dim, begin, end - begin);
    sub_iter.for_each(loop);
  });
}

void TensorIteratorBase::foreach_reduced_elt(
    loop_subiter_t loop,
    bool parallelize) {
  TP_CHECK(ninputs() == 1, "one input required");
  TP_CHECK(noutputs() >= 1, "at least one output required");

  auto shape = this->shape();
  if (output(0).numel() == 0) {
    return;
  }
  if (output(0).numel() == 1) {
    loop(*this);
  } else if (
      numel() < parallel::GRAIN_SIZE || parallel::get_num_threads() == 1 ||
      parallel::in_parallel_region() || !parallelize) {
    auto reduce_dims = num_reduce_dims();

    DimVector non_reduced_shape(
        shape.begin() + (shape.size() - reduce_dims), shape.end());

    int64_t non_reduced_numel = 1;
    for (const auto i : non_reduced_shape) {
      non_reduced_numel *= i;
    }
    DimCounter dims{non_reduced_shape, {0, non_reduced_numel}};
    while (!dims.is_done()) {
      TensorIterator reduced = *this;
      reduced.select_all_keeping_dim(reduce_dims, dims.values);
      loop(reduced);
      dims.increment({1, 1});
    }
  } else {
    int dim = find_split_dim(*this);
    int64_t cols = shape[dim];
    parallel::parallel_for(0, cols, 1, [&](int64_t begin, int64_t end) {
      if (begin == end) {
        return;
      }

      TensorIterator sub_iter(*this);

      sub_iter.narrow(dim, begin, end - begin);
      // On some broken setups, `#ifdef _OPENMP` is true,
      // and `get_num_threads` returns > 1, but
      // `#pragma omp parallel` is ignored.
      // There is no API to check for this, so we need to explicitly
      // stop trying to parallelize if we've already gotten here.
      //
      // (If we are on one of those broken setups, we will
      //  only have one thread here, and end - begin == cols.)
      sub_iter.foreach_reduced_elt(loop, false);
    });
  }
}

SplitUntil32Bit::iterator::iterator(const TensorIteratorBase& iter) {
  vec.emplace_back(new TensorIterator(iter));
  vec.emplace_back(nullptr); // ++ first pops the last element
  ++(*this);
}

SplitUntil32Bit::iterator& SplitUntil32Bit::iterator::operator++() {
  vec.pop_back();
  while (!vec.empty() && !vec.back()->can_use_32bit_indexing()) {
    auto& iter = *vec.back();
    auto split_dim = iter.get_dim_to_split();
    vec.emplace_back(iter.split(split_dim));
  }
  return *this;
}

TensorIterator& SplitUntil32Bit::iterator::operator*() const {
  return *vec.back();
}

SplitUntil32Bit::iterator SplitUntil32Bit::begin() const {
  return SplitUntil32Bit::iterator(iter);
}

SplitUntil32Bit::iterator SplitUntil32Bit::end() const {
  return SplitUntil32Bit::iterator();
}

DimCounter::DimCounter(const DimVector& shape, Range range)
  : shape(shape)
  , range(range)
  , values(shape.size())
  , offset(range.begin) {
  std::fill(values.begin(), values.end(), 0);
  if (range.begin == 0) {
    return;
  }

  int64_t linear_offset = range.begin;
  auto ndim = values.size();
  for (const auto dim : irange(ndim)) {
    int64_t size = shape[dim];
    if (size > 0) {
      values[dim] = linear_offset % size;
      linear_offset /= size;
    }
  }
  TP_CHECK(linear_offset == 0, "invalid range begin");
}

bool DimCounter::is_done() const {
  return offset >= range.end;
}

void DimCounter::increment(const std::array<int64_t, 2>& step) {
  offset += step[0] * step[1];
  auto ndim = values.size();
  int64_t overflow = step[0];
  size_t i = 0;
  if (step[1] != 1) {
    TP_CHECK(step[0] == shape[0] && values[0] == 0, "invalid step");
    i = 1;
    overflow = step[1];
  }
  for (; i < ndim && overflow > 0; i++) {
    auto size = shape[i];
    auto prev = values[i];
    auto value = prev + overflow;
    if (value >= size) {
      overflow = 1;
      value -= size;
      TP_CHECK(value < size, "value overflow");
    } else {
      overflow = 0;
    }
    values[i] = static_cast<int64_t>(value);
  }
  TP_CHECK(overflow == 0 || overflow == 1, "invalid overflow");
}

std::array<int64_t, 2> DimCounter::max_2d_step() const {
  int64_t step0 = std::min(shape[0] - values[0], range.end - offset);
  int64_t step1 = 1;
  if (!shape.empty() && step0 == shape[0]) {
    step1 = std::min(shape[1] - values[1], (range.end - offset) / shape[0]);
  }
  return {step0, step1};
}

} // namespace tensorplay