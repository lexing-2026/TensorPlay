#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"

#include <vector>

namespace tensorplay {
namespace mps {

// ---------------------------------------------------------------------------
// constant_pad_nd
//
// Negative padding crops the input. Positive padding fills an output
// canvas and copies the retained input into its interior.
// ---------------------------------------------------------------------------

Tensor constant_pad_nd_kernel(const Tensor& self,
                               const std::vector<int64_t>& pad,
                               const Scalar& value) {
  if (!self.device().is_mps()) {
    TP_THROW(RuntimeError, "constant_pad_nd(): input must be on MPS");
  }
  const std::vector<int64_t> input_sizes =
      static_cast<std::vector<int64_t>>(self.shape());
  const int64_t l_inp = self.dim();
  const int64_t l_pad = static_cast<int64_t>(pad.size()) / 2;

  if (pad.size() % 2 != 0) {
    TP_THROW(ValueError, "Length of pad must be even but instead it equals ",
             pad.size());
  }
  if (l_inp < l_pad) {
    TP_THROW(ValueError,
             "Length of pad should be no more than twice the number of "
             "dimensions of the input. Pad length is ",
             pad.size(), " while the input has ", l_inp, " dimensions.");
  }

  // Output shape: leading dims stay, the last l_pad dims get the (possibly
  // negative) padding applied.
  std::vector<int64_t> new_shape;
  for (int64_t i = 0; i < l_inp - l_pad; ++i) {
    new_shape.push_back(self.size(i));
  }
  for (int64_t i = 0; i < l_pad; ++i) {
    const int64_t dim = l_inp - l_pad + i;
    const int64_t pad_idx = 2 * (l_pad - i - 1);
    const int64_t new_dim = self.size(dim) + pad[pad_idx] + pad[pad_idx + 1];
    if (new_dim < 0) {
      TP_THROW(ValueError, "The input size ", input_sizes[l_inp - l_pad + i],
               ", plus negative padding ", pad[pad_idx], " and ",
               pad[pad_idx + 1],
               " resulted in a negative output size, which is invalid. Check "
               "dimension ",
               l_inp - l_pad + i, " of your input.");
    }
    new_shape.push_back(new_dim);
  }

  Tensor output = Tensor(new_shape, self.dtype(), self.device());
  if (output.numel() == 0) {
    return output;
  }

  Tensor cropped = self;
  Tensor interior = output;
  for (int64_t i = 0; i < l_pad; ++i) {
    const int64_t dim = l_inp - l_pad + i;
    const int64_t pad_idx = 2 * (l_pad - i - 1);
    const int64_t left = pad[pad_idx];
    const int64_t right = pad[pad_idx + 1];
    if (left < 0) {
      cropped = cropped.slice(dim, -left, cropped.size(dim));
    }
    if (right < 0) {
      cropped = cropped.slice(dim, 0, cropped.size(dim) + right);
    }
    if (left > 0) {
      interior = interior.slice(dim, left, interior.size(dim));
    }
    if (right > 0) {
      interior = interior.slice(dim, 0, interior.size(dim) - right);
    }
  }
  output.fill_(value);
  interior.copy_(cropped);
  return output;
}

TENSORPLAY_LIBRARY_IMPL(MPS, PadKernels) {
  m.impl("constant_pad_nd", constant_pad_nd_kernel);
}

} // namespace mps
} // namespace tensorplay
