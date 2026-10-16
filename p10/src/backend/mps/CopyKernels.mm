#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "DistributionDispatch.h"
#include "Parallel.h"
#include "Quantizer.h"

#include "backend/mps/MPSAllocator.h"
#include "backend/mps/MPSDevice.h"
#include "backend/mps/MPSRuntime.h"
#include "backend/mps/MPSStream.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <vector>

namespace tensorplay {
namespace mps {
namespace {

template <typename self_t, typename src_t>
void strided_copy_mps(self_t* dst, const std::vector<int64_t>& dst_strides,
                      const src_t* src, const std::vector<int64_t>& src_strides,
                      const std::vector<int64_t>& sizes) {
  const int ndim = static_cast<int>(sizes.size());
  if (ndim == 0) {
    dst[0] = cast_value<self_t>(src[0]);
    return;
  }
  std::vector<int64_t> o_sizes, o_dstr, o_sstr;
  o_sizes.reserve(ndim);
  o_dstr.reserve(ndim);
  o_sstr.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    if (sizes[i] == 1) continue;  // stride irrelevant; contributes one element
    if (sizes[i] == 0) return;    // nothing to copy
    o_sizes.push_back(sizes[i]);
    o_dstr.push_back(dst_strides[i]);
    o_sstr.push_back(src_strides[i]);
  }
  const int m = static_cast<int>(o_sizes.size());
  if (m == 0) {
    *dst = cast_value<self_t>(src[0]);
    return;
  }
  // Recurse dimension-by-dimension with the last dimension innermost so the
  // hot loop touches consecutive elements on both sides when strides allow.
  std::function<void(int, int64_t, int64_t)> walk =
      [&](int dim, int64_t dst_off, int64_t src_off) {
        if (dim == m - 1) {
          const int64_t len = o_sizes[dim];
          const int64_t d_stride = o_dstr[dim];
          const int64_t s_stride = o_sstr[dim];
          for (int64_t i = 0; i < len; ++i) {
            dst[dst_off + i * d_stride] =
                cast_value<self_t>(src[src_off + i * s_stride]);
          }
          return;
        }
        for (int64_t i = 0; i < o_sizes[dim]; ++i) {
          walk(dim + 1, dst_off + i * o_dstr[dim],
               src_off + i * o_sstr[dim]);
        }
      };
  walk(0, 0, 0);
}

// Copies `src` into `self` through the host-visible mappings of the MPS
// shared-memory buffers. Works for any combination of CPU/MPS tensors on the
// source side; the destination must be MPS.
void copy_via_host(Tensor& self, const Tensor& src) {
  const auto rm_strides = SizesAndStrides::compute_contiguous_strides(
      static_cast<std::vector<int64_t>>(self.shape()));
  const bool self_rm =
      static_cast<std::vector<int64_t>>(self.strides()) == rm_strides;
  const bool src_rm =
      static_cast<std::vector<int64_t>>(src.strides()) == rm_strides;

  distribution::dispatch_dtype(self.dtype(), [&](auto self_tag) {
    using self_t = decltype(self_tag);
    distribution::dispatch_dtype(src.dtype(), [&](auto src_tag) {
      using src_t = decltype(src_tag);
      if (self_rm && src_rm) {
        if (std::is_same_v<self_t, src_t>) {
          std::memcpy(self.data_ptr(), src.data_ptr(),
                      static_cast<size_t>(self.numel()) * self.itemsize());
          return;
        }
        self_t* d = self.data_ptr<self_t>();
        const src_t* s = src.data_ptr<src_t>();
        for (int64_t i = 0; i < self.numel(); ++i) {
          d[i] = cast_value<self_t>(s[i]);
        }
        return;
      }
      strided_copy_mps(
          self.data_ptr<self_t>(),
          static_cast<std::vector<int64_t>>(self.strides()),
          src.data_ptr<src_t>(),
          static_cast<std::vector<int64_t>>(src.strides()),
          static_cast<std::vector<int64_t>>(self.shape()));
    });
  });
}

} // namespace

Tensor& copy_kernel(Tensor& self, const Tensor& src_in, bool non_blocking) {
  const bool self_q = self.impl()->quantizer() != nullptr;
  const bool src_q = src_in.impl()->quantizer() != nullptr;
  if (self_q != src_q) {
    TP_THROW(NotImplementedError,
             "mixed quantized/non-quantized copy is not implemented on the "
             "MPS backend");
  }
  if (self_q) {
    // Copy the underlying integer codes; the destination keeps its own
    // quantizer (the codes view aliases the quantized tensor's storage).
    Tensor self_codes = quantized::strip_quantizer(self);
    Tensor src_codes = quantized::strip_quantizer(src_in);
    return copy_kernel(self_codes, src_codes, non_blocking);
  }
  // The source broadcasts to the destination's shape; a source that does
  // not is refused rather than read past its own elements.
  const Tensor src = src_in.shape() == self.shape()
      ? src_in : src_in.expand(static_cast<std::vector<int64_t>>(self.shape()));

  if (!self.device().is_mps()) {
    TP_THROW(RuntimeError, "copy_kernel dispatched to MPS but dst is not MPS");
  }
  if (self.numel() == 0) {
    return self;
  }

  if (src.device().is_mps()) {
    // Make sure any GPU work that produced `src` has completed before the
    // host mapping is read, then copy (blit for contiguous same-dtype,
    // host-visible walk otherwise).
    MPSStream* stream = getCurrentMPSStream();
    if (self.dtype() == src.dtype() && self.is_contiguous() &&
        src.is_contiguous()) {
      const size_t nbytes =
          static_cast<size_t>(self.numel()) * self.itemsize();
      void* src_buf = getMTLBufferFromPointer(src.data_ptr());
      void* dst_buf = getMTLBufferFromPointer(self.data_ptr());
      if (src_buf && dst_buf) {
        stream->copy((__bridge id<MTLBuffer>)src_buf,
                     (__bridge id<MTLBuffer>)dst_buf,
                     nbytes, 0, 0, SyncType::COMMIT_AND_WAIT);
        return self;
      }
      // Foreign buffers (not MPS-managed): fall back to the host mapping.
      stream->synchronize(SyncType::COMMIT_AND_WAIT);
      std::memcpy(self.data_ptr(), src.data_ptr(), nbytes);
      return self;
    }
    stream->synchronize(SyncType::COMMIT_AND_WAIT);
    copy_via_host(self, src);
    return self;
  }

  if (src.device().is_cpu()) {
    copy_via_host(self, src);
    return self;
  }

  TP_THROW(RuntimeError,
           "copy_kernel on MPS only supports CPU or MPS source, got ",
           src.device().toString());
}

TENSORPLAY_LIBRARY_IMPL(MPS, CopyKernels) {
  m.impl("copy_", copy_kernel);
}

} // namespace mps
} // namespace tensorplay