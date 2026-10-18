#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "TypePromotion.h"

#include "backend/mps/MPSDevice.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSStream.h"

#include <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace mps {

// The bundled Metal shader library (ScanSort.metal).
static auto& lib = MetalShaderLibrary::getBundledLibrary();

namespace {

int64_t wrap_scan_dim(int64_t dim, int64_t ndim) {
  if (ndim == 0) return 0;
  int64_t d = dim;
  if (d < 0) d += ndim;
  if (d < 0 || d >= ndim) {
    TP_THROW(IndexError, "Dimension out of range");
  }
  return d;
}

void outer_inner(const std::vector<int64_t>& shape, int64_t dim,
                 int64_t& outer, int64_t& inner) {
  outer = 1;
  inner = 1;
  for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
  for (size_t i = static_cast<size_t>(dim) + 1; i < shape.size(); ++i) {
    inner *= shape[i];
  }
}

DType resolve_scan_dtype(const Tensor& self, std::optional<DType> dtype) {
  return dtype.value_or(isIntegralType(self.dtype(), true) ? DType::Int64
                                                           : self.dtype());
}

void dispatch_scan(const std::string& kernel_name, const Tensor& src,
               const Tensor& result, int64_t d_size, int64_t inner,
               int64_t outer) {
  auto pso = lib.getPipelineStateForFunc(kernel_name);
  const int d_size_i = static_cast<int>(d_size);
  const int inner_i = static_cast<int>(inner);
  const int outer_i = static_cast<int>(outer);
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, src, result, d_size_i, inner_i);
      const int max_tptg = [pso maxTotalThreadsPerThreadgroup];
      const int tg = std::min(std::max(inner_i, 1), max_tptg);
      [encoder dispatchThreads:MTLSizeMake(inner_i, outer_i, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
  });
}

// Scans `rows` rows of `len` consecutive elements, one threadgroup per row.
// The shader reads four elements per thread, so the group width follows the
// row length up to a full pass; only a whole number of SIMD groups is valid.
void dispatch_row_scan(const std::string& kernel_name, const Tensor& src,
                       const Tensor& result, int64_t rows, int64_t len) {
  auto pso = lib.getPipelineStateForFunc(kernel_name);
  const std::array<uint32_t, 2> sizes = {static_cast<uint32_t>(rows),
                                         static_cast<uint32_t>(len)};
  constexpr uint32_t kReadsPerThread = 4;
  const uint32_t wanted =
      static_cast<uint32_t>((len + kReadsPerThread - 1) / kReadsPerThread);
  uint32_t tg = std::min<uint32_t>(wanted, 256);
  tg = (tg + 31) / 32 * 32;
  tg = std::max<uint32_t>(tg, 32);
  tg = std::min<uint32_t>(
      tg, static_cast<uint32_t>([pso maxTotalThreadsPerThreadgroup]) / 32 * 32);
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      // One partial per lane, sized for the widest accumulator.
      [encoder setThreadgroupMemoryLength:32 * sizeof(long) atIndex:0];
      mtl_setArgs(encoder, src, result, sizes);
      [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows),
                                                1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
  });
}

// Whether the scan can run down rows of the contiguous buffer: only the
// innermost axis lands on consecutive elements, and the shader addresses rows
// and elements with 32-bit indices. Returns the row count in `rows`.
bool innermost_rows(const Tensor& src, int64_t inner, int64_t d_size,
                    int64_t outer, uint64_t& rows) {
  if (inner != 1 || src.storage_offset() != 0) {
    return false;
  }
  rows = static_cast<uint64_t>(outer);
  return d_size > 0 && d_size <= std::numeric_limits<uint32_t>::max() &&
         rows <= std::numeric_limits<uint32_t>::max();
}

} // namespace

// ---------------------------------------------------------------------------
// cumsum / cumprod
// ---------------------------------------------------------------------------

Tensor cumsum_kernel(const Tensor& self, int64_t dim,
                     std::optional<DType> dtype) {
  const int64_t nd = self.dim();
  dim = wrap_scan_dim(dim, nd);
  const DType out_dtype = resolve_scan_dtype(self, dtype);
  Tensor src = (self.dtype() == out_dtype) ? self.contiguous()
                                           : self.to(out_dtype).contiguous();
  if (nd == 0) {
    Tensor result = Tensor::empty({}, out_dtype, src.device());
    result.copy_(src);
    return result;
  }
  Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(src.shape()),
                                out_dtype, src.device());
  const int64_t d_size = src.size(dim);
  if (d_size == 0 || src.numel() == 0) return result;
  int64_t outer = 1;
  int64_t inner = 1;
  outer_inner(static_cast<std::vector<int64_t>>(src.shape()), dim, outer,
              inner);
  uint64_t rows = 0;
  if (innermost_rows(src, inner, d_size, outer, rows)) {
    dispatch_row_scan("scan_rows_sum_" + scalarToMetalTypeString(out_dtype),
                      src, result, static_cast<int64_t>(rows), d_size);
  } else {
    dispatch_scan("cumsum_" + scalarToMetalTypeString(out_dtype), src, result,
                  d_size, inner, outer);
  }
  return result;
}

Tensor cumprod_kernel(const Tensor& self, int64_t dim,
                      std::optional<DType> dtype) {
  const int64_t nd = self.dim();
  dim = wrap_scan_dim(dim, nd);
  const DType out_dtype = resolve_scan_dtype(self, dtype);
  Tensor src = (self.dtype() == out_dtype) ? self.contiguous()
                                           : self.to(out_dtype).contiguous();
  if (nd == 0) {
    Tensor result = Tensor::empty({}, out_dtype, src.device());
    result.copy_(src);
    return result;
  }
  Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(src.shape()),
                                out_dtype, src.device());
  const int64_t d_size = src.size(dim);
  if (d_size == 0 || src.numel() == 0) return result;
  int64_t outer = 1;
  int64_t inner = 1;
  outer_inner(static_cast<std::vector<int64_t>>(src.shape()), dim, outer,
              inner);
  uint64_t rows = 0;
  if (innermost_rows(src, inner, d_size, outer, rows)) {
    dispatch_row_scan("scan_rows_prod_" + scalarToMetalTypeString(out_dtype),
                      src, result, static_cast<int64_t>(rows), d_size);
  } else {
    dispatch_scan("cumprod_" + scalarToMetalTypeString(out_dtype), src, result,
                  d_size, inner, outer);
  }
  return result;
}

Tensor& cumsum_inplace_kernel(Tensor& self, int64_t dim,
                              std::optional<DType> dtype) {
  Tensor r = cumsum_kernel(self, dim, dtype);
  self.copy_(r);
  return self;
}

Tensor& cumprod_inplace_kernel(Tensor& self, int64_t dim,
                               std::optional<DType> dtype) {
  Tensor r = cumprod_kernel(self, dim, dtype);
  self.copy_(r);
  return self;
}

// ---------------------------------------------------------------------------
// sort / argsort
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor> sort_kernel(const Tensor& self, int64_t dim,
                                       bool descending) {
  const int64_t nd = self.dim();
  if (nd == 0) {
    TP_THROW(RuntimeError, "sort: expects at least 1 dimension");
  }
  dim = wrap_scan_dim(dim, nd);
  Tensor self_c = self.contiguous();
  const int64_t d_size = self_c.size(dim);
  int64_t outer = 1;
  int64_t inner = 1;
  outer_inner(static_cast<std::vector<int64_t>>(self_c.shape()), dim, outer,
            inner);
  Tensor values = Tensor::empty(static_cast<std::vector<int64_t>>(self_c.shape()),
                                 self_c.dtype(), self_c.device());
  Tensor indices = Tensor::empty(static_cast<std::vector<int64_t>>(self_c.shape()),
                                  DType::Int64, self_c.device());
  if (d_size == 0 || self_c.numel() == 0) {
    return std::make_tuple(values, indices);
  }

  // Each thread keeps four elements; the threadgroup covers the slice, and a
  // slice beyond 256 threads stays on the error path for now.
  constexpr int64_t kSortLanes = 4;
  constexpr int64_t kSortMaxTptg = 256;
  constexpr int64_t kSortMaxSlice = kSortLanes * kSortMaxTptg;
  if (d_size > kSortMaxSlice) {
    TP_THROW(NotImplementedError,
             "sort: slices longer than ", kSortMaxSlice,
             " are not supported on MPS yet");
  }
  int64_t tptg = 32;
  while (tptg * kSortLanes < d_size) {
    tptg <<= 1;
  }
  auto pso = lib.getPipelineStateForFunc(
      "sort_merge_" + scalarToMetalTypeString(self_c.dtype()) + "_tptg" +
      std::to_string(tptg));

  const DType dt = self_c.dtype();
  const int desc_i = descending ? 1 : 0;
  // Sentinel value: NaN ascending (it sorts past everything, including +inf),
  // -inf descending; min/max for integers.
  float f_sent = 0.0f;
  Half h_sent(0.0f);
  BFloat16 b_sent(0.0f);
  int64_t l_sent = 0;
  int32_t i_sent = 0;
  int16_t s_sent = 0;
  int8_t c_sent = 0;
  uint8_t uc_sent = 0;
  const void* sentinel_ptr = nullptr;
  size_t sentinel_size = 0;
  switch (dt) {
    case DType::Float32:
      f_sent = descending ? -std::numeric_limits<float>::infinity()
                        : std::numeric_limits<float>::quiet_NaN();
      sentinel_ptr = &f_sent;
      sentinel_size = sizeof(float);
      break;
    case DType::Float16:
      h_sent = Half(descending ? -std::numeric_limits<float>::infinity()
                               : std::numeric_limits<float>::quiet_NaN());
      sentinel_ptr = &h_sent;
      sentinel_size = sizeof(Half);
      break;
    case DType::BFloat16:
      b_sent = BFloat16(descending ? -std::numeric_limits<float>::infinity()
                                    : std::numeric_limits<float>::quiet_NaN());
      sentinel_ptr = &b_sent;
      sentinel_size = sizeof(BFloat16);
      break;
    case DType::Int64:
      l_sent = descending ? std::numeric_limits<int64_t>::lowest()
                          : std::numeric_limits<int64_t>::max();
      sentinel_ptr = &l_sent;
      sentinel_size = sizeof(int64_t);
      break;
    case DType::Int32:
      i_sent = descending ? std::numeric_limits<int32_t>::lowest()
                          : std::numeric_limits<int32_t>::max();
      sentinel_ptr = &i_sent;
      sentinel_size = sizeof(int32_t);
      break;
    case DType::Int16:
      s_sent = descending ? std::numeric_limits<int16_t>::lowest()
                          : std::numeric_limits<int16_t>::max();
      sentinel_ptr = &s_sent;
      sentinel_size = sizeof(int16_t);
      break;
    case DType::Int8:
      c_sent = descending ? std::numeric_limits<int8_t>::lowest()
                          : std::numeric_limits<int8_t>::max();
      sentinel_ptr = &c_sent;
      sentinel_size = sizeof(int8_t);
      break;
    case DType::UInt8:
      uc_sent = descending ? std::numeric_limits<uint8_t>::lowest()
                           : std::numeric_limits<uint8_t>::max();
      sentinel_ptr = &uc_sent;
      sentinel_size = sizeof(uint8_t);
      break;
    default:
      TP_THROW(NotImplementedError, "sort: unsupported dtype on MPS");
  }

  const int d_size_i = static_cast<int>(d_size);
  const int inner_i = static_cast<int>(inner);
  const int num_slices_i = static_cast<int>(outer * inner);
  const size_t val_bytes = tptg * kSortLanes * self_c.itemsize();
  const size_t idx_bytes = tptg * kSortLanes * sizeof(uint32_t);

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, self_c, values, indices, d_size_i, inner_i, desc_i);
      [encoder setBytes:sentinel_ptr length:sentinel_size atIndex:6];
      [encoder setThreadgroupMemoryLength:val_bytes atIndex:0];
      [encoder setThreadgroupMemoryLength:idx_bytes atIndex:1];
      [encoder dispatchThreadgroups:MTLSizeMake(num_slices_i, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(tptg), 1, 1)];
    }
  });
  return std::make_tuple(values, indices);
}

Tensor argsort_kernel(const Tensor& self, int64_t dim, bool descending) {
  return std::get<1>(sort_kernel(self, dim, descending));
}

TENSORPLAY_LIBRARY_IMPL(MPS, ScanSortKernels) {
  m.impl("cumsum", cumsum_kernel);
  m.impl("cumprod", cumprod_kernel);
  m.impl("cumsum_", cumsum_inplace_kernel);
  m.impl("cumprod_", cumprod_inplace_kernel);
  m.impl("sort", sort_kernel);
  m.impl("argsort", argsort_kernel);
}

} // namespace mps
} // namespace tensorplay