#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "SparseKernels.h"
#include "Exception.h"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

namespace tensorplay {
namespace cuda {

// Torch permits tensors with up to 64 dimensions.  Matmul can reach that
// rank through broadcasted batch dimensions, so the strided clone path must
// not impose the old eight-dimension limit.
static constexpr int MAX_DIMS = 64;

struct TensorInfo {
    int64_t sizes[MAX_DIMS];
    int64_t strides[MAX_DIMS];
    int ndim;
};

TensorInfo get_tensor_info(const Tensor& t) {
    TensorInfo info;
    info.ndim = t.dim();
    if (info.ndim > MAX_DIMS) {
         TP_THROW(RuntimeError, "Tensor dimension exceeds MAX_DIMS (64) for CUDA copy");
    }
    for (int i = 0; i < info.ndim; ++i) {
        info.sizes[i] = t.size(i);
        info.strides[i] = t.stride(i);
    }
    return info;
}

__device__ int64_t get_linear_offset(int64_t idx, const int64_t* sizes, const int64_t* strides, int ndim) {
    int64_t offset = 0;
    for (int i = ndim - 1; i >= 0; --i) {
        int64_t mod = idx % sizes[i];
        idx /= sizes[i];
        offset += mod * strides[i];
    }
    return offset;
}

template <typename DstT, typename SrcT>
__global__ void copy_cast_kernel_impl(int64_t numel, DstT* dst, TensorInfo dst_info, const SrcT* src, TensorInfo src_info) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    int64_t dst_offset = get_linear_offset(idx, dst_info.sizes, dst_info.strides, dst_info.ndim);
    // Use dst_info.sizes for src logic as well, assuming shapes match (which is enforced in wrapper)
    // If src was expanded, its strides handle the mapping correctly.
    int64_t src_offset = get_linear_offset(idx, dst_info.sizes, src_info.strides, src_info.ndim); 
    
    dst[dst_offset] = static_cast<DstT>(src[src_offset]);
}

template <typename ComplexT>
__global__ void copy_complex_strided_kernel(
    int64_t numel, ComplexT* dst, TensorInfo dst_info,
    const ComplexT* src, TensorInfo src_info) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    const int64_t dst_offset = get_linear_offset(
        idx, dst_info.sizes, dst_info.strides, dst_info.ndim);
    const int64_t src_offset = get_linear_offset(
        idx, dst_info.sizes, src_info.strides, src_info.ndim);
    dst[dst_offset] = src[src_offset];
}

Tensor& copy_kernel(Tensor& self, const Tensor& src, bool non_blocking) {
    if (self.numel() != src.numel()) {
        TP_THROW(RuntimeError, "Sizes do not match for copy");
    }
    
    Device dst_dev = self.device();
    Device src_dev = src.device();
    
    if (!dst_dev.is_cuda()) {
         TP_THROW(RuntimeError, "copy_kernel dispatched to CUDA but dst is CPU?");
    }

    bool src_cuda = src_dev.is_cuda();
    auto stream = getCurrentCUDAStream(static_cast<int>(dst_dev.index()));

    // Optimize: Contiguous copy (both src and dst must be contiguous AND same dtype)
    if (self.dtype() == src.dtype() && self.is_contiguous() && src.is_contiguous()) {
        size_t nbytes = self.numel() * self.itemsize();
        if (src_cuda && src_dev.index() != dst_dev.index()) {
            // Establish both directions of the lifetime/order relationship:
            // destination work waits for prior source work, and the source
            // stream waits for the copy before its allocator may recycle src.
            auto src_stream = getCurrentCUDAStream(static_cast<int>(src_dev.index()));
            CUDAEvent source_ready;
            source_ready.record(src_stream);
            source_ready.block(stream);
            checkCuda(cudaMemcpyPeerAsync(
                          self.data_ptr(), static_cast<int>(dst_dev.index()),
                          src.data_ptr(), static_cast<int>(src_dev.index()),
                          nbytes, stream.stream()),
                      "cudaMemcpyPeerAsync");
            CUDAEvent copy_complete;
            copy_complete.record(stream);
            copy_complete.block(src_stream);
        } else {
            const cudaMemcpyKind kind = src_cuda
                ? cudaMemcpyDeviceToDevice
                : cudaMemcpyHostToDevice;
            checkCuda(cudaMemcpyAsync(self.data_ptr(), src.data_ptr(), nbytes, kind, stream.stream()),
                      "cudaMemcpyAsync");
            if (!src_cuda) {
                if (non_blocking && src.is_pinned()) {
                    recordPinnedStream(
                        const_cast<void*>(src.unsafeGetTensorImpl()->storage().data()), stream);
                } else {
                    // Pageable host storage cannot safely outlive the call and
                    // cudaMemcpyAsync may stage it synchronously anyway.
                    stream.synchronize();
                }
            }
        }
        return self;
    }
    
    // Strided copy or Casting copy
    // If src is CPU, we must move it to CUDA first (to a contiguous buffer)
    Tensor src_cuda_tensor = src;
    if (!src_cuda) {
        // Create a contiguous CUDA tensor
        // Note: we can't easily use "empty" then copy because we might recurse.
        // We manually allocate and copy from host.
        
        // 1. Ensure src is contiguous on host
        Tensor src_contig = src.is_contiguous() ? src : src.contiguous();
        
        // 2. Allocate temp CUDA memory
        src_cuda_tensor = Tensor(static_cast<std::vector<int64_t>>(src.shape()), src.dtype(), self.device());
        
        // 3. Copy H2D (contiguous)
        checkCuda(cudaMemcpyAsync(
                      src_cuda_tensor.data_ptr(), src_contig.data_ptr(),
                      src_contig.numel() * src_contig.itemsize(),
                      cudaMemcpyHostToDevice, stream.stream()),
                  "cudaMemcpyAsync (strided H2D staging)");
        if (non_blocking && src_contig.is_pinned()) {
            recordPinnedStream(
                const_cast<void*>(src_contig.unsafeGetTensorImpl()->storage().data()), stream);
        } else {
            stream.synchronize();
        }
    }
    
    // Now src_cuda_tensor is on CUDA. 
    // self is on CUDA.
    
    int64_t numel = self.numel();
    if (numel == 0) return self;

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    TensorInfo dst_info = get_tensor_info(self);
    TensorInfo src_info = get_tensor_info(src_cuda_tensor);

    // std::complex is a host-side type in this project.  Keep the CUDA copy
    // path explicit for complex tensors so expanded/strided complex batches
    // can be materialized without asking nvcc to instantiate std::complex
    // casts in the generic dtype conversion kernel.
    if (self.dtype() == src_cuda_tensor.dtype() && self.dtype() == DType::ComplexFloat) {
        copy_complex_strided_kernel<cuFloatComplex><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            numel, reinterpret_cast<cuFloatComplex*>(self.data_ptr<std::complex<float>>()), dst_info,
            reinterpret_cast<const cuFloatComplex*>(src_cuda_tensor.data_ptr<std::complex<float>>()), src_info);
        checkCuda(cudaGetLastError(), "CUDA complex float copy kernel");
        return self;
    }
    if (self.dtype() == src_cuda_tensor.dtype() && self.dtype() == DType::ComplexDouble) {
        copy_complex_strided_kernel<cuDoubleComplex><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            numel, reinterpret_cast<cuDoubleComplex*>(self.data_ptr<std::complex<double>>()), dst_info,
            reinterpret_cast<const cuDoubleComplex*>(src_cuda_tensor.data_ptr<std::complex<double>>()), src_info);
        checkCuda(cudaGetLastError(), "CUDA complex double copy kernel");
        return self;
    }
    
    // Define a local macro to avoid recursion of TENSORPLAY_FORALL_SCALAR_TYPES
    #define LOCAL_FORALL_SCALAR_TYPES(_) \
        _(uint8_t, UInt8) \
        _(int8_t, Int8) \
        _(int16_t, Int16) \
        _(int32_t, Int32) \
        _(int64_t, Int64) \
        _(uint16_t, UInt16) \
        _(uint32_t, UInt32) \
        _(uint64_t, UInt64) \
        _(float, Float32) \
        _(double, Float64) \
        _(tensorplay::Half, Float16) \
        _(tensorplay::BFloat16, BFloat16) \
        _(bool, Bool)

    #define SRC_CASE(src_ctype, src_name) \
    case DType::src_name: \
        copy_cast_kernel_impl<DstT, src_ctype><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(numel, self.data_ptr<DstT>(), dst_info, src_cuda_tensor.data_ptr<src_ctype>(), src_info); \
        break;

    #define DST_CASE(dst_ctype, dst_name) \
    case DType::dst_name: { \
        using DstT = dst_ctype; \
        switch (src_cuda_tensor.dtype()) { \
            LOCAL_FORALL_SCALAR_TYPES(SRC_CASE) \
            default: TP_THROW(NotImplementedError, "Unsupported src dtype for casting"); \
        } \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(DST_CASE)
        default: TP_THROW(NotImplementedError, "Unsupported dst dtype for casting");
    }
    #undef DST_CASE
    #undef SRC_CASE
    #undef LOCAL_FORALL_SCALAR_TYPES
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
         TP_THROW(RuntimeError, std::string("CUDA Copy Kernel Error: ") + cudaGetErrorString(err));
    }

    return self;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, CopyKernels) {
    m.impl("copy_", copy_kernel);
    m.impl("sparse_coo_tensor", sparse_coo_tensor_cuda);
    m.impl("sparse_mask", sparse_mask_cuda);
}

} // namespace cuda
} // namespace tensorplay
