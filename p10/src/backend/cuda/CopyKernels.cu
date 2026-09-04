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

template <typename T> struct real_of;
template <> struct real_of<cuFloatComplex> { using type = float; };
template <> struct real_of<cuDoubleComplex> { using type = double; };

template <typename C> struct dtype_of_complex;
template <> struct dtype_of_complex<cuFloatComplex> {
    static constexpr DType value = DType::ComplexFloat;
};
template <> struct dtype_of_complex<cuDoubleComplex> {
    static constexpr DType value = DType::ComplexDouble;
};

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

// Mixed real<->complex casts.  Storage is interleaved component scalars, so
// offsets are computed in each tensor's own element units and the complex
// side addresses its two components through a doubled pointer.
template <typename C>
__global__ void cast_real_to_complex_kernel(
    int64_t numel, C* __restrict__ dst, TensorInfo dst_info,
    const typename real_of<C>::type* __restrict__ src, TensorInfo src_info) {
    using R = typename real_of<C>::type;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    const int64_t dst_offset = get_linear_offset(idx, dst_info.sizes, dst_info.strides, dst_info.ndim);
    const int64_t src_offset = get_linear_offset(idx, dst_info.sizes, src_info.strides, src_info.ndim);
    dst[dst_offset] = C{src[src_offset], R(0)};
}

template <typename C>
__global__ void cast_complex_to_real_kernel(
    int64_t numel, typename real_of<C>::type* __restrict__ dst, TensorInfo dst_info,
    const C* __restrict__ src, TensorInfo src_info) {
    using R = typename real_of<C>::type;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    const int64_t dst_offset = get_linear_offset(idx, dst_info.sizes, dst_info.strides, dst_info.ndim);
    const int64_t src_offset = get_linear_offset(idx, dst_info.sizes, src_info.strides, src_info.ndim);
    dst[dst_offset] = src[src_offset].x;
}

template <typename D, typename S>
__global__ void cast_complex_to_complex_kernel(
    int64_t numel, D* __restrict__ dst, TensorInfo dst_info,
    const S* __restrict__ src, TensorInfo src_info) {
    using RD = typename real_of<D>::type;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    const int64_t dst_offset = get_linear_offset(idx, dst_info.sizes, dst_info.strides, dst_info.ndim);
    const int64_t src_offset = get_linear_offset(idx, dst_info.sizes, src_info.strides, src_info.ndim);
    dst[dst_offset] = D{static_cast<RD>(src[src_offset].x),
                        static_cast<RD>(src[src_offset].y)};
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

    // --- mixed real<->complex casts ----------------------------------------
    // the real component.  Width pairs only (f32<->c64, f64<->c128).
    #define TP_CUDA_CPLX_CAST_R2C(DT_REAL, CU_C)                                \
        if (self.dtype() == dtype_of_complex<CU_C>::value &&                    \
            src_cuda_tensor.dtype() == DType::DT_REAL) {                        \
            cast_real_to_complex_kernel<CU_C>                                   \
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(      \
                    numel,                                                      \
                    reinterpret_cast<CU_C*>(self.data_ptr()), dst_info,         \
                    reinterpret_cast<const real_of<CU_C>::type*>(      \
                        src_cuda_tensor.data_ptr()), src_info);                 \
            checkCuda(cudaGetLastError(), "CUDA complex cast kernel");          \
            return self;                                                        \
        }
    #define TP_CUDA_CPLX_CAST_C2R(CU_C, DT_REAL)                                \
        if (self.dtype() == DType::DT_REAL &&                                   \
            src_cuda_tensor.dtype() == dtype_of_complex<CU_C>::value) {         \
            cast_complex_to_real_kernel<CU_C>                                   \
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(      \
                    numel,                                                      \
                    reinterpret_cast<real_of<CU_C>::type*>(            \
                        self.data_ptr()), dst_info,                             \
                    reinterpret_cast<const CU_C*>(                              \
                        src_cuda_tensor.data_ptr()), src_info);                 \
            checkCuda(cudaGetLastError(), "CUDA complex cast kernel");          \
            return self;                                                        \
        }
    #define TP_CUDA_CPLX_CAST_C2C(CU_DST, CU_SRC)                               \
        if (self.dtype() == dtype_of_complex<CU_DST>::value &&                  \
            src_cuda_tensor.dtype() == dtype_of_complex<CU_SRC>::value) {       \
            cast_complex_to_complex_kernel<CU_DST, CU_SRC>                      \
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(      \
                    numel,                                                      \
                    reinterpret_cast<CU_DST*>(self.data_ptr()), dst_info,       \
                    reinterpret_cast<const CU_SRC*>(                            \
                        src_cuda_tensor.data_ptr()), src_info);                 \
            checkCuda(cudaGetLastError(), "CUDA complex cast kernel");          \
            return self;                                                        \
        }
    TP_CUDA_CPLX_CAST_R2C(Float32, cuFloatComplex)
    TP_CUDA_CPLX_CAST_R2C(Float64, cuDoubleComplex)
    TP_CUDA_CPLX_CAST_C2R(cuFloatComplex, Float32)
    TP_CUDA_CPLX_CAST_C2R(cuDoubleComplex, Float64)
    TP_CUDA_CPLX_CAST_C2C(cuFloatComplex, cuDoubleComplex)
    TP_CUDA_CPLX_CAST_C2C(cuDoubleComplex, cuFloatComplex)
    #undef TP_CUDA_CPLX_CAST_R2C
    #undef TP_CUDA_CPLX_CAST_C2R
    #undef TP_CUDA_CPLX_CAST_C2C
    
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

// Extract the single element of a 1-element device tensor.  The value is
// staged through a synchronous device-to-host copy so the result reflects all
// work queued on the current stream.
Scalar item_cuda(const Tensor& self) {
    if (!self.defined()) {
        TP_THROW(RuntimeError, "Tensor not defined");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    if (impl->is_sparse()) {
        TP_THROW(RuntimeError, "item() is not supported for sparse tensors");
    }
    if (impl->numel() != 1) {
        TP_THROW(ValueError, "item() only supported for 1-element tensors");
    }
    if (!impl->device().is_cuda()) {
        TP_THROW(RuntimeError, "item(): expected a CUDA tensor but got ",
                 impl->device().toString());
    }

    const void* src = self.data_ptr();
    // A 1-element tensor addresses its only element directly; strides are
    // irrelevant and data_ptr() already includes the storage offset.
    switch (impl->dtype()) {
        case DType::Float32: {
            float v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<double>(v));
        }
        case DType::Float64: {
            double v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(v);
        }
        case DType::Float16: {
            Half v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<float>(v));
        }
        case DType::BFloat16: {
            BFloat16 v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<float>(v));
        }
        case DType::Float8_e4m3fn: {
            Float8_e4m3fn v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<float>(v));
        }
        case DType::Float8_e5m2: {
            Float8_e5m2 v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<float>(v));
        }
        case DType::Int8: {
            int8_t v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<int64_t>(v));
        }
        case DType::Int16: {
            int16_t v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<int64_t>(v));
        }
        case DType::Int32: {
            int32_t v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<int64_t>(v));
        }
        case DType::Int64: {
            int64_t v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(v);
        }
        case DType::UInt8: {
            uint8_t v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<uint64_t>(v));
        }
        case DType::UInt16: {
            uint16_t v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<uint64_t>(v));
        }
        case DType::UInt32: {
            uint32_t v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<uint64_t>(v));
        }
        case DType::UInt64: {
            uint64_t v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(v);
        }
        case DType::Bool: {
            bool v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(v);
        }
        case DType::ComplexHalf: {
            std::complex<Half> v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(std::complex<float>(static_cast<float>(v.real()),
                                              static_cast<float>(v.imag())));
        }
        case DType::ComplexFloat: {
            std::complex<float> v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(v);
        }
        case DType::ComplexDouble: {
            std::complex<double> v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(v);
        }
        case DType::BComplex32: {
            std::complex<BFloat16> v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(std::complex<float>(static_cast<float>(v.real()),
                                              static_cast<float>(v.imag())));
        }
        default:
            TP_THROW(NotImplementedError, "item() not implemented for this dtype");
    }
}

TENSORPLAY_LIBRARY_IMPL(CUDA, CopyKernels) {
    m.impl("copy_", copy_kernel);
    m.impl("item", item_cuda);
    m.impl("sparse_coo_tensor", sparse_coo_tensor_cuda);
    m.impl("sparse_mask", sparse_mask_cuda);
    m.impl("to_dense", to_dense_sparse_cuda);
    m.impl("to_sparse", to_sparse_coo_cuda);
    m.impl("to_sparse_csr", to_sparse_csr_cuda);
    m.impl("_nnz", sparse_nnz_cuda);
    m.impl("sparse_mm", sparse_mm_cuda);
    m.impl("sparse_sum", sparse_sum_cuda);
    m.impl("sparse_add", sparse_add_cuda);
    m.impl("sparse_mul", sparse_mul_cuda);
    m.impl("spdiags", spdiags_cuda);
}

} // namespace cuda
} // namespace tensorplay
