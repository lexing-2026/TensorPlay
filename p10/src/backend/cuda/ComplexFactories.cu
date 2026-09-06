// Complex factory kernels on CUDA: real, imag, conj, adjoint, complex,
// polar.  One elementwise pass each over contiguous buffers.

#include "Tensor.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"
#include "CUDALoops.cuh"
#include "Complex.h"
#include "Exception.h"
#include "Utils.h"

#include <cuda_runtime.h>

#include <vector>
#include <complex>

namespace tensorplay {
namespace cuda {

#define CUDA_CHECK(condition)                                                 \
    do {                                                                      \
        cudaError_t error = condition;                                        \
        if (error != cudaSuccess) {                                           \
            TP_THROW(RuntimeError,                                            \
                     std::string("CUDA Error: ") + cudaGetErrorString(error)); \
        }                                                                     \
    } while (0)

namespace {

constexpr int kThreads = 256;

inline dim3 make_grid(int64_t work) {
    return dim3(static_cast<unsigned>((work + kThreads - 1) / kThreads));
}

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

bool is_cplx(DType d) { return isComplexType(d); }

bool is_factory_real_dtype(DType d) {
    return d == DType::Float16 || d == DType::Float32 ||
           d == DType::Float64 || d == DType::BFloat16;
}

void check_factory_inputs(const Tensor& a, const Tensor& b, const char* name) {
    if (!is_factory_real_dtype(a.dtype()) || !is_factory_real_dtype(b.dtype())) {
        TP_THROW(NotImplementedError, name,
                 " expects floating-point inputs");
    }
    if (a.dtype() != b.dtype()) {
        TP_THROW(RuntimeError, name, " expects inputs with the same dtype");
    }
    if (a.device() != b.device()) {
        TP_THROW(DeviceMismatchError, name,
                 " expects inputs on the same device");
    }
}

template <typename CT, typename RT>
__global__ void cplx_real_kernel(int64_t n, const CT* src, RT* dst) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) dst[i] = src[i].real();
}
template <typename CT, typename RT>
__global__ void cplx_imag_kernel(int64_t n, const CT* src, RT* dst) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) dst[i] = src[i].imag();
}
template <typename CT>
__global__ void cplx_conj_kernel(int64_t n, CT* data) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        const CT value = data[i];
        data[i] = CT(value.real(), -value.imag());
    }
}
template <typename CT, typename RT>
__global__ void cplx_pack_kernel(int64_t n, const RT* re, const RT* im, CT* dst) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        dst[i] = CT(re[i], im[i]);
    }
}
template <typename CT, typename RT>
__global__ void polar_kernel(int64_t n, const RT* ab, const RT* ang, CT* dst) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    using scalar_t = decltype(CT{}.real());
    for (; i < n; i += stride) {
        const RT radius = ab[i];
        const RT angle = ang[i];
        dst[i] = CT(static_cast<scalar_t>(radius * std::cos(angle)),
                    static_cast<scalar_t>(radius * std::sin(angle)));
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

Tensor real_cuda(const Tensor& self) {
    // Real input is its own real part (zero-copy view, as the op contract
    // states); complex input materializes the real component at its paired
    // real precision.
    if (!is_cplx(self.dtype())) return self;
    DType rt = toRealValueType(self.dtype());
    Tensor out = Tensor::empty(shape_of(self), rt, self.device());
    if (out.numel() == 0) return out;
    switch (self.dtype()) {
        case DType::ComplexHalf:
        {
            Tensor sc = self.contiguous();
            const int64_t n = sc.numel();
            auto stream = getCurrentCUDAStream();
            dim3 grid = make_grid(n), block(kThreads);
            cplx_real_kernel<tensorplay::complex<Half>, Half>
                <<<grid, block, 0, stream.stream()>>>(
                    n,
                    static_cast<const tensorplay::complex<Half>*>(sc.data_ptr()),
                    out.data_ptr<Half>());
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case DType::ComplexFloat:
        {
            TensorIterator iter = TensorIteratorConfig()
                .check_all_same_dtype(false)
                .add_output(out)
                .add_input(self)
                .build();
            gpu_kernel(iter, [] __device__ (std::complex<float> value) -> float {
                return value.real();
            });
            break;
        }
        case DType::ComplexDouble:
        {
            TensorIterator iter = TensorIteratorConfig()
                .check_all_same_dtype(false)
                .add_output(out)
                .add_input(self)
                .build();
            gpu_kernel(iter, [] __device__ (std::complex<double> value) -> double {
                return value.real();
            });
            break;
        }
        case DType::BComplex32:
        {
            Tensor sc = self.contiguous();
            const int64_t n = sc.numel();
            auto stream = getCurrentCUDAStream();
            dim3 grid = make_grid(n), block(kThreads);
            cplx_real_kernel<tensorplay::complex<BFloat16>, BFloat16>
                <<<grid, block, 0, stream.stream()>>>(
                    n,
                    static_cast<const tensorplay::complex<BFloat16>*>(
                        sc.data_ptr()),
                    out.data_ptr<BFloat16>());
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        default:
            TP_THROW(NotImplementedError, "real does not support this dtype");
    }
    return out;
}

Tensor imag_cuda(const Tensor& self) {
    if (!is_cplx(self.dtype())) {
        return Tensor::zeros(shape_of(self), self.dtype(), self.device());
    }
    DType rt = toRealValueType(self.dtype());
    Tensor out = Tensor::empty(shape_of(self), rt, self.device());
    if (out.numel() == 0) return out;
    switch (self.dtype()) {
        case DType::ComplexHalf:
        {
            Tensor sc = self.contiguous();
            const int64_t n = sc.numel();
            auto stream = getCurrentCUDAStream();
            dim3 grid = make_grid(n), block(kThreads);
            cplx_imag_kernel<tensorplay::complex<Half>, Half>
                <<<grid, block, 0, stream.stream()>>>(
                    n,
                    static_cast<const tensorplay::complex<Half>*>(sc.data_ptr()),
                    out.data_ptr<Half>());
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case DType::ComplexFloat:
        {
            TensorIterator iter = TensorIteratorConfig()
                .check_all_same_dtype(false)
                .add_output(out)
                .add_input(self)
                .build();
            gpu_kernel(iter, [] __device__ (std::complex<float> value) -> float {
                return value.imag();
            });
            break;
        }
        case DType::ComplexDouble:
        {
            TensorIterator iter = TensorIteratorConfig()
                .check_all_same_dtype(false)
                .add_output(out)
                .add_input(self)
                .build();
            gpu_kernel(iter, [] __device__ (std::complex<double> value) -> double {
                return value.imag();
            });
            break;
        }
        case DType::BComplex32:
        {
            Tensor sc = self.contiguous();
            const int64_t n = sc.numel();
            auto stream = getCurrentCUDAStream();
            dim3 grid = make_grid(n), block(kThreads);
            cplx_imag_kernel<tensorplay::complex<BFloat16>, BFloat16>
                <<<grid, block, 0, stream.stream()>>>(
                    n,
                    static_cast<const tensorplay::complex<BFloat16>*>(
                        sc.data_ptr()),
                    out.data_ptr<BFloat16>());
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        default:
            TP_THROW(NotImplementedError, "imag does not support this dtype");
    }
    return out;
}

Tensor conj_cuda(const Tensor& self) {
    if (!is_cplx(self.dtype())) return self.clone();
    if (self.dtype() == DType::ComplexFloat ||
        self.dtype() == DType::ComplexDouble) {
        Tensor out = Tensor::empty(shape_of(self), self.dtype(), self.device());
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(out)
            .add_input(self)
            .build();
        if (self.dtype() == DType::ComplexFloat) {
            gpu_kernel(
                iter, [] __device__ (std::complex<float> value)
                    -> std::complex<float> {
                    return std::complex<float>(value.real(), -value.imag());
                });
        } else {
            gpu_kernel(
                iter, [] __device__ (std::complex<double> value)
                    -> std::complex<double> {
                    return std::complex<double>(value.real(), -value.imag());
                });
        }
        return out;
    }
    Tensor out = detail::contiguous_clone(self);
    int64_t n = out.numel();
    if (n == 0) return out;
    auto stream = getCurrentCUDAStream();
    dim3 grid = make_grid(n), block(kThreads);
    switch (self.dtype()) {
        case DType::ComplexHalf:
            cplx_conj_kernel<tensorplay::complex<Half>>
                <<<grid, block, 0, stream.stream()>>>(
                    n,
                    static_cast<tensorplay::complex<Half>*>(out.data_ptr()));
            break;
        case DType::BComplex32:
            cplx_conj_kernel<tensorplay::complex<BFloat16>>
                <<<grid, block, 0, stream.stream()>>>(
                    n,
                    static_cast<tensorplay::complex<BFloat16>*>(
                        out.data_ptr()));
            break;
        default:
            TP_THROW(NotImplementedError, "conj does not support this dtype");
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor complex_cuda(const Tensor& re, const Tensor& im) {
    check_factory_inputs(re, im, "complex");
    const DType fdt = re.dtype();
    const DType cdt = toComplexType(fdt);
    std::vector<int64_t> shape = broadcast_shapes(shape_of(re), shape_of(im));
    Tensor out = Tensor::empty(shape, cdt, re.device());
    if (out.numel() == 0) return out;
    switch (fdt) {
        case DType::Float16:
        {
            Tensor rc = re.expand(shape).contiguous();
            Tensor ic = im.expand(shape).contiguous().to(rc.dtype());
            const int64_t n = out.numel();
            auto stream = getCurrentCUDAStream();
            dim3 grid = make_grid(n), block(kThreads);
            cplx_pack_kernel<tensorplay::complex<Half>, Half>
                <<<grid, block, 0, stream.stream()>>>(
                    n, rc.data_ptr<Half>(), ic.data_ptr<Half>(),
                    static_cast<tensorplay::complex<Half>*>(out.data_ptr()));
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case DType::Float32:
        {
            TensorIterator iter = TensorIteratorConfig()
                .check_all_same_dtype(false)
                .add_output(out)
                .add_input(re)
                .add_input(im)
                .build();
            gpu_kernel(iter, [] __device__ (float real, float imag)
                -> std::complex<float> {
                return std::complex<float>(real, imag);
            });
            break;
        }
        case DType::Float64:
        {
            TensorIterator iter = TensorIteratorConfig()
                .check_all_same_dtype(false)
                .add_output(out)
                .add_input(re)
                .add_input(im)
                .build();
            gpu_kernel(iter, [] __device__ (double real, double imag)
                -> std::complex<double> {
                return std::complex<double>(real, imag);
            });
            break;
        }
        case DType::BFloat16:
        {
            Tensor rc = re.expand(shape).contiguous();
            Tensor ic = im.expand(shape).contiguous().to(rc.dtype());
            const int64_t n = out.numel();
            auto stream = getCurrentCUDAStream();
            dim3 grid = make_grid(n), block(kThreads);
            cplx_pack_kernel<tensorplay::complex<BFloat16>, BFloat16>
                <<<grid, block, 0, stream.stream()>>>(
                    n, rc.data_ptr<BFloat16>(), ic.data_ptr<BFloat16>(),
                    static_cast<tensorplay::complex<BFloat16>*>(
                        out.data_ptr()));
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        default:
            TP_THROW(NotImplementedError, "complex does not support this dtype");
    }
    return out;
}

Tensor polar_cuda(const Tensor& abs_, const Tensor& angle_) {
    check_factory_inputs(abs_, angle_, "polar");
    const DType fdt = abs_.dtype();
    const DType cdt = toComplexType(fdt);
    std::vector<int64_t> shape = broadcast_shapes(shape_of(abs_), shape_of(angle_));
    Tensor out = Tensor::empty(shape, cdt, abs_.device());
    if (out.numel() == 0) return out;
    switch (cdt) {
        case DType::ComplexHalf:
        {
            Tensor a = abs_.expand(shape).contiguous().to(DType::Float32);
            Tensor th = angle_.expand(shape).contiguous().to(DType::Float32);
            const int64_t n = out.numel();
            auto stream = getCurrentCUDAStream();
            dim3 grid = make_grid(n), block(kThreads);
            polar_kernel<tensorplay::complex<Half>, float>
                <<<grid, block, 0, stream.stream()>>>(
                    n, a.data_ptr<float>(), th.data_ptr<float>(),
                    static_cast<tensorplay::complex<Half>*>(out.data_ptr()));
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case DType::ComplexFloat:
        {
            TensorIterator iter = TensorIteratorConfig()
                .check_all_same_dtype(false)
                .add_output(out)
                .add_input(abs_)
                .add_input(angle_)
                .build();
            gpu_kernel(iter, [] __device__ (float radius, float angle)
                -> std::complex<float> {
                return std::complex<float>(radius * ::cosf(angle),
                                           radius * ::sinf(angle));
            });
            break;
        }
        case DType::ComplexDouble:
        {
            TensorIterator iter = TensorIteratorConfig()
                .check_all_same_dtype(false)
                .add_output(out)
                .add_input(abs_)
                .add_input(angle_)
                .build();
            gpu_kernel(iter, [] __device__ (double radius, double angle)
                -> std::complex<double> {
                return std::complex<double>(radius * ::cos(angle),
                                            radius * ::sin(angle));
            });
            break;
        }
        case DType::BComplex32:
        {
            Tensor a = abs_.expand(shape).contiguous().to(DType::Float32);
            Tensor th = angle_.expand(shape).contiguous().to(DType::Float32);
            const int64_t n = out.numel();
            auto stream = getCurrentCUDAStream();
            dim3 grid = make_grid(n), block(kThreads);
            polar_kernel<tensorplay::complex<BFloat16>, float>
                <<<grid, block, 0, stream.stream()>>>(
                    n, a.data_ptr<float>(), th.data_ptr<float>(),
                    static_cast<tensorplay::complex<BFloat16>*>(
                        out.data_ptr()));
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        default:
            TP_THROW(NotImplementedError, "polar does not support this dtype");
    }
    return out;
}


// composed with conj(); ndim <= 1 is plain conj.  conj_cuda materializes the
// conjugate for complex inputs and aliases real ones.
Tensor adjoint_cuda(const Tensor& self) {
    if (self.dim() <= 1) return conj_cuda(self);
    return conj_cuda(self.transpose(-2, -1));
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ComplexFactories) {
    m.impl("real", real_cuda);
    m.impl("imag", imag_cuda);
    m.impl("conj", conj_cuda);
    m.impl("adjoint", adjoint_cuda);
    m.impl("complex", complex_cuda);
    m.impl("polar", polar_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
