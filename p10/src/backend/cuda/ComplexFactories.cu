// Complex factory kernels on CUDA: real, imag, conj, adjoint, complex,
// polar.  One elementwise pass each over contiguous buffers.

#include "Tensor.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"
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

bool is_cplx(DType d) { return d == DType::ComplexFloat || d == DType::ComplexDouble; }

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
    for (; i < n; i += stride) data[i] = std::conj(data[i]);
}
template <typename CT, typename RT>
__global__ void cplx_pack_kernel(int64_t n, const RT* re, const RT* im, CT* dst) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) dst[i] = CT(re[i], im[i]);
}
template <typename CT, typename RT>
__global__ void polar_kernel(int64_t n, const RT* ab, const RT* ang, CT* dst) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) dst[i] = CT(ab[i] * std::cos(ang[i]), ab[i] * std::sin(ang[i]));
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

    DType rt = self.dtype() == DType::ComplexDouble ? DType::Float64 : DType::Float32;
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(shape_of(sc), rt, self.device());
    int64_t n = sc.numel();
    auto stream = getCurrentCUDAStream();
    dim3 grid = make_grid(n), block(kThreads);
    if (self.dtype() == DType::ComplexFloat)
        cplx_real_kernel<std::complex<float>, float><<<grid, block, 0, stream.stream()>>>(
            n, sc.data_ptr<std::complex<float>>(), out.data_ptr<float>());
    else
        cplx_real_kernel<std::complex<double>, double><<<grid, block, 0, stream.stream()>>>(
            n, sc.data_ptr<std::complex<double>>(), out.data_ptr<double>());
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor imag_cuda(const Tensor& self) {
    if (!is_cplx(self.dtype())) {
        return Tensor::zeros(shape_of(self), self.dtype(), self.device());
    }
    DType rt = self.dtype() == DType::ComplexDouble ? DType::Float64 : DType::Float32;
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(shape_of(sc), rt, self.device());
    int64_t n = sc.numel();
    auto stream = getCurrentCUDAStream();
    dim3 grid = make_grid(n), block(kThreads);
    if (self.dtype() == DType::ComplexFloat)
        cplx_imag_kernel<std::complex<float>, float><<<grid, block, 0, stream.stream()>>>(
            n, sc.data_ptr<std::complex<float>>(), out.data_ptr<float>());
    else
        cplx_imag_kernel<std::complex<double>, double><<<grid, block, 0, stream.stream()>>>(
            n, sc.data_ptr<std::complex<double>>(), out.data_ptr<double>());
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor conj_cuda(const Tensor& self) {
    if (!is_cplx(self.dtype())) return self.clone();
    if (self.dtype() != DType::ComplexFloat &&
        self.dtype() != DType::ComplexDouble)
        TP_THROW(NotImplementedError,
                 "CUDA conj: half complexes are not supported yet");
    Tensor out = detail::contiguous_clone(self);
    int64_t n = out.numel();
    if (n == 0) return out;
    auto stream = getCurrentCUDAStream();
    dim3 grid = make_grid(n), block(kThreads);
    if (self.dtype() == DType::ComplexFloat)
        cplx_conj_kernel<std::complex<float>><<<grid, block, 0, stream.stream()>>>(
            n, out.data_ptr<std::complex<float>>());
    else
        cplx_conj_kernel<std::complex<double>><<<grid, block, 0, stream.stream()>>>(
            n, out.data_ptr<std::complex<double>>());
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor complex_cuda(const Tensor& re, const Tensor& im) {
    DType fdt = promoteTypes(re.dtype(), im.dtype());
    DType cdt = fdt == DType::Float64 ? DType::ComplexDouble : DType::ComplexFloat;
    std::vector<int64_t> shape = broadcast_shapes(shape_of(re), shape_of(im));
    Tensor rc = re.expand(shape).contiguous().to(fdt == DType::Float64 ? DType::Float64
                                                                       : DType::Float32);
    Tensor ic = im.expand(shape).contiguous().to(rc.dtype());
    Tensor out = Tensor::empty(shape, cdt, re.device());
    int64_t n = out.numel();
    if (n == 0) return out;
    auto stream = getCurrentCUDAStream();
    dim3 grid = make_grid(n), block(kThreads);
    if (cdt == DType::ComplexFloat)
        cplx_pack_kernel<std::complex<float>, float><<<grid, block, 0, stream.stream()>>>(
            n, rc.data_ptr<float>(), ic.data_ptr<float>(),
            out.data_ptr<std::complex<float>>());
    else
        cplx_pack_kernel<std::complex<double>, double><<<grid, block, 0, stream.stream()>>>(
            n, rc.data_ptr<double>(), ic.data_ptr<double>(),
            out.data_ptr<std::complex<double>>());
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor polar_cuda(const Tensor& abs_, const Tensor& angle_) {
    DType fdt = promoteTypes(abs_.dtype(), angle_.dtype());
    if (fdt != DType::Float64) fdt = DType::Float32;
    DType cdt = fdt == DType::Float64 ? DType::ComplexDouble : DType::ComplexFloat;
    std::vector<int64_t> shape = broadcast_shapes(shape_of(abs_), shape_of(angle_));
    Tensor a = abs_.expand(shape).contiguous().to(fdt);
    Tensor th = angle_.expand(shape).contiguous().to(fdt);
    Tensor out = Tensor::empty(shape, cdt, abs_.device());
    int64_t n = out.numel();
    if (n == 0) return out;
    auto stream = getCurrentCUDAStream();
    dim3 grid = make_grid(n), block(kThreads);
    if (fdt == DType::Float64)
        polar_kernel<std::complex<double>, double><<<grid, block, 0, stream.stream()>>>(
            n, a.data_ptr<double>(), th.data_ptr<double>(),
            out.data_ptr<std::complex<double>>());
    else
        polar_kernel<std::complex<float>, float><<<grid, block, 0, stream.stream()>>>(
            n, a.data_ptr<float>(), th.data_ptr<float>(),
            out.data_ptr<std::complex<float>>());
    CUDA_CHECK(cudaGetLastError());
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
