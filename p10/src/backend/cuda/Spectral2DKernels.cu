#include "Spectral2DKernels.h"

#include "CUDARuntime.h"
#include "Dispatcher.h"
#include "Exception.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <cmath>
#include <climits>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {

#define TP_SPECTRAL2D_CUDA_CHECK(condition) \
    do { \
        cudaError_t error = (condition); \
        if (error != cudaSuccess) { \
            TP_THROW(RuntimeError, "CUDA error: ", cudaGetErrorString(error)); \
        } \
    } while (0)

inline std::string cufft_error_name(cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
        case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
        default: return "unknown cuFFT error";
    }
}

#define TP_SPECTRAL2D_CUFFT_CHECK(condition) \
    do { \
        cufftResult error = (condition); \
        if (error != CUFFT_SUCCESS) { \
            TP_THROW(RuntimeError, "cuFFT error: ", cufft_error_name(error)); \
        } \
    } while (0)

namespace {

constexpr int kThreads = 256;

struct FFT2Args {
    int64_t first_dim;
    int64_t last_dim;
    int64_t first_size;
    int64_t last_size;
};

enum class FFTNorm { none, by_root_n, by_n };

int64_t wrap_fft_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    TP_CHECK(dim >= 0 && dim < ndim, "FFT dimension is out of range");
    return dim;
}

FFT2Args canonicalize_fft2(const Tensor& self,
                           const std::optional<std::vector<int64_t>>& s,
                           const std::vector<int64_t>& dim,
                           bool inverse_real) {
    TP_CHECK(self.dim() >= 2, "FFT2 expects an input with at least two dimensions");
    TP_CHECK(dim.size() == 2, "FFT2 expects exactly two transform dimensions");
    const int64_t first_dim = wrap_fft_dim(dim[0], self.dim());
    const int64_t last_dim = wrap_fft_dim(dim[1], self.dim());
    TP_CHECK(first_dim != last_dim, "FFT transform dimensions must be unique");
    if (s.has_value()) {
        TP_CHECK(s->size() == 2,
                 "FFT2 shape and dimension arguments must have the same length");
    }

    auto resolve_size = [&](size_t index, int64_t source_dim) {
        int64_t size = s.has_value() ? (*s)[index] : -1;
        if (size == -1) {
            size = inverse_real && index == 1
                ? 2 * (self.size(source_dim) - 1)
                : self.size(source_dim);
        }
        TP_CHECK(size > 0, "FFT size must be positive");
        return size;
    };

    return {first_dim, last_dim,
            resolve_size(0, first_dim), resolve_size(1, last_dim)};
}

FFTNorm norm_from_string(const std::string& norm, bool forward) {
    if (norm == "backward") return forward ? FFTNorm::none : FFTNorm::by_n;
    if (norm == "forward") return forward ? FFTNorm::by_n : FFTNorm::none;
    if (norm == "ortho") return FFTNorm::by_root_n;
    TP_THROW(RuntimeError, "Invalid normalization mode: \"", norm, "\"");
    return FFTNorm::none;
}

bool is_complex(DType dtype) {
    return dtype == DType::ComplexFloat || dtype == DType::ComplexDouble;
}

DType complex_dtype(DType dtype) {
    TP_CHECK(dtype == DType::Float32 || dtype == DType::Float64,
             "Unsupported real dtype for FFT");
    return dtype == DType::Float64 ? DType::ComplexDouble : DType::ComplexFloat;
}

DType real_dtype(DType dtype) {
    TP_CHECK(dtype == DType::ComplexFloat || dtype == DType::ComplexDouble,
             "Unsupported complex dtype for FFT");
    return dtype == DType::ComplexDouble ? DType::Float64 : DType::Float32;
}

std::vector<int64_t> sizes_of(const Tensor& tensor) {
    return static_cast<std::vector<int64_t>>(tensor.shape());
}

std::pair<Tensor, std::vector<int64_t>> move_fft_dims_last(const Tensor& input,
                                                            int64_t first_dim,
                                                            int64_t last_dim) {
    const int64_t ndim = input.dim();
    std::vector<int64_t> permutation;
    permutation.reserve(static_cast<size_t>(ndim));
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != first_dim && i != last_dim) permutation.push_back(i);
    }
    permutation.push_back(first_dim);
    permutation.push_back(last_dim);

    bool identity = true;
    for (int64_t i = 0; i < ndim; ++i) {
        if (permutation[static_cast<size_t>(i)] != i) {
            identity = false;
            break;
        }
    }
    if (identity) return {input, {}};

    Tensor moved = input.permute(permutation).contiguous();
    std::vector<int64_t> inverse(static_cast<size_t>(ndim));
    for (int64_t i = 0; i < ndim; ++i) {
        inverse[static_cast<size_t>(permutation[static_cast<size_t>(i)])] = i;
    }
    return {std::move(moved), std::move(inverse)};
}

Tensor finish_fft_layout(Tensor output, const std::vector<int64_t>& inverse) {
    if (inverse.empty()) return output;
    return output.permute(inverse).contiguous();
}

template <typename R, typename C>
__global__ void promote_real_kernel(int64_t count, const R* source, C* destination) {
    const int64_t index = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (index < count) {
        destination[index].x = source[index];
        destination[index].y = R(0);
    }
}

template <typename T>
Tensor promote_complex(const Tensor& input) {
    Tensor source = input.contiguous();
    Tensor output(sizes_of(source), complex_dtype(source.dtype()), source.device());
    const int64_t count = source.numel();
    if (count == 0) return output;
    auto stream = getCurrentCUDAStream().stream();
    if constexpr (std::is_same_v<T, double>) {
        promote_real_kernel<double, cufftDoubleComplex>
            <<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                count, static_cast<const double*>(source.data_ptr()),
                static_cast<cufftDoubleComplex*>(output.data_ptr()));
    } else {
        promote_real_kernel<float, cufftComplex>
            <<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                count, static_cast<const float*>(source.data_ptr()),
                static_cast<cufftComplex*>(output.data_ptr()));
    }
    TP_SPECTRAL2D_CUDA_CHECK(cudaGetLastError());
    return output;
}

template <typename T>
__global__ void resize_plane_kernel(int64_t batch, int64_t source_height,
                                    int64_t source_width, int64_t output_height,
                                    int64_t output_width, const T* source, T* output) {
    const int64_t batch_index = blockIdx.x;
    if (batch_index >= batch) return;
    const int64_t output_plane = output_height * output_width;
    const int64_t source_plane = source_height * source_width;
    for (int64_t index = threadIdx.x; index < output_plane; index += blockDim.x) {
        const int64_t row = index / output_width;
        const int64_t column = index - row * output_width;
        if (row < source_height && column < source_width) {
            output[batch_index * output_plane + index] =
                source[batch_index * source_plane + row * source_width + column];
        } else {
            output[batch_index * output_plane + index] = T{};
        }
    }
}

Tensor resize_fft_plane(const Tensor& input, int64_t height, int64_t width) {
    const std::vector<int64_t> input_sizes = sizes_of(input);
    const int64_t source_height = input_sizes[input_sizes.size() - 2];
    const int64_t source_width = input_sizes.back();
    if (source_height == height && source_width == width) return input;
    TP_CHECK(height > 0 && width > 0, "FFT size must be positive");

    std::vector<int64_t> output_sizes = input_sizes;
    output_sizes[output_sizes.size() - 2] = height;
    output_sizes.back() = width;
    Tensor output(output_sizes, input.dtype(), input.device());
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < input_sizes.size(); ++i) batch *= input_sizes[i];
    if (batch == 0) return output;
    auto stream = getCurrentCUDAStream().stream();
    if (input.dtype() == DType::Float64) {
        resize_plane_kernel<double><<<batch, kThreads, 0, stream>>>(
            batch, source_height, source_width, height, width,
            static_cast<const double*>(input.data_ptr()),
            static_cast<double*>(output.data_ptr()));
    } else if (input.dtype() == DType::Float32) {
        resize_plane_kernel<float><<<batch, kThreads, 0, stream>>>(
            batch, source_height, source_width, height, width,
            static_cast<const float*>(input.data_ptr()),
            static_cast<float*>(output.data_ptr()));
    } else if (input.dtype() == DType::ComplexDouble) {
        resize_plane_kernel<cufftDoubleComplex><<<batch, kThreads, 0, stream>>>(
            batch, source_height, source_width, height, width,
            static_cast<const cufftDoubleComplex*>(input.data_ptr()),
            static_cast<cufftDoubleComplex*>(output.data_ptr()));
    } else {
        resize_plane_kernel<cufftComplex><<<batch, kThreads, 0, stream>>>(
            batch, source_height, source_width, height, width,
            static_cast<const cufftComplex*>(input.data_ptr()),
            static_cast<cufftComplex*>(output.data_ptr()));
    }
    TP_SPECTRAL2D_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor resize_fft_support(const Tensor& input, int64_t first_dim,
                          int64_t last_dim, int64_t first_size,
                          int64_t last_size) {
    auto [moved, inverse] = move_fft_dims_last(input.contiguous(), first_dim, last_dim);
    Tensor resized = resize_fft_plane(moved, first_size, last_size);
    return finish_fft_layout(std::move(resized), inverse);
}

struct PlanKey {
    int type;
    int height;
    int width;
    int batch;

    bool operator<(const PlanKey& other) const {
        if (type != other.type) return type < other.type;
        if (height != other.height) return height < other.height;
        if (width != other.width) return width < other.width;
        return batch < other.batch;
    }
};

cufftHandle acquire_plan(cufftType type, int64_t height, int64_t width,
                         int64_t batch, int64_t input_width, int64_t output_width) {
    const int64_t input_distance = height * input_width;
    const int64_t output_distance = height * output_width;
    TP_CHECK(height <= INT_MAX && width <= INT_MAX && batch <= INT_MAX &&
                 input_distance <= INT_MAX && output_distance <= INT_MAX,
             "FFT dimensions exceed cuFFT limits");
    static std::mutex mutex;
    static std::map<PlanKey, cufftHandle> plans;
    const PlanKey key{static_cast<int>(type), static_cast<int>(height),
                      static_cast<int>(width), static_cast<int>(batch)};
    std::lock_guard<std::mutex> lock(mutex);
    auto found = plans.find(key);
    if (found != plans.end()) {
        TP_SPECTRAL2D_CUFFT_CHECK(cufftSetStream(found->second,
                                                  getCurrentCUDAStream().stream()));
        return found->second;
    }

    int dimensions[2] = {static_cast<int>(height), static_cast<int>(width)};
    int input_embed[2] = {static_cast<int>(height), static_cast<int>(input_width)};
    int output_embed[2] = {static_cast<int>(height), static_cast<int>(output_width)};
    cufftHandle plan;
    TP_SPECTRAL2D_CUFFT_CHECK(cufftPlanMany(
        &plan, 2, dimensions, input_embed, 1, static_cast<int>(input_distance),
        output_embed, 1, static_cast<int>(output_distance), type,
        static_cast<int>(batch)));
    TP_SPECTRAL2D_CUFFT_CHECK(cufftSetStream(plan, getCurrentCUDAStream().stream()));
    plans.emplace(key, plan);
    return plan;
}

template <typename C, typename S>
__global__ void scale_complex_kernel(int64_t count, C* data, S scale) {
    const int64_t index = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (index < count) {
        data[index].x *= scale;
        data[index].y *= scale;
    }
}

template <typename C, typename S>
__global__ void scale_interior_bins_kernel(int64_t count, int64_t width,
                                           int64_t interior_end, C* data, S scale) {
    const int64_t index = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (index < count) {
        const int64_t column = index % width;
        if (column > 0 && column < interior_end) {
            data[index].x *= scale;
            data[index].y *= scale;
        }
    }
}

template <typename T>
__global__ void scale_real_kernel(int64_t count, T* data, T scale) {
    const int64_t index = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (index < count) data[index] *= scale;
}

template <typename C, typename T>
__global__ void extract_real_kernel(int64_t count, const C* source, T* destination) {
    const int64_t index = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (index < count) destination[index] = source[index].x;
}

template <typename T>
Tensor extract_real_2d(const Tensor& input) {
    using C = std::conditional_t<std::is_same_v<T, double>, cufftDoubleComplex, cufftComplex>;
    Tensor source = input.contiguous();
    Tensor output(sizes_of(source), real_dtype(source.dtype()), source.device());
    const int64_t count = source.numel();
    if (count == 0) return output;
    auto stream = getCurrentCUDAStream().stream();
    extract_real_kernel<C, T><<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        count, static_cast<const C*>(source.data_ptr()), static_cast<T*>(output.data_ptr()));
    TP_SPECTRAL2D_CUDA_CHECK(cudaGetLastError());
    return output;
}

template <typename T>
T norm_factor(FFTNorm mode, int64_t n) {
    switch (mode) {
        case FFTNorm::none: return T(1);
        case FFTNorm::by_root_n: return T(1) / std::sqrt(T(n));
        case FFTNorm::by_n: return T(1) / T(n);
    }
    return T(1);
}

template <typename T>
Tensor c2c_plane(const Tensor& input, bool forward, FFTNorm norm,
                 int64_t transform_size) {
    using C = std::conditional_t<std::is_same_v<T, double>, cufftDoubleComplex, cufftComplex>;
    const std::vector<int64_t> sizes = sizes_of(input);
    const int64_t height = sizes[sizes.size() - 2];
    const int64_t width = sizes.back();
    const int64_t batch = input.numel() / (height * width);
    Tensor output(sizes, input.dtype(), input.device());
    if (batch > 0) {
        cufftHandle plan = acquire_plan(std::is_same_v<T, double> ? CUFFT_Z2Z : CUFFT_C2C,
                                        height, width, batch, width, width);
        if constexpr (std::is_same_v<T, double>) {
            TP_SPECTRAL2D_CUFFT_CHECK(cufftExecZ2Z(
                plan, static_cast<cufftDoubleComplex*>(input.data_ptr()),
                static_cast<cufftDoubleComplex*>(output.data_ptr()),
                forward ? CUFFT_FORWARD : CUFFT_INVERSE));
        } else {
            TP_SPECTRAL2D_CUFFT_CHECK(cufftExecC2C(
                plan, static_cast<cufftComplex*>(input.data_ptr()),
                static_cast<cufftComplex*>(output.data_ptr()),
                forward ? CUFFT_FORWARD : CUFFT_INVERSE));
        }
    }
    const T scale = norm_factor<T>(norm, transform_size);
    if (scale != T(1) && output.numel() > 0) {
        auto stream = getCurrentCUDAStream().stream();
        if constexpr (std::is_same_v<T, double>) {
            scale_complex_kernel<cufftDoubleComplex, double>
                <<<(output.numel() + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    output.numel(), static_cast<cufftDoubleComplex*>(output.data_ptr()), scale);
        } else {
            scale_complex_kernel<cufftComplex, float>
                <<<(output.numel() + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    output.numel(), static_cast<cufftComplex*>(output.data_ptr()), scale);
        }
        TP_SPECTRAL2D_CUDA_CHECK(cudaGetLastError());
    }
    return output;
}

template <typename T>
Tensor r2c_plane(const Tensor& input, FFTNorm norm, int64_t transform_size) {
    using C = std::conditional_t<std::is_same_v<T, double>, cufftDoubleComplex, cufftComplex>;
    const std::vector<int64_t> input_sizes = sizes_of(input);
    const int64_t height = input_sizes[input_sizes.size() - 2];
    const int64_t width = input_sizes.back();
    const int64_t bins = width / 2 + 1;
    const int64_t batch = input.numel() / (height * width);
    std::vector<int64_t> output_sizes = input_sizes;
    output_sizes.back() = bins;
    Tensor output(output_sizes, complex_dtype(input.dtype()), input.device());
    if (batch > 0) {
        cufftHandle plan = acquire_plan(std::is_same_v<T, double> ? CUFFT_D2Z : CUFFT_R2C,
                                        height, width, batch, width, bins);
        if constexpr (std::is_same_v<T, double>) {
            TP_SPECTRAL2D_CUFFT_CHECK(cufftExecD2Z(
                plan, static_cast<cufftDoubleReal*>(input.data_ptr()),
                static_cast<cufftDoubleComplex*>(output.data_ptr())));
        } else {
            TP_SPECTRAL2D_CUFFT_CHECK(cufftExecR2C(
                plan, static_cast<cufftReal*>(input.data_ptr()),
                static_cast<cufftComplex*>(output.data_ptr())));
        }
    }
    const T scale = norm_factor<T>(norm, transform_size);
    if (scale != T(1) && output.numel() > 0) {
        auto stream = getCurrentCUDAStream().stream();
        if constexpr (std::is_same_v<T, double>) {
            scale_complex_kernel<cufftDoubleComplex, double>
                <<<(output.numel() + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    output.numel(), static_cast<cufftDoubleComplex*>(output.data_ptr()), scale);
        } else {
            scale_complex_kernel<cufftComplex, float>
                <<<(output.numel() + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    output.numel(), static_cast<cufftComplex*>(output.data_ptr()), scale);
        }
        TP_SPECTRAL2D_CUDA_CHECK(cudaGetLastError());
    }
    return output;
}

template <typename T>
Tensor c2r_plane(const Tensor& input, int64_t output_width, FFTNorm norm,
                 int64_t transform_size) {
    using C = std::conditional_t<std::is_same_v<T, double>, cufftDoubleComplex, cufftComplex>;
    const std::vector<int64_t> input_sizes = sizes_of(input);
    const int64_t height = input_sizes[input_sizes.size() - 2];
    const int64_t bins = input_sizes.back();
    const int64_t batch = input.numel() / (height * bins);
    std::vector<int64_t> output_sizes = input_sizes;
    output_sizes.back() = output_width;
    Tensor output(output_sizes, real_dtype(input.dtype()), input.device());
    if (batch > 0) {
        cufftHandle plan = acquire_plan(std::is_same_v<T, double> ? CUFFT_Z2D : CUFFT_C2R,
                                        height, output_width, batch, bins, output_width);
        if constexpr (std::is_same_v<T, double>) {
            TP_SPECTRAL2D_CUFFT_CHECK(cufftExecZ2D(
                plan, static_cast<cufftDoubleComplex*>(input.data_ptr()),
                static_cast<cufftDoubleReal*>(output.data_ptr())));
        } else {
            TP_SPECTRAL2D_CUFFT_CHECK(cufftExecC2R(
                plan, static_cast<cufftComplex*>(input.data_ptr()),
                static_cast<cufftReal*>(output.data_ptr())));
        }
    }
    const T scale = norm_factor<T>(norm, transform_size);
    if (scale != T(1) && output.numel() > 0) {
        auto stream = getCurrentCUDAStream().stream();
        scale_real_kernel<T><<<(output.numel() + kThreads - 1) / kThreads,
                                kThreads, 0, stream>>>(
            output.numel(), static_cast<T*>(output.data_ptr()), scale);
        TP_SPECTRAL2D_CUDA_CHECK(cudaGetLastError());
    }
    return output;
}

template <typename T>
Tensor fft2_c2c_impl(const Tensor& self, const FFT2Args& args,
                     FFTNorm norm, bool forward) {
    TP_CHECK(self.dtype() == DType::Float32 || self.dtype() == DType::Float64 ||
                 is_complex(self.dtype()),
             "Unsupported dtype for FFT");
    Tensor input = is_complex(self.dtype())
        ? self.contiguous()
        : promote_complex<T>(self);
    auto [moved, inverse] = move_fft_dims_last(input, args.first_dim, args.last_dim);
    Tensor resized = resize_fft_plane(moved, args.first_size, args.last_size);
    Tensor output = c2c_plane<T>(resized, forward, norm,
                                 args.first_size * args.last_size);
    return finish_fft_layout(std::move(output), inverse);
}

template <typename T>
Tensor fft2_r2c_impl(const Tensor& self, const FFT2Args& args,
                     FFTNorm norm) {
    TP_CHECK(self.dtype() == DType::Float32 || self.dtype() == DType::Float64,
             "RFFT2 expects a real input");
    auto [moved, inverse] = move_fft_dims_last(self.contiguous(),
                                               args.first_dim, args.last_dim);
    Tensor resized = resize_fft_plane(moved, args.first_size, args.last_size);
    Tensor output = r2c_plane<T>(resized, norm, args.first_size * args.last_size);
    return finish_fft_layout(std::move(output), inverse);
}

template <typename T>
Tensor fft2_c2r_impl(const Tensor& self, const FFT2Args& args,
                     FFTNorm norm) {
    TP_CHECK(is_complex(self.dtype()), "IRFFT2 expects a complex input");
    auto [moved, inverse] = move_fft_dims_last(self.contiguous(),
                                               args.first_dim, args.last_dim);
    Tensor resized = resize_fft_plane(moved, args.first_size, args.last_size / 2 + 1);
    Tensor output = c2r_plane<T>(resized, args.last_size, norm,
                                 args.first_size * args.last_size);
    return finish_fft_layout(std::move(output), inverse);
}

template <typename T>
Tensor fft2_c2c_backward_impl(const Tensor& grad, const Tensor& self,
                              const FFT2Args& args, const std::string& norm,
                              bool forward_was) {
    TP_CHECK(is_complex(grad.dtype()), "FFT2 backward expects a complex gradient");
    auto [moved, inverse] = move_fft_dims_last(grad.contiguous(),
                                               args.first_dim, args.last_dim);
    Tensor transformed = c2c_plane<T>(moved, !forward_was,
                                      norm_from_string(norm, forward_was),
                                      args.first_size * args.last_size);
    Tensor output = finish_fft_layout(std::move(transformed), inverse);
    output = resize_fft_support(output, args.first_dim, args.last_dim,
                                self.size(args.first_dim), self.size(args.last_dim));
    return is_complex(self.dtype()) ? output : extract_real_2d<T>(output);
}

template <typename T>
Tensor fft2_rfft_backward_impl(const Tensor& grad, const Tensor& self,
                               const FFT2Args& args, const std::string& norm) {
    TP_CHECK(is_complex(grad.dtype()), "RFFT2 backward expects a complex gradient");
    auto [moved, inverse] = move_fft_dims_last(grad.contiguous(),
                                               args.first_dim, args.last_dim);
    Tensor full = resize_fft_plane(moved, args.first_size, args.last_size);
    Tensor transformed = c2c_plane<T>(full, false,
                                      norm_from_string(norm, true),
                                      args.first_size * args.last_size);
    Tensor output = extract_real_2d<T>(transformed);
    output = finish_fft_layout(std::move(output), inverse);
    return resize_fft_support(output, args.first_dim, args.last_dim,
                              self.size(args.first_dim), self.size(args.last_dim));
}

template <typename T>
Tensor fft2_irfft_backward_impl(const Tensor& grad, const Tensor& self,
                                const FFT2Args& args, const std::string& norm) {
    TP_CHECK(!is_complex(grad.dtype()), "IRFFT2 backward expects a real gradient");
    Tensor spectrum = fft2_r2c_impl<T>(grad, args,
                                       norm_from_string(norm, false));
    auto [moved, inverse] = move_fft_dims_last(spectrum, args.first_dim,
                                               args.last_dim);
    Tensor resized = resize_fft_plane(moved, self.size(args.first_dim),
                                      self.size(args.last_dim));
    const int64_t interior_end = (args.last_size + 1) / 2;
    const int64_t count = resized.numel();
    if (count > 0) {
        auto stream = getCurrentCUDAStream().stream();
        if constexpr (std::is_same_v<T, double>) {
            scale_interior_bins_kernel<cufftDoubleComplex, double>
                <<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    count, resized.size(-1), interior_end,
                    static_cast<cufftDoubleComplex*>(resized.data_ptr()), 2.0);
        } else {
            scale_interior_bins_kernel<cufftComplex, float>
                <<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    count, resized.size(-1), interior_end,
                    static_cast<cufftComplex*>(resized.data_ptr()), 2.0f);
        }
        TP_SPECTRAL2D_CUDA_CHECK(cudaGetLastError());
    }
    return finish_fft_layout(std::move(resized), inverse);
}

}  // namespace

Tensor fft_fft2_cuda(const Tensor& self, std::optional<std::vector<int64_t>> s,
                    const std::vector<int64_t>& dim, std::string norm) {
    const FFT2Args args = canonicalize_fft2(self, s, dim, false);
    return self.dtype() == DType::ComplexDouble || self.dtype() == DType::Float64
        ? fft2_c2c_impl<double>(self, args, norm_from_string(norm, true), true)
        : fft2_c2c_impl<float>(self, args, norm_from_string(norm, true), true);
}

Tensor fft_ifft2_cuda(const Tensor& self, std::optional<std::vector<int64_t>> s,
                     const std::vector<int64_t>& dim, std::string norm) {
    const FFT2Args args = canonicalize_fft2(self, s, dim, false);
    return self.dtype() == DType::ComplexDouble || self.dtype() == DType::Float64
        ? fft2_c2c_impl<double>(self, args, norm_from_string(norm, false), false)
        : fft2_c2c_impl<float>(self, args, norm_from_string(norm, false), false);
}

Tensor fft_rfft2_cuda(const Tensor& self, std::optional<std::vector<int64_t>> s,
                     const std::vector<int64_t>& dim, std::string norm) {
    const FFT2Args args = canonicalize_fft2(self, s, dim, false);
    return self.dtype() == DType::Float64
        ? fft2_r2c_impl<double>(self, args, norm_from_string(norm, true))
        : fft2_r2c_impl<float>(self, args, norm_from_string(norm, true));
}

Tensor fft_irfft2_cuda(const Tensor& self, std::optional<std::vector<int64_t>> s,
                      const std::vector<int64_t>& dim, std::string norm) {
    const FFT2Args args = canonicalize_fft2(self, s, dim, true);
    return self.dtype() == DType::ComplexDouble
        ? fft2_c2r_impl<double>(self, args, norm_from_string(norm, false))
        : fft2_c2r_impl<float>(self, args, norm_from_string(norm, false));
}

Tensor fft_fft2_backward_cuda(const Tensor& grad, const Tensor& self,
                              std::optional<std::vector<int64_t>> s,
                              const std::vector<int64_t>& dim, std::string norm) {
    const FFT2Args args = canonicalize_fft2(self, s, dim, false);
    return grad.dtype() == DType::ComplexDouble
        ? fft2_c2c_backward_impl<double>(grad, self, args, norm, true)
        : fft2_c2c_backward_impl<float>(grad, self, args, norm, true);
}

Tensor fft_ifft2_backward_cuda(const Tensor& grad, const Tensor& self,
                               std::optional<std::vector<int64_t>> s,
                               const std::vector<int64_t>& dim, std::string norm) {
    const FFT2Args args = canonicalize_fft2(self, s, dim, false);
    return grad.dtype() == DType::ComplexDouble
        ? fft2_c2c_backward_impl<double>(grad, self, args, norm, false)
        : fft2_c2c_backward_impl<float>(grad, self, args, norm, false);
}

Tensor fft_rfft2_backward_cuda(const Tensor& grad, const Tensor& self,
                               std::optional<std::vector<int64_t>> s,
                               const std::vector<int64_t>& dim, std::string norm) {
    const FFT2Args args = canonicalize_fft2(self, s, dim, false);
    return grad.dtype() == DType::ComplexDouble
        ? fft2_rfft_backward_impl<double>(grad, self, args, norm)
        : fft2_rfft_backward_impl<float>(grad, self, args, norm);
}

Tensor fft_irfft2_backward_cuda(const Tensor& grad, const Tensor& self,
                                std::optional<std::vector<int64_t>> s,
                                const std::vector<int64_t>& dim, std::string norm) {
    const FFT2Args args = canonicalize_fft2(self, s, dim, true);
    return grad.dtype() == DType::Float64
        ? fft2_irfft_backward_impl<double>(grad, self, args, norm)
        : fft2_irfft_backward_impl<float>(grad, self, args, norm);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, Spectral2DKernels) {
    m.impl("fft_fft2", fft_fft2_cuda);
    m.impl("fft_ifft2", fft_ifft2_cuda);
    m.impl("fft_rfft2", fft_rfft2_cuda);
    m.impl("fft_irfft2", fft_irfft2_cuda);
    m.impl("fft_fft2_backward", fft_fft2_backward_cuda);
    m.impl("fft_ifft2_backward", fft_ifft2_backward_cuda);
    m.impl("fft_rfft2_backward", fft_rfft2_backward_cuda);
    m.impl("fft_irfft2_backward", fft_irfft2_backward_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
