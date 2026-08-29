// fractional_max_pool2d / fractional_max_pool3d CPU kernels.
//
// (shared interval logic from FractionalMaxPooling.h): pooling window starts
// follow generate_intervals(sample, ...) driven by the caller-provided
// random_samples tensor, so the kernel itself is RNG-free and deterministic.
// indices are flat in-plane int64 offsets.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"
#include "Half.h"
#include "BFloat16.h"
#include <vector>
#include <tuple>
#include <cmath>
#include <limits>

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

namespace {

template <typename compute_t>
static std::vector<int64_t> generate_intervals(compute_t sample, int64_t inputSize,
                                               int64_t outputSize, int64_t poolSize) {
    std::vector<int64_t> sequence(outputSize, 0);
    if (outputSize > 1) {
        compute_t alpha = static_cast<compute_t>(inputSize - poolSize) /
                          static_cast<compute_t>(outputSize - 1);
        for (int64_t i = 0; i < outputSize - 1; ++i) {
            sequence[i] = static_cast<int64_t>(
                static_cast<int>((i + sample) * alpha) - static_cast<int>(sample * alpha));
        }
    }
    if (outputSize > 0) {
        sequence[outputSize - 1] = inputSize - poolSize;
    }
    return sequence;
}

template <typename storage_t, typename compute_t>
static void fractional_max_pool2d_single_batch(
        const storage_t* input, storage_t* output, int64_t* indices,
        const storage_t* randomSamples,
        int64_t numPlanes, int64_t inputW, int64_t inputH,
        int64_t outputW, int64_t outputH, int64_t poolSizeW, int64_t poolSizeH) {
    parallel_for(0, numPlanes, 1, [&](int64_t begin, int64_t end) {
        for (int64_t plane = begin; plane < end; ++plane) {
            const storage_t* samplesForPlane = randomSamples + plane * 2;
            auto sequenceW = generate_intervals<compute_t>(
                static_cast<compute_t>(samplesForPlane[0]), inputW, outputW, poolSizeW);
            auto sequenceH = generate_intervals<compute_t>(
                static_cast<compute_t>(samplesForPlane[1]), inputH, outputH, poolSizeH);

            const storage_t* inputForPlane = input + plane * inputW * inputH;
            storage_t* outputForPlane = output + plane * outputW * outputH;
            int64_t* indicesForPlane = indices + plane * outputW * outputH;

            for (int64_t h = 0; h < outputH; ++h) {
                const int64_t inputHStart = sequenceH[h];
                for (int64_t w = 0; w < outputW; ++w) {
                    const int64_t inputWStart = sequenceW[w];
                    compute_t maxVal = -std::numeric_limits<compute_t>::infinity();
                    int64_t maxIndex = inputHStart * inputW + inputWStart;
                    for (int64_t h2 = inputHStart; h2 < inputHStart + poolSizeH; ++h2) {
                        for (int64_t w2 = inputWStart; w2 < inputWStart + poolSizeW; ++w2) {
                            const int64_t planeIndex = h2 * inputW + w2;
                            const compute_t val = static_cast<compute_t>(inputForPlane[planeIndex]);
                            if (val > maxVal || std::isnan(val)) {
                                maxVal = val;
                                maxIndex = planeIndex;
                            }
                        }
                    }
                    outputForPlane[h * outputW + w] = static_cast<storage_t>(maxVal);
                    indicesForPlane[h * outputW + w] = maxIndex;
                }
            }
        }
    });
}

template <typename storage_t, typename compute_t>
static std::tuple<Tensor, Tensor> fractional_max_pool2d_cpu_impl(
        const Tensor& input, const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& output_size, const Tensor& random_samples) {
    const bool batched = input.dim() == 4;
    const int64_t numBatch = batched ? input.size(0) : 1;
    const int64_t numPlanes = input.size(batched ? 1 : 0);
    const int64_t inputH = input.size(batched ? 2 : 1);
    const int64_t inputW = input.size(batched ? 3 : 2);
    const int64_t outputH = output_size[0];
    const int64_t outputW = output_size[1];
    const int64_t poolSizeH = kernel_size[0];
    const int64_t poolSizeW = kernel_size[1];

    std::vector<int64_t> out_shape = batched
        ? std::vector<int64_t>{numBatch, numPlanes, outputH, outputW}
        : std::vector<int64_t>{numPlanes, outputH, outputW};
    Tensor output = Tensor::empty(out_shape, input.dtype(), input.device());
    Tensor indices = Tensor::empty(out_shape, DType::Int64, input.device());
    if (output.numel() == 0) return {output, indices};

    const Tensor ic = input.contiguous();
    const Tensor sc = random_samples.contiguous();
    for (int64_t batch = 0; batch < numBatch; ++batch) {
        fractional_max_pool2d_single_batch<storage_t, compute_t>(
            ic.data_ptr<storage_t>() + batch * numPlanes * inputH * inputW,
            output.data_ptr<storage_t>() + batch * numPlanes * outputH * outputW,
            indices.data_ptr<int64_t>() + batch * numPlanes * outputH * outputW,
            sc.data_ptr<storage_t>() + batch * numPlanes * 2,
            numPlanes, inputW, inputH, outputW, outputH, poolSizeW, poolSizeH);
    }
    return {output, indices};
}

template <typename storage_t>
static void fractional_max_pool2d_backward_single_batch(
        storage_t* gradInput, const storage_t* gradOutput, const int64_t* indices,
        int64_t numPlanes, int64_t inputW, int64_t inputH,
        int64_t outputW, int64_t outputH) {
    parallel_for(0, numPlanes, 1, [&](int64_t begin, int64_t end) {
        for (int64_t plane = begin; plane < end; ++plane) {
            storage_t* gradInputForPlane = gradInput + plane * inputW * inputH;
            const storage_t* gradOutputForPlane = gradOutput + plane * outputW * outputH;
            const int64_t* indicesForPlane = indices + plane * outputW * outputH;
            for (int64_t h = 0; h < outputH; ++h) {
                for (int64_t w = 0; w < outputW; ++w) {
                    const int64_t outputIndex = h * outputW + w;
                    const int64_t index = indicesForPlane[outputIndex];
                    if (index < 0 || index >= inputW * inputH)
                        continue;
                    gradInputForPlane[index] += gradOutputForPlane[outputIndex];
                }
            }
        }
    });
}

template <typename storage_t>
static Tensor fractional_max_pool2d_backward_cpu_impl(
        const Tensor& grad_output, const Tensor& input,
        const std::vector<int64_t>& output_size, const Tensor& indices) {
    const bool batched = input.dim() == 4;
    const int64_t numBatch = batched ? input.size(0) : 1;
    const int64_t numPlanes = input.size(batched ? 1 : 0);
    const int64_t inputH = input.size(batched ? 2 : 1);
    const int64_t inputW = input.size(batched ? 3 : 2);
    const int64_t outputH = output_size[0];
    const int64_t outputW = output_size[1];

    std::vector<int64_t> in_shape = batched
        ? std::vector<int64_t>{numBatch, numPlanes, inputH, inputW}
        : std::vector<int64_t>{numPlanes, inputH, inputW};
    Tensor grad_input = Tensor::zeros(in_shape, input.dtype(), input.device());
    if (grad_output.numel() == 0) return grad_input;

    const Tensor goc = grad_output.contiguous();
    const Tensor ic = indices.contiguous();
    for (int64_t batch = 0; batch < numBatch; ++batch) {
        fractional_max_pool2d_backward_single_batch<storage_t>(
            grad_input.data_ptr<storage_t>() + batch * numPlanes * inputH * inputW,
            goc.data_ptr<storage_t>() + batch * numPlanes * outputH * outputW,
            ic.data_ptr<int64_t>() + batch * numPlanes * outputH * outputW,
            numPlanes, inputW, inputH, outputW, outputH);
    }
    return grad_input;
}

template <typename storage_t, typename compute_t>
static void fractional_max_pool3d_single_batch(
        const storage_t* input, storage_t* output, int64_t* indices,
        const storage_t* randomSamples,
        int64_t numPlanes, int64_t inputW, int64_t inputH, int64_t inputT,
        int64_t outputW, int64_t outputH, int64_t outputT,
        int64_t poolSizeW, int64_t poolSizeH, int64_t poolSizeT) {
    parallel_for(0, numPlanes, 1, [&](int64_t begin, int64_t end) {
        for (int64_t plane = begin; plane < end; ++plane) {
            const storage_t* samplesForPlane = randomSamples + plane * 3;
            auto sequenceT = generate_intervals<compute_t>(
                static_cast<compute_t>(samplesForPlane[0]), inputT, outputT, poolSizeT);
            auto sequenceH = generate_intervals<compute_t>(
                static_cast<compute_t>(samplesForPlane[1]), inputH, outputH, poolSizeH);
            auto sequenceW = generate_intervals<compute_t>(
                static_cast<compute_t>(samplesForPlane[2]), inputW, outputW, poolSizeW);

            const storage_t* inputForPlane = input + plane * inputT * inputH * inputW;
            storage_t* outputForPlane = output + plane * outputT * outputH * outputW;
            int64_t* indicesForPlane = indices + plane * outputT * outputH * outputW;

            for (int64_t t = 0; t < outputT; ++t) {
                const int64_t inputTStart = sequenceT[t];
                for (int64_t h = 0; h < outputH; ++h) {
                    const int64_t inputHStart = sequenceH[h];
                    for (int64_t w = 0; w < outputW; ++w) {
                        const int64_t inputWStart = sequenceW[w];
                        compute_t maxVal = -std::numeric_limits<compute_t>::infinity();
                        int64_t maxIndex = (inputTStart * inputH + inputHStart) * inputW + inputWStart;
                        for (int64_t t2 = inputTStart; t2 < inputTStart + poolSizeT; ++t2) {
                            for (int64_t h2 = inputHStart; h2 < inputHStart + poolSizeH; ++h2) {
                                for (int64_t w2 = inputWStart; w2 < inputWStart + poolSizeW; ++w2) {
                                    const int64_t planeIndex = (t2 * inputH + h2) * inputW + w2;
                                    const compute_t val = static_cast<compute_t>(inputForPlane[planeIndex]);
                                    if (val > maxVal || std::isnan(val)) {
                                        maxVal = val;
                                        maxIndex = planeIndex;
                                    }
                                }
                            }
                        }
                        outputForPlane[(t * outputH + h) * outputW + w] = static_cast<storage_t>(maxVal);
                        indicesForPlane[(t * outputH + h) * outputW + w] = maxIndex;
                    }
                }
            }
        }
    });
}

template <typename storage_t, typename compute_t>
static std::tuple<Tensor, Tensor> fractional_max_pool3d_cpu_impl(
        const Tensor& input, const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& output_size, const Tensor& random_samples) {
    const bool batched = input.dim() == 5;
    const int64_t numBatch = batched ? input.size(0) : 1;
    const int64_t numPlanes = input.size(batched ? 1 : 0);
    const int64_t inputT = input.size(batched ? 2 : 1);
    const int64_t inputH = input.size(batched ? 3 : 2);
    const int64_t inputW = input.size(batched ? 4 : 3);
    const int64_t outputT = output_size[0];
    const int64_t outputH = output_size[1];
    const int64_t outputW = output_size[2];
    const int64_t poolSizeT = kernel_size[0];
    const int64_t poolSizeH = kernel_size[1];
    const int64_t poolSizeW = kernel_size[2];

    std::vector<int64_t> out_shape = batched
        ? std::vector<int64_t>{numBatch, numPlanes, outputT, outputH, outputW}
        : std::vector<int64_t>{numPlanes, outputT, outputH, outputW};
    Tensor output = Tensor::empty(out_shape, input.dtype(), input.device());
    Tensor indices = Tensor::empty(out_shape, DType::Int64, input.device());
    if (output.numel() == 0) return {output, indices};

    const Tensor ic = input.contiguous();
    const Tensor sc = random_samples.contiguous();
    const int64_t in_vol = inputT * inputH * inputW;
    const int64_t out_vol = outputT * outputH * outputW;
    for (int64_t batch = 0; batch < numBatch; ++batch) {
        fractional_max_pool3d_single_batch<storage_t, compute_t>(
            ic.data_ptr<storage_t>() + batch * numPlanes * in_vol,
            output.data_ptr<storage_t>() + batch * numPlanes * out_vol,
            indices.data_ptr<int64_t>() + batch * numPlanes * out_vol,
            sc.data_ptr<storage_t>() + batch * numPlanes * 3,
            numPlanes, inputW, inputH, inputT, outputW, outputH, outputT,
            poolSizeW, poolSizeH, poolSizeT);
    }
    return {output, indices};
}

template <typename storage_t>
static void fractional_max_pool3d_backward_single_batch(
        storage_t* gradInput, const storage_t* gradOutput, const int64_t* indices,
        int64_t numPlanes, int64_t inputW, int64_t inputH, int64_t inputT,
        int64_t outputW, int64_t outputH, int64_t outputT) {
    const int64_t inputVol = inputT * inputH * inputW;
    parallel_for(0, numPlanes, 1, [&](int64_t begin, int64_t end) {
        for (int64_t plane = begin; plane < end; ++plane) {
            storage_t* gradInputForPlane = gradInput + plane * inputVol;
            const storage_t* gradOutputForPlane = gradOutput + plane * outputT * outputH * outputW;
            const int64_t* indicesForPlane = indices + plane * outputT * outputH * outputW;
            const int64_t out_vol = outputT * outputH * outputW;
            for (int64_t i = 0; i < out_vol; ++i) {
                const int64_t index = indicesForPlane[i];
                if (index < 0 || index >= inputVol)
                    continue;
                gradInputForPlane[index] += gradOutputForPlane[i];
            }
        }
    });
}

template <typename storage_t>
static Tensor fractional_max_pool3d_backward_cpu_impl(
        const Tensor& grad_output, const Tensor& input,
        const std::vector<int64_t>& output_size, const Tensor& indices) {
    const bool batched = input.dim() == 5;
    const int64_t numBatch = batched ? input.size(0) : 1;
    const int64_t numPlanes = input.size(batched ? 1 : 0);
    const int64_t inputT = input.size(batched ? 2 : 1);
    const int64_t inputH = input.size(batched ? 3 : 2);
    const int64_t inputW = input.size(batched ? 4 : 3);
    const int64_t outputT = output_size[0];
    const int64_t outputH = output_size[1];
    const int64_t outputW = output_size[2];

    std::vector<int64_t> in_shape = batched
        ? std::vector<int64_t>{numBatch, numPlanes, inputT, inputH, inputW}
        : std::vector<int64_t>{numPlanes, inputT, inputH, inputW};
    Tensor grad_input = Tensor::zeros(in_shape, input.dtype(), input.device());
    if (grad_output.numel() == 0) return grad_input;

    const Tensor goc = grad_output.contiguous();
    const Tensor ic = indices.contiguous();
    const int64_t in_vol = inputT * inputH * inputW;
    const int64_t out_vol = outputT * outputH * outputW;
    for (int64_t batch = 0; batch < numBatch; ++batch) {
        fractional_max_pool3d_backward_single_batch<storage_t>(
            grad_input.data_ptr<storage_t>() + batch * numPlanes * in_vol,
            goc.data_ptr<storage_t>() + batch * numPlanes * out_vol,
            ic.data_ptr<int64_t>() + batch * numPlanes * out_vol,
            numPlanes, inputW, inputH, inputT, outputW, outputH, outputT);
    }
    return grad_input;
}

void fractional_pool_check(const Tensor& input, const Tensor& random_samples,
                           int64_t ndim_spatial, const char* name) {
    if (input.dtype() != random_samples.dtype())
        TP_THROW(RuntimeError, std::string(name) + ": expect random_samples to have the same dtype as input");
    if (random_samples.dim() != 3)
        TP_THROW(RuntimeError, std::string(name) + ": expect random_samples to have 3 dimensions");
    const int64_t input_batch = input.dim() == ndim_spatial + 2 ? input.size(0) : 1;
    const int64_t input_channel = input.dim() == ndim_spatial + 2 ? input.size(1) : input.size(0);
    if (random_samples.size(0) < input_batch)
        TP_THROW(RuntimeError, std::string(name) + ": random_samples.size(0) must be >= input batch size");
    if (random_samples.size(1) != input_channel)
        TP_THROW(RuntimeError, std::string(name) + ": random_samples.size(1) must equal input channels");
    if (random_samples.size(2) != ndim_spatial)
        TP_THROW(RuntimeError, std::string(name) + ": random_samples.size(2) must equal the number of spatial dims");
}

}  // namespace

std::tuple<Tensor, Tensor> fractional_max_pool2d_cpu(
        const Tensor& self, const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& output_size, const Tensor& random_samples) {
    if (kernel_size.size() != 2 || output_size.size() != 2)
        TP_THROW(RuntimeError, "fractional_max_pool2d: kernel_size and output_size must have 2 elements");
    if (self.dim() != 3 && self.dim() != 4)
        TP_THROW(RuntimeError, "fractional_max_pool2d: expected 3D or 4D input");
    if (kernel_size[0] <= 0 || kernel_size[1] <= 0)
        TP_THROW(RuntimeError, "fractional_max_pool2d: kernel_size must be positive");
    const int64_t inputH = self.size(self.dim() - 2);
    const int64_t inputW = self.size(self.dim() - 1);
    if (output_size[0] + kernel_size[0] - 1 > inputH || output_size[1] + kernel_size[1] - 1 > inputW)
        TP_THROW(RuntimeError, "fractional_max_pool2d: kernel too large relative to input");
    fractional_pool_check(self, random_samples, 2, "fractional_max_pool2d");
    switch (self.dtype()) {
        case DType::Float32: return fractional_max_pool2d_cpu_impl<float, float>(self, kernel_size, output_size, random_samples);
        case DType::Float64: return fractional_max_pool2d_cpu_impl<double, double>(self, kernel_size, output_size, random_samples);
        case DType::Float16: return fractional_max_pool2d_cpu_impl<Half, float>(self, kernel_size, output_size, random_samples);
        case DType::BFloat16: return fractional_max_pool2d_cpu_impl<BFloat16, float>(self, kernel_size, output_size, random_samples);
        default: TP_THROW(TypeError, "fractional_max_pool2d: unsupported dtype");
    }
}

Tensor fractional_max_pool2d_backward_cpu(
        const Tensor& grad_output, const Tensor& self,
        const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& output_size, const Tensor& indices) {
    if (output_size.size() != 2)
        TP_THROW(RuntimeError, "fractional_max_pool2d_backward: output_size must have 2 elements");
    if (self.dim() != 3 && self.dim() != 4)
        TP_THROW(RuntimeError, "fractional_max_pool2d_backward: expected 3D or 4D input");
    switch (self.dtype()) {
        case DType::Float32: return fractional_max_pool2d_backward_cpu_impl<float>(grad_output, self, output_size, indices);
        case DType::Float64: return fractional_max_pool2d_backward_cpu_impl<double>(grad_output, self, output_size, indices);
        case DType::Float16: return fractional_max_pool2d_backward_cpu_impl<Half>(grad_output, self, output_size, indices);
        case DType::BFloat16: return fractional_max_pool2d_backward_cpu_impl<BFloat16>(grad_output, self, output_size, indices);
        default: TP_THROW(TypeError, "fractional_max_pool2d_backward: unsupported dtype");
    }
}

std::tuple<Tensor, Tensor> fractional_max_pool3d_cpu(
        const Tensor& self, const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& output_size, const Tensor& random_samples) {
    if (kernel_size.size() != 3 || output_size.size() != 3)
        TP_THROW(RuntimeError, "fractional_max_pool3d: kernel_size and output_size must have 3 elements");
    if (self.dim() != 4 && self.dim() != 5)
        TP_THROW(RuntimeError, "fractional_max_pool3d: expected 4D or 5D input");
    if (kernel_size[0] <= 0 || kernel_size[1] <= 0 || kernel_size[2] <= 0)
        TP_THROW(RuntimeError, "fractional_max_pool3d: kernel_size must be positive");
    const int64_t inputT = self.size(self.dim() - 3);
    const int64_t inputH = self.size(self.dim() - 2);
    const int64_t inputW = self.size(self.dim() - 1);
    // (strict, unlike the 2D variant which allows equality).
    if (output_size[0] + kernel_size[0] - 1 >= inputT ||
        output_size[1] + kernel_size[1] - 1 >= inputH ||
        output_size[2] + kernel_size[2] - 1 >= inputW)
        TP_THROW(RuntimeError, "fractional_max_pool3d: kernel too large relative to input");
    fractional_pool_check(self, random_samples, 3, "fractional_max_pool3d");
    switch (self.dtype()) {
        case DType::Float32: return fractional_max_pool3d_cpu_impl<float, float>(self, kernel_size, output_size, random_samples);
        case DType::Float64: return fractional_max_pool3d_cpu_impl<double, double>(self, kernel_size, output_size, random_samples);
        case DType::Float16: return fractional_max_pool3d_cpu_impl<Half, float>(self, kernel_size, output_size, random_samples);
        case DType::BFloat16: return fractional_max_pool3d_cpu_impl<BFloat16, float>(self, kernel_size, output_size, random_samples);
        default: TP_THROW(TypeError, "fractional_max_pool3d: unsupported dtype");
    }
}

Tensor fractional_max_pool3d_backward_cpu(
        const Tensor& grad_output, const Tensor& self,
        const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& output_size, const Tensor& indices) {
    if (output_size.size() != 3)
        TP_THROW(RuntimeError, "fractional_max_pool3d_backward: output_size must have 3 elements");
    if (self.dim() != 4 && self.dim() != 5)
        TP_THROW(RuntimeError, "fractional_max_pool3d_backward: expected 4D or 5D input");
    switch (self.dtype()) {
        case DType::Float32: return fractional_max_pool3d_backward_cpu_impl<float>(grad_output, self, output_size, indices);
        case DType::Float64: return fractional_max_pool3d_backward_cpu_impl<double>(grad_output, self, output_size, indices);
        case DType::Float16: return fractional_max_pool3d_backward_cpu_impl<Half>(grad_output, self, output_size, indices);
        case DType::BFloat16: return fractional_max_pool3d_backward_cpu_impl<BFloat16>(grad_output, self, output_size, indices);
        default: TP_THROW(TypeError, "fractional_max_pool3d_backward: unsupported dtype");
    }
}

TENSORPLAY_LIBRARY_IMPL(CPU, FractionalMaxPoolKernels) {
    m.impl("fractional_max_pool2d", fractional_max_pool2d_cpu);
    m.impl("fractional_max_pool2d_backward", fractional_max_pool2d_backward_cpu);
    m.impl("fractional_max_pool3d", fractional_max_pool3d_cpu);
    m.impl("fractional_max_pool3d_backward", fractional_max_pool3d_backward_cpu);
}

}  // namespace cpu
}  // namespace tensorplay
