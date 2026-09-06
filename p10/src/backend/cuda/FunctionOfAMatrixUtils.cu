#include "Tensor.h"
#include "Dispatcher.h"
#include "DType.h"
#include "TensorIterator.h"
#include "Exception.h"
#include "CUDALoops.cuh"
#include "Complex.h"
#include "CUDARuntime.h"

#include <cuda_runtime.h>

#include <vector>

namespace tensorplay {
namespace cuda {
namespace {

template <typename T>
struct linear_combination_coefficient_type {
    using type = T;
};

template <typename T>
struct linear_combination_coefficient_type<tensorplay::complex<T>> {
    using type = T;
};

template <typename scalar_t>
void compute_linear_combination_kernel(
        TensorIterator& iter,
        int64_t input_stride,
        int64_t coefficient_stride,
        int64_t summations) {
    if (iter.numel() == 0) return;
    if (!iter.can_use_32bit_indexing()) {
        for (auto& sub_iter : iter.with_32bit_indexing()) {
            compute_linear_combination_kernel<scalar_t>(
                sub_iter, input_stride, coefficient_stride, summations);
        }
        return;
    }

    using coefficient_t = typename linear_combination_coefficient_type<scalar_t>::type;
    auto offset_calculator = make_offset_calculator<3>(iter);
    char* output_data = static_cast<char*>(iter.data_ptr(0));
    char* input_data = static_cast<char*>(iter.data_ptr(1));
    char* coefficient_data = static_cast<char*>(iter.data_ptr(2));

    auto loop = [=] __host__ __device__(int index) {
        const auto offsets = offset_calculator.get(static_cast<uint32_t>(index));
        auto* output = reinterpret_cast<scalar_t*>(output_data + offsets[0]);
        const auto* input = reinterpret_cast<const scalar_t*>(input_data + offsets[1]);
        const auto* coefficients = reinterpret_cast<const coefficient_t*>(
            coefficient_data + offsets[2]);
        for (int64_t j = 0; j < summations; ++j) {
            if constexpr (tensorplay::is_complex_type_v<scalar_t>) {
                const scalar_t coefficient_value(
                    coefficients[j * coefficient_stride], coefficient_t(0));
                *output += input[j * input_stride] * coefficient_value;
            } else {
                *output += input[j * input_stride] * coefficients[j * coefficient_stride];
            }
        }
    };

    launch_legacy_kernel<kLoopNumThreads, kLoopThreadWorkSize>(
        iter.numel(), loop);
}

void compute_linear_combination_dispatch(
        TensorIterator& iter,
        int64_t input_stride,
        int64_t coefficient_stride,
        int64_t summations) {
#define TP_LINEAR_COMBINATION_CASE(ctype, name) \
    case DType::name: \
        compute_linear_combination_kernel<ctype>( \
            iter, input_stride, coefficient_stride, summations); \
        break;
    switch (iter.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_LINEAR_COMBINATION_CASE)
        case DType::ComplexHalf:
            compute_linear_combination_kernel<tensorplay::complex<Half>>(
                iter, input_stride, coefficient_stride, summations);
            break;
        case DType::ComplexFloat:
            compute_linear_combination_kernel<tensorplay::complex<float>>(
                iter, input_stride, coefficient_stride, summations);
            break;
        case DType::ComplexDouble:
            compute_linear_combination_kernel<tensorplay::complex<double>>(
                iter, input_stride, coefficient_stride, summations);
            break;
        case DType::BComplex32:
            compute_linear_combination_kernel<tensorplay::complex<BFloat16>>(
                iter, input_stride, coefficient_stride, summations);
            break;
        default:
            TP_THROW(TypeError, "_compute_linear_combination: unsupported dtype");
    }
#undef TP_LINEAR_COMBINATION_CASE
}

Tensor& compute_linear_combination_out_cuda(
        const Tensor& input, const Tensor& coefficients, Tensor& output) {
    const int64_t output_first_dim_size = coefficients.size(0);
    const int64_t input_first_dim_size = coefficients.size(1);

    Tensor output_broadcast = output.unsqueeze(1);
    std::vector<int64_t> output_sizes =
        static_cast<std::vector<int64_t>>(output_broadcast.shape());
    std::vector<int64_t> output_strides =
        static_cast<std::vector<int64_t>>(output_broadcast.strides());
    output_sizes[1] = 1;
    output_strides[1] = 0;
    Tensor output_restrided = output.as_strided(output_sizes, output_strides);

    Tensor input_broadcast = input.unsqueeze(0);
    std::vector<int64_t> input_sizes =
        static_cast<std::vector<int64_t>>(input_broadcast.shape());
    std::vector<int64_t> input_strides =
        static_cast<std::vector<int64_t>>(input_broadcast.strides());
    input_sizes[1] = 1;
    input_strides[1] = 0;
    Tensor input_restrided = input.as_strided(input_sizes, input_strides);

    std::vector<int64_t> coefficient_sizes(input.dim() + 1, 1);
    coefficient_sizes[0] = output_first_dim_size;
    std::vector<int64_t> coefficient_strides(input.dim() + 1, 0);
    coefficient_strides[0] = coefficients.stride(0);
    Tensor coefficients_restrided = coefficients.as_strided(
        coefficient_sizes, coefficient_strides);

    TensorIterator iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .add_output(output_restrided)
        .add_input(input_restrided)
        .add_input(coefficients_restrided)
        .build();

    compute_linear_combination_dispatch(
        iter, input.stride(0), coefficients.stride(1), input_first_dim_size);
    return output;
}

}  // namespace

Tensor compute_linear_combination_cuda(
        const Tensor& input, const Tensor& coefficients) {
    if (input.dim() <= 0 || input.numel() == 0) {
        TP_THROW(RuntimeError, "Empty tensor not supported");
    }
    std::vector<int64_t> output_shape =
        static_cast<std::vector<int64_t>>(input.shape());
    output_shape[0] = coefficients.size(0);
    Tensor output = Tensor::zeros(output_shape, input.dtype(), input.device());
    compute_linear_combination_out_cuda(input, coefficients, output);
    return output;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, FunctionOfAMatrixUtils) {
    m.impl("_compute_linear_combination", compute_linear_combination_cuda);
    m.impl("_compute_linear_combination.out", compute_linear_combination_out_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
