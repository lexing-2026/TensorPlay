#include "CUDAReduce.cuh"

#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

using namespace tensorplay;
using namespace tensorplay::cuda;
using namespace tensorplay::cuda::reduction;

class PreservedDimensions : public TensorIteratorBase {
public:
    PreservedDimensions(const TensorIterator& original, int64_t columns,
                        bool reduce_fastest) : TensorIteratorBase(original) {
        // Preserve singleton axes that normal iterator construction merges.
        shape_.assign(70, 1);
        view_offsets_.assign(70, 0);
        operands_[0].stride_bytes.assign(70, 0);
        operands_[1].stride_bytes.assign(70, 0);
        shape_[0] = reduce_fastest ? 4 : 3;
        shape_[65] = reduce_fastest ? 129 : 10923;
        shape_[68] = columns;
        const int64_t rows = shape_[0] * shape_[65];
        operands_[1].stride_bytes[0] = sizeof(float) * (reduce_fastest ? 1 : columns);
        operands_[1].stride_bytes[65] = shape_[0] * operands_[1].stride_bytes[0];
        operands_[1].stride_bytes[68] = sizeof(float) * (reduce_fastest ? rows : 1);
        operands_[0].stride_bytes[68] = sizeof(float);
        operands_[0].stride_bytes[69] = columns * sizeof(float);
    }
};

void check_case(int64_t columns, bool reduce_fastest) {
    const int64_t rows = reduce_fastest ? 516 : 32769;
    const Device device(DeviceType::CUDA, 0);
    auto input = Tensor::empty({rows, columns}, DType::Float32, device);
    auto output = Tensor::empty({columns}, DType::Float32, device);
    auto indices = Tensor::empty({columns}, DType::Int64, device);
    auto output_view = output.as_strided({rows, columns}, {0, 1});
    auto original = TensorIterator::reduce_op(output_view, input);
    TensorIterator iter(PreservedDimensions(original, columns, reduce_fastest));
    const auto config = make_reduce_config<float, float, float>(iter);
    TP_CHECK(config.ndim == 70 && config.num_reduce_dims == 68,
             "dimension test must preserve runtime axes");
    TP_CHECK(config.output_vec_size == (reduce_fastest ? 1 : (columns % 4 == 0 ? 4 : columns % 2 == 0 ? 2 : 1)),
             "dimension test output vector width");
    std::vector<float> values(rows * columns);
    std::iota(values.begin(), values.end(), 0.f);
    checkCuda(cudaMemcpyAsync(input.data_ptr(), values.data(), values.size() * sizeof(float),
                              cudaMemcpyHostToDevice, getCurrentCUDAStream().stream()), "dimension test input");
    using Sum = SumOps<float, float, float>;
    if (reduce_fastest) {
        TP_CHECK(config.input_vec_size == 4, "dimension test input vector width");
        launch_reduce<float, float, float, Sum, 4, 4>(iter, {}, 0.f);
    } else {
        launch_reduce<float, float, float, Sum, 4, 1>(iter, {}, 0.f);
    }
    getCurrentCUDAStream().synchronize();
    std::vector<float> got(columns);
    checkCuda(cudaMemcpy(got.data(), output.data_ptr(), columns * sizeof(float),
                         cudaMemcpyDeviceToHost), "dimension test sum");
    for (int64_t column = 0; column < columns; ++column) {
        const float expected = reduce_fastest
            ? rows * (rows - 1) / 2.f + column * rows * rows
            : columns * rows * (rows - 1) / 2.f + column * rows;
        TP_CHECK(std::abs(got[column] - expected) <= std::abs(expected) * 1e-6f,
                 "runtime dimension sum mismatch");
    }
    using Maximum = ExtremumOps<float, float, float, true>;
    const ArgPair<float> identity{reduction_lower_bound<float>(), 0};
    if (reduce_fastest) {
        launch_reduce<float, ArgPair<float>, float, Maximum, 4, 4, int64_t>(
            iter, {}, identity, static_cast<int64_t*>(indices.data_ptr()));
    } else {
        launch_reduce<float, ArgPair<float>, float, Maximum, 4, 1, int64_t>(
            iter, {}, identity, static_cast<int64_t*>(indices.data_ptr()));
    }
    getCurrentCUDAStream().synchronize();
    std::vector<int64_t> got_indices(columns);
    checkCuda(cudaMemcpy(got.data(), output.data_ptr(), columns * sizeof(float),
                         cudaMemcpyDeviceToHost), "dimension test maximum");
    checkCuda(cudaMemcpy(got_indices.data(), indices.data_ptr(), columns * sizeof(int64_t),
                         cudaMemcpyDeviceToHost), "dimension test indices");
    for (int64_t column = 0; column < columns; ++column) {
        const float expected = reduce_fastest ? (column + 1) * rows - 1
                                             : (rows - 1) * columns + column;
        TP_CHECK(got[column] == expected && got_indices[column] == rows - 1,
                 "runtime dimension extremum mismatch");
    }
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    for (int64_t columns : {4, 6, 5}) check_case(columns, false);
    check_case(4, true);
    CUDAStreamGuard guard(getStreamFromPool());
    for (int64_t columns : {4, 6, 5}) check_case(columns, false);
    check_case(4, true);
    std::puts("runtime dimension reductions passed");
}
