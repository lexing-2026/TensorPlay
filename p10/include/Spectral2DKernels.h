#pragma once

#include "Tensor.h"

#include <optional>
#include <string>
#include <vector>

namespace tensorplay {
namespace cpu {

Tensor fft_fft2_cpu(const Tensor& self, std::optional<std::vector<int64_t>> s,
                   const std::vector<int64_t>& dim, std::string norm);
Tensor fft_ifft2_cpu(const Tensor& self, std::optional<std::vector<int64_t>> s,
                    const std::vector<int64_t>& dim, std::string norm);
Tensor fft_rfft2_cpu(const Tensor& self, std::optional<std::vector<int64_t>> s,
                    const std::vector<int64_t>& dim, std::string norm);
Tensor fft_irfft2_cpu(const Tensor& self, std::optional<std::vector<int64_t>> s,
                     const std::vector<int64_t>& dim, std::string norm);

}  // namespace cpu

namespace cuda {

Tensor fft_fft2_cuda(const Tensor& self, std::optional<std::vector<int64_t>> s,
                    const std::vector<int64_t>& dim, std::string norm);
Tensor fft_ifft2_cuda(const Tensor& self, std::optional<std::vector<int64_t>> s,
                     const std::vector<int64_t>& dim, std::string norm);
Tensor fft_rfft2_cuda(const Tensor& self, std::optional<std::vector<int64_t>> s,
                     const std::vector<int64_t>& dim, std::string norm);
Tensor fft_irfft2_cuda(const Tensor& self, std::optional<std::vector<int64_t>> s,
                      const std::vector<int64_t>& dim, std::string norm);

}  // namespace cuda
}  // namespace tensorplay
