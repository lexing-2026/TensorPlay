#pragma once

// Device rule of iterator-backed operators.
//
// The common device is the first non-CPU device among the operands.  A
// single zero-dim CPU input may accompany a non-CPU computation: its value
// folds into the kernel as a scalar.  Every other defined tensor, outputs
// included, must sit on the common device.

#include "Device.h"
#include "Exception.h"
#include "Tensor.h"

#include <optional>
#include <vector>

namespace tensorplay::detail {

class IteratorDeviceCheck {
public:
    void add(const Tensor& tensor, bool is_output) {
        if (!tensor.defined()) return;
        entries_.push_back({tensor.device(), tensor.dim() == 0, is_output});
    }

    void add(const std::optional<Tensor>& tensor, bool is_output) {
        if (tensor.has_value()) add(*tensor, is_output);
    }

    void add(const std::vector<Tensor>& tensors, bool is_output) {
        for (const auto& tensor : tensors) add(tensor, is_output);
    }

    void add(const std::vector<std::optional<Tensor>>& tensors, bool is_output) {
        for (const auto& tensor : tensors) add(tensor, is_output);
    }

    void check() const {
        Device common(DeviceType::CPU);
        for (const auto& entry : entries_) {
            if (!entry.device.is_cpu()) {
                common = entry.device;
                break;
            }
        }
        bool scalar_taken = false;
        for (const auto& entry : entries_) {
            if (!common.is_cpu() && !scalar_taken && !entry.is_output &&
                entry.zero_dim && entry.device.is_cpu()) {
                scalar_taken = true;
                continue;
            }
            if (entry.device != common) {
                TP_THROW(DeviceMismatchError,
                         "Expected all tensors to be on the same device, but found "
                         "at least two devices, " + common.toString() + " and " +
                         entry.device.toString() + "!");
            }
        }
    }

private:
    struct Entry {
        Device device;
        bool zero_dim;
        bool is_output;
    };
    std::vector<Entry> entries_;
};

} // namespace tensorplay::detail
