#pragma once

#include "Autograd.h"
#include "SavedVariable.h"

namespace tensorplay {
namespace tpx {

struct CtcLossBackward : Node {
    SavedVariable log_probs_, targets_, input_lengths_, target_lengths_, nll_, alpha_;
    int64_t blank_;
    bool zero_infinity_;
    CtcLossBackward(Tensor log_probs, Tensor targets, Tensor input_lengths,
                    Tensor target_lengths, int64_t blank, bool zero_infinity,
                    Tensor neg_log_likelihood, Tensor log_alpha);
    size_t num_inputs() const override { return 2; }
    variable_list apply(variable_list&& inputs) override;
    void release_variables() override;
};

struct CtcLossBackwardBackward : Node {
    SavedVariable grad_output_, log_probs_, targets_, input_lengths_, target_lengths_, nll_, alpha_;
    int64_t blank_;
    bool zero_infinity_;
    CtcLossBackwardBackward(Tensor grad_output, Tensor log_probs, Tensor targets,
                            Tensor input_lengths, Tensor target_lengths,
                            Tensor neg_log_likelihood, Tensor log_alpha,
                            int64_t blank, bool zero_infinity);
    variable_list apply(variable_list&& inputs) override;
    void release_variables() override;
};

}  // namespace tpx
}  // namespace tensorplay
