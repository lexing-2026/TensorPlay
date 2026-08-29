// Exact low-precision foreach bodies for the native optimizer MTA path.
//
// This file is included by OptimizerMTA.cuh after the common metadata and
// load/store helpers have been declared.  Keeping these bodies in a separate
// foreach functors, while still allowing the CUDA compiler to inline the
// body into the one MTA kernel template.

// result back to the tensor scalar type after every operation.  The reduced
// opmath bodies in OptimizerMTA.cuh intentionally keep everything in opmath,
// which is the right behavior for the explicit fused API but not for the
// ordinary foreach optimizer path on Half/BFloat16.  Exact bodies below use
// this helper at each foreach boundary while retaining one horizontal MTA
// launch.
template <typename scalar_t, typename math_t>
__device__ __forceinline__ math_t round_to_scalar(math_t value) {
    return static_cast<math_t>(static_cast<scalar_t>(value));
}

// These helpers reuse the CUDA foreach pointwise implementation.  In
// particular, addcmul(alpha=1) is the only path that fuses the two tensor
// operands with the input; addcdiv(alpha=1) deliberately remains divide then
// add.  Keeping the distinction here avoids accidentally changing the
// low-precision rounding boundary while the optimizer stays one MTA launch.
template <typename scalar_t, typename math_t>
__device__ __forceinline__ math_t exact_addcmul(
        math_t input, math_t tensor1, math_t tensor2, math_t alpha) {
    if (alpha == math_t(1)) {
        return round_to_scalar<scalar_t>(fma(tensor1, tensor2, input));
    }
    return round_to_scalar<scalar_t>(
        fma(alpha, tensor1 * tensor2, input));
}

template <typename scalar_t, typename math_t>
__device__ __forceinline__ math_t exact_addcdiv(
        math_t input, math_t tensor1, math_t tensor2, math_t alpha) {
    const math_t quotient = tensor1 / tensor2;
    if (alpha == math_t(1)) {
        return round_to_scalar<scalar_t>(input + quotient);
    }
    return round_to_scalar<scalar_t>(fma(alpha, quotient, input));
}

template <typename scalar_t, typename math_t>
__device__ __forceinline__ math_t exact_add(
        math_t input, math_t value, math_t alpha) {
    return round_to_scalar<scalar_t>(input + alpha * value);
}

template <typename scalar_t, typename math_t>
__device__ __forceinline__ math_t exact_lerp(
        math_t start, math_t end, math_t weight) {
    const math_t value = fabs(weight) < math_t(0.5)
        ? start + weight * (end - start)
        : end - (end - start) * (math_t(1) - weight);
    return round_to_scalar<scalar_t>(value);
}

template <typename scalar_t, typename math_t, bool Centered, bool HasMomentum>
struct RmspropExactBody {
    math_t lr, alpha, one_minus_alpha, eps, weight_decay, momentum;
    bool maximize;

    template <int Depth, typename S, typename Metadata>
    __device__ __forceinline__ void prepare(const Metadata&, int) {}
    __device__ __forceinline__ bool should_load(int) const { return true; }
    __device__ __forceinline__ bool should_store(int depth) const {
        return depth != 1;
    }
    __device__ __forceinline__ void operator()(
            math_t values[][kILP], int lane) const {
        math_t g = values[1][lane];
        if (maximize) g = round_to_scalar<scalar_t>(-g);
        const math_t p = values[0][lane];
        if (weight_decay != math_t(0)) {
            g = round_to_scalar<scalar_t>(g + weight_decay * p);
        }

        math_t square = round_to_scalar<scalar_t>(
            values[2][lane] * alpha);
        square = exact_addcmul<scalar_t>(
            square, g, g, one_minus_alpha);
        values[2][lane] = square;

        math_t average = square;
        if constexpr (Centered) {
            constexpr int kGradAvgIndex = 3;
            const math_t old_mean = values[kGradAvgIndex][lane];
            const math_t mean = exact_lerp<scalar_t>(
                old_mean, g, one_minus_alpha);
            values[kGradAvgIndex][lane] = mean;
            average = exact_addcmul<scalar_t>(
                square, mean, mean, math_t(-1));
        }

        math_t denominator = round_to_scalar<scalar_t>(sqrt(average));
        denominator = round_to_scalar<scalar_t>(denominator + eps);
        if constexpr (HasMomentum) {
            constexpr int kMomentumIndex = Centered ? 4 : 3;
            math_t buffer = round_to_scalar<scalar_t>(
                values[kMomentumIndex][lane] * momentum);
            buffer = exact_addcdiv<scalar_t>(
                buffer, g, denominator, math_t(1));
            values[kMomentumIndex][lane] = buffer;
            values[0][lane] = exact_add<scalar_t>(p, buffer, -lr);
        } else {
            values[0][lane] = exact_addcdiv<scalar_t>(
                p, g, denominator, -lr);
        }
    }
};

template <typename scalar_t, typename math_t>
struct AdadeltaExactBody {
    math_t lr, rho, one_minus_rho, eps, weight_decay;
    bool maximize;

    template <int Depth, typename S, typename Metadata>
    __device__ __forceinline__ void prepare(const Metadata&, int) {}
    __device__ __forceinline__ bool should_load(int) const { return true; }
    __device__ __forceinline__ bool should_store(int depth) const {
        return depth != 1;
    }
    __device__ __forceinline__ void operator()(
            math_t values[][kILP], int lane) const {
        math_t g = values[1][lane];
        if (maximize) g = round_to_scalar<scalar_t>(-g);
        const math_t p = values[0][lane];
        if (weight_decay != math_t(0)) {
            g = round_to_scalar<scalar_t>(g + weight_decay * p);
        }

        math_t square = round_to_scalar<scalar_t>(
            values[2][lane] * rho);
        square = exact_addcmul<scalar_t>(
            square, g, g, one_minus_rho);
        values[2][lane] = square;

        math_t std = round_to_scalar<scalar_t>(sqrt(
            round_to_scalar<scalar_t>(square + eps)));
        math_t delta = round_to_scalar<scalar_t>(sqrt(
            round_to_scalar<scalar_t>(values[3][lane] + eps)));
        delta = round_to_scalar<scalar_t>(delta / std);
        delta = round_to_scalar<scalar_t>(delta * g);

        math_t acc = round_to_scalar<scalar_t>(
            values[3][lane] * rho);
        acc = exact_addcmul<scalar_t>(
            acc, delta, delta, one_minus_rho);
        values[3][lane] = acc;
        const math_t update = round_to_scalar<scalar_t>(-lr * delta);
        values[0][lane] = exact_add<scalar_t>(p, update, math_t(1));
    }
};

template <typename scalar_t, typename math_t>
struct AdagradExactBody {
    math_t corrected_lr, eps, weight_decay;
    bool maximize;

    template <int Depth, typename S, typename Metadata>
    __device__ __forceinline__ void prepare(
            const Metadata& metadata, int tensor_index) {
        // Host metadata contains the Python double corrected learning rate,
        // rather than a step value that would be recomputed in float.
        corrected_lr = static_cast<math_t>(
            metadata.step_metadata.host.step_sizes[tensor_index]);
    }
    __device__ __forceinline__ bool should_load(int) const { return true; }
    __device__ __forceinline__ bool should_store(int depth) const {
        return depth != 1;
    }
    __device__ __forceinline__ void operator()(
            math_t values[][kILP], int lane) const {
        math_t g = values[1][lane];
        if (maximize) g = round_to_scalar<scalar_t>(-g);
        const math_t p = values[0][lane];
        if (weight_decay != math_t(0)) {
            g = round_to_scalar<scalar_t>(g + weight_decay * p);
        }
        const math_t sum = round_to_scalar<scalar_t>(
            exact_addcmul<scalar_t>(
                values[2][lane], g, g, math_t(1)));
        values[2][lane] = sum;
        math_t denominator = round_to_scalar<scalar_t>(sqrt(sum));
        denominator = round_to_scalar<scalar_t>(denominator + eps);
        const math_t numerator = round_to_scalar<scalar_t>(
            corrected_lr * g);
        values[0][lane] = exact_addcdiv<scalar_t>(
            p, numerator, denominator, math_t(1));
    }
};

template <typename scalar_t, typename math_t>
struct AdamaxExactBody {
    math_t beta1, beta2, one_minus_beta1, eps, weight_decay;
    bool maximize;
    math_t step_size;

    template <int Depth, typename S, typename Metadata>
    __device__ __forceinline__ void prepare(
            const Metadata& metadata, int tensor_index) {
        // This is the negative Python scalar-list coefficient used by the
        // final foreach_addcdiv_ call.
        step_size = static_cast<math_t>(
            metadata.step_metadata.host.step_sizes[tensor_index]);
    }
    __device__ __forceinline__ bool should_load(int) const { return true; }
    __device__ __forceinline__ bool should_store(int depth) const {
        return depth != 1;
    }
    __device__ __forceinline__ void operator()(
            math_t values[][kILP], int lane) const {
        math_t g = values[1][lane];
        if (maximize) g = round_to_scalar<scalar_t>(-g);
        const math_t p = values[0][lane];
        if (weight_decay != math_t(0)) {
            g = round_to_scalar<scalar_t>(g + weight_decay * p);
        }
        values[2][lane] = exact_lerp<scalar_t>(
            values[2][lane], g, one_minus_beta1);
        math_t inf = round_to_scalar<scalar_t>(
            values[3][lane] * beta2);
        const math_t abs_g = round_to_scalar<scalar_t>(fabs(g));
        const math_t candidate = round_to_scalar<scalar_t>(abs_g + eps);
        inf = inf < candidate ? candidate : inf;
        values[3][lane] = round_to_scalar<scalar_t>(inf);
        values[0][lane] = round_to_scalar<scalar_t>(
            fma(step_size, values[2][lane] / values[3][lane], p));
    }
};

template <typename scalar_t, typename math_t>
struct RpropExactBody {
    math_t step_size_min, step_size_max, etaminus, etaplus;
    bool maximize;

    template <int Depth, typename S, typename Metadata>
    __device__ __forceinline__ void prepare(const Metadata&, int) {}
    __device__ __forceinline__ bool should_load(int) const { return true; }
    __device__ __forceinline__ bool should_store(int depth) const {
        return depth != 1;
    }
    __device__ __forceinline__ void operator()(
            math_t values[][kILP], int lane) const {
        math_t g = values[1][lane];
        if (maximize) g = round_to_scalar<scalar_t>(-g);

        // _foreach_mul(grads, prevs) writes a low-precision temporary before
        // sign() is evaluated.  Keep the direction code separately so the
        // etaminus comparison is independent of scalar representation.
        const math_t raw_grad = values[1][lane];
        math_t product = round_to_scalar<scalar_t>(
            raw_grad * values[2][lane]);
        if (maximize) product = round_to_scalar<scalar_t>(-product);
        const int direction = product > math_t(0) ? 1 :
            (product < math_t(0) ? -1 : 0);
        const math_t sign = direction > 0 ?
            round_to_scalar<scalar_t>(etaplus) :
            (direction < 0 ? round_to_scalar<scalar_t>(etaminus) :
                             math_t(1));
        math_t step_size = round_to_scalar<scalar_t>(
            values[3][lane] * sign);
        step_size = round_to_scalar<scalar_t>(fmin(
            step_size_max, fmax(step_size_min, step_size)));
        values[3][lane] = step_size;

        math_t stored_grad = maximize
            ? round_to_scalar<scalar_t>(-raw_grad) : raw_grad;
        const math_t masked_grad = direction < 0 ? math_t(0) : stored_grad;
        const math_t grad_sign = masked_grad > math_t(0) ? math_t(1) :
            (masked_grad < math_t(0) ? math_t(-1) : math_t(0));
        values[0][lane] = exact_addcmul<scalar_t>(
            values[0][lane], grad_sign, step_size, math_t(-1));
        values[2][lane] = round_to_scalar<scalar_t>(masked_grad);
    }
};

template <typename scalar_t, typename math_t>
struct NadamExactBody {
    math_t beta1, beta2, one_minus_beta1, one_minus_beta2;
    math_t eps, weight_decay, lr, adamw_factor;
    bool maximize, decoupled_weight_decay;
    double lr_value, beta1_value, beta2_value, momentum_decay_value;
    math_t step_size_grads, step_size_expavg, correction2_sqrt;

    template <int Depth, typename S, typename Metadata>
    __device__ __forceinline__ void prepare(
            const Metadata& metadata, int tensor_index) {
        // Keep the host step and the already-rounded CPU mu_product in the
        // two metadata arrays.  The scalar-list coefficients and bias
        // opmath type by the foreach kernel; reproduce that conversion here.
        const double step = metadata.step_metadata.host.step_sizes[
            tensor_index];
        const double mu_product = metadata.step_metadata.host.correction2_sqrts[
            tensor_index];
        const double mu = beta1_value * (1.0 - 0.5 * pow(
            0.96, step * momentum_decay_value));
        const double mu_next = beta1_value * (1.0 - 0.5 * pow(
            0.96, (step + 1.0) * momentum_decay_value));
        step_size_grads = static_cast<math_t>(
            -lr_value * (1.0 - mu) / (1.0 - mu_product));
        step_size_expavg = static_cast<math_t>(
            -lr_value * mu_next /
            (1.0 - mu_product * mu_next));
        correction2_sqrt = static_cast<math_t>(sqrt(
            1.0 - pow(beta2_value, step)));
    }
    __device__ __forceinline__ bool should_load(int) const { return true; }
    __device__ __forceinline__ bool should_store(int depth) const {
        return depth != 1;
    }
    __device__ __forceinline__ void operator()(
            math_t values[][kILP], int lane) const {
        math_t g = values[1][lane];
        if (maximize) g = round_to_scalar<scalar_t>(-g);
        math_t p = values[0][lane];
        if (weight_decay != math_t(0)) {
            if (decoupled_weight_decay) {
                p = round_to_scalar<scalar_t>(p * adamw_factor);
            } else {
                g = round_to_scalar<scalar_t>(g + weight_decay * p);
            }
        }

        values[2][lane] = exact_lerp<scalar_t>(
            values[2][lane], g, one_minus_beta1);
        math_t second = round_to_scalar<scalar_t>(
            values[3][lane] * beta2);
        second = exact_addcmul<scalar_t>(
            second, g, g, one_minus_beta2);
        values[3][lane] = second;

        math_t denominator = round_to_scalar<scalar_t>(sqrt(second));
        denominator = round_to_scalar<scalar_t>(
            denominator / correction2_sqrt);
        denominator = round_to_scalar<scalar_t>(denominator + eps);
        p = exact_addcdiv<scalar_t>(
            p, g, denominator, step_size_grads);
        p = exact_addcdiv<scalar_t>(
            p, values[2][lane], denominator, step_size_expavg);
        values[0][lane] = p;
    }
};

template <typename scalar_t, typename math_t>
struct RadamExactBody {
    math_t beta1, beta2, one_minus_beta1, one_minus_beta2;
    math_t eps, weight_decay, lr, adamw_factor;
    bool maximize, decoupled_weight_decay;
    math_t unrectified_step_size, rectified_coefficient;

    template <int Depth, typename S, typename Metadata>
    __device__ __forceinline__ void prepare(
            const Metadata& metadata, int tensor_index) {
        unrectified_step_size = static_cast<math_t>(
            metadata.step_metadata.host.step_sizes[tensor_index]);
        rectified_coefficient = static_cast<math_t>(
            metadata.step_metadata.host.correction2_sqrts[tensor_index]);
    }
    __device__ __forceinline__ bool should_load(int) const { return true; }
    __device__ __forceinline__ bool should_store(int depth) const {
        return depth != 1;
    }
    __device__ __forceinline__ void operator()(
            math_t values[][kILP], int lane) const {
        math_t g = values[1][lane];
        if (maximize) g = round_to_scalar<scalar_t>(-g);
        math_t p = values[0][lane];
        if (weight_decay != math_t(0)) {
            if (decoupled_weight_decay) {
                p = round_to_scalar<scalar_t>(p * adamw_factor);
            } else {
                g = round_to_scalar<scalar_t>(g + weight_decay * p);
            }
        }

        values[2][lane] = exact_lerp<scalar_t>(
            values[2][lane], g, one_minus_beta1);
        math_t second = round_to_scalar<scalar_t>(
            values[3][lane] * beta2);
        second = exact_addcmul<scalar_t>(
            second, g, g, one_minus_beta2);
        values[3][lane] = second;

        // sqrt(v), add eps, divide by the rectified coefficient, reciprocal,
        // then add the unrectified coefficient.
        math_t buffer = round_to_scalar<scalar_t>(sqrt(second));
        buffer = round_to_scalar<scalar_t>(buffer + eps);
        buffer = round_to_scalar<scalar_t>(
            buffer / rectified_coefficient);
        buffer = round_to_scalar<scalar_t>(math_t(1) / buffer);
        buffer = round_to_scalar<scalar_t>(
            buffer + unrectified_step_size);
        values[0][lane] = exact_addcmul<scalar_t>(
            p, values[2][lane], buffer, math_t(1));
    }
};
