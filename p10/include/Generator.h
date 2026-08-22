#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include "Macros.h"
#include "MT19937RNGEngine.h"

namespace tensorplay {

class Tensor;

// Matches c10::default_rng_seed_val: the seed torch.Generator() starts from.
constexpr uint64_t default_rng_seed_val = 67280421310721ULL;

class P10_API Generator {
public:
    explicit Generator(uint64_t seed_val = default_rng_seed_val);

    // Seeds the engine (resetting any cached normal sample) and returns the seed.
    uint64_t manual_seed(uint64_t seed);

    // Reseeds with a nondeterministic value and returns it.
    uint64_t seed();

    uint64_t initial_seed() const;
    uint64_t current_seed() const;

    // Raw engine draws; the distribution layer consumes these directly.
    uint32_t random();
    uint64_t random64();

    // Box-Muller produces samples in pairs; the unconsumed half is cached here
    // and participates in get_state/set_state.
    std::optional<float> next_float_normal_sample() const;
    std::optional<double> next_double_normal_sample() const;
    void set_next_float_normal_sample(std::optional<float> randn);
    void set_next_double_normal_sample(std::optional<double> randn);

    // Serializes the full engine state to a CPU UInt8 tensor using the same
    // POD layout as torch.get_rng_state (5056 bytes).
    Tensor get_state() const;
    void set_state(const Tensor& new_state);

private:
    mt19937_engine engine_;
    std::optional<float> next_float_normal_sample_;
    std::optional<double> next_double_normal_sample_;
};

namespace detail {
P10_API uint64_t getNonDeterministicRandom();
} // namespace detail

P10_API Generator& default_generator();
P10_API void manual_seed(uint64_t seed);

} // namespace tensorplay
