#include "Generator.h"
#include "Device.h"
#include "DType.h"
#include "Tensor.h"
#include "Exception.h"

#include <algorithm>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

#ifdef USE_CUDA
namespace tensorplay { namespace cuda {
void manual_seed(uint64_t seed);
void manual_seed_all(uint64_t seed);
} }
#endif

namespace tensorplay {

namespace {

// POD layout mirrors torch's CPUGeneratorImplState so RNG states are
// byte-compatible with torch.get_rng_state / torch.set_rng_state (5056 bytes).
// The legacy state array holds 64-bit entries even though the engine now uses
// 32-bit ones; torch keeps the same quirk for checkpoint compatibility.
struct CPUGeneratorStateLegacy {
    uint64_t the_initial_seed;
    int left;
    int seeded;
    uint64_t next;
    uint64_t state[MERSENNE_STATE_N];

    double normal_x;
    double normal_y;
    double normal_rho;
    int normal_is_valid;
};

struct CPUGeneratorState {
    CPUGeneratorStateLegacy legacy_pod;
    float next_float_normal_sample;
    bool is_next_float_normal_sample_valid;
};

static_assert(std::is_standard_layout_v<CPUGeneratorState>, "CPUGeneratorState must be a POD type");

inline uint64_t make64BitsFrom32Bits(uint32_t hi, uint32_t lo) {
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

} // namespace

Generator::Generator(uint64_t seed_val) : engine_(seed_val) {}

uint64_t Generator::manual_seed(uint64_t seed) {
    next_float_normal_sample_.reset();
    next_double_normal_sample_.reset();
    engine_ = mt19937_engine(seed);
    return seed;
}

uint64_t Generator::seed() {
    auto random = detail::getNonDeterministicRandom();
    manual_seed(random);
    return random;
}

uint64_t Generator::initial_seed() const {
    return engine_.seed();
}

uint64_t Generator::current_seed() const {
    return engine_.seed();
}

uint32_t Generator::random() {
    return engine_();
}

uint64_t Generator::random64() {
    uint32_t random1 = engine_();
    uint32_t random2 = engine_();
    return make64BitsFrom32Bits(random1, random2);
}

std::optional<float> Generator::next_float_normal_sample() const {
    return next_float_normal_sample_;
}

std::optional<double> Generator::next_double_normal_sample() const {
    return next_double_normal_sample_;
}

void Generator::set_next_float_normal_sample(std::optional<float> randn) {
    next_float_normal_sample_ = randn;
}

void Generator::set_next_double_normal_sample(std::optional<double> randn) {
    next_double_normal_sample_ = randn;
}

Tensor Generator::get_state() const {
    constexpr size_t size = sizeof(CPUGeneratorState);

    Tensor state_tensor(std::vector<int64_t>{static_cast<int64_t>(size)}, DType::UInt8,
                        Device(DeviceType::CPU));
    auto rng_state = state_tensor.data_ptr<uint8_t>();

    CPUGeneratorState accum_state{};
    auto rng_data = engine_.data();
    accum_state.legacy_pod.the_initial_seed = rng_data.seed_;
    accum_state.legacy_pod.left = rng_data.left_;
    accum_state.legacy_pod.seeded = rng_data.seeded_;
    accum_state.legacy_pod.next = rng_data.next_;
    std::copy(rng_data.state_.begin(), rng_data.state_.end(), std::begin(accum_state.legacy_pod.state));
    accum_state.legacy_pod.normal_x = 0.0; // unused, kept for layout compatibility
    accum_state.legacy_pod.normal_rho = 0.0;
    accum_state.legacy_pod.normal_is_valid = false;
    accum_state.legacy_pod.normal_y = 0.0;
    accum_state.next_float_normal_sample = 0.0f;
    accum_state.is_next_float_normal_sample_valid = false;
    if (next_double_normal_sample_) {
        accum_state.legacy_pod.normal_is_valid = true;
        accum_state.legacy_pod.normal_y = *(next_double_normal_sample_);
    }
    if (next_float_normal_sample_) {
        accum_state.is_next_float_normal_sample_valid = true;
        accum_state.next_float_normal_sample = *(next_float_normal_sample_);
    }

    std::memcpy(rng_state, &accum_state, size);
    return state_tensor;
}

void Generator::set_state(const Tensor& new_state) {
    constexpr size_t size = sizeof(CPUGeneratorState);

    if (new_state.device().type() != DeviceType::CPU || new_state.dtype() != DType::UInt8) {
        TP_THROW(RuntimeError, "RNG state must be a CPU UInt8 tensor");
    }
    if (static_cast<size_t>(new_state.numel()) != size) {
        TP_THROW(RuntimeError,
                 "Expected a CPUGeneratorState of size " + std::to_string(size) +
                 " but found the input RNG state size to be " + std::to_string(new_state.numel()));
    }

    const auto* rng_state = reinterpret_cast<const CPUGeneratorState*>(new_state.data_ptr<uint8_t>());
    const auto& legacy_pod = rng_state->legacy_pod;

    mt19937_data_pod rng_data{};
    std::copy(std::begin(legacy_pod.state), std::end(legacy_pod.state), rng_data.state_.begin());
    rng_data.seed_ = legacy_pod.the_initial_seed;
    rng_data.left_ = legacy_pod.left;
    rng_data.seeded_ = legacy_pod.seeded;
    rng_data.next_ = static_cast<uint32_t>(legacy_pod.next);
    mt19937_engine engine;
    engine.set_data(rng_data);
    if (!engine.is_valid()) {
        TP_THROW(RuntimeError, "Invalid mt19937 state");
    }
    engine_ = engine;
    next_float_normal_sample_ = rng_state->is_next_float_normal_sample_valid
        ? std::optional<float>(rng_state->next_float_normal_sample)
        : std::optional<float>();
    next_double_normal_sample_ = legacy_pod.normal_is_valid
        ? std::optional<double>(legacy_pod.normal_y)
        : std::optional<double>();
}

namespace detail {

// Nondeterministic seed from the platform entropy source (torch uses
// c10::detail::getNonDeterministicRandom for its default generators).
uint64_t getNonDeterministicRandom() {
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) | rd();
}

} // namespace detail

Generator& default_generator() {
    static Generator* gen = new Generator(detail::getNonDeterministicRandom());
    return *gen;
}

void manual_seed(uint64_t seed) {
    default_generator().manual_seed(seed);
#ifdef USE_CUDA
    // torch semantics: torch.manual_seed seeds every device RNG. The CUDA
    // backend applies this lazily (never initializing CUDA) and skips it in
    // bad-fork children.
    tensorplay::cuda::manual_seed_all(seed);
#endif
}

} // namespace tensorplay
