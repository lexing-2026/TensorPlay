#include "Generator.h"
#include "Device.h"

#ifdef USE_CUDA
namespace tensorplay { namespace cuda {
void manual_seed(uint64_t seed);
void manual_seed_all(uint64_t seed);
} }
#endif

namespace tensorplay {

Generator::Generator(uint64_t seed_val) : initial_seed_(seed_val) {
    if (seed_val == 0) {
        // Generate a random seed
        std::random_device rd;
        initial_seed_ = rd();
    }
    mt_.seed(static_cast<unsigned int>(initial_seed_));
}

uint64_t Generator::manual_seed(uint64_t seed) {
    initial_seed_ = seed;
    mt_.seed(static_cast<unsigned int>(seed));
    return seed;
}

uint64_t Generator::seed() {
    std::random_device rd;
    return manual_seed(rd());
}

uint64_t Generator::initial_seed() const {
    return initial_seed_;
}

std::mt19937& Generator::engine() {
    return mt_;
}

Generator& default_generator() {
    static Generator* gen = new Generator(2023); 
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
