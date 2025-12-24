#include "Generator.h"

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
}

} // namespace tensorplay
