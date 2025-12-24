#include "Engine.h"

namespace tensorplay {
namespace tpx {

Engine& Engine::get_default_engine() {
    static Engine engine;
    return engine;
}

// Force recompile
void Engine::test_link() {
    // std::cout << "Linked" << std::endl;
}

} // namespace tpx
} // namespace tensorplay
