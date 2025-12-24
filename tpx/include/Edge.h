#pragma once
#include <memory>
#include <cstdint>
#include "Macros.h"

namespace tensorplay {
namespace tpx {
class Node;

struct TENSORPLAY_API Edge {
    std::shared_ptr<Node> function;
    uint32_t input_nr;

    Edge(std::shared_ptr<Node> function, uint32_t input_nr)
        : function(std::move(function)), input_nr(input_nr) {}
    
    Edge() : function(nullptr), input_nr(0) {}
    
    bool is_valid() const { return function != nullptr; }
};
}
}
