#include "Fusion.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

namespace tensorplay {
namespace stax {

// --- Core Passes ---

class DeadCodeElimination : public Pass {
public:
    std::string name() const override { return "DeadCodeElimination"; }
    bool run(Graph& graph) override {
        bool changed = false;
        // Naive Mark-Sweep
        // 1. Mark outputs as live
        std::vector<bool> live_values(graph.values.size(), false);
        std::vector<ValueNode*> worklist = graph.outputs;
        
        for (auto* v : worklist) {
            // Find value index by pointer (slow, but robust for now)
            // Or assume value->id maps to index if we don't delete from vector
            // Let's use ID if it's stable.
            // Actually, we can just traverse up from outputs.
        }
        
        // Actually, let's just check uses.
        // If a value has no uses and is not a graph output, it's dead.
        // We need to iterate backwards or repeatedly.
        
        // Simplified: Remove OpNodes whose outputs are unused.
        auto& nodes = graph.nodes;
        auto it = std::remove_if(nodes.begin(), nodes.end(), [&](const std::unique_ptr<OpNode>& node) {
            bool has_users = false;
            for (auto* out : node->outputs) {
                // Check if 'out' is a graph output
                bool is_graph_out = false;
                for(auto* go : graph.outputs) if(go == out) is_graph_out = true;
                
                if (!out->uses.empty() || is_graph_out) {
                    has_users = true;
                    break;
                }
            }
            if (!has_users) {
                // If we remove this node, we should remove it from its inputs' uses
                for(auto* in : node->inputs) {
                    auto& uses = in->uses;
                    uses.erase(std::remove(uses.begin(), uses.end(), node.get()), uses.end());
                }
                changed = true;
                return true;
            }
            return false;
        });
        
        if (it != nodes.end()) {
            nodes.erase(it, nodes.end());
        }
        
        return changed;
    }
};
REGISTER_STAX_PASS("dce", DeadCodeElimination);

class FusionPass : public Pass {
public:
    std::string name() const override { return "FusionPass"; }
    bool run(Graph& graph) override {
        bool changed = false;
        // Implement the same logic as fuseGraph
        for (auto& node : graph.nodes) {
            if (node->op_type == "add") {
                for (auto* input : node->inputs) {
                    if (input->producer) {
                         if (input->producer->op_type == "mul") {
                            if (input->uses.size() == 1) {
                                // Fuse
                                node->op_type = "fused_mul_add";
                                
                                OpNode* mul_node = input->producer;
                                
                                // Rewire inputs
                                std::vector<ValueNode*> new_inputs;
                                for (auto* original_input : node->inputs) {
                                    if (original_input == input) {
                                        for (auto* mul_input : mul_node->inputs) {
                                            new_inputs.push_back(mul_input);
                                            mul_input->uses.push_back(node.get());
                                        }
                                    } else {
                                        new_inputs.push_back(original_input);
                                    }
                                }
                                node->inputs = new_inputs;
                                input->uses.clear(); // Disconnect mul output
                                changed = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
        return changed;
    }
};
REGISTER_STAX_PASS("fusion", FusionPass);

// --- Optimizer Implementation ---

void Optimizer::addPass(const std::string& pass_name) {
    passes_.push_back(pass_name);
}

void Optimizer::run(Graph& graph) {
    for (const auto& pass_name : passes_) {
        auto pass = PassRegistry::instance().createPass(pass_name);
        if (pass) {
            std::cout << "[Stax] Running Pass: " << pass->name() << std::endl;
            bool changed = pass->run(graph);
            if (changed) {
                std::cout << "       -> Graph modified." << std::endl;
            }
        } else {
            std::cerr << "[Stax] Warning: Pass '" << pass_name << "' not found." << std::endl;
        }
    }
}

// --- Legacy Wrapper ---
void fuseGraph(Graph& graph) {
    Optimizer opt;
    opt.addPass("fusion");
    opt.addPass("dce");
    opt.run(graph);
}

} // namespace stax
} // namespace tensorplay
