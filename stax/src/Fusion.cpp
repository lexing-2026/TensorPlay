#include "Fusion.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>

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

                                // Preserve scalar operands while removing the
                                // intermediate mul value.  This follows the
                                // pointwise fusion convention: constants
                                // remain attributes of the fused node.
                                if (auto it = mul_node->attrs.find("scalar_value");
                                    it != mul_node->attrs.end()) {
                                    node->attrs["mul_scalar_value"] = it->second;
                                    if (auto pos = mul_node->attrs.find("scalar_position");
                                        pos != mul_node->attrs.end()) {
                                        node->attrs["mul_scalar_position"] = pos->second;
                                    }
                                }
                                if (auto it = node->attrs.find("scalar_value");
                                    it != node->attrs.end()) {
                                    node->attrs["add_scalar_value"] = it->second;
                                    if (auto pos = node->attrs.find("scalar_position");
                                        pos != node->attrs.end()) {
                                        node->attrs["add_scalar_position"] = pos->second;
                                    }
                                }
                                
                                // Rewire inputs with the multiplied operand
                                // first.  Addition is commutative, and the
                                // fused executor's contract is
                                // ``mul(input0, scalar) + input1``.  Keeping
                                // the original add order would compile
                                // ``input0 + mul(input1, scalar)`` as
                                // ``mul(input0, scalar) + input1``.
                                std::vector<ValueNode*> new_inputs;
                                for (auto* mul_input : mul_node->inputs) {
                                    new_inputs.push_back(mul_input);
                                    mul_input->uses.push_back(node.get());
                                }
                                for (auto* original_input : node->inputs) {
                                    if (original_input != input) {
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
    const bool verbose = std::getenv("TENSORPLAY_STAX_VERBOSE") != nullptr;
    for (const auto& pass_name : passes_) {
        auto pass = PassRegistry::instance().createPass(pass_name);
        if (pass) {
            if (verbose) {
                std::cout << "[Stax] Running Pass: " << pass->name() << std::endl;
            }
            bool changed = pass->run(graph);
            if (verbose && changed) {
                std::cout << "       -> Graph modified." << std::endl;
            }
        } else {
            if (verbose) {
                std::cerr << "[Stax] Warning: Pass '" << pass_name << "' not found." << std::endl;
            }
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
