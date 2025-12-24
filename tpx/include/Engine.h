#pragma once
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <stack>
#include <iostream>
#include "Macros.h"
#include "TPXTensor.h"
#include "Edge.h"
#include "Node.h"

namespace tensorplay {
namespace tpx {

class TENSORPLAY_API Engine {
private:
    struct NodeTask {
        std::shared_ptr<Node> fn;
        
        // Max heap by sequence_nr
        bool operator<(const NodeTask& other) const {
            return fn->sequence_nr() < other.fn->sequence_nr();
        }
    };

public:
    static Engine& get_default_engine();
    
    // capture_grads: map from Node* to input gradients (vector of tensors).
    // If a node is in this map, we capture its input gradients and do NOT execute the node.
    void execute_graph(const std::vector<tensorplay::tpx::Edge>& roots, 
                       const std::vector<tensorplay::tpx::Tensor>& inputs, 
                       bool keep_graph, 
                       bool create_graph,
                       std::unordered_map<Node*, std::vector<Tensor>>* capture_grads = nullptr) {
        if (roots.size() != inputs.size()) {
            TP_THROW(RuntimeError, "Engine::execute: roots and inputs must have same size");
        }

        // 1. Compute dependencies
        // Count incoming edges for nodes reachable from roots
        std::unordered_map<Node*, int> dependencies;
        std::unordered_set<Node*> visited;
        std::vector<Node*> stack;
        
        for (const auto& edge : roots) {
            if (edge.function) {
                stack.push_back(edge.function.get());
            }
        }
        
        while (!stack.empty()) {
            Node* node = stack.back();
            stack.pop_back();
            
            if (visited.count(node)) continue;
            visited.insert(node);
            
            for (const auto& edge : node->next_edges()) {
                 if (auto next_node = edge.function) {
                     dependencies[next_node.get()]++;
                     stack.push_back(next_node.get());
                 }
            }
        }
        
        struct NodeState {
            std::unordered_map<uint32_t, Tensor> input_grads;
            int outstanding_deps = 0;
        };
        
        std::unordered_map<Node*, NodeState> graph_state;
        
        for (auto& [node, dep] : dependencies) {
            graph_state[node].outstanding_deps = dep;
        }
        
        // 2. Initialize Queue
        std::priority_queue<NodeTask> ready_queue;
        std::unordered_set<Node*> in_queue;

        // 3. Feed Roots
        for (size_t i = 0; i < roots.size(); ++i) {
            auto node_ptr = roots[i].function;
            if (!node_ptr) continue;
            
            Node* node = node_ptr.get();
            auto& state = graph_state[node];
            uint32_t input_nr = roots[i].input_nr;
            
            // Accumulate root gradients
            if (state.input_grads[input_nr].defined()) {
                 // Use in-place accumulation to avoid deep copy/allocation overhead
                 state.input_grads[input_nr] += inputs[i];
            } else {
                 state.input_grads[input_nr] = inputs[i];
            }
            
            // Mark as autograd shared to enable optimizations (e.g. OneDNN cache sharing)
            if (state.input_grads[input_nr].defined()) {
                state.input_grads[input_nr].unsafeGetTensorImpl()->set_autograd_shared(true);
            }
            
            if (state.outstanding_deps == 0) {
                if (in_queue.find(node) == in_queue.end()) {
                    ready_queue.push({node_ptr});
                    in_queue.insert(node);
                }
            }
        }

        // 4. Execution Loop
        while (!ready_queue.empty()) {
            auto task = ready_queue.top();
            ready_queue.pop();
            Node* node = task.fn.get();
            
            auto& state = graph_state[node];
            
            // Construct input list
            uint32_t max_idx = 0;
            for(auto& kv : state.input_grads) {
                if (kv.first > max_idx) max_idx = kv.first;
            }
            
            variable_list grad_inputs;
            if (!state.input_grads.empty()) {
                grad_inputs.resize(max_idx + 1);
                for(auto& kv : state.input_grads) {
                    grad_inputs[kv.first] = kv.second;
                }
            }
            
            // Check if we should capture this node's input gradients instead of executing
            if (capture_grads && capture_grads->count(node)) {
                (*capture_grads)[node] = grad_inputs;
                continue;
            }

            // Execute Node
            variable_list grad_outputs = node->apply(std::move(grad_inputs));
            
            // Propagate to next edges
            const auto& edges = node->next_edges();
            
            // If mismatch, we just iterate min length
            for (size_t i = 0; i < edges.size(); ++i) {
                if (i >= grad_outputs.size()) break;
                
                const auto& edge = edges[i];
                const auto& grad = grad_outputs[i];
                
                if (!edge.is_valid()) continue;
                if (!grad.defined()) continue;
                
                Node* next_node = edge.function.get();
                auto& next_state = graph_state[next_node];
                
                // Accumulate
                if (next_state.input_grads[edge.input_nr].defined()) {
                    next_state.input_grads[edge.input_nr] = next_state.input_grads[edge.input_nr].add(grad);
                } else {
                    next_state.input_grads[edge.input_nr] = grad;
                }
                
                next_state.outstanding_deps--;
                if (next_state.outstanding_deps == 0) {
                    if (in_queue.find(next_node) == in_queue.end()) {
                        ready_queue.push({edge.function});
                        in_queue.insert(next_node);
                    }
                }
            }

            // Cleanup graph if needed
            if (!keep_graph) {
                node->release_variables();
            }
        }
    }
    
    void test_link();
};

}
}
