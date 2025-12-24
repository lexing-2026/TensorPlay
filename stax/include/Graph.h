#pragma once
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <optional>
#include <iostream>
#include <variant>
#include "Tensor.h" 
#include "Macros.h"

namespace tensorplay {
namespace stax {

struct OpNode;
struct Graph;

struct STAX_API ValueNode {
    size_t id;
    OpNode* producer; // Producer op
    size_t producer_output_index; // Output index of producer
    
    // Type info (Metadata)
    std::string dtype = "float32"; // Simplified for now
    std::vector<int64_t> shape;
    std::string device = "cpu";
    
    std::vector<OpNode*> uses;
    
    ValueNode(size_t id_, OpNode* n, size_t off) : id(id_), producer(n), producer_output_index(off) {}
};

using Attribute = std::variant<int64_t, double, std::string, std::vector<int64_t>>;

struct STAX_API OpNode {
    std::string op_type; // e.g., "add", "matmul"
    std::string name;    // unique name
    std::vector<ValueNode*> inputs;
    std::vector<ValueNode*> outputs;
    Graph* owningGraph;
    
    // Attributes
    std::unordered_map<std::string, Attribute> attrs;
    
    OpNode(Graph* g, std::string type, std::string n);
    
    void addInput(ValueNode* v);
    ValueNode* addOutput();
    
    // Helpers
    void setAttr(const std::string& key, Attribute val);
    template<typename T> T getAttr(const std::string& key) const;
};

struct STAX_API Graph {
    std::vector<std::unique_ptr<OpNode>> nodes;
    std::vector<std::unique_ptr<ValueNode>> values;
    std::vector<ValueNode*> inputs;
    std::vector<ValueNode*> outputs;
    
    ValueNode* addInput();
    OpNode* createNode(std::string op_type, std::string name = "");
    void registerOutput(ValueNode* v);
    
    void print() const;

    Graph() = default;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
};

class STAX_API IRBuilder {
public:
    IRBuilder(Graph& g) : graph_(g) {}
    
    ValueNode* createInput(const std::vector<int64_t>& shape, const std::string& dtype = "float32");
    
    ValueNode* createOp(const std::string& op_type, 
                        const std::vector<ValueNode*>& inputs, 
                        const std::vector<int64_t>& out_shape = {},
                        const std::string& name = "");
                        
    void markOutput(ValueNode* v);

private:
    Graph& graph_;
    size_t op_counter_ = 0;
};

// Template implementation
template<typename T>
T OpNode::getAttr(const std::string& key) const {
    if (attrs.count(key)) {
        if (std::holds_alternative<T>(attrs.at(key))) {
            return std::get<T>(attrs.at(key));
        }
    }
    throw std::runtime_error("Attribute not found or type mismatch: " + key);
}

} // namespace stax
} // namespace tensorplay
