#pragma once
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <optional>
#include <iostream>
#include <variant>
#include <cstddef>
#include "Tensor.h"
#include "Macros.h"

namespace tensorplay {
namespace stax {

struct OpNode;
struct Graph;

struct STAX_API CaptureState {
    size_t compile_depth = 0;
    size_t disabled_depth = 0;
    size_t exporting_depth = 0;
};

STAX_API void enterCaptureState(
    bool compiling,
    bool exporting,
    bool disabled);
STAX_API void exitCaptureState(
    bool compiling,
    bool exporting,
    bool disabled);
STAX_API CaptureState currentCaptureState();

// a "custom_op" node carries its qualified "ns::op" name as a string
// attribute and hands its tensor inputs to the installed executor — the
// Python↔dispatcher bridge in the bindings.  Kernels keep full eager
// semantics (device dispatch, autograd) because the bridge re-enters the
// Python operator entry instead of a raw callable.
using CustomOpExecutor =
    std::function<std::vector<Tensor>(const std::string&, const std::vector<Tensor>&)>;

STAX_API void setCustomOpExecutor(CustomOpExecutor executor);
STAX_API CustomOpExecutor& customOpExecutor();

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

using Attribute = std::variant<
    int64_t,
    double,
    std::string,
    std::vector<int64_t>,
    std::vector<double>>;

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
    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) const;
    
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
