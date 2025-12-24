
#include "Graph.h"
#include <iostream>
#include <sstream>

namespace tensorplay {
namespace stax {

OpNode::OpNode(Graph* g, std::string type, std::string n) 
    : owningGraph(g), op_type(type), name(n) {}

void OpNode::addInput(ValueNode* v) {
    inputs.push_back(v);
    v->uses.push_back(this);
}

ValueNode* OpNode::addOutput() {
    auto v = std::make_unique<ValueNode>(owningGraph->values.size(), this, outputs.size());
    ValueNode* ptr = v.get();
    owningGraph->values.push_back(std::move(v));
    outputs.push_back(ptr);
    return ptr;
}

void OpNode::setAttr(const std::string& key, Attribute val) {
    attrs[key] = val;
}

ValueNode* Graph::addInput() {
    auto v = std::make_unique<ValueNode>(values.size(), nullptr, 0);
    ValueNode* ptr = v.get();
    values.push_back(std::move(v));
    inputs.push_back(ptr);
    return ptr;
}

OpNode* Graph::createNode(std::string op_type, std::string name) {
    if (name.empty()) {
        name = op_type + "_" + std::to_string(nodes.size());
    }
    auto n = std::make_unique<OpNode>(this, op_type, name);
    OpNode* ptr = n.get();
    nodes.push_back(std::move(n));
    return ptr;
}

void Graph::registerOutput(ValueNode* v) {
    outputs.push_back(v);
}

void Graph::print() const {
    std::cout << "Graph(" << inputs.size() << " inputs, " << outputs.size() << " outputs):" << std::endl;
    for (auto& n : nodes) {
        std::cout << "  %" << n->outputs[0]->id << " = " << n->op_type << "(";
        for (size_t i = 0; i < n->inputs.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << "%" << n->inputs[i]->id;
        }
        std::cout << ") [name=" << n->name << "]" << std::endl;
    }
}

// --- IRBuilder Implementation ---

ValueNode* IRBuilder::createInput(const std::vector<int64_t>& shape, const std::string& dtype) {
    ValueNode* val = graph_.addInput();
    val->shape = shape;
    val->dtype = dtype;
    return val;
}

ValueNode* IRBuilder::createOp(const std::string& op_type, 
                               const std::vector<ValueNode*>& inputs, 
                               const std::vector<int64_t>& out_shape,
                               const std::string& name) {
    std::string actual_name = name;
    if (actual_name.empty()) {
        actual_name = op_type + "_" + std::to_string(op_counter_++);
    }
    
    OpNode* node = graph_.createNode(op_type, actual_name);
    for (auto* in : inputs) {
        node->addInput(in);
    }
    
    ValueNode* out = node->addOutput();
    out->shape = out_shape;
    // Assume dtype propagation for now (same as input 0)
    if (!inputs.empty()) {
        out->dtype = inputs[0]->dtype;
    }
    
    return out;
}

void IRBuilder::markOutput(ValueNode* v) {
    graph_.registerOutput(v);
}

} // namespace stax
} // namespace tensorplay
