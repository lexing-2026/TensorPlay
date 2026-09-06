
#include "Graph.h"
#include "StaxPointwise.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <tuple>

namespace tensorplay {
namespace stax {

namespace {

thread_local CaptureState capture_state;

const std::vector<int64_t>* int_list_attr(
    const OpNode& node,
    const std::string& key) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end() || !std::holds_alternative<std::vector<int64_t>>(it->second)) {
        throw std::runtime_error("Stax fused pointwise attribute is missing: " + key);
    }
    return &std::get<std::vector<int64_t>>(it->second);
}

const std::vector<double>* float_list_attr(
    const OpNode& node,
    const std::string& key) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end()) {
        static const std::vector<double> empty;
        return &empty;
    }
    if (!std::holds_alternative<std::vector<double>>(it->second)) {
        throw std::runtime_error("Stax fused pointwise attribute has an invalid type: " + key);
    }
    return &std::get<std::vector<double>>(it->second);
}

const std::vector<int64_t>& required_int_list_attr(
    const OpNode& node,
    const std::string& key) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end() ||
        !std::holds_alternative<std::vector<int64_t>>(it->second)) {
        throw std::runtime_error("Stax native node integer-list attribute is missing: " + key);
    }
    return std::get<std::vector<int64_t>>(it->second);
}

int64_t required_int_attr(const OpNode& node, const std::string& key) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end() || !std::holds_alternative<int64_t>(it->second)) {
        throw std::runtime_error("Stax native node integer attribute is missing: " + key);
    }
    return std::get<int64_t>(it->second);
}

double required_float_attr(const OpNode& node, const std::string& key) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end() || !std::holds_alternative<double>(it->second)) {
        throw std::runtime_error("Stax native node float attribute is missing: " + key);
    }
    return std::get<double>(it->second);
}

Tensor execute_fused_pointwise_cpu(
    const OpNode& node,
    const std::vector<Tensor>& operands) {
    auto input_count_it = node.attrs.find("input_count");
    if (input_count_it == node.attrs.end() ||
        !std::holds_alternative<int64_t>(input_count_it->second)) {
        throw std::runtime_error("Stax fused pointwise input_count is missing");
    }
    const int64_t input_count = std::get<int64_t>(input_count_it->second);
    if (input_count < 1 || static_cast<size_t>(input_count) != operands.size()) {
        throw std::runtime_error("Stax fused pointwise input count mismatch");
    }
    const auto* program_ptr = int_list_attr(node, "program");
    const auto* constants_ptr = float_list_attr(node, "constants");
    return cpu::stax_fused_pointwise_cpu(operands, *program_ptr, *constants_ptr);
}

} // namespace

void enterCaptureState(bool compiling, bool exporting, bool disabled) {
    if (compiling) {
        ++capture_state.compile_depth;
    }
    if (exporting) {
        ++capture_state.exporting_depth;
    }
    if (disabled) {
        ++capture_state.disabled_depth;
    }
}

void exitCaptureState(bool compiling, bool exporting, bool disabled) {
    if (compiling && capture_state.compile_depth == 0) {
        throw std::runtime_error("Stax capture compile state underflow");
    }
    if (exporting && capture_state.exporting_depth == 0) {
        throw std::runtime_error("Stax capture export state underflow");
    }
    if (disabled && capture_state.disabled_depth == 0) {
        throw std::runtime_error("Stax capture disabled state underflow");
    }
    if (compiling) {
        --capture_state.compile_depth;
    }
    if (exporting) {
        --capture_state.exporting_depth;
    }
    if (disabled) {
        --capture_state.disabled_depth;
    }
}

CaptureState currentCaptureState() {
    return capture_state;
}

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

static CustomOpExecutor& customOpExecutorSlot() {
    static CustomOpExecutor executor;
    return executor;
}

void setCustomOpExecutor(CustomOpExecutor executor) {
    customOpExecutorSlot() = std::move(executor);
}

CustomOpExecutor& customOpExecutor() {
    return customOpExecutorSlot();
}

std::vector<Tensor> Graph::execute(const std::vector<Tensor>& inputs) const {
    if (inputs.size() != this->inputs.size()) {
        throw std::runtime_error(
            "Stax Graph::execute expected " + std::to_string(this->inputs.size()) +
            " inputs, got " + std::to_string(inputs.size()));
    }

    // ValueNode ids are dense and stable for the lifetime of the graph.  A
    // vector avoids hashing every operand on every native execution; this is
    // small compared with a convolution but material for the many pointwise
    // and residual nodes around each convolution.
    std::vector<Tensor> env(this->values.size());
    for (size_t i = 0; i < this->inputs.size(); ++i) {
        env[this->inputs[i]->id] = inputs[i];
    }

    // Keep only values that still have a downstream consumer.  TensorPlay's
    // allocator can recycle the storage as soon as the last use retires,
    // retaining every intermediate until the whole graph returns.
    std::vector<size_t> remaining_uses(this->values.size(), 0);
    std::vector<bool> keep_alive(this->values.size(), false);
    for (const auto& node_ptr : nodes) {
        for (const ValueNode* input : node_ptr->inputs) {
            ++remaining_uses[input->id];
        }
    }
    for (const ValueNode* output : outputs) {
        keep_alive[output->id] = true;
    }

    auto release_inputs = [&](const OpNode& node) {
        for (const ValueNode* input : node.inputs) {
            if (remaining_uses[input->id] > 0) {
                --remaining_uses[input->id];
            }
            if (remaining_uses[input->id] == 0 && !keep_alive[input->id]) {
                env[input->id] = Tensor();
            }
        }
    };

    auto value = [&env](const ValueNode* v) -> const Tensor& {
        if (v->id >= env.size() || !env[v->id].defined()) {
            throw std::runtime_error("Stax Graph::execute encountered an undefined value");
        }
        return env[v->id];
    };

    auto scalar_attr = [](const OpNode& node, const std::string& key) -> std::optional<Scalar> {
        auto it = node.attrs.find(key);
        if (it == node.attrs.end()) {
            return std::nullopt;
        }
        if (std::holds_alternative<int64_t>(it->second)) {
            return Scalar(std::get<int64_t>(it->second));
        }
        if (std::holds_alternative<double>(it->second)) {
            return Scalar(std::get<double>(it->second));
        }
        throw std::runtime_error("Stax scalar attribute has an invalid type: " + key);
    };

    auto scalar_position = [](const OpNode& node, const std::string& key) -> int64_t {
        auto it = node.attrs.find(key);
        if (it == node.attrs.end()) {
            return 1;
        }
        if (!std::holds_alternative<int64_t>(it->second)) {
            throw std::runtime_error("Stax scalar position has an invalid type: " + key);
        }
        return std::get<int64_t>(it->second);
    };

    for (const auto& node_ptr : nodes) {
        const OpNode& node = *node_ptr;
        if (node.outputs.empty()) {
            throw std::runtime_error("Stax operation has no output: " + node.op_type);
        }

        Tensor result;
        bool handled_by_custom_op = false;
        if (node.op_type == "channels_last") {
            // tensor with NHWC physical storage (see
            // the generated empty_strided/reinterpret_tensor wrapper).  The
            // native graph keeps that same logical shape and stride contract.
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax channels_last expects one input");
            }
            const Tensor& input = value(node.inputs[0]);
            if (input.dim() != 4) {
                throw std::runtime_error("Stax channels_last expects a 4D input");
            }
            const int64_t n = input.size(0);
            const int64_t c = input.size(1);
            const int64_t h = input.size(2);
            const int64_t w = input.size(3);
            const std::vector<int64_t> target_strides{
                c * h * w, 1, w * c, c};
            if (input.strides() == target_strides) {
                result = input;
            } else {
                // Clone the logical NHWC view, then reinterpret its
                // contiguous storage as logical NCHW with channels-last
                // strides.  This is the same one-copy reorder emitted by
                // before a convolution when the user input is
                // still contiguous NCHW.
                const std::vector<int64_t> physical_shape{n, h, w, c};
                const std::vector<int64_t> physical_strides{
                    input.stride(0), input.stride(2), input.stride(3), input.stride(1)};
                Tensor physical = input.as_strided(
                    physical_shape, physical_strides).clone();
                result = physical.as_strided(
                    {n, c, h, w}, target_strides);
            }
        } else if (node.op_type == "add_relu") {
            if (node.inputs.size() != 2) {
                throw std::runtime_error("Stax add_relu expects two tensor inputs");
            }
            result = tpx::ops::add_relu(value(node.inputs[0]), value(node.inputs[1]));
        } else if (node.op_type == "add" || node.op_type == "sub" ||
                   node.op_type == "mul" || node.op_type == "div") {
            auto scalar = scalar_attr(node, "scalar_value");
            if (scalar.has_value()) {
                if (node.inputs.size() != 1) {
                    throw std::runtime_error("Stax scalar binary op expects one tensor input");
                }
                const Tensor& tensor = value(node.inputs[0]);
                const bool scalar_first = scalar_position(node, "scalar_position") == 0;
                if (node.op_type == "add") {
                    result = tpx::ops::add(tensor, *scalar);
                } else if (node.op_type == "sub") {
                    result = scalar_first
                        ? tpx::ops::sub(
                            tpx::ops::full({}, *scalar, tensor.dtype(), tensor.device()), tensor)
                        : tpx::ops::sub(tensor, *scalar);
                } else if (node.op_type == "mul") {
                    result = tpx::ops::mul(tensor, *scalar);
                } else {
                    result = scalar_first
                        ? tpx::ops::div(
                            tpx::ops::full({}, *scalar, tensor.dtype(), tensor.device()), tensor)
                        : tpx::ops::div(tensor, *scalar);
                }
            } else {
                if (node.inputs.size() != 2) {
                    throw std::runtime_error("Stax binary op expects two tensor inputs");
                }
                if (node.op_type == "add") {
                    result = tpx::ops::add(value(node.inputs[0]), value(node.inputs[1]));
                } else if (node.op_type == "sub") {
                    result = tpx::ops::sub(value(node.inputs[0]), value(node.inputs[1]));
                } else if (node.op_type == "mul") {
                    result = tpx::ops::mul(value(node.inputs[0]), value(node.inputs[1]));
                } else {
                    result = tpx::ops::div(value(node.inputs[0]), value(node.inputs[1]));
                }
            }
        } else if (node.op_type == "neg" || node.op_type == "pos" ||
                   node.op_type == "abs" || node.op_type == "sin" ||
                   node.op_type == "cos" || node.op_type == "exp" ||
                   node.op_type == "log" || node.op_type == "sigmoid" ||
                   node.op_type == "sqrt" || node.op_type == "square" ||
                   node.op_type == "tanh" || node.op_type == "relu") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax unary op expects one tensor input");
            }
            const Tensor& input = value(node.inputs[0]);
            if (node.op_type == "neg") {
                result = tpx::ops::neg(input);
            } else if (node.op_type == "pos") {
                result = input;
            } else if (node.op_type == "abs") {
                result = tpx::ops::abs(input);
            } else if (node.op_type == "sin") {
                result = tpx::ops::sin(input);
            } else if (node.op_type == "cos") {
                result = tpx::ops::cos(input);
            } else if (node.op_type == "exp") {
                result = tpx::ops::exp(input);
            } else if (node.op_type == "log") {
                result = tpx::ops::log(input);
            } else if (node.op_type == "sigmoid") {
                result = tpx::ops::sigmoid(input);
            } else if (node.op_type == "sqrt") {
                result = tpx::ops::sqrt(input);
            } else if (node.op_type == "square") {
                result = tpx::ops::square(input);
            } else if (node.op_type == "tanh") {
                result = tpx::ops::tanh(input);
            } else {
                // The functional schema is authoritative: a plain relu must
                // not mutate its input merely because the value has one
                // consumer.  Only an explicit inplace=True capture carries
                // the write-alias bit into this native graph.
                const auto inplace_it = node.attrs.find("inplace");
                const bool inplace_requested =
                    inplace_it != node.attrs.end() &&
                    std::holds_alternative<int64_t>(inplace_it->second) &&
                    std::get<int64_t>(inplace_it->second) != 0;
                const bool reusable = inplace_requested &&
                    node.inputs[0]->producer != nullptr &&
                    node.inputs[0]->uses.size() == 1 &&
                    std::find(outputs.begin(), outputs.end(), node.inputs[0]) == outputs.end();
                if (reusable) {
                    Tensor inplace = input;
                    tpx::ops::relu_(inplace);
                    result = std::move(inplace);
                } else {
                    result = tpx::ops::relu(input);
                }
            }
        } else if (node.op_type == "pow") {
            auto scalar = scalar_attr(node, "scalar_value");
            if (scalar.has_value()) {
                if (node.inputs.size() != 1) {
                    throw std::runtime_error("Stax scalar pow expects one tensor input");
                }
                result = tpx::ops::pow(value(node.inputs[0]), *scalar);
            } else {
                if (node.inputs.size() != 2) {
                    throw std::runtime_error("Stax tensor pow expects two tensor inputs");
                }
                result = tpx::ops::pow(value(node.inputs[0]), value(node.inputs[1]));
            }
        } else if (node.op_type == "matmul") {
            if (node.inputs.size() != 2) {
                throw std::runtime_error("Stax matmul expects two tensor inputs");
            }
            result = tpx::ops::matmul(value(node.inputs[0]), value(node.inputs[1]));
        } else if (node.op_type == "mm") {
            if (node.inputs.size() != 2) {
                throw std::runtime_error("Stax mm expects two tensor inputs");
            }
            result = tpx::ops::mm(value(node.inputs[0]), value(node.inputs[1]));
        } else if (node.op_type == "linear") {
            // One node rather than a transpose, a product and an addition:
            // the bias belongs in the product's epilogue, and splitting it
            // out costs a full pass over the output to add it back.
            if (node.inputs.size() != 2 && node.inputs.size() != 3) {
                throw std::runtime_error(
                    "Stax linear expects an input, a weight and an optional bias");
            }
            std::optional<Tensor> bias;
            if (node.inputs.size() == 3) {
                bias = value(node.inputs[2]);
            }
            result = tpx::ops::linear(
                value(node.inputs[0]), value(node.inputs[1]), bias);
        } else if (node.op_type == "t") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax t expects one tensor input");
            }
            result = tpx::ops::t(value(node.inputs[0]));
        } else if (node.op_type == "conv2d") {
            if (node.inputs.size() != 2 && node.inputs.size() != 3) {
                throw std::runtime_error(
                    "Stax conv2d expects input and weight, with optional bias");
            }
            Tensor bias;
            if (required_int_attr(node, "has_bias") != 0) {
                if (node.inputs.size() != 3) {
                    throw std::runtime_error("Stax conv2d bias input is missing");
                }
                bias = value(node.inputs[2]);
            }
            result = tpx::ops::conv2d(
                value(node.inputs[0]),
                value(node.inputs[1]),
                bias,
                required_int_list_attr(node, "stride"),
                required_int_list_attr(node, "padding"),
                required_int_list_attr(node, "dilation"),
                required_int_attr(node, "groups"));
        } else if (node.op_type == "conv2d_relu") {
            if (node.inputs.size() != 2 && node.inputs.size() != 3) {
                throw std::runtime_error(
                    "Stax conv2d_relu expects input and weight, with optional bias");
            }
            Tensor bias;
            if (required_int_attr(node, "has_bias") != 0) {
                if (node.inputs.size() != 3) {
                    throw std::runtime_error("Stax conv2d_relu bias input is missing");
                }
                bias = value(node.inputs[2]);
            }
            result = tpx::ops::conv2d_relu(
                value(node.inputs[0]),
                value(node.inputs[1]),
                bias,
                required_int_list_attr(node, "stride"),
                required_int_list_attr(node, "padding"),
                required_int_list_attr(node, "dilation"),
                required_int_attr(node, "groups"));
        } else if (node.op_type == "batch_norm") {
            // Inputs are emitted in the same order as the optional fields in
            // the Python functional signature: running_mean, running_var,
            // weight, bias.  Presence flags make None a real optional value
            // rather than a dummy Tensor input.
            if (node.inputs.empty()) {
                throw std::runtime_error("Stax batch_norm is missing its input");
            }
            size_t input_index = 1;
            std::optional<Tensor> running_mean;
            std::optional<Tensor> running_var;
            std::optional<Tensor> weight;
            std::optional<Tensor> bias;
            auto take_optional = [&](const char* attr_name) -> std::optional<Tensor> {
                if (required_int_attr(node, attr_name) == 0) {
                    return std::nullopt;
                }
                if (input_index >= node.inputs.size()) {
                    throw std::runtime_error("Stax batch_norm optional input is missing");
                }
                return value(node.inputs[input_index++]);
            };
            running_mean = take_optional("has_running_mean");
            running_var = take_optional("has_running_var");
            weight = take_optional("has_weight");
            bias = take_optional("has_bias");
            if (input_index != node.inputs.size()) {
                throw std::runtime_error("Stax batch_norm has unexpected inputs");
            }
            result = tpx::ops::batch_norm(
                value(node.inputs[0]),
                weight,
                bias,
                running_mean,
                running_var,
                required_int_attr(node, "training") != 0,
                required_float_attr(node, "momentum"),
                required_float_attr(node, "eps"));
        } else if (node.op_type == "max_pool2d") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax max_pool2d expects one input");
            }
            result = tpx::ops::max_pool2d(
                value(node.inputs[0]),
                required_int_list_attr(node, "kernel_size"),
                required_int_list_attr(node, "stride"),
                required_int_list_attr(node, "padding"),
                required_int_list_attr(node, "dilation"),
                required_int_attr(node, "ceil_mode") != 0);
        } else if (node.op_type == "adaptive_avg_pool2d") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax adaptive_avg_pool2d expects one input");
            }
            result = tpx::ops::adaptive_avg_pool2d(
                value(node.inputs[0]),
                required_int_list_attr(node, "output_size"));
        } else if (node.op_type == "threshold_backward") {
            if (node.inputs.size() != 2) {
                throw std::runtime_error("Stax threshold_backward expects grad and output");
            }
            result = tpx::ops::threshold_backward(
                value(node.inputs[0]),
                value(node.inputs[1]),
                *scalar_attr(node, "threshold"));
        } else if (node.op_type == "conv2d_grad_input" ||
                   node.op_type == "conv2d_grad_weight" ||
                   node.op_type == "conv2d_grad_bias") {
            if (node.inputs.size() != 3) {
                throw std::runtime_error(
                    "Stax conv2d backward expects grad, input, and weight");
            }
            const Tensor& grad_output = value(node.inputs[0]);
            const Tensor& input = value(node.inputs[1]);
            const Tensor& weight = value(node.inputs[2]);
            const auto stride = required_int_list_attr(node, "stride");
            const auto padding = required_int_list_attr(node, "padding");
            const auto dilation = required_int_list_attr(node, "dilation");
            const int64_t groups = required_int_attr(node, "groups");
            if (node.op_type == "conv2d_grad_input") {
                result = tpx::ops::conv2d_grad_input(
                    grad_output, input, weight, stride, padding, dilation, groups);
            } else if (node.op_type == "conv2d_grad_weight") {
                result = tpx::ops::conv2d_grad_weight(
                    grad_output, input, weight, stride, padding, dilation, groups);
            } else {
                result = tpx::ops::conv2d_grad_bias(
                    grad_output, input, weight, stride, padding, dilation, groups);
            }
        } else if (node.op_type == "matmul_backward_self" ||
                   node.op_type == "matmul_backward_other") {
            if (node.inputs.size() != 3) {
                throw std::runtime_error(
                    "Stax matmul backward expects grad, self, and other");
            }
            if (node.op_type == "matmul_backward_self") {
                result = tpx::ops::matmul_backward_self(
                    value(node.inputs[0]), value(node.inputs[1]), value(node.inputs[2]));
            } else {
                result = tpx::ops::matmul_backward_other(
                    value(node.inputs[0]), value(node.inputs[1]), value(node.inputs[2]));
            }
        } else if (node.op_type == "max_pool2d_backward") {
            if (node.inputs.size() != 2) {
                throw std::runtime_error("Stax max_pool2d_backward expects grad and input");
            }
            result = tpx::ops::max_pool2d_backward(
                value(node.inputs[0]),
                value(node.inputs[1]),
                required_int_list_attr(node, "kernel_size"),
                required_int_list_attr(node, "stride"),
                required_int_list_attr(node, "padding"),
                required_int_list_attr(node, "dilation"),
                required_int_attr(node, "ceil_mode") != 0);
        } else if (node.op_type == "adaptive_avg_pool2d_backward") {
            if (node.inputs.size() != 2) {
                throw std::runtime_error(
                    "Stax adaptive_avg_pool2d_backward expects grad and input");
            }
            result = tpx::ops::adaptive_avg_pool2d_backward(
                value(node.inputs[0]), value(node.inputs[1]));
        } else if (node.op_type == "batch_norm_backward") {
            if (node.inputs.size() < 2) {
                throw std::runtime_error("Stax batch_norm_backward is missing grad/input");
            }
            size_t input_index = 2;
            std::optional<Tensor> weight;
            std::optional<Tensor> running_mean;
            std::optional<Tensor> running_var;
            auto take_optional = [&](const char* attr_name) -> std::optional<Tensor> {
                if (required_int_attr(node, attr_name) == 0) {
                    return std::nullopt;
                }
                if (input_index >= node.inputs.size()) {
                    throw std::runtime_error(
                        "Stax batch_norm_backward optional input is missing");
                }
                return value(node.inputs[input_index++]);
            };
            weight = take_optional("has_weight");
            running_mean = take_optional("has_running_mean");
            running_var = take_optional("has_running_var");
            if (input_index != node.inputs.size()) {
                throw std::runtime_error(
                    "Stax batch_norm_backward has unexpected inputs");
            }
            auto backward = tpx::ops::batch_norm_backward(
                value(node.inputs[0]),
                value(node.inputs[1]),
                weight,
                running_mean,
                running_var,
                required_int_attr(node, "training") != 0,
                required_float_attr(node, "eps"));
            if (node.outputs.size() != 3) {
                throw std::runtime_error(
                    "Stax batch_norm_backward must have three outputs");
            }
            env[node.outputs[0]->id] = std::get<0>(backward);
            env[node.outputs[1]->id] = std::get<1>(backward);
            env[node.outputs[2]->id] = std::get<2>(backward);
            release_inputs(node);
            continue;
        } else if (node.op_type == "reshape") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax reshape expects one input");
            }
            result = tpx::ops::reshape(
                value(node.inputs[0]), required_int_list_attr(node, "shape"));
        } else if (node.op_type == "sum") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax sum expects one input");
            }
            const auto dim_it = node.attrs.find("dim");
            if (dim_it == node.attrs.end()) {
                result = tpx::ops::sum(value(node.inputs[0]));
            } else {
                if (!std::holds_alternative<std::vector<int64_t>>(dim_it->second)) {
                    throw std::runtime_error("Stax sum dim has an invalid type");
                }
                result = tpx::ops::sum(
                    value(node.inputs[0]),
                    std::get<std::vector<int64_t>>(dim_it->second),
                    required_int_attr(node, "keepdim") != 0);
            }
        } else if (node.op_type == "flatten") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax flatten expects one input");
            }
            const Tensor& input = value(node.inputs[0]);
            int64_t start_dim = required_int_attr(node, "start_dim");
            int64_t end_dim = required_int_attr(node, "end_dim");
            const int64_t ndim = input.dim();
            if (start_dim < 0) start_dim += ndim;
            if (end_dim < 0) end_dim += ndim;
            if (start_dim < 0 || end_dim < start_dim || end_dim >= ndim) {
                throw std::runtime_error("Stax flatten has invalid dimensions");
            }
            std::vector<int64_t> shape;
            shape.reserve(static_cast<size_t>(ndim - (end_dim - start_dim)));
            for (int64_t dim = 0; dim < start_dim; ++dim) {
                shape.push_back(input.size(dim));
            }
            int64_t flattened = 1;
            for (int64_t dim = start_dim; dim <= end_dim; ++dim) {
                flattened *= input.size(dim);
            }
            shape.push_back(flattened);
            for (int64_t dim = end_dim + 1; dim < ndim; ++dim) {
                shape.push_back(input.size(dim));
            }
            result = tpx::ops::reshape(input, shape);
        } else if (node.op_type == "fused_pointwise") {
            std::vector<Tensor> operands;
            operands.reserve(node.inputs.size());
            for (const ValueNode* input : node.inputs) {
                operands.push_back(value(input));
            }
            result = execute_fused_pointwise_cpu(node, operands);
        } else if (node.op_type == "fused_mul_add") {
            auto mul_scalar = scalar_attr(node, "mul_scalar_value");
            auto add_scalar = scalar_attr(node, "add_scalar_value");
            if (mul_scalar.has_value()) {
                if (node.inputs.size() != 1 && node.inputs.size() != 2) {
                    throw std::runtime_error("Stax fused scalar mul-add has invalid inputs");
                }
                if (add_scalar.has_value() && node.inputs.size() == 1) {
                    // Scalar constants stay as IR attributes, so the scalar
                    // overload avoids materializing two intermediate tensors.
                    result = tpx::ops::fused_mul_add(
                        value(node.inputs[0]), *mul_scalar, *add_scalar);
                    env[node.outputs[0]->id] = std::move(result);
                    release_inputs(node);
                    continue;
                }
                Tensor product = tpx::ops::mul(value(node.inputs[0]), *mul_scalar);
                result = add_scalar.has_value()
                    ? tpx::ops::add(product, *add_scalar)
                    : tpx::ops::add(product, value(node.inputs[1]));
            } else {
                if (node.inputs.size() != 2 && node.inputs.size() != 3) {
                    throw std::runtime_error("Stax fused mul-add has invalid inputs");
                }
                if (!add_scalar.has_value() && node.inputs.size() == 3) {
                    // keeps the generated autograd contract attached while
                    // dispatching the single CPU/CUDA kernel in p10.
                    result = tpx::ops::fused_mul_add(
                        value(node.inputs[0]), value(node.inputs[1]), value(node.inputs[2]));
                } else {
                    Tensor product = tpx::ops::mul(value(node.inputs[0]), value(node.inputs[1]));
                    result = add_scalar.has_value()
                        ? tpx::ops::add(product, *add_scalar)
                        : tpx::ops::add(product, value(node.inputs[2]));
                }
            }
        } else if (node.op_type == "custom_op") {
            // User-defined operator: re-enter the Python dispatcher bridge
            // (device dispatch + autograd preserved) with the tensor values.
            const auto& executor = customOpExecutor();
            if (!executor) {
                throw std::runtime_error(
                    "Stax Graph::execute found a custom_op node but no "
                    "executor is installed");
            }
            std::vector<Tensor> op_inputs;
            op_inputs.reserve(node.inputs.size());
            for (const ValueNode* input : node.inputs) {
                op_inputs.push_back(value(input));
            }
            std::vector<Tensor> op_outputs =
                executor(node.getAttr<std::string>("op_name"), op_inputs);
            if (op_outputs.size() != node.outputs.size()) {
                throw std::runtime_error(
                    "custom op '" + node.name + "' produced " +
                    std::to_string(op_outputs.size()) + " outputs but the "
                    "native graph reserved " +
                    std::to_string(node.outputs.size()));
            }
            for (size_t oi = 0; oi < op_outputs.size(); ++oi) {
                env[node.outputs[oi]->id] = std::move(op_outputs[oi]);
            }
            release_inputs(node);
            handled_by_custom_op = true;
        } else {
            throw std::runtime_error("Stax Graph::execute does not support op: " + node.op_type);
        }

        if (!handled_by_custom_op) {
            if (node.outputs.size() != 1) {
                throw std::runtime_error(
                    "Stax native operation has multiple outputs but no output handler: " +
                    node.op_type);
            }
            env[node.outputs[0]->id] = std::move(result);
            release_inputs(node);
        }
    }

    std::vector<Tensor> result;
    result.reserve(outputs.size());
    for (const ValueNode* output : outputs) {
        result.push_back(value(output));
    }
    return result;
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
