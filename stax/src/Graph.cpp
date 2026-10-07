
#include "Graph.h"
#include "GradMode.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <tuple>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <set>
#include <thread>
#include <exception>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tensorplay {
namespace stax {

namespace {

thread_local CaptureState capture_state;

struct GradModeGuard {
    explicit GradModeGuard(bool enabled) : previous(GradMode::is_enabled()) {
        GradMode::set_enabled(enabled);
    }

    ~GradModeGuard() { GradMode::set_enabled(previous); }

    bool previous;
};

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

    // The dispatch body is wrapped so the scheduler below can run
    // independent nodes concurrently; the eager engine gets that overlap at
    // residual forks for free and a sequential walk forfeits it.
    auto run_node = [&](const OpNode& node) -> void {
        if (node.outputs.empty()) {
            throw std::runtime_error("Stax operation has no output: " + node.op_type);
        }

        Tensor result;
        std::vector<Tensor> multi_result;
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
                // Materialize the logical NHWC view into an NHWC-ordered
                // buffer, then reinterpret its contiguous storage as logical
                // NCHW with channels-last strides.  A stride-preserving copy
                // would keep the source's physical order and the final
                // reinterpretation would read elements in the wrong places.
                const std::vector<int64_t> physical_shape{n, h, w, c};
                const std::vector<int64_t> physical_strides{
                    input.stride(0), input.stride(2), input.stride(3), input.stride(1)};
                Tensor physical = input.as_strided(
                    physical_shape, physical_strides).contiguous();
                result = physical.as_strided(
                    {n, c, h, w}, target_strides);
            }
        } else if (node.op_type == "add_relu") {
            if (node.inputs.size() != 2) {
                throw std::runtime_error("Stax add_relu expects two tensor inputs");
            }
            result = tpx::ops::add_relu(value(node.inputs[0]), value(node.inputs[1]));
        } else if (node.op_type == "add" || node.op_type == "sub" ||
                   node.op_type == "mul" || node.op_type == "div" ||
                   node.op_type == "eq" || node.op_type == "ne" ||
                   node.op_type == "lt" || node.op_type == "le" ||
                   node.op_type == "gt" || node.op_type == "ge") {
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
                } else if (node.op_type == "div") {
                    result = scalar_first
                        ? tpx::ops::div(
                            tpx::ops::full({}, *scalar, tensor.dtype(), tensor.device()), tensor)
                        : tpx::ops::div(tensor, *scalar);
                } else if (node.op_type == "eq") {
                    result = tpx::ops::eq(tensor, *scalar);
                } else if (node.op_type == "ne") {
                    result = tpx::ops::ne(tensor, *scalar);
                } else if (node.op_type == "lt") {
                    result = scalar_first ? tpx::ops::gt(tensor, *scalar)
                                          : tpx::ops::lt(tensor, *scalar);
                } else if (node.op_type == "le") {
                    result = scalar_first ? tpx::ops::ge(tensor, *scalar)
                                          : tpx::ops::le(tensor, *scalar);
                } else if (node.op_type == "gt") {
                    result = scalar_first ? tpx::ops::lt(tensor, *scalar)
                                          : tpx::ops::gt(tensor, *scalar);
                } else {
                    result = scalar_first ? tpx::ops::le(tensor, *scalar)
                                          : tpx::ops::ge(tensor, *scalar);
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
                } else if (node.op_type == "div") {
                    result = tpx::ops::div(value(node.inputs[0]), value(node.inputs[1]));
                } else if (node.op_type == "eq") {
                    result = tpx::ops::eq(value(node.inputs[0]), value(node.inputs[1]));
                } else if (node.op_type == "ne") {
                    result = tpx::ops::ne(value(node.inputs[0]), value(node.inputs[1]));
                } else if (node.op_type == "lt") {
                    result = tpx::ops::lt(value(node.inputs[0]), value(node.inputs[1]));
                } else if (node.op_type == "le") {
                    result = tpx::ops::le(value(node.inputs[0]), value(node.inputs[1]));
                } else if (node.op_type == "gt") {
                    result = tpx::ops::gt(value(node.inputs[0]), value(node.inputs[1]));
                } else {
                    result = tpx::ops::ge(value(node.inputs[0]), value(node.inputs[1]));
                }
            }
        } else if (node.op_type == "where") {
            if (node.inputs.empty() || node.inputs.size() > 3) {
                throw std::runtime_error("Stax where has invalid inputs");
            }
            auto self_scalar = scalar_attr(node, "self_scalar");
            auto other_scalar = scalar_attr(node, "other_scalar");
            const Tensor& condition = value(node.inputs[0]);
            if (self_scalar.has_value() && other_scalar.has_value()) {
                result = tpx::ops::where(condition, *self_scalar, *other_scalar);
            } else if (self_scalar.has_value()) {
                if (node.inputs.size() != 2) {
                    throw std::runtime_error("Stax where self input is missing");
                }
                result = tpx::ops::where(
                    condition, *self_scalar, value(node.inputs[1]));
            } else if (other_scalar.has_value()) {
                if (node.inputs.size() != 2) {
                    throw std::runtime_error("Stax where other input is missing");
                }
                result = tpx::ops::where(
                    condition, value(node.inputs[1]), *other_scalar);
            } else {
                if (node.inputs.size() != 3) {
                    throw std::runtime_error("Stax where tensor inputs are missing");
                }
                result = tpx::ops::where(
                    condition, value(node.inputs[1]), value(node.inputs[2]));
            }
        } else if (node.op_type == "clamp" || node.op_type == "clamp_min" ||
                   node.op_type == "clamp_max") {
            if (node.inputs.empty() || node.inputs.size() > 3) {
                throw std::runtime_error("Stax clamp has invalid inputs");
            }
            const Tensor& input = value(node.inputs[0]);
            auto min_scalar = scalar_attr(node, "min_scalar");
            auto max_scalar = scalar_attr(node, "max_scalar");
            const bool min_tensor = required_int_attr(node, "min_tensor") != 0;
            const bool max_tensor = required_int_attr(node, "max_tensor") != 0;
            size_t input_index = 1;
            std::optional<Tensor> min_value;
            std::optional<Tensor> max_value;
            if (min_tensor) {
                if (input_index >= node.inputs.size()) {
                    throw std::runtime_error("Stax clamp minimum input is missing");
                }
                min_value = value(node.inputs[input_index++]);
            }
            if (max_tensor) {
                if (input_index >= node.inputs.size()) {
                    throw std::runtime_error("Stax clamp maximum input is missing");
                }
                max_value = value(node.inputs[input_index++]);
            }
            if (input_index != node.inputs.size()) {
                throw std::runtime_error("Stax clamp has unexpected inputs");
            }
            if (node.op_type == "clamp_min") {
                result = min_tensor ? tpx::ops::clamp_min(input, *min_value)
                                     : tpx::ops::clamp_min(input, *min_scalar);
            } else if (node.op_type == "clamp_max") {
                result = max_tensor ? tpx::ops::clamp_max(input, *max_value)
                                     : tpx::ops::clamp_max(input, *max_scalar);
            } else if (min_tensor || max_tensor) {
                result = tpx::ops::clamp(input, min_value, max_value);
            } else {
                result = tpx::ops::clamp(input, min_scalar, max_scalar);
            }
        } else if (node.op_type == "gelu") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax gelu expects one tensor input");
            }
            const auto approximate_it = node.attrs.find("approximate");
            const std::string approximate = approximate_it == node.attrs.end()
                ? "none"
                : node.getAttr<std::string>("approximate");
            result = tpx::ops::gelu(value(node.inputs[0]), approximate);
        } else if (node.op_type == "softmax" || node.op_type == "log_softmax") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax softmax expects one tensor input");
            }
            const auto dtype_it = node.attrs.find("dtype");
            const DType dtype = dtype_it == node.attrs.end()
                ? DType::Undefined
                : static_cast<DType>(required_int_attr(node, "dtype"));
            result = node.op_type == "softmax"
                ? tpx::ops::softmax(value(node.inputs[0]), required_int_attr(node, "dim"), dtype)
                : tpx::ops::log_softmax(value(node.inputs[0]), required_int_attr(node, "dim"), dtype);
        } else if (node.op_type == "layer_norm") {
            if (node.inputs.empty() || node.inputs.size() > 3) {
                throw std::runtime_error("Stax layer_norm has invalid inputs");
            }
            size_t input_index = 1;
            std::optional<Tensor> weight;
            std::optional<Tensor> bias;
            if (required_int_attr(node, "has_weight") != 0) {
                if (input_index >= node.inputs.size()) {
                    throw std::runtime_error("Stax layer_norm weight is missing");
                }
                weight = value(node.inputs[input_index++]);
            }
            if (required_int_attr(node, "has_bias") != 0) {
                if (input_index >= node.inputs.size()) {
                    throw std::runtime_error("Stax layer_norm bias is missing");
                }
                bias = value(node.inputs[input_index++]);
            }
            if (input_index != node.inputs.size()) {
                throw std::runtime_error("Stax layer_norm has unexpected inputs");
            }
            result = tpx::ops::layer_norm(
                value(node.inputs[0]),
                required_int_list_attr(node, "normalized_shape"),
                weight,
                bias,
                required_float_attr(node, "eps"));
        } else if (node.op_type == "neg" || node.op_type == "pos" ||
                   node.op_type == "abs" || node.op_type == "sin" ||
                   node.op_type == "cos" || node.op_type == "exp" ||
                   node.op_type == "log" || node.op_type == "sigmoid" ||
                   node.op_type == "sqrt" || node.op_type == "square" ||
                   node.op_type == "tanh" || node.op_type == "relu" ||
                   node.op_type == "conj" || node.op_type == "rsqrt" ||
                   node.op_type == "sign" || node.op_type == "silu" ||
                   node.op_type == "acos" || node.op_type == "acosh" ||
                   node.op_type == "asin" || node.op_type == "asinh" ||
                   node.op_type == "atan" || node.op_type == "atanh" ||
                   node.op_type == "ceil" || node.op_type == "cosh" ||
                   node.op_type == "erf" || node.op_type == "erfc" ||
                   node.op_type == "exp2" || node.op_type == "expm1" ||
                   node.op_type == "floor" || node.op_type == "log1p" ||
                   node.op_type == "log2" || node.op_type == "reciprocal" ||
                   node.op_type == "round" || node.op_type == "sinh" ||
                   node.op_type == "tan" || node.op_type == "trunc" ||
                   node.op_type == "erfinv" || node.op_type == "erfcx" ||
                   node.op_type == "lgamma" || node.op_type == "i0" ||
                   node.op_type == "tanhshrink") {
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
            } else if (node.op_type == "silu") {
                result = tpx::ops::silu(input);
            } else if (node.op_type == "conj") {
                result = tpx::ops::conj(input);
            } else if (node.op_type == "rsqrt") {
                result = tpx::ops::rsqrt(input);
            } else if (node.op_type == "sign") {
                result = tpx::ops::sign(input);
            } else if (node.op_type == "acos") {
                result = tpx::ops::acos(input);
            } else if (node.op_type == "acosh") {
                result = tpx::ops::acosh(input);
            } else if (node.op_type == "asin") {
                result = tpx::ops::asin(input);
            } else if (node.op_type == "asinh") {
                result = tpx::ops::asinh(input);
            } else if (node.op_type == "atan") {
                result = tpx::ops::atan(input);
            } else if (node.op_type == "atanh") {
                result = tpx::ops::atanh(input);
            } else if (node.op_type == "ceil") {
                result = tpx::ops::ceil(input);
            } else if (node.op_type == "cosh") {
                result = tpx::ops::cosh(input);
            } else if (node.op_type == "erf") {
                result = tpx::ops::erf(input);
            } else if (node.op_type == "erfc") {
                result = tpx::ops::erfc(input);
            } else if (node.op_type == "exp2") {
                result = tpx::ops::exp2(input);
            } else if (node.op_type == "expm1") {
                result = tpx::ops::expm1(input);
            } else if (node.op_type == "floor") {
                result = tpx::ops::floor(input);
            } else if (node.op_type == "log1p") {
                result = tpx::ops::log1p(input);
            } else if (node.op_type == "log2") {
                result = tpx::ops::log2(input);
            } else if (node.op_type == "reciprocal") {
                result = tpx::ops::reciprocal(input);
            } else if (node.op_type == "round") {
                result = tpx::ops::round(input);
            } else if (node.op_type == "sinh") {
                result = tpx::ops::sinh(input);
            } else if (node.op_type == "tan") {
                result = tpx::ops::tan(input);
            } else if (node.op_type == "trunc") {
                result = tpx::ops::trunc(input);
            } else if (node.op_type == "erfinv") {
                result = tpx::ops::erfinv(input);
            } else if (node.op_type == "erfcx") {
                result = tpx::ops::erfcx(input);
            } else if (node.op_type == "lgamma") {
                result = tpx::ops::lgamma(input);
            } else if (node.op_type == "i0") {
                result = tpx::ops::i0(input);
            } else if (node.op_type == "tanhshrink") {
                result = tpx::ops::tanhshrink(input);
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
                const bool scalar_first = scalar_position(node, "scalar_position") == 0;
                result = scalar_first
                    ? tpx::ops::pow(*scalar, value(node.inputs[0]))
                    : tpx::ops::pow(value(node.inputs[0]), *scalar);
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
        } else if (node.op_type == "linear_backward") {
            if (node.inputs.size() != 3 || node.outputs.size() != 3) {
                throw std::runtime_error(
                    "Stax linear_backward expects input, grad, weight and three outputs");
            }
            const auto mask = required_int_list_attr(node, "output_mask");
            if (mask.size() != 3) {
                throw std::runtime_error(
                    "Stax linear_backward output_mask must have three entries");
            }
            std::vector<bool> output_mask;
            output_mask.reserve(mask.size());
            for (int64_t item : mask) {
                output_mask.push_back(item != 0);
            }
            auto backward = tpx::ops::linear_backward(
                value(node.inputs[0]),
                value(node.inputs[1]),
                value(node.inputs[2]),
                output_mask);
            env[node.outputs[0]->id] = std::get<0>(backward);
            env[node.outputs[1]->id] = std::get<1>(backward);
            env[node.outputs[2]->id] = std::get<2>(backward);
            return;
        } else if (node.op_type == "t") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax t expects one tensor input");
            }
            result = tpx::ops::t(value(node.inputs[0]));
        } else if (node.op_type == "transpose") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax transpose expects one tensor input");
            }
            result = tpx::ops::transpose(
                value(node.inputs[0]),
                required_int_attr(node, "dim0"),
                required_int_attr(node, "dim1"));
        } else if (node.op_type == "select") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax select expects one tensor input");
            }
            result = tpx::ops::select(
                value(node.inputs[0]),
                required_int_attr(node, "dim"),
                required_int_attr(node, "index"));
        } else if (node.op_type == "slice") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax slice expects one tensor input");
            }
            const auto start_it = node.attrs.find("has_start");
            const auto end_it = node.attrs.find("has_end");
            const bool has_start = start_it != node.attrs.end() &&
                std::holds_alternative<int64_t>(start_it->second) &&
                std::get<int64_t>(start_it->second) != 0;
            const bool has_end = end_it != node.attrs.end() &&
                std::holds_alternative<int64_t>(end_it->second) &&
                std::get<int64_t>(end_it->second) != 0;
            std::optional<int64_t> start;
            std::optional<int64_t> end;
            if (has_start) start = required_int_attr(node, "start");
            if (has_end) end = required_int_attr(node, "end");
            result = tpx::ops::slice(
                value(node.inputs[0]),
                required_int_attr(node, "dim"),
                start,
                end,
                required_int_attr(node, "step"));
        } else if (node.op_type == "stack") {
            if (node.inputs.empty()) {
                throw std::runtime_error("Stax stack expects tensor inputs");
            }
            std::vector<Tensor> tensors;
            tensors.reserve(node.inputs.size());
            for (const ValueNode* input : node.inputs) {
                tensors.push_back(value(input));
            }
            result = tpx::ops::stack(tensors, required_int_attr(node, "dim"));
        } else if (node.op_type == "repeat") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax repeat expects one tensor input");
            }
            result = tpx::ops::repeat(
                value(node.inputs[0]), required_int_list_attr(node, "repeats"));
        } else if (node.op_type == "index_select") {
            if (node.inputs.size() != 2) {
                throw std::runtime_error("Stax index_select expects tensor and index");
            }
            result = tpx::ops::index_select(
                value(node.inputs[0]), required_int_attr(node, "dim"), value(node.inputs[1]));
        } else if (node.op_type == "index_select_backward") {
            if (node.inputs.size() != 2) {
                throw std::runtime_error(
                    "Stax index_select_backward expects gradient and index");
            }
            result = tpx::ops::index_select_backward(
                value(node.inputs[0]),
                required_int_list_attr(node, "self_sizes"),
                required_int_attr(node, "dim"),
                value(node.inputs[1]));
        } else if (node.op_type == "gather") {
            if (node.inputs.size() != 2) {
                throw std::runtime_error("Stax gather expects tensor and index");
            }
            result = tpx::ops::gather(
                value(node.inputs[0]), required_int_attr(node, "dim"), value(node.inputs[1]));
        } else if (node.op_type == "gather_backward") {
            if (node.inputs.size() != 3) {
                throw std::runtime_error(
                    "Stax gather_backward expects gradient, input, and index");
            }
            result = tpx::ops::gather_backward(
                value(node.inputs[0]),
                value(node.inputs[1]),
                required_int_attr(node, "dim"),
                value(node.inputs[2]),
                required_int_attr(node, "sparse_grad") != 0);
        } else if (node.op_type == "embedding") {
            if (node.inputs.size() != 2) {
                throw std::runtime_error("Stax embedding expects weight and indices");
            }
            result = tpx::ops::embedding(
                value(node.inputs[0]),
                value(node.inputs[1]),
                required_int_attr(node, "padding_idx"),
                required_int_attr(node, "scale_grad_by_freq") != 0,
                required_int_attr(node, "sparse") != 0);
        } else if (node.op_type == "constant_pad_nd") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax constant_pad_nd expects one tensor input");
            }
            result = tpx::ops::constant_pad_nd(
                value(node.inputs[0]),
                required_int_list_attr(node, "pad"),
                *scalar_attr(node, "value"));
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
        } else if (node.op_type == "max_pool1d_with_indices" ||
                   node.op_type == "max_pool2d_with_indices" ||
                   node.op_type == "max_pool3d_with_indices") {
            if (node.inputs.size() != 1 || node.outputs.size() != 2) {
                throw std::runtime_error(
                    "Stax max_pool_with_indices expects one input and two outputs");
            }
            const auto& input = value(node.inputs[0]);
            const auto kernel = required_int_list_attr(node, "kernel_size");
            const auto stride = required_int_list_attr(node, "stride");
            const auto padding = required_int_list_attr(node, "padding");
            const auto dilation = required_int_list_attr(node, "dilation");
            const bool ceil_mode = required_int_attr(node, "ceil_mode") != 0;
            std::tuple<Tensor, Tensor> pooled;
            if (node.op_type == "max_pool1d_with_indices") {
                pooled = tpx::ops::max_pool1d_with_indices(
                    input, kernel, stride, padding, dilation, ceil_mode);
            } else if (node.op_type == "max_pool2d_with_indices") {
                pooled = tpx::ops::max_pool2d_with_indices(
                    input, kernel, stride, padding, dilation, ceil_mode);
            } else {
                pooled = tpx::ops::max_pool3d_with_indices(
                    input, kernel, stride, padding, dilation, ceil_mode);
            }
            env[node.outputs[0]->id] = std::get<0>(pooled);
            env[node.outputs[1]->id] = std::get<1>(pooled);
            return;
        } else if (node.op_type == "adaptive_max_pool1d" ||
                   node.op_type == "adaptive_max_pool2d" ||
                   node.op_type == "adaptive_max_pool3d") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax adaptive_max_pool expects one input");
            }
            const auto output_size = required_int_list_attr(node, "output_size");
            if (node.op_type == "adaptive_max_pool1d") {
                auto pooled = tpx::ops::adaptive_max_pool1d(
                    value(node.inputs[0]), output_size);
                result = std::get<0>(pooled);
            } else if (node.op_type == "adaptive_max_pool2d") {
                result = tpx::ops::adaptive_max_pool2d(
                    value(node.inputs[0]), output_size);
            } else {
                auto pooled = tpx::ops::adaptive_max_pool3d(
                    value(node.inputs[0]), output_size);
                result = std::get<0>(pooled);
            }
        } else if (node.op_type == "max_pool1d" ||
                   node.op_type == "max_pool2d" ||
                   node.op_type == "max_pool3d") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax max_pool expects one input");
            }
            const auto& input = value(node.inputs[0]);
            const auto kernel = required_int_list_attr(node, "kernel_size");
            const auto stride = required_int_list_attr(node, "stride");
            const auto padding = required_int_list_attr(node, "padding");
            const auto dilation = required_int_list_attr(node, "dilation");
            const bool ceil_mode = required_int_attr(node, "ceil_mode") != 0;
            if (node.op_type == "max_pool1d") {
                result = tpx::ops::max_pool1d(input, kernel, stride, padding, dilation, ceil_mode);
            } else if (node.op_type == "max_pool2d") {
                result = tpx::ops::max_pool2d(input, kernel, stride, padding, dilation, ceil_mode);
            } else {
                result = tpx::ops::max_pool3d(input, kernel, stride, padding, dilation, ceil_mode);
            }
        } else if (node.op_type == "adaptive_avg_pool1d" ||
                   node.op_type == "adaptive_avg_pool2d" ||
                   node.op_type == "adaptive_avg_pool3d") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax adaptive_avg_pool expects one input");
            }
            const auto& input = value(node.inputs[0]);
            const auto output_size = required_int_list_attr(node, "output_size");
            if (node.op_type == "adaptive_avg_pool1d") {
                result = tpx::ops::adaptive_avg_pool1d(input, output_size);
            } else if (node.op_type == "adaptive_avg_pool2d") {
                result = tpx::ops::adaptive_avg_pool2d(input, output_size);
            } else {
                result = tpx::ops::adaptive_avg_pool3d(input, output_size);
            }
        } else if (node.op_type == "avg_pool1d" ||
                   node.op_type == "avg_pool2d" ||
                   node.op_type == "avg_pool3d") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax avg_pool expects one input");
            }
            std::optional<int64_t> divisor_override;
            const auto divisor_it = node.attrs.find("divisor_override");
            if (divisor_it != node.attrs.end()) {
                if (!std::holds_alternative<int64_t>(divisor_it->second)) {
                    throw std::runtime_error(
                        "Stax avg_pool2d divisor_override has an invalid type");
                }
                divisor_override = std::get<int64_t>(divisor_it->second);
            }
            const auto& input = value(node.inputs[0]);
            const auto kernel = required_int_list_attr(node, "kernel_size");
            const auto stride = required_int_list_attr(node, "stride");
            const auto padding = required_int_list_attr(node, "padding");
            const bool ceil_mode = required_int_attr(node, "ceil_mode") != 0;
            const bool count_include_pad = required_int_attr(node, "count_include_pad") != 0;
            if (node.op_type == "avg_pool1d") {
                result = tpx::ops::avg_pool1d(
                    input, kernel, stride, padding, ceil_mode, count_include_pad);
            } else if (node.op_type == "avg_pool2d") {
                result = tpx::ops::avg_pool2d(
                    input, kernel, stride, padding, ceil_mode, count_include_pad,
                    divisor_override);
            } else {
                result = tpx::ops::avg_pool3d(
                    input, kernel, stride, padding, ceil_mode, count_include_pad,
                    divisor_override);
            }
        } else if (node.op_type == "interpolate") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax interpolate expects one input");
            }
            const auto& input = value(node.inputs[0]);
            const auto output_size = required_int_list_attr(node, "output_size");
            const auto rank_it = node.attrs.find("spatial_rank");
            const int64_t spatial_rank = rank_it == node.attrs.end()
                ? input.dim() - 2
                : required_int_attr(node, "spatial_rank");
            if (spatial_rank == 1) {
                result = tpx::ops::upsample_nearest1d(input, output_size);
            } else if (spatial_rank == 2) {
                result = tpx::ops::upsample_nearest2d(input, output_size);
            } else if (spatial_rank == 3) {
                result = tpx::ops::upsample_nearest3d(input, output_size);
            } else {
                throw std::runtime_error("Stax interpolate expects one to three spatial dimensions");
            }
        } else if (node.op_type == "group_norm") {
            if (node.inputs.empty() || node.inputs.size() > 3) {
                throw std::runtime_error("Stax group_norm has invalid inputs");
            }
            size_t input_index = 1;
            std::optional<Tensor> weight;
            std::optional<Tensor> bias;
            if (required_int_attr(node, "has_weight") != 0) {
                if (input_index >= node.inputs.size()) {
                    throw std::runtime_error("Stax group_norm weight is missing");
                }
                weight = value(node.inputs[input_index++]);
            }
            if (required_int_attr(node, "has_bias") != 0) {
                if (input_index >= node.inputs.size()) {
                    throw std::runtime_error("Stax group_norm bias is missing");
                }
                bias = value(node.inputs[input_index++]);
            }
            if (input_index != node.inputs.size()) {
                throw std::runtime_error("Stax group_norm has unexpected inputs");
            }
            result = tpx::ops::group_norm(
                value(node.inputs[0]),
                required_int_attr(node, "num_groups"),
                weight,
                bias,
                required_float_attr(node, "eps"));
        } else if (node.op_type == "native_group_norm") {
            // The native spelling reports the per-group mean and reciprocal
            // standard deviation next to its result, so a gradient that needs
            // those statistics reads them instead of reducing the input again.
            if (node.inputs.empty() || node.inputs.size() > 3 || node.outputs.size() != 3) {
                throw std::runtime_error("Stax native_group_norm has invalid arity");
            }
            size_t input_index = 1;
            std::optional<Tensor> weight;
            std::optional<Tensor> bias;
            if (required_int_attr(node, "has_weight") != 0) {
                if (input_index >= node.inputs.size()) {
                    throw std::runtime_error("Stax native_group_norm weight is missing");
                }
                weight = value(node.inputs[input_index++]);
            }
            if (required_int_attr(node, "has_bias") != 0) {
                if (input_index >= node.inputs.size()) {
                    throw std::runtime_error("Stax native_group_norm bias is missing");
                }
                bias = value(node.inputs[input_index++]);
            }
            if (input_index != node.inputs.size()) {
                throw std::runtime_error("Stax native_group_norm has unexpected inputs");
            }
            auto fused = tpx::ops::native_group_norm(
                value(node.inputs[0]),
                weight,
                bias,
                required_int_attr(node, "N"),
                required_int_attr(node, "C"),
                required_int_attr(node, "HxW"),
                required_int_attr(node, "group"),
                required_float_attr(node, "eps"));
            env[node.outputs[0]->id] = std::get<0>(fused);
            env[node.outputs[1]->id] = std::get<1>(fused);
            env[node.outputs[2]->id] = std::get<2>(fused);
            return;
        } else if (node.op_type == "dropout") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax dropout expects one input");
            }
            result = tpx::ops::dropout(
                value(node.inputs[0]),
                required_float_attr(node, "p"),
                required_int_attr(node, "training") != 0);
        } else if (node.op_type == "scaled_dot_product_attention") {
            if (node.inputs.size() != 3) {
                throw std::runtime_error(
                    "Stax scaled_dot_product_attention expects query, key, and value");
            }
            // The forward form takes a mask, a dropout rate, a scale and a
            // grouped-query flag; the graph carries none of them here, so each
            // is passed at the value that means "not asked for".
            result = tpx::ops::scaled_dot_product_attention(
                value(node.inputs[0]),
                value(node.inputs[1]),
                value(node.inputs[2]),
                std::nullopt,
                0.0,
                required_int_attr(node, "is_causal") != 0,
                std::nullopt,
                false);
        } else if (node.op_type == "_scaled_dot_product_attention_with_lse") {
            // The fused forward also hands back the softmax normalizer, so a
            // gradient that needs it does not have to rebuild the score
            // matrix.  The node keeps both results; the caller registers the
            // normalizer as a saved value when a gradient reads it.
            if (node.inputs.size() != 3 || node.outputs.size() != 2) {
                throw std::runtime_error(
                    "Stax _scaled_dot_product_attention_with_lse has invalid arity");
            }
            auto fused = tpx::ops::_scaled_dot_product_attention_with_lse(
                value(node.inputs[0]),
                value(node.inputs[1]),
                value(node.inputs[2]),
                required_int_attr(node, "is_causal") != 0,
                required_int_attr(node, "impl"));
            env[node.outputs[0]->id] = std::get<0>(fused);
            env[node.outputs[1]->id] = std::get<1>(fused);
            return;
        } else if (node.op_type == "threshold_backward") {
            if (node.inputs.size() != 2) {
                throw std::runtime_error("Stax threshold_backward expects grad and output");
            }
            result = tpx::ops::threshold_backward(
                value(node.inputs[0]),
                value(node.inputs[1]),
                *scalar_attr(node, "threshold"));
        } else if (node.op_type == "convolution_backward") {
            if (node.inputs.size() != 3 || node.outputs.size() != 3) {
                throw std::runtime_error(
                    "Stax convolution_backward expects three inputs and outputs");
            }
            const auto mask = required_int_list_attr(node, "output_mask");
            if (mask.size() != 3) {
                throw std::runtime_error(
                    "Stax convolution_backward output_mask must have three entries");
            }
            std::vector<bool> output_mask;
            output_mask.reserve(mask.size());
            for (int64_t item : mask) output_mask.push_back(item != 0);
            const auto stride = required_int_list_attr(node, "stride");
            const auto padding = required_int_list_attr(node, "padding");
            const auto dilation = required_int_list_attr(node, "dilation");
            const auto output_padding = required_int_list_attr(node, "output_padding");
            const bool transposed = required_int_attr(node, "transposed") != 0;
            const int64_t groups = required_int_attr(node, "groups");
            auto backward = tpx::ops::convolution_backward(
                value(node.inputs[0]), value(node.inputs[1]), value(node.inputs[2]),
                std::nullopt, stride, padding, dilation, transposed, output_padding,
                groups, output_mask);
            env[node.outputs[0]->id] = std::get<0>(backward);
            env[node.outputs[1]->id] = std::get<1>(backward);
            env[node.outputs[2]->id] = std::get<2>(backward);
            return;
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
        } else if (node.op_type == "max_pool1d_backward" ||
                   node.op_type == "max_pool2d_backward" ||
                   node.op_type == "max_pool3d_backward") {
            if (node.inputs.size() != 2) {
                throw std::runtime_error("Stax max_pool backward expects grad and input");
            }
            const auto& grad = value(node.inputs[0]);
            const auto& input = value(node.inputs[1]);
            const auto kernel = required_int_list_attr(node, "kernel_size");
            const auto stride = required_int_list_attr(node, "stride");
            const auto padding = required_int_list_attr(node, "padding");
            const auto dilation = required_int_list_attr(node, "dilation");
            const bool ceil_mode = required_int_attr(node, "ceil_mode") != 0;
            if (node.op_type == "max_pool1d_backward") {
                if (kernel.size() != 1 || stride.size() != 1 ||
                    padding.size() != 1 || dilation.size() != 1) {
                    throw std::runtime_error("Stax max_pool1d backward expects one-value spatial attributes");
                }
                result = tpx::ops::max_pool2d_backward(
                    tpx::ops::unsqueeze(grad, -2),
                    tpx::ops::unsqueeze(input, -2),
                    {1, kernel[0]}, {1, stride[0]}, {0, padding[0]},
                    {1, dilation[0]}, ceil_mode);
                result = tpx::ops::squeeze(result, -2);
            } else if (node.op_type == "max_pool2d_backward") {
                result = tpx::ops::max_pool2d_backward(
                    grad, input, kernel, stride, padding, dilation, ceil_mode);
            } else {
                result = tpx::ops::max_pool3d_backward(
                    grad, input, kernel, stride, padding, dilation, ceil_mode);
            }
        } else if (node.op_type == "adaptive_avg_pool1d_backward" ||
                   node.op_type == "adaptive_avg_pool2d_backward" ||
                   node.op_type == "adaptive_avg_pool3d_backward") {
            if (node.inputs.size() != 2) {
                throw std::runtime_error("Stax adaptive_avg_pool backward expects grad and input");
            }
            if (node.op_type == "adaptive_avg_pool1d_backward") {
                result = tpx::ops::adaptive_avg_pool2d_backward(
                    tpx::ops::unsqueeze(value(node.inputs[0]), -2),
                    tpx::ops::unsqueeze(value(node.inputs[1]), -2));
                result = tpx::ops::squeeze(result, -2);
            } else if (node.op_type == "adaptive_avg_pool2d_backward") {
                result = tpx::ops::adaptive_avg_pool2d_backward(
                    value(node.inputs[0]), value(node.inputs[1]));
            } else {
                result = tpx::ops::adaptive_avg_pool3d_backward(
                    value(node.inputs[0]), value(node.inputs[1]));
            }
        } else if (node.op_type == "avg_pool1d_backward" ||
                   node.op_type == "avg_pool2d_backward" ||
                   node.op_type == "avg_pool3d_backward") {
            if (node.inputs.size() != 2) {
                throw std::runtime_error("Stax avg_pool backward expects grad and input");
            }
            std::optional<int64_t> divisor_override;
            const auto divisor_it = node.attrs.find("divisor_override");
            if (divisor_it != node.attrs.end()) {
                if (!std::holds_alternative<int64_t>(divisor_it->second)) {
                    throw std::runtime_error(
                        "Stax avg_pool2d_backward divisor_override has an invalid type");
                }
                divisor_override = std::get<int64_t>(divisor_it->second);
            }
            const auto& grad = value(node.inputs[0]);
            const auto& input = value(node.inputs[1]);
            const auto kernel = required_int_list_attr(node, "kernel_size");
            const auto stride = required_int_list_attr(node, "stride");
            const auto padding = required_int_list_attr(node, "padding");
            const bool ceil_mode = required_int_attr(node, "ceil_mode") != 0;
            const bool count_include_pad = required_int_attr(node, "count_include_pad") != 0;
            if (node.op_type == "avg_pool1d_backward") {
                if (kernel.size() != 1 || stride.size() != 1 || padding.size() != 1) {
                    throw std::runtime_error("Stax avg_pool1d backward expects one-value spatial attributes");
                }
                result = tpx::ops::avg_pool2d_backward(
                    tpx::ops::unsqueeze(grad, -2),
                    tpx::ops::unsqueeze(input, -2),
                    {1, kernel[0]}, {1, stride[0]}, {0, padding[0]},
                    ceil_mode, count_include_pad, divisor_override);
                result = tpx::ops::squeeze(result, -2);
            } else if (node.op_type == "avg_pool2d_backward") {
                result = tpx::ops::avg_pool2d_backward(
                    grad, input, kernel, stride, padding, ceil_mode, count_include_pad,
                    divisor_override);
            } else {
                result = tpx::ops::avg_pool3d_backward(
                    grad, input, kernel, stride, padding, ceil_mode, count_include_pad,
                    divisor_override);
            }
        } else if (node.op_type == "upsample_nearest1d_backward" ||
                   node.op_type == "upsample_nearest2d_backward" ||
                   node.op_type == "upsample_nearest3d_backward") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax upsample_nearest backward expects one input");
            }
            const auto& grad = value(node.inputs[0]);
            const auto output_size = required_int_list_attr(node, "output_size");
            const auto input_size = required_int_list_attr(node, "input_size");
            if (node.op_type == "upsample_nearest1d_backward") {
                result = tpx::ops::upsample_nearest1d_backward(
                    grad, output_size, input_size);
            } else if (node.op_type == "upsample_nearest2d_backward") {
                result = tpx::ops::upsample_nearest2d_backward(
                    grad, output_size, input_size);
            } else {
                result = tpx::ops::upsample_nearest3d_backward(
                    grad, output_size, input_size);
            }
        } else if (node.op_type == "group_norm_backward") {
            if (node.inputs.size() < 2 || node.inputs.size() > 4) {
                throw std::runtime_error(
                    "Stax group_norm_backward has invalid inputs");
            }
            size_t input_index = 2;
            std::optional<Tensor> weight;
            std::optional<Tensor> bias;
            if (required_int_attr(node, "has_weight") != 0) {
                if (input_index >= node.inputs.size()) {
                    throw std::runtime_error(
                        "Stax group_norm_backward weight is missing");
                }
                weight = value(node.inputs[input_index++]);
            }
            if (required_int_attr(node, "has_bias") != 0) {
                if (input_index >= node.inputs.size()) {
                    throw std::runtime_error(
                        "Stax group_norm_backward bias is missing");
                }
                bias = value(node.inputs[input_index++]);
            }
            if (input_index != node.inputs.size() || node.outputs.size() != 3) {
                throw std::runtime_error(
                    "Stax group_norm_backward has inconsistent inputs/outputs");
            }
            auto backward = tpx::ops::group_norm_backward(
                value(node.inputs[0]),
                value(node.inputs[1]),
                required_int_attr(node, "num_groups"),
                weight,
                bias,
                required_float_attr(node, "eps"));
            env[node.outputs[0]->id] = std::get<0>(backward);
            env[node.outputs[1]->id] = std::get<1>(backward);
            env[node.outputs[2]->id] = std::get<2>(backward);
            return;
        } else if (node.op_type == "scaled_dot_product_attention_backward") {
            if (node.inputs.size() != 4 || node.outputs.size() != 3) {
                throw std::runtime_error(
                    "Stax scaled_dot_product_attention_backward has invalid arity");
            }
            // The gradient form takes a mask, a dropout rate, a scale and a
            // grouped-query flag as well; the graph carries none of them, so
            // each is passed at the value that means "not asked for".
            auto backward = tpx::ops::scaled_dot_product_attention_backward(
                value(node.inputs[0]),
                value(node.inputs[1]),
                value(node.inputs[2]),
                value(node.inputs[3]),
                std::nullopt,
                0.0,
                required_int_attr(node, "is_causal") != 0,
                std::nullopt,
                false);
            env[node.outputs[0]->id] = std::get<0>(backward);
            env[node.outputs[1]->id] = std::get<1>(backward);
            env[node.outputs[2]->id] = std::get<2>(backward);
            return;
        } else if (node.op_type == "_scaled_dot_product_attention_backward_with_lse") {
            // Reading the softmax normalizer the fused forward produced lets
            // the gradient reuse the score statistics instead of rebuilding
            // the whole score matrix in a separate pass.
            if (node.inputs.size() != 6 || node.outputs.size() != 3) {
                throw std::runtime_error(
                    "Stax _scaled_dot_product_attention_backward_with_lse has invalid arity");
            }
            auto backward =
                tpx::ops::_scaled_dot_product_attention_backward_with_lse(
                    value(node.inputs[0]),
                    value(node.inputs[1]),
                    value(node.inputs[2]),
                    value(node.inputs[3]),
                    value(node.inputs[4]),
                    value(node.inputs[5]),
                    required_int_attr(node, "is_causal") != 0,
                    required_int_attr(node, "impl"));
            env[node.outputs[0]->id] = std::get<0>(backward);
            env[node.outputs[1]->id] = std::get<1>(backward);
            env[node.outputs[2]->id] = std::get<2>(backward);
            return;
        } else if (node.op_type == "native_group_norm_backward") {
            if (node.inputs.size() != 5 || node.outputs.size() != 3) {
                throw std::runtime_error(
                    "Stax native_group_norm_backward has invalid arity");
            }
            const auto mask_it = node.attrs.find("output_mask");
            if (mask_it == node.attrs.end() ||
                !std::holds_alternative<std::vector<int64_t>>(mask_it->second)) {
                throw std::runtime_error(
                    "Stax native_group_norm_backward is missing output_mask");
            }
            const auto& mask = std::get<std::vector<int64_t>>(mask_it->second);
            if (mask.size() != 3) {
                throw std::runtime_error(
                    "Stax native_group_norm_backward output_mask has three entries");
            }
            size_t input_index = 4;
            std::optional<Tensor> weight;
            if (mask[1] != 0) {
                if (input_index < node.inputs.size()) {
                    weight = value(node.inputs[input_index++]);
                }
            }
            if (input_index != node.inputs.size()) {
                throw std::runtime_error(
                    "Stax native_group_norm_backward has unexpected inputs");
            }
            auto backward = tpx::ops::native_group_norm_backward(
                value(node.inputs[0]),
                value(node.inputs[1]),
                value(node.inputs[2]),
                value(node.inputs[3]),
                weight,
                required_int_attr(node, "N"),
                required_int_attr(node, "C"),
                required_int_attr(node, "HxW"),
                required_int_attr(node, "group"),
                {mask[0] != 0, mask[1] != 0, mask[2] != 0});
            for (size_t index = 0; index < 3; ++index) {
                if (mask[index] == 0) {
                    continue;
                }
                if (index == 0) {
                    env[node.outputs[0]->id] = std::get<0>(backward);
                } else if (index == 1) {
                    env[node.outputs[1]->id] = std::get<1>(backward);
                } else {
                    env[node.outputs[2]->id] = std::get<2>(backward);
                }
            }
            return;
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
            return;
        } else if (node.op_type == "chunk" || node.op_type == "split" ||
                   node.op_type == "split_with_sizes" || node.op_type == "unbind") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error(
                    "Stax view partition expects one tensor input");
            }
            const Tensor& input = value(node.inputs[0]);
            const int64_t dim = required_int_attr(node, "dim");
            if (node.op_type == "chunk") {
                multi_result = tpx::ops::chunk(
                    input, required_int_attr(node, "chunks"), dim);
            } else if (node.op_type == "split") {
                auto split_size = node.attrs.find("split_size");
                if (split_size != node.attrs.end()) {
                    if (!std::holds_alternative<int64_t>(split_size->second)) {
                        throw std::runtime_error(
                            "Stax split size has an invalid type");
                    }
                    multi_result = tpx::ops::split(
                        input, std::get<int64_t>(split_size->second), dim);
                } else {
                    multi_result = tpx::ops::split(
                        input, required_int_list_attr(node, "split_sizes"), dim);
                }
            } else if (node.op_type == "split_with_sizes") {
                multi_result = tpx::ops::split_with_sizes(
                    input, required_int_list_attr(node, "split_sizes"), dim);
            } else {
                multi_result = tpx::ops::unbind(input, dim);
            }
            if (multi_result.size() != node.outputs.size()) {
                throw std::runtime_error(
                    "Stax view partition produced an unexpected output count");
            }
            for (size_t index = 0; index < multi_result.size(); ++index) {
                env[node.outputs[index]->id] = std::move(multi_result[index]);
            }
            return;
        } else if (node.op_type == "cat") {
            if (node.inputs.empty()) {
                throw std::runtime_error("Stax cat expects tensor inputs");
            }
            std::vector<Tensor> tensors;
            tensors.reserve(node.inputs.size());
            for (const ValueNode* input : node.inputs) {
                tensors.push_back(value(input));
            }
            result = tpx::ops::cat(
                tensors, required_int_attr(node, "dim"));
        } else if (node.op_type == "zeros_like") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax zeros_like expects one tensor input");
            }
            result = tpx::ops::zeros(
                required_int_list_attr(node, "shape"),
                value(node.inputs[0]).dtype(),
                value(node.inputs[0]).device());
        } else if (node.op_type == "expand") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax expand expects one tensor input");
            }
            const auto implicit_it = node.attrs.find("implicit");
            const bool implicit = implicit_it != node.attrs.end() &&
                std::holds_alternative<int64_t>(implicit_it->second) &&
                std::get<int64_t>(implicit_it->second) != 0;
            result = tpx::ops::expand(
                value(node.inputs[0]),
                required_int_list_attr(node, "shape"),
                implicit);
        } else if (node.op_type == "unsqueeze") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax unsqueeze expects one input");
            }
            result = tpx::ops::unsqueeze(
                value(node.inputs[0]), required_int_attr(node, "dim"));
        } else if (node.op_type == "squeeze") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax squeeze expects one input");
            }
            const auto dim_it = node.attrs.find("dim");
            if (dim_it == node.attrs.end()) {
                const auto dims_it = node.attrs.find("dims");
                if (dims_it == node.attrs.end()) {
                    result = tpx::ops::squeeze(value(node.inputs[0]));
                } else {
                    if (!std::holds_alternative<std::vector<int64_t>>(dims_it->second)) {
                        throw std::runtime_error("Stax squeeze dimensions have an invalid type");
                    }
                    result = tpx::ops::squeeze(
                        value(node.inputs[0]),
                        std::get<std::vector<int64_t>>(dims_it->second));
                }
            } else {
                if (!std::holds_alternative<int64_t>(dim_it->second)) {
                    throw std::runtime_error("Stax squeeze dimension has an invalid type");
                }
                result = tpx::ops::squeeze(
                    value(node.inputs[0]), std::get<int64_t>(dim_it->second));
            }
        } else if (node.op_type == "permute_backward") {
            if (node.inputs.size() != 2) {
                throw std::runtime_error(
                    "Stax permute_backward expects gradient and input");
            }
            result = tpx::ops::permute_backward(
                value(node.inputs[0]),
                value(node.inputs[1]),
                required_int_list_attr(node, "dims"));
        } else if (node.op_type == "reshape") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax reshape expects one input");
            }
            result = tpx::ops::reshape(
                value(node.inputs[0]), required_int_list_attr(node, "shape"));
        } else if (node.op_type == "permute") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax permute expects one input");
            }
            result = tpx::ops::permute(
                value(node.inputs[0]), required_int_list_attr(node, "dims"));
        } else if (node.op_type == "contiguous") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax contiguous expects one input");
            }
            result = tpx::ops::contiguous(
                value(node.inputs[0]), required_int_attr(node, "memory_format"));
        } else if (node.op_type == "float") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax float expects one input");
            }
            result = value(node.inputs[0]).to(DType::Float32);
        } else if (node.op_type == "cast") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax cast expects one input");
            }
            const auto dtype_it = node.attrs.find("dtype");
            if (dtype_it == node.attrs.end() ||
                !std::holds_alternative<std::string>(dtype_it->second)) {
                throw std::runtime_error("Stax cast requires a dtype attribute");
            }
            const auto& name = std::get<std::string>(dtype_it->second);
            DType target = DType::Undefined;
            if (name == "float16" || name == "half") {
                target = DType::Float16;
            } else if (name == "bfloat16") {
                target = DType::BFloat16;
            } else if (name == "float32" || name == "float") {
                target = DType::Float32;
            } else if (name == "float64" || name == "double") {
                target = DType::Float64;
            } else if (name == "bool") {
                target = DType::Bool;
            } else if (name == "int64" || name == "long") {
                target = DType::Int64;
            } else if (name == "int32" || name == "int") {
                target = DType::Int32;
            }
            if (target == DType::Undefined) {
                throw std::runtime_error("Stax cast dtype is unsupported: " + name);
            }
            result = value(node.inputs[0]).to(target);
        } else if (node.op_type == "sum" || node.op_type == "mean") {
            if (node.inputs.size() != 1) {
                throw std::runtime_error("Stax reduction expects one input");
            }
            const auto dim_it = node.attrs.find("dim");
            if (dim_it == node.attrs.end()) {
                result = node.op_type == "sum"
                    ? tpx::ops::sum(value(node.inputs[0]))
                    : tpx::ops::mean(value(node.inputs[0]));
            } else {
                if (!std::holds_alternative<std::vector<int64_t>>(dim_it->second)) {
                    throw std::runtime_error("Stax reduction dim has an invalid type");
                }
                const auto& dims = std::get<std::vector<int64_t>>(dim_it->second);
                const bool keepdim = required_int_attr(node, "keepdim") != 0;
                result = node.op_type == "sum"
                    ? tpx::ops::sum(value(node.inputs[0]), dims, keepdim)
                    : tpx::ops::mean(value(node.inputs[0]), dims, keepdim);
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
                    return;
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
            return;
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
        }
    };

    // Fork-point overlap: with several workers, independent ready nodes run
    // concurrently and the residual-branch chains stop serializing.  Graphs
    // without forks (a single ready node at any time) gain nothing and only
    // pay the synchronization, so they keep the sequential walk.
    bool has_custom_op = false;
    for (const auto& node_ptr : nodes) {
        if (node_ptr->op_type == "custom_op") {
            has_custom_op = true;
            break;
        }
    }
    const bool has_cuda_input = std::any_of(
        inputs.begin(), inputs.end(), [](const Tensor& input) {
            return input.defined() && input.device().is_cuda();
        });
#ifdef _OPENMP
    const unsigned worker_budget = static_cast<unsigned>(omp_get_max_threads());
#else
    const unsigned worker_budget = 1;
#endif
    if (nodes.size() >= 24 && !has_custom_op && worker_budget > 1 &&
        !has_cuda_input) {
        std::vector<int> producer_of(this->values.size(), -1);
        for (size_t i = 0; i < nodes.size(); ++i) {
            for (const auto& o : nodes[i]->outputs) {
                producer_of[o->id] = static_cast<int>(i);
            }
        }
        std::vector<int> indegree(nodes.size(), 0);
        std::vector<std::vector<int>> succs(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            std::set<int> distinct;
            for (const ValueNode* in : nodes[i]->inputs) {
                int producer = producer_of[in->id];
                if (producer >= 0) {
                    distinct.insert(producer);
                }
            }
            indegree[i] = static_cast<int>(distinct.size());
            for (int producer : distinct) {
                succs[producer].push_back(static_cast<int>(i));
            }
        }
        std::mutex sched_mtx;
        std::condition_variable sched_cv;
        std::deque<size_t> ready;
        std::vector<int> pending(nodes.size(), 0);
        for (size_t i = 0; i < nodes.size(); ++i) {
            pending[i] = indegree[i];
            if (indegree[i] == 0) {
                ready.push_back(i);
            }
        }
        size_t completed = 0;
        bool aborted = false;
        std::exception_ptr error;
        const bool grad_enabled = GradMode::is_enabled();
        auto worker = [&]() {
            GradModeGuard grad_mode_guard(grad_enabled);
            for (;;) {
                size_t idx;
                {
                    std::unique_lock<std::mutex> lock(sched_mtx);
                    sched_cv.wait(lock, [&] {
                        return aborted || !ready.empty() || completed == nodes.size();
                    });
                    if (aborted || ready.empty()) {
                        return;
                    }
                    idx = ready.front();
                    ready.pop_front();
                }
                try {
                    run_node(*nodes[idx]);
                } catch (const std::exception& error_value) {
                    std::lock_guard<std::mutex> lock(sched_mtx);
                    if (!error) {
                        error = std::make_exception_ptr(std::runtime_error(
                            "Stax node " + nodes[idx]->name + " (" +
                            nodes[idx]->op_type + "): " + error_value.what()));
                    }
                    aborted = true;
                    sched_cv.notify_all();
                    return;
                } catch (...) {
                    std::lock_guard<std::mutex> lock(sched_mtx);
                    if (!error) {
                        error = std::make_exception_ptr(std::runtime_error(
                            "Stax node " + nodes[idx]->name + " (" +
                            nodes[idx]->op_type + ") failed with an unknown error"));
                    }
                    aborted = true;
                    sched_cv.notify_all();
                    return;
                }
                {
                    std::lock_guard<std::mutex> lock(sched_mtx);
                    release_inputs(*nodes[idx]);
                    for (int succ : succs[idx]) {
                        if (--pending[succ] == 0) {
                            ready.push_back(static_cast<size_t>(succ));
                        }
                    }
                    ++completed;
                }
                sched_cv.notify_all();
            }
        };
        size_t extra = static_cast<size_t>(worker_budget - 1);
        if (extra > 8) {
            extra = 8;
        }
        std::vector<std::thread> threads;
        threads.reserve(extra);
        for (size_t t = 0; t < extra; ++t) {
            threads.emplace_back(worker);
        }
        worker();
        for (auto& thread : threads) {
            thread.join();
        }
        if (error) {
            std::rethrow_exception(error);
        }
    } else {
        for (const auto& node_ptr : nodes) {
            const OpNode& node = *node_ptr;
            try {
                run_node(node);
            } catch (const std::exception& error_value) {
                throw std::runtime_error(
                    "Stax node " + node.name + " (" + node.op_type + "): " +
                    error_value.what());
            }
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
