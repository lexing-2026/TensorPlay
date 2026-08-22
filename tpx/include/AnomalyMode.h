#pragma once
#include <functional>
#include <memory>
#include <string>
#include "Macros.h"

namespace tensorplay {
namespace tpx {

class Node;

// Mirrors torch/csrc/autograd/anomaly_mode.h. The enabling of anomaly mode is
// global: as soon as one guard enables it, every computation and thread is
// affected (this is also torch's semantics). It carries a significant
// performance penalty and should only be used for debugging.
class TENSORPLAY_API AnomalyMode {
public:
    static bool is_enabled() { return _enabled; }
    static bool should_check_nan() { return _check_nan; }
    static void set_enabled(bool enabled, bool check_nan = true) {
        _enabled = enabled;
        _check_nan = check_nan;
    }

private:
    static bool _enabled;
    static bool _check_nan;
};

// Per-node debug metadata, created lazily on first use when AnomalyMode is on.
// Holds the stack captured at forward time plus the parent node that induced
// this node (populated when the node is created during a backward with
// create_graph=True), so an error in the backward can be traced back to the
// forward ops that produced it. Mirrors torch::autograd::AnomalyMetadata.
class TENSORPLAY_API AnomalyMetadata {
public:
    virtual ~AnomalyMetadata();

    // Captures the current stack. The default implementation records the C++
    // stack via tensorplay::get_stacktrace(); the Python bindings install a
    // capture hook so the *Python* traceback of the op call site is recorded
    // instead (same as torch's PyAnomalyMetadata override).
    virtual void store_stack();
    virtual void print_stack(const std::string& current_node_name);

    void assign_parent(const std::shared_ptr<Node>& parent_node) { parent_ = parent_node; }

private:
    std::string traceback_;
    std::weak_ptr<Node> parent_;
};

// Install a custom stack capturer used by AnomalyMetadata::store_stack().
TENSORPLAY_API void set_anomaly_stack_capture(std::function<std::string()> capture);

// The node currently being evaluated by the engine on this thread. New nodes
// created while it is set (i.e. during create_graph backwards) record it as
// their parent. Mirrors torch's tls_current_evaluating_node.
TENSORPLAY_API std::shared_ptr<Node> get_current_evaluating_node();
TENSORPLAY_API void set_current_evaluating_node(const std::shared_ptr<Node>& node);

// RAII guard mirroring torch::autograd::NodeGuard.
class CurrentEvalNodeGuard {
public:
    explicit CurrentEvalNodeGuard(const std::shared_ptr<Node>& node)
        : last_(get_current_evaluating_node()) {
        set_current_evaluating_node(node);
    }
    ~CurrentEvalNodeGuard() { set_current_evaluating_node(last_); }

    CurrentEvalNodeGuard(const CurrentEvalNodeGuard&) = delete;
    CurrentEvalNodeGuard& operator=(const CurrentEvalNodeGuard&) = delete;

private:
    std::shared_ptr<Node> last_;
};

} // namespace tpx
} // namespace tensorplay
