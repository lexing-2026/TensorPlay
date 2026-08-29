#include "AnomalyMode.h"
#include "Node.h"
#include "Stacktrace.h"
#include <cstdio>

namespace tensorplay {
namespace tpx {

bool AnomalyMode::_enabled = false;
bool AnomalyMode::_check_nan = true;

namespace {

std::function<std::string()>& stack_capture_fn() {
    static std::function<std::string()> fn;
    return fn;
}

// The engine may evaluate several nodes of one graph concurrently; the parent
thread_local std::weak_ptr<Node> tls_current_evaluating_node;

} // namespace

void set_anomaly_stack_capture(std::function<std::string()> capture) {
    stack_capture_fn() = std::move(capture);
}

std::shared_ptr<Node> get_current_evaluating_node() {
    return tls_current_evaluating_node.lock();
}

void set_current_evaluating_node(const std::shared_ptr<Node>& node) {
    tls_current_evaluating_node = node;
}

AnomalyMetadata::~AnomalyMetadata() = default;

void AnomalyMetadata::store_stack() {
    if (auto& fn = stack_capture_fn()) {
        traceback_ = fn();
        return;
    }
    traceback_ = tensorplay::get_stacktrace();
}

void AnomalyMetadata::print_stack(const std::string& current_node_name) {
    fprintf(stderr,
            "Error detected in %s. Traceback of forward call that caused the error:\n%s",
            current_node_name.c_str(),
            traceback_.c_str());

    auto cur_parent = parent_.lock();
    while (cur_parent) {
        auto* parent_metadata = cur_parent->anomaly_metadata();
        if (!parent_metadata) break;
        fprintf(stderr,
                "\n\nPrevious calculation was induced by %s. Traceback of forward call "
                "that induced the previous calculation:\n%s",
                cur_parent->name().c_str(),
                parent_metadata->traceback_.c_str());
        cur_parent = parent_metadata->parent_.lock();
    }
}

} // namespace tpx
} // namespace tensorplay
