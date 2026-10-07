#pragma once
#include <vector>
#include <cstddef>
#include <optional>
#include <utility>
#include "Tensor.h"
#include "Autograd.h"
#include "Node.h"
#include "Stream.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#ifdef USE_CUDA
#include "CUDARuntime.h"
#endif

namespace tensorplay {
namespace tpx {

// True if `v` is a "vanilla" contiguous tensor that we hold the last reference
// to (both the TensorImpl and its Storage), so accumulating into it in-place
// with add_() is safe.
inline bool can_accumulate_inplace(const Tensor& v) {
    if (!v.is_contiguous()) return false;
    // impl() hands back a reference to the handle's own pointer, so a
    // use_count of 1 means `v` is the only holder.  Any second holder (a
    // leaf's stored grad, a captured gradient, another node's buffer slot)
    // would observe the in-place update.
    if (v.impl().use_count() != 1) return false;
    if (!v.impl()->has_storage()) return false;
    if (v.impl()->storage().use_count() != 1) return false;
    return true;
}

// `grad_mode` is passed explicitly (rather than read from thread-local
// GradMode) because with the multithreaded engine accumulation may run on a
// worker thread whose TLS does not describe this GraphTask.
inline void accumulate(std::vector<Tensor>& buffer, size_t pos, Tensor&& var, bool grad_mode) {
    auto& old_var = buffer[pos];
    if (old_var.is_sparse() && var.is_sparse() &&
        !old_var.is_sparse_compressed() && !var.is_sparse_compressed()) {
        buffer[pos] = tensorplay::tpx::ops::sparse_add(old_var, var);
    } else if (grad_mode) {
        // Under GradMode (e.g. create_graph backward) accumulate through the
        // autograd-aware ops so the second-order graph is built.
        buffer[pos] = tensorplay::tpx::ops::add(old_var, var);
    } else if (can_accumulate_inplace(old_var)) {
        buffer[pos] = old_var.add_(var);
    } else {
        buffer[pos] = old_var + var;
    }
}

#ifdef USE_CUDA
namespace {
// Records a gradient's storage on the given CUDA stream so the caching
// allocator does not reuse it while work on that stream may still read it.
void record_stream_any_impl(Tensor& var, const Stream& stream) {
    if (stream.device_index() != var.device().index()) {
        return;
    }
    if (var.unsafeGetTensorImpl()->has_storage()) {
        void* base_ptr = var.unsafeGetTensorImpl()->storage().data();
        if (base_ptr != nullptr) {
            tensorplay::cuda::recordStream(base_ptr, cuda::toCUDAStream(stream));
        }
    }
}
} // namespace
#endif

// Accumulates gradients for a single Node input at a fixed index (input_nr).
//
// When several producers feed one consumer slot, the producers may run on
// different CUDA streams.  The first producer records the accumulation stream
// and a ready event; every later producer waits on the previous event before
// adding, then records a fresh event for the next producer (and for the
// consumer).  A consuming node waits on the final ready event before reading
// the accumulated gradient, and the engine switches to the consumer's stream
// while evaluating it.
struct InputBuffer {
    InputBuffer() = default;
    explicit InputBuffer(size_t size)
        : buffer(size),
          opt_accum_streams(size),
          ready_events(size),
          ready_streams(size) {}
    InputBuffer(variable_list&& inputs) : buffer(std::move(inputs)) {}
    InputBuffer(InputBuffer&&) = default;
    InputBuffer& operator=(InputBuffer&&) = default;
    InputBuffer(const InputBuffer&) = delete;
    InputBuffer& operator=(InputBuffer&) = delete;

    // Convenience overload used by the engine for single-producer slots
    // (roots) where no stream bookkeeping is required yet.
    void add(size_t pos, Tensor&& var, bool grad_mode = false) {
        add(pos, std::move(var), std::nullopt, std::nullopt, nullptr, grad_mode);
    }

    // The full signature records which stream produced the gradient and which
    // stream the consumer runs on, so accelerator accumulation is ordered
    // correctly.  `fn` is the consumer node (used only for diagnostics).
    void add(
        size_t pos,
        Tensor&& var,
        const std::optional<Stream>& opt_producer_stream,
        const std::optional<Stream>& opt_consumer_stream,
        Node* fn,
        bool grad_mode = false) {
        (void)fn;
        if (pos >= buffer.size() || !var.defined()) return;

        const bool is_accelerator =
            var.device().type() == DeviceType::CUDA ||
            var.device().type() == DeviceType::Vulkan;
        if (!is_accelerator) {
            if (!buffer[pos].defined()) {
                buffer[pos] = std::move(var);
            } else {
                accumulate(buffer, pos, std::move(var), grad_mode);
            }
            return;
        }

#ifdef USE_CUDA
        const auto device = var.device();
        const auto device_type = device.type();
        const std::optional<Stream> opt_producer =
            opt_producer_stream.has_value()
                ? opt_producer_stream
                : std::optional<Stream>(tensorplay::cuda::getCurrentStream(
                      static_cast<int>(device.index())));

        std::optional<Stream> opt_consumer;
        if (opt_overridden_consumer_stream.has_value()) {
            opt_consumer = opt_overridden_consumer_stream;
        } else if (opt_consumer_stream.has_value()) {
            opt_consumer = opt_consumer_stream;
        } else {
            opt_consumer = tensorplay::cuda::getCurrentStream(
                static_cast<int>(device.index()));
        }

        // First producer: determine the accumulation stream, record the
        // ready event the consumer (or a later producer) will wait on.
        if (!opt_accum_streams[pos].has_value()) {
            if (opt_consumer->device() == device) {
                opt_accum_streams[pos] = opt_consumer;
                if (*opt_consumer != *opt_producer) {
                    record_stream_any_impl(var, *opt_consumer);
                }
            } else if (opt_producer->device() == device) {
                opt_accum_streams[pos] = opt_producer;
            } else {
                opt_accum_streams[pos] = tensorplay::cuda::getCurrentStream(
                    static_cast<int>(device.index()));
            }
            buffer[pos] = std::move(var);
            const auto& accum_stream = opt_accum_streams[pos];
            if (*opt_consumer != *opt_producer ||
                *accum_stream != *opt_producer) {
                cuda::CUDAEvent event;
                event.record(cuda::toCUDAStream(*opt_producer));
                ready_events[pos] = std::move(event);
            }
            ready_streams[pos] = opt_producer;
            return;
        }

        // Nth producer: wait for the earlier producer's event, accumulate on
        // the accumulation stream, then record a fresh event for the
        // consumer (and any later producer).
        const auto accum_stream = opt_accum_streams[pos];
        const auto& ready_event = ready_events[pos];
        const auto& ready_stream = ready_streams[pos];
        if (*accum_stream != *opt_producer) {
            cuda::CUDAEvent event;
            event.record(cuda::toCUDAStream(*opt_producer));
            event.block(cuda::toCUDAStream(*accum_stream));
            record_stream_any_impl(var, *accum_stream);
        }
        if (*accum_stream != *ready_stream) {
            if (ready_event.has_value()) {
                ready_event->block(cuda::toCUDAStream(*accum_stream));
            }
            record_stream_any_impl(buffer[pos], *accum_stream);
        }
        cuda::CUDAStreamGuard stream_guard(cuda::toCUDAStream(*accum_stream));
        accumulate(buffer, pos, std::move(var), grad_mode);
        if (*opt_consumer != *accum_stream) {
            cuda::CUDAEvent event;
            event.record(cuda::toCUDAStream(*accum_stream));
            ready_events[pos] = std::move(event);
        }
        ready_streams[pos] = accum_stream;
#else
        // Without CUDA, accelerator accumulation cannot be ordered through
        // events; fall back to plain accumulation (Vulkan has no event
        // support in this file).
        if (!buffer[pos].defined()) {
            buffer[pos] = std::move(var);
        } else {
            accumulate(buffer, pos, std::move(var), grad_mode);
        }
#endif
    }

    Tensor operator[](size_t pos) { return buffer[pos]; }

    // Device of the first defined input; used by the engine to route a
    // NodeTask to the owning device's ready queue.
    int device_index() const {
        for (const auto& t : buffer) {
            if (t.defined() && t.device().is_cuda()) {
                return static_cast<int>(t.device().index());
            }
        }
        return -1; // CPU / unspecified
    }

    static variable_list variables(InputBuffer&& g) {
        return std::move(g.buffer);
    }

    std::vector<Tensor> buffer;
    // The stream used for accumulation when a slot receives multiple
    // producers.
    std::vector<std::optional<Stream>> opt_accum_streams;
    // Events the consumer must wait on before reading each slot; updated as
    // producers accumulate.
    std::vector<std::optional<cuda::CUDAEvent>> ready_events;
    // The streams the ready events were recorded on.
    std::vector<std::optional<Stream>> ready_streams;
    // The stream the consumer is moved to once its slots have started
    // filling.  Nothing in this engine moves a consumer yet, so it stays
    // empty and the stream passed to add() is the one used.
    std::optional<Stream> opt_overridden_consumer_stream;
};

} // namespace tpx
} // namespace tensorplay
