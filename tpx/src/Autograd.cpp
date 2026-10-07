#include "Autograd.h"
#include "ForwardFallback.h"
#include "TensorImpl.h"
#include "AccumulateGrad.h"
#include "Engine.h"
#include "InputBuffer.h"
#include "ManualNodes.h" // For AsStridedBackward
#include "LocalDispatchKeySet.h"
#include "TransformDispatch.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "tensorplay/ops/AutogradNodesGenerated.h"
#ifdef USE_CUDA
#include "CUDAGenerator.h"
#include "CUDARuntime.h"
#endif

#include <set>

namespace tensorplay {
namespace tpx {

namespace impl {

AutogradMeta* get_autograd_meta(const Tensor& t) {
    if (auto* impl = t.unsafeGetTensorImpl().get()) {
        return static_cast<AutogradMeta*>(impl->autograd_meta());
    }
    return nullptr;
}

AutogradMeta* get_or_create_autograd_meta(const Tensor& t) {
    auto impl = t.unsafeGetTensorImpl();
    if (!impl) return nullptr;
    if (auto* meta = impl->autograd_meta()) {
        return static_cast<AutogradMeta*>(meta);
    }
    auto meta = std::make_shared<AutogradMeta>();
    auto* raw = meta.get();
    impl->set_autograd_meta(std::move(meta));
    return raw;
}

std::shared_ptr<Node> grad_fn(const Tensor& t) {
    auto* meta = get_autograd_meta(t);
    if (!meta) return nullptr;
    if (!meta->has_view_info()) return meta->grad_fn();

    std::lock_guard<std::mutex> lock(meta->view_mutex());
    const auto& base = meta->view_base();
    if (!base.defined()) return meta->grad_fn();

    const uint32_t current_version = t.unsafeGetTensorImpl()->version();
    if (meta->attr_version() == current_version) return meta->grad_fn();
    if (!meta->grad_fn() && !base.requires_grad()) {
        meta->set_attr_version(current_version);
        return nullptr;
    }
    if (meta->creation_meta() != CreationMeta::DEFAULT) {
        TP_THROW(RuntimeError,
                 "a view was modified after its backward history became stale");
    }

    std::shared_ptr<Node> refreshed;
    if (meta->has_view_fn()) {
        const bool previous_grad_mode = GradMode::is_enabled();
        GradMode::set_enabled(true);
        try {
            Tensor replay = meta->view_fn()(base);
            refreshed = grad_fn(replay);
        } catch (...) {
            GradMode::set_enabled(previous_grad_mode);
            throw;
        }
        GradMode::set_enabled(previous_grad_mode);
    } else {
        // The view's current geometry, absolute on the storage it shares
        // with the base: an in-place update of the view's own shape
        // (t_, squeeze_) is part of what the view now reads.
        refreshed = std::make_shared<AsStridedBackward>(
            base, static_cast<std::vector<int64_t>>(t.shape()), t.strides(),
            std::optional<int64_t>(
                static_cast<int64_t>(t.unsafeGetTensorImpl()->storage_offset())));
        refreshed->set_view_fn(true);
        refreshed->add_next_edge_list(collect_next_edges(base));
    }
    std::shared_ptr<Node> previous = meta->retains_grad() ? meta->grad_fn() : nullptr;
    meta->set_grad_fn(std::move(refreshed));
    meta->set_attr_version(current_version);
    if (meta->retains_grad()) move_retains_grad_hook(t, previous, meta->output_nr());
    return meta->grad_fn();
}

namespace {
// Adds each gradient a node receives for `output_nr` into the .grad of the
// tensor it produced.  The tensor is held weakly -- the node is reachable
// from it -- and the gradient is handed on unchanged.  The first gradient is
// copied, so a later in-place step of the backward cannot change it.
Node::PreHookFn retains_grad_hook(const Tensor& t, uint32_t output_nr) {
    weak_intrusive_ptr<TensorImpl> weak(t.unsafeGetTensorImpl());
    return [weak, output_nr](variable_list&& grads) -> variable_list {
        if (output_nr >= grads.size() || !grads[output_nr].defined()) return std::move(grads);
        intrusive_ptr<TensorImpl> owner = weak.lock();
        if (!owner) return std::move(grads);
        const Tensor self(std::move(owner));
        auto* meta = get_autograd_meta(self);
        if (meta == nullptr) return std::move(grads);
        const Tensor& incoming = grads[output_nr];
        const Tensor current = meta->grad();
        meta->set_grad(current.defined() ? ops::add(current, incoming) : ops::clone(incoming));
        return std::move(grads);
    };
}
} // namespace

void retain_grad(const Tensor& t) {
    if (!t.requires_grad()) {
        TP_THROW(RuntimeError, "can't retain_grad on Tensor that has requires_grad=False");
    }
    std::shared_ptr<Node> fn = grad_fn(t);
    if (!fn) return;
    auto* meta = get_or_create_autograd_meta(t);
    if (meta == nullptr || meta->retains_grad()) return;
    const uint32_t nr = output_nr(t);
    fn->retains_grad_hooks()[nr] = retains_grad_hook(t, nr);
    meta->set_retains_grad(true);
}

void move_retains_grad_hook(const Tensor& t, const std::shared_ptr<Node>& previous,
                            uint32_t previous_nr) {
    auto* meta = get_autograd_meta(t);
    if (meta == nullptr || !meta->retains_grad()) return;
    const std::shared_ptr<Node> current = meta->grad_fn();
    const uint32_t nr = meta->output_nr();
    if (current == previous && nr == previous_nr) return;
    if (previous) previous->retains_grad_hooks().erase(previous_nr);
    if (current) current->retains_grad_hooks()[nr] = retains_grad_hook(t, nr);
}

void set_requires_grad(const Tensor& t, bool requires_grad) {
    auto impl = t.unsafeGetTensorImpl();
    if (!impl) return;
    if (!requires_grad && !impl->autograd_meta()) return;

    // The p10 layer performs the invariant check before metadata allocation.
    impl->set_requires_grad(requires_grad);
    if (requires_grad && !impl->autograd_meta()) {
        if (auto* meta = get_or_create_autograd_meta(t)) {
            meta->set_requires_grad(true);
        }
    }
}

void set_view_metadata(
    const Tensor& view,
    const Tensor& base,
    CreationMeta creation_meta,
    std::function<Tensor(const Tensor&)> view_fn,
    bool force_view_fn) {
    if (!view.defined() || !base.defined()) return;
    auto view_impl = view.unsafeGetTensorImpl();
    auto base_impl = base.unsafeGetTensorImpl();
    if (!view_impl || !base_impl || view_impl == base_impl) return;
    if (view_impl->is_inference()) return;
    if (!view_impl->has_storage() || !base_impl->has_storage() ||
        !view_impl->storage().is_same(base_impl->storage())) {
        return;
    }
    Tensor root_base = base;
    auto* base_meta = get_autograd_meta(base);
    if (base_meta && base_meta->has_view_info() &&
        base_meta->view_base().defined()) {
        root_base = base_meta->view_base();
    }
    if (!root_base.defined()) return;

    if (!force_view_fn && view.dtype() == base.dtype()) {
        view_fn = {};
    } else if (view_fn && base_meta && base_meta->has_view_info()) {
        std::function<Tensor(const Tensor&)> parent_view_fn;
        if (base_meta->has_view_fn()) {
            parent_view_fn = base_meta->view_fn();
        } else {
            const auto parent_size =
                static_cast<std::vector<int64_t>>(base.shape());
            const auto parent_stride = base.strides();
            const int64_t parent_offset = static_cast<int64_t>(
                base_impl->storage_offset());
            const int64_t root_offset = static_cast<int64_t>(
                root_base.unsafeGetTensorImpl()->storage_offset());
            const int64_t relative_offset = parent_offset - root_offset;
            parent_view_fn = [parent_size, parent_stride, relative_offset](
                                 const Tensor& root) {
                return root.as_strided(
                    parent_size, parent_stride, relative_offset);
            };
        }
        auto current_view_fn = std::move(view_fn);
        view_fn = [parent_view_fn = std::move(parent_view_fn),
                   current_view_fn = std::move(current_view_fn)](
                      const Tensor& root) {
            return current_view_fn(parent_view_fn(root));
        };
    }
    if (auto* meta = get_or_create_autograd_meta(view)) {
        meta->set_view_info(view, root_base, creation_meta, std::move(view_fn));
    }
}

bool has_view_metadata(const Tensor& t) {
    auto* meta = get_autograd_meta(t);
    return meta != nullptr && meta->has_view_info();
}

void rebase_history(const Tensor& self, std::shared_ptr<Node> grad_fn) {
    TP_CHECK(grad_fn != nullptr, "rebase_history requires a backward node");
    auto* meta = get_autograd_meta(self);
    if (!meta || !meta->has_view_info()) {
        set_grad_fn(self, std::move(grad_fn));
        return;
    }
    if (meta->creation_meta() != CreationMeta::DEFAULT) {
        TP_THROW(RuntimeError,
                 "a view with restricted mutation history cannot be modified inplace");
    }
    const Tensor base = meta->view_base();
    TP_CHECK(base.defined(), "view metadata has no base tensor");
    std::function<Tensor(const Tensor&)> view_fn;
    if (meta->has_view_fn()) view_fn = meta->view_fn();
    auto copy_slices = std::make_shared<CopySlices>(
        base, self, std::move(view_fn), std::move(grad_fn));
    set_grad_fn(base, copy_slices);
    (void)impl::grad_fn(self);
}

Tensor fw_grad(const Tensor& t, uint64_t level) {
    if (!t.defined()) return ForwardGrad::undef_grad();
    auto* meta = get_autograd_meta(t);
    if (!meta) return ForwardGrad::undef_grad();
    return meta->fw_grad(level, t);
}

void set_fw_grad(const Tensor& t, const Tensor& new_grad, uint64_t level,
                 bool is_inplace_op) {
    auto* meta = get_or_create_autograd_meta(t);
    TP_CHECK(meta != nullptr, "cannot attach a forward gradient to this tensor");
    meta->set_fw_grad(new_grad, t, level, is_inplace_op);
}

std::vector<Tensor> fw_grads_from_backward(const char* op_name,
                                           const std::vector<Tensor>& inputs,
                                           const ForwardRerun& rerun) {
    // Under the caller's grad mode, a primal that is part of the caller's
    // graph stays in it, so the tangent is differentiable in it too; any
    // other primal becomes a fresh leaf.  Only inputs with a tangent are
    // differentiated; the rest run as they are.
    const bool caller_records = GradMode::is_enabled();
    std::vector<Tensor> args;
    args.reserve(inputs.size());
    std::vector<Tensor> diff_inputs, tangents;
    bool needs_graph = false;
    for (const Tensor& in : inputs) {
        const Tensor tangent = in.defined() ? fw_grad(in, 0) : Tensor();
        if (!tangent.defined() || !isFloatingOrComplexType(in.dtype())) {
            args.push_back(in);
            continue;
        }
        Tensor primal = to_non_opt_primal(in);
        if (caller_records && primal.requires_grad()) {
            needs_graph = true;
        } else {
            primal = primal.detach();
            set_requires_grad(primal, true);
        }
        needs_graph = needs_graph || (caller_records && tangent.requires_grad());
        diff_inputs.push_back(primal);
        tangents.push_back(tangent);
        args.push_back(std::move(primal));
    }
    TP_CHECK(!InferenceMode::is_enabled() || diff_inputs.empty(),
             "Trying to use forward AD with ", op_name,
             " in inference mode, which records no backward to take its tangent from.");

    // The products are recorded whatever the caller's mode: forward AD
    // answers under no_grad too.
    struct ModeRestore {
        bool previous;
        ~ModeRestore() { GradMode::set_enabled(previous); }
    } restore{caller_records};
    GradMode::set_enabled(true);

    const std::vector<Tensor> outs = rerun(args);
    std::vector<Tensor> result(outs.size());
    if (diff_inputs.empty()) return result;

    std::vector<Tensor> roots, cotangents;
    std::vector<size_t> where;
    for (size_t i = 0; i < outs.size(); ++i) {
        const Tensor& out = outs[i];
        if (!out.defined() || !out.requires_grad()) continue;
        Tensor u = ops::zeros_like(out);
        set_requires_grad(u, true);
        roots.push_back(out);
        cotangents.push_back(std::move(u));
        where.push_back(i);
    }
    if (roots.empty()) return result;

    // vjp(u) = J^H u, recorded; its derivative in u along t is J t.
    const std::vector<Tensor> vjp = tensorplay::tpx::grad(
        roots, diff_inputs, cotangents,
        /*retain_graph=*/true, /*create_graph=*/true, /*allow_unused=*/true);
    std::vector<Tensor> products, product_tangents;
    for (size_t k = 0; k < vjp.size(); ++k) {
        if (vjp[k].defined() && vjp[k].requires_grad()) {
            products.push_back(vjp[k]);
            product_tangents.push_back(tangents[k]);
        }
    }
    std::vector<Tensor> jt;
    if (!products.empty()) {
        jt = tensorplay::tpx::grad(products, cotangents, product_tangents,
                                   /*retain_graph=*/needs_graph, /*create_graph=*/needs_graph,
                                   /*allow_unused=*/true);
    }
    GradMode::set_enabled(caller_records);
    // A backward that hands its gradient straight through (clone, a copy)
    // returns the input tangent itself; the output gets its own, so an
    // in-place update of one does not write the other.
    auto shares_an_input_tangent = [&](const Tensor& r) {
        for (const Tensor& t : tangents) {
            if (r.unsafeGetTensorImpl()->storage().is_same(t.unsafeGetTensorImpl()->storage())) {
                return true;
            }
        }
        return false;
    };
    for (size_t j = 0; j < where.size(); ++j) {
        const size_t i = where[j];
        if (j < jt.size() && jt[j].defined()) {
            result[i] = shares_an_input_tangent(jt[j]) ? jt[j].clone() : jt[j];
        } else {
            result[i] = ops::zeros_like(outs[i]);
        }
    }
    return result;
}

RandomReplay::RandomReplay(const std::vector<Tensor>& inputs,
                           const std::optional<Generator>& generator)
    : generator_(generator) {
    cpu_state_ = (generator_ ? *generator_ : default_generator()).get_state();
#ifdef USE_CUDA
    std::set<int> devices;
    for (const Tensor& t : inputs) {
        if (t.defined() && t.device().is_cuda()) {
            devices.insert(t.device().index() >= 0 ? static_cast<int>(t.device().index())
                                                   : tensorplay::cuda::currentDevice());
        }
    }
    for (int device : devices) {
        tensorplay::cuda::CUDAGuard guard(device);
        cuda_states_.emplace_back(device, tensorplay::cuda::get_rng_state());
    }
#else
    (void)inputs;
#endif
}

std::vector<Tensor> RandomReplay::run(const std::function<std::vector<Tensor>()>& fn) const {
    // Generators are handles: this one shares the stream it names.
    Generator cpu = generator_ ? *generator_ : default_generator();
    struct Streams {
        Generator cpu;
        Tensor cpu_state;
        std::vector<std::pair<int, Tensor>> cuda_states;
        ~Streams() {
            cpu.set_state(cpu_state);
#ifdef USE_CUDA
            for (const auto& [device, state] : cuda_states) {
                tensorplay::cuda::CUDAGuard guard(device);
                tensorplay::cuda::set_rng_state(state);
            }
#endif
        }
    } now{cpu, cpu.get_state(), {}};
#ifdef USE_CUDA
    for (const auto& [device, state] : cuda_states_) {
        tensorplay::cuda::CUDAGuard guard(device);
        now.cuda_states.emplace_back(device, tensorplay::cuda::get_rng_state());
        tensorplay::cuda::set_rng_state(state);
    }
#endif
    cpu.set_state(cpu_state_);
    return fn();
}

void set_output_fw_grad(const Tensor& out, const Tensor& tangent) {
    if (!tangent.defined() || !out.defined() || !isFloatingOrComplexType(out.dtype())) return;
    if (is_fw_grad_defined(out, 0)) return;
    set_fw_grad(out, tangent.dtype() == out.dtype() ? tangent : tangent.to(out.dtype()),
                /* level */ 0, /* is_inplace_op */ false);
}

void set_inplace_fw_grad(const Tensor& self, const Tensor& tangent) {
    if (!tangent.defined() || !self.defined() || !isFloatingOrComplexType(self.dtype())) return;
    const Tensor value = tangent.dtype() == self.dtype() ? tangent : tangent.to(self.dtype());
    Tensor existing = fw_grad(self, 0);
    if (existing.defined()) {
        if (existing.unsafeGetTensorImpl() != value.unsafeGetTensorImpl()) {
            ops::copy_(existing, value);
        }
        return;
    }
    set_fw_grad(self, value, /* level */ 0, /* is_inplace_op */ true);
}

void refuse_forward_ad(const char* op_name, bool out_variant) {
    TP_THROW(NotImplementedError, "Trying to use forward AD with ", op_name,
             " that does not support it because ",
             out_variant ? "it is an out= function." : "it has not been implemented yet.");
}

Tensor to_non_opt_fw_grad(const Tensor& t) {
    return t.defined() ? fw_grad(t, 0) : Tensor();
}

Tensor to_non_opt_primal(const Tensor& t) {
    if (t.defined()) {
        if (t.unsafeGetTensorImpl()->is_wrapped_number()) {
            return t;
        }
        return ops::_fw_primal(t, 0);
    }
    return Tensor();
}

// Checks whether the tangent has the same layout (sizes, strides, storage
// offset) as the primal; a mismatch is repaired with a fresh zero buffer so
// in-place tangent updates stay valid.
bool has_same_fw_meta(const Tensor& base, const Tensor& other) {
    if (!base.defined() || !other.defined()) return false;
    if (base.dim() != other.dim()) return false;
    if (base.sizes() != other.sizes()) return false;
    if (base.numel() == 0 && other.numel() == 0) return true;
    if (base.unsafeGetTensorImpl()->storage_offset() !=
        other.unsafeGetTensorImpl()->storage_offset()) {
        return false;
    }
    const auto base_strides = base.unsafeGetTensorImpl()->strides();
    const auto other_strides = other.unsafeGetTensorImpl()->strides();
    const auto base_sizes = base.sizes();
    for (size_t i = 0; i < base_strides.size(); ++i) {
        if (base_strides[i] != other_strides[i] && base_sizes[i] != 1 &&
            base_sizes[i] != 0) {
            return false;
        }
    }
    return true;
}

} // namespace impl

void AutogradMeta::accum_grad(const tensorplay::Tensor& grad) {
    if (!grad_.defined()) {
        // The first gradient is kept as-is only when the caller's handle is
        // its sole holder and no second-order graph is being recorded.  A
        // gradient that fans out (an addend's gradient reaching two leaves,
        // or a leaf and a buffered interior node) is deep-copied, so later
        // in-place updates of this slot or of the other holder stay private.
        if (!GradMode::is_enabled() && grad.impl().use_count() <= 1) {
            grad_ = grad;
        } else {
            grad_ = grad.clone();
        }
    } else if (grad_.is_sparse() && grad.is_sparse() &&
               !grad_.is_sparse_compressed() && !grad.is_sparse_compressed()) {
        grad_ = ops::sparse_add(grad_, grad);
    } else if (!GradMode::is_enabled()) {
        // First-order accumulation keeps the stored tensor's identity:
        // handles to `.grad` taken earlier observe the running sum.
        grad_ += grad;
    } else {
        grad_ = grad_ + grad;
    }
}

AutogradMeta::~AutogradMeta() {
    if (fw_grad_) {
        fw_grad_->clear();
    }
}

void AutogradMeta::set_fw_grad(
    const tensorplay::Tensor& new_grad_base,
    const tensorplay::Tensor& self_base,
    uint64_t level,
    bool is_inplace_op) {
    TP_CHECK(
        !impl::fw_grad(new_grad_base, level).defined(),
        "Setting a forward grad that itself has a forward gradient at the "
        "same level is not supported.");
    TP_CHECK(
        isFloatingOrComplexType(new_grad_base.dtype()) &&
            isFloatingOrComplexType(self_base.dtype()),
        "Expected both tensor and its forward grad to be floating point or complex");
    // Lazy initialization
    {
        std::lock_guard<std::mutex> lock(fw_mutex_);
        if (!fw_grad_) {
            fw_grad_ = std::make_shared<ForwardGrad>();
        }
    }
    if (fw_grad_->contains(level)) {
        // Setting the forward grad again is only allowed if it is a no-op.
        // In-place ops may re-set it so their code generation stays simple.
        TP_CHECK(
            new_grad_base.defined(),
            "Cannot set a forward grad that is an undefined Tensor. Use "
            "_fw_primal(level) to get a new Tensor with this forward grad unset.");
        TP_CHECK(
            is_inplace_op,
            "Only inplace operations can re-set the forward grad of a Tensor "
            "that already has one.");
        TP_CHECK(
            fw_grad_->value(level).unsafeGetTensorImpl() ==
                new_grad_base.unsafeGetTensorImpl(),
            "Cannot set a value of a forward grad if it already exists. "
            "Inplace operations should modify it inplace.");
    } else {
        Tensor new_grad = new_grad_base;

        TP_CHECK(
            self_base.sizes() == new_grad.sizes(),
            "Trying to set a forward gradient that has a different size than "
            "that of the original Tensor, this is not supported.");

        if (is_inplace_op && has_view_info_ && view_base_.defined()) {
            // In-place op on a view without a prior tangent: propagate the
            // tangent to the base and make this tensor's tangent a view of
            // the base's tangent, keeping the view relation consistent.
            const Tensor& base = view_base_;
            if (!impl::fw_grad(base, level).defined()) {
                Tensor new_base_fw_grad;
                if (impl::has_same_fw_meta(new_grad, base) &&
                    impl::has_same_fw_meta(new_grad, self_base)) {
                    new_base_fw_grad = new_grad;
                } else {
                    new_base_fw_grad =
                        ops::_new_zeros_with_same_feature_meta(new_grad, base);

                    Tensor new_fw_grad_value;
                    if (has_view_fn()) {
                        new_fw_grad_value = view_fn()(new_base_fw_grad);
                    } else {
                        new_fw_grad_value = new_base_fw_grad.as_strided(
                            self_base.shape(), self_base.strides(),
                            static_cast<int64_t>(
                                self_base.unsafeGetTensorImpl()->storage_offset()));
                    }

                    new_fw_grad_value.copy_(new_grad);
                    new_grad = std::move(new_fw_grad_value);
                }
                impl::set_fw_grad(base, new_base_fw_grad, level,
                                  /* is_inplace_op */ false);
            }
        }

        // Enforce the basic layout constraint: the tangent must share the
        // primal's shape so in-place tangent updates stay valid.  The fresh
        // buffer is contiguous -- reproducing a degenerate primal layout
        // (e.g. stride-0 broadcast dims) would make the copy collapse.
        if (!impl::has_same_fw_meta(new_grad, self_base)) {
            auto res = ops::zeros(
                static_cast<std::vector<int64_t>>(new_grad.shape()),
                new_grad.dtype(), new_grad.device());
            res.copy_(new_grad);
            new_grad = std::move(res);
        }

        fw_grad_->set_value(std::move(new_grad), level);
    }
}

const tensorplay::Tensor& AutogradMeta::fw_grad(
    uint64_t level,
    const tensorplay::Tensor& self) const {
    if (!FwGradMode::is_enabled()) {
        return ForwardGrad::undef_grad();
    }

    std::lock_guard<std::mutex> lock(fw_mutex_);

    const Tensor& direct_fw_grad =
        fw_grad_ ? fw_grad_->value(level) : ForwardGrad::undef_grad();

    if (!direct_fw_grad.defined() && has_view_info_ && view_base_.defined()) {
        // A view without its own tangent reads the base's tangent through the
        // view relation; the value is cached so later reads are direct.
        const Tensor& base = view_base_;
        const Tensor& base_val = impl::fw_grad(base, level);
        if (base_val.defined()) {
            fw_grad_ = std::make_shared<ForwardGrad>();

            Tensor new_val;
            if (has_view_fn()) {
                new_val = view_fn()(base_val);
            } else {
                new_val = base_val.as_strided(
                    self.shape(), self.strides(),
                    static_cast<int64_t>(
                        self.unsafeGetTensorImpl()->storage_offset()));
            }

            fw_grad_->set_value(std::move(new_val), level);
            return fw_grad_->value(level);
        }
    }
    return direct_fw_grad;
}

std::vector<Edge> collect_next_edges(const Tensor& t) {
    std::vector<Edge> edges;
    if (impl::requires_grad(t)) {
        // Record the forward shape on every edge regardless of target kind:
        // the engine reduces broadcast-inflated grads back to it.  The dtype
        // casts floating gradients to it before the consumer node runs.
        // Dimension-by-dimension copy: no intermediate Size/vector materialize.
        const size_t ndim = t.dim();
        std::vector<int64_t> shape(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            shape[i] = t.size(i);
        }
        const DType dt = t.dtype();
        auto fill_metadata = [&](Edge& edge) {
            edge.grad_dtype = dt;
            edge.device_type_hint = t.device().type();
            edge.device_index_hint = t.device().index();
#ifdef USE_CUDA
            if (t.device().is_cuda()) {
                edge.stream = tensorplay::cuda::getCurrentStream(
                    static_cast<int>(t.device().index()));
            } else {
                edge.stream = Stream(Stream::DEFAULT, t.device());
            }
#else
            edge.stream = Stream(Stream::DEFAULT, t.device());
#endif
        };
        auto fn = impl::grad_fn(t);
        if (fn) {
            Edge edge(std::move(fn), impl::output_nr(t), std::move(shape));
            fill_metadata(edge);
            edges.push_back(std::move(edge));
        } else {
            // Leaf
            auto* meta = impl::get_autograd_meta(t);
            if (meta) {
                // Hold a strong reference locally: the graph edge becomes its
                // owner, while the tensor only keeps a weak cache reference.
                std::shared_ptr<Node> acc = meta->grad_accumulator();
                if (!acc) {
                    acc = std::make_shared<AccumulateGrad>(t);
                    meta->set_grad_accumulator(acc);
                }
                Edge edge(std::move(acc), 0, std::move(shape));
                fill_metadata(edge);
                edges.push_back(std::move(edge));
            } else {
                edges.emplace_back();
            }
        }
    } else {
        edges.emplace_back();
    }
    return edges;
}

std::vector<Edge> collect_next_edges(const std::optional<Tensor>& t) {
    if (t.has_value()) {
        return collect_next_edges(*t);
    }
    return {Edge()};
}

namespace impl {
bool is_view_of_leaf(const Tensor& t) {
    // Walk the grad_fn chain through view nodes; if it terminates at an
    // AccumulateGrad the (transitive) base is a leaf that requires grad.
    auto fn = grad_fn(t);
    while (fn && fn->is_view_fn()) {
        const auto& edges = fn->next_edges();
        if (edges.empty()) return false;
        fn = edges[0].function;
    }
    return fn != nullptr && dynamic_cast<AccumulateGrad*>(fn.get()) != nullptr;
}
} // namespace impl

void backward(const std::vector<Tensor>& tensors, const std::vector<Tensor>& gradients, bool retain_graph, bool create_graph,
              const std::vector<Tensor>& inputs) {
    if (!gradients.empty() && tensors.size() != gradients.size()) {
        TP_THROW(RuntimeError, "Mismatch in tensors and gradients size");
    }

    std::vector<Edge> roots;
    std::vector<Tensor> root_grads;
    roots.reserve(tensors.size());
    root_grads.reserve(tensors.size());

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& tensor = tensors[i];
        if (!tensor.requires_grad()) {
            TP_THROW(RuntimeError, "Tensor does not require grad and does not have a grad_fn");
        }

        // Prepare gradient
        Tensor grad;
        if (i < gradients.size() && gradients[i].defined()) {
            grad = gradients[i];
        } else {
            if (tensor.numel() != 1) {
                TP_THROW(RuntimeError, "grad can be implicitly created only for scalar outputs");
            }
            // Create scalar tensor on the same device and fill with 1.0.
            // The ones_like factory is used (rather than an empty tensor plus
            // an asynchronous fill_) so the value is visible to the engine's
            // kernels on the current stream as soon as the root is consumed.
            grad = Tensor::ones_like(tensor);
        }
        root_grads.push_back(grad);

        // Prepare root
        if (auto fn = impl::grad_fn(tensor)) {
            roots.emplace_back(fn, impl::output_nr(tensor));
        } else if (tensor.requires_grad()) {
            // Leaf node
            auto* meta = impl::get_autograd_meta(tensor);
            if (meta) {
                std::shared_ptr<Node> acc = meta->grad_accumulator();
                if (!acc) {
                    acc = std::make_shared<AccumulateGrad>(tensor);
                    meta->set_grad_accumulator(acc);
                }
                roots.emplace_back(std::move(acc), 0);
            }
        }
    }

    // Named inputs restrict the pass to the graph leading to them; the
    // engine runs their nodes, so a leaf's accumulator fills its .grad and a
    // non-leaf's node hands its gradient to the retained-gradient hook.
    std::vector<Edge> output_edges;
    output_edges.reserve(inputs.size());
    for (const auto& input : inputs) {
        if (!input.requires_grad()) {
            TP_THROW(RuntimeError, "One of the differentiated Tensors does not require grad");
        }
        if (auto fn = impl::grad_fn(input)) {
            impl::retain_grad(input);
            output_edges.emplace_back(fn, impl::output_nr(input));
        } else if (auto* meta = impl::get_autograd_meta(input)) {
            std::shared_ptr<Node> acc = meta->grad_accumulator();
            if (!acc) {
                acc = std::make_shared<AccumulateGrad>(input);
                meta->set_grad_accumulator(acc);
            }
            output_edges.emplace_back(std::move(acc), 0);
        } else {
            TP_THROW(RuntimeError, "Could not determine gradient edge for input");
        }
    }

    Engine::get_default_engine().execute(roots, root_grads, retain_graph, create_graph,
                                         /*accumulate_grad=*/true, output_edges);
}

std::vector<Tensor> grad(
    const std::vector<Tensor>& outputs,
    const std::vector<Tensor>& inputs,
    const std::vector<Tensor>& grad_outputs,
    bool retain_graph,
    bool create_graph,
    bool allow_unused) {

    if (outputs.empty()) {
        TP_THROW(RuntimeError, "grad requires at least one output tensor");
    }
    if (inputs.empty()) {
        TP_THROW(RuntimeError, "grad requires at least one input tensor");
    }

    // 1. Prepare roots
    std::vector<Edge> roots;
    roots.reserve(outputs.size());
    std::vector<Tensor> root_grads;
    root_grads.reserve(outputs.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& output = outputs[i];
        if (!output.requires_grad()) {
            TP_THROW(RuntimeError, "element " + std::to_string(i) + " of tensors does not require grad and does not have a grad_fn");
        }

        // Prepare grad
        Tensor gradient;
        if (i < grad_outputs.size() && grad_outputs[i].defined()) {
            gradient = grad_outputs[i];
        } else {
            if (output.numel() != 1) {
                TP_THROW(RuntimeError, "grad can be implicitly created only for scalar outputs");
            }
            std::vector<float> data = {1.0f};
            gradient = Tensor::tensor(data, output.dtype(), output.device()).reshape({});
        }
        root_grads.push_back(gradient);

        // Prepare edge
        if (auto fn = impl::grad_fn(output)) {
            roots.emplace_back(fn, impl::output_nr(output));
        } else {
            // Leaf
            auto* meta = impl::get_autograd_meta(output);
            if (meta) {
                std::shared_ptr<Node> acc = meta->grad_accumulator();
                if (!acc) {
                    acc = std::make_shared<AccumulateGrad>(output);
                    meta->set_grad_accumulator(acc);
                }
                roots.emplace_back(std::move(acc), 0);
            }
        }
    }

    // 2. Build the output edges (the tensors w.r.t. which we differentiate).
    // The engine captures the input gradients of these edges' functions.
    std::vector<Edge> output_edges;
    output_edges.reserve(inputs.size());

    for (const auto& input : inputs) {
        if (!input.requires_grad()) {
            TP_THROW(RuntimeError, "One of the differentiated Tensors does not require grad");
        }

        Edge edge;
        if (auto fn = impl::grad_fn(input)) {
            edge = Edge(fn, impl::output_nr(input));
        } else {
            // Leaf
            auto* meta = impl::get_autograd_meta(input);
            if (meta) {
                std::shared_ptr<Node> acc = meta->grad_accumulator();
                if (!acc) {
                    acc = std::make_shared<AccumulateGrad>(input);
                    meta->set_grad_accumulator(acc);
                }
                edge = Edge(std::move(acc), 0);
            }
        }

        if (!edge.is_valid()) {
            TP_THROW(RuntimeError, "Could not determine gradient edge for input");
        }
        output_edges.push_back(std::move(edge));
    }

    // 3. Execute
    auto captured = Engine::get_default_engine().execute(roots, root_grads, retain_graph,
                                                         create_graph, /*accumulate_grad=*/false,
                                                         output_edges);

    // 4. Collect results
    std::vector<Tensor> results;
    results.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        Tensor res;
        if (i < captured.size() && captured[i].defined()) {
            res = captured[i];
        } else {
            if (!allow_unused) {
                TP_THROW(RuntimeError, "One of the differentiated Tensors was not used in the graph");
            }
        }
        results.push_back(std::move(res));
    }

    return results;
}

void backward(const Tensor& tensor, const Tensor& gradient, bool retain_graph, bool create_graph) {
    std::vector<Tensor> tensors = {tensor};
    std::vector<Tensor> gradients;
    if (gradient.defined()) {
        gradients.push_back(gradient);
    }
    backward(tensors, gradients, retain_graph, create_graph);
}

Tensor as_strided(const Tensor& self, const std::vector<int64_t>& size,
                  const std::vector<int64_t>& stride,
                  std::optional<int64_t> storage_offset) {
    // The dispatched operator records the view and its backward.
    return ops::as_strided(self, size, stride, storage_offset);
}

Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
    // routing through the generated slice op carries the gradient via
    // SliceBackward.
    if (length < 0) {
        TP_THROW(RuntimeError, "narrow(): length cannot be negative but got ", length);
    }
    return tensorplay::tpx::ops::slice(self, dim, start, start + length, 1);
}

// expand() moved to the generated dispatcher surface;
// the derivative formulas in derivatives.yaml now resolve against
// tensorplay::tpx::ops::expand, which carries autograd routing.

// back to the source tensor's dtype and device.
struct ToCopyBackward : public Node {
    DType dtype_;
    Device device_;

    ToCopyBackward(DType dtype, Device device) : dtype_(dtype), device_(device) {}

    size_t num_inputs() const override { return 1; }

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) return {Tensor()};
        Tensor grad = inputs[0];
        // A real source receives the real part of a complex gradient.
        if (isComplexType(grad.dtype()) && !isComplexType(dtype_)) {
            grad = ops::real(grad);
        }
        // Through the recording conversion so a second derivative sees it.
        return {record_conversion(grad, grad.to(device_, dtype_))};
    }
};

// Records a conversion of ``self`` into ``result``.  Only a floating or
// complex result is differentiable; a conversion that changed nothing
// returned ``self`` itself and has nothing to record.  The gradient flows
// back converted to the source's dtype and device, and the tangent is
// converted forward like the value.
Tensor record_conversion(const Tensor& self, Tensor result) {
    if (!result.defined() ||
        result.unsafeGetTensorImpl() == self.unsafeGetTensorImpl() ||
        !isFloatingOrComplexType(result.dtype())) {
        return result;
    }
    const bool requires_grad =
        GradMode::is_enabled() && !InferenceMode::is_enabled() &&
        self.requires_grad() && !autograd_dispatch_excluded();
    if (requires_grad) {
        auto grad_fn = std::make_shared<ToCopyBackward>(self.dtype(), self.device());
        grad_fn->add_next_edge_list(collect_next_edges(self));
        impl::set_requires_grad(result, true);
        impl::set_grad_fn(result, grad_fn);
    }
    if (impl::is_fw_grad_defined(self, /* level */ 0)) {
        Tensor tangent = impl::to_non_opt_fw_grad(self);
        if (tangent.defined()) {
            impl::set_fw_grad(
                result,
                record_conversion(tangent, tangent.to(result.device(), result.dtype())),
                /* level */ 0, /* is_inplace_op */ false);
        }
    }
    return result;
}

namespace {

// The conversion records its own node, so the call it makes runs beneath
// autograd: nothing under it records history a second time, and a dispatch
// mode is handed the conversion itself rather than the copy it is made of.
//
// Not under a function transform: a batched value is unwrapped and the call
// made again one level down, where the conversion is that level's own and has
// to record history there.  Excluding autograd for the whole call would reach
// that inner call too, and the gradient a transform asks for would vanish.
struct BelowConversionAutograd {
    std::optional<tensorplay::impl::ExcludeDispatchKeyGuard> guard;
    BelowConversionAutograd() {
        if (!tensorplay::transform::are_transforms_active()) {
            guard.emplace(
                DispatchKeySet::make(DispatchKey::AutogradCPU) |
                DispatchKeySet::make(DispatchKey::AutogradCUDA) |
                DispatchKeySet::make(DispatchKey::AutogradVulkan) |
                DispatchKeySet::make(DispatchKey::AutogradSparseCPU) |
                DispatchKeySet::make(DispatchKey::AutogradSparseCUDA) |
                DispatchKeySet::make(DispatchKey::AutogradSparse));
        }
    }
};

} // namespace

Tensor to(const Tensor& self, DType dtype, bool non_blocking, bool copy) {
    Tensor result;
    {
        BelowConversionAutograd below;
        result = self.to(dtype, non_blocking, copy);
    }
    return record_conversion(self, std::move(result));
}

Tensor to(const Tensor& self, Device device, bool non_blocking, bool copy) {
    Tensor result;
    {
        BelowConversionAutograd below;
        result = self.to(device, non_blocking, copy);
    }
    return record_conversion(self, std::move(result));
}

Tensor to(const Tensor& self, Device device, DType dtype, bool non_blocking, bool copy) {
    Tensor result;
    {
        BelowConversionAutograd below;
        result = self.to(device, dtype, non_blocking, copy);
    }
    return record_conversion(self, std::move(result));
}

} // namespace tpx
} // namespace tensorplay
