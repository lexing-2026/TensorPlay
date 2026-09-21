#include "python_bindings.h"

#include "ForwardGrad.h"

// Forward-mode AD entry points: level management, the tangent-read switch,
// and direct tangent accessors used by the Python jvp transformation and the
// custom-Function forward hook wiring.

void init_forward_ad(py::module_& m) {
    using tensorplay::Tensor;
    using tensorplay::tpx::FwGradMode;
    using tensorplay::tpx::ForwardADLevel;
    namespace timpl = tensorplay::tpx::impl;

    // Enters a new forward AD level and returns its handle.  Levels cannot
    // nest: a second entry while one is active raises.
    m.def("_enter_dual_level", []() -> int64_t {
        return static_cast<int64_t>(ForwardADLevel::get_next_idx());
    });

    // Exits the given level, erasing every tangent registered with it.  The
    // level must be the most recently created one.
    m.def("_exit_dual_level", [](int64_t level) {
        ForwardADLevel::release_idx(static_cast<uint64_t>(level));
    }, "level"_a);

    m.def("_get_fwd_grad_enabled", []() -> bool {
        return FwGradMode::is_enabled();
    });

    m.def("_set_fwd_grad_enabled", [](bool enabled) {
        FwGradMode::set_enabled(enabled);
    }, "enabled"_a);

    // Reads the tangent stored on a tensor at the given level; the result is
    // undefined when the tensor carries no tangent there.
    m.def("_fw_grad", [](const Tensor& self, int64_t level) -> Tensor {
        return timpl::fw_grad(self, static_cast<uint64_t>(level));
    }, "self"_a, "level"_a);

    // Attaches a tangent to a tensor at the given level.  Re-attaching is
    // only allowed for in-place ops that set the very same tangent object.
    m.def("_set_fw_grad",
          [](const Tensor& self, const Tensor& new_grad, int64_t level,
             bool is_inplace_op) {
              timpl::set_fw_grad(self, new_grad,
                                 static_cast<uint64_t>(level),
                                 is_inplace_op);
          },
          "self"_a, "new_grad"_a, "level"_a, "is_inplace_op"_a);
}
