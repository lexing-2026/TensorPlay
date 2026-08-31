#include "python_bindings.h"

#include "TransformDispatch.h"

namespace {

tensorplay::transform::Randomness parse_randomness(const std::string& value) {
    if (value == "error") return tensorplay::transform::Randomness::Error;
    if (value == "same") return tensorplay::transform::Randomness::Same;
    if (value == "different") return tensorplay::transform::Randomness::Different;
    TP_THROW(ValueError,
             "randomness must be 'error', 'same', or 'different'");
}

} // namespace

void init_transforms(py::module_& m) {
    m.def("_transform_push_vmap", [](int64_t batch_size,
                                      const std::string& randomness) {
        return tensorplay::transform::push_vmap(
            batch_size, parse_randomness(randomness));
    }, "batch_size"_a, "randomness"_a = "error");

    m.def("_transform_pop", []() {
        const auto layer = tensorplay::transform::pop_layer();
        return py::make_tuple(static_cast<int>(layer.kind), layer.level,
                              layer.batch_size);
    });

    m.def("_transform_current", []() -> py::object {
        const auto layer = tensorplay::transform::current_layer();
        if (!layer.has_value()) return py::none();
        return py::make_tuple(static_cast<int>(layer->kind), layer->level,
                              layer->batch_size,
                              static_cast<int>(layer->randomness));
    });

    m.def("_transform_make_batched",
          &tensorplay::transform::make_batched,
          "value"_a, "dim"_a, "level"_a);

    m.def("_transform_unwrap", [](const Tensor& value, int64_t level) {
        auto result = tensorplay::transform::unwrap_at_level(value, level);
        py::object dim = std::get<1>(result).has_value()
            ? py::cast(*std::get<1>(result))
            : py::none();
        return py::make_tuple(std::get<0>(result), std::move(dim));
    }, "value"_a, "level"_a);

    m.def("_transform_unwrap_all", &tensorplay::transform::unwrap_all,
          "value"_a);
    m.def("_transform_is_batched", &tensorplay::Tensor::is_batched,
          "value"_a);
    m.def("_transform_batch_dim", &tensorplay::Tensor::batch_dim,
          "value"_a);
    m.def("_transform_batch_level", &tensorplay::Tensor::batch_level,
          "value"_a);
}
