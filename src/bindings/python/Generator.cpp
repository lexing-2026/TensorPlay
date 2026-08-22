#include "python_bindings.h"
#include "Generator.h"

void init_generator(py::module_& m) {
    py::class_<Generator>(m, "Generator")
        .def(py::init<uint64_t>(), "seed"_a = tensorplay::default_rng_seed_val)
        .def("manual_seed", &Generator::manual_seed, "seed"_a)
        .def("seed", [](Generator& g) { return g.seed(); })
        .def("initial_seed", &Generator::initial_seed)
        .def("get_state", &Generator::get_state)
        .def("set_state", &Generator::set_state, "new_state"_a)
        .def_property_readonly("device", [](const Generator&) { return Device(DeviceType::CPU); });

    m.def("default_generator", &default_generator, py::return_value_policy::reference);
    m.def("manual_seed", &manual_seed, "seed"_a);
    // torch.seed(): reseed the default generator nondeterministically and
    // return the new seed.
    m.def("seed", []() { return default_generator().seed(); });
    m.def("initial_seed", []() { return default_generator().initial_seed(); });
    m.def("get_rng_state", []() { return default_generator().get_state(); },
          "get_rng_state() -> Tensor: returns the default generator state as a UInt8 Byte tensor.");
    m.def("set_rng_state", [](const Tensor& new_state) { default_generator().set_state(new_state); },
          "new_state"_a);
}
