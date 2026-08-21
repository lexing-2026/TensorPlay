#include "python_bindings.h"

void init_generator(py::module_& m) {
    py::class_<Generator>(m, "Generator")
        .def(py::init<uint64_t>(), "seed"_a = 0)
        .def("manual_seed", &Generator::manual_seed, "seed"_a)
        .def("seed", &Generator::seed)
        .def("initial_seed", &Generator::initial_seed)
        .def_property_readonly("device", [](const Generator&) { return Device(DeviceType::CPU); });

    m.def("default_generator", &default_generator, py::return_value_policy::reference);
    m.def("manual_seed", &manual_seed, "seed"_a);
    m.def("seed", &manual_seed, "seed"_a); // Alias
    m.def("initial_seed", []() { return default_generator().initial_seed(); });
}