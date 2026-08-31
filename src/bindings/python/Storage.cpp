#include "python_bindings.h"

#include "Allocator.h"
#include "Storage.h"

#include <cstdint>

namespace {

tensorplay::Storage make_storage(int64_t nbytes,
                                 const std::optional<Device>& device) {
    if (nbytes < 0) {
        TP_THROW(ValueError, "storage size must be non-negative");
    }
    const Device target = device.value_or(Device(DeviceType::CPU));
    return tensorplay::Storage(
        static_cast<size_t>(nbytes),
        tensorplay::getAllocator(target.type()),
        target);
}

uintptr_t storage_address(const tensorplay::Storage& storage) {
    return reinterpret_cast<uintptr_t>(storage.unsafeGetStorageImpl().get());
}

} // namespace

void init_storage(py::module_& m) {
    m.def("is_storage", [](py::handle obj) {
        return py::isinstance<tensorplay::Storage>(obj);
    }, "obj"_a);

    auto storage = py::class_<tensorplay::Storage>(m, "UntypedStorage");
    storage.attr("__module__") = "tensorplay._C";
    storage
        .def(py::init<>())
        .def(py::init([](int64_t nbytes, std::optional<Device> device) {
            return make_storage(nbytes, device);
        }), "nbytes"_a = 0, "device"_a = py::none())
        .def("__len__", [](const tensorplay::Storage& self) {
            return self.nbytes();
        })
        .def("size", [](const tensorplay::Storage& self) {
            return self.nbytes();
        })
        .def("nbytes", &tensorplay::Storage::nbytes)
        .def_property_readonly("device", &tensorplay::Storage::device)
        .def_property_readonly("is_cuda", [](const tensorplay::Storage& self) {
            return self.device().is_cuda();
        })
        .def("resizable", &tensorplay::Storage::resizable)
        .def_property_readonly("_cdata", &storage_address)
        .def("data_ptr", [](const tensorplay::Storage& self) {
            return reinterpret_cast<uintptr_t>(self.data());
        })
        .def("resize_", [](tensorplay::Storage& self, int64_t nbytes)
             -> tensorplay::Storage& {
            if (nbytes < 0) {
                TP_THROW(ValueError, "storage size must be non-negative");
            }
            self.set_nbytes(static_cast<size_t>(nbytes));
            return self;
        }, "nbytes"_a)
        .def("__bool__", &tensorplay::Storage::defined)
        .def("__repr__", [](const tensorplay::Storage& self) {
            return "<tensorplay.UntypedStorage device=" +
                   self.device().toString() + " nbytes=" +
                   std::to_string(self.nbytes()) + ">";
        });
}
