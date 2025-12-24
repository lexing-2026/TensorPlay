#pragma once

#ifndef TP_STATIC_BUILD
#define TP_STATIC_BUILD
#endif

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/operators.h>
#include <nanobind/ndarray.h>
#include <nanobind/make_iterator.h>

#include "TPXTensor.h"
#include "Device.h"
#include "DType.h"
#include "Exception.h"
#include "Generator.h"
#include "Autograd.h"

namespace nb = nanobind;
using namespace nb::literals;

using tensorplay::Device;
using tensorplay::DeviceType;
using tensorplay::DType;
using tensorplay::Size;
using tensorplay::Scalar;
using tensorplay::Generator;
using tensorplay::default_generator;
using tensorplay::manual_seed;
using Tensor = tensorplay::tpx::Tensor;

// using namespace tensorplay;

void init_tensor(nb::module_& m);
void init_device(nb::module_& m);
void init_dtype(nb::module_& m);
void init_size(nb::module_& m);
void init_generator(nb::module_& m);
void init_autograd(nb::module_& m);
void init_ops(nb::module_& m);
void init_scalar(nb::module_& m);
void init_stax(nb::module_& m);
