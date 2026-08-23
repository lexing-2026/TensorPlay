#pragma once

#ifndef TP_STATIC_BUILD
#define TP_STATIC_BUILD
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/complex.h>

#include "Autograd.h"
#include "Device.h"
#include "DType.h"
#include "Exception.h"
#include "Generator.h"

namespace py = pybind11;
using namespace py::literals;

using tensorplay::Device;
using tensorplay::DeviceType;
using tensorplay::DType;
using tensorplay::Size;
using tensorplay::Scalar;
using tensorplay::Generator;
using tensorplay::default_generator;
using tensorplay::manual_seed;
using Tensor = tensorplay::Tensor;

// using namespace tensorplay;

void init_tensor(py::module_& m);
void init_device(py::module_& m);
void init_dtype(py::module_& m);
void init_size(py::module_& m);
void init_generator(py::module_& m);
void init_autograd(py::module_& m);
void init_autocast(py::module_& m);
void init_ops(py::module_& m);
void init_scalar(py::module_& m);
void init_stax(py::module_& m);
void init_parallel(py::module_& m);
void init_distributed(py::module_& m);
void init_cuda_graph(py::module_& m);
