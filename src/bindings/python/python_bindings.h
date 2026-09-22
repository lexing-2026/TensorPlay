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

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Autograd.h"
#include "Device.h"
#include "DType.h"
#include "Exception.h"
#include "Generator.h"
#include "SymBool.h"
#include "SymFloat.h"
#include "SymInt.h"

namespace py = pybind11;
using namespace py::literals;

using tensorplay::Device;
using tensorplay::DeviceType;
using tensorplay::DType;
using tensorplay::Size;
using tensorplay::Scalar;
using tensorplay::Generator;
using tensorplay::SymBool;
using tensorplay::SymFloat;
using tensorplay::SymInt;
using tensorplay::default_generator;
using tensorplay::manual_seed;
using Tensor = tensorplay::Tensor;

// using namespace tensorplay;

void init_tensor(py::module_& m);

// numpy-to-tensor load adapters and native codec entry points (io.cpp):
// fused layout conversion + normalization kernels backing
// tensorplay.vision / tensorplay.audio, plus the native image/audio codecs.
// Codec functions are declared unconditionally; the module init registers
// only the ones whose codec library was found at configure time, so the
// Python layer can probe the extension attributes to pick a fallback.
Tensor decode_jpeg(Tensor data, int64_t mode);
Tensor encode_jpeg(Tensor data, int64_t quality);
Tensor decode_png(Tensor data, int64_t mode);
Tensor encode_png(Tensor data, int64_t compression_level);
Tensor decode_jpeg_cuda(Tensor data, int64_t mode);
std::vector<Tensor> decode_jpeg_batch(std::vector<Tensor> data, int64_t mode,
                                      const std::string& device);
std::vector<Tensor> decode_png_batch(std::vector<Tensor> data, int64_t mode);
std::vector<Tensor> encode_jpeg_batch(std::vector<Tensor> data, int64_t quality);
std::vector<Tensor> encode_png_batch(std::vector<Tensor> data, int64_t compression_level);
std::pair<Tensor, int64_t> decode_wav(Tensor data, int64_t frame_offset, int64_t num_frames);
std::vector<std::pair<Tensor, int64_t>> decode_wav_batch(std::vector<Tensor> data,
                                                         int64_t frame_offset,
                                                         int64_t num_frames);
Tensor encode_wav(Tensor data, int64_t sample_rate, int64_t bits);
std::vector<Tensor> encode_wav_batch(std::vector<Tensor> data, int64_t sample_rate,
                                     int64_t bits);
std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> wav_info(Tensor data);
Tensor audio_to_tensor(py::object obj);
void init_device(py::module_& m);
void init_dtype(py::module_& m);
void init_size(py::module_& m);

// Materialize a tensorplay.Size (C tuple subclass) for Python.
PyObject* Size_New(const tensorplay::Size& size);
bool Size_Check(PyObject* obj);
void init_generator(py::module_& m);
void init_storage(py::module_& m);
void init_autograd(py::module_& m);
void init_autocast(py::module_& m);
void init_transforms(py::module_& m);
void init_ops(py::module_& m);
void init_dispatch(py::module_& m);
void init_python_dispatch(py::module_& m);
void init_scalar(py::module_& m);
void init_symint(py::module_& m);
void init_stax(py::module_& m);
void init_stax_static_launcher(py::module_& m);
void init_parallel(py::module_& m);
void init_env_allocator(py::module_& m);
#ifdef TP_USE_DISTRIBUTED
void init_distributed(py::module_& m);
#endif
void init_cuda_graph(py::module_& m);
void init_cuda_jiterator(py::module_& m);
void init_forward_ad(py::module_& m);
void init_futures(py::module_& m);
#ifdef TP_USE_RPC
void init_rpc(py::module_& m);
void init_distributed_autograd(py::module_& m);
#endif

namespace tensorplay {
class Exception;

// Single source of truth for C++ -> Python exception translation, shared by
// the pybind11 translator and the METH_FASTCALL bridge (defined in
// CPythonBridge.cpp): maps p10 exception kinds to their Python types (incl.
// the registered DeviceMismatchError subclass). Callers pass the result to
// PyErr_SetString together with the message.
PyObject* translate_exception(const Exception& e);

// Registers the Python type object for DeviceMismatchError (called once by
// init.cpp during module init; before that, translation falls back to plain
// RuntimeError).
void set_device_mismatch_error_type(PyObject* type);
} // namespace tensorplay
