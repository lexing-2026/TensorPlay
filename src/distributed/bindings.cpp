// Python bindings for the CPU process-group backends (gloo transport and
// MPI), exposed next to the NCCL communicator surface.

#include <pybind11/chrono.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "python_bindings.h"

#include "gloo/ProcessGroupGloo.h"
#include "mpi/ProcessGroupMPI.h"
#include "store/FileStore.h"
#include "store/HashStore.h"
#include "store/PrefixStore.h"
#include "store/TCPStore.h"

namespace py = pybind11;
using namespace tensorplay;
using tensorplay::distributed::FileStore;
using tensorplay::distributed::GlooOptions;
using tensorplay::distributed::GlooWork;
using tensorplay::distributed::HashStore;
using tensorplay::distributed::PrefixStore;
using tensorplay::distributed::ProcessGroupGloo;
using tensorplay::distributed::ProcessGroupMPI;
using tensorplay::distributed::Store;
using tensorplay::distributed::TCPStore;
using tensorplay::distributed::Tensor;

namespace {

std::chrono::milliseconds toTimeout(int64_t timeout_ms) {
  return timeout_ms < 0 ? std::chrono::milliseconds(-1)
                        : std::chrono::milliseconds(timeout_ms);
}

std::vector<uint8_t> toBytes(const py::bytes& value) {
  const char* data = nullptr;
  Py_ssize_t size = 0;
  if (PyBytes_AsStringAndSize(value.ptr(), const_cast<char**>(&data), &size) != 0) {
    TP_THROW(RuntimeError, "store: expected a bytes value");
  }
  return std::vector<uint8_t>(data, data + size);
}

py::bytes fromBytes(const std::vector<uint8_t>& value) {
  return py::bytes(reinterpret_cast<const char*>(value.data()),
                   static_cast<Py_ssize_t>(value.size()));
}

// Python-facing `get` blocks for the key the way the pure-Python store did:
// poll until present or the timeout expires, then either return the value
// or surface the failure. The polling loop runs without the GIL so other
// Python threads (e.g. the one publishing the key) can make progress.
std::vector<uint8_t> waitOrThrow(
    Store& store,
    const std::string& key,
    int64_t timeout_ms) {
  const auto timeout = toTimeout(timeout_ms);
  if (timeout < std::chrono::milliseconds(0)) {
    return store.get(key);
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  {
    py::gil_scoped_release release;
    for (;;) {
      if (store.check({key})) {
        return store.get(key);
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  TP_THROW(RuntimeError, "store: timed out waiting for key ", key);
}

// Sleep granularity for the polling loops exposed through Python, where a
// timed-out wait reports failure instead of throwing.
constexpr auto kStorePollInterval = std::chrono::milliseconds(10);

void bindWork(py::module_& dist) {
  py::class_<GlooWork, std::shared_ptr<GlooWork>>(dist, "GlooWork")
      .def("wait",
           [](GlooWork& work, int64_t timeout_ms) {
             return work.wait(timeout_ms);
           },
           py::arg("timeout_ms") = -1)
      .def("is_completed", &GlooWork::is_completed)
      .def("source_rank", &GlooWork::source_rank)
      .def("op_name", [](const GlooWork& work) { return work.op_name(); })
      .def("result", [](GlooWork& work) { return work.result(); });
}

// Binds the key/value stores. Values cross the boundary as `bytes`; the
// Python layer keeps its seconds-based timeout convention, so the store
// methods take milliseconds and the Python shims convert. Operations are
// declared once on the `Store` base; subclasses add constructors and any
// store-specific surface.
void bindStores(py::module_& dist) {
  py::class_<Store, std::shared_ptr<Store>>(dist, "Store")
      .def("set",
           [](Store& store, const std::string& key, const py::bytes& v) {
             store.set(key, toBytes(v));
           },
           py::arg("key"), py::arg("value"))
      .def("get",
           [](Store& store,
              const std::string& key,
              int64_t timeout_ms) -> py::bytes {
             return fromBytes(waitOrThrow(store, key, timeout_ms));
           },
           py::arg("key"), py::arg("timeout_ms") = -1)
      .def("add",
           [](Store& store, const std::string& key, int64_t amount) {
             return store.add(key, amount);
           })
      .def("compare_and_swap",
           [](Store& store,
              const std::string& key,
              const py::bytes& expected,
              const py::bytes& value) {
             auto current =
                 store.compareSet(key, toBytes(expected), toBytes(value));
             return py::make_tuple(
                 current == toBytes(expected), fromBytes(current));
           })
      .def("compare_set",
           [](Store& store,
              const std::string& key,
              const py::bytes& expected,
              const py::bytes& value) {
             auto current =
                 store.compareSet(key, toBytes(expected), toBytes(value));
             return current == toBytes(expected) ? value : fromBytes(current);
           })
      .def("has", [](Store& store, const std::string& key) {
        return store.check({key});
      })
      .def("delete_key", [](Store& store, const std::string& key) {
        return store.deleteKey(key);
      })
      .def("wait",
           [](Store& store,
              const std::vector<std::string>& keys,
              int64_t timeout_ms) {
             bool done = false;
             {
               py::gil_scoped_release release;
               done = timeout_ms < 0
                   ? store.wait(keys)
                   : store.wait(keys, std::chrono::milliseconds(timeout_ms));
             }
             return done;
           },
           py::arg("keys"), py::arg("timeout_ms") = -1)
      .def("num_keys", &Store::getNumKeys);

  py::class_<FileStore, std::shared_ptr<FileStore>, Store>(
      dist, "FileStore")
      .def(
          py::init([](const std::string& path) {
            return std::make_shared<FileStore>(path);
          }),
          py::arg("file_name"));

  py::class_<HashStore, std::shared_ptr<HashStore>, Store>(dist, "HashStore")
      .def(py::init<>());

  py::class_<TCPStore, std::shared_ptr<TCPStore>, Store>(dist, "TCPStore")
      .def(
          py::init([](const std::string& host,
                      int port,
                      bool is_master,
                      int64_t timeout_ms) {
            return std::make_shared<TCPStore>(
                host,
                static_cast<uint16_t>(port),
                is_master,
                timeout_ms < 0 ? Store::kDefaultTimeout
                               : std::chrono::milliseconds(timeout_ms));
          }),
          py::arg("host_name"),
          py::arg("port") = 0,
          py::arg("is_master") = false,
          py::arg("timeout_ms") = -1)
      .def("host", &TCPStore::host)
      .def("port", &TCPStore::port)
      .def("stop", &TCPStore::stop);

  py::class_<PrefixStore, std::shared_ptr<PrefixStore>, Store>(
      dist, "PrefixStore")
      .def(
          py::init([](const std::string& prefix, std::shared_ptr<Store> s) {
            return std::make_shared<PrefixStore>(prefix, std::move(s));
          }),
          py::arg("prefix"), py::arg("store"));
}

void bindGloo(py::module_& dist) {
  py::class_<GlooOptions>(
      dist, "GlooOptions")
      .def(py::init<>())
      .def_readwrite("timeout_ms", &GlooOptions::timeout)
      .def_readwrite("threads", &GlooOptions::threads)
      .def_readwrite("group_name", &GlooOptions::group_name)
      .def(
          "add_device",
          [](GlooOptions& options,
             std::shared_ptr<::gloo::transport::Device> device) {
            options.devices.push_back(std::move(device));
          });

  py::class_<
      ::gloo::transport::Device,
      std::shared_ptr<::gloo::transport::Device>>(dist, "GlooDevice");

  py::class_<ProcessGroupGloo, std::shared_ptr<ProcessGroupGloo>>(
      dist, "ProcessGroupGloo")
      .def_static(
          "create_device",
          [](const std::string& hostname, const std::string& interface) {
            if (!hostname.empty()) {
              return ProcessGroupGloo::createDeviceForHostname(hostname);
            }
            if (!interface.empty()) {
              return ProcessGroupGloo::createDeviceForInterface(interface);
            }
            TP_THROW(
                ValueError, "Specify either `hostname` or `interface`.");
          },
          py::arg("hostname") = "", py::arg("interface") = "")
      .def_static(
          "create_default_device",
          [](bool lazy_init) {
            return ProcessGroupGloo::createDefaultDevice(lazy_init);
          },
          py::arg("lazy_init") = false)
      .def(
          py::init([](std::shared_ptr<Store> store,
                      int rank,
                      int size,
                      GlooOptions options) {
            return std::make_shared<ProcessGroupGloo>(
                std::move(store), rank, size, std::move(options));
          }),
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          py::arg("options"))
      .def("rank", &ProcessGroupGloo::getRank)
      .def("size", &ProcessGroupGloo::getSize)
      .def("group_name", [](const ProcessGroupGloo& pg) {
        return pg.groupName();
      })
      .def(
          "set_timeout",
          [](ProcessGroupGloo& pg, int64_t timeout_ms) {
            pg.setTimeout(std::chrono::milliseconds(timeout_ms));
          })
      .def(
          "sequence_number",
          [](ProcessGroupGloo& pg) { return pg.getSequenceNumberForGroup(); })
      .def(
          "broadcast",
          [](ProcessGroupGloo& pg,
             std::vector<Tensor>& tensors,
             int root_rank,
             int root_tensor,
             int64_t timeout_ms) {
            return pg.broadcast(
                tensors, root_rank, root_tensor, toTimeout(timeout_ms));
          },
          py::arg("tensors"),
          py::arg("root_rank"),
          py::arg("root_tensor") = 0,
          py::arg("timeout_ms") = -1)
      .def(
          "allreduce",
          [](ProcessGroupGloo& pg,
             std::vector<Tensor>& tensors,
             int op,
             int64_t timeout_ms) {
            return pg.allreduce(tensors, op, toTimeout(timeout_ms));
          },
          py::arg("tensors"), py::arg("op"), py::arg("timeout_ms") = -1)
      .def(
          "allreduce_coalesced",
          [](ProcessGroupGloo& pg,
             std::vector<Tensor>& tensors,
             int op,
             int64_t timeout_ms) {
            return pg.allreduce_coalesced(tensors, op, toTimeout(timeout_ms));
          },
          py::arg("tensors"), py::arg("op"), py::arg("timeout_ms") = -1)
      .def(
          "reduce",
          [](ProcessGroupGloo& pg,
             std::vector<Tensor>& tensors,
             int root_rank,
             int root_tensor,
             int op,
             int64_t timeout_ms) {
            return pg.reduce(
                tensors, root_rank, root_tensor, op, toTimeout(timeout_ms));
          },
          py::arg("tensors"),
          py::arg("root_rank"),
          py::arg("root_tensor") = 0,
          py::arg("op"),
          py::arg("timeout_ms") = -1)
      .def(
          "allgather",
          [](ProcessGroupGloo& pg,
             std::vector<std::vector<Tensor>>& outputs,
             std::vector<Tensor>& inputs,
             int64_t timeout_ms) {
            return pg.allgather(outputs, inputs, toTimeout(timeout_ms));
          },
          py::arg("output_tensor_lists"),
          py::arg("input_tensor_list"),
          py::arg("timeout_ms") = -1)
      .def(
          "all_gather_into_tensor",
          [](ProcessGroupGloo& pg,
             Tensor& output,
             Tensor& input,
             int64_t timeout_ms) {
            return pg.all_gather_into_tensor(output, input, toTimeout(timeout_ms));
          },
          py::arg("output"), py::arg("input"), py::arg("timeout_ms") = -1)
      .def(
          "gather",
          [](ProcessGroupGloo& pg,
             std::vector<std::vector<Tensor>>& outputs,
             std::vector<Tensor>& inputs,
             int root_rank,
             int64_t timeout_ms) {
            return pg.gather(outputs, inputs, root_rank, toTimeout(timeout_ms));
          },
          py::arg("output_tensor_lists"),
          py::arg("input_tensor_list"),
          py::arg("root_rank"),
          py::arg("timeout_ms") = -1)
      .def(
          "scatter",
          [](ProcessGroupGloo& pg,
             std::vector<Tensor>& outputs,
             std::vector<std::vector<Tensor>>& inputs,
             int root_rank,
             int64_t timeout_ms) {
            return pg.scatter(outputs, inputs, root_rank, toTimeout(timeout_ms));
          },
          py::arg("output_tensor_list"),
          py::arg("input_tensor_lists"),
          py::arg("root_rank"),
          py::arg("timeout_ms") = -1)
      .def(
          "reduce_scatter",
          [](ProcessGroupGloo& pg,
             std::vector<Tensor>& outputs,
             std::vector<std::vector<Tensor>>& inputs,
             int op,
             int64_t timeout_ms) {
            return pg.reduce_scatter(outputs, inputs, op, toTimeout(timeout_ms));
          },
          py::arg("output_tensor_list"),
          py::arg("input_tensor_lists"),
          py::arg("op"),
          py::arg("timeout_ms") = -1)
      .def(
          "reduce_scatter_tensor",
          [](ProcessGroupGloo& pg,
             Tensor& output,
             Tensor& input,
             int op,
             int64_t timeout_ms) {
            return pg.reduce_scatter_tensor(output, input, op, toTimeout(timeout_ms));
          },
          py::arg("output"), py::arg("input"), py::arg("op"),
          py::arg("timeout_ms") = -1)
      .def(
          "all_to_all_single",
          [](ProcessGroupGloo& pg,
             Tensor& output,
             Tensor& input,
             std::vector<int64_t> output_counts,
             std::vector<int64_t> input_counts,
             int64_t timeout_ms) {
            return pg.all_to_all_single(
                output,
                input,
                std::move(output_counts),
                std::move(input_counts),
                toTimeout(timeout_ms));
          },
          py::arg("output"),
          py::arg("input"),
          py::arg("output_counts") = std::vector<int64_t>{},
          py::arg("input_counts") = std::vector<int64_t>{},
          py::arg("timeout_ms") = -1)
      .def(
          "alltoall",
          [](ProcessGroupGloo& pg,
             std::vector<Tensor>& outputs,
             std::vector<Tensor>& inputs,
             int64_t timeout_ms) {
            return pg.alltoall(outputs, inputs, toTimeout(timeout_ms));
          },
          py::arg("output_tensor_list"),
          py::arg("input_tensor_list"),
          py::arg("timeout_ms") = -1)
      .def(
          "send",
          [](ProcessGroupGloo& pg, std::vector<Tensor>& tensors, int dst, int tag) {
            return pg.send(tensors, dst, tag);
          },
          py::arg("tensors"), py::arg("dst"), py::arg("tag"))
      .def(
          "recv",
          [](ProcessGroupGloo& pg, std::vector<Tensor>& tensors, int src, int tag) {
            return pg.recv(tensors, src, tag);
          },
          py::arg("tensors"), py::arg("src"), py::arg("tag"))
      .def(
          "recv_anysource",
          [](ProcessGroupGloo& pg, std::vector<Tensor>& tensors, int tag) {
            return pg.recvAnysource(tensors, tag);
          },
          py::arg("tensors"), py::arg("tag"))
      .def(
          "barrier",
          [](ProcessGroupGloo& pg, int64_t timeout_ms) {
            return pg.barrier(toTimeout(timeout_ms));
          },
          py::arg("timeout_ms") = -1)
      .def(
          "monitored_barrier",
          [](ProcessGroupGloo& pg, int64_t timeout_ms, bool wait_all_ranks) {
            pg.monitoredBarrier(toTimeout(timeout_ms), wait_all_ranks);
          },
          py::arg("timeout_ms") = -1,
          py::arg("wait_all_ranks") = false);
}

void bindMPI(py::module_& dist) {
#ifdef USE_P10D_MPI
  py::class_<ProcessGroupMPI, std::shared_ptr<ProcessGroupMPI>>(
      dist, "ProcessGroupMPI")
      .def_static(
          "create",
          [](std::vector<int> ranks) {
            return ProcessGroupMPI::createProcessGroupMPI(std::move(ranks));
          },
          py::arg("ranks") = std::vector<int>{})
      .def("rank", &ProcessGroupMPI::getRank)
      .def("size", &ProcessGroupMPI::getSize)
      .def(
          "broadcast",
          [](ProcessGroupMPI& pg,
             std::vector<Tensor>& tensors,
             int root_rank,
             int64_t timeout_ms) {
            return pg.broadcast(tensors, root_rank, toTimeout(timeout_ms));
          },
          py::arg("tensors"), py::arg("root_rank"), py::arg("timeout_ms") = -1)
      .def(
          "allreduce",
          [](ProcessGroupMPI& pg,
             std::vector<Tensor>& tensors,
             int op,
             int64_t timeout_ms) {
            return pg.allreduce(tensors, op, toTimeout(timeout_ms));
          },
          py::arg("tensors"), py::arg("op"), py::arg("timeout_ms") = -1)
      .def(
          "allreduce_coalesced",
          [](ProcessGroupMPI& pg,
             std::vector<Tensor>& tensors,
             int op,
             int64_t timeout_ms) {
            return pg.allreduce_coalesced(tensors, op, toTimeout(timeout_ms));
          },
          py::arg("tensors"), py::arg("op"), py::arg("timeout_ms") = -1)
      .def(
          "reduce",
          [](ProcessGroupMPI& pg,
             std::vector<Tensor>& tensors,
             int root_rank,
             int op,
             int64_t timeout_ms) {
            return pg.reduce(tensors, root_rank, op, toTimeout(timeout_ms));
          },
          py::arg("tensors"), py::arg("root_rank"), py::arg("op"),
          py::arg("timeout_ms") = -1)
      .def(
          "allgather",
          [](ProcessGroupMPI& pg,
             std::vector<std::vector<Tensor>>& outputs,
             std::vector<Tensor>& inputs,
             int64_t timeout_ms) {
            return pg.allgather(outputs, inputs, toTimeout(timeout_ms));
          },
          py::arg("output_tensor_lists"),
          py::arg("input_tensor_list"),
          py::arg("timeout_ms") = -1)
      .def(
          "all_gather_into_tensor",
          [](ProcessGroupMPI& pg,
             Tensor& output,
             Tensor& input,
             int64_t timeout_ms) {
            return pg.all_gather_into_tensor(output, input, toTimeout(timeout_ms));
          },
          py::arg("output"), py::arg("input"), py::arg("timeout_ms") = -1)
      .def(
          "gather",
          [](ProcessGroupMPI& pg,
             std::vector<std::vector<Tensor>>& outputs,
             std::vector<Tensor>& inputs,
             int root_rank,
             int64_t timeout_ms) {
            return pg.gather(outputs, inputs, root_rank, toTimeout(timeout_ms));
          },
          py::arg("output_tensor_lists"),
          py::arg("input_tensor_list"),
          py::arg("root_rank"),
          py::arg("timeout_ms") = -1)
      .def(
          "scatter",
          [](ProcessGroupMPI& pg,
             std::vector<Tensor>& outputs,
             std::vector<std::vector<Tensor>>& inputs,
             int root_rank,
             int64_t timeout_ms) {
            return pg.scatter(outputs, inputs, root_rank, toTimeout(timeout_ms));
          },
          py::arg("output_tensor_list"),
          py::arg("input_tensor_lists"),
          py::arg("root_rank"),
          py::arg("timeout_ms") = -1)
      .def(
          "reduce_scatter",
          [](ProcessGroupMPI& pg,
             std::vector<Tensor>& outputs,
             std::vector<std::vector<Tensor>>& inputs,
             int op,
             int64_t timeout_ms) {
            return pg.reduce_scatter(outputs, inputs, op, toTimeout(timeout_ms));
          },
          py::arg("output_tensor_list"),
          py::arg("input_tensor_lists"),
          py::arg("op"),
          py::arg("timeout_ms") = -1)
      .def(
          "reduce_scatter_tensor",
          [](ProcessGroupMPI& pg,
             Tensor& output,
             Tensor& input,
             int op,
             int64_t timeout_ms) {
            return pg.reduce_scatter_tensor(output, input, op, toTimeout(timeout_ms));
          },
          py::arg("output"), py::arg("input"), py::arg("op"),
          py::arg("timeout_ms") = -1)
      .def(
          "all_to_all_single",
          [](ProcessGroupMPI& pg,
             Tensor& output,
             Tensor& input,
             std::vector<int64_t> output_counts,
             std::vector<int64_t> input_counts,
             int64_t timeout_ms) {
            return pg.all_to_all_single(
                output,
                input,
                std::move(output_counts),
                std::move(input_counts),
                toTimeout(timeout_ms));
          },
          py::arg("output"),
          py::arg("input"),
          py::arg("output_counts") = std::vector<int64_t>{},
          py::arg("input_counts") = std::vector<int64_t>{},
          py::arg("timeout_ms") = -1)
      .def(
          "alltoall",
          [](ProcessGroupMPI& pg,
             std::vector<Tensor>& outputs,
             std::vector<Tensor>& inputs,
             int64_t timeout_ms) {
            return pg.alltoall(outputs, inputs, toTimeout(timeout_ms));
          },
          py::arg("output_tensor_list"),
          py::arg("input_tensor_list"),
          py::arg("timeout_ms") = -1)
      .def(
          "send",
          [](ProcessGroupMPI& pg, std::vector<Tensor>& tensors, int dst, int tag) {
            return pg.send(tensors, dst, tag);
          },
          py::arg("tensors"), py::arg("dst"), py::arg("tag"))
      .def(
          "recv",
          [](ProcessGroupMPI& pg, std::vector<Tensor>& tensors, int src, int tag) {
            return pg.recv(tensors, src, tag);
          },
          py::arg("tensors"), py::arg("src"), py::arg("tag"))
      .def(
          "recv_anysource",
          [](ProcessGroupMPI& pg, std::vector<Tensor>& tensors, int tag) {
            return pg.recvAnysource(tensors, tag);
          },
          py::arg("tensors"), py::arg("tag"))
      .def(
          "barrier",
          [](ProcessGroupMPI& pg, int64_t timeout_ms) {
            return pg.barrier(toTimeout(timeout_ms));
          },
          py::arg("timeout_ms") = -1);
#else
  py::class_<ProcessGroupMPI, std::shared_ptr<ProcessGroupMPI>>(
      dist, "ProcessGroupMPI")
      .def_static(
          "create",
          [](std::vector<int> ranks) {
            (void)ranks;
            TP_THROW(
                RuntimeError,
                "Distributed package doesn't have MPI built in. MPI is only "
                "included if the package is built on a host that has MPI "
                "installed.");
          },
          py::arg("ranks") = std::vector<int>{});
#endif
}

} // namespace

namespace tensorplay {
namespace distributed {

void init_gloo_bindings(py::module_& dist);
void init_mpi_bindings(py::module_& dist);

void init_gloo_bindings(py::module_& dist) {
  bindStores(dist);
  bindWork(dist);
  bindGloo(dist);
}

void init_mpi_bindings(py::module_& dist) {
  bindMPI(dist);
}

} // namespace distributed
} // namespace tensorplay
