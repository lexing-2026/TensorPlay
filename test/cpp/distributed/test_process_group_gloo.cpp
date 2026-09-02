// Multi-rank verification for the gloo process group.
//
// Pattern: the suite forks one worker per rank, each worker embeds a Python
// interpreter, rendezvouses through the Python FileStore (the same store the
// Python layer uses), and exercises collectives on real tensors. A worker
// exits non-zero when any expectation fails; the parent joins all workers
// and asserts every exit status is zero.
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>

#include <chrono>
#include <complex>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include "gloo/ProcessGroupGloo.h"

namespace py = pybind11;
using namespace tensorplay;
using tensorplay::distributed::GlooOptions;
using tensorplay::distributed::ProcessGroupGloo;
using tensorplay::distributed::ReduceOp;
using T = tensorplay::Tensor;

namespace {

constexpr auto kTimeout = std::chrono::milliseconds(30000);

// Forks `size` workers and runs `body(rank, size)` in each. The parent waits
// for every child and fails the assertion if any worker exited abnormally.
template <typename Body>
void forkAndRun(int size, const std::string& storePath, Body&& body) {
  std::vector<pid_t> pids;
  for (int rank = 0; rank < size; ++rank) {
    pid_t pid = fork();
    ASSERT_GE(pid, 0) << "fork failed";
    if (pid == 0) {
      // The child reports failure through its exit status: gtest assertions
      // would only abort this copy of the process anyway.
      bool ok = body(rank, size, storePath);
      _Exit(ok ? 0 : 1);
    }
    pids.push_back(pid);
  }
  for (pid_t pid : pids) {
    int status = -1;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status)) << "worker killed by signal";
    ASSERT_EQ(WEXITSTATUS(status), 0) << "worker exited with failure";
  }
}

// A store file unique to this run: a file left over from an earlier aborted
// run would answer rendezvous lookups with dead peer addresses.
std::string freshStorePath(const std::string& base) {
  std::string path = ::testing::TempDir() + base + "_XXXXXX";
  std::vector<char> buf(path.begin(), path.end());
  buf.push_back('\0');
  int fd = ::mkstemp(buf.data());
  EXPECT_GE(fd, 0) << "mkstemp failed";
  if (fd >= 0) {
    ::close(fd);
  }
  return std::string(buf.data());
}

// Worker-side setup: interpreter, store, and one process group.
class WorkerEnv {
 public:
  WorkerEnv(const std::string& storePath, int rank, int size)
      : interpreter_() {
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, TENSORPLAY_ROOT);
    py::module_ dist = py::module_::import("tensorplay.distributed");
    py::object pyStore = dist.attr("FileStore")(storePath, size);
    auto store = pyStore.cast<std::shared_ptr<tensorplay::distributed::Store>>();

    GlooOptions options;
    options.threads = 2;
    options.group_name = "test";
    options.devices.push_back(
        ProcessGroupGloo::createDeviceForHostname("127.0.0.1"));
    pg_ = std::make_shared<ProcessGroupGloo>(
        std::move(store), rank, size, std::move(options));
  }

  ProcessGroupGloo& pg() {
    return *pg_;
  }

 private:
  py::scoped_interpreter interpreter_;
  std::shared_ptr<ProcessGroupGloo> pg_;
};

// Runs `step` on each forked rank with a fresh worker environment.
template <typename Step>
void runCollective(Step&& step) {
  const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
  std::string storePath =
      freshStorePath(std::string(info->test_suite_name()) + "_" +
                     info->name());
  forkAndRun(2, storePath, [step](int rank, int size, const std::string& p) {
    WorkerEnv env(p, rank, size);
    step(env.pg(), rank, size);
    return !::testing::Test::HasFailure();
  });
}

T fullLike(double value, std::initializer_list<int64_t> shape = {3}) {
  return T::full(shape, value, tensorplay::ScalarType::Float32);
}

T complexFullLike(
    std::complex<float> value,
    std::initializer_list<int64_t> shape = {3}) {
  return T::full(
      shape, value, tensorplay::ScalarType::ComplexFloat);
}

} // namespace

TEST(ProcessGroupGlooTest, TestRankAndSize) {
  runCollective([](ProcessGroupGloo& pg, int rank, int size) {
    EXPECT_EQ(pg.getRank(), rank);
    EXPECT_EQ(pg.getSize(), size);
  });
}

TEST(ProcessGroupGlooTest, TestBroadcast) {
  runCollective([](ProcessGroupGloo& pg, int rank, int) {
    std::vector<T> tensors = {
        fullLike(rank == 0 ? 99.0 : -1.0)};
    pg.broadcast(tensors, 0, 0, kTimeout)->wait();
    EXPECT_EQ(tensors[0].select(0, 0).item<double>(), 99.0);
  });
}

TEST(ProcessGroupGlooTest, TestAllreduceSum) {
  runCollective([](ProcessGroupGloo& pg, int rank, int) {
    std::vector<T> tensors = {fullLike(rank + 1.0)};
    pg.allreduce(tensors, ReduceOp::SUM, kTimeout)->wait();
    EXPECT_EQ(tensors[0].select(0, 0).item<double>(), 3.0);
  });
}

TEST(ProcessGroupGlooTest, TestAllreduceAvg) {
  runCollective([](ProcessGroupGloo& pg, int rank, int) {
    std::vector<T> tensors = {fullLike(2.0 * (rank + 1.0))};
    pg.allreduce(tensors, ReduceOp::AVG, kTimeout)->wait();
    EXPECT_EQ(tensors[0].select(0, 0).item<double>(), 3.0);
  });
}

TEST(ProcessGroupGlooTest, TestAllreduceProduct) {
  runCollective([](ProcessGroupGloo& pg, int rank, int) {
    std::vector<T> tensors = {fullLike(rank + 1.0)};
    pg.allreduce(tensors, ReduceOp::PRODUCT, kTimeout)->wait();
    EXPECT_EQ(tensors[0].select(0, 0).item<double>(), 2.0);
  });
}

TEST(ProcessGroupGlooTest, TestAllreduceMinMax) {
  runCollective([](ProcessGroupGloo& pg, int rank, int) {
    std::vector<T> lo = {fullLike(rank + 1.0)};
    std::vector<T> hi = {fullLike(rank + 1.0)};
    pg.allreduce(lo, ReduceOp::MIN, kTimeout)->wait();
    pg.allreduce(hi, ReduceOp::MAX, kTimeout)->wait();
    EXPECT_EQ(lo[0].select(0, 0).item<double>(), 1.0);
    EXPECT_EQ(hi[0].select(0, 0).item<double>(), 2.0);
  });
}

TEST(ProcessGroupGlooTest, TestAllreduceScalar) {
  runCollective([](ProcessGroupGloo& pg, int rank, int) {
    std::vector<T> tensors = {fullLike(rank + 1.0, {})};
    pg.allreduce(tensors, ReduceOp::SUM, kTimeout)->wait();
    EXPECT_EQ(tensors[0].item<double>(), 3.0);
  });
}

TEST(ProcessGroupGlooTest, TestComplexCollectives) {
  runCollective([](ProcessGroupGloo& pg, int rank, int size) {
    std::vector<T> reduced = {
        complexFullLike({rank + 1.0f, rank + 2.0f})};
    pg.allreduce(reduced, ReduceOp::SUM, kTimeout)->wait();
    const auto reducedValue = reduced[0].data_ptr<std::complex<float>>()[0];
    EXPECT_FLOAT_EQ(reducedValue.real(), 3.0f);
    EXPECT_FLOAT_EQ(reducedValue.imag(), 5.0f);

    T gatheredInput = T::full(
        {}, std::complex<float>(rank + 10.0f, rank + 20.0f),
        tensorplay::ScalarType::ComplexFloat);
    T gatheredOutput = T::full(
        {size}, std::complex<float>(-1.0f, -1.0f),
        tensorplay::ScalarType::ComplexFloat);
    pg.all_gather_single(gatheredOutput, gatheredInput, kTimeout)->wait();
    auto* gatheredData = gatheredOutput.data_ptr<std::complex<float>>();
    for (int i = 0; i < size; ++i) {
      EXPECT_FLOAT_EQ(gatheredData[i].real(), i + 10.0f);
      EXPECT_FLOAT_EQ(gatheredData[i].imag(), i + 20.0f);
    }

    std::vector<T> coalescedInputs = {complexFullLike(
        {rank + 30.0f, rank + 40.0f}, {})};
    std::vector<T> coalescedOutputs = {T::full(
        {size}, std::complex<float>(-1.0f, -1.0f),
        tensorplay::ScalarType::ComplexFloat)};
    pg.all_gather_single_coalesced(
          coalescedOutputs, coalescedInputs, kTimeout)
        ->wait();
    auto* coalescedData =
        coalescedOutputs[0].data_ptr<std::complex<float>>();
    for (int i = 0; i < size; ++i) {
      EXPECT_FLOAT_EQ(coalescedData[i].real(), i + 30.0f);
      EXPECT_FLOAT_EQ(coalescedData[i].imag(), i + 40.0f);
    }
  });
}

TEST(ProcessGroupGlooTest, TestComplexSparseAllreduce) {
  runCollective([](ProcessGroupGloo& pg, int rank, int) {
    const auto indices = T::tensor<int64_t>(
        {static_cast<int64_t>(rank)}, tensorplay::ScalarType::Int64)
                             .reshape({1, 1});
    const auto values = T::tensor<std::complex<float>>(
        {std::complex<float>(rank + 1.0f, rank + 2.0f)},
        tensorplay::ScalarType::ComplexFloat);
    std::vector<T> tensors = {
        T::make_sparse_coo_tensor(indices, values, {2}, true)};
    pg.allreduce(tensors, ReduceOp::SUM, kTimeout)->wait();
    const auto dense = tensors[0].to_dense();
    const auto* denseData = dense.data_ptr<std::complex<float>>();
    EXPECT_FLOAT_EQ(denseData[0].real(), 1.0f);
    EXPECT_FLOAT_EQ(denseData[0].imag(), 2.0f);
    EXPECT_FLOAT_EQ(denseData[1].real(), 2.0f);
    EXPECT_FLOAT_EQ(denseData[1].imag(), 3.0f);
  });
}

TEST(ProcessGroupGlooTest, TestAllreduceMultipleTensors) {
  runCollective([](ProcessGroupGloo& pg, int rank, int) {
    std::vector<T> tensors = {
        fullLike(rank + 1.0),
        fullLike(2.0 * (rank + 1.0)),
    };
    pg.allreduce(tensors, 0, kTimeout)->wait();
    EXPECT_EQ(tensors[0].select(0, 0).item<double>(), 9.0);
    EXPECT_EQ(tensors[1].select(0, 0).item<double>(), 9.0);
  });
}

TEST(ProcessGroupGlooTest, TestReduce) {
  runCollective([](ProcessGroupGloo& pg, int rank, int) {
    std::vector<T> tensors = {fullLike(1.0)};
    pg.reduce(tensors, 1, 0, 0, kTimeout)->wait();
    if (rank == 1) {
      EXPECT_EQ(tensors[0].select(0, 0).item<double>(), 2.0);
    }
  });
}

TEST(ProcessGroupGlooTest, TestAllgather) {
  runCollective([](ProcessGroupGloo& pg, int rank, int size) {
    std::vector<T> inputs = {fullLike(rank)};
    // One output group per input; each group holds one tensor per rank.
    std::vector<T> output_group;
    output_group.reserve(size);
    for (int i = 0; i < size; ++i) {
      output_group.push_back(
          T::full({3}, -1.0, tensorplay::ScalarType::Float32));
    }
    std::vector<std::vector<T>> outputs{std::move(output_group)};
    pg.allgather(outputs, inputs, kTimeout)->wait();
    for (int i = 0; i < size; ++i) {
      EXPECT_EQ(outputs[0][i].select(0, 0).item<double>(), (double)i);
    }
  });
}

TEST(ProcessGroupGlooTest, TestAllGatherIntoTensor) {
  runCollective([](ProcessGroupGloo& pg, int rank, int size) {
    T input = fullLike(rank + 1.0);
    T output = T::full(
        {size * 3}, -1.0, tensorplay::ScalarType::Float32);
    pg.all_gather_into_tensor(output, input, kTimeout)->wait();
    for (int i = 0; i < size; ++i) {
      for (int64_t j = 0; j < 3; ++j) {
        EXPECT_EQ(output.select(0, i * 3 + j).item<double>(), (double)(i + 1));
      }
    }
  });
}

TEST(ProcessGroupGlooTest, TestAllGatherSingle) {
  runCollective([](ProcessGroupGloo& pg, int rank, int size) {
    T input = fullLike(rank + 1.0);
    T output = T::full(
        {size * 3}, -1.0, tensorplay::ScalarType::Float32);
    pg.all_gather_single(output, input, kTimeout)->wait();
    for (int i = 0; i < size; ++i) {
      for (int64_t j = 0; j < 3; ++j) {
        EXPECT_EQ(output.select(0, i * 3 + j).item<double>(),
                  (double)(i + 1));
      }
    }
  });
}

TEST(ProcessGroupGlooTest, TestAllGatherSingleCoalesced) {
  runCollective([](ProcessGroupGloo& pg, int rank, int size) {
    std::vector<T> inputs = {
        fullLike(rank + 1.0),
        fullLike(2.0 * (rank + 1.0), {2}),
    };
    std::vector<T> outputs = {
        T::full({size * 3}, -1.0, tensorplay::ScalarType::Float32),
        T::full({size * 2}, -1.0, tensorplay::ScalarType::Float32),
    };
    pg.all_gather_single_coalesced(outputs, inputs, kTimeout)->wait();
    for (int i = 0; i < size; ++i) {
      EXPECT_EQ(outputs[0].select(0, i * 3).item<double>(),
                (double)(i + 1));
      EXPECT_EQ(outputs[1].select(0, i * 2).item<double>(),
                2.0 * (double)(i + 1));
    }
  });
}

TEST(ProcessGroupGlooTest, TestAllGatherSingleCoalescedScalar) {
  runCollective([](ProcessGroupGloo& pg, int rank, int size) {
    std::vector<T> inputs = {
        T::full({}, static_cast<double>(rank + 1),
                tensorplay::ScalarType::Float32),
    };
    std::vector<T> outputs = {
        T::full({size}, -1.0, tensorplay::ScalarType::Float32),
    };
    pg.all_gather_single_coalesced(outputs, inputs, kTimeout)->wait();
    for (int i = 0; i < size; ++i) {
      EXPECT_EQ(outputs[0].select(0, i).item<double>(),
                static_cast<double>(i + 1));
    }
  });
}

TEST(ProcessGroupGlooTest, TestGather) {
  runCollective([](ProcessGroupGloo& pg, int rank, int size) {
    std::vector<T> inputs = {fullLike(rank)};
    std::vector<std::vector<T>> outputs;
    if (rank == 0) {
      std::vector<T> output_group;
      output_group.reserve(size);
      for (int i = 0; i < size; ++i) {
        output_group.push_back(
            T::full({3}, -1.0, tensorplay::ScalarType::Float32));
      }
      outputs.push_back(std::move(output_group));
    }
    pg.gather(outputs, inputs, 0, kTimeout)->wait();
    if (rank == 0) {
      for (int i = 0; i < size; ++i) {
        EXPECT_EQ(outputs[0][i].select(0, 0).item<double>(), (double)i);
      }
    }
  });
}

TEST(ProcessGroupGlooTest, TestGatherSingle) {
  runCollective([](ProcessGroupGloo& pg, int rank, int size) {
    T input = fullLike(rank);
    T output = T::full(
        rank == 0 ? std::vector<int64_t>{size * 3}
                  : std::vector<int64_t>{0},
        -1.0,
        tensorplay::ScalarType::Float32);
    pg.gather_single(output, input, 0, kTimeout)->wait();
    if (rank == 0) {
      for (int i = 0; i < size; ++i) {
        EXPECT_EQ(output.select(0, i * 3).item<double>(), (double)i);
      }
    }
  });
}

TEST(ProcessGroupGlooTest, TestScatter) {
  runCollective([](ProcessGroupGloo& pg, int rank, int size) {
    std::vector<T> outputs = {
        T::full({3}, -1.0, tensorplay::ScalarType::Float32)};
    std::vector<std::vector<T>> inputs;
    if (rank == 0) {
      // One group holding one contribution per rank.
      std::vector<T> group;
      for (int i = 0; i < size; ++i) {
        group.push_back(fullLike(i));
      }
      inputs.emplace_back(std::move(group));
    }
    pg.scatter(outputs, inputs, 0, kTimeout)->wait();
    EXPECT_EQ(outputs[0].select(0, 0).item<double>(), (double)rank);
  });
}

TEST(ProcessGroupGlooTest, TestAlltoall) {
  runCollective([](ProcessGroupGloo& pg, int rank, int size) {
    std::vector<T> outputs;
    std::vector<T> inputs;
    for (int i = 0; i < size; ++i) {
      outputs.push_back(
          T::full({3}, -1.0, tensorplay::ScalarType::Float32));
      inputs.push_back(fullLike(rank * 10 + i));
    }
    pg.alltoall(outputs, inputs, kTimeout)->wait();
    for (int i = 0; i < size; ++i) {
      EXPECT_EQ(outputs[i].select(0, 0).item<double>(), (double)(i * 10 + rank));
    }
  });
}

TEST(ProcessGroupGlooTest, TestAllToAllSingle) {
  runCollective([](ProcessGroupGloo& pg, int rank, int size) {
    T input = fullLike(rank + 1.0, {size * 3});
    T output = T::full({size * 3}, -1.0, tensorplay::ScalarType::Float32);
    pg.all_to_all_single(output, input, {}, {}, kTimeout)->wait();
    for (int i = 0; i < size; ++i) {
      for (int64_t j = 0; j < 3; ++j) {
        EXPECT_EQ(output.select(0, i * 3 + j).item<double>(), (double)(i + 1));
      }
    }
  });
}

TEST(ProcessGroupGlooTest, TestReduceScatter) {
  runCollective([](ProcessGroupGloo& pg, int, int size) {
    std::vector<T> outputs = {
        T::full({2}, -1.0, tensorplay::ScalarType::Float32)};
    std::vector<std::vector<T>> inputs{
        std::vector<T>(size, T::full({2}, 1.0, tensorplay::ScalarType::Float32))};
    pg.reduce_scatter(outputs, inputs, 0, kTimeout)->wait();
    EXPECT_EQ(outputs[0].select(0, 0).item<double>(), (double)size);
  });
}

TEST(ProcessGroupGlooTest, TestReduceScatterTensor) {
  runCollective([](ProcessGroupGloo& pg, int, int size) {
    T input = T::full({size * 2}, 1.0, tensorplay::ScalarType::Float32);
    T output = T::full({2}, -1.0, tensorplay::ScalarType::Float32);
    pg.reduce_scatter_tensor(output, input, 0, kTimeout)->wait();
    EXPECT_EQ(output.select(0, 0).item<double>(), (double)size);
  });
}

TEST(ProcessGroupGlooTest, TestReduceScatterSingle) {
  runCollective([](ProcessGroupGloo& pg, int, int size) {
    T input = T::full({size * 2}, 1.0, tensorplay::ScalarType::Float32);
    T output = T::full({2}, -1.0, tensorplay::ScalarType::Float32);
    pg.reduce_scatter_single(output, input, 0, kTimeout)->wait();
    EXPECT_EQ(output.select(0, 0).item<double>(), (double)size);
  });
}

TEST(ProcessGroupGlooTest, TestReduceScatterSingleCoalesced) {
  runCollective([](ProcessGroupGloo& pg, int, int size) {
    std::vector<T> inputs = {
        T::full({size * 2}, 1.0, tensorplay::ScalarType::Float32),
        T::full({size * 3}, 2.0, tensorplay::ScalarType::Float32),
    };
    std::vector<T> outputs = {
        T::full({2}, -1.0, tensorplay::ScalarType::Float32),
        T::full({3}, -1.0, tensorplay::ScalarType::Float32),
    };
    pg.reduce_scatter_single_coalesced(outputs, inputs, 0, kTimeout)->wait();
    EXPECT_EQ(outputs[0].select(0, 0).item<double>(), (double)size);
    EXPECT_EQ(outputs[1].select(0, 0).item<double>(), 2.0 * (double)size);
  });
}

TEST(ProcessGroupGlooTest, TestReduceScatterSingleCoalescedScalarOutput) {
  runCollective([](ProcessGroupGloo& pg, int rank, int size) {
    T input = T::full(
        {size}, static_cast<double>(rank + 1),
        tensorplay::ScalarType::Float32);
    T output = T::full({}, -1.0, tensorplay::ScalarType::Float32);
    pg.reduce_scatter_single(output, input, 0, kTimeout)->wait();
    EXPECT_EQ(output.item<double>(),
              static_cast<double>(size * (size + 1) / 2));
  });
}

TEST(ProcessGroupGlooTest, TestSendRecvRing) {
  runCollective([](ProcessGroupGloo& pg, int rank, int size) {
    const int dst = (rank + 1) % size;
    const int src = (rank + size - 1) % size;
    std::vector<T> out = {fullLike(100.0 + rank)};
    std::vector<T> in = {
        T::full({3}, -1.0, tensorplay::ScalarType::Float32)};
    // Post both directions before waiting: waiting on the send first would
    // deadlock when both ranks queue sends simultaneously.
    auto sendWork = pg.send(out, dst, rank);
    auto recvWork = pg.recv(in, src, src);
    sendWork->wait();
    recvWork->wait();
    EXPECT_EQ(in[0].select(0, 0).item<double>(), 100.0 + src);
  });
}

TEST(ProcessGroupGlooTest, TestBarrier) {
  runCollective([](ProcessGroupGloo& pg, int, int) {
    pg.barrier(kTimeout)->wait();
    EXPECT_EQ(pg.getSequenceNumberForGroup() > 0, true);
  });
}

TEST(ProcessGroupGlooTest, TestMonitoredBarrier) {
  runCollective([](ProcessGroupGloo& pg, int, int) {
    pg.monitoredBarrier(kTimeout, false);
  });
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
