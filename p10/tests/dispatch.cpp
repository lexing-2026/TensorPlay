// Dispatcher behavior: registration lookup, backend dispatch, error paths.

#include <gtest/gtest.h>

#include "Dispatcher.h"
#include "Exception.h"
#include "FileCheck.h"
#include "Tensor.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

using namespace tensorplay;
namespace ops = tensorplay::tpx::ops;

TEST(DispatchTest, RegisteredOpHasKernel) {
    auto handle = Dispatcher::singleton().findHandle("uniform_");
    KernelFunction kernel = handle.getKernel(DispatchKey::CPU);
    EXPECT_NE(kernel, nullptr);
}

TEST(DispatchTest, UnregisteredOpHasNoKernel) {
    auto handle = Dispatcher::singleton().findHandle("no_such_op_exists_at_all");
    KernelFunction kernel = handle.getKernel(DispatchKey::CPU);
    EXPECT_EQ(kernel, nullptr);
}

TEST(DispatchTest, CompositeFallsBackToCompositeKey) {
    // An op registered only under the Composite key is resolvable from a
    // backend key through the composite fallthrough.
    auto handle = Dispatcher::singleton().findHandle("isin_Tensor_Tensor");
    KernelFunction composite_kernel = handle.getKernel(DispatchKey::Composite);
    if (composite_kernel != nullptr) {
        KernelFunction cpu_kernel = handle.getKernel(DispatchKey::CPU);
        EXPECT_NE(cpu_kernel, nullptr);
    }
}

TEST(DispatchTest, MissingKernelThrowsWithDiagnostic) {
    // Calling an op with no kernel on any key raises with an actionable
    // message naming the op and the backend.
    Tensor t = ops::ones({2}, DType::Float32, Device(DeviceType::CPU));
    bool threw = false;
    try {
        DispatchStub<Tensor&, Tensor&>::call("no_such_op_exists_at_all",
                                            DispatchKey::CPU, t);
    } catch (const NotImplementedError& e) {
        threw = true;
        FileCheck().check("Kernel not found for op")->run(e.msg());
        FileCheck().check("no_such_op_exists_at_all")->run(e.msg());
    }
    EXPECT_TRUE(threw);
}

TEST(DispatchTest, OpCallThroughGeneratedWrapper) {
    // The generated wrappers resolve through the dispatcher at runtime.
    Tensor t = ops::ones({2, 2}, DType::Float32, Device(DeviceType::CPU));
    Tensor r = ops::matmul(t, t);
    EXPECT_EQ(r.shape(), Size({2, 2}));
}
