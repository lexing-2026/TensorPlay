// Unit tests for the intrusive reference-counting core and its use by the
// tensor handle types.
#include <gtest/gtest.h>

#include "IntrusivePtr.h"
#include "Storage.h"
#include "Tensor.h"

#include <utility>
#include <vector>

namespace {

using tensorplay::intrusive_ptr;
using tensorplay::make_intrusive;
using tensorplay::Storage;
using tensorplay::Tensor;
using tensorplay::TensorImpl;
using tensorplay::weak_intrusive_ptr;

struct Dummy : tensorplay::IntrusivePtrTarget {
    static int live;
    std::vector<int> payload{1, 2, 3};
    Dummy() { ++live; }
    ~Dummy() override { --live; }
};
int Dummy::live = 0;

TEST(IntrusivePtrTest, MakeCopyMove) {
    ASSERT_EQ(Dummy::live, 0);
    {
        intrusive_ptr<Dummy> a = make_intrusive<Dummy>();
        ASSERT_EQ(Dummy::live, 1);
        EXPECT_EQ(a.use_count(), 1u);
        EXPECT_TRUE(a.unique());

        intrusive_ptr<Dummy> b = a;
        EXPECT_EQ(a.use_count(), 2u);
        EXPECT_EQ(b.use_count(), 2u);
        EXPECT_EQ(a.get(), b.get());

        intrusive_ptr<Dummy> c = std::move(b);
        EXPECT_EQ(b.get(), nullptr);
        EXPECT_EQ(a.use_count(), 2u);
        EXPECT_EQ(c->payload.size(), 3u);
    }
    EXPECT_EQ(Dummy::live, 0);
}

TEST(IntrusivePtrTest, ResetAndAssign) {
    auto a = make_intrusive<Dummy>();
    auto b = make_intrusive<Dummy>();
    Dummy* raw = a.get();
    b = a;  // drops b's old target, shares a's
    EXPECT_EQ(b.get(), raw);
    EXPECT_EQ(a.use_count(), 2u);
    b.reset();
    EXPECT_EQ(a.use_count(), 1u);
    EXPECT_EQ(Dummy::live, 1);
}

TEST(IntrusivePtrTest, WeakLockAndExpiry) {
    intrusive_ptr<Dummy> strong = make_intrusive<Dummy>();
    weak_intrusive_ptr<Dummy> weak = strong;

    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(weak.use_count(), 1u);
    intrusive_ptr<Dummy> revived = weak.lock();
    EXPECT_EQ(revived.get(), strong.get());
    EXPECT_EQ(strong.use_count(), 2u);
    revived.reset();

    strong.reset();
    EXPECT_TRUE(weak.expired());
    EXPECT_FALSE(weak.lock().defined());
}

TEST(IntrusivePtrTest, ReleaseResourcesBeforeBodyDestruction) {
    // With a weak observer alive, the strong count hitting zero must hand
    // the payload over immediately; the object body is destroyed only when
    // the weak reference goes away too.
    auto strong = make_intrusive<Dummy>();
    weak_intrusive_ptr<Dummy> weak = strong;
    EXPECT_EQ(Dummy::live, 1);
    strong.reset();
    EXPECT_EQ(Dummy::live, 1);  // body kept for the weak reference
    weak.reset();
    EXPECT_EQ(Dummy::live, 0);
}

TEST(IntrusivePtrTest, TensorHandleSharesImpl) {
    Tensor a({2, 3}, tensorplay::DType::Float32);
    ASSERT_TRUE(a.defined());
    Tensor b = a;  // handle copy: same impl, one more reference
    EXPECT_EQ(a.impl().get(), b.impl().get());
    EXPECT_EQ(a.impl().use_count(), 2u);

    Tensor c({2, 3}, tensorplay::DType::Float32);
    EXPECT_NE(a.impl().get(), c.impl().get());

    Tensor empty;
    EXPECT_FALSE(empty.defined());
    EXPECT_EQ(empty.impl().use_count(), 0u);
}

TEST(IntrusivePtrTest, StorageHandleSharesImpl) {
    Storage s(64);
    Storage t = s;
    EXPECT_TRUE(s.is_same(t));
    EXPECT_EQ(s.use_count(), 2u);
    EXPECT_FALSE(s.is_same(Storage(64)));
}

TEST(IntrusivePtrTest, ViewSharesStorageAcrossImpls) {
    Tensor base = tensorplay::Tensor({4}, tensorplay::DType::Float32);
    Tensor view = base.slice(0, 1, 3);
    EXPECT_NE(base.impl().get(), view.impl().get());
    EXPECT_TRUE(base.impl()->storage().is_same(view.impl()->storage()));
}

} // namespace
