#pragma once

#include "Macros.h"

namespace tensorplay {

class Tensor;

// Opaque extension point carried by TensorImpl.
// The concrete autograd implementation (tensorplay::tpx::AutogradMeta) lives in
// the tpx library and is reachable through this interface, keeping p10 free of
// any dependency on the autograd layer.
class P10_API AutogradMetaBase {
public:
    virtual ~AutogradMetaBase() = default;

    virtual bool requires_grad() const = 0;
    virtual void set_requires_grad(bool requires_grad) = 0;

    virtual Tensor grad() const = 0;
    virtual void set_grad(const Tensor& grad) = 0;

    virtual bool retains_grad() const = 0;
    virtual void set_retains_grad(bool retains_grad) = 0;
};

} // namespace tensorplay