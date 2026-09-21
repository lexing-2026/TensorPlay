(extending-tensorplay)=

# Extending TensorPlay

TensorPlay's built-in operator set and modules cover most workloads, but a
few situations call for adding your own: an operation autograd cannot
record automatically, a computation that leaves the framework (a native
library, a GPU kernel written by hand), or a type that should behave like
a tensor inside TensorPlay operations. This note walks through the four
extension points and when to reach for each.

## Adding new operators

The recommended path is {func}`tensorplay.library.custom_op`, which takes
a schema-annotated Python function plus an explicit mutation contract and
produces a real registered operator — visible to the dispatcher, the
compiler, and the testing utilities:

```python
import tensorplay as tp
from tensorplay.library import custom_op, register_fake, register_autograd


@custom_op("mylib::square_op", mutates_args=())
def square_op(x: tp.Tensor) -> tp.Tensor:
    return x * x


register_fake("mylib::square_op", lambda x: x.new_empty(x.shape))


def square_backward(ctx, grad):
    x, = ctx.saved_tensors
    return 2 * x * grad


register_autograd(
    "mylib::square_op",
    square_backward,
    setup_context=lambda ctx, inputs, output: ctx.save_for_backward(*inputs),
)
```

The operator now carries its own autograd formula: calling
`square_op(x)` on a tensor with `requires_grad=True` and back-propagating
yields exactly `2 * x`. The `setup_context` callback receives
`(ctx, inputs, output)` and is where tensors get saved for the backward
pass; when it is omitted the context stays empty and the backward must
compute from the gradient alone.

Choosing the mutation and aliasing contract (functional, in-place, `out=`,
or general mutation) is a schema-level decision described in detail on the
{doc}`library reference page </library>`. For kernels written in Triton or
TileLang, `tensorplay.library.triton_op` /
`tensorplay.library.tile_lang_op` wrap the kernel launch with the same
operator machinery. New operators can be validated with
{func}`tensorplay.library.opcheck`, and their gradients with
{func}`tensorplay.autograd.gradcheck.gradcheck`.

## C and C++ kernels (JIT)

Kernels written in C++ or CUDA enter through
{mod}`tensorplay.utils.cpp_jit`, a thin frontend over the `tvm-ffi`
compile engine. String sources are compiled into a shared library whose
exported functions accept any DLPack-compatible tensor; because
`tensorplay.Tensor` implements the DLPack protocol, tensors cross the
boundary zero-copy — the kernel sees a plain view (data pointer, shape,
strides, dtype, device) and no adapter layer is involved:

```python
import tensorplay as tp
from tensorplay.utils import cpp_jit

mod = cpp_jit.load_inline(
    name="myext",
    cpp_sources=r"""
    void scale_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
      for (int64_t i = 0; i < x.size(0); ++i) {
        static_cast<float*>(y.data_ptr())[i] =
            static_cast<float*>(x.data_ptr())[i] * 3.0f;
      }
    }
    """,
    functions=["scale_cpu"],
)

x = tp.tensor([1.0, 2.0, 3.0])
y = tp.empty_like(x)
mod.scale_cpu(x, y)          # y == 3 * x, no copies
```

### Writing the kernel

A kernel is an ordinary typed C++ function. The FFI enforces the
declared signature at the boundary — a call with mismatched argument
types fails before the kernel runs — and the string entry points
prepend the common headers (tensor view, dtype, error, function, env
API), so only extras like `<string>` need explicit inclusion.

Tensors arrive as `tvm::ffi::TensorView`, a non-owning view over the
caller's memory exposing `data_ptr()`, `shape()`, `strides()`,
`numel()`, `size(i)`, `ndim()`, `dtype()` and `device()`. Scalars pass
as plain C++ types — `double`, `int64_t`, `bool`, `std::string` — by
value or const reference, and any convertible type may be returned (a
`void` kernel returns `None`):

```python
mod = cpp_jit.load_inline(
    name="myext2",
    cpp_sources=r"""
    #include <string>

    double weighted_sum(tvm::ffi::TensorView x, double factor,
                        int64_t offset, const std::string& tag) {
      double s = static_cast<double>(offset);
      for (int64_t i = 0; i < x.numel(); ++i)
        s += static_cast<float*>(x.data_ptr())[i];
      return s * factor + tag.size();
    }
    """,
    functions=["weighted_sum"],
)
mod.weighted_sum(tp.tensor([1.0, 2.0, 3.0]), 10.0, 5, "abcd")  # -> 114.0
```

The view is untyped memory: nothing checks that a `float*` cast matches
the tensor's dtype, so validating `dtype()` and `device()` is the
kernel's responsibility. Raise through `TVM_FFI_THROW`, which crosses
the boundary as the matching Python exception:

```cpp
if (x.dtype().code != kDLFloat || x.dtype().bits != 32) {
  TVM_FFI_THROW(TypeError) << "expected float32";
}
```

Outputs can follow either of two conventions. The *out-parameter* form
has the caller allocate with `tp.empty_like` / `tp.empty` and pass the
output as another view, which the kernel writes into — the `scale_cpu`
example above does exactly this. Kernels can also allocate their own
outputs through the environment allocator; `cpp_jit` installs
TensorPlay's allocator automatically on first engine use (an
already-installed host allocator takes precedence), and the request is
served natively inside the compiled extension — no interpreter lock is
involved, so allocations from kernels running on worker threads do not
serialize with Python:

```cpp
tvm::ffi::Tensor doubled(tvm::ffi::TensorView x) {
  tvm::ffi::Tensor y = tvm::ffi::Tensor::FromEnvAlloc(
      TVMFFIEnvTensorAlloc, x.shape(), x.dtype(), x.device());
  tvm::ffi::TensorView yv(y);
  // write into yv ...
  return y;
}
```

The returned tensor arrives as the engine's tensor wrapper; bring it
back zero-copy with `tp.from_dlpack(mod.doubled(x))`. The out-parameter
form skips one wrapper hop and stays the cheaper option.

Repeated calls with the same `name` reuse the cached build, and passing
`cuda_sources=` compiles CUDA sources into the same library. The engine
is an optional dependency: `cpp_jit.is_available()` reports whether it is
importable, and the loading entry points raise an error with install
instructions when it is not.

An FFI kernel by itself is an opaque callable — it records no autograd
node and stays invisible to `tensorplay.compile` until wrapped. Combine
it with `tensorplay.library.custom_op` as in the previous section: the
operator body allocates outputs and calls the kernel, `register_fake`
propagates shapes, and `register_autograd` attaches the derivative.
Under `tensorplay.compile` the whole operator is captured as one opaque
node, so the kernel boundary survives compilation.

To ship a kernel to machines without a compiler, build it ahead of time
and load the artifact by path:

```python
lib = cpp_jit.build_inline(
    name="myext", cpp_sources=..., functions=["scale_cpu"])
# later, possibly in another process:
mod = cpp_jit.load_module(lib)
```

For real source trees, `cpp_jit.build` and `cpp_jit.load` take source
*files* instead of strings. Nothing is generated on the files' behalf:
they must carry their own header preamble and export macros
(`TVM_FFI_DLL_EXPORT_TYPED_FUNC`). Kernels linked straight into the
process — compiled into the executable rather than loaded from an
artifact — are reached through `cpp_jit.system_lib()` without dynamic
loading.

An artifact carries the kernels and nothing else. The operator wrapper —
schema, fake kernel, autograd formula — is Python and is re-declared on
the target machine with `custom_op`, exactly as in the JIT flow above;
only the compilation step is skipped. When the full operator definition
itself must ride with the artifact (registered kernels, no Python
re-declaration), build a compile-time op module instead: a yaml schema
processed by the repository's code generator into an op library linked
against the core. The two AOT forms are complementary: `cpp_jit` ships
kernels cheaply, the compile-time path ships whole operators.

FFI kernels see DLPack views, not the TensorPlay object model: they
cannot call back into the dispatcher or register operators themselves.
The same division applies — iteration and kernel shipping through
`cpp_jit`, deep framework integration through the compile-time path.

## Extending autograd

```{currentmodule} tensorplay.autograd
```

Operations that autograd cannot record — anything that runs outside the
framework, or whose derivative you want to hand-write — become part of the
graph by subclassing {class}`~tensorplay.autograd.function.Function`. Functions are
how autograd encodes the operation history: `apply` runs the forward and
installs a node that calls your `backward` during the backward pass.

### When to use

Implement a custom function when the computation is not differentiable as
written (argmax-style operations, numerically unstable expressions you
want to stabilize in the derivative), when it leaves the framework (a
native library call), or when fusing several operations lets you save
fewer buffers for backward than the un-fused graph would.

### When not to use

If the function can be written with built-in operations, autograd already
records it — write a plain Python function. If you need trainable state,
write a module instead. If you only want to observe or modify gradients,
a tensor hook or a module hook is the lighter tool.

### How to use

There are two styles. The *combined* style takes `ctx` as the first
argument of `forward` and saves onto it directly:

```python
import tensorplay as tp
from tensorplay.autograd import Function, gradcheck


class Exp(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.exp()

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        return grad_out * x.exp()


x = tp.randn(5, dtype=tp.float64, requires_grad=True)
out = Exp.apply(x)
```

The *separate* style keeps `forward` free of framework plumbing, which
lets it be called directly and reused; the context is filled in a
dedicated `setup_context` classmethod receiving `(ctx, inputs, output)`:

```python
class Mul(Function):
    @staticmethod
    def forward(a, b):
        return a * b

    @staticmethod
    def setup_context(ctx, inputs, output):
        a, b = inputs
        ctx.save_for_backward(a, b)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        return grad_out * b, grad_out * a
```

Either way, one backward argument is returned per forward input, `None`
for inputs that need no gradient. `ctx.needs_input_grad` is a tuple of
booleans saying which inputs actually require gradients, so the backward
can skip work (and avoid building graphs for tensors nobody will
differentiate against).

The saved context gives you more than storage:

- `ctx.save_for_backward(*tensors)` saves inputs or outputs without
  extending their lifetime in the forward graph; they come back as
  `ctx.saved_tensors`.
- `ctx.mark_non_differentiable(*tensors)` declares outputs that carry no
  gradient — backward through them is an error instead of a silent zero.
- `ctx.mark_dirty(*tensors)` marks in-place-modified inputs so autograd
  can detect version mismatches.
- `ctx.set_materialize_grads(False)` opts out of receiving zero-filled
  gradients for outputs whose gradient was not produced.

Because the backward itself runs under autograd, higher-order derivatives
come for free as long as the backward formula is written in
differentiable operations. The `Exp` above supports double backward: the
saved `x` flows through `x.exp()` again in the second pass. Verify both
orders with {func}`tensorplay.autograd.gradcheck.gradcheck`:

```python
a = tp.randn(5, dtype=tp.float64, requires_grad=True)
b = tp.randn(5, dtype=tp.float64, requires_grad=True)
assert gradcheck(Mul.apply, (a, b))
```

:::{note}
Forward-mode AD ({func}`tensorplay.autograd.jvp`) works for built-in
operations, but a custom `Function` cannot define a `jvp` yet — the
method exists for API compatibility and raises when the engine reaches
it. Custom functions are backward-mode only for now.
:::

### Function or custom_op?

Both attach a hand-written derivative to a computation. Use a
`Function` for a one-off differentiable step inside your own code: it is
local, takes no schema, and cannot be registered per-backend. Use
`custom_op` when the operation should behave like a real operator —
named, callable from compiled graphs, registrable per device type, with
a fake kernel for shape propagation — and register its autograd formula
with `register_autograd` as shown above.

## Extending the Python API

Operations dispatch through the `__tensorplay_function__` protocol when
any argument overrides it. A wrapper type that implements the protocol
participates in TensorPlay operations without subclassing `Tensor`:

```python
import tensorplay as tp
from tensorplay.overrides import (
    handle_tensorplay_function,
    has_tensorplay_function,
)


class MyArray:
    def __init__(self, tensor):
        self.tensor = tensor

    def __tensorplay_function__(self, func, types, args, kwargs):
        if func is tp.add:
            return MyArray(tp.add(args[0].tensor, args[1].tensor))
        return NotImplemented


m1, m2 = MyArray(tp.ones(2)), MyArray(tp.ones(2))
assert has_tensorplay_function((m1, m2))
result = handle_tensorplay_function(tp.add, (m1, m2), m1, m2)
```

The handler receives the public operation, the tuple of overridable
arguments, and the original call arguments; the method itself gets the
operation, the types of the overridable arguments, and the call
arguments. Returning `NotImplemented` defers to the remaining
overridable arguments, and if none handles the call, the operation
fails with a "no implementation found" error. Implement the method on
the class; the machinery discovers it per argument.

For a cross-cutting change of behavior — logging every operation,
swapping one implementation for another, enforcing a policy — use a mode
instead of editing types. `tensorplay.overrides.TensorPlayFunctionMode`
subclasses of it enter the dispatch path for all operations inside their
context:

```python
from tensorplay.overrides import TensorPlayFunctionMode


class DoubleInputsMode(TensorPlayFunctionMode):
    def __tensorplay_function__(self, func, types, args, kwargs):
        if func is tp.add:
            return tp.add(args[0] * 2, args[1] * 2)
        return super().__tensorplay_function__(func, types, args, kwargs)


with DoubleInputsMode():
    tp.add(tp.ones(2), tp.ones(2))   # -> tensor([4., 4.])
tp.add(tp.ones(2), tp.ones(2))       # -> tensor([2., 2.])
```

Modes stack (the innermost sees calls first) and restore the previous
behavior when the context exits.
