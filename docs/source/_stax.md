# tensorplay._stax

`tensorplay._stax` is TensorPlay's private compilation package. It owns
capture orchestration, specialization guards, backend registration, code
caching, and the native Stax and TVM lowering implementations.

Graph values and graph transformations are intentionally separate:

- `tensorplay.graph` contains `Graph`, `Node`, `GraphModule`, `Proxy`, and
  `Tracer`.
- `tensorplay.graph.passes` contains graph transformations and pass
  composition utilities.
- `tensorplay._stax` consumes those graph objects and produces executable
  callables.

The supported public entry point is `tensorplay.compile`. The `_stax`
namespace is private and is exposed only for backend registration and
diagnostic tooling.

## Private services

- `CodeCache` — compiled artifact caching
- `Guard` and `GuardChain` — specialization validation
- `CudaGraphManager` — CUDA graph capture and replay
- `AOTError` and `build_aot` — ahead-of-time compilation helpers
