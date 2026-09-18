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

## Compiler backends

`tensorplay.compile(fn, backend=...)` accepts a registered name or any
callable with the contract
`backend(graph_module, example_inputs, **kwargs) -> callable`.
`tensorplay.compiler.list_backends()` lists the names; debug-tagged backends
appear with `list_backends(exclude_tags=None)`. An unknown name raises
`tensorplay.compiler.InvalidBackend`, which suggests close matches.

| Name | Tags | Purpose |
| --- | --- | --- |
| `stax` | | Native Stax lowering (default) |
| `tvm` | | Apache-TVM lowering; needs `apache-tvm`. Options: `target`, `parallel` |
| `cudagraphs` | | Capture each input layout once as a CUDA graph and replay it |
| `eager` | debug | Run the captured graph on its Python executor |
| `eager_noexcept` | debug | As `eager`; graph exceptions surface as compiler failures |
| `eager_debug` | debug | Run node by node; errors name the failing node |
| `*_TESTING_ONLY` | debug | Inject compile, run-time or accuracy failures on `relu` |

`cudagraphs` leaves a region uncaptured (and logs
`skipping cudagraphs due to ...`) when it mutates an input, touches the CPU
or several devices, or contains host-synchronizing or data-dependent-shape
operations. With `strict_native=True` those cases raise instead.
Calls that record autograd history run uncaptured.

### Third-party backends

A package makes a backend available by name through the
`tensorplay_compiler_backends` entry-point group:

```toml
[project.entry-points.tensorplay_compiler_backends]
my_compiler = "my_backend.compiler:my_compiler_function"
```

The entry point is imported the first time its name is looked up. Built-in
and explicitly registered names take precedence over entry points. A backend
that holds process-wide state may define `reset()`; `tensorplay.compiler.reset()`
calls it.
