# Environment variables

TensorPlay reads a small set of environment variables to adjust runtime
behavior: where downloaded files are cached, whether out-of-tree device
backends load at import, how errors are reported, and how the experimental
graph translator validates itself.

Unless noted otherwise, set a variable **before** the process starts (or at
least before the module that reads it is imported — several are read once
and cached). Values are strings; `1` and `0` enable and disable.

## Paths and caching

| Variable | Default | Effect |
| --- | --- | --- |
| `TENSORPLAY_HOME` | `~/.cache/tensorplay/datasets/vision` | Root directory for weights downloaded by the vision model loaders. |

`TENSORPLAY_HOME` is consulted by
`tensorplay.vision.load_state_dict_from_url` (and the model zoo helpers
built on it) when deciding where to cache downloaded checkpoints. The hub
cache used by `tensorplay.hub` is a different tree — it lives under
`~/.cache/tensorplay` and is relocated through the `tensorplay.hub.set_dir`
API, not through this variable.

## Device and CUDA

| Variable | Default | Effect |
| --- | --- | --- |
| `TENSORPLAY_DEVICE_BACKEND_AUTOLOAD` | `1` | Import out-of-tree device backends at `import tensorplay`. |
| `TENSORPLAY_NVML_BASED_CUDA_CHECK` | unset | Make `cuda.is_available()` use NVML. |
| `TENSORPLAY_DEVICE_NAME` | `cpu` | Device key for the sparse operator tuning store. |

Setting `TENSORPLAY_DEVICE_BACKEND_AUTOLOAD=0` skips the out-of-tree
backend scan during import, for startup time or isolation. The current
state is visible through the helper the import path itself uses:

```python
import tensorplay
tensorplay._is_device_backend_autoload_enabled()
# True, unless the variable is set to something other than 1
```

With `TENSORPLAY_NVML_BASED_CUDA_CHECK=1`, `tensorplay.cuda.is_available()`
probes the driver through NVML instead of initializing a CUDA context —
useful because a context created before `fork` poisons the child. The
check is therefore fork-safe under this flag.

`TENSORPLAY_DEVICE_NAME` selects the device-name key under which the
sparse operator tuning metadata is stored and looked up; set it to the
accelerator in use (e.g. a GPU name) to read the matching tuning entries.

## Debugging

| Variable | Default | Effect |
| --- | --- | --- |
| `TENSORPLAY_SHOW_CPP_STACKTRACES` | unset | Attach captured native stack traces to errors raised from C++. |

When set to `1`, errors raised on the C++ side of the library carry the
native stack trace captured at the throw site. The flag is read once at
first use and cached — set it before the first error occurs, which in
practice means before the workload starts.

## Serialization

| Variable | Default | Effect |
| --- | --- | --- |
| `TENSORPLAY_SERIALIZATION_WORKERS` | `min(4, cpu count)` | Upper bound on threads staging MEGA tensor payloads. |

When saving a MEGA checkpoint, the tensor payloads are staged by a thread
pool whose size is `min(TENSORPLAY_SERIALIZATION_WORKERS, number of
tensors)` — never more threads than tensors, and never more than requested.
The default uses at most four workers regardless of core count. An invalid
(non-integer) value falls back to the default.

## Graph translation (experimental)

| Variable | Default | Effect |
| --- | --- | --- |
| `TENSORPLAY_TRANSLATION_VALIDATION` | `0` | Validate translated graphs against the original. |
| `TENSORPLAY_TRANSLATION_VALIDATION_TIMEOUT` | `600000` | Wall-clock budget for validation, in milliseconds. |
| `TENSORPLAY_TRANSLATION_NO_BISECT` | `0` | Skip bisection when validation fails. |

These configure the experimental graph translator's self-check: when
`TENSORPLAY_TRANSLATION_VALIDATION=1`, each translated program's guards are
checked symbolically (via sympy, and z3 when installed) to admit exactly
the executions of the original program. The timeout bounds the symbolic
check; when a mismatch is found, a bisection pass pinpoints the first
divergent guard unless `TENSORPLAY_TRANSLATION_NO_BISECT=1` skips it. All
three are read when `tensorplay.graph.experimental` is imported.
