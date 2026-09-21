# tensorplay.cuda.tunable

```{eval-rst}
.. automodule:: tensorplay.cuda.tunable
```

TunableOp records, for each signature of the runtime-tunable GEMM family,
which underlying kernel implementation is fastest on the current hardware —
and replays that choice on later runs instead of re-searching. The tuning
results are written to a CSV file that can be read back, so a machine is
tuned once and reused.

## What is tuned

Matrix products dispatched through the cuBLASLt plan path can execute with
more than one library algorithm, and the library heuristic's top estimate is
not always the measured winner for a given GPU. While TunableOp is enabled,
each such GEMM first consults the results database:

* If a winner is recorded for its signature, that algorithm runs
  immediately — no measurement happens, even when tuning is also enabled.
* If no winner is recorded and tuning is enabled, every candidate algorithm
  is timed once and the fastest is recorded. The measurement cost is paid
  only on the first call per signature.
* If no winner is recorded and tuning is disabled, the library heuristic's
  top choice runs.

The tunable family covers the GEMMs that use the cuBLASLt plan path:
`Float32`/`Float64` matrix products (`mm`, `addmm`, `linear`-shaped calls)
and every bias-fused product, whose epilogue always runs on cuBLASLt. Paths
that never reach the plan path are not tunable and keep their fixed
dispatch: complex dtypes and the pinned classic-API fast paths, the custom
half-precision GEMV kernels, strided-batched GEMM and grouped GEMM.

While TunableOp is disabled (the default), dispatch is exactly the untuned
behavior; in particular the in-process selection that times the cuBLASLt
candidates once per shape still applies.

## Enabling the feature and tuning separately

The feature is enabled separately from the measurement phase itself:

```python
from tensorplay.cuda import tunable

tunable.enable()            # recorded winners are replayed; see below
tunable.tuning_enable()     # untuned shapes are measured (default on)
```

With the feature on and tuning off, a workload runs entirely on recorded
results (or the heuristic default where none are recorded) and performs no
measurement — the right mode for production replay. Turning tuning on as
well makes the first run double as the measurement pass.

## Environment variables

Launch-time configuration is available for runs whose code cannot call the
API. The variables are read once, when the tuning context is first touched
(the first API call or the first GEMM that consults it), and set the
initial state; any later API call overrides them.

| Variable | Meaning | Default |
| --- | --- | --- |
| `TP_TUNABLEOP_ENABLED` | initial state of the master switch | off |
| `TP_TUNABLEOP_TUNING` | initial state of the measurement switch | on |
| `TP_TUNABLEOP_RECORD_UNTUNED` | initial state of untuned logging | off |
| `TP_TUNABLEOP_VERBOSE` | initial state of diagnostic logging | off |
| `TP_TUNABLEOP_MAX_TUNING_DURATION_MS` | per-candidate time budget | `30` |
| `TP_TUNABLEOP_MAX_TUNING_SAMPLES` | per-candidate sample budget | `100` |
| `TP_TUNABLEOP_FILENAME` | results file; the device ordinal is embedded as with an explicit `set_filename` call | `tunableop_results<device>.csv` |

Boolean values accept `1`/`true`/`on` and `0`/`false`/`off`
(case-insensitive); a value that does not parse is reported with a warning
and ignored. A typical production deployment sets two variables and nothing
else:

```bash
TP_TUNABLEOP_ENABLED=1 TP_TUNABLEOP_FILENAME=/shared/tunableop_results.csv python train.py
```

## File input and output

The first time a GEMM consults the context, the results database is
prepared by reading the results file. The default filename is
`tunableop_results.csv` with the current device ordinal embedded
(`tunableop_results0.csv`), so one-process-per-device runs never share a
file; `set_filename` controls the name explicitly.

If tuning is enabled and new winners are found during the workload, they
are appended to this same file as they are measured, so a results file can
be built up across many workloads by reusing one name. `write_file()`
rewrites the file on demand with every recorded winner.

The file is CSV with validator lines followed by result lines:

```
Validator,TP_TUNABLEOP_FORMAT,1
Validator,CUDA_DEVICE,8.9:NVIDIA GeForce RTX 4090
Validator,CUBLASLT_VERSION,120300
GemmTunableOp_Float32,taN_m512_n1024_k256_bias0_dev0,lt_id7_tile12_stages3_splitk0_red0_swizzle0_custom0_inner1_cluster1,0.033
GemmTunableOp_Tf32,taT_m4096_n4096_k4096_bias0_dev0,Default,1.262
```

The "Validator" lines record the file format generation, the device and the
cuBLASLt build a winner was measured on. `read_file` rejects a file whose
validators do not match the current build and device, since its entries
would no longer describe runnable choices.

A recorded winner whose configuration no longer runs on the current build
is not treated as authoritative either: with tuning enabled the shape is
measured again and the stale entry is replaced in the database and
rewritten in the file, so the file heals on the next run instead of
accumulating dead lines. With tuning disabled, or under a CUDA graph
capture, the library heuristic's top choice runs instead. Verbose logging
reports every such rejection.

Each result line consists of four comma-separated fields: operator name,
operator parameters, kernel identifier and average execution time. The file
can be edited, with caution: setting the kernel field (field 3) to
`Default` falls back to the library heuristic's top choice for that
signature. The operator name and parameters (fields 1 and 2) are internal
keys and should not be modified.

## Measurement budget and diagnostics

Each candidate algorithm is timed for a bounded number of samples.
`set_max_tuning_duration` bounds the time in milliseconds and
`set_max_tuning_samples` bounds the sample count, each per candidate; a
value of zero disables a limit, the smaller bound wins when both are set,
and at least one timed sample always runs. Defaults are 30 ms and 100
samples.

`set_verbose(True)` turns on diagnostic logging to stderr covering state
changes, measurement passes and file activity. It is intended for
debugging; otherwise TunableOp is silent besides file output and warnings.

## Recording untuned GEMMs

GEMMs that ran without a tuned choice can be logged to a separate file with
`record_untuned_enable()`. Every unique signature is appended once to
`tunableop_untuned<device>.csv` in the working directory, which sizes how
much a workload would benefit from tuning and on which shapes:

```python
tunable.tuning_disable()
tunable.record_untuned_enable()
tunable.enable()
# ... run workload ...
tunable.record_untuned_disable()
```

Entries accumulate across runs; re-running the same workload with tuning
enabled then measures exactly the shapes the file lists.

## Interaction with CUDA graphs

No measurement happens while a CUDA graph capture is live (the timing
events would abort the capture). A shape whose first use happens under
capture runs the library heuristic's top choice, pinned for the process, so
eager reruns stay bit-identical to the captured replay. A signature with a
recorded winner keeps using it under capture, since that choice is already
fixed.

## Measured effect

`benchmark/bench_cuda_tunable.py` compares, per shape, the default dispatch
(in-process selection), a tuning run and a replay run, each in a fresh
process. On a GeForce RTX 3090 the recorded winners replay with no
steady-state penalty, and the per-shape first-call cost collapses because
the candidate measurement is skipped entirely:

| Shape (m×n×k, dtype) | Default first call | Replay first call |
| --- | --- | --- |
| 4096×4096×4096 float32 | 278 ms | 5.9 ms |
| 8192×4096×4096 float32 + bias | 464 ms | 12 ms |
| 4096×11008×4096 float16 + bias | 221 ms | 102 ms |
| 512×4096×4096 float16 + bias | 111 ms | 104 ms |
| 256×256×256 float64 + bias | 21 ms | 23 ms |

First-call times exclude one-time CUDA and library initialization. The
remaining first-call cost of the half-precision shapes is dominated by
loading that precision's kernels, which every mode pays once.

Steady-state throughput is unchanged on these shapes: the replay process
matches the default's per-iteration time, because the in-process selection
already lands on nearly the same algorithm as the larger tuning budget.
TunableOp's value here is the removed startup measurement and the
cross-process reuse, not a faster kernel. The tuning pass itself is the
most expensive first call — the default budget allows up to 30 ms of
measurement per candidate — and is paid once per shape per machine.

## API

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    enable
    disable
    is_enabled
    tuning_enable
    tuning_disable
    tuning_is_enabled
    record_untuned_enable
    record_untuned_disable
    record_untuned_is_enabled
    set_verbose
    is_verbose
    set_max_tuning_duration
    get_max_tuning_duration
    set_max_tuning_samples
    get_max_tuning_samples
    set_filename
    get_filename
    get_results
    read_file
    write_file
```

The `enable` pair is the master switch; the `tuning_*` pair controls
whether untuned shapes are measured (and new winners appended to the
results file as they are found); the `record_untuned_*` pair controls
logging of GEMMs that ran without a tuned choice; the `*_max_tuning_*`
pair bounds how long and how many samples a single candidate may be timed
for; and `read_file` / `write_file` load or persist the results database.
