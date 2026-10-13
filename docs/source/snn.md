# Spiking neural networks

`tensorplay.snn` provides the primitives for building and training spiking
neural networks (SNNs). The neuron dynamics run in fused native CPU/CUDA
operators (`snn_lif_step` / `snn_lif_step_backward` for a single step and
`snn_lif_seq` / `snn_lif_seq_backward` for whole spike trains); the Python
layer wires them into the autograd engine and the `nn.Module` API.

```python
import tensorplay as tp
from tensorplay import snn

lif = snn.LIFNode(threshold=1.0, decay=0.5)
x = tp.rand(4, 16)
spike = lif(x)          # single-step: [4, 16] binary spikes
lif.reset()

seq = snn.LIFNode(step_mode="m")
out = seq(tp.rand(8, 4, 16))   # multi-step: [8, 4, 16] spike train
```

## Neuron models

All neuron models share one membrane dynamic per step:

```
h     = decay * v + x
spike = heaviside(h - threshold)
v_new = h - threshold * spike             (soft reset)
v_new = h * (1 - spike) + v_reset * spike (hard reset)
```

| Module | Description |
| --- | --- |
| `IFNode` | Integrate-and-fire, `decay = 1` |
| `LIFNode` | Leaky integrate-and-fire, `decay` in `[0, 1]` |
| `ParametricLIFNode` | LIF with a learnable per-neuron `decay` parameter |

Each module keeps its membrane potential in the buffer `v`, supports
`step_mode="s"` (single step) and `step_mode="m"` (spike trains of shape
`[T, ...]`), and exposes `reset()` to clear the state.

## Surrogate gradients

The spike step is the Heaviside function, whose derivative is zero almost
everywhere. Surrogate-gradient training replaces that derivative with a
smooth approximation during backward:

| Surrogate | Backward weight |
| --- | --- |
| `StraightThroughSurrogate` | `1` |
| `SigmoidSurrogate(alpha)` | `alpha * sigmoid(alpha*x) * (1 - sigmoid(alpha*x))` |
| `FastSigmoidSurrogate(alpha)` | `alpha / (1 + alpha*|x|)^2` |
| `PiecewiseLinearSurrogate(alpha)` | `1` inside `(-alpha, alpha)`, else `0` |

`detach_reset=True` treats the spike in the voltage-update path as a constant
during backward, which is the standard stabilizing choice.

## Functional primitives

* `snn.heaviside(x)` -- the spike step.
* `snn.spike(x, surrogate)` -- Heaviside forward, surrogate backward.
* `snn.surrogate_grad(x, surrogate, alpha)` -- evaluate a surrogate gradient.
* `snn.lif_step(x, v, decay, threshold, v_reset, surrogate, detach_reset)` --
  one fused neuron update, returns `(spike, v_new)`.
* `snn.lif_seq(x, v_init, decay, threshold, v_reset, surrogate, detach_reset)` --
  a fused multi-step update on `[T, B, ...]` input, returns
  `(spikes, v_final)`. The time loop runs inside one native kernel, so
  multi-step training (BPTT) does not pay a per-step launch overhead.

## Native implementation

The neuron dynamics are implemented as fused CPU/CUDA kernels.  The single-
step operators (`snn_lif_step` / `snn_lif_step_backward`) process every
element in one pass (CPU: OpenMP ``parallel_for``; CUDA: grid-stride threads)
and support ``float32``, ``float64``, ``float16`` and ``bfloat16`` inputs,
with per-neuron or scalar ``decay`` / ``threshold`` broadcasting.  The
multi-step operators (`snn_lif_seq` / ``snn_lif_seq_backward``) iterate the
membrane dynamics over ``T`` inside the same kernel, so the whole spike-train
forward/backward is a single kernel launch per device.  The backward
recomputes the membrane trajectory from the stored spikes (no full forward
state is saved) and back-propagates the voltage gradient across time
(BPTT), so ``decay``, ``threshold``, ``v_init`` and every input frame receive
gradients.

``v_reset`` selects the reset mode: ``None`` for the soft reset
(subtract the threshold) and a scalar for the hard reset (reset to that
voltage).  ``detach_reset`` controls whether the spike inside the
voltage-update path is treated as a constant during backward; the default
``True`` is the standard stabilizing choice.

A throughput benchmark is provided in ``benchmark/snn_benchmark.py``; it
compares the fused multi-step operator with a Python time loop over
``snn.lif_step`` and reports spikes per second on both CPU and CUDA.

## Spike encoders

* `snn.rate_encode(x, T)` -- Bernoulli spikes over `T` steps.
* `snn.poisson_encode(x, T)` -- Poisson-process encoding (same draw).
* `snn.temporal_encode(x, T, tau)` -- time-to-first-spike encoding.