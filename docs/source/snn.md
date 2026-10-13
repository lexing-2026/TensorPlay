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
| `ALIFNode` | LIF with an adaptive threshold (`a` term added to the base threshold) |
| `SpikeLinear` | Linear layer intended for binary spike-train inputs |

Each neuron module keeps its membrane potential in the buffer `v`, supports
`step_mode="s"` (single step) and `step_mode="m"` (spike trains of shape
`[T, ...]`), and exposes `reset()` to clear the state.  `LIFNode` and its
subclasses accept:

* `learnable_threshold=True` -- registers `threshold` as a trainable
  `Parameter` instead of a plain number.
* `return_v_seq=True` -- in `step_mode="m"`, `forward` returns
  `(spikes, v_seq)` where `v_seq` is the per-step voltage trajectory of
  shape `[T, ...]`.

`ALIFNode` adds the parameters `base_threshold` and `decay_th` (the leak of
the adaptation variable).  Its effective threshold is
`a + base_threshold`, with `a = decay_th * a + v` after each step, so the
threshold rises after strong activity and decays back to the base value.
Because the effective threshold changes every step, `ALIFNode` runs the
native single-step operator in a tight loop for multi-step inputs.

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
| `ATanSurrogate(alpha)` | `alpha / (1 + (pi*alpha*x)^2)` |
| `ErfSurrogate(alpha)` | `alpha / sqrt(pi) * exp(-(alpha*x)^2)` |
| `PiecewiseQuadraticSurrogate(alpha)` | `-alpha^2*|x| + alpha` inside `|x| <= 1/alpha`, else `0` |
| `PiecewiseExpSurrogate(alpha)` | `alpha/2 * exp(-alpha*|x|)` |
| `SuperSpikeSurrogate(alpha)` | `alpha / (1 + |x|)^2` |
| `NonzeroSignLogAbsSurrogate(alpha)` | `alpha / (1 + alpha*|x|)` |
| `PiecewiseLeakyRELUSurrogate(w, c)` | `1/(2w)` inside `|x| < w`, else `c` |
| `LeakyKRELUSurrogate(leak, k)` | `k` for `x >= 0`, `leak` for `x < 0` |
| `S2NNSurrogate(alpha, beta)` | sigmoid gradient for `x < 0`, `beta/(x+1)` else |
| `QPseudoSpikeSurrogate(alpha)` | `(1 + 2/(alpha-1)*|x|)^(-alpha)` |
| `SquarewaveFourierSeriesSurrogate(n, T)` | `4/T * sum_i cos((2i-1)*2*pi/T*x)` |

The fused LIF kernels support the single-parameter families (ids 0-9); the
multi-parameter surrogates (ids 10-14) are available through
``snn.spike()`` and ``snn.surrogate_grad()``.

`detach_reset=True` treats the spike in the voltage-update path as a constant
during backward, which is the standard stabilizing choice.

## Functional primitives

* `snn.heaviside(x)` -- the spike step.
* `snn.spike(x, surrogate)` -- Heaviside forward, surrogate backward.
* `snn.surrogate_grad(x, surrogate, alpha, beta, w, c, n, T_period)` --
  evaluate a surrogate gradient (all 15 surrogate ids).
* `snn.lif_step(x, v, decay, threshold, v_reset, surrogate, detach_reset)` --
  one fused neuron update, returns `(spike, v_new)`.
* `snn.lif_seq(x, v_init, decay, threshold, v_reset, surrogate,
  detach_reset, return_v_seq)` -- a fused multi-step update on
  `[T, B, ...]` input, returns `(spikes, v_final)` (or
  `(spikes, v_seq, v_final)` with `return_v_seq=True`).  The time loop runs
  inside one native kernel, so multi-step training (BPTT) does not pay a
  per-step launch overhead.  The returned ``v_seq`` is differentiable: its
  gradient is injected into the per-step voltage back-propagation.
* `snn.spike_rate(spikes, dim)` -- average firing rate along the time axis.
* `snn.spike_rate_loss(spikes, target_rate, reduction)` -- mean-squared-error
  between the firing rate and a target rate.

## Native implementation

The neuron dynamics are implemented as fused CPU/CUDA kernels.  The single-
step operators (`snn_lif_step` / `snn_lif_step_backward`) process every
element in one pass (CPU: OpenMP ``parallel_for``; CUDA: grid-stride threads)
and support ``float32``, ``float64``, ``float16`` and ``bfloat16`` inputs,
with per-neuron or scalar ``decay`` / ``threshold`` / ``v_reset``
broadcasting.  The multi-step operators (`snn_lif_seq` /
``snn_lif_seq_backward``) iterate the membrane dynamics over ``T`` inside the
same kernel, so the whole spike-train forward/backward is a single kernel
launch per device.  The backward recomputes the membrane trajectory from the
stored spikes (no full forward state is saved) and back-propagates the
voltage gradient across time (BPTT), so ``decay``, ``threshold``, ``v_init``
and every input frame receive gradients.

``v_reset`` selects the reset mode: ``None`` for the soft reset (subtract
the threshold) and a scalar or broadcastable tensor for the hard reset (reset
to that voltage).  ``detach_reset`` controls whether the spike inside the
voltage-update path is treated as a constant during backward; the default
``True`` is the standard stabilizing choice.

A throughput benchmark is provided in ``benchmark/snn_benchmark.py``; it
compares the fused multi-step operator with a Python time loop over
``snn.lif_step`` and reports spikes per second on both CPU and CUDA.

## Spike encoders

* `snn.rate_encode(x, T)` -- Bernoulli spikes over `T` steps.
* `snn.poisson_encode(x, T)` -- Poisson-process encoding (same draw).
* `snn.temporal_encode(x, T, tau)` -- time-to-first-spike encoding.
* `snn.PeriodicEncoder(encoding)` -- repeats a fixed periodic spike pattern.
* `snn.GaussianTuningCurveEncoder(mu, sigma)` -- population encoding where
  each input feature drives a bank of Gaussian-tuned Bernoulli neurons.
* `snn.LatencyEncoder(T, enc_function)` -- one-hot time-to-first-spike
  encoder with linear or logarithmic firing-time map.
* `snn.WeightedPhaseEncoder(T)` -- binary-decomposition weighted-phase
  encoding.

## Deep spiking layers

* `snn.SeqToANNContainer(module)` -- apply any ANN layer to every time step
  of a `[T, B, ...]` input.
* `snn.SpikingConv2d(...)` -- `Conv2d` (applied per time step) followed by a
  multi-step `LIFNode`, for `[T, B, C, H, W]` spike trains.
* `snn.SpikingBatchNorm1d` / `snn.SpikingBatchNorm2d` -- batch normalization
  with the time axis folded into the batch dimension, the standard
  stabilizer for deep SNN training.

## Monitoring

`snn.SpikeMonitor(module)` attaches a forward hook to any `nn.Module`,
records every spike tensor the module returns, and exposes
`spike_rate()`, `firing_rate(dim)`, `total_spikes()` and `last_spike()`.
Call `reset()` to clear the records and `detach()` to remove the hook.

## Training example

A complete MNIST training recipe (Poisson encoding, two-layer spiking
network, surrogate-gradient BPTT) can be assembled from this package:

```python
import tensorplay as tp
from tensorplay import snn

class SpikingNetwork(tp.nn.Module):
    def __init__(self, n_input=784, n_hidden=256, n_class=10):
        super().__init__()
        self.fc1 = snn.SpikeLinear(n_input, n_hidden, bias=True)
        self.lif1 = snn.LIFNode(threshold=1.0, decay=0.5, step_mode="m")
        self.fc2 = snn.SpikeLinear(n_hidden, n_class, bias=True)
        self.lif2 = snn.LIFNode(threshold=1.0, decay=0.5, step_mode="m")

    def forward(self, x):        # x: [T, B, 784] Poisson spikes
        return self.lif2(self.fc2(self.lif1(self.fc1(x)))).mean(0)

    def reset(self):
        self.lif1.reset()
        self.lif2.reset()

# per batch: encode -> forward -> MSE loss -> surrogate-gradient BPTT
# spikes = snn.poisson_encode(x_flat, T=8)   # [8, B, 784]
# out = model(spikes)                        # [B, 10] firing rates
# loss = ((out - y) ** 2).mean(); loss.backward(); optimizer.step()
```