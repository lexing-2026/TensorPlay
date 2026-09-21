```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

(quantization-doc)=

# Quantization - tensorplay.ao.quantization

`tensorplay.ao` is the model optimization toolkit, and quantization is its
largest part: it converts a float model to run arithmetic on 8-bit integers
(roughly a 4x shrink in memory for weights), trading a controlled amount of
accuracy for it. The quantization stack lives in
`tensorplay.ao.quantization`, with the modules it produces under
`tensorplay.ao.nn` and the weight-pruning and numeric-comparison utilities
in sibling packages (see [pruning](pruning.md)).

Three workflows are supported, in increasing order of effort:

- **Dynamic quantization** — weights are quantized up front; activations
  are quantized on the fly at each use. One call, no calibration data:
  {func}`~tensorplay.ao.quantization.quantize_dynamic`.
- **Static (post-training) quantization** — weights and activations are both
  quantized ahead of time: insert observers with
  {func}`~tensorplay.ao.quantization.prepare`, calibrate on representative
  data, then bake the scales in with
  {func}`~tensorplay.ao.quantization.convert`.
- **Quantization-aware training (QAT)** — the same stubs, but with fake
  quantize in the training loop, so the weights adapt to the quantization
  noise before {func}`~tensorplay.ao.quantization.convert` runs.

```python
import tensorplay as tp
from tensorplay.ao.quantization import quantize_dynamic

model = tp.nn.Sequential(
    tp.nn.Linear(128, 64),
    tp.nn.ReLU(),
    tp.nn.Linear(64, 10),
)

# dynamic quantization: Linear weights go to int8, biases stay in float
qmodel = quantize_dynamic(model)
```

A quantized tensor carries its own scale and zero point; converting between
the quantized and float views is the job of the eager operators below, and
`float_model - qmodel` round trips are what the numeric-suite helpers
measure (see the last section).

## Eager operators

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.ao.quantization.quantize_per_tensor
    tensorplay.ao.quantization.quantize_per_channel
    tensorplay.ao.quantization.quantize_per_tensor_dynamic
    tensorplay.ao.quantization.dequantize
    tensorplay.ao.quantization.int_repr
    tensorplay.ao.quantization.is_quantized
    tensorplay.ao.quantization.q_scale
    tensorplay.ao.quantization.q_zero_point
    tensorplay.ao.quantization.q_per_channel_scales
    tensorplay.ao.quantization.q_per_channel_zero_points
    tensorplay.ao.quantization.q_per_channel_axis
    tensorplay.ao.quantization.qscheme
    tensorplay.ao.quantization.fake_quantize_per_tensor
    tensorplay.ao.quantization.fake_quantize_per_channel
    tensorplay.ao.quantization.quantized_linear
    tensorplay.ao.quantization.quantized_linear_dynamic
```

- {func}`~tensorplay.ao.quantization.quantize_per_tensor` /
  {func}`~tensorplay.ao.quantization.quantize_per_channel` map a float
  tensor to the integer range with one `(scale, zero_point)` pair, or one
  pair per channel along `axis`;
  {func}`~tensorplay.ao.quantization.quantize_per_tensor_dynamic` derives
  the scale from the tensor's own range instead of a calibrated value.
- {func}`~tensorplay.ao.quantization.dequantize` converts back to float;
  {func}`~tensorplay.ao.quantization.int_repr` shows the stored integer
  values.
- The `q_*` accessors read the quantization parameters back off a quantized
  tensor, and {func}`~tensorplay.ao.quantization.qscheme` reports which
  scheme (per-tensor or per-channel) it uses.
- `fake_quantize_*` simulate the rounding without leaving float — they are
  what training-time (QAT) quantization is built on.
- `quantized_linear` / `quantized_linear_dynamic` are the integer GEMM
  kernels behind quantized linear layers (static with calibrated input
  scales, dynamic with per-call input quantization).

## Observers and fake quantize

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.ao.quantization.MinMaxObserver
    tensorplay.ao.quantization.PerChannelMinMaxObserver
    tensorplay.ao.quantization.HistogramObserver
    tensorplay.ao.quantization.MovingAverageMinMaxObserver
    tensorplay.ao.quantization.MovingAveragePerChannelMinMaxObserver
    tensorplay.ao.quantization.PlaceholderObserver
    tensorplay.ao.quantization.FixedQParamsObserver
    tensorplay.ao.quantization.FakeQuantize
    tensorplay.ao.quantization.PerChannelFakeQuantize
    tensorplay.ao.quantization.default_observer
    tensorplay.ao.quantization.default_weight_observer
    tensorplay.ao.quantization.default_dynamic_quant_observer
    tensorplay.ao.quantization.get_observer_state_dict
    tensorplay.ao.quantization.load_observer_state_dict
```

Observers watch tensors pass through the model during calibration and
produce the `(scale, zero_point)` the conversion step needs.
`MinMaxObserver` tracks the running min/max (the default);
`HistogramObserver` keeps a histogram of values, giving a better scale when
outliers matter; the `MovingAverage*` variants smooth the min/max; and
`PlaceholderObserver` records nothing (for flows where parameters are
already known). `FakeQuantize` wraps an observer with the fake-quantize op,
which is the module QAT inserts into the graph.
{func}`~tensorplay.ao.quantization.get_observer_state_dict` /
{func}`~tensorplay.ao.quantization.load_observer_state_dict` save and
restore calibration results, so a calibration run need not be repeated.

## Configuration and the eager workflow

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.ao.quantization.QConfig
    tensorplay.ao.quantization.QConfigMapping
    tensorplay.ao.quantization.default_qconfig
    tensorplay.ao.quantization.default_dynamic_qconfig
    tensorplay.ao.quantization.default_per_channel_qconfig
    tensorplay.ao.quantization.default_weight_only_qconfig
    tensorplay.ao.quantization.QuantStub
    tensorplay.ao.quantization.DeQuantStub
    tensorplay.ao.quantization.prepare
    tensorplay.ao.quantization.convert
    tensorplay.ao.quantization.quantize
    tensorplay.ao.quantization.quantize_dynamic
    tensorplay.ao.quantization.fuse_modules
```

A {class}`~tensorplay.ao.quantization.QConfig` pairs an activation observer
with a weight observer (the `default_*` functions provide the common
combinations); a {class}`~tensorplay.ao.quantization.QConfigMapping`
assigns QConfigs per submodule type or name. `QuantStub` / `DeQuantStub`
mark where activations enter and leave the quantized region.

The static workflow composes these:

```python
from tensorplay.ao.quantization import prepare, convert

model.train()
m = prepare(model)               # inserts observers at the stubs
for x, _ in calibration_data:    # run representative data through
    m(x)
model = convert(m)               # observers become scale constants,
                                 # float layers become quantized layers
```

{func}`~tensorplay.ao.quantization.quantize` runs
prepare→calibrate (`run_fn`)→convert in one call for eager models;
{func}`~tensorplay.ao.quantization.quantize_dynamic` needs no calibration
step at all. {func}`~tensorplay.ao.quantization.fuse_modules` merges
adjacent layers (`Conv`+`BN`+`ReLU`, `Linear`+`ReLU`) into the single
fused modules below, which quantize better than the pieces separately.

## Quantized modules

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.ao.nn.quantized.Linear
    tensorplay.ao.nn.quantized.QuantizedLinear
    tensorplay.ao.nn.quantized.Conv1d
    tensorplay.ao.nn.quantized.Conv2d
    tensorplay.ao.nn.quantized.Conv3d
    tensorplay.ao.nn.quantized.BatchNorm2d
    tensorplay.ao.nn.quantized.BatchNorm3d
    tensorplay.ao.nn.quantized.ReLU
    tensorplay.ao.nn.quantized.ReLU6
    tensorplay.ao.nn.quantized.ELU
    tensorplay.ao.nn.quantized.LeakyReLU
    tensorplay.ao.nn.quantized.Hardswish
    tensorplay.ao.nn.quantized.Hardsigmoid
    tensorplay.ao.nn.quantized.Sigmoid
    tensorplay.ao.nn.quantized.Tanh
    tensorplay.ao.nn.quantized.MaxPool1d
    tensorplay.ao.nn.quantized.MaxPool2d
    tensorplay.ao.nn.quantized.MaxPool3d
    tensorplay.ao.nn.quantized.Quantize
    tensorplay.ao.nn.quantized.DeQuantize
    tensorplay.ao.nn.quantized.FloatFunctional
    tensorplay.ao.nn.quantized.QFunctional
    tensorplay.ao.nn.quantized.FXFloatFunctional
    tensorplay.ao.nn.quantized.dynamic.Linear
```

These are the integer kernels `convert` maps float layers onto: the
`Linear`/`Conv*` compute layers, quantized `BatchNorm`, the
activation functions, and `MaxPool`. `Quantize`/`DeQuantize` are the module
forms of the eager ops at the boundary stubs. `FloatFunctional` /
`QFunctional` wrap arithmetic (add, cat, mul, ...) so the same model code
runs its ops through quantized or float kernels;
`FXFloatFunctional` is the graph-capture-friendly variant.

## Fused modules

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.ao.nn.intrinsic.LinearReLU
    tensorplay.ao.nn.intrinsic.ConvReLU1d
    tensorplay.ao.nn.intrinsic.ConvReLU2d
    tensorplay.ao.nn.intrinsic.ConvReLU3d
    tensorplay.ao.nn.intrinsic.BNReLU2d
    tensorplay.ao.nn.intrinsic.BNReLU3d
```

`fuse_modules` produces these single-kernel combinations (`Conv`+`ReLU`,
`Linear`+`ReLU`, `BN`+`ReLU`), and `convert` then maps them onto their
fused quantized counterparts — one quantized kernel where the float model
had two or three layers.

## Quantization-aware training modules

For QAT, `tensorplay.ao.nn.qat` provides the float modules with fake
quantize built in — `ao.nn.qat.Linear` and `ao.nn.qat.Conv1d`/`Conv2d`/
`Conv3d` — which expose their weight and activation fake-quantize modules
as `weight_fake_quant` and `activation_post_process`. Under the QAT
workflow, `prepare` (in QAT mode) swaps the model's Linear and Conv layers
for these, training proceeds with quantization noise simulated in the
forward pass, and `convert` then turns them into the true quantized
modules. Access them through the `tensorplay.ao.nn.qat` namespace after
`tensorplay.ao.quantization` is imported.

## Numeric comparison utilities

The `tensorplay.ao.ns` subpackage holds the helpers for measuring what
quantization actually cost: `compare_weights` (float vs quantized weight
tables side by side), `prepare_model_with_stubs` + `Shadow` +
`compare_model_stub` (a shadow copy of the float model runs next to the
quantized one), and `prepare_model_outputs` + `compare_model_outputs`
(per-activation comparisons). They are reached through the
`tensorplay.ao.ns._numeric_suite` module.

## Where to go next

- [pruning](pruning.md) — the other model-optimization axis: sparsity
  instead of precision.
- [nn](nn.md) — the float modules the quantized stack starts from.
- [the main namespace](tensorplay.md) — the tensor APIs the eager
  quantize operators are part of.