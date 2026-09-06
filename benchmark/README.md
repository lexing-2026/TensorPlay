# ResNet classification benchmark

`benchmark_resnet_classification.py` trains a reference-framework ResNet-18 and a
TensorPlay ResNet-18 on the animal dataset in `test/data`. Both runs use the
same initial state, ImageNet normalization, deterministic epoch order,
optimizer settings, and batch size. The script reports train/evaluate/test
accuracy and training throughput for both frameworks.

It then copies the final reference state into a fresh TensorPlay model and runs
the same test batches through both models. This strict check compares logits
with `allclose` and requires identical Top-1 predictions; it is the accuracy
correctness gate. Inference p50/p95 latency and images/s are measured on
preprocessed batches and the TensorPlay/reference throughput ratio is printed.
The default `--device all` run does the complete comparison once on CPU and
once on CUDA. It also compares the reference compiled training path
with `tensorplay.compile(backend="stax", fullgraph=True, strict_native=True)`.
Compiled training follows the same public boundary as the reference: only the
model is compiled; loss, `backward()`, and SGD remain outside. The report
records the first-step compile cost, steady training throughput, and whether
the backward path is compiled or TensorPlay eager autograd. If Stax cannot
lower the graph to its native executor or Triton, the compiled result is marked
unavailable instead of falling back to the Python `GraphModule` executor.

The complete run is intentionally compute-heavy:

```bash
python benchmark/benchmark_resnet_classification.py \
  --data-root test/data \
  --epochs 10 \
  --image-size 224 \
  --batch-size 32 \
  --threads 8 \
  --json-out benchmark/results/resnet18.json
```

For a wiring/correctness smoke test, use one epoch and smaller inputs:

```bash
python benchmark/benchmark_resnet_classification.py \
  --device all \
  --epochs 1 --image-size 32 --batch-size 16 --threads 2 \
  --warmup 1 --repeats 1 \
  --json-out /tmp/resnet18-smoke.json
```

Use `--device cpu` or `--device cuda` to isolate one device. Use
`--no-compile` when only eager training/inference comparison is needed. Use
`--compiled-training-epochs 0` to keep compiled inference while skipping the
extra compiled-training run.

The training timer includes image decode and preprocessing so it reflects
the end-to-end classification loop. The separate inference timer excludes
filesystem work to make the model performance comparison reproducible.
