#!/usr/bin/env python3
"""Train and benchmark ResNet-18 on ``test/data`` against PyTorch.

The benchmark has two complementary checks:

1. Torch and TensorPlay start from the same state and train independently on
   the same preprocessed samples and batch order.  Their train/evaluate/test
   accuracy and wall time are reported separately.
2. The final Torch state is copied into a fresh TensorPlay model.  The two
   models then run on the same test batches.  This is the strict operator
   parity gate: logits must be numerically close and Top-1 predictions must
   be identical.  The same model-only boundary is used for compiled training
   so TorchInductor's and TensorPlay's AOTAutograd-style forward/backward
   paths are reported separately.

Image decoding and preprocessing are deliberately implemented once here and
shared by both frameworks.  The training timer includes that work, while the
inference benchmark uses preprocessed test batches so model throughput is not
blurred by filesystem variance.

Example (full benchmark)::

    python benchmark/benchmark_resnet_classification.py \
        --data-root test/data --epochs 10 --image-size 224 --batch-size 32

Example (fast smoke)::

    python benchmark/benchmark_resnet_classification.py \
        --epochs 1 --image-size 32 --batch-size 16 --threads 2 \
        --warmup 1 --repeats 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# Make ``python benchmark/benchmark_resnet_classification.py`` use the
# checkout's TensorPlay package rather than requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import Torch before TensorPlay.  This avoids runtime-library load-order
# conflicts on installations that contain both CUDA-enabled packages.
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18 as torchvision_resnet18

import tensorplay as tp
import tensorplay.nn as tp_nn
import tensorplay.optim as tp_optim
from tensorplay.utils.data import Dataset as TensorPlayDataset
from tensorplay.vision.models import resnet18 as tensorplay_resnet18


NUM_CLASSES = 3
CLASS_NAMES = ("cats", "dogs", "snakes")
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True)
class DeviceContext:
    """The matching Torch and TensorPlay device used by one benchmark run."""

    name: str
    torch_device: torch.device
    tensorplay_device: tp.Device


def synchronize_torch(context: DeviceContext) -> None:
    if context.name == "cuda":
        torch.cuda.synchronize(context.torch_device)


def synchronize_tensorplay(context: DeviceContext) -> None:
    if context.name == "cuda":
        tp.cuda.synchronize(context.tensorplay_device)


def tensorplay_to_numpy(value: tp.Tensor) -> np.ndarray:
    """Materialize a TensorPlay value on the host for comparison/reporting."""

    return value.detach().cpu().numpy()


@dataclass(frozen=True)
class SampleSplit:
    name: str
    root: Path
    paths: tuple[Path, ...]
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.paths)


class IndexedImageDataset(TensorPlayDataset):
    """Small Dataset wrapper used to validate the TensorPlay data path.

    The actual training loop uses explicit, shared batch indices so that the
    Torch and TensorPlay runs see exactly the same order.  This class keeps a
    normal ``Dataset`` implementation available for callers that want to
    benchmark the project DataLoader separately.
    """

    def __init__(self, split: SampleSplit, image_size: int):
        self.split = split
        self.image_size = image_size

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        image = preprocess_image(self.split.paths[index], self.image_size)
        return tp.tensor(image), int(self.split.labels[index])


def read_split(data_root: Path, name: str) -> SampleSplit:
    csv_path = data_root / f"{name}.csv"
    image_root = data_root / name
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing split manifest: {csv_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"missing split directory: {image_root}")

    paths: list[Path] = []
    labels: list[int] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            filename = row.get("name")
            if not filename:
                raise ValueError(f"{csv_path} contains a row without a name")
            path = image_root / filename
            if not path.is_file():
                raise FileNotFoundError(f"manifest entry does not exist: {path}")
            label = int(row["label"])
            if not 0 <= label < NUM_CLASSES:
                raise ValueError(f"invalid label {label} in {csv_path}")
            paths.append(path)
            labels.append(label)

    if not paths:
        raise ValueError(f"empty split: {csv_path}")
    return SampleSplit(name, image_root, tuple(paths), np.asarray(labels, dtype=np.int64))


def preprocess_image(path: Path, image_size: int) -> np.ndarray:
    """Decode one image and return a contiguous normalized CHW float32 array."""

    resampling = getattr(Image, "Resampling", Image).BILINEAR
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), resampling)
        array = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    array = np.ascontiguousarray(array.transpose(2, 0, 1))
    array = (array - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return np.ascontiguousarray(array, dtype=np.float32)


def prepare_numpy_batch(
    split: SampleSplit, indices: Sequence[int], image_size: int
) -> tuple[np.ndarray, np.ndarray]:
    images = np.stack(
        [preprocess_image(split.paths[int(index)], image_size) for index in indices],
        axis=0,
    )
    labels = np.ascontiguousarray(split.labels[np.asarray(indices, dtype=np.int64)])
    return np.ascontiguousarray(images, dtype=np.float32), labels


def make_epoch_indices(
    split_size: int, batch_size: int, seed: int, epoch: int, shuffle: bool
) -> list[np.ndarray]:
    if shuffle:
        order = np.random.default_rng(seed + epoch).permutation(split_size)
    else:
        order = np.arange(split_size, dtype=np.int64)
    return [order[start : start + batch_size] for start in range(0, split_size, batch_size)]


def torch_state_to_tensorplay(
    state_dict: dict[str, torch.Tensor], device: tp.Device
) -> dict[str, tp.Tensor]:
    """Copy a Torch state dict without retaining Torch storage or autograd."""

    converted: dict[str, tp.Tensor] = {}
    for name, value in state_dict.items():
        array = value.detach().cpu().numpy().copy()
        # The current CUDA factory path does not safely consume a NumPy
        # buffer directly.  Materialize on CPU first, then use TensorPlay's
        # tested device-copy path.
        converted[name] = tp.tensor(array).to(device)
    return converted


def load_torch_state_into_tensorplay(
    model: tp_nn.Module, state_dict: dict[str, torch.Tensor], device: tp.Device
) -> None:
    # Use an explicit copy here instead of Module.load_state_dict.  The
    # checkout currently carries an incomplete torch-compatibility shim for
    # ``tensorplay.__future__``; the benchmark must still be able to provide
    # a strict, auditable cross-framework weight transfer.
    source = torch_state_to_tensorplay(state_dict, device)
    target = model.state_dict(keep_vars=True)
    missing = sorted(set(target) - set(source))
    unexpected = sorted(set(source) - set(target))
    mismatched = sorted(
        name
        for name in target.keys() & source.keys()
        if tuple(target[name].shape) != tuple(source[name].shape)
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"state transfer failed: missing={missing}, unexpected={unexpected}, "
            f"shape_mismatch={mismatched}"
        )
    with tp.no_grad():
        for name, value in source.items():
            target[name].copy_(value)


def framework_batches(
    split: SampleSplit,
    batch_indices: Iterable[Sequence[int]],
    image_size: int,
    framework: str,
    context: DeviceContext,
):
    for indices in batch_indices:
        images, labels = prepare_numpy_batch(split, indices, image_size)
        if framework == "torch":
            yield (
                torch.from_numpy(images).to(context.torch_device),
                torch.from_numpy(labels).to(context.torch_device),
            )
        elif framework == "tensorplay":
            tensorplay_images = tp.tensor(images)
            tensorplay_labels = tp.tensor(labels, dtype=tp.int64)
            if context.name == "cuda":
                tensorplay_images = tensorplay_images.to(context.tensorplay_device)
                tensorplay_labels = tensorplay_labels.to(context.tensorplay_device)
            yield (
                tensorplay_images,
                tensorplay_labels,
            )
        else:
            raise ValueError(f"unknown framework: {framework}")


def train_torch_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    split: SampleSplit,
    batch_indices: list[np.ndarray],
    image_size: int,
    context: DeviceContext,
) -> dict[str, float]:
    model.train()
    synchronize_torch(context)
    started = time.perf_counter()
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, labels in framework_batches(
        split, batch_indices, image_size, "torch", context
    ):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = int(labels.shape[0])
        loss_sum += float(loss.item()) * batch_size
        correct += int((logits.argmax(1) == labels).sum().item())
        total += batch_size
    synchronize_torch(context)
    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "samples_per_second": total / elapsed,
    }


def train_tensorplay_epoch(
    model: tp_nn.Module,
    optimizer: tp_optim.SGD,
    criterion: tp_nn.Module,
    split: SampleSplit,
    batch_indices: list[np.ndarray],
    image_size: int,
    context: DeviceContext,
) -> dict[str, float]:
    model.train()
    synchronize_tensorplay(context)
    started = time.perf_counter()
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, labels in framework_batches(
        split, batch_indices, image_size, "tensorplay", context
    ):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = int(labels.shape[0])
        loss_sum += float(loss.item()) * batch_size
        correct += int(
            (
                tensorplay_to_numpy(logits.argmax(1))
                == tensorplay_to_numpy(labels)
            ).sum()
        )
        total += batch_size
    synchronize_tensorplay(context)
    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "samples_per_second": total / elapsed,
    }


def evaluate_torch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    split: SampleSplit,
    batch_size: int,
    image_size: int,
    context: DeviceContext,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in framework_batches(
            split,
            make_epoch_indices(len(split), batch_size, 0, 0, False),
            image_size,
            "torch",
            context,
        ):
            logits = model(images)
            batch_size_now = int(labels.shape[0])
            loss_sum += float(criterion(logits, labels).item()) * batch_size_now
            correct += int((logits.argmax(1) == labels).sum().item())
            total += batch_size_now
    return {"loss": loss_sum / total, "accuracy": correct / total}


def evaluate_tensorplay(
    model: tp_nn.Module,
    criterion: tp_nn.Module,
    split: SampleSplit,
    batch_size: int,
    image_size: int,
    context: DeviceContext,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with tp.no_grad():
        for images, labels in framework_batches(
            split,
            make_epoch_indices(len(split), batch_size, 0, 0, False),
            image_size,
            "tensorplay",
            context,
        ):
            logits = model(images)
            batch_size_now = int(labels.shape[0])
            loss_sum += float(criterion(logits, labels).item()) * batch_size_now
            correct += int(
                (
                    tensorplay_to_numpy(logits.argmax(1))
                    == tensorplay_to_numpy(labels)
                ).sum()
            )
            total += batch_size_now
    return {"loss": loss_sum / total, "accuracy": correct / total}


def prepared_test_batches(
    split: SampleSplit,
    batch_size: int,
    image_size: int,
    context: DeviceContext,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[tp.Tensor, tp.Tensor]]]:
    indices = make_epoch_indices(len(split), batch_size, 0, 0, False)
    torch_batches = list(
        framework_batches(split, indices, image_size, "torch", context)
    )
    tensorplay_batches = list(
        framework_batches(split, indices, image_size, "tensorplay", context)
    )
    return torch_batches, tensorplay_batches


def collect_logits(
    model: torch.nn.Module | tp_nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]]
    | list[tuple[tp.Tensor, tp.Tensor]],
    framework: str,
) -> tuple[np.ndarray, np.ndarray]:
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    if framework == "torch":
        model.eval()
        with torch.inference_mode():
            for images, batch_labels in batches:
                logits.append(model(images).detach().cpu().numpy())
                labels.append(batch_labels.detach().cpu().numpy())
    else:
        model.eval()
        with tp.no_grad():
            for images, batch_labels in batches:
                logits.append(tensorplay_to_numpy(model(images)))
                labels.append(tensorplay_to_numpy(batch_labels))
    return np.concatenate(logits, axis=0), np.concatenate(labels, axis=0)


def benchmark_inference(
    torch_model: torch.nn.Module,
    tensorplay_model: tp_nn.Module,
    torch_batches: list[tuple[torch.Tensor, torch.Tensor]],
    tensorplay_batches: list[tuple[tp.Tensor, tp.Tensor]],
    warmup: int,
    repeats: int,
    context: DeviceContext,
) -> dict[str, dict[str, float]]:
    def run_torch() -> None:
        with torch.inference_mode():
            for images, _ in torch_batches:
                torch_model(images)

    def run_tensorplay() -> None:
        with tp.no_grad():
            for images, _ in tensorplay_batches:
                tensorplay_model(images)

    torch_model.eval()
    tensorplay_model.eval()
    for _ in range(warmup):
        run_torch()
        synchronize_torch(context)
        run_tensorplay()
        synchronize_tensorplay(context)

    torch_times: list[float] = []
    tensorplay_times: list[float] = []
    for _ in range(repeats):
        synchronize_torch(context)
        started = time.perf_counter()
        run_torch()
        synchronize_torch(context)
        torch_times.append(time.perf_counter() - started)
        synchronize_tensorplay(context)
        started = time.perf_counter()
        run_tensorplay()
        synchronize_tensorplay(context)
        tensorplay_times.append(time.perf_counter() - started)

    samples = sum(int(labels.shape[0]) for _, labels in torch_batches)

    def summarize(times: list[float]) -> dict[str, float]:
        return {
            "p50_seconds": float(np.percentile(times, 50)),
            "p95_seconds": float(np.percentile(times, 95)),
            "images_per_second_p50": samples / float(np.percentile(times, 50)),
            "batch_count": len(torch_batches),
            "samples": samples,
        }

    return {"torch": summarize(torch_times), "tensorplay": summarize(tensorplay_times)}


def benchmark_compiled_inference(
    torch_model: torch.nn.Module,
    tensorplay_model: tp_nn.Module,
    torch_batches: list[tuple[torch.Tensor, torch.Tensor]],
    tensorplay_batches: list[tuple[tp.Tensor, tp.Tensor]],
    eager_logits: dict[str, np.ndarray],
    warmup: int,
    repeats: int,
    context: DeviceContext,
    torch_backend: str,
    tensorplay_backend: str,
    compile_mode: str,
) -> dict[str, dict[str, object]]:
    """Compare lazy compile cost and steady-state inference throughput.

    Both frontends compile lazily on their first call, so the report keeps
    wrapper construction and first-call (actual lowering/code generation)
    time separate.  The compiled output is also checked against that
    framework's eager output before the timing result is accepted.
    """

    def run_torch(compiled: Callable[..., object]) -> None:
        with torch.inference_mode():
            for images, _ in torch_batches:
                compiled(images)

    def run_tensorplay(compiled: Callable[..., object]) -> None:
        with tp.no_grad():
            for images, _ in tensorplay_batches:
                compiled(images)

    def collect_compiled(
        compiled: Callable[..., object],
        batches: list[tuple[torch.Tensor, torch.Tensor]]
        | list[tuple[tp.Tensor, tp.Tensor]],
        framework: str,
    ) -> np.ndarray:
        outputs: list[np.ndarray] = []
        if framework == "torch":
            with torch.inference_mode():
                for images, _ in batches:
                    outputs.append(compiled(images).detach().cpu().numpy())
        else:
            with tp.no_grad():
                for images, _ in batches:
                    outputs.append(tensorplay_to_numpy(compiled(images)))
        return np.concatenate(outputs, axis=0)

    def one_framework(framework: str) -> dict[str, object]:
        codegens: list[str] = []
        if framework == "torch":
            model = torch_model
            batches = torch_batches
            compile_fn = lambda: torch.compile(
                model,
                backend=torch_backend,
                mode=compile_mode,
                fullgraph=True,
            )
            run = run_torch
        else:
            model = tensorplay_model
            batches = tensorplay_batches
            compile_fn = lambda: tp.compile(
                model,
                backend=tensorplay_backend,
                mode=compile_mode,
                # Python is only the capture frontend (as with Dynamo/FX).
                # The benchmark must reject a Python GraphModule fallback so
                # the compiled comparison measures a native executor.
                fullgraph=True,
                strict_native=True,
            )
            run = run_tensorplay

        try:
            model.eval()
            if framework == "torch":
                synchronize_torch(context)
            else:
                synchronize_tensorplay(context)
            started = time.perf_counter()
            compiled = compile_fn()
            wrapper_seconds = time.perf_counter() - started

            if framework == "torch":
                synchronize_torch(context)
            else:
                synchronize_tensorplay(context)
            started = time.perf_counter()
            run(compiled)
            if framework == "torch":
                synchronize_torch(context)
            else:
                synchronize_tensorplay(context)
            first_pass_seconds = time.perf_counter() - started

            if framework == "tensorplay":
                codegens = [
                    str(getattr(lowering, "_tensorplay_codegen", "unknown"))
                    for lowering in getattr(compiled, "_tensorplay_cache", {}).values()
                ]
                backward_codegens = [
                    str(getattr(lowering, "_tensorplay_backward_codegen", "eager-autograd"))
                    for lowering in getattr(compiled, "_tensorplay_cache", {}).values()
                ]
                if not codegens or any(
                    codegen not in {"stax-native", "stax-fused-cpu", "triton"}
                    for codegen in codegens
                ):
                    raise RuntimeError(
                        "TensorPlay compiled benchmark did not produce a native "
                        f"Stax/Triton executable: {codegens!r}"
                    )

            compiled_logits = collect_compiled(compiled, batches, framework)
            reference_logits = eager_logits[framework]
            max_abs_error = float(np.max(np.abs(reference_logits - compiled_logits)))
            prediction_match = bool(
                np.array_equal(
                    reference_logits.argmax(axis=1), compiled_logits.argmax(axis=1)
                )
            )
            logits_close = bool(
                np.allclose(reference_logits, compiled_logits, atol=5e-3, rtol=5e-3)
            )

            for _ in range(warmup):
                run(compiled)
                if framework == "torch":
                    synchronize_torch(context)
                else:
                    synchronize_tensorplay(context)

            times: list[float] = []
            for _ in range(repeats):
                if framework == "torch":
                    synchronize_torch(context)
                else:
                    synchronize_tensorplay(context)
                started = time.perf_counter()
                run(compiled)
                if framework == "torch":
                    synchronize_torch(context)
                else:
                    synchronize_tensorplay(context)
                times.append(time.perf_counter() - started)

            samples = sum(int(labels.shape[0]) for _, labels in batches)
            p50_seconds = float(np.percentile(times, 50))
            return {
                "available": True,
                "backend": torch_backend if framework == "torch" else tensorplay_backend,
                "mode": compile_mode,
                "wrapper_seconds": wrapper_seconds,
                "first_pass_seconds": first_pass_seconds,
                "compile_and_first_pass_seconds": wrapper_seconds + first_pass_seconds,
                "p50_seconds": p50_seconds,
                "p95_seconds": float(np.percentile(times, 95)),
                "images_per_second_p50": samples / p50_seconds,
                "samples": samples,
                "batch_count": len(batches),
                **({"codegen": codegens} if framework == "tensorplay" else {}),
                "max_abs_logit_error_vs_eager": max_abs_error,
                "logits_close_vs_eager": logits_close,
                "predictions_identical_vs_eager": prediction_match,
            }
        except Exception as exc:  # Keep one compiler failure visible in JSON.
            import traceback; traceback.print_exc()
            return {
                "available": False,
                "backend": torch_backend if framework == "torch" else tensorplay_backend,
                "mode": compile_mode,
                **({"codegen": []} if framework == "tensorplay" else {}),
                "error": f"{type(exc).__name__}: {exc}",
            }

    return {"torch": one_framework("torch"), "tensorplay": one_framework("tensorplay")}


def benchmark_compiled_training(
    torch_reference_model: torch.nn.Module,
    train_split: SampleSplit,
    epoch_orders: list[list[np.ndarray]],
    image_size: int,
    context: DeviceContext,
    lr: float,
    momentum: float,
    weight_decay: float,
    torch_backend: str,
    tensorplay_backend: str,
    compile_mode: str,
    optimizer_foreach: bool | None,
) -> dict[str, dict[str, object]]:
    """Compare ``torch.compile(model)`` and ``tp.compile(model)`` in training.

    This deliberately mirrors the PyTorch API boundary instead of compiling a
    hand-written training loop: the model is the only compiled object, while
    CrossEntropyLoss, ``backward()``, and the SGD update remain outside.  It
  keeps the API boundary visible: both frameworks compile the model, while
  TorchInductor and Stax each own a separate AOT forward/backward graph.
    """

    if not epoch_orders or not epoch_orders[0]:
        raise ValueError("compiled training requires at least one non-empty epoch")

    torch_state = torch_reference_model.state_dict()

    def run_epoch(
        compiled: Callable[..., Any],
        model: torch.nn.Module | tp_nn.Module,
        optimizer: Any,
        criterion: torch.nn.Module | tp_nn.Module,
        framework: str,
        batch_indices: list[np.ndarray],
    ) -> dict[str, float]:
        model.train()
        if framework == "torch":
            synchronize_torch(context)
        else:
            synchronize_tensorplay(context)
        started = time.perf_counter()
        loss_sum = 0.0
        correct = 0
        total = 0
        for images, labels in framework_batches(
            train_split, batch_indices, image_size, framework, context
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = compiled(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = int(labels.shape[0])
            loss_sum += float(loss.item()) * batch_size
            if framework == "torch":
                correct += int((logits.argmax(1) == labels).sum().item())
            else:
                correct += int(
                    (
                        tensorplay_to_numpy(logits.argmax(1))
                        == tensorplay_to_numpy(labels)
                    ).sum()
                )
            total += batch_size
        if framework == "torch":
            synchronize_torch(context)
        else:
            synchronize_tensorplay(context)
        elapsed = time.perf_counter() - started
        return {
            "seconds": elapsed,
            "loss": loss_sum / total,
            "accuracy": correct / total,
            "samples_per_second": total / elapsed,
        }

    def one_framework(framework: str) -> dict[str, object]:
        codegens: list[str] = []
        if framework == "torch":
            model = torchvision_resnet18(weights=None, num_classes=NUM_CLASSES).to(
                context.torch_device
            )
            model.load_state_dict(torch_state)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                foreach=optimizer_foreach,
            )
            compile_fn = lambda: torch.compile(
                model,
                backend=torch_backend,
                mode=compile_mode,
                fullgraph=True,
            )
        else:
            model = tensorplay_resnet18(num_classes=NUM_CLASSES).to(
                context.tensorplay_device
            )
            load_torch_state_into_tensorplay(model, torch_state, context.tensorplay_device)
            criterion = tp_nn.CrossEntropyLoss()
            optimizer = tp_optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                foreach=optimizer_foreach,
            )
            compile_fn = lambda: tp.compile(
                model,
                backend=tensorplay_backend,
                mode=compile_mode,
                fullgraph=True,
                strict_native=True,
            )

        try:
            model.train()
            if framework == "torch":
                synchronize_torch(context)
            else:
                synchronize_tensorplay(context)
            started = time.perf_counter()
            compiled = compile_fn()
            wrapper_seconds = time.perf_counter() - started

            first_batch = next(
                framework_batches(
                    train_split,
                    [epoch_orders[0][0]],
                    image_size,
                    framework,
                    context,
                )
            )
            if framework == "torch":
                synchronize_torch(context)
            else:
                synchronize_tensorplay(context)
            started = time.perf_counter()
            images, labels = first_batch
            optimizer.zero_grad(set_to_none=True)
            logits = compiled(images)
            loss = criterion(logits, labels)
            first_step_loss = float(loss.item())
            loss.backward()
            optimizer.step()
            if framework == "torch":
                synchronize_torch(context)
            else:
                synchronize_tensorplay(context)
            first_step_seconds = time.perf_counter() - started

            if framework == "tensorplay":
                codegens = [
                    str(getattr(lowering, "_tensorplay_codegen", "unknown"))
                    for lowering in getattr(compiled, "_tensorplay_cache", {}).values()
                ]
                backward_codegens = [
                    str(getattr(lowering, "_tensorplay_backward_codegen", "eager-autograd"))
                    for lowering in getattr(compiled, "_tensorplay_cache", {}).values()
                ]
                if not codegens or any(
                    codegen not in {
                        "stax-native",
                        "stax-aot-native",
                        "stax-fused-cpu",
                        "triton",
                    }
                    for codegen in codegens
                ):
                    raise RuntimeError(
                        "TensorPlay compiled training did not produce a native "
                        f"Stax/Triton executable: {codegens!r}"
                    )

            history = [
                run_epoch(
                    compiled,
                    model,
                    optimizer,
                    criterion,
                    framework,
                    batch_indices,
                )
                for batch_indices in epoch_orders
            ]
            total_seconds = sum(row["seconds"] for row in history)
            total_samples = len(epoch_orders) * len(train_split)
            return {
                "available": True,
                "backend": torch_backend if framework == "torch" else tensorplay_backend,
                "mode": compile_mode,
                "wrapper_seconds": wrapper_seconds,
                "first_step_seconds": first_step_seconds,
                "compile_and_first_step_seconds": wrapper_seconds + first_step_seconds,
                "first_step_loss": first_step_loss,
                "epochs": history,
                "seconds": total_seconds,
                "samples": total_samples,
                "samples_per_second": total_samples / total_seconds,
                "forward_codegen": (
                    codegens if framework == "tensorplay" else ["inductor"]
                ),
                "backward_codegen": (
                    backward_codegens
                    if framework == "tensorplay"
                    else ["inductor-aotautograd"]
                ),
                **({"codegen": codegens} if framework == "tensorplay" else {}),
            }
        except Exception as exc:
            import traceback; traceback.print_exc()
            return {
                "available": False,
                "backend": torch_backend if framework == "torch" else tensorplay_backend,
                "mode": compile_mode,
                **({"codegen": codegens} if framework == "tensorplay" else {}),
                "error": f"{type(exc).__name__}: {exc}",
            }

    return {"torch": one_framework("torch"), "tensorplay": one_framework("tensorplay")}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1] / "test" / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "all"),
        default="all",
        help="run on CPU, CUDA, or both (default: all)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer-foreach",
        choices=("default", "true", "false"),
        default="true",
        help=(
            "use the same Torch foreach dispatch on both optimizers; "
            "default=true, 'default' leaves Torch's device default unchanged"
        ),
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--torch-compile-backend", default="inductor")
    parser.add_argument("--tensorplay-compile-backend", default="stax")
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="default",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip the compile-vs-eager phase",
    )
    parser.add_argument(
        "--compiled-training-epochs",
        type=int,
        default=1,
        help="compiled-training epochs after eager training (default: 1; 0 skips)",
    )
    parser.add_argument("--logit-atol", type=float, default=5e-3)
    parser.add_argument("--logit-rtol", type=float, default=5e-3)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--no-fail-on-mismatch",
        action="store_true",
        help="report parity failures without returning a non-zero status",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1 because this benchmark trains")
    if args.image_size < 8 or args.batch_size < 1 or args.threads < 1:
        parser.error("image size, batch size, and threads must be positive")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be non-negative and --repeats must be positive")
    if args.compiled_training_epochs < 0:
        parser.error("--compiled-training-epochs must be non-negative")
    return args


def make_device_context(name: str) -> DeviceContext:
    if name == "cpu":
        return DeviceContext("cpu", torch.device("cpu"), tp.device("cpu"))
    if name != "cuda":
        raise ValueError(f"unknown device: {name}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available")
    if not tp.cuda.is_available():
        raise RuntimeError("TensorPlay CUDA is not available")
    if torch.cuda.device_count() < 1 or tp.cuda.device_count() < 1:
        raise RuntimeError("CUDA is reported available but no device was found")
    return DeviceContext("cuda", torch.device("cuda:0"), tp.device("cuda:0"))


def selected_device_contexts(requested: str) -> list[DeviceContext]:
    contexts = [make_device_context("cpu")] if requested in ("cpu", "all") else []
    if requested in ("cuda", "all"):
        try:
            contexts.append(make_device_context("cuda"))
        except RuntimeError as exc:
            if requested == "cuda":
                raise
            print(f"SKIP cuda: {exc}", file=sys.stderr)
    return contexts


def run_benchmark(
    args: argparse.Namespace,
    data_root: Path,
    train_split: SampleSplit,
    evaluate_split: SampleSplit,
    test_split: SampleSplit,
    context: DeviceContext,
) -> dict:
    """Run one complete CPU or CUDA comparison."""

    optimizer_foreach = {
        "default": None,
        "true": True,
        "false": False,
    }[args.optimizer_foreach]

    # Reset both frameworks before each device run.  The TensorPlay model is
    # then explicitly loaded from the Torch initialization, so initialization
    # differences cannot masquerade as training differences.
    torch.manual_seed(args.seed)
    if context.name == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    tp.manual_seed(args.seed)

    torch_model = torchvision_resnet18(weights=None, num_classes=NUM_CLASSES).to(
        context.torch_device
    )
    tensorplay_model = tensorplay_resnet18(num_classes=NUM_CLASSES).to(
        context.tensorplay_device
    )
    load_torch_state_into_tensorplay(
        tensorplay_model, torch_model.state_dict(), context.tensorplay_device
    )

    torch_criterion = torch.nn.CrossEntropyLoss()
    tensorplay_criterion = tp_nn.CrossEntropyLoss()
    torch_optimizer = torch.optim.SGD(
        torch_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        foreach=optimizer_foreach,
    )
    tensorplay_optimizer = tp_optim.SGD(
        tensorplay_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        foreach=optimizer_foreach,
    )

    epoch_orders = [
        make_epoch_indices(
            len(train_split),
            args.batch_size,
            args.seed,
            epoch,
            not args.no_shuffle,
        )
        for epoch in range(args.epochs)
    ]
    compiled_epoch_orders = [
        make_epoch_indices(
            len(train_split),
            args.batch_size,
            args.seed,
            epoch,
            not args.no_shuffle,
        )
        for epoch in range(args.compiled_training_epochs)
    ]

    print("=" * 78)
    print(f"TensorPlay / PyTorch ResNet-18 classification benchmark [{context.name}]")
    print(f"data_root={data_root}")
    print(
        f"splits=train:{len(train_split)}, evaluate:{len(evaluate_split)}, "
        f"test:{len(test_split)}; classes={CLASS_NAMES}"
    )
    print(
        f"image_size={args.image_size}, batch_size={args.batch_size}, "
        f"epochs={args.epochs}, threads={args.threads}, "
        f"shuffle={not args.no_shuffle}, optimizer_foreach={optimizer_foreach!r}"
    )
    print(
        f"torch_device={context.torch_device}, "
        f"tensorplay_device={context.tensorplay_device}"
    )
    if context.name == "cuda":
        print(f"GPU={torch.cuda.get_device_name(context.torch_device)}")
    print(f"TensorPlay threads: {tp.get_num_threads()}")
    print("=" * 78)

    torch_history: list[dict[str, float]] = []
    tensorplay_history: list[dict[str, float]] = []

    print("[1/6] Training PyTorch reference")
    for epoch, batch_indices in enumerate(epoch_orders, start=1):
        train_metrics = train_torch_epoch(
            torch_model,
            torch_optimizer,
            torch_criterion,
            train_split,
            batch_indices,
            args.image_size,
            context,
        )
        evaluate_metrics = evaluate_torch(
            torch_model,
            torch_criterion,
            evaluate_split,
            args.batch_size,
            args.image_size,
            context,
        )
        test_metrics = evaluate_torch(
            torch_model,
            torch_criterion,
            test_split,
            args.batch_size,
            args.image_size,
            context,
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "evaluate": evaluate_metrics,
            "test": test_metrics,
        }
        torch_history.append(row)
        print(
            f"  epoch {epoch:>2}: loss={train_metrics['loss']:.4f}, "
            f"train={train_metrics['accuracy']:.3%}, "
            f"evaluate={evaluate_metrics['accuracy']:.3%}, "
            f"test={test_metrics['accuracy']:.3%}, "
            f"{train_metrics['seconds']:.2f}s "
            f"({train_metrics['samples_per_second']:.1f} samples/s)"
        )

    print("[2/6] Training TensorPlay from the same initial state")
    for epoch, batch_indices in enumerate(epoch_orders, start=1):
        train_metrics = train_tensorplay_epoch(
            tensorplay_model,
            tensorplay_optimizer,
            tensorplay_criterion,
            train_split,
            batch_indices,
            args.image_size,
            context,
        )
        evaluate_metrics = evaluate_tensorplay(
            tensorplay_model,
            tensorplay_criterion,
            evaluate_split,
            args.batch_size,
            args.image_size,
            context,
        )
        test_metrics = evaluate_tensorplay(
            tensorplay_model,
            tensorplay_criterion,
            test_split,
            args.batch_size,
            args.image_size,
            context,
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "evaluate": evaluate_metrics,
            "test": test_metrics,
        }
        tensorplay_history.append(row)
        print(
            f"  epoch {epoch:>2}: loss={train_metrics['loss']:.4f}, "
            f"train={train_metrics['accuracy']:.3%}, "
            f"evaluate={evaluate_metrics['accuracy']:.3%}, "
            f"test={test_metrics['accuracy']:.3%}, "
            f"{train_metrics['seconds']:.2f}s "
            f"({train_metrics['samples_per_second']:.1f} samples/s)"
        )

    print("[3/6] Strict Torch-checkpoint -> TensorPlay parity")
    strict_tensorplay_model = tensorplay_resnet18(num_classes=NUM_CLASSES).to(
        context.tensorplay_device
    )
    load_torch_state_into_tensorplay(
        strict_tensorplay_model, torch_model.state_dict(), context.tensorplay_device
    )
    torch_batches, tensorplay_batches = prepared_test_batches(
        test_split, args.batch_size, args.image_size, context
    )
    torch_logits, labels = collect_logits(torch_model, torch_batches, "torch")
    tensorplay_logits, tensorplay_labels = collect_logits(
        strict_tensorplay_model, tensorplay_batches, "tensorplay"
    )
    max_abs_logit_error = float(np.max(np.abs(torch_logits - tensorplay_logits)))
    max_rel_logit_error = float(
        np.max(
            np.abs(torch_logits - tensorplay_logits)
            / np.maximum(np.abs(torch_logits), np.float32(1e-8))
        )
    )
    torch_predictions = torch_logits.argmax(axis=1)
    tensorplay_predictions = tensorplay_logits.argmax(axis=1)
    torch_accuracy = float(np.mean(torch_predictions == labels))
    tensorplay_accuracy = float(np.mean(tensorplay_predictions == tensorplay_labels))
    prediction_match = bool(np.array_equal(torch_predictions, tensorplay_predictions))
    label_match = bool(np.array_equal(labels, tensorplay_labels))
    logits_close = bool(
        np.allclose(
            torch_logits,
            tensorplay_logits,
            atol=args.logit_atol,
            rtol=args.logit_rtol,
        )
    )
    print(f"  max_abs_logit_error={max_abs_logit_error:.6g}")
    print(f"  max_rel_logit_error={max_rel_logit_error:.6g}")
    print(f"  predictions_identical={prediction_match}; labels_identical={label_match}")
    print(
        f"  Torch test accuracy={torch_accuracy:.3%}; "
        f"TensorPlay transferred accuracy={tensorplay_accuracy:.3%}"
    )

    print("[4/6] Inference performance on the same transferred weights")
    inference = benchmark_inference(
        torch_model,
        strict_tensorplay_model,
        torch_batches,
        tensorplay_batches,
        args.warmup,
        args.repeats,
        context,
    )
    for framework in ("torch", "tensorplay"):
        metrics = inference[framework]
        print(
            f"  {framework:>10}: p50={metrics['p50_seconds'] * 1000:.2f} ms, "
            f"p95={metrics['p95_seconds'] * 1000:.2f} ms, "
            f"throughput={metrics['images_per_second_p50']:.1f} images/s"
        )
    throughput_ratio = (
        inference["tensorplay"]["images_per_second_p50"]
        / inference["torch"]["images_per_second_p50"]
    )
    latency_ratio = (
        inference["tensorplay"]["p50_seconds"]
        / inference["torch"]["p50_seconds"]
    )
    print(f"  TensorPlay/PyTorch latency ratio={latency_ratio:.3f}x (target < 1)")

    compiled_training: dict[str, dict[str, object]] | None = None
    compiled_training_ok = True
    compiled_inference: dict[str, dict[str, object]] | None = None
    compile_ok = True
    if args.no_compile:
        print("[5/6] Compiled training/inference comparison skipped (--no-compile)")
    else:
        if args.compiled_training_epochs:
            print("[5/6] Compiled training: TorchInductor/AOTAutograd contract")
            compiled_training = benchmark_compiled_training(
                torch_model,
                train_split,
                compiled_epoch_orders,
                args.image_size,
                context,
                args.lr,
                args.momentum,
                args.weight_decay,
                args.torch_compile_backend,
                args.tensorplay_compile_backend,
                args.compile_mode,
                optimizer_foreach,
            )
            training_available = True
            for framework in ("torch", "tensorplay"):
                metrics = compiled_training[framework]
                if not metrics["available"]:
                    training_available = False
                    print(f"  {framework:>10}: unavailable ({metrics['error']})")
                    continue
                print(
                    f"  {framework:>10}: wrapper={metrics['wrapper_seconds']:.3f}s, "
                    f"first_step={metrics['first_step_seconds']:.3f}s, "
                    f"steady={metrics['seconds']:.3f}s, "
                    f"throughput={metrics['samples_per_second']:.1f} samples/s, "
                    f"forward={metrics['forward_codegen']}, "
                    f"backward={metrics['backward_codegen']}"
                )
            if training_available:
                torch_first_loss = float(compiled_training["torch"]["first_step_loss"])
                tensorplay_first_loss = float(
                    compiled_training["tensorplay"]["first_step_loss"]
                )
                first_loss_close = bool(
                    np.isclose(
                        torch_first_loss,
                        tensorplay_first_loss,
                        atol=5e-3,
                        rtol=5e-3,
                    )
                )
                compiled_training_ok = first_loss_close
                print(
                    f"  first_step_loss: Torch={torch_first_loss:.6g}, "
                    f"TensorPlay={tensorplay_first_loss:.6g}, "
                    f"close={first_loss_close}"
                )
            else:
                compiled_training_ok = False
        else:
            print("[5/6] Compiled training skipped (--compiled-training-epochs 0)")

        compile_ok = compiled_training_ok
        print("[6/6] Compiled inference: first-call cost and steady state")
        compiled_inference = benchmark_compiled_inference(
            torch_model,
            strict_tensorplay_model,
            torch_batches,
            tensorplay_batches,
            {"torch": torch_logits, "tensorplay": tensorplay_logits},
            args.warmup,
            args.repeats,
            context,
            args.torch_compile_backend,
            args.tensorplay_compile_backend,
            args.compile_mode,
        )
        for framework in ("torch", "tensorplay"):
            metrics = compiled_inference[framework]
            if not metrics["available"]:
                compile_ok = False
                print(f"  {framework:>10}: unavailable ({metrics['error']})")
                continue
            compile_ok = compile_ok and bool(
                metrics["logits_close_vs_eager"]
                and metrics["predictions_identical_vs_eager"]
            )
            eager_throughput = inference[framework]["images_per_second_p50"]
            compiled_throughput = metrics["images_per_second_p50"]
            codegen_text = (
                f"codegen={metrics['codegen']}, " if framework == "tensorplay" else ""
            )
            print(
                f"  {framework:>10}: wrapper={metrics['wrapper_seconds']:.3f}s, "
                f"first_pass={metrics['first_pass_seconds']:.3f}s, "
                f"p50={metrics['p50_seconds'] * 1000:.2f} ms, "
                f"throughput={compiled_throughput:.1f} images/s, "
                f"compiled/eager={compiled_throughput / eager_throughput:.3f}x, "
                f"{codegen_text}"
                f"max_abs_vs_eager={metrics['max_abs_logit_error_vs_eager']:.6g}"
            )
        if (
            compiled_inference["torch"]["available"]
            and compiled_inference["tensorplay"]["available"]
        ):
            compiled_latency_ratio = (
                compiled_inference["tensorplay"]["p50_seconds"]
                / compiled_inference["torch"]["p50_seconds"]
            )
            print(
                "  TensorPlay/PyTorch compiled inference latency ratio="
                f"{compiled_latency_ratio:.3f}x (target < 1)"
            )
        if compile_ok:
            print("  PASS: compiled training and inference checks passed")
        else:
            print("  FAIL: compiled training or inference phase was unavailable or changed outputs")

    result = {
        "config": {
            "data_root": str(data_root),
            "device": context.name,
            "torch_device": str(context.torch_device),
            "tensorplay_device": str(context.tensorplay_device),
            "epochs": args.epochs,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "optimizer_foreach": optimizer_foreach,
            "threads": args.threads,
            "seed": args.seed,
            "shuffle": not args.no_shuffle,
            "torch_compile_backend": args.torch_compile_backend,
            "tensorplay_compile_backend": args.tensorplay_compile_backend,
            "compile_mode": args.compile_mode,
            "compile_enabled": not args.no_compile,
            "compiled_training_epochs": args.compiled_training_epochs,
        },
        "splits": {
            "train": len(train_split),
            "evaluate": len(evaluate_split),
            "test": len(test_split),
        },
        "torch_training": torch_history,
        "tensorplay_training": tensorplay_history,
        "strict_parity": {
            "max_abs_logit_error": max_abs_logit_error,
            "max_rel_logit_error": max_rel_logit_error,
            "logits_close": logits_close,
            "predictions_identical": prediction_match,
            "labels_identical": label_match,
            "torch_test_accuracy": torch_accuracy,
            "tensorplay_transferred_test_accuracy": tensorplay_accuracy,
        },
        "inference": inference,
        "throughput_ratio_tensorplay_over_torch": throughput_ratio,
        "latency_ratio_tensorplay_over_torch": latency_ratio,
        "compiled_training": compiled_training,
        "compiled_inference": compiled_inference,
    }
    if (
        compiled_inference is not None
        and compiled_inference["torch"]["available"]
        and compiled_inference["tensorplay"]["available"]
    ):
        result["compiled_inference_latency_ratio_tensorplay_over_torch"] = (
            compiled_inference["tensorplay"]["p50_seconds"]
            / compiled_inference["torch"]["p50_seconds"]
        )
    result["compile_ok"] = compile_ok
    result["compiled_training_ok"] = compiled_training_ok
    parity_ok = logits_close and prediction_match and label_match
    if not parity_ok or not compile_ok:
        if parity_ok and not compile_ok:
            message = "FAIL: compiled Torch/TensorPlay ResNet comparison did not pass"
        else:
            message = "FAIL: strict Torch/TensorPlay ResNet parity check did not pass"
        print(message, file=sys.stderr)
        result["parity_ok"] = False
        return result
    print("PASS: strict accuracy/logit parity check passed")
    result["parity_ok"] = True
    return result


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    train_split = read_split(data_root, "train")
    evaluate_split = read_split(data_root, "evaluate")
    test_split = read_split(data_root, "test")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(max(1, min(args.threads, 4)))
    tp.set_num_threads(args.threads)
    contexts = selected_device_contexts(args.device)
    if not contexts:
        print("No requested device is available", file=sys.stderr)
        return 1

    results: dict[str, dict] = {}
    for context in contexts:
        results[context.name] = run_benchmark(
            args,
            data_root,
            train_split,
            evaluate_split,
            test_split,
            context,
        )
        if context.name == "cuda":
            tp.cuda.empty_cache()
            torch.cuda.empty_cache()

    output = {
        "config": {
            "data_root": str(data_root),
            "device_request": args.device,
            "epochs": args.epochs,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "threads": args.threads,
            "seed": args.seed,
            "shuffle": not args.no_shuffle,
        },
        "splits": {
            "train": len(train_split),
            "evaluate": len(evaluate_split),
            "test": len(test_split),
        },
        "devices": results,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")

    parity_ok = all(
        result["parity_ok"] and result["compile_ok"]
        for result in results.values()
    )
    if not parity_ok:
        print(
            "FAIL: Torch/TensorPlay ResNet parity or compile comparison did not pass",
            file=sys.stderr,
        )
        return 0 if args.no_fail_on_mismatch else 1
    print("PASS: strict accuracy/logit parity check passed on all requested devices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
