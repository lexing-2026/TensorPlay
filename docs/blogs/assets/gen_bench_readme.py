#!/usr/bin/env python3
"""Generate README benchmark overview chart (data: docs/whitepaper section 9)."""
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark_readme.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.patch.set_facecolor("#fbfbfd")

    # -- panel 1: op-level relative throughput spread (whitepaper fig:benchhist) --
    ax = axes[0]
    ax.barh(
        ["lagging\n(>20% behind)", "within 1.5x", "leading\n(>=1.5x)"],
        [6, 24, 28],
        color=["#ef9a9a", "#cfd8dc", "#a5d6a7"],
        edgecolor="#455a64",
        lw=1.0,
        height=0.55,
    )
    for y, v in enumerate([6, 24, 28]):
        ax.text(v + 0.4, y, str(v), va="center", fontsize=10, color="#263238", weight="bold")
    ax.set_xlim(0, 34)
    ax.set_xlabel("op / shape pairs (of 58, 8 threads)", fontsize=9)
    ax.set_title("CPU per-op throughput vs reference", fontsize=11, weight="bold", color="#212121")
    ax.text(
        0.02, -0.34,
        "wins cluster by kernel file: exp 2.4x, gelu 2.3x (1M),\n"
        "sqrt 10.8x, norm2 9.9x on scalar-loop shapes",
        transform=ax.transAxes, fontsize=7.5, color="#546e7a",
    )

    # -- panel 2: backend op coverage (whitepaper tab:backends) --
    ax = axes[1]
    backends = ["CPU", "CUDA", "Vulkan"]
    ops = [1325, 1274, 145]
    colors = ["#1565c0", "#ef6c00", "#6a1b9a"]
    bars = ax.bar(backends, ops, color=colors, width=0.55, edgecolor="#37474f", lw=1.0)
    for b, v in zip(bars, ops):
        ax.text(b.get_x() + b.get_width() / 2, v + 28, f"{v:,}", ha="center", fontsize=10, weight="bold", color="#263238")
    ax.set_ylim(0, 1560)
    ax.set_ylabel("unique ops", fontsize=9)
    ax.set_title("Backend op coverage", fontsize=11, weight="bold", color="#212121")
    ax.text(0.98, 0.86, "CUDA = 96% of CPU surface\nVulkan = teaching backend (4.5k GLSL)",
            transform=ax.transAxes, ha="right", fontsize=7.5, color="#546e7a")

    # -- panel 3: dispatch overhead (whitepaper microbench) --
    ax = axes[2]
    labels = ["C++ dispatch", "binding + runtime\n(eager op total ~890 ns)"]
    vals = [4, 886]
    ax.barh(labels, vals, color=["#a5d6a7", "#cfd8dc"], edgecolor="#455a64", lw=1.0, height=0.5)
    ax.text(10, 0, "~4 ns  (<0.5%)", va="center", fontsize=10, weight="bold", color="#1b5e20")
    ax.text(893, 1, "~890 ns", va="center", fontsize=10, color="#263238", weight="bold")
    ax.set_xlim(0, 1400)
    ax.set_xlabel("host time per eager op (ns)", fontsize=9)
    ax.set_title("Where the time goes", fontsize=11, weight="bold", color="#212121")
    ax.text(
        0.02, -0.34,
        "dispatcher adds the same sub-1% sliver on CPU, CUDA\nand Vulkan paths — stax batching removes ~0.9 ms/step",
        transform=ax.transAxes, fontsize=7.5, color="#546e7a",
    )

    fig.suptitle("TensorPlay Benchmarks — readability without a runtime tax", fontsize=13, weight="bold", color="#212121", y=1.02)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
