#!/usr/bin/env python3
"""Generate 01-architecture four pillars diagram."""
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="01_arch.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    fig.patch.set_facecolor("#fbfbfd"); ax.set_facecolor("#fbfbfd")

    # foundation
    ax.add_patch(patches.FancyBboxPatch((1, 0.5), 10, 0.6, boxstyle="round,pad=0.08", facecolor="#e8eaf6", edgecolor="#3949ab", lw=1.2))
    ax.text(6, 0.8, "CMake + codegen  ( native_functions.yaml  ->  10 artifacts )", ha="center", va="center", fontsize=8, color="#283593")

    pillars = [
        (1.5, "p10\nTensor / Storage\nDispatcher / Kernels", "#e3f2fd", "#1565c0"),
        (4.2, "tpx\nAutograd DAG\nEngine / Node", "#fff3e0", "#ef6c00"),
        (6.9, "stax\nGraph IR\nPass / Interpreter", "#f3e5f5", "#6a1b9a"),
        (9.2, "_C  +  tp_python", "#e8f5e9", "#2e7d32"),
    ]
    for x, label, bg, fg in pillars:
        ax.add_patch(patches.FancyBboxPatch((x, 1.4), 1.8, 2.6, boxstyle="round,pad=0.08", facecolor=bg, edgecolor=fg, lw=1.4))
        ax.text(x+0.9, 2.7, label, ha="center", va="center", fontsize=8, color=fg, weight="bold", linespacing=1.4)

    # dependency arrows (downward only)
    for sx, ex in [(2.4, 5.1), (5.1, 7.8), (7.8, 10.1)]:
        ax.annotate("", xy=(ex, 2.0), xytext=(sx+1.8, 2.0), arrowprops=dict(arrowstyle="->", color="#78909c", lw=1.2, connectionstyle="arc3,rad=0.12"))
    ax.text(6, 1.15, "dependency flows down only  p10 -> tpx -> stax -> _C  (acyclic)", ha="center", fontsize=7, color="#546e7a", style="italic")

    # Python roof
    ax.add_patch(patches.FancyBboxPatch((1, 4.6), 10, 0.8, boxstyle="round,pad=0.08", facecolor="#fffde7", edgecolor="#f9a825", lw=1.4))
    ax.text(6, 5.0, 'Python glue  tensorplay/__init__.py  ·  nn / optim / autograd / compiler / graph', ha="center", fontsize=8, color="#f57f17", weight="bold")
    for x in [2.4, 5.1, 7.8, 10.1]:
        ax.annotate("", xy=(x, 4.6), xytext=(x, 4.0), arrowprops=dict(arrowstyle="->", color="#f9a825", lw=1.1, ls="--"))

    ax.set_title("TensorPlay Four-Pillar Architecture — Physical Isolation for Readability", fontsize=12, weight="bold", color="#212121", pad=12)
    plt.tight_layout(); plt.savefig(args.out, dpi=220, bbox_inches="tight"); print(f"saved to {args.out}")

if __name__ == "__main__":
    main()
