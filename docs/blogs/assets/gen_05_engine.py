#!/usr/bin/env python3
import argparse, matplotlib.pyplot as plt, matplotlib.patches as patches
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out", default="05_engine.png"); args=ap.parse_args()
    fig, ax=plt.subplots(figsize=(11,3.6)); ax.set_xlim(0,11); ax.set_ylim(0,3.4); ax.axis("off")
    fig.patch.set_facecolor("#fbfbfd"); ax.set_facecolor("#fbfbfd")
    steps=[
        (0.4, "Root(s)\nGraphRoot\nif multi-root", "#e3f2fd", "#1565c0"),
        (2.6, "compute_\ndependencies\nBFS + min_topo", "#fff3e0", "#ef6c00"),
        (4.8, "Enqueue\nroot grad\nto CPU queue", "#e8f5e9", "#2e7d32"),
        (7.0, "Evaluate\nnode.apply()\nInputBuffer\naccumulate", "#fce4ec", "#ad1457"),
        (9.2, "Done?\noutstanding\n==0 -> return", "#f3e5f5", "#6a1b9a"),
    ]
    for x, txt, bg, fg in steps:
        ax.add_patch(patches.FancyBboxPatch((x,1.1),1.6,1.1, boxstyle="round,pad=0.06", facecolor=bg, edgecolor=fg, lw=1.1))
        ax.text(x+0.8,1.65, txt, ha="center", va="center", fontsize=6.5, color=fg, weight="bold", linespacing=1.25)
    for sx, ex in [(2.0,2.6),(4.2,4.8),(6.4,7.0),(8.6,9.2)]:
        ax.annotate("", xy=(ex,1.65), xytext=(sx,1.65), arrowprops=dict(arrowstyle="->", color="#78909c", lw=1.2))
    # annotations
    ax.text(3.4,2.45, "Engine.cpp:621 execute", ha="center", fontsize=5, color="#546e7a", family="monospace")
    ax.text(7.8,2.45, "sum_to_shape + dtype cast", ha="center", fontsize=5, color="#ad1457", style="italic")
    ax.text(5.5,0.7, "CPU queue: priority by sequence_nr (max-heap) · CUDA queues: per-device worker · nested_depth local_queue", ha="center", fontsize=6, color="#37474f", style="italic")
    ax.set_title("Engine Execute Flow: From Roots to ReadyQueues to Completion", fontsize=11, weight="bold", color="#212121")
    plt.tight_layout(); plt.savefig(args.out, dpi=220, bbox_inches="tight"); print(f"saved to {args.out}")
if __name__=="__main__": main()
