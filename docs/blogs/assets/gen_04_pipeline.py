#!/usr/bin/env python3
import argparse, matplotlib.pyplot as plt, matplotlib.patches as patches
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out", default="04_pipeline.png"); args=ap.parse_args()
    fig, ax=plt.subplots(figsize=(12,3.2)); ax.set_xlim(0,12); ax.set_ylim(0,3); ax.axis("off")
    fig.patch.set_facecolor("#fbfbfd"); ax.set_facecolor("#fbfbfd")
    steps=["populate\noperands","compute\nshape\n(broadcast)","compute\ntypes\n(promote)","compute\nstrides","reorder\ndims","coalesce\ndims","allocate\noutput"]
    for i, s in enumerate(steps):
        x=0.4+i*1.62; col="#e3f2fd" if i<3 else "#fff3e0" if i<6 else "#e8f5e9"; ec="#1565c0" if i<3 else "#ef6c00" if i<6 else "#2e7d32"
        ax.add_patch(patches.FancyBboxPatch((x,0.7),1.4,1.5, boxstyle="round,pad=0.06", facecolor=col, edgecolor=ec, lw=1.1))
        ax.text(x+0.7,1.45, s, ha="center", va="center", fontsize=6.5, color=ec, weight="bold", linespacing=1.2)
        ax.text(x+0.7,0.85, f"step {i+1}", ha="center", fontsize=5, color="#78909c")
        if i<6:
            ax.annotate("", xy=(x+1.4,1.45), xytext=(x+1.55,1.45), arrowprops=dict(arrowstyle="->", color="#78909c", lw=1.1))
    # fast path
    ax.annotate("", xy=(9.8,2.5), xytext=(1.8,2.5), arrowprops=dict(arrowstyle="->", color="#4caf50", lw=1, ls="--", connectionstyle="arc3,rad=0.18"))
    ax.text(6,2.75, "fast path: all same shape & contiguous -> collapse to 1D", ha="center", fontsize=6, color="#2e7d32", style="italic")
    ax.set_title("TensorIterator Build Pipeline: 7 Steps from Tensors to parallel_for", fontsize=11, weight="bold", color="#212121")
    plt.tight_layout(); plt.savefig(args.out, dpi=220, bbox_inches="tight"); print(f"saved to {args.out}")
if __name__=="__main__": main()
