#!/usr/bin/env python3
import argparse, matplotlib.pyplot as plt, matplotlib.patches as patches
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out", default="02_view.png"); args=ap.parse_args()
    fig, ax=plt.subplots(figsize=(11,4.2)); ax.set_xlim(0,11); ax.set_ylim(0,4); ax.axis("off")
    fig.patch.set_facecolor("#fbfbfd"); ax.set_facecolor("#fbfbfd")
    # storage block
    ax.add_patch(patches.FancyBboxPatch((1,0.4),9,0.7, boxstyle="round,pad=0.08", facecolor="#fff3e0", edgecolor="#ef6c00", lw=1.3))
    ax.text(5.5,0.9, "Shared Storage  (6 elements, contiguous bytes)", ha="center", fontsize=7, color="#e65100", weight="bold")
    for i in range(6):
        x=2.2+i*1.05; ax.add_patch(patches.Rectangle((x,0.55),0.85,0.35, facecolor="white", edgecolor="#ffb74d", lw=1))
        ax.text(x+0.42,0.72, str(i), ha="center", fontsize=6, color="#4e342e")
    # TensorImpl a
    ax.add_patch(patches.FancyBboxPatch((0.7,1.7),4.2,1.2, boxstyle="round,pad=0.08", facecolor="#e3f2fd", edgecolor="#1565c0", lw=1.3))
    ax.text(2.8,2.65, "Tensor a", ha="center", fontsize=8, weight="bold", color="#0d47a1")
    ax.text(2.8,2.35, "shape [2, 3]  stride [3, 1]  offset 0", ha="center", fontsize=6, family="monospace", color="#1565c0")
    ax.text(2.8,2.05, "is_view=false  version -> counter #42", ha="center", fontsize=6, color="#37474f")
    # TensorImpl b
    ax.add_patch(patches.FancyBboxPatch((6.1,1.7),4.2,1.2, boxstyle="round,pad=0.08", facecolor="#e8f5e9", edgecolor="#2e7d32", lw=1.3))
    ax.text(8.2,2.65, "View b = a.view([3,2])", ha="center", fontsize=8, weight="bold", color="#1b5e20")
    ax.text(8.2,2.35, "shape [3, 2]  stride [2, 1]  offset 0", ha="center", fontsize=6, family="monospace", color="#2e7d32")
    ax.text(8.2,2.05, "is_view=true  version -> SAME #42", ha="center", fontsize=6, color="#37474f")
    # arrows to storage
    ax.annotate("", xy=(4.5,1.1), xytext=(2.8,1.7), arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.2, connectionstyle="arc3,rad=-0.2"))
    ax.annotate("", xy=(6.5,1.1), xytext=(8.2,1.7), arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.2, connectionstyle="arc3,rad=0.2"))
    # shared version
    ax.add_patch(patches.FancyBboxPatch((4.3,2.9),2.4,0.35, boxstyle="round,pad=0.04", facecolor="#fffde7", edgecolor="#f9a825", lw=1))
    ax.text(5.5,3.08, "share_version_counter", ha="center", fontsize=6, color="#f57f17", family="monospace")
    ax.plot([4.3,2.8],[2.9,2.9], color="#f9a825", lw=1, ls="--"); ax.plot([6.7,8.2],[2.9,2.9], color="#f9a825", lw=1, ls="--")
    ax.text(5.5,1.45, "as_strided reuses storage + shares version; b[0,0]=99 mutates a[0,0]", ha="center", fontsize=6, color="#546e7a", style="italic")
    ax.set_title("View Zero-Copy: Two Tags, One Warehouse, Shared Version", fontsize=11, weight="bold", color="#212121")
    plt.tight_layout(); plt.savefig(args.out, dpi=220, bbox_inches="tight"); print(f"saved to {args.out}")
if __name__=="__main__": main()
