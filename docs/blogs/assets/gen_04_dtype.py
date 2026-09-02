#!/usr/bin/env python3
import argparse, matplotlib.pyplot as plt, matplotlib.patches as patches
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out", default="04_dtype.png"); args=ap.parse_args()
    fig, ax=plt.subplots(figsize=(11,4.2)); ax.set_xlim(0,11); ax.set_ylim(0,4); ax.axis("off")
    fig.patch.set_facecolor("#fbfbfd"); ax.set_facecolor("#fbfbfd")
    # tiers
    tiers=[
        (3.0, "Complex", ["ComplexDouble","ComplexFloat","ComplexHalf"], "#fce4ec", "#ad1457"),
        (2.1, "Float", ["Float64","Float32","BFloat16 / Float16"], "#e3f2fd", "#1565c0"),
        (1.2, "Int", ["Int64 > Int32 > Int16 > Int8","UInt8"], "#e8f5e9", "#2e7d32"),
        (0.5, "Bool", ["Bool"], "#fffde7", "#f57f17"),
    ]
    for y, label, items, bg, fg in tiers:
        ax.add_patch(patches.FancyBboxPatch((0.5,y),1.2,0.45, boxstyle="round,pad=0.04", facecolor=bg, edgecolor=fg, lw=1))
        ax.text(1.1,y+0.22, label, ha="center", fontsize=7, weight="bold", color=fg)
        for i, it in enumerate(items):
            x=2.0+i*2.6; ax.add_patch(patches.FancyBboxPatch((x,y),2.4,0.45, boxstyle="round,pad=0.04", facecolor="white", edgecolor=fg, lw=1))
            ax.text(x+1.2,y+0.22, it, ha="center", fontsize=6, color=fg)
    # arrows promotion
    ax.annotate("", xy=(5.5,3.0), xytext=(5.5,2.55), arrowprops=dict(arrowstyle="->", color="#ad1457", lw=1.2))
    ax.annotate("", xy=(5.5,2.1), xytext=(5.5,1.65), arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.2))
    ax.annotate("", xy=(5.5,1.2), xytext=(5.5,0.95), arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.2))
    ax.text(7.5,2.6, "promote direction", fontsize=6, color="#546e7a", style="italic")
    # special
    ax.add_patch(patches.FancyBboxPatch((7.5,1.9),3.0,0.6, boxstyle="round,pad=0.06", facecolor="#fff3e0", edgecolor="#ef6c00", lw=1.1))
    ax.text(9.0,2.25, "Float16 + BFloat16 -> Float32", ha="center", fontsize=6, weight="bold", color="#e65100", family="monospace")
    ax.text(9.0,1.98, "avoid precision loss", ha="center", fontsize=5, color="#8d6e63", style="italic")
    ax.set_title("DType Promotion Lattice (TypePromotion.h:12) — Bool < Int < Float < Complex", fontsize=11, weight="bold", color="#212121")
    plt.tight_layout(); plt.savefig(args.out, dpi=220, bbox_inches="tight"); print(f"saved to {args.out}")
if __name__=="__main__": main()
