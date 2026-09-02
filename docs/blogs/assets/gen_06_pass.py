#!/usr/bin/env python3
import argparse, matplotlib.pyplot as plt, matplotlib.patches as patches
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out", default="06_pass.png"); args=ap.parse_args()
    fig, ax=plt.subplots(figsize=(11,3.8)); ax.set_xlim(0,11); ax.set_ylim(0,3.5); ax.axis("off")
    fig.patch.set_facecolor("#fbfbfd"); ax.set_facecolor("#fbfbfd")
    # Before
    ax.text(2.0,3.0, "Before (dirty FX Graph)", ha="center", fontsize=7, weight="bold", color="#37474f")
    ax.add_patch(patches.FancyBboxPatch((0.5,1.0),3.0,1.6, boxstyle="round,pad=0.06", facecolor="#fff3e0", edgecolor="#ef6c00", lw=1))
    ax.text(2.0,2.2, "x  -> mul(2) -> add(1)\n     -> mul(2) dup\n2+3 const\n  dead node", ha="center", fontsize=6, color="#4e342e", linespacing=1.25)
    ax.text(2.0,1.3, "nodes: 6, edges messy", ha="center", fontsize=5, color="#8d6e63", style="italic")
    # PassManager
    ax.add_patch(patches.FancyBboxPatch((4.2,1.1),2.6,1.4, boxstyle="round,pad=0.06", facecolor="#e3f2fd", edgecolor="#1565c0", lw=1.2))
    ax.text(5.5,2.15, "PassManager", ha="center", fontsize=7, weight="bold", color="#0d47a1")
    ax.text(5.5,1.85, "ConstFold", ha="center", fontsize=5, color="#1565c0")
    ax.text(5.5,1.65, "CSE / DCE", ha="center", fontsize=5, color="#1565c0")
    ax.text(5.5,1.45, "PointwiseFusionHint", ha="center", fontsize=5, color="#1565c0")
    ax.text(5.5,1.25, "Normalize / ShapeProp", ha="center", fontsize=5, color="#1565c0")
    ax.annotate("", xy=(4.2,1.8), xytext=(3.5,1.8), arrowprops=dict(arrowstyle="->", color="#78909c", lw=1.2))
    # After
    ax.text(8.8,3.0, "After (clean + segmented)", ha="center", fontsize=7, weight="bold", color="#37474f")
    ax.add_patch(patches.FancyBboxPatch((7.2,1.0),3.4,1.6, boxstyle="round,pad=0.06", facecolor="#e8f5e9", edgecolor="#2e7d32", lw=1.1))
    ax.text(8.9,2.2, "fused group: mul+add+relu\n  (single kernel)\nconst 5, dead gone", ha="center", fontsize=6, color="#1b5e20", linespacing=1.25)
    ax.text(8.9,1.3, "nodes: 3, 1 fused group", ha="center", fontsize=5, color="#388e3c", style="italic")
    ax.annotate("", xy=(7.2,1.8), xytext=(6.8,1.8), arrowprops=dict(arrowstyle="->", color="#78909c", lw=1.2))
    # highlight fused
    ax.add_patch(patches.Rectangle((7.4,1.6),3.0,0.7, fill=False, edgecolor="#00c853", lw=1.2, ls="--"))
    ax.set_title("PassManager Washing: ConstFold + CSE + DCE + FusionHint", fontsize=11, weight="bold", color="#212121")
    plt.tight_layout(); plt.savefig(args.out, dpi=220, bbox_inches="tight"); print(f"saved to {args.out}")
if __name__=="__main__": main()
