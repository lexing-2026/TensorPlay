#!/usr/bin/env python3
import argparse, matplotlib.pyplot as plt, matplotlib.patches as patches
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out", default="02_version.png"); args=ap.parse_args()
    fig, ax=plt.subplots(figsize=(11,3.6)); ax.set_xlim(0,11); ax.set_ylim(0,3.2); ax.axis("off")
    fig.patch.set_facecolor("#fbfbfd"); ax.set_facecolor("#fbfbfd")
    steps=[
        (0.5, "Save\nx (v0) ->\nSavedVariable\n(saved v0)", "#e3f2fd", "#1565c0"),
        (3.0, "Forward\ny = x*2\nNode holds\nSavedVariable(v0)", "#fff3e0", "#e65100"),
        (5.5, "Inplace\nx.add_(1)\nbump_version()\nx now v1", "#fce4ec", "#ad1457"),
        (8.0, "Backward\ny.backward()\nunpack checks\nv0 vs v1 -> ERROR", "#ffebee", "#c62828"),
    ]
    for x, txt, bg, fg in steps:
        ax.add_patch(patches.FancyBboxPatch((x,0.7),2.0,1.5, boxstyle="round,pad=0.08", facecolor=bg, edgecolor=fg, lw=1.2))
        ax.text(x+1.0,1.45, txt, ha="center", va="center", fontsize=7, color=fg, weight="bold", linespacing=1.3)
    for sx, ex in [(2.5,3.0),(5.0,5.5),(7.5,8.0)]:
        ax.annotate("", xy=(ex,1.45), xytext=(sx,1.45), arrowprops=dict(arrowstyle="->", color="#78909c", lw=1.3))
    ax.text(9.2,2.55, "Error: modified by\nan inplace op", ha="center", fontsize=6, color="#c62828", weight="bold")
    ax.text(5.5,2.6, "Without check -> silent wrong grad", ha="center", fontsize=6, color="#546e7a", style="italic")
    ax.set_title("Version Guard Timeline: How SavedVariable Catches Silent Errors", fontsize=11, weight="bold", color="#212121")
    plt.tight_layout(); plt.savefig(args.out, dpi=220, bbox_inches="tight"); print(f"saved to {args.out}")
if __name__=="__main__": main()
