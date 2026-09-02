#!/usr/bin/env python3
import argparse, matplotlib.pyplot as plt, matplotlib.patches as patches
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out", default="01_python.png"); args=ap.parse_args()
    fig, ax=plt.subplots(figsize=(11,3.2)); ax.set_xlim(0,11); ax.set_ylim(0,3); ax.axis("off")
    fig.patch.set_facecolor("#fbfbfd"); ax.set_facecolor("#fbfbfd")
    steps=[
        (0.4, "1. DLL preload\nWindows: LoadLibraryExW\nLinux: ctypes.CDLL + glob", "#e8eaf6", "#283593"),
        (3.9, "2. Symbol rename\n_C.add  ->  tensorplay.add\n__module__ rewrite", "#fff3e0", "#e65100"),
        (7.4, "3. Lazy subpkgs\n__getattr__ on demand\nlinalg / sparse / fft", "#e8f5e9", "#2e7d32"),
    ]
    for x, txt, bg, fg in steps:
        ax.add_patch(patches.FancyBboxPatch((x,0.7),3.0,1.5, boxstyle="round,pad=0.08", facecolor=bg, edgecolor=fg, lw=1.2))
        ax.text(x+1.5,1.45, txt, ha="center", va="center", fontsize=7.5, color=fg, weight="bold", linespacing=1.35)
    for sx, ex in [(3.4,3.9),(6.9,7.4)]:
        ax.annotate("", xy=(ex,1.45), xytext=(sx,1.45), arrowprops=dict(arrowstyle="->", color="#78909c", lw=1.4))
    ax.set_title("Python glue: 3 dirty jobs in 1200 lines for a smooth import", fontsize=11, weight="bold", color="#212121")
    plt.tight_layout(); plt.savefig(args.out, dpi=220, bbox_inches="tight"); print(f"saved to {args.out}")
if __name__=="__main__": main()
