#!/usr/bin/env python3
import argparse, matplotlib.pyplot as plt, matplotlib.patches as patches
def box(ax,x,y,w,h,txt,bg,fg,fs=6.5):
    ax.add_patch(patches.FancyBboxPatch((x,y),w,h, boxstyle="round,pad=0.06", facecolor=bg, edgecolor=fg, lw=1.1))
    ax.text(x+w/2,y+h/2, txt, ha="center", va="center", fontsize=fs, color=fg, weight="bold", linespacing=1.25)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out", default="06_pipeline.png"); args=ap.parse_args()
    fig, ax=plt.subplots(figsize=(12,4.5)); ax.set_xlim(0,12); ax.set_ylim(0,4.5); ax.axis("off")
    fig.patch.set_facecolor("#fbfbfd"); ax.set_facecolor("#fbfbfd")
    # Stage 1 Capture
    box(ax,0.5,2.8,3.2,1.2,"Stage 1  Capture\nTracer + Proxy\nPython fn -> FX Graph", "#e3f2fd","#0d47a1",7)
    ax.text(2.1,2.55,"tensorplay/graph", ha="center", fontsize=5, color="#1565c0", family="monospace")
    # Stage 2 Host
    box(ax,4.3,2.8,3.4,1.2,"Stage 2  Host (_stax)\nPassManager (5 passes)\nGuardChain + scheduler", "#fff3e0","#e65100",7)
    ax.text(6.0,2.55,"_stax/api.py", ha="center", fontsize=5, color="#ef6c00", family="monospace")
    # Stage 3 Codegen
    box(ax,8.3,2.8,3.2,1.2,"Stage 3  Codegen\nC++ (4x unroll+AVX)\nTriton template", "#e8f5e9","#1b5e20",7)
    ax.text(9.9,2.55,"codegen/cpp.py, triton.py", ha="center", fontsize=5, color="#2e7d32", family="monospace")
    # arrows
    ax.annotate("", xy=(4.3,3.4), xytext=(3.7,3.4), arrowprops=dict(arrowstyle="->", color="#78909c", lw=1.4))
    ax.annotate("", xy=(8.3,3.4), xytext=(7.7,3.4), arrowprops=dict(arrowstyle="->", color="#78909c", lw=1.4))
    # side boxes
    box(ax,4.3,1.4,1.6,0.7,"CodeCache\nfile_lock + rename", "#f3e5f5","#6a1b9a",6)
    box(ax,6.2,1.4,1.6,0.7,"CudaGraph\nManager", "#e0f2f1","#00695c",6)
    box(ax,8.4,1.4,1.5,0.7,"Backend\nstax / tvm", "#fffde7","#f9a825",6)
    ax.annotate("", xy=(5.1,1.4), xytext=(5.6,2.8), arrowprops=dict(arrowstyle="->", color="#ab47bc", lw=1, ls="--"))
    ax.annotate("", xy=(7.0,1.4), xytext=(6.4,2.8), arrowprops=dict(arrowstyle="->", color="#00897b", lw=1, ls="--"))
    ax.annotate("", xy=(9.1,1.4), xytext=(9.9,2.8), arrowprops=dict(arrowstyle="->", color="#f9a025", lw=1, ls="--"))
    # bottom note
    ax.text(6,0.8, "compile() is manager, _stax is scheduler, stax/tvm/inductor are machines", ha="center", fontsize=6, color="#546e7a", style="italic")
    ax.set_title("Compiler Pipeline: Capture -> Host (Guard+Pass) -> Codegen (C++/Triton)", fontsize=11, weight="bold", color="#212121")
    plt.tight_layout(); plt.savefig(args.out, dpi=220, bbox_inches="tight"); print(f"saved to {args.out}")
if __name__=="__main__": main()
