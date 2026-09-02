#!/usr/bin/env python3
import argparse, matplotlib.pyplot as plt, matplotlib.patches as patches
def box(ax,x,y,w,h,txt,bg,fg,fs=7):
    ax.add_patch(patches.FancyBboxPatch((x,y),w,h, boxstyle="round,pad=0.06", facecolor=bg, edgecolor=fg, lw=1.1))
    ax.text(x+w/2,y+h/2, txt, ha="center", va="center", fontsize=fs, color=fg, weight="bold", linespacing=1.25)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out", default="05_dag.png"); args=ap.parse_args()
    fig, ax=plt.subplots(figsize=(11,4.5)); ax.set_xlim(0,11); ax.set_ylim(0,4.2); ax.axis("off")
    fig.patch.set_facecolor("#fbfbfd"); ax.set_facecolor("#fbfbfd")
    # DAG top
    box(ax,4.2,3.0,2.6,0.6,"z = y.sum()\nSumBackward", "#fff3e0","#e65100",6.5)
    box(ax,4.2,2.1,2.6,0.6,"y = x*2\nMulBackward", "#e3f2fd","#1565c0",6.5)
    box(ax,2.0,1.1,2.4,0.6,"AccumulateGrad(x)\nleaf x", "#e8f5e9","#2e7d32",6.5)
    box(ax,6.6,1.1,2.4,0.6,"CatBackward\n(branch merge)", "#f3e5f5","#6a1b9a",6.5)
    ax.annotate("", xy=(5.5,2.7), xytext=(5.5,3.0), arrowprops=dict(arrowstyle="->", color="#78909c", lw=1.2))
    ax.annotate("", xy=(4.2,1.7), xytext=(5.0,2.1), arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.1, connectionstyle="arc3,rad=0.2"))
    ax.annotate("", xy=(7.2,1.7), xytext=(6.0,2.1), arrowprops=dict(arrowstyle="->", color="#6a1b9a", lw=1.1, connectionstyle="arc3,rad=-0.2"))
    ax.text(5.5,3.75, "Explicit DAG: Node.next_edges (no global Tape)", ha="center", fontsize=7, color="#37474f", style="italic")
    # Engine bottom
    ax.add_patch(patches.FancyBboxPatch((0.5,0.15),5.0,0.75, boxstyle="round,pad=0.06", facecolor="white", edgecolor="#78909c", lw=1))
    ax.text(3.0,0.65, "Engine  ·  ReadyQueue CPU (priority by sequence_nr)", ha="center", fontsize=6, color="#37474f", weight="bold")
    ax.text(3.0,0.35, "GraphTask: dependencies / InputBuffer / outstanding_tasks", ha="center", fontsize=5, color="#546e7a", family="monospace")
    ax.add_patch(patches.FancyBboxPatch((6.0,0.15),4.5,0.75, boxstyle="round,pad=0.06", facecolor="white", edgecolor="#00838f", lw=1))
    ax.text(8.25,0.65, "ReadyQueue CUDA  ·  worker_main thread", ha="center", fontsize=6, color="#006064", weight="bold")
    ax.text(8.25,0.35, "queue_for_device(id)  lazy thread", ha="center", fontsize=5, color="#00838f", family="monospace")
    ax.annotate("", xy=(3.0,0.9), xytext=(3.5,1.1), arrowprops=dict(arrowstyle="->", color="#78909c", lw=1, ls="--"))
    ax.annotate("", xy=(8.25,0.9), xytext=(7.5,1.1), arrowprops=dict(arrowstyle="->", color="#00838f", lw=1, ls="--"))
    ax.set_title("Autograd DAG + Engine ReadyQueue (tpx/include/Node.h:26, Engine.h:23)", fontsize=11, weight="bold", color="#212121")
    plt.tight_layout(); plt.savefig(args.out, dpi=220, bbox_inches="tight"); print(f"saved to {args.out}")
if __name__=="__main__": main()
