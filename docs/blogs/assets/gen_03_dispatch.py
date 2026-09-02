#!/usr/bin/env python3
import argparse, matplotlib.pyplot as plt, matplotlib.patches as patches
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out", default="03_dispatch.png"); args=ap.parse_args()
    fig, ax=plt.subplots(figsize=(11,4.2)); ax.set_xlim(0,11); ax.set_ylim(0,4); ax.axis("off")
    fig.patch.set_facecolor("#fbfbfd"); ax.set_facecolor("#fbfbfd")
    # Tensor left
    ax.add_patch(patches.FancyBboxPatch((0.5,1.5),2.2,1.2, boxstyle="round,pad=0.08", facecolor="#e3f2fd", edgecolor="#1565c0", lw=1.3))
    ax.text(1.6,2.4, "Tensor (CUDA)", ha="center", fontsize=8, weight="bold", color="#0d47a1")
    ax.text(1.6,2.1, "key_set() ->", ha="center", fontsize=6, color="#1565c0", family="monospace")
    ax.text(1.6,1.85, "{ CUDA, AutogradCUDA }", ha="center", fontsize=6.5, color="#0d47a1", weight="bold", family="monospace")
    # Dispatcher middle
    ax.add_patch(patches.FancyBboxPatch((3.6,0.8),4.0,2.2, boxstyle="round,pad=0.08", facecolor="#fffde7", edgecolor="#f9a825", lw=1.3))
    ax.text(5.6,2.75, "Dispatcher  unordered_map<string, DispatchTable>", ha="center", fontsize=6.5, weight="bold", color="#f57f17", family="monospace")
    # Table for add
    ax.text(4.0,2.45, '"add"', ha="center", fontsize=7, color="#37474f", family="monospace", weight="bold")
    # 13 slots
    slot_names=["CPU","CUDA","Vulkan","AutogradCPU","AutogradCUDA","...","Composite"]
    colors=["#e3f2fd","#e3f2fd","#e3f2fd","#fff3e0","#fff3e0","#f5f5f5","#f3e5f5"]
    for i, (name, col) in enumerate(zip(slot_names, colors)):
        x=3.9+i*0.5; ax.add_patch(patches.Rectangle((x,1.5),0.42,0.45, facecolor=col, edgecolor="#90a4ae", lw=0.8))
        ax.text(x+0.21,1.72, name[:4], ha="center", fontsize=4.5, color="#37474f")
        # pointer dot if has kernel
        if name in ["CPU","CUDA","AutogradCUDA","Composite"]:
            ax.text(x+0.21,1.58, "●", ha="center", fontsize=6, color="#2e7d32")
    ax.text(5.6,1.25, "getKernel(AutogradCUDA) hit -> call   |   miss -> fallback to Composite", ha="center", fontsize=6, color="#546e7a", style="italic")
    ax.text(5.6,1.0, "Dispatcher.h:67  void* kernels[13] + atomic load", ha="center", fontsize=5, color="#78909c", family="monospace")
    # arrow tensor -> dispatcher
    ax.annotate("", xy=(3.6,2.1), xytext=(2.7,2.1), arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.3))
    ax.text(3.15,2.3, "highest_priority_key()", ha="center", fontsize=5, color="#1565c0", family="monospace")
    # Result right
    ax.add_patch(patches.FancyBboxPatch((8.4,1.5),2.0,1.2, boxstyle="round,pad=0.08", facecolor="#e8f5e9", edgecolor="#2e7d32", lw=1.3))
    ax.text(9.4,2.25, "Kernel", ha="center", fontsize=8, weight="bold", color="#1b5e20")
    ax.text(9.4,1.95, "add_cuda(Tensor...)", ha="center", fontsize=6, family="monospace", color="#2e7d32")
    ax.text(9.4,1.7, "reinterpret_cast<Func*>", ha="center", fontsize=5.5, color="#4caf50", family="monospace")
    ax.annotate("", xy=(8.4,2.0), xytext=(7.6,2.0), arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.3))
    ax.set_title("Dispatcher Lookup: 13-Slot Table + Composite Fallback (214 lines)", fontsize=11, weight="bold", color="#212121")
    plt.tight_layout(); plt.savefig(args.out, dpi=220, bbox_inches="tight"); print(f"saved to {args.out}")
if __name__=="__main__": main()
