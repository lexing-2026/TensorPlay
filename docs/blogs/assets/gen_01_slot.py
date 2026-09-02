#!/usr/bin/env python3
import argparse, matplotlib.pyplot as plt, matplotlib.patches as patches
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out", default="01_slot.png"); args=ap.parse_args()
    fig, ax=plt.subplots(figsize=(11,4.5)); ax.set_xlim(0,11); ax.set_ylim(0,4); ax.axis("off")
    fig.patch.set_facecolor("#fbfbfd"); ax.set_facecolor("#fbfbfd")
    # left: p10 slot
    ax.add_patch(patches.FancyBboxPatch((0.5,1.0),4.2,2.2, boxstyle="round,pad=0.08", facecolor="#e3f2fd", edgecolor="#1565c0", lw=1.4))
    ax.text(2.6,3.0, "p10  ·  TensorImpl", ha="center", fontsize=9, weight="bold", color="#0d47a1")
    ax.text(2.6,2.5, "shared_ptr<AutogradMetaBase>", ha="center", fontsize=7, color="#1565c0", family="monospace")
    ax.add_patch(patches.FancyBboxPatch((1.2,1.35),2.8,0.65, boxstyle="round,pad=0.06", facecolor="white", edgecolor="#90a4ae", lw=1, ls="--"))
    ax.text(2.6,1.68, "USB slot (pure virtual)\nshape only, no content", ha="center", fontsize=7, color="#37474f", linespacing=1.3)
    # boundary
    ax.plot([5.2,5.2],[0.6,3.6], color="#78909c", lw=1.2, ls="--")
    ax.text(5.2,3.75, "shared-lib boundary", ha="center", fontsize=7, color="#546e7a")
    ax.text(5.2,0.35, "p10 headers never #include tpx", ha="center", fontsize=6, color="#78909c", style="italic")
    # right: tpx plug
    ax.add_patch(patches.FancyBboxPatch((6.0,1.0),4.5,2.2, boxstyle="round,pad=0.08", facecolor="#fff3e0", edgecolor="#ef6c00", lw=1.4))
    ax.text(8.25,3.0, "tpx  ·  AutogradMeta", ha="center", fontsize=9, weight="bold", color="#e65100")
    ax.text(8.25,2.5, "grad_fn_ / grad_ / weak_ptr<AccumulateGrad>", ha="center", fontsize=6.5, color="#ef6c00", family="monospace")
    ax.add_patch(patches.FancyBboxPatch((6.7,1.35),3.1,0.65, boxstyle="round,pad=0.06", facecolor="white", edgecolor="#ffab40", lw=1))
    ax.text(8.25,1.68, "USB plug (concrete)\nstatic_cast handshake", ha="center", fontsize=7, color="#4e342e", linespacing=1.3)
    # arrow
    ax.annotate("", xy=(6.0,1.68), xytext=(4.7,1.68), arrowprops=dict(arrowstyle="->", color="#ff6f00", lw=1.6, connectionstyle="arc3,rad=0.18"))
    ax.text(5.35,1.95, "get_or_create_autograd_meta()", ha="center", fontsize=6, color="#ff6f00", family="monospace")
    ax.set_title("Virtual-Interface Decoupling: p10 slot, tpx plug", fontsize=11, weight="bold", color="#212121")
    plt.tight_layout(); plt.savefig(args.out, dpi=220, bbox_inches="tight"); print(f"saved to {args.out}")
if __name__=="__main__": main()
