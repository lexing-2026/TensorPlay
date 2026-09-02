#!/usr/bin/env python3
import argparse, matplotlib.pyplot as plt, matplotlib.patches as patches
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out", default="02_layers.png"); args=ap.parse_args()
    fig, ax=plt.subplots(figsize=(11,4.5)); ax.set_xlim(0,11); ax.set_ylim(0,4); ax.axis("off")
    fig.patch.set_facecolor("#fbfbfd"); ax.set_facecolor("#fbfbfd")
    # StorageImpl bottom
    ax.add_patch(patches.FancyBboxPatch((0.5,0.5),10,0.7, boxstyle="round,pad=0.08", facecolor="#fff8e1", edgecolor="#f57f17", lw=1.3))
    ax.text(5.5,0.95, "Storage / StorageImpl  —  raw bytes, DataPtr (64B cache), allocator, nbytes, device", ha="center", fontsize=7.5, color="#e65100", weight="bold")
    ax.text(5.5,0.65, "p10/include/StorageImpl.h:13  ·  p10/include/DataPtr.h:29  ·  p10/src/Allocator.cpp:52", ha="center", fontsize=5.5, color="#8d6e63", family="monospace")
    # TensorImpl middle
    ax.add_patch(patches.FancyBboxPatch((0.5,1.6),10,1.2, boxstyle="round,pad=0.08", facecolor="#e3f2fd", edgecolor="#1565c0", lw=1.3))
    ax.text(5.5,2.55, "TensorImpl  —  SizesAndStrides, dtype, device, is_contiguous, is_view, version", ha="center", fontsize=7.5, color="#0d47a1", weight="bold")
    ax.text(5.5,2.2, "memory_format  ·  inference_tensor_  ·  SparseState  ·  Quantizer  ·  transform_value_ (vmap)", ha="center", fontsize=6, color="#1976d2")
    ax.text(5.5,1.85, "p10/include/TensorImpl.h:28  ·  Storage + SharedState + version_counter", ha="center", fontsize=5.5, color="#546e7a", family="monospace")
    # Tensor top
    ax.add_patch(patches.FancyBboxPatch((3.5,3.2),4,0.6, boxstyle="round,pad=0.08", facecolor="#fce4ec", edgecolor="#ad1457", lw=1.3))
    ax.text(5.5,3.55, "Tensor  —  shared_ptr<TensorImpl>  (value semantics, shallow copy)", ha="center", fontsize=7.5, color="#880e4f", weight="bold")
    ax.text(5.5,3.3, "p10/include/Tensor.h:80  ·  Tensor b = a  shares impl", ha="center", fontsize=5.5, color="#6a1b9a", family="monospace")
    # arrows
    ax.annotate("", xy=(5.5,3.2), xytext=(5.5,2.8), arrowprops=dict(arrowstyle="->", color="#ad1457", lw=1.3))
    ax.annotate("", xy=(5.5,1.6), xytext=(5.5,1.2), arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.3))
    ax.text(8.8,2.8, "shallow copy", fontsize=6, color="#ad1457", style="italic")
    ax.text(8.8,1.35, "owns / shares", fontsize=6, color="#1565c0", style="italic")
    ax.set_title("Three-Layer Model: Storage -> TensorImpl -> Tensor", fontsize=11, weight="bold", color="#212121")
    plt.tight_layout(); plt.savefig(args.out, dpi=220, bbox_inches="tight"); print(f"saved to {args.out}")
if __name__=="__main__": main()
