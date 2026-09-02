#!/usr/bin/env python3
import argparse, matplotlib.pyplot as plt, matplotlib.patches as patches
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out", default="03_pipeline.png"); args=ap.parse_args()
    fig, ax=plt.subplots(figsize=(12,4.5)); ax.set_xlim(0,12); ax.set_ylim(0,4.5); ax.axis("off")
    fig.patch.set_facecolor("#fbfbfd"); ax.set_facecolor("#fbfbfd")
    # YAMLs
    ax.add_patch(patches.FancyBboxPatch((0.4,1.5),2.0,1.5, boxstyle="round,pad=0.08", facecolor="#e8f5e9", edgecolor="#2e7d32", lw=1.3))
    ax.text(1.4,2.6, "native_functions.yaml", ha="center", fontsize=7, weight="bold", color="#1b5e20", family="monospace")
    ax.text(1.4,2.3, "479 schemas", ha="center", fontsize=6, color="#2e7d32")
    ax.text(1.4,2.0, "func / dispatch / variants", ha="center", fontsize=5.5, color="#4caf50")
    ax.add_patch(patches.FancyBboxPatch((0.4,0.5),2.0,0.8, boxstyle="round,pad=0.08", facecolor="#fff3e0", edgecolor="#ef6c00", lw=1.3))
    ax.text(1.4,0.95, "derivatives.yaml", ha="center", fontsize=7, weight="bold", color="#e65100", family="monospace")
    ax.text(1.4,0.65, "187 formulas", ha="center", fontsize=6, color="#ef6c00")
    # Factory
    ax.add_patch(patches.FancyBboxPatch((3.6,1.0),2.4,1.5, boxstyle="round,pad=0.08", facecolor="#e3f2fd", edgecolor="#1565c0", lw=1.4))
    ax.text(4.8,2.15, "tools/codegen", ha="center", fontsize=8, weight="bold", color="#0d47a1")
    ax.text(4.8,1.85, "7 generators", ha="center", fontsize=6, color="#1565c0")
    ax.text(4.8,1.55, "gen_api / gen_tpx / gen_autograd\n gen_bindings / gen_python_c ...", ha="center", fontsize=5.5, color="#1976d2", linespacing=1.3)
    ax.text(4.8,1.2, "main.py:32 CodegenContext", ha="center", fontsize=5, color="#546e7a", family="monospace")
    # Arrows yaml -> factory
    ax.annotate("", xy=(3.6,2.2), xytext=(2.4,2.2), arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.4))
    ax.annotate("", xy=(3.6,1.2), xytext=(2.4,0.9), arrowprops=dict(arrowstyle="->", color="#ef6c00", lw=1.4))
    # Artifacts
    artifacts=["TensorGenerated.h/.cpp","TPXOpsGenerated","AutogradNodes","Bindings","PythonCAPI","... 10 artifacts"]
    ax.add_patch(patches.FancyBboxPatch((7.2,0.6),2.0,2.4, boxstyle="round,pad=0.08", facecolor="#f3e5f5", edgecolor="#6a1b9a", lw=1.3))
    for i, art in enumerate(artifacts):
        ax.text(8.2,2.7-i*0.28, art, ha="center", fontsize=5.5, color="#4a148c", family="monospace")
    ax.text(8.2,3.2, "build/generated/tensorplay/ops/", ha="center", fontsize=6, weight="bold", color="#6a1b9a")
    ax.annotate("", xy=(7.2,1.75), xytext=(6.0,1.75), arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.4))
    # Dispatcher
    ax.add_patch(patches.FancyBboxPatch((10.0,1.2),1.6,1.2, boxstyle="round,pad=0.08", facecolor="#e0f2f1", edgecolor="#00695c", lw=1.3))
    ax.text(10.8,2.0, "Dispatcher", ha="center", fontsize=7, weight="bold", color="#004d40")
    ax.text(10.8,1.7, "map<name, 13-slot>", ha="center", fontsize=5.5, color="#00695c", family="monospace")
    ax.text(10.8,1.45, "214 lines", ha="center", fontsize=6, color="#00897b")
    ax.annotate("", xy=(10.0,1.75), xytext=(9.2,1.75), arrowprops=dict(arrowstyle="->", color="#6a1b9a", lw=1.4))
    ax.set_title("From 2 YAMLs to 10 Artifacts to 1 Dispatcher — Code Generation Pipeline", fontsize=11, weight="bold", color="#212121")
    plt.tight_layout(); plt.savefig(args.out, dpi=220, bbox_inches="tight"); print(f"saved to {args.out}")
if __name__=="__main__": main()
