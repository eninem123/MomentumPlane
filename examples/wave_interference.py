"""
wave_interference.py — compare different injection geometries
Run: python examples/wave_interference.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from momentum_plane import MomentumPlanePipeline, SimulationConfig

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(out_dir, exist_ok=True)
    configs = [
        ("Fine lattice (stride=4)", SimulationConfig(grid_size=64, stride=4, n_steps=20, seed=42)),
        ("Coarse lattice (stride=16)", SimulationConfig(grid_size=64, stride=16, n_steps=20, seed=42)),
        ("Diagonal wavevector", SimulationConfig(grid_size=64, stride=8, wavevector=(0.5, 0.5), n_steps=20, seed=42)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    for ax, (label, cfg) in zip(axes, configs):
        print(f"Running: {label} ...")
        result = MomentumPlanePipeline(cfg).run()
        data = result["final_momentum_intensity"] + 1e-12
        im = ax.imshow(data, cmap="inferno", origin="lower")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("kx"); ax.set_ylabel("ky")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("Momentum-Plane Interference Patterns", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "wave_interference.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"-> saved {out_path}")

if __name__ == "__main__":
    main()
