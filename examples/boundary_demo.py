"""
boundary_demo.py — 拓扑边界约束演示

Demonstrates geometric boundary confinement: quantum walks trapped inside
rings, polygons, and strips. Inspired by topological photonics where
boundary states propagate defect-immune along lattice edges.

Run: python examples/boundary_demo.py
Output: assets/boundary_*.png (side-by-side position + momentum for each shape)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from momentum_plane import MomentumPlanePipeline, SimulationConfig


def run_and_plot(cfg, title, out_path):
    """Run simulation and plot position density + momentum intensity + boundary overlay."""
    result = MomentumPlanePipeline(cfg).run()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pos = result["final_position"]
    axes[0].imshow(pos, cmap="inferno", origin="lower")
    if result["boundary_mask"] is not None:
        axes[0].contour(result["boundary_mask"], levels=[0.5], colors="cyan",
                        linewidths=1.5, alpha=0.8)
    axes[0].set_title(f"{title}\nPosition |ψ|²")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    mom = result["final_momentum_intensity"]
    axes[1].imshow(np.log10(mom + 1e-10), cmap="viridis", origin="lower")
    axes[1].set_title(f"Momentum Plane\n({result['n_peaks'][-1]} peaks)")
    axes[1].set_xlabel("kx")
    axes[1].set_ylabel("ky")

    if result["boundary_mask"] is not None:
        axes[2].imshow(result["boundary_mask"], cmap="Blues", origin="lower")
        axes[2].set_title("Boundary Mask\n(cyan outline in left panel)")
    else:
        axes[2].text(0.5, 0.5, "No boundary", ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=16)
        axes[2].set_title("Boundary Mask")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → saved {os.path.basename(out_path)}")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(out_dir, exist_ok=True)

    common = dict(grid_size=96, stride=10, wavevector=(0.5, 0.3),
                  sigma=2.5, n_steps=35, coin_type="hadamard",
                  record_every=35, seed=42)

    print("=" * 60)
    print("  MomentumPlane — Topological Boundary Confinement Demo")
    print("=" * 60)

    print("\n[1/4] No boundary (free propagation)...")
    cfg = SimulationConfig(**common)
    run_and_plot(cfg, "Free (no boundary)",
                  os.path.join(out_dir, "boundary_free.png"))

    print("[2/4] Ring boundary...")
    cfg = SimulationConfig(boundary_mask={"shape": "ring", "inner_radius": 15, "outer_radius": 35}, **common)
    run_and_plot(cfg, "Ring (annular waveguide)",
                  os.path.join(out_dir, "boundary_ring.png"))

    print("[3/4] Hexagon boundary...")
    cfg = SimulationConfig(boundary_mask={"shape": "polygon", "n_sides": 6, "radius": 30}, **common)
    run_and_plot(cfg, "Hexagon (6-sided cavity)",
                  os.path.join(out_dir, "boundary_hexagon.png"))

    print("[4/4] Triangle boundary...")
    cfg = SimulationConfig(boundary_mask={"shape": "polygon", "n_sides": 3, "radius": 35}, **common)
    run_and_plot(cfg, "Triangle (3-sided cavity)",
                  os.path.join(out_dir, "boundary_triangle.png"))

    print("\n" + "=" * 60)
    print("  Done! Check assets/boundary_*.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
