"""
generate_demo_assets.py — generate high-quality demo assets
Optimized parameter combinations for the most visually striking output.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from momentum_plane import MomentumPlanePipeline, SimulationConfig, Visualizer

OUT = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(OUT, exist_ok=True)

def run_config(name, cfg, make_gif=True):
    print(f"[{name}] grid={cfg.grid_size} stride={cfg.stride} steps={cfg.n_steps} coin={cfg.coin_type}...")
    result = MomentumPlanePipeline(cfg).run()
    viz = Visualizer(dpi=150)
    viz.side_by_side(
        result["final_position"], result["final_momentum_intensity"],
        step_label=f"(t={cfg.n_steps})",
        save_path=os.path.join(OUT, f"{name}_final.png"),
    )
    viz.convergence_plot(
        result["peak_intensities"],
        title=f"Momentum Peak Convergence - {name}",
        save_path=os.path.join(OUT, f"{name}_convergence.png"),
    )
    if make_gif and len(result["position_history"]) > 2:
        viz.animate_evolution(
            result["position_history"], result["momentum_history"],
            save_path=os.path.join(OUT, f"{name}_evolution.gif"),
            fps=10,
        )
    print(f"  -> done. peaks={result['n_peaks'][-1]}, final_peak={result['peak_intensities'][-1]:.4f}")
    return result

# Config 1: classic convergence
run_config("classic", SimulationConfig(
    grid_size=96, stride=8, wavevector=(0.8, 0.0),
    sigma=2.5, n_steps=50, coin_type="hadamard",
    record_every=2, seed=42,
))

# Config 2: diagonal wavevector - rotated diffraction
run_config("diagonal", SimulationConfig(
    grid_size=96, stride=10, wavevector=(0.5, 0.5),
    sigma=2.0, n_steps=40, coin_type="hadamard",
    record_every=2, seed=42,
))

# Config 3: Grover coin - diffusion evolution
run_config("grover", SimulationConfig(
    grid_size=96, stride=12, wavevector=(0.6, 0.2),
    sigma=3.0, n_steps=45, coin_type="grover",
    record_every=3, seed=42,
))

# Config 4: chiral bias - asymmetric convergence
run_config("chiral", SimulationConfig(
    grid_size=96, stride=10, wavevector=(0.7, 0.0),
    sigma=2.5, n_steps=50, coin_type="hadamard",
    coin_angle=0.3, record_every=2, seed=42,
))

# Config 5: sparse lattice - sharp isolated peaks
run_config("sparse", SimulationConfig(
    grid_size=96, stride=24, wavevector=(0.9, 0.0),
    sigma=2.0, n_steps=35, coin_type="hadamard",
    record_every=2, seed=42,
), make_gif=False)

print("\n=== All assets generated ===")
for f in sorted(os.listdir(OUT)):
    if f.endswith(('.png', '.gif')):
        size = os.path.getsize(os.path.join(OUT, f))
        print(f"  {f}: {size/1024:.0f} KB")
