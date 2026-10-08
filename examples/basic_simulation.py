"""
basic_simulation.py — minimal end-to-end demo
Run: python examples/basic_simulation.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from momentum_plane import MomentumPlanePipeline, SimulationConfig, Visualizer

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(out_dir, exist_ok=True)
    cfg = SimulationConfig(grid_size=64, stride=8, wavevector=(0.8, 0.0),
                            sigma=2.5, n_steps=40, coin_type="hadamard",
                            record_every=2, seed=42)
    print(f"Running: {cfg.grid_size}x{cfg.grid_size}, stride={cfg.stride}, {cfg.n_steps} steps...")
    result = MomentumPlanePipeline(cfg).run()
    viz = Visualizer()
    viz.side_by_side(result["final_position"], result["final_momentum_intensity"],
                      step_label=f"(t={cfg.n_steps})", save_path=os.path.join(out_dir, "basic_final.png"))
    print("  -> saved assets/basic_final.png")
    viz.convergence_plot(result["peak_intensities"], save_path=os.path.join(out_dir, "basic_convergence.png"))
    print("  -> saved assets/basic_convergence.png")
    if len(result["position_history"]) > 1:
        viz.animate_evolution(result["position_history"], result["momentum_history"],
                              save_path=os.path.join(out_dir, "basic_evolution.gif"), fps=8)
        print("  -> saved assets/basic_evolution.gif")
    print("Done.")

if __name__ == "__main__":
    main()
