"""
inverse_design_demo.py — 动量平面逆向设计演示

Demonstrates inverse design: specify a target momentum distribution, then use
gradient descent (JAX autodiff) to find injection parameters that reproduce it.

Run: python examples/inverse_design_demo.py
Output: assets/inverse_design.png (target vs initial vs final + loss curve)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from momentum_plane.inverse_design import (
    InverseDesignConfig,
    optimize,
    target_ring,
    target_with_peaks,
)


def plot_result(result, title, out_path):
    """Plot target, initial, final momentum planes + loss curve."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Target
    axes[0].imshow(result.target_momentum, cmap="viridis", origin="lower")
    axes[0].set_title("Target\n(desired momentum plane)")
    axes[0].set_xlabel("kx")
    axes[0].set_ylabel("ky")

    # Initial (random params)
    axes[1].imshow(result.initial_momentum, cmap="viridis", origin="lower")
    axes[1].set_title("Initial\n(random parameters)")
    axes[1].set_xlabel("kx")

    # Final (optimized)
    axes[2].imshow(result.final_momentum, cmap="viridis", origin="lower")
    axes[2].set_title(
        f"Optimized\n(kx={result.params['kx']:.3f}, ky={result.params['ky']:.3f})"
    )
    axes[2].set_xlabel("kx")

    # Loss curve
    iters = np.arange(len(result.loss_history)) * 10
    axes[3].plot(iters, result.loss_history, "b-o", markersize=4)
    axes[3].set_yscale("log")
    axes[3].set_xlabel("Iteration")
    axes[3].set_ylabel("Loss (log scale)")
    axes[3].set_title(
        f"Convergence\n{result.loss_history[0]:.2e} → {result.loss_history[-1]:.2e}"
    )
    axes[3].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → saved {os.path.basename(out_path)}")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  MomentumPlane — Inverse Design Demo")
    print("  (gradient descent over injection parameters)")
    print("=" * 60)

    # --- Demo 1: Two horizontal peaks ---
    print("\n[1/2] Target: two horizontal peaks...")
    target1 = target_with_peaks(48, [(16, 14), (16, 34)], peak_sigma=4.0)
    cfg1 = InverseDesignConfig(
        grid_size=48, stride=8, n_steps=15,
        n_iter=150, lr=0.03, loss_type="mse", seed=42,
    )
    result1 = optimize(target1, cfg1)
    print(f"  Loss: {result1.loss_history[0]:.2e} → {result1.loss_history[-1]:.2e}")
    print(f"  Params: kx={result1.params['kx']:.4f}, ky={result1.params['ky']:.4f}, "
          f"sigma={result1.params['sigma']:.4f}")
    plot_result(result1, "Inverse Design — Two Peak Target",
                os.path.join(out_dir, "inverse_design_peaks.png"))

    # --- Demo 2: Ring target ---
    print("\n[2/2] Target: ring distribution...")
    target2 = target_ring(48, radius=12, thickness=2.5)
    cfg2 = InverseDesignConfig(
        grid_size=48, stride=8, n_steps=15,
        n_iter=150, lr=0.03, loss_type="correlation", seed=123,
    )
    result2 = optimize(target2, cfg2)
    print(f"  Loss: {result2.loss_history[0]:.4f} → {result2.loss_history[-1]:.4f}")
    print(f"  Params: kx={result2.params['kx']:.4f}, ky={result2.params['ky']:.4f}, "
          f"sigma={result2.params['sigma']:.4f}")
    plot_result(result2, "Inverse Design — Ring Target (correlation loss)",
                os.path.join(out_dir, "inverse_design_ring.png"))

    print("\n" + "=" * 60)
    print("  Done! Check assets/inverse_design_*.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
