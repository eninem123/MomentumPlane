"""
benchmark.py — NumPy vs JAX 性能基准测试

Measures wall-clock time for the DTQW evolution loop across grid sizes
and step counts. Run with:

    python examples/benchmark.py

Outputs a markdown table and saves a comparison plot to assets/.
On CPU, JAX's jit-compiled lax.scan loop is typically 2-5x faster than
NumPy for large grids. On GPU (install jax[cuda]), expect 10-100x.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np

from momentum_plane.injector import Injector
from momentum_plane.lattice import LatticeHop
from momentum_plane.lattice_jax import LatticeHopJAX


def time_evolution(lattice, psi, n_steps, n_warmup=1, n_runs=3):
    """Time the evolve() method, returning best-of-n_runs seconds."""
    for _ in range(n_warmup):
        lattice.evolve(psi, n_steps)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        lattice.evolve(psi, n_steps)
        times.append(time.perf_counter() - t0)
    return min(times)


def run_benchmark(grid_sizes=(32, 64, 128, 256), n_steps=50):
    print(f"\n{'='*70}")
    print(f"  MomentumPlane Benchmark: NumPy vs JAX  (n_steps={n_steps})")
    print(f"{'='*70}\n")

    results = []
    for N in grid_sizes:
        print(f"  Grid {N}x{N} ...", end=" ", flush=True)

        inj = Injector(grid_size=N, stride=max(4, N // 8), wavevector=(0.6, 0.3),
                        sigma=2.5, seed=42)
        psi0 = inj.inject()

        lat_np = LatticeHop(grid_size=N, coin_type="hadamard")
        psi_np = lat_np.initialize_state(psi0)
        t_np = time_evolution(lat_np, psi_np, n_steps, n_warmup=1, n_runs=3)

        lat_jax = LatticeHopJAX(grid_size=N, coin_type="hadamard")
        psi_jax = lat_jax.initialize_state(psi0)
        t_jax = time_evolution(lat_jax, psi_jax, n_steps, n_warmup=2, n_runs=3)

        speedup = t_np / t_jax if t_jax > 0 else float("inf")
        results.append((N, t_np, t_jax, speedup))
        print(f"NumPy={t_np:.3f}s  JAX={t_jax:.3f}s  speedup={speedup:.1f}x")

    print(f"\n{'='*70}")
    print("  Results (markdown table):\n")
    print("| Grid Size | NumPy (s) | JAX (s) | Speedup |")
    print("|-----------|-----------|---------|---------|")
    for N, t_np, t_jax, sp in results:
        print(f"| {N}x{N} | {t_np:.3f} | {t_jax:.3f} | {sp:.1f}x |")
    print()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grids = [r[0] for r in results]
        np_times = [r[1] for r in results]
        jax_times = [r[2] for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(grids, np_times, "o-", label="NumPy", color="#e74c3c", linewidth=2)
        ax1.plot(grids, jax_times, "s-", label="JAX (jit)", color="#3498db", linewidth=2)
        ax1.set_xlabel("Grid Size (N x N)")
        ax1.set_ylabel("Time (s)")
        ax1.set_title(f"DTQW Evolution Time ({n_steps} steps)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        speedups = [r[3] for r in results]
        ax2.bar([str(g) for g in grids], speedups, color="#2ecc71", alpha=0.8)
        ax2.set_xlabel("Grid Size")
        ax2.set_ylabel("Speedup (x)")
        ax2.set_title("JAX vs NumPy Speedup")
        ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        for i, v in enumerate(speedups):
            ax2.text(i, v + 0.05, f"{v:.1f}x", ha="center", fontweight="bold")

        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), "..", "assets", "benchmark.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {out_path}")
    except Exception as e:
        print(f"  (Plot skipped: {e})")

    print(f"\n{'='*70}\n")
    return results


if __name__ == "__main__":
    run_benchmark()
