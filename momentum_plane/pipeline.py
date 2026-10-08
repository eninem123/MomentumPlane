"""
pipeline.py — 端到端仿真流水线

Convenience class that wires Injector → LatticeHop → FieldPlane → Visualizer
into a single callable pipeline.  This is the entry point used by the
Streamlit dashboard and the example scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .injector import Injector
from .lattice import LatticeHop
from .synthesizer import FieldPlane
from .visualizer import Visualizer


@dataclass
class SimulationConfig:
    """All tunable parameters for one MomentumPlane run."""

    grid_size: int = 128
    stride: int = 16
    wavevector: tuple[float, float] = (0.8, 0.0)
    sigma: float = 3.0
    amplitude: float = 1.0
    phase_jitter: float = 0.0
    n_steps: int = 30
    coin_type: str = "hadamard"
    coin_angle: float = 0.0
    boundary: str = "periodic"
    record_every: int = 1
    seed: int | None = 42
    backend: str = "numpy"  # "numpy" or "jax" (JAX = GPU + jit + autodiff)
    # Geometric boundary confinement (topological photonics inspired)
    # Set to a dict like {"shape": "ring", "inner_radius": 20, "outer_radius": 40}
    # or None for no confinement. See momentum_plane.boundary.BoundaryMask for options.
    boundary_mask: dict | None = None


class MomentumPlanePipeline:
    """End-to-end simulation: inject → hop → synthesise → visualise.

    Usage
    -----
    >>> cfg = SimulationConfig(grid_size=64, n_steps=20)
    >>> pipe = MomentumPlanePipeline(cfg)
    >>> result = pipe.run()
    >>> result["final_momentum_intensity"].shape
    (64, 64)
    """

    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.injector = Injector(
            grid_size=config.grid_size,
            stride=config.stride,
            wavevector=config.wavevector,
            sigma=config.sigma,
            amplitude=config.amplitude,
            phase_jitter=config.phase_jitter,
            seed=config.seed,
        )
        self.field = FieldPlane(grid_size=config.grid_size)
        self.visualizer = Visualizer()

        # Geometric boundary mask (topological confinement)
        self._boundary_mask = None
        if config.boundary_mask is not None:
            from .boundary import BoundaryMask
            self._boundary_mask = BoundaryMask(
                grid_size=config.grid_size, **config.boundary_mask
            )

        # Select backend: NumPy (default) or JAX (GPU + jit + autodiff)
        if config.backend == "jax":
            from .lattice_jax import LatticeHopJAX
            self.lattice = LatticeHopJAX(
                grid_size=config.grid_size,
                coin_type=config.coin_type,
                coin_angle=config.coin_angle,
                boundary=config.boundary,
            )
            self._use_jax = True
        elif config.backend == "numpy":
            self.lattice = LatticeHop(
                grid_size=config.grid_size,
                coin_type=config.coin_type,
                coin_angle=config.coin_angle,
                boundary=config.boundary,
            )
            self._use_jax = False
        else:
            raise ValueError(f"Unknown backend: {config.backend}. Use 'numpy' or 'jax'.")

    def run(self) -> dict:
        """Execute the full pipeline and return all intermediate and final data.

        Returns
        -------
        dict with keys:
            initial_position      — (N,N) complex injected field
            initial_momentum      — (N,N) momentum intensity before evolution
            position_history      — list of (N,N) density arrays
            momentum_history      — list of (N,N) intensity arrays
            final_position        — (N,N) final density
            final_momentum        — (N,N) final momentum complex field
            final_momentum_intensity — (N,N) final intensity
            peak_intensities      — list of dominant peak intensity per step
            n_peaks               — list of detected peak count per step
        """
        cfg = self.cfg

        # Helper: convert JAX arrays to NumPy if using JAX backend
        def _to_np(arr):
            return np.asarray(arr) if self._use_jax else arr

        # 1. Inject
        psi0 = self.injector.inject()
        initial_momentum = self.field.intensity(psi0)

        # 2. Initialise quantum walk state
        psi = self.lattice.initialize_state(psi0)

        # 3. Evolve with history
        position_history = []
        momentum_history = []
        peak_intensities = []
        n_peaks_list = []

        record_every = max(1, cfg.record_every)
        for t in range(cfg.n_steps):
            psi = self.lattice.step(psi)
            # Apply geometric boundary mask (absorbing confinement)
            if self._boundary_mask is not None:
                psi = _to_np(psi)
                psi = self._boundary_mask.apply(psi)
                if self._use_jax:
                    import jax.numpy as jnp
                    psi = jnp.asarray(psi)
            if (t + 1) % record_every == 0:
                pos_density = _to_np(self.lattice.probability_density(psi))
                pos_field = _to_np(self.lattice.position_field(psi))
                mom_intensity = self.field.intensity(pos_field)
                peaks = self.field.peak_positions(pos_field, threshold=0.3, min_distance=4)
                position_history.append(pos_density)
                momentum_history.append(mom_intensity)
                peak_intensities.append(float(mom_intensity.max()))
                n_peaks_list.append(len(peaks))

        final_position = _to_np(self.lattice.probability_density(psi))
        final_field = _to_np(self.lattice.position_field(psi))
        final_momentum = self.field.transform(final_field)
        final_momentum_intensity = self.field.intensity(final_field)

        return {
            "initial_position": psi0,
            "initial_momentum": initial_momentum,
            "position_history": position_history,
            "momentum_history": momentum_history,
            "final_position": final_position,
            "final_momentum": final_momentum,
            "final_momentum_intensity": final_momentum_intensity,
            "peak_intensities": peak_intensities,
            "n_peaks": n_peaks_list,
            "boundary_mask": self._boundary_mask.mask if self._boundary_mask is not None else None,
        }
