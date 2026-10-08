"""
pipeline.py — end-to-end simulation orchestration
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .injector import Injector
from .lattice import LatticeHop
from .synthesizer import FieldPlane
from .visualizer import Visualizer

@dataclass
class SimulationConfig:
    grid_size: int = 128
    stride: int = 16
    wavevector: tuple = (0.8, 0.0)
    sigma: float = 3.0
    amplitude: float = 1.0
    phase_jitter: float = 0.0
    n_steps: int = 30
    coin_type: str = "hadamard"
    coin_angle: float = 0.0
    boundary: str = "periodic"
    record_every: int = 1
    seed: Optional[int] = 42

class MomentumPlanePipeline:
    def __init__(self, config):
        self.cfg = config
        self.injector = Injector(
            grid_size=config.grid_size, stride=config.stride,
            wavevector=config.wavevector, sigma=config.sigma,
            amplitude=config.amplitude, phase_jitter=config.phase_jitter,
            seed=config.seed,
        )
        self.lattice = LatticeHop(
            grid_size=config.grid_size, coin_type=config.coin_type,
            coin_angle=config.coin_angle, boundary=config.boundary,
        )
        self.field = FieldPlane(grid_size=config.grid_size)
        self.visualizer = Visualizer()

    def run(self):
        cfg = self.cfg
        psi0 = self.injector.inject()
        initial_momentum = self.field.intensity(psi0)
        psi = self.lattice.initialize_state(psi0)
        position_history = []
        momentum_history = []
        peak_intensities = []
        n_peaks_list = []
        record_every = max(1, cfg.record_every)
        for t in range(cfg.n_steps):
            psi = self.lattice.step(psi)
            if (t + 1) % record_every == 0:
                pos_density = self.lattice.probability_density(psi)
                pos_field = self.lattice.position_field(psi)
                mom_intensity = self.field.intensity(pos_field)
                peaks = self.field.peak_positions(pos_field, threshold=0.3, min_distance=4)
                position_history.append(pos_density)
                momentum_history.append(mom_intensity)
                peak_intensities.append(float(mom_intensity.max()))
                n_peaks_list.append(len(peaks))
        final_position = self.lattice.probability_density(psi)
        final_field = self.lattice.position_field(psi)
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
        }
