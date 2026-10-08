"""
MomentumPlane — 动量平面引擎
A physics-inspired high-performance simulation framework for periodic coherent
injection and discrete quantum hopping on momentum-space lattices.

Modules:
    injector    — periodic wave-packet injection on a spatial grid
    lattice     — discrete-time quantum walk (DTQW) unitary evolution
    synthesizer — momentum-plane field synthesis via 2D FFT
    visualizer  — heatmap, phase-space trajectory, and animation rendering
"""

__version__ = "0.1.0"
__author__ = "MomentumPlane Contributors"

from .injector import Injector
from .lattice import LatticeHop
from .synthesizer import FieldPlane
from .visualizer import Visualizer
from .pipeline import MomentumPlanePipeline, SimulationConfig

__all__ = [
    "Injector", "LatticeHop", "FieldPlane", "Visualizer",
    "MomentumPlanePipeline", "SimulationConfig",
]
