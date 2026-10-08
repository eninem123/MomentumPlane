import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from momentum_plane.injector import Injector

class TestInjector:
    def test_grid_shape(self):
        inj = Injector(grid_size=32, stride=8)
        psi = inj.inject()
        assert psi.shape == (32, 32)
        assert psi.dtype == np.complex128

    def test_injection_sites_count(self):
        inj = Injector(grid_size=64, stride=16)
        assert inj.n_sites > 0

    def test_amplitude_scaling(self):
        inj = Injector(grid_size=32, stride=8, amplitude=2.0)
        psi = inj.inject()
        assert np.abs(psi).max() > 1.0

    def test_phase_jitter_reproducible(self):
        inj1 = Injector(grid_size=32, stride=8, phase_jitter=0.5, seed=123)
        inj2 = Injector(grid_size=32, stride=8, phase_jitter=0.5, seed=123)
        np.testing.assert_array_equal(inj1.inject(), inj2.inject())

    def test_zero_jitter_deterministic(self):
        inj = Injector(grid_size=32, stride=8, phase_jitter=0.0, seed=None)
        np.testing.assert_array_equal(inj.inject(), inj.inject())

    def test_n_packets_selection(self):
        inj = Injector(grid_size=64, stride=8, seed=42)
        psi = inj.inject(n_packets=5)
        psi_full = inj.inject()
        assert np.linalg.norm(psi) <= np.linalg.norm(psi_full) + 1e-10
