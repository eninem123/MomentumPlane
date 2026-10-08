import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from momentum_plane.synthesizer import FieldPlane

class TestFieldPlane:
    def test_transform_shape(self):
        fp = FieldPlane(grid_size=32)
        psi = np.random.randn(32, 32) + 1j * np.random.randn(32, 32)
        assert fp.transform(psi).shape == (32, 32)

    def test_intensity_normalised(self):
        fp = FieldPlane(grid_size=32)
        psi = np.random.randn(32, 32) + 1j * np.random.randn(32, 32)
        inten = fp.intensity(psi)
        np.testing.assert_allclose(inten.max(), 1.0)
        assert np.all(inten >= 0)

    def test_parseval_energy(self):
        fp = FieldPlane(grid_size=64, apply_shift=False)
        psi = np.random.randn(64, 64) + 1j * np.random.randn(64, 64)
        energy_pos = np.sum(np.abs(psi) ** 2)
        psi_k = fp.transform(psi)
        energy_mom = np.sum(np.abs(psi_k) ** 2) / (64 * 64)
        np.testing.assert_allclose(energy_pos, energy_mom, rtol=1e-10)

    def test_phase_range(self):
        fp = FieldPlane(grid_size=32)
        psi = np.random.randn(32, 32) + 1j * np.random.randn(32, 32)
        phase = fp.phase(psi)
        assert np.all(phase >= -np.pi - 1e-10)
        assert np.all(phase <= np.pi + 1e-10)

    def test_peak_detection(self):
        fp = FieldPlane(grid_size=64)
        x = np.arange(64)
        X, Y = np.meshgrid(x, x, indexing="ij")
        psi = np.exp(1j * (0.5 * X + 0.3 * Y))
        peaks = fp.peak_positions(psi, threshold=0.5, min_distance=4)
        assert len(peaks) >= 1

    def test_window_application(self):
        fp = FieldPlane(grid_size=32, window="hann")
        psi = np.ones((32, 32), dtype=np.complex128)
        assert np.abs(fp.transform(psi)).max() > 0
