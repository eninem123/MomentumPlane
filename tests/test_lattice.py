import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from momentum_plane.lattice import LatticeHop, hadamard_coin, grover_coin

class TestCoinOperators:
    def test_hadamard_unitary(self):
        H = hadamard_coin()
        np.testing.assert_allclose(H @ H.conj().T, np.eye(4), atol=1e-10)

    def test_grover_unitary(self):
        G = grover_coin()
        np.testing.assert_allclose(G @ G.conj().T, np.eye(4), atol=1e-10)

class TestLatticeHop:
    def test_state_shape(self):
        lat = LatticeHop(grid_size=32)
        psi_pos = np.random.randn(32, 32) + 1j * np.random.randn(32, 32)
        psi = lat.initialize_state(psi_pos)
        assert psi.shape == (32, 32, 4)

    def test_unitarity_preserves_norm(self):
        lat = LatticeHop(grid_size=32, boundary="periodic")
        psi_pos = np.zeros((32, 32), dtype=np.complex128)
        psi_pos[16, 16] = 1.0
        psi = lat.initialize_state(psi_pos)
        norm_initial = np.sum(np.abs(psi) ** 2)
        psi_final = lat.evolve(psi, n_steps=10)
        norm_final = np.sum(np.abs(psi_final) ** 2)
        np.testing.assert_allclose(norm_initial, norm_final, atol=1e-10)

    def test_probability_density(self):
        lat = LatticeHop(grid_size=16)
        psi = np.random.randn(16, 16, 4) + 1j * np.random.randn(16, 16, 4)
        rho = lat.probability_density(psi)
        assert rho.shape == (16, 16)
        assert np.all(rho >= 0)

    def test_step_changes_state(self):
        lat = LatticeHop(grid_size=32)
        psi_pos = np.zeros((32, 32), dtype=np.complex128)
        psi_pos[16, 16] = 1.0
        psi = lat.initialize_state(psi_pos)
        assert not np.allclose(psi, lat.step(psi))

    def test_invalid_coin_raises(self):
        with pytest.raises(ValueError):
            LatticeHop(grid_size=16, coin_type="invalid")
