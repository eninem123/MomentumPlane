"""
test_lattice_jax.py — Unit tests for the JAX-accelerated DTQW backend.
These tests verify that the JAX backend produces identical physics to the
NumPy backend, and that jit compilation works correctly.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from momentum_plane.lattice import LatticeHop
from momentum_plane.lattice_jax import (
    LatticeHopJAX, hadamard_coin_jax, grover_coin_jax, batch_evolve,
)
from momentum_plane.injector import Injector
from momentum_plane import MomentumPlanePipeline, SimulationConfig


class TestCoinOperatorsJAX:
    def test_hadamard_unitary(self):
        H = hadamard_coin_jax()
        np.testing.assert_allclose(H @ H.conj().T, np.eye(4), atol=1e-10)

    def test_grover_unitary(self):
        G = grover_coin_jax()
        np.testing.assert_allclose(G @ G.conj().T, np.eye(4), atol=1e-10)

    def test_matches_numpy_coins(self):
        from momentum_plane.lattice import hadamard_coin, grover_coin
        np.testing.assert_allclose(hadamard_coin_jax(), hadamard_coin(), atol=1e-12)
        np.testing.assert_allclose(grover_coin_jax(), grover_coin(), atol=1e-12)


class TestLatticeHopJAX:
    def test_state_shape(self):
        lat = LatticeHopJAX(grid_size=32)
        psi_pos = np.random.randn(32, 32) + 1j * np.random.randn(32, 32)
        psi = lat.initialize_state(psi_pos)
        assert psi.shape == (32, 32, 4)

    def test_unitarity_preserves_norm(self):
        """JAX DTQW must preserve total probability (unitary evolution)."""
        lat = LatticeHopJAX(grid_size=32)
        psi_pos = np.zeros((32, 32), dtype=np.complex128)
        psi_pos[16, 16] = 1.0
        psi = lat.initialize_state(psi_pos)
        norm_initial = float(jnp.sum(jnp.abs(psi) ** 2))
        psi_final = lat.evolve(psi, n_steps=10)
        norm_final = float(jnp.sum(jnp.abs(psi_final) ** 2))
        np.testing.assert_allclose(norm_initial, norm_final, atol=1e-10)

    def test_matches_numpy_backend(self):
        """JAX and NumPy backends must produce identical results."""
        grid_size = 48
        n_steps = 15

        inj = Injector(grid_size=grid_size, stride=8, wavevector=(0.6, 0.3), sigma=2.0, seed=42)
        psi0 = inj.inject()

        lat_np = LatticeHop(grid_size=grid_size, coin_type="hadamard")
        psi_np = lat_np.initialize_state(psi0)
        psi_np = lat_np.evolve(psi_np, n_steps=n_steps)
        final_np = lat_np.position_field(psi_np)

        lat_jax = LatticeHopJAX(grid_size=grid_size, coin_type="hadamard")
        psi_jax = lat_jax.initialize_state(psi0)
        psi_jax = lat_jax.evolve(psi_jax, n_steps=n_steps)
        final_jax = np.asarray(lat_jax.position_field(psi_jax))

        np.testing.assert_allclose(final_np, final_jax, atol=1e-10)

    def test_grover_matches_numpy(self):
        grid_size = 32
        inj = Injector(grid_size=grid_size, stride=8, seed=42)
        psi0 = inj.inject()

        lat_np = LatticeHop(grid_size=grid_size, coin_type="grover")
        psi_np = lat_np.evolve(lat_np.initialize_state(psi0), n_steps=10)
        final_np = lat_np.position_field(psi_np)

        lat_jax = LatticeHopJAX(grid_size=grid_size, coin_type="grover")
        psi_jax = lat_jax.evolve(lat_jax.initialize_state(psi0), n_steps=10)
        final_jax = np.asarray(lat_jax.position_field(psi_jax))

        np.testing.assert_allclose(final_np, final_jax, atol=1e-10)

    def test_chiral_bias_matches_numpy(self):
        grid_size = 32
        inj = Injector(grid_size=grid_size, stride=8, seed=42)
        psi0 = inj.inject()

        lat_np = LatticeHop(grid_size=grid_size, coin_type="hadamard", coin_angle=0.5)
        psi_np = lat_np.evolve(lat_np.initialize_state(psi0), n_steps=8)
        final_np = lat_np.position_field(psi_np)

        lat_jax = LatticeHopJAX(grid_size=grid_size, coin_type="hadamard", coin_angle=0.5)
        psi_jax = lat_jax.evolve(lat_jax.initialize_state(psi0), n_steps=8)
        final_jax = np.asarray(lat_jax.position_field(psi_jax))

        np.testing.assert_allclose(final_np, final_jax, atol=1e-10)

    def test_reflective_raises(self):
        with pytest.raises(ValueError, match="periodic"):
            LatticeHopJAX(grid_size=16, boundary="reflective")

    def test_pipeline_jax_backend(self):
        cfg = SimulationConfig(grid_size=32, stride=8, n_steps=5, backend="jax", seed=42)
        result = MomentumPlanePipeline(cfg).run()
        assert isinstance(result["final_position"], np.ndarray)
        assert isinstance(result["final_momentum_intensity"], np.ndarray)
        assert result["final_position"].shape == (32, 32)

    def test_pipeline_numpy_vs_jax_identical(self):
        common = dict(grid_size=48, stride=8, wavevector=(0.7, 0.2), sigma=2.5,
                      n_steps=12, coin_type="hadamard", seed=42)
        cfg_np = SimulationConfig(backend="numpy", **common)
        cfg_jax = SimulationConfig(backend="jax", **common)
        r_np = MomentumPlanePipeline(cfg_np).run()
        r_jax = MomentumPlanePipeline(cfg_jax).run()
        np.testing.assert_allclose(
            r_np["final_momentum_intensity"], r_jax["final_momentum_intensity"], atol=1e-10
        )


class TestBatchEvolve:
    def test_batch_shape(self):
        params = [
            {"wavevector": (0.5, 0.0), "stride": 8, "sigma": 2.5},
            {"wavevector": (0.0, 0.5), "stride": 8, "sigma": 2.5},
            {"wavevector": (0.3, 0.3), "stride": 12, "sigma": 2.0},
        ]
        intensities = batch_evolve(params, grid_size=32, n_steps=10)
        assert intensities.shape == (3, 32, 32)
        assert np.all(intensities >= 0)
        assert np.all(intensities <= 1.0 + 1e-10)
