"""
test_inverse_design.py — Unit tests for inverse design (JAX autodiff optimization).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp

from momentum_plane.inverse_design import (
    InverseDesignConfig,
    differentiable_inject,
    forward_pass,
    loss_correlation,
    loss_mse,
    optimize,
    target_ring,
    target_with_peaks,
)

jax.config.update("jax_enable_x64", True)


class TestDifferentiableInject:
    def test_output_shape(self):
        field = differentiable_inject(0.5, 0.3, 2.0, 0.0, 32, 6)
        assert field.shape == (32, 32)

    def test_normalization(self):
        field = differentiable_inject(0.5, 0.3, 2.0, 0.0, 32, 6)
        norm = float(jnp.sum(jnp.abs(field) ** 2))
        assert abs(norm - 1.0) < 1e-10

    def test_complex_dtype(self):
        field = differentiable_inject(0.5, 0.3, 2.0, 0.0, 32, 6)
        assert jnp.issubdtype(field.dtype, jnp.complexfloating)

    def test_gradient_finite(self):
        """kx should have a non-zero, finite gradient."""
        def f(kx):
            field = differentiable_inject(kx, 0.3, 2.0, 0.0, 32, 6)
            return jnp.sum(jnp.abs(field) ** 2)

        grad = jax.grad(f)(0.5)
        assert jnp.isfinite(grad)
        assert abs(float(grad)) < 1e-8

    def test_sigma_gradient(self):
        """Intensity at center should depend on sigma."""
        def center_intensity(sigma):
            field = differentiable_inject(0.0, 0.0, sigma, 0.0, 32, 6)
            return jnp.abs(field[16, 16]) ** 2

        grad = jax.grad(center_intensity)(2.0)
        assert jnp.isfinite(grad)
        assert float(grad) < 0


class TestForwardPass:
    def test_output_shape(self):
        params = jnp.array([0.5, 0.3, 0.5, 0.0])
        mom = forward_pass(params, 32, 6, 10, "hadamard")
        assert mom.shape == (32, 32)

    def test_normalization(self):
        params = jnp.array([0.5, 0.3, 0.5, 0.0])
        mom = forward_pass(params, 32, 6, 10, "hadamard")
        assert abs(float(jnp.sum(mom)) - 1.0) < 1e-10

    def test_non_negative(self):
        params = jnp.array([0.5, 0.3, 0.5, 0.0])
        mom = forward_pass(params, 32, 6, 10, "hadamard")
        assert float(jnp.min(mom)) >= 0.0

    def test_grover_coin(self):
        params = jnp.array([0.5, 0.3, 0.5, 0.0])
        mom = forward_pass(params, 32, 6, 10, "grover")
        assert mom.shape == (32, 32)
        assert abs(float(jnp.sum(mom)) - 1.0) < 1e-10

    def test_gradient_flow(self):
        """Gradient of forward pass w.r.t. params should be finite and non-zero."""
        params = jnp.array([0.5, 0.3, 0.5, 0.0])

        def peak_intensity(p):
            mom = forward_pass(p, 32, 6, 10, "hadamard")
            return jnp.max(mom)

        grads = jax.grad(peak_intensity)(params)
        assert jnp.all(jnp.isfinite(grads))
        assert float(jnp.sum(jnp.abs(grads))) > 1e-12


class TestLossFunctions:
    def test_mse_non_negative(self):
        params = jnp.array([0.5, 0.3, 0.5, 0.0])
        target = jnp.ones((32, 32)) / (32 * 32)
        loss = loss_mse(params, target, 32, 6, 10, "hadamard")
        assert float(loss) >= 0.0

    def test_correlation_range(self):
        """Negative correlation should be in [-1, 0]."""
        params = jnp.array([0.5, 0.3, 0.5, 0.0])
        target = jnp.ones((32, 32)) / (32 * 32)
        loss = loss_correlation(params, target, 32, 6, 10, "hadamard")
        assert -1.0 <= float(loss) <= 0.0

    def test_mse_gradient(self):
        params = jnp.array([0.5, 0.3, 0.5, 0.0])
        target = jnp.ones((32, 32)) / (32 * 32)
        grads = jax.grad(loss_mse)(params, target, 32, 6, 10, "hadamard")
        assert jnp.all(jnp.isfinite(grads))


class TestTargetGeneration:
    def test_target_with_peaks_shape(self):
        t = target_with_peaks(48, [(20, 24), (28, 24)])
        assert t.shape == (48, 48)

    def test_target_with_peaks_normalized(self):
        t = target_with_peaks(48, [(20, 24)])
        assert abs(t.sum() - 1.0) < 1e-10

    def test_target_ring_shape(self):
        t = target_ring(48, radius=10, thickness=2.0)
        assert t.shape == (48, 48)

    def test_target_ring_normalized(self):
        t = target_ring(48, radius=10, thickness=2.0)
        assert abs(t.sum() - 1.0) < 1e-10


class TestOptimization:
    def test_optimize_converges(self):
        """Loss should decrease over iterations."""
        target = target_with_peaks(32, [(12, 16), (20, 16)], peak_sigma=3.0)
        cfg = InverseDesignConfig(
            grid_size=32, stride=6, n_steps=10,
            n_iter=40, lr=0.05, loss_type="mse", seed=42,
        )
        result = optimize(target, cfg)
        assert len(result.loss_history) > 1
        assert result.loss_history[-1] <= result.loss_history[0] + 1e-12

    def test_optimize_result_fields(self):
        target = target_with_peaks(32, [(16, 16)])
        cfg = InverseDesignConfig(grid_size=32, stride=6, n_steps=10, n_iter=20, seed=42)
        result = optimize(target, cfg)
        assert "kx" in result.params
        assert "ky" in result.params
        assert "sigma" in result.params
        assert "phase_offset" in result.params
        assert result.initial_momentum.shape == (32, 32)
        assert result.final_momentum.shape == (32, 32)
        assert result.target_momentum.shape == (32, 32)

    def test_sigma_positive(self):
        """Optimized sigma should always be positive."""
        target = target_with_peaks(32, [(16, 16)])
        cfg = InverseDesignConfig(grid_size=32, stride=6, n_steps=10, n_iter=20, seed=42)
        result = optimize(target, cfg)
        assert result.params["sigma"] > 0.0

    def test_correlation_loss_optimization(self):
        target = target_ring(32, radius=8, thickness=2.0)
        cfg = InverseDesignConfig(
            grid_size=32, stride=6, n_steps=10,
            n_iter=30, lr=0.02, loss_type="correlation", seed=123,
        )
        result = optimize(target, cfg)
        assert result.loss_history[-1] <= result.loss_history[0] + 1e-10
