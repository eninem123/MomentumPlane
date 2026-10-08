"""
test_boundary.py — Unit tests for geometric boundary masks.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from momentum_plane import MomentumPlanePipeline, SimulationConfig
from momentum_plane.boundary import BoundaryMask


class TestBoundaryMaskShapes:
    def test_circle_mask(self):
        m = BoundaryMask(grid_size=64, shape="circle", radius=20)
        assert m.mask.shape == (64, 64)
        assert m.mask.dtype == bool
        assert m.mask[32, 32]
        assert not m.mask[0, 0]
        expected = np.pi * 20**2 / 64**2
        assert abs(m.coverage() - expected) < 0.05

    def test_ring_mask(self):
        m = BoundaryMask(grid_size=64, shape="ring", inner_radius=10, outer_radius=25)
        assert not m.mask[32, 32]
        assert m.mask[32, 50]
        assert not m.mask[0, 0]

    def test_polygon_hexagon(self):
        m = BoundaryMask(grid_size=64, shape="polygon", n_sides=6, radius=20)
        assert m.mask[32, 32]
        assert not m.mask[0, 0]
        assert m.coverage() > 0.1

    def test_polygon_triangle(self):
        m = BoundaryMask(grid_size=64, shape="polygon", n_sides=3, radius=25)
        assert m.mask[32, 32]
        m_hex = BoundaryMask(grid_size=64, shape="polygon", n_sides=6, radius=25)
        assert m.coverage() < m_hex.coverage()

    def test_strip_horizontal(self):
        m = BoundaryMask(grid_size=64, shape="strip", strip_axis="horizontal", strip_width=10)
        assert m.mask[32, 0]
        assert m.mask[32, 63]
        assert not m.mask[0, 32]
        assert not m.mask[63, 32]

    def test_strip_vertical(self):
        m = BoundaryMask(grid_size=64, shape="strip", strip_axis="vertical", strip_width=10)
        assert m.mask[0, 32]
        assert not m.mask[32, 0]

    def test_custom_mask(self):
        custom = np.zeros((64, 64), dtype=bool)
        custom[10:20, 10:20] = True
        m = BoundaryMask(grid_size=64, shape="custom", custom_mask=custom)
        np.testing.assert_array_equal(m.mask, custom)

    def test_custom_mask_wrong_shape_raises(self):
        custom = np.zeros((32, 32), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            BoundaryMask(grid_size=64, shape="custom", custom_mask=custom)

    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError, match="Unknown shape"):
            BoundaryMask(grid_size=64, shape="sphere")


class TestBoundaryApply:
    def test_apply_2d(self):
        m = BoundaryMask(grid_size=32, shape="circle", radius=10)
        psi = np.ones((32, 32), dtype=np.complex128)
        result = m.apply(psi)
        assert result.shape == (32, 32)
        np.testing.assert_allclose(result[m.mask], 1.0)
        np.testing.assert_allclose(result[~m.mask], 0.0)

    def test_apply_3d_coin_state(self):
        m = BoundaryMask(grid_size=32, shape="circle", radius=10)
        psi = np.ones((32, 32, 4), dtype=np.complex128)
        result = m.apply(psi)
        assert result.shape == (32, 32, 4)
        for d in range(4):
            np.testing.assert_allclose(result[:, :, d][m.mask], 1.0)
            np.testing.assert_allclose(result[:, :, d][~m.mask], 0.0)

    def test_apply_preserves_dtype(self):
        m = BoundaryMask(grid_size=16, shape="circle", radius=5)
        psi = np.ones((16, 16), dtype=np.complex64)
        result = m.apply(psi)
        assert result.dtype == np.complex64

    def test_apply_wrong_ndim_raises(self):
        m = BoundaryMask(grid_size=16, shape="circle", radius=5)
        with pytest.raises(ValueError, match="2D or 3D"):
            m.apply(np.ones((16,)))


class TestBoundarySites:
    def test_boundary_sites_exist(self):
        m = BoundaryMask(grid_size=32, shape="circle", radius=10)
        sites = m.boundary_sites()
        assert len(sites) > 0
        for r, c in sites:
            assert m.mask[r, c]

    def test_boundary_sites_adjacent_to_outside(self):
        m = BoundaryMask(grid_size=32, shape="circle", radius=10)
        sites = set(map(tuple, m.boundary_sites()))
        for r, c in sites:
            neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
            has_outside = any(
                0 <= nr < 32 and 0 <= nc < 32 and not m.mask[nr, nc]
                for nr, nc in neighbors
            )
            assert has_outside, f"Site ({r},{c}) has no outside neighbor"


class TestBoundaryPipeline:
    def test_pipeline_with_ring_boundary(self):
        cfg = SimulationConfig(
            grid_size=48, stride=8, n_steps=15, seed=42,
            boundary_mask={"shape": "ring", "inner_radius": 8, "outer_radius": 18},
        )
        result = MomentumPlanePipeline(cfg).run()
        assert result["boundary_mask"] is not None
        assert result["boundary_mask"].shape == (48, 48)
        assert result["final_position"].shape == (48, 48)

    def test_pipeline_without_boundary(self):
        cfg = SimulationConfig(grid_size=48, stride=8, n_steps=15, seed=42)
        result = MomentumPlanePipeline(cfg).run()
        assert result["boundary_mask"] is None

    def test_boundary_confines_probability(self):
        cfg = SimulationConfig(
            grid_size=48, stride=8, n_steps=20, record_every=5, seed=42,
            boundary_mask={"shape": "circle", "radius": 15},
        )
        result = MomentumPlanePipeline(cfg).run()
        mask = result["boundary_mask"]
        p_first = result["position_history"][0][mask].sum()
        p_last = result["position_history"][-1][mask].sum()
        assert p_last <= p_first + 1e-10

    def test_polygon_boundary_pipeline(self):
        cfg = SimulationConfig(
            grid_size=48, stride=8, n_steps=15, seed=42,
            boundary_mask={"shape": "polygon", "n_sides": 6, "radius": 15},
        )
        result = MomentumPlanePipeline(cfg).run()
        assert result["boundary_mask"] is not None
        assert result["final_momentum_intensity"].shape == (48, 48)
