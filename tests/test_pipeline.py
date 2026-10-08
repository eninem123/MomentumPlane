import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from momentum_plane import MomentumPlanePipeline, SimulationConfig

class TestPipeline:
    def test_full_run(self):
        cfg = SimulationConfig(grid_size=32, stride=8, n_steps=5, record_every=1, seed=42)
        result = MomentumPlanePipeline(cfg).run()
        for key in ["initial_position", "initial_momentum", "position_history",
                    "momentum_history", "final_position", "final_momentum",
                    "final_momentum_intensity", "peak_intensities", "n_peaks"]:
            assert key in result

    def test_history_length(self):
        cfg = SimulationConfig(grid_size=32, stride=8, n_steps=10, record_every=2, seed=42)
        result = MomentumPlanePipeline(cfg).run()
        assert len(result["position_history"]) == 5
        assert len(result["momentum_history"]) == 5

    def test_shapes(self):
        cfg = SimulationConfig(grid_size=48, stride=12, n_steps=3, seed=42)
        result = MomentumPlanePipeline(cfg).run()
        assert result["final_position"].shape == (48, 48)
        assert result["final_momentum_intensity"].shape == (48, 48)

    def test_reproducible(self):
        cfg = SimulationConfig(grid_size=32, stride=8, n_steps=5, seed=99)
        r1 = MomentumPlanePipeline(cfg).run()
        r2 = MomentumPlanePipeline(cfg).run()
        np.testing.assert_allclose(r1["final_momentum_intensity"], r2["final_momentum_intensity"])
