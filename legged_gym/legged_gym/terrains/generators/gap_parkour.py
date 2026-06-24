from __future__ import annotations

import numpy as np

from legged_gym.terrains.generators.mix import MixTerrainGenerator


class GapParkourTerrainGenerator(MixTerrainGenerator):
    """MGDP-style gap parkour terrain entry point.

    The generator keeps parkour-specific metadata in TerrainData.extras so
    rewards, observations, or debug visualization can consume it through
    explicit mdp terms instead of mesh_type checks.
    """

    def __init__(self, cfg, num_robots: int, num_goals: int | None = None, **kwargs) -> None:
        super().__init__(cfg, num_robots, **kwargs)
        self.num_goals = num_goals if num_goals is not None else getattr(cfg, "num_goals", None)

    def generate(self):
        data = super().generate()
        data.extras["terrain_kind"] = "gap_parkour"
        if self.num_goals is not None:
            shape = (self.num_rows, self.num_cols, 3)
            data.extras.setdefault("goals", np.zeros(shape, dtype=np.float32))
            data.extras.setdefault("goals_stone", np.zeros(shape, dtype=np.float32))
            data.extras.setdefault("goals_narrow", np.zeros(shape, dtype=np.float32))
        return data
