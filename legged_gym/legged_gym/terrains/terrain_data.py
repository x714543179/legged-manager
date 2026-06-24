from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SubTerrainResult:
    terrain: object
    terrain_type: int = -1
    extras: dict = field(default_factory=dict)


@dataclass
class TerrainData:
    cfg: object
    mesh_type: str
    height_samples: np.ndarray | None = None
    vertices: np.ndarray | None = None
    triangles: np.ndarray | None = None
    env_origins: np.ndarray | None = None
    terrain_types: np.ndarray | None = None
    extras: dict = field(default_factory=dict)
    env_length: float = 0.0
    env_width: float = 0.0
    tot_rows: int = 0
    tot_cols: int = 0

    @property
    def heightsamples(self):
        return self.height_samples

    @property
    def type(self):
        return self.mesh_type

    @property
    def terrain_type(self):
        return self.terrain_types
