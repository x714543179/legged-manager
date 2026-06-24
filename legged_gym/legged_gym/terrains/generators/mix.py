from __future__ import annotations

from legged_gym.terrains.generators.rough import RoughTerrainGenerator


_ROUGH = "legged_gym.terrains.generators.rough"
_MGDP = "legged_gym.terrains.generators.mgdp"


def _terrain_dict_term(name: str, terrain_type: int) -> dict:
    normalized = name.strip().lower().replace("_", "-")
    terrain_terms = {
        "plane": {"class_name": f"{_ROUGH}:DiscreteObstaclesTerrain", "height_range": (0.0, 0.0)},
        "slope down": {"class_name": f"{_ROUGH}:PyramidSlopeTerrain", "inverted": True},
        "slope-down": {"class_name": f"{_ROUGH}:PyramidSlopeTerrain", "inverted": True},
        "slope": {"class_name": f"{_ROUGH}:PyramidSlopeTerrain"},
        "pyramid": {"class_name": f"{_ROUGH}:RandomRoughSlopeTerrain"},
        "rough slope": {"class_name": f"{_ROUGH}:RandomRoughSlopeTerrain"},
        "rough-slope": {"class_name": f"{_ROUGH}:RandomRoughSlopeTerrain"},
        "stairs down": {"class_name": f"{_ROUGH}:PyramidStairsTerrain", "inverted": True},
        "stairs-down": {"class_name": f"{_ROUGH}:PyramidStairsTerrain", "inverted": True},
        "down-stairs": {"class_name": f"{_ROUGH}:PyramidStairsTerrain", "inverted": True},
        "stairs up": {"class_name": f"{_ROUGH}:PyramidStairsTerrain", "inverted": False},
        "stairs-up": {"class_name": f"{_ROUGH}:PyramidStairsTerrain", "inverted": False},
        "up-stairs": {"class_name": f"{_ROUGH}:PyramidStairsTerrain", "inverted": False},
        "new stairs down": {
            "class_name": f"{_ROUGH}:PyramidStairsTerrain",
            "inverted": True,
            "step_width": 0.5,
            "step_height_range": (0.25, 0.55),
        },
        "new-stairs-down": {
            "class_name": f"{_ROUGH}:PyramidStairsTerrain",
            "inverted": True,
            "step_width": 0.5,
            "step_height_range": (0.25, 0.55),
        },
        "discrete obstacles": {"class_name": f"{_ROUGH}:DiscreteObstaclesTerrain"},
        "discrete-obstacles": {"class_name": f"{_ROUGH}:DiscreteObstaclesTerrain"},
        "discrete": {"class_name": f"{_ROUGH}:DiscreteObstaclesTerrain"},
        "pit": {"class_name": f"{_ROUGH}:PitTerrain"},
        "gap": {"class_name": f"{_MGDP}:ParkourGapTerrain"},
        "single-gap": {"class_name": f"{_MGDP}:ParkourGapTerrain", "terrain_type": 3},
        "ramp": {"class_name": f"{_MGDP}:RampTerrain"},
        "hurdle": {"class_name": f"{_MGDP}:HurdleTerrain"},
        "bream": {"class_name": f"{_MGDP}:BeamTerrain"},
        "beam": {"class_name": f"{_MGDP}:BeamTerrain"},
        "single-bridge": {"class_name": f"{_MGDP}:BeamTerrain", "terrain_type": 8},
        "step-beams": {"class_name": f"{_MGDP}:BeamTerrain", "terrain_type": 9},
        "rotation-beams": {"class_name": f"{_MGDP}:BeamTerrain", "terrain_type": 10},
        "narrow-beams": {"class_name": f"{_MGDP}:BeamTerrain", "terrain_type": 11},
        "cross-beams": {"class_name": f"{_MGDP}:BeamTerrain", "terrain_type": 12},
        "air-beams": {"class_name": f"{_MGDP}:BeamTerrain", "terrain_type": 13},
        "step-stone": {"class_name": f"{_MGDP}:ParkourStepTerrain", "terrain_type": 4},
        "stones-1rows": {"class_name": f"{_MGDP}:ParkourStepTerrain", "num_stones": 1, "terrain_type": 7},
        "stones-2rows": {"class_name": f"{_MGDP}:ParkourStepTerrain", "num_stones": 2, "terrain_type": 5},
        "balance-2stones": {"class_name": f"{_MGDP}:ParkourStepTerrain", "num_stones": 2, "terrain_type": 6},
        "air-stone": {"class_name": f"{_MGDP}:AirStoneTerrain"},
        "air-stone-terrain": {"class_name": f"{_MGDP}:AirStoneTerrain"},
        "corridor": {"class_name": f"{_MGDP}:NarrowCorridorTerrain"},
    }
    term_cfg = dict(terrain_terms.get(normalized, {"class_name": f"{_ROUGH}:DiscreteObstaclesTerrain"}))
    term_cfg.setdefault("terrain_type", terrain_type)
    return term_cfg


class MixTerrainGenerator(RoughTerrainGenerator):
    """MGDP-style mixed terrain entry point.

    This class intentionally reuses the tiled generator protocol. Task configs can
    compose a mix terrain by listing sub_terrains; concrete MGDP terrain pieces
    can be added here without touching robot env code.
    """

    def __init__(self, cfg, num_robots: int, terrain_dict: dict | None = None, **kwargs) -> None:
        self.terrain_dict_cfg = terrain_dict
        super().__init__(cfg, num_robots, **kwargs)

    def _build_sub_terrains(self):
        if self.sub_terrains_cfg:
            return super()._build_sub_terrains()
        if self.terrain_dict_cfg:
            return [
                (
                    name,
                    _terrain_dict_term(name, idx),
                    proportion,
                )
                for idx, (name, proportion) in enumerate(self.terrain_dict_cfg.items())
                if proportion > 0.0
            ]
        return super()._build_sub_terrains()

    def generate(self):
        data = super().generate()
        data.extras.setdefault("terrain_kind", "mix")
        return data
