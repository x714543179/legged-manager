from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from isaacgym import terrain_utils

from legged_gym.terrains.generator import TerrainGenerator
from legged_gym.terrains.terrain_data import SubTerrainResult


class RoughTerrainGenerator(TerrainGenerator):
    """Default legged_gym rough-terrain generator."""

    def _build_sub_terrains(self):
        sub_terrains = super()._build_sub_terrains()
        if sub_terrains:
            return sub_terrains

        proportions = list(getattr(self.cfg, "terrain_proportions", []))
        defaults = [
            ("slope", {"class_name": "legged_gym.terrains.generators.rough:PyramidSlopeTerrain"}, 1.0),
            ("rough_slope", {"class_name": "legged_gym.terrains.generators.rough:RandomRoughSlopeTerrain"}, 1.0),
            ("stairs_down", {"class_name": "legged_gym.terrains.generators.rough:PyramidStairsTerrain", "inverted": True}, 1.0),
            ("stairs_up", {"class_name": "legged_gym.terrains.generators.rough:PyramidStairsTerrain", "inverted": False}, 1.0),
            ("discrete", {"class_name": "legged_gym.terrains.generators.rough:DiscreteObstaclesTerrain"}, 1.0),
            ("stepping_stones", {"class_name": "legged_gym.terrains.generators.rough:SteppingStonesTerrain"}, 1.0),
            ("gap", {"class_name": "legged_gym.terrains.generators.rough:GapTerrain"}, 1.0),
            ("pit", {"class_name": "legged_gym.terrains.generators.rough:PitTerrain"}, 1.0),
        ]
        if proportions:
            defaults = [
                (name, cfg, proportions[idx])
                for idx, (name, cfg, _) in enumerate(defaults)
                if idx < len(proportions) and proportions[idx] > 0.0
            ]
        return defaults


@dataclass
class PyramidSlopeTerrain:
    platform_size: float = 3.0
    slope_range: tuple[float, float] = (0.0, 0.4)
    inverted: bool | None = None
    terrain_type: int = 0

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        slope = self.slope_range[0] + (self.slope_range[1] - self.slope_range[0]) * difficulty
        if self.inverted is True or (self.inverted is None and rng.random() < 0.5):
            slope *= -1.0
        terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=self.platform_size)
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class RandomRoughSlopeTerrain:
    platform_size: float = 3.0
    slope_range: tuple[float, float] = (0.0, 0.4)
    min_height: float = -0.05
    max_height: float = 0.05
    step: float = 0.005
    downsampled_scale: float = 0.2
    terrain_type: int = 1

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        slope = self.slope_range[0] + (self.slope_range[1] - self.slope_range[0]) * difficulty
        terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=self.platform_size)
        terrain_utils.random_uniform_terrain(
            terrain,
            min_height=self.min_height,
            max_height=self.max_height,
            step=self.step,
            downsampled_scale=self.downsampled_scale,
        )
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class PyramidStairsTerrain:
    step_width: float = 0.31
    step_height_range: tuple[float, float] = (0.05, 0.23)
    platform_size: float = 3.0
    inverted: bool = False
    terrain_type: int = 2

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        step_height = self.step_height_range[0] + (self.step_height_range[1] - self.step_height_range[0]) * difficulty
        if self.inverted:
            step_height *= -1.0
        terrain_utils.pyramid_stairs_terrain(
            terrain,
            step_width=self.step_width,
            step_height=step_height,
            platform_size=self.platform_size,
        )
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class DiscreteObstaclesTerrain:
    height_range: tuple[float, float] = (0.05, 0.25)
    rectangle_min_size: float = 1.0
    rectangle_max_size: float = 2.0
    num_rectangles: int = 20
    platform_size: float = 3.0
    terrain_type: int = 4

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        height = self.height_range[0] + (self.height_range[1] - self.height_range[0]) * difficulty
        terrain_utils.discrete_obstacles_terrain(
            terrain,
            height,
            self.rectangle_min_size,
            self.rectangle_max_size,
            self.num_rectangles,
            platform_size=self.platform_size,
        )
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class SteppingStonesTerrain:
    stone_size_range: tuple[float, float] = (1.5, 0.075)
    stone_distance: float = 0.1
    platform_size: float = 4.0
    terrain_type: int = 5

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        stone_size = self.stone_size_range[0] + (self.stone_size_range[1] - self.stone_size_range[0]) * difficulty
        distance = 0.05 if difficulty == 0 else self.stone_distance
        terrain_utils.stepping_stones_terrain(
            terrain,
            stone_size=stone_size,
            stone_distance=distance,
            max_height=0.0,
            platform_size=self.platform_size,
        )
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class GapTerrain:
    gap_size_range: tuple[float, float] = (0.0, 1.0)
    platform_size: float = 3.0
    terrain_type: int = 6

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        gap_size = self.gap_size_range[0] + (self.gap_size_range[1] - self.gap_size_range[0]) * difficulty
        _gap_terrain(terrain, gap_size=gap_size, platform_size=self.platform_size)
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class PitTerrain:
    depth_range: tuple[float, float] = (0.0, 1.0)
    platform_size: float = 4.0
    terrain_type: int = 7

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        depth = self.depth_range[0] + (self.depth_range[1] - self.depth_range[0]) * difficulty
        _pit_terrain(terrain, depth=depth, platform_size=self.platform_size)
        return SubTerrainResult(terrain, self.terrain_type)


def _gap_terrain(terrain, gap_size, platform_size=1.0):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x - x2 : center_x + x2, center_y - y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x - x1 : center_x + x1, center_y - y1 : center_y + y1] = 0


def _pit_terrain(terrain, depth, platform_size=1.0):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

