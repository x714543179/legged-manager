from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from legged_gym.terrains.terrain_data import SubTerrainResult


def _height_units(value, terrain):
    return int(value / terrain.vertical_scale)


def _length_units(value, terrain):
    return int(value / terrain.horizontal_scale)


def _height_range(max_height, terrain, step=4):
    max_height = max(1, _height_units(max_height, terrain))
    return np.arange(1, max_height + 1, step=max(1, step), dtype=np.int16)


@dataclass
class ParkourStepTerrain:
    x_range: tuple[float, float] = (0.2, 0.4)
    hurdle_height_range: tuple[float, float] = (0.1, 0.2)
    platform_size: float = 1.5
    num_stones: int = 8
    terrain_type: int = 5

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        platform_size = _length_units(self.platform_size, terrain)
        dis_x_min = max(1, _length_units(self.x_range[0], terrain))
        dis_x_max = max(dis_x_min + 1, _length_units(self.x_range[1], terrain))
        height_min = _height_units(self.hurdle_height_range[0], terrain)
        height_max = max(height_min + 1, _height_units(self.hurdle_height_range[1], terrain))
        step_height = rng.integers(height_min, height_max)

        start_y = 2
        end_y = terrain.length - 2
        max_x = terrain.width + platform_size
        current_height = 0
        for _ in range(self.num_stones):
            rand_x = rng.integers(dis_x_min, dis_x_max)
            start_x = int(rng.integers(max(1, terrain.width // 2 - 5), max(2, terrain.width // 2)))
            current_height += step_height
            end_x = np.clip(start_x + rand_x, start_x, max_x)
            terrain.height_field_raw[start_x:end_x, start_y:end_y] = current_height
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class ParkourGapTerrain:
    depth: float = 0.5
    platform_size: float = 2.0
    terrain_type: int = 6

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        gap_size = np.clip(int((0.5 * difficulty if difficulty < 0.1 else 0.1 + difficulty) / terrain.horizontal_scale), 1, 13)
        depth = _height_units(self.depth, terrain)
        platform_size = _length_units(self.platform_size, terrain)
        start_y = 0
        end_y = int(terrain.length - platform_size / 8)
        start_x = platform_size
        center_x = terrain.width
        terrain.height_field_raw[start_x:center_x, start_y:end_y] = -depth
        terrain.height_field_raw[start_x + gap_size : center_x - gap_size, start_y + gap_size : end_y - gap_size] = 0
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class RampTerrain:
    platform_size: float = 2.0
    depth: float = 0.5
    slope_scale: float = 7.0
    terrain_type: int = 7

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        platform_size = _length_units(self.platform_size, terrain)
        terrain.height_field_raw[:, :] = -_height_units(self.depth, terrain)
        slope_start = 5
        slope_end = int((terrain.width - platform_size) / 2)
        height2width_ratio = 2 * int(difficulty * self.slope_scale + 1)
        xs = np.arange(slope_start, slope_end)
        max_height = height2width_ratio * max(1, slope_end - slope_start)
        heights = (height2width_ratio * (xs - slope_start)).clip(max=max_height).astype(np.int16)
        terrain.height_field_raw[slope_start:slope_end, :] = heights[:, None]
        x1 = slope_end
        x2 = int((terrain.width + platform_size) / 2)
        terrain.height_field_raw[x1:x2, :] = max_height
        slope_start = x2
        slope_end = terrain.width - 5
        xs = np.arange(slope_start, slope_end)
        heights = (height2width_ratio * (xs - slope_start)).clip(max=max_height).astype(np.int16)
        for idx, x in enumerate(range(slope_start, slope_end)):
            terrain.height_field_raw[x, :] = heights[len(heights) - idx - 1]
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class BeamTerrain:
    stone_size_range: tuple[float, float] = (1.0, 0.5)
    stone_distance: float = 0.1
    max_height_range: tuple[float, float] = (0.05, 0.23)
    platform_size: float = 2.0
    depth: float = 0.5
    terrain_type: int = 8

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        stone_size = self.stone_size_range[0] + (self.stone_size_range[1] - self.stone_size_range[0]) * difficulty
        beam_length = max(1, _length_units(stone_size, terrain))
        distance = 0.1 if difficulty < 0.2 else self.stone_distance + 0.4 * int(10 * difficulty) / 10
        beam_distance = np.clip(_length_units(distance, terrain), 1, 5)
        platform_size = _length_units(self.platform_size, terrain)
        height_values = _height_range(
            self.max_height_range[0] + (self.max_height_range[1] - self.max_height_range[0]) * difficulty,
            terrain,
        )
        terrain.height_field_raw[:, :] = -_height_units(self.depth, terrain)
        platform_y = terrain.length // 2 - platform_size // 2
        start_x = terrain.width // 2 - platform_size // 2 - 1
        while start_x >= 0:
            beam_width = int(rng.integers(15, 31))
            row_y = int(platform_y + platform_size / 2 - beam_width / 2)
            stop_x = max(0, start_x - beam_length)
            terrain.height_field_raw[stop_x:start_x, row_y : row_y + beam_width] = rng.choice(height_values)
            start_x -= beam_length + beam_distance
        start_x = terrain.width // 2 + platform_size // 2 + 1
        while start_x < terrain.width:
            beam_width = int(rng.integers(15, 31))
            row_y = int(platform_y + platform_size / 2 - beam_width / 2)
            stop_x = min(terrain.width, start_x + beam_length)
            terrain.height_field_raw[start_x:stop_x, row_y : row_y + beam_width] = rng.choice(height_values)
            start_x += beam_length + beam_distance
        x1 = terrain.width // 2 - platform_size // 2
        x2 = terrain.width // 2 + platform_size // 2
        terrain.height_field_raw[x1:x2, platform_y : platform_y + platform_size] = 0
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class HurdleTerrain:
    platform_size: float = 2.0
    depth: float = 0.6
    hurdle_height_range: tuple[float, float] = (0.1, 0.5)
    terrain_type: int = 15

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        platform_size = _length_units(self.platform_size, terrain)
        platform_y = (terrain.length - platform_size) // 2
        terrain.height_field_raw[:, :] = -_height_units(self.depth, terrain)
        row_y = int(platform_y + platform_size / 2 - 17 / 2)
        terrain.height_field_raw[:, row_y : row_y + 18] = 0
        h0, h1 = self.hurdle_height_range
        height_min = _height_units(h0, terrain)
        height_max = max(height_min + 1, _height_units(h1, terrain))
        step_height = rng.integers(height_min, height_max)
        start_x = platform_size + 20
        while start_x < terrain.width - 20:
            size_x = int(np.clip(12 + 18 * (1.0 - difficulty), 12, 30))
            stop_x = min(terrain.width, start_x + size_x)
            size_y = int(rng.integers(17, 30))
            y = int(platform_y + platform_size / 2 - size_y / 2)
            terrain.height_field_raw[start_x:stop_x, y : y + size_y] = step_height
            start_x += size_x + int(rng.integers(17, 30))
        return SubTerrainResult(terrain, self.terrain_type)


@dataclass
class AirStoneTerrain:
    depth: float = 0.6
    terrain_type: int = 14

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        terrain.height_field_raw[:] = -_height_units(self.depth, terrain)
        start_y, end_y = 20, terrain.length - 20
        terrain.height_field_raw[:, start_y:end_y] = 0
        center = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        return SubTerrainResult(terrain, self.terrain_type, {"goals_stone": center})


@dataclass
class NarrowCorridorTerrain:
    depth: float = 0.6
    platform_size: float = 2.0
    wall_height: float = 0.5
    terrain_type: int = 17

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        platform_size = _length_units(self.platform_size, terrain)
        terrain.height_field_raw[:] = -_height_units(self.depth, terrain)
        start_y, end_y = 20, terrain.length - 20
        terrain.height_field_raw[:, start_y:end_y] = 0
        wall_abs_height = _height_units(self.wall_height, terrain)
        center_y = terrain.length // 2
        narrow_gap = int(np.clip(8 - (8 - 2) * difficulty, 2, 8))
        terrain.height_field_raw[platform_size + 20 :, center_y + narrow_gap : end_y] = wall_abs_height
        terrain.height_field_raw[platform_size + 20 :, start_y : center_y - narrow_gap] = wall_abs_height
        center = np.array([1.0, 0.5, 0.32], dtype=np.float32)
        return SubTerrainResult(terrain, self.terrain_type, {"goals_narrow": center})
