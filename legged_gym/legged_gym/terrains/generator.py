from __future__ import annotations

import numpy as np
from isaacgym import terrain_utils

from .terrain_data import TerrainData, SubTerrainResult
from .registry import build_from_cfg, cfg_to_dict


class TerrainGenerator:
    """Build a tiled heightfield from pluggable sub-terrain generators."""

    def __init__(
        self,
        cfg,
        num_robots: int,
        sub_terrains: dict | None = None,
        difficulty_range: list[float] | tuple[float, float] = (0.0, 1.0),
        seed: int | None = None,
        **_,
    ) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.sub_terrains_cfg = sub_terrains
        self.difficulty_range = difficulty_range
        self.rng = np.random.default_rng(seed)

        self.mesh_type = getattr(cfg, "mesh_type", "trimesh")
        self.env_length = getattr(cfg, "terrain_length", getattr(cfg, "size", [8.0, 8.0])[0])
        self.env_width = getattr(cfg, "terrain_width", getattr(cfg, "size", [8.0, 8.0])[1])
        self.num_rows = getattr(cfg, "num_rows", 1)
        self.num_cols = getattr(cfg, "num_cols", 1)
        self.horizontal_scale = getattr(cfg, "horizontal_scale", 0.1)
        self.vertical_scale = getattr(cfg, "vertical_scale", 0.005)
        self.border_size = getattr(cfg, "border_size", 0.0)
        self.curriculum = getattr(cfg, "curriculum", False)
        self.flat_ratio = getattr(cfg, "flat_ratio", 0.0)
        self.tile_spacing = getattr(cfg, "terrain_spacing", 0.0)

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)
        self.tile_spacing_pixels = int(self.tile_spacing / self.horizontal_scale)
        self.tile_width_stride = self.width_per_env_pixels + self.tile_spacing_pixels
        self.tile_length_stride = self.length_per_env_pixels + self.tile_spacing_pixels
        self.border = int(self.border_size / self.horizontal_scale)
        self.tot_cols = int(self.num_cols * self.width_per_env_pixels + max(0, self.num_cols - 1) * self.tile_spacing_pixels) + 2 * self.border
        self.tot_rows = int(self.num_rows * self.length_per_env_pixels + max(0, self.num_rows - 1) * self.tile_spacing_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.env_origins = np.zeros((self.num_rows, self.num_cols, 3), dtype=np.float32)
        self.terrain_types = np.zeros((self.num_rows, self.num_cols), dtype=np.float32)
        self.extras: dict = {}
        self._extra_shapes: dict[str, tuple[int, ...]] = {}
        self.sub_terrains = self._build_sub_terrains()

    def generate(self) -> TerrainData:
        if self.mesh_type in ("none", "plane", None):
            return TerrainData(self.cfg, self.mesh_type, env_origins=self.env_origins)

        self._generate_tiles()
        data = TerrainData(
            cfg=self.cfg,
            mesh_type=self.mesh_type,
            height_samples=self.height_field_raw,
            env_origins=self.env_origins,
            terrain_types=self.terrain_types,
            extras=self.extras,
            env_length=self.env_length,
            env_width=self.env_width,
            tot_rows=self.tot_rows,
            tot_cols=self.tot_cols,
        )
        if self.mesh_type == "trimesh":
            data.vertices, data.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.horizontal_scale,
                self.vertical_scale,
                getattr(self.cfg, "slope_treshold", getattr(self.cfg, "slope_threshold", 0.75)),
            )
        return data

    def _generate_tiles(self) -> None:
        for col in range(self.num_cols):
            for row in range(self.num_rows):
                if self.flat_ratio > 0.0 and self.rng.random() < self.flat_ratio:
                    result = SubTerrainResult(self._flat_subterrain())
                else:
                    difficulty = self._difficulty(row)
                    term_name, term_cfg = self._select_sub_terrain(col)
                    result = self._make_subterrain(term_name, term_cfg, difficulty, row, col)
                self._add_terrain_to_map(result, row, col)

    def _build_sub_terrains(self):
        if self.sub_terrains_cfg:
            items = []
            for name, cfg in self.sub_terrains_cfg.items():
                cfg_dict = cfg_to_dict(cfg)
                proportion = float(cfg_dict.pop("proportion", 1.0))
                items.append((name, cfg_dict, proportion))
            return items
        return []

    def _select_sub_terrain(self, col: int):
        if not self.sub_terrains:
            return "legacy", None
        total = sum(proportion for _, _, proportion in self.sub_terrains)
        if self.curriculum:
            cursor = ((col + 0.5) / max(1, self.num_cols)) * total
        else:
            cursor = self.rng.uniform(0.0, total)
        acc = 0.0
        for name, cfg, proportion in self.sub_terrains:
            acc += proportion
            if cursor <= acc:
                return name, cfg
        name, cfg, _ = self.sub_terrains[-1]
        return name, cfg

    def _make_subterrain(self, name: str, cfg: dict | None, difficulty: float, row: int, col: int) -> SubTerrainResult:
        if cfg is None:
            return self._make_legacy_terrain(row, col, difficulty)
        cfg = dict(cfg)
        class_name = cfg.pop("class_name")
        generator = build_from_cfg({"class_name": class_name, **cfg})
        terrain = self._empty_subterrain(name)
        return generator.generate(terrain, difficulty=difficulty, row=row, col=col, rng=self.rng)

    def _make_legacy_terrain(self, row: int, col: int, difficulty: float) -> SubTerrainResult:
        raise NotImplementedError

    def _difficulty(self, row: int) -> float:
        lo, hi = self.difficulty_range
        if self.curriculum:
            alpha = row / max(1, self.num_rows - 1)
        else:
            alpha = self.rng.choice([0.5, 0.75, 0.9])
        return float(lo + (hi - lo) * alpha)

    def _empty_subterrain(self, name: str):
        return terrain_utils.SubTerrain(
            name,
            width=self.length_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.vertical_scale,
            horizontal_scale=self.horizontal_scale,
        )

    def _flat_subterrain(self):
        return self._empty_subterrain("flat")

    def _add_terrain_to_map(self, result: SubTerrainResult, row: int, col: int) -> None:
        terrain = result.terrain
        start_x = self.border + row * self.tile_length_stride
        end_x = start_x + self.length_per_env_pixels
        start_y = self.border + col * self.tile_width_stride
        end_y = start_y + self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = row * (self.env_length + self.tile_spacing) + 0.5 * self.env_length
        env_origin_y = col * (self.env_width + self.tile_spacing) + 0.5 * self.env_width
        x1 = int((self.env_length / 2.0 - 1.0) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2.0 + 1.0) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 1.0) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 1.0) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[row, col] = [env_origin_x, env_origin_y, env_origin_z]
        self.terrain_types[row, col] = result.terrain_type
        for key, value in result.extras.items():
            self._set_extra(key, value, row, col)

    def _set_extra(self, key: str, value, row: int, col: int) -> None:
        if isinstance(value, np.ndarray):
            value_array = value.astype(np.float32, copy=False)
        else:
            value_array = np.asarray(value, dtype=np.float32)
        if value_array.ndim == 1 and value_array.shape[0] >= 2:
            value_array = value_array.copy()
            value_array[0] += row * (self.env_length + self.tile_spacing)
            value_array[1] += col * (self.env_width + self.tile_spacing)

        if key not in self.extras:
            self._extra_shapes[key] = value_array.shape
            self.extras[key] = np.zeros((self.num_rows, self.num_cols, *value_array.shape), dtype=np.float32)
        if self._extra_shapes[key] != value_array.shape:
            raise ValueError(
                f"Terrain extra '{key}' has shape {value_array.shape}, "
                f"but previous tiles used {self._extra_shapes[key]}."
            )
        self.extras[key][row, col] = value_array
