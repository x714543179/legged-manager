from __future__ import annotations

import numpy as np
from isaacgym import gymapi
import torch

from .registry import build_from_cfg, cfg_to_dict


class TerrainImporter:
    """IsaacGym terrain importer with IsaacLab-style generator indirection."""

    def __init__(self, cfg, terrain_cfg, num_envs: int, device: str) -> None:
        self.cfg = cfg_to_dict(cfg)
        self.terrain_cfg = terrain_cfg
        self.num_envs = num_envs
        self.device = device
        self.terrain_type = self.cfg.get("terrain_type", "generator")
        self.mesh_type = self.cfg.get("mesh_type", getattr(terrain_cfg, "mesh_type", "trimesh"))
        self.max_init_terrain_level = self.cfg.get(
            "max_init_terrain_level", getattr(terrain_cfg, "max_init_terrain_level", 0)
        )
        self.use_terrain_origins = self.cfg.get("use_terrain_origins", True)
        self.terrain = None
        self.data = None

    def build(self):
        if self.mesh_type in ("none", "plane", None):
            return None
        if self.terrain_type != "generator":
            raise ValueError(f"Unsupported terrain_type '{self.terrain_type}'. Only 'generator' is implemented.")
        generator_cfg = self.cfg.get("generator")
        if generator_cfg is None:
            generator_cfg = {
                "class_name": "legged_gym.terrains.generators.rough:RoughTerrainGenerator",
            }
        generator = build_from_cfg(generator_cfg, self.terrain_cfg, self.num_envs)
        self.data = generator.generate()
        self.terrain = self.data
        return self.data

    def add_to_sim(self, gym, sim):
        if self.mesh_type == "plane":
            return self._add_ground_plane(gym, sim)
        if self.mesh_type == "heightfield":
            return self._add_heightfield(gym, sim)
        if self.mesh_type == "trimesh":
            return self._add_trimesh(gym, sim)
        if self.mesh_type in ("none", None):
            return
        raise ValueError(f"Terrain mesh type '{self.mesh_type}' is not supported.")

    def configure_env_origins(self, env):
        if self.mesh_type not in ("heightfield", "trimesh") or self.data is None or not self.use_terrain_origins:
            env.custom_origins = False
            env.env_origins = torch.zeros(env.num_envs, 3, device=env.device, requires_grad=False)
            num_cols = np.floor(np.sqrt(env.num_envs))
            num_rows = np.ceil(env.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = env.cfg.env.env_spacing
            env.env_origins[:, 0] = spacing * xx.flatten()[: env.num_envs]
            env.env_origins[:, 1] = spacing * yy.flatten()[: env.num_envs]
            env.env_origins[:, 2] = 0.0
            return

        env.custom_origins = True
        env.env_origins = torch.zeros(env.num_envs, 3, device=env.device, requires_grad=False)
        max_init_level = self.max_init_terrain_level
        if not getattr(env.cfg.terrain, "curriculum", False):
            max_init_level = env.cfg.terrain.num_rows - 1
        env.terrain_levels = torch.randint(0, max_init_level + 1, (env.num_envs,), device=env.device)
        env.terrain_types = torch.div(
            torch.arange(env.num_envs, device=env.device),
            (env.num_envs / env.cfg.terrain.num_cols),
            rounding_mode="floor",
        ).to(torch.long)
        env.max_terrain_level = env.cfg.terrain.num_rows
        env.terrain_origins = torch.from_numpy(self.data.env_origins).to(env.device).to(torch.float)
        env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
        if self.data.terrain_types is not None:
            env.terrain_class = torch.from_numpy(self.data.terrain_types).to(env.device).to(torch.float)
            env.env_class = env.terrain_class[env.terrain_levels, env.terrain_types]

    def update_env_origins(self, env, env_ids, move_up, move_down):
        env.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        env.terrain_levels[env_ids] = torch.where(
            env.terrain_levels[env_ids] >= env.max_terrain_level,
            torch.randint_like(env.terrain_levels[env_ids], env.max_terrain_level),
            torch.clip(env.terrain_levels[env_ids], 0),
        )
        env.env_origins[env_ids] = env.terrain_origins[env.terrain_levels[env_ids], env.terrain_types[env_ids]]
        if hasattr(env, "terrain_class"):
            env.env_class[env_ids] = env.terrain_class[env.terrain_levels[env_ids], env.terrain_types[env_ids]]

    def attach_to_env(self, env):
        env.terrain_importer = self
        env.terrain = self.data
        if self.data is None:
            return
        for key, value in self.data.extras.items():
            if isinstance(value, np.ndarray):
                setattr(env, key, torch.from_numpy(value).to(env.device))
            else:
                setattr(env, key, value)
        if self.data.height_samples is None:
            return
        env.height_samples = torch.tensor(self.data.height_samples).view(self.data.tot_rows, self.data.tot_cols).to(env.device)
        if self.data.terrain_types is not None:
            env.terrain_type = torch.from_numpy(self.data.terrain_types).to(env.device)

    def _add_ground_plane(self, gym, sim):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self._fill_material(plane_params)
        gym.add_ground(sim, plane_params)

    def _add_heightfield(self, gym, sim):
        if self.data is None:
            self.build()
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain_cfg.horizontal_scale
        hf_params.row_scale = self.terrain_cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain_cfg.vertical_scale
        hf_params.nbRows = self.data.tot_cols
        hf_params.nbColumns = self.data.tot_rows
        hf_params.transform.p.x = -self.terrain_cfg.border_size
        hf_params.transform.p.y = -self.terrain_cfg.border_size
        hf_params.transform.p.z = 0.0
        self._fill_material(hf_params)
        gym.add_heightfield(sim, self.data.height_samples, hf_params)

    def _add_trimesh(self, gym, sim):
        if self.data is None:
            self.build()
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.data.vertices.shape[0]
        tm_params.nb_triangles = self.data.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain_cfg.border_size
        tm_params.transform.p.y = -self.terrain_cfg.border_size
        tm_params.transform.p.z = 0.0
        self._fill_material(tm_params)
        gym.add_triangle_mesh(
            sim,
            self.data.vertices.flatten(order="C"),
            self.data.triangles.flatten(order="C"),
            tm_params,
        )

    def _fill_material(self, params):
        params.static_friction = getattr(self.terrain_cfg, "static_friction", 1.0)
        params.dynamic_friction = getattr(self.terrain_cfg, "dynamic_friction", 1.0)
        params.restitution = getattr(self.terrain_cfg, "restitution", 0.0)


def get_importer_cfg(terrain_cfg):
    importer_cfg = getattr(terrain_cfg, "importer", None)
    if importer_cfg is None:
        return {
            "terrain_type": "generator",
            "mesh_type": getattr(terrain_cfg, "mesh_type", "trimesh"),
            "max_init_terrain_level": getattr(terrain_cfg, "max_init_terrain_level", 0),
            "use_terrain_origins": True,
            "generator": {
                "class_name": "legged_gym.terrains.generators.rough:RoughTerrainGenerator",
            },
        }
    return cfg_to_dict(importer_cfg)
