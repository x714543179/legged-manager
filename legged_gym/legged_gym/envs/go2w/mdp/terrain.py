"""Terrain helper terms for the go2w task."""

from __future__ import annotations

import torch

from legged_gym.utils.math import quat_apply_yaw


def init_height_points(env):
    y = torch.tensor(env.cfg.terrain.measured_points_y, device=env.device, requires_grad=False)
    x = torch.tensor(env.cfg.terrain.measured_points_x, device=env.device, requires_grad=False)
    grid_x, grid_y = torch.meshgrid(x, y)
    env.num_height_points = grid_x.numel()
    points = torch.zeros(env.num_envs, env.num_height_points, 3, device=env.device, requires_grad=False)
    points[:, :, 0] = grid_x.flatten()
    points[:, :, 1] = grid_y.flatten()
    return points


def get_heights(env, env_ids=None):
    if env.cfg.terrain.mesh_type == "plane":
        return torch.zeros(env.num_envs, env.num_height_points, device=env.device, requires_grad=False)
    if env.cfg.terrain.mesh_type == "none":
        raise NameError("Can't measure height with terrain mesh type 'none'")

    if env_ids is not None and len(env_ids) > 0:
        points = quat_apply_yaw(
            env.base_quat[env_ids].repeat(1, env.num_height_points), env.height_points[env_ids]
        ) + env.root_states[env_ids, :3].unsqueeze(1)
    else:
        points = quat_apply_yaw(env.base_quat.repeat(1, env.num_height_points), env.height_points)
        points += env.root_states[:, :3].unsqueeze(1)

    points += env.terrain.cfg.border_size
    points = (points / env.terrain.cfg.horizontal_scale).long()
    px = points[:, :, 0].view(-1)
    py = points[:, :, 1].view(-1)
    px = torch.clip(px, 0, env.height_samples.shape[0] - 2)
    py = torch.clip(py, 0, env.height_samples.shape[1] - 2)

    heights1 = env.height_samples[px, py]
    heights2 = env.height_samples[px + 1, py]
    heights3 = env.height_samples[px, py + 1]
    heights = torch.min(torch.min(heights1, heights2), heights3)
    return heights.view(env.num_envs, -1) * env.terrain.cfg.vertical_scale
