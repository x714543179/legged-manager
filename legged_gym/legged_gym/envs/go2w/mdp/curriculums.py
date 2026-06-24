"""Curriculum terms for the go2w task."""

from __future__ import annotations

import numpy as np
import torch


def terrain_levels(env, env_ids):
    if not env.init_done:
        return
    distance = torch.norm(env.root_states[env_ids, :2] - env.env_origins[env_ids, :2], dim=1)
    move_up = distance > env.terrain.env_length / 2
    move_down = (
        distance < torch.norm(env.commands[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    ) * ~move_up
    if hasattr(env, "terrain_importer"):
        env.terrain_importer.update_env_origins(env, env_ids, move_up, move_down)
        return
    env.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
    env.terrain_levels[env_ids] = torch.where(
        env.terrain_levels[env_ids] >= env.max_terrain_level,
        torch.randint_like(env.terrain_levels[env_ids], env.max_terrain_level),
        torch.clip(env.terrain_levels[env_ids], 0),
    )
    env.env_origins[env_ids] = env.terrain_origins[env.terrain_levels[env_ids], env.terrain_types[env_ids]]


def command_ranges(env, env_ids):
    if torch.mean(env.episode_sums["tracking_lin_vel"][env_ids]) / env.max_episode_length > (
        0.8 * env.reward_scales["tracking_lin_vel"]
    ):
        env.command_ranges["lin_vel_x"][0] = np.clip(
            env.command_ranges["lin_vel_x"][0] - 0.5, -env.cfg.commands.max_curriculum, 0.0
        )
        env.command_ranges["lin_vel_x"][1] = np.clip(
            env.command_ranges["lin_vel_x"][1] + 0.5, 0.0, env.cfg.commands.max_curriculum
        )
