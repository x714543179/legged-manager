"""Command terms for the go2w task."""

from __future__ import annotations

import torch
from isaacgym.torch_utils import torch_rand_float


def resample_commands(env, env_ids):
    if len(env_ids) == 0:
        return
    env.commands[env_ids, 0] = torch_rand_float(
        env.command_ranges["lin_vel_x"][0], env.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=env.device
    ).squeeze(1)
    env.commands[env_ids, 1] = torch_rand_float(
        env.command_ranges["lin_vel_y"][0], env.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=env.device
    ).squeeze(1)
    if env.cfg.commands.heading_command:
        env.commands[env_ids, 3] = torch_rand_float(
            env.command_ranges["heading"][0], env.command_ranges["heading"][1], (len(env_ids), 1), device=env.device
        ).squeeze(1)
    else:
        env.commands[env_ids, 2] = torch_rand_float(
            env.command_ranges["ang_vel_yaw"][0],
            env.command_ranges["ang_vel_yaw"][1],
            (len(env_ids), 1),
            device=env.device,
        ).squeeze(1)
    env.commands[env_ids, :2] *= (torch.norm(env.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

