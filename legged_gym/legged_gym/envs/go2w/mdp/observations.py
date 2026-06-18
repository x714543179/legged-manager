"""Observation terms for the go2w task."""

from __future__ import annotations

import torch
from isaacgym.torch_utils import quat_rotate_inverse

from .utils import manager_term_params


def imu(env, latency_enabled=True, randomize_latency=True, latency_range=(1, 3)):
    if latency_enabled:
        env.obs_imu = env.obs_imu_latency_buffer[
            env._env_indices, :, env.obs_imu_latency_simstep.long()
        ]
    else:
        env.obs_imu = torch.cat(
            (env.base_ang_vel * env.obs_scales.ang_vel, env.base_euler_xyz * env.obs_scales.quat),
            dim=1,
        )
    return env.obs_imu


def motor(env, latency_enabled=True, randomize_latency=True, latency_range=(1, 3)):
    env.dof_err = env.dof_pos - env.default_dof_pos
    env.dof_err[:, env.wheel_indices] = 0
    q = env.dof_err * env.obs_scales.dof_pos
    dq = env.dof_vel * env.obs_scales.dof_vel
    if latency_enabled:
        env.obs_motor = env.obs_motor_latency_buffer[
            env._env_indices, :, env.obs_motor_latency_simstep.long()
        ]
    else:
        env.obs_motor = torch.cat((q, dq), dim=1)
    return env.obs_motor


def dof_pos(env):
    dof_pos = env.dof_pos.clone()
    dof_pos[:, env.wheel_indices] = 0
    return dof_pos


def imu_noise(env, term_value):
    noise_scales = env.cfg.noise.noise_scales
    noise_level = env.cfg.noise.noise_level
    ang_vel_noise = torch.full(
        (3,),
        noise_scales.ang_vel * noise_level * env.obs_scales.ang_vel,
        device=env.device,
        dtype=term_value.dtype,
    )
    quat_noise = torch.full(
        (3,),
        noise_scales.quat,
        device=env.device,
        dtype=term_value.dtype,
    )
    return torch.cat((ang_vel_noise, quat_noise), dim=0)


def motor_noise(env, term_value):
    noise_scales = env.cfg.noise.noise_scales
    noise_level = env.cfg.noise.noise_level
    dof_pos_noise = torch.full(
        (env.num_actions,),
        noise_scales.dof_pos * noise_level * env.obs_scales.dof_pos,
        device=env.device,
        dtype=term_value.dtype,
    )
    dof_vel_noise = torch.full(
        (env.num_actions,),
        noise_scales.dof_vel * noise_level * env.obs_scales.dof_vel,
        device=env.device,
        dtype=term_value.dtype,
    )
    return torch.cat((dof_pos_noise, dof_vel_noise), dim=0)


def dof_pos_noise(env, term_value):
    return torch.full(
        (term_value.shape[-1],),
        env.cfg.noise.noise_scales.dof_pos * env.cfg.noise.noise_level * env.obs_scales.dof_pos,
        device=env.device,
        dtype=term_value.dtype,
    )


def update_latency_buffers(env, env_ids=None):
    motor_latency_params = manager_term_params(env, "observation", "motor")
    imu_latency_params = manager_term_params(env, "observation", "imu")
    if motor_latency_params.get("latency_enabled", False):
        latency_range = motor_latency_params.get("latency_range", [1, 3])
        max_latency = int(latency_range[1])
        dof_error = env.dof_pos - env.default_dof_pos
        dof_error[:, env.wheel_indices] = 0
        q = dof_error * env.obs_scales.dof_pos
        dq = env.dof_vel * env.obs_scales.dof_vel
        env.obs_motor_latency_buffer[:, :, 1:] = env.obs_motor_latency_buffer[:, :, :max_latency].clone()
        env.obs_motor_latency_buffer[:, :, 0] = torch.cat((q, dq), dim=1).clone()
    if imu_latency_params.get("latency_enabled", False):
        latency_range = imu_latency_params.get("latency_range", [1, 3])
        max_latency = int(latency_range[1])
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.base_quat[:] = env.root_states[:, 3:7]
        env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
        env.base_euler_xyz = env.get_euler_xyz_tensor(env.base_quat)
        env.obs_imu_latency_buffer[:, :, 1:] = env.obs_imu_latency_buffer[:, :, :max_latency].clone()
        env.obs_imu_latency_buffer[:, :, 0] = torch.cat(
            (env.base_ang_vel * env.obs_scales.ang_vel, env.base_euler_xyz * env.obs_scales.quat),
            dim=1,
        ).clone()
