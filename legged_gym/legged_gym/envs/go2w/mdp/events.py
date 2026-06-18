"""Event terms for the go2w task."""

from __future__ import annotations

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import torch_rand_float

from .utils import cfg_term_params, sequence_value


def randomize_friction(env, props=None, env_id=None, env_ids=None, enabled=True, friction_range=(0.2, 1.25)):
    if props is None or env_id is None or not enabled:
        return props
    if env_id == 0:
        num_buckets = 64
        bucket_ids = torch.randint(0, num_buckets, (env.num_envs, 1))
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device="cpu")
        env.friction_coeffs = friction_buckets[bucket_ids]

    for shape_prop in props:
        shape_prop.friction = env.friction_coeffs[env_id]
    return props


def randomize_rigid_body_props(
    env,
    props=None,
    env_id=None,
    env_ids=None,
    randomize_base_mass=True,
    added_mass_range=(-1.0, 2.0),
    randomize_link_mass=True,
    multiplied_link_mass_range=(0.9, 1.1),
    randomize_base_com=True,
    added_base_com_range=(-0.03, 0.03),
):
    if props is None or env_id is None:
        return props

    if randomize_base_mass:
        props[0].mass += np.random.uniform(added_mass_range[0], added_mass_range[1])

    if randomize_link_mass:
        env.multiplied_link_masses_ratio = torch_rand_float(
            multiplied_link_mass_range[0],
            multiplied_link_mass_range[1],
            (1, env.num_bodies - 1),
            device=env.device,
        )
        for body_id in range(1, len(props)):
            props[body_id].mass *= env.multiplied_link_masses_ratio[0, body_id - 1]

    if randomize_base_com:
        env.added_base_com = torch_rand_float(
            added_base_com_range[0], added_base_com_range[1], (1, 3), device=env.device
        )
        props[0].com += gymapi.Vec3(
            env.added_base_com[0, 0], env.added_base_com[0, 1], env.added_base_com[0, 2]
        )

    return props


def init_dof_props(
    env,
    props=None,
    env_id=None,
    env_ids=None,
    default_joint_friction=None,
    default_joint_stiffness=None,
    default_joint_damping=None,
    default_joint_armature=None,
):
    if props is None or env_id is None:
        return props

    if env_id == 0:
        env.dof_pos_limits = torch.zeros(env.num_dof, 2, dtype=torch.float, device=env.device, requires_grad=False)
        env.dof_vel_limits = torch.zeros(env.num_dof, dtype=torch.float, device=env.device, requires_grad=False)
        env.torque_limits = torch.zeros(env.num_dof, dtype=torch.float, device=env.device, requires_grad=False)
        for dof_id in range(len(props)):
            env.dof_pos_limits[dof_id, 0] = props["lower"][dof_id].item()
            env.dof_pos_limits[dof_id, 1] = props["upper"][dof_id].item()
            env.dof_vel_limits[dof_id] = props["velocity"][dof_id].item()
            env.torque_limits[dof_id] = props["effort"][dof_id].item()
            props["friction"][dof_id] = sequence_value(default_joint_friction, dof_id)
            props["stiffness"][dof_id] = sequence_value(default_joint_stiffness, dof_id)
            props["damping"][dof_id] = sequence_value(default_joint_damping, dof_id)
            props["armature"][dof_id] = sequence_value(default_joint_armature, dof_id)

            midpoint = (env.dof_pos_limits[dof_id, 0] + env.dof_pos_limits[dof_id, 1]) / 2
            dof_range = env.dof_pos_limits[dof_id, 1] - env.dof_pos_limits[dof_id, 0]
            env.dof_pos_limits[dof_id, 0] = midpoint - 0.5 * dof_range * env.cfg.rewards.soft_dof_pos_limit
            env.dof_pos_limits[dof_id, 1] = midpoint + 0.5 * dof_range * env.cfg.rewards.soft_dof_pos_limit
    return props


def randomize_motor_zero_offset(env, env_ids=None, env_id=None, enabled=True, offset_range=(-0.035, 0.035)):
    if not enabled:
        return
    if env_id is not None:
        env_ids = torch.tensor([env_id], device=env.device, dtype=torch.long)
    if env_ids is None or len(env_ids) == 0:
        return
    env.motor_zero_offsets[env_ids, :] = torch_rand_float(
        offset_range[0], offset_range[1], (len(env_ids), env.num_actions), device=env.device
    )


def randomize_pd_gains(
    env,
    env_ids=None,
    env_id=None,
    enabled=True,
    stiffness_multiplier_range=(0.9, 1.1),
    damping_multiplier_range=(0.9, 1.1),
):
    if not enabled:
        return
    if env_id is not None:
        env_ids = torch.tensor([env_id], device=env.device, dtype=torch.long)
    if env_ids is None or len(env_ids) == 0:
        return
    env.p_gains_multiplier[env_ids, :] = torch_rand_float(
        stiffness_multiplier_range[0],
        stiffness_multiplier_range[1],
        (len(env_ids), env.num_actions),
        device=env.device,
    )
    env.d_gains_multiplier[env_ids, :] = torch_rand_float(
        damping_multiplier_range[0],
        damping_multiplier_range[1],
        (len(env_ids), env.num_actions),
        device=env.device,
    )


def push_robots(env, env_ids=None, enabled=False, interval_s=15, max_vel_xy=1.0):
    if not enabled:
        return
    interval = max(1, int(np.ceil(interval_s / env.dt)))
    if env.common_step_counter % interval == 0:
        env.root_states[:, 7:9] = torch_rand_float(
            -max_vel_xy, max_vel_xy, (env.num_envs, 2), device=env.device
        )
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))


def update_height_measurements(env, env_ids=None):
    env._event_measure_heights(env_ids)


def reset_latency_buffers(env, env_ids=None):
    if env_ids is not None:
        action_params = cfg_term_params(env, "actions", "command_latency")
        if action_params.get("enabled", False):
            latency_range = action_params.get("latency_range", [1, 3])
            env.cmd_action_latency_buffer[env_ids, :, :] = 0.0
            if action_params.get("randomize", False):
                env.cmd_action_latency_simstep[env_ids] = torch.randint(
                    latency_range[0], latency_range[1] + 1, (len(env_ids),), device=env.device
                )
            else:
                env.cmd_action_latency_simstep[env_ids] = latency_range[1]
        else:
            env.cmd_action_latency_simstep[env_ids] = 0

        motor_params = cfg_term_params(env, "observations", "motor")
        if motor_params.get("latency_enabled", False):
            latency_range = motor_params.get("latency_range", [1, 3])
            env.obs_motor_latency_buffer[env_ids, :, :] = 0.0
            if motor_params.get("randomize_latency", False):
                env.obs_motor_latency_simstep[env_ids] = torch.randint(
                    latency_range[0], latency_range[1] + 1, (len(env_ids),), device=env.device
                )
            else:
                env.obs_motor_latency_simstep[env_ids] = latency_range[1]
        else:
            env.obs_motor_latency_simstep[env_ids] = 0

        imu_params = cfg_term_params(env, "observations", "imu")
        if imu_params.get("latency_enabled", False):
            latency_range = imu_params.get("latency_range", [1, 3])
            env.obs_imu_latency_buffer[env_ids, :, :] = 0.0
            if imu_params.get("randomize_latency", False):
                env.obs_imu_latency_simstep[env_ids] = torch.randint(
                    latency_range[0], latency_range[1] + 1, (len(env_ids),), device=env.device
                )
            else:
                env.obs_imu_latency_simstep[env_ids] = latency_range[1]
        else:
            env.obs_imu_latency_simstep[env_ids] = 0


def randomize_joint_friction(env, env_ids=None, enabled=False, friction_range=(0.9, 1.1)):
    if env_ids is None or len(env_ids) == 0:
        return
    resample_joint_friction(env, env_ids, enabled=enabled, friction_range=friction_range)
    refresh_actor_dof_props(env, env_ids)


def resample_joint_friction(env, env_ids, enabled=False, friction_range=(0.9, 1.1)):
    if len(env_ids) == 0:
        return
    if not enabled:
        env.joint_friction_coeffs[env_ids, :] = 1.0
        return
    env.joint_friction_coeffs[env_ids, :] = torch_rand_float(
        friction_range[0],
        friction_range[1],
        (len(env_ids), env.num_dof),
        device=env.device,
    )


def refresh_actor_dof_props(env, env_ids):
    if len(env_ids) == 0:
        return
    if isinstance(env_ids, torch.Tensor):
        env_ids = env_ids.tolist()

    dof_params = cfg_term_params(env, "events", "dof_props")
    default_joint_friction = dof_params.get("default_joint_friction", [0.0] * env.num_dof)
    joint_friction_params = cfg_term_params(env, "events", "joint_friction")
    randomize_friction = joint_friction_params.get("enabled", False)

    for env_id in env_ids:
        env_handle = env.envs[env_id]
        actor_handle = env.actor_handles[env_id]
        dof_props = env.gym.get_actor_dof_properties(env_handle, actor_handle)
        for dof_id in range(env.num_dof):
            default_friction = sequence_value(default_joint_friction, dof_id)
            if randomize_friction:
                dof_props["friction"][dof_id] = default_friction * env.joint_friction_coeffs[env_id, dof_id].item()
            else:
                dof_props["friction"][dof_id] = default_friction
        env.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
