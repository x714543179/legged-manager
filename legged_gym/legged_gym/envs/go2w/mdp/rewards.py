"""Reward terms for the go2w task."""

from __future__ import annotations

import torch


def lin_vel_z(env):
    return torch.square(env.base_lin_vel[:, 2])


def ang_vel_xy(env):
    return torch.sum(torch.square(env.base_ang_vel[:, :2]), dim=1)


def orientation(env):
    return torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1)


def base_height(env):
    base_height_error = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    return torch.square(base_height_error - env.cfg.rewards.base_height_target)


def torques(env):
    return torch.sum(torch.square(env.torques), dim=1)


def dof_vel(env):
    joint_vel = env.dof_vel.clone()
    joint_vel[:, env.wheel_indices] = 0
    return torch.sum(torch.square(joint_vel), dim=1)


def dof_acc(env):
    return torch.sum(torch.square((env.last_dof_vel - env.dof_vel) / env.dt), dim=1)


def action_rate(env):
    return torch.sum(torch.square(env.last_actions - env.actions), dim=1)


def collision(env):
    return torch.sum(
        1.0 * (torch.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1) > 0.1),
        dim=1,
    )


def dof_pos_limits(env):
    out_of_limits = -(env.dof_pos - env.dof_pos_limits[:, 0]).clip(max=0.0)
    out_of_limits += (env.dof_pos - env.dof_pos_limits[:, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def dof_vel_limits(env):
    return torch.sum(
        (torch.abs(env.dof_vel) - env.dof_vel_limits * env.cfg.rewards.soft_dof_vel_limit).clip(min=0.0, max=1.0),
        dim=1,
    )


def torque_limits(env):
    return torch.sum(
        (torch.abs(env.torques) - env.torque_limits * env.cfg.rewards.soft_torque_limit).clip(min=0.0),
        dim=1,
    )


def tracking_lin_vel(env):
    lin_vel_error = torch.sum(torch.square(env.commands[:, :2] - env.base_lin_vel[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / env.cfg.rewards.tracking_sigma)


def tracking_ang_vel(env):
    ang_vel_error = torch.square(env.commands[:, 2] - env.base_ang_vel[:, 2])
    return torch.exp(-ang_vel_error / env.cfg.rewards.tracking_sigma)


def feet_air_time(env):
    contact = env.contact_forces[:, env.feet_indices, 2] > 1.0
    contact_filt = torch.logical_or(contact, env.last_contacts)
    env.last_contacts = contact
    first_contact = (env.feet_air_time > 0.0) * contact_filt
    env.feet_air_time += env.dt
    reward = torch.sum((env.feet_air_time - 0.5) * first_contact, dim=1)
    reward *= torch.norm(env.commands[:, :2], dim=1) > 0.1
    env.feet_air_time *= ~contact_filt
    return -reward


def feet_stumble(env):
    return torch.any(
        torch.norm(env.contact_forces[:, env.feet_indices, :2], dim=2)
        > 5 * torch.abs(env.contact_forces[:, env.feet_indices, 2]),
        dim=1,
    )


def stand_still(env):
    dof_err = env.dof_pos - env.default_dof_pos
    dof_err[:, env.wheel_indices] = 0
    return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(env.commands[:, :2], dim=1) < 0.1)


def feet_contact_forces(env):
    return torch.sum(
        (torch.norm(env.contact_forces[:, env.feet_indices, :], dim=-1) - env.cfg.rewards.max_contact_force).clip(min=0.0),
        dim=1,
    )


def orientation_quat(env):
    orientation_error = torch.sum(torch.square(env.root_states[:, :7] - env.base_init_state[0:7]), dim=1)
    return torch.exp(-orientation_error / env.cfg.rewards.tracking_sigma)


def hip_action_l2(env):
    return torch.sum(env.actions[:, [0, 4, 8, 12]] ** 2, dim=1)


def joint_power(env):
    return torch.sum(
        torch.abs(env.dof_vel[:, env.joint_indices]) * torch.abs(env.torques[:, env.joint_indices]),
        dim=1,
    )


def power_distribution(env):
    power = env.torques * env.dof_vel
    return torch.var(torch.abs(power), dim=1)


def lr_symmetry(env):
    left_ids = torch.tensor([0, 1, 2, 8, 9, 10], device=env.device)
    right_ids = torch.tensor([4, 5, 6, 12, 13, 14], device=env.device)
    mirror_sign = torch.tensor([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0], device=env.device)
    dof_pos_rel = env.dof_pos - env.default_dof_pos
    dof_pos_rel[:, env.wheel_indices] = 0
    sym_err = torch.mean((dof_pos_rel[:, left_ids] - mirror_sign * dof_pos_rel[:, right_ids]) ** 2, dim=1)
    return torch.exp(-10.0 * sym_err)


def default_pos(env):
    joint_pos = env.dof_pos.clone()
    joint_pos[:, env.wheel_indices] = 0
    return torch.sum(torch.abs(joint_pos - env.default_dof_pos), dim=1)
