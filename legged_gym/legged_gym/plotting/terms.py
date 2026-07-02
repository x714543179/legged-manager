import re
from typing import Iterable, List, Optional

import torch


def joint_state(env, env_index=0, joint_index=0, joint_name=None, key_prefix="joint"):
    """Return position, velocity and torque for one joint."""
    joint_id, _ = _resolve_joint(env, joint_index=joint_index, joint_name=joint_name)
    values = {
        f"{key_prefix}_pos": _scalar(env.dof_pos[env_index, joint_id]),
        f"{key_prefix}_vel": _scalar(env.dof_vel[env_index, joint_id]),
    }
    if hasattr(env, "torques"):
        values[f"{key_prefix}_torque"] = _scalar(env.torques[env_index, joint_id])
    return values


def joint_position(env, env_index=0, joint_index=0, joint_name=None, key="joint_pos"):
    """Return position for one joint."""
    joint_id, _ = _resolve_joint(env, joint_index=joint_index, joint_name=joint_name)
    return {key: _scalar(env.dof_pos[env_index, joint_id])}


def joint_velocity(env, env_index=0, joint_index=0, joint_name=None, key="joint_vel"):
    """Return velocity for one joint."""
    joint_id, _ = _resolve_joint(env, joint_index=joint_index, joint_name=joint_name)
    return {key: _scalar(env.dof_vel[env_index, joint_id])}


def joint_torque(env, env_index=0, joint_index=0, joint_name=None, key="joint_torque"):
    """Return torque for one joint."""
    joint_id, _ = _resolve_joint(env, joint_index=joint_index, joint_name=joint_name)
    return {key: _scalar(env.torques[env_index, joint_id])}


def joint_states(env, env_index=0, joint_indices=None, joint_names=None, key_prefix="joint"):
    """Return position, velocity and torque for multiple joints."""
    joints = _resolve_joints(env, joint_indices=joint_indices, joint_names=joint_names)
    values = {}
    for joint_id, joint_name in joints:
        key_name = _safe_key(joint_name)
        values[f"{key_prefix}_{key_name}_pos"] = _scalar(env.dof_pos[env_index, joint_id])
        values[f"{key_prefix}_{key_name}_vel"] = _scalar(env.dof_vel[env_index, joint_id])
        if hasattr(env, "torques"):
            values[f"{key_prefix}_{key_name}_torque"] = _scalar(env.torques[env_index, joint_id])
    return values


def _resolve_joints(env, joint_indices=None, joint_names=None):
    if joint_names is not None:
        return [_resolve_joint(env, joint_name=name) for name in _as_list(joint_names)]
    if joint_indices is not None:
        return [_resolve_joint(env, joint_index=index) for index in _as_list(joint_indices)]
    return [_resolve_joint(env, joint_index=index) for index in range(getattr(env, "num_dof", env.num_actions))]


def _resolve_joint(env, joint_index=None, joint_name=None):
    dof_names = list(getattr(env, "dof_names", []))
    if joint_name is not None:
        if joint_name in dof_names:
            joint_index = dof_names.index(joint_name)
        else:
            matches = [idx for idx, name in enumerate(dof_names) if joint_name in name]
            if len(matches) != 1:
                raise ValueError(f"Could not uniquely resolve joint name '{joint_name}'.")
            joint_index = matches[0]
    if joint_index is None:
        joint_index = 0
    joint_index = int(joint_index)
    joint_label = dof_names[joint_index] if joint_index < len(dof_names) else f"dof_{joint_index}"
    return joint_index, joint_label


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _safe_key(value):
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
