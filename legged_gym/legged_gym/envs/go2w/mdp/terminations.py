"""Termination terms for the go2w task."""

from __future__ import annotations

import torch


def base_height_contact(env):
    contact_flag = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    env.base_contact_buf = torch.any(contact_flag.unsqueeze(1) < 0.20, dim=1)
    return env.base_contact_buf
