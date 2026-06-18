"""Action terms for the go2w task."""

from __future__ import annotations


def command_latency(env, actions, enabled=True, randomize=True, latency_range=(1, 3)):
    if enabled:
        max_latency = int(latency_range[1])
        env.cmd_action_latency_buffer[:, :, 1:] = env.cmd_action_latency_buffer[:, :, :max_latency].clone()
        env.cmd_action_latency_buffer[:, :, 0] = actions.clone()
        return env.cmd_action_latency_buffer[
            env._env_indices, :, env.cmd_action_latency_simstep.long()
        ]
    return actions.clone()

