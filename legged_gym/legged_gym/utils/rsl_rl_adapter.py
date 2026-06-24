from __future__ import annotations

import torch


class RslRlVecEnvAdapter:
    """Adapt the legacy legged_gym VecEnv API to the newer rsl_rl VecEnv API."""

    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length
        self.device = env.device
        self.cfg = env.cfg
        self.num_obs = env.num_obs
        self.num_privileged_obs = env.num_privileged_obs
        self.num_obs_hist = env.num_obs_hist
        self.reset()

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_observations(self):
        actor_obs, obs_history = self.env.get_observations()
        privileged_obs, prev_privileged_obs = self.env.get_privileged_observations()
        if privileged_obs is None:
            privileged_obs = actor_obs
        if prev_privileged_obs is None:
            prev_privileged_obs = privileged_obs
        return self._make_obs_tensordict(actor_obs, privileged_obs, prev_privileged_obs, obs_history)

    def get_privileged_observations(self):
        return self.env.get_privileged_observations()

    def reset(self):
        self.env.reset()
        return self.get_observations()

    def step(self, actions: torch.Tensor):
        actor_obs, privileged_obs, prev_privileged_obs, obs_history, rewards, dones, extras = self.env.step(actions)
        if privileged_obs is None:
            privileged_obs = actor_obs
        if prev_privileged_obs is None:
            prev_privileged_obs = privileged_obs
        obs = self._make_obs_tensordict(actor_obs, privileged_obs, prev_privileged_obs, obs_history)
        return obs, rewards, dones, extras

    def _make_obs_tensordict(self, actor_obs, privileged_obs, prev_privileged_obs, obs_history):
        from tensordict import TensorDict

        obs_dict = {
            "actor": actor_obs,
            "critic": privileged_obs,
            "prev_critic": prev_privileged_obs,
            "history": obs_history,
        }
        obs_dict.update(getattr(self.env, "obs_term_bufs", {}))
        obs_dict.update(getattr(self.env, "privileged_obs_term_bufs", {}))
        obs_dict.update(self._previous_privileged_terms(prev_privileged_obs))
        return TensorDict(obs_dict, batch_size=[self.num_envs], device=self.device)

    def _previous_privileged_terms(self, prev_privileged_obs):
        observation_manager = getattr(self.env, "observation_manager", None)
        names = getattr(observation_manager, "_privileged_term_names", [])
        widths = getattr(observation_manager, "_privileged_term_widths", [])
        if not names or prev_privileged_obs is None:
            return {}
        chunks = torch.split(prev_privileged_obs, widths, dim=-1)
        return {f"prev_{name}": chunk for name, chunk in zip(names, chunks)}
