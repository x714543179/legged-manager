from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.modules import HiddenState, MLP
from rsl_rl.utils import unpad_trajectories


class DreamWaQActorBackbone(nn.Module):
    """DreamWaQ actor backbone compatible with the ActorModel shell."""

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        actor_hidden_dims=(512, 256, 128),
        encoder_hidden_dims=(128, 64),
        decoder_hidden_dims=(64, 128),
        latent_dim=19,
        velocity_dim=3,
        decoder_output_group="actor",
        velocity_target_group="prev_critic_base_lin_vel",
        activation="elu",
        autoencoder_loss_coef=1.0,
        velocity_loss_coef=1.0,
        reconstruction_loss_coef=1.0,
        kl_loss_coef=1.0,
        **_,
    ) -> None:
        super().__init__()
        self.obs_groups = obs_groups[obs_set]
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.velocity_dim = velocity_dim
        self.decoder_output_group = decoder_output_group
        self.velocity_target_group = velocity_target_group
        self.autoencoder_loss_coef = autoencoder_loss_coef
        self.velocity_loss_coef = velocity_loss_coef
        self.reconstruction_loss_coef = reconstruction_loss_coef
        self.kl_loss_coef = kl_loss_coef

        self.actor_obs_dim = self._obs_dim(obs, self.obs_groups)
        actor_input_dim = self.actor_obs_dim + latent_dim
        history_dim = obs["history"].shape[-1]
        decoder_output_dim = obs[decoder_output_group].shape[-1]

        encoder_output_dim = encoder_hidden_dims[-1]
        latent_only_dim = latent_dim - velocity_dim
        if latent_only_dim <= 0:
            raise ValueError("latent_dim must be greater than velocity_dim.")

        self.actor = MLP(actor_input_dim, output_dim, actor_hidden_dims, activation)
        self.encoder = MLP(history_dim, encoder_output_dim, encoder_hidden_dims[:-1], activation)
        self.encode_mean_latent = nn.Linear(encoder_output_dim, latent_only_dim)
        self.encode_logvar_latent = nn.Linear(encoder_output_dim, latent_only_dim)
        self.encode_mean_vel = nn.Linear(encoder_output_dim, velocity_dim)
        self.encode_logvar_vel = nn.Linear(encoder_output_dim, velocity_dim)
        self.decoder = MLP(latent_dim, decoder_output_dim, decoder_hidden_dims, activation)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        train_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        code, code_vel, decode, _, _, mean_latent, logvar_latent = self.cenet_forward(obs["history"])
        actor_obs = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        actions = self.actor(torch.cat((code, actor_obs), dim=-1))
        output = {"actions": actions}
        if train_mode:
            output["aux_losses"] = {
                "autoencoder": self._autoencoder_loss(
                    code_vel,
                    decode,
                    mean_latent,
                    logvar_latent,
                    obs,
                )
            }
        return output

    def cenet_forward(self, obs_history):
        distribution = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = self.encode_logvar_latent(distribution)
        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = self.encode_logvar_vel(distribution)
        code_latent = self._reparameterize(mean_latent, logvar_latent)
        code_vel = self._reparameterize(mean_vel, logvar_vel)
        code = torch.cat((code_vel, code_latent), dim=-1)
        decode = self.decoder(code)
        return code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        pass

    def as_jit(self) -> nn.Module:
        return _DreamWaQJitWrapper(self)

    @staticmethod
    def _obs_dim(obs: TensorDict, groups: list[str]) -> int:
        return sum(obs[group].shape[-1] for group in groups)

    @staticmethod
    def _reparameterize(mean, logvar):
        std = torch.exp(logvar * 0.5)
        return mean + std * torch.randn_like(std)

    def _autoencoder_loss(self, code_vel, decode, mean_latent, logvar_latent, obs):
        vel_target = obs[self.velocity_target_group][..., : self.velocity_dim].detach()
        decode_target = obs[self.decoder_output_group].detach()
        velocity_loss = F.mse_loss(code_vel, vel_target)
        reconstruction_loss = F.mse_loss(decode, decode_target)
        kl_loss = -0.5 * torch.sum(1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp())
        loss = (
            self.velocity_loss_coef * velocity_loss
            + self.reconstruction_loss_coef * reconstruction_loss
            + self.kl_loss_coef * kl_loss
        )
        return self.autoencoder_loss_coef * loss


class _DreamWaQJitWrapper(nn.Module):
    """TorchScript wrapper with the legacy DreamWaQ export signature."""

    def __init__(self, backbone: DreamWaQActorBackbone) -> None:
        super().__init__()
        self.actor = copy.deepcopy(backbone.actor)
        self.encoder = copy.deepcopy(backbone.encoder)
        self.encode_mean_latent = copy.deepcopy(backbone.encode_mean_latent)
        self.encode_logvar_latent = copy.deepcopy(backbone.encode_logvar_latent)
        self.encode_mean_vel = copy.deepcopy(backbone.encode_mean_vel)
        self.encode_logvar_vel = copy.deepcopy(backbone.encode_logvar_vel)

    def forward(self, observations: torch.Tensor, history_observations: torch.Tensor) -> torch.Tensor:
        distribution = self.encoder(history_observations)
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = self.encode_logvar_latent(distribution)
        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = self.encode_logvar_vel(distribution)
        code_latent = self._reparameterize(mean_latent, logvar_latent)
        code_vel = self._reparameterize(mean_vel, logvar_vel)
        return self.actor(torch.cat((code_vel, code_latent, observations), dim=-1))

    @staticmethod
    def _reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(logvar * 0.5)
        return mean + std * torch.randn_like(std)
