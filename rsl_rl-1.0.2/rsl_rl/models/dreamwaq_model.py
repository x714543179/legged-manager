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
        collision_dim=0,
        wheel_ground_dist_dim=0,
        decoder_output_group="actor",
        velocity_target_group="prev_critic_base_lin_vel",
        collision_target_group="prev_critic_collision_labels",
        wheel_ground_dist_target_group="prev_critic_wheel_ground_dist",
        velocity_target_scale=1.0,
        wheel_ground_dist_target_scale=1.0,
        activation="elu",
        autoencoder_loss_coef=1.0,
        velocity_loss_coef=1.0,
        collision_loss_coef=1.0,
        wheel_ground_dist_loss_coef=1.0,
        reconstruction_loss_coef=1.0,
        kl_loss_coef=1.0,
        kl_loss_reduction="sum",
        **_,
    ) -> None:
        super().__init__()
        self.obs_groups = obs_groups[obs_set]
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.velocity_dim = velocity_dim
        self.collision_dim = collision_dim
        self.wheel_ground_dist_dim = wheel_ground_dist_dim
        self.decoder_output_group = decoder_output_group
        self.velocity_target_group = velocity_target_group
        self.collision_target_group = collision_target_group
        self.wheel_ground_dist_target_group = wheel_ground_dist_target_group
        self.velocity_target_scale = velocity_target_scale
        self.wheel_ground_dist_target_scale = wheel_ground_dist_target_scale
        self.autoencoder_loss_coef = autoencoder_loss_coef
        self.velocity_loss_coef = velocity_loss_coef
        self.collision_loss_coef = collision_loss_coef
        self.wheel_ground_dist_loss_coef = wheel_ground_dist_loss_coef
        self.reconstruction_loss_coef = reconstruction_loss_coef
        self.kl_loss_coef = kl_loss_coef
        self.kl_loss_reduction = kl_loss_reduction

        self.actor_obs_dim = self._obs_dim(obs, self.obs_groups)
        actor_input_dim = self.actor_obs_dim + latent_dim
        history_dim = obs["history"].shape[-1]
        decoder_output_dim = obs[decoder_output_group].shape[-1]
        encoder_output_dim = encoder_hidden_dims[-1]

        self.actor = MLP(actor_input_dim, output_dim, actor_hidden_dims, activation)
        self.encoder = MLP(history_dim, encoder_output_dim, encoder_hidden_dims[:-1], activation)
        latent_only_dim = latent_dim - velocity_dim - collision_dim - wheel_ground_dist_dim
        if latent_only_dim <= 0:
            raise ValueError("latent_dim must be greater than velocity_dim + collision_dim + wheel_ground_dist_dim.")
        self.latent_only_dim = latent_only_dim
        self.encode_mean_latent = nn.Linear(encoder_output_dim, latent_only_dim)
        self.encode_logvar_latent = nn.Linear(encoder_output_dim, latent_only_dim)
        self.encode_mean_vel = nn.Linear(encoder_output_dim, velocity_dim)
        self.encode_logvar_vel = nn.Linear(encoder_output_dim, velocity_dim)
        if collision_dim > 0:
            self.encode_collision_logits = nn.Linear(encoder_output_dim, collision_dim)
        if wheel_ground_dist_dim > 0:
            self.encode_mean_wheel_ground_dist = nn.Linear(encoder_output_dim, wheel_ground_dist_dim)
            self.encode_logvar_wheel_ground_dist = nn.Linear(encoder_output_dim, wheel_ground_dist_dim)
        self.decoder = MLP(latent_dim, decoder_output_dim, decoder_hidden_dims, activation)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        train_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        code, aux, decode = self.cenet_forward(obs["history"])
        actor_obs = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        actions = self.actor(torch.cat((code, actor_obs), dim=-1))
        output = {"actions": actions}
        if train_mode:
            output["aux_losses"] = self._autoencoder_losses(aux, decode, obs)
        return output

    def cenet_forward(self, obs_history):
        distribution = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = torch.clamp(self.encode_logvar_latent(distribution), min=-5.0, max=5.0)
        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = torch.clamp(self.encode_logvar_vel(distribution), min=-5.0, max=5.0)
        code_latent = self._reparameterize(mean_latent, logvar_latent)
        code_vel = self._reparameterize(mean_vel, logvar_vel)
        code_terms = [code_vel]
        aux = {
            "code_vel": code_vel,
            "mean_latent": mean_latent,
            "logvar_latent": logvar_latent,
        }
        if self.collision_dim > 0:
            collision_logits = self.encode_collision_logits(distribution)
            collision_prob = torch.sigmoid(collision_logits)
            code_terms.append(collision_prob)
            aux["collision_logits"] = collision_logits
        if self.wheel_ground_dist_dim > 0:
            mean_wheel_ground_dist = self.encode_mean_wheel_ground_dist(distribution)
            logvar_wheel_ground_dist = torch.clamp(
                self.encode_logvar_wheel_ground_dist(distribution), min=-5.0, max=5.0
            )
            code_wheel_ground_dist = self._reparameterize(mean_wheel_ground_dist, logvar_wheel_ground_dist)
            code_terms.append(code_wheel_ground_dist)
            aux["code_wheel_ground_dist"] = code_wheel_ground_dist
        code_terms.append(code_latent)
        code = torch.cat(code_terms, dim=-1)
        decode = self.decoder(code)
        return code, aux, decode

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

    def _autoencoder_losses(self, aux, decode, obs):
        losses = {}
        vel_target = obs[self.velocity_target_group][..., : self.velocity_dim].detach() * self.velocity_target_scale
        decode_target = obs[self.decoder_output_group].detach()
        losses["velocity_estimation"] = self.velocity_loss_coef * F.mse_loss(aux["code_vel"], vel_target)
        if self.collision_dim > 0:
            collision_target = obs[self.collision_target_group][..., : self.collision_dim].detach()
            losses["collision_estimation"] = self.collision_loss_coef * F.binary_cross_entropy_with_logits(
                aux["collision_logits"], collision_target
            )
        if self.wheel_ground_dist_dim > 0:
            wheel_target = (
                obs[self.wheel_ground_dist_target_group][..., : self.wheel_ground_dist_dim].detach()
                * self.wheel_ground_dist_target_scale
            )
            losses["wheel_ground_dist_estimation"] = self.wheel_ground_dist_loss_coef * F.mse_loss(
                aux["code_wheel_ground_dist"], wheel_target
            )
        losses["reconstruction"] = self.reconstruction_loss_coef * F.mse_loss(decode, decode_target)
        mean_latent = aux["mean_latent"]
        logvar_latent = aux["logvar_latent"]
        kl_per_sample = torch.sum(1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp(), dim=-1)
        if self.kl_loss_reduction == "mean":
            kl_loss = -0.5 * torch.mean(kl_per_sample)
        elif self.kl_loss_reduction == "sum":
            kl_loss = -0.5 * torch.sum(kl_per_sample)
        else:
            raise ValueError(f"Unsupported kl_loss_reduction: {self.kl_loss_reduction}")
        losses["kl"] = self.kl_loss_coef * kl_loss
        return {name: self.autoencoder_loss_coef * loss for name, loss in losses.items()}


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
        self.collision_dim = backbone.collision_dim
        self.wheel_ground_dist_dim = backbone.wheel_ground_dist_dim
        if self.collision_dim > 0:
            self.encode_collision_logits = copy.deepcopy(backbone.encode_collision_logits)
        if self.wheel_ground_dist_dim > 0:
            self.encode_mean_wheel_ground_dist = copy.deepcopy(backbone.encode_mean_wheel_ground_dist)
            self.encode_logvar_wheel_ground_dist = copy.deepcopy(backbone.encode_logvar_wheel_ground_dist)

    def forward(self, observations: torch.Tensor, history_observations: torch.Tensor) -> torch.Tensor:
        distribution = self.encoder(history_observations)
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = torch.clamp(self.encode_logvar_latent(distribution), min=-5.0, max=5.0)
        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = torch.clamp(self.encode_logvar_vel(distribution), min=-5.0, max=5.0)
        code_latent = self._reparameterize(mean_latent, logvar_latent)
        code_vel = self._reparameterize(mean_vel, logvar_vel)
        code_terms = [code_vel]
        if self.collision_dim > 0:
            code_terms.append(torch.sigmoid(self.encode_collision_logits(distribution)))
        if self.wheel_ground_dist_dim > 0:
            mean_wheel_ground_dist = self.encode_mean_wheel_ground_dist(distribution)
            logvar_wheel_ground_dist = torch.clamp(
                self.encode_logvar_wheel_ground_dist(distribution), min=-5.0, max=5.0
            )
            code_terms.append(self._reparameterize(mean_wheel_ground_dist, logvar_wheel_ground_dist))
        code_terms.append(code_latent)
        code = torch.cat(code_terms, dim=-1)
        return self.actor(torch.cat((code, observations), dim=-1))

    @staticmethod
    def _reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(logvar * 0.5)
        return mean + std * torch.randn_like(std)
