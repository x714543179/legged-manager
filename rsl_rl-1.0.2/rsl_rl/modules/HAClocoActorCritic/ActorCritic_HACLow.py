# rsl_rl/modules/ActorCritic_HACLow.py
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic_DWAQ import ActorCritic_DWAQ



class ActorCritic_HACLow(ActorCritic_DWAQ):
    """
    HAC-LOCO Stage 1: Low-level policy with additional velocity and force estimation heads.
    Structure strictly matches paper Fig.2:
        Encoder: [256,128,64]
        f_head: [32,16]
        v_head: [32,16]
        Decoder: [512,256,128]
    """
    def __init__(self, num_actor_obs, num_critic_obs, num_actions,
                 cenet_in_dim, cenet_out_dim, activation="elu", init_noise_std=1.0):
        super().__init__(num_actor_obs, num_critic_obs, num_actions,
                         cenet_in_dim, cenet_out_dim, activation, init_noise_std)

        # === Replace old VAE-style heads with paper-accurate heads ===
        hidden_f = 32
        hidden_v = 32

        self.f_head = nn.Sequential(
            nn.Linear(cenet_out_dim, hidden_f),
            self.activation,
            nn.Linear(hidden_f, 16),
            self.activation,
            nn.Linear(16, 3)   # 输出外力 (Fx,Fy,Fz)
        )

        self.v_head = nn.Sequential(
            nn.Linear(cenet_out_dim, hidden_v),
            self.activation,
            nn.Linear(hidden_v, 16),
            self.activation,
            nn.Linear(16, 3)   # 输出速度估计 (Vx,Vy,Vz)
        )

        # Decoder 维度保持一致 [512,256,128] → 输出维度=73 (DreamWaQ原有obs维度)
        self.decoder = nn.Sequential(
            nn.Linear(cenet_out_dim + 6, 512),  # z + f + v
            self.activation,
            nn.Linear(512, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, 73)
        )

    def cenet_forward(self, obs_history):
        """Encoder → f_head/v_head → concat → Decoder"""
        z_t = self.encoder(obs_history)           # [256,128,64]
        f_hat = self.f_head(z_t)                  # [32,16]→3
        v_hat = self.v_head(z_t)                  # [32,16]→3
        l_t = torch.cat([z_t, f_hat, v_hat], dim=-1)
        o_hat_next = self.decoder(l_t)
        return z_t, f_hat, v_hat, o_hat_next

    def act(self, observations, obs_history, deterministic_for_grad=False, **kwargs):
        z_t, f_hat, v_hat, o_hat_next = self.cenet_forward(obs_history)
        latent = torch.cat((z_t, f_hat, v_hat), dim=-1)
        obs_aug = torch.cat((latent, observations), dim=-1)
        self.update_distribution(obs_aug)
        # self.last_force_est = f_hat.detach()

        if deterministic_for_grad:
            return self.distribution.rsample()
        else:
            return self.distribution.sample()

    def act_inference(self, observations, obs_history):
        z_t, f_hat, v_hat, o_hat_next = self.cenet_forward(obs_history)
        latent = torch.cat((z_t, f_hat, v_hat), dim=-1)
        obs_aug = torch.cat((latent, observations), dim=-1)
        actions_mean = self.actor(obs_aug)
        return actions_mean