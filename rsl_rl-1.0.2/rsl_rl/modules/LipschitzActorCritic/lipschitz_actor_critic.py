# ============================================================
# LipschitzActorCritic.py —— 修复版（可直接替换）
# 变更：
# 1) cenet_forward()：对 obs_history 做 clone().detach()，避免 inference tensor 参与反向图保存
# 2) _lipschitz_mean()：防御性 clone + requires_grad_(True)，并返回 grad_norm
# 3) update_distribution()：返回 grad_norm
# 4) act()：返回 (action, grad_norm) 供 PPO 存储
# 5) reset()：空实现以兼容框架
# ============================================================

from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Normal


# ---------- LCN: K(x) 网络（输出正数） ----------
class LipschitzConstantNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 128),     nn.Tanh(),
            nn.Linear(128, 1),       nn.Softplus()
        )
    def forward(self, x):
        return self.net(x) + 1e-6   # 防零


def get_activation(name):
    return {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid()
    }.get(name, nn.ELU())


class LipschitzActorCritic(nn.Module):
    """
    与 ActorCritic_DWAQ 完全同款接口：
      - cenet_forward(obs_history)  -> code, code_vel, decode, ...
      - act(observations, obs_history, deterministic_for_grad=False) -> (action, grad_norm)
      - get_actions_log_prob(actions)
      - act_inference(observations, obs_history)
      - evaluate(critic_observations)
      - 属性: action_mean, action_std, entropy
    但内部把 MGN: mean = 2*K*mu_raw/(||∇_obs mu_raw|| + eps) 融合为可导的策略均值。
    """
    def __init__(self,
                 num_actor_obs: int,
                 num_critic_obs: int,
                 num_actions: int,
                 cenet_in_dim: int,
                 cenet_out_dim: int,
                 activation: str = "elu",
                 init_noise_std: float = 1.0,
                 eps: float = 1e-6):
        super().__init__()

        self.act_fn = get_activation(activation)
        self.eps = eps

        # ---------- 与 DWAQ 相同的主体 ----------
        self.actor = nn.Sequential(
            nn.Linear(num_actor_obs, 512), self.act_fn,
            nn.Linear(512, 256),           self.act_fn,
            nn.Linear(256, 128),           self.act_fn,
            nn.Linear(128, num_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(num_critic_obs, 512), self.act_fn,
            nn.Linear(512, 256),            self.act_fn,
            nn.Linear(256, 128),            self.act_fn,
            nn.Linear(128, 1)
        )

        # ---------- VAE-like 编码器 ----------
        self.encoder = nn.Sequential(
            nn.Linear(cenet_in_dim, 128), self.act_fn,
            nn.Linear(128, 64),           self.act_fn,
        )
        self.encode_mean_latent   = nn.Linear(64, cenet_out_dim - 3)
        self.encode_logvar_latent = nn.Linear(64, cenet_out_dim - 3)
        self.encode_mean_vel      = nn.Linear(64, 3)
        self.encode_logvar_vel    = nn.Linear(64, 3)

        self.decoder = nn.Sequential(
            nn.Linear(cenet_out_dim, 64), self.act_fn,
            nn.Linear(64, 128),           self.act_fn,
            nn.Linear(128, 73)
        )

        # ---------- Lipschitz K(x) ----------
        # 注意：这里的 obs_dim = code_dim + obs_dim（与外部拼接保持一致）
        self.lcn = LipschitzConstantNet(num_actor_obs)

        # ---------- 分布参数（与 DWAQ 一致） ----------
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args = False

    # ===== DWAQ 同款工具 =====
    def reparameterise(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def cenet_forward(self, obs_history):
        # ✅ 关键修复：把上一轮 inference_mode 产生的 obs_history 转为普通张量
        #    这样 encoder 的中间保存不会被禁止（否则会报 Inference tensors cannot be saved for backward）
        obs_history = obs_history.clone().detach()

        h = self.encoder(obs_history)
        mean_latent   = self.encode_mean_latent(h)
        logvar_latent = self.encode_logvar_latent(h)
        mean_vel      = self.encode_mean_vel(h)
        logvar_vel    = self.encode_logvar_vel(h)

        code_latent = self.reparameterise(mean_latent, logvar_latent)
        code_vel    = self.reparameterise(mean_vel,    logvar_vel)
        code        = torch.cat((code_vel, code_latent), dim=-1)
        decode      = self.decoder(code)
        return code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent

    # ===== 关键点：把 MGN 融合成 “可导的策略均值” =====
    def _lipschitz_mean(self, obs_concat):
        """
        输入：obs_concat = concat(code, obs)，形状与 DWAQ 完全一致
        输出：smooth_mean, grad_norm
        """
        # ✅ 防御性：确保 obs_concat 可导且不是 inference tensor
        obs_concat = obs_concat.clone().detach().requires_grad_(True)


        # 1) 原始均值（不经平滑）
        mu_raw = self.actor(obs_concat)  # [B, A]

        # 2) ||∇_{obs_concat} mu_raw||
        grad = torch.autograd.grad(
            outputs=mu_raw,
            inputs=obs_concat,
            grad_outputs=torch.ones_like(mu_raw),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_norm = grad.norm(2, dim=1, keepdim=True)
        grad_norm = grad_norm.clamp(0.2, 5.0)  # ✅ 非原地操作

        # 3) K(x) 来自 LCN(obs_concat)
        K = self.lcn(obs_concat)

        # 4) MGN 平滑均值
        smooth_mean = 2.0 * K * mu_raw / (grad_norm + self.eps)
        return smooth_mean, grad_norm

    def update_distribution(self, observations_concat):
        """
        与 DWAQ 保持一致：传入的就是 concat(code, obs) 后的特征。
        但这里把分布的均值改成了“可导的 Lipschitz 均值”，并返回 grad_norm。
        """
        mean, grad_norm = self._lipschitz_mean(observations_concat)  # ✅ 可导
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        return grad_norm

    # ===== 与 DWAQ 完全一致的外部接口 =====
    def act(self, observations, obs_history, deterministic_for_grad=False, **kwargs):
        code, _, _, _, _, _, _ = self.cenet_forward(obs_history)
        obs_concat = torch.cat((code, observations), dim=-1)
        grad_norm = self.update_distribution(obs_concat)

        # 训练时：deterministic_for_grad=True 让 rsample 保持梯度
        action = self.distribution.rsample() if deterministic_for_grad else self.distribution.sample()

        # ✅ 同时返回 grad_norm，供 PPO 存到 storage
        return action, grad_norm.detach()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, obs_history):
        # 推理用均值（已包含 Lipschitz 平滑）
        code, _, _, _, _, _, _ = self.cenet_forward(obs_history)
        obs_concat = torch.cat((code, observations), dim=-1)
        mean, _ = self._lipschitz_mean(obs_concat)
        return mean

    def evaluate(self, critic_observations, **kwargs):
        critic_observations = critic_observations.clone().detach()
        return self.critic(critic_observations)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        """为兼容 PPO/RNN 接口，这里为空实现"""
        pass











# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ----------------------------------------------------------------------
# # 论文定义的 LCN 结构: 4-layer MLP, hidden [256,128], tanh activation
# # ----------------------------------------------------------------------
# class LipschitzConstantNet(nn.Module):
#     def __init__(self, obs_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(obs_dim, 256),
#             nn.Tanh(),
#             nn.Linear(256, 128),
#             nn.Tanh(),
#             nn.Linear(128, 1),
#             nn.Softplus()  # 保证输出为正
#         )

#     def forward(self, obs):
#         return self.net(obs) + 1e-6  # 防止为0
        

# class LipschitzActorCritic(nn.Module):
#     """Actor-Critic with MGN + LCN structure"""
#     def __init__(self, base_actor_critic, k_obs):
#         super().__init__()
#         self.actor_critic = base_actor_critic
#         self.actor = base_actor_critic.actor
#         self.critic = base_actor_critic.critic

#         self.lcn = LipschitzConstantNet(k_obs)






#     def forward(self, obs):
#         raw_action = self.actor(obs)
#         # compute gradient norm
#         obs.requires_grad_(True)
#         grad = torch.autograd.grad(
#             outputs=raw_action,
#             inputs=obs,
#             grad_outputs=torch.ones_like(raw_action),
#             create_graph=True,
#             retain_graph=True,
#             only_inputs=True
#         )[0]
#         grad_norm = grad.norm(2, dim=1, keepdim=True)
#         K = self.lcn(obs)
#         smoothed_action = K * raw_action / (grad_norm + 1e-6)
#         return smoothed_action
