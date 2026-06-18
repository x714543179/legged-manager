# ============================================================
# PDGainLipschitzNet —— 独立 PD 修正网络（带 MGN + LCN Lipschitz 操作）
# 与你上传的 lipschitz_actor_critic.py 完全一致逻辑，只是抽离成独立网络。
# ============================================================

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
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
                 num_actions: int,
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

        # ---------- Lipschitz K(x) ----------
        # 注意：这里的 obs_dim = code_dim + obs_dim（与外部拼接保持一致）
        # self.lcn = LipschitzConstantNet(num_actor_obs)

        # ---------- 分布参数（与 DWAQ 一致） ----------
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args = False

    # ===== DWAQ 同款工具 =====
    def reparameterise(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)


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

        mean = self.actor(observations_concat)
        # std = torch.ones_like(mean) * self.std
        # self.distribution = Normal(mean, std)

        self.distribution = Normal(mean, mean * 0.0 + self.std)



        # mean, grad_norm = self._lipschitz_mean(observations_concat)  # ✅ 可导
        # self.distribution = Normal(mean, mean * 0.0 + self.std)
        # return grad_norm

    # ===== 与 DWAQ 完全一致的外部接口 =====
    def act(self, obs_concat, deterministic_for_grad=False, **kwargs):

        self.update_distribution(obs_concat)
        action = self.distribution.sample()

        return action


        # grad_norm = self.update_distribution(obs_concat)

        # 训练时：deterministic_for_grad=True 让 rsample 保持梯度
        # action = self.distribution.rsample() if deterministic_for_grad else self.distribution.sample()

        # ✅ 同时返回 grad_norm，供 PPO 存到 storage
        # return action, grad_norm.detach()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_concat):
        # 推理用均值（已包含 Lipschitz 平滑）
        mean = self.actor(obs_concat)



        # mean, _ = self._lipschitz_mean(obs_concat)
        return mean


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




class DWAQPDActorCritic(nn.Module):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, cenet_in_dim, cenet_out_dim, PDnet_out_dim, activation="elu", init_noise_std=1.0,):
        super().__init__()

        self.activation = get_activation(activation)
        actor_input_dim = num_actor_obs
        critic_input_dim = num_critic_obs

        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim,512),
            self.activation,
            nn.Linear(512,256),
            self.activation,
            nn.Linear(256,128),
            self.activation,
            nn.Linear(128,num_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim,512),
            self.activation,
            nn.Linear(512,256),
            self.activation,
            nn.Linear(256,128),
            self.activation,
            nn.Linear(128,1)
        )

        self.encoder = nn.Sequential(
            nn.Linear(cenet_in_dim,128),
            self.activation,
            nn.Linear(128,64),
            self.activation,
        )
        self.encode_mean_latent = nn.Linear(64,cenet_out_dim-3)
        self.encode_logvar_latent = nn.Linear(64,cenet_out_dim-3)
        self.encode_mean_vel = nn.Linear(64,3)
        self.encode_logvar_vel = nn.Linear(64,3)

        self.decoder = nn.Sequential(
            nn.Linear(cenet_out_dim,64),
            self.activation,
            nn.Linear(64,128),
            self.activation,
            nn.Linear(128,num_actor_obs - cenet_out_dim)
        )

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False


        self.LipPDnet = LipschitzActorCritic(num_actor_obs, PDnet_out_dim)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def reparameterise(self,mean,logvar):
        var = torch.exp(logvar*0.5)
        code_temp = torch.randn_like(var)
        code = mean + var*code_temp
        return code
    
    def cenet_forward(self,obs_history):
        obs_history = obs_history.clone().detach()
        distribution = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = self.encode_logvar_latent(distribution)
        # var = torch.exp(logvar_latent*0.5)
        # code_temp = torch.randn_like(var)
        # code = mean_latent + var*code_temp
        # print("latent : ",code[0])
        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = self.encode_mean_vel(distribution)
        code_latent = self.reparameterise(mean_latent,logvar_latent)
        code_vel = self.reparameterise(mean_vel,logvar_vel)
        code = torch.cat((code_vel,code_latent),dim=-1)
        decode = self.decoder(code)
        return code,code_vel,decode,mean_vel,logvar_vel,mean_latent,logvar_latent

    @property
    def action_mean(self):

        self._joint_mean = self.distribution.mean
        self._pd_mean = self.LipPDnet.action_mean
        self._full_mean = torch.cat([self._joint_mean, self._pd_mean], dim=-1)

        return self._full_mean

    @property
    def action_std(self):
        joint_std = self.distribution.stddev
        pd_std = self.LipPDnet.action_std
        return torch.cat([joint_std, pd_std], dim=-1)

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        # std = torch.ones_like(mean) * self.std
        # self.distribution = Normal(mean, std)

        self.distribution = Normal(mean, mean * 0.0 + self.std)

        

    def act(self, observations, obs_history, deterministic_for_grad=False, **kwargs,):
        code,_,decode,_,_,_,_ = self.cenet_forward(obs_history)
        observations = torch.cat((code,observations),dim=-1)
        self.update_distribution(observations)

        pd_action  = self.LipPDnet.act(observations)

        if deterministic_for_grad:
            # ✅ 返回均值而不是采样，保持可导
            joint_action = self.distribution.rsample()
        else:
            joint_action = self.distribution.sample()
            # PPO 正常采样

        full_action = torch.cat([joint_action,pd_action],dim=-1)




        return full_action




    def get_actions_log_prob(self, actions):
        joint_actions, pd_actions = self.split_action(actions)
        log_prob_joint = self.distribution.log_prob(joint_actions).sum(dim=-1)
        log_prob_pd = self.LipPDnet.get_actions_log_prob(pd_actions)

        log_prob = log_prob_joint + log_prob_pd
        return log_prob
    
    def split_action(self, full_action):
        """
        full_action: (..., num_actions + PDnet_out_dim)
        return: joint_action, pd_action
        """
        joint = full_action[..., :self.std.numel()]  # num_actions
        pd    = full_action[..., self.std.numel():]
        return joint, pd
    

    def act_inference(self, observations,obs_history):
        code,_,decode,_,_,_,_ = self.cenet_forward(obs_history)
        observations = torch.cat((code,observations),dim=-1)
        actions_mean = self.actor(observations)
        pd_action_mean = self.LipPDnet.act_inference(observations)

        full_action = torch.cat([actions_mean,pd_action_mean],dim=-1)
        return full_action

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


