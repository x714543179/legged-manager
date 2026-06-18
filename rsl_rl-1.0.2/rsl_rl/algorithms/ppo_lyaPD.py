# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules.LyapdActorCritic.lya_critic import LyaCritic
from rsl_rl.modules.LyapdActorCritic.DWAQpdActorCritic import DWAQPDActorCritic


from rsl_rl.storage import RolloutStorage

class LyaPDPPO:
    actor_critic: DWAQPDActorCritic
    def __init__(self,
                 actor_critic,
                 state_dim,
                 action_dim,
                 lyapunov_cfg,
                 lr_k_scale = 0.01,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.99,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        # k_lr = learning_rate * 0.01
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # LyaNet parameters
        self.alpha3 = lyapunov_cfg.get("alpha3", 0.05)
        self.lambda_lr = lyapunov_cfg.get("lambda_lr", 1e-4)
        self.lambda_tolerance = lyapunov_cfg.get("delta_tolerance", 0.01)
        self.lambda_ = torch.tensor(lyapunov_cfg.get("lambda_init", 1.0), device=self.device)
        self.lyapunov_critic = LyaCritic(state_dim).to(self.device)
        self.lyapunov_target = LyaCritic(state_dim).to(self.device)
        self.lyapunov_optimizer = torch.optim.Adam(self.lyapunov_critic.parameters(),
                                                   lr=lyapunov_cfg.get("lc_lr", 3e-4))
        self.soft_tau = lyapunov_cfg.get("lc_tau", 0.01)

        # PDnet的Lip parameters
        self.lr_k_scale = lr_k_scale
        # self._build_optimizer()

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, obs_hist_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, obs_hist_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, prev_critic_obs, obs_history):
        # if self.actor_critic.is_recurrent:
        #     self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values

        # self.transition.actions = self.actor_critic.act(obs,obs_history).detach()
        # 替换为：
        result = self.actor_critic.act(obs, obs_history)
        if isinstance(result, tuple):
            actions, grad_norm = result
            self.transition.actions = actions
            # self.transition.grad_norm = grad_norm
        else:
            self.transition.actions = result
            # 兼容非 Lipschitz 策略
            # if hasattr(self.transition, "grad_norm"):
                # self.transition.grad_norm = torch.zeros(1, device=self.device)

        with torch.inference_mode():
            self.transition.values = self.actor_critic.evaluate(critic_obs)
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.observation_history = obs_history
        self.transition.critic_observations = critic_obs
        self.transition.prev_critic_obs = prev_critic_obs
        # print("第0个机器人的动作是",torch.squeeze(self.transition.actions[0]))
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, next_obs, next_obs_hist, costs):

        self.transition.next_observations = next_obs
        self.transition.next_observations_hist = next_obs_hist
        self.transition.costs = costs

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    

    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self,beta=1):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_autoenc_loss = 0
        # if self.actor_critic.is_recurrent:
        #     generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        # else:
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, next_obs_batch, critic_obs_batch, prev_critic_obs_batch, obs_hist_batch, next_obs_hist_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, costs_batch, K_batch, hid_states_batch, masks_batch in generator:


                self.actor_critic.act(obs_batch, obs_hist_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                #Beta VAE loss
                code,code_vel,decode,mean_vel,logvar_vel,mean_latent,logvar_latent = self.actor_critic.cenet_forward(obs_hist_batch)
                
                vel_target = prev_critic_obs_batch[:,105:108]   # 73:76
                decode_target = obs_batch
                vel_target.requires_grad = False
                decode_target.requires_grad = False
                autoenc_loss = (nn.MSELoss()(code_vel,vel_target) + nn.MSELoss()(decode,decode_target) + beta*(-0.5 * torch.sum(1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp())))/self.num_mini_batches
                # estimation_loss = (code[:,0:3] - prev_critic_obs_batch[:,45:48]).pow(2).mean()
                # reconst_loss = (decode - obs_batch).pow(2).mean()
                # latent_loss = beta*(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()))/mean.shape[0]
                # autoenc_loss = estimation_loss + reconst_loss + latent_loss


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()



                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()


                # ====== Lyapunov 更新 ======

                with torch.no_grad():
                    L_next = self.lyapunov_target(next_obs_batch)
                L_curr = self.lyapunov_critic(obs_batch)
                c_s = costs_batch.detach()
                lyapunov_loss = F.mse_loss(L_curr, c_s + self.gamma * L_next)

                self.lyapunov_optimizer.zero_grad()
                lyapunov_loss.backward()
                self.lyapunov_optimizer.step()
                # === ② 计算 Actor 的约束项 (有梯度路径) ===
                for p in self.lyapunov_critic.parameters():
                    p.requires_grad = False

                L_next_pi = self.lyapunov_critic(next_obs_batch)              
                L_curr_detached = self.lyapunov_critic(obs_batch.detach()).detach()
                c_s_detached = c_s.detach()
                # Lyapunov 约束项：只对 actor 方向有梯度
                delta_L = L_next_pi - L_curr_detached + self.alpha3 * c_s_detached
                delta_L = torch.clamp(delta_L, -10, 10) 
                self.delta_L = delta_L
                for p in self.lyapunov_critic.parameters():
                    p.requires_grad = True


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()


                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + autoenc_loss + self.lambda_ * delta_L.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()


                # print("==== Gradient Check ====")
                # for name, param in self.actor_critic.named_parameters():
                #     if param.grad is None:
                #         print(f"{name:40s} | grad = None")
                #     else:
                #         print(f"{name:40s} | grad_norm = {param.grad.data.norm():.4e}")
                # === ✅ 打印梯度统计 ===
                # if hasattr(self.actor_critic, "lcn"):
                #     total_grad = 0.0
                #     count = 0
                #     for name, p in self.actor_critic.named_parameters():
                #         if p.grad is not None:
                #             grad_mean = p.grad.mean().item()
                #             grad_absmean = p.grad.abs().mean().item()
                #             if "lcn" in name:
                #                 print(f"[LCN] {name:40s} grad_mean={grad_mean:+.4e} | abs_mean={grad_absmean:.4e}")
                #             elif "actor" in name:
                #                 print(f"[ACTOR] {name:38s} grad_mean={grad_mean:+.4e} | abs_mean={grad_absmean:.4e}")
                #             elif "critic" in name:
                #                 print(f"[CRITIC] {name:37s} grad_mean={grad_mean:+.4e} | abs_mean={grad_absmean:.4e}")
                #             total_grad += grad_absmean
                #             count += 1
                #     if count > 0:
                #         print(f"🧠 mean(|grad|) across network = {total_grad / count:.4e}")
                #     print("-" * 100)

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()


                # 软更新目标网络
                for tp, p in zip(self.lyapunov_target.parameters(), self.lyapunov_critic.parameters()):
                    tp.data.copy_(self.soft_tau * p.data + (1 - self.soft_tau) * tp.data)

                self.lambda_ = torch.clamp(self.lambda_ + self.lambda_lr * (delta_L.mean() - self.lambda_tolerance), 0, 10)


                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_autoenc_loss += autoenc_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_autoenc_loss
    

    def get_LyaLoss(self):
        return self.lambda_.item(), self.delta_L.mean().item(), self.lambda_.item() * self.delta_L.mean().item()


    def _build_optimizer(self):
        """
        重新构建 Adam 优化器，给 LCN 单独学习率
        """
        if not hasattr(self.actor_critic.LipPDnet, "lcn"):
            # 如果当前网络没有 LCN，仍保持单一优化器
            print("⚠️ Warning: actor_critic has no LCN; using default optimizer.")
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
            return

        # ✅ 分组：LCN 单独学习率，其余保持原有 learning_rate
        lr_k = self.learning_rate * self.lr_k_scale


                # ✅ 拆出 LCN 参数，其余全部按默认学习率
        lcn_params = list(self.actor_critic.LipPDnet.lcn.parameters())
        other_params = [p for n, p in self.actor_critic.named_parameters() if not n.startswith("LipPDnet.lcn")]

        self.optimizer = torch.optim.Adam([
            {"params": other_params, "lr": self.learning_rate},     # actor、critic、log_std 等都在这里
            {"params": lcn_params,   "lr": lr_k},        # 单独设置 LCN 学习率
        ])


        for i, group in enumerate(self.optimizer.param_groups):
            print(f"Param group {i}: lr={group['lr']:.2e}, num_params={len(group['params'])}")
