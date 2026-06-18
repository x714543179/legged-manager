import torch
import torch.nn.functional as F
import torch.nn as nn
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.LyaCritic.lya_critic import LyaCritic
from rsl_rl.modules.LyaruActorCritic.Fheadnet import DisturbanceNet

class LyapunovRuPPO(PPO):
    def __init__(self, actor_critic, state_dim, action_dim, lyapunov_cfg, fhead_input_dim, **kwargs):
        super().__init__(actor_critic, **kwargs)
        self.alpha3 = lyapunov_cfg.get("alpha3", 0.05)
        self.alpha_d = lyapunov_cfg.get("alpha_d", 0.002)  # 扰动项权重（可调）
        self.lambda_lr = lyapunov_cfg.get("lambda_lr", 1e-4)
        self.lambda_tolerance = lyapunov_cfg.get("delta_tolerance", 0.01)
        self.lambda_ = torch.tensor(lyapunov_cfg.get("lambda_init", 1.0), device=self.device)
        self.lyapunov_critic = LyaCritic(state_dim, action_dim).to(self.device)
        self.lyapunov_target = LyaCritic(state_dim, action_dim).to(self.device)
        self.lyapunov_optimizer = torch.optim.Adam(self.lyapunov_critic.parameters(),
                                                   lr=lyapunov_cfg.get("lc_lr", 3e-4))
        self.soft_tau = lyapunov_cfg.get("lc_tau", 0.01)

        # 扰动力估计网络
        self.fhead_net = DisturbanceNet(fhead_input_dim).to(self.device)
        self.disturbance_optimizer = torch.optim.Adam(self.fhead_net.parameters(),lr=3e-4)

    def update(self,beta=1):
        mean_value_loss, mean_surrogate_loss, mean_autoenc_loss = 0, 0, 0
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
                
                vel_target = prev_critic_obs_batch[:,73:76]
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


                # ====== fhead 更新 ======
                # 外力目标 (若环境包含特权obs最后3维)
                f_hat = self.fhead_net(obs_hist_batch.detach())
                if critic_obs_batch.shape[1] >= 323:
                    f_target = critic_obs_batch[:, -3:]
                else:
                    f_target = torch.zeros_like(f_hat)

                fEst_loss = (0.5 * F.mse_loss(f_hat, f_target) )

                self.disturbance_optimizer.zero_grad()
                fEst_loss.backward()
                self.disturbance_optimizer.step()


                # ====== Lyapunov 更新 ======
                with torch.no_grad():
                    a_next = self.actor_critic.act(next_obs_batch, next_obs_hist_batch)
                    L_next = self.lyapunov_target(next_obs_batch, a_next)
                L_curr = self.lyapunov_critic(obs_batch, actions_batch)
                c_s = costs_batch.detach()
                lyapunov_loss = F.mse_loss(L_curr, c_s + self.gamma * L_next)

                self.lyapunov_optimizer.zero_grad()
                lyapunov_loss.backward()
                self.lyapunov_optimizer.step()


                # === ② 计算 Actor 的约束项 (有梯度路径) ===
                for p in self.lyapunov_critic.parameters():
                    p.requires_grad = False
                a_next_pi = self.actor_critic.act(next_obs_batch, next_obs_hist_batch)            # ✅ 不加 no_grad！
                L_next_pi = self.lyapunov_critic(next_obs_batch, a_next_pi)              
                L_curr_detached = self.lyapunov_critic(obs_batch.detach(), actions_batch.detach()).detach()
                c_s_detached = c_s.detach()

                # --- 扰动项：用估计的外力范数作为非负“坏项”加入约束 ---
                with torch.no_grad():
                    f_hat_det = self.fhead_net(obs_hist_batch.detach())   # (N,3)
                    d_term = self.alpha_d * torch.norm(f_hat_det, dim=1, keepdim=True)  # (N,1), >=0

                # Lyapunov 约束项：只对 actor 方向有梯度
                # print('当前的扰动项为：',d_term)
                delta_L = L_next_pi - L_curr_detached + self.alpha3 * c_s_detached - d_term
                delta_L = torch.clamp(delta_L, -10, 10) 
                self.delta_L = delta_L
                for p in self.lyapunov_critic.parameters():
                    p.requires_grad = True



                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

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