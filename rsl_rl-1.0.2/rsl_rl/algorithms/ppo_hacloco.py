import torch
import torch.nn as nn
from rsl_rl.algorithms.ppo import PPO

class HACLocoPPO(PPO):
    """
    PPO subclass compatible with ActorCritic_HACLow.
    Keeps all PPO losses identical to the original DreamWaQ implementation,
    only modifies the autoencoder (HAC-Low) term to use (z_t, f_hat, v_hat, o_hat_next).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, beta=1.0):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_autoenc_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        mse = nn.MSELoss()

        for obs_batch, next_obs_batch, critic_obs_batch, prev_critic_obs_batch, obs_hist_batch, next_obs_hist_batch, \
            actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, costs_batch, K_batch, hid_states_batch, masks_batch in generator:

            # === actor-critic 前向 ===
            self.actor_critic.act(obs_batch, obs_hist_batch,
                                  masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch,
                                                     masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # === KL 自适应学习率（保持不变） ===
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5)
                        + (torch.square(old_sigma_batch)
                           + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # === HAC-Low 自监督损失 ===
            z_t, f_hat, v_hat, o_hat_next = self.actor_critic.cenet_forward(obs_hist_batch)

            # 速度目标 (prev critic obs 中的 [73:76])
            v_target = prev_critic_obs_batch[:, 73:76]
            v_target.requires_grad = False

            # 外力目标 (若环境包含特权obs最后3维)
            if critic_obs_batch.shape[1] >= 323:
                f_target = critic_obs_batch[:, -3:]
            else:
                f_target = torch.zeros_like(f_hat)

            f_target.requires_grad = False
            decode_target = next_obs_batch.detach()

            # print("v_hat的损失是：",mse(v_hat, v_target))
            # print("f_hat的原损失是：",mse(f_hat, f_target))
            # print("f_hat的损失是：",mse(f_hat/100, f_target/100))
            # print("o_hat_next的损失是：",mse(o_hat_next, decode_target))
                  


            autoenc_loss = (
                mse(v_hat, v_target) +
                0.5 * mse(f_hat, f_target) +
                mse(o_hat_next, decode_target)
            ) / self.num_mini_batches   # 力除以100N，使各项量级接近

            # === PPO 核心部分（保持完全一致） ===
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss \
                   - self.entropy_coef * entropy_batch.mean() \
                   + beta * autoenc_loss * 0.2

            # === 反向传播与优化 ===
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_autoenc_loss += autoenc_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_autoenc_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_autoenc_loss
