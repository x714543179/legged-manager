from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms.plugins.base import PPOPlugin
from rsl_rl.algorithms.plugins.amp.discriminator import Discriminator
from rsl_rl.algorithms.plugins.amp.replay_buffer import ReplayBuffer
from rsl_rl.algorithms.plugins.amp.motion_loader import AMPLoader
from rsl_rl.algorithms.plugins.amp.normalizer import Normalizer


class AMPPlugin(PPOPlugin):
    """AMP（Adversarial Motion Priors）判别器插件。

    将判别器奖励整合进 PPO 训练：
    - rollout 期间：用判别器预测替换任务奖励
    - update 期间：同时优化策略损失和判别器损失（共享同一 optimizer）

    子类可覆盖的扩展点：
    - ``_compute_amp_reward``：自定义奖励混合策略
    - ``_update_normalizer``：自定义 normalizer 更新频率/方式
    - ``_build_discriminator``：自定义判别器网络结构
    """

    def __init__(
        self,
        amp_reward_coef: float,
        amp_discr_hidden_dims: list[int],
        amp_task_reward_lerp: float = 0.0,
        amp_replay_buffer_size: int = 100_000,
        amp_motion_files: str | list[str] | None = None,
        amp_body_names: list[str] | None = None,
        amp_anchor_name: str = "pelvis",
        amp_loss_coef: float = 1.0,
        **kwargs,
    ):
        """初始化参数，重型对象在 on_init 中按需创建。

        Args:
            amp_reward_coef: 判别器奖励的缩放系数。
            amp_discr_hidden_dims: 判别器 MLP 各隐藏层维度。
            amp_task_reward_lerp: 任务奖励插值权重（0 = 纯 AMP 奖励，1 = 纯任务奖励）。
            amp_replay_buffer_size: 策略轨迹回放 buffer 的最大容量。
            amp_motion_files: 专家动作文件路径（.npz 或目录）。
            amp_body_names: 用于计算 AMP 观测的身体链接名称列表。
            amp_anchor_name: 锚点身体名称（用于坐标对齐）。
            amp_loss_coef: 判别器损失相对 PPO 损失的权重系数。
        """
        if "min_normalized_std" in kwargs:
            raise ValueError(
                "AMPPlugin no longer clamps policy std. Move 'min_normalized_std' to "
                "the PPO algorithm config as 'min_policy_std'."
            )
        self.amp_reward_coef = amp_reward_coef
        self.amp_discr_hidden_dims = amp_discr_hidden_dims
        self.amp_task_reward_lerp = amp_task_reward_lerp
        self.amp_replay_buffer_size = amp_replay_buffer_size
        self.amp_motion_files = amp_motion_files
        self.amp_body_names = amp_body_names
        self.amp_anchor_name = amp_anchor_name
        self.amp_loss_coef = amp_loss_coef

        # 重型对象在 on_init 中创建（此时 ppo 和 env 均已就绪）
        self.discriminator: Discriminator | None = None
        self.amp_storage: ReplayBuffer | None = None
        self.amp_data: AMPLoader | None = None
        self.amp_normalizer: Normalizer | None = None

        self._current_amp_obs: torch.Tensor | None = None
        self.amp_policy_generator = None
        self.amp_expert_generator = None
        self._policy_pred_sum: float = 0.0
        self._expert_pred_sum: float = 0.0
        self._pred_count: int = 0

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def on_init(self, ppo, env) -> None:
        robot = env.unwrapped.scene["robot"]
        all_body_names = list(robot.body_names)

        # >>> AMP BODY ID DEBUG START
        print("\n========== AMP BODY ID DEBUG: RUNTIME ==========")
        print("[AMPDBG] runtime num bodies:", len(all_body_names))
        for body_id, body_name in enumerate(all_body_names):
            print(f"[AMPDBG] runtime body[{body_id:02d}] = {body_name}")
        resolved_body_ids, resolved_body_names = robot.find_bodies(
            list(self.amp_body_names),
            preserve_order=True,
        )
        resolved_anchor_ids, resolved_anchor_names = robot.find_bodies(
            [self.amp_anchor_name],
            preserve_order=True,
        )
        print("[AMPDBG] configured amp_anchor_name:", self.amp_anchor_name)
        print("[AMPDBG] resolved anchor ids:", resolved_anchor_ids)
        print("[AMPDBG] resolved anchor names:", resolved_anchor_names)
        print("[AMPDBG] configured amp_body_names:", self.amp_body_names)
        print("[AMPDBG] resolved body ids:", resolved_body_ids)
        print("[AMPDBG] resolved body names:", resolved_body_names)
        print("================================================\n")
        # <<< AMP BODY ID DEBUG END

        self.amp_data = AMPLoader(
            motion_file=self.amp_motion_files,
            body_names=self.amp_body_names,
            anchor_name=self.amp_anchor_name,
            all_body_names=all_body_names,
            device=ppo.device,
        )
        # >>> AMP BODY ID DEBUG START
        print("\n========== AMP BODY ID DEBUG: LOADER ==========")
        print("[AMPDBG] loader anchor index in source motion:", self.amp_data._anchor_indexes)
        print("[AMPDBG] loader anchor name:", self.amp_data._anchor_name)
        print("[AMPDBG] loader body indexes in AMP selected order:", self.amp_data._body_indexes)
        print("[AMPDBG] loader body names:", self.amp_data._body_names)
        print("[AMPDBG] loader observation_dim:", self.amp_data.observation_dim)
        print("===============================================\n")
        # <<< AMP BODY ID DEBUG END
        obs_dim: int = self.amp_data.observation_dim

        self.discriminator = self._build_discriminator(obs_dim, ppo.device)

        # 判别器参数加入 PPO optimizer，使单次 backward 同时更新策略和判别器
        ppo.optimizer.add_param_group(
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 10e-4}
        )
        ppo.optimizer.add_param_group(
            {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 10e-2}
        )

        self.amp_storage = ReplayBuffer(obs_dim, self.amp_replay_buffer_size, ppo.device)
        self.amp_normalizer = Normalizer(obs_dim)

    def _build_discriminator(self, obs_dim: int, device: str) -> Discriminator:
        """构建判别器网络，子类可覆盖以替换为自定义结构。"""
        return Discriminator(
            input_dim=obs_dim * 2,
            amp_reward_coef=self.amp_reward_coef,
            hidden_layer_sizes=self.amp_discr_hidden_dims,
            device=device,
            task_reward_lerp=self.amp_task_reward_lerp,
        ).to(device)

    # ------------------------------------------------------------------
    # Rollout hooks
    # ------------------------------------------------------------------

    def on_after_act(self, _runner, obs) -> None:
        amp_obs = obs.get("amp", obs.get("amp_obs"))
        if amp_obs is not None:
            self._current_amp_obs = amp_obs.detach().clone()

    def on_after_step(self, _runner, obs, rewards, dones, extras) -> torch.Tensor:
        if self._current_amp_obs is None or self.discriminator is None:
            return rewards
        next_amp_obs = obs.get("amp", obs.get("amp_obs"))
        if next_amp_obs is None:
            return rewards

        # Mirror local amp_ppo: for terminal envs use pre-step obs as terminal next state.
        next_amp_obs_with_term = next_amp_obs.clone()
        if dones is not None:
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if reset_ids.numel() > 0:
                next_amp_obs_with_term[reset_ids] = self._current_amp_obs[reset_ids]

        self.amp_storage.insert(self._current_amp_obs, next_amp_obs_with_term)
        reward_components = self._compute_amp_reward(self._current_amp_obs, next_amp_obs_with_term, rewards)
        step_metrics = extras.setdefault("step_metrics", {})
        step_metrics.update({k: v.reshape(-1, 1) for k, v in reward_components.items()})
        return reward_components["mixed_reward"]

    def _compute_amp_reward(
        self,
        amp_obs: torch.Tensor,
        next_amp_obs: torch.Tensor,
        task_rewards: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """计算判别器奖励，子类可覆盖以实现自定义混合策略。"""
        reward_components, _ = self.discriminator.predict_amp_reward_components(
            amp_obs, next_amp_obs, task_rewards, normalizer=self.amp_normalizer
        )
        return reward_components

    # ------------------------------------------------------------------
    # Update hooks
    # ------------------------------------------------------------------

    def on_update_start(self, ppo) -> None:
        n = ppo.num_learning_epochs * ppo.num_mini_batches
        bs = ppo.storage.num_envs * ppo.storage.num_transitions_per_env // ppo.num_mini_batches
        self.amp_policy_generator = self.amp_storage.feed_forward_generator(n, bs)
        self.amp_expert_generator = self.amp_data.feed_forward_generator(n, bs)
        self._policy_pred_sum = 0.0
        self._expert_pred_sum = 0.0
        self._pred_count = 0

    def on_per_batch_extra_loss(self, ppo, _batch) -> dict[str, torch.Tensor]:
        pol_s, pol_ns = next(self.amp_policy_generator)
        exp_s, exp_ns = next(self.amp_expert_generator)

        pol_s = pol_s.to(ppo.device)
        pol_ns = pol_ns.to(ppo.device)
        exp_s = exp_s.to(ppo.device)
        exp_ns = exp_ns.to(ppo.device)

        if self.amp_normalizer is not None:
            with torch.no_grad():
                pol_s = self.amp_normalizer.normalize_torch(pol_s, ppo.device)
                pol_ns = self.amp_normalizer.normalize_torch(pol_ns, ppo.device)
                exp_s = self.amp_normalizer.normalize_torch(exp_s, ppo.device)
                exp_ns = self.amp_normalizer.normalize_torch(exp_ns, ppo.device)

        B = pol_s.size(0)
        disc_out = self.discriminator(
            torch.cat([
                torch.cat([pol_s, pol_ns], dim=-1),
                torch.cat([exp_s, exp_ns], dim=-1),
            ], dim=0)
        )
        policy_d = disc_out[:B]
        expert_d = disc_out[B:]

        amp_loss, grad_pen = self.discriminator.compute_loss(
            policy_d, expert_d, sample_amp_expert=(exp_s, exp_ns)
        )

        # Mirror local amp_ppo: update normalizer per-batch from normalized states.
        if self.amp_normalizer is not None:
            self.amp_normalizer.update(pol_s.detach().cpu().numpy())
            self.amp_normalizer.update(exp_s.detach().cpu().numpy())

        # Accumulate discriminator prediction stats for logging.
        self._policy_pred_sum += policy_d.detach().mean().item()
        self._expert_pred_sum += expert_d.detach().mean().item()
        self._pred_count += 1

        return {
            "amp":  amp_loss,
            "amp_grad_pen":   grad_pen,
        }

    def on_post_backward(self, ppo) -> None:
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), ppo.max_grad_norm)

    def on_post_update(self, _ppo) -> dict[str, float]:
        n = max(self._pred_count, 1)
        return {
            "amp_policy_pred": self._policy_pred_sum / n,
            "amp_expert_pred": self._expert_pred_sum / n,
        }

    def _update_normalizer(self) -> None:
        """更新 normalizer 的运行统计量，子类可覆盖以调整更新频率。"""
        if self.amp_normalizer is not None and self.amp_storage is not None and self.amp_storage.num_samples > 0:
            n = min(self.amp_storage.num_samples, 1024)
            idxs = torch.randperm(self.amp_storage.num_samples)[:n]
            states = self.amp_storage.states[idxs]
            next_states = self.amp_storage.next_states[idxs]
            self.amp_normalizer.update_with_tensors(states, next_states)

    # ------------------------------------------------------------------
    # Train/eval mode
    # ------------------------------------------------------------------

    def on_train_mode(self, _ppo) -> None:
        if self.discriminator is not None:
            self.discriminator.train()

    def on_eval_mode(self, _ppo) -> None:
        if self.discriminator is not None:
            self.discriminator.eval()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def on_save(self, _ppo, saved_dict: dict) -> None:
        if self.discriminator is not None:
            saved_dict["discriminator_state_dict"] = self.discriminator.state_dict()

    def on_load(self, _ppo, loaded_dict: dict) -> None:
        if self.discriminator is not None and "discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
