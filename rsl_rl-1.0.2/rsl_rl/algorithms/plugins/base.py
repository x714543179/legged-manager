from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rsl_rl.algorithms.ppo import PPO
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner
    from rsl_rl.env import VecEnv


class PPOPlugin:
    """PPO 插件基类。所有 hook 默认为空操作，子类按需覆盖。

    Hook 调用顺序（每次训练迭代）::

        construct_algorithm():
            plugin.on_init(ppo, env)          ← 初始化插件，env 和 ppo 均已就绪

        OnPolicyRunner.learn() 内每步:
            alg.act(obs)
            plugin.on_after_act(runner, obs)  ← 存储跨步需要的观测
            env.step(actions)
            plugin.on_after_step(...)         ← 可修改 rewards，返回修改后版本
            alg.process_env_step(...)

        PPO.update() 内:
            plugin.on_update_start(ppo)       ← batch 循环前，初始化 generator 等
            for batch in generator:
                _compute_loss(...)
                plugin.on_per_batch_extra_loss(ppo, batch) ← 返回额外 loss dict
                loss.backward()
                plugin.on_post_backward(ppo)  ← optimizer.step() 前裁剪插件参数梯度
                optimizer.step()
            plugin.on_post_update(ppo)        ← 返回额外 metric dict

        ppo.train_mode() / eval_mode():
            plugin.on_train_mode(ppo) / on_eval_mode(ppo)

        ppo.save() / load():
            plugin.on_save(ppo, saved_dict) / on_load(ppo, loaded_dict)
    """

    def on_init(self, ppo: "PPO", env: "VecEnv") -> None:
        """初始化插件，此时 ppo 和 env 都已就绪。

        可向 ``ppo.optimizer`` 添加参数组（用于判别器等附加网络），
        也可从 ``env`` 读取机器人身体名称等环境信息。
        """

    def on_after_act(self, runner: "OnPolicyRunner", obs: dict) -> None:
        """动作采样后、env.step 前调用。

        通常用于存储需要跨步保留的观测（例如 AMP 的当前帧状态）。
        """

    def on_after_step(
        self,
        runner: "OnPolicyRunner",
        obs: dict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
    ) -> torch.Tensor:
        """env.step 后、process_env_step 前调用。

        返回修改后的 rewards（例如用判别器奖励替换任务奖励）。
        默认直接返回原始 rewards，不做修改。
        """
        return rewards

    def on_update_start(self, ppo: "PPO") -> None:
        """update() 中 batch 循环前调用。

        通常用于初始化跨 batch 共享的 generator（如 AMP 的回放 buffer 生成器）。
        """

    def on_per_batch_extra_loss(self, ppo: "PPO", batch) -> dict[str, torch.Tensor]:
        """每个 mini-batch 内，PPO loss 计算后调用。

        返回额外 loss 的字典；字典中所有 loss 会被求和并加到总 loss 上，
        也会被记录到日志中。返回空字典表示无额外 loss。
        """
        return {}

    def on_post_backward(self, ppo: "PPO") -> None:
        """loss.backward() 后、optimizer.step() 前调用。

        通常用于裁剪插件自有参数（如判别器）的梯度范数。
        """

    def on_post_update(self, ppo: "PPO") -> dict[str, float]:
        """update() 结束后调用。

        返回需要记录到日志的额外标量 metric（如 normalizer 统计量）。
        返回空字典表示无额外 metric。
        """
        return {}

    def on_train_mode(self, ppo: "PPO") -> None:
        """切换到训练模式时调用（对应 ppo.train_mode()）。"""

    def on_eval_mode(self, ppo: "PPO") -> None:
        """切换到推理模式时调用（对应 ppo.eval_mode()）。"""

    def on_save(self, ppo: "PPO", saved_dict: dict) -> None:
        """保存 checkpoint 时调用。将插件状态写入 saved_dict。"""

    def on_load(self, ppo: "PPO", loaded_dict: dict) -> None:
        """加载 checkpoint 时调用。从 loaded_dict 恢复插件状态。"""
