import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.LipschitzActorCritic.lipschitz_actor_critic import LipschitzActorCritic



class LipPPO(PPO):
    """基于 PPO 的 Lipschitz 扩展版，继承原版 PPO，变量名保持一致"""


    def __init__(self, *args, lr_k_scale=0.01, **kwargs):
        """
        参数:
        - lr_k_scale: K 网络的学习率相对于主学习率的比例 (默认 0.1)
        """
        super().__init__(*args, **kwargs)
        self.lr_k_scale = lr_k_scale
        self._build_optimizer()

    def _build_optimizer(self):
        """
        重新构建 Adam 优化器，给 LCN 单独学习率
        """
        if not hasattr(self.actor_critic, "lcn"):
            # 如果当前网络没有 LCN，仍保持单一优化器
            print("⚠️ Warning: actor_critic has no LCN; using default optimizer.")
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
            return

        # ✅ 分组：LCN 单独学习率，其余保持原有 learning_rate
        lr_k = self.learning_rate * self.lr_k_scale


                # ✅ 拆出 LCN 参数，其余全部按默认学习率
        lcn_params = list(self.actor_critic.lcn.parameters())
        other_params = [p for n, p in self.actor_critic.named_parameters() if not n.startswith("lcn")]

        self.optimizer = torch.optim.Adam([
            {"params": other_params, "lr": self.learning_rate},     # actor、critic、log_std 等都在这里
            {"params": lcn_params,   "lr": lr_k},        # 单独设置 LCN 学习率
        ])


        for i, group in enumerate(self.optimizer.param_groups):
            print(f"Param group {i}: lr={group['lr']:.2e}, num_params={len(group['params'])}")
