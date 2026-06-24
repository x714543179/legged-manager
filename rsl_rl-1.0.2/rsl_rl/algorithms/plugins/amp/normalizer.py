from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class RunningMeanStd:
    """在线计算数据流的均值和方差（Welford 并行算法）。"""

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = m_2 / tot_count
        self.count = tot_count


class Normalizer(RunningMeanStd):
    """基于运行均值/方差的观测归一化器，支持 numpy 和 torch 两种接口。"""

    def __init__(self, input_dim: int, epsilon: float = 1e-4, clip_obs: float = 10.0):
        super().__init__(shape=input_dim)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def normalize_torch(self, x: torch.Tensor, device: str | torch.device) -> torch.Tensor:
        mean = torch.tensor(self.mean, device=device, dtype=torch.float32)
        std = torch.sqrt(torch.tensor(self.var + self.epsilon, device=device, dtype=torch.float32))
        return torch.clamp((x - mean) / std, -self.clip_obs, self.clip_obs)

    def update_with_tensors(self, *tensors: torch.Tensor) -> None:
        """用若干 torch.Tensor 更新运行统计量（自动转 numpy）。"""
        arr = torch.cat(tensors, dim=0).cpu().numpy()
        self.update(arr)
