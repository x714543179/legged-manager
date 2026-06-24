from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING

import torch

from rsl_rl.algorithms.plugins.base import PPOPlugin
from rsl_rl.utils import construct_actor_with_shell

if TYPE_CHECKING:
    from rsl_rl.algorithms.ppo import PPO
    from rsl_rl.env import VecEnv


class TeacherKLPlugin(PPOPlugin):
    """Add a frozen teacher-policy KL term to PPO mini-batch updates."""

    def __init__(
        self,
        teacher_checkpoint_path: str,
        teacher_actor: dict,
        teacher_obs_groups: dict[str, list[str]],
        start_loss_coef: float = 0.5,
        end_loss_coef: float = 0.0,
        end_step: int = 2000,
        kl_direction: str = "student_teacher",
        teacher_obs_aliases: dict[str, str] | None = None,
        checkpoint_key: str = "actor_state_dict",
        strict: bool = True,
    ) -> None:
        self.teacher_checkpoint_path = os.path.expanduser(teacher_checkpoint_path)
        self.teacher_actor_cfg = copy.deepcopy(teacher_actor)
        self.teacher_obs_groups = copy.deepcopy(teacher_obs_groups)
        self.teacher_obs_aliases = copy.deepcopy(teacher_obs_aliases or {})
        self.start_loss_coef = float(start_loss_coef)
        self.end_loss_coef = float(end_loss_coef)
        self.end_step = int(end_step)
        self.kl_direction = kl_direction
        self.checkpoint_key = checkpoint_key
        self.strict = bool(strict)

        self.teacher = None
        self.step = 0
        self._coef = self.start_loss_coef
        self._kl_sum = 0.0
        self._loss_sum = 0.0
        self._batch_count = 0

        if self.end_step < 0:
            raise ValueError(f"end_step must be >= 0, got {self.end_step}")
        if self.kl_direction not in {"student_teacher", "teacher_student"}:
            raise ValueError(
                f"Unsupported kl_direction={self.kl_direction!r}; "
                "expected 'student_teacher' or 'teacher_student'."
            )

    def on_init(self, ppo: "PPO", env: "VecEnv") -> None:
        if not os.path.isfile(self.teacher_checkpoint_path):
            raise FileNotFoundError(f"Teacher checkpoint not found: {self.teacher_checkpoint_path}")

        self.teacher = construct_actor_with_shell(
            self._teacher_obs(ppo.storage.observations[0]),
            self.teacher_obs_groups,
            self.teacher_actor_cfg,
            env.num_actions,
        ).to(ppo.device)

        checkpoint = torch.load(self.teacher_checkpoint_path, weights_only=False, map_location=ppo.device)
        if self.checkpoint_key not in checkpoint:
            raise KeyError(
                f"Teacher checkpoint {self.teacher_checkpoint_path} does not contain key "
                f"{self.checkpoint_key!r}. Available keys: {list(checkpoint.keys())}"
            )
        self.teacher.load_state_dict(checkpoint[self.checkpoint_key], strict=self.strict)
        self._freeze_teacher()

        print(
            "[TeacherKLPlugin] Loaded frozen teacher actor from "
            f"{self.teacher_checkpoint_path}; obs_groups={self.teacher_obs_groups}, "
            f"obs_aliases={self.teacher_obs_aliases}, "
            f"start_loss_coef={self.start_loss_coef}, end_loss_coef={self.end_loss_coef}, "
            f"end_step={self.end_step}, kl_direction={self.kl_direction}"
        )

    def _freeze_teacher(self) -> None:
        if self.teacher is None:
            return
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    def _current_coef(self) -> float:
        if self.end_step <= 0:
            return self.end_loss_coef
        progress = min(max(self.step / self.end_step, 0.0), 1.0)
        return self.start_loss_coef + progress * (self.end_loss_coef - self.start_loss_coef)

    def on_update_start(self, _ppo: "PPO") -> None:
        self._coef = self._current_coef()
        self._kl_sum = 0.0
        self._loss_sum = 0.0
        self._batch_count = 0
        self._freeze_teacher()

    def _teacher_obs(self, obs):
        if not self.teacher_obs_aliases:
            return obs
        teacher_obs = obs.clone()
        for teacher_key, source_key in self.teacher_obs_aliases.items():
            if source_key not in obs.keys():
                raise KeyError(
                    f"Teacher obs alias source {source_key!r} is missing. "
                    f"Available observation keys: {list(obs.keys())}"
                )
            teacher_obs[teacher_key] = obs[source_key]
        return teacher_obs

    def _distribution_params(self, actor, obs, *, with_grad: bool) -> tuple[torch.Tensor, ...]:
        if with_grad:
            actor(obs, stochastic_output=False, train_mode=True)
            return actor.output_distribution_params

        with torch.no_grad():
            actor(obs, stochastic_output=False, train_mode=False)
            return tuple(p.detach() for p in actor.output_distribution_params)

    def on_per_batch_extra_loss(self, ppo: "PPO", batch) -> dict[str, torch.Tensor]:
        if self.teacher is None:
            raise RuntimeError("TeacherKLPlugin.on_init() must run before loss computation.")
        if ppo.actor.is_recurrent:
            raise NotImplementedError("TeacherKLPlugin currently supports feed-forward actors only.")

        student_params = self._distribution_params(ppo.actor, batch.observations, with_grad=True)
        teacher_params = self._distribution_params(self.teacher, self._teacher_obs(batch.observations), with_grad=False)

        if self.kl_direction == "student_teacher":
            kl = ppo.actor.get_kl_divergence(student_params, teacher_params)
        else:
            kl = ppo.actor.get_kl_divergence(teacher_params, student_params)

        raw_kl = kl.mean()
        loss = raw_kl * self._coef

        self._kl_sum += raw_kl.detach().item()
        self._loss_sum += loss.detach().item()
        self._batch_count += 1

        return {"teacher_kl_loss": loss}

    def on_post_update(self, _ppo: "PPO") -> dict[str, float]:
        n = max(self._batch_count, 1)
        metrics = {
            "teacher_kl_raw": self._kl_sum / n,
            "teacher_kl_coef": self._coef,
            "teacher_kl_loss_metric": self._loss_sum / n,
        }
        self.step += 1
        return metrics

    def on_train_mode(self, _ppo: "PPO") -> None:
        self._freeze_teacher()

    def on_eval_mode(self, _ppo: "PPO") -> None:
        self._freeze_teacher()

    def on_save(self, _ppo: "PPO", saved_dict: dict) -> None:
        saved_dict["teacher_kl_plugin_state"] = {
            "step": self.step,
            "coef": self._coef,
            "teacher_checkpoint_path": self.teacher_checkpoint_path,
        }

    def on_load(self, _ppo: "PPO", loaded_dict: dict) -> None:
        state = loaded_dict.get("teacher_kl_plugin_state")
        if not state:
            return
        self.step = int(state.get("step", self.step))
        self._coef = float(state.get("coef", self._current_coef()))
