# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import pathlib
from dataclasses import asdict, is_dataclass
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("wandb package is required to log to Weights and Biases.") from None


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for W&B."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        """Initialize a W&B run for logging."""
        super().__init__(log_dir, flush_secs=flush_secs)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        # Get wandb project and entity
        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.") from None
        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None
        group = cfg.get("wandb_group")
        mode = cfg.get("wandb_mode", "online")
        tags = cfg.get("wandb_tags", [])

        # Initialize wandb
        if wandb.run is None:
            wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                group=group,
                mode=mode,
                dir=os.path.dirname(log_dir),
                tags=tags,
                config={"log_dir": log_dir},
                settings=wandb.Settings(start_method="thread"),
            )
        else:
            wandb.config.update({"log_dir": log_dir}, allow_val_change=True)

        # Initialize set to keep track of logged videos
        self.logged_videos: set[str] = set()

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Upload environment and training configuration to W&B."""
        wandb.config.update(
            {
                "train_cfg": self._to_wandb_config(train_cfg),
                "env_cfg": self._to_wandb_config(env_cfg),
            },
            allow_val_change=True,
        )

    def _to_wandb_config(self, obj):
        """Convert config objects into W&B-safe built-in containers."""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, pathlib.Path):
            return str(obj)
        if is_dataclass(obj) and not isinstance(obj, type):
            return self._to_wandb_config(asdict(obj))
        if isinstance(obj, dict):
            return {str(key): self._to_wandb_config(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._to_wandb_config(value) for value in obj]
        if callable(obj):
            return getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj)))
        if hasattr(obj, "to_dict"):
            return self._to_wandb_config(obj.to_dict())
        if hasattr(obj, "__dict__"):
            return {
                key: self._to_wandb_config(value)
                for key, value in vars(obj).items()
                if not key.startswith("_")
            }
        return str(obj)

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        """Log a scalar to both TensorBoard and W&B."""
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({tag: scalar_value}, step=global_step)

    def stop(self) -> None:
        """Finish the active W&B run."""
        wandb.finish()

    def save_model(self, model_path: str, it: int) -> None:
        """Upload a model checkpoint artifact to W&B."""
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        """Upload an arbitrary file artifact to W&B."""
        wandb.save(path, base_path=os.path.dirname(path))

    def save_video(self, video: pathlib.Path, it: int) -> None:
        """Upload a video artifact once per filename to W&B."""
        if video.name not in self.logged_videos:
            wandb.log({"video": wandb.Video(str(video), format="mp4")}, step=it)
            self.logged_videos.add(video.name)
