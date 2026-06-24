# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import ActorModel, MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer, construct_actor_with_shell


class PPO:
    """Proximal Policy Optimization algorithm.

    Reference:
        - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
    """

    actor: MLPModel | ActorModel
    """The actor model."""

    critic: MLPModel
    """The critic model."""

    def __init__(
        self,
        actor: ActorModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        min_policy_std: list[float] | float | None = None,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Legacy compatibility argument (ignored).
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Plugin list (instantiated by construct_algorithm, on_init called separately)
        plugins: list | None = None,
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Keep signature backward-compatible, but symmetry is disabled in this PPO.
        _ = symmetry_cfg

        # PPO components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Create the optimizer
        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic.parameters()),
            lr=learning_rate,
        )  # type: ignore

        # Add storage
        self.storage = storage
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
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.min_policy_std = min_policy_std

        self.plugins: list = list(plugins) if plugins else []
        # on_init is NOT called here — construct_algorithm calls it after env is available


    # region Rollout Environment Interaction and Return Computation 

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Record the hidden states for recurrent policies
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        # Compute the actions and values
        self.transition.actions = self.actor(obs, stochastic_output=True)["actions"].detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update the normalizers."""
        # Update the normalizers
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

        # post-rollout hook (currently unused; reserved for plugins that need per-step processing)


    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage
        # Compute value for the last step
        last_values = self.critic(obs).detach()
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        st.advantages = st.returns - st.values
        # Normalize the advantages if per minibatch normalization is not used
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)
    #endregion 
    # region resolve Loss Computation and Optimization
    def _register_loss_metrics(self) -> dict[str, float]:
        """Register running metric buffer for one update iteration.

        Metric keys are inferred dynamically from ``loss_results`` in
        ``_accumulate_loss_metrics``.
        """
        return {}

    def _extract_metric_values_from_loss_results(
        self,
        obj: dict | torch.Tensor | float | int,
        prefix: str = "",
    ) -> dict[str, float]:
        """Recursively parse nested loss dicts into scalar metrics.

        Rules:
        - dict: recurse with path-like keys
        - Tensor: scalar -> item, otherwise mean().item()
        - float/int: cast to float
        """
        parsed: dict[str, float] = {}
        if isinstance(obj, dict):
            for key, value in obj.items():
                child_prefix = f"{prefix}/{key}" if prefix else str(key)
                parsed.update(self._extract_metric_values_from_loss_results(value, child_prefix))
            return parsed

        if isinstance(obj, torch.Tensor):
            if obj.numel() == 0:
                return parsed
            parsed[prefix] = obj.item() if obj.numel() == 1 else obj.mean().item()
            return parsed

        if isinstance(obj, (float, int)):
            parsed[prefix] = float(obj)
            return parsed

        return parsed

    def _accumulate_loss_metrics(
        self,
        metrics: dict[str, float],
        loss_results: dict,
    ) -> None:
        """Accumulate per-batch loss values into running metric sums.

        This parser accepts nested dictionaries (e.g. output of ``_compute_loss``)
        and resolves metric values by key, avoiding repetitive variable unpacking
        at the call site.
        """
        parsed_metrics = self._extract_metric_values_from_loss_results(loss_results)

        for key, value in parsed_metrics.items():
            metrics[key] = metrics.get(key, 0.0) + value

    @staticmethod
    def _finalize_loss_metrics(metrics: dict[str, float], num_updates: int) -> dict[str, float]:
        """Normalize running sums by update count and return logging dict."""
        if num_updates <= 0:
            raise ValueError(f"num_updates must be > 0, got {num_updates}")
        return {name: value / num_updates for name, value in metrics.items()}
    # endregion
    
    def _forward_model(self, batch: TensorDict, original_batch_size: int) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        """Run actor/critic forward pass for one mini-batch."""
        forward_dict = self.actor(
            batch.observations,
            masks=batch.masks,
            hidden_state=batch.hidden_states[0],
            stochastic_output=True,
            train_mode=True,
        )
        actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
        values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
        distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
        entropy = self.actor.output_entropy[:original_batch_size]

        extra = forward_dict.get("extra", {}) if isinstance(forward_dict, dict) else {}

        # Prefer the named-loss dict; fall back to legacy scalar so old backbones still work.
        aux_losses: dict = extra.get("aux_losses") or {}
        if not aux_losses and extra.get("aux_loss") is not None:
            aux_losses = {"aux_loss": extra["aux_loss"]}

        return {
            "actions_log_prob": actions_log_prob,
            "values": values,
            "distribution_params": distribution_params,
            "entropy": entropy,
            "aux_losses": aux_losses,
        }

    def _adjust_learning_rate_based_on_kl(self, batch: TensorDict, distribution_params: tuple[torch.Tensor, ...]) -> None:
        """Adapt learning rate based on KL divergence under adaptive schedule."""
        if self.desired_kl is None or self.schedule != "adaptive":
            return

        with torch.inference_mode():
            kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
            kl_mean = torch.mean(kl)

            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size

            if self.gpu_global_rank == 0:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            if self.is_multi_gpu:
                lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(lr_tensor, src=0)
                self.learning_rate = lr_tensor.item()

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def _compute_ppo_loss(self, forward_results: dict, mb_rollout_data: dict) -> dict[str, torch.Tensor]:
        """Compute PPO objective for one mini-batch."""
        batch: TensorDict = mb_rollout_data["batch"]
        original_batch_size: int = mb_rollout_data["original_batch_size"]

        actions_log_prob = forward_results["actions_log_prob"]
        values = forward_results["values"]
        entropy = forward_results["entropy"]
        distribution_params = forward_results["distribution_params"]

        self._adjust_learning_rate_based_on_kl(batch, distribution_params)

        ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
        surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
        surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        if self.use_clipped_value_loss:
            value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - batch.returns).pow(2)
            value_losses_clipped = (value_clipped - batch.returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (batch.returns - values).pow(2).mean()

        ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
        out_dict = {
            "ppo_loss": ppo_loss,
            "surrogate_loss": surrogate_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }

        return out_dict

    def _compute_loss(self, mb_forward_results: dict, mb_rollout_data: dict) -> dict:
        """Compute total loss and return nested loss dicts for logging/extension."""
        ppo_loss_dict = self._compute_ppo_loss(mb_forward_results, mb_rollout_data)
        loss = ppo_loss_dict["ppo_loss"]
        aux_loss_dict: dict[str, torch.Tensor] = dict(mb_forward_results.get("aux_losses", {}))
        if aux_loss_dict:
            loss = loss + sum(aux_loss_dict.values())  # type: ignore[arg-type]

        return {
            "loss": loss,
            "ppo_loss_dict": ppo_loss_dict,
            "aux_losses": aux_loss_dict,
        }

    def _expand_min_policy_std(self, target: torch.Tensor) -> torch.Tensor:
        """Return the configured minimum std broadcast to a distribution parameter tensor."""
        if self.min_policy_std is None:
            raise RuntimeError("min_policy_std is not configured")

        min_std = torch.as_tensor(self.min_policy_std, device=target.device, dtype=target.dtype)
        if min_std.ndim == 0:
            min_std = min_std.unsqueeze(0)
        min_std = torch.clamp_min(min_std, 1e-6)

        if min_std.numel() == 1:
            return min_std.expand_as(target)
        if min_std.numel() != target.numel():
            return min_std.min().expand_as(target)
        return min_std.reshape_as(target)

    def _clamp_policy_std(self) -> None:
        """Clamp the actor policy std after optimizer updates to avoid std collapse."""
        if self.min_policy_std is None:
            return

        dist = getattr(self.actor, "distribution", None)
        if dist is None:
            return

        with torch.no_grad():
            std_type = getattr(dist, "std_type", None)
            if std_type == "scalar" and hasattr(dist, "std_param"):
                target = dist.std_param
                target.clamp_(min=self._expand_min_policy_std(target))
            elif std_type == "log" and hasattr(dist, "log_std_param"):
                target = dist.log_std_param
                min_std = self._expand_min_policy_std(target)
                target.clamp_(min=torch.log(min_std))

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        metrics = self._register_loss_metrics()

        # Get mini batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for plugin in self.plugins:
            plugin.on_update_start(self)

        # Iterate over batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]
            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            mb_forward_results = self._forward_model(batch, original_batch_size)
            mb_rollout_data = {
                "batch": batch,
                "original_batch_size": original_batch_size,
            }
            loss_results = self._compute_loss(mb_forward_results, mb_rollout_data)

            # 收集各插件的额外 loss，合并到总 loss 和日志中
            for plugin in self.plugins:
                extra = plugin.on_per_batch_extra_loss(self, batch)
                if extra:
                    loss_results["loss"] = loss_results["loss"] + sum(extra.values())
                    loss_results.update(extra)

            loss = loss_results["loss"]

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()     

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

            # 让插件在 backward 后裁剪自有参数（如判别器）的梯度
            for plugin in self.plugins:
                plugin.on_post_backward(self)

            self.optimizer.step()
            self._clamp_policy_std()
            # Store the losses.
            self._accumulate_loss_metrics(
                metrics,
                loss_results=loss_results,
            )

        # Divide the losses by the number of updates.
        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = self._finalize_loss_metrics(metrics, num_updates)

        # 插件可在此追加额外 metric（如 normalizer 统计量）
        for plugin in self.plugins:
            loss_dict.update(plugin.on_post_update(self))

        # Clear the storage
        self.storage.clear()

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()
        for plugin in self.plugins:
            plugin.on_train_mode(self)

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()
        for plugin in self.plugins:
            plugin.on_eval_mode(self)
        
    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        for plugin in self.plugins:
            plugin.on_save(self, saved_dict)
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        for plugin in self.plugins:
            plugin.on_load(self, loaded_dict)
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel | ActorModel:
        """Get the policy model."""
        return self.actor


    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct the PPO algorithm."""
        actor_cfg = copy.deepcopy(cfg["actor"])
        critic_cfg = copy.deepcopy(cfg["critic"])
        algorithm_cfg = copy.deepcopy(cfg["algorithm"])
        obs_groups = copy.deepcopy(cfg.get("obs_groups", {}))

        # Resolve class callables
        alg_class: type[PPO] = resolve_callable(algorithm_cfg.pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(critic_cfg.pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        obs_groups = resolve_obs_groups(obs, obs_groups, default_sets)

        # Explicitly ignore legacy symmetry settings in this PPO variant.
        algorithm_cfg.pop("symmetry_cfg", None)
        algorithm_cfg.pop("aux_modules", None)

        # Initialize the policy
        actor = construct_actor_with_shell(obs, obs_groups, actor_cfg, env.num_actions).to(device)
        print(f"Actor Model: {actor}")
        if algorithm_cfg.pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            critic_cfg["cnns"] = actor.backbone.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, obs_groups, "critic", 1, **critic_cfg).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # 从 config 实例化插件（on_init 在此处调用，可访问 env）
        plugins_cfgs: list[dict] = algorithm_cfg.pop("plugins", [])
        plugins: list = []
        for pcfg in plugins_cfgs:
            pcfg = dict(pcfg)
            plugin_cls = resolve_callable(pcfg.pop("class_name"))
            plugins.append(plugin_cls(**pcfg))

        # Initialize the algorithm
        alg: PPO = alg_class(
            actor, critic, storage,
            device=device,
            plugins=plugins,
            **algorithm_cfg,
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        for plugin in alg.plugins:
            plugin.on_init(alg, env)

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.actor.state_dict(), self.critic.state_dict()]
        
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.actor.load_state_dict(model_params[0])
        self.critic.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = chain(self.actor.parameters(), self.critic.parameters())

        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
