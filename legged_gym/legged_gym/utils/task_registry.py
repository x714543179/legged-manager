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

from __future__ import annotations

import os
import copy
from datetime import datetime
from typing import TYPE_CHECKING, Tuple

from rsl_rl.utils import resolve_callable

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.base_config import BaseConfig

if TYPE_CHECKING:
    from rsl_rl.env import VecEnv

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: "VecEnv", env_cfg: BaseConfig, train_cfg: BaseConfig):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> "VecEnv":
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[BaseConfig, BaseConfig]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple["VecEnv", BaseConfig]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, BaseConfig]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        train_cfg_dict = copy.deepcopy(class_to_dict(train_cfg))
        self._fill_runner_top_level_cfg(train_cfg_dict)
        self._inject_obs_groups_from_env(train_cfg_dict, env)
        tags = list(train_cfg_dict.get("wandb_tags", []))
        tags.extend([f"task_{name}", getattr(env.cfg, "task_name", "")])
        train_cfg_dict["wandb_tags"] = [tag for tag in tags if tag]

        # === 根据 config 中的 runner.class_name 选择 runner ===
        runner_class_name = getattr(train_cfg.runner, "runner_class_name", "rsl_rl.runners:OnPolicyRunner")
        runner_class = resolve_callable(runner_class_name)

        runner_env = env
        if "actor" in train_cfg_dict and "critic" in train_cfg_dict:
            from legged_gym.utils.rsl_rl_adapter import RslRlVecEnvAdapter

            runner_env = RslRlVecEnvAdapter(env)
        runner = runner_class(runner_env, train_cfg_dict, log_dir, device=args.rl_device)
        # runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)

        #save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if resume:
            # load previously trained model
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner, train_cfg




    @staticmethod
    def _fill_runner_top_level_cfg(cfg):
        runner_cfg = cfg.get("runner", {})
        cfg["num_steps_per_env"] = runner_cfg.get("num_steps_per_env", cfg.get("num_steps_per_env", 24))
        cfg["save_interval"] = runner_cfg.get("save_interval", cfg.get("save_interval", 500))
        cfg["run_name"] = runner_cfg.get("run_name", cfg.get("run_name", ""))
        cfg["logger"] = runner_cfg.get("logger", cfg.get("logger", "tensorboard"))
        cfg["wandb_project"] = runner_cfg.get(
            "wandb_project",
            cfg.get("wandb_project", runner_cfg.get("experiment_name", "legged_gym")),
        )
        cfg["wandb_group"] = runner_cfg.get("wandb_group", cfg.get("wandb_group", None))
        cfg["wandb_mode"] = runner_cfg.get("wandb_mode", cfg.get("wandb_mode", "online"))
        cfg["wandb_tags"] = runner_cfg.get("wandb_tags", cfg.get("wandb_tags", []))
        cfg["multi_gpu"] = cfg.get("multi_gpu", None)

    @staticmethod
    def _inject_obs_groups_from_env(train_cfg_dict, env):

        if train_cfg_dict.get("obs_groups"):
            return
        observation_manager = getattr(env, "observation_manager", None)
        obs_groups = getattr(observation_manager, "obs_groups", None)
        if obs_groups:
            train_cfg_dict["obs_groups"] = obs_groups


# make global task registry
task_registry = TaskRegistry()
