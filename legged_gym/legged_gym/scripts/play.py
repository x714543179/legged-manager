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

import os
import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.plotting import create_plot_manager
from legged_gym.utils import  get_args, task_registry
from legged_gym.utils.helpers import export_vae_policy_as_jit, disable_manager_randomization

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if getattr(args, "viewer", "native") == "viser":
        args.headless = True
    # override some parameters for testing
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    # env_cfg.terrain.num_rows = 5
    # env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    disable_manager_randomization(env_cfg)

    # command
    env_cfg.commands.ranges.lin_vel_x = [0, 0]
    env_cfg.commands.ranges.lin_vel_y = [0, 0]
    env_cfg.commands.ranges.ang_vel_yaw = [0, 0]
    env_cfg.commands.ranges.heading = [0, 0]


    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, obs_hist = env.get_observations()






    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    uses_tensordict_policy = not hasattr(ppo_runner.alg, "actor_critic")
    if uses_tensordict_policy:
        obs = ppo_runner.env.get_observations().to(env.device)
    
    # export policy as a jit module (used to run it from C++)

    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        if hasattr(ppo_runner, "export_policy_to_jit") and uses_tensordict_policy:
            ppo_runner.export_policy_to_jit(path)
        else:
            export_vae_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    # if EXPORT_POLICY:
    #     path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    #     export_policy_as_jit_actor(ppo_runner.alg.actor_critic, path)
    #     export_policy_as_jit_encoder(ppo_runner.alg.actor_critic,path)
    #     print('Exported policy as jit script to: ', path)

    robot_index = 0 # which robot is used for logging
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    plot_output_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "plots")
    plot_manager = create_plot_manager(env, args=args, dt=env.dt, output_dir=plot_output_dir)

    viser_viewer = None
    if getattr(args, "viewer", "native") == "viser":
        from legged_gym.utils.viser_viewer import create_viser_viewer

        viser_viewer = create_viser_viewer(env, port=args.viser_port, robot_index=robot_index)
        print(f"Viser web viewer started at http://localhost:{args.viser_port}")
     
    try:
        for i in range(10*int(env.max_episode_length)):
            if viser_viewer is not None:
                cmd = viser_viewer.get_command()
                cmd_tensor = torch.as_tensor(cmd, device=env.device, dtype=env.commands.dtype)
                env.commands[:, :3] = cmd_tensor

            if uses_tensordict_policy:
                actions = policy(obs.detach())["actions"]
                obs, rews, dones, infos = ppo_runner.env.step(actions.detach())
                obs = obs.to(env.device)
            else:
                actions = policy(obs.detach(),obs_hist.detach())
                obs, _, _, obs_hist, rews, dones, infos = env.step(actions.detach())
            # obs[:,6] = 0.0
            # obs[:,7] = 2.0
            # obs[:,8] = 0.0


            if viser_viewer is not None:
                viser_viewer.update_from_env(env, robot_index)

            plot_manager.step(env, actions=actions, rews=rews, dones=dones, infos=infos)

            if RECORD_FRAMES:
                if i % 2:
                    filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
                    img_idx += 1 
            if MOVE_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)
    finally:
        plot_manager.close()
        if viser_viewer is not None:
            viser_viewer.stop()





if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
