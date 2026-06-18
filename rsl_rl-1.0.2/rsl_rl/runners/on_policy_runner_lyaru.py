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
  
import time
import os
from collections import deque
import statistics
  
from torch.utils.tensorboard import SummaryWriter
import torch
import wandb
import numpy as np

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCritic_DWAQ
from rsl_rl.modules import LipschitzActorCritic
from rsl_rl.env.__init__ import VecEnv

from rsl_rl.algorithms.ppo_lya import LyapunovPPO
from rsl_rl.algorithms.ppo_lyaRu import LyapunovRuPPO



class OnPolicyRunnerLyaRU:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cuda:0'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        cenet_in_dim = self.env.num_obs_hist * self.env.num_obs
        cenet_out_dim = 19
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic

        # ✅ 移除类型注解 (ActorCritic_DWAQ)，防止编辑器类型固定
        actor_critic = actor_critic_class(
            num_actor_obs=self.env.num_obs + cenet_out_dim,
            num_critic_obs=num_critic_obs,
            num_actions=self.env.num_actions,
            cenet_in_dim=cenet_in_dim,
            cenet_out_dim=cenet_out_dim,
            **self.policy_cfg).to(self.device)

        

        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg = LyapunovRuPPO(actor_critic,
                                state_dim=self.env.num_obs,
                                action_dim=self.env.num_actions,
                                lyapunov_cfg=train_cfg["lyapunov"],
                                fhead_input_dim = cenet_in_dim,
                                device=self.device,
                                **self.alg_cfg)


        
        print("用到的算法程序是", self.cfg["algorithm_class_name"])
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_obs_hist*self.env.num_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _, _, _ = self.env.reset()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!RESETRESETRESETRESETRESETRESETRESETRESETRESETRESETRESETRESETRESETRESETRESETRESETRESETRESETRESET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs,obs_hist = self.env.get_observations()
        privileged_obs,prev_critic_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs,prev_critic_obs, obs_hist = obs.to(self.device), critic_obs.to(self.device),prev_critic_obs.to(self.device),obs_hist.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)


        torque_cost_buffer = deque(maxlen=100)
        jerk_cost_buffer = deque(maxlen=100)
        orientation_cost_buffer = deque(maxlen=100)
        velocity_cost_buffer = deque(maxlen=100)
        slip_cost_buffer = deque(maxlen=100)
        cur_torque_cost_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_jerk_cost_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_orientation_cost_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_velocity_cost_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_slip_cost_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)



        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    
                    actions = self.alg.act(obs, critic_obs,prev_critic_obs,obs_hist)

                    #prev_critic_obs = critic_obs
                    # print("######prev_critic_obs =====",prev_critic_obs)
                    obs, privileged_obs, prev_privileged_obs, obs_hist, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    prev_critic_obs = prev_privileged_obs

                    next_obs = obs
                    next_obs_hist = obs_hist
                    costs,torque_cost, jerk_cost, orientation_cost, velocity_cost, slip_cost = self.env.compute_cost()

                    obs, critic_obs, prev_critic_obs, obs_hist, rewards, dones, next_obs, next_obs_hist = obs.to(self.device), critic_obs.to(self.device), prev_critic_obs.to(self.device), obs_hist.to(self.device), rewards.to(self.device), dones.to(self.device), next_obs.to(self.device), next_obs_hist.to(self.device)
                    # print("######prev_critic_obs =====",prev_critic_obs[0,0],'\n',"#####critic_obs =====",critic_obs[0,0])
                    # print("######obs_hist =====",obs_hist[180,0],'\n',"#####obs =====",obs[0,0])
                    self.alg.process_env_step(rewards, dones, infos, next_obs, next_obs_hist, costs)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                        if self.cfg["algorithm_class_name"] == "LyapunovRuPPO":
                            cur_torque_cost_sum += torque_cost.squeeze(-1)
                            cur_jerk_cost_sum += jerk_cost.squeeze(-1)
                            cur_orientation_cost_sum += orientation_cost.squeeze(-1)
                            cur_velocity_cost_sum += velocity_cost.squeeze(-1)
                            cur_slip_cost_sum += slip_cost.squeeze(-1)
                            torque_cost_buffer.extend(cur_torque_cost_sum[new_ids][:, 0].cpu().numpy().tolist())
                            jerk_cost_buffer.extend(cur_jerk_cost_sum[new_ids][:, 0].cpu().numpy().tolist())
                            orientation_cost_buffer.extend(cur_orientation_cost_sum[new_ids][:, 0].cpu().numpy().tolist())
                            velocity_cost_buffer.extend(cur_velocity_cost_sum[new_ids][:, 0].cpu().numpy().tolist())
                            slip_cost_buffer.extend(cur_slip_cost_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_torque_cost_sum[new_ids] = 0
                            cur_jerk_cost_sum[new_ids] = 0
                            cur_orientation_cost_sum[new_ids] = 0
                            cur_velocity_cost_sum[new_ids] = 0
                            cur_slip_cost_sum[new_ids] = 0




                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss, mean_autoenc_loss = self.alg.update()
            if self.cfg["algorithm_class_name"] == "LyapunovRuPPO":
                Lya_lambda, Lya_deltaL, Lya_loss = self.alg.get_LyaLoss()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']
        wandb_dict = {}

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

                # wandb指令
                rew_key = key[4:]
                if rew_key in self.env.reward_scales:
                    wandb_dict['Episode_rew/' + key] = value / np.clip(np.abs(self.env.reward_scales[rew_key]),1e-11,None)
                    wandb_dict['Episode_rew_without_scale/' + key] = value



        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/autoenc_function', locs['mean_autoenc_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        wandb_dict['Loss/value_function'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/learning_time'] = locs['learn_time']

        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])


        if self.cfg["algorithm_class_name"] == "LyapunovRuPPO":
            wandb_dict['Lya/lambda'] = locs['Lya_lambda']
            wandb_dict['Lya/delta_L'] = locs['Lya_deltaL']
            wandb_dict['Lya/Lya_loss'] = locs['Lya_loss']

            if len(locs['torque_cost_buffer']) > 0:
                wandb_dict['LyaCost/mean_torque_cost'] = statistics.mean(locs['torque_cost_buffer'])
                wandb_dict['LyaCost/mean_jerk_cost'] = statistics.mean(locs['jerk_cost_buffer'])
                wandb_dict['LyaCost/mean_orientation_cost'] = statistics.mean(locs['orientation_cost_buffer'])
                wandb_dict['LyaCost/mean_velocity_cost'] = statistics.mean(locs['velocity_cost_buffer'])
                wandb_dict['LyaCost/mean_slip_cost'] = statistics.mean(locs['slip_cost_buffer'])

        wandb.log(wandb_dict, step=locs['it'])


        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Autoenc function loss:':>{pad}} {locs['mean_autoenc_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Autoenc function loss:':>{pad}} {locs['mean_autoenc_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
