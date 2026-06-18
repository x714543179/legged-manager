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
import sys  
sys.path.append("/home/hu/csq/DreamWaQ/legged_gym")
import matplotlib
import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry , Logger
from legged_gym.utils.helpers import export_policy_as_jit_actor,export_policy_as_jit_encoder,class_to_dict, disable_manager_randomization
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import quat_rotate

import numpy as np
import torch
import pickle
import math



def quat_to_yaw(q):
    """
    输入: q = [x, y, z, w]
    返回: yaw (弧度)
    """
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # yaw = atan2(2(wz + xy), 1 - 2(y² + z²))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return yaw


def rotate_yaw_only(vec, yaw):
    """
    只绕 Z 轴旋转 vec = [vx, vy, vz]
    """
    c = torch.cos(yaw)
    s = torch.sin(yaw)

    vx = vec[..., 0]
    vy = vec[..., 1]

    wx = c * vx - s * vy
    wy = s * vx + c * vy
    wz = vec[..., 2]

    return torch.stack([wx, wy, wz], dim=-1)


# === 手动四元数旋转函数（与 quat_rotate 等价） ===
def quat_apply(q, v):
    """
    q: shape (..., 4)  四元数 (x, y, z, w)
    v: shape (..., 3)  向量
    返回值: v 在世界坐标下的旋转结果
    """
    q_w = q[..., 3]
    q_vec = q[..., 0:3]
    cross1 = torch.cross(q_vec, v, dim=-1)
    cross2 = torch.cross(q_vec, cross1 + q_w.unsqueeze(-1) * v, dim=-1)
    return v + 2 * (cross1 + cross2)


def trans_matrix_ba(m, t):
    r = np.array([[np.cos(t[2]) * np.cos(t[1]),
                   np.cos(t[2]) * np.sin(t[1]) * np.sin(t[0]) - np.sin(t[2]) * np.cos(t[0]),
                   np.cos(t[2]) * np.sin(t[1]) * np.cos(t[0]) + np.sin(t[2]) * np.sin(t[0])],
                  [np.sin(t[2]) * np.cos(t[1]),
                   np.sin(t[2]) * np.sin(t[1]) * np.sin(t[0]) + np.cos(t[2]) * np.cos(t[0]),
                   np.sin(t[2]) * np.sin(t[1]) * np.cos(t[0]) - np.cos(t[2]) * np.sin(t[0])],
                  [-np.sin(t[1]), np.cos(t[1]) * np.sin(t[0]), np.cos(t[1]) * np.cos(t[0])]])
    trans = np.hstack([r, np.array(m)[:, np.newaxis]])
    trans = np.vstack([trans, np.array([[0, 0, 0, 1]])])
    return trans


def quaternion2rpy(q):
    # Isaac Gym: q = [x, y, z, w]
    x, y, z, w = q[0], q[1], q[2], q[3]

    # 重新排成 RPY 常用顺序 [w, x, y, z]
    qw = w
    qx = x
    qy = y
    qz = z

    # roll (x-axis rotation)
    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (qw * qy - qz * qx)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (qw * qz + qx * qy)
    t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(t3, t4)

    return [roll, pitch, yaw]





def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    class_to_dict(env_cfg)
    class_to_dict(train_cfg)
    
    with open('env_cfg.pkl', 'wb') as f:
        pickle.dump(class_to_dict(env_cfg), f)
    with open('train_cfg.pkl', 'wb') as f:
        pickle.dump(train_cfg, f)
    # override some parameters for testing
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    # env_cfg.terrain.num_rows = 5
    # env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    disable_manager_randomization(env_cfg)

    # 设置reset时间
    env_cfg.env.episode_length_s = 60 # 单位秒

    # 设置地面为plane
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.selected = True
    env_cfg.terrain.terrain_kwargs = {
            "type": "test1_rugged_terrain",
            "amplitude": 0.04,
            "triangle_scale": 0.50
        }

    # 种子


    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs,obs_hist = env.get_observations()


    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit_actor(ppo_runner.alg.actor_critic, path)
        export_policy_as_jit_encoder(ppo_runner.alg.actor_critic,path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 2 # which joint is used for logging

    stop_state_log = 3000 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    

    # 
    base_origin_x = None
    base_origin_y = None

    base_origin_yaw = None


    MAX_FORCE = 50.0  # 扰动力最大值（N）
    # 固定随机种子，让所有策略使用相同的扰动序列
    DISTURB_SEED = 1122
    rng = np.random.RandomState(DISTURB_SEED)

    # 10*int(env.max_episode_length)
    for i in range(3001):          
        actions = policy(obs.detach(),obs_hist.detach())
        obs, _, _, obs_hist, rews, dones, infos = env.step(actions.detach())
        # obs[:,6] = 0.0
        # obs[:,7] = 2.0
        # obs[:,8] = 0.0

        if i > 10 :
            # ===============================
            # 💥 每个仿真步施加一次随机扰动
            # ===============================
            # 1) 幅值 [0, MAX_FORCE]
            mag = rng.uniform(50, 50 + MAX_FORCE)   # numpy float

            # 2) 随机方向 [0, 2π)
            theta = rng.uniform(0, 2 * np.pi)


            # 3) 世界坐标方向
            world_dir = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float32)

            # 4) 最终施力（numpy → torch）
            current_force = torch.tensor(mag * world_dir, device=env.device).float()

            # 5) 构建力张量
            forces = torch.zeros((env.num_envs, env.num_bodies, 3), device=env.device)
            forces[:, 0, :] = current_force

            env.gym.apply_rigid_body_force_tensors(
                env.sim,
                gymtorch.unwrap_tensor(forces),
                None,
                gymapi.ENV_SPACE
            )


        if i % 100 == 0 and i > 0:
            print(f"[Step {i}] Disturbance force magnitude = {mag:.4f} N,  direction(deg)={theta*57.296:.2f}")


        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)



        if base_origin_x is None:
            base_origin_x = env.base_pos[robot_index, 0].item()
            base_origin_y = env.base_pos[robot_index, 1].item()


        _,_,base_yaw  = quaternion2rpy(env.base_quat[robot_index].cpu().numpy())
        base_yaw *= 57.296
        if base_origin_yaw is None:
            base_origin_yaw = base_yaw



        if i < stop_state_log:



            logger.log_states(
                {
                    'com_xy_pos': env.base_pos[robot_index, 0].item(),
                    'base_pos_x': env.base_pos[robot_index, 0].item() - base_origin_x,
                    'base_pos_y': env.base_pos[robot_index, 1].item() - base_origin_y,

                    'base_yaw': base_yaw,


                }

            )
            
        elif i==stop_state_log:
            logger.PlotStable()

        # if i < stop_state_log:
        #     logger.log_states(
        #         {
        #             'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
        #             'dof_pos': env.dof_pos[robot_index, joint_index].item(),
        #             'dof_vel': env.dof_vel[robot_index, joint_index].item(),
        #             'dof_torque': env.torques[robot_index, joint_index].item(),
        #             'command_x': env.commands[robot_index, 0].item(),
        #             'command_y': env.commands[robot_index, 1].item(),
        #             'command_yaw': env.commands[robot_index, 2].item(),
        #             'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
        #             'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
        #             'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
        #             'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
        #             'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
        #             'dof_torque_0': env.torques[robot_index, 0].item(),
        #             'dof_torque_1': env.torques[robot_index, 1].item(),
        #             'dof_torque_2': env.torques[robot_index, 2].item(),
        #             'dof_torque_3': env.torques[robot_index, 3].item(),
        #             'dof_torque_4': env.torques[robot_index, 4].item(),
        #             'dof_torque_5': env.torques[robot_index, 5].item(),
        #             'dof_torque_6': env.torques[robot_index, 6].item(),
        #             'dof_torque_7': env.torques[robot_index, 7].item(),
        #             'dof_torque_8': env.torques[robot_index, 8].item(),
        #             'dof_torque_9': env.torques[robot_index, 9].item(),
        #             'dof_torque_10': env.torques[robot_index, 10].item(),
        #             'dof_torque_11': env.torques[robot_index, 11].item(),
        #             'dof_torque_12': env.torques[robot_index, 12].item(),
        #             'dof_torque_13': env.torques[robot_index, 13].item(),
        #             'dof_torque_14': env.torques[robot_index, 14].item(),
        #             'dof_torque_15': env.torques[robot_index, 15].item(),



        #             'dof_pos_0': env.dof_pos[robot_index, 0].item(),
        #             'dof_pos_1': env.dof_pos[robot_index, 1].item(),
        #             'dof_pos_2': env.dof_pos[robot_index, 2].item(),
        #             'dof_pos_3': env.dof_pos[robot_index, 3].item(),
        #             'dof_pos_4': env.dof_pos[robot_index, 4].item(),
        #             'dof_pos_5': env.dof_pos[robot_index, 5].item(),
        #             'dof_pos_6': env.dof_pos[robot_index, 6].item(),
        #             'dof_pos_7': env.dof_pos[robot_index, 7].item(),
        #             'dof_pos_8': env.dof_pos[robot_index, 8].item(),
        #             'dof_pos_9': env.dof_pos[robot_index, 9].item(),
        #             'dof_pos_10': env.dof_pos[robot_index, 10].item(),
        #             'dof_pos_11': env.dof_pos[robot_index, 11].item(),
                    
        #         }
        #     )
        # elif i==stop_state_log:
        #     logger.plot_states()
        # if  0 < i < stop_rew_log:
        #     if infos["episode"]:
        #         num_episodes = torch.sum(env.reset_buf).item()
        #         if num_episodes>0:
        #             logger.log_rewards(infos["episode"], num_episodes)
        # elif i==stop_rew_log:
        #     logger.print_rewards()





if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
