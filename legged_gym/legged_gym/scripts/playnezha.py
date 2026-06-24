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
import matplotlib
import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry , Logger
from legged_gym.utils.helpers import export_vae_policy_as_jit, disable_manager_randomization

import numpy as np
import torch


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
    x, y, z, w = q[0], q[1], q[2], q[3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [roll, pitch, yaw]


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    # env_cfg.terrain.num_rows = 5
    # env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    disable_manager_randomization(env_cfg)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs,obs_hist = env.get_observations()

    # # === 创建撞击方块 ===
    # gym = env.gym
    # sim_box = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    # asset_options = gymapi.AssetOptions()
    # asset_options.disable_gravity = False
    # asset_options.fix_base_link = False
    # asset_box = gym.create_box(sim_box, 0.1, 0.1, 0.1, asset_options)  # 方块尺寸 

    # # 创建一个额外的 env 容器来管理方块
    # env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
    # env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
    # env_box = gym.create_env(sim_box, env_lower, env_upper, 1)

    # pose = gymapi.Transform()
    # pose.p = gymapi.Vec3(1.0, 0.0, 0.5)  # 初始位置（前方 1m，高度 0.5m）
    # pose.r = gymapi.Quat(0, 0, 0, 1)
    # box_handle = gym.create_actor(env_box, asset_box, pose, "impact_box", 0, 0)


    # # 设置方块物理参数（反弹、摩擦等）
    # shape_props = gym.get_actor_rigid_shape_properties(env_box, box_handle)
    # shape_props[0].restitution = 1
    # shape_props[0].compliance = 0.5
    # gym.set_actor_rigid_shape_properties(env_box, box_handle, shape_props)
    # gym.set_rigid_body_color(env_box, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(200/255, 200/255, 10/255))

    # # 保存方块状态
    # box_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim_box)).clone()






    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_vae_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    
    # export policy as a jit module (used to run it from C++)
    # if EXPORT_POLICY:
    #     path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    #     export_policy_as_jit_actor(ppo_runner.alg.actor_critic, path)
    #     export_policy_as_jit_encoder(ppo_runner.alg.actor_critic,path)
    #     print('Exported policy as jit script to: ', path)


    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 2 # which joint is used for logging
    stop_state_log = 1000 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
     
    for i in range(10*int(env.max_episode_length)):          
        actions = policy(obs.detach(),obs_hist.detach())
        obs, _, _, obs_hist, rews, dones, infos = env.step(actions.detach())
        # obs[:,6] = 0.0
        # obs[:,7] = 2.0
        # obs[:,8] = 0.0

        # === 💥 每隔 100 步，对单个机器人施加一次“速度冲击”来模拟外力干扰 ===
        # if i % 100 == 0:
        #     root_states = gymtorch.wrap_tensor(env.gym.acquire_actor_root_state_tensor(env.sim))
        #     quat = root_states[0, 3:7].unsqueeze(0)
        #     local_push = torch.tensor([[0.0, 2.5, 0.0]], device=env.device)
        #     world_push = quat_apply(quat, local_push)[0]
        #     root_states[0, 7:10] += world_push * 1.0   # 推力大小可调
        #     env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(root_states))
        #     print("💨 速度脉冲已施加")


        # ===================================================
        # 💥 每隔 200 步，对机器人质心施加一次持续外力干扰（机身方向）
        # ===================================================
        # if i == 0:
        #     push_active = False        # 当前是否在施力
        #     push_until = 0             # 推力结束的步数
        #     current_force = torch.zeros(3, device=env.device)
        #     push_duration = 20         # 持续 40 帧（约 0.1 秒）
        #     push_interval = 200        # 每 200 步触发一次（约 0.5 秒）
        #     push_force = 150.0        # 推力大小（N）
        #     use_local_direction = True # 是否随机身姿态施力方向变化

        # # === 1️⃣ 触发新推力 ===
        # if (i % push_interval == 0) and (i > 0):
        #     push_active = True
        #     push_until = i + push_duration

        #     # 获取机器人姿态
        #     root_states = gymtorch.wrap_tensor(env.gym.acquire_actor_root_state_tensor(env.sim))
        #     quat = root_states[0, 3:7].unsqueeze(0)

        #     # 推力方向：机身坐标的 +Y（也可改为 [1,0,0] 表示前向）
        #     local_push = torch.tensor([[0.0, 1.0, 0.0]], device=env.device)

        #     if use_local_direction:
        #         # 将局部方向旋转到世界坐标系下
        #         q_vec = quat[..., :3]
        #         q_w = quat[..., 3]
        #         cross1 = torch.cross(q_vec, local_push, dim=-1)
        #         cross2 = torch.cross(q_vec, cross1 + q_w.unsqueeze(-1) * local_push, dim=-1)
        #         world_dir = local_push + 2.0 * (cross1 + cross2)
        #         world_dir = world_dir[0] / (torch.norm(world_dir) + 1e-6)
        #     else:
        #         world_dir = torch.tensor([0.0, 1.0, 0.0], device=env.device)

        #     # 计算最终施力向量
        #     current_force = world_dir * push_force
        #     print(f"💥 Step {i}: 启动持续推力 {current_force.cpu().numpy()}，持续 {push_duration} 帧")

        # # === 2️⃣ 持续施力阶段 ===
        # if push_active:
        #     # 构建力张量：只对根刚体（质心）施力
        #     forces = torch.zeros((env.num_envs, env.num_bodies, 3), device=env.device)
        #     forces[:, 0, :] = current_force

        #     # 在当前仿真步施加力
        #     env.gym.apply_rigid_body_force_tensors(
        #         env.sim,
        #         gymtorch.unwrap_tensor(forces),
        #         None,
        #         gymapi.ENV_SPACE
        #     )

        #     # 判断是否结束持续施力
        #     if i >= push_until:
        #         push_active = False
        #         print(f"🕓 Step {i}: 推力结束")



        # === 方块撞击 ===
        # root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        # robot_pos = root_states[0, :3].cuda().cpu().numpy()
        # robot_quat = root_states[0, 3:7].cuda().cpu().numpy()
        # robot_ang = np.array(quaternion2rpy(robot_quat))
        # box_pos_bias = np.array([np.random.randint(-30, 30)*0.1, np.random.randint(-20, 20)*0.1, np.random.randint(0, 20)*0.1])
        # box_position = (trans_matrix_ba(robot_pos, [0, 0, robot_ang[-1]]) @ np.append(box_pos_bias, 1))[:-1]
        # if i % 300 == 0:  # 每隔 300 步触发一次
        #     print("💥 发射方块撞击狗")
        #     box_pos = torch.tensor([box_position[0], box_position[1], box_position[2]]).to(box_states.device)
        #     box_vel = (robot_pos - box_position) * np.random.randint(3, 7)
        #     box_vel = torch.tensor([box_vel[0], box_vel[1], box_vel[2]]).to(box_states.device)
        #     root_states[1, :3] = box_pos
        #     root_states[1, 7:10] = box_vel
        #     # 选择方块的索引（通常是机器人actor数之后）
        #     # box_idx = env.num_envs  # 如果每个env只有一个actor，这样不会覆盖机器人
        #     # env_idx_t = torch.tensor([box_idx], dtype=torch.int32, device=env.device)

        #     gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))


        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)


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
