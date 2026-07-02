from legged_gym.envs.base.base_config import BaseConfig
from legged_gym.managers import ManagerTermCfg, ObsGroup
from legged_gym.plotting import PlotTermCfg
from legged_gym.envs.go2w import mdp

class GO2WRoughCfg( BaseConfig ):
    task_name = 'DreamWaQ_go2w'
    # 训练环境类
    class env:
        num_envs = 4096 # 强化学习同时训练智能体的数量
        num_actions = 16 # 可操控的动作数量
        num_observations = 73 # 强化学习观测值的数量  
        num_obs_hist = 5
        num_privileged_obs = 320
        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 20
      
    # 机器人指令类
    class commands:
        curriculum = True # 是否使用课程学习
        max_curriculum = 1.5 # 课程难度最高级
        num_commands = 4 # 指令的个数：x轴方向线速度，y轴方向线速度，角速度以及航向
        resampling_time = 10. # 指令更改的时间
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s] x轴方向线速度
            lin_vel_y = [-1.0, 1.0]   # min max [m/s] y轴方向线速度
            ang_vel_yaw = [-1, 1]    # min max [r ad/s] 角速度
            heading = [-3.14, 3.14] # 航向 实际上没有使用这个维度

            # lin_vel_x = [0, 0] # min max [m/s] x轴方向线速度
            # lin_vel_y = [0, 0]   # min max [m/s] y轴方向线速度
            # ang_vel_yaw = [0, 0]    # min max [r ad/s] 角速度
            # heading = [0, 0] # 航向 实际上没有使用这个维度

        resample = ManagerTermCfg(func=mdp.resample_commands, env_arg=True)

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True  
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.5
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = None # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        # terrain_kwargs = {
        #     "type": "test1_rugged_terrain",
        #     "amplitude": 0.035,
        #     "triangle_scale": 0.40
        # }
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        class importer:
            terrain_type = "generator"
            mesh_type = "trimesh"
            max_init_terrain_level = 5
            use_terrain_origins = True
            rough_template = {
                "class_name": "legged_gym.terrains.generators.rough:RoughTerrainGenerator",
                "difficulty_range": [0.0, 1.0],
                "sub_terrains": {
                    "slope": {
                        "class_name": "legged_gym.terrains.generators.rough:PyramidSlopeTerrain",
                        "proportion": 0.1,
                        "terrain_type": 0,
                    },
                    "rough_slope": {
                        "class_name": "legged_gym.terrains.generators.rough:RandomRoughSlopeTerrain",
                        "proportion": 0.1,
                        "terrain_type": 1,
                    },
                    "stairs_down": {
                        "class_name": "legged_gym.terrains.generators.rough:PyramidStairsTerrain",
                        "proportion": 0.35,
                        "inverted": True,
                        "terrain_type": 2,
                    },
                    "stairs_up": {
                        "class_name": "legged_gym.terrains.generators.rough:PyramidStairsTerrain",
                        "proportion": 0.25,
                        "inverted": False,
                        "terrain_type": 3,
                    },
                    "discrete": {
                        "class_name": "legged_gym.terrains.generators.rough:DiscreteObstaclesTerrain",
                        "proportion": 0.2,
                        "terrain_type": 4,
                    },
                },
            }
            mgdp_mix_template = {
                "class_name": "legged_gym.terrains.generators.mix:MixTerrainGenerator",
                "difficulty_range": [0.0, 1.0],
                "terrain_dict": {
                    "slope down": 0.2,
                    "pyramid": 0.2,
                    "stairs down": 0.2,
                    "stairs up": 0.2,
                    "discrete obstacles": 1.1,
                    "hurdle": 0.2,
                    "gap": 1.2,
                    "ramp": 1.1,
                    "bream": 0.0,
                    "new stairs down": 0.3,
                    "pit": 1.0,
                },
            }
            mgdp_gap_parkour_template = {
                "class_name": "legged_gym.terrains.generators.gap_parkour:GapParkourTerrainGenerator",
                "difficulty_range": [0.0, 1.0],
                "num_goals": 10,
                "terrain_dict": {
                    "plane": 0.0,
                    "up_stairs": 0.0,
                    "down_stairs": 0.0,
                    "single-gap": 0.002,
                    "step-stone": 0.101,
                    "Stones-2Rows": 0.101,
                    "balance-2Stones": 0.0,
                    "stones-1Rows": 0.101,
                    "single-bridge": 0.101,
                    "step-Beams": 0.0,
                    "Rotation-Beams": 0.0,
                    "narrow-Beams": 0.0,
                    "cross-Beams": 0.0,
                    "air-Beams": 0.101,
                    "air_stone": 0.101,
                    "hurdle": 0.101,
                    "ramp": 0.101,
                    "corridor": 1.1,
                },
            }
            generator = rough_template

    # 机器人初始状态
    class init_state:
        pos = [0.0, 0.0, 0.5] # x,y,z [m] 初始位置 四元数表示
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        # 初始关节位置
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 0.8,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
            
            'FL_foot_joint':0.0,
            'RL_foot_joint':0.0,
            'FR_foot_joint':0.0,
            'RR_foot_joint':0.0,

        }
        init_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad] 机身
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad] 大腿
            'RL_thigh_joint': 0.8,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 0.8,   # [rad]

            'FL_calf_joint': -1.5,   # [rad] 小腿
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad] 

            'FL_foot_joint':0.0, # 轮足
            'RL_foot_joint':0.0,
            'FR_foot_joint':0.0,
            'RR_foot_joint':0.0,
        }

    # 机器人关节电机控制模式、参数
    class control:
        # PD Drive parameters:
        control_type = 'P' # 位置控制、速度控制、扭矩控制
        
        stiffness = {'hip_joint': 50.,'thigh_joint': 50.,'calf_joint': 50.,"foot_joint":0}  # [N*m/rad] 刚度系数k_p 
        damping = {'hip_joint': 1,'thigh_joint': 1,'calf_joint': 1,"foot_joint":0.5}     # [N*m*s/rad] 阻尼系数k_d
        # action scale: target angle = actionScale * action + defaultAngle
        # 乘一个缩放因子，目的是让动作值适应不同关节的运动范围
        action_scale = 0.25
        vel_scale = 10.0 # 轮子的速度缩放超参数
        # decimation: Number of control action updates @ sim DT per policy DT
        # 仿真环境的控制频率/decimation = 实际环境中的控制频率
        decimation = 4


    # 与机器人urdf相关参数
    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2w/urdf/go2w.urdf' # 存放位置
        name = "go2w"
        foot_name = "foot"
        wheel_name =["foot"] 
        joint_name = ["hip", "thigh", "calf"] 
        penalize_contacts_on = ["thigh", "calf", "base"] # 惩罚接触
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter "base","calf","hip","thigh"
        replace_cylinder_with_capsule = False
        flip_visual_attachments = True
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        thickness = 0.01


    # action与obs

    class actions:
        command_latency = ManagerTermCfg(
            func=mdp.command_latency,
            mode="decimation",
            env_arg=True,
            params={
                "enabled": True,
                "randomize": True,
                "latency_range": [1, 3],
            },
        )

    class observations:
        class actor(ObsGroup):
            imu = ManagerTermCfg(
                func=mdp.imu,
                env_arg=True,
                noise=mdp.imu_noise,
                params={
                    "latency_enabled": True,
                    "randomize_latency": True,
                    "latency_range": [1, 3],
                },
            )
            command = ManagerTermCfg(func="_obs_commands")
            motor = ManagerTermCfg(
                func=mdp.motor,
                env_arg=True,
                noise=mdp.motor_noise,
                params={
                    "latency_enabled": True,
                    "randomize_latency": True,
                    "latency_range": [1, 3],
                },
            )
            dof_pos = ManagerTermCfg(func=mdp.dof_pos, env_arg=True, noise=mdp.dof_pos_noise)
            action = ManagerTermCfg(func="_obs_actions")

        class critic(ObsGroup):
            policy = ManagerTermCfg(func="_obs_policy")
            base_lin_vel = ManagerTermCfg(func="_obs_base_lin_vel")
            contact_forces = ManagerTermCfg(func="_obs_contact_forces")
            heights = ManagerTermCfg(func="_obs_height_measurements")

    # 奖励配置

    class rewards:
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.4 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 1.
        base_height_target = 0.34
        max_contact_force = 100. # forces above this value are penalized

    class rewards_manager:
        tracking_lin_vel = ManagerTermCfg(func=mdp.tracking_lin_vel, scale=3.0, env_arg=True)
        tracking_ang_vel = ManagerTermCfg(func=mdp.tracking_ang_vel, scale=1.5, env_arg=True)
        lin_vel_z = ManagerTermCfg(func=mdp.lin_vel_z, scale=-0.1, env_arg=True)
        ang_vel_xy = ManagerTermCfg(func=mdp.ang_vel_xy, scale=-0.05, env_arg=True)
        orientation = ManagerTermCfg(func=mdp.orientation, scale=-2.0, env_arg=True)
        torques = ManagerTermCfg(func=mdp.torques, scale=-0.0002, env_arg=True)
        dof_vel = ManagerTermCfg(func=mdp.dof_vel, scale=-1e-7, env_arg=True)
        dof_acc = ManagerTermCfg(func=mdp.dof_acc, scale=-1e-7, env_arg=True)
        base_height = ManagerTermCfg(func=mdp.base_height, scale=-0.5, env_arg=True)
        feet_air_time = ManagerTermCfg(func=mdp.feet_air_time, scale=-0.5, env_arg=True)
        collision = ManagerTermCfg(func=mdp.collision, scale=-0.1, env_arg=True)
        feet_stumble = ManagerTermCfg(func=mdp.feet_stumble, scale=-0.5, env_arg=True)
        action_rate = ManagerTermCfg(func=mdp.action_rate, scale=-0.0002, env_arg=True)
        stand_still = ManagerTermCfg(func=mdp.stand_still, scale=-0.01, env_arg=True)
        dof_pos_limits = ManagerTermCfg(func=mdp.dof_pos_limits, scale=-0.9, env_arg=True)
        hip_action_l2 = ManagerTermCfg(func=mdp.hip_action_l2, scale=-0.1, env_arg=True)
        joint_power = ManagerTermCfg(func=mdp.joint_power, scale=-2e-5, env_arg=True)
        default_pos = ManagerTermCfg(func=mdp.default_pos, scale=-0.05, env_arg=True)




    class terminations:
        illegal_contact = ManagerTermCfg(func="_termination_illegal_contact")
        base_height_contact = ManagerTermCfg(func=mdp.base_height_contact, env_arg=True)

    class events:
        latency_update = ManagerTermCfg(func=mdp.update_latency_buffers, mode="decimation", env_arg=True)
        height_measurement = ManagerTermCfg(func=mdp.update_height_measurements, mode="step", env_arg=True)
        push_robot = ManagerTermCfg(
            func=mdp.push_robots,
            mode="step",
            env_arg=True,
            params={
                "enabled": False,
                "interval_s": 15,
                "max_vel_xy": 1.0,
            },
        )
        latency_reset = ManagerTermCfg(func=mdp.reset_latency_buffers, mode="reset", env_arg=True)
        friction = ManagerTermCfg(
            func=mdp.randomize_friction,
            mode="asset_init",
            env_arg=True,
            params={
                "enabled": True,
                "friction_range": [0.2, 1.25],
            },
        )
        rigid_body_props = ManagerTermCfg(
            func=mdp.randomize_rigid_body_props,
            mode="asset_init",
            env_arg=True,
            params={
                "randomize_base_mass": True,
                "added_mass_range": [-1.0, 2.0],
                "randomize_link_mass": True,
                "multiplied_link_mass_range": [0.9, 1.1],
                "randomize_base_com": True,
                "added_base_com_range": [-0.03, 0.03],
            },
        )
        dof_props = ManagerTermCfg(
            func=mdp.init_dof_props,
            mode="asset_init",
            env_arg=True,
            params={
                "default_joint_friction": [0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0],
                "default_joint_stiffness": [0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0],
                "default_joint_damping": [0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0],
                "default_joint_armature": [0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0],
            },
        )
        motor_zero_offset = ManagerTermCfg(
            func=mdp.randomize_motor_zero_offset,
            mode="asset_init",
            env_arg=True,
            params={
                "enabled": True,
                "offset_range": [-0.035, 0.035],
            },
        )
        pd_gains = ManagerTermCfg(
            func=mdp.randomize_pd_gains,
            mode="asset_init",
            env_arg=True,
            params={
                "enabled": True,
                "stiffness_multiplier_range": [0.9, 1.1],
                "damping_multiplier_range": [0.9, 1.1],
            },
        )
        joint_friction = ManagerTermCfg(
            func=mdp.randomize_joint_friction,
            mode="reset",
            env_arg=True,
            params={
                "enabled": False,
                "friction_range": [0.9, 1.1],
            },
        )

    class normalization:
        contact_force_range = [0.0, 100.0]
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            quat = 1.
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            quat = 0.1
            height_measurements = 0.1

    class viewer:
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]

    class plots:
        enabled = False
        backend = "matplotlib"
        interval = 1
        max_steps = 1000
        show = True
        terms = {
            "joint_state": PlotTermCfg(
                func="legged_gym.plotting.terms:joint_state",
                params={
                    "env_index": 0,
                    "joint_index": 2,
                    "key_prefix": "joint",
                },
            ),
        }
        figures = [
            {
                "name": "joint_state",
                "title": "Joint State",
                "layout": [3, 1],
                "plots": [
                    {"title": "position", "series": ["joint_pos"], "ylabel": "rad"},
                    {"title": "velocity", "series": ["joint_vel"], "ylabel": "rad/s"},
                    {"title": "torque", "series": ["joint_torque"], "ylabel": "Nm"},
                ],
            }
        ]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2

class GO2WRoughCfgPPO( BaseConfig ):
    seed = 5

    actor = {
        "class_name": "ActorModel",
        "backbone": {
            "class_name": "rsl_rl.models:DreamWaQActorBackbone",
            "actor_hidden_dims": [512, 256, 128],
            "encoder_hidden_dims": [128, 64],
            "decoder_hidden_dims": [64, 128],
            "latent_dim": 19,
            "velocity_dim": 3,
            "velocity_target_group": "prev_critic_base_lin_vel",
            "decoder_output_group": "actor",
            "activation": "elu",
            "autoencoder_loss_coef": 0.25,
            "velocity_loss_coef": 1.0,
            "reconstruction_loss_coef": 1.0,
            "kl_loss_coef": 1.0,
        },
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    }

    critic = {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": False,
    }

    class algorithm:
        class_name = "PPO"
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        entropy_coef = 0.003

        # obs = [imu(6), cmd(3), motor(q,dq,32), dof_pos(16), action(16)] -> 73
        plugins = [
            {
                "class_name": "rsl_rl.algorithms.plugins:SymmetryLossPlugin",
                "obs_permutation": [
                    -0.0001, -1, 2, -3, 4, -5, 6, -7, -8,
                    -13, 14, 15, 16, -9.0001, 10, 11, 12, -21, 22, 23, 24, -17, 18, 19, 20,
                    -29, 30, 31, 32, -25.0001, 26, 27, 28, -37, 38, 39, 40, -33, 34, 35, 36,
                    -45, 46, 47, 48, -41.0001, 42, 43, 44, -53, 54, 55, 56, -49, 50, 51, 52,
                    -61, 62, 63, 64, -57.0001, 58, 59, 60, -69, 70, 71, 72, -65, 66, 67, 68,
                ],
                "act_permutation": [-4, 5, 6, 7, -0.0001, 1, 2, 3, -12, 13, 14, 15, -8, 9, 10, 11],
                "frame_stack": 5,
                "sym_coef": 1.0,
            }
        ]
        
    class runner:
        runner_class_name = "rsl_rl.runners:OnPolicyRunner"
        logger = "wandb"
        wandb_project = "rough_go2w"
        wandb_mode = "online"
        save_interval = 500 # check for potential saves every this many iterations
        run_name = 'blind_50_1_40_0.5_trmeish'
        experiment_name = 'rough_go2w'
        num_steps_per_env = 24 # per iteration
        max_iterations = 10000
        load_run = -1
        checkpoint = -1
        resume = False
        resume_path = None
  
