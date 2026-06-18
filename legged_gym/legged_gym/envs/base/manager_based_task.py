import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.managers import (
    ActionManager,
    CommandManager,
    EventManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
)
from legged_gym.utils.math import get_scale_shift


class ManagerBasedTask(BaseTask):
    """Thin task base that owns the RL lifecycle and manager dispatch."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False

        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._init_managers()
        self.init_done = True

    def step(self, actions):
        if hasattr(self, "disturbance_force"):
            self.disturbance_force = self.disturbance_force.to(self.device)
        self.observation_manager.update_history_before_step()
        self.prev_privileged_obs_buf = self.privileged_obs_buf
        if hasattr(self, "foot_velocities"):
            self.prev_foot_velocities = self.foot_velocities

        self.action_manager.process(actions)
        self.render()
        for _ in range(self.cfg.control.decimation):
            control_actions = self._actions_for_torque()
            self.torques = self._compute_torques(control_actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.event_manager.apply("decimation")
            self._post_decimation_step()
        self.post_physics_step()

        self.observation_manager.finalize()
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.prev_privileged_obs_buf,
            self.obs_hist_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def post_physics_step(self):
        self._refresh_sim_tensors()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._update_common_quantities()
        self._refresh_task_tensors()
        self.command_manager.compute()
        self.event_manager.apply("step")

        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()
        self._update_post_step_buffers()

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        return self.termination_manager.compute()

    def _check_termination_impl(self):
        self.reset_buf = self._termination_illegal_contact()
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.command_manager.reset(env_ids)
        self.event_manager.reset(env_ids)

        if hasattr(self, "slast_actions"):
            self.slast_actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        if hasattr(self, "last_joint_pos_target"):
            self.last_joint_pos_target[env_ids] = 0.0
        if hasattr(self, "last_last_joint_pos_target"):
            self.last_last_joint_pos_target[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.action_manager.reset(env_ids)
        self.observation_manager.reset(env_ids)
        self._reset_task_buffers(env_ids)

        self.extras["episode"] = {}
        self.extras["episode"].update(self.reward_manager.reset(env_ids))
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        return self.reward_manager.compute()

    def compute_observations(self):
        return self.observation_manager.compute()

    def _obs_base_ang_vel(self):
        return self.base_ang_vel * self.obs_scales.ang_vel

    def _obs_projected_gravity(self):
        return self.projected_gravity

    def _obs_commands(self):
        return self.commands[:, :3] * self.commands_scale

    def _obs_dof_pos_error(self):
        return (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos

    def _obs_dof_vel(self):
        return self.dof_vel * self.obs_scales.dof_vel

    def _obs_actions(self):
        return self.actions

    def _obs_policy(self):
        return self.obs_buf

    def _obs_base_lin_vel(self):
        return self.base_lin_vel * self.obs_scales.lin_vel

    def _obs_contact_forces(self):
        contact_forces_scale, contact_forces_shift = get_scale_shift(self.cfg.normalization.contact_force_range)
        return (self.contact_forces.view(self.num_envs, -1) - contact_forces_shift) * contact_forces_scale

    def _obs_height_measurements(self):
        return torch.clip(
            self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0
        ) * self.obs_scales.height_measurements

    def _post_process_observations(self):
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _termination_illegal_contact(self):
        if len(self.termination_contact_indices) == 0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)

    def _event_resample_commands(self, env_ids=None):
        self.command_manager.compute()

    def _event_measure_heights(self, env_ids=None):
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

    def _actions_for_torque(self):
        return self.actions

    def _post_decimation_step(self):
        pass

    def _refresh_sim_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _update_common_quantities(self):
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if hasattr(self, "foot_positions"):
            self.foot_positions[:] = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[
                :, self.feet_indices, 0:3
            ]
        if hasattr(self, "foot_velocities"):
            self.foot_velocities[:] = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[
                :, self.feet_indices, 7:10
            ]

    def _refresh_task_tensors(self):
        pass

    def _update_post_step_buffers(self):
        if hasattr(self, "slast_actions"):
            self.slast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        if hasattr(self, "last_last_joint_pos_target"):
            self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        if hasattr(self, "last_joint_pos_target"):
            self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def _init_managers(self):
        self.action_manager = ActionManager(self, getattr(self.cfg, "actions", None))
        self.command_manager = CommandManager(self, getattr(self.cfg, "commands", None))
        self.observation_manager = ObservationManager(self, getattr(self.cfg, "observations", None))
        self.reward_manager = RewardManager(self, getattr(self.cfg, "rewards_manager", None))
        self.termination_manager = TerminationManager(self, getattr(self.cfg, "terminations", None))
        self.event_manager = EventManager(self, getattr(self.cfg, "events", None))
        self.managers = {
            "action": self.action_manager,
            "command": self.command_manager,
            "observation": self.observation_manager,
            "reward": self.reward_manager,
            "termination": self.termination_manager,
            "event": self.event_manager,
        }
        self.reward_scales = self.reward_manager.reward_scales
        self.episode_sums = self.reward_manager.episode_sums
        self.reward_names = [name for name in self.reward_manager.active_terms if name != "termination"]
        self.reward_functions = [
            self.reward_manager._resolve_callable(term.func)
            for name, term in zip(self.reward_manager.active_terms, self.reward_manager._terms)
            if name != "termination"
        ]
