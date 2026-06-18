import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic_DWAQ import ActorCritic_DWAQ
from .him_estimator import HIMEstimator



class ActorCritic_HIM(ActorCritic_DWAQ):
    """
    HAC-LOCO Stage 1: Low-level policy with additional velocity and force estimation heads.
    Structure strictly matches paper Fig.2:
        Encoder: [256,128,64]
        f_head: [32,16]
        v_head: [32,16]
        Decoder: [512,256,128]
    """
    def __init__(self, num_actor_obs, num_critic_obs, num_actions,
                 cenet_in_dim, cenet_out_dim, activation="elu", init_noise_std=1.0, ):
        super().__init__(num_actor_obs, num_critic_obs, num_actions,
                         cenet_in_dim, cenet_out_dim, activation, init_noise_std)

        self.num_one_step_obs = (num_actor_obs - cenet_out_dim)   # 单步本体obs
        self.history_size = cenet_in_dim / self.num_one_step_obs
        
        # Estimator
        self.estimator = HIMEstimator(temporal_steps = self.history_size, num_one_step_obs = self.num_one_step_obs) 



    def act(self, observations, obs_history, deterministic_for_grad=False, **kwargs):
        with torch.no_grad():
            vel, latent = self.estimator(obs_history)
        actor_input = torch.cat((observations, vel, latent), dim=-1)
        self.update_distribution(actor_input)
        # self.last_force_est = f_hat.detach()

        if deterministic_for_grad:
            return self.distribution.rsample()
        else:
            return self.distribution.sample()

    def act_inference(self, observations, obs_history):
        vel, latent = self.estimator(obs_history)
        actor_input = torch.cat((observations, vel, latent), dim=-1)
        actions_mean = self.actor(actor_input)
        return actions_mean