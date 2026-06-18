import copy
import os
import torch

def export_vae_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        # copy policy parameters
        if hasattr(policy, "actor") and hasattr(policy, "vae"):
            self.actor = copy.deepcopy(policy.actor)
            self.vae = copy.deepcopy(policy.vae)
        else:
            raise ValueError("Policy does not have an actor/vae module.")

        # # copy normalizer if exists
        # if normalizer:
        #     self.normalizer = copy.deepcopy(normalizer)
        # else:
        #     self.normalizer = torch.nn.Identity()


    def forward(self, observations, history_observations):
        estimation, latent_params = self.vae(history_observations)
        z, v = estimation
        actions = self.actor(torch.cat((z, v, observations), dim=-1))
        return actions


    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
