from __future__ import annotations

import torch  
import torch.nn as nn
from torch.distributions import Normal



class DisturbanceNet(nn.Module):

    def __init__(self, cenet_in_dim,  activation="elu"):
        super().__init__()
        self.activation = get_activation(activation)

        self.f_head = nn.Sequential(
            nn.Linear(cenet_in_dim, 128),
            self.activation,
            nn.Linear(128, 16),
            self.activation,
            nn.Linear(16, 3)   # 输出外力 (Fx,Fy,Fz)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor (batch_size, cenet_in_dim)

        Returns:
            f_hat: Tensor (batch_size, 3)
                   estimated external disturbance force
        """
        return self.f_head(x)




def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None