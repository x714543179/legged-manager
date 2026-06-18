# lyapunov_critic.py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class LyaCritic(nn.Module):
    def __init__(self,  state_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim , hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    
    def forward(self, state):
        x = torch.cat([state], 1)   
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        L = F.softplus(self.fc3(x))
        return L