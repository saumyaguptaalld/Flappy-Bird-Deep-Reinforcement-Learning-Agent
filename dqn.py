import torch
import torch.nn as nn

class DQN(nn.Module):
    # input dim, output dim, hidden dim
    def __init__(self, state_dim = 12, action_dim = 2, hidden_dim = 256): # 12 options constitute 1 action state(but unrelated to this 12 , we take 12 input dim), there are two action 0(no flap) and 1 (flap)
        super(DQN,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        return self.model(x)
