import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_acts, hidden, lr = 3e-4):
        super(ActorCritic, self).__init__()

        self.learning_rate = lr
        
        self.num_acts = num_acts
        self.critic_lin1 = nn.Linear(num_inputs, hidden)
        self.critic_lin2 = nn.Linear(hidden, 1)

        self.actor_lin1 = nn.Linear(num_inputs, hidden)
        self.actor_lin2 = nn.Linear(hidden, num_acts)
        
    def forward(self, state):
        val = F.relu(self.critic_lin1(state))
        val = self.critic_lin2(val)

        policy_dist = F.relu(self.actor_lin1(state))
        policy_dist = F.softmax(self.actor_lin2(policy_dist))

        return val, policy_dist

