# In discrete_policy.py

from torch.distributions import Categorical
import torch.nn as nn
import torch
import numpy as np

class DiscreteFF(nn.Module):
    def __init__(self, input_shape, n_actions, shared_layer_sizes, policy_layer_sizes, device):
        super().__init__()
        self.device = device

        shared_layers = []
        last_size = input_shape
        for size in shared_layer_sizes:
            shared_layers.append(nn.Linear(last_size, size))
            shared_layers.append(nn.LayerNorm(size))
            shared_layers.append(nn.ReLU())
            last_size = size
        
        self.shared_head = nn.Sequential(*shared_layers).to(self.device)


        policy_layers = []

        for size in policy_layer_sizes:
            policy_layers.append(nn.Linear(last_size, size))
            policy_layers.append(nn.LayerNorm(size))
            policy_layers.append(nn.ReLU())
            last_size = size

        policy_layers.append(nn.Linear(last_size, n_actions))
        policy_layers.append(nn.Softmax(dim=-1))

        self.policy = nn.Sequential(*policy_layers).to(self.device)

        self.n_actions = n_actions

    def forward(self, obs):

        shared_output = self.shared_head(obs)
        policy_output = self.policy(shared_output)
        return policy_output

    def get_output(self, obs):
        t = type(obs)
        if t != torch.Tensor:
            if t != np.array:
                obs = np.asarray(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.forward(obs)


    def get_action(self, obs, deterministic=True):
        probs = self.get_output(obs)
        probs = probs.view(-1, self.n_actions)
        probs = torch.clamp(probs, min=1e-11, max=1)
        if deterministic:
            return probs.cpu().numpy().argmax(), 1
        action = torch.multinomial(probs, 1, True)
        log_prob = torch.log(probs).gather(-1, action)
        return action.flatten().cpu(), log_prob.flatten().cpu()

    def get_backprop_data(self, obs, acts):
        acts = acts.long()
        probs = self.get_output(obs)
        probs = probs.view(-1, self.n_actions)
        probs = torch.clamp(probs, min=1e-11, max=1)
        log_probs = torch.log(probs)
        action_log_probs = log_probs.gather(-1, acts)
        entropy = -(log_probs * probs).sum(dim=-1)
        return action_log_probs.to(self.device), entropy.to(self.device).mean()