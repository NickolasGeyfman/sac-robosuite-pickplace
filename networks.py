import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np


class CriticNetwork(nn.Module):
    """
    The CriticNetwork computes Q(s, a).
    It has two fully-connected layers, then outputs a single Q-value.
    """
    def __init__(
        self,
        beta,                     # learning rate
        input_dims,               # state shape (e.g. [obs_dim])
        n_actions,                # number of action dimensions
        fc1_dims=256,
        fc2_dims=256,
        name="critic",
        chkpt_dir="tmp/sac"
    ):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_sac")

        # Layers
        self.fc1 = nn.Linear(self.input_dims[0] + self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        # Device handling
        if T.backends.mps.is_available():
            self.device = T.device("mps")
        elif T.cuda.is_available():
            self.device = T.device("cuda")
        else:
            self.device = T.device("cpu")
        self.to(self.device)

    def forward(self, state, action):
        """
        Forward pass: Q(s, a)
        """
        x = T.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q(x)
        return q_value

    def save_checkpoint(self):
        print("...saving checkpoint for Critic...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("...loading checkpoint for Critic...")
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    """
    The ValueNetwork computes V(s).
    It has two fully-connected layers, then outputs a single state-value.
    """
    def __init__(
        self,
        beta,              # learning rate
        input_dims,
        fc1_dims=256,
        fc2_dims=256,
        name="value",
        chkpt_dir="tmp/sac"
    ):
        super(ValueNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_sac")

        # Layers
        self.fc1 = nn.Linear(self.input_dims[0], self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        # Device handling
        if T.backends.mps.is_available():
            self.device = T.device("mps")
        elif T.cuda.is_available():
            self.device = T.device("cuda")
        else:
            self.device = T.device("cpu")
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass: V(s)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.v(x)
        return v

    def save_checkpoint(self):
        print("...saving checkpoint for Value...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("...loading checkpoint for Value...")
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    """
    The ActorNetwork outputs a policy in the form of a Gaussian (mu, sigma),
    and then samples actions via a reparameterized trick.
    """
    def __init__(
        self,
        alpha,                   # learning rate
        input_dims,
        max_action=1.0,          # scale for the final tanh-squashed action
        fc1_dims=256,
        fc2_dims=256,
        n_actions=2,
        name="actor",
        chkpt_dir="tmp/sac"
    ):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_sac")
        self.max_action = max_action

        # Reparameterization trick
        self.reparam_noise = 1e-6

        # Layers
        self.fc1 = nn.Linear(self.input_dims[0], self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # Device handling
        if T.backends.mps.is_available():
            self.device = T.device("mps")
        elif T.cuda.is_available():
            self.device = T.device("cuda")
        else:
            self.device = T.device("cpu")
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass: produce mu, sigma for each action dimension.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = self.mu(x)
        sigma = self.sigma(x)
        # clamp sigma so it doesn't blow up
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1.0)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        """
        Sample an action from the Gaussian (mu, sigma) distribution,
        then tanh-squash it to be in [-max_action, +max_action].
        Returns both the action and the log-likelihood of that action.
        """
        mu, sigma = self.forward(state)
        dist = Normal(mu, sigma)

        if reparameterize:
            actions = dist.rsample()
        else:
            actions = dist.sample()

        # Tanh-squash
        out = T.tanh(actions) * T.tensor(self.max_action).to(self.device)

        # Compute log_probs (accounting for tanh's Jacobian)
        log_probs = dist.log_prob(actions)
        log_probs -= T.log(1 - out.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(dim=1, keepdim=True)

        return out, log_probs

    def save_checkpoint(self):
        print("...saving checkpoint for Actor...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("...loading checkpoint for Actor...")
        self.load_state_dict(T.load(self.checkpoint_file))