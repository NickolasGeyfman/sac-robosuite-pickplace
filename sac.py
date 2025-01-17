import os
import torch as T
import torch.nn.functional as F
import numpy as np

from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    def __init__(
        self,
        alpha=0.0003,
        beta=0.0003,
        input_dims=[88],
        gamma=0.99,
        n_actions=2,
        max_size=1000000,
        tau=0.005,
        layer1_size=256,
        layer2_size=256,
        batch_size=256,
        reward_scale=2,
        # Pass in max_action explicitly, rather than relying on env
        max_action=None
    ):
        """
        Initializes the SAC agent.

        Parameters:
        -----------
        alpha : float
            Learning rate for the actor.
        beta : float
            Learning rate for the critics and value networks.
        input_dims : list or tuple
            Observation space dimensions (e.g., [obs_dim]).
        gamma : float
            Discount factor.
        n_actions : int
            Number of continuous action dimensions.
        max_size : int
            Maximum size of the replay buffer.
        tau : float
            Soft update parameter for target value network.
        layer1_size : int
            Number of neurons in the first hidden layer.
        layer2_size : int
            Number of neurons in the second hidden layer.
        batch_size : int
            Batch size for training updates.
        reward_scale : float
            A factor to scale rewards (commonly used in SAC).
        max_action : float or np.ndarray
            The maximum action bound(s). Should match env.action_space.high
            if you're using a continuous Gym or Robosuite environment.
        """
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.scale = reward_scale

        # Replay buffer for off-policy experience
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        # Create networks
        self.actor = ActorNetwork(
            alpha=alpha,
            input_dims=input_dims,
            n_actions=n_actions,
            max_action=max_action,  # <--- Replaced env.action_space.high
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name='actor',
            chkpt_dir='tmp/sac'
        )
        self.critic_1 = CriticNetwork(
            beta, input_dims, n_actions,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name='critic_1'
        )
        self.critic_2 = CriticNetwork(
            beta, input_dims, n_actions,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name='critic_2'
        )
        self.value = ValueNetwork(
            beta, input_dims,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name='value'
        )
        self.target_value = ValueNetwork(
            beta, input_dims,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name='target_value'
        )

        # Initialize target_value network to match value network weights
        self.update_network_parameters(tau=1.0)

    def choose_action(self, observation):
        """
        Given a single observation (shape [obs_dim]),
        returns a single action (shape [n_actions]) as a np.array.
        """
        obs_array = np.array(observation, dtype=np.float32)
        state_t = T.from_numpy(obs_array).unsqueeze(0).to(self.actor.device)

        actions, _ = self.actor.sample_normal(state_t, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        """
        Store a single transition in the replay buffer.
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        """
        Perform a soft update on the target value network.
        target_value = tau * value + (1 - tau) * target_value
        """
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = (
                tau * value_state_dict[name].clone() +
                (1 - tau) * target_value_state_dict[name].clone()
            )

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('...saving models...')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        try:
            self.actor.load_checkpoint()
            self.critic_1.load_checkpoint()
            self.critic_2.load_checkpoint()
            self.value.load_checkpoint()
            self.target_value.load_checkpoint()
            print("Loaded existing checkpoints.")
        except FileNotFoundError:
            print("No checkpoints found. Starting from scratch.")

    def learn(self):
        """
        Sample a batch from replay buffer, compute losses,
        and update the actor, critics, and value networks.
        """
        if self.memory.mem_cntr < self.batch_size:
            return

        # 1) Sample from the replay buffer
        states, actions, rewards, next_states, dones = \
            self.memory.sample_buffer(self.batch_size)

        # 2) Convert to PyTorch tensors
        states_t = T.tensor(states, dtype=T.float).to(self.actor.device)
        next_states_t = T.tensor(next_states, dtype=T.float).to(self.actor.device)
        actions_t = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards_t = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        dones_t = T.tensor(dones).to(self.actor.device)

        # 3) Current value V(s) and target value V(s')
        value = self.value(states_t).view(-1)
        value_ = self.target_value(next_states_t).view(-1)
        value_[dones_t] = 0.0

        # 4) Sample actions from the current policy
        new_actions, log_probs = self.actor.sample_normal(
            states_t, reparameterize=False
        )
        log_probs = log_probs.view(-1)

        q1_new_policy = self.critic_1(states_t, new_actions).view(-1)
        q2_new_policy = self.critic_2(states_t, new_actions).view(-1)
        critic_value = T.min(q1_new_policy, q2_new_policy)

        # 5) Update Value network
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # 6) Update Actor (policy)
        self.actor.optimizer.zero_grad()
        new_actions, log_probs = self.actor.sample_normal(
            states_t, reparameterize=True
        )
        log_probs = log_probs.view(-1)
        q1_new = self.critic_1(states_t, new_actions).view(-1)
        q2_new = self.critic_2(states_t, new_actions).view(-1)
        critic_value = T.min(q1_new, q2_new)
        actor_loss = (log_probs - critic_value).mean()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # 7) Update Critics
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q_hat = self.scale * rewards_t + self.gamma * value_
        q1_old_policy = self.critic_1(states_t, actions_t).view(-1)
        q2_old_policy = self.critic_2(states_t, actions_t).view(-1)

        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # 8) Soft update target value network
        self.update_network_parameters()