import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

#alpha and beta are learning rates for actor and critic networks
#gamma is the discount factor (controls how much the agent values future rewards of immediate rewards)
#tau controls the soft update of target value network

class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[88], env=None, gamma=0.99, n_actions=2, max_size=1000000,
                 tau=0.005, layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        
        #Create networks
        self.actor = ActorNetwork(alpha, input_dims, n_actions, name='actor', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions, name='critic_2')
        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    #Forward our current observation through the policy (actor network) to get an action. Then we return a numpy array
    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]
    
    #Saves transition into the replay buffer for later sampling
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    #Soft update of target value network
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()
        
        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                (1-tau)*target_value_state_dict[name].clone()
            
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
        except FileNotFoundError as e:
            print("No checkpoints found. Starting from scratch.")


    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # 1) Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = \
            self.memory.sample_buffer(self.batch_size)

        # 2) Convert to tensors on correct device
        states_t = T.tensor(states, dtype=T.float).to(self.actor.device)
        next_states_t = T.tensor(next_states, dtype=T.float).to(self.actor.device)
        actions_t = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards_t = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        dones_t = T.tensor(dones).to(self.actor.device)

        # 3) Compute current value V(s) and target value V(s') (with the target network)
        value = self.value(states_t).view(-1)
        value_ = self.target_value(next_states_t).view(-1)
        value_[dones_t] = 0.0

        # 4) Sample action from current policy for the value loss
        new_actions, log_probs = self.actor.sample_normal(states_t, reparameterize=False)
        log_probs = log_probs.view(-1)

        # Evaluate Q(s, new_actions)
        q1_new_policy = self.critic_1(states_t, new_actions).view(-1)
        q2_new_policy = self.critic_2(states_t, new_actions).view(-1)
        critic_value = T.min(q1_new_policy, q2_new_policy)

        # 5) Value network loss
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # 6) Actor (policy) loss
        self.actor.optimizer.zero_grad()
        new_actions, log_probs = self.actor.sample_normal(states_t, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1(states_t, new_actions).view(-1)
        q2_new_policy = self.critic_2(states_t, new_actions).view(-1)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        actor_loss = (log_probs - critic_value).mean()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # 7) Critic networks loss
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * rewards_t + self.gamma * value_
        q1_old_policy = self.critic_1(states_t, actions_t).view(-1)
        q2_old_policy = self.critic_2(states_t, actions_t).view(-1)

        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        # Make sure you use q2_old_policy here:
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # 8) Soft update target network
        self.update_network_parameters()



