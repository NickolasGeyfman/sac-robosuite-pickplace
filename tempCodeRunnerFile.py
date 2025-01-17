import time
import os 
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from networks import CriticNetwork, ActorNetwork, ValueNetwork
from buffer import ReplayBuffer
from sac import Agent

import torch as T

if __name__ == "__main__":

    if not os.path.exists("tmp/sac"):
        os.makedirs("tmp/sac")

    env_name = "PickPlace"

    env = suite.make(
        env_name, 
        robots=["Panda"],
        controller_configs=suite.load_controller_config(
            default_controller="JOINT_VELOCITY"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        horizon=100,
    )
    
    env = GymWrapper(env)
    


    alpha = 0.0003           # Actor (policy) learning rate
    beta = 0.0003            # Critic/Value networks learning rate
    gamma = 0.99             # Discount factor
    tau = 0.005              # Soft update parameter
    batch_size = 256
    layer1_size = 256
    layer2_size = 256
    reward_scale = 2.0       # Common in SAC to scale the rewards
    n_games = 100000

    agent = Agent(
        alpha=alpha,
        beta=beta,
        input_dims=env.observation_space.shape,
        env=env,
        gamma=gamma,
        n_actions=env.action_space.shape[0],
        max_size=1000000,
        tau=tau,
        layer1_size=layer1_size,
        layer2_size=layer2_size,
        batch_size=batch_size,
        reward_scale=reward_scale
    )

    if T.cuda.is_available():
        device = T.device("cuda")
    elif T.backends.mps.is_available():
        device = T.device("mps")
    else:
        device = T.device("cpu")

    print("Training on device:", device)

    agent.actor.device = device
    agent.critic_1.device = device
    agent.critic_2.device = device
    agent.value.device = device
    agent.target_value.device = device

    # Initialize TensorBoard
    writer = SummaryWriter("logs")

    best_score = -np.inf
    episode_identifier = (
        f"alpha={alpha} beta={beta} batch_size={batch_size} "
        f"layer1={layer1_size} layer2={layer2_size} tau={tau} "
        f"env={env_name} reward_scale={reward_scale} "
        f"{int(time.time())}"
    )

    agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            # 1) Agent chooses action
            action = agent.choose_action(observation)
            # 2) Step environment
            next_observation, reward, done, info = env.step(action)
            score += reward
            # 3) Store transition and learn
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()
            # 4) Move to next state
            observation = next_observation

        # Log the score in TensorBoard
        writer.add_scalar(f"score/{episode_identifier}", score, global_step=i)

        # Save models every 10 episodes (adjust as desired)
        if i % 10 == 0:
            agent.save_models()

        print(f"Episode {i}, Score {score}")