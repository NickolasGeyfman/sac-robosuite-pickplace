import time
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from sac import Agent
import torch as T

if __name__ == "__main__":

    # Make sure the same directory structure exists
    if not os.path.exists("tmp/sac"):
        os.makedirs("tmp/sac")

    env_name = "PickPlace"

    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(
            default_controller="JOINT_VELOCITY"),
        has_renderer=True,              
        render_camera="frontview",       
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        horizon=300,                    
    )
    env = GymWrapper(env)

    alpha = 0.0003
    beta = 0.0003
    gamma = 0.99
    tau = 0.005
    batch_size = 256
    layer1_size = 256
    layer2_size = 256
    reward_scale = 2.0

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
    agent.actor.device = device
    agent.critic_1.device = device
    agent.critic_2.device = device
    agent.value.device = device
    agent.target_value.device = device

    print("Testing on device:", device)

    # Load the models from disk
    agent.load_models()

    writer = SummaryWriter("logs_test")

    n_test_episodes = 10

    for i in range(n_test_episodes):
        observation = env.reset()
        done = False
        score = 0

        while not done:

            action = agent.choose_action(observation)
            observation, reward, done, info = env.step(action)
            score += reward


            env.render()

            time.sleep(0.02)


        writer.add_scalar("test_score", score, global_step=i)
        print(f"[Test] Episode {i}/{n_test_episodes}, Score = {score}")

    writer.close()