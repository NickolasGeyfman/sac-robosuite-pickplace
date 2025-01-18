import multiprocessing as mp
import numpy as np
import torch as T
import time
import os

import robosuite as suite
from robosuite.wrappers import GymWrapper

from sac import Agent  # your SAC agent class (unchanged)
from torch.utils.tensorboard import SummaryWriter  # For logging


###################################
# Worker function for one environment
###################################
def env_worker(child_conn, env_config):
    """
    Each worker:
      - Creates a robosuite environment
      - Waits for an action from the main process (msg)
      - Steps the environment with that action
      - Sends (next_obs, reward, done) back
      - If it receives "CLOSE", it exits
    """
    try:
        print("[Worker] Starting environment creation...", flush=True)
        env = suite.make(
            "PickPlace",
            robots=env_config["robots"],
            controller_configs=env_config["controller_configs"],
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=env_config["control_freq"],
            reward_shaping=True,
            horizon=env_config["horizon"],
            single_object_mode=env_config["single_object_mode"]
        )
        env = GymWrapper(env)
        print("[Worker] Environment created successfully.", flush=True)

        # Initial reset
        obs = env.reset()
        child_conn.send(obs)
        print("[Worker] Sent initial observation to main.", flush=True)

        while True:
            msg = child_conn.recv()  # blocking

            # If message is the string "CLOSE", we shut down
            if isinstance(msg, str) and msg == "CLOSE":
                print("[Worker] Received CLOSE, exiting worker.", flush=True)
                child_conn.close()
                break
            else:
                # Otherwise, interpret msg as a NumPy array for the action
                action = msg  
                next_obs, reward, done, info = env.step(action)

                # IMPORTANT BUG FIX:
                # Send the transition (including the *true* next_obs if done)
                # before resetting the environment. This way the main process
                # stores the correct final transition in the replay buffer.
                child_conn.send((next_obs, reward, done))

                if done:
                    # Only now do we reset
                    next_obs = env.reset()
                    print("[Worker] Episode ended, resetting environment.", flush=True)

                # Prepare for next iteration
                obs = next_obs

    except Exception as e:
        print(f"[Worker] Caught exception: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        child_conn.close()
        print("[Worker] Connection closed, worker exiting.", flush=True)


###################################
# Main parallel training loop
###################################
def main_parallel():
    # Ensure the 'tmp/sac' directory exists for saving models
    if not os.path.exists("tmp/sac"):
        os.makedirs("tmp/sac")

    # -------------------------
    # 1) TensorBoard Setup
    # -------------------------
    writer = SummaryWriter(log_dir="logs")

    # 2) Create a dummy environment to infer obs/action shapes
    test_env = suite.make(
        "PickPlace",
        robots=["Panda"],
        # You can switch to "OSC_POSITION" if you prefer a higher-level controller:
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        reward_shaping=True,
        horizon=200,
        single_object_mode=1
    )
    test_env = GymWrapper(test_env)

    obs_dim = test_env.observation_space.shape  # e.g., (39,)
    act_dim = test_env.action_space.shape[0]    # e.g., 7 or 8 (depending on the gripper)
    max_action = test_env.action_space.high     # array of size act_dim

    print(f"Observation Dim: {obs_dim}, Action Dim: {act_dim}, Max Action: {max_action}")

    # 3) Create your Agent
    agent = Agent(
        input_dims=obs_dim,
        n_actions=act_dim,
        max_action=max_action,
        alpha=0.0003,
        beta=0.0003,
        gamma=0.99,
        tau=0.005,
        layer1_size=256,
        layer2_size=256,
        batch_size=256,
        reward_scale=2.0,
        max_size=1_000_000
    )

    # Optionally load existing models
    agent.load_models()

    # 4) Environment config for workers
    env_config = {
        "robots": ["Panda"],
        "controller_configs": suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        "control_freq": 20,
        "horizon": 200,
        "single_object_mode": 1
    }

    # 5) Spawn parallel environments
    n_envs = 8
    parent_conns = []
    processes = []
    for i in range(n_envs):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=env_worker, args=(child_conn, env_config))
        processes.append(p)
        parent_conns.append(parent_conn)
        p.start()

    # 6) Collect initial observations
    obs_list = []
    for conn in parent_conns:
        first_obs = conn.recv()  # initial obs from each worker
        obs_list.append(first_obs)
    obs_array = np.array(obs_list, dtype=np.float32)  # shape: [n_envs, obs_dim]

    # -------------------------
    # Track Episode Rewards
    # -------------------------
    episode_rewards = [0.0] * n_envs
    episode_counts = [0] * n_envs

    # 7) Training loop
    total_steps = 10000000000  # example target, you can set higher if needed
    step_count = 0
    total_start_time = time.time()

    # Warm-up config:
    warm_up_steps = 5000  # number of steps to sample random actions (aggregated across all envs)

    while step_count < total_steps:
        iteration_start = time.time()

        # a) For each env, choose action
        actions = []
        for i in range(n_envs):
            if step_count < warm_up_steps:
                # During warm-up, use random actions
                action = test_env.action_space.sample()
            else:
                # Use the agent's policy
                action = agent.choose_action(obs_array[i])
            actions.append(action)
        actions = np.array(actions, dtype=np.float32)

        # b) Send each action
        for i, conn in enumerate(parent_conns):
            conn.send(actions[i])

        # c) Receive next_obs, reward, done from each worker
        next_obs_list = []
        rewards_list = []
        dones_list = []
        for i, conn in enumerate(parent_conns):
            next_obs, reward, done = conn.recv()
            next_obs_list.append(next_obs)
            rewards_list.append(reward)
            dones_list.append(done)

        next_obs_array = np.array(next_obs_list, dtype=np.float32)
        rewards_array = np.array(rewards_list, dtype=np.float32)
        dones_array = np.array(dones_list, dtype=bool)

        # d) Store transitions, track episode rewards
        for i in range(n_envs):
            agent.remember(
                obs_array[i],
                actions[i],
                rewards_array[i],
                next_obs_array[i],
                dones_array[i]
            )

            episode_rewards[i] += rewards_array[i]
            if dones_array[i]:
                ep_count = episode_counts[i]
                final_ep_reward = episode_rewards[i]
                # Log to TensorBoard
                writer.add_scalar("score", final_ep_reward, global_step=ep_count)
                print(f"[Main] Env {i} finished episode {ep_count} with reward={final_ep_reward:.2f}")
                # Reset
                episode_rewards[i] = 0.0
                episode_counts[i] += 1

        # e) Optionally learn
        # Only learn if we are past the warm-up phase
        if step_count >= warm_up_steps:
            agent.learn()

        # f) Update obs_array
        obs_array = next_obs_array
        step_count += n_envs  # we count 1 step per env => n_envs steps total

        # g) Optionally save models every 50k steps
        if step_count % 50000 == 0 and step_count > 0:
            agent.save_models()
            print(f"[Main] Models saved at step_count={step_count}")

        # Time metrics
        iter_duration = time.time() - iteration_start
        writer.add_scalar("time/iteration_duration", iter_duration, global_step=step_count)

        # Print info every 1000 steps
        if step_count % 1000 == 0:
            avg_batch_reward = np.mean(rewards_array)
            print(f"[Main] Step={step_count}, mean reward for this batch={avg_batch_reward:.2f}")
            print(f"[Main] Iteration time={iter_duration:.3f}s")

    # After the loop, final save
    agent.save_models()
    print("[Main] Final models saved. Exiting loop.")

    # 9) Close workers
    for conn in parent_conns:
        conn.send("CLOSE")
    for p in processes:
        p.join()

    # Close TensorBoard writer
    writer.close()

    total_elapsed = time.time() - total_start_time
    print(f"Finished parallel training. Total time = {total_elapsed:.2f}s")
    print("TensorBoard logs saved in 'logs'.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main_parallel()