import matplotlib.pyplot as plt
import gym
import os
import time
import random
import time
import numpy as np
from skimage import transform
from IPython.display import display, clear_output
import tensorflow as tf
from utils import FrameStack, Scheduler, compute_returns, compute_gae
from ppo import PPO
from vec_env.subproc_vec_env import SubprocVecEnv

env_name = "CarRacing-v0"

def preprocess_frame(frame):
    frame = frame[:-12, 6:-6] # Crop to 84x84
    #frame = transform.resize(frame, [84, 84])
    frame = np.dot(frame[..., 0:3], [0.299, 0.587, 0.114])
    frame = frame / 255.0
    frame = frame * 2 - 1
    return frame

def make_env():
    return gym.make(env_name)

def evaluate(model, test_env, num_runs=1):
    total_reward = 0
    for _ in range(num_runs):
        initial_frame = test_env.reset()
        frame_stack = FrameStack(initial_frame, preprocess_fn=preprocess_frame)
        done = False
        while not done:
            # Predict action given state: π(a_t | s_t; θ)
            state = frame_stack.get_state()
            action, _ = model.predict(np.expand_dims(state, axis=0), greedy=False)
            frame, reward, done, info = test_env.step(action[0])
            test_env.render()
            total_reward += reward
            frame_stack.add_frame(frame)
            time.sleep(0.016)
    return total_reward / num_runs

def main():
    # Create test env
    print("Creating test environment")
    test_env = gym.make(env_name)

    # Traning parameters
    lr_scheduler     = Scheduler(initial_value=3e-4, interval=1000, decay_factor=1)#0.75)
    std_scheduler    = Scheduler(initial_value=2.0, interval=1000, decay_factor=0.75)
    discount_factor  = 0.99
    gae_lambda       = 0.95
    ppo_epsilon      = 0.2
    t_max            = 180
    num_epochs       = 10
    batch_size       = 64
    save_interval    = 100
    eval_interval    = 20
    training         = True

    # Environment constants
    frame_stack_size = 4
    input_shape      = (84, 84, frame_stack_size)
    num_actions      = test_env.action_space.shape[0]
    action_min       = np.array([-1.0, 0.0, 0.0])
    action_max       = np.array([ 1.0, 1.0, 1.0])

    # Create model
    print("Creating model")
    model_checkpoint = None#"./models/CarRacing-v0/run2/episode0_step455000.ckpt"
    model = PPO(num_actions, input_shape, action_min, action_max, ppo_epsilon,
                value_scale=0.5, entropy_scale=0.0001,
                model_checkpoint=model_checkpoint, model_name="CarRacing-v0")

    if training:
        print("Creating environments")
        num_envs = 4
        envs = SubprocVecEnv([make_env for _ in range(num_envs)])

        initial_frames = envs.reset()
        envs.get_images()
        frame_stacks = [FrameStack(initial_frames[i], preprocess_fn=preprocess_frame) for i in range(num_envs)]

        print("Main loop")
        step = 0
        while training:
            if step % eval_interval == 0:
                avg_reward = evaluate(model, test_env, 1)
                model.write_to_summary("eval_avg_reward", avg_reward)

            # While there are running environments
            print("Training...")
            states, taken_actions, values, rewards, dones = [], [], [], [], []
            learning_rate = np.maximum(lr_scheduler.get_value(), 1e-6)
            std = np.maximum(std_scheduler.get_value(), 0.2)
            
            # Simulate game for some number of steps
            for _ in range(t_max):
                # Predict and value action given state
                # π(a_t | s_t; θ_old)
                states_t = [frame_stacks[i].get_state() for i in range(num_envs)]
                actions_t, values_t = model.predict(states_t, use_old_policy=True, std=std)

                # Sample action from a Gaussian distribution
                envs.step_async(actions_t)
                frames, rewards_t, dones_t, infos = envs.step_wait()
                envs.get_images() # render
                
                # Store state, action and reward
                states.append(states_t)                      # [T, N, 84, 84, 1]
                taken_actions.append(actions_t)              # [T, N, 3]
                values.append(np.squeeze(values_t, axis=-1)) # [T, N]
                rewards.append(rewards_t)                    # [T, N]
                dones.append(dones_t)                        # [T, N]
                
                # Get new state
                for i in range(num_envs):
                    frame_stacks[i].add_frame(frames[i])

            # Calculate last values (bootstrap values)
            states_last = [frame_stacks[i].get_state() for i in range(num_envs)]
            last_values = np.squeeze(model.predict(states_last)[-1], axis=-1) # [N]

            # Compute returns
            returns = compute_returns(rewards, last_values, dones, discount_factor)
            
            # Compute advantages
            advantages = compute_gae(rewards, values, last_values, dones, discount_factor, gae_lambda)

            # Normalize advantages
            advantages = (advantages - np.mean(advantages)) / np.std(advantages)

            # Flatten arrays
            states        = np.array(states).reshape((-1, *input_shape))       # [T x N, 84, 84, 1]
            taken_actions = np.array(taken_actions).reshape((-1, num_actions)) # [T x N, 3]
            returns       = returns.flatten()                                  # [T x N]
            advantages    = advantages.flatten()                               # [T X N]

            # Train for some number of epochs
            model.update_old_policy() # θ_old <- θ
            for _ in range(num_epochs):
                # Sample mini-batch randomly and train
                mb_idx = np.random.choice(len(states), batch_size, replace=False)

                # Optimize network
                model.train(states[mb_idx], taken_actions[mb_idx],
                            returns[mb_idx], advantages[mb_idx],
                            learning_rate=learning_rate, std=std)

            # Reset environment's frame stack if done
            for i, done in enumerate(dones_t):
                if done:
                    for _ in range(frame_stack_size):
                        frame_stacks[i].add_frame(frames[i])

            # Save model
            step += 1
            if step % save_interval == 0:
                model.save()
    
    # Training complete, evaluate model
    avg_reward = evaluate(model, test_env, 10)
    print("Model achieved a final reward of:", avg_reward)

if __name__ == "__main__":
    main()