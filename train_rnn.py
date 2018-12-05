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
from ppo_rnn import PPO
from vec_env.subproc_vec_env import SubprocVecEnv
import cv2

env_name = "CarRacing-v0"

def preprocess_frame(frame):
    frame = frame[:-12, 6:-6] # Crop to 84x84
    frame = np.dot(frame[..., 0:3], [0.299, 0.587, 0.114])
    frame = frame / 255.0
    frame = frame * 2 - 1
    return frame

def make_env():
    return gym.make(env_name)

def evaluate(model, test_env, make_video=False):
    total_reward = 0
    test_env.seed(0)
    state = np.expand_dims(preprocess_frame(test_env.reset()), axis=-1)
    rendered_frame = test_env.render(mode="rgb_array")
    features = np.zeros((1, *model.get_feature_shape()))
    done = False
    if make_video:
        video_dir = "./videos/{}/run{}".format(model.model_name, model.run_idx)
        if not os.path.isdir(video_dir): os.makedirs(video_dir)
        video_writer = cv2.VideoWriter("./videos/{}/run{}/step{}.avi".format(model.model_name, model.run_idx, model.step_idx),
                                       cv2.VideoWriter_fourcc(*"MPEG"), 30,
                                       (rendered_frame.shape[1], rendered_frame.shape[0]))
    while not done:
        # Predict action given state: π(a_t | s_t; θ)
        action, _, features = model.predict(np.expand_dims(state, axis=0),
                                            features,
                                            greedy=False)
        state, reward, done, _ = test_env.step(action[0])
        state = np.expand_dims(preprocess_frame(state), axis=-1)
        rendered_frame = test_env.render(mode="rgb_array")
        total_reward += reward
        if make_video: video_writer.write(cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
        # time.sleep(0.016)
    if make_video: video_writer.release()
    return total_reward

def train():
    # Create test env
    print("Creating test environment")
    test_env = gym.make(env_name)

    # Traning parameters
    lr_scheduler    = Scheduler(initial_value=3e-4, interval=1000, decay_factor=1)#0.75)
    discount_factor = 0.99
    gae_lambda      = 0.95
    ppo_epsilon     = 0.2
    t_max           = 180
    num_epochs      = 10
    batch_size      = 64
    save_interval   = 100
    eval_interval   = 20
    training        = True

    # Environment constants
    input_shape      = (84, 84, 1)
    num_actions      = test_env.action_space.shape[0]
    action_min       = np.array([-1.0, 0.0, 0.0])
    action_max       = np.array([ 1.0, 1.0, 1.0])

    # Create model
    print("Creating model")
    model_checkpoint = None
    model = PPO(num_actions, input_shape, action_min, action_max, ppo_epsilon,
                value_scale=0.5, entropy_scale=0.01,
                model_checkpoint=model_checkpoint, model_name="CarRacing-v0-rnn")

    if training:
        print("Creating environments")
        num_envs = 4
        envs = SubprocVecEnv([make_env for _ in range(num_envs)])

        states_t = [preprocess_frame(f) for f in envs.reset()]
        features_t = [np.zeros(model.get_feature_shape()) for _ in range(num_envs)]
        envs.get_images()
        
        print("Training loop")
        while True:
            global_step = model.step_idx // num_epochs
            if global_step % eval_interval == 0:
                print("Running evaluation...")
                avg_reward = evaluate(model, test_env, make_video=True)
                model.write_to_summary("eval_avg_reward", avg_reward)

            # While there are running environments
            print("Training (global step {})...".format(global_step))
            states, prev_features, taken_actions, values, rewards, dones = [], [], [], [], [], []
            learning_rate = np.maximum(lr_scheduler.get_value(), 1e-6)
            
            # Simulate game for some number of steps
            for _ in range(t_max):
                # Predict and value action given state
                # π(a_t | s_t; θ_old)
                prev_features.append(features_t) # [T, N, 2592]
                actions_t, values_t, features_t = model.predict(np.expand_dims(states_t, axis=-1),
                                                                features_t,
                                                                use_old_policy=True)

                # Sample action from a Gaussian distribution
                envs.step_async(actions_t)
                frames_t, rewards_t, dones_t, _ = envs.step_wait()
                states_t = [preprocess_frame(f) for f in frames_t]
                envs.get_images() # render
                
                # Get new state
                for i in range(num_envs):
                    # Reset environment's recurrent feature if done
                    if dones_t[i]:
                        features_t[i] = np.zeros(model.get_feature_shape())
                
                # Store state, action and reward
                states.append(states_t)                      # [T, N, 84, 84, 1]
                taken_actions.append(actions_t)              # [T, N, 3]
                values.append(np.squeeze(values_t, axis=-1)) # [T, N]
                rewards.append(rewards_t)                    # [T, N]
                dones.append(dones_t)                        # [T, N]

            # Calculate last values (bootstrap values)
            last_values = np.squeeze(model.predict(np.expand_dims(states_t, axis=-1),
                                     features_t,
                                     use_old_policy=True)[1], axis=-1) # [N]

            # Compute returns
            returns = compute_returns(rewards, last_values, dones, discount_factor)
            
            # Compute advantages
            advantages = compute_gae(rewards, values, last_values, dones, discount_factor, gae_lambda)

            # Normalize advantages
            advantages = (advantages - np.mean(advantages)) / np.std(advantages)

            # Flatten arrays
            states        = np.array(states).reshape((-1, *input_shape))       # [T x N, 84, 84, 1]
            prev_features = np.array(prev_features).reshape((-1, *model.get_feature_shape()))
            taken_actions = np.array(taken_actions).reshape((-1, num_actions)) # [T x N, 3]
            returns       = returns.flatten()                                  # [T x N]
            advantages    = advantages.flatten()                               # [T X N]

            # Train for some number of epochs
            model.update_old_policy() # θ_old <- θ
            for _ in range(num_epochs):
                # Sample mini-batch randomly and train
                mb_idx = np.random.choice(len(states), batch_size, replace=False)

                # Optimize network
                model.train(states[mb_idx], prev_features[mb_idx], taken_actions[mb_idx],
                            returns[mb_idx], advantages[mb_idx],
                            learning_rate=learning_rate)

            # Save model
            if (global_step + 1) % save_interval == 0:
                model.save()
    
    # Training complete, evaluate model
    avg_reward = evaluate(model, test_env, 10)
    print("Model achieved a final reward of:", avg_reward)

if __name__ == "__main__":
    train()