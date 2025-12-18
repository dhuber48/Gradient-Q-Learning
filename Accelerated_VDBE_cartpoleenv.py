#!/usr/bin/env python3

from collections import defaultdict
import gymnasium as gym
import random, numpy as np
from tqdm import tqdm
from Accelerated_VDBE_cartPoleAgent import Agent
import time
import matplotlib.pyplot as plt

#THIS IS FOR ACCELERATED VDBE Q-LEARNING (MY ALGORITHM)

#based on farama gym tutorial code:
learning_rate = 0.2        # How fast to learn (higher = faster but less stable)
n_episodes = 100000       # Max number of episodes
start_epsilon = 1.0         # Start with 100% random actions
sigma = 20    #parameter for VDBE, approx equal to max discounted reward 
reward_list = []

# Stop training criteria
SOLVED_AVG_REWARD = 200 #SET THIS TO WHATEVER YOU WANT FOR TESTING
SOLVED_WINDOW = 100

# Create training environment - a cart with a pole that needs balancing
env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# Initialize agent
agent = Agent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    sigma=sigma
)


start_train_time = time.time()
trained_episodes = 0

#training loop
for episode in tqdm(range(n_episodes)):
    observation, info = env.reset()
    episode_over = False
    episode_total_reward = 0

    while not episode_over:
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        episode_total_reward += reward

        agent.update(observation, action, reward, terminated, next_observation)
        observation = next_observation
        episode_over = terminated or truncated

    reward_list.append(episode_total_reward)
    trained_episodes += 1

    #Print epsilon every 250 episodes
    if (episode + 1) % 250 == 0: 
        recent_avg = np.mean(reward_list[-100:]) if len(reward_list) >= 100 else np.mean(reward_list) # Calculate the average of the last 100 just for the display
        tqdm.write(f"Episode {episode+1} | Epsilon: {agent.epsilon:.4f} | Avg Reward (last 100): {recent_avg:.1f}")

    #CHECK STOPPING CONDITION (Avg >= solved_avg_reward over last solved_window episodes)
    if len(reward_list) >= SOLVED_WINDOW:
        avg_reward = np.mean(reward_list[-SOLVED_WINDOW:]) #Calculate average of the last solved_window rewards
        
        if avg_reward >= SOLVED_AVG_REWARD: #end condition
            print(f"\n\nEnvironment solved in {episode + 1} episodes!")
            print(f"Average Reward over last {SOLVED_WINDOW} episodes: {avg_reward}")
            break

env.close()


# --- REPORTING & PLOTTING ---
total_train_time = time.time() - start_train_time
print(f"Total training time: {total_train_time:.2f} seconds")

window_size = 50 #Moving average size
if len(reward_list) >= window_size:
    moving_avg = np.convolve(reward_list, np.ones(window_size)/window_size, mode='valid')
else:
    moving_avg = reward_list # Fallback if run is very short

# GRAPH CODE
plt.figure(figsize=(10, 5))
plt.plot(reward_list, label='Episode Reward', alpha=0.5)

# Adjust x-axis for moving average to match data points
plt.plot(range(window_size-1, len(reward_list)) if len(reward_list) >= window_size else range(len(reward_list)), 
         moving_avg, color='red', label=f'{window_size}-Episode Moving Avg')

plt.axhline(y=475, color='g', linestyle='--', label='Solved Threshold (475)')
plt.title(f'Training Progress (Stopped at {trained_episodes} Episodes)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.show()

# Show final trained agent (may delete later)
eval_env = gym.make("CartPole-v1", render_mode="human")
for i in range(5):
    observation, info = eval_env.reset()
    episode_over = False
    start_time = time.time()
    current_reward = 0
    agent.epsilon = 0

    while not episode_over:
        action = agent.get_action(observation)
        observation, reward, terminated, truncated, info = eval_env.step(action)
        current_reward += reward
        episode_over = terminated or truncated
    
    elapsed_ms = (time.time() - start_time) * 1000
    print(f"Eval Episode {i+1}: Reward: {current_reward} | Duration: {elapsed_ms:.0f}ms")
    time.sleep(1)

eval_env.close()