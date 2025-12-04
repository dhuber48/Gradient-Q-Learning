from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from mountainCarAgent import Agent


#based on farama gym tutorial code:
learning_rate = 0.25        # How fast to learn (higher = faster but less stable)
n_episodes = 1000       # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / n_episodes  # Reduce exploration over time
final_epsilon = 0.05        # Always keep some exploration
# Create our training environment - MountainCarContinuous environment
env = gym.make("MountainCarContinuous-v0", render_mode=None, goal_velocity=0.1)
#env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# Initialize agent
agent = Agent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# Training loop
for episode in tqdm(range(n_episodes), desc="Training"):
    observation, info = env.reset()
    episode_over = False
    
    while not episode_over:
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        #agent.update(observation, action, reward, terminated, next_observation)
        # Aggressive reward shaping: position is the main learning signal
        shaped_reward = reward + next_observation[0] * 10 + next_observation[1] * 0.1
        agent.update(observation, action, shaped_reward, terminated, next_observation)
        observation = next_observation
        episode_over = terminated or truncated
    
    agent.decay_epsilon()

env.close()

# Show final trained agent
import time
env = gym.make("MountainCarContinuous-v0", render_mode="human", goal_velocity=0.1)

for _ in range(5):
    observation, info = env.reset()
    episode_over = False
    
    while not episode_over:
        action = agent.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
    
    time.sleep(1)

env.close()