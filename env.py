from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from agent import Agent

#based on farama gym tutorial code:
learning_rate = 0.2        # How fast to learn (higher = faster but less stable)
n_episodes = 10000        # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / n_episodes  # Reduce exploration over time
final_epsilon = 0.0        # Always keep some exploration

# Create our training environment - a cart with a pole that needs balancing
env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# Initialize agent
agent = Agent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# Training loop
for episode in tqdm(range(n_episodes)):
    observation, info = env.reset()
    episode_over = False
    
    while not episode_over:
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        agent.update(observation, action, reward, terminated, next_observation)
        observation = next_observation
        episode_over = terminated or truncated
    
    agent.decay_epsilon()

env.close()

# Show final trained agent
import time
env = gym.make("CartPole-v1", render_mode="human")

for _ in range(5):
    observation, info = env.reset()
    episode_over = False
    
    while not episode_over:
        action = agent.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
    
    time.sleep(1)

env.close()