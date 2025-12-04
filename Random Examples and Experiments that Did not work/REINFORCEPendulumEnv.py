# import gymnasium as gym
# import numpy as np
# from tqdm import tqdm
# import time

# # CRITICAL: Import the Agent using the exact local file name
# from reinforce_Agent import REINFORCE_Agent, Transition 

# # --- REINFORCE Hyperparameters ---
# learning_rate = 0.05      # Critical for stability
# n_episodes = 5000          
# discount_factor = 0.99      

# # Create the training environment
# env = gym.make("Pendulum-v1")
# env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# # Access the max steps using the standard Gymnasium environment spec, 
# max_episode_steps = env.unwrapped.spec.max_episode_steps 
# if max_episode_steps is None:
#     max_episode_steps = 200 # Default steps for Pendulum-v1

# # Initialize REINFORCE agent
# agent = REINFORCE_Agent(
#     env=env,
#     learning_rate=learning_rate,
#     discount_factor=discount_factor,
# )

# # Training loop
# print(f"Starting REINFORCE training for {n_episodes} episodes...")
# for episode in tqdm(range(n_episodes)):
#     observation, info = env.reset()
    
#     # Store transitions for the current episode
#     episode_data = [] 
    
#     for t in range(max_episode_steps):
#         # 1. Select action from the Gaussian policy
#         continuous_action, log_prob = agent.get_action(observation)
        
#         # Defensive Check: Ensure action is a 1D array of floats for environment
#         if continuous_action.ndim == 0:
#             continuous_action = np.array([continuous_action])
        
#         # 2. Take step in environment
#         # Must unpack 5 values required by Gymnasium:
#         next_observation, reward, terminated, truncated, info = env.step(continuous_action)
        
#         # 3. Store transition (state, action, reward, log_prob)
#         # Store the action as a 1D array (Transition dtype requires shape (1,))
#         episode_data.append(
#             (observation, continuous_action, reward, log_prob)
#         )
            
#         observation = next_observation
#         # Check for episode termination
#         episode_over = terminated or truncated
        
#         if episode_over:
#             break
    
#     # --- EPISODIC UPDATE ---
#     transitions = np.array(episode_data, dtype=Transition)
#     agent.train_episode(transitions)

# env.close()
# print("Training complete. Showing agent performance.")

# # --- Show Final Trained Agent ---
# # Re-create environment for rendering
# env = gym.make("Pendulum-v1", render_mode="human")

# for i in range(5):
#     observation, info = env.reset()
#     episode_over = False
#     start_time = time.time()
#     steps = 0
#     total_reward = 0
    
#     for t in range(max_episode_steps):
#         # Use a deterministic (non-sampled) action for testing/rendering: the mean mu
#         mu, _, _ = agent._forward(observation)
#         # Get the mean action and ensure it's a 1D array (shape (1,))
#         continuous_action = mu.flatten()
        
#         # Defensive Check for testing loop
#         if continuous_action.ndim == 0:
#             continuous_action = np.array([continuous_action])

#         # Must unpack 5 values here as well
#         observation, reward, terminated, truncated, info = env.step(continuous_action)
#         total_reward += reward
#         steps += 1
        
#         episode_over = terminated or truncated
#         if episode_over:
#             break
    
#     elapsed_ms = (time.time() - start_time) * 1000
#     print(f"Test Run {i+1} | Steps: {steps} | Total Reward: {total_reward:.2f}")
#     time.sleep(1)

# env.close()

import gymnasium as gym
import numpy as np
from tqdm import tqdm
import time

# CRITICAL: Import the Agent using the exact local file name
from reinforce_Agent import REINFORCE_Agent, Transition 

# --- REINFORCE Hyperparameters ---
learning_rate = 0.0005      # Critical for stability
n_episodes = 10000          
discount_factor = 0.99      

# Create the training environment
env = gym.make("Pendulum-v1")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# Access the max steps using the standard Gymnasium environment spec, 
max_episode_steps = env.unwrapped.spec.max_episode_steps 
if max_episode_steps is None:
    max_episode_steps = 200 # Default steps for Pendulum-v1

# Initialize REINFORCE agent
agent = REINFORCE_Agent(
    env=env,
    learning_rate=learning_rate,
    discount_factor=discount_factor,
)

# Training loop
print(f"Starting REINFORCE training for {n_episodes} episodes...")
for episode in tqdm(range(n_episodes)):
    observation, info = env.reset()
    
    # Store transitions for the current episode
    episode_data = [] 
    
    for t in range(max_episode_steps):
        # 1. Select action from the Gaussian policy
        continuous_action, log_prob = agent.get_action(observation)
        
        # Defensive Check: Ensure action is a 1D array of floats for environment
        if continuous_action.ndim == 0:
            continuous_action = np.array([continuous_action])
        
        # 2. Take step in environment
        # Must unpack 5 values required by Gymnasium:
        next_observation, reward, terminated, truncated, info = env.step(continuous_action)
        
        # 3. Store transition (state, action, reward, log_prob)
        # Store the action as a 1D array (Transition dtype requires shape (1,))
        episode_data.append(
            (observation, continuous_action, reward, log_prob)
        )
            
        observation = next_observation
        # Check for episode termination
        episode_over = terminated or truncated
        
        if episode_over:
            break
    
    # --- EPISODIC UPDATE ---
    transitions = np.array(episode_data, dtype=Transition)
    agent.train_episode(transitions)

env.close()
print("Training complete. Showing agent performance.")

# --- Show Final Trained Agent ---
# Re-create environment for rendering
env = gym.make("Pendulum-v1", render_mode="human")

for i in range(5):
    observation, info = env.reset()
    episode_over = False
    start_time = time.time()
    steps = 0
    total_reward = 0
    
    for t in range(max_episode_steps):
        # Use a deterministic (non-sampled) action for testing/rendering: the mean mu
        mu, _, _ = agent._forward(observation)
        # Get the mean action and ensure it's a 1D array (shape (1,))
        continuous_action = mu.flatten()
        
        # Defensive Check for testing loop
        if continuous_action.ndim == 0:
            continuous_action = np.array([continuous_action])

        # Must unpack 5 values here as well
        observation, reward, terminated, truncated, info = env.step(continuous_action)
        total_reward += reward
        steps += 1
        
        episode_over = terminated or truncated
        if episode_over:
            break
    
    elapsed_ms = (time.time() - start_time) * 1000
    print(f"Test Run {i+1} | Steps: {steps} | Total Reward: {total_reward:.2f}")
    time.sleep(1)

env.close()