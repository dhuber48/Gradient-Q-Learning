import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
LEARNING_RATE = 1e-3
GAMMA = 0.99          # Discount factor
HIDDEN_SIZE = 128
NUM_EPISODES = 1000
# MountainCarContinuous is solved when avg reward > 90 over 100 trials.
# It is a difficult environment for vanilla REINFORCE due to sparse rewards.
PRINT_EVERY = 50

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        
        # Output for Mean (mu)
        self.fc_mu = nn.Linear(hidden_size, action_dim)
        
        # Output for Standard Deviation (sigma)
        # We learn the log_std to ensure std is always positive when exponentiated
        self.fc_log_std = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        mu = self.fc_mu(x)
        
        # Clamp log_std to prevent numerical instability
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        
        return mu, std

class REINFORCE_Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = PolicyNetwork(self.state_dim, self.action_dim, HIDDEN_SIZE).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        
        # Storage for current episode
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        mu, std = self.policy(state)
        
        # Create a normal distribution based on the predicted mean and std
        dist = Normal(mu, std)
        
        # Sample an action (this allows for exploration)
        action = dist.sample()
        
        # Save the log probability of the action for the update step
        log_prob = dist.log_prob(action)
        self.log_probs.append(log_prob)
        
        # Return the action as a numpy array
        # Note: MountainCarContinuous expects actions in [-1, 1]
        # We clamp purely for the environment interaction, but keep the gradient flow on the raw action
        return action.detach().cpu().numpy()[0]

    def update_policy(self):
        """
        Performs the Monte Carlo Policy Gradient update.
        loss = - sum(log_prob * Gt)
        """
        discounted_returns = []
        R = 0
        
        # Calculate returns-to-go (G_t) backwards
        for r in reversed(self.rewards):
            R = r + GAMMA * R
            discounted_returns.insert(0, R)
            
        discounted_returns = torch.tensor(discounted_returns).to(self.device)
        
        # Normalize returns
        # This is CRITICAL for convergence in continuous environments with high variance
        if len(discounted_returns) > 1:
            discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-9)
        
        policy_loss = []
        for log_prob, Gt in zip(self.log_probs, discounted_returns):
            # We want to maximize return, so we minimize negative return
            # Summing the log_probs for the action dimensions (if > 1D action)
            policy_loss.append(-log_prob.sum() * Gt)
            
        self.optimizer.zero_grad()
        # Sum all losses for the episode and backpropagate
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Reset memory
        self.log_probs = []
        self.rewards = []

def train():
    env = gym.make('MountainCarContinuous-v0')
    agent = REINFORCE_Agent(env)
    
    running_reward = 0
    reward_history = []
    
    print(f"Starting training on {agent.device}...")
    
    for i_episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(state)
            
            # Step the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # MountainCarContinuous specific: 
            # The raw reward is -0.1 * action^2, plus +100 if it reaches goal.
            # It's helpful to not modify the reward too much to prove the algorithm works,
            # but standard REINFORCE struggles with the negative drift without normalization (handled in update).
            
            agent.rewards.append(reward)
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        agent.update_policy()
        
        # Smoothing for display
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        reward_history.append(episode_reward)
        
        if i_episode % PRINT_EVERY == 0:
            print(f"Episode {i_episode}\tLast Reward: {episode_reward:.2f}\tAvg Reward (Smooth): {running_reward:.2f}")
            
        # Optional: Save checkpoint or stop if solved
        if running_reward > 90:
            print(f"Solved at episode {i_episode}!")
            torch.save(agent.policy.state_dict(), "reinforce_mountaincar_policy.pth")
            break

    env.close()
    
    # Plotting results
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

def test_model():
    """ Runs a visualization of the trained model """
    # Re-initialize environment with render mode
    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    agent = REINFORCE_Agent(env)
    
    try:
        agent.policy.load_state_dict(torch.load("reinforce_mountaincar_policy.pth"))
        agent.policy.eval() # Set to evaluation mode
        print("Loaded saved model.")
    except FileNotFoundError:
        print("No saved model found, running with random weights.")

    state, _ = env.reset()
    total_reward = 0
    
    # Run for one episode
    while True:
        # In test mode, we usually just take the mean (deterministic) or sample with very low std
        # Here we just use the mean for best performance
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
        mu, _ = agent.policy(state_tensor)
        action = mu.detach().cpu().numpy()[0]
        
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
            
    print(f"Test Episode Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    # 1. Train the agent
    train()
    
    # 2. Visualize the result (uncomment to run)
    # test_model()