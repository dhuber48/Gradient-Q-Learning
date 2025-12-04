# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import matplotlib.pyplot as plt

# # ----------------------------------------
# # 1. Custom Environment with Continuous Reward
# # ----------------------------------------
# class Continuous1DEnv(gym.Env):
#     """
#     A simple 1D environment where the agent moves left/right to reach 0.
#     Reward is strictly continuous: e^(-distance^2).
#     """
#     def __init__(self):
#         super(Continuous1DEnv, self).__init__()
#         # State: Position between -5.0 and 5.0
#         self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        
#         # Actions: 0 (Left), 1 (Stay), 2 (Right)
#         self.action_space = spaces.Discrete(3)
        
#         self.state = None
#         self.max_steps = 100
#         self.current_step = 0

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         # Start at a random position away from the center
#         self.state = np.array([np.random.uniform(-4, 4)], dtype=np.float32)
#         self.current_step = 0
#         return self.state, {}

#     def step(self, action):
#         # Actions: 0 -> -0.5 move, 1 -> 0.0 move, 2 -> +0.5 move
#         move = 0
#         if action == 0: move = -0.5
#         if action == 2: move = 0.5
        
#         # Update state with clip
#         self.state[0] += move
#         self.state[0] = np.clip(self.state[0], -5.0, 5.0)
        
#         # --- THE KEY PART: Continuous Reward ---
#         # Using a Gaussian curve so reward is 1.0 at 0, and decays smoothly to 0.
#         dist = abs(self.state[0])
#         reward = np.exp(-(dist**2))
        
#         self.current_step += 1
#         terminated = False # Endless for simplicity, or stop if very close
#         truncated = self.current_step >= self.max_steps
        
#         if dist < 0.1:
#             terminated = True # Reached goal
#             reward += 10.0 # Bonus for finishing
            
#         return self.state, reward, terminated, truncated, {}

# # ----------------------------------------
# # 2. Tabular Q-Learning Agent
# # ----------------------------------------
# class TabularQAgent:
#     def __init__(self, action_space_size, state_bins=50, low=-5.0, high=5.0):
#         self.action_space_size = action_space_size
#         self.lr = 0.1  # Initial Learning Rate (Alpha)
#         self.gamma = 0.95
#         self.epsilon = 1.0
#         self.epsilon_decay = 0.995
#         self.min_epsilon = 0.01
        
#         # Discretize the continuous state into bins
#         self.bins = np.linspace(low, high, state_bins)
#         # Q-Table: [num_bins, num_actions]
#         self.q_table = np.zeros((state_bins + 1, action_space_size))

#     def get_discrete_state(self, state):
#         # Determine which bin the continuous state falls into
#         return np.digitize(state[0], self.bins)

#     def select_action(self, state):
#         if np.random.rand() < self.epsilon:
#             return np.random.choice(self.action_space_size)
        
#         state_idx = self.get_discrete_state(state)
#         return np.argmax(self.q_table[state_idx])

#     def update(self, state, action, reward, next_state):
#         state_idx = self.get_discrete_state(state)
#         next_state_idx = self.get_discrete_state(next_state)
        
#         # Standard Q-Learning update rule
#         # Q(s,a) = Q(s,a) + lr * [R + gamma * max(Q(s',a')) - Q(s,a)]
#         best_next_q = np.max(self.q_table[next_state_idx])
#         current_q = self.q_table[state_idx, action]
        
#         target = reward + self.gamma * best_next_q
#         error = target - current_q
        
#         self.q_table[state_idx, action] += self.lr * error
        
#     def decay_epsilon(self):
#         self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# # ----------------------------------------
# # 3. Training Loop with Adaptive LR
# # ----------------------------------------
# def train_adaptive_q():
#     env = Continuous1DEnv()
#     agent = TabularQAgent(env.action_space.n)
    
#     num_episodes = 500
    
#     # Tracking metrics
#     rewards_history = []
#     lr_history = []
    
#     # For derivative calculation
#     prev_episode_reward = 0
    
#     print("Starting Adaptive Learning Rate Training...")
    
#     for episode in range(num_episodes):
#         state, _ = env.reset()
#         total_reward = 0
#         done = False
        
#         while not done:
#             action = agent.select_action(state)
#             next_state, reward, terminated, truncated, _ = env.step(action)
            
#             agent.update(state, action, reward, next_state)
            
#             state = next_state
#             total_reward += reward
#             done = terminated or truncated
        
#         agent.decay_epsilon()
#         rewards_history.append(total_reward)
        
#         # --- ADAPTIVE LEARNING RATE LOGIC ---
#         # Calculate derivative of reward (Change in performance)
#         reward_derivative = total_reward - prev_episode_reward
        
#         # Example Logic: 
#         # If reward is increasing fast (derivative > 0), boost LR to learn faster?
#         # Or if reward is unstable (derivative high absolute value), lower LR?
#         # Here: We boost LR if improving, dampen if getting worse.
        
#         base_lr = 0.1
        
#         # Simple Sigmoid-like scaling based on derivative
#         # If derivative is +10 (big improvement), factor > 1. 
#         # If derivative is -10 (getting worse), factor < 1.
#         factor = 1.0 + np.tanh(reward_derivative * 0.1) 
        
#         # Update Agent's LR
#         agent.lr = base_lr * factor
#         # Clamp to reasonable values
#         agent.lr = np.clip(agent.lr, 0.01, 0.5)
        
#         lr_history.append(agent.lr)
#         prev_episode_reward = total_reward
        
#         if episode % 50 == 0:
#             print(f"Ep {episode}: Reward={total_reward:.2f}, dR={reward_derivative:.2f}, New LR={agent.lr:.4f}")

#     # Visualization
#     fig, ax1 = plt.subplots(figsize=(10, 6))

#     ax1.set_xlabel('Episode')
#     ax1.set_ylabel('Total Reward', color='tab:blue')
#     ax1.plot(rewards_history, color='tab:blue', alpha=0.6, label='Reward')
#     ax1.tick_params(axis='y', labelcolor='tab:blue')

#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#     ax2.set_ylabel('Adaptive Learning Rate', color='tab:red')
#     ax2.plot(lr_history, color='tab:red', label='Learning Rate')
#     ax2.tick_params(axis='y', labelcolor='tab:red')

#     plt.title('Reward vs Adaptive Learning Rate')
#     plt.show()

# if __name__ == "__main__":
#     train_adaptive_q()
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import time
import os
#from IPython.display import clear_output # For better visualization

# ----------------------------------------
# 1. Custom Environment with Continuous Reward
# ----------------------------------------
class Continuous1DEnv(gym.Env):
    """
    A simple 1D environment where the agent moves left/right to reach 0.
    Reward is strictly continuous: e^(-distance^2).
    """
    def __init__(self):
        super(Continuous1DEnv, self).__init__()
        # State: Position between -5.0 and 5.0
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        
        # Actions: 0 (Left), 1 (Stay), 2 (Right)
        self.action_space = spaces.Discrete(3)
        
        self.state = None
        self.max_steps = 100
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start at a random position away from the center
        self.state = np.array([np.random.uniform(-4, 4)], dtype=np.float32)
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        # Actions: 0 -> -0.5 move, 1 -> 0.0 move, 2 -> +0.5 move
        move = 0
        if action == 0: move = -0.5
        if action == 2: move = 0.5
        
        # Update state with clip
        self.state[0] += move
        self.state[0] = np.clip(self.state[0], -5.0, 5.0)
        
        # --- THE KEY PART: Continuous Reward ---
        # Using a Gaussian curve so reward is 1.0 at 0, and decays smoothly to 0.
        dist = abs(self.state[0])
        reward = np.exp(-(dist**2))
        
        self.current_step += 1
        terminated = False # Endless for simplicity, or stop if very close
        truncated = self.current_step >= self.max_steps
        
        if dist < 0.1:
            terminated = True # Reached goal
            reward += 10.0 # Bonus for finishing
            
        return self.state, reward, terminated, truncated, {}
    
    # ----------------------------------------
    # MODIFICATION: Render function
    # ----------------------------------------
    def render(self, mode='human'):
        if mode == 'human':
            # Clear previous output for dynamic view
            #clear_output(wait=True)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Create a simple 1D plot visualization
            fig, ax = plt.subplots(figsize=(8, 1))
            
            # Set limits and hide y-axis
            ax.set_xlim(-5.0, 5.0)
            ax.set_ylim(0, 1)
            ax.yaxis.set_visible(False)
            ax.set_title(f"1D Environment: Position={self.state[0]:.2f}")
            
            # Draw the goal area (0 +/- 0.1)
            ax.axvspan(-0.1, 0.1, color='green', alpha=0.3, label='Goal')
            
            # Draw the agent's current position
            ax.plot(self.state[0], 0.5, 'o', color='red', markersize=10, label='Agent')
            
            # Print the plot
            plt.show()

# ----------------------------------------
# 2. Tabular Q-Learning Agent
# ----------------------------------------
class TabularQAgent:
    def __init__(self, action_space_size, state_bins=50, low=-5.0, high=5.0):
        self.action_space_size = action_space_size
        self.lr = 0.1  # Initial Learning Rate (Alpha)
        self.gamma = 0.95
        # Set epsilon to 0 for a greedy evaluation policy in test mode
        self.epsilon = 1.0 
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        # Discretize the continuous state into bins
        self.bins = np.linspace(low, high, state_bins)
        # Q-Table: [num_bins, num_actions]
        self.q_table = np.zeros((state_bins + 1, action_space_size))

    def get_discrete_state(self, state):
        # Determine which bin the continuous state falls into
        return np.digitize(state[0], self.bins)

    def select_action(self, state, exploit_only=False): # MODIFICATION: Added exploit_only flag
        # If exploit_only is True, or if random number is too low (standard explore)
        if not exploit_only and np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space_size)
        
        state_idx = self.get_discrete_state(state)
        # Handle case where Q-values are all zero (e.g., initial state)
        if np.all(self.q_table[state_idx] == 0):
             return np.random.choice(self.action_space_size)
             
        return np.argmax(self.q_table[state_idx])

    def update(self, state, action, reward, next_state):
        state_idx = self.get_discrete_state(state)
        next_state_idx = self.get_discrete_state(next_state)
        
        # Standard Q-Learning update rule
        # Q(s,a) = Q(s,a) + lr * [R + gamma * max(Q(s',a')) - Q(s,a)]
        best_next_q = np.max(self.q_table[next_state_idx])
        current_q = self.q_table[state_idx, action]
        
        target = reward + self.gamma * best_next_q
        error = target - current_q
        
        self.q_table[state_idx, action] += self.lr * error
        
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# ----------------------------------------
# 3. Training Loop with Adaptive LR
# ----------------------------------------
def train_adaptive_q():
    env = Continuous1DEnv()
    agent = TabularQAgent(env.action_space.n)
    
    num_episodes = 500
    
    # Tracking metrics
    rewards_history = []
    lr_history = []
    
    # For derivative calculation
    prev_episode_reward = 0
    
    print("Starting Adaptive Learning Rate Training...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Use standard select_action which respects epsilon
            action = agent.select_action(state) 
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            agent.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            done = terminated or truncated
        
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        
        # --- ADAPTIVE LEARNING RATE LOGIC ---
        # Calculate derivative of reward (Change in performance)
        reward_derivative = total_reward - prev_episode_reward
        
        # Here: We boost LR if improving, dampen if getting worse.
        base_lr = 0.1
        
        # Simple Sigmoid-like scaling based on derivative
        factor = 1.0 + np.tanh(reward_derivative * 0.1) 
        
        # Update Agent's LR
        agent.lr = base_lr * factor
        # Clamp to reasonable values
        agent.lr = np.clip(agent.lr, 0.01, 0.5)
        
        lr_history.append(agent.lr)
        prev_episode_reward = total_reward
        
        if episode % 50 == 0:
            print(f"Ep {episode}: Reward={total_reward:.2f}, dR={reward_derivative:.2f}, New LR={agent.lr:.4f}")

    # Visualization
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color='tab:blue')
    ax1.plot(rewards_history, color='tab:blue', alpha=0.6, label='Reward')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Adaptive Learning Rate', color='tab:red')
    ax2.plot(lr_history, color='tab:red', label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Reward vs Adaptive Learning Rate')
    plt.show()
    
    return agent, env # MODIFICATION: Return agent and env
    
# ----------------------------------------
# 4. Testing/Evaluation Loop (New Function)
# ----------------------------------------
def test_agent(agent, env, num_test_episodes=5):
    """
    Evaluates the trained agent and visualizes its performance.
    """
    print("\n--- Starting Evaluation (Greedy Policy) ---")
    
    for episode in range(num_test_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        print(f"Test Episode {episode + 1}: Starting at {state[0]:.2f}")
        
        while not done:
            # Select action using only the learned Q-values (exploit_only=True)
            action = agent.select_action(state, exploit_only=True) 
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Render the environment to visualize the movement
            env.render()
            time.sleep(0.1) # Slow down for human observation
            
            state = next_state
            total_reward += reward
            step_count += 1
            done = terminated or truncated

        print(f"Test Episode {episode + 1} finished in {step_count} steps with total reward: {total_reward:.2f}")

if __name__ == "__main__":
    # Train the agent and get the final agent/environment instances
    trained_agent, env_instance = train_adaptive_q()
    
    # Test the agent with visualization
    test_agent(trained_agent, env_instance)