from collections import defaultdict
import gymnasium as gym
import numpy as np


#Note: This q-learning agent adapted for Pendulum:
class Agent:
    def __init__(
            self,
            env: gym.Env,  # env: The training environment
            learning_rate: float, #learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: float, #initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: float, #epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: float, #final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: float = 0.95, #discount_factor: How much to value future rewards (0-1)
        ):
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        # Use 5 discrete actions for continuous action space
        self.n_actions = 5
        self.q_values = defaultdict(lambda: np.zeros(self.n_actions))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []
        self.last_reward = 0

    def _discrete_to_continuous(self, action: int) -> np.ndarray:
        """Convert discrete action (0-4) to continuous action (-2 to 2)."""
        return np.array([4.0 * action / (self.n_actions - 1) - 2.0])

    #Pendulum observation space: [cos(theta), sin(theta), angular_velocity]
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # Discretize continuous Pendulum observation with bins for each dimension
        obs = (
            np.digitize(obs[0], np.linspace(-1, 1, 15)),
            np.digitize(obs[1], np.linspace(-1, 1, 15)),
            np.digitize(obs[2], np.linspace(-8, 8, 15)),
        )
        
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            discrete_action = np.random.randint(0, self.n_actions)
        # With probability (1-epsilon): exploit (best known action)
        else:
            discrete_action = int(np.argmax(self.q_values[obs]))
        
        continuous_action = self._discrete_to_continuous(discrete_action)
        return continuous_action

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        next_obs: np.ndarray,
    ):
        """Update Q-values based on agent's experience."""
        # Convert continuous action back to discrete (0-4)
        discrete_action = int(np.clip(np.round((action[0] + 2.0) / 4.0 * (self.n_actions - 1)), 0, self.n_actions - 1))
        
        # Discretize continuous Pendulum observations with bins for each dimension
        obs = (
            np.digitize(obs[0], np.linspace(-1, 1, 15)),
            np.digitize(obs[1], np.linspace(-1, 1, 15)),
            np.digitize(obs[2], np.linspace(-8, 8, 15)),
        )
        next_obs = (
            np.digitize(next_obs[0], np.linspace(-1, 1, 15)),
            np.digitize(next_obs[1], np.linspace(-1, 1, 15)),
            np.digitize(next_obs[2], np.linspace(-8, 8, 15)),
        )
        
        future_q_value = (not terminated) * np.max(self.q_values[next_obs]) #max_a: Q(S_t+1, a)

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value #R_t+1 + gamma * max_a Q(S_t+1, a)

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][discrete_action] #TD error: target - Q(S_t, A_t)

        #updating learning rate with respect to derivative of reward: 
        self.lr *= (1 - 0.01 * (reward - self.last_reward)); self.last_reward = reward
        
        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs][discrete_action] = (
            self.q_values[obs][discrete_action] + self.lr * temporal_difference
        ) #Q(S_t, A_t) += alpha * TD error

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
