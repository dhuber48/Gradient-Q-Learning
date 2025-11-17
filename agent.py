from collections import defaultdict
import gymnasium as gym
import numpy as np


#Note: This q-learning agent taken from farama gym's tutorial:
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
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    #will need to change obs and action type to match cartpole observation space
    def get_action(self, obs: tuple[int, int, bool]) -> int:
        # Discretize continuous CartPole observation with finer bins for each dimension
        obs = (
            np.digitize(obs[0], np.linspace(-2.4, 2.4, 20)),
            np.digitize(obs[1], np.linspace(-4, 4, 20)),
            np.digitize(obs[2], np.linspace(-0.2, 0.2, 20)),
            np.digitize(obs[3], np.linspace(-4, 4, 20)),
        )
        
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Update Q-values based on agent's experience."""
        # Discretize continuous CartPole observations with finer bins for each dimension
        obs = (
            np.digitize(obs[0], np.linspace(-2.4, 2.4, 20)),
            np.digitize(obs[1], np.linspace(-4, 4, 20)),
            np.digitize(obs[2], np.linspace(-0.2, 0.2, 20)),
            np.digitize(obs[3], np.linspace(-4, 4, 20)),
        )
        next_obs = (
            np.digitize(next_obs[0], np.linspace(-2.4, 2.4, 20)),
            np.digitize(next_obs[1], np.linspace(-4, 4, 20)),
            np.digitize(next_obs[2], np.linspace(-0.2, 0.2, 20)),
            np.digitize(next_obs[3], np.linspace(-4, 4, 20)),
        )
        
        future_q_value = (not terminated) * np.max(self.q_values[next_obs]) #max_a: Q(S_t+1, a)

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value #R_t+1 + gamma * max_a Q(S_t+1, a)

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action] #TD error: target - Q(S_t, A_t)

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        ) #Q(S_t, A_t) += alpha * TD error

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)