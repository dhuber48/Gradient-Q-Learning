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
        self.reward_sum = 0
        self.last_action = 0

        self.step_counter = 1
        self.discretization_bins = 10

    #will need to change obs and action type to match cartpole observation space
    def get_action(self, obs: tuple[int, int, bool]) -> int:
        # Discretize continuous CartPole observation with finer bins for each dimension
        obs = (
            np.digitize(obs[0], np.linspace(-2.4, 2.4, self.discretization_bins)),
            np.digitize(obs[1], np.linspace(-4, 4, self.discretization_bins)),
            np.digitize(obs[2], np.linspace(-0.2, 0.2, self.discretization_bins)),
            np.digitize(obs[3], np.linspace(-4, 4, self.discretization_bins)),
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

        # Discretize continuous CartPole observations with bins for each dimension
        obs = (
            np.digitize(obs[0], np.linspace(-2.4, 2.4, self.discretization_bins)),
            np.digitize(obs[1], np.linspace(-4, 4, self.discretization_bins)),
            np.digitize(obs[2], np.linspace(-0.2, 0.2, self.discretization_bins)),
            np.digitize(obs[3], np.linspace(-4, 4, self.discretization_bins)),
        )
        next_obs = (
            np.digitize(next_obs[0], np.linspace(-2.4, 2.4, self.discretization_bins)),
            np.digitize(next_obs[1], np.linspace(-4, 4, self.discretization_bins)),
            np.digitize(next_obs[2], np.linspace(-0.2, 0.2, self.discretization_bins)),
            np.digitize(next_obs[3], np.linspace(-4, 4, self.discretization_bins)),
        )
        
        if not terminated:
            self.step_counter += 1
        else:
            self.step_counter = 1

        #Update Q-values based on agent's experience
        future_q_value = (not terminated) * np.max(self.q_values[next_obs]) #max_a: Q(S_t+1, a)

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value #R_t+1 + gamma * max_a Q(S_t+1, a)

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action] #TD error: target - Q(S_t, A_t)

        #udpating learning rate with respect to derivative of reward:

        # if action != self.last_action:  #avoid division by zero
        #     self.lr *= (reward + self.reward_sum - self.reward_sum)/(action-self.last_action) #+ 1e-5)  #add small constant to avoid division by zero
        # print(f"Action: {action}")
        # print(f"Updated learning rate: {self.lr}")

        # self.reward_sum += reward
        # self.last_action = action

        self.lr = max(0.01, min(1.0, self.lr*(1- .1* 0.01 * self.step_counter**(-0.99)) ))  #differentiable learning rate decays based on derivative of reward, with truncation between 0.01 and 1.0.
        #print(f"Differentiable learning rate: {self.lr}")

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

    def differentiable_epsilon(self): #this is not going to work because we can't change epsilon so drastically within each episode.
        self.epsilon = self.epsilon *(1+ 0.01 * self.step_counter**(-0.99))  #example decay based on step count
        print(f"Differentiable epsilon: {self.epsilon}")