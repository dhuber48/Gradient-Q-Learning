#!/usr/bin/env python3


from collections import defaultdict
import gymnasium as gym
import numpy as np

#THIS IS ACCELERATED VDBE Q-LEARNING (MY ALGORITHM)

#Note: This q-learning agent is a modified version of the one from farama gym's tutorial:
class Agent:
    def __init__(
            self,
            env: gym.Env,  # env: The training environment
            learning_rate: float, #learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: float, #initial_epsilon: Starting exploration rate (usually 1.0)
            sigma: float, #sigma: parameter for VDBE
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

        self.sigma = sigma

        self.step_counter = 1
        self.discretization_bins = 6

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        # Discretize continuous CartPole observation with bins for each dimension
        obs = (
            np.digitize(obs[0], np.linspace(-2.4, 2.4, self.discretization_bins)),
            np.digitize(obs[1], np.linspace(-4, 4, self.discretization_bins)),
            np.digitize(obs[2], np.linspace(-0.2, 0.2, self.discretization_bins)),
            np.digitize(obs[3], np.linspace(-4, 4, self.discretization_bins)),
        )
        
        # With probability epsilon: explore random action
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit best known action
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

        #Bellman equation to update Q-values
        target = reward + self.discount_factor * future_q_value #R_t+1 + gamma * max_a Q(S_t+1, a)

        #How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action] #TD error: target - Q(S_t, A_t)

        #Implementing basic VDBE
        #Note: can replace tanh with a different squashing function later. VDBE uses tanh(lr*x/2)
        #surprise = -np.tanh(-np.abs(self.lr*temporal_difference)/self.sigma) SAME THING
        surprise = np.tanh(np.abs(temporal_difference)/self.sigma)

        #ACCELERATION IMPLEMENTATION
        surprise_gradient = 1-surprise**2/self.sigma
        responsiveness = surprise_gradient #delta in VDBE paper
        self.epsilon = (1 - responsiveness) * self.epsilon + (responsiveness * surprise)

        # Update estimate in the direction of the error
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        ) #Q(S_t, A_t) += alpha * TD error


