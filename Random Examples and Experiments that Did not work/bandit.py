# action_reward_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ActionRewardEnv(gym.Env):
    """
    Simple env: discrete actions and reward is analytic function of action.
    Observation is a dummy scalar (keeps compatibility with tabular Q-learning).
    Episode length is 1 by default.
    Reward: r(a) = - (a - 7)^2  (max at a=7).
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, n_actions=11, target=7, noise_std=0.0, max_steps=1):
        super().__init__()
        self.n_actions = n_actions
        self.action_space = spaces.Discrete(n_actions)           # actions 0..n_actions-1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.target = target
        self.noise_std = noise_std
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.step_count = 0
        # return a dummy observation
        return np.array([0.0], dtype=np.float32)

    def step(self, action):
        self.step_count += 1

        # reward defined directly as function of action (deterministic + optional noise)
        base_reward = - (float(action) - float(self.target)) ** 2
        noise = np.random.normal(scale=self.noise_std) if self.noise_std > 0 else 0.0
        reward = base_reward + noise

        done = (self.step_count >= self.max_steps)
        obs = np.array([0.0], dtype=np.float32)
        info = {"base_reward": base_reward, "noise": noise}
        return obs, float(reward), bool(done), info

    def render(self, mode="human"):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    env = ActionRewardEnv()
    obs = env.reset()
    obs, r, done, info = env.step(7)
    print("action 7 reward:", r, info)
