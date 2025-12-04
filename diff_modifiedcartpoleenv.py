from collections import defaultdict
import gymnasium as gym
import random, numpy as np
from tqdm import tqdm
from cartPoleAgent import Agent

# Using wrapper to make reward differentiable
class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.k = 0

    def reset(self, **kwargs):
        #print(self.k)
        self.k = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action) 
        if not terminated and not truncated: #increment k only if not done
            self.k += 1
        #else:
        modified_reward = self.k**.01 #this seems to work
        #float(np.exp(self.k)) #increased too fast so didn't work
        #print(modified_reward)
        return observation, modified_reward, terminated, truncated, info

#based on farama gym tutorial code:
learning_rate = 0.2        # How fast to learn (higher = faster but less stable)
n_episodes = 2000       # Number of episodes to train on
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / n_episodes  # Reduce exploration over time
final_epsilon = 0.0        # Always keep some exploration
reward_list = []
episode_counter = 0

# Create our training environment - a cart with a pole that needs balancing
env = gym.make("CartPole-v1")
env = RewardWrapper(env)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# Initialize agent
agent = Agent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

##Idea: decay epsilon based on rate of change of total reward for a given episode. 
# If reward is increasing, decay epsilon faster to exploit. 
# If reward is decreasing, decay epsilon slower (or even increase epsilon) to explore more.
# If reward is stagnant, decay epsilon at normal rate or keep epsilon constant.
def diff_epsilon():
    if len(reward_list) < n_episodes/2:
        agent.decay_epsilon()
        return
    
    if reward_list[-1] > reward_list[-2]:
        agent.epsilon -= agent.epsilon_decay * 1.5  #decay faster
        print("decay faster")
    elif reward_list[-1] < reward_list[-2]:
        agent.epsilon += agent.epsilon_decay * 0.5  #decay slower or increase
        print("decay slower")
   
    agent.epsilon = max(final_epsilon, min(1.0, agent.epsilon)) #cap epsilon between final_epsilon and 1.0

def record_reward():
    reward_list.append(agent.step_counter)

# Training loop
for episode in tqdm(range(n_episodes)):
    observation, info = env.reset()
    episode_over = False
    
    episode_total_reward = 0

    while not episode_over:
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        episode_total_reward += reward

        agent.update(observation, action, reward, terminated, next_observation)
        observation = next_observation
        episode_over = terminated or truncated
    
    #agent.decay_epsilon() 
    #record_reward()

    reward_list.append(episode_total_reward)
    diff_epsilon()
    print(f"Episode {episode+1}: Epsilon = {agent.epsilon:.4f}")

env.close()



# Show final trained agent
import time
#env = gym.make("CartPole-v1", render_mode="human")
eval_env = RewardWrapper(gym.make("CartPole-v1", render_mode="human"))


for i in range(5):
    observation, info = eval_env.reset()
    episode_over = False
    start_time = time.time()
    
    while not episode_over:
        action = agent.get_action(observation)
        observation, reward, terminated, truncated, info = eval_env.step(action)
        episode_over = terminated or truncated
    
    elapsed_ms = (time.time() - start_time) * 1000
    print(f"Episode {i+1} duration: {elapsed_ms:.0f}ms")
    time.sleep(1)

eval_env.close()