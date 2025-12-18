import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from collections import defaultdict

#THIS FILE WAS USED TO TEST ALL OF THE AGENTS IN THIS PROJECT AND COMPARE THEIR RESULTS

from normal_cartPoleAgent import Agent as NormalAgent
from VDBE_cartPoleAgent import Agent as VDBEAgent
from Accelerated_VDBE_cartPoleAgent import Agent as AcceleratedAgent

# --- CONFIGURATION ---
NUM_RUNS = 10           # Runs per agent
MAX_EPISODES = 5000    # Hard cap
SOLVED_AVG_REWARD = 225.0 #475.0
SOLVED_WINDOW = 100 #100

BASE_PARAMS = {
    'learning_rate': 0.2,
    'initial_epsilon': 1.0,

}

# --- AGENT SETUP ---
AGENTS_TO_TEST = [
    {
        'class': NormalAgent,
        'label': 'Normal Q-Learning',
        'color': 'red',
        'linestyle': '-',  # Solid
        'params': {'epsilon_decay': 1.0 / MAX_EPISODES, 'final_epsilon': 0.00}
    },
    
    # --- STANDARD VDBE (Dashed, Blue Scales) ---
    {
        'class': VDBEAgent,
        'label': 'Standard VDBE (σ=1)',
        'color': 'cyan',
        'linestyle': '--', 
        'params': {'sigma': 1, 'delta': 0.1}
    },
    {
        'class': VDBEAgent,
        'label': 'Standard VDBE (σ=5)',
        'color': 'deepskyblue',
        'linestyle': '--',
        'params': {'sigma': 5, 'delta': 0.1}
    },
    {
        'class': VDBEAgent,
        'label': 'Standard VDBE (σ=20)',
        'color': 'blue',
        'linestyle': '--',
        'params': {'sigma': 20, 'delta': 0.1}
    },
    {
        'class': VDBEAgent,
        'label': 'Standard VDBE (σ=100)',
        'color': 'navy',
        'linestyle': '--',
        'params': {'sigma': 100, 'delta': 0.1}
    },

    # --- ACCELERATED VDBE (Solid, Orange/Gold Scales) ---
    {
        'class': AcceleratedAgent,
        'label': 'Accel VDBE (σ=1)',
        'color': 'gold',
        'linestyle': '-',
        'params': {'sigma': 1}
    },
    {
        'class': AcceleratedAgent,
        'label': 'Accel VDBE (σ=5)',
        'color': 'orange',
        'linestyle': '-',
        'params': {'sigma': 5}
    },
    {
        'class': AcceleratedAgent,
        'label': 'Accel VDBE (σ=20)',
        'color': 'darkorange',
        'linestyle': '-',
        'params': {'sigma': 20}
    },
        {
        'class': AcceleratedAgent,
        'label': 'Accel VDBE (σ=100)',
        'color': 'saddlebrown',
        'linestyle': '-',
        'params': {'sigma': 100}
    },
]

# --- One training cycle ---
def run_single_trial(AgentClass, specific_params):
    env = gym.make("CartPole-v1")
    try:
        agent = AgentClass(env=env, **BASE_PARAMS, **specific_params)
    except TypeError:
        agent = AgentClass(env=env, **BASE_PARAMS)

    rewards = []
    start_time = time.time()
    solved_ep = -1

    for episode in tqdm(range(MAX_EPISODES), desc="  Training", leave=False):
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

        if hasattr(agent, "decay_epsilon"): #For standard Q-learning only
            agent.decay_epsilon()

        rewards.append(episode_total_reward)

        if len(rewards) >= SOLVED_WINDOW:
            if np.mean(rewards[-SOLVED_WINDOW:]) >= SOLVED_AVG_REWARD:
                solved_ep = episode + 1
                break
    
    env.close()
    duration = time.time() - start_time
    if solved_ep == -1: solved_ep = MAX_EPISODES
    
    return rewards, solved_ep, duration

def process_rewards(all_runs_rewards):
    max_len = max(len(r) for r in all_runs_rewards)
    padded = []
    for r in all_runs_rewards:
        pad_width = max_len - len(r)
        if pad_width > 0:
            padded.append(r + [r[-1]] * pad_width)
        else:
            padded.append(r)
    return np.array(padded)


results = {}

plt.style.use('default') 
plt.figure(figsize=(12, 8))

print(f"Starting Benchmark: {len(AGENTS_TO_TEST)} Agents, {NUM_RUNS} runs each.")

# --- MAIN LOOP (runs trials for each agent) ---
for agent_config in AGENTS_TO_TEST:
    name = agent_config['label']
    AgentCls = agent_config['class']
    params = agent_config['params']
    color = agent_config['color']
    
    print(f"\nTraining {name}...")
    
    run_rewards = []
    solve_episodes = []
    run_times = []

    for i in tqdm(range(NUM_RUNS), desc=f"Runs ({name})"):
        r, s, t = run_single_trial(AgentCls, params)
        run_rewards.append(r)
        solve_episodes.append(s)
        run_times.append(t)

    # Process Data
    padded_data = process_rewards(run_rewards)
    mean_curve = np.mean(padded_data, axis=0)
    std_curve = np.std(padded_data, axis=0)
    
    # Smooth for plotting
    window = 50
    smooth_mean = np.convolve(mean_curve, np.ones(window)/window, mode='valid')
    smooth_std = std_curve[window-1:]
    x_axis = np.arange(len(smooth_mean)) + window

    results[name] = {
        'avg_solve': np.mean(solve_episodes),
        'std_solve': np.std(solve_episodes),
        'avg_time': np.mean(run_times),
        'min_solve': np.min(solve_episodes)
    }

    # --- PLOTTING LOGIC ---
    # 1. Plot the Mean Line
    plt.plot(x_axis, smooth_mean, color=color, linewidth=2, label=f'{name} (Mean)')
    
    # 2. Plot the Shaded Region (Std Dev) with a separate label entry in legend logic
    plt.fill_between(x_axis, 
                     smooth_mean - smooth_std, 
                     smooth_mean + smooth_std, 
                     color=color, alpha=0.15, 
                     label=f'{name} (Smoothed mean +- std dev between all runs at a given episode)')
    
    # 3. Mark the average solved point with a Star
    avg_solve_point = int(np.mean(solve_episodes))
    if avg_solve_point < len(smooth_mean) + window:
        # Find the y-value at that specific x-point
        # We adjust index because smooth_mean is shorter than x_axis by 'window'
        idx = avg_solve_point - window 
        if 0 <= idx < len(smooth_mean):
            plt.plot(avg_solve_point, smooth_mean[idx], marker='*', color=color, markersize=15, markeredgecolor='black')
            plt.text(avg_solve_point, smooth_mean[idx]+15, f"{avg_solve_point} eps", color=color, fontweight='bold')

# --- FINALIZE PLOT ---
plt.axhline(y=SOLVED_AVG_REWARD, color='black', linestyle='--', alpha=0.5, label='Solved Threshold (475)')

plt.ylim(0, 520) 

plt.title(f'Agent Comparison ({NUM_RUNS} runs averaged)')
plt.xlabel('Episode')
plt.ylabel('Average Reward (Smoothed Over Last 50 Episodes)')

# Legend
plt.legend(loc='upper left', frameon=True, fontsize='medium')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# --- PRINT TEXT REPORT ---
print("\n" + "="*65)
print(f"{'AGENT':<20} | {'SOLVED (Avg Eps)':<20} | {'TIME (Avg)':<15} | {'BEST RUN':<10}")
print("-" * 65)
for name, stats in results.items():
    eps_str = f"{stats['avg_solve']:.1f} ± {stats['std_solve']:.1f}"
    time_str = f"{stats['avg_time']:.2f}s"
    print(f"{name:<20} | {eps_str:<20} | {time_str:<15} | {stats['min_solve']:<10}")
print("="*65)

plt.show()