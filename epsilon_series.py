import numpy as np
import matplotlib.pyplot as plt

#THIS FILE WAS USED TO VISUALIZE THE PROGRESSION OF EXPLORATION RATE


# --- Parameters ---
steps = 200
sigma = 50.0          # Scaling factor
epsilon_0 = 0.2       # Starting value
delta_fixed = 0.1     # Fixed rate for comparison (ungraphed)

# --- Run Simulation ---
def run_simulation(T_vals):
    eps_fixed_hist = [epsilon_0]
    eps_adaptive_hist = [epsilon_0]
    delta_adaptive_hist = []
    
    for t in range(len(T_vals)):
        current_T = T_vals[t]
        
        # U(T) = tanh(|T|/sigma)
        u_val = np.tanh(np.abs(current_T) / sigma)
        
        # --- Base VDBE ---
        prev_eps_fixed = eps_fixed_hist[-1]
        next_eps_fixed = delta_fixed * u_val + (1 - delta_fixed) * prev_eps_fixed 
        eps_fixed_hist.append(next_eps_fixed)
        
        # --- Adaptive Delta (Accelerated VDBE) ---
        delta_adaptive = (1 - u_val**2) / sigma
        
        prev_eps_adaptive = eps_adaptive_hist[-1]
        next_eps_adaptive = delta_adaptive * u_val + (1 - delta_adaptive) * prev_eps_adaptive
        
        eps_adaptive_hist.append(next_eps_adaptive)
        delta_adaptive_hist.append(delta_adaptive)
        
    return eps_fixed_hist, eps_adaptive_hist, delta_adaptive_hist

# --- 1. Generate Random T values ---
T_random = np.random.uniform(0, 100, steps)

# --- 2. Generate Bell Curve T values  ---
x = np.linspace(0, steps, steps)
center = steps / 2
width = steps / 6  
T_bell = 100 * np.exp(-((x - center)**2) / (2 * width**2))

# --- Run Simulations ---
fixed_rand, adapt_rand, delta_rand = run_simulation(T_random)
fixed_bell, adapt_bell, delta_bell = run_simulation(T_bell)

# --- Plotting Function ---
def plot_results(T_vals, eps_fixed, eps_adaptive, delta_vals, title_suffix):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top Plot: Epsilon Output
    ax1.plot(eps_fixed, color='blue', linestyle='--', alpha=0.6, label=f'Original VDBE ($\delta$={delta_fixed})')
    ax1.plot(eps_adaptive, color='red', linewidth=2.5, label='Accelerated VDBE $\delta_*$')
    ax1.set_ylabel(r'$\varepsilon$ Value')
    ax1.set_title(f'Comparison: Original VDBE vs. Accelerated VDBE Update Rule ({title_suffix})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom Plot: Input T and Delta
    ax2.set_ylabel('Input |T|', color='gray')
    ax2.plot(T_vals, color='gray', alpha=0.3, label='Input |T|', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, 110) # Give some headroom
    
    # Twin axis for Delta
    ax2_right = ax2.twinx()
    
    # Plotting the Accelerated Delta 
    ax2_right.plot(delta_vals, color='green', linestyle=':', label=r'Accelerated $\delta_*$')
    ax2_right.set_ylabel(r'Accelerated $\delta_*$ (Slope)', color='green')
    ax2_right.tick_params(axis='y', labelcolor='green')
    
    # Legends
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    ax2.set_xlabel('Time Step')
    plt.tight_layout()
    plt.show()

# --- Display Plots ---
plot_results(T_random, fixed_rand, adapt_rand, delta_rand, "Random T")
plot_results(T_bell, fixed_bell, adapt_bell, delta_bell, "Bell Curve T")