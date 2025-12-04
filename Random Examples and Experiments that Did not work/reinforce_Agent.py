import numpy as np
import gymnasium as gym

# Define the structure for storing episodic data
# log_prob is critical for the REINFORCE update
Transition = np.dtype([
    ('state', np.float32, (3,)), 
    ('action', np.float32, (1,)), 
    ('reward', np.float32), 
    ('log_prob', np.float32)
])

class REINFORCE_Agent:
    """A simplified REINFORCE agent (Monte Carlo Policy Gradient) using NumPy."""

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        discount_factor: float = 0.99,
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        
        # State space size (cos(theta), sin(theta), theta_dot)
        self.input_size = env.observation_space.shape[0]
        self.hidden_size = 64
        self.action_limit = env.action_space.high[0] # Typically 2.0 for Pendulum

        self._init_network()

    def _init_network(self):
        """Initializes the policy network (simple 2-layer MLP weights)."""
        
        def init_weights(input_dim, output_dim):
            # He/Xavier approximation for weight initialization
            return np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)

        # Policy Network Weights
        self.W1 = init_weights(self.input_size, self.hidden_size)
        self.b1 = np.zeros(self.hidden_size)
        
        # Output layer splits into Mean (mu) and Log Standard Deviation (log_std)
        self.W_mu = init_weights(self.hidden_size, 1)
        self.b_mu = np.zeros(1)
        self.W_logstd = init_weights(self.hidden_size, 1)
        self.b_logstd = np.zeros(1)
        
    def _forward(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Performs a forward pass to get mu and log_std."""
        
        if state.ndim == 1:
            state = state[np.newaxis, :]
            
        self.h = np.dot(state, self.W1) + self.b1
        h_relu = np.maximum(0, self.h)
        
        mu = np.dot(h_relu, self.W_mu) + self.b_mu
        log_std = np.dot(h_relu, self.W_logstd) + self.b_logstd
        
        mu = np.tanh(mu) * self.action_limit
        log_std = np.clip(log_std, -20, 2) 
        
        return mu, log_std, h_relu 

    def get_action(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Samples an action from the policy distribution and returns its log probability."""
        
        mu, log_std, _ = self._forward(obs)
        std = np.exp(log_std)
        
        # 1. Sample action from Gaussian N(mu, std^2)
        action = np.random.normal(mu, std)
        
        # 2. Clip action to environment limits
        action = np.clip(action, -self.action_limit, self.action_limit)
        
        # 3. Calculate log probability of the sampled action
        log_prob = -0.5 * np.log(2.0 * np.pi) - log_std - 0.5 * ((action - mu) / std)**2
        
        return action.squeeze(), log_prob.squeeze()
    
    def _calculate_discounted_rewards(self, rewards):
        """Calculates the discounted cumulative reward G_t for each time step t."""
        G = np.zeros_like(rewards, dtype=np.float32)
        running_sum = 0
        
        for t in reversed(range(len(rewards))):
            running_sum = rewards[t] + self.gamma * running_sum
            G[t] = running_sum
            
        # Normalize the rewards (baseline subtraction) for stability
        G = (G - np.mean(G)) / (np.std(G) + 1e-8)
        return G

    def train_episode(self, episode_transitions: np.ndarray):
        """
        Performs the REINFORCE update using all transitions from one episode.
        """
        
        states = np.array(episode_transitions['state'])
        actions = np.array(episode_transitions['action'])
        rewards = np.array(episode_transitions['reward'])

        # 1. Calculate Discounted Cumulative Rewards (G_t)
        G = self._calculate_discounted_rewards(rewards)
        
        # 2. Forward pass to get policy parameters (needed for gradients)
        mu, log_std, h_relu = self._forward(states)
        std = np.exp(log_std)
        
        # 3. Calculate the Policy Gradient Scaling factors
        mu_error = (actions - mu) / (std**2 + 1e-8)
        log_std_error = ( ((actions - mu)**2 / (std**2 + 1e-8)) - 1 ) * 0.5 
        
        # Policy Gradient: G_t * Gradient of Log_Prob
        grad_mu = mu_error * G[:, np.newaxis]
        grad_logstd = log_std_error * G[:, np.newaxis]
        
        # --- SIMPLIFIED BACKPROPAGATION ---
        
        # 4. Update W_mu and W_logstd (Output Weights)
        dW_mu = np.dot(h_relu.T, grad_mu) / len(states)
        db_mu = np.mean(grad_mu, axis=0)
        
        dW_logstd = np.dot(h_relu.T, grad_logstd) / len(states)
        db_logstd = np.mean(grad_logstd, axis=0)

        self.W_mu += self.lr * dW_mu
        self.b_mu += self.lr * db_mu
        self.W_logstd += self.lr * dW_logstd
        self.b_logstd += self.lr * db_logstd
        
        # 5. Update W1 (Hidden Weights)
        error_mu = np.dot(grad_mu, self.W_mu.T)
        error_logstd = np.dot(grad_logstd, self.W_logstd.T)
        
        error_hidden = (error_mu + error_logstd) * (self.h > 0)
        
        dW1 = np.dot(states.T, error_hidden) / len(states)
        db1 = np.mean(error_hidden, axis=0)
        
        self.W1 += self.lr * dW1
        self.b1 += self.lr * db1
        
        return np.mean(np.abs(grad_mu)) + np.mean(np.abs(grad_logstd))