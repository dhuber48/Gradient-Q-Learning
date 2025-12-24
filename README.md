# Accelerated VDBE

Please see the pdf of my paper for an in-depth technical explanation of this project.

## Abstract

> This project is called **Accelerated VDBE**, a novel modification to the VDBE algorithm [1] for Q-learning. An agent running VDBE uses the absolute value of temporal difference, which is the difference between currently observed and previously estimated Q-values, to update its exploration rate, $\varepsilon$, at every time step. Accelerated VDBE injects the derivative of a normalized temporal difference function with respect to temporal difference into the update equation for $\varepsilon$. Testing on Farama Gymnasium's cartpole environment [3] shows that Accelerated VDBE outperforms original VDBE and normal Q-learning, by learning to keep a pole upright at least 13\% faster.

---

## How To Run My Code

### Testing Individual Agents
To test my three individual agents, run the `(agent name)_cartpoleenv.py` files. 
* These execute one training cycle of their respective agents and will plot a graph of the agent's performance (y-axis) versus episode number (x-axis). 
* After exiting the graph, an animation will appear of the trained agent operating the cartpole. 

**Configuration:** If you want to change the average number of episodes or solving window required to consider an agent, you can do that at the top of the `(agent name)_cartpoleenv.py` files (except for normal Q-learning, which runs for a set number of episodes, which can also be changed at the top of the `normal_cartpoleenv.py` script).

### File Descriptions

* **`(agent name)_cartPoleAgent.py`**
    These are the code for the individual agents. Running them individually will not do anything. 

* **`benchmarking.py`**
    This file tests all of my agents 10 times each, which four different scaling parameters for each version of VBDE. This script runs the tests I used for my analysis. Please note that this script can take a while to run.

* **`epsilon_series.py`**
    This file runs two simulations to help visualize how exploration rate evolves over time in Accelerated VDBE and normal VDBE.

---

## References

1.  Tokic, Michel, and Palm, Gunther. Value-Difference Based Exploration: Adaptive Control between Epsilon-Greedy and Softmax. KI’11: Proceedings of the 34th Annual German conference on Advances in artificial intelligence, 4 Oct. 2011. https://www.tokic.com/www/tokicm/publikationen/papers/KI2011.pdf
2.  Sutton, R. S., & Barto, A. G. (2021). Reinforcement Learning an Introduction. MTM.
3.  Farama Foundation. (2025). Gymnasium documentation. Cart Pole -Gymnasium Documentation. https://gymnasium.farama.org/environments/
