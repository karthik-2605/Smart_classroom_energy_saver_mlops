"""
agent.py
========
Q-Learning Agent for the Smart Classroom RL system.

Implements:
  - Q-table initialisation
  - Epsilon-greedy action selection
  - Q-value update (Bellman equation)
  - Model save / load (pickle)
"""

import numpy as np
import pickle
import os


class QLearningAgent:
    """
    Tabular Q-Learning agent with epsilon-greedy exploration.

    The Q-table maps (state_index, action) -> expected future reward.
    Over many episodes the agent learns which device combination
    minimises energy waste while keeping students comfortable.

    Attributes:
        state_size  (int)  : Total number of discrete states.
        num_actions (int)  : Number of possible actions (16).
        lr          (float): Learning rate alpha.
        gamma       (float): Discount factor.
        epsilon     (float): Current exploration rate.
        epsilon_min (float): Minimum epsilon after decay.
        epsilon_decay(float): Multiplicative decay per episode.
        q_table     (ndarray): Shape (state_size, num_actions).
    """

    def __init__(self, state_size, num_actions,
                 learning_rate=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        """
        Initialise the Q-Learning agent.

        Args:
            state_size   (int)  : Number of distinct states.
            num_actions  (int)  : Number of actions available.
            learning_rate(float): Alpha for Q-update.
            gamma        (float): Discount factor for future rewards.
            epsilon      (float): Starting exploration probability.
            epsilon_min  (float): Floor for epsilon.
            epsilon_decay(float): Per-episode decay multiplier.
        """
        self.state_size   = state_size
        self.num_actions  = num_actions
        self.lr           = learning_rate
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.epsilon_min  = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialise Q-table with small random values to break symmetry
        self.q_table = np.random.uniform(low=-0.01, high=0.01,
                                         size=(state_size, num_actions))

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state_index):
        """
        Epsilon-greedy action selection.

        With probability epsilon -> explore (random action).
        With probability 1-epsilon -> exploit (best known action).

        Args:
            state_index (int): Flat index of current state.

        Returns:
            int: Chosen action (0-15).
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)   # explore
        return int(np.argmax(self.q_table[state_index])) # exploit

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------

    def learn(self, state_idx, action, reward, next_state_idx, done):
        """
        Update Q-table using the Bellman equation:
            Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]

        Args:
            state_idx      (int)  : Index of current state.
            action         (int)  : Action taken.
            reward         (float): Reward received.
            next_state_idx (int)  : Index of next state.
            done           (bool) : Whether episode ended.
        """
        current_q = self.q_table[state_idx, action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state_idx])

        # Bellman update
        self.q_table[state_idx, action] += self.lr * (target_q - current_q)

    def decay_epsilon(self):
        """Decay exploration rate after each episode (floor at epsilon_min)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath):
        """
        Save the Q-table and hyperparameters to a pickle file.

        Args:
            filepath (str): Destination path (e.g. 'models/q_table.pkl').
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "q_table":       self.q_table,
            "epsilon":       self.epsilon,
            "lr":            self.lr,
            "gamma":         self.gamma,
            "state_size":    self.state_size,
            "num_actions":   self.num_actions,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"[Agent] Model saved to {filepath}")

    def load(self, filepath):
        """
        Load Q-table and hyperparameters from a pickle file.

        Args:
            filepath (str): Source path.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table     = data["q_table"]
        self.epsilon     = data["epsilon"]
        self.lr          = data["lr"]
        self.gamma       = data["gamma"]
        self.state_size  = data["state_size"]
        self.num_actions = data["num_actions"]
        print(f"[Agent] Model loaded from {filepath}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_policy_summary(self):
        """
        Return best action per state as a summary string.

        Returns:
            str: Human-readable policy table (first 10 states).
        """
        lines = ["State Index | Best Action | Max Q-Value"]
        lines.append("-" * 42)
        for s in range(min(10, self.state_size)):
            best_a = int(np.argmax(self.q_table[s]))
            max_q  = round(float(np.max(self.q_table[s])), 4)
            lines.append(f"{s:>11} | {best_a:>11} | {max_q:>11}")
        return "\n".join(lines)
