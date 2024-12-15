import numpy as np
import os
import torch

class QLearningAgent():
    def __init__(self, config):
        """
        Initialize the Q-learning agent with a Q-table and hyperparameters.

        Args:
            config (dict): Configuration dictionary containing:
                - num_actions (int): Number of possible actions.
                - learning_rate (float): Learning rate for Q-value updates.
                - gamma (float): Discount factor for future rewards.
                - epsilon (float): Exploration probability for epsilon-greedy policy.
                - num_states (int): Number of possible states.
        """
        self.a_dim = config['num_actions']
        self.lr = config['learning_rate']
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.s_dim = config['num_states']
        self.Q = np.zeros((self.s_dim, self.a_dim))


    def select_action(self, state, deterministic):
        """
        Select an action based on the current policy using epsilon-greedy or deterministic selection.

        Args:
            state (np.ndarray or int): The current state.
                - If numpy array, it should represent a one-hot encoded state.
                - If integer, it represents a state index.
            deterministic (bool): Whether to select actions deterministically.

        Returns:
            int: The selected action index.
        """
        # Convert the state to an integer index if it is a numpy array
        if isinstance(state, np.ndarray):
            
            s = int(np.argmax(state))
        else:
            
            s = int(state)

        if deterministic:
            # Deterministic policy: choose the action with the highest Q-value
            return np.argmax(self.Q[s, :])
        else:
            # Epsilon-greedy policy
            if np.random.uniform(0, 1) < self.epsilon:
                # Random action for exploration
                return np.random.randint(0, self.a_dim)
            else:
                # Greedy action based on the Q-table
                return np.argmax(self.Q[s, :])

    def train(self, s, a, r, s_next, dw):
        """
        Update Q-table using Q-learning update rule.
        Args:
            s (int): current state
            a (int): action taken
            r (float): reward received
            s_next (int): next state
            dw (bool): done flag (True if episode terminated)
        """
        Q_sa = self.Q[s, a] # Current Q-value for state-action pair
        # Target Q-value using the Bellman equation
        target_Q = r + (1 - dw) * self.gamma * np.max(self.Q[s_next, :])
        # Update the Q-value with the learning rate
        self.Q[s, a] += self.lr * (target_Q - Q_sa)

    def predict_one(self, state):
        """
        Return Q-values for a single given state.
        Args:
            state (int or array): state index
        Returns:
            np.ndarray: Q-values for all actions
        """
        if isinstance(state, int):
            return self.Q[state, :]
        else:
            return self.Q[state, :]

    def save(self, name, steps, path):
        """
        Save Q-table to a .pth file.
        Args:
            name (str): model name
            steps (int): number of training steps (or episodes)
            path (str): directory to save model
        """
        if not os.path.exists(path):
            os.makedirs(path)

        # Construct the file path and save the Q-table using PyTorch
        save_path = os.path.join(path, "{}_{}.pth".format(name, steps))
        
        torch.save(self.Q, save_path)

        print("Saved Q-table at {}".format(save_path))

    def load(self, name, steps, path):
        """
        Load Q-table from a .pth file.
        Args:
            name (str): model name
            steps (int): number of training steps (or episodes) to load
            path (str): directory where model is saved
        """
        load_path = os.path.join(path, "{}_{}.pth".format(name, steps))
        if os.path.exists(load_path):
            self.Q = torch.load(load_path)
            print("Loaded Q-table from {}".format(load_path))
        else:
            raise FileNotFoundError("No Q-table found at {}".format(load_path))


