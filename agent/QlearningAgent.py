import numpy as np
import os
import torch

class QLearningAgent():
    def __init__(self, config):
        self.a_dim = config['num_actions']
        self.lr = config['learning_rate']
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.s_dim = config['num_states']
        self.Q = np.zeros((self.s_dim, self.a_dim))


    def select_action(self, state, deterministic):
        # 如果 state 是 numpy 数组(长度为80)
        if isinstance(state, np.ndarray):
            # 将 (80,) 的状态向量转换为整数索引
            s = int(np.argmax(state))
        else:
            # 如果 state 本身就是整数(根据实际情况使用)
            s = int(state)

        if deterministic:
            return np.argmax(self.Q[s, :])
        else:
            if np.random.uniform(0, 1) < self.epsilon:
                return np.random.randint(0, self.a_dim)
            else:
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
        Q_sa = self.Q[s, a]
        target_Q = r + (1 - dw) * self.gamma * np.max(self.Q[s_next, :])
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
        save_path = os.path.join(path, "{}_{}.pth".format(name, steps))
        # 使用 torch.save 存储 Q 表
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


