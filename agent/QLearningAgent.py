import os

import numpy as np
import torch


class QLearningAgent:
    def __init__(self, config, memory):
        self.a_dim = config['num_actions']
        self.lr = config['learning_rate']
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.s_dim = config['num_states']
        self.Q = np.zeros((self.s_dim, self.a_dim))
        self.memory = memory
        self.batch_size = config['batch_size']

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

    def train(self):
        if self.memory.size_now() < self.batch_size:
            return
            # print(self.memory.sample(self.batch_size))
        states, actions, rewards, next_states = self.memory.get_sample_batch(self.batch_size)

        # Loop through each sample in the batch to update Q-values
        for state, action, reward, next_state in  zip(states, actions, rewards, next_states):
            # Get the current Q-value for the state-action pair
            print(state, action)
            Q_sa = self.Q[state, action]

            # Compute the target Q-value using the Bellman equation

            target_Q = reward + self.gamma * np.max(self.Q[next_state, :])

            # Update the Q-value for the state-action pair
            self.Q[state, action] += self.lr * (target_Q - Q_sa)

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
