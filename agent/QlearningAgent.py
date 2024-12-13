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

    def encode_state(self, state):
        """
        将一个 (80,) 形状的状态向量映射为单一整数索引。
        假设 state 是 one-hot 编码的，即在80个位置中有一个位置为1，其余为0。
        则此位置的下标即为状态索引。

        Args:
            state (np.ndarray): 当前状态，形状为 (80,) 的一维数组（one-hot向量）。

        Returns:
            int: 离散整数索引。
        """
        # 检查state的形状
        if state.shape == (80,):
            # 使用argmax获取one-hot向量中值为1的索引
            return int(np.argmax(state))
        else:
            raise ValueError("State shape is not (80,). Received shape: {}".format(state.shape))

    # def select_action(self, s, deterministic):
    #     """
    #     Select an action based on Q-table.
    #     Args:
    #         s (int): current state index
    #         deterministic (bool): whether use deterministic (greedy) policy
    #     Returns:
    #         int: selected action
    #     """
    #     if deterministic:
    #         return np.argmax(self.Q[s, :])
    #     else:
    #         # epsilon-greedy policy
    #         if np.random.uniform(0, 1) < self.epsilon:
    #             return np.random.randint(0, self.a_dim)
    #         else:
    #             return np.argmax(self.Q[s, :])

    # def select_action(self, s, deterministic):
    #     s = int(s)  # 确保索引为整数
    #     if deterministic:
    #         return np.argmax(self.Q[s, :])
    #     else:
    #         if np.random.uniform(0, 1) < self.epsilon:
    #             return np.random.randint(0, self.a_dim)
    #         else:
    #             return np.argmax(self.Q[s, :])

    # def select_action(self, state, deterministic):
    #     """
    #     Q-learning 的 select_action 方法修改版本示例:
    #     1. 检查 state 类型。
    #     2. 如果 state 是 np.ndarray，则根据情况将其转换为整数索引。
    #     3. 然后使用 Q 表查找 Q 值并选择动作。
    #     """
    #
    #     # 如果状态是一个 numpy 数组并且代表一个单一离散状态（例如 [0]）
    #     # 可以尝试使用 state.item() 将其转换为 python 标量
    #     if isinstance(state, np.ndarray):
    #         # 如果是单元素数组，例如 [0]，转换为 int
    #         if state.size == 1:
    #             s = int(state.item())
    #         else:
    #             # 如果是多维状态，需要自定义映射函数
    #             # s = self.encode_state(state)  # 用户需要实现这一步，将多维状态映射到单一整数索引
    #             print(state.shape)
    #
    #     else:
    #         # 如果本身就是整数索引
    #         s = int(state)
    #
    #     # s 现在是一个整数索引，可用于查询 Q 表
    #     if deterministic:
    #         # 确定性策略，返回 Q 值最大的动作
    #         return np.argmax(self.Q[s, :])
    #     else:
    #         # epsilon-贪婪策略
    #         if np.random.uniform(0, 1) < self.epsilon:
    #             return np.random.randint(0, self.a_dim)
    #         else:
    #             return np.argmax(self.Q[s, :])
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
