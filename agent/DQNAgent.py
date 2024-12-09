import copy
import os.path

import numpy as np
import torch

import torch.nn.functional as F

from RLfinal.agent.utils.Q_Net import DuelingQNetwork, QNetwork


class DQNAgent(object):
    def __init__(self,config, memory):
        """
        Initialize the DQN agent with specific parameters.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_layer_size (int): Size of the hidden layers.
            device (torch.device): Device to run the computations on.
            learning_rate (float): Learning rate for the optimizer.
            exploration_noise (float): Noise level for exploration during action selection.
            batch_size (int): Batch size for training.
            discount_factor (float): Discount factor for future rewards (gamma).
            use_double_dqn (bool): Whether to use Double DQN.
            use_dueling_network (bool): Whether to use a dueling architecture for Q-network.
            replay_buffer (ReplayBuffer): Replay buffer for sampling experiences.
        """
        # Initialize parameters
        self.state_dim = config['num_states']
        self.action_dim = config['num_actions']
        self.hidden_layers = config['hidden_layers']
        self.device = config['device']
        self.learning_rate = config['learning_rate']
        self.exploration_noise = config['epsilon']
        self.batch_size = config['batch_size']
        self.discount_factor = config['gamma']
        self.use_double_dqn = config['use_double_dqn']
        self.use_dueling_network = config['use_dueling_network']
        # self.replay_buffer = replay_buffer
        self.memory = memory
        self.polyak_factor = 0.005  # Polyak averaging factor for target network updates

        # Initialize Q-network
        if self.use_dueling_network:
            self.q_network = DuelingQNetwork(self.state_dim, self.action_dim, self.hidden_layers).to(self.device)
        else:
            self.q_network = QNetwork(self.state_dim, self.action_dim, self.hidden_layers).to(self.device)

        # Optimizer for Q-network
        self.q_network_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Initialize target Q-network
        self.target_q_network = copy.deepcopy(self.q_network)
        for param in self.target_q_network.parameters():
            param.requires_grad = False  # Target network is frozen

    def select_action(self, state, deterministic):
        """
        Select an action based on the current policy.

        Args:
            state (np.ndarray): The current state.
            deterministic (bool): Whether to select actions deterministically.

        Returns:
            int: Selected action.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            if deterministic:
                action = self.q_network(state_tensor).argmax().item()
            else:
                if np.random.rand() < self.exploration_noise:
                    action = np.random.randint(0, self.action_dim)
                else:
                    action = self.q_network(state_tensor).argmax().item()
        return action

    def train(self):
        """
        Train the agent using data from the replay buffer.
        """
        # states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # print(self.batch_size, type(self.batch_size))
        # print(self.memory.size_now(), type(self.memory.size_now()))
        if self.memory.size_now() < self.batch_size:
            return
        # print(self.memory.sample(self.batch_size))
        states, actions, rewards, next_states= self.memory.get_sample_batch(self.batch_size)

        # Compute the target Q-value
        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.q_network(next_states).argmax(dim=1).unsqueeze(-1)
                max_next_q_values = self.target_q_network(next_states).gather(1, next_actions)
            else:
                max_next_q_values = self.target_q_network(next_states).max(1)[0].unsqueeze(1)

            target_q_values = rewards + self.discount_factor * max_next_q_values  # ~dones: not done

        # Compute the current Q-values
        current_q_values = self.q_network(states).gather(1, actions)

        # Calculate the loss
        q_loss = F.mse_loss(current_q_values, target_q_values)
        self.q_network_optimizer.zero_grad()
        q_loss.backward()
        self.q_network_optimizer.step()

        # Update the target network using Polyak averaging
        for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()):
            target_param.data.copy_(self.polyak_factor * param.data + (1 - self.polyak_factor) * target_param.data)
        # print("1111111111111")

    def predict_one(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return q_values.cpu().detach().numpy()


    def save(self, name, steps, path):
        torch.save(self.q_network.state_dict(), path + "{}_{}.pth".format(name, steps))

    def load(self, name, steps, path):
        self.q_network.load_state_dict(
            torch.load(os.path.join(path,"{}_{}.pth".format(name, steps)), map_location=self.device))
        self.target_q_network.load_state_dict(
            torch.load(os.path.join(path,"{}_{}.pth".format(name, steps)), map_location=self.device))


