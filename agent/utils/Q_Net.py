import torch
from torch import nn

<<<<<<< HEAD
from RLfinal.agent.utils.utils import build_net
=======
from agent.utils.utils import build_net
>>>>>>> 7ea8810 (Jinfan Xiang updated Q-learning model)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        """
        Standard Q-network with fully connected layers.
        """
        super(QNetwork, self).__init__()
        layers = [state_dim] + hidden_layers + [action_dim]
        self.q_layers = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state):
        """
        Forward pass to compute Q-values for all actions.
        """
        return self.q_layers(state)


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        """
        Dueling Q-network with separate streams for state-value and advantage.
        """
        super(DuelingQNetwork, self).__init__()
        layers = [state_dim] + hidden_layers
        self.shared_hidden = build_net(layers, nn.ReLU, nn.ReLU)
        self.value_stream = nn.Linear(hidden_layers[-1], 1)
        self.advantage_stream = nn.Linear(hidden_layers[-1], action_dim)

    def forward(self, state):
        """
        Forward pass to compute Q-values using dueling architecture.
        Q(s, a) = V(s) + A(s, a) - mean(A(s, a))
        """
        shared_representation = self.shared_hidden(state)
        state_value = self.value_stream(shared_representation)
        advantage = self.advantage_stream(shared_representation)
        q_values = state_value + (advantage - torch.mean(advantage, dim=-1, keepdim=True))
        return q_values

class Double_Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(Double_Q_Net, self).__init__()
        layers = [state_dim] + hidden_layers + [action_dim]

        self.Q1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        q1 = self.Q1(s)
        q2 = self.Q2(s)
        return q1,q2