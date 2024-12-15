import torch
from torch import nn


from agent.utils.utils import build_net



class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
         """
        Initialize a standard Q-network with fully connected layers.

        Args:
            state_dim (int): Dimension of the input state space.
            action_dim (int): Dimension of the output action space.
            hidden_layers (list): List defining the size of each hidden layer.
        """
        super(QNetwork, self).__init__()

        # Create a fully connected neural network using build_net utility.
        # Layers: [state_dim] -> hidden_layers -> [action_dim]
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
        Initialize a Dueling Q-network with shared layers for state representation.

        Args:
            state_dim (int): Dimension of the input state space.
            action_dim (int): Dimension of the output action space.
            hidden_layers (list): List defining the size of each hidden layer.
        """
        super(DuelingQNetwork, self).__init__()
        # Shared layers for feature extraction
        layers = [state_dim] + hidden_layers
        self.shared_hidden = build_net(layers, nn.ReLU, nn.ReLU)
        # Separate streams for state-value and advantage
        self.value_stream = nn.Linear(hidden_layers[-1], 1) # Outputs a single value V(s)
        self.advantage_stream = nn.Linear(hidden_layers[-1], action_dim) # Outputs advantage values A(s, a)

    def forward(self, state):
        """
        Perform a forward pass through the Dueling Q-network.

        Args:
            state (torch.Tensor): Input state tensor with shape (batch_size, state_dim).

        Returns:
            torch.Tensor: Q-values for all actions with shape (batch_size, action_dim).

        Note:
            Q(s, a) = V(s) + A(s, a) - mean(A(s, a))
        """
        # Extract shared representation of the state
        shared_representation = self.shared_hidden(state)

        # Compute state value and advantages
        state_value = self.value_stream(shared_representation) # V(s)
        advantage = self.advantage_stream(shared_representation) # A(s, a)

        # Combine state value and advantage to compute Q-values
        q_values = state_value + (advantage - torch.mean(advantage, dim=-1, keepdim=True))
        return q_values

class Double_Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        """
        Initialize a Double Q-network with two independent Q-functions.

        Args:
            state_dim (int): Dimension of the input state space.
            action_dim (int): Dimension of the output action space.
            hidden_layers (list): List defining the size of each hidden layer.
        """
        super(Double_Q_Net, self).__init__()

        # Define two independent Q-networks
        layers = [state_dim] + hidden_layers + [action_dim]

        self.Q1 = build_net(layers, nn.ReLU, nn.Identity) # First Q-network
        self.Q2 = build_net(layers, nn.ReLU, nn.Identity) # Second Q-network

    def forward(self, s):
        """
        Perform a forward pass through both Q-networks.

        Args:
            s (torch.Tensor): Input state tensor with shape (batch_size, state_dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - Q-values from the first Q-network (batch_size, action_dim).
                - Q-values from the second Q-network (batch_size, action_dim).
        """
        q1 = self.Q1(s) # Compute Q-values from the first Q-network
        q2 = self.Q2(s) # Compute Q-values from the second Q-network
        return q1,q2
