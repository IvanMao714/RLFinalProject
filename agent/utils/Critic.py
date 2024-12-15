import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim,net_width):

        """
        Initialize the Critic network.

        Args:
            state_dim (int): Dimension of the input state.
            net_width (int): Width of the hidden layers, i.e., the number of neurons in each hidden layer.
        """
        super(Critic, self).__init__()

        # Define the first hidden layer: maps the state input to a higher-dimensional space
        self.C1 = nn.Linear(state_dim, net_width)

        # Define the second hidden layer: further processes the feature representation
        self.C2 = nn.Linear(net_width, net_width)

        # Define the output layer: maps the features to a single scalar value (state value estimate)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):

        """
        Perform a forward pass through the Critic network to estimate the value of a given state.

        Args:
            state (torch.Tensor): Input state tensor with shape `(batch_size, state_dim)`.

        Returns:
            torch.Tensor: Scalar value (state value) for each input state, with shape `(batch_size, 1)`.
        """
        # Pass the input state through the first hidden layer with ReLU activation
        v = torch.relu(self.C1(state))

        # Pass through the second hidden layer with ReLU activation
        v = torch.relu(self.C2(v))

        # Pass through the output layer to produce the scalar state value
        v = self.C3(v)
        return v
