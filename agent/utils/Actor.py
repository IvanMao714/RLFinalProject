import torch
import torch.nn.functional as F
import torch.nn as nn


from agent.utils.utils import build_net


# Actor class for policy-based reinforcement learning
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):

        """
        Initialize the Actor network.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the output action.
            hidden_layers (list): List defining the size of each hidden layer.
        """
        super(Actor, self).__init__()

        # Placeholder for a feed-forward network with linear layers
        # Uncomment the following lines to define a specific architecture if needed
        # self.l1 = nn.Linear(state_dim, net_width)
        # self.l2 = nn.Linear(net_width, net_width)
        # self.l3 = nn.Linear(net_width, action_dim)


    def forward(self, state):
        """
        Forward pass of the network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Output of the final hidden layer.
        """
        # Pass the state through hidden layers with Tanh activation
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n

    def pi(self, state, softmax_dim = 0):

        """
        Compute the policy probabilities for the given state.

        Args:
            state (torch.Tensor): Input state tensor.
            softmax_dim (int): Dimension along which to apply the softmax function.

        Returns:
            torch.Tensor: Action probabilities computed from the policy.
        """
        # Pass the state through the network
        n = self.forward(state)

        # Compute action probabilities using softmax
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        return prob

# Stochastic Actor Design (SAD) Actor class with customizable hidden layers
class SAD_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        """
        Initialize the SAD_Actor network.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the output action.
            hid_shape (list): List defining the sizes of the hidden layers.
        """
        super(SAD_Actor, self).__init__()

        # Build a fully connected feed-forward network using the utility function
        # Layers are defined as [input_dim] -> hidden_layers -> [output_dim]
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.P = build_net(layers, nn.ReLU, nn.Identity) # ReLU for hidden layers, Identity for output layer

    def forward(self, s):

        """
        Forward pass of the SAD_Actor network.

        Args:
            s (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Action probabilities computed using softmax.
        """
        # Compute logits (unnormalized probabilities) using the network
        logits = self.P(s)

        # Apply softmax to convert logits into probabilities
        probs = F.softmax(logits, dim=1)
        return probs
