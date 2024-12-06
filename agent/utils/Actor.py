import torch
import torch.nn.functional as F
import torch.nn as nn

from RLfinal.agent.utils.utils import build_net


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(Actor, self).__init__()

        # self.l1 = nn.Linear(state_dim, net_width)
        # self.l2 = nn.Linear(net_width, net_width)
        # self.l3 = nn.Linear(net_width, action_dim)


    def forward(self, state):
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n

    def pi(self, state, softmax_dim = 0):
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        return prob

class SAD_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(SAD_Actor, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.P = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        logits = self.P(s)
        probs = F.softmax(logits, dim=1)
        return probs