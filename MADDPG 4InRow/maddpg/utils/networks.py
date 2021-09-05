import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tanh
from numpy import prod

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out


class MLPNetwork_CNN(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=False, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork_CNN, self).__init__()

        self.cnn1 = nn.Conv2d(2, 8, 2)
        self.cnn2 = nn.Conv2d(8, 8, 2)
        self.obs_dim_fl = prod(input_dim)
        self.cnn_out_dim = self.calc_input_size(input_dim)

        self.fc1 = nn.Linear(self.obs_dim_fl + self.cnn_out_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def calc_input_size(self, input_dim):
        m = self.cnn1(torch.zeros((1,) + input_dim))
        m = self.cnn2(m)
        return int(prod(m.size()))

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.cnn1(X))
        h1 = self.nonlin(self.cnn2(h1))
        h1 = torch.cat((X.reshape(-1, self.obs_dim_fl), h1.reshape(-1, self.cnn_out_dim)), dim=1)
        h1 = self.nonlin(self.fc1(h1), inplace=True)
        h1 = self.nonlin(self.fc2(h1), inplace=True)
        h1 = self.nonlin(self.fc3(h1), inplace=True)
        out = self.out_fn(self.fc4(h1))
        return out