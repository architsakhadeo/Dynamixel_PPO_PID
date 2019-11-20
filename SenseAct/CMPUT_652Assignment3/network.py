import torch
from torch import nn
from torch.distributions.normal import Normal


# TODO: create your network here. Your network should inherit from nn.Module.
# It is recommended that your policy and value networks not share the same core network. This can be
# done easily within the same class or you can create separate classes.

class PPONetwork(nn.Module):
    def __init__(self, action_space, in_size):
        """
        Feel free to modify as you like.

        The policy should be parameterized by a normal distribution (torch.distributions.normal.Normal).
        To be clear your policy network should output the mean and stddev which are then fed into the Normal which
        can then be sampled. Care should be given to how your network outputs the stddev and how it is initialized.
        Hint: stddev should not be negative, but there are numerous ways to handle that. Large values of stddev will
        be problematic for learning.

        :param action_space: Action space of the environment. Gym action space. May have more than one action.
        :param in_size: Size of the input
        """
        super(PPONetwork, self).__init__()
        self.action_count = action_space.shape[0]
        self.in_size

        # TODO: fill me in

    def forward(self, inputs):
        # TODO: fill me in

