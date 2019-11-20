"""
Place your PPO agent code in here.
"""

import torch
from torch.utils.tensorboard import SummaryWriter


class PPO:
    def __init__(self,
                 device,  # cpu or cuda
                 network,  # your network
                 state_size,  # size of your state vector
                 batch_size,  # size of batch
                 mini_batch_div,
                 epoch_count,
                 gamma=0.99,  # discounting
                 l=0.95,  # lambda used in lambda-return
                 eps=0.2,  # epsilon value used in PPO clipping
                 summary_writer: SummaryWriter = None):
        self.device = device

        self.batch_size = batch_size
        self.mini_batch_div = mini_batch_div
        self.epoch_count = epoch_count
        self.gamma = gamma
        self.l = l
        self.eps = eps
        self.summary_writer = summary_writer

        self.state_size = state_size
        self.network = network
        self.optimizer = torch.optim.Adam(self.network.parameters())

    def step(self, state, r, terminal):
        """
        You will need some step function which returns the action.
        This is where I saved my transition data in my own code.
        :param state:
        :param r:
        :param terminal:
        :return:
        """

    @staticmethod
    def compute_return(r_buffer, v_buffer, t_buffer, l, gamma):
        """

        Compute the return. Unit test this function

        :param r_buffer: rewards
        :param v_buffer: values
        :param t_buffer: terminal
        :param l: lambda value
        :param gamma: gamma value
        :return:  the return
        """
        raise NotImplementedError

    def compute_advantage(self, g, v):
        """
        Compute the advantage
        :return: the advantage
        """
        raise NotImplementedError

    def compute_rho(self, actions, old_pi, new_pi):
        """
        Compute the ratio between old and new pi
        :param actions:
        :param old_pi:
        :param new_pi:
        :return:
        """
        raise NotImplementedError

    def learn(self, t):
        """
        Here's where you should do your learning and logging.
        :param t: The total number of transitions observed.
        :return:
        """
        raise NotImplementedError
