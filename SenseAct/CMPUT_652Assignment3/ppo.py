#!/usr/bin/env python

import argparse
import os
import torch
from torch.utils.tensorboard import SummaryWriter

from env import ReacherEnv
from ppo_agent import PPO

import time
import senseact.devices.dxl.dxl_utils as dxl


def main(cycle_time, idn, baud, port_str, batch_size, mini_batch_size, epoch_count, gamma, l, max_action, outdir,
         ep_time):
    """
    :param cycle_time: sense-act cycle time
    :param idn: dynamixel motor id
    :param baud: dynamixel baud
    :param batch_size: How many sample to record for each learning update
    :param mini_batch_size: How many samples to sample from each batch
    :param epoch_count: Number of epochs to train each batch on. Is this the number of mini-batches?
    :param gamma: Usual discount value
    :param l: lambda value for lambda returns.


    In the original paper PPO runs N agents each collecting T samples.
    I need to think about how environment resets are going to work. To calculate things correctly we'd technically
    need to run out the episodes to termination. How should we handle termination? We might want to have a max number
    of steps. In our setting we're going to be following a sine wave - I don't see any need to terminate then. So we
    don't need to run this in an episodic fashion, we'll do a continuing task. We'll collect a total of T samples and
    then do an update. I think I will implement the environment as a gym environment just to permit some
    interoperability. If there was an env that had a terminal then we would just track that terminal and reset the env
    and carry on collecting. Hmmm, actually I'm not sure how to think about this as a gym env. So SenseAct uses this
    RTRLBaseEnv, but I'm not sure I want to do that.

    So the changes listed from REINFORCE:
    1. Drop γ^t from the update, but not from G_t
    2. Batch Updates
    3. Multiple Epochs over the same batch
    4. Mini-batch updates
    5. Surrogate objective: - π_θ/π_θ_{old} * G_t
    6. Add Baseline
    7. Use λ-return: can you the real lambda returns or use generalized advantage estimation like they do in the paper.
    8. Normalize the advantage estimates: H = G^λ - v
    9. Proximity constraint:
        ρ = π_θ/π_θ_{old}
        objective:
        -min[ρΗ, clip(ρ, 1-ε, 1+ε)H]

    Also, there is the value function loss and there is an entropy bonus given.

    """

    tag = f"{time.time()}"
    summaries_dir = f"./summaries/{tag}"
    returns_dir = "./returns"
    networks_dir = "./networks"
    if outdir:
        summaries_dir = os.path.join(outdir, f"summaries/{tag}")
        returns_dir = os.path.join(outdir, "returns")
        networks_dir = os.path.join(outdir, "networks")

    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(returns_dir, exist_ok=True)
    os.makedirs(networks_dir, exist_ok=True)

    summary_writer = SummaryWriter(log_dir=summaries_dir)

    env = ReacherEnv(cycle_time, ep_time, dxl.get_driver(False), idn, port_str, baud, 100.0)
    obs_len = env.observation_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = None  # TODO: create your network
    network.to(device)

    agent = None  # TODO: create your agent

    # TODO: implement your main loop here. You will want to collect batches of transitions

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle_time", type=float, default=0.040, help="sense-act cycle time")
    parser.add_argument("--idn", type=int, default=1, help="Dynamixel ID")
    parser.add_argument("--baud", type=int, default=1000000, help="Dynamixel Baud")
    parser.add_argument("--port_str", type=str, default=None,
                        help="Default of None will use the first device it finds. Set this to override.")
    parser.add_argument("--batch_size", type=int, default=20000,
                        help="How many samples to record for each learning update")
    parser.add_argument("--mini_batch_div", type=int, default=32, help="Number of division to divide batch into")
    parser.add_argument("--epoch_count", type=int, default=10,
                        help="Number of times to train over the entire batch per update.")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount")
    parser.add_argument("--l", type=float, default=0.95, help="lambda for lambda return")
    parser.add_argument("--max_action", type=float, default=100.0,
                        help="The maximum value you will output to the motor. "
                             "This should be dependent on the control mode which you select.")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--ep_time", type=float, default=2.0, help="number of seconds to run for each episode.")

    args = parser.parse_args()
    main(**args.__dict__)
