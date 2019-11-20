import argparse

import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

import senseact.devices.dxl.dxl_utils as dxl
from env import ReacherEnv


class PIDReacherEnv(ReacherEnv):

    def __init__(self, cycle_time, ep_time, driver, idn, port_str, baud, max_action):
        super(PIDReacherEnv, self).__init__(cycle_time, ep_time, driver, idn, port_str, baud, max_action)
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=(2,))

    def _make_observation(self, dxl_observation: dict):
        current_pos = dxl_observation["present_pos"]
        return np.array([current_pos, self.target])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle_time", type=float, default=0.040, help="cycle time")
    parser.add_argument("--ep_time", type=float, default=2.0, help="number of seconds to run for each episode.")
    parser.add_argument("--idn", type=int, default=1, help="Dynamixel ID")
    parser.add_argument("--baud", type=int, default=1000000, help="Dynamixel Baud")
    parser.add_argument("--port_str", type=str, default=None,
                        help="Default of None will use the first device it finds. Set this to override.")
    parser.add_argument("--ep_count", type=int, default=100, help="How many episodes to average returns over.")

    args = parser.parse_args()

    max_action = 100  # TODO:100
    use_ctypes = True  # TODO: False

    idn = args.idn
    env = PIDReacherEnv(args.cycle_time, args.ep_time, dxl.get_driver(use_ctypes), idn, args.port_str, args.baud,
                        max_action)

    # TODO
    kp = 1.2
    ki = 1.0
    kd = 0.001

    returns = []
    
    error_prior = 0
    integral = 0
    derivative = 0
    flag = 0
    for i in range(args.ep_count):
        observation = env.reset()
        g = 0.0  # return
        # TODO: Run one episode of PID control
        while True:
            error = observation[2]

            integral = integral + error
            derivative = error - error_prior
            
            action = kp*error + ki*integral + kd*derivative # + bias
            
            observation, reward, done = env.step(action)
            error_prior = error
            
            g += reward
            
            if flag == 1:
                break

            if done == True:
                flag = 1
                

        returns.append(g)

    env.close()

    returns = np.array(returns)
    avg_return = returns.mean()

    print(f"Avg return: {avg_return}.")
