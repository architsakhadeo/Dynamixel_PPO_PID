import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

import senseact.devices.dxl.dxl_utils as dxl
from env import ReacherEnv
import math


class PIDReacherEnv(ReacherEnv):

    def __init__(self, cycle_time, ep_time, driver, idn, port_str, baud, max_action):
        super(PIDReacherEnv, self).__init__(cycle_time, ep_time, driver, idn, port_str, baud, max_action)
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=(3,))

    #def _make_observation(self, dxl_observation: dict):
    #    current_pos = dxl_observation["present_pos"]
    #    return np.array([current_pos, self.target])


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

    max_action = np.pi  # TODO:100
    use_ctypes = True  # TODO: False

    idn = args.idn
    env = PIDReacherEnv(args.cycle_time, args.ep_time, dxl.get_driver(use_ctypes), idn, args.port_str, args.baud, max_action)

    # TODO
    kp = 1
    ki = 0.0000001
    kd = 0.0000001

    returns = []
    rewards = []
    
    error_prior = 0
    integral = 0
    derivative = 0
    action_prior = 0
    for i in range(args.ep_count):
        print(i)
        start = time.time()
        flag = 0
        observation = env.reset()
        error = observation[2]

        g = 0.0  # return
        r = []
        # TODO: Run one episode of PID control
        while True:
            
            integral = integral + (error*env.cycle_time)
            derivative = derivative +(error)/env.cycle_time
            
            
            action = kp*error + ki*integral + kd*derivative # + bias
            print('Action ', action, end = '\t')
            #action = 1
            
            action = action + action_prior

            observation, reward, done = env.step(action)
            action_prior = action
            
            #error_prior = error
            error = observation[2]
            print('Observation ', observation, end='\t')
            
            #error_prior = error
            
            g += reward
            r.append(reward)
            print('Reward ', reward, end='\n')
            
            if flag == 1:
                break

            if done == True:
                flag = 1
                print('Done')
            
            #time.sleep(env.cycle_time)    

        returns.append(g)
        rewards.append(r)
        end = time.time()
        print(end-start)

    env.close()

    returns = np.array(returns)
    np.save('./returnsoverallepisodes.npy', returns)
    np.save('./rewardsoverallepisodes.npy', rewards)
    avg_return = returns.mean()

    print(f"Avg return: {avg_return}.")
