import unittest
import numpy as np
import torch
from ppo_agent import PPO

"""
python -m unittest discover
"""


class TestPPOAgent(unittest.TestCase):

    def test_return(self):
        # TODO: Run unit tests on your return.
        # How does it handle the end of an episode and the start of the next one?
        # How does it handle the end of the batch?
        # Does it handle gamma and lambda properly?


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPPOAgent)
    unittest.TextTestRunner(verbosity=2).run(suite)
