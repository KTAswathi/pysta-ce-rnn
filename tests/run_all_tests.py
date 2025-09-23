"""Code for running each of the test suites"""

import os
import pysta
os.chdir(f"{pysta.basedir}")
os.makedirs(f"{pysta.basedir}/figures/tests", exist_ok = True)

print("\nRunning all test!")

import test_envs

import test_agents

import test_training

