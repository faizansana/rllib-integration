#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""DQN Algorithm. Tested with CARLA.
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
from __future__ import print_function

import argparse
import os
import yaml

from stable_baselines3 import DQN

from rllib_integration.carla_env import CarlaEnv

from dqn_example.dqn_experiment import DQNExperiment

# Set the experiment to EXPERIMENT_CLASS so that it is passed to the configuration
EXPERIMENT_CLASS = DQNExperiment


def parse_config(file_path):
    """
    Parses the .yaml configuration file into a readable dictionary
    """
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config["env"] = CarlaEnv
        config["env_config"]["experiment"]["type"] = EXPERIMENT_CLASS

    return config


def train_algorithm():

    config = parse_config("dqn_example/dqn_config.yaml")

    # Run gym loop
    env = CarlaEnv(config["env_config"])

    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_example/tensorboard/", buffer_size=1000)
    model.learn(total_timesteps=100_000)


def main():
    # argparser = argparse.ArgumentParser(description=__doc__)
    # argparser.add_argument("configuration_file",
    #                        help="Configuration file (*.yaml)")

    # args = argparser.parse_args()
    config = parse_config("dqn_example/dqn_config.yaml")

    # Run gym loop
    env = CarlaEnv(config["env_config"])
    # env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")
        if done:
            env.reset()


if __name__ == '__main__':

    try:
        train_algorithm()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
