"""

This script is used to train a reinforcement learning agent using the Stable Baselines3 library in a Gymnasium environment (RandomPointAviary).

-------
In a terminal, run as:

    $ python learn.py

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.RandomPointAviary import RandomPointAviary

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')

DEFAULT_AGENTS = 1 # Number of agents in the environment
DEFAULT_MA = False
ENVIRONMENT = RandomPointAviary # Environment to use
N_ENVS = 10 # Number of parallel environments                 

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):

    filename = os.path.join(output_folder, 'name') # Name of the folder where results will be saved inside output_folder, switch "name" to a specific name if desired
    if not os.path.exists(filename):
        os.makedirs(filename+'/')
    
    # Set up parallel environments
    train_env = make_vec_env(
        ENVIRONMENT,
        env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
        seed=0
    )
    
    eval_env = make_vec_env(           
        ENVIRONMENT,
        env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        seed=123
    )

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    model = PPO('MlpPolicy',
                train_env,
                n_steps=2048,
                batch_size=256,
                tensorboard_log=filename+'/tb/',
                verbose=1)

    #### Target cumulative rewards (problem-dependent) ##########
    target_reward = 100.0 # Target cumulative reward for stopping training
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    model.learn(total_timesteps=int(5e7) if local else int(1e2), # shorter training in GitHub Actions pytest
                callback=eval_callback,
                log_interval=100)

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
