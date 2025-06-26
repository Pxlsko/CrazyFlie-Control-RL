"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

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
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.WaypointAviary import WaypointAviary
from gym_pybullet_drones.envs.RandomPointAviary import RandomPointAviary


from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1
DEFAULT_MA = False
ENVIRONMENT = RandomPointAviary
N_ENVS = 8                       # o los núcleos que quieras usar



def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):

    filename = os.path.join(output_folder, 'GeneralizePointv2')
    if not os.path.exists(filename):
        os.makedirs(filename+'/')
    
    # train_env = make_vec_env(ENVIRONMENT,
    #                             env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
    #                             n_envs=N_ENVS,
    #                             vec_env_cls=SubprocVecEnv, 
    #                             seed=0
    #                             )
    # eval_env = ENVIRONMENT(obs=DEFAULT_OBS, act=DEFAULT_ACT)

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
                tensorboard_log=filename+'/tb/',
                verbose=1)

    #### Target cumulative rewards (problem-dependent) ##########
    if DEFAULT_ACT == ActionType.ONE_D_RPM:
        target_reward = 474.15 
    else:
        target_reward = 667.
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
    model.learn(total_timesteps=int(1e7) if local else int(1e2), # shorter training in GitHub Actions pytest
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
