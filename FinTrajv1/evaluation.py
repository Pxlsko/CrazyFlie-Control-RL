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
import pybullet as p
import itertools


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.TestRandomPointAviary import TestRandomPointAviary
from gym_pybullet_drones.envs.TrajectoryAviary import TrajectoryAviary
from gym_pybullet_drones.envs.HoverRandomPointAviary import HoverRandomPointAviary

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
ENVIRONMENT = TestRandomPointAviary


def _draw_target_line(env, drone_pos):
    """Dibuja (y renueva) una línea roja del dron al objetivo."""
    p.addUserDebugLine(
        lineFromXYZ=drone_pos,
        lineToXYZ=env.TARGET_POS,
        lineColorRGB=[1, 0, 0],
        lineWidth=2,
        lifeTime=1 / env.CTRL_FREQ,        
        physicsClientId=env.CLIENT
    )

def _draw_target_text(env):
    """Texto flotante con las coordenadas del objetivo (una sola vez por reset)."""
    if hasattr(env, "_text_uid"):
        p.removeUserDebugItem(env._text_uid, physicsClientId=env.CLIENT)
    env._text_uid = p.addUserDebugText(
        text=f"Target: {env.TARGET_POS.round(2)}",
        textPosition=env.TARGET_POS + [0, 0, 0.15],
        textColorRGB=[1, 0.6, 0],
        textSize=1.2,
        lifeTime=0,                         
        physicsClientId=env.CLIENT
    )

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):

    filename = os.path.join(output_folder + '/Trajectoryresultsv4',)
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    if local:
        input("Press Enter to continue...")

    if os.path.isfile(filename+'/best_model.zip'):
       path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    #### Show (and record a video of) the model's performance ##
    test_env = ENVIRONMENT(gui=gui,
                            obs=DEFAULT_OBS,
                            act=DEFAULT_ACT,
                            record=record_video)
    test_env_nogui = ENVIRONMENT(obs=DEFAULT_OBS, act=DEFAULT_ACT)

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=DEFAULT_AGENTS if multiagent else 1,
                output_folder=output_folder,
                colab=colab
                )

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    test_env._draw_waypoints()
    print("Target Position:", test_env.TARGET_POS)  # Imprime la posición objetivo

    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)

        obs2 = obs.squeeze()
        
        ############################
        drone_xyz = obs2[0:3]           # (x,y,z) del dron en este paso
        _draw_target_line(test_env, drone_xyz)
        ############################
        p.resetDebugVisualizerCamera(
            cameraDistance=2.5,           # Distancia de la cámara al dron
            cameraYaw=45,                 # Ángulo horizontal
            cameraPitch=-30,              # Ángulo vertical
            cameraTargetPosition=drone_xyz,
            physicsClientId=test_env.CLIENT
        )
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)

        logger.log(drone=0,
            timestamp=i/test_env.CTRL_FREQ,
            state=np.hstack([obs2[0:3],
                                np.zeros(4),
                                obs2[3:15],
                                act2
                                ]),
            control=np.zeros(12)
            )
            
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs, _ = test_env.reset(seed=42, options={})
            _draw_target_text(test_env)
            print("Target Position:", test_env.TARGET_POS)  # Imprime la posición objetivo al reiniciar

    test_env.close()
    if hasattr(test_env, "reached_errors") and len(test_env.reached_errors) > 0:
        errors = np.array(test_env.reached_errors)
        print(f"\n[RESULTADO] Precisión al llegar a los puntos:")
        print(f"  - Error medio: {errors.mean():.4f} m")
        print(f"  - Error máximo: {errors.max():.4f} m")
        print(f"  - Error mínimo: {errors.min():.4f} m")
    else:
        print("\n[RESULTADO] No se registraron llegadas a waypoints.")
    #############################################################
    # Validación: comprobar que el agente alcanza los 216 puntos del grid
    
    # grid = np.arange(0.5, 3.1, 0.5)
    # targets = np.array(list(itertools.product(grid, grid, grid)), dtype=np.float32)

    # print("\n[VALIDACIÓN] Comprobando que el agente alcanza los 216 puntos…")
    # successes = 0

    # val_env = RandomPointAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)   # sin GUI
    # max_steps = int(20 * val_env.CTRL_FREQ)      # 20 s de margen por punto

    # for tgt in targets:
    #     obs, _ = val_env.reset()
    #     val_env.TARGET_POS = tgt
    #     # Si existe el marcador, lo recolocamos (inútil sin GUI, pero por si acaso)
    #     if hasattr(val_env, "_update_target_marker"):
    #         val_env._update_target_marker()

    #     for _ in range(max_steps):
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, _, terminated, _, _ = val_env.step(action)
    #         if terminated:
    #             successes += 1
    #             break

    # val_env.close()
    # print(f"\nResultado: {successes}/216 puntos alcanzados "
    #     f"({successes/216:.1%} de éxito)\n")

    ################################################

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot(reference=test_env.WAYPOINTS)
if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))


