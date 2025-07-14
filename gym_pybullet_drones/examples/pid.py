"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
You can select between a rectangular, a smooth circular or an helix trajectory by
changing the variable TRAJ_TYPE.

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 150 #  140 for helix, 110 for circle and 150 for rectangle
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def generate_smooth_circle_waypoints(center, radius, num_points, smooth_points):
    '''
    Generates smoothed points in a circle using a spline and rounds to 2 decimals.

    Parameters:
    - center: Center of the circle.
    - radius: Radius of the circle.
    - num_points: Number of points to generate.
    - smooth_points: Number of points for the smoothed trajectory.
    '''    
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    z = np.full_like(x, center[2])
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    z = np.append(z, z[0])
    t = np.linspace(0, 1, num_points+1)
    t_smooth = np.linspace(0, 1, smooth_points)
    cs_x = CubicSpline(t, x, bc_type='periodic')
    cs_y = CubicSpline(t, y, bc_type='periodic')
    cs_z = CubicSpline(t, z, bc_type='periodic')
    waypoints = [
        np.round(np.array([cs_x(ti), cs_y(ti), cs_z(ti)], dtype=np.float32), 2)  
        for ti in t_smooth
    ]
    return np.array(waypoints)

def generate_helix_waypoints(center, radius, height, turns, num_points):
    '''
    Generates points of a helical trajectory.

    Parameters:
    - center: Center of the helix.
    - radius: Horizontal radius of the helix.
    - height: Total height it reaches.
    - turns: Number of turns.
    - num_points: Total number of points of the helix.
    '''
    t = np.linspace(0, 2 * np.pi * turns, num_points)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    z = center[2] + np.linspace(0, height, num_points)
    waypoints = [np.round(np.array([xi, yi, zi], dtype=np.float32), 2) for xi, yi, zi in zip(x, y, z)]
    return np.array(waypoints)

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    #### Select trajectory type ################################
    TRAJ_TYPE = "rectangle"  # "rectangle", "circle", "helix"

    NUM_WAYPOINTS = 16
    RADIUS = 1.0
    CENTER = np.array([1.5, 1.5, 1.5])
    NUM_WP = control_freq_hz * duration_sec

    if TRAJ_TYPE == "rectangle":
        RECTANGLE_POINTS = [
            np.array([0.5, 0.5, 1.5]),
            np.array([0.5, 2.5, 1.5]),
            np.array([2.5, 2.5, 1.5]),
            np.array([2.5, 0.5, 1.5]),
            np.array([0.5, 0.5, 1.5])  
        ]
        NUM_SEGMENTS = len(RECTANGLE_POINTS) - 1
        wps_per_segment = NUM_WP // NUM_SEGMENTS
        TARGET_POS = np.zeros((NUM_WP, 3))
        for i in range(NUM_SEGMENTS):
            start = RECTANGLE_POINTS[i]
            end = RECTANGLE_POINTS[i + 1]
            for j in range(wps_per_segment):
                alpha = j / wps_per_segment
                idx = i * wps_per_segment + j
                TARGET_POS[idx] = (1 - alpha) * start + alpha * end
        for k in range(NUM_SEGMENTS * wps_per_segment, NUM_WP):
            TARGET_POS[k] = RECTANGLE_POINTS[-1]
        wp_counters = np.zeros(num_drones, dtype=int)
        H = .1
        H_STEP = .05
        R = .3
        INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2),
                               R*np.sin((i/6)*2*np.pi+np.pi/2)-R,
                               H+i*H_STEP] for i in range(num_drones)])
    elif TRAJ_TYPE == "circle":
        TARGET_POS = generate_smooth_circle_waypoints(CENTER, RADIUS, NUM_WAYPOINTS, NUM_WP)
        wp_counters = np.zeros(num_drones, dtype=int)
        H = .1
        H_STEP = .05
        R = .3
        INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2),
                               R*np.sin((i/6)*2*np.pi+np.pi/2)-R,
                               H+i*H_STEP] for i in range(num_drones)])
    elif TRAJ_TYPE == "helix":
        HELIX_HEIGHT = 1.5
        TURNS = 2
        N_HELIX = int(NUM_WP * 0.7) 
        traj_points = generate_helix_waypoints(CENTER, RADIUS, HELIX_HEIGHT, TURNS, N_HELIX)
        TARGET_POS = np.zeros((NUM_WP, 3))
        TARGET_POS[:N_HELIX, :] = traj_points
        TARGET_POS[N_HELIX:, :] = traj_points[-1]
        wp_counters = np.zeros(num_drones, dtype=int)
        INIT_XYZS = np.array([TARGET_POS[0] for i in range(num_drones)])
    else:
        raise ValueError("TRAJ_TYPE must be 'rectangle', 'circle' or 'helix'.")

    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                    state=obs[j],
                                                                    target_pos=TARGET_POS[wp_counters[j]],
                                                                    target_rpy=INIT_RPYS[j, :]
                                                                    )

        #### Go to the next way point and loop #####################
        for j in range(num_drones):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(
                        drone=j,
                        timestamp=i/env.CTRL_FREQ,
                        state=obs[j],
                        control = np.hstack([
                            TARGET_POS[wp_counters[j]],      # x, y, z deseado
                            INIT_RPYS[j, :],                 # orientación deseada
                            np.zeros(6)                      # velocidades
                        ]),
                        target_pos=TARGET_POS[wp_counters[j]] 
                    )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot_custom()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Trajectory flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))