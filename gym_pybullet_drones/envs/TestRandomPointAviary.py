from turtle import distance
import numpy as np
import pybullet as p
from gymnasium import spaces   
import itertools                                          # ① NUEVO
from scipy.interpolate import CubicSpline

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class TestRandomPointAviary(BaseRLAviary):
    '''
    Test environment for RandomPointAviary policy.

    This environment includes: random point or points reach and hover, and trajectory following.
    '''

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 trajectory_type: str = "helix"  # "circle", "square", "random", "helix" or "random_and_hover"
                 ):
        
        self.NUM_WAYPOINTS = 16 # 1 for random, 7 for random_and_hover , 16 for circle and square
        self.RADIUS = 1.0 # Trajectories radius
        self.CENTER = np.array([1.5, 1.5, 1.5]) # Center of the trajectory

        self.trajectory_type = trajectory_type

        if self.trajectory_type == "circle":
            self.WAYPOINTS = self._generate_smooth_circle_waypoints(self.CENTER, self.RADIUS, self.NUM_WAYPOINTS, smooth_points=300)
        elif self.trajectory_type == "square":
            self.WAYPOINTS = self._generate_square_waypoints(self.CENTER, self.RADIUS, points_per_edge=35)
        elif self.trajectory_type in ["random", "random_and_hover"]:
            self.WAYPOINTS = self._generate_random_waypoints(self.CENTER, self.RADIUS,  self.NUM_WAYPOINTS)
        elif self.trajectory_type == "helix":
            self.WAYPOINTS = self._generate_helix_waypoints(self.CENTER, self.RADIUS, height=1.5, turns=2, num_points=100)
        else:
            raise ValueError(f"Trayectoria '{self.trajectory_type}' no soportada")

        ######################
        # For random_and_hover:
        self.HOVER_TIME_SEC = 5.0  # Hover time over each point in seconds
        self.hover_counter = 0     # Hover step counter
        self.in_hover = False      # Is in hover mode?
        ######################

        self.current_target_idx = 0
        self.TARGET_POS = self.WAYPOINTS[self.current_target_idx]
        self.EPISODE_LEN_SEC = 140 # 20 for random, 60 for random_and_hover, 150 for rectangular, 110 for circular, 140 for helix
        self.reached_errors = []

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    def _generate_square_waypoints(self, center, size, points_per_edge=10):
        '''
        Generates a square trajectory composed of more interpolated points between the corners.

        Parameters:
        - center: Center of the square.
        - size: Size of the square.
        - points_per_edge: Number of points per edge.
        '''
        half = size
        z = center[2]

        # Corners of the square
        corners = [
            [center[0] - half, center[1] - half, z],
            [center[0] - half, center[1] + half, z],
            [center[0] + half, center[1] + half, z],
            [center[0] + half, center[1] - half, z],
            [center[0] - half, center[1] - half, z],  # Returns to start to close the square
        ]

        # Interpolate points between corners
        waypoints = []
        for i in range(len(corners) - 1):
            start = np.array(corners[i])
            end = np.array(corners[i + 1])
            for t in np.linspace(0, 1, points_per_edge, endpoint=True):
                point = (1 - t) * start + t * end
                waypoints.append(np.round(point, 2))
        
        return waypoints

    
    def _generate_helix_waypoints(self, center, radius, height, turns, num_points):
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
        return waypoints

    def _update_target_marker(self):
        '''
        Creates or updates a translucent orange sphere at the current target position for random or random_and_hover trajectories.
        '''
        # Delete the previous marker if it exists
        if hasattr(self, "_target_marker_id") and self._target_marker_id is not None:
            try:
                p.removeBody(self._target_marker_id, physicsClientId=self.CLIENT)
            except Exception:
                pass
            self._target_marker_id = None

        # Only draw sphere for random trajectories
        if self.trajectory_type == "random" or self.trajectory_type == "random_and_hover":
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                rgbaColor=[1, 0.5, 0, 0.4], 
                radius=0.05,
                physicsClientId=self.CLIENT
            )
            self._target_marker_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=self.TARGET_POS,
                physicsClientId=self.CLIENT
            )
        else:
            self._target_marker_id = None
    
    def _generate_random_waypoints(self, center, radius, num_points):
        '''
        Generates random points within a cube centered at center.

        Parameters:
        - center: Center of the cube.
        - radius: Half the side length of the cube.
        - num_points: Number of random points to generate.
        '''

        waypoints = []
        for _ in range(num_points):
            point = center + np.random.uniform(-radius, radius, size=3)
            waypoints.append(np.round(point, 1))
        return waypoints

    def _generate_smooth_circle_waypoints(self, center, radius, num_points, smooth_points=100):
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
        # Close the circle for the spline
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        z = np.append(z, z[0])
        t = np.linspace(0, 1, num_points+1)
        t_smooth = np.linspace(0, 1, smooth_points)
        cs_x = CubicSpline(t, x, bc_type='periodic')
        cs_y = CubicSpline(t, y, bc_type='periodic')
        cs_z = CubicSpline(t, z, bc_type='periodic')
        waypoints = [
            np.round(np.array([cs_x(ti), cs_y(ti), cs_z(ti)], dtype=np.float32), 3)  
            for ti in t_smooth
        ]
        return waypoints
    
    def reset(self, seed=None, options=None):
        '''
        Resets the environment to its initial state.
        '''

        self.current_target_idx = 0
        self.TARGET_POS = self.WAYPOINTS[self.current_target_idx]
        self.reached_errors = []
        self.hover_counter = 0
        self.in_hover = False
        obs = super().reset(seed=seed, options=options)
        self._update_target_marker()
        return obs
    
    def _observationSpace(self):
        '''
        Defines the observation space of the environment.
        '''

        parent = super()._observationSpace()
        low  = np.hstack([parent.low[0],  [-np.inf]*3])
        high = np.hstack([parent.high[0], [ np.inf]*3])
        low  = np.tile(low,  (self.NUM_DRONES, 1))
        high = np.tile(high, (self.NUM_DRONES, 1))
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self):
        '''
        Adds to observation the distance to the target position.
        '''

        kin   = super()._computeObs()[0]
        delta = self.TARGET_POS - self._getDroneStateVector(0)[0:3]
        return np.hstack([kin, delta]).reshape(1, -1).astype('float32')
    
    def _computeReward(self):
        '''Computes the current reward value.

        Returns
        -------
        float
            The reward.

        '''
        state = self._getDroneStateVector(0)
        
        # Penalization parameters:
        pos_k = 0.05 # constant for position penalization
        ori_k = 0.002 # constant for orientation penalization
        lin_vel_k = 0.005 # constant for linear velocity penalization
        ang_vel_k = 0.002 # constant for angular velocity penalization
        act_k = 0.003 # constant for action penalization

        # Survival reward
        survival_r = 0.1 # constant for survival reward

        # Check if the drone is out of bounds or crashed
        if (abs(state[0]) > 3.5 or abs(state[1]) > 3.5 or state[2] > 3.5 or  
            abs(state[7]) > 0.6 or abs(state[8]) > 0.6 or state[2] <= 0.075):
            return -50
        
        # Check if the episode has timed out
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return -10
        
        # Desired position and orientation
        target_pos = self.TARGET_POS
        target_ori = np.array([0, 0, 0])  # Desired orientation (roll, pitch, yaw)

        # Desired velocities
        target_lin_vel = np.array([0, 0, 0])  # Desired linear velocity
        target_ang_vel = np.array([0, 0, 0])  # Desired angular velocity

        # Position penalization
        pos_r = -pos_k * np.linalg.norm(target_pos - state[0:3])**2

        # Orientation penalization
        ori_r = -ori_k * np.linalg.norm(target_ori - state[7:10])**2

        # Linear velocity penalization
        lin_vel_r = -lin_vel_k * np.linalg.norm(target_lin_vel - state[10:13])**2

        # Angular velocity penalization
        ang_vel_r = -ang_vel_k * np.linalg.norm(target_ang_vel - state[13:16])**2

        # Action penalization
        if len(self.action_buffer) > 1:
            last_action = self.action_buffer[-1][0]      # Actual last action
            prev_action = self.action_buffer[-2][0]      # Previous action
            jerk = last_action - prev_action
            jerk_r = -act_k * np.linalg.norm(jerk)**2    # Quadratic penalization
        else:
            jerk_r = 0

        # Target reached reward
        # If the drone is close enough to the target position and has low velocity, give a high reward based on the steps taken
        if np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.05 and np.linalg.norm(state[10:13]) < 0.1:
            max_reward = 100.0
            initial_dist = np.linalg.norm(self.TARGET_POS - np.array([0, 0, 0]))
            norm_steps = self.step_counter / (initial_dist + 1e-6)
            reward = max_reward * (1 - norm_steps / (self.PYB_FREQ * self.EPISODE_LEN_SEC))
            reward = max(reward, 0)
            return reward

        # Total reward calculation
        total_r = pos_r + ori_r + lin_vel_r + ang_vel_r + jerk_r + survival_r

        return total_r

    ################################################################################
    
    def _computeTerminated(self):
        '''
        Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        '''

        state = self._getDroneStateVector(0)

        # Check if the drone is close enough to the target position
        pos_error = np.linalg.norm(self.TARGET_POS - state[0:3])
        if self.trajectory_type == "random_and_hover":
                    if not self.in_hover:
                        if pos_error < 0.1:
                            print(f"[INFO] Target reached {self.current_target_idx}: {self.TARGET_POS}, inicia hover")
                            self.reached_errors.append(pos_error)
                            self.in_hover = True
                            self.hover_counter = 0
                        # Dont terminate episode, until hover is complete
                        return False
                    else:
                        # Already hovering
                        self.hover_counter += 1
                        if self.hover_counter >= int(self.HOVER_TIME_SEC * self.CTRL_FREQ):
                            # Hover finished, move to the next waypoint if any
                            self.in_hover = False
                            self.hover_counter = 0
                            if self.current_target_idx < len(self.WAYPOINTS) - 1:
                                self.current_target_idx += 1
                                self.TARGET_POS = self.WAYPOINTS[self.current_target_idx]
                                self._update_target_marker()
                            else:
                                # Last target reached, remove marker
                                if hasattr(self, "_target_marker_id") and self._target_marker_id is not None:
                                    try:
                                        p.removeBody(self._target_marker_id, physicsClientId=self.CLIENT)
                                    except Exception:
                                        pass
                                    self._target_marker_id = None
                            return False
                        else:
                            # Still hovering
                            return False

        # Original behavior for the rest of the trajectories
        pos_error = np.linalg.norm(self.TARGET_POS - state[0:3])
        if pos_error < 0.1:
            print(f"[INFO] Se alcanzó el waypoint {self.current_target_idx}: {self.TARGET_POS}")
            self.reached_errors.append(pos_error)
            if self.current_target_idx < len(self.WAYPOINTS) - 1:
                self.current_target_idx += 1
                self.TARGET_POS = self.WAYPOINTS[self.current_target_idx]
                self._update_target_marker()
            else:
                # Last target reached, remove marker
                if hasattr(self, "_target_marker_id") and self._target_marker_id is not None:
                    try:
                        p.removeBody(self._target_marker_id, physicsClientId=self.CLIENT)
                    except Exception:
                        pass
                    self._target_marker_id = None
            return False
        
        # Check if the drone is out of bounds or crashed
        elif (abs(state[0]) > 3.5 or abs(state[1]) > 3.5 or state[2] > 3.5 or  
            abs(state[7]) > 0.6 or abs(state[8]) > 0.6 or state[2] <= 0.075):
            # If the drone crashes or goes out of bounds, remove marker
            if hasattr(self, "_target_marker_id") and self._target_marker_id is not None:
                try:
                    p.removeBody(self._target_marker_id, physicsClientId=self.CLIENT)
                except Exception:
                    pass
                self._target_marker_id = None
            return True
        else:
            return False
    
    ################################################################################
    
    def _computeTruncated(self):
        '''
        Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        '''
        # Check if the episode has timed out
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        '''
        Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        '''
    
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
    
    def _draw_waypoints(self):
        '''
        Draw spheres for random and random_and_hover trajectories and lines for other types.
        '''

        if self.trajectory_type != "random" and self.trajectory_type != "random_and_hover":
            if hasattr(self, "_waypoint_line_ids"):
                for lid in self._waypoint_line_ids:
                    try:
                        p.removeUserDebugItem(lid, physicsClientId=self.CLIENT)
                    except Exception:
                        pass
            self._waypoint_line_ids = []
            for i in range(len(self.WAYPOINTS)):
                start = self.WAYPOINTS[i]
                end = self.WAYPOINTS[(i + 1) % len(self.WAYPOINTS)]
                line_id = p.addUserDebugLine(
                    lineFromXYZ=start,
                    lineToXYZ=end,
                    lineColorRGB=[0, 0, 1], 
                    lineWidth=2,
                    lifeTime=0,              
                    physicsClientId=self.CLIENT
                )
                self._waypoint_line_ids.append(line_id)