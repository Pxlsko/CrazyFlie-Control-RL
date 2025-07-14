from turtle import distance
import numpy as np
import pybullet as p
from gymnasium import spaces   
import itertools                                         
from scipy.interpolate import CubicSpline

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class ConstantPerturbation(BaseRLAviary):
    '''
    Test environment for RandomPointAviary policy with constant perturbation.

    In this environment at some point in the episode a constant perturbation is applied
    to the drone, which is expected to reject it and return to the target position.
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
                 trajectory_type: str = "random" 
                 ):
        self.NUM_WAYPOINTS = 1
        self.RADIUS = 1.0
        self.CENTER = np.array([1.5, 1.5, 1.5])

        self.trajectory_type = trajectory_type
        self.WAYPOINTS = self._generate_random_waypoints(self.CENTER, self.RADIUS, self.NUM_WAYPOINTS)
     

        self.current_target_idx = 0
        self.TARGET_POS = self.WAYPOINTS[self.current_target_idx]
        self.EPISODE_LEN_SEC = 20
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
        
        # Perturbation parameters
        self.perturbation_applied = False
        self.PERTURBATION_FORCE = np.array([0.0, 0.0, 0.5])  # Perturbation force
        self.PERTURBATION_DURATION_STEPS = int(self.CTRL_FREQ * 0.5) # Duration of the perturbation in control steps
        self.perturbation_step_counter = 0
        self.PERTURBATION_TIME_SEC = 10.0 # Time in seconds to apply the perturbation
        self.perturbation_trigger_step = int(self.PERTURBATION_TIME_SEC * self.PYB_FREQ) # Simulation step for the perturbation
    
    
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
            waypoints.append(np.round(point, 2))
        return waypoints

    def reset(self, seed=None, options=None):
        '''
        Resets the environment to its initial state.
        '''
        self.current_target_idx = 0
        self.TARGET_POS = self.WAYPOINTS[self.current_target_idx]
        self.reached_errors = []
        self.perturbation_applied = False  # Reset perturbation flag
        self.perturbation_step_counter = 0 # Reset perturbation step counter
        return super().reset(seed=seed, options=options)
    
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
        state = self._getDroneStateVector(0)
        pos_error = np.linalg.norm(self.TARGET_POS - state[0:3])

        # Perturbation initialization:
        if not self.perturbation_applied and self.step_counter >= self.perturbation_trigger_step:
            print(f"[INFO] Perturbation at {self.PERTURBATION_TIME_SEC} seconds!")
            self.perturbation_applied = True
            p.applyExternalForce(self.DRONE_IDS[0], 
                                 -1, 
                                 self.PERTURBATION_FORCE, 
                                 state[0:3], 
                                 p.WORLD_FRAME, 
                                 physicsClientId=self.CLIENT)
        # Keep perturbation for a while if already applied
        if self.perturbation_applied:
            self.perturbation_step_counter += 1
            if self.perturbation_step_counter >= self.PERTURBATION_DURATION_STEPS:
                self.perturbation_applied = False # Perturbation ends
                self.perturbation_step_counter = 0

        # Check if the drone is close enough to the target position
        if pos_error < 0.1:
            print(f"[INFO] Target reached {self.current_target_idx}: {self.TARGET_POS}")
            self.reached_errors.append(pos_error)
            
            if self.current_target_idx < len(self.WAYPOINTS) - 1:
                self.current_target_idx += 1
                self.TARGET_POS = self.WAYPOINTS[self.current_target_idx]
            return False 
        
        # Check if the drone is out of bounds or crashed
        elif (abs(state[0]) > 3.5 or abs(state[1]) > 3.5 or state[2] > 3.5 or
              abs(state[7]) > 0.6 or abs(state[8]) > 0.6 or state[2] <= 0.075):
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        # Check if the episode has timed out
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
    
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years