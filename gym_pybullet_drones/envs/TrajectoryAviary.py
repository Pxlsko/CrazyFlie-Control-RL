from turtle import distance
import numpy as np
import pybullet as p
from gymnasium import spaces   
import itertools                                   

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class TrajectoryAviary(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################

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
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.NUM_WAYPOINTS = 5
        self.TARGET_POS_LIST = [self._sample_grid_point() for _ in range(self.NUM_WAYPOINTS)]
        self.current_target_idx = 0
        self.TARGET_POS = self.TARGET_POS_LIST[self.current_target_idx]
        self.EPISODE_LEN_SEC = 15


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

    def reset(self, seed=None, options=None):
        self.current_target_idx = 0
        self.TARGET_POS_LIST = [self._sample_grid_point() for _ in range(self.NUM_WAYPOINTS)]
        self.TARGET_POS = self.TARGET_POS_LIST[self.current_target_idx]
        self.survival_reward = 0  # Reinicia la recompensa de supervivencia
        return super().reset(seed=seed, options=options)


    def _sample_grid_point(self, exclude=None, min_dist=1.5):
            GRID = np.arange(0.5, 3.1, 0.5)
            while True:
                point = np.array([np.random.choice(GRID) for _ in range(3)], dtype=np.float32)
                if exclude is None or np.linalg.norm(point - exclude) > min_dist:
                    return point
    def _next_waypoint(self):
        # Genera un nuevo punto suficientemente alejado del actual
        prev = self.TARGET_POS
        new_point = self._sample_grid_point(exclude=prev, min_dist=1.5)
        return new_point
    
    def _observationSpace(self):
        parent = super()._observationSpace()
        low  = np.hstack([parent.low[0],  [-np.inf]*3])
        high = np.hstack([parent.high[0], [ np.inf]*3])
        low  = np.tile(low,  (self.NUM_DRONES, 1))
        high = np.tile(high, (self.NUM_DRONES, 1))
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self):
        kin   = super()._computeObs()[0]
        delta = self.TARGET_POS - self._getDroneStateVector(0)[0:3]
        return np.hstack([kin, delta]).reshape(1, -1).astype('float32')
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        
        # Parámetros de penalización
        pos_k = 0.008
        ori_k = 0.002
        lin_vel_k = 0.004
        ang_vel_k = 0.002
        act_k = 0.002
        survival_r = 0.1      # Recompensa de supervivencia por paso
        step_penalty = -0.0005  # Penalización por cada paso

        # Posición deseada y orientación deseada
        target_pos = self.TARGET_POS
        target_ori = np.array([0, 0, 0])  # Orientación deseada (roll, pitch, yaw)

        # Velocidades deseadas
        target_lin_vel = np.array([0, 0, 0])  # Velocidad lineal deseada
        target_ang_vel = np.array([0, 0, 0])  # Velocidad angular deseada

        # Penalización por posición
        pos_r = -pos_k * np.linalg.norm(target_pos - state[0:3])**2

        # Penalización por orientación
        ori_r = -ori_k * np.linalg.norm(target_ori - state[7:10])**2

        # Penalización por velocidad lineal
        lin_vel_r = -lin_vel_k * np.linalg.norm(target_lin_vel - state[10:13])**2

        # Penalización por velocidad angular
        ang_vel_r = -ang_vel_k * np.linalg.norm(target_ang_vel - state[13:16])**2

        # Penalización por acciones tomadas
        if len(self.action_buffer) > 0:
            last_action = self.action_buffer[-1][0]  # Última acción tomada
            act_r = -act_k * np.linalg.norm(last_action)
        else:
            act_r = 0
        
        if (abs(state[0]) > 3.5 or abs(state[1]) > 3.5 or state[2] > 3.5 or  
            abs(state[7]) > 0.6 or abs(state[8]) > 0.6 or state[2] <= 0.075):
            return -100
        
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return -50
                
        # Check if reached current waypoint with low velocity
        pos_error = np.linalg.norm(self.TARGET_POS - state[0:3])
        vel_norm = np.linalg.norm(state[10:13])
        if pos_error < 0.1 and vel_norm < 0.1:
            # Reward increases with waypoint index
            reward = 60.0 * (self.current_target_idx + 1)
            self.survival_reward = 0  # Reinicia la recompensa de supervivencia
            return reward


        # Recompensa total
        total_r = pos_r + ori_r + lin_vel_r + ang_vel_r + act_r + step_penalty
        self.survival_reward += survival_r + step_penalty


        return total_r

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        pos_error = np.linalg.norm(self.TARGET_POS - state[0:3])
        vel_norm = np.linalg.norm(state[10:13])

        if pos_error < 0.1 and vel_norm < 0.1:
                if self.current_target_idx < self.NUM_WAYPOINTS - 1:
                    self.current_target_idx += 1
                    # Genera un nuevo punto alejado del anterior
                    self.TARGET_POS = self._sample_grid_point(exclude=self.TARGET_POS, min_dist=1.5)
                    self.TARGET_POS_LIST[self.current_target_idx] = self.TARGET_POS
                    print(f"[INFO] Waypoint {self.current_target_idx} alcanzado: {self.TARGET_POS}")
                    return False
                else:
                    print(f"[INFO] Todos los waypoints alcanzados.")
                    return True

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
        # state = self._getDroneStateVector(0)
        # if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
        #      or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        # ):
        #     return True
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
    