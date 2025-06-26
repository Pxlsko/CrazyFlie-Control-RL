from turtle import distance
import numpy as np
import pybullet as p
from gymnasium import spaces   
import itertools                                          # ① NUEVO
from scipy.interpolate import CubicSpline

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class TestRandomPointAviary(BaseRLAviary):
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
        self.NUM_WAYPOINTS = 16  # Número de puntos en el círculo
        self.RADIUS = 1.0        # Radio del círculo
        self.CENTER = np.array([1.5, 1.5, 1.5])  # Centro del círculo (ajusta a tu escenario)
        self.WAYPOINTS = self._generate_smooth_circle_waypoints(self.CENTER, self.RADIUS, self.NUM_WAYPOINTS, smooth_points=100)       
        self.current_target_idx = 0
        self.TARGET_POS = self.WAYPOINTS[self.current_target_idx]
        # self.TARGET_POS = self._sample_grid_point()           
        self.EPISODE_LEN_SEC = 60
        self.reached_errors = []  # Guarda el error de cada waypoint alcanzado



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

    def _generate_smooth_circle_waypoints(self, center, radius, num_points, smooth_points=100):
        """Genera puntos suavizados en un círculo usando una spline."""
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        z = np.full_like(x, center[2])
        # Cierra el círculo para la spline
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        z = np.append(z, z[0])
        t = np.linspace(0, 1, num_points+1)
        t_smooth = np.linspace(0, 1, smooth_points)
        cs_x = CubicSpline(t, x, bc_type='periodic')
        cs_y = CubicSpline(t, y, bc_type='periodic')
        cs_z = CubicSpline(t, z, bc_type='periodic')
        waypoints = [np.array([cs_x(ti), cs_y(ti), cs_z(ti)], dtype=np.float32) for ti in t_smooth]
        return waypoints
    
    def reset(self, seed=None, options=None):
            self.current_target_idx = 0
            self.TARGET_POS = self.WAYPOINTS[self.current_target_idx]
            self.reached_errors = []  # Reinicia los errores al empezar episodio
            return super().reset(seed=seed, options=options)
    
    def _sample_grid_point(self):
        GRID = np.arange(0.5, 3.1, 0.1)                           #   [0.5,1,1.5,2,2.5,3]

        """Devuelve un XYZ aleatorio de la rejilla 0.5 – 3 m."""
        return np.array([np.random.choice(GRID) for _ in range(3)], dtype=np.float32)
    
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
        pos_k = 0.01
        ori_k = 0.002
        lin_vel_k = 0.002
        ang_vel_k = 0.002
        act_k = 0.004
        survival_r = 0.1  # Recompensa por supervivencia

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

        if np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.1 and np.linalg.norm(state[10:13]) < 0.1:

            return 100.0


        # Recompensa total
        total_r = pos_r + ori_r + lin_vel_r + ang_vel_r + act_r + survival_r

    

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
        # if np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.15:
        #     print(f"[INFO] Se alcanzó el waypoint {self.current_target_idx}: {self.TARGET_POS}")
        #     self.current_target_idx += 1
        #     if self.current_target_idx >= len(self.WAYPOINTS):
        #         print("[INFO] Trayectoria circular completada.")
        #         return True  # Termina el episodio tras completar la vuelta
        #     self.TARGET_POS = self.WAYPOINTS[self.current_target_idx]
        #     return False
        pos_error = np.linalg.norm(self.TARGET_POS - state[0:3])
        if pos_error < 0.15:
            print(f"[INFO] Se alcanzó el waypoint {self.current_target_idx}: {self.TARGET_POS}")
            self.reached_errors.append(pos_error)  # Guarda el error al llegar
            if self.current_target_idx < len(self.WAYPOINTS) - 1:
                self.current_target_idx += 1
                self.TARGET_POS = self.WAYPOINTS[self.current_target_idx]
            return False
    
        
        elif (abs(state[0]) > 3.5 or abs(state[1]) > 3.5 or state[2] > 3.5 or  # Condiciones de truncamiento
            abs(state[7]) > 0.6 or abs(state[8]) > 0.6 or state[2] <= 0.075):
            return True  # Termina el episodio si se cumplen las condiciones de truncamiento
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
    
    def _draw_waypoints(self):
        """Dibuja la trayectoria circular de referencia en azul."""
        for i in range(len(self.WAYPOINTS)):
            start = self.WAYPOINTS[i]
            end = self.WAYPOINTS[(i + 1) % len(self.WAYPOINTS)]
            p.addUserDebugLine(
                lineFromXYZ=start,
                lineToXYZ=end,
                lineColorRGB=[0, 0, 1],  # Azul
                lineWidth=2,
                lifeTime=0,              # Permanente
                physicsClientId=self.CLIENT
            )