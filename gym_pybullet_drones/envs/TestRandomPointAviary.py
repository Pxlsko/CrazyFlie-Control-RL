from turtle import distance
import numpy as np
import pybullet as p
from gymnasium import spaces   
import itertools                                          # ① NUEVO
from scipy.interpolate import CubicSpline

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class TestRandomPointAviary(BaseRLAviary):
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
                 trajectory_type: str = "helix"  # "circle", "square", "random", "helix", "random_and_hover"
                 ):
        self.NUM_WAYPOINTS = 16 # 1 para random, 16 para circle y square
        self.RADIUS = 1.0

        self.CENTER = np.array([1.5, 1.5, 1.5])

        self.trajectory_type = trajectory_type

        if self.trajectory_type == "circle":
            self.WAYPOINTS = self._generate_smooth_circle_waypoints(self.CENTER, self.RADIUS, self.NUM_WAYPOINTS, smooth_points=300)
        elif self.trajectory_type == "square":
            # self.WAYPOINTS = self._generate_square_waypoints(self.CENTER, self.RADIUS)
            self.WAYPOINTS = self._generate_square_waypoints(self.CENTER, self.RADIUS, points_per_edge=35)
        elif self.trajectory_type in ["random", "random_and_hover"]:
            self.WAYPOINTS = self._generate_random_waypoints(self.CENTER, self.RADIUS,  self.NUM_WAYPOINTS)
        elif self.trajectory_type == "helix":
            self.WAYPOINTS = self._generate_helix_waypoints(self.CENTER, self.RADIUS, height=1.5, turns=2, num_points=100)
        else:
            raise ValueError(f"Trayectoria '{self.trajectory_type}' no soportada")

        self.HOVER_TIME_SEC = 5.0  # Tiempo de hover sobre cada punto en segundos
        self.hover_counter = 0     # Contador de pasos de hover
        self.in_hover = False      # ¿Está en modo hover?

        self.current_target_idx = 0
        self.TARGET_POS = self.WAYPOINTS[self.current_target_idx]
        self.EPISODE_LEN_SEC = 140 # 20 para la random, 150 para la rectangular, 110 para la circular, 140 para la hélice 
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

    # def _generate_square_waypoints(self, center, size):
    #     """Genera 4 puntos formando un cuadrado en el plano XY."""
    #     half = size
    #     z = center[2]
    #     points = [
    #         [center[0] - half, center[1] - half, z],
    #         [center[0] - half, center[1] + half, z],
    #         [center[0] + half, center[1] + half, z],
    #         [center[0] + half, center[1] - half, z]
    #     ]
    #     return [np.round(np.array(p, dtype=np.float32), 2) for p in points]
    def _generate_square_waypoints(self, center, size, points_per_edge=10):
        """Genera una trayectoria cuadrada compuesta por más puntos interpolados entre las esquinas."""
        half = size
        z = center[2]
        
        # Esquinas del cuadrado
        corners = [
            [center[0] - half, center[1] - half, z],
            [center[0] - half, center[1] + half, z],
            [center[0] + half, center[1] + half, z],
            [center[0] + half, center[1] - half, z],
            [center[0] - half, center[1] - half, z],  # Vuelve al inicio para cerrar el cuadrado
        ]

        # Interpolar puntos entre las esquinas
        waypoints = []
        for i in range(len(corners) - 1):
            start = np.array(corners[i])
            end = np.array(corners[i + 1])
            for t in np.linspace(0, 1, points_per_edge, endpoint=True):
                point = (1 - t) * start + t * end
                waypoints.append(np.round(point, 2))
        
        return waypoints

    
    def _generate_helix_waypoints(self, center, radius, height, turns, num_points):
        """
        Genera puntos de una trayectoria helicoidal.

        Parámetros:
        - center: Centro de la hélice.
        - radius: Radio horizontal de la hélice.
        - height: Altura total que alcanza.
        - turns: Número de vueltas.
        - num_points: Número total de puntos de la hélice.

        Retorna:
        - Lista de waypoints en forma de hélice.
        """
        t = np.linspace(0, 2 * np.pi * turns, num_points)
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        z = center[2] + np.linspace(0, height, num_points)
        waypoints = [np.round(np.array([xi, yi, zi], dtype=np.float32), 2) for xi, yi, zi in zip(x, y, z)]
        return waypoints

    def _update_target_marker(self):
        """Crea o actualiza una esfera traslúcida naranja en el objetivo actual."""
        # Elimina el marcador anterior si existe
        if hasattr(self, "_target_marker_id") and self._target_marker_id is not None:
            try:
                p.removeBody(self._target_marker_id, physicsClientId=self.CLIENT)
            except Exception:
                pass
            self._target_marker_id = None

        # Solo dibuja esfera para trayectorias random
        if self.trajectory_type == "random" or self.trajectory_type == "random_and_hover":
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                rgbaColor=[1, 0.5, 0, 0.4],  # Naranja translúcido
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
            self._target_marker_id = None  # No hay marcador para otros tipos

    def get_position_match_percentage(self, tolerance=0.0):
        """
        Calcula el porcentaje de waypoints alcanzados dentro de una tolerancia dada.
        """
        if not hasattr(self, "reached_errors") or len(self.reached_errors) == 0:
            return 0.0
        total = len(self.reached_errors)
        matched = sum(e <= tolerance for e in self.reached_errors)
        return 100.0 * matched / total if total > 0 else 0.0
    
    def _generate_random_waypoints(self, center, radius, num_points):
        """Genera puntos aleatorios dentro de un cubo centrado en center."""
        waypoints = []
        for _ in range(num_points):
            point = center + np.random.uniform(-radius, radius, size=3)
            waypoints.append(np.round(point, 1))
        return waypoints

    def _generate_smooth_circle_waypoints(self, center, radius, num_points, smooth_points=100):
        """Genera puntos suavizados en un círculo usando una spline y redondea a 2 decimales."""
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
        waypoints = [
            np.round(np.array([cs_x(ti), cs_y(ti), cs_z(ti)], dtype=np.float32), 3)  
            for ti in t_smooth
        ]
        return waypoints
    
    def reset(self, seed=None, options=None):
        self.current_target_idx = 0
        self.TARGET_POS = self.WAYPOINTS[self.current_target_idx]
        self.reached_errors = []
        self.hover_counter = 0
        self.in_hover = False
        obs = super().reset(seed=seed, options=options)
        self._update_target_marker()
        return obs

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
        pos_k = 0.05
        ori_k = 0.002
        lin_vel_k = 0.005
        ang_vel_k = 0.002
        act_k = 0.003
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
        # if len(self.action_buffer) > 0:
        #     last_action = self.action_buffer[-1][0]  # Última acción tomada
        #     act_r = -act_k * np.linalg.norm(last_action)
        # else:
        #     act_r = 0
        if len(self.action_buffer) > 1:
            last_action = self.action_buffer[-1][0]      # Acción actual
            prev_action = self.action_buffer[-2][0]      # Acción anterior
            jerk = last_action - prev_action
            jerk_r = -act_k * np.linalg.norm(jerk)**2    # Penalización cuadrática
        else:
            jerk_r = 0

        if np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.05 and np.linalg.norm(state[10:13]) < 0.1:
            max_reward = 100.0
            initial_dist = np.linalg.norm(self.TARGET_POS - np.array([0, 0, 0]))
            norm_steps = self.step_counter / (initial_dist + 1e-6)
            reward = max_reward * (1 - norm_steps / (self.PYB_FREQ * self.EPISODE_LEN_SEC))
            reward = max(reward, 0)
            return reward

        # Recompensa total
        # total_r = pos_r + ori_r + lin_vel_r + ang_vel_r + act_r + survival_r
        total_r = pos_r + ori_r + lin_vel_r + ang_vel_r + jerk_r + survival_r
    

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
        # if np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.1:
        #     print(f"[INFO] Se alcanzó el waypoint {self.current_target_idx}: {self.TARGET_POS}")
        #     self.current_target_idx += 1
        #     if self.current_target_idx >= len(self.WAYPOINTS):
        #         print("[INFO] Trayectoria circular completada.")
        #         return True  # Termina el episodio tras completar la vuelta
        #     self.TARGET_POS = self.WAYPOINTS[self.current_target_idx]
        #     return False
        pos_error = np.linalg.norm(self.TARGET_POS - state[0:3])
        if self.trajectory_type == "random_and_hover":
                    if not self.in_hover:
                        if pos_error < 0.1:
                            print(f"[INFO] Llegó al waypoint {self.current_target_idx}: {self.TARGET_POS}, inicia hover")
                            self.reached_errors.append(pos_error)
                            self.in_hover = True
                            self.hover_counter = 0
                        # No termina episodio, sigue hasta que haga hover
                        return False
                    else:
                        # Ya está haciendo hover
                        self.hover_counter += 1
                        if self.hover_counter >= int(self.HOVER_TIME_SEC * self.CTRL_FREQ):
                            # Hover terminado, avanza al siguiente waypoint si hay más
                            self.in_hover = False
                            self.hover_counter = 0
                            if self.current_target_idx < len(self.WAYPOINTS) - 1:
                                self.current_target_idx += 1
                                self.TARGET_POS = self.WAYPOINTS[self.current_target_idx]
                                self._update_target_marker()
                            else:
                                # Último objetivo alcanzado, elimina marcador
                                if hasattr(self, "_target_marker_id") and self._target_marker_id is not None:
                                    try:
                                        p.removeBody(self._target_marker_id, physicsClientId=self.CLIENT)
                                    except Exception:
                                        pass
                                    self._target_marker_id = None
                            return False
                        else:
                            # Sigue haciendo hover
                            return False

        # --- Comportamiento original para el resto de trayectorias
        pos_error = np.linalg.norm(self.TARGET_POS - state[0:3])
        if pos_error < 0.1:
            print(f"[INFO] Se alcanzó el waypoint {self.current_target_idx}: {self.TARGET_POS}")
            self.reached_errors.append(pos_error)
            if self.current_target_idx < len(self.WAYPOINTS) - 1:
                self.current_target_idx += 1
                self.TARGET_POS = self.WAYPOINTS[self.current_target_idx]
                self._update_target_marker()
            else:
                # Último objetivo alcanzado, elimina marcador
                if hasattr(self, "_target_marker_id") and self._target_marker_id is not None:
                    try:
                        p.removeBody(self._target_marker_id, physicsClientId=self.CLIENT)
                    except Exception:
                        pass
                    self._target_marker_id = None
            return False

        elif (abs(state[0]) > 3.5 or abs(state[1]) > 3.5 or state[2] > 3.5 or  
            abs(state[7]) > 0.6 or abs(state[8]) > 0.6 or state[2] <= 0.075):
            # Si termina abruptamente, elimina marcador
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
        """No dibuja nada para random, mantiene compatibilidad."""
        # Para trayectorias no random, puedes mantener las líneas azules si quieres
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
                    lineColorRGB=[0, 0, 1],  # Azul
                    lineWidth=2,
                    lifeTime=0,              # Permanente
                    physicsClientId=self.CLIENT
                )
                self._waypoint_line_ids.append(line_id)