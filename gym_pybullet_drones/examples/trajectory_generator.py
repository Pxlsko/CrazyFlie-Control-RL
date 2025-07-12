import numpy as np
from scipy.interpolate import CubicSpline

class TrajectoryGenerator:
    """
    Clase para generar diferentes tipos de trayectorias de waypoints.
    """
    def __init__(self):
        pass # No se necesita inicialización especial para este generador

    def generate_smooth_circle_waypoints(self, center: np.ndarray, radius: float, num_points: int, smooth_points: int = 100):
        """
        Genera puntos suavizados en un círculo usando una spline cúbica.
        Args:
            center (np.ndarray): Coordenadas (x, y, z) del centro del círculo.
            radius (float): Radio del círculo.
            num_points (int): Número de puntos originales para definir el círculo (antes del suavizado).
            smooth_points (int): Número de puntos finales de la trayectoria suavizada.
        Returns:
            list[np.ndarray]: Lista de waypoints [x, y, z] redondeados a 2 decimales.
        """
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        z = np.full_like(x, center[2]) # Z se mantiene constante

        # Cierra el círculo para la spline (agrega el primer punto al final)
        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])
        z_closed = np.append(z, z[0])

        t_original = np.linspace(0, 1, num_points + 1)
        t_smooth = np.linspace(0, 1, smooth_points)

        cs_x = CubicSpline(t_original, x_closed, bc_type='periodic')
        cs_y = CubicSpline(t_original, y_closed, bc_type='periodic')
        cs_z = CubicSpline(t_original, z_closed, bc_type='periodic')

        waypoints = [
            np.round(np.array([cs_x(ti), cs_y(ti), cs_z(ti)], dtype=np.float32), 2)
            for ti in t_smooth
        ]
        return waypoints

    def generate_helix_waypoints(self, center: np.ndarray, radius: float, height: float, turns: float, num_points: int):
        """
        Genera puntos para una trayectoria helicoidal.
        Args:
            center (np.ndarray): Coordenadas (x, y, z) del inicio de la hélice.
            radius (float): Radio de la hélice.
            height (float): Altura total que la hélice asciende/desciende.
            turns (float): Número de vueltas de la hélice.
            num_points (int): Número de puntos para la trayectoria.
        Returns:
            list[np.ndarray]: Lista de waypoints [x, y, z] redondeados a 2 decimales.
        """
        t = np.linspace(0, 2 * np.pi * turns, num_points)
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        z = center[2] + np.linspace(0, height, num_points) # Elevación lineal en Z

        waypoints = [np.round(np.array([xi, yi, zi], dtype=np.float32), 2) for xi, yi, zi in zip(x, y, z)]
        return waypoints

    def generate_square_waypoints(self, center: np.ndarray, side_length: float, num_points: int, smooth_points: int = 100):
        """
        Genera puntos suavizados en una trayectoria cuadrada.
        Args:
            center (np.ndarray): Coordenadas (x, y, z) del centro del cuadrado.
            side_length (float): Longitud del lado del cuadrado.
            num_points (int): Número de puntos principales para definir el cuadrado (debe ser al menos 4).
            smooth_points (int): Número de puntos finales de la trayectoria suavizada.
        Returns:
            list[np.ndarray]: Lista de waypoints [x, y, z] redondeados a 2 decimales.
        """
        if num_points < 4:
            raise ValueError("num_points must be at least 4 for a square trajectory.")

        half_side = side_length / 2
        
        # Puntos de las esquinas del cuadrado (ajustados al centro)
        corners_x = [center[0] + half_side, center[0] - half_side, center[0] - half_side, center[0] + half_side, center[0] + half_side]
        corners_y = [center[1] + half_side, center[1] + half_side, center[1] - half_side, center[1] - half_side, center[1] + half_side]
        corners_z = [center[2], center[2], center[2], center[2], center[2]]

        # Interpolar para obtener los puntos de control del spline
        original_x = []
        original_y = []
        original_z = []

        # Puntos distribuidos a lo largo de cada lado, incluyendo las esquinas
        points_per_side = int(num_points / 4) # Puntos para cada lado sin contar esquinas duplicadas
        
        for i in range(4): # Para cada lado del cuadrado
            start_x, start_y, start_z = corners_x[i], corners_y[i], corners_z[i]
            end_x, end_y, end_z = corners_x[i+1], corners_y[i+1], corners_z[i+1]

            # Generar puntos intermedios para cada lado (excluyendo el punto final para evitar duplicados en las esquinas)
            if i < 3:
                interp_frac = np.linspace(0, 1, points_per_side, endpoint=False) # Excluye el último para no duplicar el inicio del siguiente segmento
            else: # Último segmento, incluye el punto final para cerrar
                interp_frac = np.linspace(0, 1, points_per_side)

            original_x.extend(start_x + interp_frac * (end_x - start_x))
            original_y.extend(start_y + interp_frac * (end_y - start_y))
            original_z.extend(start_z + interp_frac * (end_z - start_z))
        
        # Añadir el primer punto al final para cerrar la spline
        original_x.append(original_x[0])
        original_y.append(original_y[0])
        original_z.append(original_z[0])

        t_original = np.linspace(0, 1, len(original_x))
        t_smooth = np.linspace(0, 1, smooth_points)

        cs_x = CubicSpline(t_original, original_x, bc_type='periodic')
        cs_y = CubicSpline(t_original, original_y, bc_type='periodic')
        cs_z = CubicSpline(t_original, original_z, bc_type='periodic')

        waypoints = [
            np.round(np.array([cs_x(ti), cs_y(ti), cs_z(ti)], dtype=np.float32), 2)
            for ti in t_smooth
        ]
        return waypoints

    # Puedes añadir un método para puntos aleatorios si lo necesitas,
    # aunque para comparación de trayectorias, las determinísticas son mejores.
    def generate_random_point_waypoints(self, center: np.ndarray, radius: float, num_points: int):
        """
        Genera puntos aleatorios en el espacio 3D.
        Args:
            center (np.ndarray): Centro alrededor del cual se generarán los puntos.
            radius (float): Radio máximo de desviación desde el centro.
            num_points (int): Número de puntos aleatorios a generar.
        Returns:
            list[np.ndarray]: Lista de waypoints [x, y, z] redondeados a 2 decimales.
        """
        waypoints = []
        for _ in range(num_points):
            offset = (np.random.rand(3) - 0.5) * 2 * radius # Aleatorio entre -radius y +radius
            point = center + offset
            waypoints.append(np.round(point, 2))
        return waypoints