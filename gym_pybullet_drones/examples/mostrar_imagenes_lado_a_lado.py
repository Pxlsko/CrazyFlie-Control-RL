import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def mostrar_y_guardar_pares_individuales(rutas_imagenes, carpeta_salida, nombre_base="comparacion", figsize=(10, 5)):
    """
    Muestra pares de imágenes una al lado de la otra en varias filas y guarda el resultado en un archivo.

    Args:
        rutas_imagenes (list): Lista de rutas a las imágenes. Debe tener un número par de elementos.
        carpeta_salida (str): Ruta donde se guardará la imagen combinada.
        nombre_salida (str): Nombre del archivo de salida.
        figsize_individual (tuple): Tamaño de cada fila de imágenes.
    """
    if len(rutas_imagenes) % 2 != 0:
        raise ValueError("La lista de imágenes debe tener un número par de elementos.")

    num_pares = len(rutas_imagenes) // 2

    for i in range(num_pares):
        img1 = mpimg.imread(rutas_imagenes[2 * i])
        img2 = mpimg.imread(rutas_imagenes[2 * i + 1])

        fig, axs = plt.subplots(1, 2, figsize=figsize)
        axs[0].imshow(img1, aspect='auto')
        axs[0].set_title("Modelo RL", fontsize=14)

        axs[0].axis('off')
        axs[1].imshow(img2, aspect='auto')
        axs[1].set_title("Modelo PID", fontsize=14)
        axs[1].axis('off')

        plt.tight_layout()

        nombre_archivo = f"{nombre_base}_{i + 1}.png"
        ruta_guardado = os.path.join(carpeta_salida, nombre_archivo)
        plt.savefig(ruta_guardado, bbox_inches='tight')
        plt.show()
        plt.close(fig)  # Cierra la figura para liberar memoria


if __name__ == "__main__":
    # === RELLENA AQUÍ LOS NOMBRES DE ARCHIVO (sin patrón) ===
    # nombres_archivos = [
    #     "posicion_deseada_vs_medida.png", "posicion_deseada_vs_medida.png",
    #     "orientacion_deseada_vs_medida.png","orientacion_deseada_vs_medida.png",
    #     "velocidad_lineal_deseada_vs_medida.png", "velocidad_lineal_deseada_vs_medida.png",
    #     "velocidad_angular_deseada_vs_medida.png", "velocidad_angular_deseada_vs_medida.png",
    #     "helixRPM_filtrado.png", "RPM_filtrado.png",
    # ]
    nombres_archivos = [
        "rectRPM_filtrado.png", "RPM_filtrado.png"
    ]

    # Carpetas base
    base_rl = r"C:\Users\pablo\OneDrive\Escritorio\UNI\4CARRERA\RL\ProgresoTFGPython\ExpTrajFinv3\Definitivo\RectangleTrajectory"
    base_pid = r"C:\Users\pablo\OneDrive\Escritorio\UNI\4CARRERA\RL\ProgresoTFGPython\ExpTrajFinv3\Definitivo\PID\Rectangular\rpm"

    # Crear lista completa de rutas alternadas (RL, PID, RL, PID...)
    rutas = []
    for i in range(0, len(nombres_archivos), 2):
        ruta_rl = os.path.join(base_rl, nombres_archivos[i])
        ruta_pid = os.path.join(base_pid, nombres_archivos[i + 1])
        rutas.append(ruta_rl)
        rutas.append(ruta_pid)

    # Carpeta de salida
    carpeta_salida = r"C:\Users\pablo\OneDrive\Escritorio\UNI\4CARRERA\RL\ProgresoTFGPython\ExpTrajFinv3\Definitivo\Comparations\RLPIDRectangle\rpm"

    mostrar_y_guardar_pares_individuales(rutas, carpeta_salida)
