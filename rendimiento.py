"""
rendimiento.py - Mediciones de tiempo y memoria (Arquitectura de Computadoras).

Ejecuta varios escenarios de cargas con los dos métodos numéricos (RK4 y
Euler) y con las dos implementaciones del cálculo (Python puro y NumPy).
Para cada combinación mide:
  - tiempo de ejecución (promedio y desviación de varias repeticiones),
  - memoria pico reservada durante el cálculo (tracemalloc),
  - memoria ocupada por los puntos de las líneas y por la estructura de cargas,
  - memoria total del proceso (RSS, con psutil si está instalado).
Guarda resultados/mediciones.csv y gráficas PNG en la carpeta resultados.
"""
import csv
import os
import platform
import statistics
import time
import tracemalloc

import numpy as np

from cargas import GestorCargas
from modelo import generar_lineas

try:
    import psutil
    PROC = psutil.Process()
except ImportError:
    PROC = None

ESCENARIOS = {
    'E1: 1 carga': [(0, 0, 2)],
    'E2: dipolo': [(-1.5, 0, 1), (1.5, 0, -1)],
    'E3: 3 cargas': [(-2, -1.5, 2), (2, -1.5, -1), (0, 2, 1)],
    'E4: 5 cargas': [(-2, -1.5, 3), (1.5, 0, -1), (1, 3.2, 1), (-2.5, 2.5, -1), (3, -3, 1)],
}
REPETICIONES = 10
SALIDA = 'resultados'


def rss_mb():
    return PROC.memory_info().rss / 2**20 if PROC else float('nan')


def medir(nombre, lista, metodo, motor, reps=REPETICIONES):
    g = GestorCargas()
    for x, y, q in lista:
        g.agregar(x, y, q)
    pos, q = g.posiciones.copy(), g.valores.copy()
    generar_lineas(pos, q, metodo, motor=motor)            # calentamiento
    tiempos = []
    for _ in range(reps):
        t0 = time.perf_counter()
        lineas = generar_lineas(pos, q, metodo, motor=motor)
        tiempos.append((time.perf_counter() - t0) * 1000)
    rss_antes = rss_mb()
    tracemalloc.start()
    lineas = generar_lineas(pos, q, metodo, motor=motor)
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    pasos = sum(len(l) - 1 for l in lineas)
    prom = statistics.mean(tiempos)
    return {
        'escenario': nombre, 'cargas': len(lista), 'metodo': metodo, 'motor': motor,
        'lineas': len(lineas), 'pasos': pasos,
        'tiempo_ms': round(prom, 2), 'desv_ms': round(statistics.stdev(tiempos), 2),
        'us_por_paso': round(prom * 1000 / pasos, 2),
        'mem_pico_kb': round(pico / 1024, 1),
        'mem_lineas_kb': round(sum(l.nbytes for l in lineas) / 1024, 1),
        'mem_cargas_b': g.bytes_usados(),
        'rss_mb': round(rss_antes, 1),
    }


def graficas(filas):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nombres = list(ESCENARIOS)
    x = np.arange(len(nombres))

    def serie(metodo, motor, campo):
        return [next(f[campo] for f in filas if f['escenario'] == n and f['metodo'] == metodo and f['motor'] == motor)
                for n in nombres]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    w = 0.38
    for k, (m, c) in enumerate((('RK4', '#1f4fb4'), ('Euler', '#e08a1e'))):
        axs[0].bar(x + (k - 0.5) * w, serie(m, 'python', 'tiempo_ms'), w, label=m, color=c)
        axs[1].bar(x + (k - 0.5) * w, serie(m, 'python', 'mem_pico_kb'), w, label=m, color=c)
    axs[0].set_title('Tiempo de cálculo (Python puro)')
    axs[0].set_ylabel('ms')
    axs[1].set_title('Memoria pico durante el cálculo')
    axs[1].set_ylabel('KB')
    for a in axs:
        a.set_xticks(x, nombres)
        a.legend()
        a.grid(axis='y', color='0.9')
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, 'grafica_tiempo_memoria.png'), dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.bar(x - 0.19, serie('RK4', 'python', 'us_por_paso'), 0.38, label='Python puro', color='#1f4fb4')
    ax.bar(x + 0.19, serie('RK4', 'numpy', 'us_por_paso'), 0.38, label='NumPy', color='#8c8c8c')
    ax.set_title('Costo por paso de RK4 según la implementación')
    ax.set_ylabel('µs por paso')
    ax.set_xticks(x, nombres)
    ax.legend()
    ax.grid(axis='y', color='0.9')
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, 'grafica_python_vs_numpy.png'), dpi=150)


def main():
    os.makedirs(SALIDA, exist_ok=True)
    print('Equipo:', platform.processor() or platform.machine(), '|', platform.system(), platform.release(),
          '| Python', platform.python_version(), '| NumPy', np.__version__)
    filas = []
    for nombre, lista in ESCENARIOS.items():
        for metodo in ('RK4', 'Euler'):
            for motor in ('python', 'numpy'):
                f = medir(nombre, lista, metodo, motor)
                filas.append(f)
                print(f"{nombre:13s} {metodo:5s} {motor:6s} líneas={f['lineas']:3d} pasos={f['pasos']:6d} "
                      f"t={f['tiempo_ms']:9.2f} ms ±{f['desv_ms']:6.2f}  {f['us_por_paso']:7.2f} µs/paso  "
                      f"pico={f['mem_pico_kb']:7.1f} KB  líneas={f['mem_lineas_kb']:6.1f} KB  RSS={f['rss_mb']} MB")
    with open(os.path.join(SALIDA, 'mediciones.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    graficas(filas)
    print('\nResultados guardados en la carpeta', SALIDA)


if __name__ == '__main__':
    main()
