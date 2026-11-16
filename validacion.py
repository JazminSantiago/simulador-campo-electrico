"""
validacion.py - Verificación del método numérico (Ecuaciones Diferenciales).

Para dos cargas sobre el eje x existe una cantidad que se conserva a lo largo
de cada línea de campo:
        C = q1·cos(θ1) + q2·cos(θ2)
donde θi es el ángulo entre el eje x y el vector que va de la carga i al punto.
Si la integración fuera exacta, C sería constante en toda la línea. Se recorre
la misma longitud de línea (1.5 m) con distintos pasos h y la diferencia
|C(final) - C(inicial)| mide el error global del método numérico.
Se compara Euler contra RK4 con varios tamaños de paso h.
"""
import os

import numpy as np

from modelo import trazar_linea

POS = np.array([[-1.0, 0.0], [1.0, 0.0]])
Q = np.array([1.0, -1.0])
PASOS_H = [0.1, 0.05, 0.02, 0.01, 0.005]
ANGULO_INICIAL = np.radians(60)
SALIDA = 'resultados'
LONGITUD = 1.5        # se compara C después de recorrer 1.5 m de la línea


def invariante(p):
    c = 0.0
    for (cx, cy), qq in zip(POS, Q):
        dx, dy = p[:, 0] - cx, p[:, 1] - cy
        c = c + qq * dx / np.hypot(dx, dy)
    return c


def error_metodo(metodo, h):
    r0 = 0.08
    semilla = POS[0] + r0 * np.array([np.cos(ANGULO_INICIAL), np.sin(ANGULO_INICIAL)])
    pasos = int(round(LONGITUD / h))
    pts, fin = trazar_linea(semilla, POS, Q, 1, metodo, h, max_pasos=pasos, origen=0)
    c = invariante(pts)
    return abs(c[-1] - c[0]), len(pts) - 1, fin


def main():
    os.makedirs(SALIDA, exist_ok=True)
    filas = []
    print(f"{'h':>6} | {'Euler: error':>13} {'pasos':>6} | {'RK4: error':>13} {'pasos':>6}")
    for h in PASOS_H:
        ee, pe, _ = error_metodo('Euler', h)
        er, pr, _ = error_metodo('RK4', h)
        filas.append((h, ee, er))
        print(f'{h:6.3f} | {ee:13.3e} {pe:6d} | {er:13.3e} {pr:6d}')
    filas = np.array(filas)
    # orden de convergencia observado: pendiente en escala log-log
    oe = np.polyfit(np.log(filas[:, 0]), np.log(filas[:, 1]), 1)[0]
    orr = np.polyfit(np.log(filas[:, 0]), np.log(filas[:, 2]), 1)[0]
    print(f'Orden observado: Euler ≈ {oe:.2f}   RK4 ≈ {orr:.2f}')
    np.savetxt(os.path.join(SALIDA, 'validacion.csv'), filas, delimiter=',',
               header='h,error_euler,error_rk4', comments='', fmt='%.6e')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    axs[0].loglog(filas[:, 0], filas[:, 1], 'o-', color='#e08a1e', label=f'Euler (pendiente {oe:.2f})')
    axs[0].loglog(filas[:, 0], filas[:, 2], 's-', color='#1f4fb4', label=f'RK4 (pendiente {orr:.2f})')
    axs[0].set_xlabel('tamaño de paso h (m)')
    axs[0].set_ylabel('error |C(1.5 m) − C(0)|')
    axs[0].set_title('Convergencia de los métodos')
    axs[0].legend()
    axs[0].grid(True, which='both', color='0.9')
    # misma línea con h = 0.2 para ver la diferencia
    r0 = 0.08
    for ang in np.radians([30, 60, 90, 120]):
        s = POS[0] + r0 * np.array([np.cos(ang), np.sin(ang)])
        for metodo, c, est in (('Euler', '#e08a1e', '--'), ('RK4', '#1f4fb4', '-')):
            pts, _ = trazar_linea(s, POS, Q, 1, metodo, 0.2, max_pasos=100, origen=0)
            axs[1].plot(pts[:, 0], pts[:, 1], est, color=c, lw=1.4,
                        label=metodo if ang == np.radians(30) else None)
    axs[1].scatter(*POS.T, c=['#d62728', '#1f4fb4'], s=90, zorder=5, edgecolor='k')
    axs[1].set_aspect('equal')
    axs[1].set_xlim(-2.5, 2.5)
    axs[1].set_ylim(-0.3, 2.8)
    axs[1].set_title('Líneas del dipolo con h = 0.2')
    axs[1].legend()
    axs[1].grid(True, color='0.9')
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, 'grafica_validacion.png'), dpi=150)
    print('Resultados guardados en la carpeta', SALIDA)


if __name__ == '__main__':
    main()
