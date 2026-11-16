"""
modelo.py - Modelo matemático y métodos numéricos.

Campo de cargas puntuales (Ley de Coulomb + superposición):
    E(r) = k * sum_i q_i (r - r_i) / |r - r_i|^3

Una línea de campo es una curva r(s) cuya tangente apunta siempre en la
dirección de E. Si se parametriza por longitud de arco s, cumple el sistema
de ecuaciones diferenciales ordinarias (EDO) de primer orden:

    dx/ds = Ex / |E|        dy/ds = Ey / |E|

con condición inicial r(0) = punto semilla cerca de una carga.
El sistema se resuelve con Euler o con Runge-Kutta de 4o orden (RK4).
"""
import math

import numpy as np

K = 8.9875e9          # constante de Coulomb (N·m²/C²)
MICRO = 1e-6          # las cargas se capturan en microcoulombs

# Parámetros de integración (valores por defecto)
PASO = 0.02           # tamaño de paso h (m)
MAX_PASOS = 3000      # límite de pasos por línea
RADIO_SEMILLA = 0.08  # distancia a la carga donde inicia cada línea (m)
RADIO_PARADA = 0.06   # si la línea llega a esta distancia de una carga, termina
LIMITE = 6.0          # la línea termina si sale del cuadro [-LIMITE, LIMITE]
LINEAS_POR_UC = 8     # líneas por cada microcoulomb de carga


def campo(x, y, pos, q):
    """Vector E en (x, y). pos: arreglo (n, 2); q: arreglo (n,) en µC."""
    dx = x - pos[:, 0]
    dy = y - pos[:, 1]
    r3 = (dx * dx + dy * dy) ** 1.5
    r3 = np.maximum(r3, 1e-12)
    f = K * q * MICRO / r3
    return np.array([np.sum(f * dx), np.sum(f * dy)])


def direccion(p, pos, q, sentido):
    """Lado derecho de la EDO: vector unitario en la dirección de E (o -E)."""
    e = campo(p[0], p[1], pos, q)
    norma = np.hypot(e[0], e[1])
    if norma < 1e-30:
        return None
    return sentido * e / norma


def paso_euler(p, h, pos, q, sentido):
    k1 = direccion(p, pos, q, sentido)
    if k1 is None:
        return None
    return p + h * k1


def paso_rk4(p, h, pos, q, sentido):
    k1 = direccion(p, pos, q, sentido)
    if k1 is None:
        return None
    k2 = direccion(p + h / 2 * k1, pos, q, sentido)
    if k2 is None:
        return None
    k3 = direccion(p + h / 2 * k2, pos, q, sentido)
    if k3 is None:
        return None
    k4 = direccion(p + h * k3, pos, q, sentido)
    if k4 is None:
        return None
    return p + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


METODOS = {'RK4': paso_rk4, 'Euler': paso_euler}


# ---------------------------------------------------------------------------
# Versión escalar (Python puro). Hace exactamente las mismas cuentas, pero con
# números de punto flotante sueltos en lugar de arreglos de NumPy. Con 5 cargas
# o menos, crear arreglos pequeños en cada evaluación cuesta más que la cuenta
# misma, por eso esta versión es varias veces más rápida (ver rendimiento.py).
# La constante K no se usa aquí porque se cancela al normalizar E / |E|.
# ---------------------------------------------------------------------------
def _dir_py(x, y, cargas, sentido):
    ex = ey = 0.0
    for cx, cy, qq in cargas:
        dx = x - cx
        dy = y - cy
        r2 = dx * dx + dy * dy
        if r2 < 1e-12:
            r2 = 1e-12
        f = qq / (r2 * math.sqrt(r2))
        ex += f * dx
        ey += f * dy
    n = math.hypot(ex, ey)
    if n < 1e-300:
        return None
    return sentido * ex / n, sentido * ey / n


def _euler_py(x, y, h, cargas, s):
    k1 = _dir_py(x, y, cargas, s)
    if k1 is None:
        return None
    return x + h * k1[0], y + h * k1[1]


def _rk4_py(x, y, h, cargas, s):
    k1 = _dir_py(x, y, cargas, s)
    if k1 is None:
        return None
    k2 = _dir_py(x + h / 2 * k1[0], y + h / 2 * k1[1], cargas, s)
    if k2 is None:
        return None
    k3 = _dir_py(x + h / 2 * k2[0], y + h / 2 * k2[1], cargas, s)
    if k3 is None:
        return None
    k4 = _dir_py(x + h * k3[0], y + h * k3[1], cargas, s)
    if k4 is None:
        return None
    return (x + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            y + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))


METODOS_PY = {'RK4': _rk4_py, 'Euler': _euler_py}


def _trazar_py(semilla, pos, q, sentido, metodo, h, max_pasos, origen):
    paso = METODOS_PY[metodo]
    cargas = [(float(pos[i, 0]), float(pos[i, 1]), float(q[i])) for i in range(len(q))]
    x, y = float(semilla[0]), float(semilla[1])
    puntos = [(x, y)]
    fin = None
    r2_parada = RADIO_PARADA * RADIO_PARADA
    for k in range(1, max_pasos + 1):
        nuevo = paso(x, y, h, cargas, sentido)
        if nuevo is None:
            break
        x, y = nuevo
        puntos.append(nuevo)
        if abs(x) > LIMITE or abs(y) > LIMITE:
            break
        for j, (cx, cy, _) in enumerate(cargas):
            if (j != origen or k >= 10) and (x - cx) ** 2 + (y - cy) ** 2 < r2_parada:
                fin = j
                break
        if fin is not None:
            break
    return np.array(puntos), fin


def trazar_linea(semilla, pos, q, sentido=1, metodo='RK4', h=PASO, max_pasos=MAX_PASOS, origen=None,
                 motor='python'):
    """
    Integra una línea de campo desde 'semilla'.
    Regresa (puntos, fin) donde fin indica en qué carga terminó (índice) o None
    si salió del área o llegó al límite de pasos.
    """
    if motor == 'python':
        return _trazar_py(semilla, pos, q, sentido, metodo, h, max_pasos, origen)
    paso = METODOS[metodo]
    puntos = np.empty((max_pasos + 1, 2))
    puntos[0] = semilla
    p = np.array(semilla, dtype=float)
    fin = None
    k = 0
    for k in range(1, max_pasos + 1):
        nuevo = paso(p, h, pos, q, sentido)
        if nuevo is None:
            break
        p = nuevo
        puntos[k] = p
        if abs(p[0]) > LIMITE or abs(p[1]) > LIMITE:
            break
        d = np.hypot(pos[:, 0] - p[0], pos[:, 1] - p[1])
        cerca = np.where(d < RADIO_PARADA)[0]
        cerca = cerca[cerca != origen] if k < 10 else cerca
        if cerca.size:
            fin = int(cerca[0])
            break
    return puntos[:k + 1].copy(), fin


def semillas(cx, cy, n, radio=RADIO_SEMILLA):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
    return np.column_stack((cx + radio * np.cos(ang), cy + radio * np.sin(ang)))


def generar_lineas(pos, q, metodo='RK4', h=PASO, motor='python'):
    """
    Genera todas las líneas de campo de la configuración.
    - Desde cada carga positiva se integra en el sentido de E (sale de la carga).
    - Desde cada carga negativa se integra en sentido contrario (-E), así se
      dibujan las líneas que llegan a ella desde lejos. Si una de esas líneas
      termina en una carga positiva, se descarta porque ya se dibujó desde la
      positiva.
    El número de líneas de cada carga es proporcional a |q|.
    Regresa una lista de arreglos (m, 2) orientados en el sentido de E.
    """
    lineas = []
    for i in range(len(q)):
        n = int(np.clip(round(LINEAS_POR_UC * abs(q[i])), 4, 40))
        sentido = 1 if q[i] > 0 else -1
        for s in semillas(pos[i, 0], pos[i, 1], n):
            pts, fin = trazar_linea(s, pos, q, sentido, metodo, h, origen=i, motor=motor)
            if sentido == -1:
                if fin is not None and q[fin] > 0:
                    continue
                pts = pts[::-1]          # se invierte para que apunte como E
            if len(pts) > 2:
                lineas.append(pts)
    return lineas
