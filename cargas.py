"""
cargas.py - Estructura de datos para las cargas eléctricas.

Las cargas se guardan en un arreglo de NumPy de tamaño fijo (MAX_CARGAS filas).
Cada fila tiene x, y (metros) y q (microcoulombs). Un contador indica cuántas
filas están en uso. Al ser memoria contigua y preasignada:
  - no se piden ni liberan bloques de memoria al agregar o quitar cargas,
  - el cálculo del campo puede leer las cargas como un bloque (vectorizado),
  - el espacio ocupado es constante: 5 x 3 x 8 bytes = 120 bytes.
"""
import numpy as np

MAX_CARGAS = 5


class GestorCargas:
    def __init__(self, maximo=MAX_CARGAS):
        self.maximo = maximo
        self.datos = np.zeros((maximo, 3), dtype=np.float64)  # columnas: x, y, q
        self.n = 0

    # ---- consultas -------------------------------------------------------
    @property
    def posiciones(self):
        return self.datos[:self.n, :2]

    @property
    def valores(self):
        return self.datos[:self.n, 2]

    def __len__(self):
        return self.n

    def lleno(self):
        return self.n >= self.maximo

    def cercana(self, x, y, radio):
        """Índice de la carga más cercana a (x, y) dentro de 'radio', o None."""
        if self.n == 0:
            return None
        d = np.hypot(self.posiciones[:, 0] - x, self.posiciones[:, 1] - y)
        i = int(np.argmin(d))
        return i if d[i] <= radio else None

    # ---- operaciones -----------------------------------------------------
    def agregar(self, x, y, q):
        if self.lleno():
            raise ValueError(f"Solo se permiten {self.maximo} cargas")
        if q == 0:
            raise ValueError("La magnitud de la carga no puede ser 0")
        self.datos[self.n] = (x, y, q)
        self.n += 1
        return self.n - 1

    def modificar(self, i, x=None, y=None, q=None):
        self._validar(i)
        if x is not None:
            self.datos[i, 0] = x
        if y is not None:
            self.datos[i, 1] = y
        if q is not None:
            if q == 0:
                raise ValueError("La magnitud de la carga no puede ser 0")
            self.datos[i, 2] = q

    def eliminar(self, i):
        self._validar(i)
        # se recorren las filas siguientes para que las cargas sigan contiguas
        self.datos[i:self.n - 1] = self.datos[i + 1:self.n]
        self.n -= 1
        self.datos[self.n] = 0

    def limpiar(self):
        self.datos[:] = 0
        self.n = 0

    def bytes_usados(self):
        return self.datos.nbytes

    def _validar(self, i):
        if not 0 <= i < self.n:
            raise IndexError("Carga inexistente")
