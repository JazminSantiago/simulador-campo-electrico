"""
interfaz.py - Interfaz gráfica con matplotlib.

Uso:
  - Clic en un lugar vacío del plano: inserta una carga con el signo y la
    magnitud elegidos en el panel (máximo 5).
  - Clic sobre una carga: la selecciona (se marca con un anillo).
  - Arrastrar una carga: la mueve; al soltar se recalculan las líneas.
  - Panel "Carga seleccionada": cambiar x, y o q y pulsar "Aplicar cambios".
  - "Eliminar carga" quita la seleccionada; "Reiniciar" borra todo.
  - "Método" cambia entre RK4 y Euler para comparar.
"""
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox

from cargas import GestorCargas, MAX_CARGAS
from modelo import generar_lineas, LIMITE

try:
    import psutil
    _proc = psutil.Process()

    def memoria_proceso():
        return _proc.memory_info().rss / 2**20     # MB
except ImportError:                                  # psutil es opcional
    def memoria_proceso():
        return None

LIM = 5.0                      # área visible: [-5, 5] x [-5, 5] m
ROJO, AZUL = '#d62728', '#1f4fb4'


class SimuladorCampo:
    def __init__(self):
        self.cargas = GestorCargas()
        self.metodo = 'RK4'
        self.sel = None
        self.arrastrando = False
        self.lineas = []
        self.stats = {'ms': 0.0, 'pasos': 0, 'kb_lineas': 0.0, 'rss': None}

        self.fig = plt.figure(figsize=(13.5, 8))
        try:
            self.fig.canvas.manager.set_window_title('Simulador de líneas de campo eléctrico')
        except Exception:
            pass
        self.ax = self.fig.add_axes([0.05, 0.07, 0.58, 0.87])
        self._panel()
        self.fig.canvas.mpl_connect('button_press_event', self._presionar)
        self.fig.canvas.mpl_connect('motion_notify_event', self._mover)
        self.fig.canvas.mpl_connect('button_release_event', self._soltar)
        self.actualizar()

    # ------------------------------------------------------------ panel
    def _panel(self):
        f = self.fig
        x0 = 0.69
        f.text(x0, 0.95, 'Nueva carga', fontsize=12, weight='bold')
        f.text(x0, 0.925, 'Clic en el plano para insertarla (máx. 5)', fontsize=9, color='0.35')
        ax = f.add_axes([x0, 0.81, 0.13, 0.09], frameon=True)
        self.r_signo = RadioButtons(ax, ('Positiva (+)', 'Negativa (−)'))
        self.t_mag = TextBox(f.add_axes([x0 + 0.21, 0.835, 0.07, 0.04]), 'Magnitud (µC) ', initial='1')

        f.text(x0, 0.76, 'Carga seleccionada', fontsize=12, weight='bold')
        self.t_sel = f.text(x0, 0.735, 'Ninguna (haga clic sobre una carga)', fontsize=9, color='0.35')
        self.t_x = TextBox(f.add_axes([x0 + 0.03, 0.675, 0.06, 0.04]), 'x (m) ', initial='')
        self.t_y = TextBox(f.add_axes([x0 + 0.13, 0.675, 0.06, 0.04]), 'y (m) ', initial='')
        self.t_q = TextBox(f.add_axes([x0 + 0.23, 0.675, 0.06, 0.04]), 'q (µC) ', initial='')
        self.b_aplicar = Button(f.add_axes([x0, 0.61, 0.135, 0.045]), 'Aplicar cambios')
        self.b_eliminar = Button(f.add_axes([x0 + 0.145, 0.61, 0.135, 0.045]), 'Eliminar carga')
        self.b_aplicar.on_clicked(self._aplicar)
        self.b_eliminar.on_clicked(self._eliminar)

        f.text(x0, 0.55, 'Método numérico', fontsize=12, weight='bold')
        ax = f.add_axes([x0, 0.465, 0.135, 0.075], frameon=True)
        self.r_metodo = RadioButtons(ax, ('RK4', 'Euler'))
        self.r_metodo.on_clicked(self._cambiar_metodo)
        self.b_reiniciar = Button(f.add_axes([x0 + 0.145, 0.48, 0.135, 0.05]), 'Reiniciar', color='#f3d6d6', hovercolor='#eab8b8')
        self.b_reiniciar.on_clicked(self._reiniciar)

        f.text(x0, 0.41, 'Cargas y mediciones', fontsize=12, weight='bold')
        self.t_info = f.text(x0, 0.39, '', fontsize=9.5, family='monospace', va='top')
        self.t_msg = f.text(0.05, 0.015, '', fontsize=10, color='#b00000')

    # ------------------------------------------------------------ dibujo
    def actualizar(self, recalcular=True):
        if recalcular:
            self._calcular()
        ax = self.ax
        ax.clear()
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_aspect('equal')
        ax.grid(True, color='0.9', lw=0.6)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'Líneas de campo eléctrico  ·  método {self.metodo}  ·  {len(self.lineas)} líneas', fontsize=11)
        for pts in self.lineas:
            ax.plot(pts[:, 0], pts[:, 1], color='0.25', lw=0.9)
            self._flechas(pts)
        pos, q = self.cargas.posiciones, self.cargas.valores
        for i in range(len(q)):
            c = ROJO if q[i] > 0 else AZUL
            ax.scatter(pos[i, 0], pos[i, 1], s=120 + 60 * min(abs(q[i]), 4), color=c, zorder=5,
                       edgecolor='k', lw=0.8)
            ax.text(pos[i, 0], pos[i, 1], '+' if q[i] > 0 else '−', color='w', ha='center', va='center',
                    fontsize=12, weight='bold', zorder=6)
            ax.annotate(f'q{i + 1} = {q[i]:+g} µC', (pos[i, 0], pos[i, 1]), xytext=(9, 9),
                        textcoords='offset points', fontsize=9, zorder=6)
            if i == self.sel:
                ax.scatter(pos[i, 0], pos[i, 1], s=520, facecolor='none', edgecolor='#e6a100', lw=2, zorder=4)
        if len(q) == 0:
            ax.text(0, 0, 'Haga clic en el plano para insertar una carga', ha='center', va='center',
                    fontsize=11, color='0.45')
        self._info()
        self.fig.canvas.draw_idle()

    def _flechas(self, pts):
        """Flechas de dirección a la mitad de la línea (y a 1/4 y 3/4 si es larga)."""
        largo = len(pts)
        marcas = [0.5] if largo < 120 else [0.25, 0.5, 0.75]
        for f in marcas:
            k = int(largo * f)
            if 1 <= k < largo - 1:
                a, b = pts[k - 1], pts[k + 1]
                if max(abs(b[0]), abs(b[1])) < LIM:
                    self.ax.annotate('', xy=b, xytext=a,
                                     arrowprops=dict(arrowstyle='-|>', color='0.25', lw=0.9, mutation_scale=11))

    def _calcular(self):
        pos, q = self.cargas.posiciones.copy(), self.cargas.valores.copy()
        t0 = time.perf_counter()
        self.lineas = generar_lineas(pos, q, self.metodo) if len(q) else []
        ms = (time.perf_counter() - t0) * 1000
        self.stats = {'ms': ms, 'pasos': sum(len(l) - 1 for l in self.lineas),
                      'kb_lineas': sum(l.nbytes for l in self.lineas) / 1024,
                      'rss': memoria_proceso()}

    def _info(self):
        pos, q = self.cargas.posiciones, self.cargas.valores
        filas = ['  #      x       y      q(µC)']
        for i in range(len(q)):
            m = '>' if i == self.sel else ' '
            filas.append(f'{m} q{i + 1}  {pos[i, 0] + 0:6.2f}  {pos[i, 1] + 0:6.2f}  {q[i]:+7.2f}')
        filas += [''] * (MAX_CARGAS + 1 - len(filas))
        s = self.stats
        filas += ['',
                  f'Cargas: {len(q)} de {MAX_CARGAS}',
                  f'Líneas: {len(self.lineas)}   Pasos: {s["pasos"]}',
                  f'Tiempo de cálculo: {s["ms"]:.1f} ms',
                  f'Puntos guardados: {s["kb_lineas"]:.1f} KB',
                  'Memoria del programa: ' + (f'{s["rss"]:.1f} MB' if s['rss'] else 'n/d (instale psutil)'),
                  f'Estructura cargas: {self.cargas.bytes_usados()} B']
        self.t_info.set_text('\n'.join(filas))

    # ------------------------------------------------------------ eventos
    def _aviso(self, txt=''):
        self.t_msg.set_text(txt)

    def _presionar(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None or ev.button != 1:
            return
        i = self.cargas.cercana(ev.xdata, ev.ydata, radio=0.3)
        if i is not None:
            self._seleccionar(i)
            self.arrastrando = True
            return
        try:
            mag = abs(float(self.t_mag.text.replace(',', '.')))
            if mag == 0:
                raise ValueError
        except ValueError:
            self._aviso('Escriba una magnitud válida (número distinto de 0).')
            self.fig.canvas.draw_idle()
            return
        q = mag if self.r_signo.value_selected.startswith('Positiva') else -mag
        try:
            i = self.cargas.agregar(round(ev.xdata, 2), round(ev.ydata, 2), q)
        except ValueError as e:
            self._aviso(str(e) + '. Elimine una o reinicie la simulación.')
            self.fig.canvas.draw_idle()
            return
        self._aviso()
        self._seleccionar(i, redibujar=False)
        self.actualizar()

    def _mover(self, ev):
        if not self.arrastrando or self.sel is None or ev.inaxes is not self.ax or ev.xdata is None:
            return
        self.cargas.modificar(self.sel, x=round(ev.xdata, 2), y=round(ev.ydata, 2))
        self._llenar_campos()
        self.actualizar(recalcular=False)   # mientras se arrastra solo se mueve la carga

    def _soltar(self, ev):
        if self.arrastrando:
            self.arrastrando = False
            self.actualizar()

    def _seleccionar(self, i, redibujar=True):
        self.sel = i
        self._llenar_campos()
        if redibujar:
            self.actualizar(recalcular=False)

    def _llenar_campos(self):
        if self.sel is None:
            self.t_sel.set_text('Ninguna (haga clic sobre una carga)')
            for t in (self.t_x, self.t_y, self.t_q):
                t.set_val('')
            return
        x, y, q = self.cargas.datos[self.sel]
        self.t_sel.set_text(f'q{self.sel + 1}')
        self.t_x.set_val(f'{x:g}')
        self.t_y.set_val(f'{y:g}')
        self.t_q.set_val(f'{q:g}')

    def _aplicar(self, _):
        if self.sel is None:
            self._aviso('Primero seleccione una carga haciendo clic sobre ella.')
            self.fig.canvas.draw_idle()
            return
        try:
            x, y, q = (float(t.text.replace(',', '.')) for t in (self.t_x, self.t_y, self.t_q))
            if abs(x) > LIM or abs(y) > LIM:
                raise ValueError(f'La posición debe estar entre -{LIM:g} y {LIM:g} m.')
            self.cargas.modificar(self.sel, x, y, q)
        except ValueError as e:
            msg = str(e)
            self._aviso(msg if ' ' in msg else 'Valores no válidos: x, y y q deben ser números (q distinto de 0).')
            self.fig.canvas.draw_idle()
            return
        self._aviso()
        self.actualizar()

    def _eliminar(self, _):
        if self.sel is None:
            return
        self.cargas.eliminar(self.sel)
        self.sel = None
        self._llenar_campos()
        self.actualizar()

    def _reiniciar(self, _):
        self.cargas.limpiar()
        self.sel = None
        self._llenar_campos()
        self._aviso()
        self.actualizar()

    def _cambiar_metodo(self, etiqueta):
        self.metodo = etiqueta
        self.actualizar()

    def mostrar(self):
        plt.show()
