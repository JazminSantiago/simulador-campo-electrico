# Simulador de líneas de campo eléctrico

Prototipo en Python que dibuja las líneas de campo eléctrico de hasta 5 cargas
puntuales. Las líneas se obtienen resolviendo numéricamente (RK4 o Euler) la
ecuación diferencial dr/ds = E/|E|.

## Requisitos
- Python 3.10 o superior
- Paquetes: numpy, matplotlib y psutil (psutil es opcional, solo muestra la memoria del programa)

## Instalación (Windows, en PowerShell dentro de esta carpeta)
    py -m pip install -r requirements.txt

## Ejecución
    py main.py

## Uso
| Acción | Cómo |
|---|---|
| Insertar carga | Elegir signo y magnitud (µC) en el panel y hacer clic en un lugar vacío del plano (máximo 5) |
| Seleccionar carga | Clic sobre la carga (queda marcada con un anillo) |
| Mover carga | Arrastrarla con el mouse; al soltar se recalculan las líneas |
| Modificar posición o magnitud | Seleccionar la carga, cambiar x, y o q y pulsar **Aplicar cambios** |
| Eliminar carga | Seleccionarla y pulsar **Eliminar carga** |
| Reiniciar | Botón **Reiniciar** (borra todas las cargas) |
| Cambiar método | Opciones **RK4** / **Euler** |

## Mediciones y validación
    py rendimiento.py     # tiempo y memoria en 4 escenarios -> resultados/mediciones.csv y gráficas
    py validacion.py      # error y orden de convergencia de Euler y RK4 -> resultados/validacion.csv y gráfica

## Estructura del código
| Archivo | Contenido |
|---|---|
| `main.py` | Punto de entrada |
| `cargas.py` | Estructura de datos de las cargas (arreglo NumPy de 5 × 3 preasignado) |
| `modelo.py` | Ley de Coulomb, EDO de las líneas de campo, métodos de Euler y RK4, generación de líneas |
| `interfaz.py` | Interfaz gráfica con matplotlib (plano, panel, eventos del mouse) |
| `rendimiento.py` | Medición de tiempo y memoria |
| `validacion.py` | Verificación del método numérico con una cantidad conservada del dipolo |
