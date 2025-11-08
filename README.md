
# 🎓 Predicción Académica — Notas y Créditos (Versión Avanzada)

Aplicación web **Flask + scikit-learn + Chart.js** para **regresión académica** con datasets grandes.  
Compara varios modelos, ejecuta **validación cruzada**, calcula **métricas avanzadas** y muestra **gráficas** interactivas.  
Incluye **descargas** del dataset y resultados.

---

## 🚀 Novedades de esta versión

- **Dataset grande y paramétrico**: genera 10k, 15k, 20k+ filas realistas (selector en la UI).
- **Re-creación bajo demanda**: casilla *Re-crear dataset* para regenerar con el tamaño elegido.
- **Más métricas**: `MAE`, `RMSE`, `MAPE%`, `R²` y baseline por promedio.
- **Validación cruzada (K-Fold, k=5)**: reporte `media ± desviación` para `R²` y `MAE`.
- **Comparación de modelos**: `LinearRegression`, `Ridge`, `Lasso`, `RandomForest`, `GradientBoosting`.
- **Gráficas nuevas**:
  - **Barras**: R² por modelo (comparación).
  - **Barras**: Importancia de variables o coeficientes del mejor modelo.
  - **Dispersión**: Real vs. Predicho (con miles de puntos muestreados).
  - **Histograma**: Distribución de residuales.
  - **Heatmap**: Matriz de correlación entre X y Y (tabla coloreada).
- **Descargas**: botones para **CSV** del dataset y **JSON** con todos los resultados.

---

## 📦 Dataset

Se genera de forma **sintética pero realista** con estas columnas:

**Entradas (X):**
- `PromedioAcumulado` (0–5)
- `AsistenciaPct` (50–100%)
- `HorasEstudioSem` (0–25 h)
- `TareasEntregadasPct` (30–100%)
- `Parcial1`, `Parcial2` (1–5)
- `DificultadMateria` (1–5)
- `IntentosReprobados` (0–2)

**Salidas (Y):**
- `NotaFinal` (0–5) — *objetivo de la regresión*
- `CreditosAsignados` — determinada por política según `PromedioAcumulado`

> La **NotaFinal** se construye con una fórmula ponderada + ruido + penalizaciones por dificultad e intentos previos.

---

## 🧠 Modelos y Métricas

Modelos evaluados automáticamente:
- `LinearRegression`
- `Ridge (α=1.0)`
- `Lasso (α=0.01)`
- `RandomForestRegressor (n_estimators=200)`
- `GradientBoostingRegressor`

Métricas reportadas por modelo (holdout 80/20):
- **MAE** (error absoluto medio)
- **RMSE** (raíz del MSE)
- **MAPE%** (error porcentual absoluto medio)
- **R²** (explicación de la varianza)

Adicionalmente, se calcula:
- **Baseline** (predecir el promedio de `NotaFinal`) para comparar.
- **K-Fold CV (k=5)**: `R²` y `MAE` con **media ± desviación estándar**.

---

## 📊 Visualizaciones

1. **Comparación de modelos (R²)** — *Barras.*  
   Muestra qué modelo generaliza mejor (más alto es mejor).
2. **Importancia/Coeficientes (mejor modelo)** — *Barras.*  
   - Si el mejor modelo es no lineal (p. ej., RandomForest), muestra `feature_importances_`.
   - Si es lineal (p. ej., Ridge), muestra los **coeficientes**.
3. **Real vs. Predicho** — *Dispersión.*  
   Nube de puntos con muestra de hasta 1500 observaciones; idealmente cerca de la línea `y = x`.
4. **Distribución de residuales** — *Histograma.*  
   Permite ver si los errores se concentran alrededor de 0 (buena señal).
5. **Matriz de correlación** — *Heatmap (tabla coloreada).*  
   Observa relaciones entre todas las variables X y Y (positivo en azul, negativo en rojo).
6. **Vistas previas X / Y** — *Tablas.*  
   Primeras filas de **entradas** (X) y **salidas** (Y).

---

## 🌐 Endpoints y Descargas

- **App**: `GET /`  
  UI con selector de tamaño, botón para correr el pipeline y visualizaciones.
- **Iniciar pipeline**: `GET /start?n=10000&force=0|1`  
  - `n`: tamaño solicitado del dataset.
  - `force=1`: regenera el CSV, aunque exista.
- **Descargar dataset**: `GET /download/dataset?n=10000&force=0|1` → `dataset_notas.csv`
- **Descargar resultados**: `GET /download/results?n=10000&force=0|1` → `resultados.json`

> Los enlaces de descarga también están disponibles como botones en la interfaz.

---

## 💻 Instalación y ejecución

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

python app.py
```

Abrir **http://127.0.0.1:5000**.  
Elegir tamaño (*10k por defecto*), marcar **Re-crear** si se desea regenerar, y pulsar **▶️ Empezar**.

---

## 🧩 Cómo leer los resultados

- **Baseline**: sirve de referencia mínima; cualquier modelo útil debe superarlo.
- **Comparación (R²)**: elige el modelo con R² más alto sin sacrificar mucho MAE/MAPE.
- **Importancia**: identifica qué variables pesan más en la predicción (útil para recomendaciones).
- **Real vs. Predicho**: puntos cercanos a la diagonal indican buena precisión.
- **Residuales**: distribución centrada en 0 y relativamente estrecha = mejor ajuste.
- **Correlaciones**: verifica relaciones lineales fuertes y multicolinealidad potencial.

---

## 📝 Estructura del proyecto

```
tu_carpeta/
├─ app.py                # Backend (Flask + scikit-learn)
├─ requirements.txt      # Dependencias
├─ templates/
│  └─ index.html         # UI principal + contenedores de gráficos
└─ static/
   ├─ style.css          # Estilos
   └─ app.js             # Lógica de front + Chart.js
```

---

## 🏁 Autor

**Juan Felipe Hernández Palacio (Drownfe)**  
Proyecto académico — *Predicción de Nota Final y Créditos con ML*.
