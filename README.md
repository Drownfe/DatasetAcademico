
# 🎓 Predicción Académica — Notas y Créditos

Este proyecto implementa un modelo de **Regresión Lineal** que analiza un dataset académico generado automáticamente para predecir la **nota final (0–5)** de un estudiante y estimar cuántos **créditos** puede matricular según su rendimiento.

---

## 🧠 Descripción del Proyecto

El sistema genera un dataset simulado con variables académicas realistas:

- `PromedioAcumulado` (promedio de semestres previos)
- `AsistenciaPct` (porcentaje de asistencia)
- `HorasEstudioSem` (horas de estudio semanales)
- `TareasEntregadasPct` (porcentaje de tareas entregadas)
- `Parcial1`, `Parcial2`
- `DificultadMateria` (nivel 1 a 5)
- `IntentosReprobados`
- `NotaFinal` (calculada por el modelo)
- `CreditosAsignados` (según política académica)

El modelo se entrena para predecir la **NotaFinal** a partir de las demás variables, mientras que los **CréditosAsignados** se determinan con base en el promedio acumulado.

---

## 📊 Variables

| Tipo | Variable | Descripción |
|------|-----------|-------------|
| 🎯 Y (Salida) | `NotaFinal` | Predicción de la nota final (escala 0–5) |
| 🎯 Y (Salida) | `CreditosAsignados` | Créditos recomendados según desempeño |
| 🔢 X (Entrada) | `PromedioAcumulado` | Promedio general del estudiante |
| 🔢 X (Entrada) | `AsistenciaPct` | Porcentaje de asistencia |
| 🔢 X (Entrada) | `HorasEstudioSem` | Horas de estudio semanales |
| 🔢 X (Entrada) | `TareasEntregadasPct` | Porcentaje de tareas entregadas |
| 🔢 X (Entrada) | `Parcial1`, `Parcial2` | Calificaciones de parciales |
| 🔢 X (Entrada) | `DificultadMateria` | Dificultad del curso (1–5) |
| 🔢 X (Entrada) | `IntentosReprobados` | Veces que reprobó previamente |

---

## ⚙️ Tecnologías Usadas

- **Python 3.11+**
- **Flask** — Interfaz web interactiva
- **Pandas / NumPy** — Manipulación de datos
- **scikit-learn** — Entrenamiento del modelo de regresión
- **HTML + CSS + JS** — Visualización dinámica y animaciones

---

## 💻 Cómo ejecutar

```bash
# 1️⃣ Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # (en Windows)

# 2️⃣ Instalar dependencias
pip install -r requirements.txt

# 3️⃣ Ejecutar el servidor Flask
python app.py
```

Abre en el navegador 👉 **http://127.0.0.1:5000**  
Presiona **“Empezar”** para ejecutar el pipeline completo.

---

## 🚀 Funcionamiento del Pipeline

1️⃣ Genera o carga automáticamente un dataset académico.  
2️⃣ Separa variables **X** (entradas) y **Y** (salidas).  
3️⃣ Divide los datos en entrenamiento y prueba (80/20).  
4️⃣ Entrena un modelo de regresión lineal.  
5️⃣ Muestra métricas de rendimiento (MSE, RMSE, R²).  
6️⃣ Visualiza las primeras filas de **X** y **Y** en tablas interactivas.  

---

## 🧾 Ejemplo de salida

**Métricas del modelo:**
- MSE: `0.52`
- RMSE: `0.72`
- R²: `0.61`

**Ejemplo de predicción:**  
> Un estudiante con 90% de asistencia, 12 horas de estudio y promedio 3.9 puede obtener una nota final estimada de **4.2**, con asignación de **18 créditos**.

---

## 🎨 Interfaz visual

El frontend muestra pasos progresivos del análisis con animaciones y emojis:  
📦 Generación del dataset → 🧩 Separación de variables → ⚙️ Entrenamiento → 📊 Resultados finales.

---

## 🏁 Autor

Desarrollado por **Juan Felipe Hernández Palacio (Drownfe)**  
💙 Proyecto académico — Politécnico Colombiano Jaime Isaza Cadavid  
📚 Materia: *Inteligencia Artificial / Minería de Datos*
