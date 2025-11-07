
# 🏡 Predicción de Casas — Proyecto Interactivo con Flask & Machine Learning

## 📘 Descripción General
Este proyecto demuestra un flujo completo de *Machine Learning* usando un **modelo de Regresión Lineal** para predecir el valor promedio de viviendas en California 🏠.  
La aplicación está construida con **Flask**, **pandas**, **scikit-learn** y **CSS animations**, ofreciendo una interfaz web moderna e interactiva que muestra cada paso del análisis y entrenamiento del modelo.

---

## 🚀 Funcionalidades
✅ Carga automática del dataset `california_housing.csv` (si no existe).  
✅ Procesamiento de datos y división en conjuntos de entrenamiento y prueba.  
✅ Entrenamiento de un modelo de **Regresión Lineal**.  
✅ Cálculo de métricas de rendimiento: **Error Cuadrático Medio (MSE)** y **Coeficiente de Determinación (R²)**.  
✅ Visualización de coeficientes del modelo.  
✅ Vista previa del dataset real.  
✅ Interfaz web con animaciones, colores, y pasos guiados.  

---

## 🧠 Tecnologías Utilizadas
- **Python 3.10+**
- **Flask** 🌐 (Backend y servidor web)
- **scikit-learn** 🤖 (Modelo predictivo)
- **pandas** 🧮 (Procesamiento de datos)
- **HTML + CSS + JS** 💅 (Interfaz animada)

---

## 🗂️ Estructura del Proyecto
```
📦 PrediccionCasas
├── app.py                  # Servidor Flask principal
├── requirements.txt        # Dependencias del proyecto
├── /templates
│   └── index.html          # Página principal con interfaz animada
├── /static
│   ├── style.css           # Estilos y animaciones
│   └── app.js              # Lógica interactiva en frontend
└── california_housing.csv  # Dataset generado automáticamente
```

---

## ⚙️ Instalación y Ejecución

### 1️⃣ Clonar el repositorio
```bash
git clone https://github.com/tuusuario/PrediccionCasas.git
cd PrediccionCasas
```

### 2️⃣ Crear entorno virtual
```bash
python -m venv venv
venv\Scripts\activate   # En Windows
source venv/bin/activate  # En Linux/Mac
```

### 3️⃣ Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4️⃣ Ejecutar servidor Flask
```bash
python app.py
```

### 5️⃣ Abrir en navegador
Ir a 👉 **http://127.0.0.1:5000**  
y presionar el botón **“Empezar”** ▶️ para ver el flujo completo del modelo.

---

## 📊 Explicación del Modelo
El modelo usa **Regresión Lineal** para predecir `MedHouseVal` (valor promedio de vivienda) a partir de variables numéricas:

| Variable (X) | Descripción |
|---------------|-------------|
| MedInc | Ingreso medio del vecindario (en decenas de miles de USD) |
| HouseAge | Edad promedio de las viviendas |
| AveRooms | Promedio de habitaciones por casa |
| AveOccup | Promedio de ocupantes por vivienda |

El resultado (`MedHouseVal`) se interpreta como el valor promedio de la vivienda en **cientos de miles de USD**.  
Ejemplo: una predicción de `3.95` equivale aproximadamente a **$395,000 USD**.

---

## ✨ Resultados Mostrados
- **📦 Dataset:** Total de filas, columnas y variables.  
- **🧠 Modelo:** Coeficientes, intercepto, MSE y R².  
- **👀 Vista previa:** Primeras filas del dataset con datos reales.  
- **✅ Animaciones:** Pasos del proceso con efectos visuales en tiempo real.

---

## 🧩 Créditos
Desarrollado por **Juan** ✨  
Proyecto educativo para visualizar la aplicación de *Machine Learning* con Python, Flask y scikit-learn.

---

## 📄 Licencia
Este proyecto es de uso educativo y libre. Puedes modificarlo y adaptarlo para tus propios experimentos con datasets o modelos.

