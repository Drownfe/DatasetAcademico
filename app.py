import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

CSV_PATH = "dataset_notas.csv"
# X (entradas) y Y (salidas)
X_COLS = [
    "PromedioAcumulado", "AsistenciaPct", "HorasEstudioSem", "TareasEntregadasPct",
    "Parcial1", "Parcial2", "DificultadMateria", "IntentosReprobados"
]
Y_COLS = ["NotaFinal", "CreditosAsignados"]

np.random.seed(42)


# ----------------------- Generación del dataset -----------------------
def _calcular_nota_final(row):
    """
    Nota final ponderada (0-5) con ruido realista + penalizaciones por dificultad e intentos previos.
    """
    p1 = row["Parcial1"]        # 0-5
    p2 = row["Parcial2"]        # 0-5
    tareas = 0.5 * (row["HorasEstudioSem"] / 20) * 5     # normaliza 0-20h a 0-5 (~max 2.5)
    asistencia = (row["AsistenciaPct"] / 100) * 5        # 0-5
    base = 0.30 * p1 + 0.40 * p2 + 0.15 * tareas + 0.15 * asistencia

    dificultad = row["DificultadMateria"]                 # 1..5
    penal_dif = (dificultad - 3) * 0.15                  # ±0.3 aprox
    penal_reprob = row["IntentosReprobados"] * 0.10      # 0, 0.1, 0.2

    ruido = np.random.normal(0, 0.15)
    nota = base - penal_dif - penal_reprob + ruido
    return float(np.clip(nota, 1.0, 5.0))


def _asignar_creditos(prom_acum):
    """
    Política de créditos por promedio acumulado (0-5).
    Ajusta aquí si tu institución usa otros topes.
    """
    if prom_acum >= 4.5:  return 22
    if prom_acum >= 4.0:  return 20
    if prom_acum >= 3.7:  return 18
    if prom_acum >= 3.3:  return 16
    if prom_acum >= 3.0:  return 14
    if prom_acum >= 2.7:  return 12
    return 10


def _crear_dataset_si_no_existe(path=CSV_PATH, n=220):
    if os.path.exists(path):
        return

    prom_acum       = np.clip(np.random.normal(3.6, 0.5, n), 2.0, 5.0)
    asistencia_pct  = np.clip(np.random.normal(85, 10, n), 50, 100)
    horas_estudio   = np.clip(np.random.normal(10, 4, n), 0, 25)
    tareas_pct      = np.clip(np.random.normal(80, 15, n), 30, 100)
    parcial1        = np.clip(np.random.normal(3.6, 0.7, n), 1.0, 5.0)
    parcial2        = np.clip(np.random.normal(3.7, 0.7, n), 1.0, 5.0)
    dificultad      = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    reprob_prev     = np.random.choice([0, 1, 2], size=n, p=[0.75, 0.2, 0.05])

    df = pd.DataFrame({
        "PromedioAcumulado":    np.round(prom_acum, 2),
        "AsistenciaPct":        np.round(asistencia_pct, 1),
        "HorasEstudioSem":      np.round(horas_estudio, 1),
        "TareasEntregadasPct":  np.round(tareas_pct, 1),
        "Parcial1":             np.round(parcial1, 2),
        "Parcial2":             np.round(parcial2, 2),
        "DificultadMateria":    dificultad.astype(int),
        "IntentosReprobados":   reprob_prev.astype(int),
    })

    df["NotaFinal"] = df.apply(_calcular_nota_final, axis=1)
    df["CreditosAsignados"] = df["PromedioAcumulado"].apply(_asignar_creditos)

    df.to_csv(path, index=False)
    print(f"✅ Dataset creado: {path} ({len(df)} filas)")


# ----------------------- Pipeline de entrenamiento -----------------------
def pipeline():
    _crear_dataset_si_no_existe(CSV_PATH)

    df = pd.read_csv(CSV_PATH)

    # Validación columnas
    faltantes = [c for c in X_COLS + Y_COLS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en CSV: {faltantes}")

    X = df[X_COLS].copy()
    Y = df[Y_COLS].copy()

    # Modelo de regresión para NotaFinal
    y_nota = Y["NotaFinal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_nota, test_size=0.2, random_state=42
    )
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    # Coeficientes por feature (impacto sobre NotaFinal)
    coefs = {feat: float(c) for feat, c in zip(X.columns, modelo.coef_)}

    # Vista previa (X e Y por separado)
    preview_X = X.head(10).round(2).to_dict(orient="records")
    preview_Y = Y.head(10).round(2).to_dict(orient="records")

    result = {
        "dataset_info": {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "X_cols": X_COLS,
            "Y_cols": Y_COLS,
            "policy_credits": [
                ">= 4.5 ➜ 22 créditos",
                ">= 4.0 ➜ 20",
                ">= 3.7 ➜ 18",
                ">= 3.3 ➜ 16",
                ">= 3.0 ➜ 14",
                ">= 2.7 ➜ 12",
                "<  2.7 ➜ 10",
            ],
            "note": "NotaFinal está en escala 0–5. Los créditos dependen del PromedioAcumulado.",
        },
        "metrics": {
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "intercept": float(modelo.intercept_),
            "coefs": coefs,
        },
        "preview_X": preview_X,
        "preview_Y": preview_Y,
    }
    return result


# ----------------------- Rutas Flask -----------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start")
def start():
    try:
        res = pipeline()
        return jsonify({"ok": True, "result": res})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
