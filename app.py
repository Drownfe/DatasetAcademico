import os
import io
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_file
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

CSV_PATH = "dataset_notas.csv"
X_COLS = [
    "PromedioAcumulado", "AsistenciaPct", "HorasEstudioSem", "TareasEntregadasPct",
    "Parcial1", "Parcial2", "DificultadMateria", "IntentosReprobados"
]
Y_COLS = ["NotaFinal", "CreditosAsignados"]

np.random.seed(42)


# ----------------------- Generación del dataset -----------------------
def _calcular_nota_final(row):
    """Nota final ponderada (0-5) + ruido + penalizaciones por dificultad e intentos previos."""
    p1 = row["Parcial1"]
    p2 = row["Parcial2"]
    tareas = 0.5 * (row["HorasEstudioSem"] / 20) * 5      # normaliza 0-20h a 0-5 (~max 2.5)
    asistencia = (row["AsistenciaPct"] / 100) * 5         # 0-5
    base = 0.30 * p1 + 0.40 * p2 + 0.15 * tareas + 0.15 * asistencia

    dificultad = row["DificultadMateria"]                 # 1..5
    penal_dif = (dificultad - 3) * 0.15                   # ±0.3 aprox
    penal_reprob = row["IntentosReprobados"] * 0.10       # 0, 0.1, 0.2

    ruido = np.random.normal(0, 0.15)
    nota = base - penal_dif - penal_reprob + ruido
    return float(np.clip(nota, 1.0, 5.0))


def _asignar_creditos(prom_acum):
    """Política de créditos por promedio acumulado (0-5). Ajustable."""
    if prom_acum >= 4.5:  return 22
    if prom_acum >= 4.0:  return 20
    if prom_acum >= 3.7:  return 18
    if prom_acum >= 3.3:  return 16
    if prom_acum >= 3.0:  return 14
    if prom_acum >= 2.7:  return 12
    return 10


def _crear_dataset(path=CSV_PATH, n=10000):
    """Genera SIEMPRE un dataset nuevo con n filas."""
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
    return df


def _ensure_dataset(path=CSV_PATH, n=10000, force=False):
    """Crea dataset si no existe o si force=True o si el existente tiene menos filas que n."""
    if not os.path.exists(path):
        return _crear_dataset(path, n)
    try:
        df = pd.read_csv(path)
    except Exception:
        return _crear_dataset(path, n)

    if force or len(df) < n:
        return _crear_dataset(path, n)
    return df


# ----------------------- Métricas y evaluación -----------------------
def _metricas(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
    r2 = r2_score(y_true, y_pred)
    return dict(mse=float(mse), rmse=float(rmse), mae=float(mae), mape=float(mape), r2=float(r2))


def _evaluar_modelo(nombre, modelo, X, y):
    # Holdout
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo.fit(X_tr, y_tr)
    y_pred = modelo.predict(X_te)
    met = _metricas(y_te, y_pred)

    # K-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(modelo, X, y, scoring="r2", cv=kf)
    cv_mae = -cross_val_score(modelo, X, y, scoring="neg_mean_absolute_error", cv=kf)
    met.update({
        "cv_r2_mean": float(cv_r2.mean()), "cv_r2_std": float(cv_r2.std()),
        "cv_mae_mean": float(cv_mae.mean()), "cv_mae_std": float(cv_mae.std()),
    })

    # Muestras para gráficas
    idx = np.arange(len(y_te))
    if len(idx) > 1500:
        idx = np.random.choice(idx, size=1500, replace=False)
    y_true_s = np.array(y_te)[idx]
    y_pred_s = np.array(y_pred)[idx]
    resid = (y_true_s - y_pred_s).tolist()

    # Importancias / Coeficientes
    coefs = None
    importance = None
    if hasattr(modelo, "coef_"):
        coefs = {feat: float(c) for feat, c in zip(X.columns, np.ravel(modelo.coef_))}
    if hasattr(modelo, "feature_importances_"):
        importance = {feat: float(v) for feat, v in zip(X.columns, modelo.feature_importances_)}

    return {
        "name": nombre,
        "metrics": met,
        "y_true": [float(v) for v in y_true_s],
        "y_pred": [float(v) for v in y_pred_s],
        "residuals": [float(v) for v in resid],
        "coefs": coefs,
        "feature_importance": importance,
    }


def pipeline(n=10000, force=False):
    df = _ensure_dataset(CSV_PATH, n=n, force=force)
    X = df[X_COLS].copy()
    y = df["NotaFinal"].copy()

    # Baseline (promedio global)
    y_base = np.full_like(y, fill_value=y.mean(), dtype=float)
    base_metrics = _metricas(y, y_base)

    modelos = [
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge(alpha=1.0, random_state=42)),
        ("Lasso", Lasso(alpha=0.01, random_state=42, max_iter=5000)),
        ("RandomForest", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
    ]
    evaluaciones = [_evaluar_modelo(nm, mdl, X, y) for nm, mdl in modelos]

    # Correlaciones para heatmap
    corr_df = df[X_COLS + Y_COLS].corr().round(3)
    corr_labels = corr_df.columns.tolist()
    corr_values = corr_df.values.tolist()

    preview_X = X.head(12).round(2).to_dict(orient="records")
    preview_Y = df[Y_COLS].head(12).round(2).to_dict(orient="records")

    return {
        "dataset_info": {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "X_cols": X_COLS,
            "Y_cols": Y_COLS,
            "note": "NotaFinal escala 0–5. Créditos se asignan por política según PromedioAcumulado.",
        },
        "baseline": base_metrics,
        "models": evaluaciones,
        "corr": {"labels": corr_labels, "matrix": corr_values},
        "preview_X": preview_X,
        "preview_Y": preview_Y,
    }


# ----------------------- Rutas Flask -----------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start")
def start():
    try:
        n = int(request.args.get("n", 10000))
        force = request.args.get("force", "0") in ("1", "true", "True")
        res = pipeline(n=n, force=force)
        return jsonify({"ok": True, "result": res})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/download/dataset")
def download_dataset():
    n = int(request.args.get("n", 10000))
    force = request.args.get("force", "0") in ("1", "true", "True")
    _ensure_dataset(CSV_PATH, n=n, force=force)
    return send_file(CSV_PATH, as_attachment=True)


@app.route("/download/results")
def download_results():
    n = int(request.args.get("n", 10000))
    force = request.args.get("force", "0") in ("1", "true", "True")
    res = pipeline(n=n, force=force)
    buf = io.BytesIO(json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8"))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="resultados.json", mimetype="application/json")


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
