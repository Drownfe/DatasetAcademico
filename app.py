import os
import pandas as pd
from flask import Flask, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import numpy as np

app = Flask(__name__)

CSV_PATH = "california_housing.csv"
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveOccup"]
TARGET = "MedHouseVal"
USD_MULT = 100_000  # MedHouseVal está en cientos de miles de USD → multiplicar por 100,000


def asegurar_dataset(csv_path: str = CSV_PATH):
    """Crea california_housing.csv si no existe usando sklearn.datasets."""
    if os.path.exists(csv_path):
        return
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.to_csv(csv_path, index=False)


def pipeline_entrenamiento():
    """Carga datos, entrena regresión lineal y devuelve métricas + preview (con USD)."""
    asegurar_dataset(CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    # Validación básica de columnas
    faltantes = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas faltantes en el CSV: {faltantes}")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    rmse_usd = float(rmse * USD_MULT)  # error típico en dólares

    # Coeficientes por feature
    coef_por_feature = {feature: float(coef) for feature, coef in zip(FEATURES, modelo.coef_)}

    # Vista previa + columna convertida a USD
    preview_df = df.head(8).copy()
    preview_df["MedHouseVal_USD"] = (preview_df["MedHouseVal"] * USD_MULT).round(0).astype(int)
    preview = preview_df.round(3).to_dict(orient="records")

    resultado = {
        "dataset_info": {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "features": FEATURES,
            "target": TARGET,
            "units_note": (
                "MedHouseVal está en cientos de miles de USD. "
                "Se añade la columna MedHouseVal_USD con el valor convertido a dólares."
            ),
        },
        "metrics": {
            "mse_native": mse,
            "r2": r2,
            "rmse_usd": rmse_usd,
            "intercept": float(modelo.intercept_),
            "coefs": coef_por_feature,
        },
        "preview": preview,
    }
    return resultado


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start")
def start():
    try:
        result = pipeline_entrenamiento()
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
