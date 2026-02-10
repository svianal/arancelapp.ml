from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# =========================
# Load model and scalers
# =========================
MODEL_DIR = os.path.join(BASE_DIR, "model")

model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
scaler_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

# =========================
# Dropdown options
# =========================
tipo_pu_pr_options = ['PU', 'PR']

sector_options = sorted({
    f.replace('Sector_', '').strip()
    for f in feature_names
    if f.startswith('Sector_')
})

estado_options = sorted({
    f.replace('Estado_', '').strip()
    for f in feature_names
    if f.startswith('Estado_')
})

tipo_predio_options = sorted({
    f.replace('Tipo predio_', '').strip()
    for f in feature_names
    if f.startswith('Tipo predio_')
})


def predict_arancel(tipo_pu_pr, estado, tipo_predio, sector, terreno):
    input_vector = []

    for feature in feature_names:
        if feature.startswith('Sector_'):
            input_vector.append(1.0 if feature.replace('Sector_', '').strip() == sector else 0.0)

        elif feature.startswith('Estado_'):
            input_vector.append(1.0 if feature.replace('Estado_', '').strip() == estado else 0.0)

        elif feature.startswith('Tipo predio_'):
            input_vector.append(1.0 if feature.replace('Tipo predio_', '').strip() == tipo_predio else 0.0)

        elif feature == 'Tipo PU/PR':
            input_vector.append(1.0 if tipo_pu_pr == 'PU' else 0.0)

        else:
            input_vector.append(float(terreno))

    input_df = pd.DataFrame([input_vector], columns=feature_names)
    input_scaled = scaler_X.transform(input_df)

    y_pred_scaled = model.predict(input_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]

    return round(float(y_pred), 2)


# =========================
# Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    form_values = {
        "tipo_pu_pr": "PU",
        "estado": estado_options[0] if estado_options else "",
        "tipo_predio": tipo_predio_options[0] if tipo_predio_options else "",
        "sector": sector_options[0] if sector_options else "",
        "terreno": "0.79",
    }

    if request.method == "POST":
        tipo_pu_pr = request.form.get("tipo_pu_pr", "PU")
        estado = request.form.get("estado", form_values["estado"])
        tipo_predio = request.form.get("tipo_predio", form_values["tipo_predio"])
        sector = request.form.get("sector", form_values["sector"])
        terreno_raw = request.form.get("terreno", form_values["terreno"])

        form_values.update(
            {
                "tipo_pu_pr": tipo_pu_pr,
                "estado": estado,
                "tipo_predio": tipo_predio,
                "sector": sector,
                "terreno": terreno_raw,
            }
        )

        try:
            terreno = float(terreno_raw)
            prediction = predict_arancel(
                tipo_pu_pr,
                estado,
                tipo_predio,
                sector,
                terreno,
            )
        except ValueError:
            error = "El valor de terreno debe ser numerico."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form_values=form_values,
        tipo_pu_pr_options=tipo_pu_pr_options,
        estado_options=estado_options,
        tipo_predio_options=tipo_predio_options,
        sector_options=sector_options
    )


if __name__ == "__main__":
    app.run(debug=True)
