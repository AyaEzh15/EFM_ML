import os
import joblib
import pandas as pd


MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
COLUMNS_PATH = "models/columns.pkl"


def load_artifacts():
    """
    Charger le modèle, le scaler et les colonnes.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Le fichier models/best_model.pkl est introuvable.")

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Le fichier models/scaler.pkl est introuvable.")

    if not os.path.exists(COLUMNS_PATH):
        raise FileNotFoundError("Le fichier models/columns.pkl est introuvable.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columns = joblib.load(COLUMNS_PATH)

    return model, scaler, columns


def predict_intrusion(input_data: dict):
    """
    Prédire si un trafic réseau est Normal ou Attack.
    """

    model, scaler, columns = load_artifacts()
    
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    df = df[columns]
    df_scaled = scaler.transform(df)
    prediction = int(model.predict(df_scaled)[0])

    if prediction == 0:
        return {
            "prediction": 0,
            "label": "Normal",
            "interpretation": "Le trafic réseau est considéré comme normal."
        }
    else:
        return {
            "prediction": 1,
            "label": "Attack",
            "interpretation": "Le trafic réseau est considéré comme une attaque potentielle."
        }