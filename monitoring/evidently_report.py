import os
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


def generate_drift_report():
    """
    Génère un rapport de data drift avec Evidently AI.
    On compare les données d'entraînement avec les données de test.
    """

    train_path = "data/processed/train_processed.csv"
    test_path = "data/processed/test_processed.csv"

    if not os.path.exists(train_path):
        raise FileNotFoundError("Le fichier data/processed/train_processed.csv est introuvable.")

    if not os.path.exists(test_path):
        raise FileNotFoundError("Le fichier data/processed/test_processed.csv est introuvable.")

    print("Chargement des données...")
    reference_data = pd.read_csv(train_path)
    current_data = pd.read_csv(test_path)

    # Pour accélérer la génération du rapport
    reference_data = reference_data.sample(n=5000, random_state=42)
    current_data = current_data.sample(n=5000, random_state=42)

    os.makedirs("monitoring/reports", exist_ok=True)

    print("Génération du rapport Evidently AI...")

    report = Report([
        DataDriftPreset()
    ])

    my_eval = report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    my_eval.save_html("monitoring/reports/evidently_data_drift_report.html")

    print("Rapport généré avec succès :")
    print("monitoring/reports/evidently_data_drift_report.html")


if __name__ == "__main__":
    generate_drift_report()