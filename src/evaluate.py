import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing import prepare_data, split_and_scale


def evaluate_best_model():
    print("Chargement et préparation des données...")
    train_encoded, test_encoded = prepare_data()
    X_train, X_test, y_train, y_test = split_and_scale(train_encoded, test_encoded)

    print("Chargement du meilleur modèle...")
    model = joblib.load("models/best_model.pkl")

    print("Prédiction sur les données de test...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n==============================")
    print("Résultats du meilleur modèle")
    print("==============================")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Attack"],
        output_dict=True
    )

    print("\nClassification Report :")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

    os.makedirs("reports/figures", exist_ok=True)

    # 1. Sauvegarder les métriques principales
    metrics_df = pd.DataFrame([{
        "model": "Random Forest",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }])

    metrics_df.to_csv("reports/evaluation_metrics.csv", index=False)

    # 2. Sauvegarder le classification report détaillé
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("reports/classification_report.csv")

    # 3. Sauvegarder la matrice de confusion en CSV
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["True Normal", "True Attack"],
        columns=["Predicted Normal", "Predicted Attack"]
    )

    cm_df.to_csv("reports/confusion_matrix.csv")

    # 4. Sauvegarder la matrice de confusion en image
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Attack"]
    )

    display.plot()
    plt.title("Matrice de confusion - Random Forest")
    plt.savefig("reports/figures/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nFichiers générés :")
    print("- reports/figures/confusion_matrix.png")
    print("- reports/evaluation_metrics.csv")
    print("- reports/classification_report.csv")
    print("- reports/confusion_matrix.csv")


if __name__ == "__main__":
    evaluate_best_model()