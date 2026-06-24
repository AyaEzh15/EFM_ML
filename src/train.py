import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from preprocessing import prepare_data, split_and_scale


def train_models():
    """
    Entraîne plusieurs modèles ML et sauvegarde le meilleur modèle.
    """

    os.makedirs("models", exist_ok=True)

    print("Préparation des données...")
    train_encoded, test_encoded = prepare_data()
    X_train, X_test, y_train, y_test = split_and_scale(train_encoded, test_encoded)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

    mlflow.set_experiment("MLOps_IDS_Cybersecurity")

    best_model = None
    best_model_name = ""
    best_f1 = 0

    results = []

    for model_name, model in models.items():
        print("\n==============================")
        print(f"Entraînement du modèle : {model_name}")
        print("==============================")

        with mlflow.start_run(run_name=model_name):

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print(f"Accuracy  : {accuracy:.4f}")
            print(f"Precision : {precision:.4f}")
            print(f"Recall    : {recall:.4f}")
            print(f"F1-score  : {f1:.4f}")

            print("\nClassification report :")
            print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(model, model_name.replace(" ", "_"))

            results.append({
                "model": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = model_name

    joblib.dump(best_model, "models/best_model.pkl")

    results_df = pd.DataFrame(results)
    results_df.to_csv("models/results.csv", index=False)

    print("\n==============================")
    print("Meilleur modèle :", best_model_name)
    print("Meilleur F1-score :", best_f1)
    print("Modèle sauvegardé dans models/best_model.pkl")
    print("Résultats sauvegardés dans models/results.csv")
    print("==============================")


if __name__ == "__main__":
    train_models()