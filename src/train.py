

import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import mlflow.xgboost

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from preprocessing import prepare_data, split_and_scale


def train_models():
    """
    Entraîne plusieurs modèles ML :
    - Logistic Regression
    - Random Forest
    - SVM
    - XGBoost

    Sauvegarde le meilleur modèle selon le F1-score.
    """

    os.makedirs("models", exist_ok=True)

    print("Préparation des données...")
    train_encoded, test_encoded = prepare_data()
    X_train, X_test, y_train, y_test = split_and_scale(train_encoded, test_encoded)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "SVM": LinearSVC(
            random_state=42,
            max_iter=5000
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric="logloss",
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
            print(classification_report(
                y_test,
                y_pred,
                target_names=["Normal", "Attack"]
            ))

            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            if model_name == "XGBoost":
                mlflow.xgboost.log_model(
                    model,
                    name=model_name.replace(" ", "_")
                )
            else:
                mlflow.sklearn.log_model(
                    model,
                    name=model_name.replace(" ", "_")
                )

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

    best_model_info = {
        "best_model_name": best_model_name,
        "best_f1_score": best_f1
    }

    with open("models/best_model_info.json", "w", encoding="utf-8") as f:
        json.dump(best_model_info, f, indent=4)

    results_df = pd.DataFrame(results)
    results_df.to_csv("models/results.csv", index=False)

    print("\n==============================")
    print("Meilleur modèle :", best_model_name)
    print("Meilleur F1-score :", best_f1)
    print("Modèle sauvegardé dans models/best_model.pkl")
    print("Informations sauvegardées dans models/best_model_info.json")
    print("Résultats sauvegardés dans models/results.csv")
    print("==============================")


if __name__ == "__main__":
    train_models()

