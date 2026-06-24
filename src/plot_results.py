import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_model_comparison():
    """
    Génère des graphiques comparatifs des modèles entraînés.
    Lit le fichier models/results.csv généré par src/train.py.
    """

    results_path = "models/results.csv"

    if not os.path.exists(results_path):
        raise FileNotFoundError(
            "Le fichier models/results.csv est introuvable. "
            "Lance d'abord : python src/train.py"
        )

    results = pd.read_csv(results_path)

    print("Résultats chargés :")
    print(results)

    os.makedirs("reports/figures", exist_ok=True)

    # Graphique Accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(results["model"], results["accuracy"])
    plt.title("Comparaison des modèles - Accuracy")
    plt.xlabel("Modèles")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("reports/figures/model_accuracy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Graphique Precision
    plt.figure(figsize=(10, 6))
    plt.bar(results["model"], results["precision"])
    plt.title("Comparaison des modèles - Precision")
    plt.xlabel("Modèles")
    plt.ylabel("Precision")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("reports/figures/model_precision_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Graphique Recall
    plt.figure(figsize=(10, 6))
    plt.bar(results["model"], results["recall"])
    plt.title("Comparaison des modèles - Recall")
    plt.xlabel("Modèles")
    plt.ylabel("Recall")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("reports/figures/model_recall_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Graphique F1-score
    plt.figure(figsize=(10, 6))
    plt.bar(results["model"], results["f1_score"])
    plt.title("Comparaison des modèles - F1-score")
    plt.xlabel("Modèles")
    plt.ylabel("F1-score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("reports/figures/model_f1_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nGraphiques générés :")
    print("- reports/figures/model_accuracy_comparison.png")
    print("- reports/figures/model_precision_comparison.png")
    print("- reports/figures/model_recall_comparison.png")
    print("- reports/figures/model_f1_comparison.png")


if __name__ == "__main__":
    plot_model_comparison()