import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def add_value_labels(ax, bars):
    """
    Ajoute la valeur exacte au-dessus de chaque barre.
    """
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )


def plot_single_metric(results, metric, title, ylabel, output_path, colors):
    """
    Génère un graphique en barres pour une seule métrique.
    """
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    bars = ax.bar(
        results["model"],
        results[metric],
        color=colors,
        edgecolor="black",
        linewidth=0.8
    )

    add_value_labels(ax, bars)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Modèles", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 1.08)

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_metrics_grouped(results, output_path):
    """
    Génère un graphique groupé qui compare tous les modèles
    selon Accuracy, Precision, Recall et F1-score.
    """
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]

    x = np.arange(len(results["model"]))
    width = 0.18

    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759"]

    plt.figure(figsize=(13, 7))
    ax = plt.gca()

    for i, metric in enumerate(metrics):
        bars = ax.bar(
            x + (i - 1.5) * width,
            results[metric],
            width,
            label=metric_labels[i],
            color=colors[i],
            edgecolor="black",
            linewidth=0.6
        )

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90
            )

    ax.set_title(
        "Comparaison globale des performances des modèles",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Modèles", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.12)

    ax.set_xticks(x)
    ax.set_xticklabels(results["model"], rotation=20, ha="right")

    ax.legend(title="Métriques", loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_best_model_summary(results, output_path):
    """
    Génère un graphique spécifique pour mettre en évidence
    le meilleur modèle selon le F1-score.
    """
    best_model = results.loc[results["f1_score"].idxmax()]

    metrics = ["accuracy", "precision", "recall", "f1_score"]
    values = [
        best_model["accuracy"],
        best_model["precision"],
        best_model["recall"],
        best_model["f1_score"]
    ]

    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759"]

    plt.figure(figsize=(9, 6))
    ax = plt.gca()

    bars = ax.bar(
        labels,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.8
    )

    add_value_labels(ax, bars)

    ax.set_title(
        f"Performances du meilleur modèle : {best_model['model']}",
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.08)

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


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

    expected_columns = ["model", "accuracy", "precision", "recall", "f1_score"]
    missing_columns = [col for col in expected_columns if col not in results.columns]

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes dans models/results.csv : {missing_columns}"
        )

    results = results.sort_values(by="f1_score", ascending=False)

    print("Résultats chargés :")
    print(results)

    os.makedirs("reports/figures", exist_ok=True)

    model_colors = {
        "Logistic Regression": "#4C78A8",
        "Random Forest": "#59A14F",
        "SVM": "#F28E2B",
        "XGBoost": "#E15759"
    }

    colors = [
        model_colors.get(model, "#777777")
        for model in results["model"]
    ]

    plot_single_metric(
        results,
        metric="accuracy",
        title="Comparaison des modèles - Accuracy",
        ylabel="Accuracy",
        output_path="reports/figures/model_accuracy_comparison.png",
        colors=colors
    )

    plot_single_metric(
        results,
        metric="precision",
        title="Comparaison des modèles - Precision",
        ylabel="Precision",
        output_path="reports/figures/model_precision_comparison.png",
        colors=colors
    )

    plot_single_metric(
        results,
        metric="recall",
        title="Comparaison des modèles - Recall",
        ylabel="Recall",
        output_path="reports/figures/model_recall_comparison.png",
        colors=colors
    )

    plot_single_metric(
        results,
        metric="f1_score",
        title="Comparaison des modèles - F1-score",
        ylabel="F1-score",
        output_path="reports/figures/model_f1_comparison.png",
        colors=colors
    )

    plot_all_metrics_grouped(
        results,
        output_path="reports/figures/model_metrics_comparison_grouped.png"
    )

    plot_best_model_summary(
        results,
        output_path="reports/figures/best_model_summary.png"
    )

    print("\nGraphiques générés :")
    print("- reports/figures/model_accuracy_comparison.png")
    print("- reports/figures/model_precision_comparison.png")
    print("- reports/figures/model_recall_comparison.png")
    print("- reports/figures/model_f1_comparison.png")
    print("- reports/figures/model_metrics_comparison_grouped.png")
    print("- reports/figures/best_model_summary.png")


if __name__ == "__main__":
    plot_model_comparison()