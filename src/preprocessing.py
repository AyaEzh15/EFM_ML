import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


URL_TRAIN = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv"
URL_TEST = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.csv"


NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]


def create_directories():
    """
    Crée les dossiers nécessaires au pipeline.
    """
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)


def load_data():
    """
    Charge les données NSL-KDD depuis GitHub.
    """
    print("Chargement du dataset train depuis GitHub...")
    train_df = pd.read_csv(URL_TRAIN, header=None)

    print("Chargement du dataset test depuis GitHub...")
    test_df = pd.read_csv(URL_TEST, header=None)

    train_df.columns = NSL_KDD_COLUMNS
    test_df.columns = NSL_KDD_COLUMNS

    print("\nDimensions des données chargées :")
    print("Train chargé :", train_df.shape)
    print("Test chargé  :", test_df.shape)

    print("\nAperçu des premières lignes du train :")
    print(train_df.head())

    return train_df, test_df


def save_raw_data(train_df, test_df):
    """
    Sauvegarde les données brutes dans data/raw.
    """
    train_df.to_csv("data/raw/KDDTrain.csv", index=False)
    test_df.to_csv("data/raw/KDDTest.csv", index=False)

    print("\nDonnées brutes sauvegardées :")
    print("- data/raw/KDDTrain.csv")
    print("- data/raw/KDDTest.csv")


def clean_data(df):
    """
    Nettoie les données et transforme la cible en classification binaire.
    normal = 0
    attack = 1
    """
    df = df.copy()

    df = df.drop_duplicates()

    if "difficulty" in df.columns:
        df = df.drop("difficulty", axis=1)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    df["label"] = df["label"].apply(
        lambda x: 0 if str(x).lower().strip() == "normal" else 1
    )

    return df


def plot_class_distribution(train_df, test_df):
    """
    Génère un graphique de distribution des classes train/test.
    """
    train_counts = train_df["label"].value_counts().sort_index()
    test_counts = test_df["label"].value_counts().sort_index()

    labels = ["Normal", "Attack"]

    train_values = [
        train_counts.get(0, 0),
        train_counts.get(1, 0)
    ]

    test_values = [
        test_counts.get(0, 0),
        test_counts.get(1, 0)
    ]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 6))
    ax = plt.gca()

    bars_train = ax.bar(
        [i - width / 2 for i in x],
        train_values,
        width=width,
        label="Train",
        color="#4C78A8",
        edgecolor="black"
    )

    bars_test = ax.bar(
        [i + width / 2 for i in x],
        test_values,
        width=width,
        label="Test",
        color="#F28E2B",
        edgecolor="black"
    )

    for bars in [bars_train, bars_test]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 500,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

    ax.set_title(
        "Distribution des classes Normal et Attack",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel("Classe", fontsize=12)
    ax.set_ylabel("Nombre d'observations", fontsize=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend()

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        "reports/figures/class_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\nGraphique de distribution sauvegardé :")
    print("- reports/figures/class_distribution.png")


def encode_data(train_df, test_df):
    """
    Encode les variables catégorielles.
    Train et test sont encodés ensemble pour garantir les mêmes colonnes.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["source"] = "train"
    test_df["source"] = "test"

    full_df = pd.concat([train_df, test_df], axis=0)

    y = full_df["label"]
    X = full_df.drop(["label", "source"], axis=1)

    X_encoded = pd.get_dummies(X)

    full_encoded = X_encoded.copy()
    full_encoded["label"] = y
    full_encoded["source"] = full_df["source"].values

    train_encoded = full_encoded[
        full_encoded["source"] == "train"
    ].drop("source", axis=1)

    test_encoded = full_encoded[
        full_encoded["source"] == "test"
    ].drop("source", axis=1)

    return train_encoded, test_encoded


def prepare_data():
    """
    Pipeline de préparation :
    - chargement train/test
    - sauvegarde raw
    - nettoyage
    - distribution des classes
    - encodage
    - sauvegarde processed
    """
    create_directories()

    train_df, test_df = load_data()

    save_raw_data(train_df, test_df)

    print("\nNettoyage des données...")
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    print("\nDistribution train :")
    print(train_df["label"].value_counts())

    print("\nDistribution test :")
    print(test_df["label"].value_counts())

    plot_class_distribution(train_df, test_df)

    print("\nEncodage des données...")
    train_encoded, test_encoded = encode_data(train_df, test_df)

    train_encoded.to_csv("data/processed/train_processed.csv", index=False)
    test_encoded.to_csv("data/processed/test_processed.csv", index=False)

    print("\nDonnées traitées sauvegardées :")
    print("- data/processed/train_processed.csv")
    print("- data/processed/test_processed.csv")

    print("\nDimensions après encodage :")
    print("Train encoded :", train_encoded.shape)
    print("Test encoded  :", test_encoded.shape)

    return train_encoded, test_encoded


def split_and_scale(train_encoded, test_encoded):
    """
    Sépare X/y et normalise les variables avec StandardScaler.
    Le scaler est entraîné uniquement sur le train.
    """
    X_train = train_encoded.drop("label", axis=1)
    y_train = train_encoded["label"]

    X_test = test_encoded.drop("label", axis=1)
    y_test = test_encoded["label"]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(X_train.columns), "models/columns.pkl")

    print("\nArtefacts sauvegardés :")
    print("- models/scaler.pkl")
    print("- models/columns.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    train_encoded, test_encoded = prepare_data()
    split_and_scale(train_encoded, test_encoded)

    print("\nPrétraitement terminé avec succès.")