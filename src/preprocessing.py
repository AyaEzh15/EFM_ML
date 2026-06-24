import os
import joblib
import pandas as pd

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


def load_data():
    """
    Charger les données NSL-KDD depuis GitHub.
    """
    print("Chargement du dataset train depuis GitHub...")
    train_df = pd.read_csv(URL_TRAIN, header=None)

    print("Chargement du dataset test depuis GitHub...")
    test_df = pd.read_csv(URL_TEST, header=None)

    train_df.columns = NSL_KDD_COLUMNS
    test_df.columns = NSL_KDD_COLUMNS

    print("Train chargé :", train_df.shape)
    print("Test chargé :", test_df.shape)

    return train_df, test_df


def clean_data(df):
    """
    Nettoyage + transformation de la cible.
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


def encode_data(train_df, test_df):
    """
    Encodage des variables catégorielles.
    On encode train et test ensemble pour avoir les mêmes colonnes.
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

    train_encoded = full_encoded[full_encoded["source"] == "train"].drop("source", axis=1)
    test_encoded = full_encoded[full_encoded["source"] == "test"].drop("source", axis=1)

    return train_encoded, test_encoded


def prepare_data():
    """
    Pipeline complet :
    - téléchargement train/test
    - nettoyage
    - encodage
    - sauvegarde CSV
    """
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    train_df, test_df = load_data()

    train_df.to_csv("data/raw/KDDTrain.csv", index=False)
    test_df.to_csv("data/raw/KDDTest.csv", index=False)

    print("Nettoyage des données...")
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    print("Distribution train :")
    print(train_df["label"].value_counts())

    print("Distribution test :")
    print(test_df["label"].value_counts())

    print("Encodage des données...")
    train_encoded, test_encoded = encode_data(train_df, test_df)

    train_encoded.to_csv("data/processed/train_processed.csv", index=False)
    test_encoded.to_csv("data/processed/test_processed.csv", index=False)

    print("Données sauvegardées :")
    print("- data/processed/train_processed.csv")
    print("- data/processed/test_processed.csv")

    return train_encoded, test_encoded


def split_and_scale(train_encoded, test_encoded):
    """
    Séparer X/y + normaliser avec StandardScaler.
    Le scaler est entraîné seulement sur train.
    """
    X_train = train_encoded.drop("label", axis=1)
    y_train = train_encoded["label"]

    X_test = test_encoded.drop("label", axis=1)
    y_test = test_encoded["label"]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(X_train.columns), "models/columns.pkl")

    print("Scaler sauvegardé dans models/scaler.pkl")
    print("Colonnes sauvegardées dans models/columns.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    train_encoded, test_encoded = prepare_data()
    split_and_scale(train_encoded, test_encoded)
    print("Prétraitement terminé avec succès.")