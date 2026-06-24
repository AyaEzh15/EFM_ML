import os
import joblib


def test_best_model_exists():
    assert os.path.exists("models/best_model.pkl")


def test_scaler_exists():
    assert os.path.exists("models/scaler.pkl")


def test_columns_exists():
    assert os.path.exists("models/columns.pkl")


def test_model_can_be_loaded():
    model = joblib.load("models/best_model.pkl")
    assert model is not None