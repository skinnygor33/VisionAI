import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import joblib


def train_isolation_forest(X):
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        max_samples=0.7,
        random_state=42
    )
    iso.fit(X)
    return iso


def evaluate_isolation_forest(model, X, y_true):
    y_pred = model.predict(X)
    y_pred = (y_pred == -1).astype(int)
    print(classification_report(y_true, y_pred))
    return y_pred


def save_model(model, path="/Users/emiliamacarenarodriguezlavarriosarriaga/Desktop/AIFO/VisionAI/models/isolation_forest.pkl"):
    joblib.dump(model, path)