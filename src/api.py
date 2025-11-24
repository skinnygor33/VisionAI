from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
#from src.data_pipeline import load_data, clean_data, feature_engineering, get_train_test, save_processed_data, normalize # type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUP_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "supervised_learning_model.pkl")
ANOM_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "isolation_forest.pkl")

MODEL_VERSION = "1.0.0"
app = FastAPI(title="Fraud Detection API")

# Transaction Schema
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

def clean_data(df):
    df['cd "Time"'] = pd.to_numeric(df['cd "Time"'], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(how='any', axis=0)
    return df

def feature_engineering(df):
    df["hour_of_day"] = (df['cd "Time"'] // 3600).astype(int)
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if h in [0,1,2,3,4,5] else 0)
    df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
    df["log_amount"] = np.log1p(df["Amount"])
    df["amount_percentile"] = df["Amount"].rank(pct=True)
    pca_cols = [f"V{i}" for i in range(1,29)]
    df["pca_sum"] = df[pca_cols].sum(axis=1)
    df["pca_abs_sum"] = df[pca_cols].abs().sum(axis=1)
    threshold_95 = df["Amount"].quantile(0.95)
    df["amount_above_95th"] = (df["Amount"] > threshold_95).astype(int)
    df["time_cycle_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["time_cycle_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    return df

def normalize(df, cols):
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

def preprocess_batch(df_batch):
    df_batch = clean_data(df_batch)
    df_batch = feature_engineering(df_batch)
    cols = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]
    df_batch = normalize(df_batch, cols)
    return df_batch

def apply_pca(df_batch, n_components=15):
    # Seleccionar todas las columnas finales que el modelo supervisado espera
    pca_cols = sup_features  # sup_features debe contener las ~40 columnas finales
    n_comp = min(n_components, df_batch.shape[0], len(pca_cols))
    pca = PCA(n_components=n_comp)
    return pca.fit_transform(df_batch)


def preprocess_supervised_batch(df_batch, n_components=15):
    df_batch = df_batch.reindex(columns=sup_features, fill_value=0)
    n_comp = min(n_components, df_batch.shape[0], df_batch.shape[1])
    pca = PCA(n_components=n_comp)
    df_pca = pca.fit_transform(df_batch)
    return df_pca


    
# Load Models Once at Startup
@app.on_event("startup")
def load_models():
    global sup_model, anom_model, sup_features

    sup_model = joblib.load(SUP_MODEL_PATH)
    anom_model = joblib.load(ANOM_MODEL_PATH)

    sup_features = None
    try:
        sup_features = sup_model.feature_names_in_.tolist()
    except:
        print("No se pudo extraer feature_names_in_ del modelo supervisado")

    print("Models loaded successfully.")
    print("Supervised model expects:", sup_features)


@app.post("/predict_batch")
def predict_batch(transactions: list[Transaction]):
    try:
        start = time.time()
        df_batch = pd.DataFrame([tx.dict() for tx in transactions])

        # Pipeline completo
        df_processed = preprocess_batch(df_batch)
        X_supervised = apply_pca(df_processed, n_components=15)
        proba = sup_model.predict_proba(X_supervised)[:,1]
        sup_flag = proba >= 0.5

        # Isolation Forest sobre features procesadas
        df_anom = df_processed[sup_features].copy()
        anom_label = anom_model.predict(df_anom)
        anom_flag = (anom_label==-1)
        anom_score = anom_model.score_samples(df_anom)
        anomaly_prob = anom_flag.astype(float)

        combined_risk = 0.75*proba + 0.25*anomaly_prob
        latency = time.time() - start

        results = [{"fraud_risk": float(combined_risk[i]),
                    "supervised_probability": float(proba[i]),
                    "supervised_flag": bool(sup_flag[i]),
                    "anomaly_flag": bool(anom_flag[i]),
                    "anomaly_score": float(anom_score[i])}
                for i in range(len(df_batch))]

        return {"results": results, "model_version": "1.0.0", "latency_seconds": latency}

    except Exception as e:
        import traceback
        print("\nERROR EN /predict_batch")
        print(e)
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/")
def root():
    return {"message": "Fraud Detection API is running", "version": MODEL_VERSION}
