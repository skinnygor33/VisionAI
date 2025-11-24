import pandas as pd
import requests
from sklearn.metrics import recall_score, confusion_matrix
import numpy as np

API_URL = "http://localhost:8000/predict_batch"

df = pd.read_csv("../data/test/synthetic_test_large.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

for col in df.columns:
    if "Time" in col:
        df = df.rename(columns={col: "cd_Time"})
        break

y_true = df["Class"].copy()
df_features = df.drop(columns=["Class"])

# Payload
transactions_payload = []
for _, row in df_features.iterrows():
    transaction = {
        "cd_Time": float(row["cd_Time"]),
        "Amount": float(row["Amount"]),
        **{f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)}
    }
    transactions_payload.append(transaction)

response = requests.post(API_URL, json=transactions_payload)
if response.status_code != 200:
    print("API error:", response.status_code, response.text)
    exit()

try:
    data = response.json()
except Exception as e:
    print("Error parsing JSON:", e)
    print("RAW RESPONSE:", response.text)
    exit()

if "results" not in data:
    print("Key 'results' not found in API response.")
    print("Full response:", data)
    exit()

results = data["results"]
preds = [1 if r["fraud_risk"] >= 0.5 else 0 for r in results]
risk_scores = [r["fraud_risk"] for r in results]

recall = recall_score(y_true, preds)
tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
FPR = fp / (fp + tn)
FNR = fn / (fn + tp)

print("\n--- SHADOW TEST RESULTS ---")
print("Recall:", recall)
print("FPR:", FPR)
print("FNR:", FNR)

df_out = df_features.copy()
df_out["true_label"] = y_true
df_out["predicted_label"] = preds
df_out["predicted_risk"] = risk_scores
df_out.to_csv("../data/test/shadow_predictions.csv", index=False)
print("Saved shadow_predictions.csv")
