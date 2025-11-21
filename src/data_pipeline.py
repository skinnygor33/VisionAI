import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)
    return df



def clean_data(df):
    # Ensure 'Time' and 'Amount' are numeric
    df["Time"] = pd.to_numeric(df['cd "Time"'], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    # Drop rows with invalid values (very few if any)
    df = df.dropna()

    return df

def feature_engineering(df):
    # 1. hour_of_day
    df["hour_of_day"] = (df['cd "Time"'] // 3600).astype(int)

    # 2. is_night (0 = no, 1 = yes)
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if h in [0,1,2,3,4,5] else 0)

    # 3. amount_zscore
    df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

    # 4. log_amount
    df["log_amount"] = np.log1p(df["Amount"])

    # 5. amount_percentile
    df["amount_percentile"] = df["Amount"].rank(pct=True)

    # 6. pca_sum: suma de V1...V28
    pca_cols = [col for col in df.columns if col.startswith("V")]
    df["pca_sum"] = df[pca_cols].sum(axis=1)

    # 7. pca_abs_sum
    df["pca_abs_sum"] = df[pca_cols].abs().sum(axis=1)

    # 8. amount_above_95th
    threshold_95 = df["Amount"].quantile(0.95)
    df["amount_above_95th"] = (df["Amount"] > threshold_95).astype(int)

    # 9–10. time cyclic encoding
    df["time_cycle_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["time_cycle_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    return df

def get_train_test(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def save_processed_data(df, path="Users/emiliamacarenarodriguezlavarriosarriaga/Desktop/AIFO/VisionAI/data/processed/clean_creditcard.csv"):
    df.to_csv(path, index=False)
