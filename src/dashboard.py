import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, confusion_matrix
import numpy as np
import io

# Configure the page
st.set_page_config(
    page_title="Fraud Detection Model Tester",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("Fraud Detection Model Tester")
st.markdown("""
Upload your test data and evaluate the fraud detection model's performance.
The app will send batch predictions to your API and display the results.
""")

# Sidebar for API configuration
st.sidebar.header("API Configuration")
api_url = st.sidebar.text_input("API URL", value="http://localhost:8000/predict_batch")
st.sidebar.markdown("---")
st.sidebar.info("""
**Note:** 
- Ensure your API is running at the specified URL
- CSV should contain transaction data with features (Time, Amount, V1-V28) and Class column
""")

# Main content area
st.header("1. Upload Test Data")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a CSV file", 
    type="csv",
    help="Should contain transaction data with Class column as target"
)
    
# Optional: Use default file
use_default = st.checkbox("Use default synthetic test file")
    
if use_default:
    try:
        df = pd.read_csv(r"C:\Users\valed\OneDrive\Desktop\VisionAI\VisionAI\data\test\synthetic_test_large.csv")
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        st.success("Loaded default test file successfully!")
    except Exception as e:
        st.error(f"Error loading default file: {e}")
        df = None
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = None
else:
    df = None
    
# Display data preview
if df is not None:
    st.subheader("Data Preview")
    st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns") 
        # Check for required columns
    required_cols = ["Class"]
    missing_cols = [col for col in required_cols if col not in df.columns]
        
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        st.dataframe(df.head(10), use_container_width=True)    
            # Show class distribution
        st.subheader("Class Distribution")
        class_counts = df["Class"].value_counts()
        st.write(f"Legitimate (0): {class_counts.get(0, 0)} transactions")
        st.write(f"Fraudulent (1): {class_counts.get(1, 0)} transactions")


st.header("2. Run Evaluation")
    
if df is not None and "Class" in df.columns:
    if st.button("Run Batch Prediction Test", type="primary"):
        with st.spinner("Sending requests to API and processing results..."):
            try:
                # Prepare data (same logic as your original code)
                y_true = df["Class"].copy()
                df_features = df.drop(columns=["Class"])
                    
                # Rename Time column if needed
                for col in df_features.columns:
                    if "Time" in col:
                        df_features = df_features.rename(columns={col: "cd_Time"})
                        break
                # Prepare payload
                transactions_payload = []
                for _, row in df_features.iterrows():
                    transaction = {
                        "cd_Time": float(row["cd_Time"]),
                        "Amount": float(row["Amount"]),
                        **{f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)}
                    }
                    transactions_payload.append(transaction)
                    
                # Send request to API
                response = requests.post(api_url, json=transactions_payload)
                    
                if response.status_code != 200:
                    st.error(f"API Error {response.status_code}: {response.text}")
                else:
                    try:
                        data = response.json()
                        if "results" not in data:
                            st.error("Key 'results' not found in API response")
                            st.write("Full response:", data)
                        else:
                            results = data["results"]
                            preds = [1 if r["fraud_risk"] >= 0.5 else 0 for r in results]
                            risk_scores = [r["fraud_risk"] for r in results]
                                
                            # Calculate metrics
                            recall = recall_score(y_true, preds)
                            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
                            FPR = fp / (fp + tn)
                            FNR = fn / (fn + tp)
                            accuracy = (tp + tn) / (tp + tn + fp + fn)
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                                
                            st.success("Evaluation completed!")
                                
                                # Metrics in columns
                            st.subheader("Performance Metrics")
                                
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                                
                            with metric_col1:
                                st.metric("Recall", f"{recall:.4f}")
                                st.metric("Accuracy", f"{accuracy:.4f}")
                                
                            with metric_col2:
                                st.metric("Precision", f"{precision:.4f}")
                                
                            with metric_col3:
                                st.metric("False Negative Rate", f"{FNR:.4f}")
                                st.metric("False Positive Rate", f"{FPR:.4f}")
                                
                            with metric_col4:
                                st.metric("True Positives", tp)
                                st.metric("False Positives", fp)
                                
                            # Confusion Matrix
                            st.subheader("Confusion Matrix")
                            cm = confusion_matrix(y_true, preds)
                            cm_df = pd.DataFrame(
                                cm, 
                                index=["Actual Legit", "Actual Fraud"], 
                                columns=["Predicted Legit", "Predicted Fraud"]
                            )
                            st.dataframe(cm_df.style.format("{:d}"), use_container_width=True)
                                
                            st.subheader("Risk Score Distribution")
                            fig_risk, ax = plt.subplots(figsize=(6, 4))
                            plt.style.use('dark_background')

                            risk_legit = [risk_scores[i] for i in range(len(risk_scores)) if y_true.iloc[i] == 0]
                            risk_fraud = [risk_scores[i] for i in range(len(risk_scores)) if y_true.iloc[i] == 1]
                            
                            ax.hist(risk_legit, bins=50, alpha=0.7, label='Legitimate', color='#4FC8E0', density=True)
                            ax.hist(risk_fraud, bins=50, alpha=0.7, label='Fraud', color='#604FE0', density=True)
                            ax.set_xlabel('Risk Score')
                            ax.set_ylabel('Density')
                            ax.set_title('Distribution of Risk Scores by True Class')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig_risk)

                                # Create results dataframe
                            df_out = df_features.copy()
                            df_out["true_label"] = y_true
                            df_out["predicted_label"] = preds
                            df_out["predicted_risk"] = risk_scores
                                
                                # Download section
                            st.subheader("Download Results")
                                
                                # Convert to CSV for download
                            csv = df_out.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions CSV",
                                data=csv,
                                file_name="shadow_predictions.csv",
                                mime="text/csv",
                                type="primary"
                            )
                                
                                # Show some predictions
                            st.subheader("Prediction Samples")
                            display_cols = ["cd_Time", "Amount", "true_label", "predicted_label", "predicted_risk"]
                            available_cols = [col for col in display_cols if col in df_out.columns]
                            st.dataframe(df_out[available_cols].head(15), use_container_width=True)
                                
                    except Exception as e:
                        st.error(f"Error processing API response: {e}")
                        st.text(f"Raw response: {response.text}")
                
            except Exception as e:
                st.error(f"Unexpected error: {e}")
    
    else:
        st.info("Please upload a CSV file with transaction data to begin testing or push the button to start the evaluation.")

# Footer
st.markdown("---")
st.markdown(
    "**Fraud Detection Model Tester** | "
    "Upload transaction data and evaluate model performance against your API"
)

