import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from detect_anomalies import load_data, detect_with_isolation_forest, detect_with_autoencoder

st.set_page_config(layout="wide")
st.title("🚨 Network Traffic Anomaly Detection")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload CSV File", type="csv")

# Load default CSV if none is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.info("✅ Using uploaded file.")
else:
    df = pd.read_csv("synthetic_network_traffic.csv", nrows=1000)
    st.warning("⚠️ No file uploaded. Using sample dataset.")

# Preview
st.write("📊 Preview of Data", df.head())

# Select Method
method = st.selectbox("🧠 Choose Detection Method", ["Isolation Forest", "Autoencoder"])

# Run detection
if st.button("🚀 Run Detection"):
    if method == "Isolation Forest":
        result_df = detect_with_isolation_forest(df.copy())
    else:
        result_df = detect_with_autoencoder(df.copy())

    st.success("✅ Detection Completed")
    st.dataframe(result_df)

    # Show graphs
    st.subheader("📈 Visualizations")

    st.image("model/anomaly_counts.png", caption="Anomaly Distribution", use_column_width=True)

    try:
        st.image("model/feature_scatter.png", caption="Feature Scatter with Anomalies", use_column_width=True)
    except:
        st.warning("Feature scatter plot not available.")

    if method == "Autoencoder":
        try:
            st.image("model/mse_distribution.png", caption="MSE Distribution (Autoencoder)", use_column_width=True)
        except:
            st.warning("MSE distribution plot not available.")

    # Download button
    st.download_button("📥 Download Results as CSV", result_df.to_csv(index=False), file_name="anomaly_results.csv")
