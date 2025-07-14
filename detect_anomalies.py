import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import joblib
from tensorflow.keras.models import load_model

# ------------------- Data Loading -------------------
def load_data(path):
    df = pd.read_csv(path)
    return df.dropna()

# ------------------- Isolation Forest -------------------
def detect_with_isolation_forest(df):
    os.makedirs('model', exist_ok=True)

    df_numeric = df.select_dtypes(include=[np.number])
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_numeric)

    model = IsolationForest(n_estimators=100, contamination=0.05)
    model.fit(df_scaled)

    df['anomaly'] = model.predict(df_scaled)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

    joblib.dump(model, 'model/isolation_forest_model.pkl')
    plot_anomaly_counts(df)
    
    # Optional: Scatter plot on first two numeric features
    numeric_cols = df_numeric.columns.tolist()
    if len(numeric_cols) >= 2:
        plot_feature_scatter(df, numeric_cols[0], numeric_cols[1])

    return df

# ------------------- Autoencoder -------------------
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation="relu")(input_layer)
    encoded = Dense(8, activation="relu")(encoded)
    decoded = Dense(16, activation="relu")(encoded)
    output_layer = Dense(input_dim, activation="linear")(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder

def detect_with_autoencoder(df):
    os.makedirs('model', exist_ok=True)

    df_numeric = df.select_dtypes(include=[np.number])
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_numeric)

    X_train, X_test = train_test_split(df_scaled, test_size=0.2, random_state=42)

    autoencoder = build_autoencoder(df_numeric.shape[1])
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

    reconstructions = autoencoder.predict(df_scaled)
    mse = np.mean(np.power(df_scaled - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 95)

    df['anomaly'] = (mse > threshold).astype(int)

    autoencoder.save('model/autoencoder_model.h5')
    plot_anomaly_counts(df)
    plot_mse_distribution(mse, threshold)

    numeric_cols = df_numeric.columns.tolist()
    if len(numeric_cols) >= 2:
        plot_feature_scatter(df, numeric_cols[0], numeric_cols[1])

    return df

# ------------------- Visualization Functions -------------------
def plot_anomaly_counts(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='anomaly', data=df)
    plt.title('Anomaly Counts')
    plt.xlabel('Class (0: Normal, 1: Anomaly)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('model/anomaly_counts.png')
    plt.close()

def plot_feature_scatter(df, x_col, y_col):
    if x_col in df.columns and y_col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue='anomaly', palette='coolwarm')
        plt.title(f'{x_col} vs {y_col} with Anomalies')
        plt.tight_layout()
        plt.savefig('model/feature_scatter.png')
        plt.close()

def plot_mse_distribution(mse, threshold):
    plt.figure(figsize=(6, 4))
    sns.histplot(mse, bins=50, kde=True)
    plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model/mse_distribution.png')
    plt.close()
