# 🚨 Network Traffic Anomaly Detection

This project uses **unsupervised machine learning techniques** such as **Isolation Forests** and **Autoencoders** to detect anomalies in network traffic data. These anomalies may indicate potential **cybersecurity threats**, **system malfunctions**, or other unusual behaviors in the network.

---

## 📌 Features

- Upload your own network traffic CSV file or use a sample dataset.
- Choose between **Isolation Forest** or **Autoencoder** for anomaly detection.
- Visualize results with:
  - Anomaly count distribution
  - Feature scatter plots with anomalies highlighted
  - MSE distribution (for autoencoders)
- Download detection results as CSV.
- Pre-trained models are saved for reuse.

---

## 🧠 Algorithms Used

### 1. Isolation Forest
- Tree-based model to isolate anomalies.
- Ideal for high-dimensional numeric datasets.

### 2. Autoencoder (Deep Learning)
- Neural network trained to reconstruct input data.
- Anomalies are identified by high reconstruction error (MSE).

---

## 🧪 Dataset

- **Source**: `Dataset-Unicauca-Version2-87Atts.csv`
- Features: 87 attributes representing various network traffic characteristics.
- Type: Unlabeled, numeric data.

---

## 🚀 How to Run the App

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/network-anomaly-detector.git
   cd network-anomaly-detector
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
3. **Run the Streamlit App**
   ```bash
   streamlit run streamlit_app2.py

## 📂 Project Structure
  ```bash
├── detect_anomalies.py         # Core logic for anomaly detection and plotting
├── streamlit_app2.py           # Streamlit UI
├── Dataset-Unicauca-...csv     # Sample dataset
├── requirements.txt            # Python package requirements
└── model/
    ├── isolation_forest_model.pkl
    ├── autoencoder_model.h5
    ├── anomaly_counts.png
    ├── feature_scatter.png
    └── mse_distribution.png

##  📈 Output Visualizations

- 📊 **Anomaly Distribution**
- 🔍 **Feature-wise Scatter Plot**
- 📉 **MSE Distribution (Autoencoder only)**

---

## ✅ Future Improvements

- Add support for real-time anomaly detection using streaming data.
- Visualize feature importance for explainability.
- Support categorical features with encoding.
- Add model performance evaluation metrics.

---

## 🙌 Acknowledgments

- Dataset provided by Universidad del Cauca.
- Uses Keras, TensorFlow, scikit-learn, and Streamlit.
