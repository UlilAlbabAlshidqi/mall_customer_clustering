import streamlit as st
import pandas as pd
import joblib

# === Load pipeline langsung ===
pipeline = joblib.load("gmm_pipeline.pkl")

st.title("Customer Segmentation with GMM (Pipeline Version)")

# === Input form ===
age = st.number_input("Age", min_value=10, max_value=100, value=30)
income = st.number_input("Annual Income (k$)", min_value=1, max_value=200, value=50)
score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict Cluster"):
    # Buat DataFrame sesuai kolom asli training
    X_new = pd.DataFrame([{
        "Age": age,
        "Annual Income": income,
        "Spending Score": score,
        "Gender": gender
    }])

    # Prediksi cluster langsung via pipeline
    label = pipeline.predict(X_new)[0]

    # ---- Mapping cluster -> deskripsi ----
    cluster_labels = {
        0: "Male - Older Low Spender",
        1: "Male - Young High Spender",
        2: "Female - Older Low Spender",
        3: "Female - Young High Spender"
    }
    cluster_text = cluster_labels.get(label, f"Cluster {label}")

    st.success(f"Pelanggan termasuk dalam cluster: {cluster_text} (Cluster #{label})")
