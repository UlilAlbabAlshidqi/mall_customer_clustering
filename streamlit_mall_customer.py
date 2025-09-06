import streamlit as st
import pandas as pd
import joblib

# ===== Load model GMM =====
gmm_model = joblib.load('gmm_model.joblib')

# ===== Mapping cluster ke label teks =====
cluster_labels = {
    0: "Male - Older Low Spender",
    1: "Male - Young High Spender",
    2: "Female - Older Low Spender",
    3: "Female - Young High Spender"
}

# ===== Mean/Std/IQR dari training (manual scaling) =====
# Sesuaikan ini dengan dataset trainingmu
age_mean, age_std = 43.0, 13.0
score_mean, score_std = 50.0, 25.0
income_median, income_iqr = 60.0, 20.0

# ===== Urutan kolom feature saat training =====
feature_columns = ['Age', 'Spending Score', 'Annual Income', 'Male', 'Female']

# ===== Streamlit UI =====
st.title("Prediksi Cluster Pelanggan Mall")

age = st.number_input("Age", min_value=1, max_value=100, value=25)
income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Prediksi Cluster"):
    # ---- Transform numeric manual ----
    age_scaled = (age - age_mean) / age_std
    score_scaled = (spending_score - score_mean) / score_std
    income_scaled = (income - income_median) / income_iqr

    # ---- One-hot encode gender ----
    male = 1.0 if gender == "Male" else 0.0
    female = 1.0 if gender == "Female" else 0.0

    # ---- Gabungkan semua menjadi dataframe ----
    X_concat = pd.DataFrame([[age_scaled, score_scaled, income_scaled, male, female]],
                            columns=feature_columns)

    # ---- Prediksi cluster ----
    label = gmm_model.predict(X_concat)[0]
    cluster_text = cluster_labels[label]

    st.success(f"Pelanggan termasuk dalam cluster: {cluster_text} (Cluster #{label})")
