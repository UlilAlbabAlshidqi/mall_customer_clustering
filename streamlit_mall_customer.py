import streamlit as st
import pandas as pd
import joblib

# ===== Load model dan transformer =====
gmm_model = joblib.load('gmm_model.joblib')
ct = joblib.load('scaler_transformer.joblib')

# Feature columns saat training (urutan harus sama)
feature_columns = ['Age', 'Spending Score', 'Annual Income', 'Male', 'Female']

st.title("Prediksi Cluster Pelanggan Mall")

# ===== Input user =====
age = st.number_input("Age", min_value=1, max_value=100, value=25)
income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Prediksi Cluster"):
    # Buat dataframe dari input user
    df_input = pd.DataFrame({
        "Age": [age],
        "Annual Income": [income],
        "Spending Score": [spending_score],
        "Gender": [gender]
    })

    # ---- Transform numeric ----
    X_numeric_scaled = pd.DataFrame(
        ct.transform(df_input[["Age", "Spending Score", "Annual Income"]]),
        columns=["Age","Spending Score","Annual Income"]
    )

    # ---- One-hot encode gender ----
    X_categoric = pd.get_dummies(df_input["Gender"]).astype(float)
    X_categoric = X_categoric.reindex(columns=["Male","Female"], fill_value=0)

    # ---- Gabung numeric + gender ----
    X_concat = pd.concat([X_numeric_scaled.reset_index(drop=True),
                          X_categoric.reset_index(drop=True)], axis=1)

    # ---- Urutkan kolom sesuai training ----
    X_concat = X_concat[feature_columns]

    # ---- Prediksi cluster ----
    label = gmm_model.predict(X_concat)[0]

    # ---- Mapping ke label teks ----
    cluster_labels = {
        0: "Male - Older Low Spender",
        1: "Male - Young High Spender",
        2: "Female - Older Low Spender",
        3: "Female - Young High Spender"
    }
    cluster_text = cluster_labels[label]

    st.success(f"Pelanggan termasuk dalam cluster: {cluster_text} (Cluster #{label})")
